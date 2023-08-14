__plugin_name__ = 'qcquant (0.2.0)'

import napari
from magicgui import widgets
import tifffile
import numpy as np
import numba as nb
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QPushButton,QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def load_tif(image_path):
    z = tifffile.imread(image_path).astype('int')
    print('Loaded %s'%(image_path),z.dtype,z.shape)
    smax = 0
    with tifffile.TiffFile(image_path) as tif:
        # for key in tif.pages[0].tags.keys():
            # print(key,tif.pages[0].tags[key])
        # print(tif.pages[0].tags['PhotometricInterpretation'].value)
        smax = int(tif.pages[0].tags['SMaxSampleValue'].value)
        print(str(tif.pages[0].tags['PhotometricInterpretation'].value))
        print('%d-bit image'%(int(np.log2(smax+1))))
        if str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISBLACK':
            pass
        elif str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISWHITE':
            z = smax - z
            print('Inverted image')
        else:
            raise Exception('Cannot interpret photometric approach')
    return z

def calculate_absorbance(z,z0,mode='trans'):
    if mode == 'trans':
        transmittance = z.astype('float')/z0.astype('float')
        absorbance = -np.log10(transmittance)
    elif mode == 'epi':
        absorbance = z.astype('float') / 4096.
    else:
        raise Exception('%s mode is not implemented yet'%(mode))
    
    ## tif files are 16bit images so limit absorbance accordingly
    limit = np.log10(2**16)
    absorbance[absorbance > limit] = limit 
    absorbance[absorbance < -limit] = -limit 
    
    return absorbance

def contrast(viewer,layername='absorbance'):
    dmin = viewer.layers[layername].data.min()
    dmax = viewer.layers[layername].data.max()
    delta = (dmax-dmin)*1.
    viewer.layers[layername].contrast_limits_range = (dmin-delta,dmax+delta)
    viewer.layers[layername].contrast_limits = (dmin,dmax)

def fxn_locate(viewer,prefs):
    if 'com' in viewer.layers:
        viewer.layers.remove('com')

    points = [layer for layer in viewer.layers if isinstance(layer, napari.layers.Points)]
    if len(points) == 0:
        print('need to pick some points')
        return

    ## use last points
    d = points[-1].data
    if d.shape[0] < 2:
        print('need to pick at least two points')
        return

    minind = ((d[-1]+.5)//1).astype('int')
    centerind = ((d[-2]+.5)//1).astype('int')
    # print(centerind,minind)
    zmin = np.median(viewer.layers['absorbance'].data[minind[0]-1:minind[0]+2,minind[1]-1:minind[1]+2])
    zcenter = np.median(viewer.layers['absorbance'].data[centerind[0]-1:centerind[0]+2,centerind[1]-1:centerind[1]+2])

    if zcenter < zmin:
        print('invert the image, please')
        return
    thresholdval = prefs['threshold']*(zcenter-zmin) + zmin

    mask = viewer.layers['absorbance'].data > thresholdval
    mask = nd.binary_closing(mask)
    mask = nd.gaussian_filter(mask.astype('float'),prefs['filter_width'])
    mask = mask > .2
    # mask = nd.binary_fill_holes(mask)

    # viewer.add_image(mask)
    labeled_mask, num_features = nd.label(mask)
    maskind = labeled_mask[centerind[0],centerind[1]]
    com = nd.center_of_mass(viewer.layers['absorbance'].data, labels=labeled_mask,index=maskind)
    area = float((labeled_mask==maskind).sum())

    r = np.sqrt(area/np.pi)
    ellipse = np.array([[com[0],com[1]],[r,r]])
    rr = prefs['extent_factor']/(prefs['calibration']/1000.)
    ellipse2 = np.array([[com[0],com[1]],[rr,rr]])
    # r2 = 1.2*r
    # box = np.array([[com[0]-r2,com[1]-r2],[com[0]-r2,com[1]+r2],[com[0]+r2,com[1]+r2],[com[0]+r2,com[1]-r2],])

    labeled_mask[labeled_mask==0] -= num_features
    if 'labeled_mask' in viewer.layers:
        viewer.layers.remove('labeled_mask')
    viewer.add_image(labeled_mask,colormap='viridis',opacity=.2)
    
    if 'zones' in viewer.layers:
        viewer.layers.remove('zones')
    shapes = viewer.add_shapes(name='zones')
    # shapes.add_rectangles([box,],face_color='blue',edge_color='darkblue')
    shapes.add_ellipses([ellipse2,],face_color='blue',edge_color='darkblue')
    shapes.add_ellipses([ellipse,],face_color='red',edge_color='darkred')
    shapes.opacity = .2
    viewer.add_points(com,name='com',face_color='red',edge_color='darkred')
    viewer.layers['com']._com = com
    viewer.layers['com']._r = r
    
    
    viewer.layers['zones'].visible = False
    viewer.layers['labeled_mask'].visible = False

    print('Locating\n====================')
    print('Threshold',thresholdval)
    print('COM',com)
    print('radius',r,'\n====================')

def fxn_radial(viewer,prefs):
    ls = [layer for layer in viewer.layers if layer.name == 'com']
    if len(ls) == 0:
        print('need to get a com layer')
        return

    # com = ls[0]._com
    # r = ls[0]._r
    com = viewer.layers['com'].data[-1]

#     x,y = radial_profile(viewer.layers['absorbance'].data, com, r*prefs['extent_factor'])
    x,y = radial_profile(viewer.layers['absorbance'].data, com,prefs['extent_factor']/(prefs['calibration']/1000.),prefs['bin_width']/(prefs['calibration']/1000.))
    x *= prefs['calibration']/1000.
#     y2 = nd.median_filter(y,int(prefs['smooth_kernel']))
    y2 = nd.gaussian_filter1d(y,prefs['smooth_kernel'])

    fig,ax = plt.subplots(1,figsize=(4,3))
    viewer.layers['com']._rad_fig = fig
    viewer.layers['com']._rad_ax = ax
    viewer.layers['com']._rad_ax.plot(x,y,color='k',lw=1.2)
    viewer.layers['com']._rad_ax.plot(x,y2,color='r',alpha=.8,lw=.8)
    viewer.layers['com']._rad_ax.set_xlim(0,x.max())
    viewer.layers['com']._rad_ax.set_ylim(0.,viewer.layers['com']._rad_ax.get_ylim()[1])
    viewer.layers['com']._rad_ax.set_xlabel('Radial Distance (mm)')
    viewer.layers['com']._rad_ax.set_ylabel('Absorption')
    viewer.layers['com']._rad_fig.canvas.draw()

    qw = QWidget()
    canvas = FigureCanvas(viewer.layers['com']._rad_fig)
    toolbar = NavigationToolbar(canvas)

    vb = QVBoxLayout()
    b = QPushButton('Save')
    b.clicked.connect(lambda e: save_radial(e,qw,x,y,y2))
    from PyQt5.QtWidgets import QSizePolicy
    canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
    vb.addWidget(canvas)
#     vb.addStretch()
    vb.addWidget(toolbar)
    vb.addWidget(b)
    qw.setLayout(vb)
    viewer.window.add_dock_widget(qw,name='Plot: Radial Average - (%.1f,%.1f)'%(com[0],com[1]))
#     qw.show()
    fig.tight_layout()
    canvas.draw()
    
    

def radial_adjust(viewer,prefs):
    ls = [layer for layer in viewer.layers if layer.name == 'com']
    if len(ls) == 0:
        print('need to get a com layer')
        return
    com = viewer.layers['com'].data[-1].copy()
    
    
    from scipy.optimize import minimize
    def minfxn(theta,viewer,prefs,com):
        i,j = theta
        ll = 64
        if i < -ll or i > ll or j < -ll or j > ll:
            return np.inf
        x,y = radial_profile(viewer.layers['absorbance'].data, com+theta, prefs['fit_extent']/(prefs['calibration']/1000.), prefs['bin_width']/(prefs['calibration']/1000.), method='norm_var')
        out = y.max()
        return out
    
    out = minimize(minfxn,np.array((0.,0.)),args=(viewer,prefs,com),method='Nelder-Mead',options={'initial_simplex':np.array(((1.,1.),(0.,1),(1.,0)))})
    print(out)
    viewer.layers['com'].data[-1] += out.x
    viewer.layers['com'].refresh()




@nb.njit
def bin_count_mean(rk,dk,dx):
    ### rk: 1d array of radii of all pixels within cutoff
    ### dk: 1d array of data values of all pixels within cutoff
    ### dx: bin size for quantization
    
    nbins = int((np.max(rk)-np.min(rk))//dx+1)
    if nbins < 1:
        nbins = 1
    y = np.zeros(nbins)
    n = np.zeros(nbins)
    out = np.zeros((2,nbins))
    ind = 0
    for i in range(rk.size):
        ind = int(rk[i]//dx)
        y[ind] += dk[i]
        n[ind] += 1.
    for i in range(nbins):
        out[0,i] = i*dx
        if n[i] > 0:
            out[1,i] = y[i]/n[i]
    return out
    
@nb.njit
def bin_count_var(rk,dk,dx):
    ### rk: 1d array of radii of all pixels within cutoff
    ### dk: 1d array of data values of all pixels within cutoff
    ### dx: bin size for quantization
    
    nbins = int((np.max(rk)-np.min(rk))//dx+1)
    if nbins < 1:
        nbins = 1
    y = np.zeros(nbins)
    n = np.zeros(nbins)
    Ex = bin_count_mean(rk,dk,dx)[1]
    out = np.zeros((2,nbins))
    ind = 0
    for i in range(rk.size):
        ind = int(rk[i]//dx)
        y[ind] += (dk[i]-Ex[ind])**2.
        n[ind] += 1.
    for i in range(nbins):
        out[0,i] = i*dx
        if n[i] > 0:
            out[1,i] = y[i]/n[i]
    return out

def radial_profile(data, center, cutoff,dx,method='mean'):
    # @nb.njit
    # def quick_count(rs,rk,dk):
    #     ### rs: 1d array of all unique radii within cutoff
    #     ### rk: 1d array of radii of all pixels within cutoff
    #     ### dk: 1d array of data values of all pixels within cutoff
    #     radial_profile = np.zeros_like(rs)
    #     rpn = np.zeros_like(rs)
    #
    #     for i in range(dk.size):
    #         for j in range(rs.size):
    #             if rk[i] == rs[j]:
    #                 radial_profile[j] += dk[i]
    #                 rpn[j] += 1
    #                 break
    #     radial_profile = radial_profile/rpn
    #     return radial_profile

    x,y = np.indices((data.shape))
    ## This rounds up/down to the nearest pixels, then flattens so that pixel access is symmetric and you actually do some averaging
    r = np.sqrt((x.astype('float') - float((0.5+center[0])//1))**2 + (y.astype('float') - float((0.5+center[1])//1))**2)
    keep = r <= cutoff
    rk = r[keep].astype('float')
    dk = data[keep].astype('float')
#     rs = np.unique(rk)
#     radial_profile = quick_count(rs,rk,dk)
    if method == 'mean':
        rs,radial_profile = bin_count_mean(rk,dk,dx)
    elif method in ['var','norm_var']:
        rs,radial_profile = bin_count_var(rk,dk,dx)
        if method == 'norm_var':
            radial_profile = np.sqrt(radial_profile)/bin_count_mean(rk,dk,dx)[1]
    return rs, radial_profile

def save_radial(event,widget,rs,profile,smoothed):
    fname = QFileDialog.getSaveFileName(parent=widget, caption="Save Radial Profile", filter="ASCII (*.txt)")[0]
    if not fname == "":
        np.savetxt(fname,np.array((rs,profile,smoothed)))
        print('Saved:',fname)

def add_flat(viewer,flat):
    if 'flat' in viewer.layers:
        viewer.layers.remove('flat')
    viewer.add_image(flat,name='flat')
    contrast(viewer,'flat')

def fxn_load_flat(viewer,prefs):
    flat = load_tif(prefs['flat'])
    # if prefs['invertdata']:
        # flat = 4096-flat
    add_flat(viewer,flat)
    
    
# # def check_symmetry(viewer,prefs):
# ll = 1
# pp = 64
# for i in range(-ll,ll+1,1):
#     for j in range(-ll,ll+1,1):
#         q = viewer.layers['com'].data[0].astype('int') + np.array([i,j],dtype='int')
#         d = viewer.layers['absorbance'].data
#         dd = d[q[0]-pp:q[0]+pp,q[1]-pp:q[1]+pp]
#         # dd = np.concatenate([dd,dd],axis=0)
#         # dd = np.concatenate([dd,dd],axis=1)
#         ff = np.abs((np.fft.fft2(dd)))
#         viewer.add_image(np.log(ff))
#
# d = viewer.layers['absorbance'].data
# com = viewer.layers['com'].data[0].astype('int')
# ringx = np.array([-2,-2,-1,0,1,2,2,2,1,0,-1,-2])*100
# ringy = np.array([0,-1,-2,-2,-2,-1,0,1,2,2,2,1])*100
# record = []
# ll = 10
# for i in range(-ll,ll+1):
#     for j in range(-ll,ll+1):
#         q = com + np.array([i,j],dtype='int')
#         ii = q[0] + ringx
#         jj = q[1] + ringy
#         # print(d[ii,jj].shape)
#         record.append([i,j,np.sum((d[ii,jj]-d[q[0],q[1]])**2.)])
# mm = np.array([ri[2] for ri in record])
# print(record[np.argmax(mm)])
# viewer.layers['com'].data[0,0] -= record[np.argmax(mm)][0]
# viewer.layers['com'].data[0,1] -= record[np.argmax(mm)][1]

# ll = 1
# pp = 20
# for i in range(-ll,ll+1,1):
#     for j in range(-ll,ll+1,1):
#         dx = np.sum((d[q[0]+i-pp:q[0]+i+1,q[1]+j-pp:q[1]+j+pp+1]-d[q[0]+i:q[0]+i+pp+1,q[1]+j-pp:q[1]+j-pp+1])**2.)
#         dy = np.sum((d[q[0]+i-pp:q[0]+i+pp+1,q[1]+j-pp:q[1]+j+1]-d[q[0]+i-pp:q[0]+i+pp+1,q[1]+j-pp:q[1]+j-pp+1])**2.)
#         print(i,j,)
#
#

def new_com(viewer,data):
    ls = [layer for layer in viewer.layers if layer.name == 'com']
    if len(ls) == 0:
        com = np.array([data.shape[0]/2.,data.shape[1]/2.])
        viewer.add_points(com,name='com',face_color='red',edge_color='darkred')

def fxn_load_data_trans(viewer,prefs):
    data = load_tif(prefs['absorbance'])
    # if prefs['invertdata']:
        # data = 4096-data
    if not 'flat' in viewer.layers:
        flat = np.zeros_like(data) + data.max()
        add_flat(viewer,flat)
    absorbance = calculate_absorbance(data,viewer.layers['flat'].data,mode='trans')
    if 'absorbance' in viewer.layers:
        viewer.layers.remove('absorbance')
    viewer.add_image(absorbance,name='absorbance')
    contrast(viewer,'absorbance')
    new_com(viewer,data)

def fxn_load_data_epi(viewer,prefs):
    data = load_tif(prefs['absorbance'])
    # if prefs['invertdata']:
        # data = 4096-data
    absorbance = calculate_absorbance(data,None,mode='epi')
    if 'absorbance' in viewer.layers:
        viewer.layers.remove('absorbance')
    viewer.add_image(absorbance,name='absorbance')
    contrast(viewer,'absorbance')
    new_com(viewer,data)

def fxn_calc_conversion(viewer,prefs):
    shapes = [layer for layer in viewer.layers if isinstance(layer, napari.layers.shapes.Shapes)]
    if len(shapes)== 0:
        print('Draw a circle! Need a shapes layer')
        return
    ss = shapes[-1] ## use last entry
    inds = np.where(np.array((ss.shape_type))=='ellipse')[0]
    if inds.size == 0:
        print('Draw a circle!')
        return
    ind = inds[-1]
    ellipse = ss.data[ind]
    
    r1 = (ellipse[3,0]-ellipse[0,0])/2.
    r2 = (ellipse[1,1]-ellipse[0,1])/2.
    r = .5*(r1+r2)
    
    factor = prefs['dishdiameter']/(2.*r)*1000  ## um/pix
    print(factor)
    viewer.window._dock_widgets[__plugin_name__].container.calibration.value = factor

def initialize_radial():
    viewer = napari.Viewer()

    w_flat = widgets.FileEdit(mode='r',label='Flat-field File',name='flat')
    b_load_flat = widgets.PushButton(text='Load Flat')
    w_data = widgets.FileEdit(mode='r',label='Data File',name='absorbance')
    b_load_data_trans = widgets.PushButton(text='Load Data (Trans-illumination)')
    b_load_data_epi = widgets.PushButton(text='Load Data (Epi-illumination)')
    # w_invert = widgets.CheckBox(text='Invert (12-bit) Data',value=False,name='invertdata')
    w_dish_diameter = widgets.FloatSpinBox(value=36.,label='Dish O.D. (mm)',min=0,name='dishdiameter')
    b_calc_conversion = widgets.PushButton(text='Calculate conversion (circle)')
    w_calibration = widgets.FloatSpinBox(value=47.4,label='Calibration (um/px)',min=0,name='calibration')
    w_extentfactor = widgets.FloatSpinBox(value=25.,label='Extent (mm)',min=0,name='extent_factor')
    w_fitextent = widgets.FloatSpinBox(value=4.,label='Fit Extent (mm)',min=0,name='fit_extent')
    w_filterwidth = widgets.FloatSpinBox(value=10.,label='Filter Width',min=0.,name='filter_width')
    w_binwidth = widgets.FloatSpinBox(value=.025,label='Bin Width (mm)',min=0.,name='bin_width',step=.001)
    w_smoothkernel = widgets.FloatSpinBox(value=2.,label='Smooth Kernel (bins)',min=0.,name='smooth_kernel')
    b_locate = widgets.PushButton(text='Locate Center')
    b_radial = widgets.PushButton(text='Calculate Radial Average')
    b_fit = widgets.PushButton(text='Find Radial Center')
    container = widgets.Container(widgets=[w_flat,b_load_flat,w_data,b_load_data_trans,b_load_data_epi,w_dish_diameter,b_calc_conversion,w_calibration, w_extentfactor, w_fitextent, w_binwidth, w_smoothkernel, b_fit, b_radial])
    
    b_locate.clicked.connect(lambda e: fxn_locate(viewer,container.asdict()))
    b_radial.clicked.connect(lambda e: fxn_radial(viewer,container.asdict()))
    b_load_flat.clicked.connect(lambda e: fxn_load_flat(viewer,container.asdict()))
    b_load_data_trans.clicked.connect(lambda e: fxn_load_data_trans(viewer,container.asdict()))
    b_load_data_epi.clicked.connect(lambda e: fxn_load_data_epi(viewer,container.asdict()))
    b_calc_conversion.clicked.connect(lambda e: fxn_calc_conversion(viewer,container.asdict()))
    b_fit.clicked.connect(lambda e: radial_adjust(viewer,container.asdict()))

    dock = viewer.window.add_dock_widget(container,name=__plugin_name__)
    dock.container = container
    
    napari.run()

if __name__ == '__main__':
    initialize_radial()
