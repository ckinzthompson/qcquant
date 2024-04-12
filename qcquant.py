__plugin_name__ = 'qcquant (0.3.0)'

############################################
######## IMPORTANT GLOBAL VARIABLES ########
viewer = None         ## Napari viewer
dock_qcquant = None   ## QCQuant config dock
dock_radial = None    ## Radial plot dock
dock_profile = None   ## Profile plot dock
############################################

import napari
from magicgui import widgets
import tifffile
import numpy as np
import numba as nb
import scipy.ndimage as nd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QPushButton,QFileDialog,QSizePolicy,QDockWidget,QHBoxLayout,QLabel,QLineEdit
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def load_tif(image_path):
    z = tifffile.imread(image_path).astype('int')
    print('Loaded %s'%(image_path),z.dtype,z.shape)
    smax = 0
    with tifffile.TiffFile(image_path) as tif:
        # for key in tif.pages[0].tags.keys():
        #     print(key,tif.pages[0].tags[key].value)
        # print(tif.pages[0].tags['PhotometricInterpretation'].value)
        smax = int(tif.pages[0].tags['SMaxSampleValue'].value)
        print(str(tif.pages[0].tags['PhotometricInterpretation'].value))
        print('%d-bit image'%(int(np.log2(smax+1))))
        if str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISBLACK' or tif.pages[0].tags['PhotometricInterpretation'].value == 1:
            pass
        elif str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISWHITE' or tif.pages[0].tags['PhotometricInterpretation'].value == 0:
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
    x,y = np.indices((data.shape))
    ## This rounds up/down to the nearest pixels, then flattens so that pixel access is symmetric and you actually do some averaging
    r = np.sqrt((x.astype('float') - float((0.5+center[0])//1))**2 + (y.astype('float') - float((0.5+center[1])//1))**2)
    keep = r <= cutoff
    rk = r[keep].astype('float')
    dk = data[keep].astype('float')
#     rs = np.unique(rk)
#     radial_profile = quick_count(rs,rk,dk)
    if method == 'mean':
        rs,radial_profile_ = bin_count_mean(rk,dk,dx)
    elif method in ['var','norm_var']:
        rs,radial_profile_ = bin_count_var(rk,dk,dx)
        if method == 'norm_var':
            radial_profile_ = np.sqrt(radial_profile_)/bin_count_mean(rk,dk,dx)[1]
    return rs, radial_profile_


def get_prefs():
    global viewer,dock_qcquant
    prefs = dock_qcquant.container.asdict()
    return prefs 

def com_to_top():
    global viewer
    if 'com' in viewer.layers:
        viewer.layers.move(viewer.layers.index('com'),-1)

def contrast(layername='absorbance'):
    global viewer
    dmin = viewer.layers[layername].data.min()
    dmax = viewer.layers[layername].data.max()
    delta = (dmax-dmin)*1.
    viewer.layers[layername].contrast_limits_range = (dmin-delta,dmax+delta)
    viewer.layers[layername].contrast_limits = (dmin,dmax)
    
def new_com(data):
    global viewer
    ls = [layer for layer in viewer.layers if layer.name == 'com']
    if len(ls) == 0:
        com = np.array([data.shape[0]/2.,data.shape[1]/2.])
        viewer.add_points(com,name='com',face_color='red',edge_color='darkred')

def add_flat(flat):
    global viewer
    if 'flat' in viewer.layers:
        viewer.layers.remove('flat')
    viewer.add_image(flat,name='flat')
    com_to_top()
    contrast('flat')

def radial_adjust():
    global viewer
    prefs = get_prefs()
    ls = [layer for layer in viewer.layers if layer.name == 'com']
    if len(ls) == 0:
        print('need to get a com layer')
        return
    com = viewer.layers['com'].data[-1].copy()
    
    def minfxn(theta,com):
        global viewer
        prefs = get_prefs()
        i,j = theta
        ll = 64
        if i < -ll or i > ll or j < -ll or j > ll:
            return np.inf
        x,y = radial_profile(viewer.layers['absorbance'].data, com+theta, prefs['fit_extent']/(prefs['calibration']/1000.), prefs['bin_width']/(prefs['calibration']/1000.), method='norm_var')
        if prefs['locate_mode'] == 'Max':
            out = y.max()
        elif prefs['locate_mode'] == 'Mean':
            out = y.mean()
        return out
    
    out = minimize(minfxn,np.array((0.,0.)),args=(com,),method='Nelder-Mead',options={'initial_simplex':np.array(((1.,1.),(0.,1),(1.,0)))})
    print(out)
    viewer.layers['com'].data[-1] += out.x
    viewer.layers['com'].refresh()

def save_radial():
    global viewer,dock_radial
    if dock_radial.x is None or dock_radial.y is None or dock_radial.y2 is None:
        return
        
    prefs = get_prefs()
    fname = QFileDialog.getSaveFileName(parent=dock_radial, caption="Save Radial Analysis")[0]
    if not fname == "":
        np.savetxt(fname+'.txt',np.array((dock_radial.x,dock_radial.y,dock_radial.y2)))
        print('Saved:',fname+'.txt')
        
        save_img(fname)
        print('Saved:',fname+'.png')

    
def save_img(fname):
    global viewer
    prefs = get_prefs()
    com = viewer.layers['com'].data[-1].astype('int')
    d = viewer.layers['absorbance'].data
    # print(com)
    
    if prefs['centerimg']:
        def minfxn(theta,com):
            global viewer
            prefs = get_prefs()
            i,j = theta
            ll = 64
            if i < -ll or i > ll or j < -ll or j > ll:
                return np.inf
            x,y = radial_profile(d, com+theta, (prefs['dishdiameter']/2.+1.)/(prefs['calibration']/1000.), prefs['bin_width']/(prefs['calibration']/1000.), method='norm_var')
            edge = 3./(prefs['calibration']/1000.)
            ddr = prefs['dishdiameter']/2./(prefs['calibration']/1000.)
            keep = np.bitwise_and(x>(ddr-edge),x<(ddr+edge))
            out = y[keep].mean()
            return out
        print('Finding Image COM')
        out = minimize(minfxn, np.array((0.,0.)), args=(com,), method='Nelder-Mead', options={'initial_simplex':np.array(((1.,1.),(0.,1),(1.,0)))})
        com += out.x.astype('int')
        print('Done',com)
    
    nx,ny = d.shape
    r = prefs['dishdiameter']/2. + 1.
    r /= prefs['calibration']/1000.
    r = int((r+1)//1)
    
    out = np.zeros((r*2,r*2))
    
    x0 = np.max((0,com[0]-r))
    x1 = np.min((nx,com[0]+r))
    y0 = np.max((0,com[1]-r))
    y1 = np.min((ny,com[1]+r))
    dx0 = x0-(com[0]-r)
    dx1 = (com[0]+r)-x1
    dy0 = y0-(com[1]-r)
    dy1 = (com[1]+r)-y1

    dd = d[x0:x1,y0:y1]
    out[dx0:out.shape[0]-dx1,dy0:out.shape[1]-dy1] = dd
    
    fig,ax = plt.subplots(1,figsize=(4,4),dpi=400)
    ax.imshow(out,origin='lower',interpolation='bicubic',cmap='Greys_r',vmin=dd.min(),vmax=dd.max())
    ax.axis('off')
    fig.subplots_adjust(left=0,right=1.,top=1.,bottom=0.)
    fig.savefig('%s.png'%(fname))
    plt.close()

def fxn_radial():
    global viewer,dock_qcquant,dock_radial
    prefs = get_prefs()
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
    
    dock_radial.x = x
    dock_radial.y = y
    dock_radial.y2 = y2

    dock_radial.ax.cla()
    dock_radial.ax.plot(x,y,color='k',lw=1.2)
    dock_radial.ax.plot(x,y2,color='r',alpha=.8,lw=.8)
    dock_radial.ax.set_xlim(0,x.max())
    dock_radial.ax.set_ylim(0.,dock_radial.ax.get_ylim()[1])
    dock_radial.ax.set_xlabel('Radial Distance (mm)')
    dock_radial.ax.set_ylabel('Absorption')
    dock_radial.fig.subplots_adjust(left=.2,bottom=.2)
    dock_radial.canvas.draw()
    dock_radial.raise_()

def fxn_load_flat():
    global viewer
    prefs = get_prefs()
    flat = load_tif(prefs['flat'])
    # if prefs['invertdata']:
        # flat = 4096-flat
    add_flat(flat)

def fxn_load_data_trans():
    global viewer
    prefs = get_prefs()
    data = load_tif(prefs['absorbance'])
    # if prefs['invertdata']:
        # data = 4096-data
    if not 'flat' in viewer.layers:
        flat = np.zeros_like(data) + data.max()
        add_flat(flat)
    absorbance = calculate_absorbance(data,viewer.layers['flat'].data,mode='trans')
    if 'absorbance' in viewer.layers:
        viewer.layers.remove('absorbance')
    viewer.add_image(absorbance,name='absorbance')
    com_to_top()
    contrast('absorbance')
    new_com(data)

def fxn_load_data_epi():
    global viewer
    prefs = get_prefs()
    data = load_tif(prefs['absorbance'])
    # if prefs['invertdata']:
        # data = 4096-data
    absorbance = calculate_absorbance(data,None,mode='epi')
    if 'absorbance' in viewer.layers:
        viewer.layers.remove('absorbance')
    viewer.add_image(absorbance,name='absorbance')
    com_to_top()
    contrast('absorbance')
    new_com(data)

def fxn_calc_conversion():
    global viewer,dock_qcquant
    prefs = get_prefs()
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
    dock_qcquant.container['calibration'].value = factor
    # print(factor)
    # viewer.window._dock_widgets[__plugin_name__].container.calibration.value = factor


def update_dcom():
    global viewer,dock_qcquant
    
    if 'com' in viewer.layers:
        prefs = get_prefs()
        com = np.array(viewer.layers['com'].data[-1])
        cursor = np.array(viewer.cursor.position)
        distance = np.linalg.norm(cursor-com)
        distance *= prefs['calibration']/1000.
        dock_qcquant.container['showdistance'].text = 'Distance from COM: %.1f mm'%(distance)
    else:
        dock_qcquant.container['showdistance'].text = 'Distance from COM: '

def fxn_showdistance():
    global viewer,dock_qcquant
    prefs = get_prefs()
    state = prefs['showdistance']
    
    if state:
        dock_qcquant._timer_distance.timeout.connect(lambda : update_dcom())
        dock_qcquant._timer_distance.start()
    else:
        dock_qcquant._timer_distance.stop()
        dock_qcquant._timer_distance.disconnect()
        dock_qcquant.container['showdistance'].text = 'Distance from COM: '


def initialize_qcquant_dock():
    global viewer

    w_flat = widgets.FileEdit(mode='r',label='Flat-field File',name='flat')
    b_load_flat = widgets.PushButton(text='Load Flat')
    w_data = widgets.FileEdit(mode='r',label='Data File',name='absorbance')
    b_load_data_trans = widgets.PushButton(text='Load Data (Trans-illumination)')
    b_load_data_epi = widgets.PushButton(text='Load Data (Epi-illumination)')
    # w_invert = widgets.CheckBox(text='Invert (12-bit) Data',value=False,name='invertdata')

    w_dish_diameter = widgets.FloatSpinBox(value=36.,label='Dish O.D. (mm)',min=0,max=1000.,name='dishdiameter')
    b_calc_conversion = widgets.PushButton(text='Calculate conversion (circle)')
    w_calibration = widgets.FloatSpinBox(value=47.4,label='Calibration (um/px)',min=0,name='calibration')
    w_extentfactor = widgets.FloatSpinBox(value=25.,label='Extent (mm)',min=0,name='extent_factor')
    w_fitextent = widgets.FloatSpinBox(value=4.,label='Locate Extent (mm)',min=0,name='fit_extent')
    w_filterwidth = widgets.FloatSpinBox(value=10.,label='Filter Width',min=0.,name='filter_width')
    w_binwidth = widgets.FloatSpinBox(value=.025,label='Radial Bin Width (mm)',min=0.,name='bin_width',step=.001)
    w_smoothkernel = widgets.FloatSpinBox(value=2.,label='Smoothing Kernel (bins)',min=0.,name='smooth_kernel')
    w_locatemode = widgets.ComboBox(value='Mean',label='Locate Mode',choices=['Max','Mean'],name='locate_mode')
    w_centerimg = widgets.CheckBox(text='Plate Image: Locate Plate?',value=True,name='centerimg')
    w_showdistance = widgets.CheckBox(text='Distance from COM:',value=False,name='showdistance')
    b_locate = widgets.PushButton(text='Locate Center')
    b_radial = widgets.PushButton(text='Calculate Radial Average')
    b_fit = widgets.PushButton(text='Find Radial Center')
    # b_save = widgets.PushButton(text='Save')
    container = widgets.Container(widgets=[
        w_flat,
        b_load_flat,
        w_data,
        b_load_data_trans,
        b_load_data_epi,
        w_dish_diameter,
        w_calibration,
        b_calc_conversion,
        w_fitextent,
        w_locatemode,
        b_fit,
        w_extentfactor,
        w_binwidth,
        w_smoothkernel,
        b_radial,
        w_centerimg,
        w_showdistance,
    ])
    
    b_locate.clicked.connect(lambda e: fxn_locate())
    b_radial.clicked.connect(lambda e: fxn_radial())
    b_load_flat.clicked.connect(lambda e: fxn_load_flat())
    b_load_data_trans.clicked.connect(lambda e: fxn_load_data_trans())
    b_load_data_epi.clicked.connect(lambda e: fxn_load_data_epi())
    b_calc_conversion.clicked.connect(lambda e: fxn_calc_conversion())
    b_fit.clicked.connect(lambda e: radial_adjust())
    w_showdistance.clicked.connect(lambda e: fxn_showdistance())

    dock = viewer.window.add_dock_widget(container,name=__plugin_name__)
    dock.container = container
    
    dock._timer_distance = QTimer()
    dock._timer_distance.setInterval(int(1000//25.))
    return dock

def initialize_radial_dock():
    global viewer
    
    fig,ax = plt.subplots(1,figsize=(4,3))
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas)
    # canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
    canvas.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
    b = QPushButton('Save')
    b.clicked.connect(lambda e: save_radial())
    b2 = QPushButton('Refresh')
    b2.clicked.connect(lambda e: fxn_radial())

    qw = QWidget()
    vb = QVBoxLayout()
    vb.addWidget(canvas)
    vb.addWidget(toolbar)
    vb.addWidget(b2)
    vb.addWidget(b)
    vb.addStretch()
    qw.setLayout(vb)

    qdw = viewer.window.add_dock_widget(qw,name='Radial Average')
    qdw.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
    qdw.fig = fig
    qdw.ax = ax
    qdw.canvas = canvas
    qdw.toolbar = toolbar
    
    qdw.x = None
    qdw.y = None
    qdw.y2 = None
    return qdw
    
    
def fxn_prof_save():
    global viewer,dock_profile
    print('save')

    fname = QFileDialog.getSaveFileName(parent=dock_radial, caption="Save Figure",filter='*.pdf')[0]
    if not fname == "":
        dock_profile.fig.savefig(fname)

def fxn_prof_clear():
    global viewer,dock_profile
    dock_profile.ax.cla()
    dock_profile.ax.set_xlabel('Radial Distance (mm)')
    dock_profile.ax.set_ylabel('Absorption')
    dock_profile.fig.subplots_adjust(left=.2,bottom=.2)
    dock_profile.canvas.draw()
    

def fxn_prof_add():
    global viewer,dock_profile
    fname = dock_profile.profile_filename.text()
    if fname == "":
        return
    # try:
    if 1:
        d = np.loadtxt(fname)
        if not d.ndim == 2 or not d.shape[0] == 3:
            print('Data is wrong',d.shape)
            return
        dock_profile.ax.plot(d[0],d[1],color='k',lw=1.)
        dock_profile.ax.set_xlim(d[0].min(),d[0].max())
        dock_profile.ax.set_ylim(0.,dock_profile.ax.get_ylim()[1])
        
        nlines = len(dock_profile.ax.lines)
        for i in range(nlines):
            line = dock_profile.ax.lines[i]
            line.set_color(plt.cm.viridis(float(i)/float(nlines)))
        
        dock_profile.ax.set_xlabel('Radial Distance (mm)')
        dock_profile.ax.set_ylabel('Absorption')
        dock_profile.fig.subplots_adjust(left=.2,bottom=.2)
        dock_profile.canvas.draw()
    # except:
        # print('Could not load %s'%(fname))
    
def fxn_prof_select():
    global viewer,dock_profile
    fname = QFileDialog.getOpenFileName(parent=dock_profile, caption="Load Profile file")[0]
    if not fname == "":
        dock_profile.profile_filename.setText(fname)
        
        
        
def initialize_profile_dock():
    global viewer

    fig,ax = plt.subplots(1,figsize=(4,3))
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas)
    # canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
    canvas.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
    b = QPushButton('Save Figure')
    b.clicked.connect(lambda e: fxn_prof_save())
    b2 = QPushButton('Clear Plot')
    b2.clicked.connect(lambda e: fxn_prof_clear())
    b_add = QPushButton('Add')
    b_add.clicked.connect(lambda e: fxn_prof_add())


    profile_filename = QLineEdit()
    profile_filename.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
    b_selectprofile = QPushButton('Select File')
    b_selectprofile.clicked.connect(lambda e: fxn_prof_select())
    qwf = QWidget()
    hbox = QHBoxLayout()
    hbox.addWidget(QLabel("Profile File"))
    hbox.addWidget(profile_filename)
    hbox.addWidget(b_selectprofile)
    hbox.addWidget(b_add)
    qwf.setLayout(hbox)

    qw = QWidget()
    vb = QVBoxLayout()
    vb.addWidget(canvas)
    vb.addWidget(toolbar)
    vb.addWidget(qwf)
    vb.addWidget(b2)
    vb.addWidget(b)
    vb.addStretch()
    qw.setLayout(vb)

    qdw = viewer.window.add_dock_widget(qw,name='Profile Plot')
    qdw.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
    qdw.fig = fig
    qdw.ax = ax
    qdw.canvas = canvas
    qdw.toolbar = toolbar

    qdw.x = None
    qdw.y = None
    qdw.y2 = None
    
    qdw.profile_filename = profile_filename
    return qdw
    
    
if __name__ == '__main__':
    viewer = napari.Viewer()
    dock_qcquant = initialize_qcquant_dock()
    dock_radial = initialize_radial_dock()
    dock_profile = initialize_profile_dock()
    viewer.window._qt_window.tabifyDockWidget(dock_qcquant,dock_radial)
    viewer.window._qt_window.tabifyDockWidget(dock_radial,dock_profile)
    dock_qcquant.raise_()
    napari.run()
