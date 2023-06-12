import os
import argparse
import numpy as np
import numba as nb
import napari
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QPushButton,QFileDialog
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvas
from magicgui import widgets
import tifffile
import scipy.ndimage as nd

	
def load_tif(image_path):
	z = tifffile.imread(image_path).astype('int')
	print('Loaded %s'%(image_path),z.dtype,z.shape)
	return z

def calculate_absorbance(z,z0):
	transmittance = z.astype('float')/z0.astype('float')
	absorbance = -np.log10(transmittance)
	
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
	ellipse2 = np.array([[com[0],com[1]],[r*prefs['extent_factor'],r*prefs['extent_factor']]])
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

	print('Locating\n====================')
	print('Threshold',thresholdval)
	print('COM',com)
	print('radius',r,'\n====================')



def fxn_radial(viewer,prefs):
	
	ls = [layer for layer in viewer.layers if layer.name == 'com']
	if len(ls) == 0:
		print('need to get a com layer')
		return

	com = ls[0]._com
	r = ls[0]._r

	x,y = radial_profile(viewer.layers['absorbance'].data, com, r*prefs['extent_factor'])
	x *= prefs['calibration']/1000
	# y2 = nd.median_filter(y,3)
	y2 = nd.gaussian_filter1d(y,prefs['smooth_kernel'])

	fig,ax = plt.subplots(1)
	viewer.layers['com']._rad_fig = fig
	viewer.layers['com']._rad_ax = ax
	viewer.layers['com']._rad_ax.plot(x,y,color='k')
	viewer.layers['com']._rad_ax.plot(x,y2,color='r')
	viewer.layers['com']._rad_ax.set_xlabel('Radial Distance (mm)')
	viewer.layers['com']._rad_ax.set_ylabel('Absorption')
	viewer.layers['com']._rad_fig.canvas.draw()


	qw = QWidget()
	canvas = FigureCanvas(viewer.layers['com']._rad_fig)
	toolbar = NavigationToolbar(canvas,qw)
	vb = QVBoxLayout()
	b = QPushButton('Save')
	b.clicked.connect(lambda e: save_radial(e,qw,x,y))
	vb.addWidget(canvas)
	vb.addWidget(b)
	vb.addWidget(toolbar)
	qw.setLayout(vb)
	viewer.window.add_dock_widget(qw,name='Radial Average - (%.1f,%.1f)'%(com[0],com[1]))

def radial_profile(data, center, cutoff):
	@nb.njit(parallel=True)
	def quick_count(rs,rk,dk):
		radial_profile = np.zeros_like(rs)
		rpn = np.zeros_like(rs)

		for i in nb.prange(dk.size):
			for j in range(rs.size):
				if rk[i] == rs[j]:
					radial_profile[j] += dk[i]
					rpn[j] += 1
					break
		radial_profile = radial_profile/rpn
		return radial_profile

	x,y = np.indices((data.shape))
	r = np.sqrt((x.astype('float') - center[0])**2 + (y.astype('float') - center[1])**2)
	keep = r <= cutoff
	rk = r[keep].astype('float')
	dk = data[keep].astype('float')
	rs = np.unique(rk)

	# radial_profile = np.zeros_like(rs)
	# for i in range(rs.size):
	# 	radial_profile[i] = np.median(dk[rk==rs[i]])
	
	radial_profile = quick_count(rs,rk,dk)
	return rs, radial_profile

def save_radial(event,widget,rs,profile):
	fname = QFileDialog.getSaveFileName(parent=widget, caption="Save Radial Profile", filter="ASCII (*.txt)")[0]
	if not fname == "":
		np.savetxt(fname,np.array((rs,profile)))
		print('Saved:',fname)


def add_flat(viewer,flat):
	if 'flat' in viewer.layers:
		viewer.layers.remove('flat')
	viewer.add_image(flat,name='flat')
	contrast(viewer,'flat')
	
def fxn_load_flat(viewer,prefs):
	flat = load_tif(prefs['flat'])
	add_flat(viewer,flat)

def fxn_load_data(viewer,prefs):
	data = load_tif(prefs['absorbance'])
	if not 'flat' in viewer.layers:
		flat = np.zeros_like(data) + data.max()
		add_flat(viewer,flat)
	absorbance = calculate_absorbance(data,viewer.layers['flat'].data)
	if 'absorbance' in viewer.layers:
		viewer.layers.remove('absorbance')
	viewer.add_image(absorbance,name='absorbance')
	contrast(viewer,'absorbance')

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
	viewer.window._dock_widgets['qcquant'].container.calibration.value = factor
		

def initialize_radial():
	viewer = napari.Viewer()
		
	w_flat = widgets.FileEdit(mode='r',label='Flat-field File',name='flat')
	b_load_flat = widgets.PushButton(text='Load Flat')
	w_data = widgets.FileEdit(mode='r',label='Data File',name='absorbance')
	b_load_data = widgets.PushButton(text='Load Data')
	w_dish_diameter = widgets.FloatSpinBox(value=36.,label='Dish O.D. (mm)',min=0,name='dishdiameter')
	b_calc_conversion = widgets.PushButton(text='Calculate conversion (circle)')
	w_calibration = widgets.FloatSpinBox(value=47.4,label='Calibration (um/px)',min=0,name='calibration')
	w_threshold = widgets.FloatSpinBox(value=0.5,label='Threshold',min=0.01,max=.99,name='threshold')
	w_extentfactor = widgets.FloatSpinBox(value=4.,label='Extent Factor',min=0,name='extent_factor')
	w_filterwidth = widgets.FloatSpinBox(value=10.,label='Filter Width',min=0.,name='filter_width')
	w_smoothkernel = widgets.FloatSpinBox(value=30.,label='Smooth Kernel',min=0.,name='smooth_kernel')
	b_locate = widgets.PushButton(text='Locate Center')
	b_radial = widgets.PushButton(text='Calculate Radial Average')
	container = widgets.Container(widgets=[w_flat,b_load_flat,w_data,b_load_data,w_dish_diameter,b_calc_conversion,w_calibration, w_threshold, w_extentfactor, w_filterwidth, b_locate, w_smoothkernel, b_radial])
	
	b_locate.clicked.connect(lambda e: fxn_locate(viewer,container.asdict()))
	b_radial.clicked.connect(lambda e: fxn_radial(viewer,container.asdict()))
	b_load_flat.clicked.connect(lambda e: fxn_load_flat(viewer,container.asdict()))
	b_load_data.clicked.connect(lambda e: fxn_load_data(viewer,container.asdict()))
	b_calc_conversion.clicked.connect(lambda e: fxn_calc_conversion(viewer,container.asdict()))

	dock = viewer.window.add_dock_widget(container,name='qcquant: Radial Averaging Plugin')
	dock.container = container
	
	napari.run()

if __name__ == '__main__':
	initialize_radial()
