__plugin_name__ = 'qcquant (0.3.0)'

############################################
######## IMPORTANT GLOBAL VARIABLES ########
viewer = None       ## Napari viewer    ####
dock_qcquant = None ## QCQuant config dock #
############################################

import napari
from magicgui import widgets
import tifffile
import numpy as np
import numba as nb
import scipy.ndimage as nd
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
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
		#	 print(key,tif.pages[0].tags[key].value)
		# print(tif.pages[0].tags['PhotometricInterpretation'].value)
		smax = int(tif.pages[0].tags['SMaxSampleValue'].value)
		print(str(tif.pages[0].tags['PhotometricInterpretation'].value))
		print('%d-bit image; Maximum counts is %d'%(int(np.log2(smax+1)),smax))
		if str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISBLACK' or tif.pages[0].tags['PhotometricInterpretation'].value == 1:
			pass
			print('Regular image')
		elif str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISWHITE' or tif.pages[0].tags['PhotometricInterpretation'].value == 0:
			print('Inverted image')
		else:
			raise Exception('Cannot interpret photometric approach')
	return z,smax

def calculate_intensity(z,z0,mode='trans',dust_filter=True):
	if mode == 'trans':
		intensity = z.astype('float')/z0.astype('float')
	elif mode == 'epi':
		intensity = z.astype('float') / z0
	else:
		raise Exception('%s mode is not implemented yet'%(mode))

	## don't play around
	intensity[intensity > 1.] = 1.
	intensity[intensity < 0 ] = 0. 
	
	if dust_filter:
		intensity = nd.median_filter(intensity,7)
		intensity = nd.gaussian_filter(intensity,2)
		intensity = nd.median_filter(intensity,7)

	return intensity

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
#	 rs = np.unique(rk)
#	 radial_profile = quick_count(rs,rk,dk)
	if method == 'mean':
		rs,radial_profile_ = bin_count_mean(rk,dk,dx)
	elif method in ['var','norm_var']:
		rs,radial_profile_ = bin_count_var(rk,dk,dx)
		if method == 'norm_var':
			radial_profile_ = np.sqrt(radial_profile_)/(bin_count_mean(rk,dk,dx)[1] + 1)
	return rs, radial_profile_

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
		
		fit_extent = 5
		x,y = radial_profile(viewer.layers['scattering'].data, com+theta, fit_extent/(prefs['calibration']/1000.), prefs['bin_width']/(prefs['calibration']/1000.), method='norm_var')
		out = np.nanmean(y)
		print(out)
		# x,y = radial_profile(viewer.layers['scattering'].data, com+theta, prefs['fit_extent']/(prefs['calibration']/1000.), prefs['bin_width']/(prefs['calibration']/1000.), method='norm_var')
		# if prefs['locate_mode'] == 'Max':
		# 	out = y.max()
		# elif prefs['locate_mode'] == 'Mean':
		# 	out = y.mean()
		return out
	
	out = minimize(minfxn,np.array((0.,0.)),args=(com,),method='Nelder-Mead',options={'initial_simplex':np.array(((1.,1.),(0.,1),(1.,0)))})
	print(out)
	viewer.layers['com'].data[-1] += out.x
	viewer.layers['com'].refresh()


def alignment_upscaled_fft_phase(d1,d2):
	'''
	from: highfret

	Find the linear shift in an image in phase space - upscales to super-resolve shift - d1 into d2

	input:
		* d1 - image 1 (Lx,Ly)
		* d2 - image 2 (Lx,Ly)
	output:
		* polynomial coefficients (K=1) with linear shift
	'''

	## Calculate Cross-correlation
	f1 = np.fft.fft2(d1-d1.mean())
	f2 = np.fft.fft2(d2-d2.mean())
	f = np.conjugate(f1)*f2 # / (np.abs(f1)*np.abs(f2))
	d = np.fft.ifft2(f).real

	## Find maximum in c.c.
	s1,s2 = np.nonzero(d==d.max())

	#### Interpolate for finer resolution
	## cut out region
	l = 5.
	xmin = int(np.max((0.,s1[0]-l)))
	xmax = int(np.min((d.shape[0],s1[0]+l)))
	ymin = int(np.max((0.,s2[0]-l)))
	ymax = int(np.min((d.shape[1],s2[0]+l)))
	dd = d[xmin:xmax,ymin:ymax]

	## calculate interpolation
	x = np.arange(xmin,xmax)
	y = np.arange(ymin,ymax)
	interp = RectBivariateSpline(x,y,dd)

	## interpolate new grid
	x = np.linspace(xmin,xmax,(xmax-xmin)*10)
	y = np.linspace(ymin,ymax,(ymax-ymin)*10)
	di = interp(x,y)
	# return di

	## get maximum in interpolated c.c. for sub-pixel resolution
	xy = np.nonzero(di==di.max())

	## get interpolated shifts
	dx_12 = x[xy[0][0]]
	dy_12 = y[xy[1][0]]
	if dx_12 > d.shape[0]/2:
		dx_12 -= d.shape[0]
	if dy_12 > d.shape[1]/2:
		dy_12 -= d.shape[1]

	return np.array((dx_12,dy_12))
	
def guess_com():
	global viewer
	prefs = get_prefs()
	com = viewer.layers['com'].data[-1].astype('int')
	d = viewer.layers['scattering'].data

	dd = np.zeros((d.shape[0]*3,d.shape[1]*3)) ## pad out neighboring cells
	dd[d.shape[0]:d.shape[0]*2,d.shape[1]:d.shape[1]*2] = d

	center = com.astype('double') + 0.
	cutoff = (prefs['dishdiameter']/2.+1.)/(prefs['calibration']/1000.)
	x,y = np.indices((dd.shape))
	r = np.sqrt((x.astype('float') - float((0.5+center[0])//1))**2 + (y.astype('float') - float((0.5+center[1])//1))**2)
	keep = r <= cutoff
	ddd = keep.astype('float')

	shift = alignment_upscaled_fft_phase(dd,ddd)
	com = com.astype("double") - shift
	com -= np.array(d.shape).astype('double')
	com = com.astype('int')
	print('Guesses COM:',com)
	return com

def get_prefs():
	global viewer,dock_qcquant
	prefs = dock_qcquant.container.asdict()
	return prefs 

def contrast(layername='scattering'):
	global viewer
	dmin = viewer.layers[layername].data.min()
	dmax = viewer.layers[layername].data.max()
	delta = (dmax-dmin)*1.
	viewer.layers[layername].contrast_limits_range = (dmin-delta,dmax+delta)
	viewer.layers[layername].contrast_limits = (dmin,dmax)

def com_to_top():
	global viewer
	if 'com' in viewer.layers:
		viewer.layers.move(viewer.layers.index('com'),-1)

def new_com(data):
	global viewer
	ls = [layer for layer in viewer.layers if layer.name == 'com']
	if len(ls) == 0:
		com = np.array([data.shape[0]/2.,data.shape[1]/2.])
		viewer.add_points(com,name='com',face_color='red',edge_color='darkred')

def find_com():
	com = guess_com()
	viewer.layers['com'].data[-1] = com
	viewer.layers['com'].refresh()

def save_radial():
	global viewer
	if dock_qcquant.x is None or dock_qcquant.y is None or dock_qcquant.y2 is None:
		return
		
	prefs = get_prefs()
	fname = QFileDialog.getSaveFileName(parent=dock_qcquant, caption="Save Radial Analysis")[0]
	if not fname == "":
		np.savetxt(fname+'.txt',np.array((dock_qcquant.x,dock_qcquant.y,dock_qcquant.y2)))
		print('Saved:',fname+'.txt')
		
		save_img(fname)
		print('Saved:',fname+'.png')

def save_img(fname):
	global viewer
	prefs = get_prefs()
	com = viewer.layers['com'].data[-1].astype('int')
	d = viewer.layers['scattering'].data
	
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
	global viewer,dock_qcquant
	prefs = get_prefs()
	ls = [layer for layer in viewer.layers if layer.name == 'com']
	if len(ls) == 0:
		print('need to get a com layer')
		return

	com = viewer.layers['com'].data[-1]

	x,y = radial_profile(viewer.layers['scattering'].data, com,prefs['extent_factor']/(prefs['calibration']/1000.),prefs['bin_width']/(prefs['calibration']/1000.))
	x *= prefs['calibration']/1000.
	y2 = nd.gaussian_filter1d(y,prefs['smooth_kernel'])
	
	dock_qcquant.x = x
	dock_qcquant.y = y
	dock_qcquant.y2 = y2

	dock_qcquant.ax.cla()
	dock_qcquant.ax.plot(x,y,color='k',lw=1.2)
	dock_qcquant.ax.plot(x,y2,color='r',alpha=.8,lw=.8)
	dock_qcquant.ax.set_xlim(0,x.max())
	dock_qcquant.ax.set_ylim(0.,1.)
	dock_qcquant.ax.set_xlabel('Radial Distance (mm)')
	dock_qcquant.ax.set_ylabel('Scattering')
	dock_qcquant.fig.subplots_adjust(left=.2,bottom=.2)
	dock_qcquant.canvas.draw()
	dock_qcquant.raise_()

def fxn_load_data_epi():
	global viewer
	prefs = get_prefs()
	data,smax = load_tif(prefs['scattering'])
	### Coming from a biorad gel doc, images seem to be inverted (both epi and trans) (i.e., so that you get positive bands)
	## this is borderline criminal.... so flip it back
	data = smax - data
	scattering = calculate_intensity(data,smax,mode='epi',dust_filter=True)
	if 'scattering' in viewer.layers:
		viewer.layers.remove('scattering')
	viewer.add_image(scattering,name='scattering')
	com_to_top()
	contrast('scattering')
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
	# print(factor)
	# print(prefs['dishdiameter'])
	# print(dock_qcquant.container['calibration'].value)
	
	dock_qcquant.container['calibration'].value = factor
	# print(factor)
	# viewer.window._dock_widgets[__plugin_name__].container.calibration.value = factor



def fxn_showdistance():
	global viewer,dock_qcquant
	prefs = get_prefs()
	state = prefs['showdistance']
	
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

	if state:
		dock_qcquant._timer_distance.timeout.connect(lambda : update_dcom())
		dock_qcquant._timer_distance.start()
	else:
		dock_qcquant._timer_distance.stop()
		dock_qcquant._timer_distance.disconnect()
		dock_qcquant.container['showdistance'].text = 'Distance from COM: '


def initialize_qcquant_dock():
	global viewer

	w_data = widgets.FileEdit(mode='r',label='Data File',name='scattering')
	b_load_data_epi = widgets.PushButton(text='Load Data (Epi-illumination)')
	w_dish_diameter = widgets.FloatSpinBox(value=100.0,label='Dish O.D. (mm)',min=0,max=1000.,name='dishdiameter')
	b_calc_conversion = widgets.PushButton(text='Calculate conversion (circle)')
	w_calibration = widgets.FloatSpinBox(value=92.2,label='Calibration (um/px)',min=0,name='calibration')
	w_extentfactor = widgets.FloatSpinBox(value=50.,label='Extent (mm)',min=0,name='extent_factor')
	w_binwidth = widgets.FloatSpinBox(value=.05,label='Radial Bin Width (mm)',min=0.,name='bin_width',step=.001)
	w_smoothkernel = widgets.FloatSpinBox(value=2.,label='Smoothing Kernel (bins)',min=0.,name='smooth_kernel')
	w_showdistance = widgets.CheckBox(text='Display distance from COM:',value=False,name='showdistance')
	b_locate = widgets.PushButton(text='Locate Plate Center of Mass (COM)')
	b_radial = widgets.PushButton(text='Calculate Radial Average')
	b_fit = widgets.PushButton(text='Locate Local Symmetric COM')

	container = widgets.Container(widgets=[
		w_data,
		b_load_data_epi,
		w_dish_diameter,
		w_calibration,
		b_calc_conversion,
		b_locate,
		b_fit,
		w_showdistance,
		w_extentfactor,
		w_binwidth,
		w_smoothkernel,
		b_radial,
	])

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
	vb.addWidget(container.native)
	vb.addWidget(canvas)
	vb.addWidget(toolbar)
	vb.addWidget(b2)
	vb.addWidget(b)
	vb.addStretch()
	qw.setLayout(vb)

	dock = viewer.window.add_dock_widget(qw,name=__plugin_name__)
	dock.container = container
	dock._timer_distance = QTimer()
	dock._timer_distance.setInterval(int(1000//25.))

	dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
	dock.fig = fig
	dock.ax = ax
	dock.canvas = canvas
	dock.toolbar = toolbar
	
	dock.x = None
	dock.y = None
	dock.y2 = None
	
	b_locate.clicked.connect(lambda e: find_com())
	b_radial.clicked.connect(lambda e: fxn_radial())
	b_load_data_epi.clicked.connect(lambda e: fxn_load_data_epi())
	b_calc_conversion.clicked.connect(lambda e: fxn_calc_conversion())
	b_fit.clicked.connect(lambda e: radial_adjust())
	w_showdistance.clicked.connect(lambda e: fxn_showdistance())
	return dock
	
def run_app():
	global viewer,dock_qcquant

	viewer = napari.Viewer()
	dock_qcquant = initialize_qcquant_dock()
	# viewer.window._qt_window.tabifyDockWidget(dock_qcquant,dock_profile)
	dock_qcquant.raise_()
	napari.run()
	
if __name__ == "__main__":
	run_app()