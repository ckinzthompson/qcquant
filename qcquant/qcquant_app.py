import os
import napari
import numpy as np
from magicgui import widgets
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas,NavigationToolbar2QT
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QFileDialog,QSizePolicy,QDockWidget
from .qcquant_process import load_tif_biorad, histrphi, guess_plate_com, plot_centered_plate_img, plot_polar
from . import __title__,__version__

viewer = None       ## Napari viewer
dock_qcquant = None ## QCQuant config dock

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
		viewer.layers.selection.active = viewer.layers['com']

def new_com(data):
	global viewer
	ls = [layer for layer in viewer.layers if layer.name == 'com']
	if len(ls) == 0:
		com = np.array([data.shape[0]/2.,data.shape[1]/2.])
		viewer.add_points(com,name='com',face_color='red',border_color='darkred')
	com_to_top()

def fxn_load(event=None):
	global viewer
	
	prefs = get_prefs()
	scattering = load_tif_biorad(prefs['scattering'])
	if 'scattering' in viewer.layers:
		viewer.layers.remove('scattering')
	viewer.add_image(scattering,name='scattering')

	com_to_top()
	contrast('scattering')
	new_com(scattering)
	

def fxn_conversion(event=None):
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

def fxn_locate(event=None):
	global viewer
	prefs = get_prefs()
	
	old_com = viewer.layers['com'].data[-1].astype('double')
	new_com = guess_plate_com(viewer.layers['scattering'].data, old_com, prefs['dishdiameter'], prefs['calibration'])
	viewer.layers['com'].data[-1] = new_com
	viewer.layers['com'].refresh()

def fxn_distribution(event=None):
	global viewer,dock_qcquant

	prefs = get_prefs()
	ls = [layer for layer in viewer.layers if layer.name == 'com']
	if len(ls) == 0:
		print('need to get a com layer')
		dock_qcquant._timer.stop()
		dock_qcquant._timer.disconnect()
		return

	com = viewer.layers['com'].data[-1].astype('double')

	rmin = prefs['rmin'] / (prefs['calibration']/1000.)
	rmax = prefs['rmax'] / (prefs['calibration']/1000.)
	nr = prefs['nr']
	nphi = prefs['nphi']
	dist = histrphi(viewer.layers['scattering'].data, com, nr, nphi, rmin, rmax)

	dock_qcquant.dist = dist
	dock_qcquant.r = np.linspace(prefs['rmin'],prefs['rmax'],nr)
	dock_qcquant.phi = np.linspace(np.rad2deg(-np.pi),np.rad2deg(np.pi),nphi)

	plot_polar(dock_qcquant.fig,dock_qcquant.ax,dock_qcquant.dist,dock_qcquant.r,dock_qcquant.phi)
	dock_qcquant.canvas.draw()
	dock_qcquant.raise_()

def fxn_live(event=None):
	global viewer,dock_qcquant

	if not dock_qcquant._timer.isActive():
		dock_qcquant.container[11].text = 'Live (Stop)'
		dock_qcquant._timer.start()
	else:
		dock_qcquant.container[11].text = 'Live (Start)'
		dock_qcquant._timer.stop()

def fxn_save(event=None):
	global viewer
	if dock_qcquant.dist is None or dock_qcquant.r is None or dock_qcquant.phi is None:
		return
		
	prefs = get_prefs()
	# fname = QFileDialog.getSaveFileName(parent=dock_qcquant, caption="Save Radial Analysis")[0]
	# if not fname == "":
	fpath = dock_qcquant.container[7].value
	fdir = os.path.dirname(fpath)
	fname = os.path.splitext(os.path.basename(fpath))[0]
	opath = os.path.join(fdir,f'qcquant_results')
	if not os.path.exists(opath):
		os.mkdir(opath)

	np.savetxt(os.path.join(opath,f'{fname}.dist.txt'),dock_qcquant.dist)
	np.savetxt(os.path.join(opath,f'{fname}.r.txt'),dock_qcquant.r)
	np.savetxt(os.path.join(opath,f'{fname}.phi.txt'),dock_qcquant.phi)
	
	fig,ax = plot_centered_plate_img(viewer.layers['scattering'].data, viewer.layers['com'].data[-1].astype('double'), prefs['dishdiameter'], prefs['calibration'])
	fig.savefig(os.path.join(opath,f'{fname}.plate.png'))
	plt.close(fig)
	print('Saved:',os.path.join(opath,f'{fname}.plate.png'))

	dock_qcquant.fig.savefig(os.path.join(opath,f'{fname}.dist.pdf'))
	print('Saved:',os.path.join(opath,f'{fname}.dist.pdf'))
	dock_qcquant.fig.savefig(os.path.join(opath,f'{fname}.dist.png'),dpi=300)
	print('Saved:',os.path.join(opath,f'{fname}.dist.png'))

def fxn_invert(event=None):
	global viewer
	if 'scattering' in viewer.layers:
		viewer.layers['scattering'].data = 1. - viewer.layers['scattering'].data

def initialize_qcquant_dock():
	global viewer

	w_data = widgets.FileEdit(mode='r',label='Data File',name='scattering')
	
	w_dish_diameter = widgets.FloatSpinBox(value=100.0,label='Dish O.D. (mm)',min=0,max=1000.,name='dishdiameter')
	w_calibration = widgets.FloatSpinBox(value=92.2,label='Calibration (um/px)',min=0,name='calibration')
	w_phi= widgets.SpinBox(value=128,label='Num. Bins: phi',min=1,name='nphi')
	w_nr= widgets.SpinBox(value=128,label='Num. Bins: r',min=1,name='nr')
	w_rmin = widgets.FloatSpinBox(value=0.0,label='Minimum r (mm)',min=0.,name='rmin')
	w_rmax = widgets.FloatSpinBox(value=50.,label='Minimum r (mm)',min=1.,name='rmax')

	b_conversion = widgets.PushButton(text='Calculate Conversion')
	b_locate = widgets.PushButton(text='Guess Plate COM')
	b_live = widgets.PushButton(text='Live (Start)')
	b_distribution = widgets.PushButton(text='Calc Polar Distribution')
	b_save = widgets.PushButton(text='Save Distribution')
	b_invert = widgets.PushButton(text='Invert Image')

	b_conversion.clicked.connect(fxn_conversion)
	b_locate.clicked.connect(fxn_locate)
	b_live.clicked.connect(fxn_live)
	b_distribution.clicked.connect(fxn_distribution)
	b_save.clicked.connect(fxn_save)
	b_invert.clicked.connect(fxn_invert)

	w_data.changed.connect(lambda widget: fxn_load())

	fig,ax = plt.subplots(2,figsize=(4,6))
	canvas = FigureCanvas(fig)
	toolbar = NavigationToolbar2QT(canvas)
	# canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
	canvas.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

	# container = widgets.Container(widgets=[w_dish_diameter,w_calibration,w_nr,w_phi,w_rmin,w_rmax, widgets.Label(label=""), w_data, b_conversion,b_locate,b_distribution,b_live,b_save])
	container = widgets.Container(widgets=[
		w_dish_diameter,
		w_calibration,
		w_nr,
		w_phi,
		w_rmin,
		w_rmax,
		widgets.Label(label=""),
		w_data,
		b_conversion,
		b_locate,
		b_distribution,
		b_live,
		b_save,
		b_invert
	])

	qw = QWidget()
	vb = QVBoxLayout()
	vb.addWidget(container.native)
	vb.addWidget(canvas)
	vb.addWidget(toolbar)
	vb.addStretch()
	qw.setLayout(vb)

	dock = viewer.window.add_dock_widget(qw,name=f'{__title__} {__version__}')
	dock.container = container
	n_updates_sec = 3
	dock._timer = QTimer()
	dock._timer.timeout.connect(fxn_distribution)
	dock._timer.setInterval(int(1000./float(n_updates_sec)))

	dock.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
	dock.fig = fig
	dock.ax = ax
	dock.canvas = canvas
	dock.toolbar = toolbar
	
	dock.dist = None
	dock.r = None
	dock.phi = None
	
	return dock
	
def run_app():
	global viewer,dock_qcquant

	viewer = napari.Viewer()
	dock_qcquant = initialize_qcquant_dock()
	dock_qcquant.raise_()
	napari.run()