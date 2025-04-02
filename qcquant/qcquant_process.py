import tifffile
import numba as nb
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline


def load_tif(image_path,dust_filter=True):
	### Coming from a biorad gel doc, images seem to be inverted (both epi and trans) (i.e., so that you get positive bands)
	## this is borderline criminal.... so flip it back

	z = tifffile.imread(image_path).astype('int')
	print('Loaded %s'%(image_path),z.dtype,z.shape)
	smax = 0
	with tifffile.TiffFile(image_path) as tif:
		# ########
		# #### USEFUL FOR DEBUGGING FILES
		# for key in tif.pages[0].tags.keys():
		# 	 print(key,tif.pages[0].tags[key].value)
		# print(tif.pages[0].tags['PhotometricInterpretation'].value)
		# # print(tif.pages[0].tags['SMaxSampleValue'].value)
		# ########


		if 51123 in tif.pages[0].tags: ## it's micromanager!
			bit_depth = tif.pages[0].tags[51123].value['BitDepth']
			smax = 2**bit_depth-1
		elif tif.pages[0].tags[256].value == 1440 and tif.pages[0].tags[257].value == 1080 and tif.pages[0].tags[270].value.startswith('ImageJ'): ## it's our thorlabs CS165M...
			smax = 2**10-1
		else:
			smax = int(tif.pages[0].tags['SMaxSampleValue'].value)
		
		print(str(tif.pages[0].tags['PhotometricInterpretation'].value))
		print('%d-bit image; Maximum counts is %d'%(int(np.log2(smax+1)),smax))
		if str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISBLACK' or tif.pages[0].tags['PhotometricInterpretation'].value == 1:
			print('Regular image')
		elif str(tif.pages[0].tags['PhotometricInterpretation'].value) == 'PHOTOMETRIC.MINISWHITE' or tif.pages[0].tags['PhotometricInterpretation'].value == 0:
			print('Inverted image')
		else:
			raise Exception('Cannot interpret photometric approach')
		

	if z.ndim == 3:
		# intensity = (float(smax)-z.astype('float').mean(0)) / float(smax)
		intensity = (float(smax)-z.astype('float')) / float(smax)
	else:
		# intensity = (float(smax)-z.astype('float')) / float(smax)
		intensity = (float(smax)-z.astype('float')[None,:,:]) / float(smax)
	
	## don't play around
	intensity[intensity > 1.] = 1.
	intensity[intensity < 0 ] = 0. 
	
	# if dust_filter:
	# 	intensity = nd.median_filter(intensity,7)
	# 	intensity = nd.gaussian_filter(intensity,2)
	# 	intensity = nd.median_filter(intensity,7)
	return intensity

	# intensity = nd.median_filter(intensity,5)
	# bg = np.fft.fft2(intensity)
	# bg[:3,:] *= 0
	# bg[-3:,:] *= 0
	# bg[:,:3] *= 0
	# bg[:,-3:] *= 0
	# # bg[1,1:] *= 0
	# intensity = np.fft.ifft2(bg).real
	# # intensity = intensity - bg
	# return intensity
	# return np.log(np.abs(bg).real)

	# from .fast_median import median_scmos
	# intensity = median_scmos((intensity[None,:,:]*4095).astype('uint16'),21,12,True)[0].astype('double') / 4095.
	# intensity[intensity > 1.] = 1.
	# intensity[intensity < 0 ] = 0. 
	return intensity

@nb.njit(cache=True)
def erode_nans(d0):
	d = np.copy(d0)
	nx,ny = d.shape

	nanmask = np.isnan(d)
	nmx,nmy = np.nonzero(nanmask)

	total = nmx.size
	while total > 0:
		## erode NaNs
		neighbors = np.zeros((3,3))
		for nmi in range(nmx.size):
			neighbors += np.nan
			xi = nmx[nmi]
			yi = nmy[nmi]
			for ii in [-1,0,1]:
				for jj in [-1,0,1]:
					## include the center b/c it could have contributions from nearest-neighbors?
					x = xi + ii
					y = yi + jj

					## wrap left/right b/c these are angular distributions
					if y < 0:
						y = ny-1
					elif y >= ny:
						y = 0
					## don't wrap up down b/c these are distances
					if x >= 0 and x < nx:
						neighbors[ii+1,jj+1] = d[x,y]
			d[xi,yi] = np.nanmean(neighbors)
		
		## check how many remain
		total = 0
		for nmi in range(nmx.size):
			if np.isnan(d[nmx[nmi],nmy[nmi]]):
				total += 1
	return d

@nb.njit(cache=True)
def histrphi(d,com,nr,nphi,rmin,rmax):
	dr = (rmax-rmin)/float(nr)
	dphi = 2*np.pi/float(nphi)

	hist_total = np.zeros((nr,nphi))
	hist_n = np.zeros((nr,nphi))
	hist = np.zeros((nr,nphi)) + np.nan

	for i in range(d.shape[0]):
		for j in range(d.shape[1]):
			x = float(i) - float((0.5+com[0])//1)
			y = float(j) - float((0.5+com[1])//1)
			r = np.sqrt(x*x+y*y)

			if r < rmax and r>=rmin:
				phi = np.arctan2(y,x) + np.pi
				if phi >= 2*np.pi: ## protect against weird wraps...
					phi = 0.
				ind_r = int((r-rmin)//dr)
				ind_phi = int(phi//dphi)

				#### DEBUG....
				if ind_r >= nr:
					print(r,nr,dr)
					raise Exception('OUT OF BOUNDS-r')
				if ind_phi >= nphi:
					print(phi,nphi,dphi)
					raise Exception('OUT OF BOUNDS-phi')

				hist_total[ind_r,ind_phi] += d[i,j]
				hist_n[ind_r,ind_phi] += 1.

	for i in range(nr):
		for j in range(nphi):
			if hist_n[i,j] > 0:
				hist[i,j] = hist_total[i,j] / hist_n[i,j]
	return hist


def moments(data,com,cutoff):
	## Calculate ellipse around COM using low-order moments
	## Reference: Teague, MR (1980). Image Analysis via the General Theory of Moments. J. Opt. Soc. Am. 70(8), 920.
	## xbar, ybar are centroid coordinates relative to COM
	## a, b are major and minor ellipse radii
	## phi is angle CCW from x-axis (x is dimension 0)

	x,y = np.indices((data.shape))

	keep = np.bitwise_and(np.abs(x-com[0])<cutoff,np.abs(y-com[1])<cutoff)
	dk = data[keep].astype('float')
	xk = x[keep].astype('float') - com[0]
	yk = y[keep].astype('float') - com[1]

	## Eqns. 1-6
	mu00 = np.sum(dk)
	mu10 = np.sum(xk*dk)
	mu01 = np.sum(yk*dk)
	mu20 = np.sum(xk**2.*dk)
	mu11 = np.sum(xk*yk*dk)
	mu02 = np.sum(yk**2.*dk)

	## in text above Eqn. 4
	xbar = mu10/mu00
	ybar = mu01/mu00

	## Eqns. 7-9
	a = np.sqrt((mu20+mu02+np.sqrt((mu20-mu02)**2.+4.*mu11**2.))/(.5*mu00))
	b = np.sqrt((mu20+mu02-np.sqrt((mu20-mu02)**2.+4.*mu11**2.))/(.5*mu00))
	phi = .5*np.arctan(2.*mu11/(mu20-mu02)) ## in radians

	return xbar,ybar,a,b,phi

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

def guess_plate_com(d,com_guess,dishdiameter,calibration):
	'''
	Find the linear shift to where the image aligns with a circle the theoretical size of a petri dish

	d is 2D np.ndarray of data
	dishdiamter is ~100 mm for a standard petri dish
	calibration is um/px (i.e, the size of a pixel in microns)
	com_guess should be in pixels
	'''

	dd = np.zeros((d.shape[0]*3,d.shape[1]*3)) ## pad out neighboring cells
	dd[d.shape[0]:d.shape[0]*2,d.shape[1]:d.shape[1]*2] = d

	com = com_guess.astype('double')
	cutoff = (dishdiameter/2.+1.)/(calibration/1000.)
	x,y = np.indices((dd.shape))
	r = np.sqrt((x.astype('float') - float((0.5+com[0])//1))**2 + (y.astype('float') - float((0.5+com[1])//1))**2)
	keep = r <= cutoff
	ddd = keep.astype('float')

	shift = alignment_upscaled_fft_phase(dd,ddd)
	com -= shift
	com -= np.array(d.shape).astype('double')
	com = com.astype('int')
	print('Guesses COM:',com)

	return com

def plot_centered_plate_img(d,com,dishdiameter,calibration):
	'''
	d is 2D np.ndarray of data
	dishdiamter is ~100 mm for a standard petri dish
	calibration is um/px (i.e, the size of a pixel in microns)
	com_guess should be in pixels
	'''
	
	nx,ny = d.shape
	r = dishdiameter/2. + 1.
	r /= calibration/1000.
	r = int((r+1)//1)
	
	out = np.zeros((r*2+1,r*2+1))
	
	x0 = int(np.max((0,com[0]-r)))
	x1 = int(np.min((nx,com[0]+r+1)))
	y0 = int(np.max((0,com[1]-r)))
	y1 = int(np.min((ny,com[1]+r+1)))
	new_x0 = int(r - (com[0] - x0))
	new_y0 = int(r - (com[1] - y0))
	
	dd = d[x0:x1,y0:y1]
	out[new_x0:new_x0 + dd.shape[0], new_y0:new_y0 + dd.shape[1]] = dd
	
	fig,ax = plt.subplots(1,figsize=(4,4),dpi=400)
	ax.imshow(out,origin='lower',interpolation='bicubic',cmap='Greys_r',vmin=dd.min(),vmax=dd.max())
	ax.axis('off')
	fig.subplots_adjust(left=0,right=1.,top=1.,bottom=0.)
	return fig,ax

def plot_polar(fig,ax,dist,r,phi):
	ax[0].cla()
	ax[0].imshow(dist,extent=[phi.min(),phi.max(),r.max(),r.min()], aspect='auto',interpolation='nearest')
	ax[0].set_xlabel(r'$\phi$ (deg)')
	ax[0].set_ylabel(r'$r$ (mm)')

	ax[1].cla()
	ax[1].plot(r, np.nanmean(dist,1), color='tab:orange',zorder=5)
	ylim = ax[1].get_ylim()
	for i in range(phi.size):
		ax[1].plot(r, dist[:,i], alpha=.1, color='k')
	ax[1].set_xlabel(r'$r$ (mm)')
	ax[1].set_ylabel('Scattering (a.u.)')
	ax[1].set_xlim(r.min(),r.max())
	ax[1].set_ylim(*ylim)
	
	fig.subplots_adjust(left=.2,bottom=.08,top=.98,right=.95,hspace=.2)