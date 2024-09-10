from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import os

import emcee
import corner


def collect_data(prefix,fdir,suffix='.txt'):
	#### File naming should be '<prefix><conc>.txt'
	#### Data should already be normalized by Imax=4095
	#### concs should be *1000 in filename to avoid decimals.
	#### concs are converted here 

	datas = []
	concs = []

	for fn in os.listdir(fdir):
		if fn.startswith(prefix) and fn.endswith(suffix):
			x,d,_ = np.loadtxt(os.path.join(fdir,fn))
			conc = int(fn[len(prefix):-4])
			concs.append(conc)
			datas.append([x,d])

	concs = np.array(concs).astype('double') / 1000.
	datas = np.array(datas).astype('double')
	order = np.argsort(concs)
	concs = concs[order]
	datas = datas[order]
	return concs,datas

def process_data(concs,datas,low,high):
	avgs = np.zeros(concs.size)

	fig,ax = plt.subplots(1,figsize=(6,4),sharex=True)
	for i in range(concs.size):
		ax.plot(datas[i][0], datas[i][1], label='%.2f'%(concs[i]))
		keep = np.bitwise_and(datas[i][0]>low, datas[i][0] < high)
		avgs[i] = datas[i][1][keep].mean()
		ax.axhline(y=avgs[i],color=ax.lines[-1].get_color(),ls='--')

	plt.legend(loc=1,title='[Agarose]\n(% w/v)')

	ax.set_ylim(0,1.02)
	ax.set_xlim(0,50.)

	ax.axvspan(0,low,color='k',alpha=.3)
	ax.axvspan(high,50.,color='k',alpha=.3)
	ax.axvline(x=low,color='k')
	ax.axvline(x=high,color='k')

	ax.set_title('Epi')
	ax.set_ylabel(r'$I/I_{max}$')
	ax.set_xlabel('Distance (mm)')
	plt.show()

	return avgs,fig,ax

def fxn_S(N,I0f,I0B,sigmaL):
	Imax = 1. ## already normalized
	S = I0f*(1-np.exp(-sigmaL*N))+I0B
	S[S>1] = 1.
	return S

def lnL(theta,N,S):
	## need to rescale I/Imax to something > 1 b/c otherwise Poisson variances make absolutely no sense
	## solution: just convert back to camera counts.... by multiplying by 4095
	## Likelihood is Gaussian approx to Poisson

	## DU to photons --> 1 DU ~ 10 photons?? tbf there is camera noise that probably dominates here 
	camera_max = 4095. * 5.
	Sq = fxn_S(N,*theta)+(1./4095) ## don't want 0 variance, so min at 1 DU
	lnLs = -.5*(np.log(2.*np.pi*Sq)+np.log(camera_max)) -.5/Sq*(S-Sq)**2. * camera_max
	lnLs = np.sum(lnLs)
	if np.isnan(lnLs):
		return -np.inf
	return lnLs

def minfxn(theta,N,S):
	## MLE
	if np.any(theta < 0):
		return np.inf
	return -lnL(theta,N,S)

def attenuance(y,theta):
	I0f,I0B,sigmaL = theta
	A = np.log(I0f/((I0f+I0B)-y))

	Imax = 1.
	Nmax = 1/sigmaL*np.log(I0f/((I0f+I0B)-Imax))
	Amax = Nmax*sigmaL

	A[A<0] = 0
	A[A>Amax] = Amax
	
	return A

def get_mcmc_samples(concs,avgs):

	x = concs
	y = avgs

	nwalkers = 100
	ndim = 3
	guess = np.array(([1.,y[0],(y[1]-y[0])/(x[1]-x[0])]))
	p0 = guess[None,:] + np.random.randn(nwalkers,ndim)*1e-6
	p0[p0 < 0] *= -1.
	# p0[:,:2][p0[:,:2]< 0] *= -1.

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL, args=[x,y])
	state = sampler.run_mcmc(p0, 1000, progress=True)
	keep = state[1].argsort()[nwalkers//2:]
	p1 = state[0][keep]
	nwalkers = nwalkers//2
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnL, args=[x,y])
	state = sampler.run_mcmc(p1,1000,progress=True)
	sampler.reset()
	state = sampler.run_mcmc(state[0],10000,progress=True)

	fig, ax = plt.subplots(3, figsize=(10, 7), sharex=True)
	samples = sampler.get_chain()
	labels = ["$I_0f/I_{max}$", r"$I_0B/I_{max}$", r"$\sigma L$"]
	for i in range(ndim):
		ax[i].plot(samples[:, :, i], "k", alpha=0.05)
		ax[i].set_xlim(0, len(samples))
		ax[i].set_ylabel(labels[i])
		ax[i].yaxis.set_label_coords(-0.1, 0.5)
	ax[-1].set_xlabel("step number");
	
	# tau = sampler.get_autocorr_time()
	# print(tau)


	log_probs = sampler.get_log_prob(flat=True)
	samples = sampler.get_chain(flat=True)
	best = samples[log_probs.argmax()]

	flat_samples = sampler.get_chain(discard=0, thin=400, flat=True)

	###### Corner
	fig2 = plt.figure()
	corner.corner(flat_samples,
		fig=fig2,
		labels=labels,
		quantiles=[0.16, 0.5, 0.84],
		show_titles=True,
		title_kwargs={"fontsize": 12},
		title_fmt='.3f',
		labelpad=.15,
		hist_kwargs={'density':True}
	)
	ax2 = fig2.axes
	return flat_samples,best,fig,ax,fig2,ax2

def calibrate_mcmc(concs,avgs):
	'''
	x should be agarose concentration %
	y should be average signal, normalized by I_max (i.e., 4095)
	'''

	x = concs.copy()
	y = avgs.copy()

	xx = np.linspace(0,x.max(),1000)

	samples,best,fig_traj,ax_traj,fig_corner,ax_corner = get_mcmc_samples(x,y)
	theta = np.mean(samples,0)
	# theta = best
	std = np.std(samples,0)
	# for i in range(3):
	# 	xlim = ax_corner[3*i+i].get_xlim()
	# 	xax = np.linspace(xlim[0],xlim[1],1000)
	# 	ax_corner[3*i+i].plot(xax,normal(xax,theta[i],std[i]),color='tab:orange')
	
	yys,aas = thetas_2_curves(xx,samples)

	fig_cal,ax_cal = plot_calibration(x,y,xx,theta,yys,aas)
	fig_samples,ax_samples = plot_marginal_samples(samples)

	print('MCMC\n========================')
	print('MAP:',theta)
	print('Sample Median:',theta)
	print('Sample Std:',std)

	return theta,std,fig_cal,ax_cal,fig_traj,ax_traj,fig_corner,ax_corner,fig_samples,ax_samples

def calibrate_laplace(concs,avgs):
	'''
	x should be agarose concentration %
	y should be average signal, normalized by I_max (i.e., 4095)
	'''

	x = concs.copy()
	y = avgs.copy()

	xx = np.linspace(0,x.max(),1000)

	guess = np.array(([1.,y[0],(y[1]-y[0])/(x[1]-x[0])]))
	out = minimize(minfxn,guess,args=(x,y),method='Nelder-Mead',options={'maxiter':10000})
	if not out.success:
		print(out)
		raise Exception('Fitting Failed')
	theta = out.x
	covar = calc_covar(theta,x,y)
	std = calc_std(theta,x,y)

	samples = np.random.multivariate_normal(mean=theta,cov=covar,size=1000)
	# theta = np.mean(samples,0)
	# std = np.std(samples,0)

	yys,aas = thetas_2_curves(xx,samples)
	fig_cal,ax_cal = plot_calibration(x,y,xx,theta,yys,aas)
	fig_samples,ax_samples = plot_marginal_samples(samples)

	print('Laplace Approximation\n========================')
	print('MAP:',theta)
	print('Std:',std)

	return theta,std,fig_cal,ax_cal,fig_samples,ax_samples

def thetas_2_curves(xx,samples):
	yys = np.zeros((samples.shape[0],xx.size))
	aas = np.zeros((samples.shape[0],xx.size))
	for i in range(samples.shape[0]):
		yy = fxn_S(xx,*samples[i])
		yys[i] = yy
		aas[i] = attenuance(yy,samples[i])

	return yys,aas

def normal(x,mu,sig):
	return 1./np.sqrt(2.*np.pi*sig**2.)*np.exp(-.5/(sig**2.)*(x-mu)**2.)

def plot_marginal_samples(samples):
	theta = np.mean(samples,0)
	std = np.std(samples,0)
	fig_samples,ax_samples = plt.subplots(1,3,figsize=(9,2.5))
	labels = ["$I_0f/I_{max}$", r"$I_0B/I_{max}$", r"$\sigma L$"]
	for i in range(3):
		hy,hx = ax_samples[i].hist(samples[:,i],bins=50,density=True)[:2]
		hx = np.linspace(hx.min(),hx.max(),1000)
		ax_samples[i].plot(hx,normal(hx,theta[i],std[i]),label='Laplace Approx')
		ax_samples[i].set_xlabel(labels[i])
		ax_samples[i].set_ylabel('Probability Density')
		# ax_samples[i].legend(loc=1)
	fig_samples.tight_layout()
	return fig_samples,ax_samples

def plot_calibration(x,y,xx,theta,yys,aas):

	lb1,med1,ub1 = np.percentile(yys,[16,50.,84.],axis=0)
	lb2,med2,ub2 = np.percentile(aas,[16.,50.,84.],axis=0)

	I0f,I0B,sigmaL = theta
	Imax = 1.
	Nmax = 1/sigmaL*np.log(I0f/((I0f+I0B)-Imax))
	A = np.log(I0f/((I0f+I0B)-y))
	Amax = Nmax*sigmaL
	ycurve = fxn_S(xx,*theta)
	Acurve = xx*sigmaL*(xx<Nmax) + Amax*(xx>=Nmax)

	fig,ax = plt.subplots(1,2,figsize=(12,4))
	
	ax[0].fill_between(xx,lb1,ub1,color='tab:blue',alpha=.3)
	ax[0].plot(x,y,'o',color='k')
	ax[0].plot(xx,ycurve,color='tab:blue',label='MAP')
	ax[0].axvline(Nmax,color='k')
	ax[0].axhline(Imax,color='gray',alpha=.5,ls='--')
	ax[0].set_ylabel(r'$I / I_{max}$')
	ax[0].set_xlabel('[Agarose] (% w/v)')
	ax[0].set_title(r'$I_0f/I_{max}=%.3f,\;I_0B/I_{max}=%.3f,\;\sigma L=%.3f\;\%%^{-1}$'%(I0f,I0B,sigmaL))
	ax[0].set_ylim(0,1.05)
	ax[0].set_xlim(0,x.max()*1.05)
	ax[0].legend(loc=2)

	ax[1].fill_between(xx,lb2,ub2,color='tab:blue',alpha=.3)
	ax[1].plot(x,A,'o',color='k')
	ax[1].plot(xx,Acurve,color='tab:blue',label='MAP')
	ax[1].axvline(Nmax,color='k')
	ax[1].axhline(Amax,color='gray',alpha=.5,ls='--')
	ax[1].set_ylabel('Sample-induced Attenuance')
	ax[1].set_xlabel('[Agarose] (% w/v)')
	ax[1].set_title('Linearized (Calibrated)')
	ax[1].set_ylim(0,Amax*1.05)
	ax[1].set_xlim(0,x.max()*1.05)
	ax[1].legend(loc=2)
	
	return fig,ax


#### FROM BIASD
def calc_hessian(fxn,x,eps = np.sqrt(np.finfo(np.float64).eps)):
	"""
	Calculate the Hessian using the finite difference approximation.

	Finite difference formulas given in Abramowitz & Stegun

		- Eqn. 25.3.24 (on-diagonal)
		- Eqn. 25.3.27 (off-diagonal)

	Input:
		* `fxn` is a function that can be evaluated at x
		* `x` is a 1D `np.ndarray`

	Returns:
		* an NxN `np.ndarray`, where N is the size of `x`

	"""

	# Notes:
	# xij is the position to evaluate the function at
	# yij is the function evaluated at xij
	#### if i or j = 0, it's the starting postion
	#### 1 or m1 are x + 1.*eps and x - 1.*eps, respetively


	h = np.zeros((x.size,x.size))
	y00 = fxn(x)

	for i in range(x.size):
		for j in range(x.size):
			#Off-diagonals below the diagonal are the same as those above.
			if j < i:
				h[i,j] = h[j,i]
			else:
				#On-diagonals
				if i == j:
					x10 = x.copy()
					xm10 = x.copy()
					x20 = x.copy()
					xm20 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x20[i] += 2*eps
					xm20[i] -= 2*eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y20 = fxn(x20)
					ym20 = fxn(xm20)

					h[i,j] = eps**(-2.)/12. * (-y20 + 16.* y10 - 30.*y00 +16.*ym10 - ym20)

				#Off-diagonals above the diagonal
				elif j > i:
					x10 = x.copy()
					xm10 = x.copy()
					x01 = x.copy()
					x0m1 = x.copy()
					x11 = x.copy()
					xm1m1 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x01[j] += eps
					x0m1[j] -= eps
					x11[i] += eps
					x11[j] += eps
					xm1m1[i] -= eps
					xm1m1[j] -= eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y01 = fxn(x01)
					y0m1 = fxn(x0m1)
					y11 = fxn(x11)
					ym1m1 = fxn(xm1m1)

					h[i,j] = -1./(2.*eps**2.) * (y10 + ym10 + y01 + y0m1 - 2.*y00 - y11 - ym1m1)
	return h

def calc_covar(theta,concs,avgs):
	#### FROM BIASD
	mu = theta.copy()
	feps = np.sqrt(np.finfo(np.float64).eps)
	feps *= 8. ## The interval is typically too small
	hessian = calc_hessian(lambda tt: lnL(tt,concs,avgs), mu,eps=feps)
	# print(hessian)
	#Ensure that the hessian is positive semi-definite by checking that all eigenvalues are positive
	#If not, expand the value of machine error in the hessian calculation and try again
	try:
		#Check eigenvalues, use pseudoinverse if ill-conditioned
		var = -np.linalg.inv(hessian)

		#Ensuring Hessian(variance) is stable
		new_feps = feps*2.
		new_hess = calc_hessian(lambda tt: lnL(tt,concs,avgs), mu,eps= new_feps)
		new_var = -np.linalg.inv(new_hess)
		it = 0

		while np.any(np.abs(new_var-var)/var > 1e-2):
			new_feps *= 2
			var = new_var.copy()
			new_hess = calc_hessian(lambda tt: lnL(tt,concs,avgs), mu,eps= new_feps)
			new_var = -np.linalg.inv(new_hess)
			it +=1
			# 2^26 times feps = 1. Arbitrary upper-limit, increase if necessary (probably not for BIASD)
			if it > 25:
				raise ValueError('Whelp, you hit the end there. bud')
		# print('Hessian iterations')
		# print(np.log2(new_feps/feps), it)

		#Ensure symmetry of covariance matrix if witin machine error
		if np.allclose(var,var.T):
			n = var.shape[0]
			var = np.tri(n,n,-1)*var+(np.tri(n,n)*var).T
			return var

	#If this didn't work, return None
	except np.linalg.LinAlgError:
		raise ValueError("Wasn't able to calculate the Hessian")

	raise ValueError("No estimate")

def calc_std(theta,concs,avgs):
	covar = calc_covar(theta,concs,avgs)
	std = np.diag(covar)**.5
	return std

# if __name__ == '__main__':
# 	## args
# 	prefix = 'epi_'
# 	fdir = '/Users/colin/Desktop/projects/qcquant/example_data/epi_vs_trans/'
# 	low = 5.
# 	high = 35.

# 	concs,datas = collect_data(prefix,fdir)
# 	avgs,fig1,ax1 = process_data(concs,datas,low,high)
# 	theta,fig2,ax2 = calibrate(concs,avgs)

# 	np.save('theta.npy',theta)
# 	fig1.savefig('calibration_preparation.pdf')
# 	fig1.savefig('calibration_preparation.png')
# 	fig2.savefig('calibration_linearization.pdf')
# 	fig2.savefig('calibration_linearization.png')