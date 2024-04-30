import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from scipy.optimize import minimize

def load(fn):
	x,d,_ = np.loadtxt(fn)
	return x,d

def _half_integral(x,a,b,c,d):
	'''
	integral of exp[+a(x-b)](1-1/(1+exp[-c(x-d)]))
	'''
	return np.exp(a*(x-b)) / a * hyp2f1(1.,a/c,(c+a)/c,-np.exp(c*(x-d)))
def integral(L,H,kg,kc,kd,xg,xd):
	return (_half_integral(H,-kg,xg,kd,xd) + _half_integral(H,kc,xg,kd,xd)) - (_half_integral(L,-kg,xg,kd,xd) + _half_integral(L,kc,xg,kd,xd))

def model(x,theta):
	rho0,b,kg,kc,kd,xg,xd = theta
	return rho0*(np.exp(-kg*(x-xg))+np.exp(kc*(x-xg)))*(1.-1./(1.+np.exp(-kd*(x-xd)))) + b
def x_min(theta):
	rho0,b,kg,kc,kd,xg,xd = theta
	return xg + 1./(kc+kg)*np.log(kg/kc)
def x_max(theta):
	''' if Eg~0 b/c x >> xg'''
	rho0,b,kg,kc,kd,xg,xd = theta
	return xd - 1./kd*np.log((kd-kc)/kc)

def _riemann(x,d,x0,x1,b=0):
	## linear interpolated center Riemann sum with baseline removed
	keep = np.bitwise_and(x>=x0,x<=x1)
	integral = np.sum((.5*(d[keep][1:]+d[keep][:-1])-b) * (x[keep][1:]-x[keep][:-1]))
	return integral

def naive(x,d,xl=None,xr=None):
	if xl is None:
		xl = x[0]
	if xr is None:
		xr = x[-1]
	keep = np.bitwise_and(x>=xl,x<=xr)

	b = d[keep].min() ####

	xx = x[keep]
	cc = (d[keep]-b).cumsum()

	lower = xx[np.argmax(cc>0.10*cc[-1])]
	upper = xx[np.argmax(cc>0.90*cc[-1])]
	delta = upper - lower
	xg = lower + 2./3.*delta #### 2/3 of 95 percent density
	xd = lower + 1.*delta #### 95% of density

	kc = 1./(xd-xg) #### slope is delta y = 1. divided by delta x
	kg = kc/5. #### slope is 1/3 of kc
	kd = 10.*kc #### slope is 10x kc

	q = integral(lower,upper,kg,kc,kd,xg,xd) # analytical integral
	qq = _riemann(x[keep],d[keep],x[0],x[-1],b) # approx integral
	rho0 = qq/q #### divide total volume by predicted volume

	theta = np.array((rho0,b,kg,kc,kd,xg,xd))
	return theta

def _minfxn(theta,x,d):
	rho0,b,kg,kc,kd,xg,xd = theta
	if rho0 < 0. or b < 0:
		return np.inf
	if kg < 0 or kc < 0 or kd < 0:
		return np.inf
	if xg >= xd:
		return np.inf

	yy = model(x,theta)
	rmsd = np.sqrt(np.sum(np.square(d-yy)))
	return rmsd


def fit(x,d,xl,xr,guess=None,maxiters=1000,repeats=10,verbose=False):
	'''
	theta = {rho0, b, kg, kc, kd, xg, xd}
	'''
	keep = np.bitwise_and(x>=xl,x<=xr)

	if guess is None:
		theta = naive(x,d,xl,xr)
	else:
		theta = guess.copy()
    
	for _ in range(repeats):
		out = minimize(_minfxn, theta, args=(x[keep],d[keep]), method='Nelder-Mead', options={'maxiter':maxiters})
		if verbose:
			print(out)
			print('Fit successful?',success)
		theta = out.x.copy()

	if verbose:
		## output the numpy formatted theta
		ot = ','.join(['%.4f'%(ti) for ti in theta.tolist()])
		print('theta = np.array((%s))'%(ot))

	return out.success,theta

def explore(x,d,xl,xr,guess=None,x_plotmin=None,x_plotmax=None):
	from ipywidgets import interact, widgets
	from IPython.display import display

	keep = np.bitwise_and(x>=xl,x<=xr)

	if guess is None:
		guess = naive(x[keep],d[keep])
	output = widgets.Output()
	display(output)

	def fxn(rho0=guess[0],b=guess[1],kg=guess[2],kc=guess[3],kd=guess[4],xg=guess[5],xd=guess[6]):
		theta = np.array((rho0,b,kg,kc,kd,xg,xd))
		fig,ax = plot(x,d,xl,xr,theta,x_plotmin=x_plotmin,x_plotmax=x_plotmax)
		plt.show()
		with output:
			output.clear_output(wait=True)
			ot = ','.join(['%.4f'%(ti) for ti in theta.tolist()])
			print('theta = np.array((%s))'%(ot))
		
	interact(fxn,rho0=(1e-6,1,.001),b=(0,1,.001), kg=(1e-2,2.,.001), kc=(1e-2,2.,.001), kd=(1e-2,20.,.001), xg=(0,100,.001), xd=(0,100,.001));

def explore_line(x,d):
	from ipywidgets import interact
	def fxn(x_ind=x.mean()):
		fig,ax = plt.subplots(1)
		ax.plot(x, d, color='tab:blue',alpha=.5)        
		ax.axvline(x_ind, color='k')
	interact(fxn, x_ind=(0.0,x.max(),.01));

def plot(x,d,xl,xr,theta,x_plotmin=None,x_plotmax=None):
	'''
	theta = {rho0, b, kg, kc, kd, xg, xd}
	'''

	rho0,b,kg,kc,kd,xg,xd = theta
	
	keep = np.bitwise_and(x>=xl,x<=xr)
	mm = model(x[keep],theta)

	dmax = np.max(np.abs((mm-d[keep])))
	k = kg+kc
	phi = kd/k

	fig,ax = plt.subplots(2,sharex=True,gridspec_kw={'height_ratios':[4,1]})
	ax[0].plot(x,d,color='k',lw=1.)
	ax[0].plot(x[keep],mm,color='tab:red',lw=2,zorder=4)
	ax[1].plot(x[keep],(mm-d[keep]),color='k',lw=1.)

	ax[0].axhline(y=b,color='k',lw=.5)
	ax[1].axhline(y=0,color='k',lw=1,zorder=-5)

	for aa in ax:
		aa.axvline(xl,color='k',lw=.5)
		aa.axvline(xr,color='k',lw=.5)

	if x_plotmin is None:
		x_plotmin = 0
	if x_plotmax is None:
		x_plotmax = x.max()
	ax[0].set_xlim(x_plotmin,x_plotmax)
	# ax[0].set_ylim(mm.min()-dmax,mm.max()+dmax)
	# ax[0].set_ylim(0,d.max()*1.05)
	ax[1].set_ylim(-dmax*1.05,dmax*1.05)

	ax[0].set_title(r'$\phi = %.3f$'%(phi))
	ax[0].set_ylabel('Scattering')
	ax[1].set_ylabel('Residual')
	ax[1].set_xlabel('Radial Distance (mm)')

	fig.subplots_adjust(hspace=.06)

	return fig,ax

def report(theta):
	from IPython.display import display, Markdown

	if theta.size == 7:
		variables = [r'$\rho_0$',r'$b_{\;\;}$',r'$k_g$',r'$k_c$',r'$k_d$',r'$x_g$',r'$x_d$']
	else:
		display('Theta is not properly formatted')
		return
	sout = r''
	for variable,value in zip(variables,theta):
		sout += r'%s = %.4f $ \\ $ '%(variable,value)
	display(Markdown(sout)) 
