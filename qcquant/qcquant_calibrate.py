from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import os

def collect_data(prefix,fdir,suffix='.txt'):
	#### File naming should be '<prefix><conc>.txt'
	#### Data should already be normalized by Imax=4095

	datas = []
	concs = []

	for fn in os.listdir(fdir):
		if fn.startswith(prefix) and fn.endswith(suffix):
			x,d,_ = np.loadtxt(os.path.join(fdir,fn))
			conc = int(fn[len(prefix):-4])
			concs.append(conc)
			datas.append([x,d])

	concs = np.array(concs).astype('double')
	datas = np.array(datas).astype('double')
	order = np.argsort(concs)
	concs = concs[order]
	datas = datas[order]
	return concs,datas

def process_data(concs,datas,low,high):
	avgs = np.zeros(concs.size)

	fig,ax = plt.subplots(1,figsize=(6,4),sharex=True)
	for i in range(concs.size):
		ax.plot(datas[i][0], datas[i][1], label='%.2f'%(concs[i]/1000.))
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

def minfxn(theta,N,S):
	if np.any(theta < 0):
		return np.inf
	return np.mean(np.square(S-fxn_S(N,*theta)))

def attenuance(y,theta):
	I0f,I0B,sigmaL = theta
	A = np.log(I0f/((I0f+I0B)-y))

	Imax = 1.
	Nmax = 1/sigmaL*np.log(I0f/((I0f+I0B)-Imax))
	Amax = Nmax*sigmaL

	A[A<0] = 0
	A[A>Amax] = Amax
	
	return A

def calibrate(x,y):
	'''
	x should be agarose concentration milli%
	y should be average signal, normalized by I_max (i.e., 4095)
	'''

	x /= 1000.

	guess = np.array(([1.,y[0],(y[1]-y[0])/(x[1]-x[0])]))
	out = minimize(minfxn,guess,args=(x,y),method='Nelder-Mead')
	# print(guess)
	print(out)

	xx = np.linspace(0,x.max(),1000)
	I0f,I0B,sigmaL = out.x
	ycurve = fxn_S(xx,*out.x)
	Imax = 1.
	Nmax = 1/sigmaL*np.log(I0f/((I0f+I0B)-Imax))
	A = np.log(I0f/((I0f+I0B)-y))
	Amax = Nmax*sigmaL
	Aline = xx*sigmaL*(xx<Nmax) + Amax*(xx>=Nmax)

	fig,ax = plt.subplots(1,2,figsize=(12,4))
	ax[0].plot(x,y,'o')
	ax[0].plot(xx,ycurve)
	ax[0].axvline(Nmax,color='k')
	ax[0].axhline(Imax,color='gray',alpha=.5,ls='--')
	ax[0].set_ylabel(r'$I / I_{max}$')
	ax[0].set_xlabel('[Agarose] (% w/v)')
	ax[0].set_title(r'$I_0f/I_{max}=%.3f,\;I_0B/I_{max}=%.3f,\;\sigma L=%.3f\;\%%^{-1}$'%(I0f,I0B,sigmaL))
	ax[0].set_ylim(0,1.2)
	ax[0].set_xlim(0,.8)

	ax[1].plot(x,A,'o')
	ax[1].plot(xx,Aline)
	ax[1].axvline(Nmax,color='k')
	ax[1].axhline(Amax,color='gray',alpha=.5,ls='--')
	ax[1].set_ylabel('Sample-induced Attenuance')
	ax[1].set_xlabel('[Agarose] (% w/v)')
	ax[1].set_title('Linearized (Calibrated)')
	ax[1].set_ylim(0,1.2)
	ax[1].set_xlim(0,.8)
	plt.show()

	return out.x,fig,ax


if __name__ == '__main__':
	## args
	prefix = 'epi_'
	fdir = '/Users/colin/Desktop/projects/qcquant/example_data/epi_vs_trans/'
	low = 5.
	high = 35.

	concs,datas = collect_data(prefix,fdir)
	avgs,fig1,ax1 = process_data(concs,datas,low,high)
	theta,fig2,ax2 = calibrate(concs,avgs)

	np.save('theta.npy',theta)
	fig1.savefig('calibration_preparation.pdf')
	fig1.savefig('calibration_preparation.png')
	fig2.savefig('calibration_linearization.pdf')
	fig2.savefig('calibration_linearization.png')