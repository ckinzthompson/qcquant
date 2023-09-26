import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ipywidgets import interact


def explore_line(x,d):
    def fxn(x_ind=x.mean()):
        fig,ax = plt.subplots(1)
        ax.plot(x, d, color='tab:blue',alpha=.5)        
        ax.axvline(x_ind, color='k')
    interact(fxn, x_ind=(0.0,x.max(),.01));

def explore_leftright(x,d):
    def fxn(x_left=0,x_right=x.max(),d_min=0.):
        fig,ax = plt.subplots(1)
        ax.plot(x, d, color='tab:blue',alpha=.5)        
        ax.axvline(x_left, color='k')
        ax.axvline(x_right, color='k')
    interact(fxn, x_left=(0.0,x.max(),.01), x_right=(0.0,x.max(),.01));

def format_report(theta):
    from IPython.display import display, Markdown

    if theta.size == 8:
        variables = [r'$\rho_g$',r'$k_g$',r'$k_c$',r'$\sigma_{peak}$',r'$k_d$',r'$x_g$',r'$x_d$',r'$b$']
    elif theta.size == 5:
        variables = [r'$\rho_d$',r'$\sigma_{peak}$',r'$k_d$',r'$x_d$',r'$b$']
    else:
        display('Theta is not properly formatted')
        return
    for variable,value in zip(variables,theta):
        display(Markdown('%s = %.3f'%(variable,value)))

########################################################################################################################
########################################################################################################################
########################################################################################################################


def model_diffusion(x,x_l,x_r,theta):
    rho_d,std_peak,k_d,x_d,b = theta
    y = np.zeros_like(x) + b

    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch

    ## GC
    keep = np.bitwise_and(x>=x_l,x<x_d)
    y[keep] = rho_d+(x[keep]*0)+b

    ## peak
    keep = np.bitwise_and(x>=x_d,x<x_switch_r)
    y[keep] = rho_d*np.exp(-.5/(std_peak**2.)*(x[keep]-x_d)**2.)+b

    ## D
    keep = np.bitwise_and(x>=x_switch_r,x<=x_r)
    rho_peak_d = rho_d*np.exp(-.5/(std_peak**2.)*(x_switch_r-x_d)**2.)
    y[keep] = rho_peak_d*np.exp(-k_d*(x[keep]-x_switch_r))+b
    
    return y

def minfxn_diffusion(theta,x,d,x_l,x_r):
    rho_d,std_peak,k_d,x_d,b = theta
    if b < 0 or rho_d < 0:
        return np.inf
    if std_peak < 0 or k_d < 0:
        return np.inf
    if x_l >= x_d or x_d >= x_r:
        return np.inf

    keep = np.bitwise_and(x>=x_l,x<=x_r)
    keep = np.bitwise_and(x>=x[keep][d[keep].argmax()],x<=x_r)
    ymodel = model_diffusion(x,x_l,x_r,theta)
    ss = np.nanmean(np.square(d[keep]-ymodel[keep]))

    ## keep the piece-wise solution smooth-ish
    ss += np.nanmean(np.square(np.gradient(ymodel[keep])))

    return ss

def guess_diffusion(x,d,x_l,x_r):
     ## guess
    keep = np.bitwise_and(x>=x_l,x<=x_r)
    x_d = x[keep][d[keep].argmax()]
    b = d[keep].min()
    
    keep = np.bitwise_and(x>=x_l,x<=x_d)
    rho_d = (d[keep]-b).mean()
    
    std_peak = 1.
    k_d = 1.
    
    theta = np.array((rho_d,std_peak,k_d,x_d,b))
    return theta

def fit_diffusion(x,d,x_l,x_r,guess=None,maxiters=1000,repeats=10,verbose=False):
    '''
    theta = {rho_d, std_peak, k_d, x_d, b}
    '''
    if guess is None:
        theta = guess_diffusion(x,d,x_l,x_r)
    else:
        theta = guess.copy()
    
    for _ in range(repeats):
        out = minimize(minfxn_diffusion,theta,args=(x,d,x_l,x_r),method='Nelder-Mead',options={'maxiter':maxiters})
        theta = out.x.copy()

    if verbose:
        print(out.success)
        print(theta)
    return out.success,theta

def plot_diffusion(x,d,x_l,x_r,theta,x_max=15.):
    '''
    theta = {rho_d, std_peak, k_d, x_d, b}
    '''
    rho_d,std_peak,k_d,x_d,b = theta
    mm = model_diffusion(x,x_l,x_r,theta)
    keep = np.bitwise_and(x>=x_l,x<=x_r)

    dmax = np.max(np.abs((mm-d)[keep]))
    
    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch

    fig,ax = plt.subplots(2,sharex=True,gridspec_kw={'height_ratios':[4,1]})
    ax[0].plot(x,d,color='k',lw=1.)
    ax[0].plot(x[keep],mm[keep],color='tab:red',lw=2,zorder=4)
    ax[1].plot(x[keep],(mm-d)[keep],color='k',lw=1.)

    for aa in ax:
        aa.axvline(x_l,color='k',lw=.5)
        aa.axvline(x_r,color='k',lw=.5)    
        aa.axvline(x_switch_r,color='k',lw=.5)
        aa.axvspan(xmin=x_switch_r,xmax=x_r,color='tab:blue',alpha=.05,zorder=-5)
    ax[0].axhline(y=b,color='k',lw=.5)
    
    ax[0].set_xlim(0,x_max)
    ax[0].set_ylim(mm.min()-dmax,mm.max()+dmax)
    ax[1].set_ylim(-dmax*1.05,dmax*1.05)
    
    ax[0].set_ylabel('Scattering')
    ax[1].set_ylabel('Residual')
    ax[1].set_xlabel('Radial Distance (mm)')
    
    fig.subplots_adjust(hspace=.06)
    return fig,ax

def explore_diffusion(x,d,x_left,x_right,guess=None):
    if guess is None:
        guess = guess_diffusion(x,d,x_left,x_right)
    
    def fxn(rho_d=guess[0],std_peak=guess[1],k_d=guess[2],x_d=guess[3],b=guess[4]):
        theta = np.array((rho_d,std_peak,k_d,x_d,b))
        fig,ax = plot_diffusion(x,d,x_left,x_right,theta)
    interact(fxn,rho_d=(0,.2,.0001),std_peak=(.25,5,.01),k_d=(1e-2,2.,.001),x_g=(0,20,.001),x_d=(0,20,.001),b=(0.,.5,.001));




########################################################################################################################
########################################################################################################################
########################################################################################################################

def model_growthexpansion(x,x_l,x_r,theta):
    rho_g,k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
    y = np.zeros_like(x) + b

    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch

    ## GC
    keep = np.bitwise_and(x>=x_l,x<x_switch_l)
    y[keep] = rho_g*(np.exp(-k_g*(x[keep]-x_g)) + np.exp(k_c*(x[keep]-x_g)))+b

    ## peak
    keep = np.bitwise_and(x>=x_switch_l,x<x_switch_r)
    rho_gc_peak = rho_g*(np.exp(-k_g*(x_switch_l-x_g)) + np.exp(k_c*(x_switch_l-x_g)))
    rho_peak = rho_gc_peak/np.exp(-.5/(std_peak**2.)*(x_switch_l-x_d)**2.)
    y[keep] = rho_peak*np.exp(-.5/(std_peak**2.)*(x[keep]-x_d)**2.)+b

    ## D
    keep = np.bitwise_and(x>=x_switch_r,x<=x_r)
    y[keep] = rho_gc_peak*np.exp(-k_d*(x[keep]-x_switch_r))+b
    
    return y

def minfxn_growthexpansion(theta,x,d,x_l,x_r):
    rho_g,k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
    if rho_g < 0. or b < 0:
        return np.inf
    if k_g < 0 or k_c < 0 or std_peak <= .25 or k_d < 0:
        return np.inf
    if x_g >= x_d:
        return np.inf

    keep = np.bitwise_and(x>=x_l,x<=x_r)
    if np.abs(x[keep][d[keep].argmax()]-x_d) > 5.:
        return np.inf
    ymodel = model_growthexpansion(x,x_l,x_r,theta)
    ss = np.nanmean(np.square(d[keep]-ymodel[keep]))

    ## keep the piece-wise solution smooth-ish
    ss += np.nanmean(np.square(np.gradient(ymodel[keep])))
    
    return ss

def plot_growthexpansion(x,d,x_l,x_r,theta,x_max=15.):
    '''
    theta = {rho_g, k_g, k_c, std_peak, k_d, x_g, x_d, b}
    '''
    rho_g,k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
    mm = model_growthexpansion(x,x_l,x_r,theta)
    keep = np.bitwise_and(x>=x_l,x<=x_r)

    dmax = np.max(np.abs((mm-d)[keep]))
    k = k_g+k_c
    phi = k_d/k
    
    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch

    fig,ax = plt.subplots(2,sharex=True,gridspec_kw={'height_ratios':[4,1]})
    ax[0].plot(x,d,color='k',lw=1.)
    ax[0].plot(x[keep],mm[keep],color='tab:red',lw=2,zorder=4)
    ax[1].plot(x[keep],(mm-d)[keep],color='k',lw=1.)
    
    ax[0].axhline(y=b,color='k',lw=.5)
    ax[1].axhline(y=0,color='k',lw=1,zorder=-5)
    for aa in ax:
        aa.axvline(x_l,color='k',lw=.5)
        aa.axvline(x_r,color='k',lw=.5)
        aa.axvline(x_switch_l,color='k',lw=.5)
        aa.axvline(x_switch_r,color='k',lw=.5)        
        aa.axvspan(xmin=x_l,xmax=x_switch_l,color='tab:orange',alpha=.05,zorder=-5)
        # aa.axvspan(xmin=x_switch_l,xmax=x_switch_r,color='tab:green',alpha=.05,zorder=-5)
        aa.axvspan(xmin=x_switch_r,xmax=x_r,color='tab:blue',alpha=.05,zorder=-5)

    ax[0].set_xlim(0,15)
    ax[0].set_ylim(mm.min()-dmax,mm.max()+dmax)
    ax[1].set_ylim(-dmax*1.05,dmax*1.05)
    
    ax[0].set_title(r'$\phi = %.3f$'%(phi))
    ax[0].set_ylabel('Scattering')
    ax[1].set_ylabel('Residual')
    ax[1].set_xlabel('Radial Distance (mm)')
    
    fig.subplots_adjust(hspace=.06)

    return fig,ax

def guess_growthexpansion(x,d,x_l,x_r):
    keep = np.bitwise_and(x>=x_l,x<=x_r)
    x_d = x[keep][d[keep].argmax()]
    b = d[keep].min()
    
    keep = np.bitwise_and(x>=x_l,x<=x_d)
    x_min = x[keep][d[keep].argmin()]
    
    keep = np.bitwise_and(x>=x_l,x<=x_r)
    residual = d*0.
    residual[keep] = np.log(d[keep] - (b-1e-6))
    keep = np.bitwise_and(x>=x_l,x<=x_min)
    pfit_g = np.polyfit(x[keep],residual[keep],1) ## Growth
    keep = np.bitwise_and(x>=x_min,x<=x_d)
    pfit_c = np.polyfit(x[keep],residual[keep],1) ## Chemotaxis
    keep = np.bitwise_and(x>=x_d,x<=x_r)
    keep = np.bitwise_and(keep,d > np.log(.01))
    pfit_d = np.polyfit(x[keep],residual[keep],1) ## Diffusion
    
    k_g = np.abs(pfit_g[0])
    k_c = np.abs(pfit_c[0])
    k_d = np.abs(pfit_d[0])
    # print('guess',k_g,k_c,k_d)
    
    std_peak = 1.

    delta = 1./(k_g+k_c)*np.log(k_g/k_c)
    
    x_g = x_min - delta
    rho_min = d[np.argmin(np.abs(x-x_min))]
    rho_g = (rho_min-b)/2.
    
    theta = np.array((rho_g,k_g,k_c,std_peak,k_d,x_g,x_d,b))
    return theta

def fit_growthexpansion(x,d,x_l,x_r,guess=None,maxiters=1000,repeats=10,verbose=False):
    '''
    theta = {rho_g, k_g, k_c, std_peak, k_d, x_g, x_d, b}
    '''
    if guess is None:
        theta = guess_growthexpansion(x,d,x_l,x_r)
    else:
        theta = guess.copy()
    
    for _ in range(repeats):
        out = minimize(minfxn_growthexpansion,theta,args=(x,d,x_l,x_r),method='Nelder-Mead',options={'maxiter':maxiters})
        # print(out)
        theta = out.x.copy()

    if verbose:
        print(out.success)
        print(theta)
    return out.success,theta

def explore_growthexpansion(x,d,x_left,x_right,guess=None):
    if guess is None:
        guess = guess_growthexpansion(x,d,x_left,x_right)
    
    def fxn(rho_g=guess[0],k_g=guess[1],k_c=guess[2],std_peak=guess[3],k_d=guess[4],x_g=guess[5],x_d=guess[6],b=guess[7]):
        theta = np.array((rho_g,k_g,k_c,std_peak,k_d,x_g,x_d,b))
        fig,ax = plot_growthexpansion(x,d,x_left,x_right,theta)
    interact(fxn,rho_g=(0,.2,.0001),k_g=(1e-2,2.,.001),k_c=(1e-2,2.,.001),std_peak=(.25,5,.01),k_d=(1e-2,2.,.001),x_g=(0,20,.001),x_d=(0,20,.001),b=(0.,.5,.001));