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

def format_report(theta,x=None,d=None,x_l=None,x_r=None):
    from IPython.display import display, Markdown

    if theta.size == 7:
        variables = [r'$k_g$',r'$k_c$',r'$\sigma_{peak}$',r'$k_d$',r'$x_g$',r'$x_d$',r'$b$']
    elif theta.size == 5:
        variables = [r'$\rho_d$',r'$\sigma_{peak}$',r'$k_d$',r'$x_d$',r'$b$']
    else:
        display('Theta is not properly formatted')
        return
    for variable,value in zip(variables,theta):
        display(Markdown('%s = %.3f'%(variable,value)))

    if theta.size == 7:
        if x is None:
            return
        k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
        delta_switch = std_peak*.5
        x_switch_l = x_d - delta_switch
        x_switch_r = x_d + delta_switch
        rho_g = riemann(x,d,x_l,x_switch_l,b) / (-1./k_g *(np.exp(-k_g*(x_switch_l-x_g))-np.exp(-k_g*(x_l-x_g))) +1./k_c*(np.exp(k_c*(x_switch_l-x_g))-np.exp(k_c*(x_l-x_g))))

        display(Markdown(r'$\rho_g = %.3f$'%(rho_g)))

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

def riemann(x,d,x0,x1,b=0):
	## linear interpolated center Riemann sum with baseline removed
	keep = np.bitwise_and(x>=x0,x<=x1)
	integral = np.sum((.5*(d[keep][1:]+d[keep][:-1])-b) * (x[keep][1:]-x[keep][:-1]))
	return integral

def model_growthexpansion(x,d,x_l,x_r,theta):
    k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
    y = np.zeros_like(x) + b

    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch

    rho_g = riemann(x,d,x_l,x_switch_l,b) / (-1./k_g *(np.exp(-k_g*(x_switch_l-x_g))-np.exp(-k_g*(x_l-x_g))) +1./k_c*(np.exp(k_c*(x_switch_l-x_g))-np.exp(k_c*(x_l-x_g))))

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
    k_g,k_c,std_peak,k_d,x_g,x_d,b = theta

    delta_switch = std_peak*.5
    x_switch_l = x_d - delta_switch
    x_switch_r = x_d + delta_switch
    rho_g = riemann(x,d,x_l,x_switch_l,b) / (-1./k_g *(np.exp(-k_g*(x_switch_l-x_g))-np.exp(-k_g*(x_l-x_g))) +1./k_c*(np.exp(k_c*(x_switch_l-x_g))-np.exp(k_c*(x_l-x_g))))

    if rho_g < 0. or b < 0:
        return np.inf
    if k_g < 0 or k_c < 0 or std_peak <= .25 or k_d < 0:
        return np.inf
    if x_g >= x_d:
        return np.inf

    keep = np.bitwise_and(x>=x_l,x<=x_r)
    # keep = np.bitwise_or(np.bitwise_and(x>=x_l,x<=x_switch_l),np.bitwise_and(x>=x_switch_r,x<=x_r))
    # if np.abs(x[keep][d[keep].argmax()]-x_d) > 5.:
        # return np.inf
    ymodel = model_growthexpansion(x,d,x_l,x_r,theta)
    ss = np.nanmean(np.square(d[keep]-ymodel[keep]))

    # ## keep the piece-wise solution smooth-ish
    # ss += np.nanmean(np.square(np.gradient(ymodel[keep])))
    
    return ss

def plot_growthexpansion(x,d,x_l,x_r,theta,x_max=None):
    '''
    theta = {k_g, k_c, std_peak, k_d, x_g, x_d, b}
    '''
    k_g,k_c,std_peak,k_d,x_g,x_d,b = theta
    mm = model_growthexpansion(x,d,x_l,x_r,theta)
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

    if x_max is None:
        x_max = x.max()
    ax[0].set_xlim(0,x_max)
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
    # x_d = x[keep][d[keep].argmax()]

    #### guess background (b)
    b = d[keep].min()

    #### prepare residual    
    keep = np.bitwise_and(x>=x_l,x<=x_r)
    residual = d*0.
    residual[keep] = np.log(d[keep] - (b-1e-6))

    #### guess peak location (x_d)
    xx=x[keep]
    yy = d[keep]-d[keep].min()
    qq = np.exp(-.5/(1.0**2.)*(xx-xx.mean())**2.)
    x_d = xx[np.convolve(qq,yy,mode='same').argmax()]
    pad = 1.
    keep = np.bitwise_and(x >= x_d-pad,x<=x_d+pad)
    i_d = d[keep].argmax()
    x_d = x[keep][i_d]
    d_d = d[keep][i_d]

    #### guess changeover between growth and chemotaxis
    keep = np.bitwise_and(x>=x_l,x<=x_d)
    x_min = x[keep][d[keep].argmin()]

    ## get diffusion slope (k_d)
    keep = np.bitwise_and(x>=x_d,x<=x_r)
    xx0 = x[keep][(d[keep]-b <= 0.9* (d_d-b)).argmax()]
    xx1 = x[keep][(d[keep]-b <= 0.1* (d_d-b)).argmax()]
    keep = np.bitwise_and(x>=xx0,x<=xx1)
    pfit_d = np.polyfit(x[keep],residual[keep],1) 
    k_d = np.abs(pfit_d[0])
    # print('guess',k_g,k_c,k_d)

    ## get peak width from HWHM into diffusion from peak (std_peak)
    keep = np.bitwise_and(x>=x_d,x<=x_r)
    xxh = x[keep][(d[keep]-b <= 0.5* (d_d-b)).argmax()]
    fwhm = 2.*(xxh-x_d)
    std_peak = fwhm/2.355

    ## set switches to 0.5 sigma (x_switch_r, x_switch_l; std_peak)
    x_switch_r = x_d + std_peak/2.
    x_switch_l = x_d - std_peak/2.
    if x_switch_l < x_min: ## fix if busted
        x_switch_l = .5*(x_d+x_min)
        std_peak = x_d-x_switch_l


    d_l = d[np.argmin((x-x_l)**2.)]
    d_min = d[np.argmin((x-x_min)**2.)]
    keep = np.bitwise_and(x>=x_l,x<=x_min)
    x_lmid = x[keep][np.argmax(d[keep]-d_min < .1*(d_l-d_min))]
    keep = np.bitwise_and(x>=x_min,x<=x_d)
    x_rmid = x[keep][np.argmax(d[keep]-d_min > .1*(d_d-d_min))]
    x_r_upper = x[keep][np.argmax(d[keep]-d_min > .9*(d_d-d_min))]
    
    keep = np.bitwise_and(x>=x_l,x<= x_lmid)
    pfit_g = np.polyfit(x[keep],residual[keep],1) ## Growth
    keep = np.bitwise_and(x>=x_rmid,x<=x_r_upper)
    pfit_c = np.polyfit(x[keep],residual[keep],1) ## Chemotaxis
    k_g = np.abs(pfit_g[0])
    k_c = np.abs(pfit_c[0])

    delta = 1./(k_g+k_c)*np.log(k_g/k_c)
    x_g = x_min - delta 

    theta = np.array((k_g,k_c,std_peak,k_d,x_g,x_d,b))
    return theta

def fit_growthexpansion(x,d,x_l,x_r,guess=None,maxiters=1000,repeats=10,verbose=False):
    '''
    theta = {k_g, k_c, std_peak, k_d, x_g, x_d, b}
    '''
    if guess is None:
        theta = guess_growthexpansion(x,d,x_l,x_r)
    else:
        theta = guess.copy()
    
    for _ in range(repeats):
        out = minimize(minfxn_growthexpansion,theta,args=(x,d,x_l,x_r),method='Nelder-Mead',options={'maxiter':maxiters})
        if verbose:
            print(out)
        theta = out.x.copy()

    if verbose:
        print(out.success)
        print(theta)
    return out.success,theta

def explore_growthexpansion(x,d,x_left,x_right,guess=None,x_max=None):
    if guess is None:
        guess = guess_growthexpansion(x,d,x_left,x_right)
    
    def fxn(k_g=guess[0],k_c=guess[1],std_peak=guess[2],k_d=guess[3],x_g=guess[4],x_d=guess[5],b=guess[6]):
        theta = np.array((k_g,k_c,std_peak,k_d,x_g,x_d,b))
        fig,ax = plot_growthexpansion(x,d,x_left,x_right,theta,x_max=x_max)
    interact(fxn,k_g=(1e-2,2.,.001),k_c=(1e-2,2.,.001),std_peak=(.25,5,.01),k_d=(1e-2,2.,.001),x_g=(0,20,.001),x_d=(0,20,.001),b=(0.,.5,.001));