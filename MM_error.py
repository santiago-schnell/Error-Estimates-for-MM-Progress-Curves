# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:17:58 2016

@author: wyliestroberg
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 18
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.special import lambertw
from scipy.stats import linregress
from scipy.interpolate import interp1d
from numpy import log10, exp
from multiprocessing import Pool
from itertools import product
from functools import partial
import collections

## FUNCTIONS ##
#-----------------------------------------------------------------------------
def MM_odes(y,t,params):
  k1 = params[0]
  km1 = params[1]
  k2 = params[2]
  e0 = params[3]
  c0 = params[4]
  
  Km = (km1+k2)/k1    # Michaelis-Menton constant
  Ks = km1/k1         # Disassociation constant
  
  s = y[0]		# Real parameters - not nondimensional!
  c = y[1]
  dsdt = k1*(-(e0+c0-c)*s+Ks*c)
  dcdt = k1*((e0+c0-c)*s-Km*c)
  
  return [dsdt, dcdt]
  
#-----------------------------------------------------------------------------
def MM_qssa(y,t,params):
  if len(params)==5:
    k1 = params[0]
    km1 = params[1]
    k2 = params[2]
    e0 = params[3]
    c0 = params[4]
    Km = (km1+k2)/k1    # Michaelis-Menton constant
    Vmax = k2*(e0+c0)

  elif len(params)==4:
    Km = params[0]
    Vmax = params[1]
    e0 = params[2]
    c0 = params[3]

  else:
    print 'Not enough input args for either k- or Km-based function!'
  
  s = y[0]
  
  dsdt = -Vmax*s/(Km+s)
  return dsdt
  
#-----------------------------------------------------------------------------
def MM_wfunc(t,S0,params):
  k1 = params[0]
  km1 = params[1]
  k2 = params[2]
  e0 = params[3]
  c0 = params[4]
  
  Km = (km1+k2)/k1
  Vmax = (e0+c0)*k2
  s = Km*lambertw(S0/Km*exp((S0-Vmax*t)/Km),tol=1e-12)
  return s

#-----------------------------------------------------------------------------
def MM_wfuncKmVm(t,S0,params):
  Km = params[0]
  Vmax = params[1]
  e0 = params[2]
  c0 = params[3]
  s = Km*lambertw(S0/Km*exp((S0-Vmax*t)/Km),tol=1e-12)
  return s

#-----------------------------------------------------------------------------
def MM_nonlin(t,S0,params):
  k1 = params[0]
  km1 = params[1]
  k2 = params[2]
  e0 = params[3]
  c0 = params[4]
  
  Km = (km1+k2)/k1
  Vmax = (e0+c0)*k2
  def implicit_s(s):
    return Km*np.log(S0/s) + S0 - s - Vmax*t
    
  s = fsolve(implicit_s,0.01)
  return s
    
#-----------------------------------------------------------------------------
def MM_ts(S0,params):
  k1 = params[0]
  km1 = params[1]
  k2 = params[2]
  E0 = params[3]
  Km = (km1+k2)/k1
  return (Km+S0)/(k2*E0)
 
#-----------------------------------------------------------------------------
def MM_tc(S0,params):
  k1 = params[0]
  km1 = params[1]
  k2 = params[2]
  e0 = params[3]
  Km = (km1+k2)/k1
  return 1./(k1*(Km+S0))

#-----------------------------------------------------------------------------
def qssa_cond_E0(s0,args,val=0.1):	# returns E0(S0) for which condition is met exactly
  km1 = args[1]
  k2 = args[2]
  k = km1/k2
  return val*(1+k)*(1.+s0)**2.
  
#-----------------------------------------------------------------------------
def rsa_cond_E0(s0,val=0.1):	# returns E0(S0) for which condtion is met exactly
  return [val*(1.+si) for si in s0]
  
#-----------------------------------------------------------------------------
def small_es_cond(s0,val=1.):	# returns E0(S0) for which condtion is met exactly
  return s0*val

#-----------------------------------------------------------------------------
def excess_s_cond(s0,val=0.1):	# returns E0(S0) for which condtion is met exactly
  return s0/val 
#-----------------------------------------------------------------------------
def calc_error(S0,E0,**kwargs):
  args = kwargs['args']
  tspan = kwargs['tspan']
  Npoints = kwargs['Npoints']
## Calcualte error for given (S0/Km,E0/Km)
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]

  params = [k1,km1,k2,E0,C0]
  ts = MM_ts(S0,params)
  time = np.linspace(0,tspan*ts,Npoints)

  y = odeint(MM_odes,[S0,C0],time,args=([k1,km1,k2,E0,C0],))
  s = y[:,0]
  c = y[:,1]
   
  ## QSSA M-M solution (using Lambert-W function) ##
 # s_qssa = np.array(MM_wfunc(time,S0,[k1,km1,k2,E0,C0]))

  s_qssa_ode = odeint(MM_qssa,S0,time,args=([k1,km1,k2,E0,C0],))
  s_qssa = s_qssa_ode[:,0]

  ## Calculate max error in substrate concentration ##
  #diff_err = s-s_qssa
  
  error = abs(s-s_qssa)/S0 # Normalized by initial substrate concentration

  return error

#-----------------------------------------------------------------------------
def calc_max_error(S0,E0,**kwargs):
  args = kwargs['args']
  tspan = kwargs['tspan']
  Npoints = kwargs['Npoints']
  error = calc_error(S0,E0,args=args,tspan=tspan,Npoints=Npoints)
  max_err = np.nanmax(error)
  return max_err
#-----------------------------------------------------------------------------
def gen_sample_data(y0,timeN,params,noise_variance=0.):
  time = np.linspace(0,timeN[-1]+1,10000)
  y = odeint(MM_odes,y0,time,args=(params,),mxstep=1000,printmessg=True)
  f_gen = interp1d(time,y[:,0],'linear')
  s = f_gen(timeN)
  #noise_variance = noise_variance*y0[0]
  if noise_variance!=0:
    noise = np.random.normal(0,noise_variance,len(s))
    #s = s + noise
    s = s*(1. + noise)
  return s

#-----------------------------------------------------------------------------
def lamw_fit_wrapper(S0,E0,C0): # returns function for fitting
  def lamw_wrapper(t,k1,km1,k2):
    fval = MM_wfunc(t,S0,[k1,km1,k2,E0,C0])
    if np.isnan(fval).any() or np.isinf(fval).any():
      fval = S0*1e5
    return fval
  return lamw_wrapper

#-----------------------------------------------------------------------------
def qssaode_fit_wrapper(S0,E0,C0): # returns function for fitting
  def qssa_wrapper(t,k_1,k_m1,k_2):
    y = odeint(MM_qssa,S0,t,args=([k_1,k_m1,k_2,E0,C0],))
    return y[:,0]
  return qssa_wrapper

#-----------------------------------------------------------------------------
def lamw_fit_wrapper2(E0,C0): # returns function for fitting
  def lamw_wrapper(t,k1,km1,k2,S0):
    fval = MM_wfunc(t,S0,[k1,km1,k2,E0,C0])
    if np.isnan(fval).any() or np.isinf(fval).any():
      fval = S0*1e5
    return fval
  return lamw_wrapper

#-----------------------------------------------------------------------------
def qssaode_fit_wrapper2(E0,C0): # returns function for fitting
  def qssa_wrapper(t,k_1,k_m1,k_2,S0):
    y = odeint(MM_qssa,S0,t,args=([k_1,k_m1,k_2,E0,C0],))
    return y[:,0]
  return qssa_wrapper
#-----------------------------------------------------------------------------
def qssaode_fit_wrapperKmVmS0(E0,C0): # returns function for fitting
  def qssa_wrapper(t,Km,Vm,S0):
    #y = odeint(MM_qssa,S0,t,args=([Km,Vm,E0,C0],))
    #return y[:,0]
    time = np.linspace(0,t[-1]+1.,10000)
    y = odeint(MM_qssa,S0,time,args=([Km,Vm,E0,C0],))
    func = interp1d(time,y[:,0],'linear')
    return func(t)
  return qssa_wrapper
#-----------------------------------------------------------------------------
def qssaode_fit_wrapperKmVm(S0,E0,C0): # returns function for fitting
  def qssa_wrapper(t,Km,Vm):
    #y = odeint(MM_qssa,S0,t,args=([Km,Vm,E0,C0],))
    #return y[:,0]
    time = np.linspace(0,t[-1]+1.,10000)
    y = odeint(MM_qssa,S0,time,args=([Km,Vm,E0,C0],))
    func = interp1d(time,y[:,0],'linear')
    return func(t)
  return qssa_wrapper
#-----------------------------------------------------------------------------
def exp_fit_wrapper(S0,E0,C0):
  def exp_fit(t,g):
    return S0*np.exp(-g*t)
  return exp_fit
#-----------------------------------------------------------------------------
def lamw_fit_wrapperKmVm(S0,E0,C0): # returns function for fitting
  def lamw_wrapper(t,Km,Vm):
    fval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    if np.isnan(fval).any() or np.isinf(fval).any():
      fval = S0*1e5
    return fval
  return lamw_wrapper

#-----------------------------------------------------------------------------
def lamw_fit_wrapperKmVmS0(E0,C0): # returns function for fitting
  def lamw_wrapper(t,Km,Vm,S0):
    fval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    if np.isnan(fval).any() or np.isinf(fval).any():
      print 'Nan or Inf returned by Lambert-W'
      print fval
      fval = S0*1e5
    return fval
  return lamw_wrapper

#-----------------------------------------------------------------------------
def pfo_fit_wrapper(S0,E0,C0):
  def pfo_wrapper(t,k1,km1,k2):
    k_psi = k1*E0
    Km_tilde = (km1+k2)/k_psi
    K_tilde = k2/k_psi
    a = k_psi*(1.+Km_tilde)
    b = 4*K_tilde/(1.+Km_tilde)**2
    lam_p = 0.5*a*(1.+np.sqrt(1.-b))
    lam_m = 0.5*a*(1.-np.sqrt(1.-b))
    A = (lam_p-k_psi)*np.exp(-lam_m*t)
    B = (lam_m-k_psi)*np.exp(-lam_p*t)
    return S0/(lam_p-lam_m)*(A+B)
  return pfo_wrapper

#-----------------------------------------------------------------------------
def fit_data(fitfunc,timeN,data,init,sigma=None):
  # fitfunc takes t and parameters to be fit, returns data estimates at t
  bound_min = [initi*1e-4 for initi in init]
  bound_max = [initi*1e4 for initi in init]
  #bounds = (bound_min,bound_max)
  bounds = (-np.inf,np.inf)
  #bounds = (0.,np.inf)
  maxfev = 1000
  ftol = 1e-8
  method = 'lm' # maxfev if lm, max_nfev else
  #method = 'trf' # maxfev if lm, max_nfev else
  abs_sig = True
  if sigma==None:
    abs_sig=False
  try:
    popt, pcov = curve_fit(fitfunc,timeN,data,p0=init,
                           sigma=sigma,
                           absolute_sigma=abs_sig,
                           bounds=bounds,
                           method=method,
                           maxfev=maxfev,
                           ftol=ftol)
  except (RuntimeError):
    popt = 1e5*np.ones(len(init))
    popt[1]=popt[1]*2
    pcov= np.ones((len(init),len(init)))
    print "Fitting did not converge in max_nfev function evals!"
  return popt, pcov

#-----------------------------------------------------------------------------
def fit_data_anneal(fitfunc,timeN,data,init):
  def cost_func(params):
    return np.linalg.norm(fitfunc(timeN,*params)-data)
  #minimizer_kwargs = {"method":"BFGS"}
  #minimizer_kwargs = {"method":"Nelder-Mead"}
  minimizer_kwargs = {"method":"SLSQP","options":{'ftol':1e-10},"bounds":((1e-6,np.inf),(1e-6,np.inf),(1e-6,np.inf))}
  #minimizer_kwargs = {"method":"COBYLA"}
  res = basinhopping(cost_func,init,niter=10,T=1e-4,minimizer_kwargs=minimizer_kwargs)
  pcov = np.zeros((len(init),len(init)))
  return res.x, pcov
#-----------------------------------------------------------------------------
def fit_data_evolution(fitfunc,timeN,data,init):
  def cost_func(params):
    return np.linalg.norm(fitfunc(timeN,*params)-data)
  bounds = ((0.,2.),(0.,1e4),(0,1e4))
  strategy = 'best1bin'
  maxiter = 10
  res = differential_evolution(cost_func,bounds,strategy=strategy,maxiter=maxiter,polish=True)
  pcov = np.zeros((len(init),len(init)))
  return res.x, pcov
#-----------------------------------------------------------------------------
def contour_wrapper(x,y,z,ax,upperlim=1.,lowerlim=0.):
    levels = MaxNLocator(nbins=10).tick_values(lowerlim,upperlim)
    cmap = plt.get_cmap('Reds')
    max_color = cmap(1.0)
    cmap.set_over(max_color)
    cf = ax.contourf(x,y,z,levels=levels,cmap=cmap,extend='max')
    fig = plt.gcf()
    fig.colorbar(cf,ax=ax)
    return cf
#-----------------------------------------------------------------------------
def contour_wrapper_logscale(x,y,z,ax,lims=(-2,3)):
    #levels = MaxNLocator(nbins=10).tick_values(0.,upperlim)
    levels = np.logspace(lims[0],lims[1],20)
    cmap = plt.get_cmap('Reds')
    max_color = cmap(1.0)
    cmap.set_over(max_color)
    maxval = 10.**lims[1]
    for zi in np.nditer(z,op_flags=['readwrite']):
      if zi >= maxval:
        zi[...] = maxval
    cf = ax.contourf(x,y,z,levels=levels,cmap=cmap,norm=LogNorm())
    formatter = LogFormatter(base=10.0)#, labelOnlyBase=False)
    nticks = 5
    clevel = np.logspace(lims[0],lims[1],nticks)
    fig = plt.gcf()
    fig.colorbar(cf,ax=ax,ticks=clevel,format=formatter)
    return cf

#-----------------------------------------------------------------------------
def calcKmVmaxError(s0i,e0j,**kwargs):# Npoints,tspan,args,init_coeff
  Npoints = kwargs["Npoints"]
  tspan = kwargs["tspan"]
  args = kwargs["args"]
  init_coeff = kwargs["init_coeff"]
  t0 = kwargs["t0"]
  includeS0 = kwargs["includeS0"]	# if true, S0 is considered a fitting parameter
  noise_variance = 0
  val = 1	# cut off for using algorithm to determine starting location for optimization
  if kwargs.__contains__("noise_variance"):
    noise_variance = kwargs["noise_variance"]
    val = 0 # use exact initial conditions
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]
  Km = (km1+k2)/k1

  S0 = s0i*Km
  E0 = e0j*Km
  Vmax = (E0+C0)*k2
  params = [k1,km1,k2,E0,C0]
  ts = MM_ts(S0,params)
  tc = MM_tc(S0,params)
  timeN = np.linspace(t0*tc,tspan*ts,Npoints)
  dt = timeN[1]-timeN[0]
  # Generate sample data
  noise_variance = noise_variance
  s = gen_sample_data([S0,C0],timeN,params,noise_variance)
  sigma = None
  #if noise_variance>1e-8:
  #  sigma = np.ones(len(s))*noise_variance*S0
  if e0j/(1.+s0i)>val: 
    init = [init_coeff[0]*Km,init_coeff[1]*Vmax]			# initial guess for parameters
    if includeS0==True:
      init.append(0.5*S0)
  else:
    init = initial_param_est(s,dt,S0bool=includeS0)
  if (includeS0==False):
    fit_func = qssaode_fit_wrapperKmVm(S0,E0,C0)
  else:
    fit_func = qssaode_fit_wrapperKmVmS0(E0,C0)
    #fit_func = lamw_fit_wrapperKmVmS0(E0,C0)
  try:
    popt, pcov = fit_data(fit_func,timeN,s,init,sigma)	# fit data using deterministic method
    #popt, pcov = fit_data_anneal(fit_func,timeN,s,init)	# fit data using basin-hopping
    #popt, pcov = fit_data_evolution(fit_func,timeN,s,[init[0],init[1],s[1]])	# fit data using differential evolution
  except (ValueError):
    print "Wylie says: Possibly bad initial condition for minimizer"
    print s[0:2]
    if (includeS0==true):
      init[2] = s[0]
    popt, pcov = fit_data(fit_func,timeN,s,init)
    #popt, pcov = fit_data_anneal(fit_func,timeN,s,[init[0],init[1],s[1]])
  perr = np.sqrt(np.diag(pcov))
  KmVmax_cov = np.sqrt(pcov[0,1])
  R12 = pcov[0,1]/np.sqrt(pcov[0,0]*pcov[1,1])
  #print R12
  #print (pcov[0,0],pcov[1,1],pcov[0,1],pcov[1,0])
  Km_cov = perr[0]
  Vmax_cov = perr[1]
  Km_est = popt[0]
  Vmax_est = popt[1]
  if len(popt)==3:
    S0_est = popt[2]
    S0_err = abs(S0-S0_est)/S0
  s_fit = fit_func(timeN,*popt)
  try:
    #sl, inter, r_val, p_val, std_err = linregress(s,s_fit)
    chi2 = sum([(s_fit[i]-s[i])**2./s[i] for i in range(len(s))])
  except (ValueError,IndexError):
    print s_fit.shape
    print s.shape
    print popt
    r_val = 0.
    chi2 = 1.
    
  # Calculate error in parameter estimation
  Km_err = abs(Km-Km_est)/Km
  Vmax_err = abs(Vmax-Vmax_est)/Vmax
  if len(popt)==3:
    return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, S0_err])
  else:
    return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, KmVmax_cov])
  #return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, R12])
  #return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, chi2])
#-----------------------------------------------------------------------------
def calcKmVmaxErrorPFO(s0i,e0j,**kwargs):# Npoints,tspan,args,init_coeff
  Npoints = kwargs["Npoints"]
  tspan = kwargs["tspan"]
  args = kwargs["args"]
  init_coeff = kwargs["init_coeff"]
  t0 = kwargs["t0"]
  includeS0 = kwargs["includeS0"]	# if true, S0 is considered a fitting parameter
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]
  Km = (km1+k2)/k1
  S0 = s0i*Km
  E0 = e0j*Km
  Vmax = (E0+C0)*k2
  params = [k1,km1,k2,E0,C0]
  ts = MM_ts(S0,params)
  tc = MM_tc(S0,params)
  timeN = np.linspace(t0*tc,tspan*ts,Npoints)
  dt = timeN[1]-timeN[0]
  # Generate sample data
  noise_variance = 0.00*S0
  s = gen_sample_data([S0,C0],timeN,params,noise_variance)
  #init = [init_coeff[0]*Km,init_coeff[1]*Vmax]			# initial guess for parameters
  init = [init_coeff[0]*k1,init_coeff[1]*km1,init_coeff[1]*k2]
  #init = initial_param_est(s,dt,S0bool=includeS0)
  if (includeS0==False):
    fit_func = pfo_fit_wrapper(S0,E0,C0)
  else:
    fit_func = pfo_fit_wrapper(E0,C0)
    #fit_func = lamw_fit_wrapperKmVmS0(E0,C0)
  try:
    popt, pcov = fit_data(fit_func,timeN,s,init)	# fit data using deterministic method
    #popt, pcov = fit_data_anneal(fit_func,timeN,s,[init[0],init[1],s[1]])	# fit data using basin-hopping
  except (ValueError):
    print "Wylie says: Possibly bad initial condition for minimizer"
    print s[0:3]
    if (includeS0==true):
      init[3] = s[0]
    popt, pcov = fit_data(fit_func,timeN,s,init)
    #popt, pcov = fit_data_anneal(fit_func,timeN,s,[init[0],init[1],s[1]])
  perr = np.sqrt(np.diag(pcov))
  Km_est = (popt[1]+popt[2])/popt[0]
  Vmax_est = popt[2]*E0
  KmVmax_cov = np.sqrt(pcov[0,1])
  Km_cov = perr[0]
  Vmax_cov = perr[1]
  s_fit = fit_func(timeN,*popt)
  try:
    #sl, inter, r_val, p_val, std_err = linregress(s,s_fit)
    chi2 = sum([(s_fit[i]-s[i])**2./s[i] for i in range(len(s))])
  except (ValueError,IndexError):
    print s_fit.shape
    print s.shape
    print popt
    r_val = 0.
    chi2 = 1.
    
  # Calculate error in parameter estimation
  Km_err = abs(Km-Km_est)/Km
  Vmax_err = abs(Vmax-Vmax_est)/Vmax
  k1_err = abs(k1-popt[0])/k1
  km1_err = abs(km1-popt[1])/km1
  k2_err = abs(k2-popt[2])/k2
  return np.array([k1_err, km1_err, k2_err, Vmax_cov, chi2])
  #return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, KmVmax_cov])
  #return np.array([Km_err, Vmax_err, Km_cov, Vmax_cov, R12])

#-----------------------------------------------------------------------------
def parallel_nested_for(iarray,jarray,func,**kwargs):
  len_func_out = kwargs['len_func_out']
  num_proc = kwargs['num_proc']
  Array = np.zeros((len(iarray),len(jarray),len_func_out))
  pool = Pool(processes=num_proc)	# create processor pool for parallel processing
  for i in range(len(iarray)):
    partial_func = partial(func,iarray[i])
    out = pool.map(partial_func,jarray)
    for j in range(len(out)):
      Array[j,i,:] = out[j]
  pool.close()
  pool.join()
  return Array

#-----------------------------------------------------------------------------
def calcTheoreticalError(s0,e0,**kwargs):# tspan,args
  tspan = kwargs["tspan"]
  args = kwargs["args"]
  Npoints = kwargs["Npoints"]
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]
  Km = (km1+k2)/k1

  func_wrapper = partial(calc_max_error,args=args,tspan=tspan,Npoints=Npoints)
  if (isinstance(s0,float) and isinstance(e0,float)):
    err = func_wrapper(s0*Km,e0*Km)
  else:
    S0 = [si*Km for si in s0] 
    E0 = [ei*Km for ei in e0] 
    err = parallel_nested_for(S0,E0,func_wrapper,len_func_out=1,num_proc=4)
  
  return err[:,:,0]
#-----------------------------------------------------------------------------
def calcPredictError(s0,e0,err_func=calcKmVmaxError,**kwargs):
  Npoints = kwargs["Npoints"]
  tspan = kwargs["tspan"]
  args = kwargs["args"]
  init_coeff = kwargs["init_coeff"]
  t0 = kwargs["t0"]
  includeS0 = kwargs["includeS0"]
  #err_func = kwargs["err_func"]
  noise_variance = 0.
  if kwargs.__contains__("noise_variance"):
    noise_variance = kwargs["noise_variance"] 
  print noise_variance
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]
  Km = (km1+k2)/k1
  func_wrapper = partial(err_func,
                 Npoints=Npoints,tspan=tspan,t0=t0,args=args,
                 init_coeff=init_coeff,includeS0=includeS0,noise_variance=noise_variance)
  Err_Array = parallel_nested_for(s0,e0,func_wrapper,len_func_out=5,num_proc=4) 

  Km_err = Err_Array[:,:,0]
  Vm_err = Err_Array[:,:,1]
  KmFit_err = Err_Array[:,:,2]
  VmFit_err = Err_Array[:,:,3]
  Chi_err = Err_Array[:,:,4]
  return Km_err, Vm_err, KmFit_err, VmFit_err, Chi_err
#-----------------------------------------------------------------------------
def plot_error_contour(**kwargs):
  # Parse input
  s0 = kwargs['s0']
  e0 = kwargs['e0']
  Err = kwargs['Err']
  scale = kwargs['scale']
  args = kwargs['args']
  val = kwargs['val']
  xmin = kwargs['xmin']
  xmax = kwargs['xmax']
  ymin = kwargs['ymin']
  ymax = kwargs['ymax']
  ax1 = kwargs['ax']
  lims = kwargs['lims']
  if scale == 'log':
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper_logscale(s0,e0,err,ax,lims)
  else:
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper(s0,e0,err,ax,lims[1],lims[0])

  e0_qssa_cond = qssa_cond_E0(s0,args,val) # e0 = E0/Km
  e0_rsa_cond = rsa_cond_E0(s0,val)
  # Plot contour
  ax1.set_xlabel(r'$s_0/K_M$')
  ax1.set_ylabel(r'$e_0/K_M$')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_xlim([xmin,xmax]) 
  ax1.set_ylim([ymin,ymax])
  
  ax1.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax1.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,Err,ax1)
#-----------------------------------------------------------------------------
def plot_KmVm_error_contours(**kwargs):
  # Parse input
  s0 = kwargs['s0']
  e0 = kwargs['e0']
  Km_err = kwargs['Km_err']
  Vm_err = kwargs['Vm_err']
  scale = kwargs['scale']
  args = kwargs['args']
  val = kwargs['val']
  xmin = kwargs['xmin']
  xmax = kwargs['xmax']
  ymin = kwargs['ymin']
  ymax = kwargs['ymax']
  if scale == 'log':
    lims = kwargs['lims']
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper_logscale(s0,e0,err,ax,lims)
  else:
    upperlim = kwargs['upperlim']
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper(s0,e0,err,ax,upperlim)

  e0_qssa_cond = qssa_cond_E0(s0,args,val) # e0 = E0/Km
  e0_rsa_cond = rsa_cond_E0(s0,val)


  # Setup figure
  figsize=(10,4)
  fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)

  # Plot contours
  ax1.set_xlabel(r'$s_0/K_M$')
  ax1.set_ylabel(r'$e_0/K_M$')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_xlim([xmin,xmax]) 
  ax1.set_ylim([ymin,ymax])
  #ax1.set_title(r'$K_M$ Prediction Error')
  
  ax1.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax1.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,Km_err,ax1)

  ax2.set_xlabel(r'$s_0/K_M$')
  ax2.set_ylabel(r'$e_0/K_M$')
  ax2.set_xscale('log')
  ax2.set_yscale('log')
  ax2.set_xlim([xmin,xmax]) 
  ax2.set_ylim([ymin,ymax])
  #ax2.set_title(r'V$_{max}$ Prediction Error')
  
  ax2.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax2.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,Vm_err,ax2)

  return fig
#-----------------------------------------------------------------------------
def plot_error_contours(**kwargs):
  # Parse input
  s0 = kwargs['s0']
  e0 = kwargs['e0']
  Max_err = kwargs['Max_err']
  Km_err = kwargs['Km_err']
  Vm_err = kwargs['Vm_err']
  KmFit_err = kwargs['KmFit_err']
  VmFit_err = kwargs['VmFit_err']
  Chi_err = kwargs['Chi_err']
  scale = kwargs['scale']
  args = kwargs['args']
  val = kwargs['val']
  xmin = kwargs['xmin']
  xmax = kwargs['xmax']
  ymin = kwargs['ymin']
  ymax = kwargs['ymax']
  if scale == 'log':
    lims = kwargs['lims']
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper_logscale(s0,e0,err,ax,lims)
  else:
    upperlim = kwargs['upperlim']
    def plot_contour(s0,e0,err,ax):
      return contour_wrapper(s0,e0,err,ax,upperlim)

  e0_qssa_cond = qssa_cond_E0(s0,args,val) # e0 = E0/Km
  e0_rsa_cond = rsa_cond_E0(s0,val)


  # Setup figure
  figsize=(16,8)
  fig3,((ax2, ax3, ax4), (ax5, ax6, ax7)) = plt.subplots(2,3,figsize=figsize)

  # Plot error contours
  ax2.set_xlabel(r'$s_0/K_M$')
  ax2.set_ylabel(r'$e_0/K_M$')
  ax2.set_xscale('log')
  ax2.set_yscale('log')
  ax2.set_xlim([xmin,xmax]) 
  ax2.set_ylim([ymin,ymax])
  ax2.set_title(r'Max Error')
  
  ax2.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax2.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,Max_err,ax2)
  ax2.legend(loc=4)

  ax3.set_xlabel(r'$s_0/K_M$')
  ax3.set_ylabel(r'$e_0/K_M$')
  ax3.set_xscale('log')
  ax3.set_yscale('log')
  ax3.set_xlim([xmin,xmax]) 
  ax3.set_ylim([ymin,ymax])
  ax3.set_title(r'Km Prediction Error')
  
  ax3.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax3.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  upperlimit = 10.
  cf = plot_contour(s0,e0,Km_err,ax3)

  ax4.set_xlabel(r'$s_0/K_M$')
  ax4.set_ylabel(r'$e_0/K_M$')
  ax4.set_xscale('log')
  ax4.set_yscale('log')
  ax4.set_xlim([xmin,xmax]) 
  ax4.set_ylim([ymin,ymax])
  ax4.set_title(r'V$_{max}$ Prediction Error')
  
  ax4.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax4.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  upperlimit = 10.
  cf = plot_contour(s0,e0,Vm_err,ax4)

  ax5.set_xlabel(r'$s_0/K_M$')
  ax5.set_ylabel(r'$e_0/K_M$')
  ax5.set_xscale('log')
  ax5.set_yscale('log')
  ax5.set_xlim([xmin,xmax]) 
  ax5.set_ylim([ymin,ymax])
  ax5.set_title(r'$\chi^{2}$')
  
  ax5.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax5.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,Chi_err,ax5)

  ax6.set_xlabel(r'$s_0/K_M$')
  ax6.set_ylabel(r'$e_0/K_M$')
  ax6.set_xscale('log')
  ax6.set_yscale('log')
  ax6.set_xlim([xmin,xmax]) 
  ax6.set_ylim([ymin,ymax])
  ax6.set_title(r'Km Fit Covariance')
  
  ax6.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax6.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,KmFit_err,ax6)

  ax7.set_xlabel(r'$s_0/K_M$')
  ax7.set_ylabel(r'$e_0/K_M$')
  ax7.set_xscale('log')
  ax7.set_yscale('log')
  ax7.set_xlim([xmin,xmax]) 
  ax7.set_ylim([ymin,ymax])
  ax7.set_title(r'V$_{max}$ Fit Covariance')
  
  ax7.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax7.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  cf = plot_contour(s0,e0,VmFit_err,ax7)

  return fig3
#-----------------------------------------------------------------------------
def calc_tlinear(S0,E0,args):
  ''' Calculate linear time scale for michaelis menton kinetics
      * Parameters: S0 - substrate concentration (not normalized)
                    E0 - initial enzyme concetration (not normalized)
                    args - (k1,km1,k2,C0)
  '''
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  Km = (km1+k2)/k1
  Vm = E0*k2
  tl = abs(S0-Km)/Vm
  return tl
#-----------------------------------------------------------------------------
def calc_tcurve(S0,E0,args):
  ''' Calculate curvature time scale for michaelis menton kinetics
      * Parameters: S0 - substrate concentration (not normalized)
                    E0 - initial enzyme concetration (not normalized)
                    args - (k1,km1,k2,C0)
  '''
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  Km = (km1+k2)/k1
  Vm = E0*k2
  s0 = S0/Km
  if s0>0.5:
    tcurve = 27./4.*Km/Vm*s0/(1.+s0)
  else:
    tcurve = Km/Vm*(1.+s0)**2.
  return tcurve
#-----------------------------------------------------------------------------
def calc_tratio(S0,E0,**kwargs):
  args = kwargs["args"]
  #tl = calc_tlinear(S0,E0,args)
  ts = MM_ts(S0,[args[0],args[1],args[2],E0])
  tc = calc_tcurve(S0,E0,args)
  val = (ts/tc)
  #val = np.log(val)
  return val
#-----------------------------------------------------------------------------
def tratio_contour(s0,e0,args):
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  Km = (km1+k2)/k1
  func_wrapper = partial(calc_tratio,args=args)
  if (isinstance(s0,float) and isinstance(e0,float)):
    trat = func_wrapper(s0*Km,e0*Km)
  else:
    S0 = [si*Km for si in s0] 
    E0 = [ei*Km for ei in e0] 
    trat = parallel_nested_for(S0,E0,func_wrapper,len_func_out=1,num_proc=4)
  return trat[:,:,0]
#-----------------------------------------------------------------------------
def initial_param_est(s,dt,S0bool='false',t=[]):
  ''' Estimate the Km and Vmax from discrete s vs t data using central difference
      to calculate first and second derivatives of s
  
      Paramters: s - concentration data
                 dt - sampling timestep
                 S0bool - boolean, if true, also estimate S0 for initial conditions
  '''
  s_dot = np.gradient(s,dt)
  s_ddot = np.gradient(s_dot,dt)
  ind = np.argmax(s<s[0]*0.25)
  index_max_curvature = s_ddot[ind:-1].argmax() + ind
  if index_max_curvature==1:
    print s[0]
  si = s[index_max_curvature-1]
  sdi = s_dot[index_max_curvature-1]
  sj = s[index_max_curvature+1]
  sdj = s_dot[index_max_curvature+1]
 
  Km = (sdj-sdi)*si*sj/(sdi*sj-sdj*si)
  Vm = -((sdi/si)*Km + sdi)
  S0 = s[0]
  if (S0bool):
    p0 = [Km,Vm,S0]
  else:
    p0 = [Km,Vm]
  p0 = [abs(pi) for pi in p0]
  if t==[]:
    t = s
  plt.figure(1)
  plt.plot(t,s)
  plt.figure(2)
  plt.plot(t,s_dot)
  plt.figure(3)
  plt.plot(t,s_ddot)
  return p0

#-----------------------------------------------------------------------------
def max_likelihood(data,fit_func,time,q):
  s_fit = fit_func(time,*q)
  err = data-s_fit
  #err2 = err*err
  #out = np.mean(err2)
  out = np.dot(err,err.T)/len(data)
  return out
#-----------------------------------------------------------------------------
def akaike_info_criteria(N,L,k):
  ''' Calculate the Akaike information criteria for a model.
      Parameters: N - length of data vector
                 L - maximum likelihood of data
                 k - number of fitting paramters
      Returns:   val - Akaike information criteria
  '''
  return N*np.log(L) + 2.*k
#-----------------------------------------------------------------------------
def baysian_info_criteria(N,L,k):
  ''' Calculate the Akaike information criteria for a model.
      Parameters: N - length of data vector
                 L - maximum likelihood of data
                 k - number of fitting paramters
      Returns:   val - Akaike information criteria
  '''
  return N*np.log(L) + np.log(N)*k
#-----------------------------------------------------------------------------
def calcInformation(s0i,e0j,**kwargs):# Npoints,tspan,args,init_coeff
  Npoints = kwargs["Npoints"]
  tspan = kwargs["tspan"]
  args = kwargs["args"]
  init_coeff = kwargs["init_coeff"]
  t0 = kwargs["t0"]
  includeS0 = kwargs["includeS0"]	# if true, S0 is considered a fitting parameter
  info_func = kwargs["info_func"]	# information criteria to use
  k1 = args[0]
  km1 = args[1]
  k2 = args[2]
  C0 = args[3]
  Km = (km1+k2)/k1

  S0 = s0i*Km
  E0 = e0j*Km
  Vmax = (E0+C0)*k2
  params = [k1,km1,k2,E0,C0]
  ts = MM_ts(S0,params)
  tc = MM_tc(S0,params)
  timeN = np.linspace(t0*tc,tspan*ts,Npoints)
  dt = timeN[1]-timeN[0]
  # Generate sample data
  s = gen_sample_data([S0,C0],timeN,params)
  
  init1 = [init_coeff[0]*Km,init_coeff[1]*Vmax]			# initial guess for parameters
  init2 = [Vmax/Km]
  #init1 = initial_param_est(s,dt,S0bool=includeS0)
  if (includeS0==False):
    fit_func1 = qssaode_fit_wrapperKmVm(S0,E0,C0)
    fit_func2 = exp_fit_wrapper(S0,E0,C0)
  else:
    fit_func1 = qssaode_fit_wrapperKmVmS0(E0,C0)
    fit_func2 = exp_fit_wrapper(S0,E0,C0)
    #fit_func = lamw_fit_wrapperKmVmS0(E0,C0)
  try:
    popt1, pcov1 = fit_data(fit_func1,timeN,s,init1)	# fit data using deterministic method
    popt2, pcov2 = fit_data(fit_func2,timeN,s,init2)	# fit data using deterministic method
  except (ValueError):
    print "Wylie says: Possibly bad initial condition for minimizer"
    print s[0:2]
    if (includeS0==true):
      init[2] = s[0]
    popt, pcov = fit_data(fit_func,timeN,s,init1)
  L1 = max_likelihood(s,fit_func1,timeN,popt1)
  L2 = max_likelihood(s,fit_func2,timeN,popt2)
    
  # Calculate info criteria
  info1 = info_func(len(s),L1,len(popt1))
  info2 = info_func(len(s),L2,len(popt2))
  #rel_prob = np.exp(-abs(info1-info2)/2)
  rel_prob = (info1-info2)
  return rel_prob
#-----------------------------------------------------------------------------
## MAIN ##
if __name__=="__main__":
  k1 = 1.0e0
  km1 = 1.0e1
  k2 = 1.0e1
  Km = (km1+k2)/k1
  Ks = km1/k1
  K = k2/k1
  
  E0 = 1.0e0
  C0 = 0.
  S0 = 1.0e-1 
  Vmax = k2*(E0+C0)

  params = [k1,km1,k2,E0,C0]
  y0 = [S0,C0]
  
  ts = MM_ts(S0,params)
  tspan = 100 # time in ts-units for simulated data/fitting
  Npoints = 10000
  time = np.linspace(0,tspan*ts,Npoints)
  
    
  y, out_info1 = odeint(MM_odes,y0,time,args=(params,),full_output=1)
  #print out_info1
  s = y[:,0]
  c = y[:,1]
  
  s_qssa, out_info2 = odeint(MM_qssa,S0,time,args=(params,),full_output=1)
  #print out_info1
  c_qssa = [(E0+C0)*si/(Km+si) for si in s_qssa]
  
  ## Lambert-W function ##
  s_lamw = MM_wfunc(time,S0,params)
  c_lamw = [(E0+C0)*si/(Km+si) for si in s_lamw]
  
  ## Nonlinear solve ##
  s_nonl = [MM_nonlin(ti,S0,params) for ti in time]
  c_nonl = [(E0+C0)*si/(Km+si) for si in s_nonl]

  ## Calc error as function of time
  args = [k1,km1,k2,C0]
  error = calc_error(S0,E0,args=args,tspan=tspan,Npoints=Npoints)
  max_err = np.nanmax(error)
  #print (S0/Km,E0/Km,max_err)

  ## Plot concentrations and error vs time for specific (S0,E0) ##
  time = [ti/ts for ti in time]
  fig, ax = plt.subplots(1,1)
  
  ax.plot(time,s,'k',linewidth=3,label="Exact")
  ax.plot(time,c,'k--',linewidth=3)
  
#  ax.plot(time,s_qssa,'b.-',linewidth=3)
#  ax.plot(time,c_qssa,'b-*',linewidth=3)
  
  ax.plot(time,s_lamw,'g-',linewidth=3,label="Michaelis-Menten")
  ax.plot(time,c_lamw,'g--',linewidth=3)
  
#  ax.plot(time,s_nonl,'r-|',linewidth=3)
#  ax.plot(time,c_nonl,'r-d',linewidth=3)
  
  ax_twin = ax.twinx()
  ax_twin.plot(time,error,linewidth=2)

  ## Plot fit vs real data
  timeN = np.linspace(0,tspan*ts,10)
  s_data = gen_sample_data([S0,C0],timeN,params)

  #---- When S0,E0,C0 are all considered know paramters ----#
  #---- and solving for k1,km1,k2 ----#

  #fit_func = lamw_fit_wrapper(S0,E0,C0)
  #fit_func = qssaode_fit_wrapper(S0,E0,C0)
  #popt = fit_data(fit_func,timeN,s_data,[k1,km1,k2])


  #----- When E0,C0 are all considered know paramters ----#
  #---- and solving for k1,km1,k2,S0 ----#

  #fit_func = qssaode_fit_wrapper2(E0,C0)
  #popt = fit_data(fit_func,timeN,s_data,[k1,km1,k2,S0])

  #Km_fit = (popt[1]+popt[2])/popt[0]



  #---- When S0,E0,C0 are all considered know paramters
  #---- and solving for Km and Vmax ----#
  #fit_func = lamw_fit_wrapperKmVm(S0,E0,C0)
  #popt = fit_data(fit_func,timeN,s_data,[Km,Vmax])
  #Km_fit = popt[0]

  #---- When E0,C0 are considered know paramters
  #---- and solving for Km, Vmax and S0 ----#
  #fit_func = lamw_fit_wrapperKmVmS0(E0,C0)
  init_Km = 1.0*Km
  init_Vmax = 1.0*Vmax
  fit_func = qssaode_fit_wrapperKmVmS0(E0,C0)
  popt, popc = fit_data(fit_func,timeN,s_data,[init_Km,init_Vmax,S0])
  Km_fit = popt[0]
  print popc

  ret = fit_data_anneal(fit_func,timeN,s_data,[init_Km,init_Vmax,S0])
  print ret
 
  Km_err, Vm_err, KmFit_err, VmFit_err, Chi_err = calcPredictError([S0/Km],[E0/Km],Npoints=100,tspan=tspan,t0=1.,init_coeff=[1.,1.],args=[k1,km1,k2,C0])

  figFit, axFit = plt.subplots(1,1)
  axFit.plot(timeN,s_data,'s',label="Exact")
  axFit.plot(timeN,fit_func(timeN,*popt),'-',label="M-M Fit")
  axFit.plot(timeN,fit_func(timeN,Km,Vmax,S0),'o-',label="M-M Fit w/ exact param")
  #axFit.plot(timeN,fit_func(timeN,Km,Vmax),'o-',label="M-M Fit w/ exact param")
  #axFit.plot(timeN,fit_func(timeN,k1,km1,k2),'o-',label="M-M Fit w/ exact param")
  #axFit.plot(timeN,fit_func(timeN,k1,km1,k2,S0),'o-',label="M-M Fit w/ exact param")

  axFit.legend()
  print ('#-------------------------------------------------#')
  Km_error = abs(Km-Km_fit)/Km
  print '(S0,E0) = ' + str([S0,E0])
  print 'Ks = ' + str(Ks)
  print 'K = ' + str(K)
  print '(Km, Km fit, Km error) = ' + str([Km,Km_fit,Km_error])
  print '(Vmax, Vmax fit) = ' + str([Vmax, popt[1]])
  print '(Km Error1, Km Error2) = ' + str([Km_error,Km_err])
  #k2_fit = popt[1]/E0
  #Ks_est = Km_fit - k2_fit
  #print(Ks,Ks_est)
  #print (k1,km1,k2)
  #print (k1_est,km1_est,k2_est)
  print ('#-------------------------------------------------#')
#-----------------------------------------------------------------------------
#------- PHASE PLANE TRAJECTORIES ---------#
  xmin = 1e-1
  xmax = 1e2
  ymin = 1e-1
  ymax = 1e2
  nx = 10
  ny = 10
  #s0 = np.logspace(np.log10(xmin),np.log10(xmax),nx)
  #c0 = np.logspace(np.log10(ymin),np.log10(ymax),ny)
  S_0 = np.linspace(xmin,xmax,nx)
  C_0 = np.linspace(ymin,ymax,ny)
  E0 = 5*Km
  Y = []

  def C_qssa(S,params):
    Km = (params[1]+params[2])/params[0]
    E0 = params[3]
    C0 = params[4]
    return (E0+C0)*S/(Km+S)

  def C_eqa(S,params):
    Km = (params[1]+params[2])/params[0]
    Ks = params[1]/params[0]
    E0 = params[3]
    C0 = params[4]
    return (E0+C0)*S/(Ks+S)

  for Si in S_0:
    C0 = 0
    y0 = [Si,C0]
    params = [k1,km1,k2,E0,C0]
    ts = MM_ts(Si,params)
    timespan = 100 # time in ts-units for simulated data/fitting
    time = np.linspace(0,timespan*ts,100000)
    y = odeint(MM_odes,y0,time,args=(params,),full_output=0)
    y[:,0] = y[:,0]/Km
    y[:,1] = y[:,1]/(E0+C0)
    Y.append(y)

  for Ci in C_0:
    y0 = [0,Ci]
    params = [k1,km1,k2,E0,Ci]
    ts = MM_ts(0,params)
    timespan = 100 # time in ts-units for simulated data/fitting
    time = np.linspace(0,timespan*ts,100000)
    y = odeint(MM_odes,y0,time,args=(params,),full_output=0)
    y[:,0] = y[:,0]/Km
    y[:,1] = y[:,1]/(E0+Ci)
    Y.append(y)

  figPhase, axPhase = plt.subplots(1,1)
  axPhase.set_xlabel(r'S')
  axPhase.set_ylabel(r'C')
  #axPhase.set_xscale('log')
  #axPhase.set_yscale('log')
  axPhase.set_xlim([1e-6,3]) 
  axPhase.set_ylim([1e-6,1])

  params = [k1,km1,k2,E0,C0]
  Km = (params[1]+params[2])/params[0]
  S_range = np.linspace(0,xmax,100)
  C_QSSA = C_qssa(S_range,params)
  C_EQA = C_eqa(S_range,params)
  c_QSSA = C_QSSA/(E0+C0)
  c_EQA = C_EQA/(E0+C0)
  s_range = S_range/Km

  axPhase.plot(s_range,c_QSSA,'--',linewidth=3)
  axPhase.plot(s_range,c_EQA,'-.',linewidth=3)

  for y in Y:
    axPhase.plot(y[:,0],y[:,1],'k-')

    
#-----------------------------------------------------------------------------
#------- CONTOUR PLOTS ---------#
#-----------------------------------------------------------------------------
  ## Set parameters for contour plot ##
  nx = 40
  ny = 40
  xmin = 1e-5
  xmax = 1e6
  ymin = 1e-5
  ymax = 1e5

  tspan = 10	# Length of simulation/data observation, measued in ts
  Km = (km1+k2)/k1
  args = [k1,km1,k2,C0]

  ## Calculate analytical validity lines ##
  s0 = np.logspace(np.log10(xmin),np.log10(xmax),nx) # normalized: s0 = S0/Km
  e0 = np.logspace(np.log10(ymin),np.log10(ymax),ny)
  val = 0.1
  e0_qssa_cond = qssa_cond_E0(s0,args,val) # e0 = E0/Km
  e0_rsa_cond = rsa_cond_E0(s0,val)
  s0_excess = []
  s0_small = []
  for si in s0:
    if si>10:
      s0_excess.append(si)
    if si<0.1:
      s0_small.append(si)
  e0_excess_s_cond = excess_s_cond(np.array(s0_excess),val)
  e0_small_es_cond = small_es_cond(np.array(s0_small),1.)

#-----------------------------------------------------------------------------
#------- THEORETICAL ERROR CONTOUR PLOTS ---------#
  ## Calculate theoretical error contours ##
  err = calcTheoreticalError(s0,e0,tspan=tspan,args=args,Npoints=Npoints)
  ## Plot theoretical error contours ##
  fig1,ax1 = plt.subplots(1,1)
  ax1.set_xlabel(r'$S_0/K_M$')
  ax1.set_ylabel(r'$E_0/K_M$')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_xlim([xmin,xmax]) 
  ax1.set_ylim([ymin,ymax])
  
  ax1.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
  ax1.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
  #ax1.plot(s0_excess,e0_excess_s_cond,'k-',linewidth=3,label=r'excess s0')
  #ax1.plot(s0_small,e0_small_es_cond,'k-',linewidth=3,label=r'Small e0, s0')
  cf = contour_wrapper(s0,e0,err,ax1)
  ax1.legend(loc=4)
  fig1.tight_layout()

  fig1.savefig('MM_error_contours.eps',edgecolor='black')
  
#-----------------------------------------------------------------------------
#------- PARAMETER ESTIMATION ERROR CONTOUR PLOTS ---------#
  ## Calculate fitting errors ##
  Npoints = 1000		# Number of data points
  t0 = 0.			# Initial timepoint, measured in tc
  init_coeff = [2.0,2.0]	# relative to exact values of parameter

  if (0):
    Km_err, Vm_err, KmFit_err, VmFit_err, Chi_err = calcPredictError(s0,e0,
            Npoints=Npoints,tspan=tspan,t0=t0,args=args,init_coeff=init_coeff)

    ## Plot error contours ##
    fig3 = plot_error_contours(s0=s0,e0=e0,Max_err=err,Km_err=Km_err,
                               Vm_err=Vm_err,KmFit_err=KmFit_err,VmFit_err=VmFit_err,
                               Chi_err=Chi_err,args=args,val=val,scale='log',lims=(-1,1),
                               xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

    fig_title = r'$K_M=$ ' + str(Km) + r', $K_S=$ ' + str(Ks) + r', $K=$ ' + str(K)
    fig_name = ('Zoomed_LM_tspan='+str(tspan)+'_Npoints='+str(Npoints)+'_Km0='+str(init_coeff[0])
                +'Km_Vm0='+str(init_coeff[1])+'Vm_KM=' + str(Km) + '_KS=' + str(Ks) 
                + '_K=' + str(K)+ '_k1=' + str(k1) + '.eps')
    fig3.suptitle(fig_title)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.90)
    #fig3.savefig('./MM_Figures/'+fig_name,edgecolor='black')

#-----------------------------------------------------------------------------
  
  plt.show()
