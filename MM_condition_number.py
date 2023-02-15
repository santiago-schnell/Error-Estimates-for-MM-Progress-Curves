# -*- coding: utf-8 -*-
"""
Created on Fri May 6 12:17:58 2016

@author: wyliestroberg

Plot condition number and timescale ratio for different values of s0/Km
and plot substrate progress curves for different values of s0/Km.
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.labelsize'] = 24
import matplotlib.pyplot as plt
from MM_error import *
from scipy.linalg import norm

#-----------------------------------------------------------------------------
def calc_cond_num(q,func):
  """ Calculates the condition number for the nonlinear function func
      at the point in parameter space q. Time is implicitly included in func

      * Returns
      Condition number and eignevalues of F'*F'
  """
  J = approx_jacobian(q,func)	# Jacobian at parameter-space point q
  JH = np.asmatrix(J).H		# conjugate transpose of J
  g = np.dot(JH,J)
  try:
    lam, v = np.linalg.eig(g)
  except:
    lam = np.zeros(len(q),)
  cond_num = max(lam)/min(lam)
 # print np.corrcoef(J.T)
  return cond_num, lam

#-----------------------------------------------------------------------------
def alt_cond_num(q,func):
  J = approx_jacobian(q,func)	# Jacobian at parameter-space point q
  JH = np.asmatrix(J).H		# conjugate transpose of J
  g = np.dot(JH,J)
  lam, v = np.linalg.eig(g)
  spectral_norm = np.sqrt(max(lam)) 	# spectral norm of Jacobian
  f_norm = norm(func(q),2)		# 2-norm of function
  q_norm = norm(q,2)			# 2-norm of parameter vector
  cond_num = spectral_norm*q_norm/f_norm
  return cond_num

#-----------------------------------------------------------------------------
def approx_jacobian(x,func,epsilon=1.e-8,*args):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         x       - The state vector at which the Jacobian matrix is desired
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences

    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args))
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
       dx[i] = epsilon
       jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
       dx[i] = 0.0
    return jac.transpose()

#-----------------------------------------------------------------------------
def lamw_wrapper_generator(E0,C0):
  def lamw_wrapper(q0,**kwargs):
    t = kwargs["time"]
    Km = q0[0]
    Vm = q0[1]
    S0 = q0[2]
    sval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    cval = E0*sval/(Km+sval)
    return np.asfarray([sval,cval])
  return lamw_wrapper

#-----------------------------------------------------------------------------
def lamw_wrapper_generator(S0,E0,C0):
  def lamw_wrapper(q0,**kwargs):
    t = kwargs["time"]
    Km = q0[0]
    Vm = q0[1]
    sval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    cval = E0*sval/(Km+sval)
    return np.asfarray([sval,cval])
  return lamw_wrapper

#-----------------------------------------------------------------------------
def lamw_wrapper_generator3(S0,E0,C0):
  def lamw_wrapper(q0,**kwargs):
    t = kwargs["time"]
    Km = q0[0]
    Vm = q0[1]
    sval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    return np.asfarray(sval)
  return lamw_wrapper

#-----------------------------------------------------------------------------
def lamw_wrapper_generator2(E0,C0):
  def lamw_wrapper(q0,**kwargs):
    t = kwargs["time"]
    Km = q0[0]
    Vm = q0[1]
    S0 = q0[2]
    sval = MM_wfuncKmVm(t,S0,[Km,Vm,E0,C0])
    return np.asfarray(sval)
  return lamw_wrapper
#-----------------------------------------------------------------------------
def qssaode_wrapper_generatorKmVm(S0,E0,C0): # returns function for fitting
  def qssa_wrapper(q0,**kwargs):
    #y = odeint(MM_qssa,S0,t,args=([Km,Vm,E0,C0],))
    #return y[:,0]
    time = kwargs["time"]
    Km = q0[0]
    Vm = q0[1]
    y = odeint(MM_qssa,S0,time,args=([Km,Vm,E0,C0],))
    func = interp1d(time,y[:,0],'linear')
    return func(t)
  return qssa_wrapper
#-----------------------------------------------------------------------------
if __name__=="__main__":

  k1 = 1.0e-0
  km1 = 0.5
  k2 = 0.5
  C0 = 0.
  E0 = 1.e-1
  S0 = 1.e-2
  params = [k1,km1,k2,E0,C0]
  args = [k1,km1,k2,C0]
  Km = (km1+k2)/k1
  Vm = E0*k2
  #-----------------------------
  # Plot condition number vs s0 for given e0
  lamW = lamw_wrapper_generator2(E0,C0)		# defines function which solves s(t;Vm,Km)
  S0 = np.logspace(np.log10(Km*1e-3),np.log10(Km*1e3),100)
  CondNum = []
  tratio = []
  E0 = [1.e-2]
  #E0 = np.logspace(-3,1,5)
  for j in range(len(E0)):
    CondNumj = []
    tratioj = []
    params = [k1,km1,k2,E0[j],C0]
    for i in range(len(S0)):
      tsi = MM_ts(S0[i],params)
      if i==0:
	print tsi
      Vm = E0[j]*k2*tsi
      t = np.linspace(0,4.,1000)#t/tsi
      q0 = [Km,Vm]
      #lamW = lamw_wrapper_generator3(S0[i],E0[j],C0)	# defines function which solves s(t;Vm,Km)
      lamW = qssaode_wrapper_generatorKmVm(S0[i],E0[j],C0)	# defines function which solves s(t;Vm,Km)
      #q0 = [Km,Vm,S0[i]]
      funcW = partial(lamW,time=t)			# defines mapping from q0 to s at measured timepoints
      cn, eigs = calc_cond_num(q0,funcW)		# calculate condition number and eigs of Jacobian
      CondNumj.append(cn)
      tratioj.append(calc_tratio(S0[i],E0[j],args=args))	# calculate ratio of linear to curvature timescales
    CondNum.append(CondNumj)
    tratio.append(tratioj)

  figsize = (14,6)
  fig1,(ax3,ax1) = plt.subplots(1,2,figsize=figsize)
  axes_fontsize = 20
  for j in range(len(CondNum)):
    ax1.plot(S0,CondNum[j],'k-',linewidth=2,label=r'$t_s/t_Q$'+str(E0[j]))
    
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_xlabel(r'$s_0/K_M$',fontsize=axes_fontsize)
  ax1.set_ylabel(r'Condition Number',fontsize=axes_fontsize)
  #ax1.legend()
  #fig1.savefig('ConditionNumber.eps',edgecolor='black')
  ax2 = ax1.twinx()
  for j in range(len(tratio)):
    ax2.plot(S0,tratio[j],'k--',linewidth=2)
  ax2.set_yscale('log')
  ax2.set_ylabel(r'$t_S/t_Q$',fontsize=axes_fontsize)
  #fig1.savefig('ConditionNumberandTRatio.eps',edgecolor='black')

  #-----------------------------
  # Plot s vs time for different substrate concentrations
  #fig, ax = plt.subplots(1,1)
  '''
  t_plot = np.linspace(0,4.,1000) # [ti/ts for ti in t]
  S0 = [Km/100.,Km,100*Km]
  E0 = Km/1.
  params = [k1,km1,k2,E0,C0]
  labels = [r'$S_0 = 0.01K_M$',r'$S_0 = K_M$',r'$S_0 = 100K_M$']
  linetypes = ['--','-',':'] 
  for i in range(len(S0)):
    tsi = MM_ts(S0[i],params)
    Vm = E0*k2*tsi
    s_plot = [MM_wfuncKmVm(ti,S0[i],[Km,Vm,E0,C0])/S0[i] for ti in t_plot]
    ax3.plot(t_plot,s_plot,linetypes[i],label=labels[i],linewidth=3)

  ax3.set_xlabel(r'$t/t_s$',fontsize=axes_fontsize)
  ax3.set_ylabel(r'$s/s_0$',fontsize=axes_fontsize)
  ax3.legend()
  '''

  Km = (km1+k2)/k1
  E0 = Km
  Vm = E0*k2
  S0 = Km*10
  tmax = 2
  t_plot = np.linspace(0,tmax,1000) # [ti/ts for ti in t]
  ts = MM_ts(S0,[k1,km1,k2,E0])
  t = [tp*ts for tp in t_plot]

  s_plot = [MM_wfuncKmVm(ti,S0,[Km,Vm,E0,C0])/S0 for ti in t]
  p_plot = [-1*(si-1) for si in s_plot]

  S0_low = 0.1*S0
  S0_hi = 10*S0
  ts_low = MM_ts(S0_low,[k1,km1,k2,E0])
  t_low = [tp*ts_low for tp in t_plot]
  ts_hi = MM_ts(S0_hi,[k1,km1,k2,E0])
  t_hi = [tp*ts_hi for tp in t_plot]
  s_plotLow = [MM_wfuncKmVm(ti,S0_low,[Km,Vm,E0,C0])/S0_low for ti in t_low]
  s_plotHi = [MM_wfuncKmVm(ti,S0_hi,[Km,Vm,E0,C0])/S0_hi for ti in t_hi]

  dp = np.gradient(s_plot,ts/1000)
  d2p = np.gradient(dp,ts/1000)
  t_m = t_plot[np.array(d2p).argmax()]
  t_q = calc_tcurve(S0,E0,[k1,km1,k2])/ts
  t_l = calc_tlinear(S0,E0,[k1,km1,k2])/ts

  tlow = t_m-t_q/2.
  thi = tlow+t_q

  #fig,ax = plt.subplots(1,1)#,figsize=(6,4))
  ax3.plot(t_plot,s_plot,'k-',linewidth=4)
  ax3.plot(t_plot,s_plotLow,'k:',linewidth=2)
  ax3.plot(t_plot,s_plotHi,'k-.',linewidth=2)
  ax3.plot([tlow,tlow],[0,1],'k--',linewidth=3)
  ax3.plot([thi,thi],[0,1],'k--',linewidth=3)


  ax3.annotate(s='',xy=(tlow,0.4),xycoords='data',xytext=(thi,0.4),
              size=20,arrowprops=dict(arrowstyle='<->',lw=4))
  ax3.annotate('$t_{Q}$',xy=(tlow+t_q/2.,0.4),xycoords='data',xytext=(0,8),
             textcoords='offset points',fontsize=20,fontweight='bold',ha='center')

  x_fill = [tlow,thi]
  y_fill = [0,1]
  ax3.fill([0,tlow,tlow,0],[0,0,1,1],fill=True,color='b',alpha=0.2)
  ax3.fill([thi,tmax,tmax,thi],[0,0,1,1],fill=True,color='b',alpha=0.2)

  ax3.set_xlabel(r'$t/t_S$',fontsize=20)
  ax3.set_ylabel(r'$s/s_0$',fontsize=20)

  left, bottom, width, height = [0.22,0.65,0.20,0.25]
  ax4 = fig1.add_axes([left,bottom,width,height])
  ax4.plot(t_plot,d2p,'k',linewidth=3)
  ax4.axvline(x=tlow,ymin=0,ymax=1,color='k',linestyle='--',linewidth=3)
  ax4.axvline(x=thi,ymin=0,ymax=1,color='k',linestyle='--',linewidth=3)

  ax4.set_ylabel(r'Curvature')
  #ax4.set_xlabel(r'$t/t_S$',fontsize=20)
  ax4.set_yticks([])  
  ax4.set_xticks([])  


  fig1.tight_layout()
  ax1.xaxis.labelpad = 10
  ax3.xaxis.labelpad = 10
  ax1.tick_params(pad=10)
  ax2.tick_params(pad=10)
  ax3.tick_params(pad=10)
  fig1.text(0.01,0.93,'A.',fontsize=20,fontweight='bold')
  fig1.text(0.47,0.93,'B.',fontsize=20,fontweight='bold')
  fig1.savefig('MM_Figures/SvsTime_cond_nums.pdf',edgecolor='black')


  plt.show()
    
  

