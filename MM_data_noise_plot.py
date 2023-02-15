import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 14
import matplotlib.pyplot as plt
from MM_error import *


if __name__=="__main__":
  ## Set parameters for contour plot ##
  k1 = 1.0e0
  km1 = 1.0e1
  k2 = 1.0e1
  C0 = 0.

  nx = 50
  ny = 50
  xmin = 1e-2
  xmax = 1e2
  ymin = 1e-2
  ymax = 1e2

  tspan = 3	# Length of simulation/data observation, measued in ts
  Km = (km1+k2)/k1
  Ks = km1/k1
  K  = k2/k1
  args = [k1,km1,k2,C0]
  Npoints = 1000		# Number of data points
  t0 = 0.			# Initial timepoint, measured in tc
  init_coeff = [1.0,1.0]	# relative to exact values of parameter
  noise_variance = 0.01

  s0 = np.logspace(np.log10(xmin),np.log10(xmax),nx) # normalized: s0 = S0/Km
  e0 = np.logspace(np.log10(ymin),np.log10(ymax),ny)

  ## Calculate analytical validity lines ##
  val = 0.1
  e0_qssa_cond = qssa_cond_E0(s0,args,val) # e0 = E0/Km
  e0_rsa_cond = rsa_cond_E0(s0,val)

  ## Calculate fitting errors for noisy data##
  Km_err_Noise, Vm_err_Noise, KmFit_err_Noise, VmFit_err_Noise, S0_err_Noise = calcPredictError(s0,e0,
            Npoints=Npoints,tspan=tspan,t0=t0,args=args,init_coeff=init_coeff,
            includeS0=False,noise_variance=noise_variance)
  ## Calculate fitting errors for noise-free data##
  Km_err, Vm_err, KmFit_err, VmFit_err, S0_err = calcPredictError(s0,e0,
            Npoints=Npoints,tspan=tspan,t0=t0,args=args,init_coeff=init_coeff,
            includeS0=False,noise_variance=0.)

  #---------------------------------------------------------------------
  # Plotting
  #---------------------------------------------------------------------
  ##---- Plot error contours for noisy fitting case ----##
  figsize=(10,4)
  fig5,(ax5_1,ax5_2) = plt.subplots(1,2,figsize=figsize)
  plot_error_contour(s0=s0,e0=e0,Err=Km_err_Noise,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax5_1)
  ax5_1.set_title(r'$K_M$ Prediction Error')
  plot_error_contour(s0=s0,e0=e0,Err=Vm_err_Noise,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax5_2)
  ax5_2.set_title(r'V$_{max}$ Prediction Error')
  fig5_name = 'Noisy_error.eps'
  fig5.tight_layout()
  fig5.text(0.01,0.93,'A.',fontsize=20,fontweight='bold')
  fig5.text(0.5,0.93,'B.',fontsize=20,fontweight='bold')
  fig5.subplots_adjust(top=0.90)
  fig5.savefig('./MM_Figures/'+fig5_name,edgecolor='black')

  ##---- Plot variance for noisy and non-noisy cases ----##
  figsize=(12,10)
  boxprops = dict(facecolor='white',pad=10,alpha=1.0)
  fig6,((ax6_1,ax6_2),(ax6_3,ax6_4)) = plt.subplots(2,2,figsize=figsize)

  plot_error_contour(s0=s0,e0=e0,Err=KmFit_err_Noise,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax6_1)
  ax6_1.set_title(r'$K_M$ Variance')
  ax6_1.text(0.1,0.85,r'$\delta = 0.01$',transform=ax6_1.transAxes,fontsize=18,bbox=boxprops)

  plot_error_contour(s0=s0,e0=e0,Err=VmFit_err_Noise,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax6_2)
  ax6_2.set_title(r'V$_{max}$ Variance')
  ax6_2.text(0.1,0.85,r'$\delta = 0.01$',transform=ax6_2.transAxes,fontsize=18,bbox=boxprops)

  plot_error_contour(s0=s0,e0=e0,Err=KmFit_err,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax6_3)
  ax6_3.text(0.1,0.85,r'$\delta = 0.00$',transform=ax6_3.transAxes,fontsize=18,bbox=boxprops)

  plot_error_contour(s0=s0,e0=e0,Err=VmFit_err,scale='log',args=args,lims=(-1,1),
                             val=0.1,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,ax=ax6_4)
  ax6_4.text(0.1,0.85,r'$\delta = 0.00$',transform=ax6_4.transAxes,fontsize=18,bbox=boxprops)

  fig6_name = 'Noisy_param_variance.eps'
  fig6.tight_layout()
  fig6.text(0.01,0.93,'A.',fontsize=20,fontweight='bold')
  fig6.text(0.50,0.93,'B.',fontsize=20,fontweight='bold')
  fig6.text(0.01,0.47,'C.',fontsize=20,fontweight='bold')
  fig6.text(0.50,0.47,'D.',fontsize=20,fontweight='bold')
  fig6.subplots_adjust(top=0.90)
  fig6.savefig('./MM_Figures/'+fig6_name,edgecolor='black')

  plt.show()
