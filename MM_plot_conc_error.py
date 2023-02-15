# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:17:58 2016

@author: wyliestroberg

Plot contours of the concentration error for Michaelis-Menten
kinetic calculations of substrate progress curves.
"""
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 14
import matplotlib.pyplot as plt
from MM_error import *

if __name__=="__main__":
#-----------------------------------------------------------------------------
#------- CONTOUR PLOTS ---------#
#-----------------------------------------------------------------------------
  ## Set parameters for contour plot ##
  nx = 40
  ny = 40
  xmin = 1e-2
  xmax = 1e4
  ymin = 1e-2
  ymax = 1e4

  tspan = 10	# Length of simulation/data observation, measued in ts
  Npoints = 1000
  s0 = np.logspace(np.log10(xmin),np.log10(xmax),nx) # normalized: s0 = S0/Km
  e0 = np.logspace(np.log10(ymin),np.log10(ymax),ny)
  C0 = 0
  val = 0.1

  k1 = 1.0
  km1 = [0.01,0.5,0.9]
  k2 = [0.99,0.5,0.1]

  figsize = (5,12)
  fig1,ax = plt.subplots(3,1,figsize=figsize)
  for i in range(len(km1)):
    args = [k1,km1[i],k2[i],C0]
    axi = ax[i]
    ## Calculate analytical validity lines ##
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

    ## Calculate theoretical error contours ##
    err = calcTheoreticalError(s0,e0,tspan=tspan,args=args,Npoints=Npoints)
    ## Plot theoretical error contours ##
    axi.set_xlabel(r'$s_0/K_M$')
    axi.set_ylabel(r'$e_0/K_M$')
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlim([xmin,xmax]) 
    axi.set_ylim([ymin,ymax])
    axi.plot(s0,e0_qssa_cond,'k--',linewidth=3,label=r'QSSA')
    axi.plot(s0,e0_rsa_cond,'k-',linewidth=3,label=r'QSSA+RSA')
    upperlim = 0.25
    cf = contour_wrapper(s0,e0,err,axi,upperlim)
    #axi.legend(loc=4)
  fig1.text(0.01,0.98,'A.',fontsize=20,fontweight='bold')
  fig1.text(0.01,0.65,'B.',fontsize=20,fontweight='bold')
  fig1.text(0.01,0.32,'C.',fontsize=20,fontweight='bold')
  fig1.tight_layout()

  fig1.savefig('MM_conc_error_contours.eps',edgecolor='black')
  
  plt.show()
#-----------------------------------------------------------------------------
