# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 08:21:49 2016

@author: Phil
"""
from __future__ import division
import numpy as np
      
        
class net(object):

    def __init__(self,N,pNI,mean_TranDelay,mean_Refractory,G=0.04,
                 p=0.2, mean_GE = 0.01, mean_GI = 0.04, ITonic=12,
                 mean_TauFall_I=0.02):
        self.N = N
        self.NI = int(self.N*pNI)
        self.NE = self.N-self.NI
        self.E_idx = np.arange(0,self.NE)
        self.I_idx = np.arange(self.NE,self.N)
        self.tau = 0.02
        #SINGLE NEURON CELL PARAMETERS
        mean_EE = 0				# reversal potential (mV) (Vogels & Abbott, 2009 Vogels et al. 2011, Science)
        mean_EI = -80 				# reversal potential (mV) (Vogels & Abbott, 2009 Vogels et al. 2011, Science)
        self.mean_VRest = -60                #resting-state potential (mV)
        mean_Theta = -50	 		# spike threshold (mV) (Vogels et al. 2011, Science)
        mean_TauFall_E = 0.02	# EPSP fall for excitatory cells (ms) (Vogels & Abbott, 2009 Vogels et al. 2011, Science)
        mean_TauFall_I = mean_TauFall_I #10 EPSP fall for inhibitory cells (ms) (Vogels & Abbott, 2009 Vogels et al. 2011, Science)		
        mean_ITonic = ITonic			#12.8 intrinsic drive to spike (1 = 1 pA)
        GaussSD = 0.01			# Gaussian parameter (sigma)
        GaussTheta = 0.01
        
        
        
        self.td = 0.02
        self.tr = 0.002
        
        self.delays = np.round(np.random.normal(mean_TranDelay,GaussSD*mean_TranDelay,(N))).astype(int) #Matrix of transmission delays from pre to post syn
        self.tau = abs(np.random.normal(self.tau,GaussSD*self.tau,(N)))                            #Set membrane capacitance                                               #Set membrane time constant
        self.GE = abs(np.random.normal(mean_GE,GaussSD*mean_GE,(N)))                          #Excitatory leaky conductance
        self.GI = abs(np.random.normal(mean_GI,GaussSD*mean_GI,(N)))                          #Inhibitory leaky conductance
        self.RE = np.random.normal(mean_EE,GaussSD,(N))                          #Excitatory reversal potential (mV)
        self.RI = np.random.normal(mean_EI,abs(GaussSD*mean_EI),(N))                  #Inhibitory reversal potential (mV)
        self.Theta = np.random.normal(mean_Theta,abs(GaussTheta*mean_Theta),(N))         #Spiking threshold
        self.VRest = np.random.normal(self.mean_VRest,abs(GaussSD*self.mean_VRest),(N))         #Resting potential
        self.TauE = np.random.normal(mean_TauFall_E,abs(GaussSD*mean_TauFall_E),(N))  #Time constant for exponential decay E
        self.TauI = np.random.normal(mean_TauFall_I,abs(GaussSD*mean_TauFall_I),(N))  #Time constant for exponential decay I
        self.ITonic = np.random.normal(mean_ITonic,abs(GaussSD*mean_ITonic),(N))      #Constant input received by each cell
        self.Refractory = np.random.normal(mean_Refractory,abs(GaussSD*mean_Refractory),(N))      #Constant input received by each cell
        self.W = np.abs(G*np.multiply(np.random.normal(0,1,(N,N)),np.random.rand(N,N)<p)/(np.sqrt(N*p)))
    