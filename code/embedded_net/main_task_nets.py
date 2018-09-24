# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from scipy import sparse
from scipy.signal import iirfilter, lfilter
import sys
sys.path.append('C:/Users/User1/Google Drive/Philippe/Python/workspace/')    #Path to working directory (no backslash)
import network


def default(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError
    
def lowpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='lowpass',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def launch_simul(net,Inets,nt,train,w_in,w_fb,input_Inets,w_inj,target,BPhi,Pinv,nb_epochs,
                     tr,td,step):
    N = net.N
    output = np.zeros((nb_epochs,nt))
    spikes = []
    spikes_Inets = []
    Inets_gEx = []
    nb_Inets = len(Inets)
    N_Inets = Inets[0].N
    NE_Inets = Inets[0].NE
    
    for i in range(nb_epochs):
        print('Epoch {}/{}.'.format(i+1,nb_epochs))
        ns = 0
        ns_Inets = 0
        r = np.zeros(net.NE)
        hr = np.zeros(net.NE)         
        tspike = np.full((nt*int(N/2),2),np.nan)     #Spike times   
        tspike_Inets = np.full((nt*int(N_Inets/2)*nb_Inets,2),np.nan)     #Spike times                             
        gEx = np.zeros(N)                                            #Conductance of excitatory neurons
        gIn = np.zeros(N)                                            #Conductance of excitatory neurons
        gEx_Inets = np.zeros(N_Inets*nb_Inets)                                            #Conductance of excitatory neurons
        gIn_Inets = np.zeros(N_Inets*nb_Inets)                                            #Conductance of excitatory neurons
        F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
        F_Inets = np.full(N_Inets*nb_Inets,np.nan)
        V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc
        V_Inets = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N_Inets*nb_Inets)   #Set initial voltage Exc
        M = np.zeros((N,nt))  
        M_Inets = np.zeros((N_Inets*nb_Inets,nt))  
        #Recording variables
        total_gEx = np.zeros(nt)
        total_gIn = np.zeros(nt)
        Inets_cur = np.zeros(nt)
        total_gEx_Inets = np.zeros((nb_Inets,nt))
             
        #Flatten input net params
        GE_Inets = np.ndarray.flatten(np.array([Inet.GE for Inet in Inets]))
        GI_Inets = np.ndarray.flatten(np.array([Inet.GI for Inet in Inets]))
        E_idx_Inets = np.ndarray.flatten(np.array([Inet.E_idx+(x*N_Inets) for x,Inet in enumerate(Inets)]))
        I_idx_Inets = np.ndarray.flatten(np.array([Inet.I_idx+(x*N_Inets) for x,Inet in enumerate(Inets)]))
        delays_Inets = np.ndarray.flatten(np.array([Inet.delays for Inet in Inets]))
        VRest_Inets = np.ndarray.flatten(np.array([Inet.VRest for Inet in Inets]))
        RE_Inets = np.ndarray.flatten(np.array([Inet.RE for Inet in Inets]))
        RI_Inets = np.ndarray.flatten(np.array([Inet.RI for Inet in Inets]))
        ITonic_Inets = np.ndarray.flatten(np.array([Inet.ITonic for Inet in Inets]))
        tau_Inets = np.ndarray.flatten(np.array([Inet.tau for Inet in Inets]))
        Refractory_Inets = np.ndarray.flatten(np.array([Inet.Refractory for Inet in Inets]))
        Theta_Inets = np.ndarray.flatten(np.array([Inet.Theta for Inet in Inets]))
        TauE_Inets = np.ndarray.flatten(np.array([Inet.TauE for Inet in Inets]))
        TauI_Inets = np.ndarray.flatten(np.array([Inet.TauI for Inet in Inets]))
        W_Inets = np.zeros((nb_Inets*N_Inets,nb_Inets*N_Inets))
        for i_Inet in range(nb_Inets):
            W_Inets[(i_Inet*N_Inets):((i_Inet+1)*N_Inets),(i_Inet*N_Inets):((i_Inet+1)*N_Inets)] = Inets[i_Inet].W

        for t in range(nt):
            #Conductuances decay exponentially to zero
            gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
            gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
            
            gEx_Inets = np.multiply(gEx_Inets,np.exp(-dt/TauE_Inets))
            gIn_Inets = np.multiply(gIn_Inets,np.exp(-dt/TauI_Inets))
        
            #Update conductance of postsyn neurons
            F_E = np.all([[t-F[net.E_idx]==net.delays[net.E_idx]],[F[net.E_idx] != 0]],axis = 0,keepdims=0)
           
            SpikesERes = net.E_idx[F_E[0,:]]          #If a neuron spikes x time-steps ago, activate post-syn 
            if len(SpikesERes ) > 0:
                gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
                gEx_Inets = gEx_Inets + np.multiply(GE_Inets,np.sum(w_fb.T[:,SpikesERes],axis=1))
            F_I = np.all([[t-F[net.I_idx]==net.delays[net.I_idx]],[F[net.I_idx] != 0]],axis = 0,keepdims=0)
            SpikesIRes = net.I_idx[F_I[0,:]]        
            if len(SpikesIRes) > 0:
                gIn = gIn + np.multiply(net.GI,np.sum(net.W[:,SpikesIRes],axis=1))  #Increase the conductance of postsyn neurons
            
#==============================================================================
#             #F_E_Inets = np.zeros(NE_Inets*nb_Inets)
#             for Inet_i in range(len(Inets)):
#                 #Update conductance of postsyn neurons
#                 Inet = Inets[Inet_i]
#                 indices = np.all([[t-F_Inets[Inet.E_idx+(Inet_i*N_Inets)]==Inet.delays[Inet.E_idx]],
#                                   [F_Inets[Inet.E_idx+(Inet_i*N_Inets)] != 0]],axis = 0,keepdims=0)
#                 F_E_Inets[(Inet_i*NE_Inets):((Inet_i+1)*NE_Inets)] = indices
#                
#                 SpikesE = Inet.E_idx[indices[0,:]]          #If a neuron spikes x time-steps ago, activate post-syn 
#                 if len(SpikesE) > 0:
#                     gEx_Inets[(Inet_i*N_Inets):((Inet_i+1)*N_Inets)] = gEx_Inets[(Inet_i*N_Inets):((Inet_i+1)*N_Inets)]
#                     + np.multiply(Inet.GE,np.sum(Inet.W[:,SpikesE],axis=1))  #Increase the conductance of postsyn neurons
#                             
#                 indices = np.all([[t-F_Inets[Inet.I_idx+(Inet_i*N_Inets)]==Inet.delays[Inet.I_idx]],
#                                   [F_Inets[Inet.I_idx+(Inet_i*N_Inets)] != 0]],axis = 0,keepdims=0)
#                 SpikesI = Inet.I_idx[indices[0,:]]        
#                 if len(SpikesI) > 0:
#                     gIn_Inets[(Inet_i*N_Inets):((Inet_i+1)*N_Inets)] = gIn_Inets[(Inet_i*N_Inets):((Inet_i+1)*N_Inets)]
#                     + np.multiply(Inet.GI,np.sum(Inet.W[:,SpikesI],axis=1))  #Increase the conductance of postsyn neurons
#==============================================================================

            F_E_Inets = np.all([[t-F_Inets[E_idx_Inets]==delays_Inets[E_idx_Inets]],
                                              [F_Inets[E_idx_Inets] != 0]],axis = 0,keepdims=0)
            SpikesEInets = E_idx_Inets[F_E_Inets[0,:]]
#==============================================================================
#             if len(SpikesEInets)>0:
#                 break
#==============================================================================
            if len(SpikesEInets) > 0:
                gEx_Inets = gEx_Inets + np.multiply(GE_Inets,np.sum(W_Inets[:,SpikesEInets],axis=1))  
                gEx = gEx + np.multiply(net.GE,np.sum(w_in[:,np.where(F_E_Inets)[0]],axis=1))
            F_I_Inets = np.all([[t-F_Inets[I_idx_Inets]==delays_Inets[I_idx_Inets]],
                                              [F_Inets[I_idx_Inets] != 0]],axis = 0,keepdims=0)
            SpikesIInets = I_idx_Inets[F_I_Inets[0,:]]
            if len(SpikesIInets) > 0:
                gIn_Inets = gIn_Inets + np.multiply(GI_Inets,np.sum(W_Inets.T[:,SpikesIInets],axis=1))  



                   
            noise = np.random.normal(0,1,N)*gain_noise
            #Leaky Integrate-and-fire
            dV_res = ((net.VRest-V) + np.multiply(gEx,net.RE-V) + 
                      np.multiply(gIn,net.RI-V) + net.ITonic + noise)                                     #Compute raw voltage change
            V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
            #M[:,t] = V
            
            dV_Inets = ((VRest_Inets-V_Inets) + np.multiply(gEx_Inets,RE_Inets-V_Inets) + 
                      np.multiply(gIn_Inets,RI_Inets-V_Inets) + np.multiply(w_inj,input_Inets[:,t]) + ITonic_Inets)                                     #Compute raw voltage change
            V_Inets = V_Inets + (dV_Inets * (dt/tau_Inets))    
            #M_Inets[:,t] = V_Inets
            Inets_cur[t] = np.sum(dV_Inets)
             
            r = r*np.exp(-dt/tr) + hr*dt
            hr = hr*np.exp(-dt/td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],tr*td)        
            z = np.dot(BPhi.T,r)
            output[i,t] = z
            err = z-target[t]
                    
            #RLMS
            if t >= train_start and train:
                if t%step == 1:
                    cd = np.dot(Pinv,r)
                    BPhi = BPhi-(cd*err)
                    Pinv = Pinv - np.divide(np.outer(cd,cd.T),1 + np.dot(r.T,cd))
                    
            #Update cells
            Refract = t <= (F + net.Refractory)
            Refract_Inets = t <= (F_Inets + Refractory_Inets)
            V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
            V_Inets[Refract_Inets] = VRest_Inets[Refract_Inets]
            spikers = np.where(V > net.Theta)[0]
            spikers_Inets = np.where(V_Inets > Theta_Inets)[0]
            F[spikers] = t                                                              #Update the last AP fired by the neuron
            F_Inets[spikers_Inets] = t
            V[spikers] = 0                                                             #Membrane potential at AP time
            V_Inets[spikers_Inets] = 0
            tspike[ns:(ns+len(spikers)),:] = np.array([spikers,np.repeat(dt*t,len(spikers))]).T
            tspike_Inets[ns_Inets:(ns_Inets+len(spikers_Inets)),:] = np.array([spikers_Inets,
                         np.repeat(dt*t,len(spikers_Inets))]).T
            ns += len(spikers)
            ns_Inets += len(spikers_Inets)
            total_gEx[t] = np.sum(gEx)
            total_gIn[t] = np.sum(gIn)
            for Inet_i in range(nb_Inets):
                total_gEx_Inets[Inet_i,t] = np.sum(gEx_Inets[(Inet_i*N_Inets):((Inet_i+1)*N_Inets)])
        spikes.append(tspike)
        spikes_Inets.append(tspike_Inets)
        Inets_gEx.append(total_gEx_Inets)
    return BPhi,spikes,spikes_Inets,output,total_gEx,total_gIn,Inets_gEx
    
#Sim params
startTime = datetime.now()
output_path = "C:/Users/Cortex/Google Drive/Philippe/Python/spiking_reservoir/results/"
#output_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/results/"


#Time parameters
dt = 5e-05                              #dt (in ms)
T = 2                      #Simulation length (in seconds)
fs = int(1/dt)
nt = np.round(T/dt).astype(int)

#General parameters
Refractory = 0.002/dt
GaussSD = 0.04

#Initialize network
mean_GE = 0.8#0.8
mean_GI = 2#3              #0.055 Conductance (0.001 = 1pS)1.5
mean_delays = 0.001/dt              # transmission delay (ms)
N = 1000
p_NI = 0.2
p = 0.1
G = 0.04
GaussW = 0
ITonic = 9
gain_noise = 0


#Simulation parameters
nb_epochs = 10
td = 0.02
tr = 0.002

#Input parameters
#gain_in_list = np.arange(0,5,0.5)
gain_in = 0.1
start_stim = 0.5
t_stim = int(start_stim/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
#p_in_list = np.arange(0,1,0.1)
p_in = 1
p_fb = 1

#Target function
N_out = 1
sigma = 30
target = np.zeros(nt)
target[t_stim:] = lowpass_filter(np.random.randn(N_out,n_step_stim)*sigma,10,fs)
#plt.plot(np.ndarray.flatten(target[0,:]))

#Input network parameters
nb_Inets = 3
N_Inets = 300
GE_Inets = 0.8
GI_Inets_list = [1.3,1.5,1.8]
G_Inets = 0.04
p_Inets = 1
TauI = 0.06

p_inj = 1
gain_inj = 23
gain_fb = 0
input_Inets = np.zeros((nb_Inets*N_Inets,nt))
input_Inets[:,t_stim:t_stim+n_step_stim] = 1
Inets = [network.net(N_Inets,p_NI,mean_delays,Refractory,G=G_Inets,p=p_Inets, 
                     mean_GE = GE_Inets, mean_GI = GI_Inets, ITonic=8.5, 
                     mean_TauFall_I=TauI) for GI_Inets in GI_Inets_list]
           


#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))

#PLot settings
width = 15
height = 8

#Initialization
net = network.net(N,p_NI,mean_delays,Refractory,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
NE = net.NE
NE_Inets = int(N_Inets-(N_Inets*p_NI))

scale_in = gain_in/np.sqrt((p_in*N*NE_Inets))
scale_fb = gain_fb/np.sqrt((p_fb*NE*N_Inets))
scale_inj = gain_inj/np.sqrt((p_inj*N_Inets))
w_in = scale_in*np.abs(np.multiply(np.random.normal(0,1,(N,NE_Inets*nb_Inets)),
                                   np.random.rand(N,NE_Inets*nb_Inets)<p_in))
w_fb = scale_fb*np.abs(np.multiply(np.random.normal(0,1,(NE,N_Inets*nb_Inets)),
                                   np.random.rand(NE,N_Inets*nb_Inets)<p_fb))
w_inj = scale_inj*np.abs(np.multiply(np.random.normal(0,1,N_Inets*nb_Inets),
                                   np.random.rand(N_Inets*nb_Inets)<p_inj))


#Training
BPhi = np.zeros(NE)
Pinv = np.eye(NE)*alpha
train = True             
BPhi,spikes,spikes_Inets_train,output,total_gEx,total_gIn,Inets_gEx_train = launch_simul(net,Inets,nt,train,w_in,
              w_fb,input_Inets,w_inj,target,BPhi,Pinv,nb_epochs,tr,td,step)

plt.figure(figsize=(width,height))
xaxis = dt*np.arange(nt)
epoch_i = 0
for i in range(nb_Inets):
    plt.subplot(nb_Inets,2,(i*2)+1)
    [plt.plot(xaxis,Inets_gEx_train[x][i,:],linewidth=2,alpha=0.5) for x in range(nb_epochs)]
    #plt.plot(xaxis,Inets_gEx_train[epoch_i][i,:],linewidth=2,alpha=0.5)
    plt.subplot(nb_Inets,2,(i*2)+2)
    Inet_idx = np.arange(i*N_Inets,(i+1)*N_Inets)
    tspike_Inet = spikes_Inets_train[epoch_i][np.where(np.isin(spikes_Inets_train[epoch_i][:,0],Inet_idx))[0],:]
    plt.plot(tspike_Inet[:,1],tspike_Inet[:,0],'k.',markersize=0.5)
    
#Test
train = False            
BPhi,spikes,spikes_Inets,output,total_gEx,total_gIn,Inets_gEx = launch_simul(net,Inets,nt,train,w_in,
              w_fb,input_Inets,w_inj,target,BPhi,Pinv,1,tr,td,step)


tspike = spikes[0]       
plt.figure(figsize=(width,height))
plt.title('Testing.')
plt.subplot(2,1,1)
plt.plot(tspike[:,1],tspike[:,0],'k.')
#plt.xlim([dt*i-0.5,dt*i])
plt.ylim([0,200])
plt.subplot(2,1,2)
xaxis = dt*np.arange(nt)
plt.plot(xaxis,target,'k--',linewidth=2)
plt.plot(xaxis,output[0,:],'r',linewidth=2,alpha=0.5)
#==============================================================================
# 
# 
# REC_TEST = np.zeros(nt)
# #TEST
# sp_mat = np.zeros((N,nt))
# PSC = np.zeros(N)
# h = np.zeros(N)
# r = np.zeros(N)
# hr = np.zeros(N)
# JD = 0*PSC
# SP = {k:[] for k in range(N)}    
# tlast = np.zeros(N)
# ns = 0
# v = vreset +np.random.normal(15,6,(N))
# count_train = 0
# for i in range(nt):
#     I = PSC + BIAS
#     dv = np.multiply((dt*i>(tlast + tref)),((vrest-v)+I+ np.dot(w_in,input_pattern[:,i])) /tm)
#     v = v+(dt*dv)
#     
#     index = np.where(v>=vpeak)[0]
#     
#     if len(index)>0:
#         JD = np.sum(OMEGA[:,index],axis=1)
#         tspike[ns:(ns+len(index)),:] = np.array([index,np.repeat(dt*i,len(index))]).T
#         ns+= len(index)
#         for ni in index:
#             SP[ni].append(i*dt)
#             sp_mat[ni,i] = 1
#             
#     tlast[index] = dt*i
#     
#     PSC = PSC*np.exp(-dt/tr) + h*dt
#     h = h*np.exp(-dt/td) + np.divide(JD*(len(index)>0),(tr*td))
#     r = r*np.exp(-dt/tr) + hr*dt
#     hr = hr*np.exp(-dt/td) + np.divide(v>=vpeak,tr*td)
#     
#     #Implement RLMS with the FORCE method
#     z = np.dot(BPhi.T,r)
#     zt = target[i]
#     err = z-target[i]
# 
# 
#     v = v + np.multiply(30 - v,v>=vpeak)
#     REC[:,i] = v[0:10]
#     v = v + np.multiply(vreset-v,v>=vpeak)
#     REC_TEST[i] = z
#     RECB[:,i] = np.ndarray.flatten(BPhi[0:10])
#     REC2[:,i] = np.ndarray.flatten(r[0:20])
#     #REC_ERR[i] = err
#            
# plt.figure(figsize=(width,height))
# plt.title('Testing.')
# plt.subplot(2,1,1)
# plt.plot(tspike[:,1],tspike[:,0],'k.')
# #plt.xlim([dt*i-0.5,dt*i])
# plt.ylim([0,200])
# plt.subplot(2,1,2)
# xaxis = dt*np.arange(nt)
# plt.plot(xaxis,target,'k--',linewidth=2)
# plt.plot(xaxis,REC_TEST,'r',linewidth=2)
# 
# #Additionnal figures
# 
# #Error with training
# error_epoch = [np.mean(np.abs(REC_ERR[i,train_start:])) for i in range(nb_epochs)]
# plt.figure(figsize=(width,height))
# plt.plot(error_epoch)
# 
# #Inp vs net contribution
# plt.figure(figsize=(width,height))
# plt.plot(inp_current[0,:],label='Input')
# plt.plot(net_current[0,:],label='Network')
# plt.legend()
# 
# #Input pattern
# fig = plt.figure(figsize=(width,height))
# ax1 = fig.add_subplot(211)
# [ax1.plot(input_pattern[x,:]) for x in range(N_in)]
# ax2 = fig.add_subplot(212)
# ax2.plot(np.sum(input_pattern,axis=0))
# 
#==============================================================================
