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
import random

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

def launch_simul(net,nt,train,input_res,w_res,target,BPhi,Pinv,nb_epochs,
                     tr,td,step):
    N = net.N
    output = np.zeros((nb_epochs,nt))
    spikes = []
    for i in range(nb_epochs):
        print('Epoch {}/{}.'.format(i+1,nb_epochs))
        ns = 0
        r = np.zeros(net.NE)
        hr = np.zeros(net.NE)         
        tspike = np.full((nt*int(N/2),2),np.nan)     #Spike times                   
        gEx = np.zeros(N)                                            #Conductance of excitatory neurons
        gIn = np.zeros(N)                                            #Conductance of excitatory neurons
        F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
        V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
        #Recording variables
        total_gEx = np.zeros(nt)
        total_gIn = np.zeros(nt)

        for t in range(nt):
            #Conductuances decay exponentially to zero
            gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
            gIn = np.multiply(gIn,np.exp(-dt/net.TauI))
        
            #Update conductance of postsyn neurons
            F_E = np.all([[t-F[net.E_idx]==net.delays[net.E_idx]],[F[net.E_idx] != 0]],axis = 0,keepdims=0)
           
            SpikesERes = net.E_idx[F_E[0,:]]          #If a neuron spikes x time-steps ago, activate post-syn 
            if len(SpikesERes ) > 0:
                gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
            F_I = np.all([[t-F[net.I_idx]==net.delays[net.I_idx]],[F[net.I_idx] != 0]],axis = 0,keepdims=0)
            SpikesIRes = net.I_idx[F_I[0,:]]        
            if len(SpikesIRes) > 0:
                gIn = gIn + np.multiply(net.GI,np.sum(net.W[:,SpikesIRes],axis=1))  #Increase the conductance of postsyn neurons
                     
            noise = np.random.normal(0,1,N)*gain_noise
            #Leaky Integrate-and-fire
            dV_res = ((net.VRest-V) + np.multiply(gEx,net.RE-V) + np.dot(w_res,input_res[:,t]) +
                      np.multiply(gIn,net.RI-V) + net.ITonic + noise)                                     #Compute raw voltage change
            V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
            #M[:,t] = V
    
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
            V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
            spikers = np.where(V > net.Theta)[0]
            F[spikers] = t                                                              #Update the last AP fired by the neuron
            V[spikers] = 0                                                             #Membrane potential at AP time
            tspike[ns:(ns+len(spikers)),:] = np.array([spikers,np.repeat(dt*t,len(spikers))]).T
            ns += len(spikers)
            total_gEx[t] = np.sum(gEx)
            total_gIn[t] = np.sum(gIn)
        spikes.append(tspike)
    return BPhi,spikes,output,total_gEx,total_gIn
    
#Sim params
random.seed(0)
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
GaussSD = 0.02

#Initialize network
mean_GE = 0.8#0.8
mean_GI = 2#3              #0.055 Conductance (0.001 = 1pS)1.5
mean_delays = 0.001/dt              # transmission delay (ms)
N = 1000
p_NI = 0.2
p = 0.1
G = 0.02
GaussW = 0
ITonic = 10
gain_noise = 0


#Simulation parameters
nb_epochs = 10
td = 0.02
tr = 0.002

#Input parameters
#gain_in_list = np.arange(0,5,0.5)
start_stim = 0.5
t_stim = int(start_stim/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
#p_in_list = np.arange(0,1,0.1)


#Target function
N_out = 1
sigma = 30
target = np.zeros(nt)
target[t_stim:] = lowpass_filter(np.random.randn(N_out,n_step_stim)*sigma,10,fs)
#plt.plot(np.ndarray.flatten(target[0,:]))

p_res = 0.3
gain_res = 25

osc_range = [5,7]           
N_osc = 3
    
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

scale_res = gain_res/np.sqrt((p_res*N))
w_res = scale_res*np.abs(np.multiply(np.random.normal(0,1,(N,N_osc)),
                                   np.random.rand(N,N_osc)<p_res))

osc_periods = np.random.uniform(osc_range[0],osc_range[1],N_osc)
input_res = np.zeros((N_osc,nt))
if N_osc > 0:
    N_in_net = int(np.round(p_res*N))
    scale_in = gain_res/((N*p_res)*N_osc)
    temp_w_in = np.random.normal(0,scale_in,(N_in_net,N_osc))  #Strength of connections to D
    w_in = np.zeros((N,N_osc))
    phase = np.random.uniform(0,1,N_osc)*np.pi 

    for inp_cell in range(N_osc):
        input_res[inp_cell,t_stim:t_stim+n_step_stim] =  (np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))) + 1)/2   
        exc_idx = np.random.choice(N,N_in_net)
        w_in[exc_idx,inp_cell] = temp_w_in[:,inp_cell]

#Training
BPhi = np.zeros(NE)
Pinv = np.eye(NE)*alpha
train = True             
BPhi,spikes,output,total_gEx,total_gIn = launch_simul(net,nt,train,
             input_res,w_res,target,BPhi,Pinv,nb_epochs,tr,td,step)
    
#Test
train = False            
BPhi,spikes,output,total_gEx,total_gIn = launch_simul(net,nt,train,
              input_res,w_res,target,BPhi,Pinv,1,tr,td,step)


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
