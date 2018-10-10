# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import sys
#sys.path.append("C:/Users/User1/Documents/Projects/Reservoir_computing/code/embedded_net/")
import network
import matplotlib.pyplot as plt
import pp
from datetime import datetime
from scipy import sparse
import os
import scipy
import pickle
from scipy.signal import iirfilter, lfilter
import itertools

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def launch_simul(net,nt,dt,input_res,save_path,dir_i,dir_j,rep_i,
                 nb_epochs):
    
    #Setting up plot and recording variables
    N = net.N
    data = {}
    for ep_i in range(nb_epochs):
        #Simulation variables        
        gEx = np.zeros(N)                                            #Conductance of excitatory neurons
        gIn = np.zeros(N)                                            #Conductance of excitatory neurons
        F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
        V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
        #Recording variables
        total_gEx = np.zeros(nt)
        total_gIn = np.zeros(nt)
        sparse_mat = scipy.sparse.lil_matrix((N,nt))
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
                     
            #Leaky Integrate-and-fire
            dV_res = ((net.VRest-V) + np.multiply(gEx,net.RE-V) + np.dot(net.w_res,input_res[:,t]) +
                      np.multiply(gIn,net.RI-V) + net.ITonic)                                     #Compute raw voltage change
            V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau

            #Update cells
            Refract = t <= (F + net.Refractory)
            V[Refract] = net.VRest[Refract]                                           #Hold resting potential of neurons in refractory period            
            spikers = np.where(V > net.Theta)[0]
            F[spikers] = t                                                              #Update the last AP fired by the neuron
            V[spikers] = 0                                                             #Membrane potential at AP time
            sparse_mat[spikers,t] = 1 
            total_gEx[t] = np.sum(gEx)
            total_gIn[t] = np.sum(gIn)
            
        sparse_mat = scipy.sparse.csr_matrix(sparse_mat)
        scipy.sparse.save_npz("{}{}/{}/{}/{}.npz".format(save_path,dir_i,dir_j,rep_i,ep_i), sparse_mat)
    with open("{}{}/{}/{}/{}".format(save_path,dir_i,dir_j,rep_i,"net.p"),'wb') as f:
        pickle.dump(data,f)

#Sim params
startTime = datetime.now()
#output_path = "C:/Users/Cortex/Google Drive/Philippe/Python/spiking_reservoir/results/"
#output_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/training/results/"
#output_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
#output_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"
output_path = "C:/Users/two/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"

#Network parameters
#N = 300
N_list = [100,300,500,1000]
pNI = 0.2
dt = 5e-05
mean_delays = 0.001/dt
mean_GE = 0.8#0.8
#mean_GI = 2#3              #0.055 Conductance (0.001 = 1pS)1.5
mean_GI_list = np.arange(0.5,4,0.25)
fs = np.int(1/dt)
tref = 2e-03/dt
p = 1
ITonic = 9
G = 0.04
#G_list = [0,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1]
#G_list = [0.02,0.04]
#Simulation parameters
T = 21
nt = np.round(T/dt).astype(int)
nb_epochs = 10
nb_repetitions = 10

#Input parameters
#gain_in = 5000
#gain_in_list = np.arange(0,50,10)
gain_in_list = [10,25]
start_stim = 1
t_stim = int(start_stim/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
p_in = 1

#Parallel init
modules_names = ('scipy.sparse','network','numpy as np',)

ppservers = ()
job_server = pp.Server(ppservers=ppservers)
print("Starting pp with " + str(job_server.get_ncpus()) + " workers")

#Initialization
nb_cond1 = len(gain_in_list)
nb_cond2 = len(G_list)

#Initialization
data = {}
data['cond1_name'] = 'N'
data['cond2_name'] = 'GI'
data['cond1'] = N_list
data['cond2'] = mean_GI_list
data['nb_steps'] = nt
data['dt'] = dt
data['nb_epochs'] = nb_epochs 
#data['N'] = N
data['T'] = T
data['nb_repetitions'] = nb_repetitions
#data['G'] = G

dir_name = data['cond1_name'] + '-' + data['cond2_name'] +'_osc/'
save_path = output_path+dir_name
    
jobs = {}
for cond_i in range(nb_cond1):
    #print('Running simulation {}/{}'.format(cond_i,nb_cond1))
    gain_in = gain_in_list[cond_i]
    jobs[cond_i] = {}
    for cond_j in range(nb_cond2):
        #print('Cond {}'.format(cond_j))
        for rep_i in range(nb_repetitions):
            ensure_dir('{}/{}/{}/{}/'.format(save_path,cond_i,cond_j,rep_i))
            data[cond_j] = {}
            G = G_list[cond_j]
            net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, mean_TauFall_I =0.06, ITonic=ITonic)
            input_res = np.zeros(nt)
            N_in_net = int(np.round(p_in*net.N))
            scale_in = gain_in/np.sqrt(N_in_net)
            w_res = scale_in*np.random.normal(0,1,N_in_net)  #Strength of connections to D
            w_res = np.abs(scale_in*np.multiply(np.random.normal(0,1,N),np.random.rand(N)<p_in))
            net.w_res = w_res
            input_res[t_stim:t_stim+n_step_stim] =  1      
            jobs[cond_i][cond_j] = job_server.submit(launch_simul,(net,nt,dt,input_res,save_path,cond_i,cond_j,rep_i,
                     nb_epochs),modules=modules_names)

[jobs[cond_i][cond_j]() for cond_i,cond_j in itertools.product(
        np.arange(nb_cond1),np.arange(nb_cond2))]
    

with open( output_path + dir_name + 'metadata.p', "wb") as f:
    pickle.dump(data,f)
print(datetime.now() - startTime)

#pickle.dump(data, open( output_path + name, "wb" ))