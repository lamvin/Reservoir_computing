# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import numpy
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
from scipy.stats import norm

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
def lowpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='lowpass',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def launch_simul(net,nt,alpha,dt,input_res,save_path,dir_i,dir_j,rep_i,target,nb_tests,step,
                 nb_epochs,train_start):
    
    #Setting up plot and recording variables
    N = net.N
    output = np.zeros((nb_epochs+nb_tests,nt))
    Pinv = np.eye(net.NE)*alpha
    BPhi = np.zeros(net.NE)
    data = {}
    total_iEx = np.zeros((nb_epochs+nb_tests,nt))
    total_iIn = np.zeros((nb_epochs+nb_tests,nt))
    total_iInp = np.zeros((nb_epochs+nb_tests,nt))
    for ep_i in range(nb_epochs+nb_tests):
        #Simulation variables
        r = np.zeros(net.NE)
        hr = np.zeros(net.NE)                       
        gEx = np.zeros(N)                                            #Conductance of excitatory neurons
        gIn = np.zeros(N)                                            #Conductance of excitatory neurons
        F = np.full(N,np.nan)                                               #Last spike times of each inhibitory cells
        V = np.random.normal(net.mean_VRest,abs(0.04*net.mean_VRest),N)   #Set initial voltage Exc 
        #Recording variables
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
            E_current = np.multiply(gEx,net.RE-V)
            I_current = np.multiply(gIn,net.RI-V)
            Inp_current = np.dot(net.w_res,input_res[:,t])
            dV_res = ((net.VRest-V) + E_current + I_current + Inp_current + net.ITonic)                                     #Compute raw voltage change
            V = V + (dV_res * (dt/net.tau))                                        #Update membrane potential based on tau
            #M[:,t] = V
    
            r = r*np.exp(-dt/net.tr) + hr*dt
            hr = hr*np.exp(-dt/net.td) + np.divide(V[net.E_idx]>=net.Theta[net.E_idx],net.tr*net.td)        
            z = np.dot(BPhi.T,r)
            output[ep_i,t] = z
            err = z-target[t]
                    
            #RLMS
            if t >= train_start and ep_i<nb_epochs:
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
            sparse_mat[spikers,t] = 1 
            total_iEx[ep_i,t] = np.sum(E_current)
            total_iIn[ep_i,t] = np.sum(I_current)
            total_iInp[ep_i,t] = np.sum(Inp_current)
            
        sparse_mat = scipy.sparse.csr_matrix(sparse_mat)
        scipy.sparse.save_npz("{}{}/{}/{}/{}.npz".format(save_path,dir_i,dir_j,rep_i,ep_i), sparse_mat)
    data['output'] = output
    data['BPhi'] = BPhi
    data['target'] = target
    data['net'] = net
    data['total_iEx'] = total_iEx
    data['total_iIn'] = total_iIn
    data['total_iInp'] = total_iInp
    data['input_res'] = input_res
    with open("{}{}/{}/{}/{}".format(save_path,dir_i,dir_j,rep_i,"readout.p"),'wb') as f:
        pickle.dump(data,f)

#Sim params
startTime = datetime.now()
#output_path = "C:/Users/Cortex/Google Drive/Philippe/Python/spiking_reservoir/results/"
output_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/training/results/"
#output_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
#output_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"

#Network parameters
N = 1000
pNI = 0.2
dt = 5e-05
mean_delays = 0.001/dt
mean_GE = 1#0.8
mean_GI = 6#3              #0.055 Conductance (0.001 = 1pS)1.5
fs = np.int(1/dt)
tref = 2e-03/dt
p = 0.1
ITonic = 9
td = 0.02
tr = 0.002
G = 0.02
#G_list = [0,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1]
#G_list = [0.02,0.04]
#Simulation parameters
#T = 2
nb_epochs = 10
nb_repetitions = 1
nb_tests = 1

#Input parameters
gain_in = 25
#gain_in_list = np.arange(0,70,10)
#gain_in_list = [10,25]
osc_range = [5,7]
N_in = 3
start_stim = 0.5
t_stim = int(start_stim/dt)

p_in = 0.25

#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))

peak_width = int(0.01/dt)
ready_level = 0.2
#peak_level = 0.5
peak_level_list = [5]
#peak_time = int(0.35/dt)
peak_T_list = [1]
T_buffer = 1

#Parallel init
modules_names = ('scipy.sparse','network','numpy as np',)

ppservers = ()
job_server = pp.Server(ppservers=ppservers)
print("Starting pp with " + str(job_server.get_ncpus()) + " workers")

#Initialization
nb_cond1 = len(peak_T_list)
nb_cond2 = len(peak_level_list)

#Initialization
data = {}
data['cond1_name'] = 'peak_T'
data['cond2_name'] = 'peak_level'
data['cond1'] = peak_T_list
data['cond2'] = peak_level_list
#data['nb_steps'] = nt
data['dt'] = dt
data['nb_epochs'] = nb_epochs 
data['N'] = N
#data['T'] = T
data['start_stim'] = start_stim
data['nb_repetitions'] = nb_repetitions
data['nb_tests'] = nb_tests
#data['G'] = G

save_path = '../../data/demo_results/'
    
jobs = {}
for cond_i in range(nb_cond1):
    #print('Running simulation {}/{}'.format(cond_i,nb_cond1))
    peak_T = peak_T_list[cond_i]
    T = peak_T+T_buffer + start_stim   
    nt = np.round(T/dt).astype(int)
    peak_time = int((peak_T + start_stim)/dt)
    len_stim = T-start_stim
    n_step_stim = int(len_stim/dt)
    
    jobs[cond_i] = {}
    for cond_j in range(nb_cond2):
        #print('Cond {}'.format(cond_j))
        jobs[cond_i][cond_j] = {}
        for rep_i in range(nb_repetitions):
            ensure_dir('{}/{}/{}/{}/'.format(save_path,cond_i,cond_j,rep_i))

            peak_level = peak_level_list[cond_j]
            net = network.net(N,pNI,mean_delays,tref,G=G,p=p, mean_GE = mean_GE, mean_GI = mean_GI, ITonic=ITonic)
            net.tr = tr
            net.td = td
            osc_periods = np.random.uniform(osc_range[0],osc_range[1],N_in)
            input_res = np.zeros((N_in,nt))
            if N_in > 0:
                N_in_net = int(np.round(p_in*net.NE))
                scale_in = gain_in/np.sqrt((N_in_net*N_in))
                temp_w_in = scale_in*np.random.normal(0,1,(N_in_net,N_in))  #Strength of connections to D
                temp_w_in = np.abs(scale_in*np.multiply(np.random.normal(0,1,(net.NE,N_in)),np.random.rand(net.NE,N_in)<p_in))
                w_res = np.zeros((N,N_in))
                w_res[net.E_idx,:] = temp_w_in
                net.w_res = w_res
                phase = np.random.uniform(0,1,N_in)*np.pi 

                for inp_cell in range(N_in):
                    input_res[inp_cell,t_stim:t_stim+n_step_stim] =  (np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))) + 1)/2   
                    #input_pattern[inp_cell,t_stim:t_stim+n_step_stim] =  (np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))))   
                    exc_idx = np.random.choice(N,N_in_net)
            #Target function
            pdf = norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), int(peak_width*2)))
            pdf = (pdf-min(pdf))/(max(pdf)-min(pdf))
            target = np.ones(nt)*ready_level
            target[peak_time-peak_width:peak_time+peak_width] = target[peak_time-peak_width:peak_time+peak_width] + (peak_level-ready_level)*pdf
            jobs[cond_i][cond_j][rep_i] = job_server.submit(launch_simul,(net,nt,alpha,dt,input_res,save_path,cond_i,cond_j,rep_i,target,nb_tests,
                     step,nb_epochs,train_start),modules=modules_names)

[jobs[cond_i][cond_j][rep_i]() for cond_i,cond_j,rep_i in itertools.product(
        np.arange(nb_cond1),np.arange(nb_cond2),np.arange(nb_repetitions))]
    

with open( save_path + 'metadata.p', "wb") as f:
    pickle.dump(data,f)
print(datetime.now() - startTime)

#pickle.dump(data, open( output_path + name, "wb" ))