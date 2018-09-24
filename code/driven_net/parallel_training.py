# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import numpy
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
        
def lowpass_filter(data, cutoff, fs):
    order = 3
    nyq  = fs/2.0
    cutoff = cutoff/nyq
    b, a = iirfilter(order, cutoff, btype='lowpass',
                     analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data    

def launch_simul(N,nt,tm,tr,td,vreset,vrest,vpeak,BIAS,alpha,OMEGA,dt,tref,w_in,
                 input_pattern,save_path,dir_i,dir_j,target,nb_tests,
                 nb_repetitions,train_start):
    
    #Setting up plot and recording variables
    output = numpy.zeros((nb_repetitions+nb_tests,nt))
    Pinv =numpy.eye(N)*alpha
    BPhi = numpy.zeros(N)
    data = {}
    for rep_i in range(nb_repetitions+nb_tests):
        #Simulation variables
        PSC = numpy.zeros(N)
        h = numpy.zeros(N)
        r = numpy.zeros(N)
        hr = numpy.zeros(N)
        JD = 0*PSC
        #SP = {k:[] for k in range(N)}  
        sparse_mat = scipy.sparse.lil_matrix((N,nt))
        #sparse_mat = numpy.zeros((N,nt))
        tlast = numpy.zeros(N)
        #REC = numpy.zeros((nt,10))
        v = vreset +numpy.random.normal(15,6,(N))
        for i in range(nt):
            I = PSC + BIAS
            dv = numpy.multiply((dt*i>(tlast + tref)),((vrest-v)+I+ numpy.dot(w_in,input_pattern[:,i])) /tm)
            v = v+(dt*dv)
            
            index = numpy.where(v>=vpeak)[0]
            
            if len(index)>0:
                JD = numpy.sum(OMEGA[:,index],axis=1)
                for ni in index:
                    sparse_mat[ni,i] = 1
            tlast[index] = dt*i
            
            PSC = PSC*numpy.exp(-dt/tr) + h*dt
            h = h*numpy.exp(-dt/td) + numpy.divide(JD*(len(index)>0),(tr*td))
            r = r*numpy.exp(-dt/tr) + hr*dt
            hr = hr*numpy.exp(-dt/td) + numpy.divide(v>=vpeak,tr*td)
            
            #Implement RLMS with the FORCE method
            z = numpy.dot(BPhi.T,r)
            err = z-target[i]
    
            #RLMS
            if rep_i < nb_repetitions:
                if i >= train_start:
                    if i%50 == 1:
                        cd = numpy.dot(Pinv,r)
                        BPhi = BPhi-(cd*err)
                        Pinv = Pinv - numpy.divide(numpy.outer(cd,cd.T),1 + numpy.dot(r.T,cd))

            output[rep_i,i] = z    
            v = v + numpy.multiply(30 - v,v>=vpeak)
            v = v + numpy.multiply(vreset-v,v>=vpeak)
            
        sparse_mat = scipy.sparse.csr_matrix(sparse_mat)
        scipy.sparse.save_npz("{}{}/{}/{}.npz".format(save_path,dir_i,dir_j,rep_i), sparse_mat)
    data['output'] = output
    data['BPhi'] = BPhi
    data['target'] = target
    with open("{}{}/{}/{}".format(save_path,dir_i,dir_j,"readout.p"),'wb') as f:
        pickle.dump(data,f)

#Sim params
startTime = datetime.now()
#output_path = "C:/Users/Cortex/Google Drive/Philippe/Python/spiking_reservoir/results/"
#output_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/training/results/"
#output_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
output_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"

#Network parameters
N = 1000
dt = 5e-05
fs = np.int(1/dt)
tref = 2e-03
tm = 0.01
vreset = -65
vrest = -65
vpeak = -40
td = 0.02
tr = 0.002
p = 0.1
#G = 0.02
G_list = [0,0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.2,0.3]

#Simulation parameters
T = 2
nt = np.round(T/dt).astype(int)
nb_repetitions = 20
nb_tests = 1

#Input parameters
BIAS = 25
#gain_in = 5000
gain_in_list = np.arange(0,20000,2000)
osc_range = [5,7]
N_in = 3
start_stim = 0.5
t_stim = int(start_stim/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
p_in = 0.25

#training params
alpha = dt*0.1
step = 50
train_start = int(np.round(start_stim/dt))

#Parallel init
modules_names = ('numpy','scipy.sparse',)

ppservers = ()
job_server = pp.Server(ppservers=ppservers)
print("Starting pp with " + str(job_server.get_ncpus()) + " workers")

#Initialization
nb_cond1 = len(gain_in_list)
nb_cond2 = len(G_list)

#Initialization
data = {}
data['cond1_name'] = 'gain_in'
data['cond2_name'] = 'G_in'
data['cond1'] = gain_in_list
data['cond2'] = G_list
data['nb_steps'] = nt
data['dt'] = dt
data['nb_repetitions'] = nb_repetitions  
data['N'] = N
data['T'] = T
data['start_stim'] = start_stim
#data['G'] = G

dir_name = data['cond1_name'] + '-' + data['cond2_name'] +'_task/'
save_path = output_path+dir_name
    
jobs = {}
for cond_i in range(nb_cond1):
    #print('Running simulation {}/{}'.format(cond_i,nb_cond1))
    gain_in = gain_in_list[cond_i]
    jobs[cond_i] = {}
    for cond_j in range(nb_cond2):
        #print('Cond {}'.format(cond_j))
        ensure_dir('{}/{}/{}/'.format(save_path,cond_i,cond_j))
        data[cond_j] = {}
        G = G_list[cond_j]
        OMEGA = G*np.multiply(np.random.normal(0,1,(N,N)),np.random.rand(N,N)<p)/(np.sqrt(N*p))
        osc_periods = np.random.uniform(osc_range[0],osc_range[1],N_in)
        input_pattern = np.zeros((N_in,nt))
        if N_in > 0:
            N_in_net = int(np.round(p_in*N))
            scale_in = gain_in/(N_in_net*N_in)
            temp_w_in = np.random.normal(0,scale_in,(N_in_net,N_in))  #Strength of connections to D
            w_in = np.zeros((N,N_in))
            phase = np.random.uniform(0,1,N_in)*np.pi 
    
            for inp_cell in range(N_in):
                input_pattern[inp_cell,t_stim:t_stim+n_step_stim] =  (np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))) + 1)/2   
                #input_pattern[inp_cell,t_stim:t_stim+n_step_stim] =  (np.sin(2*np.pi*osc_periods[inp_cell]*(np.linspace(phase[inp_cell],len_stim+phase[inp_cell],n_step_stim))))   
                exc_idx = np.random.choice(N,N_in_net)
                w_in[exc_idx,inp_cell] = temp_w_in[:,inp_cell]
        #Target function
        sigma = 30
        target = np.zeros(nt)
        target[t_stim:] = lowpass_filter(np.random.randn(n_step_stim)*sigma,10,fs)
        
        jobs[cond_i][cond_j] = job_server.submit(launch_simul,(N,nt,tm,tr,td,vreset,
                        vrest,vpeak,BIAS,alpha,OMEGA,dt,tref,w_in,input_pattern,
                        save_path,cond_i,cond_j,target,nb_tests,nb_repetitions,
                        train_start),modules=modules_names)

[jobs[cond_i][cond_j]() for cond_i,cond_j in itertools.product(
        np.arange(nb_cond1),np.arange(nb_cond2))]
    

with open( output_path + dir_name + 'metadata.p', "wb") as f:
    pickle.dump(data,f)
print(datetime.now() - startTime)

#pickle.dump(data, open( output_path + name, "wb" ))