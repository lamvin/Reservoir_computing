# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:36:50 2018

@author: User1
"""
import numpy as np
import numpy
import sys
#sys.path.append('C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir')    #Path to working directory (no backslash)
import matplotlib.pyplot as plt
#import cPickle as pickle
import json
import pp
from datetime import datetime
from scipy import sparse
import os
import scipy
import multiprocessing as mp
import pickle

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def launch_simul(N,nt,tm,tr,td,vreset,vrest,vpeak,BIAS,OMEGA,dt,tref,w_in,
                 input_pattern,save_path,dir_i,dir_j,rep):
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
                #SP[ni].append(i)
                sparse_mat[ni,i] = 1
        tlast[index] = dt*i
        
        PSC = PSC*numpy.exp(-dt/tr) + h*dt
        h = h*numpy.exp(-dt/td) + numpy.divide(JD*(len(index)>0),(tr*td))
        r = r*numpy.exp(-dt/tr) + hr*dt
        hr = hr*numpy.exp(-dt/td) + numpy.divide(v>=vpeak,tr*td)
        
        v = v + numpy.multiply(30 - v,v>=vpeak)
        v = v + numpy.multiply(vreset-v,v>=vpeak)
        
    sparse_mat = scipy.sparse.csr_matrix(sparse_mat)
    scipy.sparse.save_npz("{}{}/{}/{}.npz".format(save_path,dir_i,dir_j,rep), sparse_mat)

#Sim params
startTime = datetime.now()
#output_path = "C:/Users/Cortex/Google Drive/Philippe/Python/spiking_reservoir/results/"
#output_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/results/"
output_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"
SP_data = {}
with_data = True

#Network parameters
N = 1000
dt = 5e-05
tref = 2e-03
tm = 0.01
vreset = -65
vrest = -65
vpeak = -40
td = 0.02
tr = 0.002
p = 0.1

#Simulation parameters
T = 2
nt = np.round(T/dt).astype(int)
Q = 10
G = 0.02
#G_list = [0.001,0.002,0.005,0.01,0.02,0.3,0.4,0.5]
#G_list = [0.04,0.05]
nb_repetitions = 100

#Input parameters
BIAS = 25
#gain_in_list = np.arange(0,10,1)
#gain_in_list = [0.5,1]
gain_in = 5000
osc_range = [5,7]
#N_in = 3
N_in_list = [1,2,3,4,5,7,10,15,20,30,40,50]
start_stim = 0.5
t_stim = int(start_stim/dt)
len_stim = T-start_stim
n_step_stim = int(len_stim/dt)
p_in_list = np.arange(0.1,1,0.1)
#p_in = 0.3

#Parallel init

modules_names = ('numpy','scipy.sparse',)

ppservers = ()
job_server = pp.Server(ppservers=ppservers)
print("Starting pp with " + str(job_server.get_ncpus()) + " workers")

#Initialization
nb_conditions1 = len(N_in_list)
nb_conditions2 = len(p_in_list)

data = {}
data['cond1_name'] = 'N_in'
data['cond2_name'] = 'p_in'
data['cond1'] = N_in_list
data['cond2'] = p_in_list
data['nb_steps'] = nt
data['dt'] = dt
data['nb_repetitions'] = nb_repetitions  
data['N'] = N
data['T'] = T
data['start_stim'] = start_stim
data['G'] = G

dir_name = data['cond1_name'] + '-' + data['cond2_name'] +'G02/'
save_path = output_path+dir_name
    
for cond_i in range(nb_conditions1):
    print('Running simulation {}/{}'.format(cond_i,nb_conditions1))
    N_in = N_in_list[cond_i]
    sparse_mat = sparse.lil_matrix((N,nt*nb_repetitions*nb_conditions2))
    for cond_j in range(nb_conditions2):
        print('Cond {}'.format(cond_j))
        ensure_dir('{}/{}/{}/'.format(save_path,cond_i,cond_j))
        data[cond_j] = {}
        p_in = p_in_list[cond_j]
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
        jobs = []
        for rep_i in range(nb_repetitions):
            jobs.append(job_server.submit(launch_simul,(N,nt,tm,tr,td,vreset,
                        vrest,vpeak,BIAS,OMEGA,dt,tref,w_in,input_pattern,
                        save_path,cond_i,cond_j,rep_i),modules=modules_names))
        [job() for job in jobs]
# =============================================================================
#         for (key, job) in enumerate(jobs):
#             sparse_mat[:,(nt*cond_j*nb_repetitions+(key*nt)):
#                 (nt*cond_j*nb_repetitions+((key+1)*nt))] = job()
# =============================================================================
    

with open( output_path + dir_name + 'metadata.p', "wb") as f:
    pickle.dump(data,f)
print(datetime.now() - startTime)

#pickle.dump(data, open( output_path + name, "wb" ))