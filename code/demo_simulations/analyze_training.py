# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:07:55 2017

@author: User1
"""
import pickle
import numpy as np
import numpy
import scipy as sp
from matplotlib import pyplot as plt
import sys
#sys.path.append( "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/")    #Path to working directory (no backslash)
import scipy.ndimage.filters 
import os
from datetime import datetime
import pp
import network

def load_arrays(path,cond1,cond2,rep,ep_i,bin_size,sigma):
    with open('{}{}/{}/{}/{}.npz'.format(path,cond1,cond2,rep,ep_i),'rb') as f:
        array = scipy.sparse.load_npz(f)
    array = array.todense()
    N,nb_steps = array.shape
    if bin_size>1:
        nb_steps_ds = int(numpy.ceil(nb_steps/bin_size))
        spikes = numpy.zeros((N,nb_steps_ds))
        for bin_i in range(nb_steps_ds):
            end_i = bin_size*(bin_i+1)
            if end_i > nb_steps:
                end_i = nb_steps
            spikes[:,bin_i] = numpy.ndarray.flatten(numpy.sum(array[:,(bin_size*bin_i):end_i],axis=1)[:,0])
    else:
        spikes = array

    if sigma>0:
        spikes = scipy.ndimage.filters.gaussian_filter1d(spikes,sigma=sigma) 
    fr = numpy.sum(numpy.mean(spikes,axis=0))
    return fr,spikes
    



data_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/results/"
#data_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"
#data_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
#data_path = "C:/Users/two/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
sim_path = '../../data/demo_results/'


with open(sim_path + 'metadata.p','rb') as f:
    metadata = pickle.load(f)


nb_cond1 = len(metadata['cond1'])
nb_cond2 = len(metadata['cond2'])
nb_repetitions = metadata['nb_repetitions']
nb_epochs = metadata['nb_epochs']
nb_tests = 1
dt = metadata['dt']
N = metadata['N']
start_stim = metadata['start_stim']
t_stim = int(start_stim/dt)
#nt = metadata['nb_steps']
T = 2
bin_size_s = 0.001   #in s
bin_size = int(bin_size_s/dt)


results = {}
width= 15
height = 8
sigma_s = 0.02
sigma = sigma_s/bin_size_s

results = {}
total_var = np.zeros((nb_cond1,nb_cond2))
avg_rate = np.zeros((nb_cond1,nb_cond2))
for cond_i in range(nb_cond1):
    print('Loading condition {}.'.format(cond_i+1))
    results[cond_i] = {}
    for cond_j in range(nb_cond2):
        results[cond_i][cond_j] = {}
        jobs = []
        for rep_i in range(nb_repetitions):
            data= []
            for ep_i in range(nb_epochs):
                fr,spikes = load_arrays(sim_path,cond_i,cond_j,rep_i,ep_i,
                            bin_size,sigma)
                data.append(spikes)
            data= np.array(data)
            test_fr,test_spikes = load_arrays(sim_path,cond_i,cond_j,rep_i,nb_epochs,
                            bin_size,sigma)
            with open('{}{}/{}/{}/{}.p'.format(sim_path,cond_i,cond_j,rep_i,"readout"),'rb') as f:
                ro = pickle.load(f)


output = ro['output'][-1]
target = ro['target']
input_res = ro['input_res']
total_iIn = ro['total_iIn']
total_iEx = ro['total_iEx']
total_iInp = ro['total_iInp']
BPhi = ro['BPhi'] 

width = 15
height = 8

avg_data = np.mean(data,axis=0)
std_data = np.std(data,axis=0)

avg_std = np.mean(std_data,axis=0)
fr = np.mean(avg_data,axis=0)

plt.figure(figsize=(width,height))
plt.title('Weights')
plt.plot(BPhi,'k.')

plt.figure(figsize=(width,height))
plt.subplot(4,1,1)
plt.plot(target,'k')
plt.plot(output,'r')
plt.subplot(4,1,2)
plt.plot(fr)
plt.title('Firing rate')
plt.subplot(4,1,3)
plt.title('Variance')
plt.plot(avg_std)
plt.subplot(4,1,4)
[plt.plot(input_res[x,:]) for x in range(input_res.shape[0])]

t_stim_ds = int(t_stim/bin_size)
fit_signal = fr[t_stim_ds:]
ps = np.abs(np.fft.fft(fit_signal))**2
freqs = np.fft.fftfreq(fit_signal.size, bin_size_s)
idx = np.where((freqs>0) & (freqs<30))[0]
peakfreq = freqs[idx][np.argmax(ps[idx])]

plt.figure(figsize=(width,height))
plt.subplot(1,2,1)
plt.plot(freqs[idx],ps[idx])
plt.subplot(1,2,2)
plt.plot(total_iIn[-1,:],'b',label='I')
plt.plot(total_iEx[-1,:],'r',label='E')
plt.plot(total_iInp[-1,:],'g',label='Inp')
plt.plot(total_iEx[-1,:]+total_iIn[-1,:],'k',label='Balance')
plt.legend()

