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
import scipy.ndimage.filters 
import os
from datetime import datetime
import pp
import network
from scipy.stats.stats import pearsonr

def spks_to_conductance(net,dt,spikes):
    gEx = np.zeros(N)                                            #Conductance of excitatory neurons
    total_gEx = np.zeros(spikes.shape[1])
    for t in range(spikes.shape[1]):
        gEx = np.multiply(gEx,np.exp(-dt/net.TauE))
        SpikesERes = net.E_idx[spikes[:,t]] 
        if len(SpikesERes ) > 0:
            gEx = gEx + np.multiply(net.GE,np.sum(net.W[:,SpikesERes],axis=1))  #Increase the conductance of postsyn neurons
        total_gEx[t] = np.sum(gEx)
    return total_gEx

data_path = "C:/Users/User1/Documents/Projects/Reservoir_computing/data/"
#data_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"
#data_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
#data_path = "C:/Users/two/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
sim_name = 'N-GI_osc'
sim_path = data_path +sim_name + '/'


with open(sim_path + 'metadata.p','rb') as f:
    metadata = pickle.load(f)


nb_cond1 = len(metadata['cond1'])
nb_cond2 = len(metadata['cond2'])
nb_repetitions = metadata['nb_repetitions']
nb_epochs = metadata['nb_epochs']
dt = metadata['dt']
N = 1000
T = 2
bin_size_s = 0.001   #in s
bin_size = int(bin_size_s/dt)
rm_start = 0.5
T_keep = T-rm_start
nb_bins_cut = np.ceil(rm_start/bin_size_s).astype(int)
start_stim = 1
t_stim = int(start_stim/dt)

results = {}
width= 15
height = 8
sigma_s = 0.003
sigma = sigma_s/bin_size_s

results = {}
corr_mat = np.zeros((nb_cond1,nb_cond2))
all_g = np.zeros((nb_cond1,nb_cond2,nb_repetitions,nb_epochs))
for cond_i in range(nb_cond1):
    print('Loading condition {}.'.format(cond_i+1))
    results[cond_i] = {}
    for cond_j in range(nb_cond2):
        for rep_i in range(nb_repetitions):
            with open("{}{}/{}/{}/{}".format(sim_path,cond_i,cond_j,rep_i,"net.p"),'rb') as f:
                data = pickle.load(f)
            total_gEx = data['total_gEx']
            avg_total_gEx = np.mean(total_gEx,axis=0)
            ss_traces = np.mean([np.sum(np.square(total_gEx[x,:]-avg_total_gEx))/total_gEx.shape[1] for x in range(nb_epochs)])
            
            fit_signal = avg_total_gEx[t_stim:]
            ps = np.abs(np.fft.fft(fit_signal))**2

            freqs = np.fft.fftfreq(fit_signal.size, dt)
            idx = np.where((freqs>0) & (freqs<30))[0]
            peakfreq = freqs[idx][np.argmax(ps[idx])]
            
            fr = np.zeros(nb_epochs)
            for ep_i in range(nb_epochs):
                with open('{}{}/{}/{}/{}.npz'.format(sim_path,cond_i,cond_j,rep_i,ep_i),'rb') as f:
                    spikes = sp.sparse.load_npz(f)
                fr[ep_i] = np.mean(np.mean(spikes[:,t_stim:],axis=0))/dt
            avg_fr = np.mean(fr)
            fig = plt.figure(figsize=(width,height))
            [plt.plot(total_gEx[x,:]) for x in range(total_gEx.shape[0])]
            plt.annotate('MSE: {}\n Peak frequency: {} \n FR: {}'.format(*np.round([ss_traces,peakfreq,avg_fr],1)),
                         xy=(0.75, 0.9),xycoords='figure fraction',horizontalalignment='left',
                         verticalalignment='top', fontsize=10)

        
total_gEx = data ['total_gEx']



#Firing rate
plt.figure(figsize=(15,8))
plt.pcolor(corr_mat)
ax = plt.axes()
plt.ylabel(metadata['cond1_name'])
plt.xlabel(metadata['cond2_name'])
plt.title('Pearson R output/target')
plt.xticks(np.arange(nb_cond2),np.round(metadata['cond2'],2))
plt.yticks(np.arange(nb_cond1),np.round(metadata['cond1'],2))
plt.colorbar()


#==============================================================================
# file_name = files_list[cond_i]
# with open('{}{}/{}'.format(data_path,sim_name,file_name),'rb') as f:
#     data = pickle.load(f)
# nb_cond2 = len(data['cond_2'])
# total_var = np.zeros((nb_cond1,nb_cond2))
# nb_repetitions = data['nb_repetitions']
# dt = data['dt']
# nb_steps = data['nb_steps']
# results[cond_i] = {}
# for cond_j in range(nb_cond2):
#     results[cond_i][cond_j] = {}
#     condition = data[cond_i][cond_j]
#     ds_factor = bin_size*(1/dt)     #bin_size in ms
#     fr_conditions = [downsample(condition[rep_i],ds_factor,dt,nb_steps) for rep_i in range(nb_repetitions)]
#     conv_spikes = np.array([convolve_raster(fr_conditions[rep_i][:,nb_bins_cut:],sigma,dt,nb_steps) for rep_i in range(nb_repetitions)])
#     results[cond_i]['std_conv'] =  np.std(conv_spikes,axis=0)
#     results[cond_i]['avg_conv'] =  np.mean(conv_spikes,axis=0)
#     total_var[cond_i,cond_j] = np.mean(np.mean(results[cond_i]['std_conv'],axis=1))
#==============================================================================
        

#==============================================================================
#     fig = plt.figure(figsize=(width,height))
#     plt.pcolor(results[cond_i]['avg_conv'])
#     plt.ylabel('Cell #')
#     plt.xlabel('Time (ms)')
#     x_ticks = np.arange(0,results[cond_i]['avg_conv'].shape[1],2)
#     plt.xticks(x_ticks,(x_ticks*bin_size*1000).astype(int))
#     plt.title(sim_name.split('.')[0])
#==============================================================================




