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
from scipy.stats.stats import pearsonr

def load_arrays(path,cond1,cond2,bin_size,sigma,T_keep,nb_bins_cut):
    with open('{}{}/{}/{}.p'.format(path,cond1,cond2,"readout"),'rb') as f:
        data = pickle.load(f)
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
    spikes = spikes[:,nb_bins_cut:]
    if sigma>0:
        spikes = scipy.ndimage.filters.gaussian_filter1d(spikes,sigma=sigma) 
    fr = numpy.sum(numpy.mean(spikes,axis=0))/T_keep
    return fr,spikes
    



#data_path = "C:/Users/User1/Google Drive/Philippe/Python/spiking_reservoir/results/"
data_path = "C:/Users/Cortex/Documents/Philippe/spiking_reservoir/"
#data_path = "C:/Users/one/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
#data_path = "C:/Users/two/Documents/Philippe Vincent-Lamarre/spiking_reservoir/results/"
sim_name = 'gain_in-G_task'
sim_path = data_path +'results/' +sim_name + '/'


with open(sim_path + 'metadata.p','rb') as f:
    metadata = pickle.load(f)


nb_cond1 = len(metadata['cond1'])
nb_cond2 = len(metadata['cond2'])
#nb_repetitions = metadata['nb_repetitions']
dt = metadata['dt']
N = 1000
T = 2
bin_size_s = 0.001   #in s
bin_size = int(bin_size_s/dt)
rm_start = 0.5
T_keep = T-rm_start
nb_bins_cut = np.ceil(rm_start/bin_size_s).astype(int)

results = {}
width= 15
height = 8
sigma_s = 0.003
sigma = sigma_s/bin_size_s

results = {}
corr_mat = np.zeros((nb_cond1,nb_cond2))
for cond_i in range(nb_cond1):
    print('Loading condition {}.'.format(cond_i+1))
    results[cond_i] = {}
    for cond_j in range(nb_cond2):
        with open('{}{}/{}/{}.p'.format(sim_path,cond_i,cond_j,"readout"),'rb') as f:
            ro = pickle.load(f)
            results[cond_i][cond_j] = np.stack((ro['output'][-1],ro['target'])) 
            corr_mat[cond_i,cond_j] = np.corrcoef(ro['output'][-1],ro['target'])[0,1]  
        

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

#Variance
plt.figure(figsize=(12,6))
plt.pcolor(total_var)
ax = plt.axes()
plt.ylabel(metadata['cond1_name'])
plt.xlabel(metadata['cond2_name'])
plt.title('Average variance')
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




