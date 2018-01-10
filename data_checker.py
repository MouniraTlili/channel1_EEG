import numpy as np
import os 
import random
import scipy as sc
from scipy import io

# data = sc.io.loadmat('data/clean_data_0.mat')
# # original = mat = sc.io.loadmat('data/data_0.mat')

# train = data['train']
# test = data['test']
# test_range = data['test_range'] 
# train_range = data['train_range']

# full = np.empty((train_range.shape[0]+test_range.shape[0]))
# print full.shape
channel = 0
name = 'data/channel'+str(channel)+'_normalized.mat'
mat = sc.io.loadmat(name)
original  = np.load('data/channel'+str(channel)+'.npy')

not_normalized_test = np.take(original, mat['test_range'][0], axis = 0)
mat['unormalized_test'] = not_normalized_test

channel_mean = np.sum(original, axis=0)/float(original.shape[0])
channel_std = np.sqrt(np.sum(np.square(original - channel_mean), axis = 0)/float(original.shape[0]))


mat['channel_mean'] = channel_mean
mat['channel_std'] = channel_std
print mat.keys()
print not_normalized_test.shape
print original.shape

# sc.io.savemat(name, mat)

dictionary = sc.io.loadmat('data.mat')
print dictionary.keys()
# dictionary['raw_test0'] = not_normalized_test
# # dictionary['raw_test0']
# sc.io.savemat('data', dictionary)
