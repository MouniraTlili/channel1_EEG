import numpy as np
import os 
import random
import scipy as sc
from scipy import io

ch0 = np.load('data/channel0_normalized.npy')
ch32 = np.load('data/channel32_normalized.npy')
multi = np.empty((ch0.shape[0], ch0.shape[1]+ch32.shape[1]))
multi[:,:ch0.shape[1]] = ch0
multi[:,ch0.shape[1]:] = ch32


data_len = ch0.shape[0]
print data_len
train_range = sc.io.loadmat('data/channel0_normalized.mat')['train_range'][0]
# train_range = random.sample(range(0, data_len), 3*data_len/4)
train_range.sort()
test_range = []
for j in range(data_len):
	if j in train_range:
		continue
	else:
		test_range.append(j)


data = multi
# data = ch0
# data = ch32
train = np.take(data, train_range, axis = 0)
test = np.take(data, test_range, axis = 0)
print train.shape
dictionnary = {}
dictionnary['train'] = train
dictionnary['test'] = test
dictionnary['test_range'] = test_range
dictionnary['train_range'] = train_range
sc.io.savemat('data/channel0_32_normalized',dictionnary)