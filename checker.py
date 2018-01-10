import numpy as np
import os
import random
import scipy as sc
from scipy import io

data = sc.io.loadmat('data/clean_data_0.mat')
# data = sc.io.loadmat('data/data_0.mat')

train = data['train']
test = data['test']
test_range = data['test_range'][0]
train_range = data['train_range'][0]
test_range.sort()
train_range.sort()
full = np.empty((len(train_range)+len(test_range), train.shape[1]))
next_train = 0
next_test = 0
for i in range(full.shape[0]):
    if i in test_range:
		full[i] = test[next_test]
		next_test += 1
    else:
		full[i] = train[next_train]
		next_train += 1
 
print "train range 0", train_range[0]
# rebuilt = np.load("rebuilt.npy")
channel0 = np.load("data/clean_channel0.npy")

# print channel0.shape, rebuilt.shape
print channel0[0][0], train[0][0]
# print train[0][0]
# print np.where(full == -0.247786820068)
print np.shape(np.unique(np.where(channel0-full == 0)[0]))

# print train_range
# train2 = np.take(channel0, train_range, axis = 0)
# print train2 - train
# print channel0[0][0],train2[0][0]

# print rebuilt - channel0
