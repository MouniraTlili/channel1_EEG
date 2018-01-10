import numpy as np

import os 
import random
import scipy as sc
from scipy import io



# users_data = np.empty((32,40*8064))
# for i in range(len(files)):
# 	patient_i = sc.io.loadmat('data_preprocessed_matlab/'+files[i])['data']
# 	patient_i = np.reshape(patient_i[:,0, :], (1,40*8064))
# 	users_data[i,:] = patient_i
# 	print patient_i.shape
# data_dict = {}
# data_dict['channel1_all_users'] = users_data





# files = filter(lambda x: ".mat" in x, os.listdir('data/data_preprocessed_matlab/'))
# files.sort()
# print files
# skip_patients = [1,7,9,3,14,21,29]
# channel_num =0
# channel = np.empty(((32-len(skip_patients))*40,8064))
# j = -1
# for i in range(len(files)):
# 	if i in skip_patients:
# 		continue
# 	j = j+1
# 	print 'loading patient', files[i]
# 	patient_i = sc.io.loadmat('data/data_preprocessed_matlab/'+files[i])['data']
# 	print patient_i.shape, 'range',40*i,40*(i+1)
# 	channel[40*j:40*(j+1), :] = patient_i[:, channel_num, :]

# channel = np.reshape(channel, ((32-len(skip_patients))*40*9, 8064/9))
# data_dict = {}
# data_dict['channel'+str(channel_num)] = channel
# sc.io.savemat('channel'+str(channel_num),data_dict)
# np.save('channel'+str(channel_num), channel)

import h5py
skip_patients = [1,7,9,3,14,21,29]
files = filter(lambda x: ".mat" in x, os.listdir('data/Cleaned/'))
files.sort()
channel_num =0
channel = np.empty(((32-len(skip_patients))*40,8064))
j = -1
for i in range(len(files)):
	if i in skip_patients:
		continue
	j = j+1
	print 'loading patient', files[i]
	patient_i = np.transpose(h5py.File('data/Cleaned/'+files[i],'r')['Data'].value)
	print patient_i.shape, 'range',40*i,40*(i+1)
	channel[40*j:40*(j+1), :] = patient_i[:, channel_num, :]

channel = np.reshape(channel, ((32-len(skip_patients))*40*9, 8064/9))
data_dict = {}
data_dict['clean_channel'+str(channel_num)] = channel
sc.io.savemat('clean_channel'+str(channel_num),data_dict)
np.save('clean_channel'+str(channel_num), channel)



















