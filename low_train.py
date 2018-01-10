import numpy as np
import os 
import random
import scipy as sc
from scipy import io


from keras.models import Model, Sequential
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import merge, Input
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, RMSprop
import h5py
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
import keras
import tensorflow as tf
import theano
import theano.tensor as T
import os 
import scipy as sc
from scipy import io
os.environ["CUDA_VISIBLE_DEVICES"]=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
	loss = tf.reduce_sum(tf.abs(y_true - y_pred))
	return loss


np.random.seed(1337)

eeg_train = np.load('channel1_train.npy')
eeg_test = np.load('channel1_test.npy')

a=750
b=627
c=477
d=269
e=180
f=90
g=50
# f_initial = 'channel1_linear_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(e)+'_'+str(f)+'.h5'
file = 'channel1_clean_linear_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(e)+'_'+str(f)+'_'+str(g)+'.h5'
vec_len = len(eeg_train[0])


eeg_path= Input(shape=(vec_len,),name='eeg_input')
encoded_eeg=Dense(a ,name="encode_eeg1")(eeg_path)
encoded_eeg=Dense(b ,name="encode_eeg2")(encoded_eeg)
# encoded_eeg=Dense(c ,name="encode_eeg3")(encoded_eeg)
# encoded_eeg=Dense(d ,name="encode_eeg4")(encoded_eeg)
# encoded_eeg=Dense(e ,name="encode_eeg5")(encoded_eeg)
# encoded_eeg=Dense(f ,name="encode_eeg6")(encoded_eeg)
# encoded_eeg=Dense(g ,name="encode_eeg7")(encoded_eeg)

# decoded_eeg=Dense(f ,name="decode_eeg1")(encoded_eeg)
# decoded_eeg=Dense(e ,name="decode_eeg2")(decoded_eeg)
# decoded_eeg=Dense(d ,name="decode_eeg3")(decoded_eeg)
# decoded_eeg=Dense(c ,name="decode_eeg4")(decoded_eeg)
# decoded_eeg=Dense(b ,name="decode_eeg5")(decoded_eeg)
decoded_eeg=Dense(a ,name="decode_eeg6")(decoded_eeg)
decoded_eeg=Dense(vec_len,name="output1")(decoded_eeg)
model = Model(input=[eeg_path], output=[decoded_eeg])


epochs = 15000
learning_rate = 0.000001
decay_rate = learning_rate / epochs
momentum = 0.8
opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
if file in os.listdir('tmp/'):
	print 'found'
	model.load_weights('tmp/'+file)
model.compile(loss=custom_objective, optimizer=opt)
history = model.fit([eeg_train],[eeg_train], epochs=epochs,batch_size=128, verbose=0)


train_pred = np.array(model.predict({'eeg_input':eeg_train}))
test_pred = np.array(model.predict({'eeg_input':eeg_test}))

train_dist= LA.norm(train_pred.astype('float64')-eeg_train)/LA.norm(eeg_train)*100
test_dist= LA.norm(test_pred.astype('float64')-eeg_test)/LA.norm(eeg_test)*100

data = {}
data['train_pred'] = train_pred
data['test_pred'] = test_pred

# sc.io.savemat('tmp/predictions_linear'+str(b),data)
print "\n\n###############################"
print 'a = ',a, 'b = ',b , 'c = ',c, 'd = ',d, 'e = ',e, 'f = ',f, 'g = ',g
print "Train distortion", train_dist
print "Test distortion", test_dist

model.save('tmp/'+file)

# x = 409



# a=np.arange(x)

# plt.plot(a,eeg_train[1424,:x],'b')
# plt.show() 
# plt.plot(a,eeg_pred[20,:x],'b',eeg_test[20,:x],'r')
# plt.show() 



