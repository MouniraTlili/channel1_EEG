import numpy as np
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import merge, Input
import keras
import tensorflow as tf
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy as sc
from scipy import io
import time
def custom_objective(y_true, y_pred):
	loss = tf.reduce_sum(tf.abs(y_true - y_pred))
	return loss
np.random.seed(1337)

eeg_train = np.load('channel1_train.npy')
eeg_test = np.load('channel1_test.npy')



compression = int(raw_input("Enter compression level "))
if compression > 2:
	compression_level = compression -1
else:
	compression_level = 2
a=750
b=627
c=477
d=269
e=180
f=90
g=50

if compression == 2 :
	a = 850
	b = 806

if compression == 1 :
	a = 870
	b = 851
if compression == 0:
	a = 896
	b = 896






encode_layers = [a,b,c,d,e,f,g]
encode_name=["encode_eeg1","encode_eeg2","encode_eeg3", "encode_eeg4","encode_eeg5","encode_eeg6","encode_eeg7"]
decode_name=["decode_eeg1","decode_eeg2","decode_eeg3", "decode_eeg4","decode_eeg5","decode_eeg6"]

vec_len = len(eeg_train[0])
eeg_path= Input(shape=(vec_len,),name='eeg_input')
decode_layers = encode_layers[:compression_level-1]
decode_layers.reverse()
for compress in range(compression_level):
	if compress == 0:
		encoded_eeg=Dense(encode_layers[compress],name=encode_name[compress])(eeg_path)

	else:
		encoded_eeg=Dense(encode_layers[compress],name=encode_name[compress])(encoded_eeg)

decoded_eeg = encoded_eeg
for decompress in range(compression_level-1):
	decoded_eeg=Dense(decode_layers[decompress],name=decode_name[decompress])(decoded_eeg)

decoded_eeg=Dense(vec_len,name="output1")(decoded_eeg)
model = Model(inputs=[eeg_path], outputs=[decoded_eeg])

file_name = 'channel1_linear_'+ '_'.join(map(lambda x: str(x),encode_layers[:compression_level]))+'.h5'
model.load_weights('tmp/'+file_name)


epochs = 1
learning_rate = 0.000001
decay_rate = learning_rate / epochs
momentum = 0.8
opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
model.compile(loss=custom_objective, optimizer=opt)
history = model.fit([eeg_train],[eeg_train], epochs=epochs,batch_size=128, verbose=0)


train_pred = np.array(model.predict({'eeg_input':eeg_train}))
test_pred = np.array(model.predict({'eeg_input':eeg_test}))

train_dist= LA.norm(train_pred.astype('float64')-eeg_train)/LA.norm(eeg_train)*100
test_dist= LA.norm(test_pred.astype('float64')-eeg_test)/LA.norm(eeg_test)*100

print "Bottleneck size", encode_layers[compression_level-1]
print "Compression Percentage: ", float("{0:.2f}".format(100*(896-encode_layers[compression_level-1])/float(896))), "%"
print "Train distortion", float("{0:.2f}".format(train_dist)), "%"
print "Test distortion", float("{0:.2f}".format(test_dist)), "%"
data = {}
data['train_pred'] = train_pred
data['test_pred'] = test_pred
sc.io.savemat('tmp/predictions_linear'+ str(float("{0:.2f}".format(100*(896-encode_layers[compression_level-1])/float(896))))+'_'.join(map(lambda x: str(x),encode_layers[:compression_level])),data)
x = 896
a=np.arange(x)

plt.plot(a,eeg_test[100,:], 'r', test_pred[100,:],'b')
plt.show() 
time.sleep(3)
plt.clf
quit()














