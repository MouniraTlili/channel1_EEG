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
a=750
b=627
c=477
d=269
e=180
f=90
g=50

file = 'channel1_linear_'+str(a)+'_'+str(b)+'_'+str(c)+'_'+str(d)+'_'+str(e)+'_'+str(f)+'_'+str(g)+'.h5'

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
model1 = Model(inputs=[eeg_path], outputs=[decoded_eeg])
model1.load_weights('tmp/'+file)




eeg_path= Input(shape=(vec_len,),name='eeg_input')
layer=Dense(a ,name="encode_eeg1")(eeg_path)
layer=Dense(b ,name="encode_eeg2")(layer)
layer=Dense(c ,name="encode_eeg3")(layer)
layer=Dense(d ,name="encode_eeg4")(layer)
layer=Dense(e ,name="encode_eeg5")(layer)
layer=Dense(f ,name="encode_eeg6")(layer)
layer=Dense(g ,name="encode_eeg7")(layer)

# layer=Dense(f ,name="decode_eeg1")(layer)
# layer=Dense(e ,name="decode_eeg2")(layer)
# layer=Dense(d ,name="decode_eeg3")(layer)
# layer=Dense(c ,name="decode_eeg4")(layer)
# layer=Dense(b ,name="decode_eeg5")(layer)
# layer=Dense(a ,name="decode_eeg6")(layer)
# layer=Dense(vec_len,name="output1")(layer)
model = Model(inputs=[eeg_path], outputs=[layer])

model.save('tmp/half_largest')

quit()

model.layers[1].set_weights(model1.layers[1].get_weights())
model.layers[2].set_weights(model1.layers[2].get_weights())
model.layers[3].set_weights(model1.layers[3].get_weights())
model.layers[4].set_weights(model1.layers[4].get_weights())
model.layers[5].set_weights(model1.layers[5].get_weights())
# model.layers[6].set_weights(model1.layers[6].get_weights())

model.layers[6].set_weights(model1.layers[10].get_weights())
model.layers[7].set_weights(model1.layers[11].get_weights())
model.layers[8].set_weights(model1.layers[12].get_weights())
model.layers[9].set_weights(model1.layers[13].get_weights())
model.layers[10].set_weights(model1.layers[14].get_weights())







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

# print "Bottleneck size", encode_layers[compression_level-1]
print "Compression Percentage: ", float("{0:.2f}".format(100*(896-e)/float(896))), "%"
print "Train distortion", float("{0:.2f}".format(train_dist)), "%"
print "Test distortion", float("{0:.2f}".format(test_dist)), "%"
# data = {}
# data['train_pred'] = train_pred
# data['test_pred'] = test_pred
# sc.io.savemat('tmp/predictions_linear'+ str(float("{0:.2f}".format(100*(896-encode_layers[compression_level-1])/float(896))))+'_'.join(map(lambda x: str(x),encode_layers[:compression_level])),data)
x = 896
a=np.arange(x)

plt.plot(a,eeg_test[100,:], 'r', test_pred[100,:],'b')
plt.show() 
# # time.sleep(3)
# # plt.clf
# # quit()














