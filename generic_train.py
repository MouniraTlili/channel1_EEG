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
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def custom_objective(y_true, y_pred):
    loss = tf.reduce_sum(tf.abs(y_true - y_pred))
    return loss

np.random.seed(1337) 
tmp = 'tmp_single/'
channel = 0
epochs = 1
denormalize_var = 0

############################. DO NOT TOUCH ####################################

mat = sc.io.loadmat('data/channel'+str(channel)+'_normalized.mat')
channel_std = mat['channel_std']
channel_mean = mat ['channel_mean']
print channel_std.shape, channel_mean.shape
train = mat['train']
test = mat['test']
print "Train shape", train.shape
print "Test shape", test.shape
def deormalize(data):
    if denormalize_var ==1:
        return data*channel_std+channel_mean
    return data
a=806
b=716
c=627
d=537
e=448
f=358
g=268
h=179
i=89
j=44
distortion_list_EEG = []
for compression_level in range(1,11):
    print "\n\n####################################################"
    print "compression level: ",compression_level
    encode_layers = [a,b,c,d,e,f,g,h,i,j]
    encode_name=["encode1","encode2","encode3", "encode4","encode5","encode6","encode7","encode8","encode9","encode10"]
    decode_name=["decode1","decode2","decode3", "decode4","decode5","decode6","decode7","decode8","decode9"]
    file_name = 'channel'+str(channel)+'_'+ '_'.join(map(lambda x: str(x),encode_layers[:compression_level]))+'.h5'
    print file_name
    vec_len = len(train[0])
    path= Input(shape=(vec_len,),name='input')
    decode_layers = encode_layers[:compression_level-1]
    decode_layers.reverse()
    for compress in range(compression_level):
        if compress == 0:
            encoded=Dense(encode_layers[compress],name=encode_name[compress])(path)
        else:
            encoded=Dense(encode_layers[compress],name=encode_name[compress])(encoded)
    decoded = encoded

    for decompress in range(compression_level-1):
        decoded=Dense(decode_layers[decompress],name=decode_name[decompress])(decoded)

    decoded=Dense(vec_len,name="output")(decoded)
    model = Model(inputs=[path], outputs=[decoded])

    if file_name in os.listdir(tmp):
        print 'found'
        model.load_weights(tmp+file_name)

    print "Bottleneck size", encode_layers[compression_level-1]
    print "Compression Percentage: ", float("{0:.2f}".format(100*(vec_len-encode_layers[compression_level-1])/float(vec_len))), "%"
    opt = keras.optimizers.rmsprop(lr=0.0000000000001, decay=1e-6)
    model.compile(loss=custom_objective, optimizer=opt)
    history = model.fit([train],[train], epochs=epochs,batch_size=128, verbose=1)

    train_pred = np.array(model.predict({'input':train}))
    test_pred = np.array(model.predict({'input':test}))
    train_dist= LA.norm(deormalize(train_pred.astype('float64'))-deormalize(train))/LA.norm(deormalize(train))*100
    test_dist= LA.norm(deormalize(test_pred.astype('float64'))-deormalize(test))/LA.norm(deormalize(test))*100

    print "Train distortion", float("{0:.2f}".format(train_dist)), "%"
    print "Test distortion", float("{0:.2f}".format(test_dist)), "%"
    distortion_list_EEG.append(float("{0:.2f}".format(test_dist)))
    # model.save(tmp+file_name)

print distortion_list_EEG
