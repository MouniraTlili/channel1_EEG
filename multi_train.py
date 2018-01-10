import numpy as np
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import merge, Input
import keras
import tensorflow as tf
from numpy import linalg as LA
import scipy as sc
from scipy import io
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def custom_objective(y_true, y_pred):
    loss = tf.reduce_sum(tf.abs(y_true - y_pred))
    return loss
np.random.seed(1337)
tmp = 'tmp_0_32/'
epochs = 1
mat = sc.io.loadmat('data/channel0_32_normalized.mat')
train = mat['train']
test = mat['test']

channel1 = 0
channel2 = 32
channel1_std = sc.io.loadmat('data/channel'+str(channel1)+'_normalized.mat')['channel_std']
channel1_mean = sc.io.loadmat('data/channel'+str(channel1)+'_normalized.mat') ['channel_mean']
channel2_std = sc.io.loadmat('data/channel'+str(channel2)+'_normalized.mat')['channel_std']
channel2_mean = sc.io.loadmat('data/channel'+str(channel2)+'_normalized.mat') ['channel_mean']
print channel1_mean.shape, channel2_mean.shape
print channel1_std.shape, channel2_std.shape

print "Train shape", train.shape
print "Test shape", test.shape
def unormalize1(data):
    # return data
    return data*channel1_std+channel1_mean
def unormalize2(data):
    # return data
    return data*channel2_std+channel2_mean
   

a = 1612
b = 1433 
c = 1254
d = 1075
e = 896
f = 716
g = 537
h = 358
i = 179
j = 89
distortion_list_EEG = []
distortion_list_EOG = []
for compression_level in range(1,11):
    print "\n\n####################################################"
    print "compression level: ",compression_level
    encode_layers = [a,b,c,d,e,f,g,h,i,j]
    encode_name=["encode1","encode2","encode3", "encode4","encode5","encode6","encode7","encode8","encode9","encode10"]
    decode_name=["decode1","decode2","decode3", "decode4","decode5","decode6","decode7","decode8","decode9"]
    file_name = 'channel0_32linear_'+ '_'.join(map(lambda x: str(x),encode_layers[:compression_level]))+'.h5'
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
    opt = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
    model.compile(loss=custom_objective, optimizer=opt)
    history = model.fit([train],[train], epochs=epochs,batch_size=128, verbose=0)

    train_pred = np.array(model.predict({'input':train}))
    test_pred = np.array(model.predict({'input':test}))
    train_dist= LA.norm(train_pred.astype('float64')-train)/LA.norm(train)*100
    test_dist= LA.norm(test_pred.astype('float64')-test)/LA.norm(test)*100

    print "Train distortion", float("{0:.2f}".format(train_dist)), "%"
    print "Test distortion", float("{0:.2f}".format(test_dist)), "%"
    print "train pred", train_pred.shape
    #unormalize
    print "EEG Train distortion", float("{0:.2f}".format(LA.norm(unormalize1(train_pred[:,:896].astype('float64'))-unormalize1(train[:,:896]))/LA.norm(unormalize1(train[:,:896]))*100)), "%"
    print "EEG Test distortion", float("{0:.2f}".format(LA.norm(unormalize1(test_pred[:,:896].astype('float64')) -unormalize1(test[:,:896]))/LA.norm(unormalize1(test[:,:896]))*100)), "%"

    print "EOG Train distortion", float("{0:.2f}".format(LA.norm(unormalize2(train_pred[:,896:].astype('float64'))-unormalize2(train[:,896:]))/LA.norm(unormalize2(train[:,896:]))*100)), "%"
    print "EOG Test distortion", float("{0:.2f}".format(LA.norm(unormalize2(test_pred[:,896:].astype('float64'))-unormalize2(test[:,896:]))/LA.norm(unormalize2(test[:,896:]))*100)), "%"

    distortion_list_EEG.append(float("{0:.2f}".format(LA.norm(unormalize1(test_pred[:,:896].astype('float64')) -unormalize1(test[:,:896]))/LA.norm(unormalize1(test[:,:896]))*100)))
    distortion_list_EOG.append(float("{0:.2f}".format(LA.norm(unormalize2(test_pred[:,896:].astype('float64'))-unormalize2(test[:,896:]))/LA.norm(unormalize2(test[:,896:]))*100)))
    # model.save(tmp+file_name)

print "EEG TEST",distortion_list_EEG
print "EOG TEST", distortion_list_EOG
    





