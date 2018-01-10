import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import theano
import os
import scipy as sc
from scipy import io
from numpy import linalg as LA
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tmp = 'tmp_rnn/'
channel = 0
epochs = 1000
skip_patients = [1,7,9,3,14,21,29]
num_experiments = 40
num_experiments = 1
label_width = 1 
file_name = 'stacked_channel'+str(channel)+'_'+str(label_width)+'.h5'

############################. DO NOT TOUCH ####################################

mat = np.load('data/channel0_normalized.npy')

if num_experiments == 1:
	#the first index is patient 0, experiment 0
	mat = np.reshape(mat,(25*40, 8064))[0,:]
	mat = np.reshape(mat,(8064, 1))
else:
	#the 40 first vectors are patient 0, experiments 0-40
	mat = np.reshape(mat,(25*40, 8064))[:40,:]
	mat = np.reshape(mat,(40*8064, 1))
print mat.shape
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-label_width):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[(i + look_back):(i + look_back + label_width), 0])
    return np.array(dataX), np.array(dataY)

look_back = 20
dataset = mat
# split into train and test sets sequentially (we assume sequential data)
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print trainX.shape, trainY.shape
print testX.shape, testY.shape
theano.config.compute_test_value = "ignore"

batch_size = 1
model = Sequential()
# for i in range(2):
    # model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    # model.add(Dropout(0.3))
model.add(LSTM(label_width, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
if file_name in os.listdir(tmp):
	print 'found', tmp+file_name
	model.load_weights(tmp+file_name)

for i in range(1):
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
print "FINISHED"
trainScore = model.evaluate(trainX, trainY, batch_size=batch_size, verbose=2)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX[:252], testY[:252], batch_size=batch_size, verbose=0)
print('Test Score: ', testScore)
model.save(tmp+file_name)
#testing
look_ahead = 20
#take the last 19 numbers from the last vector from train, append to it the last label 
# so we have a 20 number vectorin train predictfrom there we'll predict the next 250 numbers 
# trainPredict = [np.vstack([trainX[-1][1:], trainY[-1]])]

predictions = np.zeros((testX.shape[0]-1,look_ahead))
print "test predict", testX.shape
# we cannot guess after the last vector so we remove the last instance 
for instance in range(testX.shape[0]-1):
	trainPredict = [testX[instance,:]]
	for i in range(look_ahead):
	    prediction = model.predict(np.array([trainPredict[-1]]), batch_size=batch_size)
	    predictions[instance,i] = prediction
	    trainPredict.append(np.vstack([trainPredict[-1][1:],prediction]))
test_pred = np.zeros((testX.shape[0]-1,2*look_ahead))
test_res = np.zeros((testX.shape[0]-1,2*look_ahead))
test_pred[:,:look_ahead] = testX[:-1,:,0]
test_pred[:,look_ahead:] = predictions
test_res[:,:look_ahead] = testX[:-1,:,0]
test_res[:,look_ahead:] = testX[1:,:,0]


prediction_dist= LA.norm(test_pred-test_res)/LA.norm(test_res)*100
print "Prediction distortion: ", float("{0:.2f}".format(prediction_dist)), "%"








# Code snippets taken from https://github.com/sachinruk/PyData_Keras_Talk/blob/master/cosine_LSTM.ipynb