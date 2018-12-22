# Time Series Forecasting (TSP2.0)

# so you make a label. that label is like, the N+1 sample, 
# the feature is the previous N-d:N datapoints

# Simple MLP

# Train on one sample

# feed unrelated data as an input and filter that way


import numpy as np

from os import getcwd, listdir

import matplotlib.pyplot as plt

from scipy.io import wavfile

from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# load audio to nparray
def loadWav(path):
	rate, data = wavfile.read(path)
	data = data[:,0]
	data = data + max(data)
	data = data / max(data)
	return(data)


def rampGen():
	yo = np.zeros([1,100])
	for i in range(yo.shape[1]):
		yo[0,i] = i
	# yo = np.arange(100)
	idx = yo

	yo /= 10

	yo = np.sin(yo) + 1

	drcmax = yo.max()
	yo /= drcmax

	return(yo)

# takes normalized to 0->1 signal. parses to contiguous non-clipping
def clippingParser(data):
	out = []

	# start, stop, flip flop
	st = 0
	sp = 0
	ff = 0

	for i in range(0,data.shape[0]):
		if data[i] == 1:
			if ff:
				sp = i
				out.append(data[st:sp])
				ff = 0
		else:
			if not ff:
				st = i
				ff = 1

	return(out)

# Split input data into data and labels
# Data is np array
# configured for [1,N] shape
def tspLabelFormat(data,bufferSize,predictionSize):
	sets = []
	labels = []

	for i in range(data.shape[0]-bufferSize-predictionSize+1):
		sets.append(data[i:i+bufferSize])
		labels.append(data[i+bufferSize:i+bufferSize+predictionSize])

	return([sets,labels])


if __name__ == "__main__":
	
	# Parameters 
	audioPath = getcwd()+"/liteSpdRKT.wav"
	bufferSize = 1
	predictionSize = 1

	# yo = rampGen()
	wv = loadWav(audioPath)

	yos = clippingParser(wv)

	dataLabel = []

	f = 1

	for yo in yos:
		if f:
			dataLabel = tspLabelFormat(yo,bufferSize,predictionSize)
			f = 0
		else:
			temp = tspLabelFormat(yo,bufferSize,predictionSize)
			# print(dataLabel[0].shape)
			# print(.shape)
			print(dataLabel)
			print(temp[0][0])
			np.append(dataLabel[0],temp[0][0].T,axis=0)
			np.append(dataLabel[1],temp[0][1].T,axis=0)



	# Zip_dataLabel = zip(dataLabel[0],dataLabel[1])

	X_train = np.asarray(dataLabel[0])
	y_train = np.asarray(dataLabel[1])

	# now classify boyo
	# layers = Sequential()
	model = Sequential()
	model.add(Dense(5000, activation='relu', input_dim=X_train.shape[1]))
	model.add(Dropout(0.1))
	model.add(Dense(600, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(y_train.shape[1], activation='sigmoid'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error',
	              optimizer=sgd)

	model.fit(X_train, y_train, epochs=100, batch_size=5000)

	preds = model.predict(X_train)
	# preds[preds>=0.5] = 1
	# preds[preds<0.5] = 0
	# score = compare preds and y_test

	preds *= drcmax
	print(preds)

	# plt.plot(idx,idx,'ro')
	plt.plot(preds)
	# plt.axis([0,100,0,100])
	plt.show()



