# Time Series Forecasting (TSP2.0)
# Josh Griffin
# 12/21/2018


# so you make a label. that label is like, the N+1 sample, 
# the feature is the previous N-d:N datapoints

# Simple MLP

# Train on one sample

# feed unrelated data as an input and filter that way


import numpy as np

from os import getcwd, listdir

import matplotlib.pyplot as plt

from math import sin, pi

from scipy.io import wavfile

from time import strftime

from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json

# load audio to nparray, return 0,1 scaled nparray
def loadWav(path):
	rate, data = wavfile.read(path)
	data = data[:,0]
	data = np.interp(data, (data.min(), data.max()),(0,1))
	return((data,rate))

def writeWav(data,rate,path):
	scale = 1 - .1
	data = np.interp(data,(data.min(), data.max()),(-scale,scale))
	wavfile.write(path,rate,data)


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
		if data[i] == 1 or data[i] == 0:
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

# predictive synthesis - signal [0->1]
def generatePredictions(model,signal,bufferSize,predictionSize,fullReplacement,duty,mix,fxEN):
	# damn so this will be kinda hard

	# Input is clipped audio

	# start with set of at least buffer size

	# predict next step

	# for each sample, insert sample from either unclipped audio
	# or from prediction.

	predBuffer = [[signal[0:bufferSize]]]
	predBuffers = [predBuffer]

	pred = model.predict(predBuffer)

	clipCtr = 0

	print(len(signal))

	for idx,sample in enumerate(signal[bufferSize:-predictionSize]):

		if idx%1000 == 0:
			print(str(idx)+'/'+str(len(signal))+'~ %'+str(100*idx/len(signal)),end="\r")

		# Append prediction or sample
		predBuffer[0][0][:-1] = predBuffer[0][0][1:]
	
		# replace corrupted signal with predicted signal

		# modify mix
		if fxEN:
			mix = (sin(idx/44100*2*pi)+1)/2

		if sample == 1 or sample == 0 or fullReplacement or idx%duty != 0:
			mixSignal = mix * pred + (1-mix)*sample
			signal[bufferSize:-predictionSize][idx] = mixSignal
			sample = mixSignal
			clipCtr += 1

		predBuffer[0][0][-1] = sample

		# Make prediction
		pred = model.predict(predBuffer)

	print("\nClip Count"+str(clipCtr))

	return(signal)

def saveModel(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model"+strftime("%Y%m%d-%H%M%S")+".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model"+strftime("%Y%m%d-%H%M%S")+".h5")
	print("Saved model to disk")

def trainModel(X_train,y_train):
	# now classify boyo
	model = Sequential()
	model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
	# model.add(Dropout(0.1))
	model.add(Dense(10, activation='relu'))
	# model.add(Dropout(0.1))
	model.add(Dense(y_train.shape[1], activation='sigmoid'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error',
	              optimizer=sgd)
	# train model
	model.fit(X_train, y_train, epochs=10, batch_size=50)

	return(model)

def trainModel1(X_train,y_train):
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
	# train model
	model.fit(X_train, y_train, epochs=1, batch_size=500)

	return(model)

def downSample(wv,percent,length):
	ctr = 0
	ff = 0
	for idx,sample in enumerate(wv):
		if ff:
			wv[idx] = 1
			ctr+=1
			if ctr == (1-percent)*length/percent:
				ff = 0
		else:
			ctr+=1
			if ctr == length:
				ff = 1
	return(wv)

# def sampleData():


def loadModel(jsonfn,h5fn):
	# load json and create model
	json_file = open(jsonfn, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(h5fn)
	print("Loaded model from disk")
	return(loaded_model)

if __name__ == "__main__":
	
	# Parameters
	# audioPathWrite = getcwd()+"/liteSpdRKT.wav"
	audioPathWrite = getcwd()+"/grounded.wav"
	# audioPath = getcwd()+"/un1.wav"
	# audioPath = getcwd()+"/keyRaw.wav"
	audioPath = getcwd()+"/grounded.wav"
	# length of prediction info
	bufferSize = 441
	# length of prediction label
	predictionSize = 1
	# Train new model
	# newModel = False
	newModel = True
	# equivalent to duty == 1	
	fullReplacement = True
	# min, 1 : max inf : effective sample rate
	duty = 1
	# prediction mix vs signal mix
	mix = .8
	# function enable
	fxEN = True

	# yo = rampGen()
	wv,rate = loadWav(audioPath)
	writeWV, writeRate = loadWav(audioPathWrite)
	print("Audio Loaded")

	# wv = wv[1100000:1200000]
	wv = wv[:100000]
	# writeWV = writeWV[:100000]
	# wv = wv[50000:80000]
	wv = downSample(wv,.5,44100*1)
	writeWV = downSample(writeWV,.5,44100*10)
	writeWV = writeWV[writeWV != 1]
	print("Parsing into non-clipped data")
	parsedClips = clippingParser(wv)
	print("Parsing Complete")

	dataLabel = []

	f = 1

	print("Formatting labels")
	# iterate clips and append labels
	for yo in parsedClips:
		if f:
			dataLabel = tspLabelFormat(yo,bufferSize,predictionSize)
			f = 0
		else:
			temp = tspLabelFormat(yo,bufferSize,predictionSize)
			for dat in temp[0]:
				dataLabel[0].append(dat)
			for dat in temp[1]:
				dataLabel[1].append(dat)
	print("Formatting Complete")

	# Zip_dataLabel = zip(dataLabel[0],dataLabel[1])


	# generate new model
	if newModel:
		X_train = np.asarray(dataLabel[0])
		y_train = np.asarray(dataLabel[1])
		model = trainModel(X_train,y_train)
		saveModel(model)
	else:
		model = loadModel("cmodel4.json","cweights4.h5")
	
	# for i in range(100):
	# preds = model.predict(X_train)
	pwv = generatePredictions(model,writeWV,bufferSize,predictionSize,fullReplacement,duty,mix,fxEN)

	writeWav(pwv,writeRate,'testOut'+strftime("%Y%m%d-%H%M%S")+'.wav')


	# preds *= drcmax
	# print(preds)

	# plt.plot(idx,idx,'ro')
	# plt.plot(preds)
	# plt.axis([0,100,0,100])
	# plt.show()



