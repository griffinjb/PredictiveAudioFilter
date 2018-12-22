# Time Series Forecasting (TSP2.0)

# load that audio training data

# load up that keras bullshit

# so you make a label. that label is like, the N+1 sample, 
# the feature is the previous N-d:N datapoints

# Simple MLP

# Train on one sample

# feed unrelated data as an input and filter that way


import numpy as np
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Split input data into data and labels
# Data is np array
# configured for [1,N] shape
def tspLabelFormat(data,bufferSize,predictionSize):
	sets = []
	labels = []

	for i in range(0,data.shape[1]-bufferSize-predictionSize+1):
		sets.append(data[0,i:i+bufferSize])
		labels.append(data[0,i+bufferSize:i+bufferSize+predictionSize])

	return([sets,labels])


if __name__ == "__main__":
	yo = np.zeros([1,100])
	for i in range(yo.shape[1]):
		yo[0,i] = i
	# yo = np.arange(100)

	drcmax = yo.max()

	yo /= drcmax


	dataLabel = tspLabelFormat(yo,5,1)

	Zip_dataLabel = zip(dataLabel[0],dataLabel[1])

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

	model.fit(X_train, y_train, epochs=5000, batch_size=5000)

	preds = model.predict(X_train)
	# preds[preds>=0.5] = 1
	# preds[preds<0.5] = 0
	# score = compare preds and y_test

	preds *= drcmax

	print(preds)
