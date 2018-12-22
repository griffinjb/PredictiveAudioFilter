# Time Series Forecasting (TSP2.0)

# load that audio training data

# load up that keras bullshit

# so you make a label. that label is like, the N+1 sample, 
# the feature is the previous N-d:N datapoints

# Simple MLP

# Train on one sample

# feed unrelated data as an input and filter that way


import numpy as np





# Split input data into data and labels
# Data is np array
def tspLabelFormat(data,bufferSize,predictionSize):
	sets = []
	labels = []

	for i in range(0,len(data)-bufferSize-predictionSize):
		sets.append(data[i:i+bufferSize])
		labels.append(data[i+bufferSize:i+bufferSize+predictionSize])



# if __name__ == "__main__":
		


