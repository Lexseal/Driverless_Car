import numpy as np
import random
import pickle
from skimage import io

def createSamples(classification):
	imgArr = []
	for i in range(int((len(classification))/1)):
		name = "/home/paperspace/Driverless_Car/samples/" + str(i) + ".jpg"
		'''hotNum = round(float(classification[i].split(" ")[0])*0.0063158-6.947)
		newClassification = []
		for n in range(7):
			if (n == hotNum):
				newClassification.append(1)
			else:
				newClassification.append(0)'''
		img = io.imread(name, "as_gray")
		value = (float(classification[i].split(" ")[0]))
		imgArr.append([np.array(img), np.array([value])])
	return imgArr

def createTrainingSet(testSize = 0.01):
	file = open("/home/paperspace/Driverless_Car/samples/log.txt", "r")
	inputs = file.readlines()
	data = createSamples(inputs)
	random.shuffle(data)

	testingSize = int(testSize*len(data))

	x = np.array(data)[:,0]
	y = np.array(data)[:,1]

	trainX = x[:-testingSize]
	trainY = y[:-testingSize]

	testX = x[-testingSize:]
	testY = y[-testingSize:]

	#print(testX)
	print(len(trainY), len(testY))

	return list(trainX), list(trainY), list(testX), list(testY)
#trainX, trainY, testX, testY = createTrainingSet()
