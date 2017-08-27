import numpy as np
import tensorflow as tf
import math
from datetime import datetime
#from testData import createTrainingSet

#trainX, trainY, testX, testY = createTrainingSet()

testX = list(np.load("/home/paperspace/Driverless_Car/Batches/testX.npy")/255)
testY = list(np.load("/home/paperspace/Driverless_Car/Batches/testY.npy"))

batchSize = 64
#trainStart = int(24960/1.0*0.99/batchSize)
#trainEnd = int(49920/1.0*1.00/batchSize)
trainStart = 0
trainEnd = 862
trainX = []
trainY = []
for i in range(trainStart, trainEnd):
	_batchX = list(np.load("/home/paperspace/Driverless_Car/Batches/trainX"+str(i)+".npy")/255)
	batchX = np.array(_batchX, dtype = np.float32)
	batchY = np.load("/home/paperspace/Driverless_Car/Batches/trainY"+str(i)+".npy")
	for n in range(len(batchY)):
		trainX.append(list(batchX[n]))
		trainY.append(list(batchY[n]))
	print(i)
classCount = 1
print("classCount = ", classCount)
epochs = 32

x = tf.placeholder("float", [None, 240, 320])
y = tf.placeholder("float")

def conv2d1(x, f):
	return tf.nn.conv2d(x, f, strides = [1, 1, 1, 1], padding = "SAME")

def conv2d2(x, f):
	return tf.nn.conv2d(x, f, strides = [1, 2, 2, 1], padding = "SAME")

def maxPool(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def CNNModel(x):
	x = tf.reshape(x, shape = [-1, 240, 320, 1])

	filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 24], stddev = 0.01))
	conBias1 = tf.Variable(tf.truncated_normal([24], stddev = 0.01))
	convLay1 = conv2d2(x, filter1) + conBias1
	convLay1 = maxPool(convLay1)
	convLay1 = tf.nn.elu(convLay1)

	filter2 = tf.Variable(tf.truncated_normal([4, 4, 24, 36], stddev = 0.01))
	conBias2 = tf.Variable(tf.truncated_normal([36], stddev = 0.01))
	convLay2 = conv2d2(convLay1, filter2) + conBias2
	convLay2 = maxPool(convLay2)
	convLay2 = tf.nn.elu(convLay2)

	filter3 = tf.Variable(tf.truncated_normal([3, 3, 36, 48], stddev = 0.01))
	conBias3 = tf.Variable(tf.truncated_normal([48], stddev = 0.01))
	convLay3 = conv2d2(convLay2, filter3) + conBias3
	convLay3 = maxPool(convLay3)
	convLay3 = tf.nn.elu(convLay3)

	filter4 = tf.Variable(tf.truncated_normal([3, 3, 48, 64], stddev = 0.01))
	conBias4 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
	convLay4 = conv2d1(convLay3, filter4) + conBias4
	convLay4 = tf.nn.elu(convLay4)

	filter5 = tf.Variable(tf.truncated_normal([2, 2, 64, 64], stddev = 0.01))
	conBias5 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
	convLay5 = conv2d1(convLay4, filter5) + conBias5
	convLay5 = tf.nn.elu(convLay5)
	convLay5 = tf.reshape(convLay5, shape = [-1, 1280]);
	tf.nn.dropout(convLay5, 0.75)

	weights1 = tf.Variable(tf.truncated_normal([1280, 100], stddev = 0.01))
	fcBias1 = tf.Variable(tf.truncated_normal([100], stddev = 0.01))
	fcLay1 = tf.matmul(convLay5, weights1) + fcBias1
	fcLay1 = tf.nn.elu(fcLay1)

	weights2 = tf.Variable(tf.truncated_normal([100, 50], stddev = 0.01))
	fcBias2 = tf.Variable(tf.truncated_normal([50], stddev = 0.01))
	fcLay2 = tf.matmul(fcLay1, weights2) + fcBias2
	fcLay2 = tf.nn.elu(fcLay2)

	weights3 = tf.Variable(tf.truncated_normal([50, 10], stddev = 0.01))
	fcBias3 = tf.Variable(tf.truncated_normal([10], stddev = 0.01))
	fcLay3 = tf.matmul(fcLay2, weights3) + fcBias3
	fcLay3 = tf.nn.elu(fcLay3)

	weights4 = tf.Variable(tf.truncated_normal([10, classCount], stddev = 0.01))
	fcBias4 = tf.Variable(tf.truncated_normal([classCount], stddev = 0.01))
	fcLay4 = tf.matmul(fcLay3, weights4) + fcBias4

	return fcLay4

def trainNet(x):
	prediction = CNNModel(x)
	cost = tf.losses.mean_squared_error(predictions = prediction, labels = y)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#saver.restore(sess, "/home/paperspace/Driverless_Car/Old_Model/model.ckpt")
		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(trainEnd-trainStart):
				start = i*batchSize
				end = start+batchSize
				if (end > len(trainY)):
					end = len(trainY)
				batchX = trainX[start:end]
				batchY = trainY[start:end]
				#batchX = list(np.load("/home/paperspace/Driverless_Car/Batches/trainX"+str(i)+".npy"))
				#batchY = list(np.load("/home/paperspace/Driverless_Car/Batches/trainY"+str(i)+".npy"))
				if ((i+1)%64 == 0):
					print(str(i+1)+" batches completed out of "+str(trainEnd-trainStart)+" "+str(datetime.now()))
				_, c = sess.run([optimizer, cost], feed_dict = {x: batchX, y: batchY})
				epoch_loss += c
			print("Epoch", epoch+1, "completed out of ", epochs, " loss: ", epoch_loss)
			saverPath = saver.save(sess, "/home/paperspace/Driverless_Car/Current_Model/model.ckpt")
			print(saverPath)

			loss = tf.losses.mean_squared_error(predictions = prediction, labels = y)
			print("Loss: ", loss.eval({x:testX, y:testY}))
			print("Avg pridiction: ", sess.run(tf.reduce_mean(prediction), feed_dict = {x: testX}))

trainNet(x)
