import numpy as np
import tensorflow as tf
from testData import createTrainingSet
import math
import time
from skimage import io

trainX, trainY, testX, testY = createTrainingSet()

batchSize = 64
classCount = len(testY[0])
print("classCount = ", classCount)
epochs = 16

x = tf.placeholder("float", [None, 240, 320])
y = tf.placeholder("float")

def conv2d1(x, f):
	return tf.nn.conv2d(x, f, strides = [1, 1, 1, 1], padding = "SAME")

def conv2d2(x, f):
    return tf.nn.conv2d(x, f, strides = [1, 2, 2, 1], padding = "SAME")

def maxPool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def testCNN(x):
	x = tf.reshape(x, shape = [-1, 240, 320, 1])
	intputWeight = {"weights":tf.Variable(tf.truncated_normal([5, 5, 1, 24], stddev = 0.1)), 
    				"biases":tf.Variable(tf.truncated_normal([24], stddev = 0.1))}
	'''weight1 = {"weights":tf.Variable(tf.truncated_normal([5, 5, 24, 36], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([36], stddev = 0.1))}
	weight2 = {"weights":tf.Variable(tf.truncated_normal([5, 5, 36, 48], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([48], stddev = 0.1))}'''
	weight3 = {"weights":tf.Variable(tf.truncated_normal([3, 3, 24, 48], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([48], stddev = 0.1))}
	weight4 = {"weights":tf.Variable(tf.truncated_normal([3, 3, 48, 64], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([64], stddev = 0.1))}		   
	weight5 = {"weights":tf.Variable(tf.truncated_normal([76800, 400], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([400], stddev = 0.1))}
	weight6 = {"weights":tf.Variable(tf.truncated_normal([400, 100], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([100], stddev = 0.1))}
	weight7 = {"weights":tf.Variable(tf.truncated_normal([100, 20], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([20], stddev = 0.1))}		   
	outputWeight = {"weights":tf.Variable(tf.truncated_normal([20, classCount], stddev = 0.1)), 
    				"biases":tf.Variable(tf.zeros([classCount]))}

	intputLayer = conv2d2(x, intputWeight["weights"] + intputWeight["biases"])
	intputLayer = tf.nn.elu(intputLayer)
	intputLayer = maxPool(intputLayer)

	'''hidLayer1 = conv2d2(intputLayer, weight1["weights"]) + weight1["biases"]
	hidLayer1 = tf.nn.elu(hidLayer1)
	hidLayer1 = maxPool(hidLayer1)

	hidLayer2 = conv2d2(hidLayer1, weight2["weights"]) + weight2["biases"]
	hidLayer2 = tf.nn.elu(hidLayer2)
	hidLayer2 = maxPool(hidLayer2)'''

	hidLayer3 = conv2d1(intputLayer, weight3["weights"]) + weight3["biases"]
	hidLayer3 = tf.nn.elu(hidLayer3)
	hidLayer3 = maxPool(hidLayer3)

	hidLayer4 = conv2d1(hidLayer3, weight4["weights"]) + weight4["biases"]
	hidLayer4 = tf.reshape(hidLayer4, shape = [-1, 76800]);
	hidLayer4 = tf.nn.elu(hidLayer4)
	tf.nn.dropout(hidLayer4, 0.5)

	hidLayer5 = tf.matmul(hidLayer4, weight5["weights"]) + weight5["biases"]
	hidLayer5 = tf.nn.elu(hidLayer5)

	hidLayer6 = tf.matmul(hidLayer5, weight6["weights"]) + weight6["biases"]
	hidLayer6 = tf.nn.elu(hidLayer6)

	hidLayer7 = tf.matmul(hidLayer6, weight7["weights"]) + weight7["biases"]
	hidLayer7 = tf.nn.elu(hidLayer7)
	#tf.nn.dropout(hidLayer3, 0.85)

	outputLayer = tf.matmul(hidLayer7, outputWeight["weights"]) + outputWeight["biases"]

	return outputLayer

def CNNModel1(x):
	x = tf.reshape(x, shape = [-1, 240, 320, 1])
	intputWeight = {"weights":tf.Variable(tf.truncated_normal([5, 5, 1, 24], stddev = 0.1)), 
    				"biases":tf.Variable(tf.truncated_normal([24], stddev = 0.1))}
	weight1 = {"weights":tf.Variable(tf.truncated_normal([5, 5, 24, 36], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([36], stddev = 0.1))}
	weight2 = {"weights":tf.Variable(tf.truncated_normal([5, 5, 36, 48], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([48], stddev = 0.1))}
	weight3 = {"weights":tf.Variable(tf.truncated_normal([3, 3, 48, 64], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([64], stddev = 0.1))}
	weight4 = {"weights":tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([64], stddev = 0.1))}		   
	weight5 = {"weights":tf.Variable(tf.truncated_normal([5120, 100], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([100], stddev = 0.1))}
	weight6 = {"weights":tf.Variable(tf.truncated_normal([100, 50], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([50], stddev = 0.1))}
	weight7 = {"weights":tf.Variable(tf.truncated_normal([50, 10], stddev = 0.1)), 
    		   "biases":tf.Variable(tf.truncated_normal([10], stddev = 0.1))}		   
	outputWeight = {"weights":tf.Variable(tf.truncated_normal([10, classCount], stddev = 0.1)), 
    				"biases":tf.Variable(tf.zeros([classCount]))}

	intputLayer = conv2d2(x, intputWeight["weights"] + intputWeight["biases"])
	intputLayer = tf.nn.elu(intputLayer)
	#intputLayer = maxPool(intputLayer)

	hidLayer1 = conv2d2(intputLayer, weight1["weights"]) + weight1["biases"]
	hidLayer1 = tf.nn.elu(hidLayer1)
	hidLayer1 = maxPool(hidLayer1)

	hidLayer2 = conv2d2(hidLayer1, weight2["weights"]) + weight2["biases"]
	hidLayer2 = tf.nn.elu(hidLayer2)
	hidLayer2 = maxPool(hidLayer2)

	hidLayer3 = conv2d1(hidLayer2, weight3["weights"]) + weight3["biases"]
	hidLayer3 = tf.nn.elu(hidLayer3)

	hidLayer4 = conv2d1(hidLayer3, weight4["weights"]) + weight4["biases"]
	hidLayer4 = tf.reshape(hidLayer4, shape = [-1, 5120]);
	hidLayer4 = tf.nn.elu(hidLayer4)
	tf.nn.dropout(hidLayer4, 0.5)

	hidLayer5 = tf.matmul(hidLayer4, weight5["weights"]) + weight5["biases"]
	hidLayer5 = tf.nn.elu(hidLayer5)

	hidLayer6 = tf.matmul(hidLayer5, weight6["weights"]) + weight6["biases"]
	hidLayer6 = tf.nn.elu(hidLayer6)

	hidLayer7 = tf.matmul(hidLayer6, weight7["weights"]) + weight7["biases"]
	hidLayer7 = tf.nn.elu(hidLayer7)
	#tf.nn.dropout(hidLayer3, 0.85)

	outputLayer = tf.matmul(hidLayer7, outputWeight["weights"]) + outputWeight["biases"]

	return outputLayer

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

def testNet(x):
	#prediction = CNNModel(x)
	prediction = CNNModel(x)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, "/Users/lexseallin/Documents/Projects/Driverless_Car/Current_Model//model.ckpt")

		print(testY, sess.run(prediction, feed_dict = {x: testX}))
		loss = tf.losses.mean_squared_error(predictions = prediction, labels = y)
		print("Loss: ", loss.eval({x:testX, y:testY}))
		#print(sess.run(prediction, feed_dict = {x: [io.imread("/Users/lexseallin/Desktop/sample/0.jpg", "as_gray")]}))
		'''startTime = time.time()
		for i in range(100):
			print(sess.run(prediction, feed_dict = {x: [testX[0]]}))
		print("each run took ", (time.time()-startTime)/100, "second")'''
		#print("Avg pridiction: ", sess.run(tf.reduce_mean(prediction), feed_dict = {x: testX}))

testNet(x)