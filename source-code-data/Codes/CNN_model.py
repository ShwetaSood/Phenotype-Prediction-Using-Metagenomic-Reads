import ipdb
import numpy as np
import tensorflow as tf

np.random.seed(9002)


class CNN(object):

	def __init__(self, learning_rate, num_classes, input_dim):
		self.X = tf.placeholder(tf.float32, [None, input_dim, input_dim, 3])
		self.Y_ = tf.placeholder(tf.float32, [None, num_classes])

		reduced_dim = (input_dim-6)/2

		K = 5
		L = 5
		M = 5
		N = 5
		O = 5
		P = 5*reduced_dim*reduced_dim

		# W1 = tf.Variable(tf.truncated_normal([3, 3, 3, K], stddev=0.1), name="W1")  # 5x5 patch, 3 input channel, K output channels
		# B1 = tf.Variable(tf.ones([K])/num_classes, name="B1")
		# W2 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=0.1), name="W2")
		# B2 = tf.Variable(tf.ones([L])/num_classes, name="B2")
		# W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1), name="W3")
		# B3 = tf.Variable(tf.ones([M])/num_classes, name="B3")
		# W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=0.1), name="W4")
		# B4 = tf.Variable(tf.ones([N])/num_classes, name="B4")
		# W5 = tf.Variable(tf.truncated_normal([3, 3, N, O], stddev=0.1), name="W5")
		# B5 = tf.Variable(tf.ones([O])/num_classes, name="B5")
		# W6 = tf.Variable(tf.truncated_normal([P, num_classes], stddev=0.1), name="W6")
		# B6 = tf.Variable(tf.ones([num_classes])/num_classes, name="B6")

		# new additions
		weight_decay = tf.constant(0.00001, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
		W1 = tf.get_variable(shape=[3, 3, 3, K], name="W1", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))  # 5x5 patch, 3 input channel, K output channels
		B1 = tf.get_variable(shape=[K], name="B1")
		W2 = tf.get_variable(shape=[3, 3, K, L], name="W2", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
		B2 = tf.get_variable(shape=[L], name="B2")
		W3 = tf.get_variable(shape=[3, 3, L, M], name="W3", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
		B3 = tf.get_variable(shape=[M], name="B3")
		W4 = tf.get_variable(shape=[3, 3, M, N], name="W4", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
		B4 = tf.get_variable(shape=[N], name="B4")
		W5 = tf.get_variable(shape=[3, 3, N, O], name="W5", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
		B5 = tf.get_variable(shape=[O], name="B5")
		W6 = tf.get_variable(shape=[P, num_classes], name="W6", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
		B6 = tf.get_variable(shape=[num_classes], name="B6")


		# Foward pass for new images

		stride = 1
		Y1 = tf.nn.relu(tf.nn.conv2d(self.X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
		stride = 1
		Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2)
		stride = 1
		Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='VALID') + B3)
		stride = 1
		Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4)
		stride = 1
		Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5)

		Pool_out = tf.layers.max_pooling2d(Y5, pool_size=[2,2], strides=2)
		
		YY = tf.reshape(Pool_out, shape=[-1, P])

		Ylogits = tf.matmul(YY, W6) + B6

		self.Y = tf.nn.softmax(Ylogits)
		self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)

		self.cost = tf.reduce_mean(self.cost)*100

		# accuracy of the trained model, between 0 (worst) and 1 (best)

		correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.training_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
