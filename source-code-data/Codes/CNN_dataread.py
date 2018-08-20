import os
import time
import ipdb
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from CNN_model import CNN
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

np.random.seed(9002)

def batch_iter(data, batch_size):
		""" Generates a batch iterator for a dataset."""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int((len(data)-1)/batch_size)
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_data = data[shuffle_indices]

		if batch_size==len(data):
			yield shuffled_data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = (batch_num + 1) * batch_size
			yield shuffled_data[start_index:end_index]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def train_test_split(dataset, labels):
	train_test_split_ratio = 0.7
	combined = np.array(zip(dataset, labels))
	shuffle_indices = np.random.permutation(np.arange(len(combined)))
	shuffled_data = combined[shuffle_indices]
	indices_of_1 = np.where(shuffled_data[:,1]==1)[0]
	indices_of_0 = np.where(shuffled_data[:,1]==0)[0]

	train_data = np.concatenate((shuffled_data[indices_of_0[0:int(0.7*len(indices_of_0))]], \
	shuffled_data[indices_of_1[0:int(0.7*len(indices_of_1))]]), axis=0)

	test_data = np.concatenate((shuffled_data[indices_of_0[int(0.7*len(indices_of_0)): ]], \
	shuffled_data[indices_of_1[int(0.7*len(indices_of_1)): ]]), axis=0)

	return train_data, test_data


def data_augmentation(train_data):
	shuffle_indices = np.random.permutation(np.arange(len(train_data)))
	train_data = train_data[shuffle_indices]

	indices_of_1 = np.where(train_data[:,1]==1)[0]
	indices_of_0 = np.where(train_data[:,1]==0)[0]
	list_indices = [indices_of_0, indices_of_1]
	len_indices = [len(indices_of_0), len(indices_of_1)]
	desired_samples, ratio, index_min = max(len_indices) - min(len_indices)* (max(len_indices)/min(len_indices)), \
	max(len_indices)/min(len_indices), np.argmin(len_indices)

	indices_to_augment  = list_indices[index_min]

	repeated_indices = np.repeat(indices_to_augment, (ratio-1))
	chosen_indices = np.random.permutation(indices_to_augment)[0:desired_samples]

	final_train_data = np.concatenate((train_data, train_data[repeated_indices], train_data[chosen_indices]), axis=0)
	shuffle_indices = np.random.permutation(np.arange(len(final_train_data)))
	final_train_data = final_train_data[shuffle_indices]

	return final_train_data 

def load_dataset(img_path):
	dataset = []
	files = os.listdir(img_path)
	for image_file in files:
		if image_file.endswith('png'):
			img = Image.open(img_path+image_file)
			dataset.append(np.array(img))

	return dataset

def main():
	disease = 'obe'
	num_epochs = 90#200#60
	batch_size = 20#16#118
	learning_rate = 5e-4
	num_classes = 2
	label_encoder = LabelEncoder()
	img_path = './images/'+disease+'/'

	dataset = load_dataset(img_path)
	labels = pd.read_csv('./datasets/'+disease+'phy_y.csv').iloc[:,1]
	train_data, test_data = train_test_split(dataset, labels)
	# train_data = data_augmentation(train_data)

	train_x, train_y = zip(*train_data)
	train_x = np.array(train_x)
	train_y = np.array(train_y)

	test_x, test_y = zip(*test_data)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	train_y = label_encoder.fit_transform(train_y) # to convert labels in range (0, num_classes) for one hot encoding to happen properly
	train_y = dense_to_one_hot(train_y, num_classes)
	test_y = label_encoder.fit_transform(test_y)
	test_y = dense_to_one_hot(test_y, num_classes)

	# train_x = (train_x - np.mean(train_x, axis=0))/(np.std(train_x, axis=0)+1e-5)  # normalization
	# test_x = (test_x - np.mean(train_x, axis=0))/(np.std(train_x, axis=0)+1e-5) # normalization

	model = CNN(learning_rate = learning_rate, num_classes = num_classes, input_dim=train_x.shape[1])
	with tf.Session() as sess:
		print('Starting training')
		sess.run(tf.global_variables_initializer())
		for epoch in range(num_epochs):

			begin  = time.time()
			train_accuracies = []
			train_batches = batch_iter(list(zip(train_x, train_y)), batch_size)
			for batch in train_batches:
				x_batch, y_batch = zip(*batch)
				feed_dict = {model.X: x_batch, model.Y_: y_batch}

				training_optimizer, cost, accuracy = sess.run([model.training_optimizer, model.cost, model.accuracy], feed_dict)
				train_accuracies.append(accuracy)
				print cost

			train_acc_mean = np.mean(train_accuracies)

			print("Epoch %d, time = %ds, loss = %.4f, train accuracy = %.4f" % (epoch, time.time()-begin, cost, train_acc_mean))
			feed_dict_test = {model.X: test_x, model.Y_: test_y}
			Y, accuracy = sess.run([model.Y, model.accuracy], feed_dict_test)
			print("Testing accuracy ", accuracy)


		feed_dict = {model.X: test_x, model.Y_: test_y}
		Y, accuracy = sess.run([model.Y, model.accuracy], feed_dict)
		print("Final testing accuracy ", accuracy)
		print(confusion_matrix(np.argmax(test_y, axis = 1), np.argmax(Y, axis = 1)))



if __name__ == "__main__":
    main()
