import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

np.random.seed(9001)
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


disease_list = ['cir','col','ibd','obe','t2d','wt2d',]
for disease in disease_list:

	data = pd.read_csv('./datasets/'+disease+'phy_x.csv')
	data = data.iloc[:,1:] # added
	data = data.T # added

	# data = np.load('./dataset.npy')
	labels = pd.read_csv('./datasets/'+disease+'phy_y.csv').iloc[:,1]

	train_data, test_data = train_test_split(np.array(data), labels)

	train_x, train_y = zip(*train_data)
	train_x = np.array(train_x)
	train_y = np.array(train_y)

	test_x, test_y = zip(*test_data)
	test_x = np.array(test_x)
	test_y = np.array(test_y)

	rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=1)
	rf.fit(train_x, train_y)

	predicted = rf.predict(test_x)

	accuracy = accuracy_score(test_y, predicted)
	print('Classifier: rdf, Disease: %s Mean accuracy score: %.4f'%(disease, accuracy))
	print(confusion_matrix(test_y,predicted))

	xgb=XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=8)
	xgb.fit(train_x, train_y)

	predicted_xgb= xgb.predict(test_x)

	accuracy_xgb = accuracy_score(test_y, predicted_xgb)
	print('Classifier: xgboost, Disease: %s Mean accuracy score: %.4f'%(disease, accuracy_xgb))
	print(confusion_matrix(test_y,predicted_xgb))
