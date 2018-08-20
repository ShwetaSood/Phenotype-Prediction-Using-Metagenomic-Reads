from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def prediction(train, test):
    feature_num = parse_feature_num(train,0)
    feature_num = parse_feature_num(test,feature_num)
    X,y = parse_feature_label(train, feature_num)
    # print(y)
    # clf = svm.SVC()
    clf = get_SVM_classifier()
    clf.fit(X, y)
    test_X, test_y = parse_feature_label(test, feature_num)
    predicted_y = clf.predict(test_X)
    y_true = np.array(test_y)
    y_pred = np.array(predicted_y)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    print("precision: "+str(precision[0]))
    print("recall: "+str(recall[0]))
    print("f1: "+str(f1[0]))
    print("accuracy: "+str(accuracy))

def get_random_forest_classifier():
    return RandomForestClassifier(n_estimators=500, oob_score=True, random_state=1)

def get_neural_network_classifier():
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50,50,50), random_state=1)

def get_SVM_classifier():
    return svm.SVC()

def parse_feature_label(train, feature_num):
    X = []
    y = []
    with open(train) as fh:
        for line in fh:
            label = line.split(' ')[0]
            y.append(label)
            features = [0] * (int(feature_num)+1)
            for i in line.split(' '):
                if ':' in i:
                    index = i.split(':')[0]
                    val = i.split(':')[1]
                    features[int(index)] = float(val)*100
            X.append(features)
    return X,y

def parse_feature_num(files, feature_num):
    with open(files) as fh:
        for line in fh:
            for i in line.split(' '):
                if ':' in i and (int(i.split(':')[0]) > int(feature_num)):
                    feature_num = i.split(':')[0]
    return feature_num

def main():
    train = sys.argv[1]
    test = sys.argv[2]
    prediction(train, test)

if __name__ == "__main__":
    main()
