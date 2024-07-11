import numpy as np
import pandas as pd
from RandomForest import RandomForest as RF
from sklearn.model_selection import train_test_split
from sklearn import datasets as ds

data = ds.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#try n_trees = 20 passing into the RF()
clf = RF(n_trees = 20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print(acc)