# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
# from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
#import accuracy score
from sklearn.metrics import accuracy_score
#import f1 score
from sklearn.metrics import f1_score
#import confusion matrix
from sklearn.metrics import confusion_matrix

class LinearRegression:
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr  = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range (self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    