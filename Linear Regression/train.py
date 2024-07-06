import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

from sklearn import datasets
# from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
#import accuracy score
from sklearn.metrics import accuracy_score
#import f1 score
from sklearn.metrics import f1_score
#import confusion matrix
from sklearn.metrics import confusion_matrix


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='blue', marker = "o", s = 30, label='Training Data')
plt.show

reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

yPred = reg.predict(X_test)

def msr(y_test, predictions):
    return np.mean((y_test - predictions)**2)

mse = msr(y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()