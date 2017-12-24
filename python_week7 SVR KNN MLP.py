#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week VII
Data Programming With Python – Fall / 2017
Machine Learning - Support Vector Regression, Multilayer Perceptron (MLP),
    K-Nearest Neighbour Regression (KNN)
"""

### Support Vector Regression

import numpy as np
from sklearn.svm import SVR
import pylab as plt
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.cos(X)
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
svr_poly= SVR(kernel='poly', degree=3)
svr_rbf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)
y_rbf = svr_rbf.predict(X)
y_lin = svr_lin.predict(X)
y_poly = svr_poly.predict(X)
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly,c='b', label='Polynomial model')
plt.legend()
plt.show()

### MLP

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
y = np.sin(2 * np.pi * x)
mlp_reg = MLPRegressor(hidden_layer_sizes=(10,3), activation='relu',
        solver='adam', learning_rate='constant', learning_rate_init=0.01,
        max_iter=1000, tol=0.0001,)
mlp_reg.fit(x, y)
test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
test_y = mlp_reg.predict(test_x)
plt.scatter(x, y, c='b', marker="s", label='real')
plt.scatter(test_x,test_y, c='r', marker="o", label='NN Prediction')
plt.legend()
plt.show()

### KNN

from sklearn import neighbors
import matplotlib.pyplot as plt
x = np.arange(0.0, 1, 0.01).reshape(-1, 1)
y = np.cos(np.sin(2 * np.pi * x))
knn_reg =neighbors.KNeighborsRegressor(3, weights='uniform')
knn_reg.fit(x, y)
test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
test_y = knn_reg.predict(test_x)
plt.scatter(x, y, c='b' , label='real')
plt.scatter(test_x,test_y, c='r', marker="o", label='KNN Prediction')
plt.legend()
plt.show()

### Examples

import sklearn.preprocessing as skp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
X = np.linspace(0,100,101)
y = np.array([(100*np.random.rand(1)+num) for num in (5*X+10)])
X = skp.scale(X, axis=0)
y = skp.scale(y, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y)
svr = SVR(kernel='linear')
ols = LinearRegression()
svr.fit(X_train.reshape(-1,1),y_train.flatten())
ols.fit(X_train.reshape(-1,1), y_train.flatten())
pred_SVR = svr.predict(X_test.reshape(-1,1))
pred_OLS = ols.predict(X_test.reshape(-1,1))
print(np.sqrt(mean_squared_error(y_test, pred_SVR)))
print(np.sqrt(mean_squared_error(y_test, pred_OLS)))
plt.plot(X, y, 'kv',label='True data')
plt.plot(X_test, pred_SVR,'ro' ,label='SVR')
plt.plot(X_test, pred_OLS, label='Linear Reg')
plt.legend(loc='upper left')
plt.show()

import sklearn.preprocessing as skp
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
X = np.arange(-10,10).reshape(-1, 1)
y = np.sinc(0.5*X)
standardized_X = skp.scale(X, axis=0)
standardized_y = skp.scale(y, axis=0)
mlp_reg = MLPRegressor(hidden_layer_sizes=(10), activation='tanh',
solver='lbfgs', learning_rate='constant')
mlp_reg.fit(X, y)
s = mlp_reg.predict(X)
plt.plot(X,y, label='real')
plt.plot(X,s, label='MLP')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn import neighbors
X = np.arange(-10,10).reshape(-1, 1)
y = np.sinc(0.5*X)
knn_reg = neighbors.KNeighborsRegressor(2, weights='uniform')
knn_reg.fit(X, y)
s = knn_reg.predict(X)
plt.plot(X, y, label='real')
plt.plot(X, s, label='KNN Reg')
plt.legend()
plt.show()

### Section I – Support Vector Regression (SVR)

# Create a data set consisting of 200 random input and the output would be the
# sin(x) of the input. Make the inputs between 0 and 5. ‘Sort’ the input in
# ascending order. Add some noise to the output and then, do the followings:
# SVR with ‘rbf’ kernel
# SVR with ‘linear’ kernel
# SVR with ‘poly’ kernel
# Plot the results for each of them together with the dataset itself.

import numpy as np
import matplotlib.pyplot as plt
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(40))
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
lw = 2
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
# ’poly’ Sometimes takes time to perform, be patient !!
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

### Section II – Multilayer Perceptron (MLP)

# For the a synthetic dataset containing 100 samples, and noisy output,
# create MLP regression for TWO activation function i.e. ‘Tanh’ and ‘Relu’,
# plot each regression separately to compare the results.
# Hint: use makeregression library to create data

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
#- Generate Data ----------------------------
X_R, y_R = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0, noise=30, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(11,8), dpi=100)
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R[0::5], y_R[0::5], random_state = 0)
#-MLP Regression ----------------------------
for dummy, MYactivation in zip(subaxes,['tanh', 'relu']):
    mlp_reg = MLPRegressor(hidden_layer_sizes = [100,100],activation = MYactivation, solver='lbfgs').fit(X_train,y_train)
    y_predict_output = mlp_reg.predict(X_predict_input)
    plt.plot(X_predict_input, y_predict_output,'^', markersize=10)
    plt.plot(X_train, y_train, 'o')
    plt.xlabel('Input feature')
    plt.ylabel('Target value')
    plt.title('MLP regression\n activation={})'.format(MYactivation))
    plt.legend()
    plt.show()
 
### Section III – K-Nearest Neighbour Regression (KNN)

# For the a synthetic dataset containing 100 samples, and noisy output,
# create KNN regression for various values for K and plot each regression
# separately to compare the results.
# Hint: use makeregression library to create data

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
#-Generate Synthetic Data ---------------
X_R, y_R = make_regression(n_samples=100, n_features=1, n_informative=1, bias=150.0, noise=30)
fig, subaxes = plt.subplots(5, 1, figsize=(11,8), dpi=100)
X = np.linspace(-3, 3, 500).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_R,y_R,random_state=0)
#- KNN -------------------------
for K, K in zip(subaxes,[1, 3, 7, 15, 59]):
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors=K)
    knn_reg.fit(X_train, y_train)
    y_predict_output = knn_reg.predict(X)
    plt.plot(X, y_predict_output)
    plt.plot(X_train, y_train, 'o', alpha=0.9, label='Train')
    plt.plot(X_test, y_test, '^', alpha=0.9, label='Test')
    plt.xlabel('Input feature')
    plt.ylabel('Target value')
    plt.title('KNN Regression (K={})\n$'.format(K))
    plt.legend()
    plt.show()