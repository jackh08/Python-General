#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week VI
Data Programming With Python – Fall / 2017
Partitioning Data, Linear Methods for Regression, Regression Metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 

### Partitioning Data

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=f)
# OR
# import sklearn.model_selection as skms
# X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=f)

from sklearn.datasets import load_diabetes

# Call the diabetes dataset
diabetes = load_diabetes()
# define the input variable
X = diabetes.data
# define the target variable
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Linear Models

### Ordinary Least Squares
reg = linear_model.LinearRegression()
reg.fit ([[0, 0.5], [1.5, 1], [1.8, 2]], [0, 1, 2])

print(reg.predict([[-1,0.1]]))

### Ridge
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0.5], [1.5, 1], [1.8, 2]], [0, 1, 2])
print(reg.predict([[-1, 0.1]]))

### LASSO
reg = linear_model.Lasso(alpha=.1)
reg.fit([[0, 0.5], [1.5, 1], [1.8, 2]], [0, 1, 2])
print(reg.predict([[-1, 0.1]]))

### Regression Metrics

y_true = [3.1, -0.5, 2.0, 7]
y_pred = [2.7, 0.0, 1.8, 8]

# Mean Absolute Error (MAE)

mean_absolute_error(y_true, y_pred)

# Mean Squared Error (MSE)
np.sqrt(mean_squared_error(y_true, y_pred))

# R2 Score
r2_score(y_true, y_pred) 

# Section I – Cross Validation for OLS, Ridge & Lasso

# 1 - Import the “boston” dataset from sklearn package.
# Do a linear regression by OLS, Ridge and Lasso. Please calculate a 10-fold
# cross validation and calculate RMSE for 10-fold cross validation of each method.

# For your information, k-fold cross validation means that “original sample is
# randomly partitioned into k equal size subsamples. Of the k subsamples,
# a single subsample is retained as the validation data for testing the model,
# and the remaining k-1 subsamples are used as training data.
# The cross-validation process is then repeated k times (the folds), with each
# of the k subsamples used exactly once as the validation data. The k results
# from the folds can then be averaged (or otherwise combined) to produce a
# single estimation. The advantage of this method is that all observations are
# used for both training and validation, and each observation is used for
# validation exactly once”.
# Hint : Call KFold library from sklearn

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()
x = boston.data
y = boston.target

#========================= OLS =============================
print('-----------------------------------------------')
linreg = LinearRegression()
linreg.fit(x, y)
p = linreg.predict(x)
### Kfold
# Compute RMSE using 10-fold cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train, test in kf:
    linreg.fit(x[train], y[train])
    p = linreg.predict(x[test])
    e = p - y[test]
    xval_err += np.dot(e, e)
    rmse_10cv = np.sqrt(xval_err / len(x))

print('OLS RMSE on 10-fold CV: %.4f' %rmse_10cv)
#======================= Ridge =============================
### ridge example
print('-----------------------------------------------')
ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(x, y)
p = ridge.predict(x)

# Compute RMSE using 10-fold Cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
    ridge.fit(x[train],y[train])
    p = ridge.predict(x[test])
    e = p-y[test]
    xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))
print('Ridge RMSE on 10-fold CV: %.4f' %rmse_10cv)
#======================== Lasso ============================
### LASSO example
print('-----------------------------------------------')
lasso = Lasso(fit_intercept=True, alpha=0.1)
lasso.fit(x, y)
p = lasso.predict(x)
# Compute RMSE using 10-fold cross-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
    lasso.fit(x[train],y[train])
    p = ridge.predict(x[test])
    e = p-y[test]
    xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))
print('Lasso RMSE on 10-fold CV: %.4f' %rmse_10cv)

# Examples

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pylab as pl
boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)
#print(boston.data)
x = boston.data
y = boston.target
#================== Split Data into Train & Test ==================
X_train, X_test, y_train, y_test = train_test_split(x, y,
test_size=0.2)
#============================= OLS =================================
# Create linear regression object
OLS = LinearRegression()
# Train the model using the training sets
OLS.fit(X_train, y_train)
p_OLS_train = OLS.predict(X_train)
print('Regression Coefficients: \n', OLS.coef_)
p_OLS_train = OLS.predict(X_train)
MAE_OLS_Train = mean_absolute_error(y_train, p_OLS_train)
RMSE_OLS_Train= np.sqrt(mean_squared_error(y_train, p_OLS_train))
R2_OLS_Train = r2_score(y_train, p_OLS_train)
p_OLS_test = OLS.predict(X_test)
MAE_OLS_Test = mean_absolute_error(y_test, p_OLS_test)
RMSE_OLS_Test= np.sqrt(mean_squared_error(y_test, p_OLS_test))
R2_OLS_Test = r2_score(y_test, p_OLS_test)

#============================= Ridge ===============================
# Create linear regression object with a ridge coefficient 0.5
#Train the model using the training set
ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(X_train, y_train)
p_ridge_train = ridge.predict(X_train)
MAE_Ridge_Train = mean_absolute_error(y_train, p_ridge_train)
RMSE_Ridge_Train= np.sqrt(mean_squared_error(y_train,p_ridge_train))
R2_Ridge_Train = r2_score(y_train, p_ridge_train)
p_ridge_test = ridge.predict(X_test)
MAE_Ridge_Test = mean_absolute_error(y_test, p_ridge_test)
RMSE_Ridge_Test= np.sqrt(mean_squared_error(y_test, p_ridge_test))
R2_Ridge_Test = r2_score(y_test, p_ridge_test)

#============================= Lasso ===============================
# Create linear regression object with a lasso coefficient 0.5
#Train the model using the training set
lasso = Lasso(fit_intercept=True, alpha=0.1)
lasso.fit(X_train, y_train)
p_lasso_train = lasso.predict(X_train)
MAE_Lasso_Train = mean_absolute_error(y_train, p_lasso_train)
RMSE_Lasso_Train= np.sqrt(mean_squared_error(y_train, p_lasso_train))
R2_Lasso_Train = r2_score(y_train, p_lasso_train)
p_lasso_test = lasso.predict(X_test)
MAE_Lasso_Test = mean_absolute_error(y_test, p_lasso_test)
RMSE_Lasso_Test= np.sqrt(mean_squared_error(y_test, p_lasso_test))
R2_Lasso_Test = r2_score(y_test, p_lasso_test)

#====================== Print Error Criteria =======================
print('-----------------OLS------------------')
print(MAE_OLS_Test)
print(RMSE_OLS_Test)
print(R2_OLS_Test)
print('----------------Ridge-----------------')
print(MAE_Ridge_Test)
print(RMSE_Ridge_Test)
print(R2_Ridge_Test)
print('----------------Lasso-----------------')
print(MAE_Lasso_Test)
print(RMSE_Lasso_Test)
print(R2_Lasso_Test)
#========================= Plot Outputs ============================
#matplotlib inline
pl.plot(p_OLS_train , y_train,'ro')
pl.plot(p_ridge_train , y_train,'bo')
pl.plot(p_lasso_train , y_train,'co')
pl.plot([0,50],[0,50] , 'g-')
pl.xlabel('Predicted')
pl.ylabel('Real Target Values')
pl.show()

