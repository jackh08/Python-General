#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week VIII
Data Programming With Python – Fall / 2017
Multi-class Classification - Data Pre-processing, SVM, NB, Decision Tree,
    Random Forest, Classification Metrics - Confusion Matrix
"""

### Dummy Coding / Encoding

from statsmodels.tools import categorical
import numpy as np
a = np.array(['Type1', 'Type2', 'Type3', 'Type1', 'Type2', 'Type3'])
cat_encod = categorical(a, dictnames=False, drop=True)
print(a.reshape(-1,1))
print(cat_encod)

### Support Vector

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
svc_linear = svm.SVC(kernel='linear', C=1)
svc_linear.fit(X_train, y_train)
predicted= svc_linear.predict(X_train)
print(predicted)
print(y_train)

### Naive Bayes

from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

NB = GaussianNB()
NB.fit(X, Y)
predicted= NB.predict([[1,2],[3,4]])
print(predicted)

### Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
boston = load_boston()
RF = RandomForestRegressor(n_estimators=3)
RF.fit(boston.data[:300], boston.target[:300])
instances = boston.data[[300, 309]]
print("Instance 0 prediction:", RF.predict(instances[[0]]))
print("Instance 1 prediction:", RF.predict(instances[[1]]))

### Confusion Matrix

from sklearn.metrics import confusion_matrix
y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
CM = confusion_matrix(y_actu, y_pred)
print(CM)

### SUPPORT VECTOR MACHINE (SVC)
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# --create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
# --Create the linear SVC model object -------------
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
# - Create the rbf SVC model object --------------
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=C)
svc.fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.subplot(122)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with RBF kernel')
plt.show()
### NAIVE BAYESIAN (NB)
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# -create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_for_plot = np.c_[xx.ravel(), yy.ravel()]
# -Create the NAÏVE Bayesian object -------------
NB = GaussianNB()
NB.fit(X, y)
Z = NB.predict(X_for_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Naive Bayesian')
plt.show()
### RANDOM FOREST (RF)
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# --Loading Dataset --------------------------
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# --create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_for_plot = np.c_[xx.ravel(), yy.ravel()]
# --Create the RF object --------------------------
RF = RandomForestClassifier(n_estimators=3, random_state=12)
RF.fit(X, y)
Z = RF.predict(X_for_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Random Forest')
plt.show()
### CONFUSION MATRIX
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# - Loading Dataset --------------------------#
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# -Create the SVM/NB/RF object --------------------------#
RF = RandomForestClassifier(n_estimators=3, random_state=12)
RF.fit(X, y)
Z_RF = RF.predict(X)
NB = GaussianNB()
NB.fit(X, y)
Z_NB = NB.predict(X)
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)
Z_SV = svc.predict(X)
# - CONFUSION MATRIX -----------------------#
CM_RF = confusion_matrix(y, Z_RF)
CM_NB = confusion_matrix(y, Z_NB)
CM_SV = confusion_matrix(y, Z_SV)
print(CM_RF)
print('----------------------------')
print(CM_NB)
print('----------------------------')
print(CM_SV)
plt.subplot(131)
sns.heatmap(CM_RF.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(132)
sns.heatmap(CM_NB.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(133)
sns.heatmap(CM_SV.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

### Classification

# Import the dataset from the following ‘url’ and do a classification with
# decision tree and Random Forest (RF) with number of trees equal to 5,
# and compare the result of testing data with confusion matrix.
# url http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tools import categorical
from sklearn import tree
# --Loading Dataset --------------------------#
dataframe = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
dataframe = dataframe .drop(dataframe .index[:-1000])
numericdf = dataframe[dataframe .columns[1:9]]
categordf = dataframe[dataframe .columns[0]]
categordf_en = categorical(categordf.values , drop=True)
categordf_en = categordf_en[:, 0:2]
numeric_arr = np.asarray(numericdf.values)
categor_arr = np.asarray(categordf_en)
Output = numeric_arr[:, 7]
Input_numeric = numeric_arr[:, 0:6]
Input_categor = categor_arr
Input = np.concatenate((Input_numeric, Input_categor), axis=1)
#---------------------------------------------------------------
RF = RandomForestClassifier(n_estimators=5, random_state=12)
RF.fit(Input, Output)
Z_RF = RF.predict(Input)
CM_RF= confusion_matrix(Output, Z_RF)
#---------------------------------------------------------------
DT = tree.DecisionTreeClassifier()
DT.fit(Input, Output)
Z_DT = DT.predict(Input)
CM_DT= confusion_matrix(Output, Z_DT)
#---------------------------------------------------------------
plt.subplot(121)
sns.heatmap(CM_RF.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.subplot(122)
sns.heatmap(CM_DT.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

