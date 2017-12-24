#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week XII
Data Programming With Python – Fall / 2017
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
#===========/////////////////////===================================== ////////////////==========#
#===========/////////////////////========= Regression ================ ////////////////==========#
#---------------------------- Functions defined --------------------------#
def feature_normalize(X):
    mean_r = []
    std_r = []
    X_norm = X
    n = X.shape[1]
    for i in range(n):
        meann = np.mean(X[:, i])
        s = np.std(X[:, i])
        mean_r.append(meann)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - meann) / s
    return X_norm, mean_r, std_r

def compute_cost(X, y, theta):
    num_of_data = y.size
    predictions = X.dot(theta)
    squareErrors = (predictions - y)
    J = (0.5 * num_of_data) * squareErrors.T.dot(squareErrors)
    return J

def gradient_descent(X, y, theta, learning_rate, num_iters):
    num_of_samples = y.size
    J = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        predictions = X.dot(theta)
        theta_size = theta.size
        for it in range(theta_size):
            temp = X[:, it]
            temp.shape = (num_of_samples, 1)
            errors_x1 = (predictions - y) * temp
            theta[it][0] = theta[it][0] - learning_rate * (1/num_of_samples) * errors_x1.sum()
        J[i, 0] = compute_cost(X, y, theta)
    return theta, J

# --------------------- Load the dataset ---------------------------------#
# Link for the TIMBER dataset used in this example==>
# http://www.statsci.org/data/oz/timber.txt
data = np.loadtxt('/Users/c_dredmond/Downloads/timber.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
y.shape = (m, 1)
# ------------ Scale features and set them to zero mean ------------------#
x, mean_r, std_r = feature_normalize(X)
# ------------ Add a column of ones to X (interception data) --------------#
un = np.ones(shape=(m, 3))
un[:, 1:3] = x #replace the last two columns with input data
print(x)
print('---------- After Adding intercept -------------')
print(un)
# ------------ Initializing Theta and Processing Gradient Descent ----------#
iterations = 300
learning_rate = 0.01
theta = np.zeros(shape=(3, 1))
theta, J = gradient_descent(un, y, theta, learning_rate, iterations)
# ---------------------------- Predict price ------------------------------ #
Density = np.array([1.0, ((0.7 - mean_r[0]) / std_r[0]), ((0.8 - mean_r[1]) /
std_r[1])]).dot(theta)
print('Predicted density of a 0.7 rigidity, 0.8 elasticity timber: %f' % (Density))
# --------------------------- Plotting ------------------------------------ #
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(np.arange(iterations), J, lw=5, c='red')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost Function')
ax2 = fig.add_subplot(122, projection='3d')
for c, m, zl, zh in [('g', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax2.scatter(xs, ys, zs, c=c, s=180, marker=m)
ax2.set_xlabel('Rigidity')
ax2.set_ylabel('Elasticity')
ax2.set_zlabel('Density')
plt.show()
#==============/////////////////////======================================//////////////////=========#
#==============/////////////////////========= Clustering ================= /////////////////=========#
#-------------------- Generating Synthetic Data -------------#
X, y_true = make_blobs(n_samples=300, n_features=3, centers=4, cluster_std=0.70,
random_state=0)
x_ax = X[:, 0]
y_ax = X[:, 1]
z_ax = X[:, 2]
#-------------------------- KMEAN ---------------------------#
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_
#-------------------------- Plotting ----------------------- #
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_ax, y_ax, z_ax, s=150)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_ax, y_ax, z_ax, c=y_kmeans, s=100, cmap='viridis')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200)
plt.show()
#==============/////////////////////======================================//////////////////===========
#==============/////////////////////========= CLASSIFIER ================= /////////////////===========
#==================== Import & Trim the Dataset ==================#
df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt', header=None)
X = df[df.columns[0:2]]
y = df[df.columns[4]]
X = np.asarray(X.values)
y = np.asarray(y.values)
#================= Processing the Classification =================#
C = 1.0
lin_svc = svm.SVC(kernel='linear', C=C).fit(X, y) # SVC with linear kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.9, C=C).fit(X, y) # SVC with RBF kernel
poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X, y) # SVC with polynomial (degree 3) kernel
lin_test = lin_svc.predict(X)
rbf_test = rbf_svc.predict(X)
pol_test = poly_svc.predict(X)
#======================= Confusion Matrix ========================#
cnf_matrix_lin = confusion_matrix(y, lin_test)
cnf_matrix_rbf = confusion_matrix(y, rbf_test)
cnf_matrix_pol = confusion_matrix(y, pol_test)
print(cnf_matrix_lin)
print(cnf_matrix_rbf)
print(cnf_matrix_pol)
#===================== Plotting Data & a Mesh ====================#
h = .02 # step size in the mesh
# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial (degree 3) kernel']
for i, clf in enumerate((lin_svc , rbf_svc , poly_svc)):
    plt.subplot(2, 3, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('INPUT 1')
    plt.ylabel('INPUT 2')
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
    plt.subplot(2, 3, i+4)
    if i==0:
        sns.heatmap(cnf_matrix_lin.T, square=True, annot=True, fmt='d', cbar=False)
    if i==1:
        sns.heatmap(cnf_matrix_rbf.T, square=True, annot=True, fmt='d', cbar=False)
    if i==2:
        sns.heatmap(cnf_matrix_pol.T, square=True, annot=True, fmt='d', cbar=False)##
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title(titles[i])
plt.show()
#==========///////////////////===================================///////////////////////==========#
#==========///////////////////========== TIME SERIES=============///////////////////////==========#
# load dataset
df = pd.read_csv('/Users/c_dredmond/Downloads/TimeSer.csv', header=0)
df['Date'] = pd.to_datetime(df['Date'])
indexed_df = df.set_index('Date')
timeseries = indexed_df['Value']
split_point = len(timeseries) - 10
dataset, validation = timeseries[0:split_point], timeseries[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
#def difference(dataset, interval=1):
# diff = list()
# for i in range(interval, len(dataset)):
# value = dataset[i] - dataset[i - interval]
# diff.append(value)
# return np.array(diff)
#X = dataset.values
#months_in_year = 12
#differenced = difference(X, months_in_year)
#fit model
history = [x for x in dataset]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(validation)):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = validation[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % ((yhat), (obs)))