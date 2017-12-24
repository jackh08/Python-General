#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week IV
Data Programming With Python – Fall / 2017
Regression, Clustering
"""

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as nppp
import scipy.cluster.vq as spcv
import scipy.stats as sps

from pylab import plot, title, show, legend
from scipy import linspace, sqrt, randn
from scipy.cluster.vq import kmeans, vq

# Section I – Regression

# 1- Import the ‘Auto Insurance in Sweden’ dataset from the following url,
# and do a linear regression to fit the data. Plot the data and the regression line.
# url - https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt
### sps lin regression
DataFrame = pd.read_csv('/users/c_dredmond/Downloads/AutoInsurSweden.txt', header=None)
DataMatrix = DataFrame.as_matrix()
InputMatrix= np.array(DataMatrix[:,0])
OutMatrix = np.array(DataMatrix[:,1])
(gradient,intercept,rvalue,pvalue,stderr) = sps.linregress(InputMatrix, OutMatrix)
Regression_line = nppp.polyval(InputMatrix,[intercept,gradient])
print ("Gradient & Intercept", gradient, intercept)
plot(InputMatrix,OutMatrix, 'vr')
plot(InputMatrix,Regression_line ,'b.-')
show()

# Section II – Clustering

# 2 - Download the ‘IRIS’ dataset from the url below, import it to Python and
# do a 3-mean clustering based on the inputs (4-dimesnion).
# Plot the members of each cluster with different colour (Red, Blue & Green )
# in a 2-axis coordinate which the horizontal axis is the first input and the
# vertical one is second input.
# url - https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
### k means
DataFrame = pd.read_csv('/users/c_dredmond/Downloads/iris.csv', header=None)
DataMatrix = DataFrame.as_matrix()
InputMatrix = np.matrix(DataMatrix[:,:4])
centroids,_ = kmeans(InputMatrix,3)
id,_ = vq(InputMatrix,centroids)
print(centroids)
print(id)
plot(InputMatrix[id==0,0], InputMatrix[id==0,1],
     '*b', InputMatrix[id==1,0], InputMatrix[id==1,1],
     'vr', InputMatrix[id==2,0], InputMatrix[id==2,1],
     'og', linewidth=5.5)
show()

# Linear Regression
x = [5.05, 6.75, 3.21, 2.66]
y = [1.65, 26.5, -5.93, 7.96]

(gradient, intercept, r_value, p_value, stderr) = sps.linregress(x,y)
print ("Gradient & Intercept", gradient, intercept)
print ("R-squared", r_value**2)
print ("p-value ", p_value)

### Polynomial Regression

# x : [-1, -0.96, ..., 0.96, 1]
x = np.linspace(-1,1,51)
# x^3 – x^2 + N(0,1)“Gaussian noise"
y = x**3 - x**2 + np.random.randn(len(x))

c, stats = nppp.polyfit(x,y,3,full=True)
Datasample = -0.77
result = nppp.polyval(Datasample,c)
print(result)

### Clustering

a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a,b),)
Num_of_clusters = 2;
Centre,Var = spcv.kmeans(X, Num_of_clusters )
id,dist = spcv.vq(X,Centre)

print id, dist

#Sample data creation

#number of points
n = 50
t = linspace(-5, 5, n)
#parameters
a = 0.8
b = -4
x = nppp.polyval(t,[a, b])
#add some noise
xn= x+randn(n)
(ar,br) = nppp.polyfit(t,xn,1)
xr = nppp.polyval(t,[ar,br])

#compute the mean square error
err = sqrt(sum((xr-xn)**2)/n)
print('Linear regression using polyfit')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, ms error= %.3f' % (a,b,ar,br,err))
print('-----------------------------------------------------')

#Linear regression using stats.linregress
(a_s,b_s,r,tt,stderr) = sps.linregress(t,xn)
print('Linear regression using stats.linregress')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, std error= %.3f' % (a,b,a_s,b_s,stderr))

#matplotlib ploting
title('Linear Regression Example')
plot(t,x,'g.--')
plot(t,xn,'k.')
plot(t,xr,'r.-')
legend(['original','plus noise', 'regression'])
show()

#generate two clusters: a with 100 points, b with 50:
# for repeatability of this example
np.random.seed(4711)
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],
size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]],
size=[50,])
X = np.concatenate((a,b),)
# 150 samples with 2 dimensions
print(X.shape)
# computing K-Means with K = 2
centroids,_ = kmeans(X,2)
# assign each sample to a cluster
id,_ = vq(X,centroids)
plot(X[id==0,0],X[id==0,1],'ob',X[id==1,0],X[id==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg', markersize=15)
show()
A,B = spcv.kmeans(X,2)
print(A)
print('----------------------------------------------')
print(B)
print('----------------------------------------------')
id,dist=spcv.vq(X,A)
print('----------------------------------------------')
print(id)
print(dist)
