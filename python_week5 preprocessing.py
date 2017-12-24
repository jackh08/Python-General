#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week V
Data Programming With Python – Fall / 2017
Standardization, Normalization, Binarization, Missing Data Imputation,
Principal Component Analysis (PCA)
"""

import numpy as np
import pandas as pd
import scipy.cluster.vq as spcv
import sklearn.decomposition as skd
import sklearn.preprocessing as skp
from sklearn.datasets import load_iris

iris = load_iris() #load the Iris dataset
X = iris.data
y = iris.target

# Standardized_X = (X – Average) / Std_Deviation
### standardize
scaler = skp.StandardScaler().fit(X)
standardized_X = scaler.transform(X)
print(standardized_X)

standardized_X_inverse = scaler.inverse_transform(standardized_X)
print(standardized_X_inverse)

# or use the following
# standardized_Dataset = skp.scale(Dataset, axis=0)

# Normalized_X = (X – min)/(max-min)

### normalise

normalizer = skp.Normalizer().fit(X)
normalized_X = normalizer.transform(X)
print(normalized_X)

# or use the following
X = [[ 1.,-1., 2.],[ 2., 0., 0.],[ 0.,1., -1.]]
normalized_X = skp.normalize(X, norm='l1')
print(normalized_X)

### Binarization of Data
# Defined as the process of finding a threshold t for numerical
# features, and assign one - out of two values - to all members of the
# same group as a result of splitting around the value of t. 

Dataset = [[1.0,2.0,3.0,4.0],[2.0,3.0,4.0, 5.0]]
binarizer = skp.Binarizer(threshold=0.0).fit(Dataset)
binarized_Dataset = binarizer.transform(Dataset)
print(binarized_Dataset)

# or use the following
Bin_Data = skp.binarize([[1.0,2.0,3.0,4.0],[2.0,3.0,4.0, 5.0]], threshold=2.5)
print(Bin_Data)

### Missing data imputation
Dataset = np.array([[23.56],[53.45],['NaN'],[44.44],[77.78],['NaN'],[234.44],[11.33],[79.87]])
imp = skp.Imputer(missing_values='NaN', strategy='mean', axis=0)
Dataset_imputed = imp.fit_transform(Dataset)
print(Dataset_imputed)

### Principal Component Analysis

# 1 Standardize the data.
# 2 Obtain the Eigenvectors and Eigenvalues from the covariance matrix
#   or correlation matrix, or perform Singular Vector Decomposition.
# 3 Sort eigenvalues in descending order and choose the k eigenvectors that
#   correspond to the k largest eigenvalues where k is the number of dimensions
#   of the new feature subspace (k≤d).
# 4 Construct the projection matrix W from the selected k eigenvectors.
# 5 Transform the original dataset X via W to obtain a k-dimensional
#   feature subspace X_reduced

pca = skd.PCA(n_components=2)
Dataset = np.array([[0.387,4878, 5.42],[0.723,12104,5.25], [1,12756,5.52],[1.524,6787,3.94],])
pca.fit(Dataset)
Dataset_Reduced_Dim = pca.transform(Dataset)
print(Dataset_Reduced_Dim )

### Examples

from sklearn.datasets import load_breast_cancer
# load cancer data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(X.shape)
print('-----------------------Original-----------------------------')
print(X)
print('----------------------Standardized--------------------------')
scaler = skp.StandardScaler().fit(X)
standardized_X = scaler.transform(X)
print(standardized_X)
Normalized_X = skp.normalize(X, norm='l2')
print('-----------------------Normalized---------------------------')
print(Normalized_X)
Binarized_X = skp.binarize(X,threshold=0.5)
print('-----------------------Binarized----------------------------')
print(Binarized_X)

DataFrame = pd.read_csv('/Users/c_dredmond/Downloads/Data_with_Missing_values.csv', header=None)
DataMatrix = DataFrame.as_matrix()
f = np.array(DataMatrix)
print(f)
imp = skp.Imputer(missing_values='NaN', strategy='mean', axis=0)
print(imp.fit_transform(f))
###pca example
# Create a signal with only 2 useful dimensions
x1 = np.random.normal(size=5)
x2 = np.random.normal(size=5)
x3 = x1 + x2
# stack column-wise array
X = np.c_[x1, x2, x3]
print(x1)
print('---------------------------')
print(x2)
print('---------------------------')
print(X)
pca = skd.PCA(n_components=3)
pca.fit(X)
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)
U, S, V = np.linalg.svd(X)
print(U)
print('----------------')
print(S)
print('----------------')
print(V)
# As we can see, only the 2 first components are useful

# Preprocessing Data with sklearn

# 1- Load the ‘diabetes’ dataset from sklearn dataset library,
# and do the followings :
# Standardize the data
# Normalize the data
# Reduce the dimension of the data to 4 columns with PCA
# Cluster the input features with k-mean clustering library of scipy package,
# to 4 clusters
### full example
from sklearn.datasets import load_diabetes

# load the dataset
diabetes = load_diabetes(return_X_y=False)

# set the dataset and targe
Dataset = diabetes.data
Target = diabetes.target

# print the shape
print(Dataset.shape)

# standardize
standardized_Dataset = skp.scale(Dataset, axis=0)
# normalize
Normalized_Dataset = skp.normalize(standardized_Dataset, norm='l2')
# compute the principal component analysis
pca = skd.PCA(n_components=4, whiten=False)

# fit the data
pca.fit(Normalized_Dataset)

# transform the data
Dataset_Reduced_Dim = pca.transform(Normalized_Dataset)

# print the new shape
print(Dataset_Reduced_Dim.shape)

### scipy kmeans
# run the kmeans and get the centroids - for 4 clusters
centroids,var = spcv.kmeans(Dataset_Reduced_Dim,4)

# calculate the cluster each point belongs to
id,dist = spcv.vq(Dataset_Reduced_Dim,centroids)

print(centroids)
print('---------------------------------------------------')
print(id)
