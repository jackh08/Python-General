#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week III
Data Programming With Python – Fall / 2017
Array & Data Frame
"""

import numpy as np
import numpy.linalg as npl
import pandas as pd

# Section I – Arrays

# 1 - Write a Python program to convert a list and tuple into arrays
# List to array  [1 2 3 4 5 6 7 8]
# Tuple to array
# [[8 4 6]
# [1 2 3]]

my_list = [1, 2, 3, 4, 5, 6, 7, 8]
print("List to array: ")
print(np.asarray(my_list))
my_tuple = ([8, 4, 6], [1, 2, 3])
print("Tuple to array: ")
print(np.asarray(my_tuple))

# 2 – Write a Python program to create a 3x3 matrix with values ranging
# from 2 to 10.

x = np.arange(2, 11).reshape(3,3)
print(x)

# 3 - Write a Python program to reverse an array (first element becomes last).

x = np.arange(12, 38)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)

# 4 - Write a Python program to convert the values of Centigrade degrees
# into Fahrenheit degrees.
# Centigrade values are stored into a NumPy array.
# Sample Array [0, 12, 45.21 ,34, 99.91]

fvalues = [0, 12, 32, 45.21, 34, 99.91]
F = np.array(fvalues)
print("Values in Fahrenheit degrees:")
print(F)
print("Values in Centigrade degrees:")
print(5*(F - 32)/9)

# 5 - Write a Python program to find the real and imaginary parts of an array
# of complex numbers
### complex numbers
x = np.sqrt([1+0j, 0+1j])
print("Original array", x)
print("Real part of the array:")
print(x.real)
print("Imaginary part of the array:")
print(x.imag)

# 6 - Write a Python program to find the union of two arrays. Union will return
# the unique, sorted array of values that are in either of the two input arrays
# Array1: [ 0 10 20 40 60 80]
# Array2: [10, 30, 40, 50, 70]

array1 = np.array([0, 10, 20, 40, 60, 80])
print("Array1: ", array1)
array2 = [10, 30, 40, 50, 70]
print("Array2: ", array2)
print("Unique sorted array of values that are in either of the two input arrays:")
print(np.union1d(array1, array2))

# 7 - Write a Python program to create a 2-D array whose diagonal equals
# [4, 5, 6, 8] and 0's elsewhere.

x = np.diagflat([4, 5, 6, 8])
print(x)

# 8 - Write a Python program to concatenate two 2-dimensional arrays.
# Sample arrays: ([[0, 1, 3], [5, 7, 9]], [[0, 2, 4], [6, 8, 10]]

a = np.array([[0, 1, 3], [5, 7, 9]])
b = np.array([[0, 2, 4], [6, 8, 10]])
c = np.concatenate((a, b), 1)
print(c)

# 9 - Consider the vector [1, 2, 3, 4, 5], how to build a new vector
# with 3 consecutive zeros interleaved between each value?
### interleaving
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)

# 10 - Create a null vector of size 10 but the fifth value which is 1

Z = np.zeros(10)
Z[4] = 1
print(Z)

# Section II – Import Data

# 11 - Import the ‘Car’ data from following url
# https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv

cars = pd.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
print(cars)

# Create an array of all zeros
A = np.zeros((2,2))
print(A)
# Prints "[[ 0. 0.] [ 0. 0.]]"

# Create an array of all ones
B = np.ones((1,2))
print(B)
# Prints "[[ 1. 1.]]“

# Create a constant array
C = np.full((2,2), 7)
print(C)
# Prints "[[ 7. 7.] [ 7. 7.]]“

# Create a 2x2 identity matrix
D = np.eye(2)
print(D)
# Prints "[[ 1. 0.] [ 0. 1.]]"

# Create an array filled with random v
E = np.random.random((2,2))
print(E) 

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# contains all elements from the beginning, to third (third excluded)
y = x[:2]
print y
# ([1. , 2.])

# starts from 2nd element to the end, prints by steps of 2 elements
y = x[1::2]
print y
# ([2. , 4.])

a = np.array([1,2,4,2,3])
# Sum - summation of elements
print(a.sum())
# Prod - product of elements
print(a.prod())
# Shape - size of array
# a.shape()
# Min/Max - min of array
print(a.min())
# Sorted - sort elements
print(a.sort())
# Mean - mean of elements
print(a.mean())
# Unique - unique elements
print(np.unique(a))
# Dot - dot product
print(np.dot(a,a))

# Arange 
print(np.arange(3,7,2) )
# Flatten
a = np.array([[1,2], [3,4]])
print(a.flatten())
# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)
# Asarray
a = [1, 2]
print(np.asarray(a))
#Len
a = np.array([1,2,3,4])
len(a)

# When standard mathematical operations are used with arrays, they are applied
# on an element-by-element basis. This means that the arrays should be the same
# size.

a = np.array([1,2,3], )
b = np.array([5,2,6], )
print(a + b)
# >>> ([6.,4., 9.])
print(a - b)
# >>> ([-4., 0., -3.])
print(a * b)
# >>> ([5., 4., 18.])
print(b / a)
# >>> ([5., 1., 2.])
print( a % b)
# >>> ([1., 0., 3.])
print( b**a)
# >>> ([5., 4., 216.])

x = np.array( ((2,3), (3, 5)) )
y = np.array( ((1,2), (5, -1)))
print(x * y)
#>>> ([[ 2, 6], [15, -5]])

x = np.matrix( ((2,3), (3, 5)) )
y = np.matrix( ((1,2), (5, -1)))
print(x * y)
# >>> ([[17, 1], [28, 1]])

a = np.matrix( ((2,3), (3,5)) )

# Diagonal >
print(a.diagonal())
# Transpose >
print(a.transpose())
# Trace > no direct way
print(sum(a.diagonal())) 

x = np.matrix([[1.0,0.5],[.5,1]])
print(npl.cond(x))
# >>> 3
# Singular Matrix
x = np.matrix([[1.0,2.0],[1.0,2.0]])
print(npl.cond(x))
# >>> inf


# det > Computes the determinant of a square matrix.
a = np.matrix([[1,.5],[.5,1]])
print(npl.det(a))
#>>> 0.75

#matrix_rank > Computes the rank of a matrix
a = np.matrix([[1,.5],[1,.5]])
print(npl.matrix_rank(a))
#>>> 1

#Inverse > Computes the inverse of matrix
a = np.matrix([[2,.5],[.5,1]])
print(npl.inv(a))
#>>> [[ 0.57142857 -0.28571429] [-0.28571429 1.14285714]]

# EXAMPLES – WEEK 3

# Write a Python program to create a 2d array with 1 on the border and 0 inside

x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)

# Write a Python program to test if specified values are present in an array.

x = np.array([[1.12, 2.0, 3.45], [2.33, 5.12, 6.0]], float)
print("Original array:")
print(x)
print(2 in x)
print(0 in x)
print(6 in x)
print(2.3 in x)
print(5.12 in x)

# Add an array at the end of another array

x = [10, 20, 30]
print("Original array:")
print(x)
x = np.append(x, [[40, 50, 60], [70, 80, 90]])
print("After append values to the end of the array:")
print(x)

# Write a Python program (using numpy) to sum of all the multiples of 3 or 5 below 100.

x = np.arange(1, 100)
# find multiple of 3 or 5
n = x[(x % 3 == 0) | (x % 5 == 0)]
print(n[:1000])
# print sum the numbers
print(n.sum())

# Write a Python program to remove the negative values in a numpy array with 0.

x = np.array([-1, -4, 0, 2, 3, 4, 5, -6])
print("Original array:")
print(x)
print("Replace the negative values of the said array with 0:")
x[x < 0] = 0
print(x)

# Write a Python program to count the frequency of unique values in numpy array

a = np.array( [10,10,20,10,20,20,20,30,30,50,40,40] )
print("Original array:")
print(a)
unique_elements, counts_elements = np.unique(a, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Solve a 3-unknown equation

# define matrix A using Numpy arrays
X = np.array([[2, 1, 1],[1, 3, 2],[1, 0, 0]])
#define matrix Y
Y = np.array([4, 5, 6])
print("Solutions:\n",npl.solve(X, Y ))

### Singular Value Decomposition

X = np.random.normal(size=[4,4])
print(X)
U, S, V = np.linalg.svd(X)
X_a = np.dot(np.dot(U, np.diag(S)), V)
print(X_a)

# Pandas (Additional Example) – this example was not covered in the lecture
# videos, but I thought might be helpful.
### dictionary
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
 "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Johannesburg"],
 "area": [8.516, 17.10, 3.286, 9.597, 1.221],
 "population": [200.4, 143.5, 1252, 1357, 52.98] }

brics = pd.DataFrame(dict)
print(brics)
