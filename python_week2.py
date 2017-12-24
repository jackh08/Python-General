# -*- coding: utf-8 -*-
"""
Exercise – Week II
Data Programming With Python – Fall / 2017
Variables, Loops, Control Flows, Functions, Plotting
"""

import unittest

# Exercise – Week II
# Data Programming With Python – Fall / 2017

# Section I - Variables

# 1-Write a Python program to count the number of strings where the string
# length is 2 or more and the first and last character are same from a given
# list of strings.

def match_words(words):
    ctr = 0
    for word in words:
        if len(word) > 1 and word[0] == word[-1]:
            ctr += 1
    return ctr

print(match_words(['abc', 'xyz', 'aba', '1221']))

class MatchWordsTest(unittest.TestCase):
    
    def test_match_words(self):
        self.assertEqual(2, match_words(['abc', 'xyz', 'aba', '1221']))

# 2-Write a Python program to multiplies all the items in a list.

def multiply_list(items):
   tot = 1
   for x in items:
       tot *= x
   return tot

print(multiply_list([1,2,-8]))

class MultiplyListTest(unittest.TestCase):
    
    def test_multiply_list(self):
        self.assertEqual(-16, multiply_list([1,2,-8]))

# 3-Write a Python program to get a list, sorted in increasing order by the
# last element in each tuple from a given list of non-empty tuples.
# Sample List : [(0, 5), (-1, 2), (4, 4), (2, 3), (3, 1)]

def last(n):
    return n[-1]

def sort_list_last(tuples):
    return sorted(tuples, key=last)

print(sort_list_last([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]))

class SortListLastTest(unittest.TestCase):
    
    def test_sort_list_last(self):
        self.assertEqual([(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)],
                sort_list_last([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]))

# 4-Write a Python program to print a specified list after removing the
# 0th, 4th and 5th elements.
# Sample List : ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']

color = ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']

# use list comprehension.
def remove_elements(values, indices):
    return [x for (i,x) in enumerate(values) if i not in indices]

removed_color = remove_elements(color, (0, 4, 5))
print(removed_color)

class RemoveElementsTest(unittest.TestCase):
    
    def test_remove_elements(self):
        self.assertEqual(['Green', 'White', 'Black'],
                remove_elements(color, (0, 4, 5)))
        
# Section II – Loops

# 5-Write a Python program which accepts a sequence of comma separated 4 digit
# binary numbers as its input and print the numbers that are divisible by 5 in
# a comma separated sequence.
# Sample Data : 0100,0011,1010,1001,1100,1001

def divisible_by_five(value):
    items = []
    num = [x for x in value.split(',')]
    for p in num:
        x = int(p)
        if not x%5:
            items.append(p)
    return ','.join(items)

print divisible_by_five('1,2,3,4,5,6,7,8,9,10,12')

class DivisibleByFiveTest(unittest.TestCase):
    
    def test_divisible_by_five(self):
        self.assertEqual('5,10', divisible_by_five('1,2,3,4,5,6,7,8,9,10,12'))

# 6-Write a Python program to get the Fibonacci series between 0 to 50.
# Note : The Fibonacci Sequence is the series of numbers :
# 0, 1, 1, 2, 3, 5, 8, 13, 21, ....
# Every next number is found by adding up the two numbers before it.

def fibonacci(max):
    items = []
    x = 1
    y = 0
    while y<max:
        items.append(y)
        x,y = y,x+y
    return items

items = fibonacci(50) 
print items

class FibonacciTest(unittest.TestCase):
    
    def test_fibonacci(self):
        self.assertEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], fibonacci(50))

# Section III – Control Flows

# 7-Write a Python program to check the validity of password input by users.
# Validation :
# Note : PLEASE IMPORT package re and use function search
# • At least 1 letter between [a-z] and 1 letter between [A-Z].
# • At least 1 number between [0-9].
# • At least 1 character from [$#@].
# • Minimum length 6 characters.
# • Maximum length 16 characters.

import re

def check_password(p, debug=False):
    if len(p)<6 or len(p)>16 or not re.search("[a-z]",p) \
            or not re.search("[0-9]",p) or not re.search("[A-Z]",p) \
            or not re.search("[$#@]",p) or re.search("\s",p):
       if debug: print("Not a Valid Password")
       return False
    if debug: print("Valid Password")
    return True

print check_password('abc')
print check_password('abcabcabcabcabcabc')
print check_password('abc123')
print check_password('Abc123')
print check_password('Abc123#')

class CheckPasswordTest(unittest.TestCase):
    
    def test_check_password(self):
        self.assertEqual(False, check_password('abc'))
        self.assertEqual(False, check_password('abcabcabcabcabcabc'))
        self.assertEqual(False, check_password('abc123'))
        self.assertEqual(False, check_password('Abc123'))
        self.assertEqual(True, check_password('Abc123#'))

# Section IV – Functions

# 8-Write a (recursive) function which calculates the factorial of a given
# number. Use exception handling to raise an appropriate exception if the
# input parameter is not a positive integer, but allow the user to enter floats
# as long as they are whole numbers.

def factorial(n):
    ni = int(n)
    if ni != n or ni <= 0:
        raise ValueError("%s is not a positive integer." % n)
    if ni == 1:
        return 1
    return ni * factorial(ni - 1)

print factorial(5)

class FactorialTest(unittest.TestCase):
    
    def test_factorial(self):
        self.assertEqual(1, factorial(1))
        self.assertRaises(ValueError, factorial, -1)
        self.assertEqual(120, factorial(5))

# Section V – Plotting

# 9-Make a program that plots the function g(y)=(e^−y)sin(4y) for y∈[0,4]
# using a red solid line.
# Use 500 intervals for evaluating points in [0,4].
# Store all coordinates and values in arrays.
# Set labels on the axis and use a title “Damped sine wave”.
# [HINT : from numpy import exp and sin both]

import numpy as np
import matplotlib.pyplot as plt
# avoid np. prefix in g(y) formula
from numpy import exp, sin

def g(y):
    return exp(-y)*sin(4*y)

def plot_function(x, y, title, output, x_label, y_label, y2=None):
    plt.figure()
    if y2 is None:
        plt.plot(x, y, 'r-')
    else:
        plt.plot(x, y, 'r-', x, y2, 'k--')
    plt.xlabel(x_label);
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(output + '.png');
    plt.savefig(output + '.pdf')
    plt.show()

x = np.linspace(0, 4, 501)
y = g(x)
plot_function(x, y, 'Damped sine wave', 'damped_sine_wave', '$y$', '$g(y)$')    

# 10-Add a to the above (two plots), a black dashed curve for the function
# h(y)=(e^−1.5y)sin(4y).
# Include a legend for each curve.

def h(y):
    return exp(-(3./2)*y)*sin(4*y)

y2 = h(x)
plot_function(x, y, 'Damped sine wave', 'damped_sine_wave', '$y$', '$g(y)$', y2)    

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)

# Plot cosine using blue color with a continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# Plot sine using green color with a continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
# Set x limits
plt.xlim(-4.0,4.0)
# Set x ticks
plt.xticks(np.linspace(-4,4,9,endpoint=True))
# Set y limits
plt.ylim(-1.0,1.0)
# Set y ticks
plt.yticks(np.linspace(-1,1,5,endpoint=True))
#set legends
plt.plot(X, C, color="red", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")
# Show result on screen
plt.show()

if __name__ == '__main__':
    unittest.main()

