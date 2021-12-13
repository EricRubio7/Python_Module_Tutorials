import numpy as np
#From: https://numpy.org/doc/stable/user/quickstart.html
"""The Basics"""
print("\nTHE BASICS\n")
a = np.arange(15).reshape(3, 5)
print("a is: a = np.arange(15).reshape(3, 5)")
print("a:",a)
print("a shape:", a.shape)
print("a.ndim:", a.ndim)
print("a.dtype.name: ", a.dtype.name)
print("a.itemsize: ", a.itemsize)
print("a.size: ", a.size)
print("type(a): ", type(a))
b = np.array([6,7,8])
print("b is: b = np.array([6,7,8])")
print("b: ", b)
print("type(b): ", type(b))
print("")

"""Array Creation"""
print("\nARRAY CREATION\n")
print("a is: a = np.array([2,3,4])")
a = np.array([2,3,4])
print("b is: b = np.array([1.2, 3.5, 5.1]")
b = np.array([1.2, 3.5, 5.1])
print("This is wrong: a = np.array(1, 2, 3, 4)")
print('''The array function can take only 1 or 2 positional arguments...
4 were provided\n''')
print("Arrays can be multidimensional")
print("b is: b = np.array([(1.5, 2, 3), (4, 5, 6)])")
b = np.array([(1.5, 2, 3), (4, 5, 6)])
print("b: ",b)
print("\nArray type can be specified at creation time")
print("c is: c = np.array([[1, 2], [3, 4]], dtype=complex)")
c = np.array([[1, 2], [3, 4]], dtype=complex)
print("c: ", c)
print("\nWe can create zero or ones arrays/matrixes")
print ("np.zeros((3, 4)): ", np.zeros((3, 4)))
print("np.ones((2, 3, 4), dtype=np.int16)\n", np.ones((2, 3, 4), dtype=np.int16))
print("np.empty((2, 3)): ", np.empty((2, 3)))
print("\nWe can create a sequence of numbers (similar to range)")
print("np.arange(10, 30, 5): ", np.arange(10, 30, 5))
print("np.arange(0, 2, 0.3): ", np.arange(0, 2, 0.3))
print("It's generally not possible to predict the size of the array when using arange")
print("With that in mind we usually use linespace")
print("Linespace receives an argument for how many number we want")
print("from numpy import pi")
from numpy import pi
print("np.linspace(0, 2, 9): ", np.linspace(0, 2, 9))
print("Linespace is useful to evaluate functinos at lots of points")
x = np.linspace(0, 2 * pi, 100)
f = np.sin(x)
print("x = np.linspace(0, 2 * pi, 100)", x)
print("f = np.sin(x): ", f)
print("\nSee also: array, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, linspace, numpy.random.Generator.rand, numpy.random.Generator.randn, fromfunction, fromfile")

"""Printing Arrays"""
print("\nPRINTING ARRAYS")
print('''When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:

the last axis is printed from left to right,
the second-to-last is printed from top to bottom,
the rest are also printed from top to bottom, with each slice separated from the next by an empty line.
One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.''')
a = np.arange(6) 
print("a = np.arange(6): ", a)
b = np.arange(12).reshape(4, 3)
print("b = np.arange(12).reshape(4, 3)\n", b)
c = np.arange(24).reshape(2, 3, 4)
print("c = np.arange(24).reshape(2, 3, 4):\n", c)
print('''\nIf an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:''')
print(np.arange(10000))
print(np.arange(10000).reshape(100, 100))

'''Basic operations'''
print("\nBASIC OPERATIONS\n")
print("Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.")
a = np.array([20, 30, 40, 50])
b = np.arange(4)
c = a - b
print("a:", a)
print("b: ", b)
print("c = a - b: ", c)
print("b**2: ", b**2)
print("10 * np.sin(a): ", 10 * np.sin(a))

print('''\n Unlike in many matrix languages, the product operator * operates elementwise in NumPy arrays. The matrix product can be performed using the @ operator (in python >=3.5) or the dot function or method:''')
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print("A: ", A)
print("B: ", B)
print("Elementwise product - A * B: ", A * B)
print("Matrix product - A @ B: ", A @ B)
print("Matrix product - A.dot(B): ", A.dot(B))

print("\n Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.")
rg = np.random.default_rng(1)
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
print(a)
b += a
print(b)
print('''\nWhen operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).''')

'''Universal Functions'''
print("\n UNIVERSAL FUNCTIONS\n")
print('''NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal functions” (ufunc). Within NumPy, these functions operate elementwise on an array, producing an array as output.''')

print('''\nSee also: all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, invert, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where''')

'''Indexing, Slicing and Iterating'''
print("\nINDEXING, SLICING AND ITERATING\n")
print('''One-dimensional arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.''')

a = np.arange(10)**3
print("a is: a = np.arange(10)**3", a)
print("a[2:5]", a[2:5])
a[:6:2] = 1000
print("a[:6:2] = 1000: ", a)
print("a[::-1]: ", a[::-1])
for i in a:
  print(i**(1 / 3.))
print('''Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:''')
def f(x, y):
  return 10 * x + y
b = np.fromfunction(f, (5, 4), dtype=int)
print("np.fromfunction(f, (5, 4), dtype=int)", b)
print("b[0:5, 1]: ", b[0:5, 1])
print("b[:, 1]: ", b[:, 1])
print("b[1:3, :]: ", b[1:3, :])
print('''When fewer indices are provided than the number of axes, the missing indices are considered complete slices:''')
print("b[-1]: ", b[-1])
print("Iterating over multidimensional arrays is done with respect to the first axis:")
for row in b:
  print(row)

print("\n However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:")
for element in b.flat:
  print(element)
