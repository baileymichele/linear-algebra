# -*- coding: utf-8 -*-
# linear_transformations.py
"""Linear Transformations.
Bailey Smith
September 27 2016
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

def stretch(A, a, b):
    """Scale the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    matrix = [[a,0],[0,b]]
    product = np.dot(matrix,A)
    return product

def shear(A, a, b):
    """Slant the points in 'A' by 'a' in the x direction and 'b' in the
    y direction.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    matrix1 = np.array([[1,float(a)],[float(b),1]])
    return np.dot(matrix1,A)


def reflect(A, a, b):
    """Reflect the points in 'A' about the line that passes through the origin
    and the point (a,b).

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    matrix = [[a**2-b**2,2*a*b],[2*a*b,b**2-a**2]]
    product = np.dot(matrix,A)
    return product/(a**2+b**2)

def rotate(A, theta):
    """Rotate the points in 'A' about the origin by 'theta' radians.

    Inputs:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    matrix_3 = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(matrix_3,A)


def solar_system(T, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (10,0) and the initial
    position of the moon is (11,0).

    Parameters:
        T (int): The final time.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.

        for loop for each number in t
    """
    #Compute location of earth at time t: p(t)
    earth_initial = np.array([10,0])
    moon_initial = np.array([11,0])
    earth_list = []#list of tuples
    moon_list = []#list..append in for loop

    t = np.linspace(0, T, 300)#0 initial time, T final time
    for i in t:
        earth_theta = i*omega_e
        moon_theta = i*omega_m
        earth_position = rotate(earth_initial,earth_theta)
        #Append to earth position list (x,y)
        earth_list.append(earth_position)

        moon_relative = rotate(moon_initial-earth_initial,moon_theta)
        moon_position = moon_relative + earth_position#translate: add
        #take the last thing added to earth position list to firgure our moon position
        #Append to moon position list
        moon_list.append(moon_position)

    Ex, Ey = zip(*earth_list)
    Mx, My = zip(*moon_list)
    plt.plot(Ex,Ey, "b-",label="Earth")
    plt.plot(Mx,My, "g-",label="Moon")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.gca().set_aspect("equal")
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in xrange(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in xrange(n)] for i in xrange(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]


def timit():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    domain = 2**np.arange(1,9)
    MM = []#Matrix-matrix
    MV = []#Matrix-vector
    for n in domain:
        """Generating random matrices"""
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        """Timing matrix-vector multiplication"""
        start = time.time()
        matrix_vector_product(A, x)
        MV.append(time.time() - start)#Putting times into a list so we can plot
        """Timing matrix-matrix multiplication"""
        start = time.time()
        matrix_matrix_product(A, B)
        MM.append(time.time() - start)

    plt.subplot(121)
    plt.plot(domain, MV, 'b.-', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("Matrix-Vector Multiplication", fontsize=15)

    plt.subplot(122)
    plt.plot(domain, MM, 'g.-', linewidth=2, markersize=15)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("Matrix-Matrix Multiplication", fontsize=15)

    plt.show()

def timit2():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    domain = 2**np.arange(1,9)
    MM = []#Matrix-matrix
    MV = []#Matrix-vector
    MM_numpy = []
    MV_numpy = []
    for n in domain:
        """Generating random matrices"""
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        """Timing matrix-vector multiplication using function"""
        start = time.time()
        matrix_vector_product(A, x)
        MV.append(time.time() - start)#Putting times into a list so we can plot
        """Timing matrix-matrix multiplication using function"""
        start = time.time()
        matrix_matrix_product(A, B)
        MM.append(time.time() - start)
        """Timing matrix-vector multiplication using np.dot()"""
        start = time.time()
        np.dot(A, x)
        MV_numpy.append(time.time() - start)
        """Timing matrix-matrix multiplication using np.dot()"""
        start = time.time()
        np.dot(A, B)
        MM_numpy.append(time.time() - start)

    plt.subplot(121)
    plt.plot(domain, MV, 'b.-', lw=2, ms=12, label="Matrix-Vector Python")
    plt.plot(domain, MM, 'g.-', lw=2, ms=12, label="Matrix-Matrix Python")
    plt.plot(domain, MV_numpy, 'm.-', lw=2, ms=12, label="Matrix-Vector Numpy")
    plt.plot(domain, MM_numpy, 'c.-', lw=2, ms=12, label="Matrix-Matrix Numpy")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("Multiplication on linear scale", fontsize=15)


    plt.subplot(122)
    plt.loglog(domain, MV, 'b.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, MM, 'g.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, MV_numpy, 'm.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, MM_numpy, 'c.-', basex=2, basey=2, lw=2, ms=12)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("Matrix-Matrix Multiplication", fontsize=15)

    plt.show()

def check(A,B):#pass in original matrix and changed matrix
    plt.subplot(121)
    plt.plot(A[0], A[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.title("Original", fontsize=18)
    plt.subplot(122)
    plt.plot(B[0], B[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.title("Altered", fontsize=18)
    plt.show()

