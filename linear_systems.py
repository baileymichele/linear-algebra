# -*- coding: utf-8 -*-
# linear_systems.py
"""Linear Systems.
Bailey Smith
October 4, 2016
"""

import numpy as np
from math import sqrt
from scipy import linalg as la
import time
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
import scipy

def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.
    """
    for i in xrange(int(sqrt(A.size))-1):
        for j in xrange(i+1,int(sqrt(A.size))):
             A[j,i:] -= (A[j,i] / A[i,i]) * A[i,i:]
    return A


def lu(A):
    """Compute the LU decomposition of the square matrix A. You may assume the
    decomposition exists and requires no row swaps.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m,n = np.shape(A)
    U = np.copy(A)#copy A
    L = np.eye(m)
    for j in xrange(0,n-1):
        for i in xrange(j+1,m):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]
    return L, U

def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may assume that A is invertible (hence square).
    Return x as np array
    """
    L, U = lu(A)
    n = np.shape(L)[0]
    y_list = []
    x_list = []
    for i in xrange(n):
        x_list.append(0)

    #Solve for y
    y_sum = 0
    y_list.append(b[0] - y_sum)
    for k in xrange(1,n):
        y_sum = 0
        for j in xrange(0,k):
            y_sum += L[k,j]*y_list[j]
        y_list.append(b[k] - y_sum)

    #Solve for x
    for k in xrange(n-1,-1,-1):
        x_sum = 0
        for j in xrange(n-1,k-1,-1):
            x_sum += U[k,j]*x_list[j]
        x_list[k] = ((1/U[k,k])*(y_list[k] - x_sum))

    return np.array(x_list)

def timing():
    """Time different scipy.linalg functions for solving square linear systems.
    Plot the system size versus the execution times. Use log scales if needed.
    """
    domain = 2**np.arange(1,9)
    one = []
    two = []
    three = []
    four = []
    for n in domain:
        """Generating random matrices"""
        A = np.random.rand(n,n)
        b = np.random.rand(n,1)
        """Timing #1"""
        start = time.time()
        inverse = la.inv(A)
        np.dot(inverse, b)
        one.append(time.time() - start)
        """Timing #2"""
        start = time.time()
        la.solve(A,b)
        two.append(time.time() - start)
        """Timing #3"""
        start = time.time()
        lu, piv = la.lu_factor(A)
        la.lu_solve((lu,piv), b)
        three.append(time.time() - start)
        """Timing #4"""
        lu, piv = la.lu_factor(A)
        start = time.time()
        la.lu_solve((lu,piv), b)
        four.append(time.time() - start)

    #plot
    plt.loglog(domain, one, 'b.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, two, 'g.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, three, 'm.-', basex=2, basey=2, lw=2, ms=12)
    plt.loglog(domain, four, 'c.-', basex=2, basey=2, lw=2, ms=12)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")
    plt.title("Timed Linear Algebra Functions", fontsize=15)

    plt.show()


def tridiag(n):
    """Return a sparse n x n tridiagonal matrix with 2's along the main
    diagonal and -1's along the first sub- and super-diagonals.
    """
    return sparse.diags([-1,2,-1], [-1,0,1], shape=(n,n))


def timing():#WORKS I Think!
    """Time regular and sparse linear system solvers. Plot the system size
    versus the execution times. As always, use log scales where appropriate.
    """
    domain = 2**np.arange(1,13)
    one = []
    two = []
    for n in domain:
        A = tridiag(n)
        b = np.random.rand(n,1)
        """Timing #1"""
        start = time.time()
        Acsr = A.tocsr()
        scipy.sparse.linalg.spsolve(Acsr,b)
        one.append(time.time() - start)
        """Timing #2"""
        start = time.time()
        new_A = np.copy(A.toarray())
        scipy.linalg.solve(new_A,b)
        two.append(time.time() - start)

    #plot
    plt.loglog(domain, one, 'b.-', basex=2, basey=2, lw=2, ms=12,label="Sparse solver")
    plt.loglog(domain, two, 'g.-', basex=2, basey=2, lw=2, ms=12,label="Regular solver")
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.legend(loc="upper left")
    plt.title("Timed Linear System Solvers", fontsize=15)

    plt.show()

