# -*- coding: utf-8 -*-
# iterative_solvers.py
"""Iterative Solvers.
Bailey Smith
18 October 2016
"""
import time
import numpy as np
from scipy import sparse
from scipy import linalg as la
import scipy.sparse.linalg as spla
from matplotlib import pyplot as plt

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant nxn matrix.

    Inputs:
        n (int): the dimension of the system.
        num_entries (int): the number of nonzero values. Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): An nxn strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in xrange(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in xrange(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A


def jacobi_method(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.
            (this is for Problem 2).

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    N = 0
    e = 1
    D = np.reshape(np.diag(A),(len(np.diag(A)),1))
    inverse = 1.0/D
    n = A.shape[0]
    x = np.zeros_like(b)#array of zeros with n dim
    error = []
    while N < maxiters and e >= tol:
        xprev = x
        x = xprev + (inverse*(b-np.dot(A,xprev)))
        e = la.norm(xprev - x, ord=np.inf)
        error.append(e)
        N += 1
    if plot is True:
        plt.semilogy(xrange(N), error, 'b-', lw=2, ms=12)
        plt.title("Convergence of Jacobi Method", fontsize=18)
        plt.xlabel("Iteration #")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x


def gauss_seidel(A, b, tol=1e-8, maxiters=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.
        plot (bool, opt): if True, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    N = 0
    e = 1
    n = A.shape[0]
    x = np.zeros_like(b)#array of zeros with n dim
    error = []
    while N < maxiters and e >= tol:
        xprev = np.copy(x)#If you don't copy it, changing x will also change xprev!
        for i in xrange(n):
            inverse = 1.0/A[i,i]
            x[i] = xprev[i] + (inverse*(b[i]-np.dot(A[i,:].T,xprev)))
        e = la.norm(xprev - x, ord=np.inf)
        error.append(e)
        N += 1

    if plot is True:
        plt.plot(xrange(N), error, 'b-', lw=2, ms=12)
        plt.title("Convergence of Gauss-Seidel Method", fontsize=18)
        plt.xlabel("Iteration #")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x.reshape((len(x),1))


def comparison():
    """For a 5000 parameter system, compare the runtimes of the Gauss-Seidel
    method and la.solve(). Print an explanation of why Gauss-Seidel is so much
    faster.
    """

    gauss = []
    la_solve = []
    for i in xrange(5,12):

        n = 2**i
        A = diag_dom(n)
        b = np.random.rand(n,1)

        """Timing Gauss-Seidel Method"""
        start = time.time()
        gauss_seidel(A,b)
        gauss.append(time.time() - start)#Putting times into a list so we can plot
        """Timing la.solve"""
        start = time.time()
        la.solve(A,b)
        la_solve.append(time.time() - start)

    plt.plot([5,6,7,8,9,10,11], gauss, 'b.-',lw=2, ms=10,label="Gauss")
    plt.plot([5,6,7,8,9,10,11], la_solve, 'g.-',lw=2, ms=10, label="la.solve")
    plt.xlabel("System Size", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("Timing Methods", fontsize=15)
    plt.legend(loc="upper left")


    plt.show()

def sparse_gauss_seidel(A, b, tol=1e-8, maxiters=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse CSR matrix.
        b ((n,) ndarray): A vector of length n.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """

    N = 0
    e = 1
    n = A.shape[0]
    x = np.zeros_like(b)#array of zeros with n dim
    error = []
    while N < maxiters and e >= tol:
        xprev = np.copy(x)#If you don't copy it, changing x will also change xprev!
        for i in xrange(n):
            inverse = 1.0/A[i,i]
            # Slice the i-th row of A and dot product the vector x.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = np.dot(A.data[rowstart:rowend], x[A.indices[rowstart:rowend]])
            x[i] = xprev[i] + (inverse*(b[i]-Aix))
        e = la.norm(xprev - x, ord=np.inf)
        N += 1
    return x.reshape((len(x),1))

def sparse_sor(A, b, omega, tol=1e-8, maxiters=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Inputs:
        A ((n,n) csr_matrix): An nxn sparse matrix.
        b ((n,) ndarray): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float, opt): the convergence tolerance.
        maxiters (int, opt): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): the solution to system Ax = b.
    """
    N = 0
    e = 1
    n = A.shape[0]
    x = np.zeros_like(b)#array of zeros with n dim
    while N < maxiters and e >= tol:
        xprev = np.copy(x)#If you don't copy it, changing x will also change xprev!
        for i in xrange(n):
            inverse = float(omega)/A[i,i]
            # Slice the i-th row of A and dot product the vector x.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            x[i] = xprev[i] + (inverse*(b[i]-(np.dot(A.data[rowstart:rowend], x[A.indices[rowstart:rowend]]))))
        e = la.norm(xprev - x, ord=np.inf)
        N += 1
    return x.reshape((len(x),1))


def finite_difference(n):
    """Return the A and b described in the finite difference problem that
    solves Laplace's equation.
    """
    offsets = [-1,0,1]
    B = sparse.diags([1,-4,1], offsets, shape=(n,n))
    I = sparse.diags([1,1], [-n,n], shape=(n**2,n**2))
    A = []
    for i in xrange(n):
        A.append(B)#Have n B matrices along the diagonal
    One = sparse.block_diag(A)#accounts for the identity matrix
    b = np.zeros(n**2)
    b[::n] = -100#-100 for the boarders of the interior
    b[n-1::n] = -100#
    return One+I, b

def compare_omega():
    """Time sparse_sor() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiters = 1000 using the A and b generated by finite_difference()
    with n = 20. Plot the times as a function of omega.
    """
    A,b = finite_difference(20)
    x = [1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95]
    #x = np.linspace(1,1.95,20)

    SOR = []
    for i in xrange(len(x)):
        # print i
        """Timing SOR Method"""
        start = time.time()
        sparse_sor(A,b,x[i],tol=1e-2,maxiters = 900)
        SOR.append(time.time() - start)#Putting times into a list so we can plot\

    plt.plot(x, SOR, 'b-',lw=2, ms=10)
    plt.xlabel("Omega", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    plt.title("SOR Method", fontsize=15)
    plt.legend(loc="upper right")

    plt.show()


def hot_plate(n):
    """Use finite_difference() to generate the system Au = b, then solve the
    system using SciPy's sparse system solver, scipy.sparse.linalg.spsolve().
    Visualize the solution using a heatmap using np.meshgrid() and
    plt.pcolormesh() ("seismic" is a good color map in this case).
    """
    A,b = finite_difference(n)
    x = spla.spsolve(A,b)
    plt.pcolormesh(np.reshape(x,(n,n)),cmap="seismic")
    plt.show()

