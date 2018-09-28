# -*- coding: utf-8 -*-
# qr_lstsq_eigs.py
"""QR 2 (Least Squares and Computing Eigenvalues).
Bailey Smith
October 25 2016"""

import numpy as np
from cmath import sqrt
from scipy import linalg as la
from matplotlib import pyplot as plt
from qr_decomposition import hessenberg


def least_squares(A, b,):
    """Calculate the least squares solutions to Ax = b using QR decomposition.

    Inputs:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equation.
    """
    Q,R = la.qr(A, mode='economic')
    y = np.dot(Q.T,b)
    return la.solve_triangular(R,y)


def line_fit():
    """Load the data from housing.npy. Use least squares to calculate the line
    that best relates height to weight.

    Plot the original data points and the least squares line together.
    """
    """year, index = np.load("housing.npy").T"""
    year, index = np.load("housing.npy").T
    n = len(year)
    ones = np.ones((n,1),np.float64)
    A = np.column_stack((year,ones))
    ls_solution = least_squares(A, index)

    plt.plot(year, index, 'k*', lw=2, ms=12, label = "Data points")#MAKE it a SCATTER PLOT
    plt.plot(year, year*ls_solution[0]+ls_solution[1], 'b-', lw=2, ms=12, label = "Least Squares Fit")
    plt.legend(loc="upper left")
    plt.show()

def polynomial_fit():
    """Load the data from housing.npy. Use least squares to calculate
    the polynomials of degree 3, 6, 9, and 12 that best fit the data.

    Plot the original data points and each least squares polynomial together
    in individual subplots.
    """
    year, index = np.load("housing.npy").T


    '''Plot'''
    plt.subplot(221)
    A = np.vander(year, 4)
    x = la.lstsq(A, index)[0]
    plt.plot(year, index, 'k*', lw=2, ms=8, label = "Data points")#MAKE it a SCATTER PLOT
    plt.plot(year, np.dot(A,x), 'b-', lw=2, ms=12, label = "Least Squares Fit")
    plt.legend(loc="lower right")

    plt.subplot(222)
    A = np.vander(year, 7)
    x = la.lstsq(A, index)[0]
    plt.plot(year, index, 'k*', lw=2, ms=8, label = "Data points")#MAKE it a SCATTER PLOT
    plt.plot(year, np.dot(A,x), 'b-', lw=2, ms=12, label = "Least Squares Fit")
    plt.legend(loc="lower right")

    plt.subplot(223)
    A = np.vander(year, 10)
    x = la.lstsq(A, index)[0]
    plt.plot(year, index, 'k*', lw=2, ms=8, label = "Data points")#MAKE it a SCATTER PLOT
    plt.plot(year, np.dot(A,x), 'b-', lw=2, ms=12, label = "Least Squares Fit")
    plt.legend(loc="lower right")

    plt.subplot(224)
    A = np.vander(year, 13)
    x = la.lstsq(A, index)[0]
    plt.plot(year, index, 'k*', lw=2, ms=8, label = "Data points")#MAKE it a SCATTER PLOT
    plt.plot(year, np.dot(A,x), 'b-', lw=2, ms=12, label = "Least Squares Fit")
    plt.legend(loc="lower right")

    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A))/(2*A)

    plt.plot(r*cos_t, r*sin_t, lw=2)
    plt.gca().set_aspect("equal", "datalim")

def ellipse_fit():
    """Load the data from ellipse.npy. Use least squares to calculate the
    ellipse that best fits the data.

    Plot the original data points and the least squares ellipse together.
    """
    xk,yk = np.load("ellipse.npy").T
    A = np.column_stack((2*xk, 2*yk, np.ones_like(xk)))
    b1 = xk**2 + yk**2
    c1, c2, c3 = la.lstsq(A, b1)[0]
    r = np.sqrt(c1**2 + c2**2 + c3)

    a = 1./(r**2-c1**2-c2**2)
    b = -2*c1/(r**2-c1**2-c2**2)
    c = 0
    d = -2*c2/(r**2-c1**2-c2**2)
    e = 1./(r**2-c1**2-c2**2)

    plot_ellipse(a,b,c,d,e)
    plt.plot(xk,yk, "k*")
    plt.show()


def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (foat): The dominant eigenvalue of A.
        ((n, ) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    x0 = np.random.rand(n)
    x0 = x0/la.norm(x0)
    for k in xrange(1,N):
        xk = A.dot(x0)
        xk = xk/la.norm(xk)
        if la.norm(x0 - xk, ord=np.inf) < tol:
            break
        x0 = xk
    return np.dot(xk.T,A.dot(xk)), xk.reshape((len(xk),1))


def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Inputs:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal block
            is 1x1 or 2x2.

    Returns:
        ((n, ) ndarray): The eigenvalues of A.
    """
    """EQUATION 8.7 should have (a+d) instead of ad"""
    m,n = A.shape
    S = la.hessenberg(A)
    for k in xrange(N):
        Q,R = la.qr(S, mode="economic")
        S = R.dot(Q)
    eigs = []
    i = 0
    while i < n:
        if i == n-1 or np.abs(S[i+1,i]) < tol:
            eigs.append(S[i,i])
        else:
            a = S[i,i]
            b = S[i,i+1]
            c = S[i+1,i]
            d = S[i+1,i+1]
            evalue1 = (a+d + sqrt((a+d)**2 - 4*(a*d - b*c)))/2
            evalue2 =(a+d - sqrt((a+d)**2 - 4*(a*d - b*c)))/2
            eigs.append(evalue1)
            eigs.append(evalue2)
            i += 1#If s is 2x2 want to skip entire thing
        i += 1
    return eigs

if __name__ == '__main__':
    A = np.random.random((10,10))
    A = A.T + A
    eigs, vecs = la.eig(A)
    loc = np.argmax(eigs)
    lamb, x = eigs[loc], vecs[:,loc]
    b = np.random.random((6,1))
    print least_squares(A,b)
    print line_fit()
    print polynomial_fit()
    print ellipse_fit()
    print power_method(A)
    print qr_algorithm(A), "\nCorrect", eigs
