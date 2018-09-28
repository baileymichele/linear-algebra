# -*- coding: utf-8 -*-
# qr_decomposition.py
"""
QR 1 (Decomposition).
"""

import numpy as np
from scipy import linalg as la


def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Inputs:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n,n))
    for i in xrange(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        for j in xrange(i+1,n):
            T = np.transpose(Q)
            R[i,j] = np.dot(T[j,:],Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q,R

def test1():
    A = np.random.random((6,4))
    Q,R = qr_gram_schmidt(A)
    Q2,R2 = la.qr(A, mode="economic")
    print A.shape, Q.shape, R.shape

    print np.allclose(np.triu(R), R)
    print np.allclose(np.dot(Q.T, Q), np.identity(4))
    print np.allclose(np.dot(Q, R), A)

    print "Q, R", Q, R
    print "correct", Q, R


def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Inputs:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the detetminant of A.
    """
    determinant = 1
    Q,R = la.qr(A, mode="economic")
    for i in xrange(np.shape(A)[1]):
        determinant = determinant*R[i,i]
    return abs(determinant)


def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Inputs:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """

    n = np.shape(A)[0]
    x = np.zeros_like(b).astype(np.float64)

    Q,R = la.qr(A, mode="economic")
    y = np.dot(np.transpose(Q),b).astype(np.float64)
    #Use back substitution to solve Rx = y
    #Shortcut
    for k in reversed(xrange(n)):
            x[k] = (y[k] - np.dot(R[k,k:], x[k:])) / (R[k,k])
    return x

def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Inputs:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m,n = np.shape(A)
    R = np.copy(A)
    Q = np.identity(m)
    for k in xrange(n):
        u = np.copy(R[k:,k])
        if u[0] >= 0:
            sign = 1
        else:
            sign = -1
        u[0] += sign*la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u,np.dot(np.transpose(u),R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u,np.dot(np.transpose(u),Q[k:,:]))

    return np.transpose(Q),R

def test4():
    A = np.random.random((5, 3))
    Q2,R2 = la.qr(A)
    Q,R = qr_householder(A)
    print A.shape, Q.shape, R.shape
    print np.allclose(Q.dot(R), A)
    print "Q, R", Q, R
    print "correct", Q, R


def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Inputs:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m,n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)
    for k in xrange(n-2):
        u = np.copy(H[k+1:,k])
        if u[0] >= 0:
            sign = 1
        else:
            sign = -1
        u[0] += sign*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,np.dot(np.transpose(u),H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2*np.outer(np.dot(H[:,k+1:],u),np.transpose(u))
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,np.dot(np.transpose(u),Q[k+1:,:]))
    return H, np.transpose(Q)

def test5():
    A = np.random.random((8,8))
    H2, Q2 = la.hessenberg(A, calc_q=True)
    H, Q = hessenberg(A)
    print H, Q
    # print np.allclose(np.triu(H, -1), H)
    # print np.allclose(np.dot(np.dot(Q, H), Q.T), A)

if __name__ == '__main__':
    print test1()
    A = np.random.random((4,4)).astype(np.float64)
    b = np.random.random((4,1)).astype(np.float64)
    print solve(A,b)

    Q,R = la.qr(A, mode="economic")
    print abs_det(A)
    print np.linalg.det(R)
    print test4()
    print test5()
