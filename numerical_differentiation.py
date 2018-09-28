# -*- coding: utf-8 -*-
# numerical_differentiation.py
"""Numerical Differentiation.
Bailey Smith
January 16 2017
"""

import numpy as np
from scipy import linalg as la

def centered_difference_quotient(f, pts, h=1e-5):
    """Compute the centered difference quotient for function (f)
    given points (pts).

    Inputs:
        f (function): the function for which the derivative will be
            approximated.
        pts (array): array of values to calculate the derivative.

    Returns:
        An array of the centered difference quotient.
    """
    Df_app = lambda x: .5*(f(x+h)-f(x-h))/h
    return Df_app(pts)

def calculate_errors(f,df,pts,h = 1e-5):
    """Compute the errors using the centered difference quotient approximation.

    Inputs:
        f (function): the function for which the derivative will be
            approximated.
        df (function): the function of the derivative
        pts (array): array of values to calculate the derivative

    Returns:
        an array of the errors for the centered difference quotient
            approximation.
    """
    return np.abs(centered_difference_quotient(f, pts) - df(pts))

def example_calc_derivative():
    """Use the centered difference quotient to approximate the derivative of
    f(x)=(sin(x)+1)^x at x= Ï€/3, Ï€/4, and Ï€/6.
    Then compute the error of each approximation

    Returns:
        an array of the derivative approximations
        an array of the errors of the approximations
    """
    h = 1e-5
    f = lambda x: (np.sin(x) + 1)**x
    pts = np.array([np.pi/3, np.pi/4,np.pi/6])
    df = lambda x: (np.sin(x) + 1)**x * (np.log(np.sin(x) + 1) + x*np.cos(x)/(np.sin(x) + 1))
    return centered_difference_quotient(f,pts), calculate_errors(f,df,pts)

def speed_of_plane():
    """Use centered difference quotients to calculate the speed v of the plane
    at t = 10 s
        assume distance is in km
        standard difference quotients
        calc da/dt, db/dt divide dy/dt by dx/dt..wolfram
        plug in beta/alpha values at 10 change to radians
        just use 9 and 11
    Returns:
        derivative at t = 10
        (float) speed v of plane, .27
        speed = sqrt ((dx/dt)^2 + (dy/dt)^2)
        50.11...
    """
    a = 500
    sec = lambda x: 1/np.cos(x)**2

    alpha = (np.pi/180.)*np.array([54.8,54.06,53.34])
    beta = (np.pi/180.)*np.array([65.59,64.59,63.62])

    da_dt = (alpha[2] - alpha[0])/2.
    db_dt = (beta[2] - beta[0])/2.

    A = alpha[1]
    B = beta[1]


    dx_dt =  a*((db_dt * sec(B))/(np.tan(B) - np.tan(A)) - (np.tan(B) * (db_dt * sec(B) - da_dt * sec(A)))/(np.tan(B) - np.tan(A))**2)

    dy_dt = a*((np.tan(B) - np.tan(A))*(np.tan(B)*(sec(A))*da_dt + np.tan(A)*(sec(B))*db_dt) - (np.tan(B)*np.tan(A))*((sec(B))*db_dt - (sec(A))*da_dt))/(np.tan(B)-np.tan(A))**2

    print "dx_dt: ", dx_dt, "\ndy_dt:", dy_dt
    return np.sqrt(dy_dt**2 + dx_dt**2)


def jacobian(f, n, m, pt, h=1e-5):
    """Compute the approximate Jacobian matrix of f at pt using the centered
    difference quotient.

    Inputs:
        f (function): the multidimensional function for which the derivative
            will be approximated.
        n (int): dimension of the domain of f.
        m (int): dimension of the range of f.
        pt (array): an n-dimensional array representing a point in R^n.
        h (float): a float to use in the centered difference approximation.

    Returns:
        (ndarray) Jacobian matrix of f at pt using the centered difference
            quotient.
    """
    Jac = np.zeros((m,n))
    for j in xrange(n):
        e = np.eye(n)
        Df_app = lambda x: .5*(f(x+h*e[:,j])-f(x-h*e[:,j]))/h
        # print Jac, Df_app(pt)
        Jac[:,j] = Df_app(pt)
    return Jac


def findError():
    """Compute the maximum error of jacobian() for the function
    f(x,y)=[(e^x)sin(y) + y^3, 3y - cos(x)] on the square [-1,1]x[-1,1].

    Returns:
        Maximum error of your jacobian function.
    """
    f = lambda x: np.array([np.exp(x[0]) * np.sin(x[1]) + x[1]**3, 3*x[1] - np.cos(x[0])])
    df = lambda x: np.array([[np.exp(x[0]) * np.sin(x[1]), np.exp(x[0]) * np.cos(x[1]) + 3*x[1]**2], [np.sin(x[0]), 3]])
    error = 0

    for x in np.linspace(-1,1,100):
        for y in np.linspace(-1,1,100):
            pt = np.array([x,y])
            Jac = jacobian(f, 2, 2, pt, h=1e-5)
            e = la.norm(Jac-df(pt))
            if e > error:
                error = e

    return error

