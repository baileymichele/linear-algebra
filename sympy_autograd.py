# sympy_autograd.py
"""Differentiation 2 (SymPy and Autograd).
Bailey Smith
January 24 2017
"""

import time
import numpy as np
import sympy as sy
from autograd import grad
import autograd.numpy as anp
from autograd import jacobian
import numerical_differentiation as nd

def myexp(n):
    """Compute e to the nth digit.

    Inputs:
        n (integer): n decimal places to calculate e.

    Returns:
        approximation (float): approximation of e.
    """
    #calculates pi to n decimal points.
    tot = sy.Rational(0, 1)
    term = 1
    bound = sy.Rational(1, 10)**(n+1)
    i=0
    while bound <= term:
        term = sy.Rational(1, sy.factorial(i))
        tot += term
        i += 1
    return sy.Float(tot, n)


def symbolic_solver():
    """Solves y = e^x + x for x.

    Returns:
        the solution (list).
    """
    x, y = sy.symbols('x, y')
    expr = sy.exp(x) + x - y
    return sy.solve(expr, x)


def symbolic_integral():
    """Computes the integral of sin(x^2) from 0 to infinity.

    Returns:
        the integral value (float).
    """
    x = sy.symbols('x')
    expr = sy.sin(x**2)
    return sy.integrate(expr, (x, 0, sy.oo))


def symbolic_derivative():
    """Calculates the derivative of e^sin(cos(x)) at x = 1.
    Times how long it takes to compute the derivative using SymPy as well as
    centered difference quotients.
    Calculates the error for each approximation.

    Prints the time it takes to compute and the error for both SymPy and
    centered difference quotients.

    Returns:
        SymPy approximation (float)
    """
    start = time.time()
    x = sy.symbols('x')
    expr = sy.exp(sy.sin(sy.cos(x)))
    diff1 = expr.diff(x,1).subs({x:1.})
    timesym = time.time() - start

    start = time.time()
    diff2 = nd.centered_difference_quotient(lambda x: sy.exp(sy.sin(sy.cos(x))) , np.array([1]))
    timecdq = time.time() - start

    true = np.sin(1) * np.cos(np.cos(1)) * (-np.exp(np.sin(np.cos(1))))
    error1 = abs(diff1 - true)
    error2 = abs(diff2 - true)

    print "Computation time SymPy:", timesym, "\nComputation time CDQ:", timecdq, "\nSymPy Error:", error1, "\nCDQ error:", error2, "\n"
    return diff1


def symbolic_diffEQ():
    """Solves the differential equation when x = 1.

    Returns:
        Solution when x = 1.
    """
    x = sy.symbols('x')
    f = sy.Function('f')
    expr = f(x).diff(x, 6) - 3*f(x).diff(x,4) + 3*f(x).diff(x,2) + f(x) - x**10*sy.exp(x) - x**11*sy.sin(x) - x**12*sy.exp(x)*sy.sin(x) + x**13*sy.cos(2*x) - x**14*sy.exp(x)*sy.cos(3*x)
    return sy.dsolve(expr).subs({x:1.})

def symbolic_derivative_2():
    """Computes the derivative of ln(sqrt(sin(sqrt(x)))) at x = pi/4.
    Times how long it take to compute using SymPy, autograd, and centered
    difference quotients. Computes the error of each approximation.

    Print the time
    Print the error

    Returns:
        derviative (float): the derivative computed using autograd.
    """

    x = sy.symbols('x')
    start = time.time()
    g = lambda x: anp.log(anp.sqrt(anp.sin(anp.sqrt(x))))
    grad_g = grad(g)
    diff1 = grad_g(np.pi/4.)
    timeauto = time.time() - start

    start = time.time()
    expr = sy.log(sy.sqrt(sy.sin(sy.sqrt(x))))
    diff2 = expr.diff(x,1).subs({x:np.pi/4.})
    timesym = time.time() - start

    start = time.time()
    diff3 = nd.centered_difference_quotient(lambda x: np.log(np.sqrt(np.sin(np.sqrt(x)))) , np.array([np.pi/4.]))
    timecdq = time.time() - start

    true = 0.2302590111469608485902344398735129932219551807224673384852491736308414890748909517397968648696359801
    error1 = abs(diff1 - true)
    error2 = abs(diff2 - true)
    error3 = abs(diff3 - true)

    print "Computation time Autograd:", timeauto, "\nComputation time SymPy:", timesym, "\nComputation time CDQ:", timecdq, "\nAutograd Error:", error1, "\nSymPy Error:", error2, "\nCDQ error:", error3, "\n"
    return float(diff1)


def symbolic_jacobian():
    """Computes Jacobian for the function
        f(x,y)=[(e^x)sin(y) + y^3, 3y - cos(x)]
    Times how long it takes to compute the Jacobian using SymPy and autograd.

    Prints the times.

    Returns:
        Jacobian (array): jacobian found using autograd at (x,y) = (1,1)
    """
    f = lambda x: anp.array([anp.exp(x[0])*anp.sin(x[1]) + x[1]**3, 3*x[1] - anp.cos(x[0])])
    start = time.time()
    jacobian_f = jacobian(f)
    timeauto = time.time() - start

    x,y = sy.symbols('x,y')


    start = time.time()
    F = sy.Matrix([sy.exp(x)*sy.sin(y) + y**3, 3*y - sy.cos(x)])
    F.jacobian([x,y])
    timesym = time.time() - start

    print "Computation time Autograd:", timeauto, "\nComputation time SymPy:", timesym, "\n"

    return jacobian_f(np.array([1.,1.]))

if __name__ == '__main__':
    print myexp(10)
    print "symbolic_solver\n",symbolic_solver(),"\n"
    print symbolic_integral()
    print "symbolic_derivative\n",symbolic_derivative(), "\n"
    print symbolic_diffEQ()#NotWorking?
    print "symbolic_derivative_2\n", symbolic_derivative_2(), "\n"
    print "symbolic_jacobian\n",symbolic_jacobian()
