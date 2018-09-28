# newtons_method.py
"""Newton's Method.
Bailey Smith
January 31 2017
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

def Newtons_method(f, x0, Df, iters=15, tol=1e-5, alpha=1):
    """Use Newton's method to approximate a zero of a function.

    Inputs:
        f (function): A function handle. Should represent a function
            from R to R.
        x0 (float): Initial guess.
        Df (function): A function handle. Should represent the derivative
             of f.
        iters (int): Maximum number of iterations before the function
            returns. Defaults to 15.
        tol (float): The function returns when the difference between
            successive approximations is less than tol.

    Returns:
        A tuple (x, converged, numiters) with
            x (float): the approximation for a zero of f.
            converged (bool): a Boolean telling whether Newton's method
                converged.
            numiters (int): the number of iterations computed.
    """
    error = 1
    converged = False
    for i in xrange(iters):
        if error < tol:
            converged = True
            break
        x = x0 - (alpha)*f(x0)/Df(x0)
        error = abs(x-x0)
        x0 = x
    if converged == True:
        iters = i
    return (x, converged, iters)


def plot_nonlinear():
    """Plot f(x) = sin(x)/x - x on [-4,4].
    Return the zero of this function to 7 digits of accuracy.
    """
    f = lambda x: np.sin(x)/float(x) - x if x != 0 else 1
    Df = lambda x: (x*np.cos(x) - np.sin(x))/x**2 - 1
    zero = Newtons_method(f, .8, Df,tol=1e-6)
    domain = np.linspace(-4,4,100)
    y = [f(i) for i in domain]

    plt.plot(domain, y)
    plt.show()

    return zero[0]

def non_converge():
    """Return a string as to what happens and why during Newton's Method for
    the function f(x) = x^(1/3) where x_0 = .01.
    """
    f = lambda x: np.sign(x)*np.power(np.abs(x), 1./3)
    Df = lambda x: np.sign(x)*np.power(np.abs(x), -2./3)*(1./3)
    x0 = .01
    zero = Newtons_method(f, x0, Df)
    # print zero
    print "Newton's Method does not converge for the function f(x) = x^(1/3) where x_0 = .01 because the correct zero was skipped over during the process of Newton's method"



def non_linear_solution():
    """Given P1[(1+r)**N1-1] = P2[1-(1+r)**(-N2)], if N1 = 30, N2 = 20,
    P1 = 2000, and P2 = 8000, use Newton's method to determine r.
    Return r.
    """
    f = lambda r: 2000 * ((1+r)**30-1) - 8000 * (1-(1+r)**(-20))
    Df = lambda r: (20000 * (3 * (r + 1)**50 - 8))/(r + 1)**21
    return Newtons_method(f, .06, Df)[0]

def finding_alpha():
    """Find an alpha < 1 so that running Newtons_method() on f(x) = x**(1/3)
    with x0 = .01 converges. Return the complete results of Newtons_method().
    """
    f = lambda x: np.sign(x)*np.power(np.abs(x), 1./3)
    Df = lambda x: np.sign(x)*np.power(np.abs(x), -2./3)*(1./3)
    x0 = .01
    zero = Newtons_method(f, x0, Df, alpha = .3)
    return zero


def Newtons_vector(f, x0, Df, iters = 15, tol = 1e-5, alpha = 1):
    """Use Newton's method to approximate a zero of a vector valued function.

    Inputs:
        f (function): A function handle.
        x0 (list): Initial guess.
        Df (function): A function handle. Should represent the derivative
             of f.
        iters (int): Maximum number of iterations before the function
            returns. Defaults to 15.
        tol (float): The function returns when the difference between
            successive approximations is less than tol.
        alpha (float): Defaults to 1.  Allows backstepping.

    Returns:
        A tuple (x_values, y_values) where x_values and y_values are lists that contain the x and y value from each iteration of Newton's method
    """
    xes = [x0[0]]
    ys = [x0[1]]
    error = 1
    converged = False
    for i in xrange(iters):
        # print "i", i
        if error < tol:
            converged = True
            break
        x = x0 - alpha*la.solve(Df(x0),f(x0))
        error = la.norm(x-x0)
        x0 = x
        xes.append(x0[0])
        ys.append(x0[1])
        # print x0
    if converged == True:
        iters = i
    # print converged
    return (xes, ys)


def plot_backtracking():
    """Solve the system using Newton's method and Newton's method with
    backtracking. Does not return anything, uses meshgrids
    """
    l = 5
    delta = 1

    f = lambda x: np.array([l*x[0]*x[1] - x[0]*(1+x[1]), -x[0]*x[1] + (delta-x[1])*(1+x[1])])
    Df = lambda x: np.array([[l*x[1] - (1+x[1]), l*x[0] - x[0]],[-x[1], -x[0] + delta - 1 - 2*x[1]]])
    x0 = np.array([-.1,0.23])
    a,b = Newtons_vector(f, x0, Df, iters=100, alpha=1)
    print "a\n", a,"\nb\n",b
    # x0 = np.array([.25,0.])
    c,d = Newtons_vector(f, x0, Df, iters=100, alpha=.5)
    # print 'b', b
    np.meshgrid()
    x = np.linspace(-6, 8, 700)
    y = np.linspace(-6, 6, 700)
    X,Y = np.meshgrid(x,y)
    # print X,Y
    Z = f([X,Y])


    plt.contour(X, Y, Z[1,:,:], 20, cmap="Spectral")
    plt.contour(X, Y, Z[0,:,:], 1, cmap="Spectral")


    plt.plot(a,b,"b.", lw=20)
    plt.plot(a,b,"r-")
    plt.plot(c,d,"k.", lw=20)
    plt.plot(c,d,"r-")
    plt.show()



def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=1000, iters=15, colormap='brg'):
    """Plot the basins of attraction of f.

    INPUTS:
        f (function): Should represent a function from C to C.
        Df (function): Should be the derivative of f.
        roots (array): An array of the zeros of f.
        xmin, xmax, ymin, ymax (float,float,float,float): Scalars that define the domain
            for the plot.
        numpoints (int): A scalar that determines the resolution of the plot. Defaults to 100.
        iters (int): Number of times to iterate Newton's method. Defaults to 15.
        colormap (str): A colormap to use in the plot. Defaults to 'brg'.
    """
    xreal = np.linspace(xmin, xmax, numpoints)
    ximag = np.linspace(ymin, ymax, numpoints)
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    Xold = Xreal+1j*Ximag

    Xnew = Xold - f(Xold)/Df(Xold)

    for i in xrange(numpoints):
        X = Xnew - f(Xnew)/Df(Xnew)
        Xnew = X


    for i in xrange(numpoints):
        for j in xrange(numpoints):
            index = np.argmin(abs(roots - Xnew[i,j]))
            Xnew[i,j] = index

    plt.pcolormesh(Xreal, Ximag, Xnew,cmap=colormap)
    plt.show()

def plot_basins():
    """Run plot_basins() on the function f(x) = x^3 - 1 on the domain
    [-1.5,1.5]x[-1.5,1.5].
    """
    f = lambda x : x**3-1
    Df = lambda x : 3*x**2
    roots = np.array([1.,-.5 + (3)**.5/2.*1j, -.5 - (3)**.5/2.*1j])
    plot_basins(f, Df, roots, -1.5, 1.5, -1.5, 1.5, iters=15, colormap='brg')

