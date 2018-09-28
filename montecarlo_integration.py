# montecarlo_integration.py
"""Monte Carlo Integration.
Bailey Smith
15 February 2017
"""
import time
import numpy as np
from scipy import stats
from scipy import linalg as la
from matplotlib import pyplot as plt

def est_volume(N=10000):
    """Return an estimate of the volume of the unit sphere using Monte
    Carlo Integration.
    take 2d change to 3d

    Input:
        N (int, optional) - The number of points to sample. Defaults
            to 10000.
    """
    points = np.random.rand(3, N)
    points = points*2-1
    pointsDistances = np.linalg.norm(points,axis=0)
    numInCircle = np.count_nonzero(pointsDistances < 1)
    return 8.*(numInCircle/float(N))


def approx_integral_1D(f, a, b, N=10000):
    """Use Monte-Carlo integration to approximate the integral of
    1-D function f on the interval [a,b].

    Inputs:
        f (function) - Function to integrate. Should take scalar input.
        a (float) - Left-hand side of interval.
        b (float) - Right-hand side of interval.
        N (int, optional) - The number of points to sample in
            the Monte-Carlo method. Defaults to 10000.

    Returns:
        estimate (float) - The result of the Monte-Carlo algorithm.

    Example:
        >>> f = lambda x: x**2
        >>> # Integral from 0 to 1. True value is 1/3.
        >>> approx_integral_1D(f, 0, 1)
        0.3333057231764805
    """
    total = 0
    points = np.random.rand(N)
    points = (points)*(b-a)+a
    for i in points:
        total += f(i)
    total /= float(N)
    return total*(b-a)**2

def est_integral_2D(f, mins, maxs, N=10000):
    """Use Monte-Carlo integration to approximate the integral of f
    on the box defined by mins and maxs.

    Inputs:
        f (function) - The function to integrate. This function should
            accept a 1-D NumPy array as input.
        mins (1-D np.ndarray) - Minimum bounds on integration.
        maxs (1-D np.ndarray) - Maximum bounds on integration.
        N (int, optional) - The number of points to sample in
            the Monte-Carlo method. Defaults to 10000.

    Returns:
        estimate (float) - The result of the Monte-Carlo algorithm.

    Example:
        >>> f = lambda x: np.hypot(x[0], x[1]) <= 1
        >>> # Integral over the square [-1,1] x [-1,1]. True value is pi.
        >>> mc_int(f, np.array([-1,-1]), np.array([1,1]))
        3.1290400000000007
    """

    # points = np.random.rand(n, N)
    # for i in xrange(N):
    #     points[:,i] = ((points)*(n))[:,i]+mins

    n = len(mins)
    my_list = []
    for i in xrange(np.shape(mins)[0]):
        my_list.append(np.random.uniform(mins[i],maxs[i],N))
    points = np.vstack(my_list)

    points = np.apply_along_axis(f,0,points)
    return abs(np.prod(maxs-mins))*np.sum(points)/float(N)


def integral_normal_dist():
    """Integrate the joint normal distribution.

    Return your Monte Carlo estimate, SciPy's answer, and (assuming SciPy is
    correct) the relative error of your Monte Carlo estimate.
    """
    mins = np.array([-1.5,0.,0.,0.])
    maxs = np.array([.75,1.,.5,1.])
    N = 4.
    f = lambda x: (1./(np.sqrt(2*np.pi)**N))*np.exp(-x.T.dot(x)*.5)
    x = est_integral_2D(f,mins,maxs,50000)

    means = np.zeros(4)
    covs = np.eye(4)
    value, inform = stats.mvn.mvnun(mins, maxs, means, covs)

    return (x, value, abs(x-value)/value)

def integration_error(numEstimates=50):
    """Plot the error of Monte Carlo Integration."""
    N = [50.,100.,500.] + list(range(1000,51000,1000))
    M = np.array(N)
    for i in xrange(numEstimates-1):
        M = np.vstack((M,N))
    true = 4/3.*np.pi
    for row in xrange(numEstimates):
        for col in xrange(53):
            M[row,col] = est_volume(M[row,col])
            M[row,col] = abs(true-M[row,col])
    avg = np.mean(M,0)/4

    plt.plot(N, 1./np.sqrt(N), "r--", label="1/sqrt(N)")
    plt.plot(N, avg, label="Error")
    plt.xlabel("N")
    plt.ylabel("Relative Error")
    plt.legend(loc="upper right")
    plt.ylim(0,.10)
    plt.show()


