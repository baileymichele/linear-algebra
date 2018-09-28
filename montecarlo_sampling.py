# montecarlo_sampling.py
"""Monte Carlo 2 (Importance Sampling).
Bailey Smith
February 28 2017
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def greaterThanThree(n):
    """Approximate the probability that a random draw from the standard
    normal distribution will be greater than 3."""
    points = np.random.normal(size=n)
    return len(points[points>3])* 1./n


def probability():
    """Answer the following question using importance sampling:
            A tech support hotline receives an average of 2 calls per
            minute. What is the probability that they will have to wait
            at least 10 minutes to receive 9 calls?
    Returns:
        IS (array) - an array of estimates using
            [5000, 10000, 15000, ..., 500000] as number of
            sample points."""
    a = 9
    theta = .5
    my_list = []
    h = lambda x : x >= 10
    f = lambda x : stats.gamma(a,scale=theta).pdf(x)
    g = lambda x : stats.gamma(a+6,scale=theta).pdf(x)

    for i in xrange(5000,505000,5000):
        X = np.random.gamma(a+6,scale=theta,size=i)
        my_list.append(1./i * np.sum(h(X)*f(X)/g(X)))

    return np.array(my_list)

def errors():
    """Plot the errors of Monte Carlo Simulation vs Importance Sampling
    for the probability()."""
    true = 1 - stats.gamma(a=9,scale=0.5).cdf(10)
    domain = xrange(5000,505000,5000)

    h = lambda x : x > 10
    MC_estimates = []
    for N in domain:
        X = np.random.gamma(9,scale=0.5,size=N)
        MC = 1./N*np.sum(h(X))
        MC_estimates.append(MC)
    MC_estimates = np.array(MC_estimates)

    plt.plot(domain, abs(MC_estimates-true), "r", label="Montecarlo")
    plt.plot(domain, abs(prob2()-true), label="Importance Sampling")
    plt.xlabel("Number of Sample Points")
    plt.legend(loc="upper right")
    plt.ylim(0,.0005)
    plt.title("Error of Estimations")
    plt.show()


def randomDraw():
    """Approximate the probability that a random draw from the
    multivariate standard normal distribution will be less than -1 in
    the x-direction and greater than 1 in the y-direction."""
    N = 10000
    h = lambda x : x[0]<-1 and x[1]>1
    f = lambda x: stats.multivariate_normal.pdf(x,mean=np.array([0,0]),cov=np.array([[1,0],[0,1]]))
    g = lambda x: stats.multivariate_normal.pdf(x,mean=np.array([-1,1]),cov=np.array([[1,0],[0,1]]))
    X = np.random.multivariate_normal(np.array([-1, 1]),np.array([[1,0],[0,1]]),N)
    return 1./N * np.sum(np.apply_along_axis(h,1,X)*np.apply_along_axis(f,1,X)/np.apply_along_axis(g,1,X))

if __name__ == '__main__':
    # print greaterThanThree(1000000)
    # print probability()
    # print errors()
    print randomDraw()
