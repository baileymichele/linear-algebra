#~*~ coding: UTF-8 ~*~
# matplotlib_intro.py
"""Intro to Matplotlib.
Bailey Smith
20 September 2016
"""
import numpy as np
from matplotlib import pyplot as plt

def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Inputs:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    my_array = np.random.randn(n,n)
    mean = np.mean(my_array,axis=1)
    return np.var(mean)


def base_array():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    results = []
    for n in xrange(100,1100,100):
        results.append(var_of_means(n))
    results = np.array(results)
    plt.plot(results)
    plt.show()


def plot_trig():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-np.pi*2, np.pi*2, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.arctan(x)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.show()


def non_linear_plot1():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Change the range of the y-axis to [-6,6].

        Go from -2 to 1 then 1 to 6
    """
    x1 = np.linspace(-2,1,100)
    x2 = np.linspace(1,6,100)
    plt.plot(x1,1/(x1-1),'m--', linewidth=6)
    plt.plot(x2,1/(x2-1),'m--', linewidth=6)
    plt.ylim(-6, 6)
    plt.show()


def subplot_trig():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, np.pi*2, 100)
    plt.subplot(221)
    plt.plot(x,np.sin(x), "g-")
    plt.xlim(0, np.pi*2)
    plt.ylim(-2,2)
    plt.title("sin(x)", fontsize=18)

    plt.subplot(222)
    plt.plot(x,np.sin(2*x), "r--")
    plt.xlim(0, np.pi*2)
    plt.ylim(-2,2)
    plt.title("sin(2x)", fontsize=18)

    plt.subplot(223)
    plt.plot(x,np.sin(x)*2, "b--")
    plt.xlim(0, np.pi*2)
    plt.ylim(-2,2)
    plt.title("2sin(x)", fontsize=18)

    plt.subplot(224)
    plt.plot(x,np.sin(x*2)*2, "m:")
    plt.xlim(0, np.pi*2)
    plt.ylim(-2,2)
    plt.title("2sin(2x)", fontsize=18)

    plt.suptitle("Sine Functions", fontsize=18)
    plt.show()


def plot_map():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes(x: want every row 2nd column) against latitudes(y: every row 3rd column). Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    data = np.load("FARS.npy")

    plt.subplot(121)
    x = data[:,1]
    y = data[:,2]
    plt.plot(x,y, "k,")
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.subplot(122)
    x2 = data[:,0]
    plt.hist(x2,bins=24, range=[-.5, 23.5])
    plt.xlabel("Hour of the Day")
    plt.xlim(-.5, 23.5)
    plt.show()


def subplot_trig_color():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-np.pi*2, np.pi*2, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.sin(Y))/(X*Y)
    print 'X\n',X,'\nY\n',Y,'\nZ\n',Z

    plt.subplot(121)
    plt.pcolormesh(X,Y,Z, cmap="cool")
    plt.colorbar()
    plt.xlim(-np.pi*2, 2*np.pi)
    plt.ylim(-np.pi*2, 2*np.pi)

    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap="gist_rainbow")
    plt.xlim(-np.pi*2, 2*np.pi)
    plt.ylim(-np.pi*2, 2*np.pi)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    print var_of_means(3)
    print base_array()
    print plot_trig()
    print non_linear_plot1()
    print subplot_trig()
    print plot_map()
    print subplot_trig_color()
