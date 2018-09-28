# profiling.py
"""Python Essentials: Profiling.
Bailey Smith
March 21 2017
"""

import time
import numpy as np
from numba import jit
from numba import double
from scipy import linalg as la


def compare_timings(f, g, *args):
    """Compare the timings of 'f' and 'g' with arguments '*args'.

    Inputs:
        f (func): first function to compare.
        g (func): second function to compare.
        *args: arguments to use when callings functions 'f' and 'g',
            i.e., call f with f(*args).
    Returns:
        comparison (str): The comparison of the runtimes of functions
            'f' and 'g' in the following format:
                Timing for <f>: <time>
                Timing for <g>: <time>
    """
    start = time.time()
    function1 = f(*args)
    time1 = time.time() - start

    start = time.time()
    function2 = g(*args)
    time2 = time.time() - start
    return "Timing for "+ str(f)+ ": "+ str(time1)+ "\n"+ "Timing for "+ str(g)+ ": "+ str(time2)


def LU(A):
    """Return the LU decomposition of a square matrix."""
    n = A.shape[0]
    U = np.array(np.copy(A), dtype=float)
    L = np.eye(n)
    for i in range(1, n):
        for j in range(i):
            L[i,j] = U[i,j] / U[j,j]
            for k in range(j, n):
                U[i,k] -= L[i,j] * U[j,k]
    return L, U

def LU_opt(A):
    """Return the LU decomposition of a square matrix."""
    n = A.shape[0]
    U = np.array(np.copy(A), dtype=float)
    L = np.eye(n)
    for j in xrange(0,n-1):
        for i in xrange(j+1,n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -= L[i,j]*U[j,j:]
    return L, U

def compare_LU(A):
    """Prints a comparison of LU and LU_opt with input of a square matrix A."""
    print compare_timings(LU, LU_opt, A)

def mysum(x):
    """Return the sum of the elements of X without using a built-in function.

    Inputs:
        x (iterable): a list, set, 1-d NumPy array, or another iterable.
    """
    my_sum = 0
    for i in xrange(np.shape(x)[0]):
        my_sum += x[i]
    return my_sum

def compare_sum(X):
    """
    Inputs:
        x (iterable): a list, set, 1-d NumPy array, or another iterable.

    Prints a comparison of mysum and sum
    Prints a comparison of mysum and np.sum
    """
    print compare_timings(mysum, sum, X)
    print compare_timings(mysum, np.sum, X)

def fibonacci(n):
    """Yield the first n Fibonacci numbers."""
    a,b = 1,1
    for i in xrange(1,n+1):
        yield a
        a,b = b, a+b

def foo(n):
    my_list = []
    for i in range(n):
        num = np.random.randint(-9,9)
        my_list.append(num)
    evens = 0
    for j in range(n):
        if my_list[j] % 2 == 0:
            evens += my_list[j]
    return my_list, evens

def foo_opt(n):
    """An optimized version of 'foo'"""
    my_list = np.random.randint(-9,9,n)
    return my_list, np.sum(my_list[my_list %2 == 0])

def compare_foo(n):
    """Prints a comparison of foo and foo_opt"""
    print compare_timings(foo, foo_opt, n)


def pymatpow(X, power):
    """Return X^{power}, the matrix product XX...X, 'power' times.

    Inputs:
        X ((n,n) ndarray): A square matrix.
        power (int): The power to which to raise X.
    """
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

@jit
def numba_matpow(X, power):
    """ Return X^{power}.

    Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.
    Returns:
        prod (ndarray):  X^{power}
    """
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        for i in xrange(size):
            for j in xrange(size):
                tot = 0.
                for k in xrange(size):
                    tot += prod[i,k] * X[k,j]
                temparr[j] = tot
            prod[i] = temparr
    return prod

@jit
def numpy_matpow(X, power):
    """ Return X^{power}.

    Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.
    Returns:
        prod (ndarray):  X^{power}
    """
    prod = X.copy()
    temparr = np.empty_like(X[0])
    size = X.shape[0]
    for n in xrange(1, power):
        X = np.dot(X,prod)
        # for i in xrange(size):
        #     for j in xrange(size):
        #         tot = 0.
        #         for k in xrange(size):
        #             tot += prod[i,k] * X[k,j]
        #         temparr[j] = tot
        #     prod[i] = temparr
    return X


def compare_matpow(X, power):
    """
    Inputs:
        X (ndarray):  A square 2-D NumPy array
        power (int):  The power to which to raise X.

    Prints a comparison of pymatpow and numba_matpow
    Prints a comparison of pymatpow and numpy_matpow
    """
    numba_matpow(X,2)
    numpy_matpow(X,2)
    print compare_timings(pymatpow, numba_matpow, X, power)
    print '\n', compare_timings(pymatpow, numpy_matpow, X, power)


def init_tridiag(n):
    """Construct a random nxn tridiagonal matrix A by diagonals.

    Inputs:
        n (int): The number of rows / columns of A.

    Returns:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        A ((n,n) ndarray): the tridiagonal matrix.
    """
    a = np.random.random_integers(-9, 9, n-1).astype("float")
    b = np.random.random_integers(-9 ,9, n  ).astype("float")
    c = np.random.random_integers(-9, 9, n-1).astype("float")

    # Replace any zeros with ones.
    a[a==0] = 1
    b[b==0] = 1
    c[c==0] = 1

    # Construct the matrix A.
    A = np.zeros((b.size,b.size))
    np.fill_diagonal(A, b)
    np.fill_diagonal(A[1:,:-1], a)
    np.fill_diagonal(A[:-1,1:], c)

    return a, b, c, A

def pytridiag(a, b, c, d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.

    Inputs:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        d ((n,) ndarray): the right side of the linear system.

    Returns:
        x ((n,) ndarray): solution to the tridiagonal system Ax = d.
    """
    n = len(b)

    # Make copies so the original arrays remain unchanged.
    aa = np.copy(a)
    bb = np.copy(b)
    cc = np.copy(c)
    dd = np.copy(d)

    # Forward sweep.
    for i in xrange(1, n):
        temp = aa[i-1] / bb[i-1]
        bb[i] = bb[i] - temp*cc[i-1]
        dd[i] = dd[i] - temp*dd[i-1]

    # Back substitution.
    x = np.zeros_like(b)
    x[-1] = dd[-1] / bb[-1]
    for i in reversed(xrange(n-1)):
        x[i] = (dd[i] - cc[i]*x[i+1]) / bb[i]

    return x

@jit
def numba_tridiag(a, b, c, d):
    """Solve the tridiagonal system Ax = d where A has diagonals a, b, and c.

    Inputs:
        a ((n-1,) ndarray): first subdiagonal of A.
        b ((n,) ndarray): main diagonal of A.
        c ((n-1,) ndarray): first superdiagonal of A.
        d ((n,) ndarray): the right side of the linear system.

    Returns:
        x ((n,) ndarray): solution to the tridiagonal system Ax = d.
    """
    n = len(b)

    # Make copies so the original arrays remain unchanged.
    aa = np.copy(a)
    bb = np.copy(b)
    cc = np.copy(c)
    dd = np.copy(d)

    # Forward sweep.
    for i in xrange(1, n):
        temp = aa[i-1] / bb[i-1]
        bb[i] = bb[i] - temp*cc[i-1]
        dd[i] = dd[i] - temp*dd[i-1]

    # Back substitution.
    x = np.zeros_like(b)
    x[-1] = dd[-1] / bb[-1]
    for i in reversed(xrange(n-1)):
        x[i] = (dd[i] - cc[i]*x[i+1]) / bb[i]

    return x


def compare_tridiag():
    """Prints a comparison of numba_tridiag and pytridiag
       prints a comparison of numba_tridiag and scipy.linalg.solve."""

    a,b,c,A = init_tridiag(1000000)
    d = np.random.rand(1000000)
    numba_tridiag(a, b, c, d)
    print compare_timings(numba_tridiag, pytridiag, a, b, c, d)

    a,b,c,A = init_tridiag(1000)
    d = np.random.rand(1000)
    numba_tridiag(a, b, c, d)

    start = time.time()
    function1 = numba_tridiag(a,b,c,d)
    time1 = time.time() - start

    start = time.time()
    function2 = la.solve(A,d)
    time2 = time.time() - start
    print "\nTiming for "+ str(numba_tridiag)+ ": "+ str(time1)+ "\n"+ "Timing for "+ str(la.solve)+ ": "+ str(time2)


if __name__ == '__main__':
    A = np.array([[9.,9., 3., 6],[5., 2., 9., 5.],[6., 3., 0., 7.],[3., 0., 7., 8.]])
    print compare_LU(A)

    X = np.random.rand(1,1000000)
    compare_sum(X)

    a = fibonacci(50)
    for i in xrange(50):
        print next(a)

    print compare_foo(200)


    print pymatpow(A, 3)
    print numpy_matpow(A, 3)

    print compare_matpow(A, 40)
    print compare_tridiag()
