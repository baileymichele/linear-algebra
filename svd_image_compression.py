# -*- coding: utf-8 -*-
# svd_image_compression.py
"""SVD and Image Compression.
Bailey Smith
15 November 2016
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

def truncated_svd(A,k=None):
    """Computes the truncated SVD of A. If r is None or equals the number
        of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        k: the number of singular values to use
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    """
    AHA = np.dot(A.conj().T, A)
    values, vectors = la.eig(AHA)#Find eigenvalues and eigenvectors of A^HA
    idx = values.argsort()[::-1]
    values = values[idx]
    vectors = vectors[:,idx]
    values = np.sqrt(values)
    if k is not None:#Take only k singular values
        values = values[:k]
    values = values[values > 10**-6]#NEED TO GET RID OF 0 EIGENVALUES
    vectors = vectors[:,:len(values)]
    sigma = np.diag(values)
    V = vectors
    U = np.dot(A,V)

    m,n = np.shape(U)
    for i in xrange(n):
        U[:,i] = (1./sigma[i,i])*U[:,i]

    return U, values, V.conj().T


def visualize_svd():
    """Plot each transformation associated with the SVD of A."""
    A = np.array([[3,1],[1,3]])
    U,s,Vh = la.svd(A)
    sigma = np.diag(s)
    e_1 = np.array([[0,0],[0,1]])
    e_2 = np.array([[0,1],[0,0]])
    domain = np.linspace(0,2*np.pi,100)

    plt.subplot(221)
    plt.plot(np.cos(domain),np.sin(domain))
    plt.plot((0,0),(0,1))
    plt.plot((0,1),(0,0),"g")
    plt.axis ("equal")
    plt.title("S")

    xy = np.vstack((np.cos(domain),np.sin(domain)))

    plt.subplot(222)
    VhS = Vh.dot(xy)
    plt.plot(VhS[0,:],VhS[1,:])
    plt.plot(Vh.dot(e_1)[0,:],Vh.dot(e_1)[1,:])
    plt.plot(Vh.dot(e_2)[0,:],Vh.dot(e_2)[1,:], "g")
    plt.axis ("equal")
    plt.title("Vh*S")

    plt.subplot(223)
    SVhS = sigma.dot(Vh).dot(xy)
    plt.plot(SVhS[0,:],SVhS[1,:])
    plt.plot(sigma.dot(Vh).dot(e_1)[0,:],sigma.dot(Vh).dot(e_1)[1,:])
    plt.plot(sigma.dot(Vh).dot(e_2)[0,:],sigma.dot(Vh).dot(e_2)[1,:], "g")
    plt.axis ("equal")
    plt.title("Sigma*Vh*S")

    plt.subplot(224)
    USVhS = U.dot(sigma).dot(Vh).dot(xy)
    plt.plot(USVhS[0,:],USVhS[1,:])
    plt.plot(U.dot(sigma).dot(Vh).dot(e_1)[0,:],U.dot(sigma).dot(Vh).dot(e_1)[1,:])
    plt.plot(U.dot(sigma).dot(Vh).dot(e_2)[0,:],U.dot(sigma).dot(Vh).dot(e_2)[1,:], "g")
    plt.axis ("equal")
    plt.title("U*Sigma*Vh*S")


    plt.show()

def svd_approx(A, k):
    """Returns best rank k approximation to A with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    k - rank

    Return:
    Ahat - the best rank k approximation
    """
    U,s,Vh = la.svd(A, full_matrices=False)
    S = np.diag(s[:k])
    return U[:,:k].dot(S).dot(Vh[:k,:])#A hat

def lowest_rank_approx(A,e):
    """Returns the lowest rank approximation of A with error less than e
    with respect to the induced 2-norm.

    Inputs:
    A - np.ndarray of size mxn
    e - error

    Return:
    Ahat - the lowest rank approximation of A with error less than e.
    """
    U,s,Vh = la.svd(A, full_matrices=False)
    rank = np.linalg.matrix_rank(A)
    Ahat = A.copy()
    norm = la.norm(A - Ahat)
    while norm < e:
        Ahat_prev = Ahat.copy()
        S = np.diag(s[:rank])
        Ahat = U[:,:rank].dot(S).dot(Vh[:rank,:])
        norm = la.norm(A - Ahat)
        rank -= 1
        if norm >= e:
            return Ahat_prev
        if rank == 0:
            break
    return Ahat


def compress_image(filename,k):
    """Plot the original image found at 'filename' and the rank k approximation
    of the image found at 'filename.'

    filename - jpg image file path
    k - rank
    """
    '''Read in file, plot original image then the best approximation
    Deal with 3 different colors separately
    Need to make same size as original: Make array of zeros
    scale photo K = np.round(K) / 255.
    Fancy indexing'''

    my_file = plt.imread(filename).astype(float)
    R = my_file[:,:,0]
    G = my_file[:,:,1]
    B = my_file[:,:,2]
    K = np.zeros_like(my_file)
    K[:,:,0] = svd_approx(R, k)
    K[:,:,1] = svd_approx(G, k)
    K[:,:,2] = svd_approx(B, k)

    my_file = np.round(my_file) / 255.
    K = np.round(K) / 255.
    K[K>1] = 1
    K[K<0] = 0

    plt.subplot(121)
    plt.imshow(my_file)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(K)
    plt.title("Rank 20 Approximation")
    plt.show()

if __name__ == '__main__':
    U,s,V = truncated_svd(np.array([[1,0,0],[0,0,3],[0,0,0],[0,2,0],[1,1,0]]))
    print U
    Vh = np.conj(V).T
    sigma = np.diag(s)
    print U.dot(sigma).dot(Vh)

    print visualize_svd()

    A = np.array([[1,0,0,0,2],[0,0,3,0,0]])
    k = 3
    print svd_approx(A, k)

    A = np.array([[-3,2,5],[4,5,9],[-5,-10,2]])
    A = np.array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,2,0,0,0]])
    A = np.array([[-10,-7,-9,-7,-2,-6,6,-8,-6,2],[2,-10,5,-9,4,1,6,-3,9,8],[6,6,-6,-1,-4,-3,0,1,9,-6],
    [6,0,-10,8,-7,-8,-9,-9,1,4],[8,8,-3,4,0,-6,-7,-5,8,7],[-1,-2,-6,8,-3,7,1,1,0,-2],[-5,-8,9,5,9,1,4,-5,-3,0],
    [-8,-7,-2,-8,-8,-9,6,3,9,-9],[1,-6,6,-2,-3,-7,-2,-1,-2,-5],[-8,8,-5,5,-6,0,-6,-2,-9,-3]])
    print lowest_rank_approx(A,.2)
    U,s,Vh = truncated_svd(A)
    print np.allclose(A,np.dot(U,np.diag(s)).dot(Vh))

    print compress_image("hubble_image.jpg",20)
