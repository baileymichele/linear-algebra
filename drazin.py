# drazin.py
"""The Drazin Inverse.
Bailey Smith
April 11 2017
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la


# Helper function 
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.allclose(la.det(A),0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        bool: True of Ad is the Drazin inverse of A, False otherwise.
    """

    one = np.allclose(A.dot(Ad),Ad.dot(A))
    two = np.allclose(np.linalg.matrix_power(A,k+1).dot(Ad),np.linalg.matrix_power(A,k))
    three = np.allclose(Ad.dot(A.dot(Ad)),Ad)

    if ((one is True) and (two is True) and (three is True)):
        return True
    else:
        return False


def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        Ad ((n,n) ndarray): The Drazin inverse of A.
    """
    f = lambda x: abs(x) > tol
    f2 = lambda x: abs(x) <= tol
    n = np.shape(A)[0]
    Q1, S, k1 = la.schur(A,sort=f)
    Q2, T, k2 = la.schur(A,sort=f2)
    U = np.hstack((S[:,:k1],T[:,:n-k1]))
    UI = la.inv(U)
    V = UI.dot(A).dot(U)
    Z = np.zeros((n,n))
    if k1 != 0:
        MI = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = MI
    return U.dot(Z).dot(UI)


def laplacian(A):
    '''
    Compute the Laplacian matrix of the adjacency matrix A.
    Inputs:
        A (array): adjacency matrix for undirected weighted graph,
             shape (n,n)
    Returns:
        L (array): Laplacian matrix of A
    '''
    n = A.shape[0]
    D = np.zeros((n,n))
    for i in xrange(n):
        # print A[:,i]
        D[i,i] = sum(A[:,i])
    return D-A

def effective_res(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ER ((n,n) ndarray): A matrix of which the ijth entry is the effective
        resistance from node i to node j.
    """
    n = np.shape(A)[0]
    L = laplacian(A)
    I = np.eye(n)
    ER = np.zeros((n,n))
    for j in xrange(n):
        Lj = np.copy(L)
        Lj[j] = I[j]
        D = drazin_inverse(Lj)
        ER[:,j] = np.diag(D)
        ER[j,j] = 0
    return ER

class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        count = set()
        connections = []
        numbered = []
        with open(filename, 'r') as network:
            for line in network:
                names = line.strip().split(',')
                connections.append(names)
                count.add(names[0])
                count.add(names[1])
        names = np.array(connections)
        sort = sorted(count)#makes the set a sorted list
        size = len(count)
        A = sparse.dok_matrix((size, size))

        for i in xrange(len(connections)):
            A[sort.index(names[i,1]),sort.index(names[i,0])] = 1.
            A[sort.index(names[i,0]),sort.index(names[i,1])] = 1.

        self.adj = A.toarray()
        self.ER = effective_res(self.adj)
        self.names = sort

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        ER_copy = np.copy(self.ER)
        mask = self.adj < 1
        ER_copy *= mask
        mask2 = ER_copy == 0
        ER_copy += np.max(ER_copy)*np.ones_like(ER_copy)*mask2
        minimum = np.min(ER_copy)
        min_loc = np.where(ER_copy==minimum)
        if node == None:
            return self.names[min_loc[0][0]], self.names[min_loc[1][0]]
        else:
            if node not in self.names:
                raise ValueError("Node not in the network")
            minimum = np.min(ER_copy[self.names.index(node)])
            min_loc = np.where(ER_copy==minimum)
            return self.names[min_loc[1][0]]


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if node1 not in self.names or node2 not in self.names:
            raise ValueError('Node not in network')
        self.adj[self.names.index(node1),self.names.index(node2)] = 1.
        self.adj[self.names.index(node2),self.names.index(node1)] = 1.
        self.ER = effective_res(self.adj)

