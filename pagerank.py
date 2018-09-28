# pagerank.py
"""The Page Rank Algorithm.
Bailey Smith
March 28 2017
"""
import numpy as np
from scipy import sparse
from scipy import linalg as la


def to_matrix(filename, n):
    """Return the nxn adjacency matrix described by datafile.

    Parameters:
        datafile (str): The name of a .txt file describing a directed graph.
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile

    Returns:
        A SciPy sparse dok_matrix.
    """
    lines = []

    matrix = sparse.dok_matrix((n,n),dtype=float)
    with open(filename, 'r') as myfile:
        for line in myfile:
            lines = line.strip().split()
            try:
                matrix[int(lines[0]),int(lines[1])] = 1
            except ValueError:
                pass
    return matrix


def calculateK(A,N):
    """Compute the matrix K as described in the lab.

    Parameters:
        A (ndarray): adjacency matrix of an array
        N (int): the datasize of the array

    Returns:
        K (ndarray)
    """

    for i in xrange(N):
        total = sum(A[i,:])
        if total == 0:
            A[i,:] += 1
            total = N
        A[i,:] = A[i,:]/float(total)#normalize
    return A.T


def iter_solve(adj, N=None, d=.85, tol=1E-5):
    """Return the page ranks of the network described by 'adj'.
    Iterate through the PageRank algorithm until the error is less than 'tol'.

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    if N is None:
        size = adj.shape[0]
    else:
        size = N
        adj = adj[:N,:N]

    K = calculateK(adj,size)
    p0 = np.ones(size)/float(size)
    p1 = d*K.dot(p0) + (1-d)/(size)*np.ones(size)
    while la.norm(p0-p1) > tol:
        p0 = p1
        p1 = d*K.dot(p0) + (1-d)/(size)*np.ones(size)

    return p1


def eig_solve(adj, N=None, d=.85, tol=1E-5):
    """Return the page ranks of the network described by 'adj'. Use SciPy's
    eigenvalue solver to calculate the steady state of the PageRank algorithm

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    if N is None:
        size = adj.shape[0]
    else:
        size = N
        adj = adj[:N,:N]

    E = np.ones((size,size))
    K = calculateK(adj,size)
    p0 = np.ones(size)/float(size)
    p1 = (d*K + (1-d)/(size)*E).dot(p0)
    while la.norm(p0-p1) > tol:
        p0 = p1
        p1 = (d*K + (1-d)/(size)*E).dot(p0)

    return p1


def team_rank(filename='ncaa2013.csv'):
    """Use iter_solve() to predict the rankings of the teams in the given
    dataset of games. The dataset should have two columns, representing
    winning and losing teams. Each row represents a game, with the winner on
    the left, loser on the right. Parse this data to create the adjacency
    matrix, and feed this into the solver to predict the team ranks.

    Parameters:
        filename (str): The name of the data file.
    Returns:
        ranks (list): The ranks of the teams from best to worst.
        teams (list): The names of the teams, also from best to worst.
    """
    count = set()
    numbered = []
    games = []
    with open(filename, 'r') as ncaafile:
        ncaafile.readline()
        for line in ncaafile:
            teams = line.strip().split(',')
            games.append(teams)
            if teams[0] not in count:
                numbered.append(teams[0])
            if teams[1] not in count:
                numbered.append(teams[1])
            count.add(teams[0])
            count.add(teams[1])
    teams = np.array(games)
    size = len(count)
    A = sparse.dok_matrix((size, size))

    for i in xrange(len(games)):
        A[numbered.index(teams[i,1]),numbered.index(teams[i,0])] = 1.

    solve = iter_solve(A.toarray(),d=.7)
    sort = np.argsort(solve)
    np.array(numbered)
    my_dict = {str(i+1):numbered[sort[i]] for i in xrange(size)}
    reverse = sort[::-1]
    ranks = []
    teams = []
    for i in xrange(size):
        teams.append(my_dict[str(i+1)])
        ranks = np.sort(solve)[::-1]
    return ranks, teams[::-1]


if __name__ == '__main__':
    A = to_matrix("matrix.txt", 8)

    # A = np.array([[ 0,  0,  0,  0,  0,  0,  0,  1],
    #           [ 1,  0,  0,  0,  0,  0,  0,  0],
    #           [ 0,  0,  0,  0,  0,  0,  0,  0],
    #           [ 1,  0,  1,  0,  0,  0,  1,  0],
    #           [ 1,  0,  0,  0,  0,  1,  1,  0],
    #           [ 1,  0,  0,  0,  0,  0,  1,  0],
    #           [ 1,  0,  0,  0,  0,  0,  0,  0],
    #           [ 1,  0,  0,  0,  0,  0,  0,  0]])

    # print calculateK(A.toarray(),8)
    # print iter_solve(A.toarray(), N=5)
    # print eig_solve(A.toarray(), N=5)
    print team_rank()
