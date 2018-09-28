# -*- coding: utf-8 -*-
# bfs_kbacon.py
"""Breadth-First Search (Kevin Bacon).
Bailey Smith
October 20, 2016
"""

from collections import deque
import networkx as nx

class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        my_string = ""
        sort = sorted(self.dictionary.keys())
        for i in sort:
            sort2 = sorted(self.dictionary[i])
            s = "; "
            my_string += i + ": " + s.join(sort2) + "\n"
        return my_string[:-1]

    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        if start not in self.dictionary.keys():
            raise ValueError(str(start) + " is not in the adjacency dictionary")
        visited = []
        marked_set = set()
        visit_queue = deque()
        marked_set.add(start)
        visit_queue.append(start)
        current = start
        while len(visit_queue) > 0:
            current = visit_queue.popleft()
            visited.append(current)
            for i in self.dictionary[current]:
                if i not in marked_set:
                    visit_queue.append(i)#? this adds everything start is connected to
                    # Since each neighbor will be visited, add them to marked as well.
                    marked_set.add(i)
        return visited

    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        pass

    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        if start not in self.dictionary.keys():
            raise ValueError(str(start) + " is not in the adjacency dictionary")
        if target not in self.dictionary.keys():
            raise ValueError(str(target) + " is not in the adjacency dictionary")

        parent_dict = {}
        visited = []
        marked_set = set()
        visit_queue = deque()
        marked_set.add(start)
        visit_queue.append(start)
        current = start
        while len(visit_queue) > 0:
            current = visit_queue.popleft()
            visited.append(current)
            if current == target:
                break#?
            for i in self.dictionary[current]:
                if i not in marked_set:
                    visit_queue.append(i)#? this adds everything start is connected to
                    # Since each neighbor will be visited, add them to marked as well.
                    marked_set.add(i)
                    parent_dict[i] = current
        path = []
        #append to list backwards
        current = target
        #Before finding path append target point to list..
        path.append(current)
        while current != start:
             path.append(parent_dict[current])
             current = parent_dict[current]
        #reverse list
        path.reverse()
        return path

def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph = nx.Graph()
    for i in dictionary.keys():
        for j in dictionary[i]:
            nx_graph.add_edge(i, j)
    return nx_graph


# Helper function
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)

    return graph


class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        self.filename = filename
        dictionary = parse(filename)
        self.graph = convert_to_networkx(dictionary)
        my_set = set()
        for i in dictionary.keys():
            for j in dictionary[i]:
                my_set.add(j)
        self.actors = my_set
        self.not_connected = 0

    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        if start not in self.actors:
            raise ValueError(start + " is not in the dictionary")
        if target not in self.actors:
            raise ValueError(target + " is not in the dictionary")

        return nx.shortest_path(self.graph, start, target)



    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        shortest_path = self.path_to_bacon(start,target)#List of actors/movies that connect start to target
        return (len(shortest_path)-1)/2.#Dont account for target or movie titles

    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
            2.664432056
        """
        self.not_connected = 0
        numbers = 0
        for actor in self.actors:
            try:
                numbers += self.bacon_number(actor, target)
            except:
                self.not_connected += 1

        not_connected = self.not_connected
        self.not_connected = 0
        return float(numbers)/(len(self.actors) - not_connected), not_connected


"""
V - All Hollywood actors
E - { (a1, a2) | a1 has costared with a2 }

Part 1
Writing own graph class: Breadth-First search
Build subset of actor graph and compute Kevin Bacon numbers

Part 2 *Start with?
Python Library for working with graphs (kinda like numpy)
Network x

A = {0:[1,3], 1:[4,0], 2:[3,4], 3:[0,2], 4:[1,2]} dictionary: directed graph
List(visited): visited = []
Set(marked): Set()#Need quick way of telling what has been visited
Queue(need to visit): visit_queue = deque()

Keeping track of shortest path: remember along the way
parent_dict = {}
#in for loop, if i not in marked_set
parent_dict[i] = current
"""
# =========================== END OF FILE =============================== #
