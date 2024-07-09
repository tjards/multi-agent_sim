#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:40:58 2024

This program manually computes the local k-connectivity of a nodes in a graph
based on the adjacency matrix. The graph can be directed or undirected.

Local k-connectivity represents the minimum number of nodes that need to be 
removed in order to disconnect that subject node from the rest of the graph.
(maximum number of vertex-disjoint paths between each 
 node and any other node in the graph)

Given  vertices (u,v), a 'path' between these vertices is a sequences of edges
that connect them with revisiting any vertex. 'Vertex-disjoint paths' refer
to paths between two vertices that do not share any vertices other than 
the endpoints.

Roughly outline of algo:
    
    1. Input Adjacency matrix (A), source, target, and k (to be tested)
    2. Count neighbours
    3. If number of neighbours < k, just store the number of neighbours 
    4. check is there exists k vertex-disjoint paths between the node and its neighbors to determine k-connectivity


This will be the reward function in upcoming RL application

@author: tjards
"""

# import stuff
# ------------
import numpy as np

# find the disjoint path between source and target
# ------------------------------------------------
def find_k_disjoint_paths(A, source, target, k):
    
    n = len(A)             # number of nodes
    visited = [False] * n  # initialize a list to store visited nodes
    paths = []             # count paths

    # we'll do a depth first search
    def dfs(node, path):
        visited[node] = True 
        if node == target:
            paths.append(path[:]) # store path, if target found
        else:
            for neighbor in range(n):
                if A[node][neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor, path + [neighbor]) # search for paths (recursively)
        visited[node] = False

    # initiate search from source node
    dfs(source, [source])
    
    # filter out paths !=  k
    #k_paths = [p for p in paths if len(p) == k + 1]
    
    # alternatively, filter out paths <  k
    #k_paths = [p for p in paths if len(p) >= k + 1]
    
    k_paths= paths

    return len(k_paths) >= k

# find the local connectivity 
# ---------------------------
def local_k_connectivity(A, k, pick = -1):
    
    n = len(A)          # number of nodes 
    
    if pick == -1:      # -1 denoes searching all nodes
        node_list = list(range(0,n))
    else:
        node_list = [pick] # else, specifies a specific node
    
    local_k = {}

    # for each node
    for node in node_list:
        local_k[node] = 0
        # search through connected neighbours
        neighbors = [i for i in range(n) if A[node][i] > 0]
        # if number of neighbours is less than k
        if len(neighbors) < k:
            # just store the number of neighbours (will be filtered out)
            local_k[node] = len(neighbors)
        else:
            # else, if there are at least k neighbours 
            for neighbor in neighbors:
                # count and store the disjointed paths 
                if find_k_disjoint_paths(A, node, neighbor, k):
                    local_k[node] = k
                    break
        
    return local_k

# Example adjacency matrix for a directed graph
A = np.array(
    [[0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
 [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
 [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
 [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
 [1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
 [0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
 [0, 0, 0, 0, 0, 1, 0, 1, 1, 0]]
    )

# Compute local k-connectivity for each node
k = len(A) #np.inf #
local_k_values = local_k_connectivity(A, k, -1)
print("Local k-connecticity path values:")
for key in local_k_values:
    print(f"node {key}: {local_k_values[key]}")
