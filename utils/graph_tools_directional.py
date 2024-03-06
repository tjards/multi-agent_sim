#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:13:33 2024

This is a module that supplements graph_tools to support direction graphs

@author: tjards
"""
# import stuff
# -------------
import numpy as np
from utils import conic_tools as sensor

#%% compute the adjacency matrix (directional)
def adj_matrix_bearing(states_q,states_p,r_matrix,aperature, headings):
    
    # initialize
    nNodes  = states_q.shape[1]             # number of agents (nodes)
    A       = np.zeros((nNodes,nNodes))     # initialize adjacency matrix as zeros
    # for each node
    for i in range(0,nNodes):  
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # grab the radius for this nodal relationship (stored in lattice size)
                r    = r_matrix[i,j] + 0.5 
                # compute the heading vector
                v_a   = np.array((np.cos(headings[0,i]), np.sin(headings[0,i]), 0 ))
                # if in sensor range
                if sensor.is_point_in_sensor_range(states_q[0:3,i], states_q[0:3,j], v_a, aperature, r):
                    # mark as neighbour
                    A[i,j] = 1
    
    # ensure A = A^T
    #assert (A == A.transpose()).all()
    # return the matrix

    return A

# find one way connected components 
def find_one_way_connected_components(matrix):
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor, connected in enumerate(matrix[node]):
            if connected == 1 and neighbor not in visited:
                dfs(neighbor, component)

    components = []
    visited = set()

    for node in range(len(matrix)):
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components

def out_degree_centrality(matrix, component):
    centrality = {}
    for node in component:
        out_degree = sum(matrix[node])
        centrality[node] = out_degree
    return centrality

def deg_matrix(centrality):
    
    
    
    

#%% Example usage with a directed adjacency matrix
# adjacency_matrix = [[0, 1, 1, 0],
#                     [0, 0, 0, 1],
#                     [0, 0, 0, 0],
#                     [1, 0, 1, 0]]

# one_way_components = find_one_way_connected_components(adjacency_matrix)
# for i, component in enumerate(one_way_components):
#     print(f"One-Way Connected Component {i+1}: {component}")
#     centrality = out_degree_centrality(adjacency_matrix, component)
#     print("Out-Degree Centrality:", centrality)


