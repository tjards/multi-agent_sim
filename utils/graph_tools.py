#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:21:07 2023

@author: tjards

Preliminaries:
    - Let us consider V nodes (vertices, agents)
    - Define E is a set of edges (links) as the set of ordered pairs
    from the Cartesian Product V x V, E = {(a,b) | a /in V and b /in V}
    - Then we consider Graph, G = {V,E} (nodes and edges)
    - G is simple: (a,a) not \in E \forall a \in V 
    - G is undirected: (a,b) \in E <=> (b,a) \in E
    - Nodes i,j are neighbours if they share an edge, (i,j) /in E
    - d1=|N_1| is the degree of Node 1, or, the number of neighbours

"""

# Import stuff
# ------------
import numpy as np
import random
from collections import defaultdict, Counter
import heapq 

# Parameters
# ----------
#r       = 5
#nNodes  = 10
#data = 10*np.random.rand(3,nNodes)

#%% Build Graph (as dictionary)
# ----------------------------
def build_graph(data, r):
    G = {}
    nNodes  = data.shape[1]     # number of agents (nodes)
    # for each node
    for i in range(0,nNodes):
        # create a set of edges
        set_i = set()
        # search through neighbours (will add itself)
        for j in range(0,nNodes):
            # compute distance
            dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
            # if close enough
            if dist < r:
                # add to set_i
                set_i.add(j)
            #else:
            #    print("debug: ", i," is ", dist, "from ", j)
        G[i] = set_i
    return G

# count all
def build_graph_all(data):
    G = {}
    nNodes  = data.shape[1]     # number of agents (nodes)
    # for each node
    for i in range(0,nNodes):
        # create a set of edges
        set_i = set()
        # search through neighbours (will add itself)
        for j in range(0,nNodes):
            # compute distance
            dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
            # if close enough
            #if dist < r:
                # add to set_i
            set_i.add(j)
            #else:
            #    print("debug: ", i," is ", dist, "from ", j)
        G[i] = set_i
    return G



#%% Djikstra's shortest path
# --------------------------
# accepts a Graph and starting node
# finds the shortest path from source to all other nodes

def search_djikstra(G, source):
    
    closed = set()                              # set of nodes not to visit (or already visited)
    parents = {}                                # stores the path back to source 
    costs = defaultdict(lambda: float('inf'))   # store the cost, with a default value of inf for unexplored nodes
    costs[source] = 0
    queue = []                                    # to store cost to the node from the source
    heapq.heappush(queue,(costs[source],source))  # push element into heap in form (cost, node)
    # note: heap is a binary tree where parents hold smaller values than their children
    
    # while there are elements in the heap
    while queue:
        
        # "i" is the index for the node being explored in here
        cost_i, i = heapq.heappop(queue)        # returns smallest element in heap, then removes it
        closed.add(i)                           # add this node to the closed set
        
        # search through neighbours
        for neighbour in G[i]:
            
            # we'll define each hop/step with a cost of 1, but this could be distance (later)
            step_cost = 1
            
            # don't explore nodes in closed set
            if neighbour in closed:
                continue
            
            # update cost
            cost_update = costs[i] + step_cost
            
            # if updated cost is less than current (default to inf, hence defaultdict)
            if  cost_update < costs[neighbour]:
                
                #store the and parents and costs
                parents[neighbour] = i
                costs[neighbour] = cost_update
                
                # add to heap
                heapq.heappush(queue, (cost_update, neighbour))

    return parents, costs
    


#%% Adjacency Matrix
# ------------------------------

# A = {a_ij} s.t. 1 if i,j are neighbours, 0 if not
def adj_matrix(data,r):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    A       = np.zeros((nNodes,nNodes)) # initialize adjacency matrix as zeros
    # for each node
    for i in range(0,nNodes):  
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    A[i,j] = 1
    # ensure A = A^T
    assert (A == A.transpose()).all()
    # return the matrix
    return A


#%% Compute the Degree Matrix
# ------------------------------
# D = diag{d1,d2,...dN}
def deg_matrix(data,r):
    # initialize
    nNodes  = data.shape[1]             # number of agents (nodes)
    D       = np.zeros((nNodes,nNodes)) # initialize degree matrix as zeros
    # for each node
    for i in range(0,nNodes):
        # search through neighbours
        for j in range(0,nNodes):
            # skip self
            if i != j:
                # compute distance
                dist = np.linalg.norm(data[0:3,j]-data[0:3,i])
                # if close enough
                if dist < r:
                    # mark as neighbour
                    D[i,i] += 1
    # return the matrix
    return D

#%% Compute the graph Laplacian
# -----------------------------
def lap_matrix(A,D):
    L = D-A
    eigs = np.linalg.eigvals(L)         # eigen values 
    # ensure L = L^T
    assert (L == L.transpose()).all()
    # ensure has zero row sum
    assert L.sum() == 0
    # ensure Positive Semi-Definite (all eigen values are >= 0)
    assert (eigs >= 0).all()
    # return the matrix
    return L

#%% Compute components
# --------------------
def compute_comp(L):
    eigs = np.linalg.eigvals(L)         # eigen values 
    # how many components (how many zero eig values)
    nComp = np.count_nonzero(eigs==0)
    #print('the graph has ', nComp, ' component(s)')
    return nComp


#%% Compute Augmented Laplacian: The number of null eigen values is 
#   the number of components in the graph that do not contain pins.
#   generally, the larger the aug connectivity, the better.
# ----------------------------------------------------------------- 
def compute_aug_lap_matrix(L,P,gamma,rho):
    L_aug = np.multiply(gamma, L) + np.multiply(rho, P)
    eigs = np.linalg.eigvals(L_aug)         # eigen values
    # ensure Positive Semi-Definite (all eigen values are >= 0)
    assert (eigs >= 0).all()
    # tell me if not fully pinned (i.e. there are null eigen values)
    if np.count_nonzero(eigs==0) > 0:
        print('note: graph is not fully pinned')
    # compute the augmented connectivity (smallest eig value)
    aug_connectivity = np.amin(eigs)
    # and the index
    aug_connectivity_i = np.argmin(eigs)
    # return the matrix, augmented connectivity, and index
    return L_aug, aug_connectivity, aug_connectivity_i


#%% compute the controlability matrix
# ---------------------------------
def func_ctrlb(Ai,Bi,horizon):
    A = np.mat(Ai)
    B = np.mat(Bi).transpose()
    n = horizon
    #n = A.shape[0]
    ctrlb = B
    for i in range(1,n):
        #ctrlb = np.hstack((ctrlb,A**i*B))
        ctrlb = np.hstack((ctrlb,np.dot(A**i,B)))
    return ctrlb


#%% compute the controlability Gram trace
# -------------------------------------
def compute_gram_trace(A,D,node,horizon):
    
    # define B
    B = np.zeros((A.shape[0]))
    #B = np.ones((A.shape[0]))
    B[node] = 1
    
    # discretize (zero order hold)
    #Ad = np.eye(A.shape[0],A.shape[0])+A*dt
    #Bd = B*dt
    
    # IAW with "transmission" from Appendix of Nozari et al. (2018)
    #D_c_in = compute_deg_matrix(A) # inmport this in
    A_dyn = np.dot(A,np.linalg.inv(D))
    
    #alpha = 1
    #A_dyn = np.exp(alpha*(-np.eye(A.shape[0],A.shape[0])+A))
    
    # compute
    C = func_ctrlb(A_dyn,B, horizon)
    W = np.dot(C,C.transpose())
    
    #test controlability
    rank = np.linalg.matrix_rank(C)
    if rank == C.shape[1]:
        ctrlable = True
    else:
        ctrlable = False
        
    # the trace is inversely prop to the energy required to control network
    trace = np.matrix.trace(W)
    
    return ctrlable, trace




#%% compute Betweenness Centrality
# ------------------------------
def betweenness(G):
    
    # store the betweenness for each node
    all_paths = {}
    k = 0
    
    # create a nested dict of all the shortest paths 
    for i in range(0,len(G)):
        
        parents, _      = search_djikstra(G,i)
        all_paths[k]    = parents
        k += 1
    
    # count all the influencers (i.e. those on shortest paths)
    influencers = count_influencers(G,all_paths)
    # sum of all paths
    summ = len(G)*(1+len(G))/2
    
    return {n: influencers[n] / summ for n in influencers}


#%% count instances of node appearing in a shortest path
# ------------------------------------------------------
def count_influencers(G,all_paths):

    influencers = defaultdict(lambda: float(0))
    
    # do all this for each destination nodes
    for k in range(0,len(G)):
        
        # create a set of all other nodes
        search = set(range(0,len(G)))
        search.remove(k)
    
        # we will search through all the other nodes
        while search:
        
            # select and remove a node
            i = search.pop()
            
            # this will be part of the search
            sub_search = set()
            sub_search.add(i)
            
            # but so will others we find along the way (sub search)
            while sub_search:
                
                j = sub_search.pop()
            
                # identify the parent 
                parent_i = all_paths[k][j]
                
                # if this this the destination
                if parent_i == k:
                    # get out
                    continue
                # else, keep looking
                else:
                
                    # add this to the subsearch
                    sub_search.add(parent_i)
            
                    # count this parent as part of a path
                    influencers[parent_i] += 1
                    #print(parent_i, 'is a parent of', j, 'of ...', k )
        
    return influencers 


# find connected components
# -------------------------
def find_connected_components_A(A):
    all_components = []                                     # stores all connected components
    visited = []                                            # stores all visisted nodes
    for node in range(0,A.shape[1]):                        # search all nodes (breadth)
        if node not in visited:                             # exclude nodes already visited
            component       = []                            # stores component nodes
            candidates = np.nonzero(A[node,:].ravel()==1)[0].tolist()    # create a set of candidates from neighbours 
            component.append(node)
            visited.append(node)
            candidates = list(set(candidates)-set(visited))
            while candidates:                               # now search depth
                candidate = candidates.pop(0)               # grab a candidate 
                visited.append(candidate)                   # it has how been visited 
                subcandidates = np.nonzero(A[:,candidate].ravel()==1)[0].tolist()
                component.append(candidate)
                #component.sort()
                candidates.extend(list(set(subcandidates)-set(candidates)-set(visited))) # add the unique nodes          
            all_components.append(component)
    return all_components


#%% testing
# --------

# G               = build_graph(data)
# parents, costs  = search_djikstra(G, 0)
# adjacency       = adj_matrix(data)
# degree          = deg_matrix(data)
# laplacian       = lap_matrix(adjacency, degree)
# betweennesses   = betweenness(G)


    
    

    
    
    
    



