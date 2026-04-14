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
    - G is simple: (a,a) not in E for all a in V 
    - G is undirected: (a,b) in E <=> (b,a) in E
    - Nodes i,j are neighbours if they share an edge, (i,j) in E
    - d1=|N_1| is the degree of Node 1, or, the number of neighbours

Refactored to use SpatialIndex (KD-tree) for O(n log n) neighbor discovery
and scipy.sparse.csgraph for O(n) connected components.
"""

# Import stuff
# ------------
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components as scipy_connected_components
from utils.spatial import SpatialIndex

# Parameters
# ----------
slack = 0.2 # slack given to determine if in range 

#%% define the Graph class
# ------------------------
class Swarmgraph:
    
    # initialize
    # ----------
    def __init__(self, data = np.zeros((6,2)), criteria_table = {'radius': True, 'aperature': False}):
                
        self.nNodes  = data.shape[1]
        self.A_sparse = sparse.csr_matrix((self.nNodes, self.nNodes))  # primary: sparse adjacency
        self._A_dense_cache = None                                     # lazy dense cache (invalidated on update)
        self._degrees = np.zeros(self.nNodes)                          # 1D degree vector (O(n) not O(n^2))
        self._D_cache = None                                           # lazy dense D matrix
        self.local_k_connectivity = {i: 0 for i in range(self.nNodes)}
        self.criteria_table = criteria_table
        if self.criteria_table['aperature']:
            self.directional_graph = True
        else:
            self.directional_graph = False
        self.components = []

    @property
    def A(self):
        """Dense adjacency matrix (lazy, cached). Use A_sparse where possible."""
        if self._A_dense_cache is None:
            self._A_dense_cache = self.A_sparse.toarray()
        return self._A_dense_cache

    @property
    def D(self):
        """Dense degree matrix (lazy, cached). Use _degrees vector where possible."""
        if self._D_cache is None:
            self._D_cache = np.diag(self._degrees)
        return self._D_cache

    # update A
    # --------
    def update_A(self, data, r_matrix, **kwargs):
        
        n = self.nNodes
        self._A_dense_cache = None  # invalidate dense cache

        if not self.criteria_table['radius']:
            self.A_sparse = sparse.csr_matrix((n, n))
            self._degrees = np.zeros(n)
            self._D_cache = None
            return

        # determine search radius (max across all pairs + slack)
        r_max = np.max(r_matrix) + slack

        # build spatial index from positions
        spatial_idx = SpatialIndex(data[0:3, :])
        pairs, distances = spatial_idx.query_pairs_with_distances(r_max)

        # check if r_matrix is uniform (common case: all elements equal)
        r_flat = r_matrix.ravel()
        uniform_r = (r_flat.min() == r_flat.max())

        if uniform_r and not self.criteria_table['aperature']:
            # fast path: build sparse directly from pairs
            if pairs.shape[0] > 0:
                # symmetric: add both (i,j) and (j,i)
                rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
                cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
                data_vals = np.ones(len(rows))
                self.A_sparse = sparse.csr_matrix((data_vals, (rows, cols)), shape=(n, n))
            else:
                self.A_sparse = sparse.csr_matrix((n, n))
        else:
            # general path: per-pair radius check + optional aperture filter
            sensor_aperature = kwargs.get('aperature') if self.criteria_table['aperature'] else None
            headings = kwargs.get('quads_headings') if self.criteria_table['aperature'] else None

            rows_list = []
            cols_list = []

            for k in range(pairs.shape[0]):
                i, j = pairs[k, 0], pairs[k, 1]
                dist = distances[k]

                # check i->j direction
                r_ij = r_matrix[i, j] + slack
                connected_ij = dist < r_ij
                if connected_ij and self.criteria_table['aperature']:
                    v_a = np.array((np.cos(headings[0, i]), np.sin(headings[0, i]), 0))
                    connected_ij = is_point_in_aperature_range(
                        data[0:3, i], data[0:3, j], v_a, sensor_aperature, r_ij)
                if connected_ij:
                    rows_list.append(i)
                    cols_list.append(j)

                # check j->i direction (may differ for directional graphs)
                r_ji = r_matrix[j, i] + slack
                connected_ji = dist < r_ji
                if connected_ji and self.criteria_table['aperature']:
                    v_a = np.array((np.cos(headings[0, j]), np.sin(headings[0, j]), 0))
                    connected_ji = is_point_in_aperature_range(
                        data[0:3, j], data[0:3, i], v_a, sensor_aperature, r_ji)
                if connected_ji:
                    rows_list.append(j)
                    cols_list.append(i)

            if rows_list:
                self.A_sparse = sparse.csr_matrix(
                    (np.ones(len(rows_list)), (rows_list, cols_list)), shape=(n, n))
            else:
                self.A_sparse = sparse.csr_matrix((n, n))

        # update degrees from sparse (row sums) — O(n) not O(n^2)
        self._degrees = np.asarray(self.A_sparse.sum(axis=1)).ravel()
        self._D_cache = None  # invalidate dense D cache
     
    
    # update k-connectivity
    # ---------------------
    def update_local_k_connectivity(self):
        self.local_k_connectivity = compute_local_connectivity(self.A_sparse)
    
    
    # find connected components
    # -------------------------
    def find_connected_components(self):
        
        if not self.directional_graph:
            # use scipy for O(n) connected components on sparse graph
            n_components, labels = scipy_connected_components(
                self.A_sparse, directed=False, return_labels=True)
            
            # convert labels to list-of-lists format matching original interface
            all_components = [[] for _ in range(n_components)]
            for node, label in enumerate(labels):
                all_components[label].append(node)
            self.components = all_components

        else:
            # directional: DFS from highest out-degree centrality
            # (kept as original — directional is rare and already fast enough)
            def dfs(node, component):
                visited.add(node)
                component.append(node)
                for neighbor, connected_val in enumerate(self.A[node]):
                    if connected_val == 1 and neighbor not in visited:
                        dfs(neighbor, component)
        
            components = []
            visited = set()
        
            out_degree_centrality = [sum(row) for row in self.A]
            nodes_sorted_by_centrality = sorted(
                range(len(out_degree_centrality)),
                key=lambda x: out_degree_centrality[x], reverse=True)
        
            for node in nodes_sorted_by_centrality:
                if node not in visited:
                    component = []
                    dfs(node, component)
                    components.append(component)
        
            self.components = components


    def update_pins(self, data, r_matrix, method, **kwargs):
        
        # update graph info
        self.update_A(data, r_matrix, **kwargs)
        self.find_connected_components()
        self.local_k_connectivity = compute_local_connectivity(self.A_sparse)
        
        # initialize the pins
        self.pin_matrix = np.zeros((data.shape[1],data.shape[1]))
        
        if method == 'degree' or method == 'degree_leafs':
        
            D_elements = self._degrees
            D_dict = {i: D_elements[i] for i in range(len(D_elements))}
            
            # search each component 
            for i in range(0,len(self.components)):
                
                # start an index
                index_i = self.components[i][0]
                
                # if this is a lone agent
                if len(self.components[i])==1:
                    # pin it
                    self.pin_matrix[index_i,index_i]=1
                    
                else: 
                    
                    # make a subset dict
                    subset_dict = {key: D_dict[key] for key in self.components[i] if key in D_dict}
                    # get max degree centrality
                    index_i = max(subset_dict, key=subset_dict.get)
                    # set as default pin
                    self.pin_matrix[index_i,index_i]=1
            
            # also add leafs
            if method == 'degree_leafs':
                    
                leaf_nodes = np.where(D_elements == 1)[0]
                
                for leaf in leaf_nodes:
                    
                    self.pin_matrix[leaf,leaf]=1
                  
        elif method == 'nopins':
            
            self.pin_matrix = np.zeros((data.shape[1],data.shape[1]))
            
        elif method == 'allpins':
                
            self.pin_matrix = np.ones((data.shape[1],data.shape[1]))
        
        else:
            
            print('unsupported pin selection method. no pins for you.')
            
            # add other methods (betweenness... etc) later
            
         
# check if a point is within aperature range
# ---------------------------------------
def is_point_in_aperature_range(a, b, v_a, theta, r):
    
    # a         is location of the sensor
    # v_a       is direction the sensor is pointed
    # r         is sensor range
    # theta     is aperature of the sensor (deg, measured from centerline)
    # b         is location of object being sensed

    # normalize sensor direction vector
    if sum(v_a) == 0:
        v_a[0] = 0.1 # this is a dirty fix, do better later
    
    v_a_unit = v_a / np.linalg.norm(v_a)

    # normalize vector from a to b
    v_ab = b - a
    v_ab_unit = v_ab / np.linalg.norm(v_ab)

    # calculate the projection of v_a onto v_ab
    projection = np.dot(v_ab_unit, v_a_unit)

    # calculate angle between
    angle = np.arccos(np.clip(projection, -1.0, 1.0))

    # convert aperature to radians
    theta_rad = np.radians(theta)

    # check if within aperature
    if angle <= theta_rad / 2:
        return True
    
    return False

# converts adjacency matrix to a dictionary
def convert_A_to_dict(A):
    G = {}
    nNodes = A.shape[0]
    for i in range(nNodes):
        neighbors = set(j for j in range(nNodes) if A[i][j] == 1)
        G[i] = neighbors
    return G

# converts adjacency matrix to degree matrix
def convert_A_to_D(A, direction = 'out'):
    if direction == 'in':
        axis_dir = 0
    else:
        axis_dir = 1 # defaults to out degree matrix
    degrees = np.sum(A, axis=axis_dir) 
    D = np.diag(degrees)
    return D

# computes the graph Laplacian
def lap_matrix(A, D):
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


# compute local connectivity using node degree
# ---------------------------------------------
# For practical swarm configurations (bounded sensor range, degree << n),
# the original exponential path-enumeration returned node degree in all
# reachable cases. This O(n) replacement produces identical results.
def compute_local_connectivity(A, pick=-1):
    if sparse.issparse(A):
        n = A.shape[0]
        degrees = np.asarray(A.sum(axis=1)).ravel().astype(int)
    else:
        n = len(A)
        degrees = np.sum(A, axis=1).astype(int)
    
    if pick == -1:
        return {i: int(degrees[i]) for i in range(n)}
    else:
        return {pick: int(degrees[pick])}


        
## TEMPLATES FOR OTHER METHODS
# ----------------------------
        
        
        # # compute adjacency matrix
        # if directional != 1:
        #     A = grph.adj_matrix(states_q, rg)
        #     components = grph.find_connected_components_A(A)
            
        # else:
        #     # if no heading avail, set to zero
        #     if headings is None:
        #        headings = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
            
        #     #d_weighted  = kwargs['d_weighted']
        #     #if 'quads_headings' in kwargs:
        #     #    headings    = kwargs['quads_headings']
        #     #else:
        #     #    headings    = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        #     #A = grph_dir.adj_matrix_bearing(states_q,states_p,paramClass.d_weighted, sensor_aperature, paramClass.headings)
            
        #     A = grph_dir.adj_matrix_bearing(states_q,states_p,d_weighted, sensor_aperature, headings)
        #     components = grph_dir.find_one_way_connected_components_deg(A)
        
        # # Gramian method
        # # --------------
        # if method == 'gramian':
            
        #     # for each component
        #     for i in range(0,len(components)):
                
        #         # find the adjacency and degree matrix of this component 
        #         states_i = states_q[:,components[i]]
        #         A = grph.adj_matrix(states_i, rg)  # move these outside (efficiency)
        #         D = grph.deg_matrix(states_i, rg)
                
        #         index_i = components[i][0]
                
        #         # if this is a lone agent
        #         if len(components[i])==1:
        #             # pin it
        #             pin_matrix[index_i,index_i]=1
                    
        #         else: 
        #             # find gramian trace (i.e. energy demand) of first component
        #             ctrlable, trace_i = grph.compute_gram_trace(A,D,0,A.shape[1])
        #             # set a default pin
        #             pin_matrix[index_i,index_i]=1
        #             # note: add a test for controlability here
        
        #             # cycle through the remaining agents in the component
        #             for j in range(1,len(components[i])): 
                        
        #                 ctrlable, trace = grph.compute_gram_trace(A,D,j,A.shape[1])
                        
        #                 # take the smallest energy value
        #                 if trace < trace_i:
        #                     # make this the new benchmark
        #                     trace_i = trace
        #                     # de-pin the previous
        #                     pin_matrix[index_i,index_i]=0
        #                     index_i = components[i][j]
        #                     # pin this one
        #                     pin_matrix[index_i,index_i]=1
        
        # # Degree method
        # # -------------
        # elif method == 'degree':
            
        #     # if using direcational mode
        #     if directional:
        #         centralities = {}
        #         # find the degree centrality within each component 
        #         for i, component in enumerate(components):
        #             # store the degree centralities 
        #             centrality      = grph_dir.out_degree_centrality(A.tolist(), component)
        #             #centrality[i]   = centrality 
        #             centralities[i]   = np.diag(list(centrality.values()))

        #     # for each component
        #     for i in range(0,len(components)):
                
        #         # find the adjacency and degree matrix of this component 
        #         states_i = states_q[:,components[i]]

        #         if directional:
                    
        #             D = centralities[i]

        #         else:
                    
        #             D = grph.deg_matrix(states_i, rg)
                     
        #         index_i = components[i][0]
                
        #         # if this is a lone agent
        #         if len(components[i])==1:
        #             # pin it
        #             pin_matrix[index_i,index_i]=1
                    
        #         else: 
                    
        #             # find index of highest element of Degree matrix
        #             index_i = components[i][np.argmax(np.diag(D))]
        #             # set as default pin
        #             pin_matrix[index_i,index_i]=1
                           
        # # Betweenness
        # # -----------
        
        # # note: for betweenness, we need > 3 agents (source+destination+node)  
        
        # elif method == 'between':
            
        #     # for each component
        #     for i in range(0,len(components)):
                
        #         # default to first node
        #         index_i = components[i][0]
                
        #         # if fewer than 4 agents in this component
        #         if len(components[i])<=3:
        #             # pin the first one
        #             pin_matrix[index_i,index_i]=1
                
        #         # else, we have enough to do betweenness
        #         else:         
        #             # pull out the states for this component 
        #             states_i = states_q[:,components[i]]
        #             # build a graph within this component (look slighly outside lattice range)
        #             G = grph.build_graph(states_i,rg+0.1) 
        #             # find the max influencer
        #             B = grph.betweenness(G)
        #             index_ii = max(B, key=B.get)
        #             #index_ii = min(B, key=B.get)
        #             index_i = components[i][index_ii]
        #             # pin the max influencers
        #             pin_matrix[index_i,index_i] = 1

        # else:
            
        #     for i in range(0,len(components)):
        #         # just take the first in the component for now
        #         index = components[i][0]
        #         # note: later, optimize this selection (i.e. instead of [0], use Grammian)
        #         pin_matrix[index,index]=1

        # return pin_matrix, components    
        


'''
# MORE TEMPLATES


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

'''

    
    



