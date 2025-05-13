#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module implements pinning control for flocking as a lattice
It is also RL-enabled, adjusting lattice scale to optimize on some user-defined objective
Default objective is to maximize k-connectivity

Preliminaries:
    - Let us consider V nodes (vertices, agents)
    - Define E is a set of edges (links) as the set of ordered pairs
    from the Cartesian Product V x V, E = {(a,b) | a /in V and b /in V}
    - Then we consider Graph, G = {V,E} (nodes and edges)
    - G is simple: (a,a) not \in E \forall a \in V 
    - G is undirected: (a,b) \in E <=> (b,a) \in E
    - Nodes i,j are neighbours if they share an edge, (i,j) /in E
    - d1=|N_1| is the degree of Node 1, or, the number of neighbours

# Pinning control is structured as follows:
    
    u =     interaction (a)            + obstacle (b) + target (g)
    u = {cohesion_term + alignment_term} + obstacle_term + navigation term

Created on Tue Dec 20 13:32:11 2022

Some related work:
    
    https://arxiv.org/pdf/1611.06485.pdf
    http://kth.diva-portal.org/smash/get/diva2:681041/FULLTEXT01.pdf
    https://ieeexplore-ieee-org.proxy.queensu.ca/stamp/stamp.jsp?tp=&arnumber=6762966
    https://ieeexplore-ieee-org.proxy.queensu.ca/document/9275901

Dev notes:
    
    24 Nov 25 - gradient estimation is working, but now I wantto feed these back to 
    pin, so it can stop the motion (i.e., pin adds up all differences and forces stop).
    The pin will then be able to bring everyone back.
    so, sum of: gradient_agent.gain_gradient_control * gradient_agent.C_filtered[0:gradient_agent.dimens, k_node, k_neigh]
    
    23 Feb 25 - now supports constrained gradient-based lattice consensus seeking 

@author: tjards

"""

#%% import stuff
# --------------
import numpy as np
import random
import os
import json
import random 
import copy

from config.configs_tools import update_configs  

#%% simulation setup
# ------------------

# learning parameters
hetero_lattice      = 1     # support heterogeneous lattice size? 1 = yes (Consensus), 0 = no
learning            = 0     # requires heterolattice, do we want to learn lattice size? 1 = yes (QL), 0 = no
learning_grid_size  = -1    # grid size for learning (nominally, -1, for 10 units x 10 units)

# future learning parameters default to zero for now
hetero_gradient     = 0     # (this doesn't work) supports heterogeneous potential functions

# define the method for lattice formation
flocking_method = 'lennard_jones'
#flocking_options = ['saber','morse','lennard_jones','gromacs_soft_core']   
flocking_options = ['saber','lennard_jones'] # only saber works for hetero, need to feed a/b updates in for lenard
 
                # 'saber'  = Olfati-saber flocking
                # 'morse'
                # 'lennard_jones'
                # 'gromacs_soft_core'
                # 'mixed' - randomly mix these
      
# variable imports
# ----------------
r       = None
d       = None
d_min   = 5         # default 5

# import olfat-saber stuff as baseline
from planner.techniques.saber_tools import velocity_alignment as alignment_term
from planner.techniques.saber_tools import navigation as navigation_term
from planner.techniques.saber_tools import compute_cmd_b as obstacle_term
from planner.techniques.saber_tools import r
from planner.techniques.saber_tools import d

# import the gradients
from planner.techniques.saber_tools import gradient as cohesion_term_sab
from planner.techniques.gradient_tools import grad_morse_gradient as cohesion_term_mor
from planner.techniques.gradient_tools import grad_lennard_jones as cohesion_term_len
from planner.techniques.gradient_tools import  grad_gromacs_soft_core as cohesion_term_gro

# build a list of gradient options    
cohesion_list = {}
cohesion_list['saber'] = cohesion_term_sab
cohesion_list['morse'] = cohesion_term_mor
cohesion_list['lennard_jones'] = cohesion_term_len
cohesion_list['gromacs_soft_core'] = cohesion_term_gro

# build out a random list of gradients (for mixed case)
if flocking_method == 'mixed':
    with open(os.path.join("config", "config_agents.json"), 'r') as agent_configs:
        agent_config = json.load(agent_configs)
        nAgents = agent_config['nAgents']
        term_indices = np.random.randint(0, len(flocking_options), size=(1, nAgents))
        term_selected = [flocking_options[i] for i in term_indices.flatten()]
        print(term_indices)
   
# learning requires heterolattice
if learning == 1 and hetero_lattice != 1:
    print('Warning: learning lattice requires hetero lattice enabled to find local consensus. Enforcing.')
    hetero_lattice = 1

#%% Hyperparameters
# -----------------

#r      = imported above
#d      = imported above
#d_min   = 5         # default 5
d_init  = d         # default d (don't mess with this)

#%% save configs
# --------------

def update_pinning_configs():
    
    configs_entries = [
    ('hetero_gradient', hetero_gradient),
    ('hetero_lattice', hetero_lattice),
    ('learning', learning),
    ('d', d),
    ('d_min', d_min),
    ('r_max', r),
    ('flocking_method', flocking_method)
    ]
    
    update_configs('pinning',  configs_entries)
    
# for externals
# -------------
def return_lattice_param():
    return d_init

def return_ranges():
    return r

#%% control systems functions
# -------------------------

# form the lattice
# -----------------
def compute_cmd_a(states_q, states_p, targets, targets_v, k_node, reward_values, **kwargs):   
    
    # pull out the args (try .get() to ignore n/a cases)
    # -----------------
    headings                = kwargs.get('quads_headings')
    consensus_agent         = kwargs.get('consensus_lattice') # rename this object 
    learning_agent          = kwargs.get('learning_lattice')
    gradient_agent          = kwargs.get('estimator_gradients')
    directional             = kwargs.get('directional_graph')
    A                       = kwargs.get('A')
    local_k_connectivity    = kwargs.get('local_k_connectivity')
    pin_matrix              = kwargs.get('pin_matrix')
    
    # safety checks
    # -------------
    # when using directional 
    if directional:
        #if headings is None:
        #    headings = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        if 'consensus_lattice' in kwargs:
            consensus_agent.headings = headings
    
    # ensure the parameters match the agents
    if consensus_agent is not None and consensus_agent.d_weighted.shape[1] != states_q.shape[1]:
        raise ValueError("Error! There are ", states_q.shape[1], 'agents, but ', consensus_agent.d_weighted.shape[1], 'lattice parameters')
    
    # learning processs (if applicable)
    # ---------------------------------    
    # execute the reinforcement learning, local case (if applicable)
    if learning == 1: 
        
        kwargs['learning_grid_size'] = learning_grid_size # consider adapting this with time
        learning_agent.update_step(reward_values, targets, states_q, states_p, k_node, consensus_agent, **kwargs)
     
    # estimate neighbouring gradients (legacy, delete later)
    # -------------------------------
    '''
    if hetero_gradient == 1:
        # compute accelerations
        gradient_agent.observed_gradients[:,k_node,:] = (states_p[0:gradient_agent.dimens,:] - gradient_agent.last_velos[0:gradient_agent.dimens, k_node, :]) #/gradient_agent.Ts
        # load last velo (for next time)
        gradient_agent.last_velos[0:gradient_agent.dimens, k_node, :] = states_p[0:gradient_agent.dimens,:]
        # now update
        observed_gradients = gradient_agent.observed_gradients[:,k_node,:]
        gradient_agent.update_estimates(states_q[0:gradient_agent.dimens,:], states_p[0:gradient_agent.dimens, :], observed_gradients, A, k_node)       
   '''  
   
   # initialize parameters
    # ----------------------
    if hetero_lattice == 1:
        d = consensus_agent.d_weighted[k_node, k_node]
    else:
        d = d_init
    
    u_int = np.zeros((3,states_q.shape[1]))     # interactions

    # search through each neighbour
    # -----------------------------
    for k_neigh in range(states_q.shape[1]):
        
        # except for itself:
        if k_node != k_neigh:
                        
            # pull the lattice parameter for this node/neighbour pair
            if hetero_lattice == 1:
                d = consensus_agent.d_weighted[k_node, k_neigh]                     
            
            # check if the neighbour is in range
            # ---------------------------------
            
            # use adjacency matrix (new)
            if A[k_node,k_neigh] == 0:
                in_range = False
            else:
                in_range = True 
            
            # if within range
            # ---------------
            if in_range:
                
                # for the case iof mixed potential functions
                if flocking_method == 'mixed':
                    
                    u_int[:,k_node] += cohesion_list[term_selected[k_node]](states_q, k_node, k_neigh, r, d)
                
                    # here cohesion_term = term_list[term_indices[0][k_node]]
                    #cohesion_term = copy.deepcopy(term_list[term_indices[0][k_node]])
                    #d = d_mixed[k_node,0]
                    
                else:
                    
                    u_int[:,k_node] += cohesion_list[flocking_method](states_q, k_node, k_neigh, r, d)
                    

                # legacy (delete later)
                '''
                # we will grab this cohesion term for gradient estimation
                if hetero_gradient == 1:
                    #u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gradient_estimates[0:gradient_agent.dimens, k_node, k_neigh]
                    u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gain_gradient_control * gradient_agent.C_filtered[0:gradient_agent.dimens, k_node, k_neigh]
                    
                    # add the addition navigation term for pin_sum
                    u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gain_gradient_control * pin_matrix[k_node,k_node]*gradient_agent.C_sum_bypin[0:gradient_agent.dimens,k_node]
                    
                    
                    # I don't think this is what I'm supposed to bring in
                    #gradient_agent.observed_gradients[0:gradient_agent.dimens,k_node, k_neigh] = u_int[0:gradient_agent.dimens,k_node]
                    
                '''
                
                u_int[:,k_node] += alignment_term(states_q, states_p, k_node, k_neigh, r, d)
                
                # I don't think this is what I'm supposed to bring in
                #gradient_agent.observed_gradients[0:gradient_agent.dimens,k_node, k_neigh] = u_int[0:gradient_agent.dimens,k_node]
                
                #if u_int[2,:].any() != 0:
                #    print('debug needed: 3D cmds in 2D')
                
                # seek consensus
                if hetero_lattice == 1:
                    consensus_agent.update(k_node, k_neigh, states_q)   # update lattice via consensus
                    consensus_agent.prox_i[k_node, k_neigh] = 1         # annotate as in range     
            else:
                if hetero_lattice == 1:
                    consensus_agent.prox_i[k_node, k_neigh] = 0      # annotate as not in range
           
    return u_int[:,k_node] 

# avoid obstacles
# ---------------
def compute_cmd_b(states_q, states_p, obstacles, walls, k_node):
    
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    u_obs[:,k_node] = obstacle_term(states_q, states_p, obstacles, walls, k_node)
    
    return u_obs[:,k_node] 

# track the target
# -----------------
def compute_cmd_g(states_q, states_p, targets, targets_v, k_node, pin_matrix):

    # initialize 
    u_nav = np.zeros((3,states_q.shape[1]))  
    u_nav[:,k_node] = pin_matrix[k_node,k_node]*navigation_term(states_q, states_p, targets, targets_v, k_node)
  
    return u_nav[:,k_node]

# consolidate control signals
# ---------------------------
def compute_cmd(centroid, states_q, states_p, obstacles, walls, targets, targets_v, k_node, **kwargs):
        
    directional         = kwargs.get('directional_graph')
    
    # ensure there are heading available, if needed
    if directional and kwargs.get('quads_headings') is None:
        kwargs['quads_headings'] = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        print('no headings avail, assuming 0')
    
    if 'learning_lattice' in kwargs:
        learning_agent  = kwargs.get('learning_lattice')
        if learning_agent.reward_method == 'landmarks':
            reward_values = obstacles
        elif learning_agent.reward_method == 'connectivity':
            reward_values_full  = kwargs.get('local_k_connectivity')
            reward_values       = reward_values_full[k_node]
    else:
        reward_values = 0
    
    # initialize 
    cmd_i = np.zeros((3,states_q.shape[1]))
    
    u_int = compute_cmd_a(states_q, states_p, targets, targets_v, k_node, reward_values, **kwargs)
    u_obs = compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
    u_nav = compute_cmd_g(states_q, states_p, targets, targets_v, k_node, kwargs.get('pin_matrix'))
       
    cmd_i[:,k_node] = u_int + u_obs + u_nav
    
    #if cmd_i[2,:].any() != 0:
    #    print('debug needed: 3D cmds in 2D')
    
    return cmd_i[:,k_node]


#%% legacy code 
# -------------
"""

#%% Select pins for each component 
# --------------------------------

# select pins randomly
def select_pins_random(states_q):
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1

    return pin_matrix

# select by components
#def select_pins_components(states_q, states_p):
def select_pins_components(states_q, states_p, **kwargs):
    
    # pull out parameters
    headings    = kwargs.get('quads_headings')
    d_weighted  = kwargs.get('d_weighted')
    
    # initialize the pins
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    
    # compute adjacency matrix
    if directional != 1:
        A = grph.adj_matrix(states_q, rg)
        components = grph.find_connected_components_A(A)
        
    else:
        # if no heading avail, set to zero
        if headings is None:
           headings = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        
        #d_weighted  = kwargs['d_weighted']
        #if 'quads_headings' in kwargs:
        #    headings    = kwargs['quads_headings']
        #else:
        #    headings    = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        #A = grph_dir.adj_matrix_bearing(states_q,states_p,paramClass.d_weighted, sensor_aperature, paramClass.headings)
        
        A = grph_dir.adj_matrix_bearing(states_q,states_p,d_weighted, sensor_aperature, headings)
        components = grph_dir.find_one_way_connected_components_deg(A)
    
    # Gramian method
    # --------------
    if method == 'gramian':
        
        # for each component
        for i in range(0,len(components)):
            
            # find the adjacency and degree matrix of this component 
            states_i = states_q[:,components[i]]
            A = grph.adj_matrix(states_i, rg)  # move these outside (efficiency)
            D = grph.deg_matrix(states_i, rg)
            
            index_i = components[i][0]
            
            # if this is a lone agent
            if len(components[i])==1:
                # pin it
                pin_matrix[index_i,index_i]=1
                
            else: 
                # find gramian trace (i.e. energy demand) of first component
                ctrlable, trace_i = grph.compute_gram_trace(A,D,0,A.shape[1])
                # set a default pin
                pin_matrix[index_i,index_i]=1
                # note: add a test for controlability here
    
                # cycle through the remaining agents in the component
                for j in range(1,len(components[i])): 
                    
                    ctrlable, trace = grph.compute_gram_trace(A,D,j,A.shape[1])
                    
                    # take the smallest energy value
                    if trace < trace_i:
                        # make this the new benchmark
                        trace_i = trace
                        # de-pin the previous
                        pin_matrix[index_i,index_i]=0
                        index_i = components[i][j]
                        # pin this one
                        pin_matrix[index_i,index_i]=1
    
    # Degree method
    # -------------
    elif method == 'degree':
        
        # if using direcational mode
        if directional:
            centralities = {}
            # find the degree centrality within each component 
            for i, component in enumerate(components):
                # store the degree centralities 
                centrality      = grph_dir.out_degree_centrality(A.tolist(), component)
                #centrality[i]   = centrality 
                centralities[i]   = np.diag(list(centrality.values()))

        # for each component
        for i in range(0,len(components)):
            
            # find the adjacency and degree matrix of this component 
            states_i = states_q[:,components[i]]

            if directional:
                
                D = centralities[i]

            else:
                
                D = grph.deg_matrix(states_i, rg)
                 
            index_i = components[i][0]
            
            # if this is a lone agent
            if len(components[i])==1:
                # pin it
                pin_matrix[index_i,index_i]=1
                
            else: 
                
                # find index of highest element of Degree matrix
                index_i = components[i][np.argmax(np.diag(D))]
                # set as default pin
                pin_matrix[index_i,index_i]=1
                       
    # Betweenness
    # -----------
    
    # note: for betweenness, we need > 3 agents (source+destination+node)  
    
    elif method == 'between':
        
        # for each component
        for i in range(0,len(components)):
            
            # default to first node
            index_i = components[i][0]
            
            # if fewer than 4 agents in this component
            if len(components[i])<=3:
                # pin the first one
                pin_matrix[index_i,index_i]=1
            
            # else, we have enough to do betweenness
            else:         
                # pull out the states for this component 
                states_i = states_q[:,components[i]]
                # build a graph within this component (look slighly outside lattice range)
                G = grph.build_graph(states_i,rg+0.1) 
                # find the max influencer
                B = grph.betweenness(G)
                index_ii = max(B, key=B.get)
                #index_ii = min(B, key=B.get)
                index_i = components[i][index_ii]
                # pin the max influencers
                pin_matrix[index_i,index_i] = 1

    else:
        
        for i in range(0,len(components)):
            # just take the first in the component for now
            index = components[i][0]
            # note: later, optimize this selection (i.e. instead of [0], use Grammian)
            pin_matrix[index,index]=1

    return pin_matrix, components
    

"""



    