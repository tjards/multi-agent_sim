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
    - G is simple: (a,a) not in E for all a in V 
    - G is undirected: (a,b) in E <=> (b,a) in E
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

Some default parameters:

    # learning parameters
    hetero_lattice      = 1     # support heterogeneous lattice size? 1 = yes (Consensus), 0 = no

    learning            = 0     # requires heterolattice, do we want to learn lattice size? 1 = yes (QL), 0 = no
    learning_grid_size  = -1    # grid size for learning (nominally, -1, for 10 units x 10 units)

    # future learning parameters default to zero for now
    #hetero_gradient     = 0     # (this doesn't work) supports heterogeneous potential functions

    # define the method for lattice formation
    flocking_method = 'lennard_jones'
    #flocking_options = ['default','morse','lennard_jones','gromacs_soft_core']   
    flocking_options = ['default','lennard_jones'] # only saber works for hetero, need to feed a/b updates in for lenard
    
                    # 'default'  = Olfati-saber flocking
                    # 'morse'
                    # 'lennard_jones'
                    # 'gromacs_soft_core'
                    # 'mixed' - randomly mix these

@author: tjards

"""

#%% import stuff
# --------------
import numpy as np
import random
import os
import json
#import random 
import copy
import config.config as cfg

from planner.techniques.pinning_gradients_default import velocity_alignment as alignment_term
from planner.techniques.pinning_gradients_default  import navigation as navigation_term
from planner.techniques.pinning_gradients_default  import compute_cmd_b as obstacle_term
from planner.techniques.pinning_gradients_default  import gradient as cohesion_term_default
from planner.techniques import pinning_gradients_others

#%% control systems functions
# -------------------------

from planner.base import BasePlanner
class Planner(BasePlanner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
  

        pinning_config =cfg.get_config(config, 'planner.techniques.pinning_lattice')

        self.hetero_lattice     = pinning_config.get('hetero_lattice', 0)
        self.learning           = pinning_config.get('learning', 0)
        self.learning_grid_size = pinning_config.get('learning_grid_size', -1)
        self.flocking_method    = pinning_config.get('flocking_method', 'lennard_jones')
        #self.r                  = pinning_config.get('r', 10)
        self.r_max              = pinning_config.get('r_max', 13)
        #self.d                  = pinning_config.get('d', 7)
        self.d_min              = pinning_config.get('d_min', 5)
        self.d_init             = pinning_config.get('d', 7)
        self.d_prime            = pinning_config.get('d_prime_ratio', 0.6)*self.d_init            
        self.r_prime            = pinning_config.get('r_prime_ratio', 1.3)*self.d_prime

        # default gains
        self.c1_a    = pinning_config.get('c1_a', 1.0) # interaction gain, position
        self.c2_a    = pinning_config.get('c2_a', 2.0) # interaction gain, velocity
        self.c1_b    = pinning_config.get('c1_b', 0.0) # obstacle avoidance gain, position
        self.c2_b    = pinning_config.get('c2_b', 0.0) # obstacle avoidance gain, velocity
        self.c1_g    = pinning_config.get('c1_g', 2.0) # navigation gain, position
        self.c2_g    = pinning_config.get('c2_g', 4.472) # navigation gain, velocity
 
        # learning requires heterolattice
        if self.learning == 1 and self.hetero_lattice != 1:
            print('Warning: learning lattice requires hetero lattice enabled to find local consensus. Enforcing.')
            self.hetero_lattice = 1

        # extract agents config 
        agents_config = cfg.get_config(config, 'agents')
        self.nAgents = agents_config.get('nAgents', None)

        # build a list of gradient options
        flocking_options = ['default','morse','lennard_jones','gromacs_soft_core']  # note: default is olfati-saber

        gradients_config = cfg.get_config(config, 'planner.techniques.gradients')
        gradient_funcs = pinning_gradients_others.create_gradient_functions(gradients_config)


        #from planner.techniques.gradient_tools import grad_morse_gradient as cohesion_term_mor
        #from planner.techniques.gradient_tools import grad_lennard_jones as cohesion_term_len
        #from planner.techniques.gradient_tools import  grad_gromacs_soft_core as cohesion_term_gro

        self.cohesion_list = {}
        self.cohesion_list['default'] = cohesion_term_default
        #self.cohesion_list['morse'] = cohesion_term_mor
        #self.cohesion_list['lennard_jones'] = cohesion_term_len
        #self.cohesion_list['gromacs_soft_core'] = cohesion_term_gro
        self.cohesion_list['morse'] = gradient_funcs['morse']
        self.cohesion_list['lennard_jones'] = gradient_funcs['lennard_jones']
        self.cohesion_list['gromacs_soft_core'] = gradient_funcs['gromacs_soft_core']

        # build out a random list of gradients (for mixed case)
        self.term_selected = None
        if self.flocking_method == 'mixed':
            term_indices = np.random.randint(0, len(flocking_options), size=(1, self.nAgents))
            self.term_selected = [flocking_options[i] for i in term_indices.flatten()]
            print(f"Mixed flocking assignments: {self.term_selected}")

        # graph parameters (standardized in base class)
        self.sensor_range_matrix = self.r_max * np.ones((self.nAgents, self.nAgents))
        self.connection_range_matrix = self.d_init * np.ones((self.nAgents, self.nAgents))
        
    # form the lattice
    # -----------------
    # NOTE: This method is intentionally NOT vectorized. The consensus learning
    # update (consensus_agent.update) modifies d_weighted[k_node, k_neigh]
    # during neighbor iteration, creating sequential dependencies between pairs
    # for the same agent. Vectorizing the force computation separately from
    # learning would alter the lattice evolution dynamics. The optimization here
    # is limited to accepting pre-built neighbor lists to skip the O(n) scan.
    def compute_cmd_a(self,states_q, states_p, targets, targets_v, k_node, reward_values, **kwargs):   
        
        # pull out the args (try .get() to ignore n/a cases)
        # -----------------
        headings                = kwargs.get('quads_headings')
        consensus_agent         = kwargs.get('consensus_lattice')
        learning_agent          = kwargs.get('learning_lattice')
        directional             = kwargs.get('directional_graph')
        A                       = self.interaction_graph
        neighbors               = kwargs.get('neighbors')  # pre-built from SpatialIndex
        
        # safety checks
        if directional:
            if 'consensus_lattice' in kwargs:
                consensus_agent.headings = headings
        
        if consensus_agent is not None and consensus_agent.d_weighted.shape[1] != states_q.shape[1]:
            raise ValueError("Error! There are ", states_q.shape[1], 'agents, but ', consensus_agent.d_weighted.shape[1], 'lattice parameters')
        
        # learning process (if applicable)
        if self.learning == 1: 
            kwargs['learning_grid_size'] = self.learning_grid_size
            learning_agent.update_step(reward_values, targets, states_q, states_p, k_node, consensus_agent, **kwargs)
    
        # initialize parameters
        if self.hetero_lattice == 1:
            d = consensus_agent.d_weighted[k_node, k_node]
        else:
            d = self.d_init
        
        # (3,) not (3, n) — heap fragmentation fix (see OPTIMIZATION.md)
        u_int = np.zeros(3)

        # determine neighbor set: use pre-built list if available, else scan
        if neighbors is not None:
            neigh_iter = neighbors
        else:
            neigh_iter = [j for j in range(states_q.shape[1]) if j != k_node and A[k_node, j] != 0]

        # iterate over neighbors (sequential: learning updates depend on prior pairs)
        for k_neigh in neigh_iter:
                            
            # pull the lattice parameter for this node/neighbour pair
            if self.hetero_lattice == 1:
                d = consensus_agent.d_weighted[k_node, k_neigh]                     
            
            # compute interaction forces
            if self.flocking_method == 'mixed':
                u_int += self.cohesion_list[self.term_selected[k_node]](self.c1_a,states_q, k_node, k_neigh, self.r_max, d)
            else:
                u_int += self.cohesion_list[self.flocking_method](self.c1_a,states_q, k_node, k_neigh, self.r_max, d)

            u_int += alignment_term(self.c2_a, states_q, states_p, k_node, k_neigh, self.r_max, d)
            
            # seek consensus (sequential: modifies d_weighted for subsequent pairs)
            if self.hetero_lattice == 1:
                consensus_agent.update(k_node, k_neigh, states_q)
                consensus_agent.prox_i[k_node, k_neigh] = 1

        # mark out-of-range neighbors (for consensus tracking)
        if self.hetero_lattice == 1 and neighbors is not None:
            neigh_set = set(neighbors)
            for j in range(states_q.shape[1]):
                if j != k_node and j not in neigh_set:
                    consensus_agent.prox_i[k_node, j] = 0
            
        return u_int

    # avoid obstacles
    # ---------------
    def compute_cmd_b(self, states_q, states_p, obstacles, walls, k_node):
        
        return obstacle_term(self.c1_b, self.c2_b, states_q, states_p, obstacles, walls, k_node, self.d_prime, self.r_prime)

    # track the target
    # -----------------
    def compute_cmd_g(self, states_q, states_p, targets, targets_v, k_node, pin_matrix):

        return pin_matrix[k_node,k_node]*navigation_term(self.c1_g, self.c2_g, states_q, states_p, targets, targets_v, k_node)

    # consolidate control signals
    # ---------------------------
    #def compute_cmd(self, centroid, states_q, states_p, obstacles, walls, targets, targets_v, k_node, **kwargs):
    def compute_cmd(self, states, targets, index, **kwargs):

        # Extract from states
        states_q = states[0:3, :]      # positions
        states_p = states[3:6, :]      # velocities
        targets_q = targets[0:3, :]    # target positions
        targets_v = targets[3:6, :]    # target velocities
        obstacles = kwargs.get('obstacles_plus')
        walls = kwargs.get('walls')
        k_node = index

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
        
        u_int = self.compute_cmd_a(states_q, states_p, targets_q, targets_v, k_node, reward_values, **kwargs)
        u_obs = self.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
        u_nav = self.compute_cmd_g(states_q, states_p, targets_q, targets_v, k_node, self.pin_assignments)
        
        return u_int + u_obs + u_nav

