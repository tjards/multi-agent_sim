#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:58:14 2024

this project implements the following:

[1] Yi Dong, Jie Huang,
Flocking with connectivity preservation of multiple double integrator systems \
subject to external disturbances by a distributed control law, Automatica
2015

[2] Zhang et al., Flocking Control Against Malicious Agent
IEEE TRANSACTIONS ON AUTOMATIC CONTROL, VOL. 69, NO. 5, MAY 2024



@author: tjards
"""

# import stuff
# -----------
import numpy as np

# parameters
# ---------- 
d =  5
r = np.divide(2*d, np.sqrt(2)) # sensor range (adjust this later, derived from desired separation now)

# must be computed using Eqn (15) from [1]:
Q = 0.01 


gain_p = 0.01
gain_v = 0.1

def return_ranges():
    return d

# compute commands 
# ----------------
def compute_cmd(targets, states_q, states_p, k_node, **kwargs):
    
    # extract adjacency matrix
    A = kwargs['A']
        
    # initialize
    cmd_i = np.zeros((3))
    # compute navigation
    cmd_i -= compute_navigation(states_q, states_p, targets, k_node)
    
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
        
        # except for itself:
        if k_node != k_neigh:
            
            # check if the neighbour is in range
            if A[k_node,k_neigh] == 0:
                in_range = False
            else:
                in_range = True 
            
            # if within range
            if in_range:
                
                # compute alignment 
                cmd_i -= compute_alignment(states_q, states_p, A, k_node, k_neigh)
                # compute cohesion
                cmd_i -= compute_cohesion(states_q, k_node, k_neigh)
                # compute repulsion
                cmd_i -= compute_repulsion(states_q, k_node, k_neigh)
    
  
    return cmd_i

# compute alignment command
# -------------------------
def compute_alignment(states_q, states_p, A, k_node, k_neigh):
             
    # add the contribution from this agent
    u_i_align = (states_p[:,k_node] - states_p[:,k_neigh])

    return u_i_align
    
# compute cohesion command
# ------------------------
def compute_cohesion(states_q, k_node, k_neigh):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
    u_i_cohes = np.divide(2*s*(r**2 + (r**2)/Q), np.square(r**2 - s**2 + (r**2)/Q ))
    
    return u_i_cohes
    
# compute repulsion command
# ------------------------
def compute_repulsion(states_q, k_node, k_neigh):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
    u_i_repul = np.divide(-2*s*(r**2 + (r**2)/Q), np.square(s**2 + (r**2)/Q ))
    
    return u_i_repul
    
# compute navigation command
# ------------------------
def compute_navigation(states_q, states_p, targets, k_node):
    
    u_i_navig = np.zeros((3))
    u_i_navig = -gain_p*(targets[:,k_node] - states_q[:,k_node]) + gain_v*(states_p[:,k_node])
    
    
    return u_i_navig
