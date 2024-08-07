#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:58:14 2024

this project implements the 

Zhang et al., Flocking Control Against Malicious Agent
IEEE TRANSACTIONS ON AUTOMATIC CONTROL, VOL. 69, NO. 5, MAY 2024

@author: tjards
"""

# import stuff
# -----------
import numpy as np

# parameters
# ---------- 
d       =  5 
def return_ranges():
    return d

# compute commands 
# ----------------
def compute_cmd(targets, states_q, states_p, k_node, **kwargs):
    
    # extract adjacency matrix
    A = kwargs['A']
    
    # compute commands
    cmd_i = compute_alignment(states_q, states_p, A, k_node)

    return cmd_i

# compute the alignment command
# -----------------------------
def compute_alignment(states_q, states_p, A, k_node):
    
    # initialize
    u_i_align = np.zeros((3))
    
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
                
                # add the contribution from this agent
                u_i_align -= (states_q[:,k_node] - states_q[:,k_neigh])
    

    return u_i_align
    
    