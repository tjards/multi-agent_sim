#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:12:59 2021

@author: tjards

This program implements Reynolds Rules of Flocking ("boids")

"""

import numpy as np

# Hyperparameters
# ----------------

v_o         = 10            # cruise speed
m           = 0.08          # agent mass (could be different per)
tau         = 0.2           # relaxation time (tunable)
del_u       = 0.1           # reaction time (to new neighbours)
s           = 0.1*del_u     # interpolation factor
#R_i        = 5             # interaction radius (initialize)
R_max       = 100           # interation radius, max
#n_i        = 0             # topical range count (initialize)
n_c         = 6.5           # "topical range" (i.e. min number of agents to pay attention to)
r_sep       = 10 #4         # separation radius
r_h         = 0.2           # hard sphere (ignore things too close for cohesion)
r_roost     = 50            # radius of roost
w_s         = 1             # weighting factor separation force 
w_c         = 0.7           # weighting factor for cohesion
w_a         = 0.2           # weighting factor for alignment
w_roost_h   = 0.2           # weighting factor for horizontal attraction to roost
w_roost_v   = 0.1           # weighting factor for vertical attraction to roost
w_rand      = 0.05          # default: 0.01, weight factor of random disturbances
C_c         = 0.35          # critical centrality below which member is interior to roost
alpha       = 0.5           # default: 0.5, between 0 and 1. modulates how tightly swarms converges into target. 0 is very loose, 1 is very tight 
eps         = 0.00001       # to stop divides by zero

sigma       = np.sqrt(np.divide(np.square(r_sep-r_h),4.60517)) #std dev of the gaussion set, such that at that separation zone, near zero
sigma_sqr   = np.square(sigma)

# Some useful functions
# ---------------------

# computes a unit vector in the direction of the agent velo (i.e. forward)
def unit_vector_fwd(velos):
    vector_out = np.divide(velos,np.linalg.norm(velos))
    return vector_out     # output is a unit vector
    
# brings agent back to cruised speed, v_0 after deviating
def to_cruise(m, tau, v_o, v_i, e_x):
    f_out = np.divide(m,tau)*(v_o-v_i)*e_x
    return f_out  # output is a force 
    
# topical interaction distance 
def update_interaction(s,R_i,R_max,n_i,n_c):
    R_new = (1-s)*R_i+s*(R_max-R_max*np.divide(n_i,n_c))
    return R_new
    
# gaussian set used for separation term
def gaussian_set(d_ij,r_h, sigma):
    if d_ij <= r_h:
        return 1
    else:
        return np.exp(-np.divide(np.square(d_ij-r_h),sigma_sqr))
    
# Compute commands for Starling Flocking
# --------------------------------------

# this is run for each node
def compute_cmd(targets, centroid, states_q, states_p, k_node, params, Ts):

    #initialize commands 
    # ------------------
    
    u_coh = np.zeros((3,1))     # cohesion
    f_coh = np.zeros((1,3))
    u_ali = np.zeros((3,1))     # alignment
    f_ali = np.zeros((1,3))
    f_ali = -unit_vector_fwd(states_p[:,k_node])
    u_sep = np.zeros((3,1))     # separation
    f_sep = np.zeros((1,3))
    u_roost_h = np.zeros((3,1)) # roosting (horizontal)
    f_roost_h = np.zeros((1,3))
    u_roost_v = np.zeros((3,1)) # roosting (vertical)
    f_roost_v = np.zeros((1,3)) 
    #u_nav = np.zeros((3,1))    # navigation (not req'd)
    cmd_i = np.zeros((3,1))     # consolidates all commands
    
    # import parameters
    # ----------------------
 
    params_i = np.zeros((1,4))
    if params[0,0] == 0:
        R_i =   5               # interaction radius (initialize)
        params[0,:] = R_i       # for the first time
    params_i = params[:,k_node]
    R_i = params_i[0]           # interaction range (previous)
    n_i = params_i[1]           # number of agents in range (previous)
    n_counter = 0               # number of agents in range (initialize for this time)
    params_i[2] += 1            # counter to update range (slower than sample time)
    
    # centrality
    C_i = params_i[3]
    n_counter_centrality = 0 # number of agents in range for measuring centrality, nominally 2xR_i (initialize for this time)

    # SOCIAL BEHAVIOURS
    # =================

    # update interaction range, if it's time
    # --------------------------------------
    if params_i[2] >= round(del_u/Ts,0)+1:
        # expand the range
        R_i = update_interaction(s,params_i[0],R_max,params_i[1],n_c)
        # reset the counter
        params_i[2] = 0
        
    # search through each neighbour
    # -----------------------------
    for k_neigh in range(states_q.shape[1]):
        
        # except for itself
        if k_node != k_neigh:
             
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_neigh]-states_q[:,k_node])
            #blind spot?
            
            # if the neighbour is within the range for computing centrality
            if dist <= 2*R_i: #yes, centrality is measured in a bigger range
            
                # increment the counter
                n_counter_centrality += 1
                
                # compute centrality
                C_i += np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist)
            
                        
            # if the neighhour is within the interaction range (add blind spot later)
            if dist <= R_i and n_counter < n_c:
                
                # increment the counter
                n_counter += 1
                
                # compute the separation force
                # ----------------------------
                f_sep += gaussian_set(dist, r_h, sigma)*np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist) 
            
                # compute the cohesion force 
                # -------------------------- 
                if dist > r_h:
                    f_coh += np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist)
                    
                # compute the alignment force
                # ---------------------------
                f_ali += unit_vector_fwd(states_p[:,k_neigh])
                          
    # compute consolidated commands 
    if n_counter_centrality > 0:
        # update the centrality for this node
        C_i = np.divide(np.linalg.norm(C_i),n_counter_centrality)
        # save 
        params[3,k_node] = C_i
        
    if n_counter > 0:
        u_sep = -m*np.divide(w_s,n_counter)*f_sep.reshape((3,1))
        u_coh = m*C_i*np.divide(w_c,n_counter)*f_coh.reshape((3,1))
        u_ali = m*w_a*np.divide(f_ali,np.linalg.norm(f_ali)).reshape((3,1))
        
        
    # ROOSTING BEHAVIOURS
    # ===================
    
    # define the vertical direction
    unit_vector_ver = np.array([0,0,1])

    # find the vector for the forward direction (2D)
    unit_fwd_2D = unit_vector_fwd(states_p[0:2,k_node])
    
    # rotate it 90 degrees inwards (counterclockwise first)
    unit_bank_2D = np.array([-unit_fwd_2D[1],unit_fwd_2D[0]])
    
    # adjust the direction to be towards target
    # if moves farther from target
    if np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))  +  unit_bank_2D.reshape((1,2))) >= np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))):
        # switch the sign (clockwise rotation)
         unit_bank_2D = np.array([unit_fwd_2D[1],-unit_fwd_2D[0]])
    
    # compute unit vector towards target
    #unit_to_target = np.divide(targets[0:3,k_node].reshape((1,3)) - states_q[0:3,k_node].reshape((1,3)),np.linalg.norm(targets[0:3,k_node].reshape((1,3)) - states_q[0:3,k_node].reshape((1,3))))
    unit_to_target_2D = np.divide(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2)),np.linalg.norm(targets[0:2,k_node].reshape((1,2)) - states_q[0:2,k_node].reshape((1,2))))
    
    # compute dot product of fwd and target direction (measure of how much it is pointing inwards)
    #proj_to_target = np.dot(unit_vector_fwd(states_p[0:3,k_node]).reshape((1,3)), -unit_to_target[:,0:3].reshape((3,1))).ravel()   
    #proj_to_target = np.dot(unit_vector_fwd(states_p[0:2,k_node]).reshape((1,2)), -unit_to_target[:,0:2].reshape((2,1))).ravel()
    #proj_to_target_2D = np.dot(unit_vector_fwd(states_p[0:2,k_node]).reshape((1,2)), -unit_to_target[:,0:2].reshape((2,1))).ravel() 
    proj_to_target_2D = np.dot(unit_vector_fwd(states_p[0:2,k_node]).reshape((1,2)), -unit_to_target_2D[:,0:2].reshape((2,1))).ravel() 

    # compute the horizontal roosting acceleration
    sign = np.sign(proj_to_target_2D)
    f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(sign*alpha + sign*(1-alpha)*proj_to_target_2D)
    u_roost_h = -m*w_roost_h*f_roost_h.reshape((3,1)) 

    # compute the vertical roosting acceleration
    f_roost_v =  (targets[2,k_node] - states_q[2,k_node])*unit_vector_ver
    u_roost_v = m*w_roost_v*f_roost_v.reshape((3,1))
    
    # RANDOM VECTOR
    # ==============
    
    np.random.seed(k_node)  # each node had its own unique noise (bias)
    u_rand = m*w_rand*2*(np.random.rand(3,1)-0.5) # random is between -1 and 1
    
    # CONSOLIDATION
    # =============
        
    params[0,k_node] = R_i
    params[1,k_node] = n_counter
    params[2,k_node] = params_i[2]
    params[3,k_node] = C_i
    
    cmd_i = u_coh + u_ali + u_sep + u_roost_h + u_roost_v + u_rand
    
    return cmd_i.ravel(), params
  

  
    