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

escort          = 1     # escort (i.e. target tracking?): 0 = no, 1 = yes
cd_1            = 0.3   # cohesion weight
cd_2            = 0.4   # alignment weight
cd_3            = 0.2   # separation weight
cd_track        = 0.1   # nominally, zero, unless escorting, then ensure >0
#cd_4           = 0     # navigation (Note: will get modified below, depending on case)
maxu            = 10    # max input (per rule)  note: dynamics *.evolve_sat must be used for constraints
maxv            = 100   # max v                 note: dynamics *.evolve_sat must be used for constraints

# recovery actions 
recovery        = 0     # recover if far away (0 = no, 1 = yes)
far_away        = 300   # recover how far away (i.e. when to go back to centroid)?

# force minimum number of agents for cohesion?
mode_min_coh    = 0     # enforce min # of agents (0 = no, 1 = yes)
agents_min_coh  = 2     # min number of agents

# key ranges  
r               = 10    # range at which neighbours can be sensed  
r_prime         = 10     # range at which obstacles can be sensed


# Some useful functions
# ---------------------

def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out

def order(states_q):
    distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
    
    # to find the radius that includes min number of agents
    if mode_min_coh == 1:
        slide = 0
        for k_node in range(states_q.shape[1]):
            #slide += 1
            for k_neigh in range(slide,states_q.shape[1]):
                if k_node != k_neigh:
                    distances[k_node,k_neigh] = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
    return distances 

# Compute commands
# ----------------

def compute_cmd(targets, centroid, states_q, states_p, k_node, distances):

    # Reynolds Flocking
    # ------------------ 
    
    #initialize commands 
    u_coh = np.zeros((3,states_q.shape[1]))  # cohesion
    u_ali = np.zeros((3,states_q.shape[1]))  # alignment
    u_sep = np.zeros((3,states_q.shape[1]))  # separation
    u_nav = np.zeros((3,states_q.shape[1]))  # navigation
    #distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
    cmd_i = np.zeros((3,states_q.shape[1])) 
       
    #initialize for this node
    temp_total = 0
    temp_total_prime = 0
    temp_total_coh = 0
    sum_poses = np.zeros((3))
    sum_velos = np.zeros((3))
    sum_obs = np.zeros((3))
    
    # adjust cohesion range for min number of agents 
    if mode_min_coh == 1:
        
        # make sure the number of vehicles is bigger than the min number of agents 
        if distances.shape[0] < agents_min_coh+2:
            raise Exception('There are an insufficient number of agents for the cohesion mode selected. Minimum number of agents for mode ',agents_min_coh ,' is ', agents_min_coh+2, ' and you have selected ', distances.shape[0] )
    
        r_coh = 0
        # pull agent distantces for this node
        node_ranges = distances[k_node,:]
        # sort the distances
        node_ranges_sorted = np.sort(node_ranges)
        # take the distance of the farthest agent that satisfies min count
        r_coh_temp = node_ranges_sorted[agents_min_coh+1]
        r_coh = r_coh_temp
    else:
        # else, just rely on default range
        r_coh = r
          
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
        # except for itself (duh):
        if k_node != k_neigh:
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
            
            if dist < 0.1:
                # print out any collisions 
                print('collision at agent: ', k_node)
                continue
    
            # if agent is within the alignment range
            if dist < np.maximum(r,r_coh):
     
                # count
                temp_total += 1                        
     
                # sum 
                sum_velos += states_p[:,k_neigh]
    
            # if within cohesion range 
            if dist < np.maximum(r,r_coh):
                
                #count
                temp_total_coh += 1
                
                #sum
                sum_poses += states_q[:,k_neigh]
    
            # if within the separation range 
            if dist < r_prime:
                
                # count
                temp_total_prime += 1
                
                # sum of obstacles 
                sum_obs += -(states_q[:,k_node]-states_q[:,k_neigh])/(dist**2)                        
    
    # norms
    # -----
    norm_coh = np.linalg.norm(sum_poses)
    norm_ali = np.linalg.norm(sum_velos)
    norm_sep = np.linalg.norm(sum_obs)
      
    if temp_total != 0:
        
        # Cohesion
        # --------
        if norm_coh != 0:
            temp_u_coh = (maxv*np.divide(((np.divide(sum_poses,temp_total_coh) - states_q[:,k_node])),norm_coh)-states_p[:,k_node])
            u_coh[:,k_node] = cd_1*norm_sat(temp_u_coh,maxu)
        
        # Alignment
        # ---------
        if norm_ali != 0:                 
            temp_u_ali = (maxv*np.divide((np.divide(sum_velos,temp_total)),norm_ali)-states_p[:,k_node])
            u_ali[:,k_node] = cd_2*norm_sat(temp_u_ali,maxu)
    
    if temp_total_prime != 0 and norm_sep != 0:
            
        # Separtion
        # ---------
        temp_u_sep = (maxv*np.divide(((np.divide(sum_obs,temp_total_prime))),norm_sep)-states_p[:,k_node]) 
        u_sep[:,k_node] = -cd_3*norm_sat(temp_u_sep,maxu)
                
    # Tracking
    # -------- 
    cd_4 = cd_track
           
    # if doing recovery
    if recovery == 1:
        # and if far away, adjust gain to drive back
        if np.linalg.norm(centroid.transpose()-states_q[:,k_node]) > far_away:
            cd_4 = 0.3 # overides set value, so recovery is always a low gain
        
    # if escorting, track the target (overrides recovery actions)
    if escort == 1:
        cd_4 = cd_track
        if cd_4 == 0:
            print('WARNING: no gain set for tracking target, please set a gain > 0')
        temp_u_nav = (targets[:,k_node]-states_q[:,k_node])
    else:
        temp_u_nav = (centroid.transpose()-states_q[:,k_node])
    
    # compute tracking 
    u_nav[:,k_node] = cd_4*norm_sat(temp_u_nav,maxu)
    
    # compute consolidated commands
    cmd_i[:,k_node] = u_coh[:,k_node] + u_ali[:,k_node] + u_sep[:,k_node] + u_nav[:,k_node] 
    
    return cmd_i[:,k_node]
  