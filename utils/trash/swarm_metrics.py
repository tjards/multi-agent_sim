#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This program computes some useful swarm metrics

@author: tjards

"""

import numpy as np
from scipy.spatial.distance import cdist

# order
# -----
def order(states_p):

    order = 0
    N = states_p.shape[1]
    # if more than 1 agent
    if N > 1:
        # for each vehicle/node in the network
        for k_node in range(states_p.shape[1]):
            # inspect each neighbour
            for k_neigh in range(states_p.shape[1]):
                # except for itself
                if k_node != k_neigh:
                    # and start summing the order quantity
                    norm_i = np.linalg.norm(states_p[:,k_node])
                    if norm_i != 0:
                        order += np.divide(np.dot(states_p[:,k_node],states_p[:,k_neigh]),norm_i**2)
            # average
            order = np.divide(order,N*(N-1))
            
    return order

# separation
# ----------
def separation(states_q,target_q,obstacles):
    
    # distance from targets or agents
    # ---------------------
    # note: replace target_q with states_q to get separation between agents
    #seps=cdist(states_q.transpose(), np.reshape(target_q[:,0],(-1,1)).transpose())
    seps=cdist(states_q.transpose(), states_q.transpose())    
    vals = np.unique(seps[np.where(seps!=0)])
    means = np.mean(vals)
    varis = np.var(vals)
    maxes = np.max(vals)
    mines = np.min(vals)
    
    # distance from obstacles
    # -----------------------
    if obstacles.shape[1] != 0:
        seps_obs=cdist(states_q.transpose(), obstacles[0:3,:].transpose()) - obstacles[3,:] # this last part is the radius of the obstacle
        means_obs = np.mean(seps_obs) 
        varis_obs = np.var(seps_obs)
    else:
        means_obs = 0
        varis_obs = 0
    
    return means, varis, means_obs, varis_obs, maxes, mines
    
def centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    sum_z = np.sum(points[:, 2])
    centroid = np.array((sum_x/length, sum_y/length, sum_z/length), ndmin = 2)
    return centroid.transpose() 

# spacing (between agents)
# -----------------------
def spacing(states_q):
    
    # visibility radius
    radius = 1.5*5
    
    seps=cdist(states_q.transpose(), states_q.transpose())    
    vals = np.unique(seps[np.where(seps!=0)])
    vals_t = vals # even those out of range
    vals = np.unique(vals[np.where(vals<radius)])
    
    # if empty, return zero
    if len(vals) == 0:
        vals = np.array([0])
    
    return vals.mean(), len(vals), vals_t.mean()
    
# energy
# ------
def energy(cmd):
    
    energy_total = np.sqrt(np.sum(cmd**2))
    energy_var =  np.var(np.sqrt((cmd**2)))
       
    return energy_total, energy_var

    
    