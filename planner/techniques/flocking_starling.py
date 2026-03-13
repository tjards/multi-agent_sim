#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:12:59 2021

@author: tjards

This program implements Starling flocking 

Parameter descriptions:

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

"""
# import stuff
# ------------
import numpy as np
import os
import json
import config.config as cfg

# Hyperparameters
# ----------------




#%save configs
# --------------
'''
from config.configs_tools import update_configs

configs_entries = [
    ('v_o', v_o),
    ('m', m),
    ('tau', tau),
    ('del_u', del_u),
    ('s', s),
    ('R_max', R_max),
    ('n_c', n_c),
    ('r_sep', r_sep),
    ('r_h', r_h),
    ('r_roost', r_roost),
    ('w_s', w_s),
    ('w_c', w_c),
    ('w_a', w_a),
    ('w_roost_h', w_roost_h),
    ('w_roost_v', w_roost_v),
    ('w_rand', w_rand),
    ('C_c', C_c),
    ('alpha', alpha),
    ('eps', eps),
    ('sigma', sigma),
    ('sigma_sqr', sigma_sqr)
]

update_configs('starling', configs_entries)
'''

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
def gaussian_set(d_ij,r_h, sigma_sqr):
    if d_ij <= r_h:
        return 1
    else:
        return np.exp(-np.divide(np.square(d_ij-r_h),sigma_sqr))
    
# custom class
from planner.base import BasePlanner
class Planner(BasePlanner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
 
        # load the configs
        starling_config =cfg.get_config(config, 'planner.techniques.flocking_starling')
        self.v_o       = starling_config['v_o']         # cruise speed
        self.m         = starling_config['m']           # agent mass (could be different per)
        self.tau       = starling_config['tau']         # relaxation time (tunable)
        self.del_u     = starling_config['del_u']       # reaction time (to new neighbours)
        self.s         = starling_config['s']           # interpolation factor
        self.R_max     = starling_config['R_max']       # interation radius, max
        self.n_c       = starling_config['n_c']         # "topical range" (i.e. min number of agents to pay attention to)
        self.r_sep     = starling_config['r_sep']       # separation radius
        self.r_h       = starling_config['r_h']         # hard sphere (ignore things too close for cohesion)
        self.r_roost   = starling_config['r_roost']     # radius of roost
        self.w_s       = starling_config['w_s']         # weighting factor separation force 
        self.w_c       = starling_config['w_c']         # weighting factor for cohesion
        self.w_a       = starling_config['w_a']         # weighting factor for alignment
        self.w_roost_h = starling_config['w_roost_h']   # weighting factor for horizontal attraction to roost
        self.w_roost_v = starling_config['w_roost_v']   # weighting factor for vertical attraction to roost
        self.w_rand    = starling_config['w_rand']      # default: 0.01, weight factor of random disturbances
        self.C_c       = starling_config['C_c']         # critical centrality below which member is interior to roost
        self.alpha     = starling_config['alpha']       # default: 0.5, between 0 and 1. modulates how tightly swarms converges into target. 0 is very loose, 1 is very tight 
        self.eps       = starling_config['eps']         # to stop divides by zero

        simulation_config =cfg.get_config(config, 'simulation')
        self.Ts        = simulation_config['Ts']        # simulation time step

        agent_config = cfg.get_config(config, 'agents')
        self.nAgents   = agent_config['nAgents']        # number of agents
        self.params = np.zeros((4, self.nAgents)) 

        # computed parameters
        self.sigma_param = starling_config['sigma_param']
        self.sigma = np.sqrt(np.divide(np.square(self.r_sep-self.r_h),self.sigma_param)) #std dev of the gaussion set, such that at that separation zone, near zero
        self.sigma_sqr = np.square(self.sigma)


    # Compute commands for Starling Flocking
    # --------------------------------------

    # this is run for each node
    #def compute_cmd(self,targets, centroid, states_q, states_p, k_node):
    def compute_cmd(self, states, targets, index, **kwargs):

        # extract
        states_q = states[0:3, :]      # positions
        states_p = states[3:6, :]      # velocities
        targets_q = targets[0:3, :]    # target positions
        centroid = kwargs.get('centroid')
        k_node = index

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
        Ts = self.Ts
        params = self.params
        
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
        if params_i[2] >= round(self.del_u/Ts,0)+1:
            # expand the range
            R_i = update_interaction(self.s,params_i[0],self.R_max,params_i[1],self.n_c)
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
                if dist <= R_i and n_counter < self.n_c:
                    
                    # increment the counter
                    n_counter += 1
                    
                    # compute the separation force
                    # ----------------------------
                    # note: this may need self.sigma vice self.sigma_sqr
                    f_sep += gaussian_set(dist, self.r_h, self.sigma_sqr)*np.divide((states_q[:,k_neigh]-states_q[:,k_node]),dist) 
                
                    # compute the cohesion force 
                    # -------------------------- 
                    if dist > self.r_h:
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
            u_sep = -self.m*np.divide(self.w_s,n_counter)*f_sep.reshape((3,1))
            u_coh = self.m*C_i*np.divide(self.w_c,n_counter)*f_coh.reshape((3,1))
            u_ali = self.m*self.w_a*np.divide(f_ali,np.linalg.norm(f_ali)).reshape((3,1))
            
            
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
        f_roost_h = np.divide(dist,1)*np.insert(unit_bank_2D,2,0)*(sign*self.alpha + sign*(1-self.alpha)*proj_to_target_2D)
        u_roost_h = -self.m*self.w_roost_h*f_roost_h.reshape((3,1)) 

        # compute the vertical roosting acceleration
        f_roost_v =  (targets[2,k_node] - states_q[2,k_node])*unit_vector_ver
        u_roost_v = self.m*self.w_roost_v*f_roost_v.reshape((3,1))
        
        # RANDOM VECTOR
        # ==============
        
        np.random.seed(k_node)  # each node had its own unique noise (bias)
        u_rand = self.m*self.w_rand*2*(np.random.rand(3,1)-0.5) # random is between -1 and 1
        
        # CONSOLIDATION
        # =============
            
        params[0,k_node] = R_i
        params[1,k_node] = n_counter
        params[2,k_node] = params_i[2]
        params[3,k_node] = C_i

        self.params = params
        
        cmd_i = u_coh + u_ali + u_sep + u_roost_h + u_roost_v + u_rand
        
        return cmd_i.ravel()
    

  
    