#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:12:59 2021

@author: tjards

This program implements Reynolds Rules of Flocking ("boids")

"""

import numpy as np
import config.config as cfg

# helpers
# -------
def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out

# custom class
from planner.base import BasePlanner
class Planner(BasePlanner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
        # load the configs
        reynolds_config =cfg.get_config(config, 'planner.techniques.flocking_reynolds')
        self.escort          = reynolds_config['escort']          # escort (i.e. target tracking?): 0 = no, 1 = yes
        self.cd_1            = reynolds_config['cd_1']            # cohesion weight
        self.cd_2            = reynolds_config['cd_2']            # alignment weight
        self.cd_3            = reynolds_config['cd_3']            # separation weight
        self.cd_track        = reynolds_config['cd_track']        # nominally, zero, unless escorting, then ensure >0
        self.maxu            = reynolds_config['maxu']            # max input (per rule)  note: dynamics *.evolve_sat must be used for constraints
        self.maxv            = reynolds_config['maxv']            # max v                 note: dynamics *.evolve_sat must be used for constraints
        self.recovery        = reynolds_config['recovery']        # recover if far away (0 = no, 1 = yes)
        self.far_away        = reynolds_config['far_away']        # recover how far away (i.e. when to go back to centroid)?
        self.mode_min_coh    = reynolds_config['mode_min_coh']    # enforce min # of agents (0 = no, 1 = yes)
        self.agents_min_coh  = reynolds_config['agents_min_coh']  # min number of agents
        self.r               = reynolds_config['r']               # range at which neighbours can be sensed  
        self.r_prime         = reynolds_config['r_prime']         # range at which obstacles can be sensed

    def order(self, states_q):
        distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
        
        # to find the radius that includes min number of agents
        if self.mode_min_coh == 1:
            slide = 0
            for k_node in range(states_q.shape[1]):
                #slide += 1
                for k_neigh in range(slide,states_q.shape[1]):
                    if k_node != k_neigh:
                        distances[k_node,k_neigh] = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
        return distances 

    # Compute commands
    # ----------------

    def compute_cmd(self, states, targets, index, **kwargs):

        # Extract 
        states_q = states[0:3, :]     # positions
        states_p = states[3:6, :]     # velocities
        targets_q = targets[0:3, :]   # target positions
        centroid = kwargs.get('centroid')
        distances = kwargs.get('distances')
        k_node = index

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
        if self.mode_min_coh == 1:
            
            # make sure the number of vehicles is bigger than the min number of agents 
            if distances.shape[0] < self.agents_min_coh+2:
                raise Exception('There are an insufficient number of agents for the cohesion mode selected. Minimum number of agents for mode ',self.agents_min_coh ,' is ', self.agents_min_coh+2, ' and you have selected ', distances.shape[0] )
        
            r_coh = 0
            # pull agent distantces for this node
            node_ranges = distances[k_node,:]
            # sort the distances
            node_ranges_sorted = np.sort(node_ranges)
            # take the distance of the farthest agent that satisfies min count
            r_coh_temp = node_ranges_sorted[self.agents_min_coh+1]
            r_coh = r_coh_temp
        else:
            # else, just rely on default range
            r_coh = self.r
            
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
                if dist < np.maximum(self.r,r_coh):
        
                    # count
                    temp_total += 1                        
        
                    # sum 
                    sum_velos += states_p[:,k_neigh]
        
                # if within cohesion range 
                if dist < np.maximum(self.r,r_coh):
                    
                    #count
                    temp_total_coh += 1
                    
                    #sum
                    sum_poses += states_q[:,k_neigh]
        
                # if within the separation range 
                if dist < self.r_prime:
                    
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
                temp_u_coh = (self.maxv*np.divide(((np.divide(sum_poses,temp_total_coh) - states_q[:,k_node])),norm_coh)-states_p[:,k_node])
                u_coh[:,k_node] = self.cd_1*norm_sat(temp_u_coh,self.maxu)
            
            # Alignment
            # ---------
            if norm_ali != 0:                 
                temp_u_ali = (self.maxv*np.divide((np.divide(sum_velos,temp_total)),norm_ali)-states_p[:,k_node])
                u_ali[:,k_node] = self.cd_2*norm_sat(temp_u_ali,self.maxu)
        
        if temp_total_prime != 0 and norm_sep != 0:
                
            # Separtion
            # ---------
            temp_u_sep = (self.maxv*np.divide(((np.divide(sum_obs,temp_total_prime))),norm_sep)-states_p[:,k_node]) 
            u_sep[:,k_node] = -self.cd_3*norm_sat(temp_u_sep,self.maxu)
                    
        # Tracking
        # -------- 
        cd_4 = self.cd_track
            
        # if doing recovery
        if self.recovery == 1:
            # and if far away, adjust gain to drive back
            if np.linalg.norm(centroid.transpose()-states_q[:,k_node]) > self.far_away:
                cd_4 = 0.3 # overides set value, so recovery is always a low gain
            
        # if escorting, track the target (overrides recovery actions)
        if self.escort == 1:
            cd_4 = self.cd_track
            if cd_4 == 0:
                print('WARNING: no gain set for tracking target, please set a gain > 0')
            temp_u_nav = (targets_q[:,k_node]-states_q[:,k_node])
        else:
            temp_u_nav = (centroid.transpose()-states_q[:,k_node])
        
        # compute tracking 
        u_nav[:,k_node] = cd_4*norm_sat(temp_u_nav,self.maxu)
        
        # compute consolidated commands
        cmd_i[:,k_node] = u_coh[:,k_node] + u_ali[:,k_node] + u_sep[:,k_node] + u_nav[:,k_node] 
        
        return cmd_i[:,k_node]
    