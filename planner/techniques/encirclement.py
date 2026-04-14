#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 21:26:07 2021

This module implements dynamic encirclement 

 delta_phi_desired = 2Pi/N

@author: tjards
"""

import numpy as np
#import os
#import json
from .utils import quaternions as quat
import config.config as cfg

# helpers

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def polar2cart(r, theta):
    #note: accepts and return radians
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def cart2polar(x, y):
    #note: return radians
    # [-pi, pi]
    #r = np.linalg.norm(x,y)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    #convert to 0 to 2Pi
    theta = np.mod(theta, 2*np.pi) 
    return r, theta 

def phi_dot_i_desired(phi_i, phi_j, phi_k, phi_dot_desired):
    gamma = 0.5 # tunable
    phi_ki = np.mod(phi_i - phi_k, 2*np.pi) # make sure between 0 and 2pi
    phi_ij = np.mod(phi_j - phi_i, 2*np.pi) # make sure between 0 and 2pi
    phi_dot_i_desired = np.divide(3*phi_dot_desired + gamma*(phi_ki-phi_ij),3)
    return phi_dot_i_desired
    
def directToCircle(A,B,r):
    # A = target center
    # B = vechicle position 
    # C = closest point between 
    C = A+r*np.divide(B-A,np.linalg.norm(B-A))
    return C

def centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    sum_z = np.sum(points[:, 2])
    centroid = np.array((sum_x/length, sum_y/length, sum_z/length), ndmin = 2)
    return centroid.transpose() 


from planner.base import BasePlanner 
class Planner(BasePlanner):
    
    def __init__(self, config_data, **kwargs):
        super().__init__(config_data, **kwargs)

        circle_config = cfg.get_config(config_data, 'planner.techniques.encirclement')
        
        # Store hyperparameters
        self.c1_d = circle_config.get('c1_d', 2)                # position gain
        self.c2_d = circle_config.get('c2_d', 2*np.sqrt(2))     # velocity gain
        self.r_max = circle_config.get('r_max', 50)             # max sensing range 
        self.r_desired = circle_config.get('r_desired', 5)      # desired encirclement radius
        self.phi_dot_d = circle_config.get('phi_dot_d', 0.05)   # desired angular speed [rad/s]
        self.ref_plane = circle_config.get('ref_plane', 'horizontal') # reference plane for encirclement 
        TEMP_quat_0_0 = circle_config.get('quat_0_0', 0.0)      # orientation of disc
        TEMP_quat_0_1 = circle_config.get('quat_0_1', 0.0)
        TEMP_quat_0_2 = circle_config.get('quat_0_2', 0.0)
        self.quat_0 = quat.e2q(np.array([TEMP_quat_0_0, TEMP_quat_0_1, TEMP_quat_0_2]))
        
        # Store as instance attributes for use in methods
        self.d = self.r_desired  # Make compatible with orchestrator pattern

        # compute desired separation (for analyzing results)
        nAgents = cfg.get_config(config_data, 'agents.nAgents')
        print(nAgents)
        self.desired_separation = self.compute_desired_sep(self.r_desired, nAgents)

        # graph parameters (standardized in base class)
        self.sensor_range_matrix = self.r_max * np.ones((nAgents, nAgents))
        self.connection_range_matrix = self.r_max * np.ones((nAgents, nAgents))

    def update_trajectory(self, Trajectory, targets, **kwargs):

        state = kwargs.get('state')

        Trajectory.trajectory, _, _ = self.encircle_target(targets, state)
                          
    def compute_cmd_vectorized(self, states, targets, neighbor_lists, **kwargs):
        """Vectorized: no neighbor interaction, pure per-agent tracking."""
        states_q = states[0:3, :]
        states_p = states[3:6, :]
        targets_q = targets[0:3, :]
        targets_p = targets[3:6, :]
        dq = states_q - targets_q
        dp = states_p - targets_p
        sigma_1_dq = dq / np.sqrt(1.0 + dq * dq)
        return -self.c1_d * sigma_1_dq - self.c2_d * dp

    def compute_cmd(self, states, targets, index, **kwargs):

        # extract
        states_q        = states[0:3, :]    # positions
        states_p        = states[3:6, :]    # velocities
        targets_enc     = targets[0:3, :]
        targets_v_enc   = targets[3:6, :]
        k_node          = index
        
        return - self.c1_d*sigma_1(states_q[:,k_node]-targets_enc[:,k_node])-self.c2_d*(states_p[:,k_node] - targets_v_enc[:,k_node])
        
    def encircle_target(self, targets, state):
            
        # desired rate of encirclement [rad/s]
        # -----------------------------------
        phi_dot_desired = self.phi_dot_d                 
        
        # initialise global stuff
        # -----------------------
        targets_encircle = targets.copy() 
        points_i = np.zeros((3,state.shape[1]))
        temp = np.zeros((3,1))
        quatern_ = quat.quatjugate(self.quat_0)
        
        # Regulation of Radius (position control)
        # ------------------------------   
        new_pos_desired_i = np.zeros((3,state.shape[1]))
        
        # iterate through each agent
        for ii in range(0,state.shape[1]):

            # to rotate with reference to horizontal
            if self.ref_plane == 'horizontal':
                # rotate down to the reference plane
                points_i[:,ii] = quat.rotate(quatern_,state[0:3,ii]-targets[0:3,ii])+targets[0:3,ii]
                # now find the desired position projected on the plane
                temp[0:2,0] = directToCircle(targets[0:2,ii],points_i[0:2,ii],self.r_desired)
                temp[2,0] = targets[2,ii] # at altitude
                # now rotate back
                new_pos_desired_i[:,ii] = quat.rotate(self.quat_0,temp.ravel()-targets[0:3,ii])+targets[0:3,ii]            
            
        # Regulation of Angular speed (velocity control)
        # ----------------------------------------------   
        # express state with reference to target
        state_shifted = state - targets
            
        # to rotate with reference to horizontal
        if self.ref_plane == 'horizontal':
            # define a new unit vector, which is perp to plane 
            unit_v = np.array([0,0,1]).reshape((3,1))
            # initialize a new state vector
            state_shifted_new = np.zeros((3,state.shape[1]))
            # rotate each agent into the reference plane
            for ij in range(0,state.shape[1]):
                state_shifted_new[:,ij] = quat.rotate(quatern_,state_shifted[0:3,ij])

            # convert to polar coordinates
            polar_r, polar_phi = cart2polar(state_shifted_new[0,:], state_shifted_new[1,:])

        # sort by phi and save the indicies so we can reassemble
        polar_phi_sorted = np.sort(polar_phi, axis=0)
        polar_phi_argsort = np.argsort(polar_phi, axis=0) 
        
        # for each vehicle, define a desired angular speed 
        phi_dot_desired_i = np.zeros((1,state_shifted.shape[1]))
        phiDot_out = np.zeros((1,state_shifted.shape[1]))
        
        # identify leading and lagging 
        for ii in range(0,state_shifted.shape[1]):
            # define leading and lagging vehicles (based on angles)
            if ii == 0:
                ij = state_shifted.shape[1]-1    
            else:
                ij = ii-1 # lagging vehicle
            
            if ii == state_shifted.shape[1]-1:
                ik = 0
            else:
                ik = ii+1 # leading vehicle 
            
            # compute distances
            dist_lag = np.linalg.norm(state_shifted[0:3,ii]-state_shifted[0:3,ij])
            dist_lead = np.linalg.norm(state_shifted[0:3,ii]-state_shifted[0:3,ik])
            
            # if neighbours too far away, default to the desired encirclement speed
            if dist_lead > self.r_max or dist_lag > self.r_max:
                phi_dot_desired_i[0,ii] = phi_dot_desired
                continue
            
            # compute the desired phi-dot       
            phi_dot_desired_i[0,ii] = phi_dot_i_desired(polar_phi_sorted[ii], polar_phi_sorted[ij], polar_phi_sorted[ik], phi_dot_desired)
        
        # convert the angular speeds back to cartesian (in the correct order)
        # ----------------------------------------------
        xy_dot_desired_i = np.zeros((3,state.shape[1]))
        
        index_proper = 0
        for ii in polar_phi_argsort:
            
            # get angular speed 
            w_vector = quat.rotate(self.quat_0,phi_dot_desired_i[0,index_proper]*unit_v)
            # find the corresponding velo vector
            v_vector = np.cross(w_vector.ravel(),state_shifted[0:3,ii])
            # break out into components
            xy_dot_desired_i[0,ii] = v_vector[0] 
            xy_dot_desired_i[1,ii] = v_vector[1] 
            xy_dot_desired_i[2,ii] = v_vector[2] 
            
            #fix phiDot
            phiDot_out[0,ii] = phi_dot_desired_i[0,index_proper]
            
            index_proper += 1

        # define new targets for encirclement
        # ----------------------------------
        # if we're rotating wrt horizontal 
        if self.ref_plane == 'horizontal':
            targets_encircle[0:3,:] = new_pos_desired_i[:,:]
            targets_encircle[3:6,:] = -xy_dot_desired_i[:,:] 

        return targets_encircle, phiDot_out, polar_phi_argsort


    def get_params(self):
        return self.r_desired, self.phi_dot_d, self.ref_plane, self.quat_0

    def compute_desired_sep(self,r, N):
        theta = np.pi / N  # Half of the central angle
        return 2 * r * np.sin(theta)
