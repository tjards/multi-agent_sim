#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements lemniscatic trajectories
Note: encirclement_tools and quaternions are dependencies 

Created on Thu Feb 18 14:20:17 2021

@author: tjards
"""
#%% Import stuff
import numpy as np
import pickle 
import matplotlib.pyplot as plt

# dependencies
from utils import quaternions as quat
from utils import encirclement_tools as encircle_tools


#%% Parameters
# -----------

# tunable
c1_d        = 1             # gain for position (q)
c2_d        = 2*np.sqrt(1)  # gain for velocity (p)
lemni_type  = 0
            
    # // Explcit definition of rotation (https://ieeexplore.ieee.org/document/9931405)
    #   0 = lemniscate of Gerono - surveillance (/^\)
    #   1 = lemniscate of Gerono - rolling (/^\ -> \_/)
    #   2 = lemniscate of Gerono - mobbing (\_/)
        
    # // Solved indirectly (see https://github.com/tjards/twisted_circles)
    #   3 = lemniscate of Gerono (with shift)
    #   4 = dumbbell curve 
    #   5 = lemniscate of Bernoulli

# import some parameters from encirclement_tools
r_desired, phi_dot_d, ref_plane, quat_0 = encircle_tools.get_params() 

# reference frames
unit_lem    = np.array([1,0,0]).reshape((3,1))  # sets twist orientation (i.e. orientation of lemniscate along x)
stretch     = -1*r_desired                      # stretch for lemni type 4 (legacy, remove later)
quat_0_ = quat.quatjugate(quat_0)               # used to untwist                               

#%% Useful functions 

def check_targets(targets):
    # if mobbing, offset targets back down
    if lemni_type == 2:
        targets[2,:] += r_desired/2
    return targets

def enforce(tactic_type):
    
    # define vector perpendicular to encirclement plane
    if ref_plane == 'horizontal':
        twist_perp = np.array([0,0,1]).reshape((3,1))
    elif tactic_type == 'lemni':
        print('Warning: Set ref_plane to horizontal for lemniscate')
    
    # enforce the orientation for lemniscate (later, expand this for the general case)
    lemni_good = 0
    if tactic_type == 'lemni':
        if quat_0[0] == 1:
            if quat_0[1] == 0:
                if quat_0[2] == 0:
                    if quat_0[3] == 0:
                        lemni_good = 1
    if tactic_type == 'lemni' and lemni_good == 0:
        print ('Warning: Set quat_0 to zeros for lemni to work')
        # travis note for later: you can do this rotation after the fact for the general case
    
    return twist_perp

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

#%% main functions

twist_perp = enforce('lemni')

def compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node):
    
    u_enc = np.zeros((3,states_q.shape[1]))     
    u_enc[:,k_node] = - c1_d*sigma_1(states_q[:,k_node]-targets_enc[:,k_node])-c2_d*(states_p[:,k_node] - targets_v_enc[:,k_node])    
    
    return u_enc[:,k_node]

def lemni_target(lemni_all,state,targets,i,t):
    
    nVeh = state.shape[1]
    
    # === for experiments only
    # global lemni_type
    # run_experiment = 1
    # # run a standardized experiment, or "dance"
    # if run_experiment == 1:
    #     # circle
    #     if t <= 10:
    #         lemni_type = 6
    #     elif t > 10 and t <= 70:
    #         lemni_type = 5
    #     # circle
    #     elif t > 70 and t <= 80:
    #         lemni_type = 6
    #     elif t > 80 and t <= 140:
    #         lemni_type = 4
    #     # circle
    #     elif t > 140 and t <= 150:
    #         lemni_type = 6
    #     elif t > 150 and t <= 210:
    #         lemni_type = 3
    #     else:
    #         lemni_type = 3
        
    # =======================
         
    # initialize the lemni twist factor
    lemni = np.zeros([1, nVeh])
    
    # if mobbing, can offset targets up
    #if lemni_type == 2:
    #    targets[2,:] += r_desired/2
    
    targets = check_targets(targets)

    # UNTWIST -  each agent has to be untwisted into a common plane
    # -------------------------------------------------------------      
    last_twist = lemni_all[i-1,:] #np.pi*lemni_all[i-1,:]
    state_untwisted = state.copy()
    
    # for each agent 
    for n in range(0,state.shape[1]):
        
        # get the last twist
        untwist = last_twist[n]
        
        # if 3D Gerono:
        if lemni_type < 3:
            untwist_quat = quat.quatjugate(quat.e2q(untwist*unit_lem.ravel()))
        
        # if gerono (with shift)
        elif lemni_type == 3: 
            untwist_quat = np.zeros(4)
            # compute quaternion
            untwist_quat[0] = -np.sqrt(2)*np.sqrt(1 - np.sin(untwist))/2
            untwist_quat[1] = -np.sqrt(2)*np.sqrt(np.sin(untwist) + 1)/2
            # rotate
            untwist_quat = quat.quatjugate(untwist_quat)
        
        # if dumbbell
        elif lemni_type == 4:
            untwist_quat = np.zeros(4)
            # compute quaternion
            untwist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(untwist)**2 + 1)/2
            untwist_quat[1] = -np.sqrt(2)*np.sqrt(-(np.cos(untwist) - 1)*(np.cos(untwist) + 1))/2
            # rotate
            untwist_quat = quat.quatjugate(untwist_quat)

        # if bernoulli
        elif lemni_type == 5:
            untwist_quat = np.zeros(4)
            # compute quaternion
            untwist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(untwist) + 1)/(2*np.sqrt(np.sin(untwist)**2 + 1))
            untwist_quat[1] = -np.sqrt(2)*np.sqrt(1 - np.cos(untwist))/(2*np.sqrt(np.sin(untwist)**2 + 1))
            #rotate    
            untwist_quat = quat.quatjugate(untwist_quat)
        
        else:
            # bypass
            untwist_quat =  np.zeros(4)
            untwist_quat[0] = 1
            twist_quat =  untwist_quat
            
        
        # pull out states
        states_q_n = state[0:3,n]
        # pull out the targets (for reference frame)
        targets_n = targets[0:3,n] 
        # untwist the agent 
        state_untwisted[0:3,n] = quat.rotate(untwist_quat,states_q_n - targets_n) + targets_n  
 
    # ENCIRCLE -  form a common untwisted circle
    # ------------------------------------------
    
    # compute the untwisted trejectory 
    targets_encircle, phi_dot_desired_i = encircle_tools.encircle_target(targets, state_untwisted)
    
    # TWIST - twist the circle
    # ------------------------
    
    # for each agent, we define a unique twist 
    for m in range(0,state.shape[1]):
 
        # pull out states/targets
        states_q_i = state[0:3,m]
        targets_i = targets[0:3,m]
        target_encircle_i = targets_encircle[0:3,m]
        
        # get the vector of agent position wrt target
        state_m_shifted = states_q_i - targets_i
        target_encircle_shifted = target_encircle_i - targets_i
        
        # just give some time to form a circle first
        if i > 0:
            
            # compute the lemni factor
            # -----------------------
            
            # rolling needs to dynamically adjust the lemni factor  
            if lemni_type == 1: 
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                #m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                m_shift = - np.pi + 0.1*t
                lemni[0,m] = m_theta + m_shift
                
            # mobbing needs to rotatate by pi (i.e. flip updside down) 
            if lemni_type == 2: 
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                lemni[0,m] = m_theta - np.pi
            
            # surveillance and others use this
            else: 
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                lemni[0,m] = m_theta 

        # twist the trajectory position and load it
        # note: twist is a misleading var name now; consider changing later
        twist = lemni[0,m] 
        
        # if 3D Gerono:
        if lemni_type < 3:
            twist_quat = quat.e2q(twist*unit_lem.ravel())
        
        # if Gerono (with shift)
        elif lemni_type == 3:
            twist_quat = np.zeros(4)
            # compute quaternion
            twist_quat[0] = -np.sqrt(2)*np.sqrt(1 - np.sin(twist))/2
            twist_quat[1] = -np.sqrt(2)*np.sqrt(np.sin(twist) + 1)/2

        # if dumbbell
        elif lemni_type == 4:
            twist_quat = np.zeros(4)
            # compute quaternion
            twist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(twist)**2 + 1)/2
            twist_quat[1] = -np.sqrt(2)*np.sqrt(-(np.cos(twist) - 1)*(np.cos(twist) + 1))/2
            
        # if bernoulli
        elif lemni_type == 5:
            twist_quat = np.zeros(4)
            # computer quaternion
            twist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(twist) + 1)/(2*np.sqrt(np.sin(twist)**2 + 1))
            twist_quat[1] = -np.sqrt(2)*np.sqrt(1 - np.cos(twist))/(2*np.sqrt(np.sin(twist)**2 + 1))
   
        #twist_quat = quat.e2q(twist*unit_lem.ravel())        
        twist_pos = quat.rotate(twist_quat,target_encircle_shifted)+targets_i  
        targets_encircle[0:3,m] = twist_pos
        
        # twist the trajectory velocity and load it
        w_vector = phi_dot_desired_i[0,m]*twist_perp                        # pretwisted
        w_vector_twisted = quat.rotate(twist_quat,w_vector)                 # twisted 
        twist_v_vector = np.cross(w_vector_twisted.ravel(),state_m_shifted)
        targets_encircle[3,m] =  - twist_v_vector[0] 
        targets_encircle[4,m] =  - twist_v_vector[1] 
        targets_encircle[5,m] =  - twist_v_vector[2]      

    return targets_encircle, lemni



