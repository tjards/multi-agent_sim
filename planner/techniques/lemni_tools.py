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
import os
import json
from .utils import quaternions as quat
from . import encirclement_tools as encircle_tools

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
        
    # // Implicit definition (see https://github.com/tjards/twisted_circles)
    #   3 = lemniscate of Gerono (with shift)
    #   4 = dumbbell curve 
    #   5 = lemniscate of Bernoulli

# import some parameters from encirclement_tools
r_desired, phi_dot_d, ref_plane, quat_0 = encircle_tools.get_params() 

# reference frames
#unit_lem    = np.array([1,0,0]).reshape((3,1))  # sets twist orientation (i.e. orientation of lemniscate along x) note: only required for explicit definition

# new
unit_lem = quat.rotate(quat_0, np.array([1, 0, 0]).reshape((3, 1))) # x-axis reference
twist_perp = quat.rotate(quat_0, np.array([0, 0, 1]).reshape((3,1))) # z-axis reference


#stretch     = -1*r_desired                      # stretch for lemni type 4 (legacy, remove later)
quat_0_ = quat.quatjugate(quat_0)               # used to untwist                               

#%% save configs
# --------------

from config.configs_tools import update_configs  
configs_entries = [
    ('c1_d', c1_d),
    ('c2_d', c2_d),
    # ('unit_lem', list(unit_lem))  # Uncomment if/when needed
]
update_configs('lemni', configs_entries)


#%% Useful functions 

def check_targets(targets):
    # if mobbing, offset targets back down
    if lemni_type == 2:
        targets[2,:] += r_desired/2
    return targets

# i'll get rid of this later
def enforce(tactic_type):
    
    # define vector perpendicular to encirclement plane
    if ref_plane == 'horizontal':
        #twist_perp = np.array([0,0,1]).reshape((3,1))
        
        # new
        #twist_perp = quat.rotate(quat_0, np.array([0,0,1]).reshape((3,1)))
        
        print('lemni reference place set to horizonatal')

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

enforce('lemni')

def compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node):
    
    u_enc = np.zeros((3,states_q.shape[1]))     
    u_enc[:,k_node] = - c1_d*sigma_1(states_q[:,k_node]-targets_enc[:,k_node])-c2_d*(states_p[:,k_node] - targets_v_enc[:,k_node])    
    
    return u_enc[:,k_node]

def lemni_target(lemni_all,state,targets,i,t):
    
    nVeh = state.shape[1]
         
    # initialize the lemni twist factor
    lemni = np.zeros([1, nVeh])
    
    # if mobbing, can offset targets up
    #if lemni_type == 2:
    #    targets[2,:] += r_desired/2
    
    targets = check_targets(targets)

    # UNTWIST -  each agent has to be untwisted into a common plane
    # -------------------------------------------------------------      
    last_twist = lemni_all[i-1,:] #np.pi*lemni_all[i-1,:]
    
    
    # new - rotate states into quat_0 frame
    #for n in range(state.shape[1]):
    #    state[0:3, n] = quat.rotate(quat_0_, state[0:3, n] - targets[0:3, n]) + targets[0:3, n]

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
            
        # normalize
        #untwist_quat /= np.linalg.norm(untwist_quat)

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
 
        # -----------------------
        # EXPLICITLY DEFINED
        # -----------------------
        
        # this logic works only for x-y planes
        # --------------------------------------
        '''
        rel_vec = state_untwisted[0:3, m] - targets[0:3, m]
        rel_vec_plane = quat.rotate(quat_0_, rel_vec.reshape(3, 1)).ravel()
        m_theta = np.arctan2(rel_vec_plane[1], rel_vec_plane[0])
        m_theta = np.mod(m_theta, 2*np.pi)
        '''
        
        # generalize to arbitrary planes
        # ------------------------------
        
        rel_vec = state_untwisted[0:3, m] - targets[0:3, m]
        # define local orthogonal basis of the embedding plane in world frame
        x_e = unit_lem.ravel()  
        x_e /= np.linalg.norm(x_e)
        z_e = twist_perp.ravel() 
        z_e /= np.linalg.norm(z_e)                  
        y_e = np.cross(z_e, x_e)  
        y_e /= np.linalg.norm(y_e)
        # project rel_vec into basis plane
        x_proj = np.dot(rel_vec, x_e)
        y_proj = np.dot(rel_vec, y_e)
        m_theta = np.arctan2(y_proj, x_proj)
        m_theta = np.mod(m_theta, 2*np.pi)

        
        # rolling
        if lemni_type == 1:  
            m_shift = -np.pi + 0.1 * t
            lemni[0, m] = m_theta + m_shift
        
        # mobbing
        elif lemni_type == 2:  
            lemni[0, m] = m_theta - np.pi
        
        # surveillance + rest
        else:
            lemni[0, m] = m_theta 
        
        twist = lemni[0,m] 
        
        # rotate
        if lemni_type < 3:
            twist_quat = quat.e2q(twist*unit_lem.ravel())
        
        # ---------------------
        # IMPLICITLY DEFINED
        # ---------------------
            
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
   
    
        # normalize
        # ---------
        #twist_quat /= np.linalg.norm(twist_quat)
    
        # twist positions
        # ---------------
        
        # pull out states/targets
        states_q_i = state[0:3,m]
        targets_i = targets[0:3,m]
        target_encircle_i = targets_encircle[0:3,m]
        
        # get the vector of agent position wrt target
        state_m_shifted = states_q_i - targets_i
        target_encircle_shifted = target_encircle_i - targets_i 
   

        #twist_quat = quat.e2q(twist*unit_lem.ravel())        
        twist_pos = quat.rotate(twist_quat,target_encircle_shifted)+targets_i 
        
        # new
        #twist_pos = quat.rotate(quat_0_, twist_pos - targets_i) + targets_i

        targets_encircle[0:3,m] = twist_pos
        
        # twist velocities
        # ----------------
        
      
        # this logic works in the x-y plane
        # ----------------------------------
        
        '''
        w_vector = phi_dot_desired_i[0,m]*twist_perp 
        w_vector_twisted =  quat.rotate(twist_quat,w_vector) # this is in embedding plane
        # find states in the embedding plane
        #state_m_shifted_rot = quat.rotate(quat_0, state_m_shifted)
        # now compute velo vector in the embedding plane
        twist_v_vector = np.cross(w_vector_twisted.ravel(),state_m_shifted)
        # rotate this vector back into the world frame
        #twist_v_vector = quat.rotate(quat_0_, twist_v_vector)
        '''
        
        # generalize to arbitrary planes
        # ------------------------------
        
        
        w_vector = phi_dot_desired_i[0,m]*twist_perp 
        w_vector_twisted =  quat.rotate(twist_quat,w_vector) 
        
        # rotate into embedding plane
        state_m_shifted_emb = quat.rotate(quat_0_, state_m_shifted.reshape(3,1)).ravel()
        w_vector_emb = quat.rotate(quat_0_, w_vector_twisted).ravel()
        v_emb = np.cross(w_vector_emb, state_m_shifted_emb)
        
        # rotate back
        twist_v_vector = quat.rotate(quat_0, v_emb.reshape(3,1)).ravel()
        


        targets_encircle[3,m] =  - twist_v_vector[0] 
        targets_encircle[4,m] =  - twist_v_vector[1] 
        targets_encircle[5,m] =  - twist_v_vector[2]      

    return targets_encircle, lemni



