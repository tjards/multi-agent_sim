#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements lemniscatic trajectories
Note: encirclement_tools and quaternions are dependencies 

Notes: 
    - always perform the rotations around x (regardless of learning)
    - Explicit learns along x and/or z
    - Implicit only allows learning around x (structural constraint)

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

# learning
learning            = 'CALA'    # None, 'CALA'
 
# tunable
c1_d        = 1             # gain for position (q)
c2_d        = 2*np.sqrt(1)  # gain for velocity (p)
lemni_type  = 0             # CALA learning needs 0
            
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


#%% safety checks
# ---------------
if learning == 'CALA':
    
    learning_coupling   = True      # options: True (default), else unlikely to work
    learning_axes       = 'xz'      # options: 'x' (prototype only), 'xz' (default)
    
    if 'x' not in learning_axes and 'z' not in learning_axes:
        raise Exception('warning: no learning axis defined')
    if learning_coupling and learning_axes != 'xz':
        raise Exception('warning: coupling only supported for xz-axis')
    if lemni_type != 0:
        raise Exception('warning: CALA learning only supported for type-0 curves')
else:
    # force no axis if not learning (redundant)
    learning_axes = ''

                            
#%% save configs
# --------------

from config.configs_tools import update_configs  
configs_entries = [
    ('c1_d', c1_d),
    ('c2_d', c2_d),
    ('learning', learning),
    # ('unit_lem', list(unit_lem))  # Uncomment if/when needed
]
update_configs('lemni', configs_entries)


#%% Useful functions 

def check_targets(targets):
    
    # if mobbing, offset targets back down
    if lemni_type == 2:
        targets[2,:] += r_desired/2
    return targets

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1


#%% main functions

def compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node):
    
    u_enc = np.zeros((3,states_q.shape[1]))     
    u_enc[:,k_node] = - c1_d*sigma_1(states_q[:,k_node]-targets_enc[:,k_node])-c2_d*(states_p[:,k_node] - targets_v_enc[:,k_node])    
    
    return u_enc[:,k_node]

def lemni_target(lemni_all,state,targets,i,t, learn_actions_coupled):
    
    
    # load
    unit_lem = quat.rotate(quat_0, np.array([1, 0, 0]).reshape((3, 1)))     # x-axis reference
    #tilt_perp = quat.rotate(quat_0, np.array([1, 0, 0]).reshape((3, 1)))    # y-axis 
    twist_perp = quat.rotate(quat_0, np.array([0, 0, 1]).reshape((3,1)))    # z-axis reference
    quat_0_ = quat.quatjugate(quat_0)                                       # used to untwist 
    nVeh = state.shape[1]
    
    # breakout x and z directions
    if learning_coupling:
        
        learn_actions = {
           'x': learn_actions_coupled['xz'][0:state.shape[1]],
           'z': learn_actions_coupled['xz'][state.shape[1]::]
           }      
    else:
        learn_actions = learn_actions_coupled
        
           
    # initialize the lemni twist factor
    #lemni = np.zeros([1, nVeh])
    lemni = np.zeros([2, nVeh])
    
    # if mobbing, can offset targets up    
    targets = check_targets(targets)

    # UNTWIST -  each agent has to be untwisted into a common plane
    # -------------------------------------------------------------      
    #last_twist = lemni_all[i-1,:] #np.pi*lemni_all[i-1,:]
    #last_twist = lemni_all[i-1,0,:]
    
    # always do stuff around x, learning or not learning
    #if learning is None or 'x' in learning_axes:
    last_twist_x = lemni_all[i-1, 0, :]
    #else:
    #    last_twist_x = np.zeros(state.shape[1])  # fallback

    # for now, z is only a learned parameter
    if 'z' in learning_axes:
        last_twist_z = lemni_all[i-1, 1, :]
    else:
        last_twist_z = np.zeros(state.shape[1])  # fallback
    
    state_untwisted = state.copy()
    
    # for each agent 
    for n in range(0,state.shape[1]):
        
        # get the last twist
        #untwist = last_twist[n]
        untwist = last_twist_x[n] # lemni_type >= 3 only uses x_axis
        
        # if 3D Gerono:
        if lemni_type < 3:
            # untwist_quat = quat.quatjugate(quat.e2q(untwist*unit_lem.ravel()))
            qx = quat.e2q(last_twist_x[n] * unit_lem.ravel())
            qz = quat.e2q(last_twist_z[n] * twist_perp.ravel())
            #qz =  quat.axis_angle_to_quat(last_twist_z[n] * twist_perp.ravel())
            #qz = quat.e2q(last_twist_z[n] * tilt_perp.ravel())
    
            untwist_quat = quat.quatjugate(quat.quat_mult(qz, qx))
        
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
    targets_encircle, phi_dot_desired_i, sorted_neighs = encircle_tools.encircle_target(targets, state_untwisted)
    
    # TWIST - twist the circle
    # ------------------------
    
    # for each agent, we define a unique twist 
    for m in range(0,state.shape[1]):
 
        # -----------------------
        # EXPLICITLY DEFINED
        # -----------------------
        
        # generalize to arbitrary planes        
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
        #m_theta = np.mod(m_theta, 2*np.pi)

        
        # ---------------------------
        # define the twist parameter
        # ---------------------------

        # rolling
        if lemni_type == 1:  
            m_shift = -np.pi + 0.1 * t
            #lemni[0, m] = m_theta + m_shift
            base_theta = m_theta + m_shift
        
        # mobbing
        elif lemni_type == 2:  
            #lemni[0, m] = m_theta - np.pi
            base_theta = m_theta - np.pi
        
        # surveillance + rest
        else:
            #lemni[0, m] = m_theta
            base_theta = m_theta #- np.pi/2
            
        
        # -------------------------- #
        # offset by learned parameter
        # -------------------------- #
        
        lemni[0, m] = base_theta  
        
        if 'x' in learning_axes:
            lemni[0, m] += learn_actions.get('x', np.zeros(nVeh))[m]
        #lemni[0, m] = np.mod(lemni[0, m], 2 * np.pi)
        #lemni[0, m] = (lemni[0, m] + np.pi) % (2 * np.pi) - np.pi # wrap -pi to pi
         
        if 'z' in learning_axes:
        
            lemni[1, m] = learn_actions.get('z', np.zeros(nVeh))[m]
        #lemni[1, m] = np.mod(lemni[1, m], 2 * np.pi)
        #lemni[1, m] = (lemni[1, m] + np.pi) % (2 * np.pi) - np.pi # wrap -pi to pi
        
        twist = lemni[0, m] # only along x (for implicit cases)
        # rotate
        
        # ---------------------
        # EXPLICITLY DEFINED
        # ---------------------
        if lemni_type < 3:
            
            # Always apply x-axis phase twist (learned or not)
            qx = quat.e2q(lemni[0, m] * unit_lem.ravel())

            # Only apply z-axis twist if learning it
            if 'z' in learning_axes:
                qz = quat.e2q(lemni[1, m] * twist_perp.ravel())
                #qz = quat.axis_angle_to_quat(lemni[1, m] * twist_perp.ravel())
                #qz = quat.e2q(lemni[1, m] * tilt_perp.ravel())
                twist_quat = quat.quat_mult(qz, qx)
            else:
                twist_quat = qx
            
 
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
        
        # generalize to arbitrary planes
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

    return targets_encircle, lemni, sorted_neighs



