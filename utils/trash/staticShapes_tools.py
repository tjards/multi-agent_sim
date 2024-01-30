#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:26:17 2022

Static Shapes 

Note: this is only sort of working and is very inefficient 



@author: tjards
"""
#%% Import stuff
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from utils import quaternions as quat
from utils import encirclement_tools as encircle_tools


#%% Parameters
# -----------
# tunable
c1_d        = 2             # position (q)
c2_d        = 2*np.sqrt(2)  # velocity (p)
eps         = 0.1           # nominally 0.1
lemni_type  = 3             # 0 = surv, 1 = rolling, 2 = mobbing, 3 = experimental  

# inherited from encirclement
r_desired, phi_dot_d, ref_plane, quat_0 = encircle_tools.get_params() 
unit_lem    = np.array([1,0,0]).reshape((3,1))    # sets twist orientation (i.e. orientation of lemniscate along x)
quat_0_ = quat.quatjugate(quat_0)                 # used to untwist   


#%% Useful functions 

#def enforce(ref_plane, tactic_type, quat_0):
def enforce(tactic_type):
    
    # define vector perpendicular to encirclement plane
    if ref_plane == 'horizontal':
        twist_perp = np.array([0,0,1]).reshape((3,1))
    elif tactic_type == 'statics':
        print('Warning: Set ref_plane to horizontal for lemniscate')
    
    # enforce the orientation for lemniscate (later, expand this for the general case)
    lemni_good = 0
    if tactic_type == 'statics':
        if quat_0[0] == 1:
            if quat_0[1] == 0:
                if quat_0[2] == 0:
                    if quat_0[3] == 0:
                        lemni_good = 1
    if tactic_type == 'statics' and lemni_good == 0:
        print ('Warning: Set quat_0 to zeros for lemni to work')
        # travis note for later: you can do this rotation after the fact for the general case
    
    return twist_perp


# sigma norm
def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

# transition function (need to do this for each agent)
def compute_fi_n1p1(states_q, targets, transition_loc, transition_rate):
    # transition_loc is desired distance from target to center sigmoid
    prox_i = sigma_norm(states_q[0:3]-targets[0:3])
    z_i = prox_i-sigma_norm(transition_loc)
    f_i = 2/(1 + np.exp(-z_i*transition_rate)) -1     
    return f_i

# transition function (need to do this for each agent)
def compute_fi_00p1(states_q, targets, transition_loc, transition_rate):
    # transition_loc is desired distance from target to center sigmoid
    prox_i = sigma_norm(states_q[0:3]-targets[0:3])
    z_i = prox_i-sigma_norm(transition_loc)
    f_i = 1/(1 + np.exp(-z_i*transition_rate))    
    return f_i

def compute_sign(states_q,targets,quatern):
    # need to compute the sign (+/-)
    unit_v = np.array([0,1,0]).reshape((3,1))
    unit_v_rotated = quat.rotate(quatern,unit_v)
    divZero = 0.0001
    # find the corresponding velo vector
    #unit_v_cross = np.cross(unit_v.ravel(),states_q_shifted) # state needs to be shifted for target
    sign = np.divide(np.dot(states_q-targets,unit_v_rotated),np.maximum(np.linalg.norm(states_q),divZero))
    return sign 

def compute_fi_x(states_q, targets, transition_loc, transition_rate):
    #prox_i = sigma_norm(states_q[0]-targets[0])
    #z_i = prox_i-sigma_norm(transition_loc)
    #f_i = 2/(1 + np.exp(-z_i*transition_rate)) -1
    ##f_i = 1/(1 + np.exp(-z_i*transition_rate))
    f_i = np.minimum(np.maximum(states_q[0] - targets[0], -1), 1)    
     
    return f_i    



#%% functions for twisting

# smush between -1 and +1 with a sigmoid
def smush_neg1pos1(z_i, transition_rate):
    f_i = 2/(1 + np.exp(-z_i*transition_rate)) -1     
    return f_i
    
# smush between 0 and +1 with a sigmoid
def smush_0pos1(z_i, transition_rate):
    f_i = 1/(1 + np.exp(-z_i*transition_rate))      
    return f_i

def compute_fi_n1p1_x(states_qx, targetsx, transition_loc, transition_rate):
    # transition_loc is desired distance from target to center sigmoid
    prox_i = sigma_norm(states_qx-targetsx)
    z_i = prox_i-sigma_norm(transition_loc)
    f_i = 2/(1 + np.exp(-z_i*transition_rate)) -1     
    return f_i

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1


#%% main functions

twist_perp = enforce('statics')


def compute_cmd(states_q, states_p, targets_enc, targets_v_enc, k_node):
    
    u_enc = np.zeros((3,states_q.shape[1]))     
    u_enc[:,k_node] = - c1_d*sigma_1(states_q[:,k_node]-targets_enc[:,k_node])-c2_d*(states_p[:,k_node] - targets_v_enc[:,k_node])    
    
    return u_enc[:,k_node]

#def lemni_target(nVeh,r_desired,lemni_type,lemni_all,state,targets,i,unit_lem,phi_dot_d,ref_plane,quat_0,t,twist_perp):
def lemni_target(nVeh,lemni_all,state,targets,i,t):
    
    # initialize the lemni twist factor
    lemni = np.zeros([1, nVeh])
    
    # if mobbing, offset targets up
    if lemni_type == 2:
        targets[2,:] += r_desired

    # UNTWIST -  each agent has to be untwisted into a common plane
    # -------------------------------------------------------------      
    last_twist = lemni_all[i-1,:] #np.pi*lemni_all[i-1,:]
    state_untwisted = state.copy()
    
    # for each agent 
    for n in range(0,state.shape[1]):
        
        # get the last twist
        untwist = last_twist[n]
        # make a quaternion from it
        untwist_quat = quat.quatjugate(quat.e2q(untwist*unit_lem.ravel()))
        
        # (un)TEST - try the un-rotate (see TEST below for forward rotate) - (pq)* = q*p* not p*q*
        #untwist_quat= quat.quat_mult(untwist_quat,quat_0_)
        # (un)TEST
        
        # pull out states
        states_q_n = state[0:3,n]
        # pull out the targets (for reference frame)
        targets_n = targets[0:3,n] 
        # untwist the agent 
        state_untwisted[0:3,n] = quat.rotate(untwist_quat,states_q_n - targets_n) + targets_n  
             
    # ENCIRCLE -  form a common untwisted circle
    # ------------------------------------------
    
    # compute the untwisted trejectory 
    #targets_encircle, phi_dot_desired_i = encircle_tools.encircle_target(targets, state_untwisted, r_desired, phi_dot_d, ref_plane, quat_0)
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
        
        # compute and store the lemniscate twist factor (tried a bunch of ways to do this)   
        #m_r = np.sqrt(state_untwisted[0,m]**2 + state_untwisted[1,m]**2)
        #m_theta = np.arctan2(state_untwisted[1,m],state_untwisted[0,m]) 
        #m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
        
        # just give some time to form a circle first
        if i > 0:
            
            # compute the lemni factor
            # -----------------------
            
            # if lemni_type == 0:
            #     # compute and store the lemniscate twist factor (tried a bunch of ways to do this)   
            #     m_r = np.sqrt(state_untwisted[0,m]**2 + state_untwisted[1,m]**2)
            #     m_theta = np.arctan2(state_untwisted[1,m],state_untwisted[0,m]) 
            #     m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
            #     lemni[0,m] = m_theta
            
            # lemniscate
            if lemni_type == 0: # travis: this or above - come back and figure out which of these is correct :(
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                lemni[0,m] = m_theta    
            
            # shifting lemninscate    
            if lemni_type == 1: 
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                #m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                m_shift = - np.pi + 0.1*t
                lemni[0,m] = m_theta + m_shift
                
            # mobbing    
            if lemni_type == 2: 
                # compute and store the lemniscate twist factor (tried a bunch of ways to do this)
                m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                lemni[0,m] = m_theta - np.pi 
                
            # experimental 
            if lemni_type == 3:
                
                m_r = np.sqrt((state_untwisted[0,m]-targets[0,m])**2 + (state_untwisted[1,m]-targets[1,m])**2)
                m_theta = np.arctan2(state_untwisted[1,m]-targets[1,m],state_untwisted[0,m]-targets[0,m]) 
                m_theta = np.mod(m_theta, 2*np.pi)  #convert to 0 to 2Pi
                
                if m_theta > np.pi:
                    lemni[0,m] = np.pi/2 
                else:
                    lemni[0,m] = 0
                
                
                

        # twist the trajectory position and load it
        twist = lemni[0,m] #np.pi*lemni[0,m]
        twist_quat = quat.e2q(twist*unit_lem.ravel())
        
        # TEST - take the quaternion product to also rotate by quat_0
        #twist_quat = quat.quat_mult(quat_0,twist_quat)
        # TEST -
        
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
