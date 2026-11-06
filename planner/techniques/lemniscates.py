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
import config.config as cfg
from .utils import quaternions as quat

#%% Parameters
# -----------
'''
lemni_type descriptions:
            
    # Implicit definition of rotation (https://ieeexplore.ieee.org/document/9931405)
    #   0 = lemniscate of Gerono - 'surveillance'
    #   1 = lemniscate of Gerono - 'rolling'
    #   2 = lemniscate of Gerono - 'mobbing'
        
    # Explicit definition (see https://github.com/tjards/twisted_circles)
    #   3 = lemniscate of Gerono (with shift)
    #   4 = dumbbell curve 
    #   5 = lemniscate of Bernoulli

'''                      

# helpers 
def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

# explicit definition of lemniscate trajectories
def twist_explicit(twist, lemni_type = 3, invert = False):

    twist_quat = np.zeros(4)

    if lemni_type <= 2:
        raise ValueError(f"lemni_type {lemni_type} <=2 not supported for explicit twist.")

    # gerono
    elif lemni_type == 3:
        # compute quaternion
        twist_quat[0] = -np.sqrt(2)*np.sqrt(1 - np.sin(twist))/2
        twist_quat[1] = -np.sqrt(2)*np.sqrt(np.sin(twist) + 1)/2
        if invert:
            twist_quat = quat.quatjugate(twist_quat)

    # dumbbell
    elif lemni_type == 4:
        # compute quaternion
        twist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(twist)**2 + 1)/2
        twist_quat[1] = -np.sqrt(2)*np.sqrt(-(np.cos(twist) - 1)*(np.cos(twist) + 1))/2
        if invert:
            twist_quat = quat.quatjugate(twist_quat)

    # bernoulli
    elif lemni_type == 5:

        # compute quaternion
        twist_quat[0] = -np.sqrt(2)*np.sqrt(np.cos(twist) + 1)/(2*np.sqrt(np.sin(twist)**2 + 1))
        twist_quat[1] = -np.sqrt(2)*np.sqrt(1 - np.cos(twist))/(2*np.sqrt(np.sin(twist)**2 + 1))
        if invert:   
            twist_quat = quat.quatjugate(twist_quat)
    
    else:

        raise ValueError(f"lemni_type {lemni_type} > 5 not supported for explicit twist.")

    return twist_quat


from planner.base import BasePlanner 
class Planner(BasePlanner):
    
    def __init__(self, config_data, **kwargs):
        super().__init__(config_data, **kwargs)

        lemni_config = cfg.get_config(config_data, 'planner.techniques.lemniscates')

        # get lemni hyperparameters
        self.c1_d               = lemni_config.get('c1_d', 1.0)         # position gain 
        self.c2_d               = lemni_config.get('c2_d', 2.0)         # velocity gain
        self.lemni_type         = lemni_config.get('lemni_type', 0)     # lemni type (see above)
        self.learning           = lemni_config.get('learning', None)    # learning method (None, 'CALA')
        self.learning_axes      = lemni_config.get('learning_axes', 'xz')       # axis along which to learn (nominally, xz)
        self.learning_coupling  = lemni_config.get('learning_coupling', True)   # whether to couple learning actions (True/False)
        self.nVeh               = cfg.get_config(config_data, 'agents.nAgents')
        self.last_twist         = np.zeros([2, self.nVeh])  # initialize twist storage

        # encirclement parameters we'll use
        self.circle = kwargs.get('encirclement')  

        # can be tuned
        self.unit_lem    = quat.rotate(self.circle.quat_0, np.array([1, 0, 0]).reshape((3, 1)))     # x-axis reference
        self.twist_perp  = quat.rotate(self.circle.quat_0, np.array([0, 0, 1]).reshape((3,1)))    # z-axis reference
        self.quat_0_     = quat.quatjugate(self.circle.quat_0)   

        # define the embedding planes
        self.x_e = self.unit_lem / np.linalg.norm(self.unit_lem)
        self.z_e = self.twist_perp / np.linalg.norm(self.twist_perp)
        self.y_e = np.cross(self.z_e, self.x_e)
        self.y_e /= np.linalg.norm(self.y_e)

        #graph parameters (standardized in base class)
        circle_config = cfg.get_config(config_data, 'planner.techniques.encirclement')
        r_max = circle_config.get('r_max', 50) 
        self.sensor_range_matrix = r_max * np.ones((self.nVeh, self.nVeh))
        self.connection_range_matrix = r_max * np.ones((self.nVeh, self.nVeh))


        #%% safety checks
        if self.learning == 'CALA':
                
            if 'x' not in self.learning_axes and 'z' not in self.learning_axes:
                raise Exception('warning: no learning axis defined')
            if self.learning_coupling and self.learning_axes != 'xz':
                raise Exception('warning: coupling only supported for xz-axis')
            if self.lemni_type != 0:
                raise Exception('warning: CALA learning only supported for type-0 curves')
        else:
            # force no axis if not learning (redundant)
            self.learning_axes = ''

    # for implicit lemni types
    def twist_implicit(self, angle_x, angle_z, n, invert=False):

        qx = quat.e2q(angle_x[n] * self.unit_lem.ravel())
        qz = quat.e2q(angle_z[n] * self.twist_perp.ravel())

        if not invert:
            quat_out = quat.quat_mult(qz, qx)
        else:        
            quat_out = quat.quatjugate(quat.quat_mult(qz, qx))

        return quat_out

    def check_targets(self, targets):
    
        # if mobbing, offset targets back down
        if self.lemni_type == 2:
            targets[2,:] += self.circle.r_desired/2
        return targets 

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

    def update_trajectory(self, Trajectory, targets, **kwargs):

        state   = kwargs.get('state')
        t       = kwargs.get('t')
        i       = kwargs.get('i')
        learn_actions = kwargs.get('lemni_learn_actions')
        
        Trajectory.trajectory, Trajectory.lemni, Trajectory.sorted_neighs = self.lemni_target(state, targets, i, t, learn_actions)

    def lemni_target(self, state,targets,i,t, learn_actions_coupled):    
        
        lemni       = np.zeros([2, self.nVeh])           # initiate lemni twist vector
        targets     = self.check_targets(targets)   # if mobbing, offset targets up  
        
        # breakout x and z directions
        if self.learning_coupling:
            
            learn_actions = {
            'x': learn_actions_coupled['xz'][0:state.shape[1]],
            'z': learn_actions_coupled['xz'][state.shape[1]::]
            }      
        else:
            learn_actions = learn_actions_coupled
            
        # UNTWIST -  each agent has to be untwisted into a common plane
        # -------------------------------------------------------------      
        last_twist_x = self.last_twist[0, :]

        if 'z' in self.learning_axes:
            last_twist_z = self.last_twist[1, :]
        else:
            last_twist_z = np.zeros(state.shape[1])  # fallback
        
        state_untwisted = state.copy()
        
        # for each agent 
        for n in range(0,state.shape[1]):
            
            # get the last twist
            untwist = last_twist_x[n] # lemni_type >= 3 only uses x_axis
            
            # -------------------------- #
            # do the untwist
            # -------------------------- #

            # untwist 
            if self.lemni_type < 3:
                untwist_quat = self.twist_implicit(last_twist_x, last_twist_z, n, invert=True)
            elif self.lemni_type >= 3 and self.lemni_type <=5:
                untwist_quat = twist_explicit(untwist, self.lemni_type, invert = True)
            else:
                # bypass
                untwist_quat =  np.zeros(4)
                untwist_quat[0] = 1

            # pull out states
            states_q_n = state[0:3,n]
            
            # pull out the targets (for reference frame)
            targets_n = targets[0:3,n] 
            
            # untwist the agent 
            state_untwisted[0:3,n] = quat.rotate(untwist_quat,states_q_n - targets_n) + targets_n
            
        # ENCIRCLE -  form a common untwisted circle
        # ------------------------------------------
        
        # compute the untwisted trejectory 
        targets_encircle, phi_dot_desired_i, sorted_neighs = self.circle.encircle_target(targets, state_untwisted)

        # TWIST - twist the circle
        # ------------------------
        
        # for each agent, we define a unique twist 
        for m in range(0,state.shape[1]):
    
            # generalize to arbitrary planes        
            rel_vec = state_untwisted[0:3, m] - targets[0:3, m]
            x_proj = np.dot(rel_vec, self.x_e)
            y_proj = np.dot(rel_vec, self.y_e)
            m_theta = np.arctan2(y_proj, x_proj)
            #m_theta = np.mod(m_theta, 2*np.pi)

            # ---------------------------
            # define the twist parameter
            # ---------------------------

            # rolling
            if self.lemni_type == 1:  
                m_shift = -np.pi + 0.1 * t
                base_theta = m_theta + m_shift
            
            # mobbing
            elif self.lemni_type == 2:  
                base_theta = m_theta - np.pi
            
            # surveillance + rest
            else:
                base_theta = m_theta #- np.pi/2
                
            # -------------------------- #
            # offset by learned parameter
            # -------------------------- #
            
            lemni[0, m] = base_theta  
            
            if 'x' in self.learning_axes:
                lemni[0, m] += learn_actions.get('x', np.zeros(self.nVeh))[m]
            
            if 'z' in self.learning_axes:
            
                lemni[1, m] = learn_actions.get('z', np.zeros(self.nVeh))[m]

            # -------------------------- #
            # do the twist
            # -------------------------- #

            # twist 
            if self.lemni_type < 3:
                twist_quat = self.twist_implicit(lemni[0, :], lemni[1, :], m, invert=False)
            elif self.lemni_type >= 3 and self.lemni_type <=5:
                twist_quat = twist_explicit(lemni[0, m], self.lemni_type, invert = False)

            # UNDEFINED - falls back to circle 
            else:
                # bypass
                twist_quat =  np.zeros(4)
                twist_quat[0] = 1
      
            # twist positions
            # ---------------
            
            # pull out states/targets
            states_q_i          = state[0:3,m]
            targets_i           = targets[0:3,m]
            target_encircle_i   = targets_encircle[0:3,m]
            
            # get the vector of agent position wrt target
            state_m_shifted             = states_q_i - targets_i
            target_encircle_shifted     = target_encircle_i - targets_i 
    
            #twist_quat = quat.e2q(twist*unit_lem.ravel())        
            twist_pos = quat.rotate(twist_quat,target_encircle_shifted)+targets_i 
            
            # new
            #twist_pos = quat.rotate(quat_0_, twist_pos - targets_i) + targets_i

            targets_encircle[0:3,m] = twist_pos
            
            # twist velocities
            # ----------------
            
            # generalize to arbitrary planes
            w_vector            = phi_dot_desired_i[0,m]*self.twist_perp 
            w_vector_twisted    =  quat.rotate(twist_quat,w_vector) 
            # rotate into embedding plane
            state_m_shifted_emb = quat.rotate(self.quat_0_, state_m_shifted.reshape(3,1)).ravel()
            w_vector_emb        = quat.rotate(self.quat_0_, w_vector_twisted).ravel()
            v_emb               = np.cross(w_vector_emb, state_m_shifted_emb)
            # rotate back
            twist_v_vector      = quat.rotate(self.circle.quat_0, v_emb.reshape(3,1)).ravel()
            
            targets_encircle[3,m] =  - twist_v_vector[0] 
            targets_encircle[4,m] =  - twist_v_vector[1] 
            targets_encircle[5,m] =  - twist_v_vector[2]      

        # store for next iteration
        self.last_twist = lemni.copy()  # store last twist for next iteration
        #self.targets_encircle = targets_encircle
        #self.lemni = lemni
        #self.sorted_neighs = sorted_neighs

        return targets_encircle, lemni, sorted_neighs

