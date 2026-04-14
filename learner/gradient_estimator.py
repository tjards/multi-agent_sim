#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:29:35 2024

This program implments a method for estimating potential function gradients of neighbouring agents


@author: tjards
"""

'''
# these things come from environment
potential_function: Function phi(x_ij) representing the potential
gradient_function: Gradient function of the potential, -∇φ(x_ij)

# this will be estimated through data methods
hessian_function: Hessian function of the potential, ∇²φ(x_ij).
'''


# %% import stuff
# ---------------

import numpy as np
import copy
import random
#random.seed(42)

#%% hyperparameters
# -----------------

gain_gradient_filter = 0.5 #0.5
gain_gradient_control= 30 # default 1
data_driven_hessian = 'linear_filter'
    # 'known'           = For case when hessian is known 
    # 'linear_filter'   = Try a linear filer
    # 'GPR'             = Gaussian Process Regressor
    
# if data_driven_hessian == 'GPR':
    
#     from sklearn.gaussian_process import GaussianProcessRegressor
#     from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

 
#%% filters
# --------
def low_pass_filter(current_value, previous_value, alpha):
    """Applies exponential moving average."""
    return alpha * current_value + (1 - alpha) * previous_value

filter_v_gain = 50
def filter_v(v_filtered, v, Ts):
    
    v_filtered_dot  = -filter_v_gain*v_filtered + filter_v_gain*v
    v_filtered      = v_filtered + Ts*v_filtered_dot
    
    return v_filtered_dot, v_filtered

def filter_C(C_filtered, C_estimate, Ts):
    
    C_filtered_dot  = -filter_v_gain*C_filtered + C_estimate 
    C_filtered      = C_filtered + Ts*C_filtered_dot
    
    return C_filtered_dot, C_filtered 



#%%
class GradientEstimator:
    
    # initialize
    def __init__(self, nAgents, dimens, Ts):
        
        """
        Arguments:
            nAgents = number of agents
            Ts      = step size
            dimens  = numbers of dimensions (2D, 3D)
        """
        self.nAgents                = nAgents
        self.Ts                     = Ts
        self.dimens                 = dimens
        self.gradient_estimates     = np.zeros((dimens, nAgents, nAgents))
        self.observed_gradients     = np.zeros((dimens, nAgents, nAgents))
        self.last_velos             = np.zeros((dimens, nAgents, nAgents))
        self.gain_gradient_control  = gain_gradient_control
        
        # for filters, each agent stores for each neighbour
        #   (dimens, observed by, observed of)
        self.C                  = np.zeros((dimens, nAgents, nAgents))
        self.C_filtered         = np.zeros((dimens, nAgents, nAgents))
        self.v_filtered         = np.zeros((dimens, nAgents, nAgents))
        self.v_dot_filtered     = np.zeros((dimens, nAgents, nAgents))
        self.C_dot_filtered     = np.zeros((dimens, nAgents, nAgents))
        
        # pin controller (sum of everything in that component)
        self.C_sum              = np.zeros((dimens, nAgents))
        self.C_sum_bypin       = np.zeros((dimens, nAgents)) 
        
    def update_estimates(self, states_q, states_p, observed_gradients, A, k_node):
        
        """
        Arguments:
            states_q    = array of agent positions, shape (dimens, nAgents)
            states_p    = array of agent velocities, shape (dimens, nAgents)
            observed_gradients   = gradients observed by this agent: ((dimens, nAgents))
            hessians             = hessians  observed by this agent (this will be from a data-driven model later) ((dimens, nAgents))
            A           = adjacency matrix
            k_node      = index for this agent 

        Returns:
            Updated gradient estimates for all agent pairs.
        """
        
        # for each neighbour
        for k_neigh in range(0, self.nAgents):
            
            # check if the neighbour is in range
            # ---------------------------------
            
            if k_neigh != k_node:
            
                # use adjacency matrix (new)
                if A[k_node,k_neigh] == 0:
                    in_range = False
                else:
                    in_range = True 
            
                # if within range
                # ---------------
                if in_range:
                    
                    # relative states
                    #x_ij = states_q[0:self.dimens,k_neigh] - states_q[0:self.dimens,k_node]
                    #v_ij = states_p[0:self.dimens,k_neigh] - states_p[0:self.dimens,k_node]
                
                    # pull current gradient 
                    current_gradient = self.gradient_estimates[0:self.dimens, k_node, k_neigh]
                    
                        
                    if data_driven_hessian == 'linear_filter':
                        
                        # v filter
                        v_dot_filtered, v_filtered = filter_v(self.v_filtered[0:self.dimens, k_node, k_neigh], states_p[:,k_neigh], self.Ts)
                        self.v_dot_filtered[0:self.dimens, k_node, k_neigh] = v_dot_filtered
                        self.v_filtered[0:self.dimens, k_node, k_neigh] = v_filtered 
                        
                        # C filter
                        C_dot_filtered, C_filtered = filter_C(self.C_filtered[0:self.dimens, k_node, k_neigh], current_gradient, self.Ts)
                        self.C_dot_filtered[0:self.dimens, k_node, k_neigh] = C_dot_filtered
                        self.C_filtered[0:self.dimens, k_node, k_neigh] = C_filtered 
                        
                        gradient_dot = C_dot_filtered
   
                    #self.gradient_estimates[0:self.dimens, k_node, k_neigh] = C_filtered
                    
    
                    # find gradient_dot via filters
                    # -----------------------------
                    
                    # predict the next gradient
                    predicted_gradient = current_gradient + gradient_dot * self.Ts
                    
                    # pull the observed gradient 
                    observed_gradient = observed_gradients[0:self.dimens, k_neigh]
                    #observed_gradient = low_pass_filter(observed_gradients[:, k_neigh], self.gradient_estimates[k_node, k_neigh],alpha=0.1)
                    
                    # correct with innovation
                    updated_gradient = (1-gain_gradient_filter) * predicted_gradient + gain_gradient_filter * (observed_gradient - predicted_gradient)
                    
                    #print(self.C_dot_filtered)
                    
                    # load
                    self.gradient_estimates[0:self.dimens, k_node, k_neigh] = updated_gradient
                    
                    # sum to the total for this node, all neighbours (will be feed to pin)
                    self.C_sum[0:self.dimens, k_node] += updated_gradient
                    
                              
                        
            
#%% Example usage with a simple potential function
def potential_function(x):
    """Example Lennard-Jones potential."""
    r = np.linalg.norm(x)
    return 4 * ((1 / r)**12 - (1 / r)**6)

def gradient_function(x):
    """Gradient of the Lennard-Jones potential."""
    r = np.linalg.norm(x)
    if r == 0:
        return np.zeros_like(x)
    factor = 4 * (12 / r**13 - 6 / r**7)
    return factor * x / r

def hessian_function(x, dimens):
    """Hessian of the Lennard-Jones potential."""
    r = np.linalg.norm(x)
    if r == 0:
        return np.zeros((dimens, dimens))
    factor1 = 4 * (156 / r**14 - 42 / r**8)
    factor2 = -4 * (12 / r**14 - 6 / r**8)
    outer = np.outer(x, x) / r**2
    return factor1 * outer + factor2 * np.eye(2)

