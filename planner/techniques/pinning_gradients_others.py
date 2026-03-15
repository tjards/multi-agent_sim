#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implments some useful gradient of potential functions

Created on Thu Nov 21 19:00:54 2024

@author: tjards
"""

#%% import stuff
import numpy as np

def create_gradient_functions(gradients_config):
    
    def grad_lennard_jones(_, states_q, k_node, k_neigh, r_range, d):
        gain = gradients_config.get('LJ_gain', 1000000)
        B = gradients_config.get('LJ_B', 1)
        A = ((d ** 6) * B) / 2
        A = gain * A
        B = gain * B
        
        r = np.linalg.norm(states_q[:, k_node] - states_q[:, k_neigh])
        r_hat = (states_q[:, k_node] - states_q[:, k_neigh]) / r
        return -(-12 * A / r ** 13 + 6 * B / r ** 7) * r_hat
    
    def grad_morse_gradient(_, states_q, k_node, k_neigh, r_range, d):
        beta = gradients_config.get('MORSE_beta', 0.5)
        D = gradients_config.get('MORSE_D', 1)
        r0 = d - 1
        
        r = np.linalg.norm(states_q[:, k_node] - states_q[:, k_neigh])
        r_hat = (states_q[:, k_node] - states_q[:, k_neigh]) / r
        return D * beta * (2 * np.exp(-2 * beta * (r - r0)) - np.exp(-beta * (r - r0))) * r_hat
    
    def grad_gromacs_soft_core(_,states_q, k_node, k_neigh, r_range, d):
        lambda_val = gradients_config.get('gromacs_lambda_val', 0.5)
        A = gradients_config.get('gromacs_A', 1)
        B = gradients_config.get('gromacs_B', 2)
        alpha = gradients_config.get('gromacs_alpha', 0.5)
        beta = gradients_config.get('gromacs_beta', 1.0)
        sigma = r_range
        
        r = np.linalg.norm(states_q[:, k_node] - states_q[:, k_neigh])
        lambda_alpha = lambda_val ** alpha
        r2 = r ** 2
        sc_denom = sigma ** (6 * beta) + (lambda_alpha * r2) ** beta
        sc_denom_3 = sigma ** (3 * beta) + (lambda_alpha * r2) ** beta
        
        grad_attr = -beta * (2 * r) * lambda_alpha * A * (lambda_alpha * r2) ** (beta - 1) / sc_denom ** 2
        grad_rep = -beta * (2 * r) * lambda_alpha * B * (lambda_alpha * r2) ** (beta - 1) / sc_denom_3 ** 2
        gradient = grad_attr - grad_rep
        
        r_hat = (states_q[:, k_node] - states_q[:, k_neigh]) / r
        return -gradient * r_hat
    
    return {
        'lennard_jones': grad_lennard_jones,
        'morse': grad_morse_gradient,
        'gromacs_soft_core': grad_gromacs_soft_core
    }

