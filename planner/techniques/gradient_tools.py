#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implments some useful gradient of potential functions

Created on Thu Nov 21 19:00:54 2024

@author: tjards
"""

#%% import stuff
import numpy as np
from config.configs_tools import update_configs  

#%%  parameters

# lennard-jones
LJ_gain = 1000000
LJ_B = 1

# morse
MORSE_beta = 0.5
MORSE_D = 1

# gromacs
gromacs_lambda_val = 0.5
gromacs_A = 1 
gromacs_B = 2
gromacs_alpha= 0.5
gromacs_beta= 1.0


#%% configs load

configs_entries = [
('LJ_B', LJ_B),
('LJ_gain', LJ_gain),
('MORSE_beta', MORSE_beta),
('MORSE_D', MORSE_D),
('gromacs_lambda_val', gromacs_lambda_val),
('gromacs_A', gromacs_A),
('gromacs_B', gromacs_B),
('gromacs_alpha', gromacs_alpha),
('gromacs_beta', gromacs_beta)
]

update_configs('gradient_parameters',  configs_entries)
    
#%% Gradients
# ----------

# GROMACS soft-core function (smooths out the potential at small distances)
# (something isn't working here)
def grad_gromacs_soft_core(states_q, k_node, k_neigh, r_range, d):
    
    lambda_val = gromacs_lambda_val #0.5
    A = gromacs_A #1 # these need to be tuned for d
    B = gromacs_B #2
    alpha = gromacs_alpha #0.5
    beta = gromacs_beta #1.0
    sigma = r_range #= 0.3
    r = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
     
    """
    Parameters:
    - r: distance between particles
    - lambda_val: coupling parameter (0 <= lambda <= 1) (higher, amplifies entire potential)
    - A: constant for the attractive term (higher rel to B, tighter)
    - B: constant for the repulsive term (higher rel to A, erratic when closer)
    - alpha: soft-core scaling parameter (higher, more sensitivit to lambda)
    - beta: soft-core exponent (lower, smoother transitions )
    - sigma: effective radius for soft-core interactions (higher, agents sense eachother more )

    """
    # Precompute factors for efficiency
    lambda_alpha = lambda_val**alpha
    r2 = r**2
    sc_denom = sigma**(6*beta) + (lambda_alpha * r2)**beta
    sc_denom_3 = sigma**(3*beta) + (lambda_alpha * r2)**beta

    # Compute the potential terms
    v_attr = A / sc_denom
    v_rep = B / sc_denom_3

    # Compute the derivative terms
    grad_attr = -beta * (2 * r) * lambda_alpha * A * (lambda_alpha * r2)**(beta - 1) / sc_denom**2
    grad_rep = -beta * (2 * r) * lambda_alpha * B * (lambda_alpha * r2)**(beta - 1) / sc_denom_3**2

    # Combine gradients
    gradient = grad_attr - grad_rep
    
    r_hat = (states_q[:,k_node]-states_q[:,k_neigh])/r
    
    
    return - gradient*r_hat

def grad_morse_gradient(states_q, k_node, k_neigh, r_range, d):
    
    beta = MORSE_beta #0.5
    D = MORSE_D #1
    r0 = d-1
    
    r = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
    r_hat = (states_q[:,k_node]-states_q[:,k_neigh])/r
    
    return D * beta * (2 * np.exp(-2 * beta * (r - r0)) - np.exp(-beta * (r - r0))) * r_hat


def grad_lennard_jones(states_q, k_node, k_neigh, r_range, d):
    
    # compute A and B
    gain = LJ_gain #1000000
    B = LJ_B #1
    A = ((d**6)*B)/2
    
    A = gain* A #1 # these need to be tuned for d
    B = gain*B #2
    r = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
    r_hat = (states_q[:,k_node]-states_q[:,k_neigh])/r
    
    return -(-12 * A / r**13 + 6 * B / r**7) * r_hat


