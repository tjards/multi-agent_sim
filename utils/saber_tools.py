#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Olfati-Saber flocking

Created on Sat Sep 11 10:17:52 2021

@author: tjards

"""

#%% Import stuff
# ------------
import numpy as np


#%% Hyperparameters
# -----------------

# parameters for later functions
a   = 5
b   = 5
c   = np.divide(np.abs(a-b),np.sqrt(4*a*b)) 
eps = 0.1
#eps = 0.5
h   = 0.2 # 0.9 for obs
pi  = 3.141592653589793

# gains
c1_a = 2                # lattice flocking
c2_a = 2*np.sqrt(2)
c1_b = 3                # obstacle avoidance
c2_b = 2*np.sqrt(3)
c1_g = 1                # navigation/target tracking
c2_g = 2*np.sqrt(1)

# key ranges 
d       = 2             # lattice scale (Saber flocking, distance between a-agents)
r       = 2*d           # range at which neighbours can be sensed (Saber flocking, interaction range of a-agents)
d_prime = 0.5 #0.6*d      # desired separation (Saber flocking, distance between a- and b-agents)
r_prime = 2*2*d_prime     # range at which obstacles can be sensed, (Saber flocking, interaction range of a- and b-agents)


#%% Useful functions
# ------------------

def regnorm(z):
    norm = np.divide(z,np.linalg.norm(z))
    return norm

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig
    
def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h
 
def phi_a(q_i, q_j, r_a, d_a): 
    z = sigma_norm(q_j-q_i)        
    phi_a = rho_h(z/r_a) * phi(z-d_a)    
    return phi_a
    
def phi(z):    
    phi = 0.5*((a+b)*sigma_1(z+c)+(a-b))    
    return phi 
        
def a_ij(q_i, q_j, r_a):        
    a_ij = rho_h(sigma_norm(q_j-q_i)/r_a)
    return a_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b
 
def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out

#%% Main functions
# ----------------

# interaction command
# -------------------
def compute_cmd_a(states_q, states_p, targets, targets_v, k_node):
        
    # initialize 
    r_a = sigma_norm(r)                         # lattice separation (sensor range)
    d_a = sigma_norm(d)                         # lattice separation (goal)   
    u_int = np.zeros((3,states_q.shape[1]))     # interactions

    # Lattice Flocking term (phi_alpha)
    # --------------------------------            
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
        # except for itself (duh):
        if k_node != k_neigh:
            # compute the euc distance between them
            dist = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
            # if it is within the interaction range
            if dist < r:
                # compute the interaction command
                u_int[:,k_node] += c1_a*phi_a(states_q[:,k_node],states_q[:,k_neigh],r_a, d_a)*n_ij(states_q[:,k_node],states_q[:,k_neigh]) + c2_a*a_ij(states_q[:,k_node],states_q[:,k_neigh],r_a)*(states_p[:,k_neigh]-states_p[:,k_node]) 

    return u_int[:,k_node]     

# navigation command
# ------------------
def compute_cmd_g(states_q, states_p, targets, targets_v, k_node):

    # initialize 
    u_nav = np.zeros((3,states_q.shape[1]))     # navigation

    # Navigation term (phi_gamma)
    # ---------------------------
    u_nav[:,k_node] = - c1_g*sigma_1(states_q[:,k_node]-targets[:,k_node])-c2_g*(states_p[:,k_node] - targets_v[:,k_node])
  
    return u_nav[:,k_node]

# obstacle avoidance command
# --------------------------
def compute_cmd_b(states_q, states_p, obstacles, walls, k_node):
      
    # initialize 
    d_b = sigma_norm(d_prime)                   # obstacle separation (goal range)
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
    
    # Obstacle Avoidance term (phi_beta)
    # ---------------------------------   
    # search through each obstacle 
    for k_obstacle in range(obstacles.shape[1]):

        # compute norm between this node and this obstacle
        normo = np.linalg.norm(states_q[:,k_node]-obstacles[0:3,k_obstacle])
        
        # ignore if overlapping
        if normo < 0.2:
            continue 
        
        # compute mu
        mu = np.divide(obstacles[3, k_obstacle],normo)
        # compute bold_a_k (for the projection matrix)
        bold_a_k = np.divide(states_q[:,k_node]-obstacles[0:3,k_obstacle],normo)
        bold_a_k = np.array(bold_a_k, ndmin = 2)
        # compute projection matrix
        P = np.identity(states_p.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute beta-agent position and velocity
        q_ik = mu*states_q[:,k_node]+(1-mu)*obstacles[0:3,k_obstacle]
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        if dist_b < r_prime:
            # compute the beta command
            p_ik = mu*np.dot(P,states_p[:,k_node])    
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])
           
    # search through each wall (a planar obstacle)
    for k_wall in range(walls.shape[1]):
        
        # define the wall
        bold_a_k = np.array(np.divide(walls[0:3,k_wall],np.linalg.norm(walls[0:3,k_wall])), ndmin=2).transpose()    # normal vector
        y_k = walls[3:6,k_wall]         # point on plane
        # compute the projection matrix
        P = np.identity(y_k.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute the beta_agent 
        q_ik = np.dot(P,states_q[:,k_node]) + np.dot((np.identity(y_k.shape[0])-P),y_k)
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        maxAlt = 10 # TRAVIS: maxAlt is for testing, only enforces walls below this altitude
        if dist_b < r_prime and states_q[2,k_node] < maxAlt:
            p_ik = np.dot(P,states_p[:,k_node])
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])

        return u_obs[:,k_node] 