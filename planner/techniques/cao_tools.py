#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:58:14 2024

this project implements the following:

[1] Yi Dong, Jie Huang,
Flocking with connectivity preservation of multiple double integrator systems \
subject to external disturbances by a distributed control law, Automatica
2015

[2] Zhang et al., Flocking Control Against Malicious Agent
IEEE TRANSACTIONS ON AUTOMATIC CONTROL, VOL. 69, NO. 5, MAY 2024

[3] Housheng Sua, Xiaofan Wanga, and Guanrong Chenb
A connectivity-preserving flocking algorithm for multi-agent systems
based only on position measurements
International Journal of Control
Vol. 82, No. 7, July 2009, 1334–1343


dev notes:
    
    - create a custom class
    - 2 x subclasses (member, malicious), inherit most common things
    
    - automatically select the agent with max degree centrality (i.e. the natural pin) as the malicious agent 
    - compute E using Eqn (3) based on initial states (i.e., when the Class is initialized)
    - regular agent implemented using Eqn (3)
    - malicious agent implemented using Eqn (4, 5)
    - HOW TO ID malicious agent?
    - agents surrounding malicious agent use Eqn (15), which relies on (14)
    - outer layer use Eqn (17)
    - HOW to decide which later you’re in?
    
    - paper validation is kind of contrived
    - us graph methods to identify agent 1-hop out from malicious, then 1 hop out from them for layer 2
    - 3d?


@author: tjards
"""

# import stuff
# -----------
import numpy as np

# parameters
# ---------- 

d =  5                          # desired separation
r = np.divide(2*d, np.sqrt(2))  # sensor range (adjust this later, derived from desired separation now)
#Q = 10 #0.01                 # could be computed using Eqn (3) from [2]
cmd_min = -10
cmd_max = 10

# gains
gain_p = 0.2
gain_v = 1

# malicious agent type
mal_type = 'collider'  # runaway, collider

if mal_type == 'runaway':
    mal_kv = -0.01
    mal_ka = -0.01
    mal_kr = 0.1
elif mal_type == 'collider':
    mal_kv = -6
    mal_ka = 15
    mal_kr = -5 
    

def return_ranges():
    return r

def return_desired_sep():
    return d

# define a custom class 
# ---------------------
class Flock:
    
    def __init__(self, states_q, states_p):
        
        self.status = ['friendly'] * states_q.shape[1] # stores status, assume all friendly initially
        self.cmd_i  = np.zeros((states_q.shape[1],3))
        self.d      = d
        self.r      = r
        self.Q      = 1.1*compute_E(states_q, states_p)
        self.assembled = 0 # keeps track of when the swarm is assembled (first time)
        
    def update_Q(self,states_q, states_p):
        
        self.Q      = 1.1*compute_E(states_q, states_p)
        
    # compute commands (Eqn (12) from [1] )
    # ----------------
    def compute_cmd(self, targets, states_q, states_p, k_node, **kwargs):
        
        # extract adjacency matrix
        A           = kwargs['A']
        pin_matrix  = kwargs['pin_matrix']
        
        # if this is the first time assembled
        if np.sum(pin_matrix) == 1 and self.assembled == 0:
            # mark as assembled
            self.assembled = 1
            # update Q
            self.Q      = 1.1*compute_E(states_q, states_p)
            # chose a malicious agent
            malicious = np.where(np.diag(pin_matrix) == 1)[0][0] 
            self.status[k_node - 1] = 'malicious'    
            
        
        # set parameters for friendly or malicious agent
        if self.status[k_node] == 'friendly':
            gains = [1,1,1]
        else:
            gains = [mal_kv, mal_ka, mal_kr]
        
        # initialize
        cmd_i = np.zeros((3))
        
        # compute navigation (if not assembled, draws towards)
        #if self.assembled == 0:
        cmd_i -= pin_matrix[k_node,k_node]*compute_navigation(states_q, states_p, targets, k_node)
        
        # search through each neighbour
        for k_neigh in range(states_q.shape[1]):
            
            # except for itself:
            if k_node != k_neigh:
                
                # check if the neighbour is in range
                if A[k_node,k_neigh] == 0:
                    in_range = False
                else:
                    in_range = True 
                
                # if within range
                if in_range:
                    
                    # compute alignment 
                    cmd_i -= gains[0]*compute_alignment(states_q, states_p, A, k_node, k_neigh)
                    # compute cohesion
                    cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                    # compute repulsion
                    cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
        
        cmd_i = np.clip(cmd_i, cmd_min, cmd_max)
        
        return cmd_i        

      
# # compute commands (Eqn (12) from [1] )
# # ----------------
# def compute_cmd(targets, states_q, states_p, k_node, **kwargs):
    
#     # extract adjacency matrix
#     A           = kwargs['A']
#     pin_matrix  = kwargs['pin_matrix']
    
#     #Q = compute_E(states_q, states_p, A)
        
#     # initialize
#     cmd_i = np.zeros((3))
    
#     # compute navigation
#     cmd_i -= pin_matrix[k_node,k_node]*compute_navigation(states_q, states_p, targets, k_node)
    
#     # search through each neighbour
#     for k_neigh in range(states_q.shape[1]):
        
#         # except for itself:
#         if k_node != k_neigh:
            
#             # check if the neighbour is in range
#             if A[k_node,k_neigh] == 0:
#                 in_range = False
#             else:
#                 in_range = True 
            
#             # if within range
#             if in_range:
                
#                 # compute alignment 
#                 cmd_i -= compute_alignment(states_q, states_p, A, k_node, k_neigh)
#                 # compute cohesion
#                 cmd_i -= compute_cohesion(states_q, k_node, k_neigh, Q)*(states_q[:,k_node] - states_q[:,k_neigh])
#                 # compute repulsion
#                 cmd_i -= compute_repulsion(states_q, k_node, k_neigh, Q)*(states_q[:,k_node] - states_q[:,k_neigh])
    
#     cmd_i = np.clip(cmd_i, cmd_min, cmd_max)
    
#     return cmd_i

# compute alignment command
# -------------------------
def compute_alignment(states_q, states_p, A, k_node, k_neigh):
             
    # add the contribution from this agent
    u_i_align = (states_p[:,k_node] - states_p[:,k_neigh])

    return u_i_align
    
# compute cohesion command
# ------------------------
def compute_cohesion(states_q, k_node, k_neigh, Q):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
    #u_i_cohes = np.divide(2*s*(r**2 + (r**2)/Q), np.square(r**2 - s**2 + (r**2)/Q ))
    u_i_cohes = np.divide(2*(r**2 + (r**2)/Q), np.square(r**2 - s**2 + (r**2)/Q ))
    
    return u_i_cohes
    
# compute repulsion command
# ------------------------
def compute_repulsion(states_q, k_node, k_neigh, Q):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
    #u_i_repul = np.divide(-2*s*(r**2 + (r**2)/Q), np.square(s**2 + (r**2)/Q ))
    u_i_repul = np.divide(-2*(r**2 + (r**2)/Q), np.square(s**2 + (r**2)/Q ))
    
    return u_i_repul
    
# compute navigation command
# ------------------------
def compute_navigation(states_q, states_p, targets, k_node):
    
    u_i_navig = np.zeros((3))
    u_i_navig = -gain_p*(targets[:,k_node] - states_q[:,k_node]) + gain_v*(states_p[:,k_node])
      
    return u_i_navig

# compute potential function bar 
# ------------------------------
def potential_function_bar(R, x):
    V = np.divide(R**2 - x**2, x**2) + np.divide(x**2, R**2 - x**2)
    return V

# compute Q (aka E) ref: Eqn (3) from [2]
# ---------------------------------------
def compute_E(states_q, states_p):
    v_sum = 0
    V_max = 0
    N = states_q.shape[1]
    # for each agent
    for k_node in range(states_q.shape[1]):
        v_sum += np.dot(states_p[:,k_node].transpose(),states_p[:,k_node])
        
        # search through each neighbour
        for k_neigh in range(states_q.shape[1]):
            # except for itself:
            if k_node != k_neigh:
                V_new =  potential_function_bar(r, np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh]))
                if V_new > V_max:
                    V_max = V_new
                # # check if the neighbour is in range
                # if A[k_node,k_neigh] == 0:
                #     in_range = False
                # else:
                #     in_range = True 
                # # if within range
                # if in_range:
    E = 0.5*v_sum + np.divide((N*(N-1)),2)*V_max
    
    return E
                    
                    

    
    
    