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
    
    - * automatically select the agent with max degree centrality (i.e. the natural pin) as the malicious agent 
    - * compute E using Eqn (3) based on initial states (i.e., when the Class is initialized)
    - * regular agent implemented using Eqn (3)
    - * malicious agent implemented using Eqn (4, 5)
    - HOW TO ID malicious agent?
    - * identify layers
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

d =  10                          # desired separation
#r = np.divide(2*d, np.sqrt(2))  # sensor range (adjust this later, derived from desired separation now)
r = np.multiply(d, np.sqrt(2))
#Q = 10 #0.01                 # could be computed using Eqn (3) from [2]
cmd_min = -10
cmd_max = 10

# gains
gain_p = 0.2
gain_v = 1

kv = 1
ka = 1
kr = 1


# malicious agent type
mal_type = 'collider'  # runaway, collider

if mal_type == 'runaway':
    mal_kv = -0.01
    mal_ka = -0.01
    mal_kr = 0.1
elif mal_type == 'collider':
    mal_kv = -5
    mal_ka = 75
    mal_kr = -10 
elif mal_type == 'cooperative':
    mal_kv = kv
    mal_ka = ka
    mal_kr = kr
    
    
def return_ranges():
    return r

def return_desired_sep():
    return d

# define a custom class 
# ---------------------

class Flock:
    
    def __init__(self, states_q, states_p):
        
        self.status = ['friendly'] * states_q.shape[1] # stores status, assume all friendly initially (use nested dicts, later, for multiple malicious agents)
        #self.status = {i:'friendly' for i in range(0,states_q.shape[1]) }
        self.layer = [0] * states_q.shape[1] # stores layer, relative to malicious agent
        self.cmd_i  = np.zeros((states_q.shape[1],3))
        self.d      = d
        self.r      = r
        self.Q      = 1.1*compute_E(states_q, states_p)
        self.assembled = 0 # keeps track of when the swarm is assembled (first time)
        
    # checks if layer 2 has at least 2 agents (Assumption 3)
    def check_assume_3(self):
        count_layer_2 = self.layer.count(2)
        if count_layer_2 > 2:  
            return True
        else:
            return False 
    
    # checks if at least 2 agents in layer 2 are neighbours
    def check_assume_4(self, A):
        layer_2_indices = [index for index, value in enumerate(self.layer) if value == 2]
        count = 0
        for node in layer_2_indices:
            for neighbour in layer_2_indices:
                if node != neighbour:
                    if A[node,neighbour] > 0:
                        count += 1
        if count > 2:
            return True
        else:
            return False
        
    def update_Q(self,states_q, states_p):
        
        self.Q      = 1.1*compute_E(states_q, states_p)
        
    # compute commands (Eqn (12) from [1] )
    # ----------------
    def compute_cmd(self, targets, states_q, states_p, k_node, **kwargs):
        
        # extract adjacency matrix
        A               = kwargs['A']
        A_connectivity  = kwargs['A_connectivity']
        pin_matrix      = kwargs['pin_matrix']
        
        # upon first assembly, induce a virtual agent
        if np.sum(pin_matrix) == 1 and self.assembled == 0:           
            # mark as assembled
            self.assembled = 1
            # update Q
            self.Q      = 1.1*compute_E(states_q, states_p)
            # chose a malicious agent (i.e., the pin)
            #malicious = np.where(np.diag(pin_matrix) == 1)[0][0] 
            malicious = 2
            self.status[malicious] = 'malicious'
            # malicious agent is layer 1
            self.layer[malicious] = 1
            
            # ********* layer assignment ************** #
            # immediate neighbours to malicious agent are layer 2
            for neighbour in range(0,states_q.shape[1]):
                if neighbour != malicious:
                    # if connected to malicious agent
                    #if A[malicious,neighbour] > 0:
                    if A_connectivity[malicious,neighbour] > 0:
                        # mark as inner layer
                        self.layer[neighbour] = 2
                        # neighbours of this immediate neighbours are layer 3
                        remaining = [index for index, value in enumerate(self.layer) if value == 0]
                        # for all unassigned agents
                        for neighbour_outer_option in remaining:
                            # if connected to this neighbour
                            #if A[neighbour,neighbour_outer_option] > 0:
                            if A_connectivity [neighbour,neighbour_outer_option] > 0:
                                # mark as outer layer
                                self.layer[neighbour_outer_option] = 3
            # check if valid conditions
            self.assumptions_valid = False
            if self.check_assume_3():
                if self.check_assume_4(A):
                    self.assumptions_valid = True
            print('Layers defined. Assumption validity: ', self.assumptions_valid)    
            
            print(self.status)
            print(self.layer)
            # ******************************************* #

    
        # set parameters for friendly or malicious agent
        if self.status[k_node] == 'friendly':
            gains = [kv,ka,kr]
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

# ***************** #
#  BASIC FLOCKING   #
# ***************** #
     
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
                    
# ***************** #
#  COUNTER MALIGN   #
# ***************** #       

#%%

'''
Definitions:
    Layer 1 = Malicious Agent
    Layer 2 = Neighbours of Malicious Agent 
    Layer 3 = All remaining agents 2 hops out from the Malicious agent
    Layer 0 = Everyone else
    
    set V_l = agents in Layer 2
    set V_f = agents in Layer 3
    set V_g = agents in Layer 1 and 2
    
Assumptions:
    (3) Layer 2 must have at least 2 agents
    (4) At least 2 agents in Layer 2 are neighbours
    
'''


mal_kv_hat = -5.5
mal_ka_hat = 75.5
mal_kr_hat = -10.5



#%%
    
H_bar = 10 # tunable
i_cont = 1 # tunable
H = H_bar + i_cont

# solution from wolframalpha (paper didn't provide)
def potential_function_hat(s, a, r, d, H):
    
    num1 = (s-a)*(np.divide(s*(s-a),np.linalg.norm(s)) + 2*(-np.linalg.norm(s) + (1/H)*(d-r)**2 +r))
    den1 = (r - np.linalg.norm(s) + (1/H)*(d-r)**2)**2
    num2 = H*(s-a)*(H*s*(a+s)+2*np.linalg.norm(s)*d**2)
    den2 = np.linalg.norm(s)*(H*np.linalg.norm(s) + d**2)**2
    
    return num1/den1 + num2/den2


# test
d =  5                          # desired separation
r = np.multiply(d, np.sqrt(2))  # range
d_bar = np.divide(d,np.sqrt(3))                   # malicious agent separation

# test
x_i = np.array((-1.44,-2.50,0))
x_j = np.array((2.80,0,0))
x_m = np.array((0,0,0))
x_ij = x_i - x_j
x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))

test = potential_function_hat(x_ij, x_ij_star, r, d_bar, H)




#part1a = np.divide(2*(-a+s),                                r-np.abs(s)+((-d+r)**2)/H )
#part1b = np.divide(s*(-a+s)**2,                             np.abs(s)*(r-np.abs(s)+((-d+r)**2)/H)**2)
#part2 = np.divide(H*(-a+s)*(H*s*(a+s)+2*(d**2)*np.abs(s)),  np.abs(s)*((d**2)+H*np.abs(s))**2)

# next step, build the vector (10) and compare to the computed_cmd above
# and build filters in (11) and (12)



#Gamma = np.diag([mal_kv_hat,mal_ka_hat,mal_kr_hat])             

    
    
    