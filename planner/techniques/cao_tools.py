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

d =  18                         # desired separation
#r = np.divide(2*d, np.sqrt(2))  # sensor range (adjust this later, derived from desired separation now)
r = np.multiply(d, np.sqrt(2))
#Q = 10 #0.01                 # could be computed using Eqn (3) from [2]
cmd_min = -5000
cmd_max = 5000

# gains
gain_p = 0.1# default zero (target tracking messes up this technique)
gain_v = 1

kv = 5
ka = 1
kr = 2

kx = 10 #2

d_bar = r/2 #np.divide(d,np.sqrt(3)) # malicious agent separation
#H_bar = 100 # tunable
i_cont = 0.2 # tunable
#H = H_bar + i_cont

gamma_kp = 2 # Layer 3 gamma constant


# malicious agent type
mal_type = 'runaway'  # runaway, collider

if mal_type == 'runaway':
    mal_kv = 0.8# 0.8
    mal_ka = -0.1 #-0.1
    mal_kr = 450000 #500
elif mal_type == 'collider':
    mal_kv = -5
    mal_ka = 200
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
        self.mode_malicious = 0 # 0 = default (normal flocking), 1 = switches to layer-based malicious mode
        self.malicious = 2 # the malicious agent
        self.gain_p = gain_p# default zero (target tracking messes up this technique)
        self.gain_v = gain_v# default zero (target tracking messes up this technique)
        self.alpha_kp = gamma_kp*np.ones((states_q.shape[1],states_q.shape[1]))#[gamma_kp] * states_q.shape[1] # stores alpha for Layer3

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
        #A  = kwargs['A_connectivity']
        pin_matrix      = kwargs['pin_matrix']
        Ts              = kwargs['Ts']
        
        # upon first assembly, induce a virtual agent
        if np.sum(pin_matrix) == 1 and self.assembled == 0:           
            # mark as assembled
            self.assembled = 1
            # update Q
            self.Q      = 1.1*compute_E(states_q, states_p)
            # chose a malicious agent (i.e., the pin)
            malicious = np.where(np.diag(pin_matrix) == 1)[0][0] 
            #malicious = self.malicious
            self.malicious = malicious 
            self.status[malicious] = 'malicious'
            # malicious agent is layer 1
            self.layer[malicious] = 1
            # activate response to malicious node ( = 1)
            self.mode_malicious = 1
            self.gain_p = 0 # turn off navigation in malicious mode (messes with results)
            self.gain_v = 1 # turn off navigation in malicious mode (messes with results)
            # ********* layer assignment ************** #
            # immediate neighbours to malicious agent are layer 2
            for neighbour in range(0,states_q.shape[1]):
                if neighbour != malicious:
                    # if connected to malicious agent
                    if A[malicious,neighbour] > 0:
                    #if A_connectivity[malicious,neighbour] > 0:
                        # mark as inner layer
                        self.layer[neighbour] = 2
                        # neighbours of this immediate neighbours are layer 3
                        remaining = [index for index, value in enumerate(self.layer) if value == 0]
                        # for all unassigned agents
                        for neighbour_outer_option in remaining:
                            # if connected to this neighbour
                            if A[neighbour,neighbour_outer_option] > 0:
                            #if A_connectivity [neighbour,neighbour_outer_option] > 0:
                                # mark as outer layer
                                self.layer[neighbour_outer_option] = 3
            layer2_indices = [index for index, value in enumerate(self.layer) if value == 2]
            self.H      = max(compute_H(states_q, states_p, self.malicious, layer2_indices, kx, np.array([mal_kv, mal_ka, mal_kr]), A), 100)
            if self.H < 0:
                raise ValueError("H cannot be negative")
                
            
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
        cmd_i -= pin_matrix[k_node,k_node]*compute_navigation(self.gain_p, self.gain_v, states_q, states_p, targets, k_node)

        # if no responding to malicious agent
        # ----------------------------------
        if self.mode_malicious == 0:
        
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
                        cmd_i -= gains[0]*compute_alignment(states_q, states_p, k_node, k_neigh)
                        # compute cohesion
                        cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                        # compute repulsion
                        cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                
        # if we are responding to a malicious node 
        # --------------------------------------
        if self.mode_malicious == 1:
            
            # for layer 2
            # ------------
            if self.layer[k_node] == 2: #and self.layer[k_neigh] == 2:
                        
                for k_neigh in [index for index, value in enumerate(self.layer) if value == 2]:
                        
                    # if this neighbour is also in layer 2 and not the agent itself (redundant)
                    if k_neigh != k_node:
                        
                        # check if the neighbour is in range
                        if A[k_node,k_neigh] == 0:
                            in_range = False
                        else:
                            in_range = True 
                    
                        #if within range
                        if in_range:
                        
                            # compute velo component 
                            cmd_i -= gains[0]*compute_alignment(states_q, states_p, k_node, k_neigh)
                        
                            # compute modified nonnegative potential function
                            x_i = states_q[:,k_node]
                            x_j = states_q[:,k_neigh]
                            x_ij = x_i - x_j
                            #x_m =  states_q[:,self.malicious]
                            
                            x_ij_star = compute_x_star(d, x_i, x_j)
                            #x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))
                            #print(np.linalg.norm(x_ij_star))
                            
                            #print('Agent ', k_node, ': ', np.linalg.norm(x_ij))
                            cmd_i -= kx*potential_function_hat(x_ij, x_ij_star, r, d_bar, self.H+i_cont) #*(x_ij)
         
                # malicious agent gain part (should be an estimate later)
                #pass_in_neighs_list = [index for index, value in enumerate(self.layer) if value == 2]
                
                malicious_k_hat = np.array([mal_kv,mal_ka,mal_kr])              
                cmd_i -=  Ck_malicious_v2(states_q, states_p, A, self.Q, self.malicious, malicious_k_hat)
               
                
                #cmd_i -= Ck_malicious(states_p, states_q, self.malicious, pass_in_neighs_list , self.Q, malicious_gains)
                #print('malicious cmd estimate node', k_node, ': ', -Ck_malicious_v2(states_q, states_p, A, self.Q, self.malicious, malicious_k_hat))
                
                #cmd_i -= np.dot(C_if,np.array([mal_kv,mal_ka,mal_kr]))
                            
            # for layer 3 
            # ----------------------------
            elif self.layer[k_node] == 3: # or self.layer[k_node] == 0:
                
                # search through each neighbour
                for k_neigh in range(states_q.shape[1]):
                #for k_neigh in [index for index, value in enumerate(self.layer) if (value == 3 or value ==0)]:
                
                    # except for itself:
                    if k_node != k_neigh:
                        
                        # check if the neighbour is in range
                        if A[k_node,k_neigh] == 0:
                            in_range = False
                        else:
                            in_range = True 
                    
                        #if within range
                        if in_range:
                            
                            alpha_dot = gamma_kp*np.sum(np.abs(states_p[:,k_node]-states_p[:,k_neigh])) # L-1 norm
                            self.alpha_kp[k_node, k_neigh] = alpha_dot #+= (alpha_dot*Ts).item()
  
                            
                            #print(self.alpha_kp[k_node, k_neigh])
                            
                            
                            # compute alignment 
                            cmd_i -= (self.alpha_kp[k_node, k_node])*np.sign(states_p[:,k_node]-states_p[:,k_neigh])
                            # compute cohesion
                            cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                            # compute repulsion
                            cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh]) 
                                

            # for layer 1 (malicious agent) 
            # ----------------------------
            elif self.layer[k_node] <= 1:
                
                # search through each neighbour
                for k_neigh in range(states_q.shape[1]):
                
                    # except for itself:
                    if k_node != k_neigh:
                        
                        # check if the neighbour is in range
                        if A[k_node,k_neigh] == 0:
                            in_range = False
                        else:
                            in_range = True 
                    
                        # if within range, do your thing
                        if in_range:
                        
                            # compute alignment 
                            cmd_i -= gains[0]*compute_alignment(states_q, states_p, k_node, k_neigh)
                            # compute cohesion
                            cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                            # compute repulsion
                            cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q)*(states_q[:,k_node] - states_q[:,k_neigh])   
                            
                #print('malicious cmd node ', k_node, ': ', cmd_i)
    
        cmd_i = np.clip(cmd_i, cmd_min, cmd_max)
        
             
        return cmd_i        

# ***************** #
#  BASIC FLOCKING   #
# ***************** #
     
# compute alignment command
# -------------------------
def compute_alignment(states_q, states_p, k_node, k_neigh):
             
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
def compute_navigation(gain_p, gain_v, states_q, states_p, targets, k_node):
    
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
    
Use CALA to expand lattice to overcome these assumptions
    
'''

# Layer 2 collective del-potential function (from wolframalpha, paper didn't provide)
def potential_function_hat(s, a, r, d, H):
    
    num1 = (s-a)*(np.divide(s*(s-a),np.linalg.norm(s)) + 2*(-np.linalg.norm(s) + (1/H)*(d-r)**2 +r))
    den1 = (r - np.linalg.norm(s) + (1/H)*(d-r)**2)**2
    num2 = H*(s-a)*(H*s*(a+s)+2*np.linalg.norm(s)*d**2)
    den2 = np.linalg.norm(s)*(H*np.linalg.norm(s) + d**2)**2
    
    return num1/den1 + num2/den2


# def compute_x_star(d_bar, x_i, x_j, x_m):
    
#     x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))

#     return x_ij_star


def compute_x_star(d, x_i, x_j):
    
    x_ij_star = d*((x_i - x_j)/np.linalg.norm(x_i - x_j))
    
    return x_ij_star
                  
            
def V_hat_prime(x_ij, x_ij_star, r):
    
    V_hat_prime = np.divide((np.linalg.norm(x_ij - x_ij_star))**2, r - np.linalg.norm(x_ij)) + np.divide((np.linalg.norm(x_ij - x_ij_star))**2, np.linalg.norm(x_ij)) 
    
    return V_hat_prime

def compute_H(states_q, states_p, malicious_index, layer2_indices, kx, k_vector, A):
    
    H = 0
    
    x_m = states_q[:,malicious_index]
    v_m = states_p[:,malicious_index]

    sum1 = 0
    for j in layer2_indices: 
    
        x_j = states_q[:,j]
        x_jm = x_j - x_m
        x_jm_star = compute_x_star(d_bar, x_j, x_m)
        
        term1 = V_hat_prime(x_jm, x_jm_star, r)
        
        sum2 = 0
        neighbours_j = [index for index, value in enumerate(A[j,:]) if value == 1]
        for i in neighbours_j:
            
            if i != malicious_index:
            
                x_i         = states_q[:,i]
                x_j         = states_q[:,j]
                x_ji        = x_j - x_i
                x_ji_star   = compute_x_star(d, x_j, x_i)
                sum2 += V_hat_prime(x_ji, x_ji_star, r)
        
        v_j = states_p[:,j]
        v_jm = v_j - v_m
        term2 = 0.5*np.dot(v_jm.transpose(), v_jm)
        
        sum1 += kx*term1 + (kx/2)*sum2 + term2
    
    Gamma = np.diag(k_vector)  
    eig_max = np.linalg.eigvalsh(np.linalg.inv(Gamma)).max()
    gain_terms = (k_vector[0] + k_vector[0])**2 + (k_vector[1] + k_vector[1])**2 + (k_vector[2] + k_vector[2])**2
    
    term3 = 0.5*eig_max*gain_terms
    
    H += sum1 + term3
    
    return H
    
            
        
# for this application, k_node = malicious agent, gains = k_hat
def Ck_malicious_v2(states_q, states_p, A, Q, k_node, gains):
    
    cmd_i = np.zeros((3))
    
    # search through each neighbour
    for k_neigh in range(states_q.shape[1]):
    
        # except for itself:
        if k_node != k_neigh:
            
            # check if the neighbour is in range
            if A[k_node,k_neigh] == 0:
                in_range = False
            else:
                in_range = True 
        
            # if within range, do your thing
            if in_range:
            
                # compute alignment 
                cmd_i -= gains[0]*compute_alignment(states_q, states_p, k_node, k_neigh)
                # compute cohesion
                cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, Q)*(states_q[:,k_node] - states_q[:,k_neigh])
                # compute repulsion
                cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, Q)*(states_q[:,k_node] - states_q[:,k_neigh])   
    
    return -cmd_i
    


# # build the vector (10) 
# def Ck_malicious(states_p, states_q, malicious_index, k_neighs, Q, mal_gains):
    
#     Ck = np.zeros((states_p.shape[0]))
    
#     # for all neighbours
#     for k_neigh in k_neighs:  
#         #C_1 = compute_alignment(states_q, states_p, malicious_index, k_neigh)
#         #C_2 = compute_cohesion(states_q, malicious_index , k_neigh, Q)*(states_q[:,malicious_index ] - states_q[:,k_neigh])
#         #C_3 = compute_repulsion(states_q, malicious_index , k_neigh, Q)*(states_q[:,malicious_index ] - states_q[:,k_neigh])
#         #C += np.stack([C_1, C_2, C_3], axis=1) # stack in cols (axis=1)
        
#         Ck += mal_gains[0]*compute_alignment(states_q, states_p, malicious_index, k_neigh)
#         Ck += mal_gains[1]*compute_cohesion(states_q, malicious_index , k_neigh, Q)*(states_q[:,malicious_index ] - states_q[:,k_neigh])
#         Ck += mal_gains[2]*compute_repulsion(states_q, malicious_index , k_neigh, Q)*(states_q[:,malicious_index ] - states_q[:,k_neigh])
  
#     return Ck


    
    
    



#%% Filtering
# -----------

#build filters in (11) 
def filter_v(filter_v_gain, v_filtered, v, Ts):
    
    v_filtered_dot  = -filter_v_gain*v_filtered + filter_v_gain*v
    v_filtered      = v_filtered + Ts*v_filtered_dot
    
    return v_filtered
    
# compute C_filterd from v (13)
def estimate_C_filtered(C_filtered, v_filtered, v, k_vector, Ts):

    # compute pseudo inverse
    k_vector_pseudoinv = np.linalg.pinv(k_vector.reshape(3,1)) 
    v_temp = (-v_filtered + v).reshape(3,1)
    
    # get new C_filtered
    C_filtered_new = -np.dot(v_temp, k_vector_pseudoinv)
    
    # compute derivative
    C_filtered_dot = np.divide(C_filtered_new-C_filtered, Ts)
    
    return C_filtered_new, C_filtered_dot  

def estimate_C(filter_v_gain, C_filtered_dot, C_filtered):
    
    C_estimate = C_filtered_dot + filter_v_gain*C_filtered
    
    return C_estimate
     
# estimate k    
def k_hat(Gamma_k, C_filtered, v, v_filtered, states_p, neighbours, k_hat, Ts):
    sum_v = np.zeros((3))
    for i in neighbours:
        sum_v += (states_p[:,i]-v)
   
    # note: Gamma shape may me off (transposed?) not clear in paper.
   
    #k_hat_dot = -np.dot(Gamma_k,C_filtered.transpose())*sum_v \
    #    - np.dot(Gamma_k,C_filtered.transpose())*(C_filtered*k_hat + v - v_filtered)
    k_hat_dot = -np.dot((Gamma_k*C_filtered.transpose()),sum_v) \
        -np.dot((Gamma_k*C_filtered.transpose()),np.dot(C_filtered,k_hat) + v - v_filtered) 
 
    k_hat = k_hat + Ts*k_hat_dot
    return k_hat
                
def k_hat2(C_filtered_dot_k, v, v_filtered, states_p, neighbours, k_hat, Ts):
    sum_v = np.zeros((3))
    for i in neighbours:
        sum_v += (states_p[:,i]-v)
   
    k_hat_dot = -C_filtered_dot_k*sum_v \
        -C_filtered_dot_k*(C_filtered_dot_k + v - v_filtered) 
 
    k_hat = k_hat + Ts*k_hat_dot
    return k_hat

def k_hat(Gamma_k, C_filtered, v, v_filtered, states_p, neighbours, k_hat, Ts):
    sum_v = np.zeros((3))
    for i in neighbours:
        sum_v += (states_p[:,i]-v)
   
    # note: Gamma shape may me off (transposed?) not clear in paper.
   
    k_hat_dot = -np.dot(np.dot(Gamma_k,C_filtered.transpose()),sum_v) \
        - np.dot(np.dot(Gamma_k,C_filtered.transpose()),(np.dot(C_filtered,k_hat) + v - v_filtered))

    k_hat = k_hat + Ts*k_hat_dot
    return k_hat

# -----------------------










'''

#%% test
    
import matplotlib.pyplot as plt

# real
mal_kv = 1 #-5 *0.1 
mal_ka = 1 #75  *0.1
mal_kr = 1 #-10 *0.1

# estimated
mal_kv_hat = mal_kv  #+1
mal_ka_hat = mal_ka #-2 
mal_kr_hat = mal_kr #+0.4

#parameters
H_bar = 10 # tunable
i_cont = 0.2 # tunable
H = H_bar + i_cont
d =  5                          # desired separation
r = np.multiply(d, np.sqrt(2))  # range
d_bar = np.divide(d,np.sqrt(3))                   # malicious agent separation
Q = 10 # ?
Ts = 0.02
filter_v_gain   = 25 # tunable

#test conditions
x_i = np.array((-1.44,-2.50,0))
x_j = np.array((2.80,0,0))
v_i = np.array((-0.1,-0.2,0.3))
v_j = np.array((0.4,0.1,-0.5))
x_m = np.array((0,0,0))
v_m = np.array((1,2,-0.5))
# stack
states_q = np.stack([x_i,x_j,x_m], axis=1)
states_p = np.stack([v_i,v_j,v_m], axis=1)
malicious_index = 2         # index for malcious agent
k_neighs = [0,1]            # index for neighbours of malicious agent

#where to put these
x_ij = x_i - x_j
x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))
#test = potential_function_hat(x_ij, x_ij_star, r, d_bar, H)


v_filtered = 0*np.array((0.1,0.2,0.1))
C_filtered = np.zeros((3,3))

k_vector_real = np.array([mal_kv,mal_ka,mal_kr])
k_vector_hat = np.array([mal_kv_hat,mal_ka_hat,mal_kr_hat]) # needs updating in loop

times = []
test = []
test2 = []
test3 = []
test4 = []
test5 = []

x_i = np.array((-1.44,-2.50,0))
x_j = np.array((2.89,0,0))
x_m = np.array((0,0,0))
x_ij = x_i - x_j
x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))


for i in range(0,240):
    
    # evolve the real malicious agent
    # -------------------------------
    # compute v_dot (actual)
    C_actual = C_malicious(states_p, states_q, malicious_index, k_neighs, Q)
    v_dot = -np.dot(C_actual,k_vector_real.reshape(3,1))
    # evolve (double integrator)
    states_p[:,malicious_index] += v_dot.ravel()*Ts
    states_q[:,malicious_index] += states_p[:,malicious_index]*Ts
    
    
    #  this is where we estimate k, v, C, ... etc
    #    assume known for now, just to get algo working
    # --------------------------------------------------
    
    # get filtered v
    #v_filtered = filter_v(filter_v_gain ,v_filtered, states_p[:,malicious_index], Ts)

    # TRAVIS -> start here. C_estimated is not converging correctly. v_filted is    
    #C_filtered, C_filtered_dot  = estimate_C_filtered(C_filtered, v_filtered, states_p[:,malicious_index], k_vector_hat, Ts)
    #C_filtered_dot_k = -(-v_filtered + states_p[:,malicious_index])

    # get new C_filtered
    ##C_filtered = -np.dot(v_dot, np.linalg.pinv(k_vector_real.reshape(3,1)) )
    
    #C_estimated = estimate_C(filter_v_gain, C_filtered_dot, C_filtered)
    
    #print(v_filtered - states_p[:,malicious_index])
    
    # update k
    #Gamma_k = np.diag(k_vector_hat)
    #Gamma_k = np.tile(k_vector_hat.reshape(1,3), (3, 1))
    #k_vector_hat = k_hat2(C_filtered_dot_k,states_p[:,malicious_index], v_filtered, states_p, k_neighs, k_vector_hat, Ts)
    #k_vector_hat = k_hat(Gamma_k,  C_estimated , states_p[:,malicious_index], v_filtered, states_p, k_neighs, k_vector_hat, Ts)
    #k_vector_hat =  k_hat(Gamma_k,  C_estimated , states_p[:,malicious_index], v_filtered, states_p, k_neighs, k_vector_hat, Ts)
    
    # test potential function 
    x_i =np.array((-1.44    +min(0.02*(240/(i+1)),8),  -2.50   -min(0.04*(240/(i+1)),5),   0-min(0.1*(240/(i+1)),3)))
    x_j =np.array(( 2.89    -0.04*(240/(i+1)),   0      +0.005*(240/(i+1)),         0+min(0.04*(240/(i+1)),2)))
    x_m = np.array((0,0,0))
    x_ij = x_i - x_j
    x_ij_star = (d_bar*(x_i-x_m)/(np.linalg.norm(x_i-x_m))) - (d_bar*(x_j-x_m)/(np.linalg.norm(x_j-x_m)))
    
    
    pfunc = potential_function_hat(x_ij, x_ij_star, r, d_bar, H)

    error_x_im = np.linalg.norm(x_i-x_m)-d_bar
    error_x_jm = np.linalg.norm(x_j-x_m)-d_bar
    error_xij_star = np.linalg.norm(x_ij - x_ij_star)
    norm_xij = np.linalg.norm(x_ij)
    
    
    # tests
    times.append(i)
    #test.append(states_p[:,malicious_index][0])
    #test2.append(v_filtered[0])
    
    #test.append(C_actual[0,2])
    #test2.append(C_estimated[0,2])
    
    #test.append(C_actual[2,1])
    #test2.append(C_estimated[2,1])
    
    test.append(np.linalg.norm(pfunc))
    #test2.append(np.linalg.norm(x_ij-x_ij_star))
    test2.append(error_x_im)
    test4.append(error_x_jm)
    test3.append(error_xij_star)
    test5.append(norm_xij)
    
    #print(v_dot)
    
# Plot the results
plt.figure(figsize=(10, 6))

# Plot actual
plt.plot(times[:], test[:], '-o', label='V_hat')

# Plot test
plt.plot(times[:], test2[:], label='error from dbar_i')
plt.plot(times[:], test4[:], label='error from dbar_j')
plt.plot(times[:], test3[:], label='error from x_ij_star')
plt.plot(times[:], test3[:], label='|x_ij|')
plt.axhline(y=r, color='r', linestyle='--', label='R')
plt.axhline(y=H, color='k', linestyle='--', label='H')
# Customize plot
plt.title('compare')
plt.xlabel('time')
plt.ylabel('var')
plt.legend()
plt.grid()
#plt.axis('equal')  # Equal scaling for both axes
#plt.xlim(-10, 10)  # Adjust limits as needed
#plt.ylim(-10, 10)  # Adjust limits as needed

# Show the plot
plt.show()

'''           

    
    
    