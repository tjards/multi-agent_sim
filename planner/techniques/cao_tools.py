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

Some default params

    # normal agents
    # -------------
    d       =  5                         # desired separation
    r       = np.multiply(d, np.sqrt(2))  # sensor range (adjust this later, derived from desired separation now)
    cmd_min = -100
    cmd_max = 100

    # navigation gains
    gain_p = 1 # default zero (else, messes up this technique)
    gain_v = 0 # default zero (else, messes up this technique)

    # swarming gains
    kv = 3 # used for counter-malign as well
    ka = 1
    kr = 2

    # counter-malign agents
    # ---------------------
    kx = 2          # layer 2 gain
    d_bar  = r/2    # np.divide(d,np.sqrt(3)) # malicious agent separation
    i_cont = 0.2    # small +ve for numerator of potential function (H+i)
    gamma_kp = 2    # layer 3 gamma constant
    H_min = 100     # lower bound on H

    # malicious agents
    # -----------------
    mode_malicious = 1          # 1= yes, 0 = no
    mal_type = 'collider'       # runaway, collider
    if mal_type == 'runaway':
        mal_kv = 0.8
        mal_ka = -0.1 
        mal_kr = 4500 
    elif mal_type == 'collider':
        mal_kv = -50
        mal_ka = 200
        mal_kr = -10
    elif mal_type == 'cooperative':
        mal_kv = kv
        mal_ka = ka
        mal_kr = kr
        
    # initial estimates 
    mal_kv_hat = 0.12*mal_kv 
    mal_ka_hat = 0.34*mal_ka 
    mal_kr_hat = 0.56*mal_kr   

    # filter parameters
    # -----------------
    filter_v_gain   = 50                        # tunable
    v_filtered      = 0*np.array((0.1,0.2,0.1)) # initial ize at zero
    C_filtered      = np.zeros((3,3))           # initialize at zero
    k_vector_hat = np.array([mal_kv_hat,mal_ka_hat,mal_kr_hat])
    C_estimate = np.zeros((3,3))


dev notes:
    
    17 Nov 24 - doesn't  work in 2D yet. Need to stop filter from causing exp growth to inf (no measurements)
    7 Feb 25 - added config loading 


@author: tjards
"""

# import stuff
# -----------
import numpy as np
import config.config as cfg


#%% CUSTOM CLASS 

class Planner:
    
    def __init__(self, config_data, states_q, states_p):
        
        # Extract cao configuration section
        cao_config = cfg.get_config(config_data, 'planner.techniques.cao')
        
        # Normal agent parameters
        self.d = cao_config.get('d', 5)
        self.r = cao_config.get('r', 7.071067811865476)
        self.cmd_min = cao_config.get('cmd_min', -100)
        self.cmd_max = cao_config.get('cmd_max', 100)
        
        # Navigation gains
        self.gain_p = cao_config.get('gain_p', 1)
        self.gain_v = cao_config.get('gain_v', 0)
        
        # Swarming gains
        self.kv = cao_config.get('kv', 3)
        self.ka = cao_config.get('ka', 1)
        self.kr = cao_config.get('kr', 2)
        
        # Counter-malign parameters
        self.kx = cao_config.get('kx', 2)
        self.d_bar = cao_config.get('d_bar', 3.5355339059327378)
        self.i_cont = cao_config.get('i_cont', 0.2)
        self.gamma_kp = cao_config.get('gamma_kp', 2)
        self.H_min = cao_config.get('H_min', 100)
        
        # Malicious agent mode and type
        self.mode_malicious = cao_config.get('mode_malicious', 1)
        self.mal_type = cao_config.get('mal_type', 'collider')
        
        # Set malicious parameters based on mal_type (hardcoded logic)
        if self.mal_type == 'runaway':
            self.mal_kv = 0.8
            self.mal_ka = -0.1
            self.mal_kr = 4500
        elif self.mal_type == 'collider':
            self.mal_kv = -50
            self.mal_ka = 200
            self.mal_kr = -10
        elif self.mal_type == 'cooperative':
            self.mal_kv = self.kv
            self.mal_ka = self.ka
            self.mal_kr = self.kr
        else:
            raise ValueError(f"Unknown mal_type: {self.mal_type}. Must be 'runaway', 'collider', or 'cooperative'")
        
        # Compute hat estimates (adaptive parameters)
        self.mal_kv_hat = 0.12 * self.mal_kv
        self.mal_ka_hat = 0.34 * self.mal_ka
        self.mal_kr_hat = 0.56 * self.mal_kr
        
        # Filter parameters
        self.filter_v_gain = cao_config.get('filter_v_gain', 50)
        
        # Initialize state with provided positions/velocities
        self.status = ['friendly'] * states_q.shape[1]
        self.layer = [0] * states_q.shape[1]
        self.cmd_i = np.zeros((states_q.shape[1], 3))
        self.Q = 1.1 * compute_E(states_q, states_p, self.r)
        self.assembled = 0
        self.malicious = 2
        self.alpha_kp = self.gamma_kp * np.ones((states_q.shape[1], states_q.shape[1]))
        
        # Filter state
        self.filter_v_gain_val = self.filter_v_gain
        self.v_filtered = np.zeros(3)
        self.C_filtered = np.zeros((3, 3))
        self.C_estimate = np.zeros((3, 3))
        self.k_vector_hat = np.array([self.mal_kv_hat, self.mal_ka_hat, self.mal_kr_hat])

        
    # checks if layer 2 has at least 2 agents (Assumption 3)
    def check_assume_3(self):
        count_layer_2 = self.layer.count(2)
        if count_layer_2 > 2:  
            return True
        else:
            return False 
    
    # checks if at least 2 agents in layer 2 are neighbours (Assumption 4)
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
        
    # computes E (a.k.a, Q) from Eqn (3)     
    def update_Q(self,states_q, states_p):
        self.Q      = 1.1*compute_E(states_q, states_p, self.r)
    
    # identify malicious agent and build layers
    # -----------------------------------------
    def build_layers(self, states_q, states_p, k_node, **kwargs):
        
        A               = kwargs['A']
        #A               = kwargs['A_connectivity'] 
        pin_matrix      = kwargs['pin_matrix']
        
        # mark as assembled
        self.assembled = 1
        # update Q
        self.Q      = 1.1*compute_E(states_q, states_p, self.r)
        # chose a malicious agent (i.e., the pin)
        malicious = np.where(np.diag(pin_matrix) == 1)[0][0] 
        #malicious = self.malicious
        self.malicious = malicious 
        self.status[malicious] = 'malicious'
        # malicious agent is layer 1
        self.layer[malicious] = 1
        # activate response to malicious node ( = 1)
        #self.mode_malicious = mode_malicious
        self.gain_p = 0 # turn off navigation in malicious mode (messes with results)
        self.gain_v = 0 # turn off navigation in malicious mode (messes with results)
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
        #self.H      = max(compute_H(states_q, states_p, self.malicious, layer2_indices, kx, np.array([mal_kv, mal_ka, mal_kr]), A), H_min)
        self.H      = max(compute_H(states_q, states_p, self.malicious, layer2_indices, self.kx, np.array([self.mal_kv, self.mal_ka, self.mal_kr]), A, self.d, self.d_bar, self.r), self.H_min)
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
        
    # compute commands (Eqn (12) from [1] )
    # -------------------------------------
    def compute_cmd(self, targets, states_q, states_p, k_node, **kwargs):
        
        # extract stuff
        A               = kwargs['A']
        pin_matrix      = kwargs['pin_matrix']
        Ts              = kwargs['Ts']
        
        # upon first assembly, build layers
        if np.sum(pin_matrix) == 1 and self.assembled == 0:           
            self.build_layers(states_q, states_p, k_node, **kwargs)
            
        # set parameters for friendly or malicious agent
        if self.status[k_node] == 'friendly':
            gains = [self.kv,self.ka,self.kr]
        else:
            gains = [self.mal_kv, self.mal_ka, self.mal_kr]
        
        # initialize
        cmd_i = np.zeros((3))
        
        # compute navigation (if not assembled, draws towards goal when gain_p, gain_v set)
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
                        cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh])
                        # compute repulsion
                        cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh])
                
        # if we are responding to a malicious node 
        # --------------------------------------
        if self.mode_malicious == 1:
            
            # for layer 2
            # ------------
            if self.layer[k_node] == 2: 
                 
                # search through layer 2
                for k_neigh in [index for index, value in enumerate(self.layer) if value == 2]:
                        
                    # exclude self
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
                            x_ij_star = compute_x_star(self.d, x_i, x_j)
                            cmd_i -= self.kx*potential_function_hat(x_ij, x_ij_star, self.r, self.d_bar, self.H+self.i_cont) #*(x_ij)

                #malicious_k_hat = np.array([mal_kv,mal_ka,mal_kr])
                malicious_k_hat = np.array([self.mal_kv_hat, self.mal_ka_hat, self.mal_kr_hat])
 
                # estimate the behaviour of malicious agent
                cmd_i -=  Ck_malicious(states_q, states_p, A, self.Q, self.malicious, malicious_k_hat, self.r)
                               
            # for layer 3 
            # -----------
            elif self.layer[k_node] == 3: 
                
                # search through each neighbour
                for k_neigh in range(states_q.shape[1]):
            
                    # except for itself:
                    if k_node != k_neigh:
                        
                        # check if the neighbour is in range
                        if A[k_node,k_neigh] == 0:
                            in_range = False
                        else:
                            in_range = True 
                    
                        #if within range
                        if in_range:
                            
                            # update the alpha term (paper says this is a der, but I think that's an error)
                            alpha_dot = self.gamma_kp*np.sum(np.abs(states_p[:,k_node]-states_p[:,k_neigh])) # L-1 norm
                            self.alpha_kp[k_node, k_neigh] = alpha_dot #+= (alpha_dot*Ts).item()
   
                            # compute alignment 
                            cmd_i -= (self.alpha_kp[k_node, k_node])*np.sign(states_p[:,k_node]-states_p[:,k_neigh])
                            # compute cohesion
                            cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh])
                            # compute repulsion
                            cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh]) 
                                

            # for layer 1 (malicious agent) + everyone else 
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
                            C1 = compute_alignment(states_q, states_p, k_node, k_neigh)
                            cmd_i -= gains[0]*C1
                            # compute cohesion
                            C2 = compute_cohesion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh])
                            cmd_i -= gains[1]*C2
                            # compute repulsion
                            C3 = compute_repulsion(states_q, k_node, k_neigh, self.Q, self.r)*(states_q[:,k_node] - states_q[:,k_neigh])  
                            cmd_i -= gains[2]*C3 
                 
                # do the filtering stuff     
                if self.layer[k_node] == 1:
 
                    # run the filter 
                    filter_stuff = True
                    if filter_stuff:
                
                        # creat vector of estimated gains
                        malicious_k_hat = np.array([self.mal_kv_hat,self.mal_ka_hat,self.mal_kr_hat]) 
                    
                        self.v_filtered = filter_v(self.filter_v_gain ,self.v_filtered, states_p[:,self.malicious], Ts)
                        #print('v error: ', self.v_filtered - states_p[:,self.malicious])
                    
                        # Estimate C
                        self.C_filtered = filter_C(self.filter_v_gain, self.C_filtered, self.C_estimate, Ts)
                        #print('C error: ', self.C_filtered - self.C_estimate)
                    
                        # # updatw gains
                        Gamma_k = np.diag(malicious_k_hat)
                        #print(Gamma_k)
                
                        malicious_k_hat_updated = k_hat(Gamma_k, self.C_filtered, self.C_estimate, states_p[:,self.malicious], self.v_filtered, states_p, [index for index, value in enumerate(self.layer) if value == 2], malicious_k_hat, Ts)
                        
                        if np.any(np.isnan(malicious_k_hat_updated)) or None in malicious_k_hat_updated:
                            
                            pass
                        
                        else:
                        
                            self.mal_kv_hat = malicious_k_hat_updated[0]
                            self.mal_ka_hat = malicious_k_hat_updated[1]
                            self.mal_kr_hat = malicious_k_hat_updated[2]
                            #malicious_k_hat = np.array([self.mal_kv_hat,self.mal_ka_hat,self.mal_kr_hat])
                    
                        #print('K (estimated)', malicious_k_hat)
                        #print('K (real)', np.array([mal_kv,mal_ka,mal_kr]) )
                        
                    # C is pulled from known part of malicious dynamics
                    self.C_estimate = np.stack([C1,C2,C3], axis = 1)
                          
        cmd_i = np.clip(cmd_i, self.cmd_min, self.cmd_max)
               
        return cmd_i        

#%% 
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
def compute_cohesion(states_q, k_node, k_neigh, Q, r):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
    u_i_cohes = np.divide(2*(r**2 + (r**2)/Q), np.square(r**2 - s**2 + (r**2)/Q ))
    
    return u_i_cohes
    
# compute repulsion command
# ------------------------
def compute_repulsion(states_q, k_node, k_neigh, Q, r):
 
    s = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
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

# compute Q (a.k.a E) ref: Eqn (3) from [2]
# ---------------------------------------
def compute_E(states_q, states_p, r):
    
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
    E = 0.5*v_sum + np.divide((N*(N-1)),2)*V_max
    
    return E
      
#%%              
# ***************** #
#  COUNTER-MALIGN   #
# ***************** #       

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

# Layer 2 collective del-potential function (del V_ij)
# -----------------------------------------------
def potential_function_hat(s, a, r, d, H):
    
    num1 = (s-a)*(np.divide(s*(s-a),np.linalg.norm(s)) + 2*(-np.linalg.norm(s) + (1/H)*(d-r)**2 +r))
    den1 = (r - np.linalg.norm(s) + (1/H)*(d-r)**2)**2
    num2 = H*(s-a)*(H*s*(a+s)+2*np.linalg.norm(s)*d**2)
    den2 = np.linalg.norm(s)*(H*np.linalg.norm(s) + d**2)**2
    
    return num1/den1 + num2/den2

# desired displacement 
# --------------------
def compute_x_star(d, x_i, x_j):
    
    x_ij_star = d*((x_i - x_j)/np.linalg.norm(x_i - x_j))
    
    return x_ij_star
                  
# Layer 2 collective potential function prime (\hat{V}'_ij)  
# ---------------------------------------------------------          
def V_hat_prime(x_ij, x_ij_star, r):
    
    V_hat_prime = np.divide((np.linalg.norm(x_ij - x_ij_star))**2, r - np.linalg.norm(x_ij)) + np.divide((np.linalg.norm(x_ij - x_ij_star))**2, np.linalg.norm(x_ij)) 
    
    return V_hat_prime

# Layer 2 designable positive constant (H bar)
# --------------------------------------------
def compute_H(states_q, states_p, malicious_index, layer2_indices, kx, k_vector, A, d, d_bar, r):
    
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
    
# estimate dynamics of malicious agent 
# ------------------------------------           

# for this application, k_node = malicious agent, gains = k_hat
def Ck_malicious(states_q, states_p, A, Q, k_node, gains, r):
    
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
                
                # if np.isnan(states_q).any():
                #     print("stop here")
                # if np.isnan(states_p).any():
                #     print('stop here')
            
                # compute alignment
                cmd_i -= gains[0]*compute_alignment(states_q, states_p, k_node, k_neigh)
                # compute cohesion
                cmd_i -= gains[1]*compute_cohesion(states_q, k_node, k_neigh, Q, r)*(states_q[:,k_node] - states_q[:,k_neigh])
                # compute repulsion
                cmd_i -= gains[2]*compute_repulsion(states_q, k_node, k_neigh, Q, r)*(states_q[:,k_node] - states_q[:,k_neigh])   
    
    return -cmd_i
    
#%% FILTERING

# build filters in (11) and (12)
def filter_v(filter_v_gain, v_filtered, v, Ts):
    
    v_filtered_dot  = -filter_v_gain*v_filtered + filter_v_gain*v
    v_filtered      = v_filtered + Ts*v_filtered_dot
    
    return v_filtered

def filter_C(filter_v_gain, C_filtered, C_estimate, Ts):
    
    C_filtered_dot  = -filter_v_gain*C_filtered + C_estimate 
    C_filtered      = C_filtered + Ts*C_filtered_dot
    
    return C_filtered 
      
# estimate k (14)   
def k_hat(Gamma_k, C_filtered, C_estimate, v, v_filtered, states_p, neighbours, k_hat, Ts):
    
    sum_v = np.zeros((3))
    for i in neighbours:
        sum_v += (states_p[:,i]-v)
    
    k_hat_dot = -np.dot((np.dot(Gamma_k,C_filtered.transpose())),sum_v) \
        -np.dot((np.dot(Gamma_k,C_filtered.transpose())),np.dot(C_filtered,k_hat) + v - v_filtered) 
 
    k_hat = k_hat + Ts*k_hat_dot
    
    return k_hat
  




    