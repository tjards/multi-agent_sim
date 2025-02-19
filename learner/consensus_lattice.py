#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:14:55 2024

@author: tjards
"""

# import stuff
# ------------
import numpy as np
import random
from scipy.spatial.distance import cdist


approach = 'gradient'
#       'communicate'   = directly share d
#       'gradient'      = use higher-level gradient minimization
#       'off'           = doesn't do the adjustments 

tcr = 1         # (<= 1) default 1 = never too close; <1 ratio of d_min too close
tfr = 1         # (>= 1) default 1 = never too far; >1 ratio of d_max too far
 
# define 
def adjust_dminmax(d_min, d_max, d):
    center = d_min + (d_max - d_min)/2 
    if d < center:
        offset = d - d_min
        d_max = d + offset
    elif d > center:
        offset = d_max - d
        d_min = d - offset
    return d_min, d_max


# define bump function gradient
p = 3
def bump_function_gradient(d_hat, d_min, d_max, d_pref):
    
    # first, adjust bounds
    #d1, d2, = adjust_dminmax(d_min, d_max, d_pref)
    d1, d2 = d_min, d_max
    
    if d1 < d_hat < d2:
        lambda_val = 2 * (d_hat - (d1 + d2) / 2) / (d2 - d1)
        #V_b = np.exp(-lambda_val**2 / (1 - lambda_val**2))
        V_b = np.exp(-((lambda_val**2 / (1 - lambda_val**2))**p))
        
        #gradient = V_b * (-4 * lambda_val / ((1 - lambda_val**2)**2 * (d2 - d1)))
        gradient = V_b * (-4 * lambda_val * p * (lambda_val**2 / (1 - lambda_val**2))**(p - 1) / ((1 - lambda_val**2)**2 * (d2 - d1)))

        return V_b, gradient
    else:
        return 0, 0  
     

class Consensuser:
    
    #def __init__(self, params_n, hetero_lattice, directional, d_min, d):
    def __init__(self, params_n, hetero_lattice, d_min, d, r_max):
        
        d_max = r_max - 0.5
        
        # select parameter ranges
        if hetero_lattice == 1:
            #self.params_range = [tcr*d_min,d]
            self.params_range = [tcr*d_min,tfr*d_max]
        else:
            self.params_range = [d,d]
        
        # parameters
        self.params_n   = params_n  # number of parameters
        #self.params     = [random.uniform(self.params_range[0], self.params_range[1]) for _ in range(self.params_n)] # options for these parameters
        self.params     = [round(random.uniform(self.params_range[0], self.params_range[1]),1) for _ in range(self.params_n)] # options for these parameters
        self.alpha      = 0.6 #0.5                 # (0,1)
        self.beta       = 1-self.alpha        # (0,1) # assume all equal now, but this can vary per agent (maybe, just touching)
        #self.gain_lower_bound = 1 #0.3 # for update_gradient(), weight of constraint term
        self.gain_gradient = 0.2 #0.2# for update_gradient(), nominally same as Ts
        
        
        # store the parameters
        self.d_weighted  = np.zeros((len(self.params),len(self.params)))   
        i = 0
        while (i < len(self.params)):
            self.d_weighted[i,:] = self.params[i]
            i+=1
        self.d_hat = self.d_weighted.copy()
        
        # set dconstraints
        mean_range = (d_min+d_max)/2 # offset proportional to deviation from center of range
        self.d_min = np.zeros((len(self.params),len(self.params)))
        self.d_max = np.zeros((len(self.params),len(self.params))) 
        i = 0
        while (i < len(self.params)):
            if hetero_lattice == 1:
                offset = self.params[i] - mean_range 
            else:
                offset == 0
            self.d_min[i,:] = np.max([d_min + random.uniform(0,1)*offset, d_min])
            self.d_min[i,i] = 0
            self.d_max[i,:] = np.min([d_max + random.uniform(0,1)*offset, d_max])
            self.d_max[i,i] = 0
            i+=1
        
        # store whether agents are in proximity to eachother (1 =  yes, 0 = no)
        self.prox_i = np.zeros((len(self.params),len(self.params)))
        
        #if directional:
        #    self.headings = np.zeros((1,self.params_n))
    
    
    def compute_violations(self, states_q):
        
        # compute separations between all agents:
        violations = np.zeros((states_q.shape[1],states_q.shape[1]))    
        seps=cdist(states_q.transpose(), states_q.transpose())
        # detect under mins
        diffs_min = seps - self.d_min
        violations[diffs_min < 0] += diffs_min[diffs_min < 0]
        diffs_max = self.d_max - seps
        violations[diffs_max < 0] -= diffs_max[diffs_max < 0]
        
        return violations 
    
    def update(self, k_node, k_neigh, states_q):
        
         
        if approach == 'communicate':
        
            #print('agent ', k_node,' d from agent ', k_neigh, ': ', d )
            self.d_weighted[k_node, k_neigh] = self.alpha * self.d_weighted[k_node, k_neigh]
            self.d_weighted[k_node, k_neigh] += (self.beta * self.d_weighted[k_neigh, k_node])
            #print("Agent ", k_node, "/ ", k_neigh, " param: ", self.d_weighted[k_node, k_neigh])
            
        elif approach == 'gradient':
            
            # constrained by d_min?
            constrained = 1        # 1 = yes, 0 = no (always constrain for now; use "communicate" or "off" for unconstrained
            
            if constrained == 0:
            
                # compute the distance
                d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node])
                term = self.d_weighted[k_node, k_neigh] - d
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*term
                       
            elif constrained == 1:
                
                # actual distance 
                d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node]) 
                
                # low-pass filter
                tau=0.5
                dot_d_hat = (1/tau)*(d - self.d_hat[k_node, k_neigh])
                self.d_hat[k_node, k_neigh] += 0.02*dot_d_hat 
                d_hat = self.d_hat[k_node, k_neigh]
                
                
                # current desired distance
                d_i = self.d_weighted[k_node, k_neigh]
                # constraints 
                d_min = self.d_min[k_node, k_neigh] # allows for different constraints per agent
                d_max = self.d_max[k_node, k_neigh]
                # preferred (not used, for now)
                d_pref = self.d_weighted[k_node, k_node] 
                # compute the PF and gradient for bump function 
                #V_b, V_b_grad = bump_function_gradient(d, d_min, d_max, d_pref)  
                V_b, V_b_grad = bump_function_gradient(d_hat, d_min, d_max, d_pref)  
                # build the total gradient function (force)
                #terms = V_b*(d_i - d) + 0.5*V_b_grad*(d_i - d)**2
                terms = V_b*(d_i - d_hat) #+ 0.5*V_b_grad*(d_i - d_hat)**2 # travis - this second term of Vdot... may come in useful later
                # use gradient descent to update desired distance
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*terms
                
                # clip
                #self.d_weighted[k_node, k_neigh]  = np.clip(self.d_weighted[k_node, k_neigh], self.d_min, self.d_max)

        elif approach == 'off':
            
            pass
                

  
        
            
  
        

        
        
        