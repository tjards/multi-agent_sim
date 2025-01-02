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


approach = 'off'
#       'communicate'   = directly share d
#       'gradient'      = use higher-level gradient minimization
#       'off'           = doesn't do the adjustments 

# too close ratio (<= 1)
tcr = 1         # default 1 = never too close; <1 ratio of d_min too close
d_max = 13        # max range for lattice options 
# d_min is imported, as it could be tied to physical constraints (maybe change this later)  


class Consensuser:
    
    #def __init__(self, params_n, hetero_lattice, directional, d_min, d):
    def __init__(self, params_n, hetero_lattice, d_min, d):
        
        # select parameter ranges
        if hetero_lattice == 1:
            #self.params_range = [tcr*d_min,d]
            self.params_range = [tcr*d_min,d_max]
        else:
            self.params_range = [d,d]
        
        # parameters
        self.d_min      = d_min
        self.params_n   = params_n  # number of parameters
        #self.params     = [random.uniform(self.params_range[0], self.params_range[1]) for _ in range(self.params_n)] # options for these parameters
        self.params     = [round(random.uniform(self.params_range[0], self.params_range[1]),1) for _ in range(self.params_n)] # options for these parameters
        self.alpha      = 0.6 #0.5                 # (0,1)
        self.beta       = 1-self.alpha        # (0,1) # assume all equal now, but this can vary per agent (maybe, just touching)
        self.gain_lower_bound = 0.3 # for update_gradient(), weight of lower d_min term
        self.gain_gradient = 1 # for update_gradient(), nominally same as Ts
        
        # store the parameters
        self.d_weighted  = np.zeros((len(self.params),len(self.params)))   
        i = 0
        while (i < len(self.params)):
            self.d_weighted[i,:] = self.params[i]
            i+=1
        
        # store whether agents are in proximity to eachother (1 =  yes, 0 = no)
        self.prox_i = np.zeros((len(self.params),len(self.params)))
        
        #if directional:
        #    self.headings = np.zeros((1,self.params_n))
    
    
    def update(self, k_node, k_neigh, states_q):
        
        if approach == 'communicate':
        
            #print('agent ', k_node,' d from agent ', k_neigh, ': ', d )
            self.d_weighted[k_node, k_neigh] = self.alpha * self.d_weighted[k_node, k_neigh]
            self.d_weighted[k_node, k_neigh] += (self.beta * self.d_weighted[k_neigh, k_node])
            #print("Agent ", k_node, "/ ", k_neigh, " param: ", self.d_weighted[k_node, k_neigh])
            
        elif approach == 'gradient':
            
            # constrained by d_min?
            constrained = 0         # 1 = yes, 0 = no
            
            if constrained == 0:
            
                # use gradient descent
                term = self.d_weighted[k_node, k_neigh] - np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node]) #+ np.divide(self.gain_lower_bound,self.d_weighted[k_node, k_neigh]-self.d_min)
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*term
            
            elif constrained == 1:
                
                # compute the main term
                d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node])
                d_min = self.d_min
                d_i = self.d_weighted[k_node, k_neigh]
                #term = d - d_i
                term = d_i - d
                
                # define sigmoid
                x_0 = 0 #0.2*self.d_min # shift parameter
                k   = 0 #1.0 # steepness (higher, steeper)
                
                # ignore constraints if too close
                if d < d_min:
                    term_constraint = 0
                else:
                    # compute constrain term
                    sigmoid = 1 / (1 + np.exp(-k * (d - d_min - x_0)))
                    sigmoid_derivative = k * sigmoid * (1 - sigmoid)
                    term_constraint = ((-1 / (d - d_min)) * sigmoid + np.log(d - d_min) * sigmoid_derivative)
                    
                # combine 
                terms = term + self.gain_lower_bound*term_constraint
                
                #print(term_constraint)
                
                # use gradient descent
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*terms
                
        elif approach == 'off':
            
            pass
                

  
        
            
  
        

        
        
        