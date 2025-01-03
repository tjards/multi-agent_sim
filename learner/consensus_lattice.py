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


approach = 'gradient'
#       'communicate'   = directly share d
#       'gradient'      = use higher-level gradient minimization
#       'off'           = doesn't do the adjustments 

# too close ratio (<= 1)
tcr = 1         # default 1 = never too close; <1 ratio of d_min too close
#d_max = 10        # max range for lattice options 
# d_min is imported, as it could be tied to physical constraints (maybe change this later)  


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
    
    # d1 = adjusted d_min
    # d2 = adjusted d_max
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
            self.params_range = [tcr*d_min,d_max]
        else:
            self.params_range = [d,d]
        
        # parameters
        #self.d_min      = d_min
        #self.d_max      = d_max
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
        
        # set dmins and dmaxes
        mean_range = (d_min+d_max)/2 # offset proportional to deviation from center of range
        self.d_min = np.zeros((len(self.params),len(self.params)))
        self.d_max = np.zeros((len(self.params),len(self.params))) 
        i = 0
        while (i < len(self.params)):
            offset = self.params[i] - mean_range 
            self.d_min[i,:] = np.max([d_min + random.uniform(0,1)*offset, d_min])
            self.d_max[i,:] = np.min([d_max + random.uniform(0,1)*offset, d_max])
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
            constrained = 1        # 1 = yes, 0 = no (always constrain for now; use "communicate" for unconstrained
            
            if constrained == 0:
            
                # compute the distance
                d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node])
                # only update if above minimum
                #if d >= self.d_min:
                    # use gradient descent
                    #term = self.d_weighted[k_node, k_neigh] - np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node]) #+ np.divide(self.gain_lower_bound,self.d_weighted[k_node, k_neigh]-self.d_min)
                term = self.d_weighted[k_node, k_neigh] - d
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*term
            
            # Travis: consider getting rid of this soft constraint all together. it sucks.
            
            elif constrained == 1:
                
                
                # combined
                # ========
                d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node])
                d_i = self.d_weighted[k_node, k_neigh]
                #d_min = self.d_min
                #d_max = self.d_max
                d_min = self.d_min[k_node, k_neigh] # allows for different constraints per agent
                d_max = self.d_max[k_node, k_neigh]
                
                
                
                d_pref = self.d_weighted[k_node, k_node] 
                V_b, V_b_grad = bump_function_gradient(d, d_min, d_max, d_pref)  # travis, I think you are passing in wrong d here
                
                terms = V_b*(d_i - d) + 0.5*V_b_grad*(d_i - d)**2
                
            
                
                #if d_i > d_max:
                #    print(d_i)
                
                
                ## compute the main term
                #d = np.linalg.norm(states_q[:,k_neigh]-states_q[:, k_node])
                #d_min = self.d_min
                
                #d_i = self.d_weighted[k_node, k_neigh]
                #term = d - d_i
                #term = d_i - d
                
                # constrauned part v1
                # ====================

                '''
                
                # define sigmoid (zeros for no sigmoid)
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
                '''    
                
                # constrauned part v2
                # ====================
                ##term_constraint = 0
                #d_pref = self.d_weighted[k_node, k_node]
                #term_constraint = bump_function_gradient(d_i, d_min, d_max, d_pref)
                #print(term_constraint)
    
                # combine 
                #terms = term + self.gain_lower_bound*term_constraint
                
                #print(term_constraint)
                
                # use gradient descent (only if not too close)
                #if d >= d_min:
                self.d_weighted[k_node, k_neigh] -= self.gain_gradient*terms
                
                # clip
                #self.d_weighted[k_node, k_neigh]  = np.clip(self.d_weighted[k_node, k_neigh], self.d_min, self.d_max)

                
        elif approach == 'off':
            
            pass
                

  
        
            
  
        

        
        
        