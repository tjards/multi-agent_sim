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


class Consensuser:
    
    def __init__(self, params_n, hetero_lattice, directional, d_min, d):
        
        # select parameter ranges
        if hetero_lattice == 1:
            self.params_range = [d_min,d]
        else:
            self.params_range = [d,d]
        
        # parameters
        self.params_n   = params_n  # number of parameters
        #self.params     = [random.uniform(self.params_range[0], self.params_range[1]) for _ in range(self.params_n)] # options for these parameters
        self.params     = [round(random.uniform(self.params_range[0], self.params_range[1]),1) for _ in range(self.params_n)] # options for these parameters
        self.alpha      = 0.6 #0.5                 # (0,1)
        self.beta       = 1-self.alpha        # (0,1) # assume all equal now, but this can vary per agent (maybe, just touching)
        
        # store the parameters
        self.d_weighted  = np.zeros((len(self.params),len(self.params)))   
        i = 0
        while (i < len(self.params)):
            self.d_weighted[i,:] = self.params[i]
            i+=1
        
        # store whether agents are in proximity to eachother (1 =  yes, 0 = no)
        self.prox_i = np.zeros((len(self.params),len(self.params)))
        
        if directional:
            self.headings = np.zeros((1,self.params_n))
    
    
    def update(self, k_node, k_neigh):
        
        #print('agent ', k_node,' d from agent ', k_neigh, ': ', d )
        self.d_weighted[k_node, k_neigh] = self.alpha * self.d_weighted[k_node, k_neigh]
        self.d_weighted[k_node, k_neigh] += (self.beta * self.d_weighted[k_neigh, k_node])
        #print("Agent ", k_node, "/ ", k_neigh, " param: ", self.d_weighted[k_node, k_neigh])