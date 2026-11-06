#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:19:27 2024

@author: tjards
"""

# import stuff
# ------------
import copy
import numpy as np
import os
import json

# define the trajectory object
# ----------------------------
class Trajectory:
    
    def __init__(self, tactic_type, targets, nAgents):
        
        self.trajectory = copy.deepcopy(targets)
        #self.lemni = np.zeros([1, nAgents])
        self.lemni = np.zeros([2, nAgents])         # consider renaming to "past" parameter or something 
        self.sorted_neighs = list(range(nAgents))
        self.tactic_type = tactic_type

    def load_planners(self, planners):
        self.planners = planners

    def update(self, tactic_type, state, targets, **kwargs):

        kwargs['state'] = state

        self.planners[tactic_type].update_trajectory(self, targets, **kwargs)


  
