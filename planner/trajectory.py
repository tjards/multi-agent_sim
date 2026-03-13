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

# custom packages
from .techniques import lemniscates

# define the trajectory object
# ----------------------------
class Trajectory:
    
    def __init__(self, tactic_type, targets, nAgents):
        
        self.trajectory = copy.deepcopy(targets)
        #self.lemni = np.zeros([1, nAgents])
        self.lemni = np.zeros([2, nAgents]) # consider renaming to "past" parameter or something 
        self.sorted_neighs = list(range(nAgents))
        self.tactic_type = tactic_type

    def load_planners(self, planners):
        self.planners = planners

    def update(self, tactic_type, state, targets, **kwargs):

        # eventually I will remove these conditionals and just call relevant str(tactic_type)
        kwargs['state'] = state

        # these two adjust teh trajectory; use polymorphism. 
        if tactic_type == 'encirclement' or \
            tactic_type == 'lemniscates' or \
                tactic_type == 'flocking_reynolds' or \
                    tactic_type == 'flocking_starling':

            self.planners[tactic_type].update_trajectory(self, targets, **kwargs)

        else:
            
            # temporary. once all techniques are under the base class, I can just call all with update_trajectory
            self.trajectory = targets.copy()
  