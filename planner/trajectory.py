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
from .techniques import lemni_tools 

# define the trajectory object
# ----------------------------
class Trajectory:
    
    def __init__(self, tactic_type, targets, nAgents):
        
        self.trajectory = copy.deepcopy(targets)
        #self.lemni = np.zeros([1, nAgents])
        self.lemni = np.zeros([2, nAgents])
        self.sorted_neighs = list(range(nAgents))
        self.tactic_type = tactic_type

    def load_planners(self, planners):
        self.planners = planners

    def update(self, tactic_type, state, targets, t, i, **kwargs):

        #if flocking
        if tactic_type == 'reynolds' or tactic_type == 'saber' or tactic_type == 'starling' or tactic_type == 'pinning' or tactic_type == 'shep':
            self.trajectory = targets.copy() 
        
        # if encircling
        if tactic_type == 'circle':
            self.trajectory, _, _ = self.planners['circle'].encircle_target(targets, state)
        
        # if lemniscating
        elif tactic_type == 'lemni':
            
            learn_actions = kwargs.get('lemni_learn_actions')
            self.trajectory, self.lemni, self.sorted_neighs = self.planners['lemni'].lemni_target(state,targets,i,t,learn_actions)
            
        elif tactic_type == 'cao':
            
            self.trajectory = targets.copy()
            
# helpers
def update_trajectory_args(Agents, Trajectory, Controller, tactic_type, my_kwargs):
    
    # we'll need the record of lemni parameters  
    if tactic_type == 'lemni':
        
        my_kwargs['sorted_neighs'] = Trajectory.sorted_neighs
        
        # new bidrirectional controller
        if 'lemni_CALA_xz' in Controller.Learners:
            
            # CASE 2: bidirectional
            my_kwargs['lemni_learn_actions'] = {
                'xz': Controller.Learners['lemni_CALA_xz'].action_set,
                }
        else:
            my_kwargs['lemni_learn_actions'] = {
               'xz': np.zeros((2*Agents.state.shape[1]))
               }
            
    return my_kwargs            
            
            