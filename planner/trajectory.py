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

    def update(self, tactic_type, state, targets, t, i, **kwargs):

        # eventually I will remove these conditionals and just call relevant str(tactic_type)
        kwargs['state'] = state

        #if flocking or similar
        if tactic_type == 'flocking_reynolds' or tactic_type == 'flocking_saber' or tactic_type == 'flocking_starling' or tactic_type == 'pinning_lattice' or tactic_type == 'shepherding':
            self.trajectory = targets.copy() 
        
        # if encircling
        elif tactic_type == 'encirclement':
            #self.trajectory, _, _ = self.planners['encirclement'].encircle_target(targets, state)
            self.planners['encirclement'].update_trajectory(self, targets, **kwargs)
        
        # if lemniscating
        elif tactic_type == 'lemniscates':
            
            learn_actions = kwargs.get('lemni_learn_actions')
            self.trajectory, self.lemni, self.sorted_neighs = self.planners['lemniscates'].lemni_target(state,targets,i,t,learn_actions)
            
        elif tactic_type == 'malicious_agent':
            
            self.trajectory = targets.copy()
            
# helpers
# def update_trajectory_args(Agents, Trajectory, Controller, tactic_type, kwargs):
    
#     #kwargs['sorted_neighs'] = Trajectory.sorted_neighs


#     # we'll need the record of lemni parameters  
#     if tactic_type == 'lemniscates':
        
#         #kwargs['sorted_neighs'] = Trajectory.sorted_neighs
        
#         # new bidrirectional controller
#         if 'lemni_CALA_xz' in Controller.Learners:
            
#             # CASE 2: bidirectional
#             kwargs['lemni_learn_actions'] = {
#                 'xz': Controller.Learners['lemni_CALA_xz'].action_set,
#                 }
#         else:
#             kwargs['lemni_learn_actions'] = {
#                'xz': np.zeros((2*Agents.state.shape[1]))
#                }
            
            
#     return kwargs            
            
            