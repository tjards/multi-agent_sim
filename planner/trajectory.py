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

# with open(os.path.join("config", "config_agents.json"), 'r') as tactic_tests:
#     tactic_test = json.load(tactic_tests)
#     tactic_type = tactic_test['tactic_type']
    
with open(os.path.join("config", "configs.json"), 'r') as tactic_tests:
    tactic_test = json.load(tactic_tests)
    tactic_type = tactic_test['simulation']['strategy']
    
if tactic_type == 'circle':
    from .techniques import encirclement_tools as encircle_tools
elif tactic_type == 'lemni':
    from .techniques import lemni_tools 

# define the trajectory object
# ----------------------------
class Trajectory:
    
    def __init__(self, tactic_type, targets, nAgents):
        
        self.trajectory = copy.deepcopy(targets)
        #self.lemni = np.zeros([1, nAgents])
        self.lemni = np.zeros([2, nAgents])
        self.sorted_neighs = list(range(nAgents))
    
    # WARNING: untested code
    '''
    def exclude(self, state, targets, lemni_all, exclusion):
        
        # [LEGACY] create a temp exlusionary set
        state_ = np.delete(state, [exclusion], axis = 1)
        targets_ = np.delete(targets, [exclusion], axis = 1)
        lemni_all_ = np.delete(lemni_all, [exclusion], axis = 1)
        
        return state_, targets_, lemni_all_
    '''
    
    # WARNING: untested code
    '''
    def unexclude(self, trajectory, targets, lemni, lemni_all, pins_all, i, exclusion):
        
        # [LEGACY] add exluded back in
        for ii in exclusion:
            trajectory = np.insert(trajectory,ii,targets[:,ii],axis = 1)
            trajectory[0:2,ii] = ii + 5 # just move away from the swarm
            trajectory[2,ii] = 15 + ii 
            lemni = np.insert(lemni,ii,lemni_all[i-1,ii],axis = 1)
            # label excluded as pins (for plotting colors only)
            pins_all[i-1,ii,ii] = 1  
            
            return trajectory, lemni, pins_all
    '''
    
    #def update(self, Agents, Targets, History, t, i):
    def update(self, tactic_type, state, targets, t, i, **kwargs):
        
        #if flocking
        if tactic_type == 'reynolds' or tactic_type == 'saber' or tactic_type == 'starling' or tactic_type == 'pinning' or tactic_type == 'shep':
            self.trajectory = targets.copy() 
        
        # if encircling
        if tactic_type == 'circle':
            #from .techniques import encirclement_tools as encircle_tools
            self.trajectory, _, _ = encircle_tools.encircle_target(targets, state)
        
        # if lemniscating
        elif tactic_type == 'lemni':
            #from .techniques import lemni_tools
            
            lemni_all = kwargs.get('lemni_all')
            #Agents = kwargs.get('lemni')
            
            if 'lemni_learn_actions' in kwargs:
                learn_actions = kwargs.get('lemni_learn_actions')
            else:
                learn_actions = {
                   'x': np.zeros((state.shape[1])),
                   'z': np.zeros((state.shape[1]))
                   }
                
            
            self.trajectory, self.lemni, self.sorted_neighs = lemni_tools.lemni_target(lemni_all,state,targets,i,t,learn_actions)
            
            
        elif tactic_type == 'cao':
            
            self.trajectory = targets.copy()
            
            
            
            