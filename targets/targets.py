#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define the goals/targets

Created on Sun Apr 14 13:07:31 2024

@author: tjards
"""

# import stuff
# ------------
import numpy as np

# define the target object
# ------------------------
class Targets:

    def __init__(self, nVeh):
        
        self.tSpeed  =   0       # speed of target
        
        self.targets = 4*(np.random.rand(6,nVeh)-0.5)
        self.targets[0,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[1,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[2,:] = 15
        self.targets[3,:] = 0
        self.targets[4,:] = 0
        self.targets[5,:] = 0
        
        #self.trajectory = self.targets.copy()
        #self.trajectory = copy.deepcopy(self.targets)
        
        self.config_targets = {'tSpeed': self.tSpeed , 'initial_target_positions': list(self.targets[:,0])} 
        
        
    def evolve(self, t):
        
        self.targets[0,:] = 100*np.sin(self.tSpeed*t)                 # targets[0,:] + tSpeed*0.002
        self.targets[1,:] = 100*np.sin(self.tSpeed*t)*np.cos(self.tSpeed*t)  # targets[1,:] + tSpeed*0.005
        self.targets[2,:] = 100*np.sin(self.tSpeed*t)*np.sin(self.tSpeed*t)+15  # targets[2,:] + tSpeed*0.0005
        