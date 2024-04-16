#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module computes the commands for various swarming strategies 

Note: we have created this separate module to permit mixing and/or sharing between tactic types

Created on Mon Jan  4 12:45:55 2021

@author: tjards

"""

#%% Import stuff
# --------------
import numpy as np
import copy
import os
import json

with open(os.path.join("config", "config_agents.json"), 'r') as tactic_tests:
    tactic_test = json.load(tactic_tests)
    tactic_type = tactic_test['tactic_type']

if tactic_type == 'circle':
    from planner.techniques import encirclement_tools as encircle_tools
    from planner.techniques import saber_tools
elif tactic_type == 'lemni':
    from planner.techniques import lemni_tools
    from planner.techniques import saber_tools
elif tactic_type == 'reynolds':
    from planner.techniques import reynolds_tools
    from planner.techniques import saber_tools
elif tactic_type == 'saber':
    from planner.techniques import saber_tools
elif tactic_type == 'starling':
    from planner.techniques import starling_tools
elif tactic_type == 'shep':
    from planner.techniques import shepherding as shep
elif tactic_type == 'pinning':
    from planner.techniques import pinning_RL_tools as pinning_tools

# from planner.techniques import encirclement_tools as encircle_tools
# from planner.techniques import lemni_tools 
# from planner.techniques import reynolds_tools
# from planner.techniques import saber_tools
# from planner.techniques import starling_tools
# from planner.techniques import shepherding as shep
# from planner.techniques import pinning_RL_tools as pinning_tools


#%% parameters
# ------------

# this needs to pull from SOMEWHERE
# do we want to update the lattice parameters at this level?
hetero_lattice = 1  # nominally, keep at 1 for now 

#%% Tactic Command Equations 
# --------------------------

class Controller:
    
    #def __init__(self, Agents):
    def __init__(self,tactic_type, nAgents, state):
                
        # commands
        # --------
        self.cmd = np.zeros((3,nAgents))
        self.cmd[0] = 0.001*np.random.rand(1,nAgents)-0.5      # command (x)
        self.cmd[1] = 0.001*np.random.rand(1,nAgents)-0.5      # command (y)
        self.cmd[2] = 0.001*np.random.rand(1,nAgents)-0.5      # command (z)

        # other Parameters
        # ----------------
        self.counter = 0                                        # controller counter (signals when to select pins)
        self.params = np.zeros((4,nAgents))                 # store dynamic parameters
        self.lattice = np.zeros((nAgents,nAgents))      # stores lattice parameters

        # initialize pin and components
        self.pin_matrix = np.zeros((nAgents,nAgents))
        self.components = []
        
        if tactic_type == 'shep':
            self.shepherdClass = shep.Shepherding(state)
        
        if tactic_type == 'pinning':
    
            self.pin_matrix, self.components = pinning_tools.select_pins_components(state[0:3,:],state[3:6,:])
            #Agents.pin_matrix = copy.deepcopy(self.pin_matrix) 
   
    # define commands
    # ---------------
    #def commands(self, Agents, Obstacles, Targets, Trajectory, History):   
    def commands(self, state, tactic_type, centroid, targets, obstacles_plus, walls, trajectory, dynamics_type, **mykwargs_cmd):  
 
        #Agents.state, Agents.tactic_type, Agents.centroid, Targets.targets, Obstacles.obstacles_plus, Obstacles.walls, Trajectory.trajectory
        
        # initialize 
        u_int = np.zeros((3,state[0:3,:].shape[1]))     # interactions
        u_obs = np.zeros((3,state[0:3,:].shape[1]))     # obstacles 
        u_nav = np.zeros((3,state[0:3,:].shape[1]))     # navigation
        u_enc = np.zeros((3,state[0:3,:].shape[1]))     # encirclement 
        cmd_i = np.zeros((3,state[0:3,:].shape[1]))     # store the commands
        self.params = np.zeros((state[0:3,:].shape[1],state[0:3,:].shape[1])) # store pins 
            
        # if doing Reynolds, reorder the agents 
        if tactic_type == 'reynolds':
            distances = reynolds_tools.order(state[0:3,:])
            
        # if doing pinning control, select pins (when it's time)
        if tactic_type == 'pinning':
            
            # increment the counter 
            self.counter += 1
            
            # only update the pins at Ts/100 (tunable)
            if self.counter == 100:
                self.counter = 0
                self.pin_matrix, self.components = pinning_tools.select_pins_components(state[0:3,:],state[3:6,:])
                    
                # pass pin_matrix up to agent as well
                #Agents.pin_matrix = copy.deepcopy(self.pin_matrix) # redundant 
    
        # for each vehicle/node in the network
        for k_node in range(state[0:3,:].shape[1]): 
                     
            # Reynolds Flocking
            # ------------------
            if tactic_type == 'reynolds':
               
               cmd_i[:,k_node] = reynolds_tools.compute_cmd( targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node, distances)
               
               # steal obstacle avoidance term from saber
               # ----------------------------------------
               u_obs[:,k_node] = saber_tools.compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
            
            # Saber Flocking
            # ---------------                                
            if tactic_type == 'saber':
                   
                # Lattice Flocking term (phi_alpha)
                # ---------------------------------  
                u_int[:,k_node] = saber_tools.compute_cmd_a(state[0:3,:], state[3:6,:],  targets[0:3,:], targets[3:6,:], k_node)    
            
                # Navigation term (phi_gamma)
                # ---------------------------
                u_nav[:,k_node] = saber_tools.compute_cmd_g(state[0:3,:], state[3:6,:],  targets[0:3,:], targets[3:6,:], k_node)
                              
                # Obstacle Avoidance term (phi_beta)
                # ---------------------------------   
                u_obs[:,k_node] = saber_tools.compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
    
            # Encirclement term (phi_delta)
            # ---------------------------- 
            if tactic_type == 'circle':       
                
                u_enc[:,k_node] = encircle_tools.compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)
                
                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = saber_tools.compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                     
            # Lemniscatic term (phi_lima)
            # ---------------------------- 
            if tactic_type == 'lemni':    
                
                u_enc[:,k_node] = lemni_tools.compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)
                
                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = saber_tools.compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                
                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = saber_tools.compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                      
            # Starling
            # --------
            if tactic_type == 'starling':
               
                # compute command 
                cmd_i[:,k_node], Controller.params = starling_tools.compute_cmd(targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node, self.params, 0.02)
            
            # Pinning
            # --------
            if tactic_type == 'pinning':
                
                # update some arguments 
                my_kwargs = {}
                my_kwargs['pin_matrix'] = self.pin_matrix
                if dynamics_type == 'quadcopter':
                    quads_headings = mykwargs_cmd['quads_headings']
                    my_kwargs['headings'] = quads_headings                    
                    
                # compute command
                cmd_i[:,k_node] = pinning_tools.compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **my_kwargs)
                
                # update the lattice parameters (needed for plots)
                if hetero_lattice == 1:
                    self.lattice = pinning_tools.get_lattices()
                
            # Shepherding
            # ------------
            if tactic_type == 'shep':
                
                # compute command, pin_matrix records shepherds = 1, herd = 0
                # ----------------------------------------------------------

                # compute the commands
                self.shepherdClass.compute_cmd(targets, k_node)
        
                # pull out results
                cmd_i[:,k_node]                 = self.shepherdClass.cmd
                self.pin_matrix[k_node, k_node] = self.shepherdClass.index[self.shepherdClass.i]
  
            # Mixer
            # -----         
            if tactic_type == 'saber':
                cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
            elif tactic_type == 'reynolds':
                cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node] # adds the saber obstacle avoidance 
            elif tactic_type == 'circle':
                cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 
            elif tactic_type == 'lemni':
                cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node]
            elif tactic_type == 'starling':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'pinning':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'shep':
                cmd_i[:,k_node] = cmd_i[:,k_node]

        # update the commands
        self.cmd = copy.deepcopy(cmd_i) 
        





