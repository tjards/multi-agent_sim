#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 

This module computes the commands for various swarming strategies 

The following parameters are used to control the orchestrator:    
    pin_update_rate = 10   # number of timesteps after which we update the pins
    pin_selection_method = 'nopins'
        # gramian   = [future] // based on controllability gramian
        # degree    = based on degree centrality  
        # between   = [future] // based on betweenness centrality (buggy at nAgents < 3)
        # degree_leafs = degree and also leaves (only one connection)
        # nopins      = no pins
        # allpins     = all are pins 
    criteria_table = {'radius': True, 'aperature': False} # for graph construction 
    sensor_aperature    = 140
    learning_ctrl = None  # None, 'CALA'; CALA unverified for now

Created on Mon Jan  4 12:45:55 2021

@author: tjards


"""

#%% Import stuff
# --------------
import numpy as np
import copy
import os
import json

# custom packages
from planner import trajectory
from planner.techniques import pinning_lattice
import utils.swarmgraph as graphical 
import config.config as cfg

# these will be removed later, after I objectify and migrate off conditional imports 
config_loaded = cfg.load_config('config/config.json')
tactic_type = cfg.get_config(config_loaded, 'simulation.strategy')

import learner.conductor


#%% Build the system
# ------------------
def build_system(config):

    if config.system == 'swarm':
    
        # instantiate the agents
        # ------------------------
        import agents.agents as agents
        #Agents = agents.Agents(config.strategy, config.dimens)
        Agents = agents.Agents()

        # instantiate the targets
        # -----------------------
        import targets.targets as targets
        Targets = targets.Targets(config.nAgents, config.dimens)   

        # instantiate the planner
        # -----------------------
        import planner.trajectory as trajectory
        Trajectory = trajectory.Trajectory(config.strategy, Targets.targets, config.nAgents)
        
        # instantiate the obstacles 
        # -------------------------
        import obstacles.obstacles as obstacles  
        #Obstacles = obstacles.Obstacles(config.strategy, Targets.targets, config.dimens)  
        Obstacles = obstacles.Obstacles(Targets.targets)  
        
        # instatiate any learning
        # -----------------------
        import learner.conductor
        #Learners = learner.conductor.initialize(Agents, config.strategy, config.learning_ctrl, config.Ts, config.config_path)
        Learners = learner.conductor.initialize(Agents, config.strategy, config.learning_ctrl, config.Ts, config._data)
                
    return Agents, Targets, Trajectory, Obstacles, Learners

#%% Master controller
# -------------------
class Controller:
    
    def __init__(self, config, state):

        # store config
        self.config = config
        self.Ts     = config.Ts

        # commands
        self.dimens  = config.dimens
        self.cmd = np.zeros((3,config.nAgents))
        self.cmd[0] = 0.001*np.random.rand(1,config.nAgents)-0.5      # command (x)
        self.cmd[1] = 0.001*np.random.rand(1,config.nAgents)-0.5      # command (y)
        self.cmd[2] = 0.001*np.random.rand(1,config.nAgents)-0.5      # command (z)
        #if self.dimens == 2:
        if config.dimens == 2:
            self.cmd[2] = 0*self.cmd[2]

        # general purpose counter (nominally, pin reset)
        self.counter = 0

        # now import the planners (leverage polymorphism as much as possible)
        self.planners            = {} 
        kwargs_init = {'states': state} 
        import importlib
        if config.strategy == 'lemniscates':
                from planner.techniques import encirclement 
                from planner.techniques import lemniscates
                self.planners['encirclement']     = encirclement.Planner(config._data)
                embedding = {'encirclement': self.planners['encirclement']}
                self.planners['lemniscates']      = lemniscates.Planner(config._data, **embedding)
        else:
            planner_module = importlib.import_module(f'planner.techniques.{config.strategy}')
            self.planners[config.strategy] = planner_module.Planner(config._data,**kwargs_init)
        obstacle_avoidance_module = importlib.import_module(f'planner.techniques.{config.obstacle_avoidance_strategy}')
        self.planners['obstacle_avoidance'] = obstacle_avoidance_module.Planner(config._data) 
        # give planners access to sample rate
        self.planners[config.strategy].Ts = self.Ts


        # initialize graphs and pins
        criteria_table = cfg.get_config(config._data, 'orchestrator.criteria_table')
        self.connectivity_slack = cfg.get_config(config._data, 'orchestrator.connectivity_slack') # some slack to assess connectedness
        self.Graphs                 = graphical.Swarmgraph(state, criteria_table)  
        self.Graphs_connectivity    = graphical.Swarmgraph(state, criteria_table)

        self.r_matrix = self.planners[config.strategy].sensor_range_matrix   # range at which agents can sense each other 
        self.lattice = self.planners[config.strategy].connection_range_matrix    # range at which agents are connected     
        self.pin_matrix = self.planners[config.strategy].pin_assignments  # initialize the pin matrix based on the planner (if applicable)

    def update_connections(self, state, tactic_type, **kwargs_cmd):

        # update connectivity
        self.counter += 1                    # increment the counter 

        if self.counter == self.config.pin_update_rate:  # only update the pins at Ts/(tunable parameter)
            self.counter = 0                 # reset counter
            
            # update the range parameters (unlikely to change)
            self.r_matrix = self.planners[tactic_type].sensor_range_matrix 

            # update the connectivity parameteres (from the learning)
            self.planners[tactic_type].connection_range_matrix = self.lattice

            # update the graphs
            self.Graphs.update_A(state[0:3,:], self.r_matrix, **kwargs_cmd)
            self.Graphs_connectivity.update_A(state[0:3,:], self.lattice+self.connectivity_slack, **kwargs_cmd)
            self.Graphs.find_connected_components()
            self.Graphs.update_pins(state[0:3,:], self.r_matrix, self.config.pin_selection_method, **kwargs_cmd)
            self.pin_matrix = self.Graphs.pin_matrix

            # load updated graphs and pins into the planner
            self.planners[tactic_type].update_graphs(A_interaction=self.Graphs.A, A_connectivity=self.Graphs_connectivity.A) 
            self.planners[tactic_type].pin_assignments = self.pin_matrix


    # integrate learninging agents (learning updates happen at the Controller object)
    # ----------------------------
    def learning_agents(self, tactic_type, Learners):
        
        self.Learners = {}

        for name, learner_obj in Learners.items():
            self.Learners[name] = learner_obj

        if not self.Learners:
            print('Note: controller has no learning agents')
        
        
    # define commands
    # --------------- 
    def commands(self, state, tactic_type, centroid, targets, obstacles_plus, walls, trajectory, dynamics_type, **kwargs_cmd):  
        
        # initialize 
        cmd_i = np.zeros((3,state[0:3,:].shape[1]))     # store the commands
        u_obs = np.zeros((3,state[0:3,:].shape[1]))     # obstacles 

        # ************* #
        # ARGUMENTS     #
        # ************* #
        
        # Update arguments needed below 
        kwargs_cmd['centroid']          = centroid 
        kwargs_cmd['obstacles_plus']    = obstacles_plus
        kwargs_cmd['walls']             = walls

        # pinning has a lot of unique needs
        if tactic_type == 'pinning_lattice':

            # parameters from consensus class
            if 'consensus_lattice' in self.Learners:
                kwargs_cmd['d_weighted'] = self.Learners['consensus_lattice'].d_weighted
            # or just pass the current lattice parameters
            else:
                kwargs_cmd['d_weighted'] = self.lattice # redundant below
            # pass in args required for learning
            kwargs_cmd = learner.conductor.pinning_update_args(self, kwargs_cmd)
            kwargs_cmd['directional_graph']         = self.Graphs.directional_graph
            kwargs_cmd['local_k_connectivity']      = self.Graphs.local_k_connectivity
        
        # these should go in reynolds (later)
        if tactic_type == 'flocking_reynolds':
            distances = self.planners['flocking_reynolds'].order(state[0:3,:])
            kwargs_cmd['distances'] = distances

        # ************* #
        # GRAPH UPDATES #
        # ************* #
    
        self.update_connections(state, tactic_type, **kwargs_cmd)
  
        # *************** #
        # COMMAND UPDATES #
        # *************** #

        # for each vehicle/node/agent in the network
        # ------------------------------------
        for k_node in range(state[0:3,:].shape[1]):

            # compute the planner command for this node
            cmd_i[:,k_node] = self.planners[tactic_type].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node, **kwargs_cmd)

            # compute the obstacle avoidance command for this node (if applicable)
            if tactic_type in ['flocking_reynolds', 'encirclement', 'lemniscates']:
                u_obs[:,k_node] = self.planners['obstacle_avoidance'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)

            # force pins to reflect shepherding (if applicable)
            if tactic_type == 'shepherding':
                self.pin_matrix[k_node, k_node] = self.planners['shepherding'].index[k_node]

            # ****************** #
            #  Learning updates  #
            # ****************** # 

            # pinning lattice learning update (if applicable)
            if tactic_type == 'pinning_lattice':
                 learner.conductor.pinning_update_lattice(self)
            
            # lemniscate orientation learn
            if tactic_type == 'lemniscates':
                learner.conductor.lemniscate_update(self, state, targets, k_node, **kwargs_cmd)
            
            # ******* #
            #  Mixer  #
            # ******* # 

            cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node]

            if self.dimens == 2 and self.cmd[2,:].any() != 0:
                
                print('warning, 3D cmds in 2D at node ', k_node)
                print(self.cmd[2,k_node])    
                
        # update the commands
        self.cmd = copy.deepcopy(cmd_i) 
        

        





