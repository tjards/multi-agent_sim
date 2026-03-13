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

        # initialize graphs and pins
        criteria_table = cfg.get_config(config._data, 'orchestrator.criteria_table')
        self.connectivity_slack = cfg.get_config(config._data, 'orchestrator.connectivity_slack') # some slack to assess connectedness
        self.Graphs                 = graphical.Swarmgraph(state, criteria_table)  
        self.Graphs_connectivity    = graphical.Swarmgraph(state, criteria_table)

        self.r_matrix = self.planners[config.strategy].sensor_range_matrix   # range at which agents can sense each other 
        self.lattice = self.planners[config.strategy].connection_range_matrix    # range at which agents are connected     
        self.pin_matrix = self.planners[config.strategy].pin_assignments  # initialize the pin matrix based on the planner (if applicable)


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
        u_int = np.zeros((3,state[0:3,:].shape[1]))     # interactions
        u_obs = np.zeros((3,state[0:3,:].shape[1]))     # obstacles 
        u_nav = np.zeros((3,state[0:3,:].shape[1]))     # navigation
        u_enc = np.zeros((3,state[0:3,:].shape[1]))     # encirclement 
        cmd_i = np.zeros((3,state[0:3,:].shape[1]))     # store the commands
        #self.params = np.zeros((state[0:3,:].shape[1],state[0:3,:].shape[1])) # store pins 
        
        # ************* #
        # GRAPH UPDATES #
        # ************* #
        
        kwargs_cmd['centroid'] = centroid 
        kwargs_cmd['obstacles_plus'] = obstacles_plus
        kwargs_cmd['walls'] = walls
        kwargs_cmd['A']     = self.Graphs.A
        kwargs_cmd['A_connectivity']     = self.Graphs_connectivity.A
        kwargs_cmd['pin_matrix']  = self.pin_matrix
        kwargs_cmd['Ts'] = self.Ts
        if tactic_type == 'pinning_lattice':
            # in pull parameters from consensus class
            if 'consensus_lattice' in self.Learners:
                kwargs_cmd['d_weighted'] = self.Learners['consensus_lattice'].d_weighted
            # or just pass the current lattice parameters
            else:
                kwargs_cmd['d_weighted'] = self.lattice # redundant below
    
        # update connectivity
        self.counter += 1                    # increment the counter 
        if self.counter == self.config.pin_update_rate:  # only update the pins at Ts/(tunable parameter)
            self.counter = 0                 # reset counter
            
            # update the range parameters (unlikely to change)
            self.r_matrix = self.planners[tactic_type].sensor_range_matrix 

            # update the graphs
            self.Graphs.update_A(state[0:3,:], self.r_matrix, **kwargs_cmd)
            self.Graphs_connectivity.update_A(state[0:3,:], self.lattice+self.connectivity_slack, **kwargs_cmd)
            self.Graphs.find_connected_components()
            self.Graphs.update_pins(state[0:3,:], self.r_matrix, self.config.pin_selection_method, **kwargs_cmd)
            self.pin_matrix = self.Graphs.pin_matrix


        # *************** #
        # COMMAND UPDATES #
        # *************** #
        

        # reynolds requires a matrix of distances between agents
        if tactic_type == 'flocking_reynolds':
            distances = self.planners['flocking_reynolds'].order(state[0:3,:])
   
        # for each vehicle/node/agent in the network
        # ------------------------------------
        for k_node in range(state[0:3,:].shape[1]):
                     
            # Reynolds Flocking
            # ------------------
            if tactic_type == 'flocking_reynolds':
               #cmd_i[:,k_node] = self.planners['flocking_reynolds'].compute_cmd(targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node, distances)
               kwargs_cmd['distances'] = distances
               cmd_i[:,k_node] = self.planners['flocking_reynolds'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node, **kwargs_cmd)

               # steal obstacle avoidance term from saber
               # ----------------------------------------
               u_obs[:,k_node] = self.planners['obstacle_avoidance'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
            
            # Saber Flocking
            # ---------------                                
            if tactic_type == 'flocking_saber':

                cmd_i[:,k_node] = self.planners['flocking_saber'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node, **kwargs_cmd)
                
                '''
                # Lattice Flocking term (phi_alpha)
                # ---------------------------------  
                u_int[:,k_node] = self.planners['flocking_saber'].compute_cmd_a(state[0:3,:], state[3:6,:], k_node)

                # Navigation term (phi_gamma)
                # ---------------------------
                u_nav[:,k_node] = self.planners['flocking_saber'].compute_cmd_g(state[0:3,:], state[3:6,:], targets[0:3,:], targets[3:6,:], k_node)              
                
                # Obstacle Avoidance term (phi_beta)
                # ---------------------------------   
                u_obs[:,k_node] = self.planners['flocking_saber'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                '''
                

            # Encirclement term (phi_delta)
            # ---------------------------- 
            if tactic_type == 'encirclement':       
                
                #u_enc[:,k_node] = self.planners['encirclement'].compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)
                u_enc[:,k_node] = self.planners['encirclement'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node)


                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = self.planners['obstacle_avoidance'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                     
            # Lemniscatic term (phi_lima)
            # ---------------------------- 
            if tactic_type == 'lemniscates':    
                
                #u_enc[:,k_node] = self.planners['lemniscates'].compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)
                u_enc[:,k_node] = self.planners['lemniscates'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node)

                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = self.planners['obstacle_avoidance'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)

            # Starling
            # --------
            if tactic_type == 'flocking_starling':
               
                # compute command 
                cmd_i[:, k_node] = self.planners['flocking_starling'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node, **kwargs_cmd)

            # Pinning
            # --------
            if tactic_type == 'pinning_lattice':
                
                # arguments (move the graph stuff up later, also, standardize the kwargs across all techniques)
                # ---------
                
                # initialize some custom arguments 
                #kwargs_pinning = {}
                # pin matrix
                kwargs_cmd['pin_matrix'] = self.pin_matrix
                # headings (if applicable)
                if dynamics_type == 'quadcopter':
                    kwargs_cmd['quads_headings'] = kwargs_cmd['quads_headings']
                
                # pass in args required for learning
                kwargs_cmd = learner.conductor.pinning_update_args(self, kwargs_cmd)
                # info about graph
                kwargs_cmd['directional_graph']         = self.Graphs.directional_graph
                kwargs_cmd['A']                         = self.Graphs.A
                kwargs_cmd['D']                         = self.Graphs.D
                kwargs_cmd['local_k_connectivity']      = self.Graphs.local_k_connectivity
                
                # compute command
                #_, u_int[:,k_node], u_nav[:,k_node], u_obs[:,k_node] = pinning_tools.compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                #_, u_int[:,k_node], u_nav[:,k_node], u_obs[:,k_node] = self.planners['pinning_lattice'].compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                cmd_i[:, k_node] = self.planners['pinning_lattice'].compute_cmd(state[0:6,:], trajectory[0:6,:], k_node, **kwargs_cmd)

                
                # learning update 
                learner.conductor.pinning_update_lattice(self)
                       
            # Shepherding
            # ------------
            if tactic_type == 'shepherding':
                
                # compute command, pin_matrix records shepherds = 1, herd = 0
                # ----------------------------------------------------------                

                cmd_i[:,k_node] = self.planners['shepherding'].compute_cmd(state[0:6,:], targets[0:6,:], k_node, **kwargs_cmd) 

                # assign the pin matrix (roles)
                self.pin_matrix[k_node, k_node] = self.planners['shepherding'].index[k_node]


            # flocking with a malicious agent 
            # -------------------------------
            if tactic_type == 'malicious_agent':
                            
                cmd_i[:,k_node] = self.planners['malicious_agent'].compute_cmd(state[0:6,:], targets[0:6,:], k_node, **kwargs_cmd)
            
            # Apply controller learning (move all this inside learner later)
            #---------------------------
     
            # controller parameter tuning (untested)
            '''if learning_ctrl == 'CALA':
                
                # apply learned gain to control input
                u_int = self.Learners['CALA_ctrl'].action_set[k_node]*u_int
                # compute reward for this step
                reward = self.Learners['CALA_ctrl'].update_reward_increment(k_node, state, centroid)
                # select action for next step
                self.Learners['CALA_ctrl'].step(k_node, reward)'''
            
            # lemniscate orientation learn
            if 'lemni_CALA_xz' in self.Learners:
                
                # define the target
                my_target = obstacles_plus
                allow_ext_reward = False
                
                self.Learners['lemni_CALA_xz'].learn_lemni(
                   state=k_node,
                   state_array=state,
                   centroid=centroid,
                   focal=targets[0:3,k_node],
                   target=my_target,
                   neighbours=kwargs_cmd['sorted_neighs'],
                   mode = 'xz',
                   allow_ext_reward = allow_ext_reward, 
                   ext_reward = 0
                   )
            
            # ******* #
            #  Mixer  #
            # ******* # 

            if tactic_type == 'flocking_saber':
                cmd_i[:,k_node] = cmd_i[:,k_node] #u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
            elif tactic_type == 'flocking_reynolds':
                cmd_i[:,k_node] = cmd_i[:,k_node] + u_obs[:,k_node] # adds the saber obstacle avoidance 
            elif tactic_type == 'encirclement':
                cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node] 
            elif tactic_type == 'lemniscates':
                cmd_i[:,k_node] = u_obs[:,k_node] + u_enc[:,k_node]
            elif tactic_type == 'flocking_starling':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'pinning_lattice':
                cmd_i[:,k_node] = cmd_i[:,k_node]
                #cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
            elif tactic_type == 'shepherding':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'malicious_agent':
                cmd_i[:,k_node] = cmd_i[:,k_node]

            
            if self.dimens == 2 and self.cmd[2,:].any() != 0:
                
                print('warning, 3D cmds in 2D at node ', k_node)
                print(self.cmd[2,k_node])    
                
        # update the commands
        self.cmd = copy.deepcopy(cmd_i) 
        

        





