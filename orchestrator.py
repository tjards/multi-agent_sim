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
from planner.techniques import pinning_RL_tools
from planner.techniques import pinning_RL_tools
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

        # initialize graphs
        criteria_table = cfg.get_config(config._data, 'orchestrator.criteria_table')
        self.Graphs                 = graphical.Swarmgraph(state, criteria_table)  
        self.Graphs_connectivity    = graphical.Swarmgraph(state, criteria_table)

        # initialize planner dictionary
        self.planners            = {} # dictionary for planners

        # commands
        # --------
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
        
        # [legacy] general purpose parameters variable (retire this eventually, just used for starling now)
        #self.params = np.zeros((4,config.nAgents))             # store dynamic parameters
        
        # lattice parameters (not always used, move into pinning) 
        self.lattice = np.zeros((config.nAgents,config.nAgents))      # stores lattice parameters
        
        # pins and components (not always used)
        self.pin_matrix = np.zeros((config.nAgents,config.nAgents))
                
        if config.strategy == 'pinning':
            from planner.techniques import pinning_RL_tools as pinning_tools
            self.planners['pinning'] = pinning_tools.Planner(config._data)
            #self.pin_matrix = np.ones((self.nAgents,self.nAgents))# make all pins
            self.pin_matrix = np.ones((config.nAgents,config.nAgents))# make all pins
            #d_init = pinning_tools.return_lattice_param()
            d_init = self.planners['pinning'].d_init
            self.d_init= d_init
            i = 0
            #while (i < self.nAgents):
            while (i < config.nAgents):
                self.lattice[i,:] = d_init
                i+=1
            #self.Graphs_connectivity = graphical.Swarmgraph(state, criteria_table)
        
        # starling
        if config.strategy == 'starling':
            from planner.techniques import starling_tools
            self.planners['starling'] = starling_tools.Planner(config._data)
            #self.params = np.zeros((4, config.nAgents)) 

        if config.strategy == 'cao':
            from planner.techniques import cao_tools
            self.planners['cao'] = cao_tools.Planner(config._data, state[0:3,:], state[3:6,:])
            #self.lattice = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1])) 
            self.lattice = self.planners['cao'].d*np.ones((state.shape[1],state.shape[1]))
            self.pin_matrix = np.ones((config.nAgents,config.nAgents))

        # sheparding has its own class (differentiating shepherd and herd)
        if config.strategy == 'shep':
            #self.shepherdClass = shep.Shepherding(state)
            from planner.techniques import shepherding
            self.planners['shep'] = shepherding.Planner(config._data, state) 
        
        # the planners that rely on saber tools for obstacle avoidance
        if config.strategy in ['saber', 'circle', 'lemni', 'reynolds']:
            from planner.techniques import saber_tools
            self.planners['saber'] = saber_tools.Planner(config._data)
            if config.strategy == 'reynolds':
                from planner.techniques import reynolds_tools
                self.planners['reynolds'] = reynolds_tools.Planner(config._data)
            if config.strategy == 'saber':
                #self.lattice = self.planners['saber'].return_ranges()*np.ones((state.shape[1],state.shape[1]))
                self.lattice = self.planners['saber'].d*np.ones((state.shape[1],state.shape[1]))
            if config.strategy == 'circle':
                from planner.techniques import encirclement_tools 
                self.planners['circle'] = encirclement_tools.Planner(config._data)
            if config.strategy == 'lemni':
                from planner.techniques import encirclement_tools 
                from planner.techniques import lemni_tools
                self.planners['circle']     = encirclement_tools.Planner(config._data)
                self.planners['lemni']      = lemni_tools.Planner(config._data, self.planners['circle'])

        
        # cao has it's own class and a separate graph for connected (in addition to in range)    
        #if config.strategy == 'cao':
        #    self.caoClass = cao_tools.Flock(state[0:3,:],state[3:6,:])
        #    self.lattice = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1])) 
        #    self.pin_matrix = np.ones((config.nAgents,config.nAgents))
            
   
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
        
        # note: a lot of this can be consolidated and cleaned up
        
        # update connectivity
        self.counter += 1                    # increment the counter 
        if self.counter == self.config.pin_update_rate:  # only update the pins at Ts/(tunable parameter)
            self.counter = 0                 # reset counter
            
            # update connectivity parameters 
            if tactic_type == 'saber':
                r_matrix = self.planners['saber'].d*np.ones((state.shape[1],state.shape[1]))

            elif tactic_type == 'circle':
                r_matrix = self.planners['circle'].desired_separation*np.ones((state.shape[1],state.shape[1]))

            elif tactic_type == 'cao':
                #r_matrix = cao_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
                r_matrix = self.planners['cao'].r*np.ones((state.shape[1],state.shape[1]))
                # new, define a different graph for "connected", which is slightly different than "in range"
                #separation_matrix = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1]))
                self.Graphs_connectivity.update_A(state[0:3,:], self.lattice+1, **kwargs_cmd)
                
            elif tactic_type == 'pinning':
                # in pull parameters from consensus class
                if 'consensus_lattice' in self.Learners:
                    kwargs_cmd['d_weighted'] = self.Learners['consensus_lattice'].d_weighted
                # or just pass the current lattice parameters
                else:
                    kwargs_cmd['d_weighted'] = self.lattice # redundant below
                kwargs_cmd['aperature'] = self.config.sensor_aperature
                #r_matrix = kwargs_cmd['d_weighted']  # if we want the graph based on lattice parameters
                #r_matrix = pinning_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
                r_matrix = self.planners['pinning'].r_max*np.ones((state.shape[1],state.shape[1]))
                self.Graphs_connectivity.update_A(state[0:3,:], self.lattice, **kwargs_cmd)
            else:
                r_matrix = 0*np.ones((state.shape[1],state.shape[1]))

            # update the graph
            self.Graphs.update_A(state[0:3,:], r_matrix, **kwargs_cmd)
            #self.Graphs.update_A(state[0:3,:], self.lattice, **kwargs_cmd)

            self.Graphs.find_connected_components()
            self.Graphs.update_pins(state[0:3,:], r_matrix, self.config.pin_selection_method, **kwargs_cmd)
            self.pin_matrix = self.Graphs.pin_matrix


        # *************** #
        # COMMAND UPDATES #
        # *************** #

        # reynolds requires a matrix of distances between agents
        if tactic_type == 'reynolds':
            distances = self.planners['reynolds'].order(state[0:3,:])
   
        # for each vehicle/node/agent in the network
        # ------------------------------------
        for k_node in range(state[0:3,:].shape[1]): 
                     
            # Reynolds Flocking
            # ------------------
            if tactic_type == 'reynolds':
               cmd_i[:,k_node] = self.planners['reynolds'].compute_cmd(targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node, distances)

               # steal obstacle avoidance term from saber
               # ----------------------------------------
               u_obs[:,k_node] = self.planners['saber'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
            
            # Saber Flocking
            # ---------------                                
            if tactic_type == 'saber':
                   
                # Lattice Flocking term (phi_alpha)
                # ---------------------------------  
                u_int[:,k_node] = self.planners['saber'].compute_cmd_a(state[0:3,:], state[3:6,:], k_node)

                # Navigation term (phi_gamma)
                # ---------------------------
                u_nav[:,k_node] = self.planners['saber'].compute_cmd_g(state[0:3,:], state[3:6,:], targets[0:3,:], targets[3:6,:], k_node)              
                
                # Obstacle Avoidance term (phi_beta)
                # ---------------------------------   
                u_obs[:,k_node] = self.planners['saber'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)

            # Encirclement term (phi_delta)
            # ---------------------------- 
            if tactic_type == 'circle':       
                
                u_enc[:,k_node] = self.planners['circle'].compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)

                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = self.planners['saber'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)
                     
            # Lemniscatic term (phi_lima)
            # ---------------------------- 
            if tactic_type == 'lemni':    
                
                u_enc[:,k_node] = self.planners['lemni'].compute_cmd(state[0:3,:], state[3:6,:], trajectory[0:3,:],trajectory[3:6,:], k_node)

                # steal obstacle avoidance term from saber
                # ----------------------------------------
                u_obs[:,k_node] = self.planners['saber'].compute_cmd_b(state[0:3,:], state[3:6,:], obstacles_plus, walls, k_node)

            # Starling
            # --------
            if tactic_type == 'starling':
               
                # compute command 
                #cmd_i[:,k_node], self.params = starling_tools.compute_cmd(targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node, self.params, 0.02)
                cmd_i[:, k_node] = self.planners['starling'].compute_cmd(targets[0:3,:], centroid, state[0:3,:], state[3:6,:], k_node)

            # Pinning
            # --------
            if tactic_type == 'pinning':
                
                # arguments (move the graph stuff up later, also, standardize the kwargs across all techniques)
                # ---------
                
                # initialize some custom arguments 
                kwargs_pinning = {}
                # pin matrix
                kwargs_pinning['pin_matrix'] = self.pin_matrix
                # headings (if applicable)
                if dynamics_type == 'quadcopter':
                    kwargs_pinning['quads_headings'] = kwargs_cmd['quads_headings']
                
                # pass in args required for learning
                kwargs_pinning = learner.conductor.pinning_update_args(self, kwargs_pinning)
                # info about graph
                kwargs_pinning['directional_graph']         = self.Graphs.directional_graph
                kwargs_pinning['A']                         = self.Graphs.A
                kwargs_pinning['D']                         = self.Graphs.D
                kwargs_pinning['local_k_connectivity']      = self.Graphs.local_k_connectivity
                
                # compute command
                #_, u_int[:,k_node], u_nav[:,k_node], u_obs[:,k_node] = pinning_tools.compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                _, u_int[:,k_node], u_nav[:,k_node], u_obs[:,k_node] = self.planners['pinning'].compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                # learning update 
                learner.conductor.pinning_update_lattice(self)
                       
            # Shepherding
            # ------------
            if tactic_type == 'shep':
                
                # compute command, pin_matrix records shepherds = 1, herd = 0
                # ----------------------------------------------------------

                # compute the commands
                self.planners['shep'].compute_cmd(targets, k_node)
                
                # pull out results
                cmd_i[:,k_node]                 = self.planners['shep'].cmd
                
                # assign the pin matrix (roles)
                self.pin_matrix[k_node, k_node] = self.planners['shep'].index[k_node]


            # Cao flocking 
            # ------------
            if tactic_type == 'cao':
                             
                kwargs_cao          = {}
                kwargs_cao['A']     = self.Graphs.A
                kwargs_cao['A_connectivity']     = self.Graphs_connectivity.A
                kwargs_cao['pin_matrix']  = self.pin_matrix
                kwargs_cao['Ts'] = self.Ts
                #cmd_i[:,k_node] = cao_tools.compute_cmd(targets[0:3,:],state[0:3,:], state[3:6,:], k_node, **kwargs_cao)
                #cmd_i[:,k_node] = self.caoClass.compute_cmd(targets[0:3,:],state[0:3,:], state[3:6,:], k_node, **kwargs_cao)
                #self.lattice = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1]))
                #print(self.caoClass.status)
                #print(self.caoClass.layer)

                cmd_i[:,k_node] = self.planners['cao'].compute_cmd(targets[0:3,:], state[0:3,:], state[3:6,:], k_node, **kwargs_cao)

            
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
                #cmd_i[:,k_node] = cmd_i[:,k_node]
                cmd_i[:,k_node] = u_int[:,k_node] + u_obs[:,k_node] + u_nav[:,k_node] 
            elif tactic_type == 'shep':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'cao':
                cmd_i[:,k_node] = cmd_i[:,k_node]

            
            if self.dimens == 2 and self.cmd[2,:].any() != 0:
                
                print('warning, 3D cmds in 2D at node ', k_node)
                print(self.cmd[2,k_node])    
                
        # update the commands
        self.cmd = copy.deepcopy(cmd_i) 
        

        





