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

# dev!
import utils.swarmgraph as graphical 

# read the configs
with open(os.path.join("config", "config_sim.json"), 'r') as tactic_tests:
    tactic_test = json.load(tactic_tests)
    tactic_type = tactic_test['strategy']

# load the modules, as appropriate
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
    with open(os.path.join("config", "config_planner_pinning.json"), 'r') as planner_pinning_tests:
        planner_configs = json.load(planner_pinning_tests)
        lattice_consensus = planner_configs['hetero_lattice']


#%% Hyperparameters 
# -----------------

pin_update_rate = 100   # number of timesteps after which we update the pins
pin_selection_method = 'degree'
    # gramian   = based on controllability gramian
    # degree    = based on degree centrality 
    # between   = based on betweenness centrality (buggy at nAgents < 3)
criteria_table = {'radius': True, 'aperature': True} # for graph construction 
sensor_aperature    = 140

#%% Build the system
# ------------------
def build_system(system, strategy):
    
    if system == 'swarm':
    
        # instantiate the agents
        # ------------------------
        import agents.agents as agents
        Agents = agents.Agents(strategy)
        with open(os.path.join("config", "config_agents.json"), 'w') as configs_agents:
            json.dump(Agents.config_agents, configs_agents)
        
        # instantiate the targets
        # -----------------------
        import targets.targets as targets
        Targets = targets.Targets(Agents.nAgents)
        with open(os.path.join("config", "config_targets.json"), 'w') as configs_targets:
            json.dump(Targets.config_targets, configs_targets)
    
        # instantiate the planner
        # -----------------------
        import planner.trajectory as trajectory
        Trajectory = trajectory.Trajectory(Agents.tactic_type, Targets.targets, Agents.nAgents)
    
        # instantiate the obstacles 
        # -------------------------
        import obstacles.obstacles as obstacles
        Obstacles = obstacles.Obstacles(Agents.tactic_type, Targets.targets)
        with open(os.path.join("config", "config_obstacles.json"), 'w') as configs_obstacles:
            json.dump(Obstacles.config_obstacles, configs_obstacles)
            
        # instatiate any learning
        # -----------------------
        Learners = {}
        
        # pinning control case
        if tactic_type == 'pinning':
            with open(os.path.join("config", "config_planner_pinning.json"), 'r') as planner_pinning_tests:
                planner_configs = json.load(planner_pinning_tests)
                
                # need one learner to achieve consensus on lattice size
                lattice_consensus = planner_configs['hetero_lattice']
                if lattice_consensus == 1:
                    import learner.consensus_lattice as consensus_lattice
                    #Consensuser = consensus_lattice.Consensuser(Agents.nAgents, 1, planner_configs['directional'], planner_configs['d_min'], planner_configs['d'])
                    Consensuser = consensus_lattice.Consensuser(Agents.nAgents, 1, planner_configs['d_min'], planner_configs['d'])
                    
                    #LOAD
                    Learners['consensus_lattice'] = Consensuser
                    
                    # we can also tune these lattice sizes (optional)
                    lattice_learner = planner_configs['learning']
                    if lattice_learner == 1:
                        import learner.QL_learning_lattice as learning_lattice
                        
                        # initiate the learning agent
                        Learning_agent = learning_lattice.q_learning_agent(Consensuser.params_n)
                        
                        # ensure parameters match controller
                        if Consensuser.d_weighted.shape[1] != len(Learning_agent.action):
                            raise ValueError("Error! Mis-match in dimensions of controller and RL parameters")
                        
                        # overide the module-level parameter selection
                        for i in range(Consensuser.d_weighted.shape[1]):
                            Learning_agent.match_parameters_i(Consensuser, i)
                            
                        # LOAD    
                        Learners['learning_lattice'] = Learning_agent
    
    return Agents, Targets, Trajectory, Obstacles, Learners

#%% Master controller
# -------------------
class Controller:
    
    def __init__(self,tactic_type, nAgents, state):
                
        # commands
        # --------
        self.cmd = np.zeros((3,nAgents))
        self.cmd[0] = 0.001*np.random.rand(1,nAgents)-0.5      # command (x)
        self.cmd[1] = 0.001*np.random.rand(1,nAgents)-0.5      # command (y)
        self.cmd[2] = 0.001*np.random.rand(1,nAgents)-0.5      # command (z)

        # other Parameters
        # ----------------
        #self.Graphs = graphical.Swarmgraph(state, criteria_table = {'radius': 5}, headings = None)
        #self.Graphs = graphical.Swarmgraph(state, criteria_table = {'radius': 5, 'aperature': 120}, **kwargs)  
        #criteria_table = {'radius': True, 'aperature': False}
        self.Graphs = graphical.Swarmgraph(state, criteria_table) # initialize 
        
        # general purpose counter (nominally, pin reset)
        self.counter = 0                
        
        # [legacy] general purpose parameters variable (retire this)
        self.params = np.zeros((4,nAgents))             # store dynamic parameters
        
        # lattice parameters (not always used, move into pinning) 
        self.lattice = np.zeros((nAgents,nAgents))      # stores lattice parameters
        
        # pins and components (not always used)
        self.pin_matrix = np.zeros((nAgents,nAgents))
        #self.components = []
                
        if tactic_type == 'pinning':
    
            #self.pin_matrix, self.components = pinning_tools.select_pins_components(state[0:3,:],state[3:6,:])
            #self.pin_matrix, self.components = np.ones((nAgents,nAgents)), list(range(0,nAgents)) # make all pins
            self.pin_matrix = np.ones((nAgents,nAgents))# make all pins
            d_init = pinning_tools.return_lattice_param()
            self.d_init= d_init
            i = 0
            while (i < nAgents):
                self.lattice[i,:] = d_init
                i+=1
        
        # sheparding has its own class (differentiating shepherd and herd)
        if tactic_type == 'shep':
            self.shepherdClass = shep.Shepherding(state)            
   
    # integrate learninging agents
    # ----------------------------
    def learning_agents(self, tactic_type, Learners):
        
        self.Learners = {}
        
        # merge learning with controllers (add this to the orchestrator later)
        # note: may be better to just run through this list, rather than explicitly loading each
        
        if tactic_type == 'pinning' and 'consensus_lattice' in Learners:
            
             self.Learners['consensus_lattice'] = Learners['consensus_lattice']
             print('Consensus-based lattice negotiation enabled')
             
             if 'learning_lattice' in Learners:
                 
                 self.Learners['learning_lattice'] = Learners['learning_lattice']
                 print('Q-Learning based lattice optimization enabled')
        else:
            
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
        self.params = np.zeros((state[0:3,:].shape[1],state[0:3,:].shape[1])) # store pins 
         
        # note:
            
            # ADD THE GRAPH COMPONENTS HERE, then the controller
            # pass in components to help direct the search
        
        # reynolds requires a matrix of distances between agents
        if tactic_type == 'reynolds':
            distances = reynolds_tools.order(state[0:3,:])
            
        # update the pins at the desired interval
        if tactic_type == 'pinning':
            # increment the counter 
            self.counter += 1
            # only update the pins at Ts/(tunable parameter)
            if self.counter == pin_update_rate:
                self.counter = 0
                
                # in pull parameters from consensus class
                if 'consensus_lattice' in self.Learners:
                    kwargs_cmd['d_weighted'] = self.Learners['consensus_lattice'].d_weighted
                # or just pass the current lattice parameters
                else:
                    kwargs_cmd['d_weighted'] = self.lattice # redundant below
                    
                # compute the pins and components   (old way)  
                #self.pin_matrix, self.components = pinning_tools.select_pins_components(state[0:3,:],state[3:6,:], **kwargs_cmd)
                
                # dev! (new way)
                #r_matrix = 5*np.ones((state.shape[1],state.shape[1])) # test
                r_matrix = kwargs_cmd['d_weighted']                             # if we want the graph based on lattice parameters
                #r_matrix= self.d_init*np.ones((state.shape[1],state.shape[1]))  # if we want based on actual sensor radius
                kwargs_cmd['aperature'] = sensor_aperature
                self.Graphs.update_A(state[0:3,:], r_matrix, **kwargs_cmd)
                self.Graphs.find_connected_components()
                self.Graphs.update_pins(state[0:3,:], r_matrix, pin_selection_method, **kwargs_cmd)
                self.pin_matrix = self.Graphs.pin_matrix
                
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
                
                # update connectivity
                # -------------------
                r_matrix = saber_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
                self.Graphs.update_A(state[0:3,:], r_matrix)
    
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
                
                # initialize some custom arguments 
                kwargs_pinning = {}
                
                # pin matrix
                kwargs_pinning['pin_matrix'] = self.pin_matrix
                
                # headings (if applicable)
                if dynamics_type == 'quadcopter':
                    kwargs_pinning['quads_headings'] = kwargs_cmd['quads_headings']
                    
                # learning stuff (if applicable)
                if 'consensus_lattice' in self.Learners:
                    kwargs_pinning['consensus_lattice'] = self.Learners['consensus_lattice']
                    if 'learning_lattice' in self.Learners:
                        kwargs_pinning['learning_lattice'] = self.Learners['learning_lattice']
                
                # info about graph
                kwargs_pinning['directional_graph'] = self.Graphs.directional_graph
                kwargs_pinning['A']      = self.Graphs.A
                kwargs_pinning['D']      = self.Graphs.D 
                            
                # compute command
                cmd_i[:,k_node] = pinning_tools.compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                
                # update the lattice parameters (note: plots relies on this)
                if 'consensus_lattice' in self.Learners:
                    self.lattice = self.Learners['consensus_lattice'].d_weighted
                
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
        





