#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module computes the commands for various swarming strategies 

Dev notes: 
    - we have created this separate module to permit mixing and/or sharing between tactic types
    - bring the graphing and pinning stuff out as an outer loop common to all planning/control 

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
import config.configs_tools as configs_tools
config_path=configs_tools.config_path

# read the configs
with open(config_path, 'r') as tactic_tests:
    tactic_test = json.load(tactic_tests)
    tactic_type = tactic_test['simulation']['strategy']

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
    pinning_tools.update_pinning_configs()
    with open(config_path, 'r') as planner_pinning_tests:
        planner_configs = json.load(planner_pinning_tests)['pinning']['hetero_lattice']
elif tactic_type == 'cao':
    from planner.techniques import cao_tools


#%% Hyperparameters 
# -----------------

pin_update_rate = 10   # number of timesteps after which we update the pins
pin_selection_method = 'degree_leafs'
    # gramian   = [future] // based on controllability gramian
    # degree    = based on degree centrality  
    # between   = [future] // based on betweenness centrality (buggy at nAgents < 3)
    # degree_leafs = degree and also leaves (only one connection)
    # nopins      = no pins
    # allpins     = all are pins 
criteria_table = {'radius': True, 'aperature': False} # for graph construction 
sensor_aperature    = 140

#twoD = True

configs_tools.update_configs('orchestrator', [
    ('pin_update_rate', pin_update_rate),
    ('pin_selection_method', pin_selection_method),
    ('criteria_table', criteria_table),
    ('sensor_aperature', sensor_aperature)
] )

#%% Build the system
# ------------------
def build_system(system, strategy, dimens, Ts):
    
    if system == 'swarm':
    
        # instantiate the agents
        # ------------------------
        import agents.agents as agents
        Agents = agents.Agents(strategy, dimens)
        configs_tools.update_orch_configs(config_path, agent_obj=Agents)

        
        # instantiate the targets
        # -----------------------
        import targets.targets as targets
        Targets = targets.Targets(Agents.nAgents, dimens)    
        configs_tools.update_orch_configs(config_path, target_obj=Targets)
    
        # instantiate the planner
        # -----------------------
        import planner.trajectory as trajectory
        Trajectory = trajectory.Trajectory(Agents.tactic_type, Targets.targets, Agents.nAgents)
        
    
        # instantiate the obstacles 
        # -------------------------
        import obstacles.obstacles as obstacles
        Obstacles = obstacles.Obstacles(Agents.tactic_type, Targets.targets, dimens)     
        configs_tools.update_orch_configs(config_path, obstacle_obj=Obstacles)
        
        
        # instatiate any learning
        # -----------------------
        Learners = {}
        
        # pinning control case
        if tactic_type == 'pinning':

            pinning_tools.update_pinning_configs()
            with open(os.path.join("config", "configs.json"), 'r') as planner_pinning_tests:
                configs = json.load(planner_pinning_tests)
                planner_configs = configs['pinning']
                
                # need one learner to achieve consensus on lattice size
                lattice_consensus = planner_configs['hetero_lattice']
                if lattice_consensus == 1:
                    import learner.consensus_lattice as consensus_lattice

                    Consensuser = consensus_lattice.Consensuser(Agents.nAgents, 1, planner_configs['d_min'], planner_configs['d'], planner_configs['r_max'])
                    
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
                
                # see if I have to integrate different potential functions
                potential_function_learner = planner_configs['hetero_gradient']
                if potential_function_learner == 1:
                    
                    # import the gradient estimator
                    import learner.gradient_estimator as gradient_estimator
                    
                    Gradient_agent = gradient_estimator.GradientEstimator(Agents.nAgents, Agents.dimens, Ts)
                    
                    # load
                    Learners['estimator_gradients'] = Gradient_agent
      
    
        configs_tools.update_orch_configs(config_path,learner_objs=Learners)
                
    return Agents, Targets, Trajectory, Obstacles, Learners

#%% Master controller
# -------------------
class Controller:
    
    def __init__(self,tactic_type, nAgents, state, dimens):
                
        # commands
        # --------
        self.dimens  = dimens
        self.nAgents = nAgents
        self.cmd = np.zeros((3,nAgents))
        self.cmd[0] = 0.001*np.random.rand(1,nAgents)-0.5      # command (x)
        self.cmd[1] = 0.001*np.random.rand(1,nAgents)-0.5      # command (y)
        self.cmd[2] = 0.001*np.random.rand(1,nAgents)-0.5      # command (z)
        if dimens == 2:
            self.cmd[2] = 0*self.cmd[2]

        # graph
        self.Graphs = graphical.Swarmgraph(state, criteria_table) # initialize 
        self.Graphs_connectivity = graphical.Swarmgraph(state, criteria_table)
        
        # general purpose counter (nominally, pin reset)
        self.counter = 0                
        
        # [legacy] general purpose parameters variable (retire this)
        self.params = np.zeros((4,nAgents))             # store dynamic parameters
        
        # lattice parameters (not always used, move into pinning) * redudany
        self.lattice = np.zeros((nAgents,nAgents))      # stores lattice parameters
        
        # pins and components (not always used)
        self.pin_matrix = np.zeros((nAgents,nAgents))
                
        if tactic_type == 'pinning':
    
            self.pin_matrix = np.ones((nAgents,nAgents))# make all pins
            d_init = pinning_tools.return_lattice_param()
            self.d_init= d_init
            i = 0
            while (i < nAgents):
                self.lattice[i,:] = d_init
                i+=1
            #self.Graphs_connectivity = graphical.Swarmgraph(state, criteria_table)
        
        # sheparding has its own class (differentiating shepherd and herd)
        if tactic_type == 'shep':
            self.shepherdClass = shep.Shepherding(state) 
        
        if tactic_type == 'saber':
            self.lattice = saber_tools.return_ranges()*np.ones((state.shape[1],state.shape[1])) 
        
        # cao has it's own class and a separate graph for connected (in addition to in range)    
        if tactic_type == 'cao':
            self.caoClass = cao_tools.Flock(state[0:3,:],state[3:6,:])
            self.lattice = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1])) 
            #self.Graphs_connectivity = graphical.Swarmgraph(state, criteria_table)
            self.pin_matrix = np.ones((nAgents,nAgents))
            
            

   
    # integrate learninging agents
    # ----------------------------
    def learning_agents(self, tactic_type, Learners):
        
        self.Learners = {}
        
        # note: may be better to just run through this list, rather than explicitly loading each
        
        if tactic_type == 'pinning' and 'consensus_lattice' in Learners:
            
             self.Learners['consensus_lattice'] = Learners['consensus_lattice']
             print('Consensus-based lattice negotiation enabled')
             
             if 'learning_lattice' in Learners:
                 
                 self.Learners['learning_lattice'] = Learners['learning_lattice']
                 print('Q-Learning based lattice optimization enabled')
    
    
        if tactic_type == 'pinning' and 'estimator_gradients' in Learners:
            
            self.Learners['estimator_gradients'] = Learners['estimator_gradients']
            print('Estimating neighbouring gradients enabled')
    
        if len(self.Learners) == 0:
            
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
        
        # ************* #
        # GRAPH UPDATES #
        # ************* #
        
        # update connectivity
        self.counter += 1                    # increment the counter 
        if self.counter == pin_update_rate:  # only update the pins at Ts/(tunable parameter)
            self.counter = 0                 # reset counter
            
            # update connectivity parameters 
            if tactic_type == 'saber':
                r_matrix = saber_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
            elif tactic_type == 'circle':
                r_matrix_1, _, _, _ = encircle_tools.get_params()
                r_matrix_2  = encircle_tools.compute_desired_sep(r_matrix_1, self.nAgents)
                r_matrix = r_matrix_2*np.ones((state.shape[1],state.shape[1]))
            elif tactic_type == 'cao':
                r_matrix = cao_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
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
                kwargs_cmd['aperature'] = sensor_aperature
                #r_matrix = kwargs_cmd['d_weighted']  # if we want the graph based on lattice parameters
                r_matrix = pinning_tools.return_ranges()*np.ones((state.shape[1],state.shape[1]))
                self.Graphs_connectivity.update_A(state[0:3,:], self.lattice, **kwargs_cmd)
            else:
                r_matrix = 0*np.ones((state.shape[1],state.shape[1]))

            # update the graph
            self.Graphs.update_A(state[0:3,:], r_matrix, **kwargs_cmd)
            #self.Graphs.update_A(state[0:3,:], self.lattice, **kwargs_cmd)

            self.Graphs.find_connected_components()
            self.Graphs.update_pins(state[0:3,:], r_matrix, pin_selection_method, **kwargs_cmd)
            self.pin_matrix = self.Graphs.pin_matrix

        # *************** #
        # COMMAND UPDATES #
        # *************** #

        # reynolds requires a matrix of distances between agents
        if tactic_type == 'reynolds':
            distances = reynolds_tools.order(state[0:3,:])
            
        # for each vehicle/node/agent in the network
        # ------------------------------------
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
                
                if 'estimator_gradients' in self.Learners:
                    kwargs_pinning['estimator_gradients'] = self.Learners['estimator_gradients']
                    # reset the sum for pins
                    self.Learners['estimator_gradients'].C_sum[0:self.dimens, 0:self.nAgents] = np.zeros((self.dimens, self.nAgents)) 
                    kwargs_pinning['pin_matrix'] = self.pin_matrix
                    
                    
                
                # info about graph
                kwargs_pinning['directional_graph']         = self.Graphs.directional_graph
                kwargs_pinning['A']                         = self.Graphs.A
                kwargs_pinning['D']                         = self.Graphs.D
                kwargs_pinning['local_k_connectivity']      = self.Graphs.local_k_connectivity
                
                            
                # compute command
                cmd_i[:,k_node] = pinning_tools.compute_cmd(centroid, state[0:3,:], state[3:6,:], obstacles_plus, walls,  targets[0:3,:], targets[3:6,:], k_node, **kwargs_pinning)
                
                # update the lattice parameters (note: plots relies on this)
                if 'consensus_lattice' in self.Learners:
                    self.lattice = self.Learners['consensus_lattice'].d_weighted
                    
                if 'estimator_gradients' in self.Learners:
                    # reset the by_pin sums
                    self.Learners['estimator_gradients'].C_sum_bypin[0:self.dimens, 0:self.nAgents] = np.zeros((self.dimens, self.nAgents)) 
                    # figure out who the pins are
                    pins_list = np.where(np.any(self.pin_matrix > 0, axis=1))[0]
                    # cycle through components
                    component_index = 0
                    for each_component in self.Graphs.components:
                        for each_node in each_component:
                            # add up all the gradients in this component (and hand to the pin)
                            self.Learners['estimator_gradients'].C_sum_bypin[0:self.dimens, pins_list[component_index]] += self.Learners['estimator_gradients'].C_sum[0:self.dimens, each_node]
                        component_index += 1
                        #print(self.Learners['estimator_gradients'].C_sum_bypin)
                    #print(self.Learners['estimator_gradients'].C_sum_bypin[:, :])

                        
            # Shepherding
            # ------------
            if tactic_type == 'shep':
                
                # compute command, pin_matrix records shepherds = 1, herd = 0
                # ----------------------------------------------------------

                # compute the commands
                self.shepherdClass.compute_cmd(targets, k_node)
        
                # pull out results
                cmd_i[:,k_node]                 = self.shepherdClass.cmd
                self.pin_matrix[k_node, k_node] = self.shepherdClass.index[self.shepherdClass.i] # note: by moving pin selection to outer loop, i may have messed this up
  
            # Cao flocking 
            # ------------
            if tactic_type == 'cao':
                             
                kwargs_cao          = {}
                kwargs_cao['A']     = self.Graphs.A
                kwargs_cao['A_connectivity']     = self.Graphs_connectivity.A
                kwargs_cao['pin_matrix']  = self.pin_matrix
                kwargs_cao['Ts'] = self.Ts
                #cmd_i[:,k_node] = cao_tools.compute_cmd(targets[0:3,:],state[0:3,:], state[3:6,:], k_node, **kwargs_cao)
                cmd_i[:,k_node] = self.caoClass.compute_cmd(targets[0:3,:],state[0:3,:], state[3:6,:], k_node, **kwargs_cao)
                #self.lattice = cao_tools.return_desired_sep()*np.ones((state.shape[1],state.shape[1]))
                #print(self.caoClass.status)
                #print(self.caoClass.layer)

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
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'shep':
                cmd_i[:,k_node] = cmd_i[:,k_node]
            elif tactic_type == 'cao':
                cmd_i[:,k_node] = cmd_i[:,k_node]
                
           
            if self.dimens == 2 and self.cmd[2,:].any() != 0:
                
                print('warning, 3D cmds in 2D at node ', k_node)
                print(self.cmd[2,k_node])    
                
        # update the commands
        self.cmd = copy.deepcopy(cmd_i) 
        

        





