#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:46:53 2025

@author: tjards

Coordinates all learning modules. Like orchestrator does for control.


"""

# import stuff
# -----------
import os
import json
import config.configs_tools as configs_tools
import numpy as np
config_path=configs_tools.config_path


# load configs
# ------------

def initialize(Agents, tactic_type, learning_ctrl, Ts):
    
    
    Learners = {}
    
    # if using CALA to tune controller parameters
    if learning_ctrl == 'CALA':
        from learner import CALA_control
        CALA = CALA_control.CALA(Agents.nAgents) # just one param per agent now (expand latter)
        
        #Load
        Learners['CALA_ctrl'] = CALA
     
        
    if tactic_type == 'lemni':
        with open(config_path, 'r') as planner_lemni_tests:
            configs = json.load(planner_lemni_tests)
        if configs['lemni']['learning'] == 'CALA':
            import learner.CALA_control as lemni_CALA
            lemni_CALA = lemni_CALA.CALA(Agents.nAgents)
           
            # LOAD
            Learners['lemni_CALA'] = lemni_CALA
           
        
    # pinning control case
    if tactic_type == 'pinning':
        
        from planner.techniques import pinning_RL_tools as pinning_tools
        
        pinning_tools.update_pinning_configs()
        #with open(os.path.join("config", "configs.json"), 'r') as planner_pinning_tests:
        with open(config_path, 'r') as planner_pinning_tests:
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
            
            # see if I have to integrate different potential functions (legacy - remove)
            potential_function_learner = planner_configs['hetero_gradient']
            if potential_function_learner == 1:
                
                # import the gradient estimator
                import learner.gradient_estimator as gradient_estimator
                
                Gradient_agent = gradient_estimator.GradientEstimator(Agents.nAgents, Agents.dimens, Ts)
                
                # load
                Learners['estimator_gradients'] = Gradient_agent
      
    
    configs_tools.update_orch_configs(config_path,learner_objs=Learners)
        
    return Learners
        
        
def pinning_update_args(Controller, kwargs_pinning):
    
        
    # learning stuff (if applicable)
    if 'consensus_lattice' in Controller.Learners:
        kwargs_pinning['consensus_lattice'] = Controller.Learners['consensus_lattice']
        if 'learning_lattice' in Controller.Learners:
            kwargs_pinning['learning_lattice'] = Controller.Learners['learning_lattice']
    
    if 'estimator_gradients' in Controller.Learners:
        kwargs_pinning['estimator_gradients'] = Controller.Learners['estimator_gradients']
        # reset the sum for pins
        Controller.Learners['estimator_gradients'].C_sum[0:Controller.dimens, 0:Controller.nAgents] = np.zeros((Controller.dimens, Controller.nAgents)) 
        kwargs_pinning['pin_matrix'] = Controller.pin_matrix
        
    return kwargs_pinning


def pinning_update_lattice(Controller):
    
    # update the lattice parameters (note: plots relies on this)
    if 'consensus_lattice' in Controller.Learners:
        Controller.lattice = Controller.Learners['consensus_lattice'].d_weighted
        
    if 'estimator_gradients' in Controller.Learners:
        # reset the by_pin sums
        Controller.Learners['estimator_gradients'].C_sum_bypin[0:Controller.dimens, 0:Controller.nAgents] = np.zeros((Controller.dimens, Controller.nAgents)) 
        # figure out who the pins are
        pins_list = np.where(np.any(Controller.pin_matrix > 0, axis=1))[0]
        # cycle through components
        component_index = 0
        for each_component in Controller.Graphs.components:
            for each_node in each_component:
                # add up all the gradients in this component (and hand to the pin)
                Controller.Learners['estimator_gradients'].C_sum_bypin[0:Controller.dimens, pins_list[component_index]] += Controller.Learners['estimator_gradients'].C_sum[0:Controller.dimens, each_node]
            component_index += 1
            #print(self.Learners['estimator_gradients'].C_sum_bypin)
        #print(self.Learners['estimator_gradients'].C_sum_bypin[:, :])

        