#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:05:28 2025

@author: tjards
"""

import os
import json

def update_orch_configs(config_path, agent_obj=None, target_obj=None, obstacle_obj=None, learner_objs=None):
    """
    Update the master configs.json file with provided objects.
    
    Parameters:
        config_path (str): Path to the configs.json file.
        agent_obj (object, optional): Agents object containing config_agents.
        target_obj (object, optional): Targets object containing config_targets.
        obstacle_obj (object, optional): Obstacles object containing config_obstacles.
        learner_objs (dict, optional): Dictionary of learner_name -> learner_object.
    """

    # Load existing configs if they exist
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)
    else:
        configs = {}

    # Update sections if provided
    if agent_obj is not None:
        configs['agents'] = agent_obj.config_agents

    if target_obj is not None:
        configs['targets'] = target_obj.config_targets

    if obstacle_obj is not None:
        configs['obstacles'] = obstacle_obj.config_obstacles

    if learner_objs:
        configs['learners'] = {}
        for learner_name, learner_obj in learner_objs.items():
            if hasattr(learner_obj, 'config'):
                configs['learners'][learner_name] = learner_obj.config
            else:
                configs['learners'][learner_name] = f"Config not available for {learner_name}"

    # Save updated configs
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4, sort_keys=False)
