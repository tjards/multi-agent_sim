#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:05:28 2025

@author: tjards
"""

import os
import json

# # load configs from JSON file
# def load_config(config_path):
  
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file(s) not found at {config_path}")
  
#     with open(config_path, 'r') as f:
#         config = json.load(f)

#     return config

# # get configs 
# def get_config(configs, path):

#     # split up the path (form: 'section.subsection.subsubsection = value')
#     keys = path.split('.')
#     value = configs

#     # move through the nested dicts to get the end value
#     for key in keys:
#         value = value[key]

#     return value





# *****************************
# OLD - to be removed later
# *****************************
config_path = os.path.join("config", "configs.json")

def initialize_configs():
    
    with open(config_path, 'w') as file:
        json.dump({}, file)


# generalized way to update configs 
def update_configs(section, entries=[]):
    # Load existing config if it exists, else start fresh
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            configs = json.load(file)
    else:
        configs = {}

    # Build the new section from the entries list
    section_dict = {}
    for key, value in entries:
        section_dict[key] = value

    # Update the configs with the new section
    configs[section] = section_dict

    # Save updated configs back to file
    with open(config_path, 'w') as file:
        json.dump(configs, file, indent=4, sort_keys=False)
    
# example:
#entries = [('Ti', 0), ('Tf', 100), ('Ts', 0.1), ('dimens', 3), ('verbose', 1), ('system', 'multi'), ('strategy', 'shepherd')]
#update_configs('simulation', entries)


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
