#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies.

The strategies requires no human invervention once the target is selected and all agents rely on local knowledge only. 
Each vehicle makes its own decisions about where to go based on its relative position to other vehicles.

The following agent dynamics are available:
    
    1. double integrator 
    2. quadrotor helicopter (quadcopter)

# strategy summaries:
    # reynolds  = Reynolds flocking + Olfati-Saber obstacle
    # saber     = Olfati-Saber flocking
    # starling  = swarm like starlings 
    # circle    = encirclement
    # lemni     = dynamic lemniscates and other closed curves
    # pinning   = pinning control (with RL)
    # shep      = shepherding
    # cao       = cao flocking

Created on Tue Dec 22 11:48:18 2020

@author: tjards

See devnotes.md for updates. 

"""

#%% Import stuff
# --------------

# official packages 
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#plt.style.use('classic')
plt.style.use('default')
#plt.style.available
#plt.style.use('Solarize_Light2')
import json
import h5py
import os
from datetime import datetime

# custom packages
import config.config as cfg

#%% Setup Simulation
# ------------------

# config path
config_directory = 'config/'
config_path = os.path.join(config_directory, 'config.json')

# create an immutable config object
config = cfg.Config(config_path)

# define data paths
data_directory = config.data_dir
data_file = config.data_file
data_file_path = os.path.join(data_directory, data_file)

# reproducibility
np.random.seed(config.random_seed)

#%% build the system
# ------------------
import orchestrator
import planner.trajectory

Agents, Targets, Trajectory, Obstacles, Learners = orchestrator.build_system(config)
Controller = orchestrator.Controller(config, Agents.state)
Controller.learning_agents(config.strategy, Learners)
Trajectory.load_planners(Controller.planners)

#%% initialize the data store
# ---------------------------
from data import data_manager
Database = data_manager.History(Agents, Targets, Obstacles, Controller, Trajectory, 
                                config.Ts, config.Tf, config.Ti, config.f)

#%% Run Simulation
# ----------------------
t = config.Ti
i = 1

if config.verbose == 1:
    print('starting simulation with ',config.nAgents,' agents.')

while round(t,3) < config.Tf:
    
    # initialize keyword arguments
    # ----------------------------
    my_kwargs = {}
    
    # Evolve the target
    # -----------------    
    Targets.evolve(t)
    
    # Update the obstacles (if required)
    # ----------------------------------
    Obstacles.evolve(Targets.targets, Agents.state, config.nAgents)

    # Evolve the states
    # -----------------
    Agents.evolve(Controller.cmd, Controller.pin_matrix, t, config.Ts)
    
    # Store results 
    # -------------
    Database.update(Agents, Targets, Obstacles, Controller, Trajectory, t, config.f, i)
    
    # Increment 
    # ---------
    t += config.Ts
    i += 1
    
    # print progress
    if config.verbose == 1 and (round(t,2)).is_integer():
        print(round(t,1),' of ',config.Tf,' sec completed.')
    
    # Compute Trajectory
    # --------------------

    my_kwargs = planner.trajectory.update_trajectory_args(Database, Agents, Trajectory, Controller, config.strategy, my_kwargs)     
    Trajectory.update(config.strategy, Agents.state, Targets.targets, t, i, **my_kwargs)
                        
    # Compute the commads (next step)
    # --------------------------------  
    if config.strategy == 'pinning' and config.dynamics == 'quadcopter':
        my_kwargs['quads_headings'] = Agents.quads_headings
                       
    Controller.commands(Agents.state, config.strategy, Agents.centroid, Targets.targets, Obstacles.obstacles_plus, Obstacles.walls, Trajectory.trajectory, config.dynamics, **my_kwargs) 
    
#%% Save data
# -----------
if config.verbose == 1:
    print('saving data.')

data_manager.save_data_HDF5(Database, data_file_path)

if config.verbose == 1:
    print('done.')
    
    
#%% Produce plots
# --------------
import visualization.plot_sim as plot_sim

if config.verbose == 1:
    print('building plots.')

plot_sim.plotMe(data_file_path)

#%% Produce animation of simulation
# --------------------------------- 
import visualization.animation_sim as animation_sim

if config.verbose == 1:
    print('building animation.')
    
with open(config_path, 'r') as configs_sim:
    config_sim = json.load(configs_sim)
    config_Ts = config_sim['simulation']['Ts']
    config_dimens = config_sim['simulation']['dimens']
    config_tactic_type = config_sim['simulation']['strategy']

ani = animation_sim.animateMe(data_file_path, config_Ts, config_dimens, config_tactic_type)

#%% experimental save
if config.experimental_save:
    from experiments.experiment_manager import save_experiment
    save_experiment()


#%% Main
# ------

def main():
    sim_time = datetime.now().strftime("%y%m%d-%H%M%S")
    print(f"m-a_s run at {sim_time}")

if __name__ == "__main__":
    main()



