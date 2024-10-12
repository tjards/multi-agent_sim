#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies including:
    
- Flocking (Reynolds, Olfati-Saber, Starling, and Cao)
- Dynamic Encirclement 
- Heterogeneous Pinning Control with Reinforcement Learning 
- Arbitrary Closed Curves (primarily lemniscates)
- Shepherding

The strategies requires no human invervention once the target is selected and all agents rely on local knowledge only. 
Each vehicle makes its own decisions about where to go based on its relative position to other vehicles.

The following agent dynamics are available:
    
    1. double integrator 
    2. quadrotor helicopter (quadcopter)

Created on Tue Dec 22 11:48:18 2020

Lasted updated on Fri Feb 02 16:23 2024

@author: tjards

dev notes:
    
    to do: 
        - standardize the planners (standard inputs, outputs, ...etc)
        - too many conditional statements throughout
        - 2D option
"""

#%% Import stuff
# --------------

# official packages 
# ------------------
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

# define data path
# ----------------
data_directory = 'data/data/'
#file_path = os.path.join(data_directory, f"data_{formatted_date}.json")
data_file_path = os.path.join(data_directory, "data.h5")

#%% Setup Simulation
# ------------------
np.random.seed(42)

Ti      = 0       # initial time
Tf      = 2     # final time (later, add a condition to break out when desirable conditions are met)
Ts      = 0.02    # sample time
f       = 0       # parameter for future use
verbose = 1       # 1 = print progress reports, 0 = silent
system   = 'swarm' 
strategy = 'cao'

    # reynolds  = Reynolds flocking + Olfati-Saber obstacle
    # saber     = Olfati-Saber flocking
    # starling  = swarm like starlings 
    # circle    = encirclement
    # lemni     = dynamic lemniscates and other closed curves
    # pinning   = pinning control (with RL)
    # shep      = shepherding
    # cao       = cao flocking

# save to config file
# -------------------
config_sim = {'Ti': Ti, 'Tf': Tf, 'Ts': Ts, 'verbose': 1, 'system': system, 'strategy': strategy}
with open(os.path.join("config", "config_sim.json"), 'w') as configs_sim:
    json.dump(config_sim, configs_sim)

#%% build the system
# ------------------
import orchestrator
Agents, Targets, Trajectory, Obstacles, Learners = orchestrator.build_system(system, strategy)
Controller = orchestrator.Controller(Agents.tactic_type, Agents.nAgents, Agents.state)
Controller.learning_agents(Agents.tactic_type, Learners)
Controller.Ts = Ts

# pull out constants
rVeh        = Agents.rVeh
tactic_type = Agents.tactic_type
dynamics    = Agents.config_agents['dynamics']

#%% initialize the data store
# ---------------------------
from data import data_manager
Database = data_manager.History(Agents, Targets, Obstacles, Controller, Trajectory, Ts, Tf, Ti, f)

#%% Run Simulation
# ----------------------
t = Ti
i = 1

if verbose == 1:
    print('starting simulation with ',Agents.nAgents,' agents.')

while round(t,3) < Tf:
    
    # initialize keyword arguments
    # ----------------------------
    my_kwargs = {}
    
    # Evolve the target
    # -----------------    
    Targets.evolve(t)
    
    # Update the obstacles (if required)
    # ----------------------------------
    Obstacles.evolve(Targets.targets, Agents.state, rVeh)

    # Evolve the states
    # -----------------
    Agents.evolve(Controller.cmd, Controller.pin_matrix, t, Ts)
    
    # Store results 
    # -------------
    Database.update(Agents, Targets, Obstacles, Controller, Trajectory, t, f, i)
    
    # Increment 
    # ---------
    t += Ts
    i += 1
    
  
    # print progress
    if verbose == 1 and (round(t,2)).is_integer():
        print(round(t,1),' of ',Tf,' sec completed.')
    
    # Compute Trajectory
    # --------------------

    # we'll need the record of lemni parameters  
    if tactic_type == 'lemni':
        # only need to pass last timestep, so reduce this later 
        my_kwargs['lemni_all'] = Database.lemni_all
        
    Trajectory.update(tactic_type, Agents.state, Targets.targets, t, i, **my_kwargs)
                        
    # Compute the commads (next step)
    # --------------------------------  
    if tactic_type == 'pinning' and dynamics == 'quadcopter':
        my_kwargs['quads_headings'] = Agents.quads_headings
                
    Controller.commands(Agents.state, tactic_type, Agents.centroid, Targets.targets, Obstacles.obstacles_plus, Obstacles.walls, Trajectory.trajectory, dynamics, **my_kwargs) 
    
#%% Save data
# -----------
if verbose == 1:
    print('saving data.')

data_manager.save_data_HDF5(Database, data_file_path)

if verbose == 1:
    print('done.')
    
    
#%% Produce plots
# --------------
import visualization.plot_sim as plot_sim

if verbose == 1:
    print('building plots.')

plot_sim.plotMe(data_file_path)

#%% Produce animation of simulation
# --------------------------------- 
import visualization.animation_sim as animation_sim

if verbose == 1:
    print('building animation.')

# pull out the relevant configs
with open(os.path.join("config", "config_sim.json"), 'r') as configs_sim:
    config_sim = json.load(configs_sim)
    config_Ts = config_sim['Ts']
with open(os.path.join("config", "config_agents.json"), 'r') as configs_agents:
    config_agents = json.load(configs_agents)
    config_tactic_type = config_agents['tactic_type']

ani = animation_sim.animateMe(data_file_path, config_Ts, config_tactic_type)






