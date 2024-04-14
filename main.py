#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies including:
    
    - Reynolds rules of flocking ("boids")
    - Olfati-Saber flocking
    - Starling flocking
    - Dynamic Encirclement 
    - Heterogeneous Lattice formation using Pinning Control 
    - Autonomous Assembly of Closed Curves
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
 steps for modulization:
     1. create config files
     2. create a data file
     3. plot off data file
     4. move learner outside
     5. add a sensors module


"""

#%% Import stuff
# --------------

# official packages 
#from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#plt.style.use('classic')
plt.style.use('default')
#plt.style.available
#plt.style.use('Solarize_Light2')
import json
import h5py
#from datetime import datetime
import os

# from root folder
#import animation 
#import swarm
#import animation
import orchestrator  

from data import data_manager

#%% initialize data
#data = {}
#data_manager.initialize_data_JSON(data)

#%% Setup Simulation
# ------------------
#np.random.seed(0)
#nAgents = 5
Ti      = 0       # initial time
Tf      = 20      # final time (later, add a condition to break out when desirable conditions are met)
Ts      = 0.02    # sample time
f       = 0       # parameter for future use
verbose = 1       # 1 = print progress reports, 0 = silent

strategy = 'pinning'
# reynolds  = Reynolds flocking + Olfati-Saber obstacle
# saber     = Olfati-Saber flocking
# starling  = swarm like starlings 
# circle    = encirclement
# lemni     = dynamic lemniscates and other closed curves
# pinning   = pinning control
# shep      = shepherding

#nObs    = 0
#exclusion = []   # [LEGACY] initialization of what agents to exclude, default empty

# save to config file
config_sim = {'Ti': Ti, 'Tf': Tf, 'Ts': Ts, 'verbose': 1}
config_path = os.path.join("config", "config_sim.json")
with open(config_path, 'w') as configs_sim:
    json.dump(config_sim, configs_sim)

#%% instantiate the agents
# ------------------------
import agents.agents as agents
Agents = agents.Agents(strategy)
with open(os.path.join("config", "config_agents.json"), 'w') as configs_agents:
    json.dump(Agents.config_agents, configs_agents)

# instantiate the orchestrator
# ----------------------------
Controller = orchestrator.Controller(Agents.tactic_type, Agents.nAgents, Agents.state)

# instantiate the targets
# -----------------------
import targets.targets as targets
Targets = targets.Targets(Agents.nAgents)
with open(os.path.join("config", "config_targets.json"), 'w') as configs_targets:
    json.dump(Targets.config_targets, configs_targets)

# instantiate the planner
# -----------------------
import planner.trajectory as trajectory
Trajectory = trajectory.Trajectory(Targets.targets)

# instantiate the obstacles 
# -------------------------
import obstacles.obstacles as obstacles
Obstacles = obstacles.Obstacles(Agents.tactic_type, Targets.targets)
with open(os.path.join("config", "config_obstacles.json"), 'w') as configs_obstacles:
    json.dump(Obstacles.config_obstacles, configs_obstacles)

# new: move the learning agent out here
#import learner.RL_tools as RL

History = data_manager.History(Agents, Targets, Obstacles, Controller, Ts, Tf, Ti, f)

#%% Run Simulation
# ----------------------
t = Ti
i = 1

if verbose == 1:
    print('starting simulation with ',Agents.nAgents,' agents.')

while round(t,3) < Tf:
    
    # Evolve the target
    # -----------------    
    Targets.evolve(t)
    
    # Update the obstacles (if required)
    # ----------------------------------
    Obstacles.evolve(Targets.targets, Agents.state, Agents.rVeh)

    # Evolve the states
    # -----------------
    #Agents.evolve(Controller,t,Ts)
    Agents.evolve(Controller.cmd, Controller.pin_matrix,t,Ts)
    
    # Store results 
    # -------------
    History.update(Agents, Targets, Obstacles, Controller, t, f, i)
    
    # Increment 
    # ---------
    t += Ts
    i += 1
    
    # print progress
    if verbose == 1 and (round(t,2)).is_integer():
        print(round(t,1),' of ',Tf,' sec completed.')
    
    #%% Compute Trajectory
    # --------------------
    #Trajectory.update(Agents, Targets, History, t, i)
    
    my_kwargs = {}
    if Agents.tactic_type == 'lemni':
        my_kwargs['lemni_all'] = History.lemni_all
        my_kwargs['lemni'] = Agents
    
    Trajectory.update(Agents.tactic_type, Agents.state, Targets.targets, t, i, **my_kwargs)
                        
    #%% Compute the commads (next step)
    # --------------------------------  
    Controller.commands(Agents, Obstacles, Targets, Trajectory, History) 
      
#%% Save data
# -----------

if verbose == 1:
    print('saving data.')

#data_manager.save_data_JSON(data, Agents, Targets, Obstacles, History)
data_manager.save_data_HDF5(History)

if verbose == 1:
    print('done.')

# test
key, values = data_manager.load_data_HDF5('History', 't_all')


#%% Produce animation of simulation
# --------------------------------- 
if verbose == 1:
    print('building animation.')

import visualization.animation_sim as animation_sim
ani = animation_sim.animateMe(Ts, History, Agents.tactic_type)


# if Agents.dynamics_type == 'quadcopter':  
#     import quadcopter_module.animation_quad as animation_quad
#     ani = animation_quad.animateMe(Ts, History, Obstacles, Agents.tactic_type)
# else:
#     ani = animation.animateMe(Ts, History, Obstacles, Agents.tactic_type)
    
#%% Produce plots
# --------------

if verbose == 1:
    print('building plots.')

# separtion 
fig, ax = plt.subplots()
ax.plot(History.t_all[4::],History.metrics_order_all[4::,1],'-b')
ax.plot(History.t_all[4::],History.metrics_order_all[4::,5],':b')
ax.plot(History.t_all[4::],History.metrics_order_all[4::,6],':b')
ax.fill_between(History.t_all[4::], History.metrics_order_all[4::,5], History.metrics_order_all[4::,6], color = 'blue', alpha = 0.1)
#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', ylabel='Mean Distance (with Min/Max Bounds) [m]',
        title='Separation between Agents')
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
ax.grid()
plt.show()

# radii from target
radii = np.zeros([History.states_all.shape[2],History.states_all.shape[0]])
for i in range(0,History.states_all.shape[0]):
    for j in range(0,History.states_all.shape[2]):
        radii[j,i] = np.linalg.norm(History.states_all[i,:,j] - History.targets_all[i,:,j])
        
fig, ax = plt.subplots()
for j in range(0,History.states_all.shape[2]):
    ax.plot(History.t_all[4::],radii[j,4::].ravel(),'-b')
ax.set(xlabel='Time [s]', ylabel='Distance from Target for Each Agent [m]',
        title='Distance from Target')
#plt.axhline(y = 5, color = 'k', linestyle = '--')
plt.show()

#%% radii from obstacles
if Obstacles.nObs >  0:

    radii_o = np.zeros([History.states_all.shape[2],History.states_all.shape[0],History.obstacles_all.shape[2]])
    radii_o_means = np.zeros([History.states_all.shape[2],History.states_all.shape[0]])
    radii_o_means2 =  np.zeros([History.states_all.shape[0]])
    
    for i in range(0,History.states_all.shape[0]):              # the time samples
        for j in range(0,History.states_all.shape[2]):          # the agents
            for k in range(0,History.obstacles_all.shape[2]):   # the obstacles
                radii_o[j,i,k] = np.linalg.norm(History.states_all[i,0:3,j] - History.obstacles_all[i,0:3,k])
    
            radii_o_means[j,i] = np.mean(radii_o[j,i,:])
        radii_o_means2[i] = np.mean(radii_o_means[:,i])
    
            
    fig, ax = plt.subplots()
    start = int(0/0.02)
    
    for j in range(0,History.states_all.shape[2]):
        ax.plot(History.t_all[start::],radii_o_means2[start::].ravel(),'-g')
    ax.set(xlabel='Time [s]', ylabel='Mean Distance from Landmarks [m]',
            title='Learning Progress')
    #plt.axhline(y = 5, color = 'k', linestyle = '--')
    
    plt.show()

