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
from datetime import datetime
import os

# from root folder
#import animation 
import swarm
import animation
import ctrl_tactic as tactic 

#%% initialize data
data = {}
current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")
data_directory = 'Data'
#file_path = os.path.join(data_directory, f"data_{formatted_date}.json")
file_path = os.path.join(data_directory, "data.json")
with open(file_path, 'w') as file:
    json.dump(data, file)

#%% Setup Simulation
# ------------------
#np.random.seed(0)
nAgents = 5
Ti      = 0       # initial time
Tf      = 300      # final time (later, add a condition to break out when desirable conditions are met)
Ts      = 0.02    # sample time
f       = 0       # parameter for future use
verbose = 1       # 1 = print progress reports, 0 = silent
nObs    = 3
#exclusion = []   # [LEGACY] initialization of what agents to exclude, default empty

#%% Instantiate the relevants objects
# ------------------------------------
Agents = swarm.Agents('pinning', nAgents)
Controller = tactic.Controller(Agents)
Targets = swarm.Targets(0, Agents.nVeh)
Trajectory = swarm.Trajectory(Targets)
Obstacles = swarm.Obstacles(Agents.tactic_type, nObs, Targets.targets)
History = swarm.History(Agents, Targets, Obstacles, Controller, Ts, Tf, Ti, f)

#%% Run Simulation
# ----------------------
t = Ti
i = 1

if verbose == 1:
    print('starting simulation with ',nAgents,' agents.')

while round(t,3) < Tf:
    
    # Evolve the target
    # -----------------    
    Targets.evolve(t)
    
    # Update the obstacles (if required)
    # ----------------------------------
    Obstacles.evolve(Targets.targets, Agents.state, Agents.rVeh)

    # Evolve the states
    # -----------------
    Agents.evolve(Controller,t,Ts)
     
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
    Trajectory.update(Agents, Targets, History, t, i)
                        
    #%% Compute the commads (next step)
    # --------------------------------  
    Controller.commands(Agents, Obstacles, Targets, Trajectory, History) 
      
#%% Produce animation of simulation
# --------------------------------- 
if verbose == 1:
    print('building animation.')

if Agents.dynamics_type == 'quadcopter':  
    import quadcopter_module.animation_quad as animation_quad
    ani = animation_quad.animateMe(Ts, History, Obstacles, Agents.tactic_type)
else:
    ani = animation.animateMe(Ts, History, Obstacles, Agents.tactic_type)
    
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
if nObs >  0:

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


#%% Save data
# -----------

if verbose == 1:
    print('saving data.')

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

data['Ti ']         = Ti      
data['Tf']          = Tf     
data['Ts']          = Ts           
data['Agents']      = Agents.__dict__
if Agents.dynamics_type == 'quadcopter':
    del data['Agents']['quadList']
    del data['Agents']['llctrlList']
    del data['Agents']['sDesList']
data['Targets']     = Targets.__dict__
data['Obstacles']   = Obstacles.__dict__
data['History']     = History.__dict__

data = convert_to_json_serializable(data)

with open(file_path, 'w') as file:
    json.dump(data, file)

with open(file_path, 'r') as file:
    data_json = json.load(file)

if verbose == 1:
    print('done.')

#%% LEGACY code (keep for reference)

    # EXPIRMENT # 1 - exclude random agents from the swarm every 10 seconds
    # select which agents to exclude (manually)
    # every 10 seconds
    #if t%10 < Ts:
        # randomly exclude
        #exclusion = [random.randint(0, nVeh-1)]
        #print(exclusion)
        
    # # EXPIRMENT # 2 - manually exclude agents from the swarm
    # # for simulation
    # if t < 20:
    #     exclusion = [2]
    # if t > 20 and t <= 45:
    #     exclusion = [2,7]
    # if t > 45 and t <= 65:
    #     exclusion = [1]
    # if t > 45 and t <= 75:
    #     exclusion = [1,6]
    # if t > 75 and t <= 90:
    #     exclusion = [3]
    # if t > 75 and t <= 100:
    #     exclusion = [3,7]
    # if t > 100 and t <= 115:
    #     exclusion = [9]
    # if t > 115 and t <= 120:
    #     exclusion = [5]
    # if t > 120 and t <= 150:
    #     exclusion = [4]
    # if t > 150 and t <= 185:
    #     exclusion = [4,8]
    # if t > 185 and t <= 200:
    #     exclusion = [6]
        
    # # Experiment #3 - remove 2 then remove 2
    # if t < 20:
    #     exclusion = []
    # if t > 20 and t <= 50:
    #     exclusion = [1]
    # if t > 50 and t <= 80:
    #     exclusion = [1,2]
    # if t > 80 and t <= 110:
    #     exclusion = [2]
    # if t > 110 and t <= 140:
    #     exclusion = []
    
# show_B_max  = 1 # highlight the max influencer? (0 = np, 1 = yes)
# if tactic_type == 'pinning' and show_B_max == 1:
#     # find the max influencer in the graph
#     G = graph_tools.build_graph(states_q, 5.1)
#     B = graph_tools.betweenness(G)
#     B_max = max(B, key=B.get)
#     pins_all[:,B_max,B_max] = 2
