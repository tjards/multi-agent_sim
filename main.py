#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies including:
    
    - Reynolds rules of flocking ("boids")
    - Olfati-Saber flocking
    - Starling flocking
    - Dynamic Encirclement 
    - Pinning Control
    - Autonomous Assembly of Closed Curves
    - Shepherding

The strategies requires no human invervention once the target is selected and all agents rely on local knowledge only. 
Each vehicle makes its own decisions about where to go based on its relative position to other vehicles.

Created on Tue Dec 22 11:48:18 2020

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
Ti = 0       # initial time
Tf = 30      # final time (later, add a condition to break out when desirable conditions are met)
Ts = 0.02    # sample time
f  = 0       # parameter for future use
#exclusion = []     # [LEGACY] initialization of what agents to exclude, default empty

#%% Instantiate the relevants objects
# ------------------------------------
Agents = swarm.Agents('pinning', 10)
Controller = tactic.Controller(Agents)
Targets = swarm.Targets(0, Agents.nVeh)
Trajectory = swarm.Trajectory(Targets)
Obstacles = swarm.Obstacles(Agents.tactic_type, 0, Targets.targets)
History = swarm.History(Agents, Targets, Obstacles, Controller, Ts, Tf, Ti, f)

#%% Run Simulation
# ----------------------
t = Ti
i = 1

while round(t,3) < Tf:
    
    # Evolve the target
    # -----------------    
    Targets.evolve(t)
    
    # Update the obstacles (if required)
    # ----------------------------------
    Obstacles.evolve(Targets.targets, Agents.state, Agents.rVeh)

    # Evolve the states
    # -----------------
    Agents.evolve(Controller,Ts)
     
    # Store results 
    # -------------
    History.update(Agents, Targets, Obstacles, Controller, t, f, i)
    
    # Increment 
    # ---------
    t += Ts
    i += 1
    
    #%% Compute Trajectory
    # --------------------
    Trajectory.update(Agents, Targets, History, t, i)
                        
    #%% Compute the commads (next step)
    # --------------------------------  
    Controller.commands(Agents, Obstacles, Targets, Trajectory, History) 
      
#%% Produce animation of simulation
# ---------------------------------       
ani = animation.animateMe(Ts, History, Obstacles, Agents.tactic_type)

#%% Produce plots
# --------------

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

#%% Save data
# -----------

data['Ti ']         = Ti      
data['Tf']          = Tf     
data['Ts']          = Ts     
data['Agents']      = Agents.__dict__
data['Targets']     = Targets.__dict__
data['Obstacles']   = Obstacles.__dict__
data['History']     = History.__dict__

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

data = convert_to_json_serializable(data)

with open(file_path, 'w') as file:
    json.dump(data, file)

with open(file_path, 'r') as file:
    data_json = json.load(file)


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
