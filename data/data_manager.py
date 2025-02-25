#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:54:11 2024

@author: tjards
"""

#%% import stuff
# ------------
import json
import h5py
import numpy as np
from datetime import datetime
import os

current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")
#data_directory = 'data'
#file_path = os.path.join(data_directory, f"data_{formatted_date}.json")
#file_path = os.path.join(data_directory, "data.json")
#file_path = os.path.join(data_directory, "data.h5")

#%% helpers
# -------

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# def initialize_data_JSON(data):
    
#     with open(file_path, 'w') as file:
#         json.dump(data, file)

# def save_data_JSON(data, Agents, Targets, Obstacles, History):

#     #data['Agents']      = Agents.__dict__
#     #if Agents.dynamics_type == 'quadcopter':
#     #    del data['Agents']['quadList']
#     #    del data['Agents']['llctrlList']
#     #    del data['Agents']['sDesList']
#     #data['Targets']     = Targets.__dict__
#     #data['Obstacles']   = Obstacles.__dict__
    
#     data['History']     = History.__dict__
    
#     data = convert_to_json_serializable(data)
    
#     with open(file_path, 'w') as file:
#         json.dump(data, file)
    
#     with open(file_path, 'r') as file:
#         data_json = json.load(file)
    
def save_data_HDF5(data, file_path):
    
    history_data = data.__dict__
    
    with h5py.File(file_path, 'w') as file:
        
        history_group = file.create_group('History')
        
        # Save data under the History group
        for key, value in history_data.items():
            
            history_group.create_dataset(key, data=value)

def load_data_HDF5(group, key, file_path_h5):
    
    # get current working directory
    #cwd = os.getcwd()

    # specify relative path to the HDF5 file
    #file_path_h5 = os.path.join(cwd,'data','data.h5')

    # open the HDF5 file
    with h5py.File(file_path_h5, 'r') as file:
        
        # check if group exists in the file
        if group in file:
            
            # access group
            history_group = file[group]
            
            # check if this key exists in group
            if key in history_group:
                
                # pull the data for that key
                dataset = history_group[key]
                # pull the values 
                values = dataset[:]
                
                # Inspect the dataset
                #print("Dataset shape:", dataset.shape)
                #print("Dataset dtype:", dataset.dtype)
                #print("Dataset value(s):", dataset[:])  # Print the values if it's a small dataset
                
            else:
                print("Key not found within group.")
        else:
            print("Group not found in the HDF5 file.")
            
    # return the key and values
    return key, values

#%% intermediate object to store data (eventually, call this interatively)
# -----------------------------------
class History:
    
    # note: break out the Metrics stuff int another class 
    
    def __init__(self, Agents, Targets, Obstacles, Controller, Trajectory, Ts, Tf, Ti, f):
        
        nSteps = int(Tf/Ts+1)
        
        # initialize a bunch of storage 
        self.t_all               = np.zeros(nSteps)
        self.states_all          = np.zeros([nSteps, len(Agents.state), Agents.nAgents])
        self.cmds_all            = np.zeros([nSteps, len(Controller.cmd), Agents.nAgents])
        self.targets_all         = np.zeros([nSteps, len(Targets.targets), Agents.nAgents])
        self.obstacles_all       = np.zeros([nSteps, len(Obstacles.obstacles), Obstacles.nObs])
        self.centroid_all        = np.zeros([nSteps, len(Agents.centroid), 1])
        self.f_all               = np.ones(nSteps)
        self.lemni_all           = np.zeros([nSteps, Agents.nAgents])
        # metrics_order_all   = np.zeros((nSteps,7))
        # metrics_order       = np.zeros((1,7))
        nMetrics            = 12 # there are 11 positions being used.    
        self.metrics_order_all   = np.zeros((nSteps,nMetrics))
        self.metrics_order       = np.zeros((1,nMetrics))
        self.pins_all            = np.zeros([nSteps, Agents.nAgents, Agents.nAgents]) 
        # note: for pinning control, pins denote pins as a 1
        # also used in lemni to denote membership in swarm as 0
        
        self.connectivity = np.zeros([nSteps, Agents.nAgents, Agents.nAgents])
        self.local_k_connectivity = [0]*nSteps
        
        self.swarm_prox = 0
        
        # if there are quadcopters
        dynamics = Agents.dynamics_type
        if dynamics == 'quadcopter':
            # initialize a state to store all the quadcopter states 
            self.quads_states_all   = np.zeros([nSteps, 21, Agents.nAgents])
            self.quad_w_cmd_all     = np.zeros([nSteps,4, Agents.nAgents])
            self.quads_sDes_all     = np.zeros([nSteps, 21, Agents.nAgents])

        # store the initial conditions
        self.t_all[0]                = Ti
        self.states_all[0,:,:]       = Agents.state
        self.cmds_all[0,:,:]         = Controller.cmd
        self.targets_all[0,:,:]      = Targets.targets
        self.obstacles_all[0,:,:]    = Obstacles.obstacles
        self.centroid_all[0,:,:]     = Agents.centroid
        self.f_all[0]                = f
        self.metrics_order_all[0,:]  = self.metrics_order
        #self.lemni                   = np.zeros([1, Agents.nAgents])
        self.lemni_all[0,:]          = Trajectory.lemni
        self.pins_all[0,:,:]         = Controller.pin_matrix  
        self.connectivity[0,:,:]     = Controller.Graphs.A
        self.local_k_connectivity[0] =list(Controller.Graphs.local_k_connectivity.values())
        
        # stores the desired lattice sizes 
        self.lattices            = np.zeros((nSteps,Agents.nAgents,Agents.nAgents))
        
        # store the walls
        self.walls_plots     = Obstacles.walls_plots
        
        # I want to start violations of censensus lattice constraints
        if 'consensus_lattice' in Controller.Learners:
            #self.lattice_mins = np.zeros([nSteps, Agents.nAgents, Agents.nAgents]) 
            #self.lattice_maxs = np.zeros([nSteps, Agents.nAgents, Agents.nAgents])
            self.lattice_violations = np.zeros([nSteps, Agents.nAgents, Agents.nAgents]) 
            #self.lattice_mins[0,:,:] = Controller.Learners['consensus_lattice'].d_min
            #self.lattice_maxs[0,:,:] = Controller.Learners['consensus_lattice'].d_max
        else:
            self.lattice_violations = np.zeros([nSteps, Agents.nAgents, Agents.nAgents]) 
        
        
    def sigma_norm(self, z): 
        
        eps = 0.5
        norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
        return norm_sig

    def update(self, Agents, Targets, Obstacles, Controller, Trajectory, t, f, i):
        
        # core 
        self.t_all[i]                = t
        self.states_all[i,:,:]       = Agents.state
        self.cmds_all[i,:,:]         = Controller.cmd
        self.targets_all[i,:,:]      = Targets.targets
        self.obstacles_all[i,:,:]    = Obstacles.obstacles
        self.centroid_all[i,:,:]     = Agents.centroid
        self.f_all[i]                = f
        self.lemni_all[i,:]          = Trajectory.lemni
        self.pins_all[i,:,:]         = Controller.pin_matrix 
        self.connectivity[i,:,:]     = Controller.Graphs.A
        self.local_k_connectivity[i] =list(Controller.Graphs.local_k_connectivity.values())
        
        # metrics
        self.metrics_order[0,0]      = Agents.order(Agents.state[3:6,:])
        #self.metrics_order[0,1:7]    = Agents.separation(Agents.state[0:3,:],Targets.targets[0:3,:],Obstacles.obstacles)
        self.metrics_order[0,1:7]    = Agents.separation(Agents.state[0:3,:],Targets.targets[0:3,:],Obstacles.obstacles, Controller.Graphs_connectivity.A)
        self.metrics_order[0,7:9]    = Agents.energy(Controller.cmd)
        self.metrics_order[0,9:12]   = Agents.spacing(Agents.state[0:3,:], Controller.lattice.min()) # legacy: retire this
        self.metrics_order_all[i,:]  = self.metrics_order
        self.swarm_prox              = self.sigma_norm(Agents.centroid.ravel()-Targets.targets[0:3,0])
        
        self.lattices[i,:,:]         = Controller.lattice # lattice parameters (diag is d_init)
        
        if 'consensus_lattice' in Controller.Learners:
            #self.lattice_mins[i,:,:] = Controller.Learners['consensus_lattice'].d_min
            #self.lattice_maxs[i,:,:] = Controller.Learners['consensus_lattice'].d_max
            self.lattice_violations[i,:,:] = Controller.Graphs.A*Controller.Learners['consensus_lattice'].compute_violations(Agents.state[0:3,:])
       
            
            
        
        # if there are quadcopters
        dynamics = Agents.dynamics_type
        if dynamics == 'quadcopter': 
            # cycle through the quadcopters
            for q in range(0,Agents.nAgents):
                # and store them
                self.quads_states_all[i,:,q] = Agents.quadList[q].state 
                self.quad_w_cmd_all[i,:,q]   = Agents.llctrlList[q].w_cmd
                self.quads_sDes_all[i,:,q]   = Agents.sDesList[q]
   
