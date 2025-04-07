#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:09:01 2025

@author: tjards

This project uses UMAP (https://github.com/lmcinnes/umap) 
    to demonstgrate the swarm dynamics outlined in 
    "Emergent Homeomorphic Curves in Swarms", Automatica, 2025
    (https://doi.org/10.1016/j.automatica.2025.112221)
    lie on a circular manifold in latent space.
    
    We reconstruct the phase of the system without explicity giving
    it to the model; this essentially "discovers" the circular
    embedding formulated in the paper using data-based methods.
    
    Accepts agent data produced from simimlations at 
    https://github.com/tjards/multi-agent_sim

"""

#%% preliminaries
# ---------------
import os
import data_manager
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# data paths
data_directory = 'data'
data_file_path = os.path.join(data_directory, "data.h5")

# labels
labels = ['x','y','z','vx','vy','vz','cmdx', 'cmdy','cmdz']

#%% extract, transform, load data
# -------------------------------
states_all_keys,    states_all_values   = data_manager.load_data_HDF5('History', 'states_all', data_file_path)
cmds_all_keys,      cmds_all_values     = data_manager.load_data_HDF5('History', 'cmds_all', data_file_path)

# create a new dataframe
agent_data = {}
nAgents = states_all_values.shape[2]

# for each agent
t_start = 100
t_end = states_all_values.shape[0]+1
for i in range(nAgents):
#for i in range(0,4):
    
    agent_data[labels[0]+'_'+str(i)] = states_all_values[t_start:t_end,0,i]
    agent_data[labels[1]+'_'+str(i)] = states_all_values[t_start:t_end,1,i]
    agent_data[labels[2]+'_'+str(i)] = states_all_values[t_start:t_end,2,i]
    agent_data[labels[3]+'_'+str(i)] = states_all_values[t_start:t_end,3,i]
    agent_data[labels[4]+'_'+str(i)] = states_all_values[t_start:t_end,4,i]
    agent_data[labels[5]+'_'+str(i)] = states_all_values[t_start:t_end,5,i]
    agent_data[labels[6]+'_'+str(i)] = cmds_all_values[t_start:t_end,0,i]
    agent_data[labels[7]+'_'+str(i)] = cmds_all_values[t_start:t_end,1,i]
    agent_data[labels[8]+'_'+str(i)] = cmds_all_values[t_start:t_end,2,i]
        
# make a dataframe
df = pd.DataFrame(agent_data)

# prep data
window_size = 200
stride = 1

# function for windowizing
def create_windows(data: np.ndarray, window_size: int, stride: int = 1):
    X = []
    T = data.shape[0]
    for i in range(0, T - window_size + 1, stride):
        x_window = data[i:i + window_size]
        X.append(x_window)
    return np.array(X)

# windowize
X = create_windows(df.values, window_size, stride)

# flatten
X_flat = X.reshape(X.shape[0], -1)

# scale
X_scaled = StandardScaler().fit_transform(X_flat)


#%% model using UMAP
# -------------------
UMAP_para= False

# is using basic UMAP
if not UMAP_para:
    
    from umap import UMAP

    # note: I use atypical UMAP parameters, which reflect the application
    # I choose a small n_neighbours, indicating the reliance on local 
    # observations described in the paper; yet, I use a high min_distance,
    # indicating I care about the global behavious

    # initialize, fit, transform
    reducer = UMAP(n_neighbors=2, min_dist=0.9, n_components=2) # 
    X_umap = reducer.fit_transform(X_scaled)
    
    # extact phase info
    X_center = X_umap.mean(axis=0)
    theta = np.arctan2(X_umap[:, 1] - X_center[1], X_umap[:, 0] - X_center[0])

    #%% plot stuff
    import matplotlib.pyplot as plt
    
    plot_type = 'theta'
    plt.figure(figsize=(8, 6))
    
    # plots over time
    if plot_type == 'time':
        plt.scatter(X_umap[:,0], X_umap[:,1], c=np.arange(len(X_umap)), cmap='plasma', s=5)
        plt.colorbar(label='Time Index (s)')
    # plots over phase
    if plot_type == 'theta':
        plt.scatter(X_umap[:,0], X_umap[:,1], c=theta, cmap='plasma', s=5)
        plt.colorbar(label='Phase Index (radians)')
    
    plt.title("UMAP of Swarm Dynamics")
    plt.grid(True)
    plt.show()

# if using parametric (useful for later ML application)
else:
    
    from umap.parametric_umap import ParametricUMAP

    # initialize
    pumap = ParametricUMAP(
        n_neighbors=2,
        min_dist=0.02, # default 0.1
        n_components=2,
        metric='euclidean',
        verbose=True,
        random_state=42
    )
    
    #early_stop = EarlyStopping(
    #    monitor='loss',
    #    patience=1,           # Stop after 3 epochs with no improvement
    #    min_delta=0.01,      # Minimum change to qualify as an improvement
    #    restore_best_weights=True
    #)
    
    # fit, transform
    #X_pumap = pumap.fit_transform(X_scaled, callbacks=[early_stop])
    X_pumap = pumap.fit_transform(X_scaled)
    
    #%% plot stuff
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pumap[:, 0], X_pumap[:, 1], c=np.arange(len(X_pumap)), cmap='plasma', s=5)
    plt.colorbar(label='Time Index')
    plt.title("UMAP of Swarm Dynamics (parametric)")
    plt.grid(True)
    plt.show()

#%% draw a convex hull
# --------------------
from scipy.spatial import ConvexHull
hull = ConvexHull(X_umap)
for simplex in hull.simplices:
    plt.plot(X_umap[simplex, 0], X_umap[simplex, 1], 'k--', lw=1.5)





