#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:09:01 2025

@author: tjards

Show swarm dynamics lie on a circular manifold in latent space.

You’ve reconstructed the phase of your system without explicitly giving it to the model.

"""

# import stuff
# ------------
import os
import data_manager
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# data paths
# -----------
data_directory = 'data'
data_file_path = os.path.join(data_directory, "data.h5")

# labels
# ------
labels = ['x','y','z','vx','vy','vz','cmdx', 'cmdy','cmdz']

# extract values
# --------------
states_all_keys,    states_all_values   = data_manager.load_data_HDF5('History', 'states_all', data_file_path)
cmds_all_keys,      cmds_all_values     = data_manager.load_data_HDF5('History', 'cmds_all', data_file_path)

# create a new dataframe
# ----------------------
agent_data = {}
nAgents = states_all_values.shape[2]

# for each agent
# --------------
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
# ----------------
df = pd.DataFrame(agent_data)

# prep data
# ---------
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


#%% do the UMAP stuff
# -------------------
UMAP_para= False

if not UMAP_para:
    

    from umap import UMAP
    #reducer = UMAP(n_neighbors=15, min_dist=0.1)
    reducer = UMAP(n_neighbors=2, min_dist=0.9, n_components=2)
    #reducer = UMAP(n_neighbors=10, min_dist=0.05, n_components=2)

    X_umap = reducer.fit_transform(X_scaled)
    
    # extact useful
    X_center = X_umap.mean(axis=0)
    theta = np.arctan2(X_umap[:, 1] - X_center[1], X_umap[:, 0] - X_center[0])

    # plot stuff
    import matplotlib.pyplot as plt
    
    # plot type (time, theta)
    plot_type = 'theta'
    
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'time':
        plt.scatter(X_umap[:,0], X_umap[:,1], c=np.arange(len(X_umap)), cmap='plasma', s=5)
        plt.colorbar(label='Time Index')

    if plot_type == 'theta':
        plt.scatter(X_umap[:,0], X_umap[:,1], c=theta, cmap='plasma', s=5)
        plt.colorbar(label='Angular Index')
    
    plt.title("UMAP of Swarm Dynamics")
    plt.grid(True)
    plt.show()

else:
    
    #conda install -c conda-forge umap-learn tensorflow
    #import umap.umap_ as umap
    #from tensorflow.keras.callbacks import EarlyStopping
    from umap.parametric_umap import ParametricUMAP

    # initialize para UMAP (uses a simple MLP by default)
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
    
    # fit to your windowed time series data
    #X_pumap = pumap.fit_transform(X_scaled, callbacks=[early_stop])
    X_pumap = pumap.fit_transform(X_scaled)
    
    #%%
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pumap[:, 0], X_pumap[:, 1], c=np.arange(len(X_pumap)), cmap='plasma', s=5)
    plt.colorbar(label='Time Index')
    plt.title("Parametric UMAP Embedding of Agent Trajectory Windows")
    plt.grid(True)
    plt.show()

#%% add a circle
# ---------------
# estimate center and radius
#x_mean = np.mean(X_umap[:, 0])
#y_mean = np.mean(X_umap[:, 1])
#center = np.array([x_mean, y_mean])
#r = np.mean(np.linalg.norm(X_umap - center, axis=1))
#theta = np.linspace(0, 2 * np.pi, 500)
#circle_x = center[0] + r * np.cos(theta)
#circle_y = center[1] + r * np.sin(theta)
#plt.plot(circle_x, circle_y, 'k--', label='Fitted Circle')

#%% draw a convex hull
# --------------------
from scipy.spatial import ConvexHull
hull = ConvexHull(X_umap)
for simplex in hull.simplices:
    plt.plot(X_umap[simplex, 0], X_umap[simplex, 1], 'k--', lw=1.5)





