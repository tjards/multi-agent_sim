#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:03:48 2024

@author: tjards
"""

# import stuff
# -------------
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#plt.style.use('classic')
plt.style.use('default')
#plt.style.available
#plt.style.use('Solarize_Light2')

from data import data_manager
import os

# paths
# -----
data_directory = 'data'
data_file_path = os.path.join(data_directory, "data.h5")


# produce plots
# -------------
def plotMe(data_file_path):
    
    _, t_all = data_manager.load_data_HDF5('History', 't_all', data_file_path)
    _, metrics_order_all = data_manager.load_data_HDF5('History', 'metrics_order_all', data_file_path)
    _, states_all = data_manager.load_data_HDF5('History', 'states_all', data_file_path)
    _, targets_all = data_manager.load_data_HDF5('History', 'targets_all', data_file_path)
    _, obstacles_all = data_manager.load_data_HDF5('History', 'obstacles_all', data_file_path)

    # separtion 
    fig, ax = plt.subplots()
    ax.plot(t_all[4::],metrics_order_all[4::,1],'-b')
    ax.plot(t_all[4::],metrics_order_all[4::,5],':b')
    ax.plot(t_all[4::],metrics_order_all[4::,6],':b')
    ax.fill_between(t_all[4::], metrics_order_all[4::,5], metrics_order_all[4::,6], color = 'blue', alpha = 0.1)
    #note: can include region to note shade using "where = Y2 < Y1
    ax.set(xlabel='Time [s]', ylabel='Mean Distance (with Min/Max Bounds) [m]',
            title='Separation between Agents')
    #ax.plot([70, 70], [100, 250], '--b', lw=1)
    #ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
    ax.grid()
    plt.show()
    
    # radii from target
    radii = np.zeros([states_all.shape[2],states_all.shape[0]])
    for i in range(0,states_all.shape[0]):
        for j in range(0,states_all.shape[2]):
            radii[j,i] = np.linalg.norm(states_all[i,:,j] - targets_all[i,:,j])
            
    fig, ax = plt.subplots()
    for j in range(0,states_all.shape[2]):
        ax.plot(t_all[4::],radii[j,4::].ravel(),'-b')
    ax.set(xlabel='Time [s]', ylabel='Distance from Target for Each Agent [m]',
            title='Distance from Target')
    #plt.axhline(y = 5, color = 'k', linestyle = '--')
    plt.show()
    
    #%% radii from obstacles
    if obstacles_all.shape[2] >  0:
    
        radii_o = np.zeros([states_all.shape[2],states_all.shape[0],obstacles_all.shape[2]])
        radii_o_means = np.zeros([states_all.shape[2],states_all.shape[0]])
        radii_o_means2 =  np.zeros([states_all.shape[0]])
        
        for i in range(0,states_all.shape[0]):              # the time samples
            for j in range(0,states_all.shape[2]):          # the agents
                for k in range(0,obstacles_all.shape[2]):   # the obstacles
                    radii_o[j,i,k] = np.linalg.norm(states_all[i,0:3,j] - obstacles_all[i,0:3,k])
        
                radii_o_means[j,i] = np.mean(radii_o[j,i,:])
            radii_o_means2[i] = np.mean(radii_o_means[:,i])
        
                
        fig, ax = plt.subplots()
        start = int(0/0.02)
        
        for j in range(0,states_all.shape[2]):
            ax.plot(t_all[start::],radii_o_means2[start::].ravel(),'-g')
        ax.set(xlabel='Time [s]', ylabel='Mean Distance from Landmarks [m]',
                title='Learning Progress')
        #plt.axhline(y = 5, color = 'k', linestyle = '--')
        
        plt.show()
        
        
    #%% local k-connectivity
    _, t_all = data_manager.load_data_HDF5('History', 't_all', data_file_path)
    _, local_k_connectivity = data_manager.load_data_HDF5('History', 'local_k_connectivity', data_file_path)
    
    # Plotting the array
    fig, ax = plt.subplots()
    start = int(0/0.02)
    #for i in range(0,len(local_k_connectivity[0,:])):
    temp_means = 0*local_k_connectivity[:,0]   
    temp_maxs = 0*local_k_connectivity[:,0]
    temp_mins = 0*local_k_connectivity[:,0] 
    for i in range(start,len(t_all)):
        temp_means[i] = np.mean(local_k_connectivity[i,:].ravel())
        temp_mins[i] = np.min(local_k_connectivity[i,:].ravel())
        temp_maxs[i] = np.max(local_k_connectivity[i,:].ravel())
    
    ax.plot(t_all[start::],temp_means[start::],'-b')
    ax.fill_between(t_all[start::], temp_mins[start::], temp_maxs[start::], color = 'blue', alpha = 0.2)
    
    plt.title('k-connectivity')
    plt.xlabel('time [s]')
    plt.ylabel('local k-connectivity [mean +/- max/min]')
    plt.show()

    
    