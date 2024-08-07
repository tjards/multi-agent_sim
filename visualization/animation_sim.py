#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:31:52 2023

@author: tjards

New and improved

dev notes:
    
    3 Jul 24: need to import new parameter to reflect connection (pull from A)


"""

# import stuff
# ------------
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os
import json

from data import data_manager

# get quadcopter parameters
# -------------------------

with open(os.path.join("config", "config_agents.json"), 'r') as quad_tests:
    quad_test = json.load(quad_tests)
    if quad_test['dynamics'] =='quadcopter':
        plot_quadcopter = 1
    else:
        plot_quadcopter = 0
        
if plot_quadcopter == 1:
    quad_params = {}
    with open(os.path.join("config", "config_agents.json"), 'r') as configs_agents:
        config_agents = json.load(configs_agents)
        dxm = config_agents['quad_params']['dxm']
        dym = config_agents['quad_params']['dym']
        dzm = config_agents['quad_params']['dzm']
        quad_params["dxm"] = dxm
        quad_params["dym"] = dym
        quad_params["dzm"] = dzm

#%% plotting parameters
# -------------------
numFrames           = 50    # frame rate (bigger = slower)
tail                = 200   # trailing trajectory length 
zoom                = 0     # do you want to adjust frames with motion? [0 = no, 1 = yes, 2 = fixed (set below), 3 = fixed_zoom (set below) ]
pan                 = 0     # camera pan? 0 = no, 1 = yes (memory-heavy)
connection          = 1     # show connections?
connection_thresh   = 5.1     # [legacy] nominally 5.1. how close do agents need to be in order to connect?
updated_connections = 1     # nominally 1 are these connections being updated? nominally, 0; some special cases use 1(RL)
head                = 10*0.2   # size of head pointing forward (shows directionality)
pins_overide        = 1     # default 0, overides using pin variable for colors
showObs             = 1     # (0 = don't show obstacles, 1 = show obstacles, 2 = show obstacles + floors/walls)

# helper function
def quat2Dcm(q):
    dcm = np.zeros([3,3])

    dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
    dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
    dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
    dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
    dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
    dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
    dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return dcm

# main animation function
# -----------------------
def animateMe(data_file_path, Ts,  tactic_type):
    
    # extract the data
    _, t_all = data_manager.load_data_HDF5('History', 't_all', data_file_path)
    _, states_all = data_manager.load_data_HDF5('History', 'states_all', data_file_path)
    _, cmds_all = data_manager.load_data_HDF5('History', 'cmds_all', data_file_path)
    _, targets_all = data_manager.load_data_HDF5('History', 'targets_all', data_file_path)
    _, obstacles_all = data_manager.load_data_HDF5('History', 'obstacles_all', data_file_path)
    _, walls_plots = data_manager.load_data_HDF5('History', 'walls_plots', data_file_path)
    _, centroid_all = data_manager.load_data_HDF5('History', 'centroid_all', data_file_path)
    _, f = data_manager.load_data_HDF5('History', 'f_all', data_file_path)
    _, pins_all = data_manager.load_data_HDF5('History', 'pins_all', data_file_path)
    
    if updated_connections == 1:
        
        _, lattices_connections = data_manager.load_data_HDF5('History', 'lattices', data_file_path)
        _, connectivity = data_manager.load_data_HDF5('History', 'connectivity', data_file_path)
        
    
    quats_all       = []
    
    # pull out key variables
    # ----------------------
    nVeh = states_all.shape[2]
    nObs = obstacles_all.shape[2]
    x = states_all[:,0,:]
    y = states_all[:,1,:]
    z = states_all[:,2,:]
    x_from0 = x
    y_from0 = y
    z_from0 = z
    
    # initialize plot and axis
    # ------------------------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # initialize quadcopter lines
    # ---------------------------
    if plot_quadcopter == 1:
        #quats_all       = History.quads_states_all
        _, quats_all = data_manager.load_data_HDF5('History', 'quads_states_all', data_file_path)
        quatColour      = 'blue'
        quat_line1      = []
        quat_line2      = []
        quat_line3      = []
        #for iVeh in range(0,History.quads_states_all.shape[2]):
        for iVeh in range(0,quats_all.shape[2]):
            quat_line1.append([])
            quat_line2.append([])
            quat_line3.append([])
            quat_line1[iVeh], = ax.plot([], [], [], lw=2, color=quatColour)
            quat_line2[iVeh], = ax.plot([], [], [], lw=2, color='gray')
            quat_line3[iVeh], = ax.plot([], [], [], '--', lw=1, color=quatColour)
  
    # create lists of lines for n-agents and n-obstacles
    # --------------------------------------------------
    lines_dots = []
    lines_tails = []
    lines_heads = []
    lines_targets = []
    lines_obstacles = []
    lattices = []
    # agents
    for i in range(0, nVeh):
        centroids = ax.plot([], [], [], 'kx')
        centroids_line = ax.plot([], [], [], '--', lw=1, color='black')
        line_dot = ax.plot([], [], [], 'bo', ms=3)
        line_tail = ax.plot([], [], [], ':', lw=1, color='blue')
        line_head = ax.plot([], [], [], '-', lw=1, color='magenta')
        line_target = ax.plot([], [], [], 'gx')
        lattice = ax.plot([], [], [], ':', lw=1, color='blue')
        lines_dots.extend(line_dot)
        lines_tails.extend(line_tail)
        lines_heads.extend(line_head)
        lines_targets.extend(line_target)       
        lattices.extend(lattice)
    # obstacles
    if showObs >= 1:
        r_o = obstacles_all[:,3,:]
        for j in range(0, nObs):    
            line_obstacle = ax.plot([], [], [], 'ro', ms = 10*r_o[0,j], markerfacecolor=(1,1,0,0.5) )
            lines_obstacles.extend(line_obstacle)    
        
    # set initial axis properties
    # ---------------------------
    if zoom == 2: 
        fixed_axis = 1000
        ax.set_xlim3d([-fixed_axis, fixed_axis])
        ax.set_ylim3d([-fixed_axis, fixed_axis])
        ax.set_zlim3d([-fixed_axis, fixed_axis])
    else:
        margins = 0.5
        maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + margins
        mid_x = 0.5*(x.max()+x.min())
        mid_y = 0.5*(y.max()+y.min())
        mid_z = 0.5*(z.max()+z.min())
        ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
        ax.set_xlabel('x-direction')
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
        ax.set_ylabel('y-direction')
        ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        ax.set_zlabel('Altitude')
    
    # set labels
    # -----------
    cd = round(np.linalg.norm(centroid_all[0,:,0].ravel()-targets_all[0,0:3,0]),1)
    if tactic_type == 'circle':
        mode = 'Mode: Encirclement'
    elif tactic_type == 'lemni':
        mode = 'Mode: Closed Curves'
    else:
        mode = 'Mode: '+tactic_type
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    titleType1 = ax.text2D(0.95, 0.95, mode, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, '%s : %s' % ("Centroid distance", cd), transform=ax.transAxes, horizontalalignment='right')
    
    # draw the ground (a stationary plane)
    # ------------------------------------
    if showObs == 2:
        for i in range(0, walls_plots.shape[1]):
            xx, yy = np.meshgrid(np.linspace(mid_x-maxRange, mid_x+maxRange, 20), np.linspace(mid_y-maxRange, mid_y+maxRange, 20))
            if walls_plots[2,i] == 0:
                walls_plots[2,i] = 0.001 # avoid divide by zero           
            zz = (-walls_plots[0,i] * xx - walls_plots[1,i] * yy + walls_plots[3,i] * 1.) / walls_plots[2,i]
            ax.plot_wireframe(xx, yy, zz, color='m', rcount=20, ccount=20)
    
    # the update function
    # -------------------
    def update(i):
        
        i-=1
        
        # update states+
        # --------------
        time = t_all[i*numFrames]
        x = states_all[i*numFrames,0,:]
        y = states_all[i*numFrames,1,:]
        z = states_all[i*numFrames,2,:]
        x_from0 = states_all[i*numFrames-tail:i*numFrames,0,:]
        y_from0 = states_all[i*numFrames-tail:i*numFrames,1,:]
        z_from0 = states_all[i*numFrames-tail:i*numFrames,2,:]
        x_v = states_all[i*numFrames,3,:]
        y_v = states_all[i*numFrames,4,:]
        z_v = states_all[i*numFrames,5,:]
        norma = np.maximum(np.sqrt(x_v**2 + y_v**2 + z_v**2),0.0001)
        x_head = x + head*x_v/norma
        y_head = y + head*y_v/norma
        z_head = z + head*z_v/norma
        x_point = np.vstack((x,x_head))
        y_point = np.vstack((y,y_head))
        z_point = np.vstack((z,z_head))
        x_t = targets_all[i*numFrames,0,:]
        y_t = targets_all[i*numFrames,1,:]
        z_t = targets_all[i*numFrames,2,:]
        x_o = obstacles_all[i*numFrames,0,:]
        y_o = obstacles_all[i*numFrames,1,:]
        z_o = obstacles_all[i*numFrames,2,:]
        r_o = obstacles_all[i*numFrames,3,:]        
        pos = states_all[i*numFrames,0:3,:]
        x_lat = np.zeros((2*nVeh,nVeh))
        y_lat = np.zeros((2*nVeh,nVeh))
        z_lat = np.zeros((2*nVeh,nVeh))
        cx = centroid_all[i*numFrames,0,:]
        cy = centroid_all[i*numFrames,1,:]
        cz = centroid_all[i*numFrames,2,:]
        cd = round(np.linalg.norm(centroid_all[i*numFrames,:,0].ravel()-targets_all[i*numFrames,0:3,0]),1)
        
        # update the pan angles (if required)
        # -----------------------------------
        if pan == 1:    
            # don't allow more than 180
            up = (i+1)%180
            # if we get over 90
            if up > 90:
                # start going down
                up = 180-up
            ax.view_init(azim=i+1, elev = up )
        
        # update the axis limits (if required)
        # ------------------------------------       
        if zoom == 1:
            margins = 0.5
            maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + margins
            mid_x = 0.5*(x.max()+x.min())
            mid_y = 0.5*(y.max()+y.min())
            mid_z = 0.5*(z.max()+z.min())
            ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
            ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
            ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        elif zoom == 3:
            fixed_zoom = 300
            cmid_x = 0.5*(cx.max()+cx.min())
            cmid_y = 0.5*(cy.max()+cy.min())
            cmid_z = 0.5*(cz.max()+cz.min())
            ax.set_xlim3d([cmid_x-fixed_zoom, cmid_x+fixed_zoom])
            ax.set_ylim3d([cmid_y-fixed_zoom, cmid_y+fixed_zoom])
            ax.set_zlim3d([cmid_z-fixed_zoom, cmid_z+fixed_zoom])        
        
        
        # update states
        # ------------------
        for j in range(0, nVeh):
            
            # quadcopter lines
            if plot_quadcopter == 1:
                
                dxm2 = quad_params["dxm"]
                dym2 = quad_params["dym"] 
                dzm2 = quad_params["dzm"] 
                quat2 = quats_all[i*numFrames,3:7,j]
                #z2          = z
                #z_from02    = z_from0
                
                # --- #
                #z2 = -z2
                #z_from02 = -z_from02
                #quat2 = np.array([quat2[0], -quat2[1], -quat2[2], quat2[3]])
                # --- #
                
                R2 = quat2Dcm(quat2)    
                motorPoints2 = np.array([[dxm2, -dym2, dzm2], [0, 0, 0], [dxm2, dym2, dzm2], [-dxm2, dym2, dzm2], [0, 0, 0], [-dxm2, -dym2, dzm2]])
                motorPoints2 = np.dot(R2, np.transpose(motorPoints2))
                motorPoints2[0,:] += x[j] 
                motorPoints2[1,:] += y[j] 
                motorPoints2[2,:] += z[j] 
                
                quat_line1[j].set_data(motorPoints2[0,0:3], motorPoints2[1,0:3])
                quat_line1[j].set_3d_properties(motorPoints2[2,0:3])
                quat_line2[j].set_data(motorPoints2[0,3:6], motorPoints2[1,3:6])
                quat_line2[j].set_3d_properties(motorPoints2[2,3:6])
                quat_line3[j].set_data(x_from0[:,j], y_from0[:,j])
                quat_line3[j].set_3d_properties(z_from0[:,j])
                    
            
            # agents
            lines_dots[j].set_data(x[j], y[j])
            lines_dots[j].set_3d_properties(z[j])
    
            # tails
            lines_tails[j].set_data(x_from0[:,j], y_from0[:,j])
            lines_tails[j].set_3d_properties(z_from0[:,j])
            
            # targets
            lines_targets[j] .set_data(x_t[j], y_t[j])
            lines_targets[j] .set_3d_properties(z_t[j]) 
            # heads
            lines_heads[j].set_data(x_point[:,j],y_point[:,j])
            lines_heads[j].set_3d_properties(z_point[:,j])

            # set colors (based on f-factor [legacy])
            if tactic_type == 3:
                if f[i*numFrames] < 0.5:    # if f dros below 0.5, we've transitioned to new tactic
                    if (j % 2) == 0:        # even vehicles go tactic 1, odd vehicles go tactic 2 
                        lines_dots[j].set_color('b')
                        lines_tails[j].set_color('b')
                    else:
                        lines_dots[j].set_color('c')
                        lines_tails[j].set_color('c')
                else:
                    lines_dots[j].set_color('b')
                    lines_tails[j].set_color('b')
                    
            # set colors (based on pins)
            if tactic_type == 'pinning' or pins_overide == 1:
            
                #if pin_matrix[j,j] == 1:
                if pins_all[i*numFrames,j,j] == 1:
                    lines_dots[j].set_color('r')
                    lines_tails[j].set_color('r')
                    if plot_quadcopter == 1:
                        quat_line1[j].set_color('r')
                        quat_line3[j].set_color('r')
                elif pins_all[i*numFrames,j,j] == 2:
                    lines_dots[j].set_color('c')
                    lines_tails[j].set_color('c')
                    if plot_quadcopter == 1:
                        quat_line1[j].set_color('c')
                        quat_line3[j].set_color('c')
                else:
                    lines_dots[j].set_color('b')
                    lines_tails[j].set_color('b')
                    if plot_quadcopter == 1:
                        quat_line1[j].set_color('b')
                        quat_line3[j].set_color('b')

        
        # build lattice (something is not right here)
        # -------------
        if connection == 1:
            for j in range(0, nVeh):
                # search through each neighbour
                for k_neigh in range(0,nVeh):
                    dist = 1000
                    # except for itself (duh):
                    if j != k_neigh:
                        # compute the euc distance between them
                        dist = np.linalg.norm(pos[:,j]-pos[:,k_neigh])
                        # if it is within the interaction range
                        
                        # update the connection threshold
                        if updated_connections == 1:
                            #connection_thresh_updated = History.lattices[i*numFrames,j,k_neigh] + 0.5                            
                            connection_thresh_updated = lattices_connections[i*numFrames,j,k_neigh] + 0.5
                        else:
                            connection_thresh_updated = connection_thresh 

                        #if dist <= connection_thresh_updated:
                        if connectivity[i*numFrames,j,k_neigh] > 0:    
                            
                            
                            # first, itself
                            x_lat[2*k_neigh,j] = pos[0,j]
                            y_lat[2*k_neigh,j] = pos[1,j]
                            z_lat[2*k_neigh,j] = pos[2,j]
                            # then it's neighbour
                            x_lat[2*k_neigh+1,j] = pos[0,k_neigh]
                            y_lat[2*k_neigh+1,j] = pos[1,k_neigh]
                            z_lat[2*k_neigh+1,j] = pos[2,k_neigh]
                        else:
                            x_lat[2*k_neigh:2*k_neigh+2,j] = pos[0,j]
                            y_lat[2*k_neigh:2*k_neigh+2,j] = pos[1,j]
                            z_lat[2*k_neigh:2*k_neigh+2,j] = pos[2,j]
                    else:
                        x_lat[2*k_neigh:2*k_neigh+2,j] = pos[0,j]
                        y_lat[2*k_neigh:2*k_neigh+2,j] = pos[1,j]
                        z_lat[2*k_neigh:2*k_neigh+2,j] = pos[2,j]  
                lattices[j].set_data(x_lat[:,j], y_lat[:,j])
                lattices[j].set_3d_properties(z_lat[:,j])         
                
                if tactic_type == 'shep' and pins_all[i*numFrames,j,j] == 1:
                    lattices[j].set_color('r')

        # centroid
        # --------
        cx = centroid_all[i*numFrames,0,:]
        cy = centroid_all[i*numFrames,1,:]
        cz = centroid_all[i*numFrames,2,:]
        centroids[0].set_data(cx,cy)
        centroids[0].set_3d_properties(cz)
        
        # line from target to centroid
        # ----------------------------
        cx_line=np.vstack((cx,x_t[0])).ravel()
        cy_line=np.vstack((cy,y_t[0])).ravel()
        cz_line=np.vstack((cz,z_t[0])).ravel()
        centroids_line[0].set_data(cx_line,cy_line)
        centroids_line[0].set_3d_properties(cz_line)
        
        # obstacles
        # ---------
        if showObs >= 1:
            for k in range(0, nObs):
                
                lines_obstacles[k].set_data(x_o[k], y_o[k])
                lines_obstacles[k].set_3d_properties(z_o[k])
                
        # labels
        # ------
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        titleType2.set_text('%s : %s' % ("Centroid Distance", cd))
        
    ani = animation.FuncAnimation(fig=fig, func=update, blit=False, frames=len(t_all[0:-2:numFrames]), interval=(Ts*100*numFrames))
    ani.save('visualization/animations/animation3D.gif')
    plt.show()

    
    return ani