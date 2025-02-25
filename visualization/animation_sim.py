#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:31:52 2023

@author: tjards

[ChatGPT] New version with improvements:
    - Consolidated redundant configuration file reads.
    - Modularized parts of the update routine into helper functions.
    - Abstracted 2D vs. 3D plotting logic where possible.
"""

# =============================================================================
# Import stuff
# =============================================================================
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os
import json
from data import data_manager

# =============================================================================
# Configuration pulls
# =============================================================================
config_path = os.path.join("config", "config_agents.json")
with open(config_path, 'r') as config_file:
    config_agents = json.load(config_file)


# pull the quadcopter config (if applicable in the config)
plot_quadcopter = 1 if config_agents.get('dynamics') == 'quadcopter' else 0
if plot_quadcopter:
    quad_params = {
        "dxm": config_agents['quad_params']['dxm'],
        "dym": config_agents['quad_params']['dym'],
        "dzm": config_agents['quad_params']['dzm']
    }

# =============================================================================
# Plotting parameters
# =============================================================================
numFrames           = 50    # frame rate (bigger = slower)
tail                = 200    # trailing trajectory length 
zoom                = 1     # zoom mode (0 = no, 1 = yes, 2 = fixed (set below), 3 = fixed_zoom (set below))
zoom_axis           = 50    # if zoom mode == 2, sets fixed zoom axis
zoom_fixed          = 20    # if zoom mode == 3, sets fixed zoom amount
pan                 = 0     # camera pan toggle
connection          = 1     # show connections between agents
connection_thresh   = 5.1   # threshold for agent connectivity  (default val, gets updates later)
updated_connections = 1     # use updated connectivity          (default 1)
head                = 5 * 0.2  # size of the directional head
pins_overide        = 1     # override colors using pin variable
showObs             = 1     # obstacle display mode (0 = don't show, 1 = show, 2 = show + floors/walls)
agent_shape         = 'prism'  # ['dot', 'prism']
prism_scale         = 1



# =============================================================================
# Helper functions 
# =============================================================================

# quaternion to cosine matrix conversion
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

# update camera settings (pan and zoom) based on current frame.
def update_camera(ax, x, y, z, cx, cy, cz, i):

    # if panning around
    if pan == 1:
        up = (i+1) % 180
        if up > 90:
            up = 180 - up
        ax.view_init(azim=i+1, elev=up)
    
    # if zooming in (automated)
    if zoom == 1:
        margins = 0.5
        maxRange = 0.5 * np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + margins
        mid_x = 0.5*(x.max()+x.min())
        mid_y = 0.5*(y.max()+y.min())
        mid_z = 0.5*(z.max()+z.min())
        if ax.name == '3d':
            ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
            ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
            ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        else:
            ax.set_xlim([mid_x-maxRange, mid_x+maxRange])
            ax.set_ylim([mid_y-maxRange, mid_y+maxRange])
    
    # if zooming in (fixed amount)
    elif zoom == 3:
        fixed_zoom = zoom_fixed
        if ax.name == '3d':
            mid_x = 0.5*(cx.max()+cx.min())
            mid_y = 0.5*(cy.max()+cy.min())
            mid_z = 0.5*(cz.max()+cz.min())
            ax.set_xlim3d([mid_x-fixed_zoom, mid_x+fixed_zoom])
            ax.set_ylim3d([mid_y-fixed_zoom, mid_y+fixed_zoom])
            ax.set_zlim3d([mid_z-fixed_zoom, mid_z+fixed_zoom])
        else:
            mid_x = 0.5*(cx.max()+cx.min())
            mid_y = 0.5*(cy.max()+cy.min())
            ax.set_xlim([mid_x-fixed_zoom, mid_x+fixed_zoom])
            ax.set_ylim([mid_y-fixed_zoom, mid_y+fixed_zoom])

# update quadcopter visual for agent j at frame i
def update_quadcopter(i, x, y, z, x_from0, y_from0, z_from0, quats_all, j, ax, quat_line1, quat_line2, quat_line3):
    dxm2 = quad_params["dxm"]
    dym2 = quad_params["dym"]
    dzm2 = quad_params["dzm"]
    quat2 = quats_all[i*numFrames, 3:7, j]
    R2 = quat2Dcm(quat2)
    motorPoints2 = np.array([
        [dxm2, -dym2, dzm2],
        [0, 0, 0],
        [dxm2, dym2, dzm2],
        [-dxm2, dym2, dzm2],
        [0, 0, 0],
        [-dxm2, -dym2, dzm2]
    ])
    motorPoints2 = np.dot(R2, motorPoints2.T)
    motorPoints2[0, :] += x[j]
    motorPoints2[1, :] += y[j]
    motorPoints2[2, :] += z[j]
    
    if ax.name == '3d':
        quat_line1[j].set_data(motorPoints2[0, 0:3], motorPoints2[1, 0:3])
        quat_line1[j].set_3d_properties(motorPoints2[2, 0:3])
        quat_line2[j].set_data(motorPoints2[0, 3:6], motorPoints2[1, 3:6])
        quat_line2[j].set_3d_properties(motorPoints2[2, 3:6])
        quat_line3[j].set_data(x_from0[:, j], y_from0[:, j])
        quat_line3[j].set_3d_properties(z_from0[:, j])
    else:
        quat_line1[j].set_data(motorPoints2[0, 0:3], motorPoints2[1, 0:3])
        quat_line2[j].set_data(motorPoints2[0, 3:6], motorPoints2[1, 3:6])
        quat_line3[j].set_data(x_from0[:, j], y_from0[:, j])

# create prisms (credit: Eric Kim)
def create_prism_points_3d(x, y, z, vx, vy, vz, scale=1):

    # direction defined by velocity vector
    direction_vector = np.array([vx, vy, vz])
    norm = np.linalg.norm(direction_vector)
    
    # avoid dividing by zero
    if norm == 0:
        return np.array([[x, y, z]])  

    # normalize the velocity vector
    direction_vector /= norm

    # default point nose pointing in +x direction
    triangle_base = np.array([[scale, 0, 0],
                              [-scale * 0.5, 0, scale * 0.5],
                              [-scale * 0.5, 0, -scale * 0.5]])

    # create rotation matrix to align x-axis with direction vector
    x_axis = np.array([1, 0, 0])
    v = np.cross(x_axis, direction_vector)
    s = np.linalg.norm(v)
    c = np.dot(x_axis, direction_vector)

    if s != 0:
        vx_matrix = np.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx_matrix + (vx_matrix @ vx_matrix) * ((1 - c) / (s ** 2))
        rotated_triangle = (rotation_matrix @ triangle_base.T).T
    else:
        rotated_triangle = triangle_base

    return rotated_triangle + np.array([x, y, z])


def create_prism_points_2d(x, y, vx, vy, scale=1.0):

    direction_vector = np.array([vx, vy])
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        return np.array([[x, y]]) 

    direction_vector /= norm

    # Define a default triangle (pointing along the +x direction)
    triangle_base = np.array([[scale, 0],
                              [-scale * 0.5, scale * 0.5],
                              [-scale * 0.5, -scale * 0.5]])

    # Calculate the rotation angle to align the triangle with the velocity.
    angle = np.arctan2(direction_vector[1], direction_vector[0])
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
    rotated_triangle = (rot_matrix @ triangle_base.T).T

    return rotated_triangle + np.array([x, y])



def update_agents_and_obstacles(i, states_all, targets_all, obstacles_all,
                                lines_dots, lines_tails, lines_heads, lines_targets,
                                lines_obstacles, nVeh, nObs, ax,
                                draw_mode='dot', prism_plots=None):

    # extract position and tail data.
    x = states_all[i*numFrames, 0, :]
    y = states_all[i*numFrames, 1, :]
    z = states_all[i*numFrames, 2, :]
    x_from0 = states_all[i*numFrames-tail:i*numFrames, 0, :]
    y_from0 = states_all[i*numFrames-tail:i*numFrames, 1, :]
    z_from0 = states_all[i*numFrames-tail:i*numFrames, 2, :]
    
    # get velocity components (used for both drawing options)
    x_v = states_all[i*numFrames, 3, :]
    y_v = states_all[i*numFrames, 4, :]
    z_v = states_all[i*numFrames, 5, :]
    norma = np.maximum(np.sqrt(x_v**2 + y_v**2 + z_v**2), 0.0001)
    
    # calculate "head" points to show velocity direction.
    x_head = x + head * x_v / norma
    y_head = y + head * y_v / norma
    z_head = z + head * z_v / norma

    # draw tails
    for j in range(nVeh):
        lines_tails[j].set_data(x_from0[:, j], y_from0[:, j])
        if ax.name == '3d':
            lines_tails[j].set_3d_properties(z_from0[:, j])

    # if using dots
    if draw_mode == 'dot':
        for j in range(nVeh):
            lines_dots[j].set_data(x[j], y[j])
            if ax.name == '3d':
                lines_dots[j].set_3d_properties(z[j])
            
            #lines_tails[j].set_data(x_from0[:, j], y_from0[:, j])
            #if ax.name == '3d':
            #    lines_tails[j].set_3d_properties(z_from0[:, j])
            
            x_t = targets_all[i*numFrames, 0, j]
            y_t = targets_all[i*numFrames, 1, j]
            z_t = targets_all[i*numFrames, 2, j]
            lines_targets[j].set_data(x_t, y_t)
            if ax.name == '3d':
                lines_targets[j].set_3d_properties(z_t)
            
            # Draw a line representing the heading/direction.
            x_point = [x[j], x_head[j]]
            y_point = [y[j], y_head[j]]
            z_point = [z[j], z_head[j]]
            lines_heads[j].set_data(x_point, y_point)
            if ax.name == '3d':
                lines_heads[j].set_3d_properties(z_point)
    
    
    elif draw_mode == 'prism' and prism_plots is not None:
     
        if ax.name != '3d':  # 2D case.
            for j in range(nVeh):
                prism_points = create_prism_points_2d(x[j], y[j], x_v[j], y_v[j], scale=prism_scale)
                # Assume each prism_plots[j] is a matplotlib.patches.Polygon.
                prism_plots[j].set_xy(prism_points)
        else:  # 3D case remains as before.
            for j in range(nVeh):
                prism_points = create_prism_points_3d(x[j], y[j], z[j],
                                                      x_v[j], y_v[j], z_v[j],
                                                      scale=prism_scale)
                prism_plots[j].set_verts([prism_points])


    '''
    elif draw_mode == 'prism' and prism_plots is not None:

        #prism_scale = 0.5  
        for j in range(nVeh):
            
            # compute the prism (triangle) points
            prism_points = create_prism_points_3d(x[j], y[j], z[j],
                                                  x_v[j], y_v[j], z_v[j],
                                                  scale=prism_scale)
            
            # update the corresponding prism plot object
            prism_plots[j].set_verts([prism_points])
    '''
    
    #  update obstacles (common to both modes)
    if showObs >= 1:
        x_o = obstacles_all[i*numFrames, 0, :]
        y_o = obstacles_all[i*numFrames, 1, :]
        z_o = obstacles_all[i*numFrames, 2, :]
        for k in range(nObs):
            lines_obstacles[k].set_data(x_o[k], y_o[k])
            if ax.name == '3d':
                lines_obstacles[k].set_3d_properties(z_o[k])
    
    return x, y, z, x_from0, y_from0, z_from0



# update the connections
def update_connectivity(i, pos, lattices, nVeh, lattices_connections, connectivity):

    x_lat = np.zeros((2 * nVeh, nVeh))
    y_lat = np.zeros((2 * nVeh, nVeh))
    z_lat = np.zeros((2 * nVeh, nVeh))
    
    # cycle through agents
    for j in range(nVeh):
        
        # and neighbours
        for k in range(nVeh):
            
            # ignore self
            if j == k:
                x_lat[2*k:2*k+2, j] = pos[0, j]
                y_lat[2*k:2*k+2, j] = pos[1, j]
                z_lat[2*k:2*k+2, j] = pos[2, j]
            
            else:
                
                # updated the lattice connection theshold
                dist = np.linalg.norm(pos[:, j] - pos[:, k])
                connection_thresh_updated = (lattices_connections[i*numFrames, j, k] + 0.5) if updated_connections == 1 else connection_thresh
                
                # if in range (i.e., via adjanceny matrix), make grey
                if connectivity[i*numFrames, j, k] > 0:
                    lattices[j].set_color('gray')
                    lattices[j].set_linestyle('--')
                    lattices[j].set_alpha(0.3)
                    
                    # if in range and also within lattice threshold, make blue
                    if dist <= connection_thresh_updated:
                        lattices[j].set_color('b')
                        lattices[j].set_alpha(0.6)
                    
                    # draw the line between agent j and agent k.
                    x_lat[2*k, j] = pos[0, j]
                    x_lat[2*k+1, j] = pos[0, k]
                    y_lat[2*k, j] = pos[1, j]
                    y_lat[2*k+1, j] = pos[1, k]
                    z_lat[2*k, j] = pos[2, j]
                    z_lat[2*k+1, j] = pos[2, k]
                else:
                    # if not in range, terminate line
                    x_lat[2*k:2*k+2, j] = pos[0, j]
                    y_lat[2*k:2*k+2, j] = pos[1, j]
                    z_lat[2*k:2*k+2, j] = pos[2, j]
        lattices[j].set_data(x_lat[:, j], y_lat[:, j])
        if lattices[j].axes.name == '3d':
            lattices[j].set_3d_properties(z_lat[:, j])


# =============================================================================
# Main animation function with modularized update logic
# =============================================================================
def animateMe(data_file_path, Ts, dimens, tactic_type):
    
    # load all the data
    _, t_all          = data_manager.load_data_HDF5('History', 't_all', data_file_path)
    _, states_all     = data_manager.load_data_HDF5('History', 'states_all', data_file_path)
    _, cmds_all       = data_manager.load_data_HDF5('History', 'cmds_all', data_file_path)
    _, targets_all    = data_manager.load_data_HDF5('History', 'targets_all', data_file_path)
    _, obstacles_all  = data_manager.load_data_HDF5('History', 'obstacles_all', data_file_path)
    _, walls_plots    = data_manager.load_data_HDF5('History', 'walls_plots', data_file_path)
    _, centroid_all   = data_manager.load_data_HDF5('History', 'centroid_all', data_file_path)
    _, f              = data_manager.load_data_HDF5('History', 'f_all', data_file_path)
    _, pins_all       = data_manager.load_data_HDF5('History', 'pins_all', data_file_path)
    
    if updated_connections == 1:
        _, lattices_connections = data_manager.load_data_HDF5('History', 'lattices', data_file_path) # lattice scale para
        _, connectivity = data_manager.load_data_HDF5('History', 'connectivity', data_file_path)     #  adjacency matrix
    else:
        lattices_connections = None
        _, connectivity = data_manager.load_data_HDF5('History', 'connectivity', data_file_path)
    
    # load quaternion data for quadcopters (if needed)
    if plot_quadcopter:
        _, quats_all = data_manager.load_data_HDF5('History', 'quads_states_all', data_file_path)
    
    # initialize stuff
    nVeh = states_all.shape[2]
    nObs = obstacles_all.shape[2]
    x_all = states_all[:, 0, :]
    y_all = states_all[:, 1, :]
    z_all = states_all[:, 2, :]
    fig = plt.figure()
    if dimens == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    
    if agent_shape == 'prism' and plot_quadcopter != 1 and dimens == 3:
        prism_plots = []
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for j in range(nVeh):
            poly = Poly3DCollection([], facecolors='blue', edgecolors='blue', alpha=0.5)
            ax.add_collection3d(poly)
            prism_plots.append(poly)
    elif agent_shape == 'prism' and plot_quadcopter != 1 and dimens == 2:
        prism_plots = []
        from matplotlib.patches import Polygon
        for j in range(nVeh):
            poly = Polygon([[0, 0]], closed=True, facecolor='blue', edgecolor='blue', alpha=0.5)
            ax.add_patch(poly)
            prism_plots.append(poly)
    else:
        prism_plots = None
        print('Note: you cant use prism in this config')
        

  
    # =============================================================================
    # Initialize plot elements for quadcopters (if needed)
    # =============================================================================
    if plot_quadcopter:
        quatColour = 'blue'
        quat_line1, quat_line2, quat_line3 = [], [], []
        for iVeh in range(nVeh):
            if dimens == 3:
                l1, = ax.plot([], [], [], lw=2, color=quatColour)
                l2, = ax.plot([], [], [], lw=2, color='gray')
                l3, = ax.plot([], [], [], '--', lw=1, color=quatColour)
            else:
                l1, = ax.plot([], [], lw=2, color=quatColour)
                l2, = ax.plot([], [], lw=2, color='gray')
                l3, = ax.plot([], [], '--', lw=1, color=quatColour)
            quat_line1.append(l1)
            quat_line2.append(l2)
            quat_line3.append(l3)
    
    # =============================================================================
    # Initialize plot elements for agents and obstacles
    # =============================================================================
    lines_dots, lines_tails, lines_heads, lines_targets, lattices = [], [], [], [], []
    for i in range(nVeh):
        if dimens == 3:
            dot, = ax.plot([], [], [], 'bo', ms=3)
            tail, = ax.plot([], [], [], ':', lw=1, color='blue')
            head_line, = ax.plot([], [], [], '-', lw=1, color='magenta')
            target, = ax.plot([], [], [], 'gx')
            lattice_line, = ax.plot([], [], [], ':', lw=1, color='blue')
        else:
            dot, = ax.plot([], [], 'bo', ms=3)
            tail, = ax.plot([], [], ':', lw=1, color='blue')
            head_line, = ax.plot([], [], '-', lw=1, color='magenta')
            target, = ax.plot([], [], 'gx')
            lattice_line, = ax.plot([], [], ':', lw=1, color='blue')
        lines_dots.append(dot)
        lines_tails.append(tail)
        lines_heads.append(head_line)
        lines_targets.append(target)
        lattices.append(lattice_line)
    
    # initialize obstacles (if required)
    lines_obstacles = []
    if showObs >= 1:
        r_o = obstacles_all[:, 3, :]
        for j in range(nObs):
            if dimens == 3:
                obstacle_line, = ax.plot([], [], [], 'ro', ms=10*r_o[0, j], markerfacecolor=(1,1,0,0.5))
            else:
                obstacle_line, = ax.plot([], [], 'ro', ms=10*r_o[0, j], markerfacecolor=(1,1,0,0.5))
            lines_obstacles.append(obstacle_line)
    
    # =============================================================================
    # Set initial axis limits and labels
    # =============================================================================
    if dimens == 3:
        
        # for fixed axis size
        if zoom == 2:
            fixed_axis = zoom_axis
            ax.set_xlim3d([-fixed_axis, fixed_axis])
            ax.set_ylim3d([-fixed_axis, fixed_axis])
            ax.set_zlim3d([-fixed_axis, fixed_axis])
        # autogenerated axis
        else:
            margins = 0.5
            maxRange = 0.5 * np.array([x_all.max()-x_all.min(), y_all.max()-y_all.min(), z_all.max()-z_all.min()]).max() + margins
            mid_x = 0.5 * (x_all.max() + x_all.min())
            mid_y = 0.5 * (y_all.max() + y_all.min())
            mid_z = 0.5 * (z_all.max() + z_all.min())
            ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
            ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
            ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
            ax.set_xlabel('x-direction')
            ax.set_ylabel('y-direction')
            ax.set_zlabel('Altitude')
    else:
        # for fixed axis size
        if zoom == 2:
            fixed_axis = zoom_axis
            ax.set_xlim([-fixed_axis, fixed_axis])
            ax.set_ylim([-fixed_axis, fixed_axis])
        # autogenerated axis
        else:
            margins = 0.5
            maxRange = 0.5 * np.array([x_all.max()-x_all.min(), y_all.max()-y_all.min()]).max() + margins
            mid_x = 0.5 * (x_all.max() + x_all.min())
            mid_y = 0.5 * (y_all.max() + y_all.min())
            ax.set_xlim([mid_x-maxRange, mid_x+maxRange])
            ax.set_ylim([mid_y-maxRange, mid_y+maxRange])
            ax.set_xlabel('x-direction')
            ax.set_ylabel('y-direction')
    
    # set text labels for simulation mode and centroid distance
    cd = round(np.linalg.norm(centroid_all[0, :, 0].ravel() - targets_all[0, 0:3, 0]), 1)
    if tactic_type == 'circle':
        mode = 'Mode: Encirclement'
    elif tactic_type == 'lemni':
        mode = 'Mode: Closed Curves'
    else:
        mode = 'Mode: ' + tactic_type
    
    if dimens == 3:
        titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        titleType1 = ax.text2D(0.95, 0.95, mode, transform=ax.transAxes, horizontalalignment='right')
        titleType2 = ax.text2D(0.95, 0.91, 'Centroid Distance : {}'.format(cd), transform=ax.transAxes, horizontalalignment='right')
    else:
        titleTime = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        titleType1 = ax.text(0.95, 0.95, mode, transform=ax.transAxes, horizontalalignment='right')
        titleType2 = ax.text(0.95, 0.91, 'Centroid Distance : {}'.format(cd), transform=ax.transAxes, horizontalalignment='right')
    
    # draw ground plane if required (3D and showObs==2)
    if showObs == 2 and dimens == 3:
        for i in range(walls_plots.shape[1]):
            xx, yy = np.meshgrid(np.linspace(mid_x-maxRange, mid_x+maxRange, 20),
                                   np.linspace(mid_y-maxRange, mid_y+maxRange, 20))
            if walls_plots[2, i] == 0:
                walls_plots[2, i] = 0.001  # avoid divide by zero
            zz = (-walls_plots[0, i] * xx - walls_plots[1, i] * yy + walls_plots[3, i]) / walls_plots[2, i]
            ax.plot_wireframe(xx, yy, zz, color='m', rcount=20, ccount=20)
    
    # =============================================================================
    # The update function for animation (calls modular helpers)
    # =============================================================================
    def update(i):
        
        i -= 1  
        time = t_all[i*numFrames]
        
        # update agents and obstacles
        # ---------------------------
        x, y, z, x_from0, y_from0, z_from0 = update_agents_and_obstacles(
            i, states_all, targets_all, obstacles_all,
            lines_dots, lines_tails, lines_heads, lines_targets, lines_obstacles,
            nVeh, nObs, ax, agent_shape, prism_plots
        )
        
        # update quadcopter visuals (if applicable)
        # -----------------------------------------
        if plot_quadcopter:
            for j in range(nVeh):
                update_quadcopter(i, x, y, z, x_from0, y_from0, z_from0, quats_all, j, ax, quat_line1, quat_line2, quat_line3)
        
        # update connectivity lines (if enabled)
        # -------------------------------------
        pos = states_all[i*numFrames, 0:3, :]
        if connection:
            update_connectivity(i, pos, lattices, nVeh, lattices_connections, connectivity)
        
        # update centroid and its line
        # ----------------------------
        cx = centroid_all[i*numFrames, 0, :]
        cy = centroid_all[i*numFrames, 1, :]
        cz = centroid_all[i*numFrames, 2, :]
        centroids = ax.lines[-2]
        centroids_line = ax.lines[-1]
        centroids.set_data(cx, cy)
        if dimens == 3:
            centroids.set_3d_properties(cz)
        
        cx_line = np.vstack((cx, targets_all[i*numFrames, 0, 0])).ravel()
        cy_line = np.vstack((cy, targets_all[i*numFrames, 1, 0])).ravel()
        cz_line = np.vstack((cz, targets_all[i*numFrames, 2, 0])).ravel()
        centroids_line.set_data(cx_line, cy_line)
        if dimens == 3:
            centroids_line.set_3d_properties(cz_line)
        
        # update camera (pan/zoom)
        # -------------------------
        update_camera(ax, x, y, z, centroid_all[i*numFrames, 0, :],
                      centroid_all[i*numFrames, 1, :],
                      centroid_all[i*numFrames, 2, :], i)
        
        # update text labels
        # -------------------
        cd = round(np.linalg.norm(centroid_all[i*numFrames, :, 0].ravel() - targets_all[i*numFrames, 0:3, 0]), 1)
        titleTime.set_text("Time = {:.2f} s".format(time))
        titleType2.set_text('Centroid Distance : {}'.format(cd))
    
    # ======================================
    # Create animation using update function
    # ======================================
    ani = animation.FuncAnimation(fig=fig, func=update, blit=False,
                                  frames=len(t_all[0:-2:numFrames]),
                                  interval=(Ts*100*numFrames))
    
    ani.save('visualization/animations/animation3D.gif')
    plt.show()
    return ani
