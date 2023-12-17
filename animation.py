#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:31:52 2023

@author: tjards

New and improved

"""


# import stuff
# ------------
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# plotting parameters
# -------------------
numFrames           = 20    # frame rate (bigger = slower)
tail                = 200   # trailing trajectory length 
zoom                = 0     # do you want to adjust frames with motion? [0 = no, 1 = yes, 2 = fixed (set below), 3 = fixed_zoom (set below) ]
pan                 = 0     # camera pan? 0 = no, 1 = yes (memory-heavy)
connection          = 1     # show connections?
connection_thresh   = 5.2   # nominally 5.1. how close do agents need to be in order to connect?
head                = 0.2   # size of head pointing forward (shows directionality)
pins_overide        = 1     # default 0, overides using pin variable for colors
showObs             = 0     # (0 = don't show obstacles, 1 = show obstacles, 2 = show obstacles + floors/walls)

# main animation function
# -----------------------
#def animateMe(Ts, t_all, states_all, cmds_all, targets_all, obstacles_all, walls_plots, showObs, centroid_all, f, tactic_type, pins_all):
def animateMe(Ts, History, Obstacles, tactic_type):
    
    # extract
    # -------
    t_all           = History.t_all
    states_all      = History.states_all
    cmds_all        = History.cmds_all
    targets_all     = History.targets_all[:,0:3,:]
    obstacles_all   = History.obstacles_all
    walls_plots     = Obstacles.walls_plots
    centroid_all    = History.centroid_all
    f               = History.f_all
    pins_all        = History.pins_all
  
    
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
        line_dot = ax.plot([], [], [], 'bo')
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
                elif pins_all[i*numFrames,j,j] == 2:
                    lines_dots[j].set_color('c')
                    lines_tails[j].set_color('c') 
                else:
                    lines_dots[j].set_color('b')
                    lines_tails[j].set_color('b')            

        
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
                        if dist <= connection_thresh: 
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
    ani.save('Figs/animation3D.gif')
    plt.show()

    
    return ani