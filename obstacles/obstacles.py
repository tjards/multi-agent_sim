#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:11:18 2024

@author: tjards
"""

# import stuff
# ------------
import numpy as np
import copy

# parameters
# ----------
nObs    = 2

# define the obstacle object
# --------------------------
class Obstacles:
    
    def __init__(self, tactic_type, targets):
        
        # note: don't let pass-in of walls yet, as it is a manual process still
        
        # initiate attributes
        # -------------------
        self.nObs    = nObs     # number of obstacles 
        self.vehObs  = 1     # include other vehicles as obstacles [0 = no, 1 = yes] 

        # if using reynolds, need make target an obstacle 
        if tactic_type == 'reynolds':
            self.targetObs = 1
        else:
            self.targetObs = 0   

        # there are no obstacle, but we need to make target an obstacle 
        if self.nObs == 0 and self.targetObs == 1:
            self.nObs = 1

        self.obstacles = np.zeros((4,self.nObs))
        oSpread = 20

        # manual (comment out if random)
        # obstacles[0,:] = 0    # position (x)
        # obstacles[1,:] = 0    # position (y)
        # obstacles[2,:] = 0    # position (z)
        # obstacles[3,:] = 0

        #random (comment this out if manual)
        if self.nObs != 0:
            self.obstacles[0,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[0,0]                   # position (x)
            self.obstacles[1,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[1,0]                   # position (y)
            self.obstacles[2,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[2,0]                  # position (z)
            #obstacles[2,:] = np.maximum(oSpread*(np.random.rand(1,nObs)-0.5),14)     # position (z)
            self.obstacles[3,:] = np.random.rand(1,self.nObs)+1                             # radii of obstacle(s)

        # manually make the first target an obstacle
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
            self.obstacles[3,0] = 2              # radii of obstacle(s)

        # Walls/Floors 
        # - these are defined manually as planes
        # --------------------------------------   
        self.nWalls = 1                      # default 1, as the ground is an obstacle 
        self.walls = np.zeros((6,self.nWalls)) 
        self.walls_plots = np.zeros((4,self.nWalls))

        # add the ground at z = 0:
        newWall0, newWall_plots0 = self.buildWall('horizontal', -2) 

        # load the ground into constraints   
        self.walls[:,0] = newWall0[:,0]
        self.walls_plots[:,0] = newWall_plots0[:,0]
        
        #self.obstacles_plus = self.obstacles.copy()
        self.obstacles_plus = copy.deepcopy(self.obstacles)
        
        self.config_obstacles = {'nObs': self.nObs, 'nWalls': self.nWalls, 'oSpread': oSpread, 'vehObs': self.vehObs} 
                
    def buildWall(self, wType, pos): 
        
        if wType == 'horizontal':
            
            # define 3 points on the plane (this one is horizontal)
            wallp1 = np.array([0, 0, pos])
            wallp2 = np.array([5, 10, pos])
            wallp3 = np.array([20, 30, pos+0.05])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                    
        if wType == 'vertical1':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([5, pos, 10])
            wallp3 = np.array([20,pos+0.05, 30])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
            
        if wType == 'vertical2':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos, 5, 10])
            wallp3 = np.array([pos+0.05, 20, 30])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                   
        if wType == 'diagonal1a':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([0, pos+5, 5])
            wallp3 = np.array([-5,pos+5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal1b':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([0, pos-5, 5])
            wallp3 = np.array([-5,pos-5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal2a':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos-5, 0, 5])
            wallp3 = np.array([pos-5, -5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal2b':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos+5, 0, 5])
            wallp3 = np.array([pos+5, -5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                    
        return walls, walls_plots
    
    def evolve(self, targets, state, rVeh):
        
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
            
         # Add other vehicles as obstacles (optional, default = 0)
         # -------------------------------------------------------  
        if self.vehObs == 0: 
            #self.obstacles_plus = self.obstacles.copy()
            self.obstacles_plus = copy.deepcopy(self.obstacles)
        elif self.vehObs == 1:
            states_plus = np.vstack((state[0:3,:], rVeh*np.ones((1,state.shape[1])))) 
            self.obstacles_plus = np.hstack((self.obstacles, states_plus))   