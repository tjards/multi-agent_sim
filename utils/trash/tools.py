#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:29:59 2020

This file defines some useful functions 

@author: tjards
"""
import numpy as np 

eps = 0.5

def centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    sum_z = np.sum(points[:, 2])
    centroid = np.array((sum_x/length, sum_y/length, sum_z/length), ndmin = 2)
    return centroid.transpose() 

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

def buildWall(wType, pos): 
    
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