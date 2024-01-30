# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# Functions get_poly_cc, minSomethingTraj, pos_waypoint_min are derived from Peter Huang's work:
# https://github.com/hbd730/quadcopter-simulation
# author: Peter Huang
# email: hbd730@gmail.com
# license: BSD
# Please feel free to use and modify this, but keep the above information. Thanks!



import numpy as np
#from numpy import pi
#from numpy.linalg import norm
from waypoints import makeWaypoints
#import config

class Trajectory:

    def __init__(self, quad, ctrlType, trajSelect, config):

        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]

        t_wps, wps, y_wps, v_wp = makeWaypoints(config)
        self.t_wps = t_wps
        self.wps   = wps
        self.y_wps = y_wps
        self.v_wp  = v_wp
        self.desVel_lemni = np.zeros(3) # np.array([[0, 0, 0]]) 
        self.end_reached = 0

        if (self.ctrlType == "xyz_pos"):
            self.T_segment = np.diff(self.t_wps)
       
        if (self.yawType == "zero"):
            self.y_wps = np.zeros(len(self.t_wps))
        
        # Get initial heading
        self.current_heading = quad.psi
        
        # Initialize trajectory setpoint
        self.desPos = np.zeros(3)    # Desired position (x, y, z)
        self.desVel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.         # Desired yaw speed
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)

    def desiredState(self, t, Ts, quad):
        
        self.desPos = np.zeros(3)    # Desired position (x, y, z)
        self.desVel = np.zeros(3)    # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)    # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)    # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)    # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)    # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.         # Desired yaw speed
        
        def pos_waypoint_lemni():           
            
            self.desPos = self.wps[1,:] 
            self.desVel = self.desVel_lemni

        def pos_waypoint_timed():
            
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("Time array and waypoint array not the same size.")
            elif (np.diff(self.t_wps) <= 0).any():
                raise Exception("Time array isn't properly ordered.")  
            
            if (t == 0):
                self.t_idx = 0
            elif (t >= self.t_wps[-1]):
                self.t_idx = -1
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1
            
            self.desPos = self.wps[self.t_idx,:]
       
        def pos_waypoint_arrived():

            dist_consider_arrived = 0.2 # Distance to waypoint that is considered as "arrived"
            if (t == 0):
                self.t_idx = 0
                self.end_reached = 0
            elif not(self.end_reached):
                distance_to_next_wp = ((self.wps[self.t_idx,0]-quad.pos[0])**2 + (self.wps[self.t_idx,1]-quad.pos[1])**2 + (self.wps[self.t_idx,2]-quad.pos[2])**2)**(0.5)
                if (distance_to_next_wp < dist_consider_arrived):
                    self.t_idx += 1
                    if (self.t_idx >= len(self.wps[:,0])):    # if t_idx has reached the end of planned waypoints
                        self.end_reached = 1
                        self.t_idx = -1
                    
            self.desPos = self.wps[self.t_idx,:]

        def yaw_waypoint_timed():
            
            if not (len(self.t_wps) == len(self.y_wps)):
                raise Exception("Time array and waypoint array not the same size.")
            
            self.desEul[2] = self.y_wps[self.t_idx]
                    
        
        if (self.ctrlType == "xyz_pos"):

            if (self.xyzType == "hover"):
                pass 
              
            else:    
      
                if (self.xyzType == "waypoint"):
                    pos_waypoint_timed()
  
                elif (self.xyzType == "lemni"):
                    pos_waypoint_lemni()
                    
                if (self.yawType == "none"):
                    pass
                elif (self.yawType == "waypoint"):
                    yaw_waypoint_timed()

                self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR, self.desYawRate)).astype(float)
        
        return self.sDes



