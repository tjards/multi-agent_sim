# -*- coding: utf-8 -*-
"""
updated by: P. Travis Jardine, PhD
email: travis.jardine@gmail.com 

original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import utils.quaternions as quat
import utils.lemni_tools as lemni_tools

class config():

    
    def __init__(self):

        # Select Orientation of Quadcopter and Reference Frame
        # ---------------------------
        # "NED" for front-right-down (frd) and North-East-Down
        # "ENU" for front-left-up (flu) and East-North-Up
        self.orient = "NED"
        
        # Select whether to use gyroscopic precession of the rotors in the quadcopter dynamics
        # ---------------------------
        # Set to False if rotor inertia isn't known (gyro precession has negigeable effect on drone dynamics)
        self.usePrecession = bool(False)
        
        # Simulation Setup
        # --------------------------- 
        self.Ti         =   0
        self.Ts         =   0.005 #default 0.005 (larger numbers could result in instability)
        self.Tf         =   10 #1500 # 26
        self.ifsave     =   1
        self.ifsavedata =   0     # subset of ifsave
        self.trialLen   =   60
        self.wpType     =   0   # [0 = fixed, 1 = random, 2 = TBD , 3 = learning (linked to doLearn below)] 
        self.nVeh       =   100     
        self.nSteps     =   int(self.Tf/self.Ts+1)
        
        
        # Choose trajectory settings
        # --------------------------- 
        # options
        ctrlOptions     = ["xyz_pos"]
        trajOptions_xyz = ["hover","waypoint","lemni"]
        trajOptions_psi = ["zero", "waypoint", "none"]
        
        # for the controller 
        self.ctrlType   = ctrlOptions[0]   
        
        # for the trajectory selector 
        self.trajSelect     = ["lemni","zero","waypoint"] 
        self.trajSelect[0]  = trajOptions_xyz[2]     # follow a lemniscatic trajectory
        self.trajSelect[1]  = trajOptions_psi[0]     # yaw pointed at zero
        self.trajSelect[2]  = "waypoint"             # legacy (keep as placeholder)
        
        # Choose Obstacle Avoidance settings
        # ----------------------------------
        self.PIC    = 0     # do we want to using planar inequality constraint shifting?
        self.PF     = 0     # do we want to use potential fields?
        
        # Create obstacle(s)
        # -----------------------------------------
        self.o1 = 0*np.array([-2.1, 0, -3.5],)         # obstacle 1 (x,y,z)
        self.o2 = 0*np.array([2, -1.2, 0.9])           # obstacle 2 (x,y,z)
        self.o3 = 0*np.array([0, 2.5, -2.5])           # obstacle 3 (x,y,z)
        
        # Learning stuff
        # ---------------
        self.doLearn = 0
        #self.nParams = 14       # moved lower based on the control architecture 
        self.nOptions = 7
        self.optionsInterval = [0.1,5] 
        self.wpRange = 1
        self.learnRate = 0.1
        self.a = 1               # weight of positive reinforcement (default one)
        self.b = 0               # weight of negative reinforcement (default zero)
        self.learnWhat = [0, 0, 0, 1]
        #                [1 = pos (2), 1 = vel (6), 1 = att (2), 1 = rate (4)]   
        self.nParams = 2*self.learnWhat[0] + 6*self.learnWhat[1] + 2*self.learnWhat[2] + 4*self.learnWhat[3] 
        if self.doLearn == 1:
            self.wpType = 3     # wp type must be 3 for learning (see waypoints.py)
    
        # Lemniscate stuff
        # ----------------
        self.tactic_type = 'lemni'  
        self.r_desired = 5                                   # desired radius of encirclement [m]
        self.ref_plane = 'horizontal'                        # defines reference plane (default horizontal)
        self.phi_dot_d = 0.2 #0.12 #0.12                                # how fast to encircle
        self.unit_lem = np.array([1,0,0]).reshape((3,1))     # sets twist orientation (i.e. orientation of lemniscate along x)
        self.lemni_type = 2                                  # 0 = surv, 1 = rolling, 2 = mobbing (reverse this order for quadcopters)
        self.quat_0 = quat.e2q(np.array([0,0,0]))           # if lemniscate, this has to be all zeros (consider expanding later to rotate the whole swarm)
        self.quat_0_ = quat.quatjugate(self.quat_0)               # used to untwist                               
        self.lemni = np.zeros([1, self.nVeh])
        self.twist_perp = lemni_tools.enforce(self.ref_plane, self.tactic_type, self.quat_0)
        self.lemni_all = np.zeros([self.nSteps, self.nVeh])
        self.lemni_all[0,:] = self.lemni
     
        # enforce
        if self.tactic_type == 'lemni':
            if self.ctrlType != "xyz_pos":
                print('warning: ctrlOptions must be set to xyz_pos for lemni to work')
            if self.trajSelect[0] != "lemni":
                print('warning: trajSelect must be set to lemni for lemni to work')
            

