#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:02:04 2020

@author: tjards
"""

import numpy as np


class quadata:

    def __init__(self, quad, traj, ctrl, fala, Ti, numTimeStep):
        
        # Initialize Result Matrixes
        # ---------------------------
        #numTimeStep = int(Tf/Ts+1)
    
        self.t_all          = np.zeros(numTimeStep)
        self.s_all          = np.zeros([numTimeStep, len(quad.state)])
        self.pos_all        = np.zeros([numTimeStep, len(quad.pos)])
        self.vel_all        = np.zeros([numTimeStep, len(quad.vel)])
        self.quat_all       = np.zeros([numTimeStep, len(quad.quat)])
        self.omega_all      = np.zeros([numTimeStep, len(quad.omega)])
        self.euler_all      = np.zeros([numTimeStep, len(quad.euler)])
        self.sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
        self.sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
        self.w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
        self.wMotor_all     = np.zeros([numTimeStep, len(quad.wMotor)])
        self.thr_all        = np.zeros([numTimeStep, len(quad.thr)])
        self.tor_all        = np.zeros([numTimeStep, len(quad.tor)])
        # learning items
        #falaError_all  = np.zeros([numTimeStep, len(fala.error_accumulated)])
        self.falaError_all  = np.zeros([numTimeStep, 1])
        
    
        self.t_all[0]            = Ti
        self.s_all[0,:]          = quad.state
        self.pos_all[0,:]        = quad.pos
        self.vel_all[0,:]        = quad.vel
        self.quat_all[0,:]       = quad.quat
        self.omega_all[0,:]      = quad.omega
        self.euler_all[0,:]      = quad.euler
        self.sDes_traj_all[0,:]  = traj.sDes
        self.sDes_calc_all[0,:]  = ctrl.sDesCalc
        self.w_cmd_all[0,:]      = ctrl.w_cmd
        self.wMotor_all[0,:]     = quad.wMotor
        self.thr_all[0,:]        = quad.thr
        self.tor_all[0,:]        = quad.tor
        #learning items
        self.falaError_all[0,:]  = fala.error_accumulated
    
    
    
    
    def collect(self, t, quad, traj, ctrl, fala, i):
                
        # print("{:.3f}".format(t))
        self.t_all[i]             = t
        self.s_all[i,:]           = quad.state
        self.pos_all[i,:]         = quad.pos
        self.vel_all[i,:]         = quad.vel
        self.quat_all[i,:]        = quad.quat
        self.omega_all[i,:]       = quad.omega
        self.euler_all[i,:]       = quad.euler
        self.sDes_traj_all[i,:]   = traj.sDes
        self.sDes_calc_all[i,:]   = ctrl.sDesCalc
        self.w_cmd_all[i,:]       = ctrl.w_cmd
        self.wMotor_all[i,:]      = quad.wMotor
        self.thr_all[i,:]         = quad.thr
        self.tor_all[i,:]         = quad.tor
        # learning items
        self.falaError_all[i,:]  = fala.error_accumulated
        
        #print('collecting...')