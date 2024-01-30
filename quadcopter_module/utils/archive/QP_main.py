#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:38:09 2020

@author: tjards
"""
import QP
import numpy as np


#%% initial Setup
# ---------------
nTrials = 20
ro      = 0.1   # obstacle safety buffer
r       = 0.1   # vehicle safety buffer
rb      = 0.3   # addition buffer
rb = np.minimum(1-(ro+r),rb)    # total buffer must be < 1
# xv = np.array([-3.7, -4.7])   # vehicle position
# xo = np.array([-1, -1.5])     # obstacle position
# xt = np.array([0,0])          # target position 
xv = np.array([-3.7, -4.7, 1], ndmin = 2)   # vehicle position
xo = np.array([-1, -1.5, 0.9], ndmin = 2)     # obstacle position
#xo = np.array([[-1, -1.5, 0.9],[1, 0.9, 0.5]])     # obstacle position
xt = np.array([0, 0, 0.8], ndmin = 2)          # target position 
nStates=xv.shape[1]
nObs = xo.shape[0]

#%% update the constraints
# ------------------------

#initialize constraint stuff
#xp = np.zeros([nObs,nStates])
#A = np.zeros([nObs,nStates])
#b = np.zeros([nObs,1])
#d = np.zeros([nObs,1])


xp, d, A, b = QP.computeCons(xv, xo, r, ro, rb)

# for i in range(0,nObs):
#     #xp_, d_ = QP.computeXp(xv,xo[[i],:],r,ro,rb)
#     xp[i,:], d[i,0] = QP.computeXp(xv,xo[[i],:],r,ro,rb)
#     #xp[i,:] = xp_
#     #d[i,0] = d_
#     A[i,:] = xo[i,:] - xp[i,:]
#     b[i,0] = np.dot(xp[i,:],(xo[i,:]-xp[i,:]))
#xp, d = QP.computeXp(xv,xo,r,ro,rb) # compute constraint centerpoint 
#A = np.zeros([1,2])                 # inequalities
#A = np.zeros([1,nStates])                 # inequalities
#A[0,:] = xo - xp
#b = np.dot(xp,(xo-xp))
ux = xt                             # target setpoint
cx = xt                             # target setpoint 
nQP = nStates*6 + 1
myData_QP = np.zeros([nTrials, nQP]) # data store

# move the vehicles and/or obstacle around a bit
# ----------------------------------------------
for i in range(0,nTrials):

    # pseudo-random movement
    # ----------------------
    xv[0,0] = xv[0,0] + 0.1*i*0.2
    xv[0,1] = xv[0,1] + 0.3
    if i > 8:    
        xv[0,1] = xv[0,1] - 0.1
     
    # Update constraints    
    # ------------------
    #xp, d = QP.computeXp(xv,xo,r,ro,rb) 
    #A[0,:] = xo - xp
    #b = np.dot(xp,(xo-xp))
    # for j in range(0,nObs):
    #     #xp_, d_ = QP.computeXp(xv,xo[[i],:],r,ro,rb)
    #     xp[j,:], d[j,0]= QP.computeXp(xv,xo[[j],:],r,ro,rb)
    #     #xp[i,:] = xp_
    #     #d[i,0] = d_
    #     A[j,:] = xo[j,:] - xp[j,:]
    #     b[j,0] = np.dot(xp[j,:],(xo[j,:]-xp[j,:]))
    
    # move target
    # ----------
    cx = QP.moveTarget(xv, xo, xt, r, ro, rb)
    
    # # compute constraints
    # # --------------------
    # xp, d, A, b = QP.computeCons(xv, xo, r, ro, rb)

    # # find new optimal point 
    # # ----------------------
    # cx = QP.runOpt(xv, xt, xp, A, b)
    ux = cx

    # update data store
    # -----------------
    myData_QP[i,0:nStates] = np.array(xv.ravel())
    myData_QP[i,nStates:2*nStates] = np.array(xo.ravel())
    myData_QP[i,2*nStates:3*nStates] = np.array(xt.ravel())
    myData_QP[i,3*nStates:4*nStates] = np.array(A)
    myData_QP[i,4*nStates:4*nStates+1] = np.array(b)
    myData_QP[i,4*nStates+1:5*nStates+1] = np.array(ux['x'][:])    # unconstrained case (delete later to save time)
    myData_QP[i,5*nStates+1:6*nStates+1] = np.array(cx['x'][:])   # constrained case
    
    # we would feed these new points into here
    # ---> feed cx here
 
QP.myAnimation(myData_QP,nStates,'xy')

# # do a little hack (with schmanzy slices) to show show altitudes :)_
# myData_QP_z = myData_QP  
# myData_QP_z[:,[1,2]] = myData_QP_z[:,[2,1]] 
# myData_QP_z[:,[4,5]] = myData_QP_z[:,[5,4]]
# myData_QP_z[:,[7,8]] = myData_QP_z[:,[8,7]]
# myData_QP_z[:,[14,15]] = myData_QP_z[:,[15,14]]  
# myData_QP_z[:,[17,18]] = myData_QP_z[:,[18,17]]      
    
# QP.myAnimation(myData_QP,nStates,'xz')

