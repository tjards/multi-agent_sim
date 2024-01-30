#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:23:23 2020

@author: tjards
"""
import numpy as np


def clamp_norm(v, n_max):
    vx, vy, vz = v
    n = np.sqrt(vx**2 + vy**2 + vz**2)
    f = np.minimum(n, n_max) / n
    return [f * vx, f * vy, f * vz]

def clamp_norm_min(v, n_min):
    vx, vy, vz = v
    n = np.sqrt(vx**2 + vy**2 + vz**2)
    f = np.maximum(n, n_min) / n
    return [f * vx, f * vy, f * vz]


def evolve_sat(Ts, state, cmd):
    
    # constraints
    vmax = 100
    vmin = 10

    #discretized doubple integrator 
    state[0:3,:] = state[0:3,:] + state[3:6,:]*Ts
    #state[3:6,:] = state[3:6,:] + cmd[:,:]*Ts
    #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, -vmax), vmax)
    #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, vmin), vmax)
    #state[3:6,:] = clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax)
    state[3:6,:] = clamp_norm_min(clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax),vmin)
    
    return state

def evolve(Ts, state, cmd):
    
    # constraints
    #vmax = 1000
    #vmin = -1000

    #discretized doubple integrator 
    state[0:3,:] = state[0:3,:] + state[3:6,:]*Ts
    state[3:6,:] = state[3:6,:] + cmd[:,:]*Ts
    #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, -vmax), vmax)
    #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, vmin), vmax)
    #state[3:6,:] = clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax)
    #state[3:6,:] = clamp_norm_min(clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax),vmin)
    
    return state






# %% Old

# # Define dynamics
# # ---------------
# def state_dot(t, state, cmd):
    
#     dynDot = np.array([
#         [state[3]],
#         [state[4]],
#         [state[5]],
#         [cmd[0]],
#         [cmd[1]],
#         [cmd[2]]])
    
#     dstate = np.zeros(6)
#     dstate[0] = dynDot[0]
#     dstate[1] = dynDot[1]
#     dstate[2] = dynDot[2]
#     dstate[3] = dynDot[3]
#     dstate[4] = dynDot[4]
#     dstate[5] = dynDot[5]
    
#     return dstate

# # Set integrator
# # -------------
# integrator = ode(state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
# integrator.set_initial_value(state, Ti)