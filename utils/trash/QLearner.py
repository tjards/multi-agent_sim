#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:00:39 2021

@author: tjards
"""
import numpy as np
import random

# Setup
# ------
nState = 2
nAction = 2
explore_rate = 0.2
learn_rate = 0.9    # alpha/lambda
discount = 0.8      # balance immediate/future rewards, (gamma): 0.8 to 0.99
Ti = 0
Tf = 10
Ts = 0.5


# Initialize Q table
# ------------------
Q = np.zeros((nState, nAction))


# Run Simulation
# --------------
t = Ti
state = 0

while round(t,3) < Tf:

    # Select
    # ------
    if random.uniform(0, 1) < explore_rate:
        # Explore (select randomly)        
        action = 0        
    else:
        # Exploit (select best)
        action = np.argmax(Q[state,:]) # not complete
    
    # Get new state, reward
    # ---------------------
    next_state = 0 # new_state, reward = environment(action) 
    reward = 0
                
    # Update Q table
    # ---------------
    next_action = np.argmax(Q[next_state,:])
    Q[state, action] += np.multiply(learn_rate, reward + discount*Q[next_state, next_action] - Q[state, action])
    
    # Increment 
    # -----------------------------------
    state = next_state 
    t += Ts
    
    
#%% old stuff
    #Q[state, action] += learn_rate * (reward + discount * np.max(Q[next_state, :]) — Q[state, action])