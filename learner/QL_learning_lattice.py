#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:28:01 2023

This program implements reinforcement learning for application on
the multi agent simulator. 

@author: tjards


Dev notes:
    
    27 Dec 2023: need to increase exploit rate over time
    31 Dec 2023: need to increase Q-table dict to include states_q (maybe as a grid???)
    01 Jan 2024: consider removing the global case (code simplify), call local case multiple times if necessary 
    01 Jan 2024: other simplification: no state-grid option, just default a (0,0,0) state
    01 Jan 2024: Q table should be Agent, State, Neighbour, Action. State and Neighbour are swapped (oops). I dont think this matters much.
    08 Jul 2024: allow for more diverse reward calculations (i.e. k-connectivity)

"""

#%% import stuff
# ------------
import numpy as np
import random
import os
import json
import copy
from scipy.spatial import distance

#%% hyper parameters
# ----------------
options_range   = [4, 15]    # range of action options [min, max]
nOptions        = 2         # number of action options (evenly spaced between [min, max])
time_horizon    = 120       # how long to apply action and await reward (eg., 1 sec/0.02 sample per sec = 50)
time_horizon_v  = 0.1       # optional, max speed constraint to permit new action (higher makes more stable)
states_grid     = 1         # represent states in Q-table as a grid? 1 = yes, 0 = no

#%% data saving
# -------------
data_directory = 'data'
file_path = os.path.join(data_directory, "data_Q_learning.json")

# converts to dict to json'able
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_to_json_serializable(key): convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return str(obj)
    else:
        return obj

#%% define the q learning agent
# ----------------------------
class q_learning_agent:
    
    def __init__(self, nAgents):
        
        # learning hyperparameters
        self.nAgents        = nAgents
        self.nOptions       = nOptions # defined above
        self.action_options = {state: np.linspace(options_range[0], options_range[1], self.nOptions) for state in range(self.nAgents)}
        #self.explore_rate   = 1     # [0,1], 1 = always learn, 0 = always exploit best option
        self.explore_rate = 1*np.ones((nAgents)) # now each agent has its own
        self.learn_rate     = 0.3   # [0,1]
        self.discount       = 0.2 #0.8   # balance immediate/future rewards, (gamma): 0.8 to 0.99
        self.time_horizon   = time_horizon
        self.time_horizon_v = time_horizon_v
        self.explore_exp_decay = 0.05 #0.03 # [0.01 (slower decay), 0.1 (faster decay)]: a, where et = e0 * e^{-at}
        
        # initialize timers (global)
        self.time_count     = 0     # initialize 
        self.Q_update_count = 0     # initialize 
        
        # initialize timer (local)
        self.time_count_i = np.zeros((nAgents))
        self.Q_update_count_i = np.zeros((nAgents))
        
        # initialize data 
        self.data           = {}
        
        # initialize state
        if states_grid != 1: 
            self.state = {}
        else:
            state_init = np.array([0,0,0])
            self.state = state_init
            self.state_next = np.zeros((3,nAgents))
            self.state_next[0:3, :] = state_init.reshape(3, 1)
            
        # initialize action
        self.action         = {}
        for i in range(self.nAgents):
            self.action["Agent " + str(i)] = {}
            # search through each neighbour
            for j in range(self.nAgents):
                # not itself
                if i != j:
                    # select an action (randomly)
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
        
        self.action_next    = {}
        self.nState         = self.nAgents
        self.nAction        = self.nAgents * self.nOptions
        self.reward         = 0
        self.Q              = {}
        
        # initalize the Q-table       
        for i in range(self.nAgents):
            
            self.Q["Agent " + str(i)] = {}
            
            for j in range(self.nAgents):
                
                self.Q["Agent " + str(i)]["Neighbour " + str(j)] = {}
                
                # if not itself
                if i != j:
                    
                    if states_grid == 1:
                        
                        # insert an extra entry for the "state"
                        self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)] = {}
                    
                    # run through the options
                    for k in range(self.nOptions):
                        
                        option_label = self.action_options[i][k]
                        
                        if states_grid != 1:
                        
                            self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)] = 0
                            
                        else:
                            
                            #self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)] = {}
                            #self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(option_label)][tuple(self.state)] = 0 
                            
                            self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)]["Option " + str(option_label)]= 0 

    # orchestration of update
    # -----------------------
    def update_step(self, landmarks, targets, states_q, states_p, k_node, consensus_agent, **kwargs):
        
        learning_grid_size = kwargs.get('learning_grid_size')
        
         # increment the counter(s)
        self.time_count_i[k_node] += 1
    
         # if we are at the end of the horizon (and, optionally, not jumping all over the place)
        if self.time_count_i[k_node] > self.time_horizon and np.max(abs(states_p[:,k_node])) < self.time_horizon_v:
             
            # this sets landmark to origin, if none avail
            if landmarks.shape[1] == 0:
                landmarks = np.append(landmarks,np.zeros((4,1))).reshape(4,-1)
                print('no landmarks detected, using origin')
                
            # update the state (note, this is rounded to grid cubes)
            # ------------------------------------------------------
            self.state = np.around(states_q[0:3,k_node]-targets[0:3,k_node],learning_grid_size)
            self.state_next = np.around(states_q[0:3,:]-targets[0:3,:],learning_grid_size)
            #print("trial length for Agent ",k_node,": ", learning_agent.time_count_i[k_node])
            
            # learn the lattice parameters
            # ----------------------------
            self.compute_reward(np.reshape(states_q[:,k_node],(3,1)), landmarks)  # compute the reward
            self.update_q_table_i(consensus_agent, k_node)                             # update the Q-table
            self.select_action_i(k_node)                                          # select the next action
            self.match_parameters_i(consensus_agent, k_node)                           # assign the selected action
            self.time_count_i[k_node] = 0                                         # reset the counter
            self.update_exploit_rate(k_node)                                      # update exploit rate
            #print('REWARD, Agent', k_node, ": ", learning_agent.reward)
            #print(learning_agent.explore_rate)
         
         
     
     # %% adjust explore/exploit rate
     # ------------------------------
     
    def update_exploit_rate(self, i):
         
        #self.explore_rate = self.explore_rate * np.exp(- self.explore_rate*t)
        #self.explore_rate = self.explore_rate * np.exp(- (self.explore_exp_decay))
        self.explore_rate[i] = self.explore_rate[i] * np.exp(- self.explore_exp_decay)
        if self.explore_rate[i] < 0.01:
            self.explore_rate[i] = 0 
        print('explore rate for Agent ',i,': ',  self.explore_rate[i]) 
         
        

    # %% select an action 
    # ---------------------
                    
    # local case
    # ---------
    
    # def select_action_next_i(self, i):
        
    #     self.action_next["Agent " + str(i)] = {}
        
    #     for j in range(self.nAgents): 

    #         if i != j:
                
    #             if states_grid != 1:
                
    #                 temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                    
    #                 self.action_next["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))

    #             else: # later: this should be next state, not current state
                    
    #                 temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)].get)
                    
    #                 self.action_next["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))

    
    def select_action_i(self, i):
        
        if random.uniform(0, 1) < self.explore_rate[i]:
            
            #self.state["Agent " + str(i)] = i 
            self.action["Agent " + str(i)] = {}

            for j in range(self.nAgents):
                
                if i != j:
                    
                    self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = self.action_options[i][np.random.choice(self.nOptions)]
                    
        else:
            
            self.action["Agent " + str(i)] = {}
            
            for j in range(self.nAgents): 

                if i != j:
                    
                    if states_grid != 1:
                    
                        temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)].get)
                        
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))

                    else:
                        
                        temp = max(self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)], key=self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)].get)
                        
                        self.action["Agent " + str(i)]["Neighbour Action " + str(j)] = float(temp.replace("Option ",""))


    #%% compute reward
    # ---------------                    
    # this can be called for 1 or multiple agents. Just ensure to pass in the state for 1 x agent if the former.
    
    def compute_reward(self, states_q, landmarks):
        
        # initialize reward signal and temp helpers
        self.reward = 0
        summerizer = 0.0001
        
        # for each agent 
        for i in range(states_q.shape[1]):
            
            normalizer = 0.0001
            
            # cycle through landmarks
            for j in range(landmarks.shape[1]):
                
                # accumulate distances between agents and landmarks
                summerizer += np.linalg.norm(states_q[0:3,i]-landmarks[0:3,j])
                normalizer += 1
        
        # compute reward signal
        self.reward = states_q.shape[1]/np.divide(summerizer,normalizer) 
 
    #%% link to parameters used by controller
    # ---------------------------------------

    # local case 
    def match_parameters_i(self,paramClass, i): 
        
        # for each neighbour
        for j in range(len(self.action)):
            
            # load the neighbour action
            if i != j:
                
                paramClass.d_weighted[i, j] = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]        
               
    #%% update q-table
    # ----------------
    
    # local case
    def update_q_table_i(self, paramClass, i):
        
        #self.select_action_next_i(i)
        
        for j in range(self.nAgents):
            
            # if not itself 
            if i != j: 
            
                # update the q table with selected action
                selected_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)]
                
                # we will use this same action for the discounted future rewards, but from the neighbour's perspective
                future_option = self.action["Agent " + str(i)]["Neighbour Action " + str(j)] 
                   
                #self.state = ["Agent " + str(i), "Neighbour " + str(j)]
                #self.action = ["Option " + str(selected_option)]
                
                if states_grid != 1:
                    
                    # Q(s,a)
                    Q_current = self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] 
                
                    # Q(s+,a)
                    Q_future = self.Q["Agent " + str(j)]["Neighbour " + str(i)]["Option " + str(future_option)] 
                
                    #self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] += np.multiply(self.learn_rate, self.reward + self.discount*Q_future - Q_current)
                    self.Q["Agent " + str(i)]["Neighbour " + str(j)]["Option " + str(selected_option)] = (1 - self.learn_rate)*Q_current + self.learn_rate*(self.reward + self.discount*Q_future)
                
                # we'll have to define next state in here later
                else:
                    
                    # check if we've never visited this state before (this will be a unqique method later)
                    if tuple(self.state) not in self.Q["Agent " + str(i)]["Neighbour " + str(j)]:
                        
                        # add a new entry
                        self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)] = {}
                        
                        for k in range(self.nOptions):
                            
                            self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)]["Option " + str(self.action_options[i][k])] = 0
 
                        # since we have a new state, let's explore more
                        self.explore_rate[i]   = min(1, self.explore_rate[i] + 0.2)                    
 
                    # Q(s,a)
                    Q_current = self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)]["Option " + str(selected_option)] 
                
                    # if the neighbour has this state_next and it is in range
                    if tuple(self.state_next[:,j]) in self.Q["Agent " + str(j)]["Neighbour " + str(i)] and paramClass.prox_i[i,j] == 1:
                
                        # Q(s+,a)
                        #print('next state found')
                        Q_future = self.Q["Agent " + str(j)]["Neighbour " + str(i)][tuple(self.state_next[:,j])]["Option " + str(future_option)] # this needs to flip i/j eventually 

                    else:
                        
                        Q_future = 0
                            
                
                    self.Q["Agent " + str(i)]["Neighbour " + str(j)][tuple(self.state)]["Option " + str(selected_option)] = (1 - self.learn_rate)*Q_current + self.learn_rate*(self.reward + self.discount*Q_future)
                                                   
        self.Q_update_count += 1
        
        self.data[0] = copy.deepcopy(self.Q)
        
        if self.Q_update_count > 10*self.nAgents:
            
            #self.Q_update_count = 0
            
            data = convert_to_json_serializable(self.data)

            with open(file_path, 'w') as file:
                json.dump(data, file)
                

     