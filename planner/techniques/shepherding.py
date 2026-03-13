#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:00:12 2023

@author: tjards

Refs:
    https://royalsocietypublishing.org/doi/10.1098/rsos.230015

Description of parameters:

    nShepherds = 5  # number of shepherds (just herding = 0)
    
    # for herding
    r_R = 3         # repulsion radius
    r_O = 5         # orientation radius
    r_A = 7         # attraction radius (r_R < r_O < r_A)
    r_I = 6.5       # agent interaction radius (nominally, slighly < r_A)
    a_R = 2         # gain,repulsion 
    a_O = 2        # gain orientation 
    a_A = 2         # gain, attraction 
    a_I = 4         # gain, agent interaction 
    a_V = 2         # gain, laziness (desire to stop)

    # for shepherding 
    r_S     = r_I - 1           # desired radius from herd
    r_Oi    = 3                 # range to view obstacles (here, nearest shepherd or shared point)
    r_Od    = 2                 # desired distance from obtacles 
    r_Or    = 1                 # radius of shepherd (uniform for all agents, for now)
    a_N     = 5                 # gain, navigation
    a_R_s   = 1                 # gain, shepards repel eachother
    a_R_s_v = 2*np.sqrt(a_R_s)  # gain, shepherds repel eachther (velo component)
    a_V_s   = 1 #np.sqrt(a_N)    # gain, laziness (desire to stop)

    # type of shepherding 
    type_shepherd = 'haver'
        #   'haver         = traditional approach to shepherding

    # type of collision avoidance for shepherds
    type_avoid = 'ref_point'
        #   'ref_shepherd' = maintains rO_d from nearest shepherd
        #   'ref_point'    = (prefered) maintains rO_d from desired location between herd and inv-target 

    # use heterogeneous strategies for capturing?
    #capture     = 1         # 0 = no, 1 = yes
    #r_c         = r_Oi      # range at which to consider breaking from neighbours
    #nNeighbours = 2         # criteria to break out (n number nearby)

    # bias unique to each
    #k_noise = 0.1
    #noise   = np.random.uniform(-1, 1, (nShepherds,3))

    # adjustments
    cmd_adjust = 0.02

Dev notes:

    Note: investigate "heterogeneous control strategies"
     - loan wolf, actually goes around the other side to catch/trap the herd
     - how to decide? what criteria? Maybe, if the network gets too big


"""


#%% import stuff
# ------------
import numpy as np
from scipy.spatial.distance import cdist
import copy
import os
import json
import config.config as cfg


#%% obstacle avoidance helpers (shepherds)
# ---------------------------------------
eps = 0.1
h   = 0.9
pi  = 3.141592653589793

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b

def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

#%% define overall class
# -----------------

from planner.base import BasePlanner
class Planner(BasePlanner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

#class Planner:
    
#    def __init__(self, config, state):
        
        state = kwargs.get('states', None)
        self.state    = state
        shep_config = cfg.get_config(config, 'planner.techniques.shepherding')
        self.nShepherds = shep_config.get('nShepherds', 3)

        agents_config = cfg.get_config(config, 'agents')
        self.nAgents  = agents_config.get('nAgents', None)

        # Herd Parameters 
        self.nHerd      = self.nAgents - self.nShepherds
        self.r_R = shep_config.get('r_R', 3)      # repulsion radius
        self.r_O = shep_config.get('r_O', 5)      # orientation radius
        self.r_A = shep_config.get('r_A', 7)      # attraction radius
        self.r_I = shep_config.get('r_I', 6.5)    # agent interaction radius
        self.a_R = shep_config.get('a_R', 2)      # repulsion gain
        self.a_O = shep_config.get('a_O', 2)      # orientation gain
        self.a_A = shep_config.get('a_A', 2)      # attraction gain
        self.a_I = shep_config.get('a_I', 4)      # interaction gain
        self.a_V = shep_config.get('a_V', 2)      # laziness (stop) gain
        
        # Shepherd Parameters
        self.r_S = shep_config.get('r_S', self.r_I - 1)     # desired radius from herd
        self.r_Oi = shep_config.get('r_Oi', 3)              # range to view obstacles
        self.r_Od = shep_config.get('r_Od', 2)              # desired distance from obstacles
        self.r_Or = shep_config.get('r_Or', 1)              # radius of shepherd
        self.a_N = shep_config.get('a_N', 5)                        # navigation gain
        self.a_R_s = shep_config.get('a_R_s', 1)                    # shepherd repulsion gain
        self.a_R_s_v = shep_config.get('a_R_s_v', 2*np.sqrt(self.a_R_s))  # shepherd velocity repulsion
        self.a_V_s = shep_config.get('a_V_s', 1)                    # shepherd laziness gain
        
        # Other Parameters
        self.type_shepherd = shep_config.get('type_shepherd', 'haver')
        self.type_avoid = shep_config.get('type_avoid', 'ref_point')
        self.cmd_adjust = shep_config.get('cmd_adjust', 0.02)
        self.spawned = False

        # instantiate the herd and shepherds
        self.build_index()
        self.herd       = self.Herd(self)
        self.shepherds  = self.Shepherds(self) 
    
    #def spawn(self):

        # compute distances between all
        self.compute_seps()
        
        # indices 
        self.i = 0 # current agent
        self.j = 0 # neighbour being explored
        
        # cmd adjustment (based on sample time, later, import this)
        self.cmd = np.zeros((1,3))

        # graph parameters (standardized in base class)
        self.sensor_range_matrix = self.r_A * np.ones((self.nAgents, self.nAgents))
        self.connection_range_matrix = self.r_S * np.ones((self.nAgents, self.nAgents))


    # define separation
    # ------------------ 
    def compute_seps(self):      
        
        self.seps_all = np.zeros((self.state.shape[1],self.state.shape[1]))
        i = 0
        while (i<self.state.shape[1]):
            self.seps_all[i:self.state.shape[1],i]=cdist(self.state[0:3,i].reshape(1,3), self.state[0:3,i:self.state.shape[1]].transpose())
            i+=1
        self.seps_all = self.seps_all + self.seps_all.transpose()
            
    
    # build an index distinguishing shepards from herd (1 = s, 0 = h)
    # --------------------------------------------------------------
    def build_index(self):

        # check to ensure herd is big enough
        #if self.nShepherds > (self.state.shape[1]-1):
        if self.nShepherds > (self.nAgents-1):
            raise ValueError("there needs to be at least one member in the herd ")
                   
        # random, for now (later, based on conditions)
        self.index = list(np.concatenate((np.ones(self.nShepherds, dtype=int), np.zeros(self.nHerd, dtype=int))))
    
        
    # compute commands (called from outside)
    # ----------------
    #def compute_cmd(self, Targets, i):
    #def compute_cmd(self, targets, i):
    def compute_cmd(self, states, targets, index, **kwargs):
        
        #if not self.spawned:
        #    self.state = state
        #    self.spawn()
        #    self.spawned = True 

        # store the agent being examined
        i = index
        self.i = i
        
        # store the targets
        #self.targets = Targets.targets[0:3,:]
        self.targets = targets[0:3,:]
        
        # compute the separations
        self.compute_seps()
        
        # compute command, if herd member
        if self.index[self.i] == 0:
    
            self.herd.compute_cmd(self)
        
        # compute command, if shepherd
        elif self.index[self.i] == 1:
            
            self.shepherds.compute_cmd(self)
        
        self.cmd = self.cmd_adjust*self.cmd

        return self.cmd     

    #%% define the herd
    # ---------------        
    class Herd():
    
        def __init__(self, outer):
            
            # radial parameters
            self.r_R = outer.r_R           # repulsion radius
            self.r_O = outer.r_O           # orientation radius
            self.r_A = outer.r_A           # attraction radius
            self.r_I = outer.r_I           # agent interaction radius
            
            # gain parameters
            self.a_R = outer.a_R           # repulsion gain
            self.a_O = outer.a_O           # orientation gain
            self.a_A = outer.a_A           # attraction gain
            self.a_I = outer.a_I           # interaction gain
            self.a_V = outer.a_V           # laziness gain

            # indices
            self.indices = [k for k, m in enumerate(outer.index) if m == 0]
    
        # compute herd commands
        # ---------------------
        def compute_cmd(self, outer):
            
            # initialize
            outer.cmd = np.zeros((1,3))
            outer.j = 0
            
            # urge to stop moving
            outer.cmd += self.a_V * (-outer.state[3:6,outer.i]) 
            
            # search through all agents
            while (outer.j < (outer.nHerd + outer.nShepherds)):
                
                # ignore self
                if outer.seps_all[outer.i,outer.j] > 0.000001:
                    
                    # urge to stop moving
                    #outer.cmd += self.a_V * (-outer.state[3:6,outer.i])                   
                    
                    # if it's a member of the herd
                    if outer.index[outer.j] == 0:
                    
                        # repulsion
                        if outer.seps_all[outer.i,outer.j] < self.r_R:
                            outer.cmd -= self.a_R * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                               
                        # orientation
                        if outer.seps_all[outer.i,outer.j] < self.r_O:
                            outer.cmd += self.a_O * np.divide(outer.state[3:6,outer.j]-outer.state[3:6,outer.i],np.linalg.norm(outer.state[3:6,outer.j]-outer.state[3:6,outer.i]))
                        
                        # attraction
                        if outer.seps_all[outer.i,outer.j] < self.r_A:
                            outer.cmd += self.a_A * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                    
                    # if it's a shepherd        
                    elif outer.index[outer.j] == 1:
                    
                        # shepherd influence
                        if outer.seps_all[outer.i,outer.j] < self.r_I:
                            outer.cmd -= self.a_I * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                
                # increment to next neighbour            
                outer.j+=1
            
    #%% define the shepherds
    # --------------------
    class Shepherds():
        
        def __init__(self, outer):
            
            # radial parameters
            self.r_S = outer.r_S           # desired radius from herd
            self.r_Oi = outer.r_Oi         # range to view obstacles
            self.r_Od = outer.r_Od         # desired distance from obstacles
            self.r_Or = outer.r_Or         # radius of shepherd
            
            # gain parameters
            self.a_N = outer.a_N           # navigation gain
            self.a_R_s = outer.a_R_s       # shepherd repulsion gain
            self.a_R_s_v = outer.a_R_s_v   # shepherd velocity repulsion
            self.a_V_s = outer.a_V_s       # shepherd laziness gain

            # indices
            self.indices = [k for k, m in enumerate(outer.index) if m == 1]

        # compute herd commands
        # ---------------------
        def compute_cmd(self, outer):
                
            # initialize
            outer.cmd = np.zeros((3,1))
            
            # pull out the distance from neighbours
            seps_list = list(outer.seps_all[outer.i,:])
            
            # make all the shepherds negative 
            for k in self.indices:
                seps_list[k] = -seps_list[k]

            # find the closest member of the herd
            closest_herd = seps_list.index(min(k for k in seps_list if k > 0))
    
            # compute the normalized vector between closest in herd and target 
            v = np.divide(outer.state[0:3,closest_herd]-outer.targets[0:3,outer.i],np.linalg.norm(outer.state[0:3,closest_herd]-outer.targets[0:3,outer.i])) 
     
            # compute the desired location to shepherd (based on closets hearder)
            q_s = outer.state[0:3,closest_herd] + self.r_S*v  # location
            d_s = np.linalg.norm(q_s-outer.state[0:3,outer.i]) # distance
            
            # find the closest shepherd
            closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
            q_cs = outer.state[0:3,closest_shepherd]                # location of closest shepherd
            d_cs = np.linalg.norm(q_cs-outer.state[0:3,outer.i])    # distance from that closest shepherd
            
            # if using havermaet technique
            # ----------------------------
            if outer.type_shepherd == 'haver':
            
                # navigate to push the herd towards targets
                outer.cmd = self.a_N * np.divide(q_s-outer.state[0:3,outer.i],np.linalg.norm(q_s-outer.state[0:3,outer.i]))
                
                # urge to slow down
                outer.cmd += self.a_V_s * (-outer.state[3:6,outer.i])
                
                # if the closet shepherd is within avoidance range
                if d_cs < self.r_Oi:
                    
                    # avoid the shepherd
                    if outer.type_avoid == 'ref_shepherd':
                    
                        # intermediate terms
                        bold_a_k = np.array(np.divide(outer.state[0:3,outer.i]-q_cs,d_cs), ndmin = 2)
                        P = np.identity(outer.state[3:6,:].shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                        mu = np.divide(self.r_Or,d_cs) 
                        p_ik = mu*np.dot(P,outer.state[3:6,outer.i]) 
                        q_ik = mu*outer.state[0:3,outer.i]+(1-mu)*q_cs
                                 
                        # compute command
                        outer.cmd += self.a_R_s*phi_b(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*n_ij(outer.state[0:3,outer.i], q_ik) + self.a_R_s_v*b_ik(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*(p_ik - outer.state[3:6,outer.i])
                 
                    # avoid the reference point (ends up working nicely)
                    elif outer.type_avoid == 'ref_point':
                        
                        # intermediate terms
                        bold_a_k = np.array(np.divide(outer.state[0:3,outer.i]-q_s,d_s), ndmin = 2)
                        P = np.identity(outer.state[3:6,:].shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                        mu = np.divide(self.r_Or,d_s) 
                        p_ik = mu*np.dot(P,outer.state[3:6,outer.i]) 
                        q_ik = mu*outer.state[0:3,outer.i]+(1-mu)*q_s
                             
                        # compute command
                        outer.cmd += self.a_R_s*phi_b(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*n_ij(outer.state[0:3,outer.i], q_ik) + self.a_R_s_v*b_ik(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*(p_ik - outer.state[3:6,outer.i])
