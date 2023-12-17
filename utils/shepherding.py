#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:00:12 2023

@author: tjards

Refs:
    https://royalsocietypublishing.org/doi/10.1098/rsos.230015


Dev notes:



"""

# Note: investigate "heterogeneous control strategies"
# - loan wolf, actually goes around the other side to catch/trap the herd
# - how to decide? what criteria? Maybe, if the network gets too big


#%% import stuff
# ------------
import numpy as np
from scipy.spatial.distance import cdist
import copy

#%% hyperparameters
# -----------------
nShepherds = 5  # number of shepherds (just herding = 0)

# for herding
r_R = 2         # repulsion radius
r_O = 3         # orientation radius
r_A = 4         # attraction radius (r_R < r_O < r_A)
r_I = 5.5       # agent interaction radius (nominally, slighly < r_A)

a_R = 2         # gain,repulsion 
a_O = 1         # gain orientation 
a_A = 1         # gain, attraction 
a_I = 4         # gain, agent interaction 
a_V = 5         # gain, laziness (desire to stop)

# for shepherding 
r_S     = r_I - 1           # desired radius from herd
r_Oi    = 3                 # range to view obstacles (here, nearest shepherd)
r_Od    = 1                 # desired distance from obtacles 
r_Or    = 0.5               # radius of shepherd (uniform for all agents, for now)

a_N     = 5                 # gain, navigation
a_R_s   = 1                 # gain, shepards repel eachother
a_R_s_v = 1*np.sqrt(a_R_s)  # gain, shepherds repel eachther (velo component)
a_V_s   = 1*np.sqrt(a_N)    # gain, laziness (desire to stop)

# type of shepherding 
type_shepherd = 'haver'
    #   'haver         = traditional approach to shepherding

# type of collision avoidance for shepherds
type_avoid = 'ref_point'
    #   'ref_shepherd' = maintains rO_d from nearest shepherd
    #   'ref_point'    = (prefered) maintains rO_d from desired location between herd and inv-target 

# use heterogeneous strategies for capturing?
capture     = 1         # 0 = no, 1 = yes
r_c         = r_Oi      # range at which to consider breaking from neighbours
nNeighbours = 2         # criteria to break out (n number nearby)

# bias unique to each
#k_noise = 0.1
#noise   = np.random.uniform(-1, 1, (nShepherds,3))

# adjustments
cmd_adjust = 0.02

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
class Shepherding:
    
    def __init__(self, state):
        
        self.nShepherds = nShepherds
        self.nHerd      = state.shape[1] - nShepherds
        
        # states of all
        self.state    = state
        
        # discern shepherds from herd
        self.build_index()
        #self.distinguish() # legacy, no longer used
        
        # instantiate the herd and shepherds
        self.herd       = self.Herd(self)
        self.shepherds  = self.Shepherds(self) 
        
        # compute distances between all
        self.compute_seps()
        
        # indices 
        self.i = 0 # current agent
        self.j = 0 # neighbour being explored
        
        # cmd adjustment (based on sample time, later, import this)
        #self.cmd_adjust = 0.02
        self.cmd = np.zeros((1,3))
        
        # store hyper params
        # ------------------
        self.type_shepherd = type_shepherd
        self.type_avoid = type_avoid 
             
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
        if self.nShepherds > (self.state.shape[1]-1):
            raise ValueError("there needs to be at least one member in the herd ")
                   
        # random, for now (later, based on conditions)
        self.index = list(np.concatenate((np.ones(self.nShepherds, dtype=int), np.zeros(self.nHerd, dtype=int))))
    
    # compute commands (called from outside)
    # ----------------
    def compute_cmd(self, Targets, i):
        
        # store the agent being examined
        self.i = i
        
        # store the targets
        self.targets = Targets.targets[0:3,:]
        
        # compute the separations
        self.compute_seps()
        
        # compute command, if herd member
        if self.index[self.i] == 0:
    
            self.herd.compute_cmd(self)
        
        # compute command, if shepherd
        elif self.index[self.i] == 1:
            
            self.shepherds.compute_cmd(self)
        
        self.cmd = cmd_adjust*self.cmd     

    #%% define the herd
    # ---------------        
    class Herd():
    
        def __init__(self, outer):
            
            # radial parameters
            self.r_R = r_R         # repulsion radius
            self.r_O = r_O         # orientation radius
            self.r_A = r_A         # attraction radius (r_R < r_O < r_A)
            self.r_I = r_I         # agent interaction radius (nominally, slighly < r_A)
    
            # gain parameters 
            self.a_R = a_R         # gain,repulsion 
            self.a_O = a_O         # gain orientation 
            self.a_A = a_A         # gain, attraction 
            self.a_I = a_I         # gain, agent interaction 
            self.a_V = a_V         # gain, laziness (desire to stop)
            
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
            self.r_S     = r_S      # desired radius from herd
            self.r_Oi    = r_Oi     # range to view obstacles (here, nearest shepherd)
            self.r_Od    = r_Od     # desired distance from obtacles 
            self.r_Or    = r_Or     # radius of shepherd (uniform for all agents, for now)
    
            # gain parameters
            self.a_N     = a_N      # gain, navigation
            self.a_R_s   = a_R_s    # gain, shepards repel eachother
            self.a_R_s_v = a_R_s_v  # gain, shepherds repel eachther (velo component)
            self.a_V_s   = a_V_s    # gain, laziness (desire to stop)

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

#%% LEGACY
            
    # # separate the shepherds from the herd (legacy - not used)
    # # -----------------------------------
    # def distinguish(self):
        
    #     # initiate
    #     # --------
    #     #self.state_shep_i = np.zeros((self.state_i.shape[0],self.nShepherds))
    #     i_s = 0
    #     #self.state_herd_i = np.zeros((self.state_i.shape[0],self.state_i.shape[1]-self.nShepherds))
    #     i_h = 0
        
    #     # distinguish between shepherds and herd
    #     # -------------------------------------
    #     for i in range(0,self.state.shape[1]):
            
    #         # shepherds
    #         if self.index[i] == 1:
    #             self.state_shep_i[:,i_s] = self.state[:,i]
    #             i_s += 1
    #         # herd
    #         else:
    #             self.state_herd_i[:,i_h] = self.state[:,i]
    #             i_h += 1      
                      
#%% More LEGACY
               
# # build an index distinguishing shepards from herd (1 = s, 0 = h)
# # --------------------------------------------------------------
# def build_index(nShepherds, state):

#     # check to ensure herd is big enough
#     # ---------------------------------
#     if nShepherds > (state.shape[1]-1):
#         raise ValueError("there needs to be at least one member in the herd ")
        
#     # compute size of herd
#     # --------------------
#     nHerd = state.shape[1] - nShepherds
    
#     # random, for now (later, based on conditions)
#     # ---------------
#     index = np.concatenate((np.ones(nShepherds, dtype=int), np.zeros(nHerd, dtype=int)))
#     # Shuffle to distribute 1's and 0's randomly
#     #np.random.shuffle(index)
    
#     return list(index)

# # separate the shepherds from the herd (not used)
# # -----------------------------------
# def distinguish(state, nShepherds, index):
    
#     # initiate
#     # --------
#     shepherds = np.zeros((state.shape[0],nShepherds))
#     i_s = 0
#     herd = np.zeros((state.shape[0],state.shape[1]-nShepherds))
#     i_h = 0
    
#     # distinguish between shepherds and herd
#     # -------------------------------------
#     for i in range(0,state.shape[1]):
        
#         # shepherds
#         if index[i] == 1:
#             shepherds[:,i_s] = state[:,i]
#             i_s += 1
#         # herd
#         else:
#             herd[:,i_h] = state[:,i]
#             i_h += 1    
    
#     return shepherds, herd

# # define separation
# # ------------------ 
# def compute_seps(state):
#     seps_all = np.zeros((state.shape[1],state.shape[1]))
#     i = 0
#     while (i<state.shape[1]):
#         seps_all[i:state.shape[1],i]=cdist(state[0:3,i].reshape(1,3), state[0:3,i:state.shape[1]].transpose())
#         i+=1
    
#     seps_all = seps_all + seps_all.transpose()
        
#     return seps_all


# # compute command - herd
# # ----------------------------
# def compute_cmd_herd(states_q, states_p, i, distinguish, seps_all):
    
#     # initialize
#     # -----------
#     #seps_all = compute_seps(states_q)
#     motion_vector = np.zeros((3,states_q.shape[1]))
    
#     # search through each agent
#     j = 0
#     while (j < states_q.shape[1]):
        
#         # but not itself
#         if i != j:
            
#             # pull distance
#             dist = seps_all[i,j]
#             #print(dist)
            
#             # I could nest these, given certain radial constraints
#             # ... but I won't, deliberately, for now (enforce above, then come back later)
#             #print(i)
            
#             # urge to stop moving
#             motion_vector[:,i] += a_V * (-states_p[:,i])
              
#             # repulsion
#             if dist < r_R and distinguish[j] == 0:
#                 motion_vector[:,i] -= a_R * np.divide(states_q[:,j]-states_q[:,i],dist)
                   
#             # orientation
#             if dist < r_O and distinguish[j] == 0:
#                 motion_vector[:,i] += a_O * np.divide(states_p[:,j]-states_p[:,i],np.linalg.norm(states_p[:,j]-states_p[:,i]))
            
#             # attraction
#             if dist < r_A and distinguish[j] == 0:
#                 motion_vector[:,i] += a_A * np.divide(states_q[:,j]-states_q[:,i],dist)
                
#             # shepherd influence
#             if dist < r_I and distinguish[j] == 1:
#                 motion_vector[:,i] -= a_I * np.divide(states_q[:,j]-states_q[:,i],dist)
                
#         j+=1
    
#     return motion_vector[:,i] 

# # this is for finding n closest shepherds
# # ---------------------------------------
# def find_n_neighbours(n, sepslist):
    
#     neighbours = []
    
#     for _ in range(0,n):
        
#         # find the min value 
#         select = sepslist.index(max(k for k in sepslist if k < 0))
        
#         # add to list 
#         neighbours.append(select)
        
#         # exclude this index for next round
#         sepslist[select] = float('inf')
        
#     return neighbours
       
# # compute commands - sheperd
# # -------------------------
# def compute_cmd_shep(targets, centroid, states_q, states_p, i, distinguish, seps_list):
    
#     # initialize
#     cmd = np.zeros((3,states_q.shape[1]))
    
#     # find the indices for the shepherds
#     indices_shep = [k for k, m in enumerate(distinguish) if m == 1]
    

    
#     # make them negative
#     for k in indices_shep:
#         seps_list[k] = -seps_list[k]
     
        
     
#     # find the closest herd
#     closest_herd        = seps_list.index(min(k for k in seps_list if k > 0))

    
    
#     # compute the normalized vector between closest in herd and target 
#     v = np.divide(states_q[:,closest_herd]-targets[:,i],np.linalg.norm(states_q[:,closest_herd]-targets[:,i])) 
     
    
#     # compute the desired location to shepard (based on closets hearder)
#     q_s = states_q[:,closest_herd] + r_S*v  # location
#     d_s = np.linalg.norm(q_s-states_q[:,i]) # distance
    
#     # find the closest shepherd
#     closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
    
#     q_cs = states_q[:,closest_shepherd]         # location of closest shepherd
#     d_cs = np.linalg.norm(q_cs-states_q[:,i])   # distance from that closest shepherd
    
    

#     # if using capturing, check if criteria met
#     # if capture == 1:
        
#     #     # find n neighbours
#     #     neighbours              = find_n_neighbours(nNeighbours, copy.deepcopy(seps_list))
#     #     neighbours_distances    = [seps_list[i] for i in neighbours]
        
#     #     # if they are all close enough
#     #     if all(k > -r_c for k in neighbours_distances):
#     #         print('note: need to objectify this module')
#     #         print('because I want this agent to break away')

       
#     # if using havermaet technique
#     # ----------------------------
#     if type_shepherd == 'haver':
    
#         # navigate to push the herd towards targets
#         # -----------------------------------------
#         cmd = a_N * np.divide(q_s-states_q[:,i],np.linalg.norm(q_s-states_q[:,i]))
        
#         # urge to slow down
#         # ----------------
#         cmd += a_V_s * (-states_p[:,i])
        
#         # if the closet shepherd is within avoidance range
#         if d_cs < r_Oi:
            
#             # avoid the shepherd
#             if type_avoid == 'ref_shepherd':
            
#                 bold_a_k = np.array(np.divide(states_q[:,i]-q_cs,d_cs), ndmin = 2)
#                 P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
#                 mu = np.divide(r_Or,d_cs) 
#                 p_ik = mu*np.dot(P,states_p[:,i]) 
#                 q_ik = mu*states_q[:,i]+(1-mu)*q_cs
                            
#                 cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])
         
#             # avoid the reference point (ends up working nicely)
#             elif type_avoid == 'ref_point':
                
#                 bold_a_k = np.array(np.divide(states_q[:,i]-q_s,d_s), ndmin = 2)
#                 P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
#                 mu = np.divide(r_Or,d_s) 
#                 p_ik = mu*np.dot(P,states_p[:,i]) 
#                 q_ik = mu*states_q[:,i]+(1-mu)*q_s
                            
#                 cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])
  
#     return cmd #+ noise[i,:]
  

# # old one
# # -------  
# def compute_cmd(targets, centroid, states_q, states_p, i):
    
#     # compute distances between all
#     # -----------------------------
#     seps_all = compute_seps(states_q)
    
#     # discern shepherds from herd
#     # ---------------------------
#     distinguish = build_index(nShepherds, states_q)
    
#     # if it is a member of the herd
#     # ----------------------------
#     if distinguish[i] == 0:
    
#         # do the herd stuff
#         # -----------------
#         cmd = compute_cmd_herd(states_q, states_p, i, distinguish, seps_all)
    
#     else:
        
#         # do the shepherd stuff
#         # ----------------------
#         cmd =  compute_cmd_shep(targets,centroid, states_q, states_p, i, distinguish, list(seps_all[i,:]))   
    
#     return cmd*0.02, distinguish[i] #note, this is Ts, because output of above is velo, model is double integrator
    

   