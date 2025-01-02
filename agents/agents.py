#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:07:07 2023

@author: tjards
"""

# import stuff
# ------------
import numpy as np
from scipy.spatial.distance import cdist
import random 

# agent dynamics
# --------------
#dynamics = 'quadcopter'
dynamics = 'double integrator' 
    # 'double integrator' 
    # 'quadcopter'

nAgents = 7    # number of agents
rAgents = 0.5   # physical radius of agents 
iSpread = 40   # initial spread of agents
init_conditions = 'evenly_spaced'   # mesh, random, evenly_spaced

# store the config
config_agents = {'nAgents': nAgents , 'rAgents': rAgents, 'initial_spread': iSpread, 'dynamics': dynamics} 

# some dependencies for quadcopter
if dynamics == 'quadcopter':
    
    from .quadcopter_module import config as quadcopter_config_file 
    from .quadcopter_module.quad import Quadcopter
    from .quadcopter_module.ctrl import Control as Quadcopter_Control
    quadcopter_config = quadcopter_config_file.config()
    config_agents['quad_orient'] = quadcopter_config.orient
    config_agents['quad_usePrecession'] = quadcopter_config.usePrecession
    config_agents['quad_Ts'] = quadcopter_config.Ts
    from .quadcopter_module.ctrl import return_gains as quadcopter_gains
    config_agents.update(quadcopter_gains())
    config_agents['heading_type']        = quadcopter_config.heading_type 
    config_agents['v_heading_adjust']    = quadcopter_config.v_heading_adjust
    config_agents['v_heading_saturate']  = quadcopter_config.v_heading_saturate
    heading_type = config_agents['heading_type']        
    v_heading_adjust = config_agents['v_heading_adjust']    
    v_heading_saturate = config_agents['v_heading_saturate']  
    
    from .quadcopter_module.initQuad import sys_params
    quad_params = sys_params(quadcopter_config)
    for key, value in quad_params.items():
        if isinstance(value, np.ndarray):
            quad_params[key] = value.tolist()
    config_agents['quad_params'] = quad_params
    

class Agents:
    
    def __init__(self,tactic_type, dimens):
        
        # initite attributes 
        # ------------------
        self.nAgents        = nAgents      # number of vehicles
        self.rVeh           = rAgents     # physical radius of vehicle
        self.tactic_type    = tactic_type    
        self.dynamics_type  = dynamics
        self.random_seeds   = [random.uniform(0, 2*np.pi) for _ in range(self.nAgents)] # random seeds for each agent
        self.dimens         = dimens
        
        config_agents.update({'tactic_type': self.tactic_type})
        
        # Vehicles states
        # ---------------
        
        d_sep = 7
        
        if init_conditions == 'evenly_spaced':
            self.state = np.zeros((6, self.nAgents))
    
            # Calculate the number of rows and columns required
            grid_size = int(np.ceil(np.sqrt(self.nAgents)))
            
            # Generate a grid of positions
            x_coords = np.arange(0, grid_size * d_sep, d_sep)
            y_coords = np.arange(0, grid_size * d_sep, d_sep)
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            
            # Flatten the grid and take the first nAgents positions
            positions = np.vstack((x_grid.flatten(), y_grid.flatten())).T[:self.nAgents]
            
            self.state[0, :] = positions[:, 0]  # position (x)
            self.state[1, :] = positions[:, 1]  # position (y)
            
            # Assign z positions (if applicable)
            self.state[2, :] = iSpread * np.random.rand(1, self.nAgents) + 15  # position (z)
            if self.dimens == 2:
                self.state[2, :] = 0 * self.state[2, :]
            
            # Random velocities
            self.state[3, :] = 0.1 * np.random.rand(1, self.nAgents)  # velocity (vx)
            self.state[4, :] = 0.1 * np.random.rand(1, self.nAgents)  # velocity (vy)
            self.state[5, :] = 0.1 * np.random.rand(1, self.nAgents)  # velocity (vz)
            if self.dimens == 2:
                self.state[5, :] = 0 * self.state[5, :]
            
            # Compute centroid and velocity centroid
            self.centroid = self.compute_centroid(self.state[0:3, :].transpose())
            self.centroid_v = self.compute_centroid(self.state[3:6, :].transpose())

        
        
        
        if init_conditions == 'random':
        
            self.state = np.zeros((6,self.nAgents))
            self.state[0,:] = iSpread*(np.random.rand(1,self.nAgents)-0.5)                   # position (x)
            self.state[1,:] = iSpread*(np.random.rand(1,self.nAgents)-0.5)                   # position (y)
            #self.state[2,:] = np.maximum((iSpread*np.random.rand(1,self.nAgents)-0.5),2)+8  # position (z)
            self.state[2,:] = iSpread*np.random.rand(1,self.nAgents) + 15   # position (z)
            if self.dimens == 2:
                self.state[2,:] = 0*self.state[2,:]
                
            self.state[3,:] = 0.1*np.random.rand(1,self.nAgents)                                                       # velocity (vx)
            self.state[4,:] = 0.1*np.random.rand(1,self.nAgents)                                                       # velocity (vy)
            self.state[5,:] = 0.1*np.random.rand(1,self.nAgents)
            if self.dimens == 2:
                self.state[5,:] = 0*self.state[5,:]
            
            # velocity (vz)
            self.centroid = self.compute_centroid(self.state[0:3,:].transpose())
            self.centroid_v = self.compute_centroid(self.state[3:6,:].transpose())
        
        # Vehicle states(mesh)
        # --------------
        mesh_distance = iSpread
        
        if init_conditions == 'mesh':

            side_length = int(np.ceil(nAgents ** (1/3)))
            x_vals, y_vals, z_vals = np.meshgrid(mesh_distance * np.arange(side_length), 
                                          mesh_distance * np.arange(side_length),
                                          mesh_distance * np.arange(side_length))
            x_vals = x_vals.flatten()[:nAgents]
            y_vals = y_vals.flatten()[:nAgents]
            z_vals = z_vals.flatten()[:nAgents]
            if self.dimens == 2:
                z_vals = 0*z_vals
                
            self.state = np.zeros((6,self.nAgents))
            self.state[0,:] = x_vals                   # position (x)
            self.state[1,:] = y_vals                    # position (y)
            #self.state[2,:] = np.maximum((iSpread*np.random.rand(1,self.nAgents)-0.5),2)+8  # position (z)
            self.state[2,:] = z_vals    # position (z)
            self.state[3,:] = 0*np.random.rand(1,self.nAgents)                                                       # velocity (vx)
            self.state[4,:] = 0*np.random.rand(1,self.nAgents)                                                       # velocity (vy)
            self.state[5,:] = 0*np.random.rand(1,self.nAgents)                                                      # velocity (vz)
            self.centroid = self.compute_centroid(self.state[0:3,:].transpose())
            self.centroid_v = self.compute_centroid(self.state[3:6,:].transpose())
        
        # agent dynamics
        # --------------
        if dynamics == 'quadcopter':
            
            # quadcopter objects
            # ------------------
            self.quadList    = []
            self.llctrlList  = []
            self.sDesList    = []
            self.quads_headings = np.zeros((1,self.nAgents))
            
            # for each quadcopter
            for quad_i in range(0,self.nAgents):
                
                # instantiate a new quadcopter
                self.quadList.append(Quadcopter(quadcopter_config))
                
                # align states with corresponding agent
                self.quadList[quad_i].state[0:3]    = self.state[0:3,quad_i]
                self.quadList[quad_i].state[7:10]   = self.state[3:6,quad_i]
                #self.quadList[quad_i].state[9]      = -self.state[5,quad_i]
        
                # save the headings
                self.quads_headings[0,quad_i] = self.quadList[quad_i].phi
        
                # low-level controllers
                # ---------------------
                self.sDesList.append(np.zeros(21))
                self.llctrlList.append(Quadcopter_Control(self.quadList[quad_i], "yaw")) # nominally, "yaw" at end
                self.llctrlList[quad_i].controller(self.quadList[quad_i], self.sDesList[quad_i], quadcopter_config.Ts)
        
        self.config_agents = config_agents    
    
    def compute_centroid(self, points):
        length = points.shape[0]
        sum_x = np.sum(points[:, 0])
        sum_y = np.sum(points[:, 1])
        sum_z = np.sum(points[:, 2])
        centroid = np.array((sum_x/length, sum_y/length, sum_z/length), ndmin = 2)
        return centroid.transpose() 
    
    # order
    # -----
    def order(self, states_p):

        order = 0
        N = states_p.shape[1]
        # if more than 1 agent
        if N > 1:
            # for each vehicle/node in the network
            for k_node in range(states_p.shape[1]):
                # inspect each neighbour
                for k_neigh in range(states_p.shape[1]):
                    # except for itself
                    if k_node != k_neigh:
                        # and start summing the order quantity
                        norm_i = np.linalg.norm(states_p[:,k_node])
                        if norm_i != 0:
                            order += np.divide(np.dot(states_p[:,k_node],states_p[:,k_neigh]),norm_i**2)
                # average
                order = np.divide(order,N*(N-1))
                
        return order

    # separation
    # ----------
    def separation(self, states_q,target_q,obstacles):
        
        # distance from targets or agents
        # ---------------------
        # note: replace target_q with states_q to get separation between agents
        #seps=cdist(states_q.transpose(), np.reshape(target_q[:,0],(-1,1)).transpose())
        seps=cdist(states_q.transpose(), states_q.transpose())    
        #vals = np.unique(seps[np.where(seps!=0)])
        vals = seps[np.where(seps>0.5)]
        means = np.mean(vals)
        varis = np.var(vals)
        maxes = np.max(vals)
        mines = np.min(vals)
        
        #print(mines)
        
        # distance from obstacles
        # -----------------------
        if obstacles.shape[1] != 0:
            seps_obs=cdist(states_q.transpose(), obstacles[0:3,:].transpose()) - obstacles[3,:] # this last part is the radius of the obstacle
            means_obs = np.mean(seps_obs) 
            varis_obs = np.var(seps_obs)
        else:
            means_obs = 0
            varis_obs = 0
        
        return means, varis, means_obs, varis_obs, maxes, mines

    # spacing (between agents)
    # -----------------------
    def spacing(self, states_q, radius):
        
        # visibility radius
        #radius = Controller.d_init + 0.5
        #radius= 5


        seps=cdist(states_q.transpose(), states_q.transpose())    
        #vals = np.unique(seps[np.where(seps!=0)])
        vals = seps[np.where(seps>0.5)]
        vals_t = vals # even those out of range
        #vals = np.unique(vals[np.where(vals<radius)])
        vals = vals[np.where(vals<radius)]
        
        # if empty, return zero
        if len(vals) == 0:
            vals = np.array([0])
        
        return vals.mean(), len(vals), vals_t.mean()
        
    # energy
    # ------
    def energy(self,cmd):
        
        energy_total = np.sqrt(np.sum(cmd**2))
        energy_var =  np.var(np.sqrt((cmd**2)))
           
        return energy_total, energy_var

    # evolve through agent dynamics
    # -----------------------------    
    def evolve(self, cmd, pin_matrix, t, Ts):
        
        # constraints
        vmax = 10
        vmin = -10

        if dynamics == 'quadcopter':
            
            # note: eventually we will move this into the quadcopter module
            
            # for each quadcopter
            for quad_i in range(0,self.nAgents):
                
                Ts_lapse = 0
                
                # define velocity setpoint (based on higher-level control inputs)
                # ------------------------
                self.sDesList[quad_i][3:6] =  10*cmd[0:3,quad_i]*Ts
                #self.sDesList[quad_i][5]   =  10*Controller.cmd[2,quad_i]*Ts
                
                # define yaw  
                # ----------
                # save the headings
                self.quads_headings[0,quad_i] = self.quadList[quad_i].phi
                
                # if pointing in direction of movement 
                if heading_type == 0:
                    
                    # define unit vector ahead
                    normv = np.maximum(np.sqrt(self.quadList[quad_i].state[7]**2 + self.quadList[quad_i].state[8]**2 + self.quadList[quad_i].state[9]**2),0.00001)
                    tarv = 1*np.divide(self.quadList[quad_i].state[7:10],normv)
                    
                    #print(normv)

                    # compute corresponding heading
                    heading = np.arctan2(tarv[1],tarv[0])
                    # load as setpoint (if we're moving fast enough)
                    if normv > v_heading_adjust and normv < v_heading_saturate:
                        self.sDesList[quad_i][14] = heading
                    else:
                        self.sDesList[quad_i][14] = self.sDesList[quad_i][14]
                
                # if using random number
                elif heading_type == 1:
                    
                    # set to the random seed
                    self.sDesList[quad_i][14] = self.random_seeds[quad_i]
                    
                    # if it is a pin, but not the only pin 
                    #if self.pin_matrix[quad_i, quad_i] == 1 and np.sum(self.pin_matrix) > 1:
                    if pin_matrix[quad_i, quad_i] == 1 and np.sum(pin_matrix) > 1:
                        
                        # increment the random seed slowly
                        self.random_seeds[quad_i] += 0*Ts
                    
                # if heading to centroid
                elif heading_type == 2:
                    
                    # vector towards centroid
                    v_centroid = self.centroid[0:3].ravel() - self.quadList[quad_i].state[0:3].ravel()
                    normv = np.maximum(np.sqrt(self.quadList[quad_i].state[7]**2 + self.quadList[quad_i].state[8]**2 + self.quadList[quad_i].state[9]**2),0.00001)
                    heading = np.arctan2(v_centroid[1],v_centroid[0])
                    
                    if normv > v_heading_adjust and normv < v_heading_saturate:
                        self.sDesList[quad_i][14] = heading
                    else:
                        self.sDesList[quad_i][14] = self.sDesList[quad_i][14]

                # all else, set to zero
                else:
                    self.sDesList[quad_i][14] = 0
                    
                # send to low-level controller
                # ----------------------------
                
                # low-level controller runs at different (faster) frequency than outer
                while Ts_lapse < Ts:
                
                    # update the state
                    self.quadList[quad_i].update(t+Ts_lapse, quadcopter_config.Ts, self.llctrlList[quad_i].w_cmd)
                
                    # update the low-level control signal
                    self.llctrlList[quad_i].controller(self.quadList[quad_i], self.sDesList[quad_i], quadcopter_config.Ts)
                    #self.quad_w_cmd  = self.llctrlList[quad_i].w_cmd
                    
                    Ts_lapse += quadcopter_config.Ts
                    
                # align states with corresponding agent
                self.state[0:3,quad_i] = self.quadList[quad_i].state[0:3]
                self.state[3:6,quad_i] = self.quadList[quad_i].state[7:10]

        else:

            #discretized double integrator 
            self.state[0:3,:] = self.state[0:3,:] + self.state[3:6,:]*Ts
            self.state[3:6,:] = self.state[3:6,:] + cmd[:,:]*Ts
            
            #clip
            self.state[3:6,:] = np.clip(self.state[3:6,:], vmin, vmax)
            #self.state[3:6,:] = clip_vector_magnitude(self.state[3:6,:], vmin, vmax)
            
            
            
        
 
        self.centroid = self.compute_centroid(self.state[0:3,:].transpose())
        self.centroid_v = self.compute_centroid(self.state[3:6,:].transpose())
        
            #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, -vmax), vmax)
            #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, vmin), vmax)
            #state[3:6,:] = clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax)
            #state[3:6,:] = clamp_norm_min(clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax),vmin)
        
 