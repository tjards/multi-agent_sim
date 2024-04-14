#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:07:07 2023

@author: tjards
"""

# import stuff
# ------------
import numpy as np
import copy
#from utils import swarm_metrics # do I really need this module?
from scipy.spatial.distance import cdist
from utils import encirclement_tools as encircle_tools
from utils import lemni_tools
import random 

# agent dynamics
# --------------
dynamics = 'double integrator'
#dynamics = 'double integrator' 
    # 'double integrator' 
    # 'quadcopter'

nAgents = 5    # number of agents
rAgents = 0.5   # physics radius of agents 
iSpread = 35    # initial spread of agents

# store the config
config_agents = {'nAgents': nAgents , 'rAgents': rAgents, 'initial_spread': iSpread, 'dynamics': dynamics} 


# some dependencies for quadcopter
if dynamics == 'quadcopter':
    
    from quadcopter_module import config as quadcopter_config_file 
    from quadcopter_module.quad import Quadcopter
    from quadcopter_module.ctrl import Control as Quadcopter_Control
    quadcopter_config = quadcopter_config_file.config()

# hyper parameters
# ---------------

# note: move quadcopter low-level controller to quad files 
heading_type        = 2 #  0 = forward, 1 = random, 2 = centroid 
v_heading_adjust    = 0.10 # min speed at which quadcopter adjust heading
v_heading_saturate  = 0.50 # max speed at which quadcopter adjust heading

class Agents:
    
    def __init__(self,tactic_type):
        
        # initite attributes 
        # ------------------
        self.nVeh    =   nAgents      # number of vehicles
        self.rVeh    =   0.5     # physical radius of vehicle
        self.tactic_type = tactic_type    
                        # reynolds  = Reynolds flocking + Olfati-Saber obstacle
                        # saber     = Olfati-Saber flocking
                        # starling  = swarm like starlings 
                        # circle    = encirclement
                        # lemni     = dynamic lemniscates and other closed curves
                        # pinning   = pinning control
                        # shep      = shepherding
        self.dynamics_type = dynamics
        self.random_seeds = [random.uniform(0, 2*np.pi) for _ in range(self.nVeh)] # random seeds for each agent
        self.config_agents = config_agents             
           
        # Vehicles states
        # ---------------
        iSpread =  5      # initial spread of vehicles
        self.state = np.zeros((6,self.nVeh))
        self.state[0,:] = iSpread*(np.random.rand(1,self.nVeh)-0.5)                   # position (x)
        self.state[1,:] = iSpread*(np.random.rand(1,self.nVeh)-0.5)                   # position (y)
        self.state[2,:] = np.maximum((iSpread*np.random.rand(1,self.nVeh)-0.5),2)+8  # position (z)
        self.state[3,:] = 0*np.random.rand(1,self.nVeh)                                                       # velocity (vx)
        self.state[4,:] = 0*np.random.rand(1,self.nVeh)                                                       # velocity (vy)
        self.state[5,:] = 0*np.random.rand(1,self.nVeh)                                                      # velocity (vz)
        self.centroid = self.compute_centroid(self.state[0:3,:].transpose())
        self.centroid_v = self.compute_centroid(self.state[3:6,:].transpose())
        
        # Other Parameters
        # ----------------
        #self.params = np.zeros((4,self.nVeh))  # store dynamic parameters
        self.lemni                   = np.zeros([1, self.nVeh])
        
        # agent dynamics
        # --------------
        if dynamics == 'quadcopter':
            
            # quadcopter objects
            # ------------------
            self.quadList    = []
            self.llctrlList  = []
            self.sDesList    = []
            self.quads_headings = np.zeros((1,self.nVeh))
            
            # for each quadcopter
            for quad_i in range(0,self.nVeh):
                
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
        vals = np.unique(seps[np.where(seps!=0)])
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
    def spacing(self, states_q):
        
        # visibility radius
        radius = 1.5*5
        
        seps=cdist(states_q.transpose(), states_q.transpose())    
        vals = np.unique(seps[np.where(seps!=0)])
        vals_t = vals # even those out of range
        vals = np.unique(vals[np.where(vals<radius)])
        
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
    def evolve(self, Controller, t, Ts):
        
        # constraints
        #vmax = 1000
        #vmin = -1000

        if dynamics == 'quadcopter':
            
            # note: eventually we will move this into the quadcopter module
            
            # for each quadcopter
            for quad_i in range(0,self.nVeh):
                
                Ts_lapse = 0
                
                # define velocity setpoint (based on higher-level control inputs)
                # ------------------------
                self.sDesList[quad_i][3:6] =  10*Controller.cmd[0:3,quad_i]*Ts
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
                    if self.pin_matrix[quad_i, quad_i] == 1 and np.sum(self.pin_matrix) > 1:
                        
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
            self.state[3:6,:] = self.state[3:6,:] + Controller.cmd[:,:]*Ts
        
 
        self.centroid = self.compute_centroid(self.state[0:3,:].transpose())
        self.centroid_v = self.compute_centroid(self.state[3:6,:].transpose())
        
            #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, -vmax), vmax)
            #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, vmin), vmax)
            #state[3:6,:] = clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax)
            #state[3:6,:] = clamp_norm_min(clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax),vmin)
        
class Targets:

    def __init__(self, tspeed, nVeh):
        
        self.tSpeed  =   tspeed       # speed of target
        
        self.targets = 4*(np.random.rand(6,nVeh)-0.5)
        self.targets[0,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[1,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[2,:] = 15
        self.targets[3,:] = 0
        self.targets[4,:] = 0
        self.targets[5,:] = 0
        
        #self.trajectory = self.targets.copy()
        #self.trajectory = copy.deepcopy(self.targets)
        
    def evolve(self, t):
        
        self.targets[0,:] = 100*np.sin(self.tSpeed*t)                 # targets[0,:] + tSpeed*0.002
        self.targets[1,:] = 100*np.sin(self.tSpeed*t)*np.cos(self.tSpeed*t)  # targets[1,:] + tSpeed*0.005
        self.targets[2,:] = 100*np.sin(self.tSpeed*t)*np.sin(self.tSpeed*t)+15  # targets[2,:] + tSpeed*0.0005
        
class Obstacles:
    
    def __init__(self,tactic_type,nObs,targets):
        
        # note: don't let pass-in of walls yet, as it is a manual process still
        
        # initiate attributes
        # -------------------
        self.nObs    = nObs     # number of obstacles 
        self.vehObs  = 1     # include other vehicles as obstacles [0 = no, 1 = yes] 

        # if using reynolds, need make target an obstacle 
        if tactic_type == 'reynolds':
            self.targetObs = 1
        else:
            self.targetObs = 0   

        # there are no obstacle, but we need to make target an obstacle 
        if self.nObs == 0 and self.targetObs == 1:
            self.nObs = 1

        self.obstacles = np.zeros((4,self.nObs))
        oSpread = 20

        # manual (comment out if random)
        # obstacles[0,:] = 0    # position (x)
        # obstacles[1,:] = 0    # position (y)
        # obstacles[2,:] = 0    # position (z)
        # obstacles[3,:] = 0

        #random (comment this out if manual)
        if self.nObs != 0:
            self.obstacles[0,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[0,0]                   # position (x)
            self.obstacles[1,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[1,0]                   # position (y)
            self.obstacles[2,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[2,0]                  # position (z)
            #obstacles[2,:] = np.maximum(oSpread*(np.random.rand(1,nObs)-0.5),14)     # position (z)
            self.obstacles[3,:] = np.random.rand(1,self.nObs)+1                             # radii of obstacle(s)

        # manually make the first target an obstacle
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
            self.obstacles[3,0] = 2              # radii of obstacle(s)

        # Walls/Floors 
        # - these are defined manually as planes
        # --------------------------------------   
        self.nWalls = 1                      # default 1, as the ground is an obstacle 
        self.walls = np.zeros((6,self.nWalls)) 
        self.walls_plots = np.zeros((4,self.nWalls))

        # add the ground at z = 0:
        newWall0, newWall_plots0 = self.buildWall('horizontal', -2) 

        # load the ground into constraints   
        self.walls[:,0] = newWall0[:,0]
        self.walls_plots[:,0] = newWall_plots0[:,0]
        
        #self.obstacles_plus = self.obstacles.copy()
        self.obstacles_plus = copy.deepcopy(self.obstacles)
                
    def buildWall(self, wType, pos): 
        
        if wType == 'horizontal':
            
            # define 3 points on the plane (this one is horizontal)
            wallp1 = np.array([0, 0, pos])
            wallp2 = np.array([5, 10, pos])
            wallp3 = np.array([20, 30, pos+0.05])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                    
        if wType == 'vertical1':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([5, pos, 10])
            wallp3 = np.array([20,pos+0.05, 30])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
            
        if wType == 'vertical2':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos, 5, 10])
            wallp3 = np.array([pos+0.05, 20, 30])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                   
        if wType == 'diagonal1a':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([0, pos+5, 5])
            wallp3 = np.array([-5,pos+5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal1b':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([0, pos, 0])
            wallp2 = np.array([0, pos-5, 5])
            wallp3 = np.array([-5,pos-5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal2a':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos-5, 0, 5])
            wallp3 = np.array([pos-5, -5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
        
        if wType == 'diagonal2b':
            
            # define 3 points on the plane (this one is vertical
            wallp1 = np.array([pos, 0, 0])
            wallp2 = np.array([pos+5, 0, 5])
            wallp3 = np.array([pos+5, -5, 5])       
            # define two vectors on the plane
            v1 = wallp3 - wallp1
            v2 = wallp2 - wallp1
            # compute vector normal to the plane
            wallcp = np.cross(v1, v2)
            walla, wallb, wallc = wallcp
            walld = np.dot(wallcp, wallp3)
            walls = np.zeros((6,1)) 
            walls[0:3,0] = np.array(wallcp, ndmin=2)#.transpose()
            walls[3:6,0] = np.array(wallp1, ndmin=2)#.transpose()
            walls_plots = np.zeros((4,1))
            walls_plots[:,0] = np.array([walla, wallb, wallc, walld])
                    
        return walls, walls_plots
    
    def evolve(self, targets, state, rVeh):
        
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
            
         # Add other vehicles as obstacles (optional, default = 0)
         # -------------------------------------------------------  
        if self.vehObs == 0: 
            #self.obstacles_plus = self.obstacles.copy()
            self.obstacles_plus = copy.deepcopy(self.obstacles)
        elif self.vehObs == 1:
            states_plus = np.vstack((state[0:3,:], rVeh*np.ones((1,state.shape[1])))) 
            self.obstacles_plus = np.hstack((self.obstacles, states_plus))   
            
class History:
    
    # note: break out the Metrics stuff int another class 
    
    def __init__(self, Agents, Targets, Obstacles, Controller, Ts, Tf, Ti, f):
        
        nSteps = int(Tf/Ts+1)
        
        # initialize a bunch of storage 
        self.t_all               = np.zeros(nSteps)
        self.states_all          = np.zeros([nSteps, len(Agents.state), Agents.nVeh])
        self.cmds_all            = np.zeros([nSteps, len(Controller.cmd), Agents.nVeh])
        self.targets_all         = np.zeros([nSteps, len(Targets.targets), Agents.nVeh])
        self.obstacles_all       = np.zeros([nSteps, len(Obstacles.obstacles), Obstacles.nObs])
        self.centroid_all        = np.zeros([nSteps, len(Agents.centroid), 1])
        self.f_all               = np.ones(nSteps)
        self.lemni_all           = np.zeros([nSteps, Agents.nVeh])
        # metrics_order_all   = np.zeros((nSteps,7))
        # metrics_order       = np.zeros((1,7))
        nMetrics            = 12 # there are 11 positions being used.    
        self.metrics_order_all   = np.zeros((nSteps,nMetrics))
        self.metrics_order       = np.zeros((1,nMetrics))
        self.pins_all            = np.zeros([nSteps, Agents.nVeh, Agents.nVeh]) 
        # note: for pinning control, pins denote pins as a 1
        # also used in lemni to denote membership in swarm as 0
        
        self.swarm_prox = 0
        
        # if there are quadcopters
        if dynamics == 'quadcopter':
            # initialize a state to store all the quadcopter states 
            self.quads_states_all   = np.zeros([nSteps, 21, Agents.nVeh])
            self.quad_w_cmd_all     = np.zeros([nSteps,4, Agents.nVeh])
            self.quads_sDes_all     = np.zeros([nSteps, 21, Agents.nVeh])

        # store the initial conditions
        self.t_all[0]                = Ti
        self.states_all[0,:,:]       = Agents.state
        self.cmds_all[0,:,:]         = Controller.cmd
        self.targets_all[0,:,:]      = Targets.targets
        self.obstacles_all[0,:,:]    = Obstacles.obstacles
        self.centroid_all[0,:,:]     = Agents.centroid
        self.f_all[0]                = f
        self.metrics_order_all[0,:]  = self.metrics_order
        #self.lemni                   = np.zeros([1, Agents.nVeh])
        self.lemni_all[0,:]          = Agents.lemni
        self.pins_all[0,:,:]         = Controller.pin_matrix  
        
        # stores the desired lattice sizes 
        self.lattices            = np.zeros((nSteps,Agents.nVeh,Agents.nVeh))
        
        # store the walls
        self.walls_plots     = Obstacles.walls_plots
        
        
    def sigma_norm(self, z): 
        
        eps = 0.5
        norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
        return norm_sig

    def update(self, Agents, Targets, Obstacles, Controller, t, f, i):
        
        # core 
        self.t_all[i]                = t
        self.states_all[i,:,:]       = Agents.state
        self.cmds_all[i,:,:]         = Controller.cmd
        self.targets_all[i,:,:]      = Targets.targets
        self.obstacles_all[i,:,:]    = Obstacles.obstacles
        self.centroid_all[i,:,:]     = Agents.centroid
        self.f_all[i]                = f
        self.lemni_all[i,:]          = Agents.lemni
        self.pins_all[i,:,:]         = Controller.pin_matrix 
        
        # metrics
        self.metrics_order[0,0]      = Agents.order(Agents.state[3:6,:])
        self.metrics_order[0,1:7]    = Agents.separation(Agents.state[0:3,:],Targets.targets[0:3,:],Obstacles.obstacles)
        self.metrics_order[0,7:9]    = Agents.energy(Controller.cmd)
        #self.metrics_order[0,9:12]   = swarm_metrics.spacing(Agents.state[0:3,:])
        self.metrics_order_all[i,:]  = self.metrics_order
        self.swarm_prox              = self.sigma_norm(Agents.centroid.ravel()-Targets.targets[0:3,0])
        
        self.lattices[i,:,:]         = Controller.lattice
        
        # if there are quadcopters
        if dynamics == 'quadcopter': 
            # cycle through the quadcopters
            for q in range(0,Agents.nVeh):
                # and store them
                self.quads_states_all[i,:,q] = Agents.quadList[q].state 
                self.quad_w_cmd_all[i,:,q]   = Agents.llctrlList[q].w_cmd
                self.quads_sDes_all[i,:,q]   = Agents.sDesList[q]
   
class Trajectory:
    
    def __init__(self, Targets):
        
        self.trajectory = copy.deepcopy(Targets.targets)
    
    # WARNING: untested code
    def exclude(self, state, targets, lemni_all, exclusion):
        
        # [LEGACY] create a temp exlusionary set
        state_ = np.delete(state, [exclusion], axis = 1)
        targets_ = np.delete(targets, [exclusion], axis = 1)
        lemni_all_ = np.delete(lemni_all, [exclusion], axis = 1)
        
        return state_, targets_, lemni_all_
    
    # WARNING: untested code
    def unexclude(self, trajectory, targets, lemni, lemni_all, pins_all, i, exclusion):
        
        # [LEGACY] add exluded back in
        for ii in exclusion:
            trajectory = np.insert(trajectory,ii,targets[:,ii],axis = 1)
            trajectory[0:2,ii] = ii + 5 # just move away from the swarm
            trajectory[2,ii] = 15 + ii 
            lemni = np.insert(lemni,ii,lemni_all[i-1,ii],axis = 1)
            # label excluded as pins (for plotting colors only)
            pins_all[i-1,ii,ii] = 1  
            
            return trajectory, lemni, pins_all
    
    def update(self, Agents, Targets, History, t, i):
        
        #if flocking
        if Agents.tactic_type == 'reynolds' or Agents.tactic_type == 'saber' or Agents.tactic_type == 'starling' or Agents.tactic_type == 'pinning' or Agents.tactic_type == 'shep':
            self.trajectory = Targets.targets.copy() 
        
        # if encircling
        if Agents.tactic_type == 'circle':
            self.trajectory, _ = encircle_tools.encircle_target(Targets.targets, Agents.state)
        
        # if lemniscating
        elif Agents.tactic_type == 'lemni':
            self.trajectory, Agents.lemni = lemni_tools.lemni_target(History.lemni_all,Agents.state,Targets.targets,i,t)
        
        