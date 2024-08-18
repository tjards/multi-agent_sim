#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module implements some useful tools for pinning control 

Preliminaries:
    - Let us consider V nodes (vertices, agents)
    - Define E is a set of edges (links) as the set of ordered pairs
    from the Cartesian Product V x V, E = {(a,b) | a /in V and b /in V}
    - Then we consider Graph, G = {V,E} (nodes and edges)
    - G is simple: (a,a) not \in E \forall a \in V 
    - G is undirected: (a,b) \in E <=> (b,a) \in E
    - Nodes i,j are neighbours if they share an edge, (i,j) /in E
    - d1=|N_1| is the degree of Node 1, or, the number of neighbours

Created on Tue Dec 20 13:32:11 2022

Some related work:
    
    https://arxiv.org/pdf/1611.06485.pdf
    http://kth.diva-portal.org/smash/get/diva2:681041/FULLTEXT01.pdf
    https://ieeexplore-ieee-org.proxy.queensu.ca/stamp/stamp.jsp?tp=&arnumber=6762966
    https://ieeexplore-ieee-org.proxy.queensu.ca/document/9275901

@author: tjards

Dev notes:
    
    01 Apr 2023 - build up cap/exp to compare ctrb grammian vs degree vs betweeness driver node
        in the context of autonmous assembly of the swarm (i.e. components)
    01 Apr 2023 - should components be drawn to eachother or target?
     - maybe a component is a pin as well? yes! The pins are drawn to meta-pins.
    03 Apr 2023 - add heuristic to the betweenness Djikstra for moving towards other components during merge?
    03 Apr 2023 - focus on low degree centrality nodes as drivers? low betweenness? hmm... invert typical logic
    08 Apr 2023 - there is a lot of inefficiency in below, move the A,D,G calcs outside the loops
    10 Apr 2023 - RL to select lattice size that maximizes connection, Reward = nConnections/(1-pins)^2(size)^2. With obs around target. 
    20 Dec 2023 - I am addressing the heterogeneous lattice size problem, using an I-controller
    20 Dec 2023 - above didn't work. instead, used a consensus-based approach. I want to adjust tgis 
            to have different distances for each agent
    21 Dec 2023 - hetero worx, cleaning up
    27 Dec 2023 - integrated Q-learning for landmark coverage 
    27 Dec 2023 - betweenness is finickly with small component sizes... don't use for now
    11 Feb 2024 - got RL working for quadcopter. now add sensor bearing constraint
    11 Feb 2024 - orientation of quadcopter needs to be imported for range test 
    (do only for quadcopter dynamics). Also, graph is no longer uniform, 
    because some may not see due to aperature
    15 Feb 2024 - FOS to statistically select dominant neighbours?
    16 Feb 2024 - bearing messes up the whole graph thing... need to revisit. Maybe, just over-ride it as a pin if it has noting in view.  
    22 Feb 2024 - pinning_RL_tools is replacing pinning_tools . I need to get the bearing info better. Pass in quadcopter actual, not just v. v is too choppy.
    27 Feb 2024 - need to pass in actual heading to pinning_tools (problematic, as is a level above quadcopter/vehicle type and assume double integrator)
    28 Feb 2024 - lattice working with aperatures, but now there are collisions *face palm*
    28 Feb 2024 - I made all agents also obstacles and that seeme dto fix it (anaologous to a collision avoidance system)
                - rg needs to vary as well 
    04 Mar 2024 - rg now updating in graph_tools_direction. Consider renaming '_variable'.
                - need to assess how the Qlearning is affected by the bearing-based RL. consider using paramClass.prox_i[i,j] in update_Q_table * this was already done
    05 Mar 2024 - ok, I think this works with RL. produced some initial results.  
    03 Jul 2024 - cleaning up the graphical representation. Moved graph up a level, feed into pinning. Should pin selection be part of the graph module or in here?
    08 Jul 2024 - add a RL feature to maximize k-connectivity: resilient self-assembly in contested swarms via learned k-connectivity 
    
"""

#%% Import stuff
# --------------
import numpy as np
import random
import os
import json

#%% simulation setup
# ------------------

# learning parameters
hetero_lattice      = 0     # support heterogeneous lattice size? 1 = yes (Consensus), 0 = no
learning            = 0     # requires heterolattice, do we want to learn lattice size? 1 = yes (QL), 0 = no
learning_grid_size  = -1    # grid size for learning (nominally, -1, for 10 units x 10 units)
       
# learning requires heterolattice
if learning == 1 and hetero_lattice != 1:
    print('Warning: learning lattice requires hetero lattice enabled to find local consensus. Enforcing.')
    hetero_lattice = 1
    
#%% Hyperparameters
# -----------------

# key ranges 
d                   = 5            # lattice scale, > d_min > 5 (desired distance between agents) note: gets overridden by RL.
r                   = 1.3*d         # range at which neighbours can be sensed 
d_prime             = 1             # desired separation from obstacles  
r_prime             = 1.3*d_prime   # range at which obstacles can be sensed
d_min               = 5             # floor on lattice scale (always 5)
#rg                  = d + 0.5       # [legacy] range for graph analysis (nominally, d + small number), this will auto adjust later

# gains
c1_a = 1               # cohesion
c2_a = 2*np.sqrt(1)
c1_b = 1             # obstacles 
c2_b = 2*np.sqrt(1)
c1_g = 2               # tracking (for the pins)
c2_g = 2*np.sqrt(5)

# constants for useful functions
a   = 5
b   = 5
c   = np.divide(np.abs(a-b),np.sqrt(4*a*b)) 
eps = 0.1
h   = 0.2 # 0.2 for lattice, for obs this should be 0.9
pi  = 3.141592653589793

#%% save configs
# --------------
config = {}
with open(os.path.join("config", "config_planner_pinning.json"), 'w') as configs:
    config['hetero_lattice']      = hetero_lattice
    config['learning']            = learning
    #config['directional']         = directional
    config['d']                   = d
    config['d_min']               = d_min
    json.dump(config, configs)

#%% Useful functions
# ----------------

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h
 
def phi_a(q_i, q_j, r_a, d_a): 
    z = sigma_norm(q_j-q_i)        
    phi_a = rho_h(z/r_a) * phi(z-d_a)    
    return phi_a
    
def phi(z):    
    phi = 0.5*((a+b)*sigma_1(z+c)+(a-b))    
    return phi 

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def a_ij(q_i, q_j, r_a):        
    a_ij = rho_h(sigma_norm(q_j-q_i)/r_a)
    return a_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b

#%% Control systems functions
# -------------------------

d_init = d

def return_lattice_param():
    
    return d_init

def return_ranges():
    
    return r

# form the lattice
#def compute_cmd_a(states_q, states_p, targets, targets_v, k_node, landmarks, **kwargs):
def compute_cmd_a(states_q, states_p, targets, targets_v, k_node, reward_values, **kwargs):   
    
    # pull out the args (try .get() to ignore n/a cases)
    # -----------------
    headings                = kwargs.get('quads_headings')
    consensus_agent         = kwargs.get('consensus_lattice') # rename this object 
    learning_agent          = kwargs.get('learning_lattice')
    directional             = kwargs.get('directional_graph')
    A                       = kwargs.get('A')
    local_k_connectivity    = kwargs.get('local_k_connectivity')
    
    # safety checks
    # -------------
    
    # when using directional 
    if directional:
        #if headings is None:
        #    headings = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        if 'consensus_lattice' in kwargs:
            consensus_agent.headings = headings
    
    # ensure the parameters match the agents
    if consensus_agent is not None and consensus_agent.d_weighted.shape[1] != states_q.shape[1]:
        raise ValueError("Error! There are ", states_q.shape[1], 'agents, but ', consensus_agent.d_weighted.shape[1], 'lattice parameters')
        
    # learning processs (if applicable)
    # ---------------------------------    
        
    # execute the reinforcement learning, local case (if applicable)
    if learning == 1: 
        
        kwargs['learning_grid_size'] = learning_grid_size # consider adapting this with time
        #learning_agent.update_step(landmarks, targets, states_q, states_p, k_node, consensus_agent, **kwargs)
        learning_agent.update_step(reward_values, targets, states_q, states_p, k_node, consensus_agent, **kwargs)
        
    # initialize parameters
    # ----------------------
    if hetero_lattice == 1:
        d = consensus_agent.d_weighted[k_node, k_node]
    else:
        d = d_init
    
    d_a = sigma_norm(d)                         # lattice separation (goal)  
    r_a = sigma_norm(r)                         # lattice separation (sensor range)
    u_int = np.zeros((3,states_q.shape[1]))     # interactions

    # search through each neighbour
    # -----------------------------
    for k_neigh in range(states_q.shape[1]):
        
        # except for itself:
        if k_node != k_neigh:
                        
            # pull the lattice parameter for this node/neighbour pair
            if hetero_lattice == 1:
                d = consensus_agent.d_weighted[k_node, k_neigh]
                d_a = sigma_norm(d)                        
            
            # check if the neighbour is in range
            # ---------------------------------
            
            # use adjacency matrix (new)
            if A[k_node,k_neigh] == 0:
                in_range = False
            else:
                in_range = True 
            
            # if within range
            # ---------------
            if in_range:
                # compute the interaction command
                u_int[:,k_node] += c1_a*phi_a(states_q[:,k_node],states_q[:,k_neigh],r_a, d_a)*n_ij(states_q[:,k_node],states_q[:,k_neigh]) + c2_a*a_ij(states_q[:,k_node],states_q[:,k_neigh],r_a)*(states_p[:,k_neigh]-states_p[:,k_node]) 
                # seek consensus
                if hetero_lattice == 1:
                    consensus_agent.update(k_node, k_neigh)          # update lattice via consensus
                    consensus_agent.prox_i[k_node, k_neigh] = 1      # annotate as in range     
            else:
                if hetero_lattice == 1:
                    consensus_agent.prox_i[k_node, k_neigh] = 0      # annotate as not in range
                
    return u_int[:,k_node] 

# avoid obstacles
def compute_cmd_b(states_q, states_p, obstacles, walls, k_node):
      
    # initialize 
    d_b = sigma_norm(d_prime)                   # obstacle separation (goal range)
    u_obs = np.zeros((3,states_q.shape[1]))     # obstacles 
      
    # search through each obstacle 
    for k_obstacle in range(obstacles.shape[1]):

        # compute norm between this node and this obstacle
        normo = np.linalg.norm(states_q[:,k_node]-obstacles[0:3,k_obstacle])
        
        # ignore if overlapping
        if normo < 0.2:
            continue 
        
        # compute mu
        mu = np.divide(obstacles[3, k_obstacle],normo)
        # compute bold_a_k (for the projection matrix)
        bold_a_k = np.divide(states_q[:,k_node]-obstacles[0:3,k_obstacle],normo)
        bold_a_k = np.array(bold_a_k, ndmin = 2)
        # compute projection matrix
        P = np.identity(states_p.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute beta-agent position and velocity
        q_ik = mu*states_q[:,k_node]+(1-mu)*obstacles[0:3,k_obstacle]
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        if dist_b < r_prime:
            # compute the beta command
            p_ik = mu*np.dot(P,states_p[:,k_node])    
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])
           
    # search through each wall (a planar obstacle)
    for k_wall in range(walls.shape[1]):
        
        # define the wall
        bold_a_k = np.array(np.divide(walls[0:3,k_wall],np.linalg.norm(walls[0:3,k_wall])), ndmin=2).transpose()    # normal vector
        y_k = walls[3:6,k_wall]         # point on plane
        # compute the projection matrix
        P = np.identity(y_k.shape[0]) - np.dot(bold_a_k,bold_a_k.transpose())
        # compute the beta_agent 
        q_ik = np.dot(P,states_q[:,k_node]) + np.dot((np.identity(y_k.shape[0])-P),y_k)
        # compute distance to beta-agent
        dist_b = np.linalg.norm(q_ik-states_q[:,k_node])
        # if it is with the beta range
        maxAlt = 10 # TRAVIS: maxAlt is for testing, only enforces walls below this altitude
        if dist_b < r_prime and states_q[2,k_node] < maxAlt:
            p_ik = np.dot(P,states_p[:,k_node])
            u_obs[:,k_node] += c1_b*phi_b(states_q[:,k_node], q_ik, d_b)*n_ij(states_q[:,k_node], q_ik) + c2_b*b_ik(states_q[:,k_node], q_ik, d_b)*(p_ik - states_p[:,k_node])

        return u_obs[:,k_node] 
    
# track the target
def compute_cmd_g(states_q, states_p, targets, targets_v, k_node, pin_matrix):

    # initialize 
    u_nav = np.zeros((3,states_q.shape[1]))  
    
    # note: the pin matrix "activates" the pins for target tracking (1 = pin)
    u_nav[:,k_node] = - pin_matrix[k_node,k_node]*c1_g*sigma_1(states_q[:,k_node]-targets[:,k_node])- pin_matrix[k_node,k_node]*c2_g*(states_p[:,k_node] - targets_v[:,k_node])
  
    return u_nav[:,k_node]

# consolidate control signals
def compute_cmd(centroid, states_q, states_p, obstacles, walls, targets, targets_v, k_node, **kwargs):
        
    directional         = kwargs.get('directional_graph')
    
    # ensure there are heading available, if needed
    if directional and kwargs.get('quads_headings') is None:
        kwargs['quads_headings'] = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        print('no headings avail, assuming 0')
    
    
    if 'learning_lattice' in kwargs:
        learning_agent  = kwargs.get('learning_lattice')
        if learning_agent.reward_method == 'landmarks':
            reward_values = obstacles
        elif learning_agent.reward_method == 'connectivity':
            reward_values_full  = kwargs.get('local_k_connectivity')
            reward_values       = reward_values_full[k_node]
    else:
        reward_values = 0
    
    # initialize 
    cmd_i = np.zeros((3,states_q.shape[1]))
    
    #u_int = compute_cmd_a(states_q, states_p, targets, targets_v, k_node, obstacles, **kwargs)
    u_int = compute_cmd_a(states_q, states_p, targets, targets_v, k_node, reward_values, **kwargs)
    u_obs = compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
    u_nav = compute_cmd_g(states_q, states_p, targets, targets_v, k_node, kwargs.get('pin_matrix'))
       
    cmd_i[:,k_node] = u_int + u_obs + u_nav
    
    return cmd_i[:,k_node]



# old methods
"""

#%% Select pins for each component 
# --------------------------------

# select pins randomly
def select_pins_random(states_q):
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1
    index = random.randint(0,states_q.shape[1])-1
    pin_matrix[index,index]=1

    return pin_matrix

# select by components
#def select_pins_components(states_q, states_p):
def select_pins_components(states_q, states_p, **kwargs):
    
    # pull out parameters
    headings    = kwargs.get('quads_headings')
    d_weighted  = kwargs.get('d_weighted')
    
    # initialize the pins
    pin_matrix = np.zeros((states_q.shape[1],states_q.shape[1]))
    
    # compute adjacency matrix
    if directional != 1:
        A = grph.adj_matrix(states_q, rg)
        components = grph.find_connected_components_A(A)
        
    else:
        # if no heading avail, set to zero
        if headings is None:
           headings = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        
        #d_weighted  = kwargs['d_weighted']
        #if 'quads_headings' in kwargs:
        #    headings    = kwargs['quads_headings']
        #else:
        #    headings    = np.zeros((states_q.shape[1])).reshape(1,states_q.shape[1])
        #A = grph_dir.adj_matrix_bearing(states_q,states_p,paramClass.d_weighted, sensor_aperature, paramClass.headings)
        
        A = grph_dir.adj_matrix_bearing(states_q,states_p,d_weighted, sensor_aperature, headings)
        components = grph_dir.find_one_way_connected_components_deg(A)
    
    # Gramian method
    # --------------
    if method == 'gramian':
        
        # for each component
        for i in range(0,len(components)):
            
            # find the adjacency and degree matrix of this component 
            states_i = states_q[:,components[i]]
            A = grph.adj_matrix(states_i, rg)  # move these outside (efficiency)
            D = grph.deg_matrix(states_i, rg)
            
            index_i = components[i][0]
            
            # if this is a lone agent
            if len(components[i])==1:
                # pin it
                pin_matrix[index_i,index_i]=1
                
            else: 
                # find gramian trace (i.e. energy demand) of first component
                ctrlable, trace_i = grph.compute_gram_trace(A,D,0,A.shape[1])
                # set a default pin
                pin_matrix[index_i,index_i]=1
                # note: add a test for controlability here
    
                # cycle through the remaining agents in the component
                for j in range(1,len(components[i])): 
                    
                    ctrlable, trace = grph.compute_gram_trace(A,D,j,A.shape[1])
                    
                    # take the smallest energy value
                    if trace < trace_i:
                        # make this the new benchmark
                        trace_i = trace
                        # de-pin the previous
                        pin_matrix[index_i,index_i]=0
                        index_i = components[i][j]
                        # pin this one
                        pin_matrix[index_i,index_i]=1
    
    # Degree method
    # -------------
    elif method == 'degree':
        
        # if using direcational mode
        if directional:
            centralities = {}
            # find the degree centrality within each component 
            for i, component in enumerate(components):
                # store the degree centralities 
                centrality      = grph_dir.out_degree_centrality(A.tolist(), component)
                #centrality[i]   = centrality 
                centralities[i]   = np.diag(list(centrality.values()))

        # for each component
        for i in range(0,len(components)):
            
            # find the adjacency and degree matrix of this component 
            states_i = states_q[:,components[i]]

            if directional:
                
                D = centralities[i]

            else:
                
                D = grph.deg_matrix(states_i, rg)
                 
            index_i = components[i][0]
            
            # if this is a lone agent
            if len(components[i])==1:
                # pin it
                pin_matrix[index_i,index_i]=1
                
            else: 
                
                # find index of highest element of Degree matrix
                index_i = components[i][np.argmax(np.diag(D))]
                # set as default pin
                pin_matrix[index_i,index_i]=1
                       
    # Betweenness
    # -----------
    
    # note: for betweenness, we need > 3 agents (source+destination+node)  
    
    elif method == 'between':
        
        # for each component
        for i in range(0,len(components)):
            
            # default to first node
            index_i = components[i][0]
            
            # if fewer than 4 agents in this component
            if len(components[i])<=3:
                # pin the first one
                pin_matrix[index_i,index_i]=1
            
            # else, we have enough to do betweenness
            else:         
                # pull out the states for this component 
                states_i = states_q[:,components[i]]
                # build a graph within this component (look slighly outside lattice range)
                G = grph.build_graph(states_i,rg+0.1) 
                # find the max influencer
                B = grph.betweenness(G)
                index_ii = max(B, key=B.get)
                #index_ii = min(B, key=B.get)
                index_i = components[i][index_ii]
                # pin the max influencers
                pin_matrix[index_i,index_i] = 1

    else:
        
        for i in range(0,len(components)):
            # just take the first in the component for now
            index = components[i][0]
            # note: later, optimize this selection (i.e. instead of [0], use Grammian)
            pin_matrix[index,index]=1

    return pin_matrix, components
    

"""



    