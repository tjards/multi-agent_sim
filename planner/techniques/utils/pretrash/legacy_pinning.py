#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:32:18 2025

@author: tjards
"""



# legacy (delete later)
'''
# we will grab this cohesion term for gradient estimation
if hetero_gradient == 1:
    #u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gradient_estimates[0:gradient_agent.dimens, k_node, k_neigh]
    u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gain_gradient_control * gradient_agent.C_filtered[0:gradient_agent.dimens, k_node, k_neigh]
    
    # add the addition navigation term for pin_sum
    u_int[0:gradient_agent.dimens,k_node] -= gradient_agent.gain_gradient_control * pin_matrix[k_node,k_node]*gradient_agent.C_sum_bypin[0:gradient_agent.dimens,k_node]
    
    
    # I don't think this is what I'm supposed to bring in
    #gradient_agent.observed_gradients[0:gradient_agent.dimens,k_node, k_neigh] = u_int[0:gradient_agent.dimens,k_node]
    
                '''

#%% legacy code 
# -------------
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



    