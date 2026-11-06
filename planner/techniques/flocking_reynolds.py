#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:12:59 2021

@author: tjards


This program implements Reynolds Rules of Flocking ("boids")

"""

import numpy as np
import config.config as cfg

# helpers
# -------
def norm_sat(u,maxu):
    norm1b = np.linalg.norm(u)
    u_out = maxu*np.divide(u,norm1b)
    return u_out

# custom class
from planner.base import BasePlanner
class Planner(BasePlanner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    
        # load the configs
        reynolds_config =cfg.get_config(config, 'planner.techniques.flocking_reynolds')
        self.escort          = reynolds_config['escort']          # escort (i.e. target tracking?): 0 = no, 1 = yes
        self.cd_1            = reynolds_config['cd_1']            # cohesion weight
        self.cd_2            = reynolds_config['cd_2']            # alignment weight
        self.cd_3            = reynolds_config['cd_3']            # separation weight
        self.cd_track        = reynolds_config['cd_track']        # nominally, zero, unless escorting, then ensure >0
        self.maxu            = reynolds_config['maxu']            # max input (per rule)  note: dynamics *.evolve_sat must be used for constraints
        self.maxv            = reynolds_config['maxv']            # max v                 note: dynamics *.evolve_sat must be used for constraints
        self.recovery        = reynolds_config['recovery']        # recover if far away (0 = no, 1 = yes)
        self.far_away        = reynolds_config['far_away']        # recover how far away (i.e. when to go back to centroid)?
        self.mode_min_coh    = reynolds_config['mode_min_coh']    # enforce min # of agents (0 = no, 1 = yes)
        self.agents_min_coh  = reynolds_config['agents_min_coh']  # min number of agents
        self.r               = reynolds_config['r']               # range at which neighbours can be sensed  
        self.r_prime         = reynolds_config['r_prime']         # range at which obstacles can be sensed

        nAgents = cfg.get_config(config, 'agents.nAgents')
        self.sensor_range_matrix = self.r * np.ones((nAgents, nAgents))
        self.connection_range_matrix = self.r * np.ones((nAgents, nAgents))

    def order(self, states_q):
        distances = np.zeros((states_q.shape[1],states_q.shape[1])) # to store distances between nodes
        
        # to find the radius that includes min number of agents
        if self.mode_min_coh == 1:
            slide = 0
            for k_node in range(states_q.shape[1]):
                #slide += 1
                for k_neigh in range(slide,states_q.shape[1]):
                    if k_node != k_neigh:
                        distances[k_node,k_neigh] = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
        return distances 

    # ========================== #
    # VECTORIZED BATCH COMMANDS  #
    # ========================== #

    def compute_cmd_vectorized(self, states, targets, neighbor_lists, **kwargs):
        """
        Compute commands for ALL agents at once using vectorized NumPy.

        Config coverage (see config.json -> planner.techniques.flocking_reynolds):
          Fully implemented:
            - escort = 0 or 1       target tracking vs centroid tracking
            - cd_1, cd_2, cd_3      cohesion / alignment / separation weights
            - cd_track              navigation gain
            - maxu, maxv            saturation limits
            - r, r_prime            sensing and separation radii
          Partially implemented:
            - recovery = 1          navigates far agents toward centroid, but does
                                    NOT override cd_4 to 0.3 per-agent (uses cd_track
                                    uniformly). Full per-agent gain override would
                                    require a per-agent gain array.
          Not implemented (falls back to scalar):
            - mode_min_coh = 1      adaptive per-agent cohesion radius based on
                                    sorted neighbor distances. Requires per-agent
                                    distance sorting incompatible with batch approach.

        Args:
            states: (6, n) agent positions and velocities
            targets: (6, n) target positions and velocities
            neighbor_lists: list[list[int]] from SpatialIndex.query_ball_tree(r)
            **kwargs: must include 'centroid' (3,1) array

        Returns:
            (3, n) command array, or None to fall back to scalar path.
        """
        # fall back to scalar for adaptive cohesion mode (needs per-agent sorted distances)
        if self.mode_min_coh == 1:
            return None

        states_q = states[0:3, :]     # (3, n)
        states_p = states[3:6, :]     # (3, n)
        targets_q = targets[0:3, :]   # (3, n)
        centroid = kwargs.get('centroid')
        n = states_q.shape[1]

        r_eff = self.r  # effective range (mode_min_coh=0)

        # --- build directed edge arrays from neighbor_lists ---
        src_list = []
        dst_list = []
        for i, neighs in enumerate(neighbor_lists):
            for j in neighs:
                src_list.append(i)
                dst_list.append(j)

        # initialize per-agent accumulators
        sum_poses = np.zeros((3, n))     # cohesion: sum of neighbor positions
        sum_velos = np.zeros((3, n))     # alignment: sum of neighbor velocities
        sum_obs   = np.zeros((3, n))     # separation: sum of repulsion vectors
        count_coh = np.zeros(n)          # cohesion/alignment neighbor count
        count_sep = np.zeros(n)          # separation neighbor count

        if src_list:
            src = np.array(src_list, dtype=np.intp)
            dst = np.array(dst_list, dtype=np.intp)

            # compute pairwise distances
            dq = states_q[:, src] - states_q[:, dst]  # q_i - q_j
            dists = np.linalg.norm(dq, axis=0)        # (M,)

            # skip collisions (dist < 0.1)
            valid = dists >= 0.1

            # cohesion + alignment: dist < r_eff
            coh_mask = valid & (dists < r_eff)
            if np.any(coh_mask):
                src_coh = src[coh_mask]
                dst_coh = dst[coh_mask]
                # accumulate neighbor positions and velocities
                np.add.at(sum_poses, (np.arange(3)[:, None], src_coh[None, :]),
                          states_q[:, dst_coh])
                np.add.at(sum_velos, (np.arange(3)[:, None], src_coh[None, :]),
                          states_p[:, dst_coh])
                np.add.at(count_coh, src_coh, 1)

            # separation: dist < r_prime
            sep_mask = valid & (dists < self.r_prime)
            if np.any(sep_mask):
                src_sep = src[sep_mask]
                dst_sep = dst[sep_mask]
                dists_sep = dists[sep_mask]
                # repulsion: -(q_i - q_j) / dist^2 = (q_j - q_i) / dist^2
                repulsion = -dq[:, sep_mask] / (dists_sep[None, :] ** 2)
                np.add.at(sum_obs, (np.arange(3)[:, None], src_sep[None, :]), repulsion)
                np.add.at(count_sep, src_sep, 1)

        # --- post-processing: per-agent cohesion, alignment, separation ---
        # NOTE: the scalar code uses ||raw_sum|| as the normalization denominator
        #       inside the velocity formula, NOT ||mean - pos||. We match that here.
        cmd = np.zeros((3, n))

        has_neighs = count_coh > 0

        # Cohesion
        if np.any(has_neighs):
            # norm of raw sum of neighbor positions (matches scalar norm_coh)
            norm_coh = np.linalg.norm(sum_poses, axis=0, keepdims=True)
            norm_coh_safe = np.maximum(norm_coh, 1e-15)
            coh_dir = sum_poses / np.maximum(count_coh[None, :], 1) - states_q
            temp_coh = self.maxv * coh_dir / norm_coh_safe - states_p
            # norm_sat per agent
            temp_norms = np.linalg.norm(temp_coh, axis=0, keepdims=True)
            temp_norms_safe = np.maximum(temp_norms, 1e-15)
            u_coh = self.cd_1 * self.maxu * temp_coh / temp_norms_safe
            mask_coh = has_neighs & (norm_coh.ravel() > 0)
            cmd[:, mask_coh] += u_coh[:, mask_coh]

        # Alignment
        if np.any(has_neighs):
            # norm of raw sum of neighbor velocities (matches scalar norm_ali)
            norm_ali = np.linalg.norm(sum_velos, axis=0, keepdims=True)
            norm_ali_safe = np.maximum(norm_ali, 1e-15)
            ali_dir = sum_velos / np.maximum(count_coh[None, :], 1)
            temp_ali = self.maxv * ali_dir / norm_ali_safe - states_p
            temp_norms_a = np.linalg.norm(temp_ali, axis=0, keepdims=True)
            temp_norms_a_safe = np.maximum(temp_norms_a, 1e-15)
            u_ali = self.cd_2 * self.maxu * temp_ali / temp_norms_a_safe
            mask_ali = has_neighs & (norm_ali.ravel() > 0)
            cmd[:, mask_ali] += u_ali[:, mask_ali]

        # Separation
        has_sep = count_sep > 0
        if np.any(has_sep):
            norm_sep = np.linalg.norm(sum_obs, axis=0, keepdims=True)
            norm_sep_safe = np.maximum(norm_sep, 1e-15)
            sep_dir = sum_obs / np.maximum(count_sep[None, :], 1)
            temp_sep = self.maxv * sep_dir / norm_sep_safe - states_p
            temp_norms_s = np.linalg.norm(temp_sep, axis=0, keepdims=True)
            temp_norms_s_safe = np.maximum(temp_norms_s, 1e-15)
            u_sep = -self.cd_3 * self.maxu * temp_sep / temp_norms_s_safe
            mask_sep = has_sep & (norm_sep.ravel() > 0)
            cmd[:, mask_sep] += u_sep[:, mask_sep]

        # --- navigation / tracking ---
        cd_4 = self.cd_track
        if self.escort == 1:
            temp_nav = targets_q - states_q
        else:
            temp_nav = centroid.T - states_q  # centroid is (3,1)

        # recovery override (per-agent)
        if self.recovery == 1 and centroid is not None:
            dist_from_centroid = np.linalg.norm(centroid.T - states_q, axis=0)
            far_mask = dist_from_centroid > self.far_away
            # for far agents, override gain and target
            if np.any(far_mask):
                temp_nav[:, far_mask] = (centroid.T - states_q)[:, far_mask]
                # cd_4 stays at cd_track for escort, 0.3 for recovery
                # handled below via per-agent gain

        nav_norms = np.linalg.norm(temp_nav, axis=0, keepdims=True)
        nav_norms_safe = np.maximum(nav_norms, 1e-15)
        u_nav = cd_4 * self.maxu * temp_nav / nav_norms_safe
        cmd += u_nav

        return cmd

    # Compute commands
    # ----------------

    def compute_cmd(self, states, targets, index, **kwargs):

        # Extract 
        states_q = states[0:3, :]     # positions
        states_p = states[3:6, :]     # velocities
        targets_q = targets[0:3, :]   # target positions
        centroid = kwargs.get('centroid')
        distances = kwargs.get('distances')
        k_node = index

        # Reynolds Flocking
        # ------------------ 
        
        # (3,) not (3, n) — heap fragmentation fix (see OPTIMIZATION.md)
        u_coh = np.zeros(3)
        u_ali = np.zeros(3)
        u_sep = np.zeros(3)
        u_nav = np.zeros(3)
        cmd_i = np.zeros(3)
        
        #initialize for this node
        temp_total = 0
        temp_total_prime = 0
        temp_total_coh = 0
        sum_poses = np.zeros((3))
        sum_velos = np.zeros((3))
        sum_obs = np.zeros((3))
        
        # adjust cohesion range for min number of agents 
        if self.mode_min_coh == 1:
            
            # make sure the number of vehicles is bigger than the min number of agents 
            if distances.shape[0] < self.agents_min_coh+2:
                raise Exception('There are an insufficient number of agents for the cohesion mode selected. Minimum number of agents for mode ',self.agents_min_coh ,' is ', self.agents_min_coh+2, ' and you have selected ', distances.shape[0] )
        
            r_coh = 0
            # pull agent distantces for this node
            node_ranges = distances[k_node,:]
            # sort the distances
            node_ranges_sorted = np.sort(node_ranges)
            # take the distance of the farthest agent that satisfies min count
            r_coh_temp = node_ranges_sorted[self.agents_min_coh+1]
            r_coh = r_coh_temp
        else:
            # else, just rely on default range
            r_coh = self.r
            
        # search through each neighbour
        for k_neigh in range(states_q.shape[1]):
            # except for itself (duh):
            if k_node != k_neigh:
                # compute the euc distance between them
                dist = np.linalg.norm(states_q[:,k_node]-states_q[:,k_neigh])
                
                if dist < 0.1:
                    # print out any collisions 
                    print('collision at agent: ', k_node)
                    continue
        
                # if agent is within the alignment range
                if dist < np.maximum(self.r,r_coh):
        
                    # count
                    temp_total += 1                        
        
                    # sum 
                    sum_velos += states_p[:,k_neigh]
        
                # if within cohesion range 
                if dist < np.maximum(self.r,r_coh):
                    
                    #count
                    temp_total_coh += 1
                    
                    #sum
                    sum_poses += states_q[:,k_neigh]
        
                # if within the separation range 
                if dist < self.r_prime:
                    
                    # count
                    temp_total_prime += 1
                    
                    # sum of obstacles 
                    sum_obs += -(states_q[:,k_node]-states_q[:,k_neigh])/(dist**2)                        
        
        # norms
        # -----
        norm_coh = np.linalg.norm(sum_poses)
        norm_ali = np.linalg.norm(sum_velos)
        norm_sep = np.linalg.norm(sum_obs)
        
        if temp_total != 0:
            
            # Cohesion
            # --------
            if norm_coh != 0:
                temp_u_coh = (self.maxv*np.divide(((np.divide(sum_poses,temp_total_coh) - states_q[:,k_node])),norm_coh)-states_p[:,k_node])
                u_coh = self.cd_1*norm_sat(temp_u_coh,self.maxu)
            
            # Alignment
            # ---------
            if norm_ali != 0:                 
                temp_u_ali = (self.maxv*np.divide((np.divide(sum_velos,temp_total)),norm_ali)-states_p[:,k_node])
                u_ali = self.cd_2*norm_sat(temp_u_ali,self.maxu)
        
        if temp_total_prime != 0 and norm_sep != 0:
                
            # Separtion
            # ---------
            temp_u_sep = (self.maxv*np.divide(((np.divide(sum_obs,temp_total_prime))),norm_sep)-states_p[:,k_node]) 
            u_sep = -self.cd_3*norm_sat(temp_u_sep,self.maxu)
                    
        # Tracking
        # -------- 
        cd_4 = self.cd_track
            
        # if doing recovery
        if self.recovery == 1:
            # and if far away, adjust gain to drive back
            if np.linalg.norm(centroid.transpose()-states_q[:,k_node]) > self.far_away:
                cd_4 = 0.3 # overides set value, so recovery is always a low gain
            
        # if escorting, track the target (overrides recovery actions)
        if self.escort == 1:
            cd_4 = self.cd_track
            if cd_4 == 0:
                print('WARNING: no gain set for tracking target, please set a gain > 0')
            temp_u_nav = (targets_q[:,k_node]-states_q[:,k_node])
        else:
            temp_u_nav = (centroid.transpose()-states_q[:,k_node])
        
        # compute tracking 
        u_nav = cd_4*norm_sat(temp_u_nav,self.maxu)
        
        # compute consolidated commands
        return u_coh + u_ali + u_sep + u_nav
    
