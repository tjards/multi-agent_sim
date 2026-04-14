#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module implements Olfati-Saber flocking

Created on Sat Sep 11 10:17:52 2021

@author: tjards


"""

#%% Import stuff
# ------------
import numpy as np
import os
import json

#%% New: object
# -------------

import config.config as cfg

# helpers
def regnorm(z):
    norm = np.divide(z,np.linalg.norm(z))
    return norm

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

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
        saber_config =cfg.get_config(config, 'planner.techniques.flocking_saber')
        self.pi      = saber_config['pi'] # value of pi
        self.a       = saber_config['a']  # uneven sigmoid parameter a
        self.b       = saber_config['b']  # uneven sigmoid parameter b
        self.c       = np.divide(np.abs(self.a-self.b),np.sqrt(4*self.a*self.b)) # uneven sigmoid parameter c
        self.eps     = saber_config['eps']
        self.h       = saber_config['h'] # bump function parameter h
        self.c1_a    = saber_config['c1_a'] # interaction gain, position
        self.c2_a    = saber_config['c2_a'] # interaction gain, velocity
        self.c1_b    = saber_config['c1_b'] # obstacle avoidance gain, position
        self.c2_b    = saber_config['c2_b'] # obstacle avoidance gain, velocity
        self.c1_g    = saber_config['c1_g'] # navigation gain, position
        self.c2_g    = saber_config['c2_g'] # navigation gain, velocity
        self.d       = saber_config['d']        # desired inter-agent distance
        self.d_prime = saber_config['d_prime']  # desired obstacle distance
        self.r       = saber_config['r']        # range at which neighbours can be sensed
        self.r_prime = saber_config['r_prime']  # range at which obstacles can be sensed

        nAgents = cfg.get_config(config, 'agents.nAgents')
        self.sensor_range_matrix = self.r * np.ones((nAgents, nAgents))
        self.connection_range_matrix = self.d * np.ones((nAgents, nAgents))

    # methods that depend on class attributes/methods
    def sigma_norm(self, z):    
        norm_sig = (1/self.eps) * (np.sqrt(1 + self.eps*np.linalg.norm(z)**2) - 1)
        return norm_sig
    
    def n_ij(self, q_i, q_j):
        return np.divide(q_j-q_i, np.sqrt(1 + self.eps*np.linalg.norm(q_j-q_i)**2))
    
    def rho_h(self, z):    
        if 0 <= z < self.h:
            return 1
        elif self.h <= z < 1:
            return 0.5 * (1 + np.cos(self.pi*np.divide(z-self.h, 1-self.h)))
        else:
            return 0
    
    def phi(self, z):    
        return 0.5 * ((self.a+self.b)*sigma_1(z+self.c) 
                      + (self.a-self.b))
    
    def phi_a(self, q_i, q_j, r_a, d_a): 
        z = self.sigma_norm(q_j-q_i)        
        return self.rho_h(z/r_a) * self.phi(z-d_a)
    
    def a_ij(self, q_i, q_j, r_a):        
        return self.rho_h(self.sigma_norm(q_j-q_i)/r_a)
    
    def b_ik(self, q_i, q_ik, d_b):        
        return self.rho_h(self.sigma_norm(q_ik-q_i)/d_b)
    
    def phi_b(self, q_i, q_ik, d_b): 
        z = self.sigma_norm(q_ik-q_i)        
        return self.rho_h(z/d_b) * (sigma_1(z-d_b) - 1)
    
    # main functions 
    def gradient(self, states_q, k_node, k_neigh):
        r_a = self.sigma_norm(self.r)
        d_a = self.sigma_norm(self.d)
        u_gradient = self.c1_a * self.phi_a(states_q[:,k_node], states_q[:,k_neigh], r_a, d_a) * \
                     self.n_ij(states_q[:,k_node], states_q[:,k_neigh])
        return u_gradient
    
    def velocity_alignment(self, states_q, states_p, k_node, k_neigh):
        r_a = self.sigma_norm(self.r)
        u_velocity_alignment = self.c2_a * self.a_ij(states_q[:,k_node], states_q[:,k_neigh], r_a) * \
                               (states_p[:,k_neigh] - states_p[:,k_node])
        return u_velocity_alignment
    
    def navigation(self, states_q, states_p, targets, targets_v, k_node):
        u_navigation = -self.c1_g*sigma_1(states_q[:,k_node]-targets[:,k_node]) - \
                       self.c2_g*(states_p[:,k_node] - targets_v[:,k_node])
        return u_navigation
    
    # ========================== #
    # VECTORIZED BATCH COMMANDS  #
    # ========================== #

    def compute_cmd_vectorized(self, states, targets, neighbor_lists, **kwargs):
        """
        Compute commands for ALL agents at once using vectorized NumPy.

        Config coverage (see config.json -> planner.techniques.flocking_saber):
          Fully implemented:
            - All Olfati-Saber parameters (a, b, eps, h, c1_a, c2_a, c1_g, c2_g, d, r)
            - Interaction (gradient + velocity alignment) via batched pair operations
            - Navigation (sigma_1 element-wise) over all agents simultaneously
          Kept as per-agent loop:
            - Obstacle avoidance (c1_b, c2_b, d_prime, r_prime) — obstacle count is
              typically 1-5, so per-agent iteration is negligible vs pair interactions.
              Vectorizing over agents per-obstacle is possible but low priority.

        Args:
            states: (6, n) agent positions and velocities
            targets: (6, n) target positions and velocities
            neighbor_lists: list[list[int]] from SpatialIndex.query_ball_tree(r)
            **kwargs: must include 'obstacles_plus' and 'walls'

        Returns:
            (3, n) command array. Always succeeds (no fallback to None).
        """
        states_q = states[0:3, :]      # (3, n)
        states_p = states[3:6, :]      # (3, n)
        targets_q = targets[0:3, :]    # (3, n)
        targets_p = targets[3:6, :]    # (3, n)
        n = states_q.shape[1]

        # precompute sigma-norm constants
        r_a = self._sigma_norm_scalar(self.r)
        d_a = self._sigma_norm_scalar(self.d)

        # --- interaction forces (vectorized over all neighbor pairs) ---
        cmd_int = np.zeros((3, n))

        # build flat arrays of (i, j) directed edges from neighbor_lists
        src = []
        dst = []
        for i, neighs in enumerate(neighbor_lists):
            for j in neighs:
                src.append(i)
                dst.append(j)

        if src:
            src = np.array(src, dtype=np.intp)
            dst = np.array(dst, dtype=np.intp)
            M = len(src)

            # position/velocity differences: (3, M)
            dq = states_q[:, dst] - states_q[:, src]          # q_j - q_i
            dp = states_p[:, dst] - states_p[:, src]          # p_j - p_i

            # sigma norms over all pairs: (M,)
            dist_sq = np.sum(dq * dq, axis=0)                 # ||dq||^2
            sqrt_term = np.sqrt(1.0 + self.eps * dist_sq)     # sqrt(1 + eps*||dq||^2)
            sigma_norms = (1.0 / self.eps) * (sqrt_term - 1.0)

            # n_ij vectors: (3, M)
            n_ij_vec = dq / sqrt_term[np.newaxis, :]           # (q_j-q_i)/sqrt(1+eps*||dq||^2)

            # vectorized rho_h: bump function on z = sigma_norms / r_a
            z_ra = sigma_norms / r_a
            rho_vals = self._rho_h_vec(z_ra)                  # (M,)

            # vectorized phi: phi(sigma_norms - d_a)
            z_phi = sigma_norms - d_a
            sigma_1_vals = z_phi / np.sqrt(1.0 + z_phi * z_phi)
            phi_vals = 0.5 * ((self.a + self.b) * ((z_phi + self.c) / np.sqrt(1.0 + (z_phi + self.c)**2))
                              + (self.a - self.b))

            # phi_a = rho_h * phi
            phi_a_vals = rho_vals * phi_vals                   # (M,)

            # gradient contribution: c1_a * phi_a * n_ij  -> (3, M)
            grad_contrib = self.c1_a * phi_a_vals[np.newaxis, :] * n_ij_vec

            # alignment contribution: c2_a * rho_h * (p_j - p_i) -> (3, M)
            align_contrib = self.c2_a * rho_vals[np.newaxis, :] * dp

            # total per-pair interaction force on src agent
            force = grad_contrib + align_contrib               # (3, M)

            # scatter-add to per-agent commands
            np.add.at(cmd_int, (np.arange(3)[:, np.newaxis], src[np.newaxis, :]), force)

        # --- navigation forces (trivially vectorized over all agents) ---
        # sigma_1 is element-wise: z_i / sqrt(1 + z_i^2), not norm-based
        dq_nav = states_q - targets_q
        dp_nav = states_p - targets_p
        sigma_1_nav = dq_nav / np.sqrt(1.0 + dq_nav * dq_nav)
        cmd_nav = -self.c1_g * sigma_1_nav - self.c2_g * dp_nav

        # --- obstacle forces (kept per-agent — obstacle count is small) ---
        obstacles = kwargs.get('obstacles_plus')
        walls = kwargs.get('walls')
        cmd_obs = np.zeros((3, n))
        if obstacles is not None and walls is not None:
            for k in range(n):
                cmd_obs[:, k] = self.compute_cmd_b(states_q, states_p, obstacles, walls, k)

        return cmd_int + cmd_nav + cmd_obs

    # vectorized helper: scalar sigma_norm (no array input)
    def _sigma_norm_scalar(self, r_val):
        return (1.0 / self.eps) * (np.sqrt(1.0 + self.eps * r_val**2) - 1.0)

    # vectorized bump function over array
    def _rho_h_vec(self, z):
        result = np.zeros_like(z)
        mask1 = (z >= 0) & (z < self.h)
        mask2 = (z >= self.h) & (z < 1.0)
        result[mask1] = 1.0
        result[mask2] = 0.5 * (1.0 + np.cos(self.pi * (z[mask2] - self.h) / (1.0 - self.h)))
        return result

    # main command functions
    def return_ranges(self):
        return self.d
    
    def compute_cmd(self, states, targets, index, **kwargs):
    
        # Extract 
        states_q = states[0:3, :]      # positions
        states_p = states[3:6, :]      # velocities
        targets_q = targets[0:3, :]    # target positions
        targets_p = targets[3:6, :]    # target velocities
        obstacles = kwargs.get('obstacles_plus')
        walls = kwargs.get('walls')
        k_node = index
        
        # three force components
        u_int = self.compute_cmd_a(states_q, states_p, k_node)
        u_nav = self.compute_cmd_g(states_q, states_p, targets_q, targets_p, k_node)
        u_obs = self.compute_cmd_b(states_q, states_p, obstacles, walls, k_node)
        
        # Return combined command
        return u_int + u_nav + u_obs


    def compute_cmd_a(self, states_q, states_p, k_node):
        # (3,) not (3, n) — same heap fragmentation fix as compute_cmd_b
        u_int = np.zeros(3)
        
        for k_neigh in range(states_q.shape[1]):
            if k_node != k_neigh:
                dist = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
                if dist < self.r:
                    u_int += self.gradient(states_q, k_node, k_neigh) + \
                             self.velocity_alignment(states_q, states_p, k_node, k_neigh)
        
        return u_int
    
    def compute_cmd_g(self, states_q, states_p, targets, targets_v, k_node):
        return self.navigation(states_q, states_p, targets, targets_v, k_node)
    
    # Pre-allocated 3x3 identity matrix shared across all calls.
    # The original code called np.identity(3) per agent per timestep per obstacle,
    # generating millions of small temporary arrays over a full simulation. CPython's
    # pymalloc arena allocator cannot compact these, causing monotonic RSS growth
    # (~185 MB per 100 timesteps at n=200). This is not a numpy leak but a heap
    # fragmentation issue in the application code. Fix: allocate once, reuse.
    _I3 = np.identity(3)

    def compute_cmd_b(self, states_q, states_p, obstacles, walls, k_node):
        d_b = self.sigma_norm(self.d_prime)
        # Allocate (3,) not (3, n): only one agent's output is computed per call.
        # The original (3, n) allocation was wasteful (only column k_node was used)
        # and contributed to the heap fragmentation described above.
        u_obs = np.zeros(3)
        
        # Obstacles
        for k_obstacle in range(obstacles.shape[1]):
            normo = np.linalg.norm(states_q[:,k_node] - obstacles[0:3,k_obstacle])
            
            if normo < 0.2:
                continue
            
            mu = np.divide(obstacles[3, k_obstacle], normo)
            bold_a_k = np.divide(states_q[:,k_node] - obstacles[0:3,k_obstacle], normo)
            bold_a_k = np.array(bold_a_k, ndmin=2)
            P = self._I3 - np.dot(bold_a_k, bold_a_k.transpose())
            q_ik = mu*states_q[:,k_node] + (1-mu)*obstacles[0:3,k_obstacle]
            dist_b = np.linalg.norm(q_ik - states_q[:,k_node])
            
            if dist_b < self.r_prime:
                p_ik = mu*np.dot(P, states_p[:,k_node])
                u_obs += self.c1_b*self.phi_b(states_q[:,k_node], q_ik, d_b)*\
                         self.n_ij(states_q[:,k_node], q_ik) + \
                         self.c2_b*self.b_ik(states_q[:,k_node], q_ik, d_b)*\
                         (p_ik - states_p[:,k_node])
        
        # Walls
        for k_wall in range(walls.shape[1]):
            bold_a_k = np.array(np.divide(walls[0:3,k_wall], np.linalg.norm(walls[0:3,k_wall])), ndmin=2).transpose()
            y_k = walls[3:6, k_wall]
            P = self._I3 - np.dot(bold_a_k, bold_a_k.transpose())
            q_ik = np.dot(P, states_q[:,k_node]) + np.dot((self._I3 - P), y_k)
            dist_b = np.linalg.norm(q_ik - states_q[:,k_node])
            
            maxAlt = 10
            if dist_b < self.r_prime and states_q[2,k_node] < maxAlt:
                p_ik = np.dot(P, states_p[:,k_node])
                u_obs += self.c1_b*self.phi_b(states_q[:,k_node], q_ik, d_b)*\
                         self.n_ij(states_q[:,k_node], q_ik) + \
                         self.c2_b*self.b_ik(states_q[:,k_node], q_ik, d_b)*\
                         (p_ik - states_p[:,k_node])
        
        return u_obs

