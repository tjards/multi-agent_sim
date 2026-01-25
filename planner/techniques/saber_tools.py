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
class Planner:

    def __init__(self, config):
 
        # load the configs
        saber_config =cfg.get_config(config, 'planner.techniques.saber')
        self.pi      = saber_config['pi']
        self.a       = saber_config['a']
        self.b       = saber_config['b']
        self.c       = np.divide(np.abs(self.a-self.b),np.sqrt(4*self.a*self.b)) 
        self.eps     = saber_config['eps']
        self.h       = saber_config['h']
        self.c1_a    = saber_config['c1_a']
        self.c2_a    = saber_config['c2_a']
        self.c1_b    = saber_config['c1_b']
        self.c2_b    = saber_config['c2_b']
        self.c1_g    = saber_config['c1_g']
        self.c2_g    = saber_config['c2_g']
        self.d       = saber_config['d']
        self.d_prime = saber_config['d_prime']
        self.r       = saber_config['r']
        self.r_prime = saber_config['r_prime']

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
    
    # main command functions
    def return_ranges(self):
        return self.d
    
    def compute_cmd_a(self, states_q, states_p, k_node):
        u_int = np.zeros((3, states_q.shape[1]))
        
        for k_neigh in range(states_q.shape[1]):
            if k_node != k_neigh:
                dist = np.linalg.norm(states_q[:,k_node] - states_q[:,k_neigh])
                if dist < self.r:
                    u_int[:,k_node] += self.gradient(states_q, k_node, k_neigh) + \
                                       self.velocity_alignment(states_q, states_p, k_node, k_neigh)
        
        return u_int[:,k_node]
    
    def compute_cmd_g(self, states_q, states_p, targets, targets_v, k_node):
        u_nav = np.zeros((3, states_q.shape[1]))
        u_nav[:,k_node] = self.navigation(states_q, states_p, targets, targets_v, k_node)
        return u_nav[:,k_node]
    
    def compute_cmd_b(self, states_q, states_p, obstacles, walls, k_node):
        d_b = self.sigma_norm(self.d_prime)
        u_obs = np.zeros((3, states_q.shape[1]))
        
        # Obstacles
        for k_obstacle in range(obstacles.shape[1]):
            normo = np.linalg.norm(states_q[:,k_node] - obstacles[0:3,k_obstacle])
            
            if normo < 0.2:
                continue
            
            mu = np.divide(obstacles[3, k_obstacle], normo)
            bold_a_k = np.divide(states_q[:,k_node] - obstacles[0:3,k_obstacle], normo)
            bold_a_k = np.array(bold_a_k, ndmin=2)
            P = np.identity(states_p.shape[0]) - np.dot(bold_a_k, bold_a_k.transpose())
            q_ik = mu*states_q[:,k_node] + (1-mu)*obstacles[0:3,k_obstacle]
            dist_b = np.linalg.norm(q_ik - states_q[:,k_node])
            
            if dist_b < self.r_prime:
                p_ik = mu*np.dot(P, states_p[:,k_node])
                u_obs[:,k_node] += self.c1_b*self.phi_b(states_q[:,k_node], q_ik, d_b)*\
                                   self.n_ij(states_q[:,k_node], q_ik) + \
                                   self.c2_b*self.b_ik(states_q[:,k_node], q_ik, d_b)*\
                                   (p_ik - states_p[:,k_node])
        
        # Walls
        for k_wall in range(walls.shape[1]):
            bold_a_k = np.array(np.divide(walls[0:3,k_wall], np.linalg.norm(walls[0:3,k_wall])), ndmin=2).transpose()
            y_k = walls[3:6, k_wall]
            P = np.identity(y_k.shape[0]) - np.dot(bold_a_k, bold_a_k.transpose())
            q_ik = np.dot(P, states_q[:,k_node]) + np.dot((np.identity(y_k.shape[0])-P), y_k)
            dist_b = np.linalg.norm(q_ik - states_q[:,k_node])
            
            maxAlt = 10
            if dist_b < self.r_prime and states_q[2,k_node] < maxAlt:
                p_ik = np.dot(P, states_p[:,k_node])
                u_obs[:,k_node] += self.c1_b*self.phi_b(states_q[:,k_node], q_ik, d_b)*\
                                   self.n_ij(states_q[:,k_node], q_ik) + \
                                   self.c2_b*self.b_ik(states_q[:,k_node], q_ik, d_b)*\
                                   (p_ik - states_p[:,k_node])
        
        return u_obs[:,k_node]

