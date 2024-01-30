#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is a custom modeller for the swarm

form is a follows:
    
    x_{k+1} = A*x_{k} = B*u_{k}
    
    x = states (pos, vel) of the swarm (not directly accessible)
    u = target (pin location)
    
    y_{k} = C*x_{k}
    
    C = linear matric that computes the centroid of the swarm



Created on Sun Dec 11 11:51:22 2022

@author: tjards
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class model:
    
    # note: this is very inefficient. Later, will pre-define the size of the array and load it it.

    # initializes a streaming model
    def __init__(self):
        
        # set initial value of y
        self.count_x          = -1
        self.count_y          = -1
        self.desired_size   = 3000 # how many samples model
        
        self.stream_y = np.zeros((12, self.desired_size))
        self.stream_x = np.zeros((15, self.desired_size))
        
        #self.stream_y       = np.array([0,0,0,0,0,0],ndmin=2).transpose()
        #self.stream_x       = np.array([0,0,0,0,0,0,0,0,0],ndmin=2).transpose()

    # updates the data stream
    def update_stream_y(self, y_n):
        
        #only update if under-sampled
        if self.count_y < self.desired_size:
        #self.stream_y = np.append(self.stream_y,y_n, axis=1)
            self.stream_y[:,self.count_y] = y_n.ravel()
            self.count_y += 1
        
        #print(self.stream_y)
        
    def update_stream_x(self, x_n):
        
        #only update if under-sampled
        if self.count_x < self.desired_size:
        #self.stream_x = np.append(self.stream_x,x_n, axis=1)
        #print(self.stream_x)
            self.stream_x[:,self.count_x] = x_n.ravel()
            self.count_x += 1
        
    def fit(self):
        
        #if self.count_x >= self.desired_size and self.count_y >= self.desired_size:
        print('fitting')
        # reset the counter
        #self.count_x    = -1
        #self.count_y    = -1
            
        # TRAVIS: start here. The inputs here are mis-shapen
            
        # X {array-like, sparse matrix} of shape (n_samples, n_features) Training data.
        # y array-like of shape (n_samples,) or (n_samples, n_targets) Target values. 
        
        reg = LinearRegression().fit(self.stream_x.transpose(), self.stream_y.transpose())
            
        eps = 0.000001
        self.coeffs = reg.coef_
        self.A = reg.coef_[:,0:12]
        self.A[self.A < eps] = 0
        self.B = reg.coef_[:,12:15]
        self.B[self.B < eps] = 0
            
        #m, c = np.linalg.lstsq(self.stream_x, self.stream_y, rcond=None)[0]
        #self.m = m
        #self.c = c

            
        
