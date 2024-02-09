#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:43:11 2024

@author: tjards
"""


import numpy as np

def is_point_in_cone(a, b, c, h, cone_angle):
    ab = b - a
    bc = c - b
    
    # Project ab onto bc
    ab_proj_bc = np.dot(ab, bc) / np.linalg.norm(bc)
    
    # Calculate the angle between ab and bc
    angle_ab_bc = np.arccos(ab_proj_bc / np.linalg.norm(ab))
    
    # Calculate half of the cone's angle
    half_cone_angle = np.radians(cone_angle / 2)
    
    # Check if the angle is less than or equal to half of the cone's angle
    return angle_ab_bc <= half_cone_angle

# Example usage
a = np.array([1, 2, 3])  # Coordinates of point a (random point)
b = np.array([4, 5, 6])  # Coordinates of point b (the sensor with conic range)
c = np.array([4, 5, 9])  # Coordinates of point c (tip of the cone)
h = 5  # Height of the cone
cone_angle = 45  # Angle of the cone

result = is_point_in_cone(a, b, c, h, cone_angle)
print("Is point a within the cone:", result)