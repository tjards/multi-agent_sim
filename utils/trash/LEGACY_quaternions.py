#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:46:57 2021

The project implements quaternion rotations

@author: tjards
"""

# import stuff
import numpy as np
#import matplotlib.pyplot as plt


# rotate p1 by quaternion q
# p2 = q * p1 * q'
# --------------------------
def rotate(q, p1):
    p2 = quat_mult(quat_mult(q, np.append([0.0],p1)), quatjugate(q))[1:]
    return p2

# quaternion conjugate
# where q = [qw qx qy qz]^T
# -------------------------
def quatjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


# multiplication between quaternions
# -----------------------------------
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])


# euler to quaternion
# --------------------
def e2q(angles):
    phi = angles[0]
    theta = angles[1]
    psi = angles[2]
    w = np.cos(phi/2) * np.cos(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    x = np.sin(phi/2) * np.cos(theta/2) * np.cos(psi/2) - np.cos(phi/2) * np.sin(theta/2) * np.sin(psi/2)
    y = np.cos(phi/2) * np.sin(theta/2) * np.cos(psi/2) + np.sin(phi/2) * np.cos(theta/2) * np.sin(psi/2)
    z = np.cos(phi/2) * np.cos(theta/2) * np.sin(psi/2) - np.sin(phi/2) * np.sin(theta/2) * np.cos(psi/2)
    return np.array([w, x, y, z])


# quaternion to euler
# ---------------------
def q2e(quat):
 
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    
    # x component
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
 
    # y component
    temp = 2 * (w * y - z * x)
    temp = 1 if temp > 1 else temp
    temp = -1 if temp < -1 else temp
    theta = np.arcsin(temp)
     
    # z component
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
 
    return np.array([phi, theta, psi])



#%% Example

# # vector to be rotated
# p1 = np.array([1,0,0])
# print(p1)

# # how to be rotated 
# phi = np.pi/2           # about x
# theta = -np.pi/4         # about y
# psi = np.pi           # about z
# angles = np.array([phi,theta,psi])
# print(angles)

# # convert to quaternion
# quat = e2q(angles)
# print(quat)

# # convert back to euler
# angles_test = q2e(quat)
# print(angles_test)

# # rotate
# p2 = rotate(quat,p1)
# print(np.round(p2, decimals=2))

# # plot  
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Cartesian axes
# ax.quiver(-1, 0, 0, 3, 0, 0, color='#aaaaaa',linestyle='dashed')
# ax.quiver(0, -1, 0, 0,3, 0, color='#aaaaaa',linestyle='dashed')
# ax.quiver(0, 0, -1, 0, 0, 3, color='#aaaaaa',linestyle='dashed')
# # Vector before rotation
# ax.quiver(0, 0, 0, p1[0], p1[1], p1[2], color='b')
# # Vector after rotation
# ax.quiver(0, 0, 0, p2[0], p2[1], p2[2], color='r')
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([-1.5, 1.5])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()