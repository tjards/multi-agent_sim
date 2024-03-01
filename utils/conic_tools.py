#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:43:11 2024

@author: tjards
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# check if a point is within sensor range
# ---------------------------------------
def is_point_in_sensor_range(a, b, v_a, theta, r):
    
    # a         is location of the sensor
    # v_a       is direction the sensor is pointed
    # r         is sensor range
    # theta     is aperature of the sensor (deg, measured from centerline)
    # b         is location of object being sensed

    # normalize sensor direction vector
    if sum(v_a) == 0:
        v_a[0] = 0.1
    
    v_a_unit = v_a / np.linalg.norm(v_a)

    # normalize vector from a to b
    v_ab = b - a
    v_ab_unit = v_ab / np.linalg.norm(v_ab)

    # calculate the projection of v_a onto v_ab
    projection = np.dot(v_ab_unit, v_a_unit)

    # calculate angle between
    angle = np.arccos(projection)

    # convert aperature to radians
    theta_rad = np.radians(theta)

    # check if within aperature
    if angle <= theta_rad / 2:
        # and in range
        if np.linalg.norm(v_ab) <= r:
            return True
    return False

# rotate points around an arbitrary axis
# --------------------------------------
def rotate_points(points, angle, axis):
    
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    
    return np.dot(points, rotation_matrix)

# plot in 3D
# ----------
def plot_3d_space(a, b, v_a, theta, r, in_range):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the locations
    ax.scatter(a[0], a[1], a[2], color='b', label='Sensor')
    if in_range:
        my_color = 'g'
    else:
        my_color = 'r'
    
    
    ax.scatter(b[0], b[1], b[2], color=my_color, label='Target')

    # plot the sensor range
    v_a_unit = v_a / np.linalg.norm(v_a)
    cone_end = a + v_a_unit * r
    ax.plot([a[0], cone_end[0]], [a[1], cone_end[1]], [a[2], cone_end[2]], color='gray', label='Sensor Range')

    # create some points
    num_points = 8                  # density of range plot
    theta_rad = np.radians(theta)
    angles = np.linspace(0, theta_rad / 2, num_points)
    edge_points = np.array([a + v_a_unit * r * np.cos(angle) + np.cross(v_a_unit, b - a) * r * np.sin(angle) for angle in angles])

    # rotate points the points
    edge_points_rotated = rotate_points(edge_points - a, np.pi / 2, v_a_unit) + a
    ax.plot(edge_points_rotated[:, 0], edge_points_rotated[:, 1], edge_points_rotated[:, 2], color='gray', linestyle='--')

    # reflect
    mirrored_edge_points_rotated = rotate_points(edge_points - a, -np.pi / 2, v_a_unit) + a
    ax.plot(mirrored_edge_points_rotated[:, 0], mirrored_edge_points_rotated[:, 1], mirrored_edge_points_rotated[:, 2], color='gray', linestyle=':')

    # connect tips back to sensor
    for i in range(len(edge_points)):
        ax.plot([a[0], edge_points_rotated[i, 0]], [a[1], edge_points_rotated[i, 1]], [a[2], edge_points_rotated[i, 2]], color='gray', linestyle=':')
        ax.plot([a[0], mirrored_edge_points_rotated[i, 0]], [a[1], mirrored_edge_points_rotated[i, 1]], [a[2], mirrored_edge_points_rotated[i, 2]], color='gray', linestyle=':')


    # Set equal aspect ratio
    max_range = np.array([edge_points[:, 0].max()-edge_points[:, 0].min(), edge_points[:, 1].max()-edge_points[:, 1].min(), edge_points[:, 2].max()-edge_points[:, 2].min()]).max()
    max_range2 = np.array([b[:, 0].max()-b[:, 0].min(), b[:, 1].max()-b[:, 1].min(), b[:, 2].max()-b[:, 2].min()]).max()
    if max_range2 > max_range:
        max_range = max_range2
    
    
    ax.set_xlim([a[0] - max_range / 2, a[0] + max_range / 2])
    ax.set_ylim([a[1] - max_range / 2, a[1] + max_range / 2])
    ax.set_zlim([a[2] - max_range / 2, a[2] + max_range / 2])


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# example
# --------------------
# a = np.array([0, 0, 0])
# b = np.array([1, 6, 7])
# v_a = np.array([1, 2, 0.9])  
# theta = 30                  # apereature [degrees]
# r = 5                       # range

# in_range = is_point_in_sensor_range(a, b, v_a, theta, r)

# if in_range:
#     print("TRUE: within sensor range")
# else:
#     print("FALSE: not within sensor ")

# plot_3d_space(a, b, v_a, theta, r, in_range)

