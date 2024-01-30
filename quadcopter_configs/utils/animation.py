# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' #my add - this path needs to be added

import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import utils
#import config


# my add - Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


numFrames = 8

#def sameAxisAnimation(config, t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave, Po, obsRad):
def sameAxisAnimation(config, myData, traj, params, obsPF, myColour = 'blue'):

    # travis cleanup
    Ts = config.Ts  
    ifsave = config.ifsave 
    t_all = myData.t_all 
    pos_all = myData.pos_all 
    quat_all = myData.quat_all 
    sDes_tr_all = myData.sDes_traj_all 
    waypoints = traj.wps 
    xyzType = traj.xyzType 
    yawType = traj.yawType 
    Po = obsPF.Po 
    obsRad = obsPF.obsRad

    x = pos_all[:,0]
    y = pos_all[:,1]
    z = pos_all[:,2]

    xDes = sDes_tr_all[:,0]
    yDes = sDes_tr_all[:,1]
    zDes = sDes_tr_all[:,2]

    x_wp = waypoints[:,0]
    y_wp = waypoints[:,1]
    z_wp = waypoints[:,2]

    if (config.orient == "NED"):
        z = -z
        zDes = -zDes
        z_wp = -z_wp

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line1, = ax.plot([], [], [], lw=2, color=myColour)
    line2, = ax.plot([], [], [], lw=2, color='gray')
    #line3, = ax.plot([], [], [], '--', lw=1, color='blue')
    line3, = ax.plot([], [], [], '--', lw=1, color=myColour)

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (config.orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
    elif (config.orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    if (xyzType == 0):
        trajType = 'Hover'
    else:
        ax.scatter(x_wp, y_wp, z_wp, color=myColour, alpha=1, marker = 'o', s = 25)
        if (xyzType == 1 or xyzType == 12):
            trajType = 'Simple Waypoints'
        else:
            ax.plot(xDes, yDes, zDes, ':', lw=1.3, color=myColour)
            if (xyzType == 2):
                trajType = 'Simple Waypoint Interpolation'
            elif (xyzType == 3):
                trajType = 'Minimum Velocity Trajectory'
            elif (xyzType == 4):
                trajType = 'Minimum Acceleration Trajectory'
            elif (xyzType == 5):
                trajType = 'Minimum Jerk Trajectory'
            elif (xyzType == 6):
                trajType = 'Minimum Snap Trajectory'
            elif (xyzType == 7):
                trajType = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType == 8):
                trajType = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType == 9):
                trajType = 'Minimum Snap Trajectory - Stop'
            elif (xyzType == 10):
                trajType = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType == 1):
                trajType = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'

    # travis add
    #o1 = np.array([-2, -1, -3])                  # obstacle 1 (x,y,z)
    #o2 = np.array([3, -2, 1])               # obstacle 2 (x,y,z)
    #Po = np.vstack((o1,o2)).transpose()     # stack obstacles
    
    ax.scatter(np.squeeze(Po[0,:]), np.squeeze(Po[1,:]), -np.squeeze(Po[2,:]), color='red', alpha=1, marker = 'o', s = 100*obsRad)    

    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_from0 = pos_all[0:i*numFrames,0]
        y_from0 = pos_all[0:i*numFrames,1]
        z_from0 = pos_all[0:i*numFrames,2]
    
        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]
        
        quat = quat_all[i*numFrames]
    
        if (config.orient == "NED"):
            z = -z
            z_from0 = -z_from0
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
    
        R = utils.quat2Dcm(quat)    
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 
        
        line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        line1.set_3d_properties(motorPoints[2,0:3])
        line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        line2.set_3d_properties(motorPoints[2,3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))
        
        return line1, line2


    def ini_plot():

        line1.set_data([], [])
        line1.set_3d_properties([])
        line2.set_data([], [])
        line2.set_3d_properties([])
        line3.set_data([], [])
        line3.set_3d_properties([])

        return line1, line2, line3

    
    #my add (init_func messes up, made a new one here)
    line_ani = animation.FuncAnimation(fig, updateLines, blit=False, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames))
 
    # Creating the Animation object
    # ORIGINAL line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    
    if (ifsave):
        # ORIGINAL line_ani.save('Gifs/Raw/animation_{0:.0f}_{1:.0f}.gif'.format(xyzType,yawType), dpi=80, writer='imagemagick', fps=25)
        line_ani.save('Gifs/Raw/animation_{0:.0f}_{1:.0f}_{2}.gif'.format(xyzType,yawType,myColour), writer=writer) #my add
 
  
        
    #plt.show()
    return line_ani