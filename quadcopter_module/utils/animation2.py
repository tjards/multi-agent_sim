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
def sameAxisAnimation2(config, myData, traj, params, myData2, traj2, params2, obsPF, myColour = 'blue', myColour2 = 'green'):

    # shared
    Ts = config.Ts  
    ifsave = config.ifsave
    Po = obsPF.Po[:,0:-1]   # cut off last one, as that represents the position of vechicle 1
    obsRad = obsPF.obsRad
    
    # first vehicle
    t_all = myData.t_all 
    pos_all = myData.pos_all 
    quat_all = myData.quat_all 
    sDes_tr_all = myData.sDes_traj_all 
    waypoints = traj.wps 
    xyzType = traj.xyzType 
    yawType = traj.yawType 

    # second vehicle
    t_all2 = myData2.t_all 
    pos_all2 = myData2.pos_all 
    quat_all2 = myData2.quat_all 
    sDes_tr_all2 = myData2.sDes_traj_all 
    waypoints2 = traj2.wps 
    xyzType2 = traj2.xyzType 
    yawType2 = traj2.yawType 


    # first vehicle 
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
        
    # second vehicle 
    x2 = pos_all2[:,0]
    y2 = pos_all2[:,1]
    z2 = pos_all2[:,2]
    xDes2 = sDes_tr_all2[:,0]
    yDes2 = sDes_tr_all2[:,1]
    zDes2 = sDes_tr_all2[:,2]
    x_wp2 = waypoints2[:,0]
    y_wp2 = waypoints2[:,1]
    z_wp2 = waypoints2[:,2]
    if (config.orient == "NED"):
        z2 = -z2
        zDes2 = -zDes2
        z_wp2 = -z_wp2    
        

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    # first vehicle
    line1, = ax.plot([], [], [], lw=2, color=myColour)
    line2, = ax.plot([], [], [], lw=2, color='gray')
    line3, = ax.plot([], [], [], '--', lw=1, color=myColour)
    
    # second vehicle
    line1b, = ax.plot([], [], [], lw=2, color=myColour2)
    line2b, = ax.plot([], [], [], lw=2, color='gray')
    line3b, = ax.plot([], [], [], '--', lw=1, color=myColour2)

    # Setting the axes properties
    extraEachSide = 0.5
    # find maxes from vehicles
    xmax = np.maximum(x.max(),x2.max())
    ymax = np.maximum(y.max(),y2.max())
    zmax = np.maximum(z.max(),z2.max())
    xmin = np.minimum(x.min(),x2.min())
    ymin = np.minimum(y.min(),x2.min())
    zmin = np.minimum(z.min(),x2.min())
    
    # maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    # mid_x = 0.5*(x.max()+x.min())
    # mid_y = 0.5*(y.max()+y.min())
    # mid_z = 0.5*(z.max()+z.min())
    
    maxRange = 0.5*np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max() + extraEachSide
    mid_x = 0.5*(xmax+xmin)
    mid_y = 0.5*(ymax+ymin)
    mid_z = 0.5*(zmax+zmin)
    
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
    titleVeh1 = ax.text2D(0.05, 0.91, "vehicle 1", color=myColour, transform=ax.transAxes)
    titleVeh2 = ax.text2D(0.05, 0.87, "vehicle 2", color=myColour2, transform=ax.transAxes)

    # first vehicle
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

    # second vehicle
    trajType2 = ''
    yawTrajType2 = ''
    if (xyzType2 == 0):
        trajType2 = 'Hover'
    else:
        ax.scatter(x_wp2, y_wp2, z_wp2, color=myColour2, alpha=1, marker = 'o', s = 25)
        if (xyzType2 == 1 or xyzType2 == 12):
            trajType2 = 'Simple Waypoints'
        else:
            ax.plot(xDes2, yDes2, zDes2, ':', lw=1.3, color=myColour2)
            if (xyzType2 == 2):
                trajType2 = 'Simple Waypoint Interpolation'
            elif (xyzType2 == 3):
                trajType2 = 'Minimum Velocity Trajectory'
            elif (xyzType2 == 4):
                trajType2 = 'Minimum Acceleration Trajectory'
            elif (xyzType2 == 5):
                trajType2 = 'Minimum Jerk Trajectory'
            elif (xyzType2 == 6):
                trajType2 = 'Minimum Snap Trajectory'
            elif (xyzType2 == 7):
                trajType2 = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType2 == 8):
                trajType2 = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType2 == 9):
                trajType2 = 'Minimum Snap Trajectory - Stop'
            elif (xyzType2 == 10):
                trajType2 = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType2 == 1):
                trajType2 = 'Minimum Snap Trajectory - Fast Stop'

    if (yawType2 == 0):
        yawTrajType2 = 'None'
    elif (yawType2 == 1):
        yawTrajType2 = 'Waypoints'
    elif (yawType2 == 2):
        yawTrajType2 = 'Interpolation'
    elif (yawType2 == 3):
        yawTrajType2 = 'Follow'
    elif (yawType2 == 4):
        yawTrajType2 = 'Zero'
    



    # travis add
    #o1 = np.array([-2, -1, -3])                  # obstacle 1 (x,y,z)
    #o2 = np.array([3, -2, 1])               # obstacle 2 (x,y,z)
    #Po = np.vstack((o1,o2)).transpose()     # stack obstacles  
    ax.scatter(np.squeeze(Po[0,:]), np.squeeze(Po[1,:]), -np.squeeze(Po[2,:]), color='red', alpha=1, marker = 'o', s = 100*obsRad)    

    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):

        
        # first vehicle
        # -------------
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
        
        # second vehicle
        # -------------
        time2 = t_all2[i*numFrames]
        pos2 = pos_all2[i*numFrames]
        x2 = pos2[0]
        y2 = pos2[1]
        z2 = pos2[2]

        x_from02 = pos_all2[0:i*numFrames,0]
        y_from02 = pos_all2[0:i*numFrames,1]
        z_from02 = pos_all2[0:i*numFrames,2]
    
        dxm2 = params2["dxm"]
        dym2 = params2["dym"]
        dzm2 = params2["dzm"]
        
        quat2 = quat_all2[i*numFrames]
    
        if (config.orient == "NED"):
            z2 = -z2
            z_from02 = -z_from02
            quat2 = np.array([quat2[0], -quat2[1], -quat2[2], quat2[3]])
    
        R2 = utils.quat2Dcm(quat2)    
        motorPoints2 = np.array([[dxm2, -dym2, dzm2], [0, 0, 0], [dxm2, dym2, dzm2], [-dxm2, dym2, dzm2], [0, 0, 0], [-dxm2, -dym2, dzm2]])
        motorPoints2 = np.dot(R2, np.transpose(motorPoints2))
        motorPoints2[0,:] += x2 
        motorPoints2[1,:] += y2 
        motorPoints2[2,:] += z2 
        
        line1b.set_data(motorPoints2[0,0:3], motorPoints2[1,0:3])
        line1b.set_3d_properties(motorPoints2[2,0:3])
        line2b.set_data(motorPoints2[0,3:6], motorPoints2[1,3:6])
        line2b.set_3d_properties(motorPoints2[2,3:6])
        line3b.set_data(x_from02, y_from02)
        line3b.set_3d_properties(z_from02)
        #titleTime.set_text(u"Time = {:.2f} s".format(time))
        

        return line1, line2, line1b, line2b


    def ini_plot():

        # vehicle 1
        line1.set_data([], [])
        line1.set_3d_properties([])
        line2.set_data([], [])
        line2.set_3d_properties([])
        line3.set_data([], [])
        line3.set_3d_properties([])
        # vehicle 2
        line1b.set_data([], [])
        line1b.set_3d_properties([])
        line2b.set_data([], [])
        line2b.set_3d_properties([])
        line3b.set_data([], [])
        line3b.set_3d_properties([])

        return line1, line2, line3, line1b, line2b, line3b

    
    #my add (init_func messes up, made a new one here)
    line_ani = animation.FuncAnimation(fig, updateLines, blit=False, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames))
 
    # Creating the Animation object
    # ORIGINAL line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    
    if (ifsave):
        # ORIGINAL line_ani.save('Gifs/Raw/animation_{0:.0f}_{1:.0f}.gif'.format(xyzType,yawType), dpi=80, writer='imagemagick', fps=25)
        line_ani.save('Gifs/Raw/animation_multi_{0}_and_{1}.gif'.format(myColour,myColour2), writer=writer) #my add
 
  
        
    plt.show()
    return line_ani