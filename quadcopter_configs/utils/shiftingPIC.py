#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This program implements shifting planar inequality constraints
for application with 3D obstacle avoidance as defined here:
    
    P. T. Jardine, S. Givigi, and S. Yousefi, 
    Planar Inequality Constraints for Stable, Collision-free Model Predictive Control of a Quadcopter ,
    IFAC-PapersOnLine, Volume 50, Issue 1, July 2017, Pages 9095-9100

author: P. Travis Jardine, PhD
email: travis.jardine@gmail.com 

Created on Wed Oct 21 20:19:56 2020

@author: tjards

"""
import numpy as np
from scipy import optimize as opt
#from matplotlib import pyplot as plt
#from matplotlib.animation import FuncAnimation
#import mpl_toolkits.mplot3d.axes3d as p3 
#from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' #my add - this path needs to be added
from matplotlib import animation

# Set up formatting for the movie files
# ------------------------------------
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

# move target
# -----------
def moveTarget(state, Po, xv, xt, r, ro, rb):
    
    # search for closest obstacle
    # --------------------------
    distance = 10    # max range of sensor
    dindex = 0      # default
    for k in range(0,len(Po[0,:])):
        distance_new = np.linalg.norm(state[0:3]-Po[:,k])
        if distance_new < distance:
            dindex = k
            
    xo = np.reshape(Po[:,dindex], (1,3))
    
    # compute constraints
    # --------------------
    xp, d, A, b = computeCons(xv, xo, r, ro, rb)

    # find new optimal point 
    # ----------------------
    cx = runOpt(xv, xt, xp, A, b)
    
    return cx

# compute constraints 
# -------------------
def computeCons(xv, xo, r, ro, rb):
    
    #initialize constraint stuff
    nStates=xv.shape[1]
    nObs = xo.shape[0]
    xp = np.zeros([nObs,nStates])
    A = np.zeros([nObs,nStates])
    b = np.zeros([nObs,1])
    d = np.zeros([nObs,1])
    
    for i in range(0,nObs):
        #xp_, d_ = QP.computeXp(xv,xo[[i],:],r,ro,rb)
        xp[i,:], d[i,0] = computeXp(xv,xo[[i],:],r,ro,rb)
        #xp[i,:] = xp_
        #d[i,0] = d_
        A[i,:] = xo[i,:] - xp[i,:]
        b[i,0] = np.dot(xp[i,:],(xo[i,:]-xp[i,:]))

    return xp, d, A, b

# computes the centerpoint of the constraint
# ------------------------------------------
def computeXp(x,xo,r,ro,rb):
               
        diff = x-xo
        d = np.linalg.norm(diff)                            # distance between obs and veh    
        #xp = xo + np.multiply(np.divide(ro+r+rb,d),(x-xo))  # compute centerpoint
        if diff.all() == 0:
            xp = xo
        else:
            xp = xo + np.multiply(np.divide(ro+r+rb,d),(x-xo))  # compute centerpoint

        return xp, d

# find an optimal point around obstacle 
# -------------------------------------    
def runOpt(x, xt, xp, A, b):
    
    # cons = {'type':'ineq',
    #         'fun':lambda x: b - np.dot(A,x),
    #         'jac':lambda x: -A}
    
    
    # I think this needs to be 1-D
    cons = {'type':'ineq',
            'fun':lambda x: - np.dot(A[0,:],x.transpose()) + b[0]}
     
    # unconstrained case
    # ux = opt.minimize(f, x, constraints=None, args = [xt[0], xt[1]])
    
    # constrained case
    #if len(x) == 2:
    if x.shape[1] == 2:
        #cx = opt.minimize(f, x, constraints=cons, args = [xt[0], xt[1]])
        cx = opt.minimize(f, xp, constraints=cons, args = [xt[0,0], xt[0,1]])
    #elif len(x) == 3:
    elif x.shape[1] == 3:
        #cx = opt.minimize(f, x, constraints=cons, args = [xt[0], xt[1], xt[2]])
        cx = opt.minimize(f, xp, constraints=cons, args = [xt[0,0], xt[0,1], xt[0,2]])

    #cx = opt.minimize(f, x0, bounds=bnds, constraints=cons)

    return cx


# the objective function
# ----------------------
def f(x, xt):
    
    x = np.array(x, ndmin = 2)
    xt = np.array(xt, ndmin = 2)
    
    #if len(x) == 2:
    if x.shape[1] == 2:
        #cost = ((x[0]-xt[0])**2 + (x[1]-xt[1])**2)
        cost = ((x[0,0]-xt[0,0])**2 + (x[0,1]-xt[0,1])**2)
    #elif len(x) == 3:
    elif x.shape[1] == 3:
        #cost = ((x[0]-xt[0])**2 + (x[1]-xt[1])**2 + (x[2]-xt[2])**2)
        cost = ((x[0,0]-xt[0,0])**2 + (x[0,1]-xt[0,1])**2 + (x[0,2]-xt[0,2])**2)
        
    return cost 
    #return ((x[0]-xt[0])**2 + (x[1]-xt[1])**2)
    #return ((x[0]-xt[0])**2 + (x[1]-xt[1])**2 + (x[2]-xt[2])**2)


# animate the obstacle avoidance
# -----------------------------
# def myAnimation(myData_QP,nStates,name):

#     # initialize data store 
#     # ---------------------
#     xv = myData_QP[0,0:nStates] 
#     xo = myData_QP[0,nStates:2*nStates] 
#     xt = myData_QP[0,2*nStates:3*nStates] 
#     A = myData_QP[0,3*nStates:4*nStates] 
#     b = myData_QP[0,4*nStates:4*nStates+1] 
#     ux = myData_QP[0,4*nStates+1:5*nStates+1] 
#     cx = myData_QP[0,5*nStates+1:6*nStates+1]

#     # initiate the figure
#     # -------------------
    
#     plotSize = 5
  
#     # if nStates == 3:
#     #     fig = plt.figure()
#     #     ax = p3.Axes3D(fig)
#     #     #ax = fig.add_subplot(111, projection ='3d')
    
# #if nStates == 2:
#     fig, ax = plt.subplots()
#     x = np.linspace(-plotSize, plotSize, 100)
#     y = np.linspace(-plotSize, plotSize, 100)
#     X, Y = np.meshgrid(x, y)
#     # if nStates == 2: # only run contours for 2-D case
#     #     Z = f(np.vstack([X.ravel(), Y.ravel()]), xt).reshape((100,100))
#     #     ax.contour(X, Y, Z, np.arange(-2,20, 1), alpha=0.3)
    
#     # initialize lines that need updating
#     # ------------------------------------
    
#     #if nStates == 2:

#     line_obs = ax.scatter(xo[0], xo[1], s=400, marker = 'o', alpha = 0.2, color = 'blue')
#     line_veh = ax.scatter(xv[0], xv[1], s=400, marker = 'o', alpha = 0.2, color = 'green')
#     line_tar = ax.scatter(xt[0], xt[1], s=400, marker = 'o', alpha = 0.2, color = 'black')
#     line_ux = ax.scatter(xt[0], xt[1], s=100, marker = 'x', alpha = 1, color = 'black')
#     line_cx = ax.scatter(xt[0], xt[1], s=100, marker = 'x', alpha = 1, color = 'red')
#     line_xv2cx, = ax.plot([xv[0],xt[0]],[xv[1],xt[1]], 'g:', linewidth=1)
#     line_xt2cx, = ax.plot([xt[0],xt[0]],[xt[1],xt[1]], 'b:', linewidth=1)
    
#     # if nStates == 3:
    
#     #     line_obs = ax.scatter(xo[0], xo[1], xo[2], s=400, marker = 'o', alpha = 0.2, color = 'blue')
#     #     line_veh = ax.scatter(xv[0], xv[1], xv[2], s=400, marker = 'o', alpha = 0.2, color = 'green')
#     #     line_tar = ax.scatter(xt[0], xt[1], xt[2], s=400, marker = 'o', alpha = 0.2, color = 'black')
#     #     line_ux = ax.scatter(xt[0], xt[1], xt[2], s=100, marker = 'x', alpha = 1, color = 'black')
#     #     line_cx = ax.scatter(xt[0], xt[1], xt[2], s=100, marker = 'x', alpha = 1, color = 'red')
#     #     line_xv2cx, = ax.plot([xv[0],xt[0]],[xv[1],xt[1]],[xv[2],xt[2]], 'g:', linewidth=1)
#     #     line_xt2cx, = ax.plot([xt[0],xt[0]],[xt[1],xt[1]],[xt[2],xt[2]], 'b:', linewidth=1)
    
    

#     #%% this is the constraint illustration
#     # --------------------------------------
#     #yy = -np.divide(A[0,0],A[0,1])*x+np.divide(b,A[0,1])
#     if nStates == 2:
#         yy = -np.divide(A[0],A[1])*x+np.divide(b,A[1])
#         line_const, = ax.plot(x, yy, 'r:', linewidth=1)
        
#     if nStates == 3:
#         yy = 0
    
#     plt.axis([-plotSize,plotSize,-plotSize,plotSize])
#     plt.xlabel('x-direction')
#     plt.ylabel('y-direction')
#     plt.title('obstacle avoid')
#     #plt.show()
    
#     #%% update the lines
#     # ------------------
#     def update(i):
      
#         # extract data
#         # --------------
        
#         xv = myData_QP[i,0:nStates] 
#         xo = myData_QP[i,nStates:2*nStates] 
#         xt = myData_QP[i,2*nStates:3*nStates] 
#         A = myData_QP[i,3*nStates:4*nStates] 
#         b = myData_QP[i,4*nStates:4*nStates+1] 
#         ux = myData_QP[i,4*nStates+1:5*nStates+1] 
#         cx = myData_QP[i,5*nStates+1:6*nStates+1]
        
#         # xv = myData_QP[i,0:2] 
#         # xo = myData_QP[i,2:4] 
#         # xt = myData_QP[i,4:6] 
#         # A = myData_QP[i,6:8] 
#         # b = myData_QP[i,8:9] 
#         # ux = myData_QP[i,9:11] 
#         # cx = myData_QP[i,11:13]
        
#         # update the lines
#         # some remnants of legacy 2D case kept
#         # (for nostalgia)
#         # -----------------
        
#         #if nStates == 2:
            
#         line_obs.set_offsets([xo[0],xo[1]])
#         line_veh.set_offsets([xv[0],xv[1]])
#         line_tar.set_offsets([xt[0],xt[1]])
#         line_ux.set_offsets([ux[0], ux[1]])
#         line_cx.set_offsets([cx[0], cx[1]])
#         line_xv2cx.set_xdata([xv[0],cx[0]])
#         line_xv2cx.set_ydata([xv[1],cx[1]])
#         line_xt2cx.set_xdata([xt[0],cx[0]])
#         line_xt2cx.set_ydata([xt[1],cx[1]])
#         #yy = -np.divide(A[0,0],A[0,1])*x+np.divide(b,A[0,1])
        
#         #if nStates == 3:
        
#             # line_obs.set_offsets([xo[0],xo[1],xo[2]])
#             # line_veh.set_offsets([xv[0],xv[1],xv[2]])
#             # line_tar.set_offsets([xt[0],xt[1],xt[1]])
#             # line_ux.set_offsets([ux[0], ux[1],ux[2]])
#             # line_cx.set_offsets([cx[0], cx[1],cx[2]])
#             # line_xv2cx.set_xdata([xv[0],cx[0]])
#             # line_xv2cx.set_ydata([xv[1],cx[1]])
#             # #line_xv2cx.set_3d_properties([xv[2],cx[2]])
#             # line_xt2cx.set_xdata([xt[0],cx[0]])
#             # line_xt2cx.set_ydata([xt[1],cx[1]])
#             # #line_xt2cx.set_3d_properties([xt[2],cx[2]])
#             # #yy = -np.divide(A[0,0],A[0,1])*x+np.divide(b,A[0,1])
                        
#             #line_obs._offsets3d = (xo[0],xo[1],xo[2])
#             #line_veh._offsets3d = (xv[0], xv[1], xv[2])
#             #line_tar._offsets3d = (xt[0],xt[1],xt[1])
#             #line_ux._offsets3d = (ux[0], ux[1],ux[2])
#             #line_cx._offsets3d = (cx[0], cx[1],cx[2])
#             #line_xv2cx.set_xdata([xv[0],cx[0]])
#             #line_xv2cx.set_ydata([xv[1],cx[1]])
#             #line_xv2cx.set_3d_properties([xv[2],cx[2]])
#             #line_xt2cx.set_xdata([xt[0],cx[0]])
#             #line_xt2cx.set_ydata([xt[1],cx[1]])
#             #line_xt2cx.set_3d_properties([xt[2],cx[2]])
#             #yy = -np.divide(A[0,0],A[0,1])*x+np.divide(b,A[0,1])
        
#         if nStates == 2:
#             yy = -np.divide(A[0],A[1])*x+np.divide(b,A[1])
#             line_const.set_xdata(x)
#             line_const.set_ydata(yy)    
        
    
    
#     #%% build the animation
#     # ---------------------
#     anim = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=200, blit=False)
#     anim.save('test{}.gif'.format(name), writer=writer) #my add







#%% LEGACY stuff

    # cons = ({'type': 'eq',
    #          'fun' : lambda x: np.array([x[0]**3 - x[1]]),
    #          'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
    #         {'type': 'ineq',
    #          'fun' : lambda x: np.array([x[1] - (x[0]-1)**4 - 2])})    
    # bnds = ((0.5, 1.5), (1.5, 2.5))



    #xo = np.reshape(obsPF.Po[:,dindex], (1,3))
    #cx = QP.moveTarget(xv, xo, xt, 0.1, 0.1, 0.7)