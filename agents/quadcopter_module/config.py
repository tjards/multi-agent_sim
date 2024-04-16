# -*- coding: utf-8 -*-
"""
updated by: P. Travis Jardine, PhD
email: travis.jardine@gmail.com 

original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!

modified for application in this larger project by P. Travis Jardine

"""

import numpy as np
#from .utils import quaternions as quat
#import utils.lemni_tools as lemni_tools

class config():

    def __init__(self):

        # Select Orientation of Quadcopter and Reference Frame
        # ---------------------------
        # "NED" for front-right-down (frd) and North-East-Down
        # "ENU" for front-left-up (flu) and East-North-Up
        self.orient = "NED"
        #self.orient = "ENU"
        
        # Select whether to use gyroscopic precession of the rotors in the quadcopter dynamics
        # ---------------------------
        # Set to False if rotor inertia isn't known (gyro precession has negigeable effect on drone dynamics)
        self.usePrecession = bool(False)
        
        # Sample rate for controller
        # --------------------------- 
        self.Ts         =   0.005  # default 0.005 
        
        # Heading-related parameters
        # --------------------------
        self.heading_type        = 2 #  0 = forward, 1 = random, 2 = centroid 
        self.v_heading_adjust    = 0.10 # min speed at which quadcopter adjust heading
        self.v_heading_saturate  = 0.50 # max speed at which quadcopter adjust heading
   
