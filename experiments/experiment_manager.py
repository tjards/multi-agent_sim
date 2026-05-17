#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:32:32 2025

@author: tjards
"""

import os
import shutil
from datetime import datetime

def save_experiment():

    # source file paths
    config_src = os.path.join("config", "config.json")
    data_src   = os.path.join("data", "data", "data.h5")
    anim_src   = os.path.join("visualization", "animations", "animation.gif")
    plots_src  = os.path.join("visualization", "plots")
    
    # generate timestamped folder name with seconds
    timestamp = datetime.now().strftime("experiment_%y%m%d-%H%M%S")
    base_folder = os.path.join("experiments", timestamp)
    
    # create the directory
    os.makedirs(base_folder, exist_ok=True)
    plots_dst = os.path.join(base_folder, "plots")
    os.makedirs(plots_dst, exist_ok=True)
    
    # destination paths
    config_dst = os.path.join(base_folder, "config.json")
    data_dst   = os.path.join(base_folder, "data.h5")
    anim_dst   = os.path.join(base_folder, "animation.gif")
    
    # copy files
    shutil.copyfile(config_src, config_dst)
    shutil.copyfile(data_src, data_dst)
    shutil.copyfile(anim_src, anim_dst)
    
    # copy all plot files
    for file_name in os.listdir(plots_src):
        src_path = os.path.join(plots_src, file_name)
        dst_path = os.path.join(plots_dst, file_name)
        if os.path.isfile(src_path):
            shutil.copyfile(src_path, dst_path)
    
    print(f"Experiment files saved to: {base_folder}")