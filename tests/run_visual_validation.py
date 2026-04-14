#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual validation: run a short sim and produce plots.

Runs flocking_saber with 13 agents for 10 seconds, saves HDF5 data,
and generates all plots. Check visualization/plots/ for output.
"""

import sys
import os
import json
import copy
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config.config as cfg

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # override for a short saber sim
    config_data['simulation']['strategy'] = 'flocking_saber'
    config_data['simulation']['Tf'] = 10
    config_data['simulation']['Ts'] = 0.02
    config_data['simulation']['Ti'] = 0
    config_data['simulation']['verbose'] = 1
    config_data['simulation']['dimens'] = 3
    config_data['simulation']['experimental_save'] = False
    config_data['agents']['nAgents'] = 13
    config_data['agents']['dynamics'] = 'double integrator'

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_visual_config.json')
    with open(temp_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_path)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # build system
        import orchestrator
        import learner.conductor

        import agents.agents as agents_mod
        agents_mod.agents_config = config_data['agents']
        agents_mod.simulation_config = config_data['simulation']

        Agents, Targets, Trajectory, Obstacles, Learners = orchestrator.build_system(config)
        Controller = orchestrator.Controller(config, Agents.state)
        Controller.learning_agents(config.strategy, Learners)
        Trajectory.load_planners(Controller.planners)

        # data storage
        from data import data_manager

        data_directory = 'data/data/'
        os.makedirs(data_directory, exist_ok=True)
        data_file_path = os.path.join(data_directory, 'data.h5')

        Database = data_manager.History(Agents, Targets, Obstacles, Controller, Trajectory,
                                         config.Ts, config.Tf, config.Ti, config.f)

        # run sim
        t = config.Ti
        i = 1
        print(f'Starting visual validation sim: {config.nAgents} agents, Tf={config.Tf}s')

        while round(t, 3) < config.Tf:
            kwargs = {}
            Targets.evolve(t)
            Obstacles.evolve(Targets.targets, Agents.state, config.nAgents)
            Agents.evolve(Controller.cmd, Controller.pin_matrix, t, config.Ts)

            Database.update(Agents, Targets, Obstacles, Controller, Trajectory, t, config.f, i)

            t += config.Ts
            i += 1

            kwargs = learner.conductor.update_args(Agents, Controller, config.strategy, kwargs)
            kwargs['sorted_neighs'] = Trajectory.sorted_neighs
            kwargs['i'] = i
            kwargs['t'] = t
            Trajectory.update(config.strategy, Agents.state, Targets.targets, **kwargs)
            Controller.commands(Agents.state, config.strategy, Agents.centroid,
                                Targets.targets, Obstacles.obstacles_plus, Obstacles.walls,
                                Trajectory.trajectory, config.dynamics, **kwargs)

            if round(t, 2).is_integer():
                print(f'  {round(t,1)} of {config.Tf} sec')

        # save data
        print('Saving data...')
        data_manager.save_data_HDF5(Database, data_file_path)

        # generate plots
        print('Generating plots...')
        plots_dir = 'visualization/plots'
        os.makedirs(plots_dir, exist_ok=True)

        import visualization.plot_sim as plot_sim
        plot_sim.plotMe(data_file_path)

        print(f'Done. Check {plots_dir}/ for output PNGs.')

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    main()
