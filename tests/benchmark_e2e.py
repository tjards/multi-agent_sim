#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end simulation benchmark.

Runs a short flocking_saber simulation at varying agent counts,
timing the full simulation loop (evolve + trajectory + commands).
Compares wall-clock time with and without vectorized planner path.
"""

import sys
import os
import time
import copy
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config.config as cfg


def run_sim(config_data, n_agents, Tf, use_vectorized=True):
    """Run a short sim and return wall-clock seconds."""

    # patch config for this run
    config_data = copy.deepcopy(config_data)
    config_data['simulation']['strategy'] = 'flocking_saber'
    config_data['simulation']['Tf'] = Tf
    config_data['simulation']['Ts'] = 0.02
    config_data['simulation']['Ti'] = 0
    config_data['simulation']['verbose'] = 0
    config_data['simulation']['dimens'] = 3
    config_data['simulation']['experimental_save'] = False
    config_data['agents']['nAgents'] = n_agents
    config_data['agents']['dynamics'] = 'double integrator'
    config_data['agents']['iSpread'] = 40
    config_data['agents']['init_conditions'] = 'random'

    # write temp config
    temp_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_benchmark_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_config_path)

        np.random.seed(config.random_seed)

        # build system components inline (avoid module-level imports that read config)
        # agents
        from agents.agents import Agents
        # monkeypatch config for agents module
        import agents.agents as agents_mod
        agents_mod.agents_config = config_data['agents']
        agents_mod.simulation_config = config_data['simulation']

        AgentsObj = Agents()

        # targets
        import targets.targets as targets_mod
        TargetsObj = targets_mod.Targets(config.nAgents, config.dimens)

        # trajectory
        from planner.trajectory import Trajectory
        TrajectoryObj = Trajectory(config.strategy, TargetsObj.targets, config.nAgents)

        # obstacles
        import obstacles.obstacles as obstacles_mod
        ObstaclesObj = obstacles_mod.Obstacles(TargetsObj.targets)

        # learner
        import learner.conductor
        Learners = learner.conductor.initialize(AgentsObj, config.strategy, config.learning_ctrl, config.Ts, config._data)

        # controller
        import orchestrator
        ControllerObj = orchestrator.Controller(config, AgentsObj.state)
        ControllerObj.learning_agents(config.strategy, Learners)
        TrajectoryObj.load_planners(ControllerObj.planners)

        # if disabling vectorized, monkeypatch the planner
        if not use_vectorized:
            ControllerObj.planners[config.strategy].compute_cmd_vectorized = lambda *a, **kw: None

        # run simulation loop
        t = config.Ti
        t0 = time.perf_counter()

        while round(t, 3) < config.Tf:
            kwargs = {}

            TargetsObj.evolve(t)
            ObstaclesObj.evolve(TargetsObj.targets, AgentsObj.state, config.nAgents)
            AgentsObj.evolve(ControllerObj.cmd, ControllerObj.pin_matrix, t, config.Ts)

            t += config.Ts

            kwargs = learner.conductor.update_args(AgentsObj, ControllerObj, config.strategy, kwargs)
            kwargs['sorted_neighs'] = TrajectoryObj.sorted_neighs
            kwargs['i'] = 0
            kwargs['t'] = t
            TrajectoryObj.update(config.strategy, AgentsObj.state, TargetsObj.targets, **kwargs)

            ControllerObj.commands(
                AgentsObj.state, config.strategy, AgentsObj.centroid,
                TargetsObj.targets, ObstaclesObj.obstacles_plus, ObstaclesObj.walls,
                TrajectoryObj.trajectory, config.dynamics, **kwargs)

        elapsed = time.perf_counter() - t0

    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    return elapsed


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    # short sim for benchmarking (2 seconds of sim time = 100 timesteps at Ts=0.02)
    Tf = 2.0

    agent_counts = [13, 30, 50, 100]

    print(f"End-to-end simulation benchmark (Tf={Tf}s, Ts=0.02, strategy=flocking_saber)")
    print(f"{'n':>6}  {'scalar (s)':>11}  {'vectorized (s)':>14}  {'speedup':>8}  {'steps/s (vec)':>13}")
    print("-" * 62)

    n_steps = int(Tf / 0.02)

    for n in agent_counts:
        # scalar path (vectorized disabled)
        try:
            t_scalar = run_sim(base_config, n, Tf, use_vectorized=False)
        except Exception as e:
            print(f"{n:>6}  ERROR (scalar): {e}")
            continue

        # vectorized path
        try:
            t_vec = run_sim(base_config, n, Tf, use_vectorized=True)
        except Exception as e:
            print(f"{n:>6}  ERROR (vectorized): {e}")
            continue

        speedup = t_scalar / t_vec if t_vec > 0 else float('nan')
        steps_per_sec = n_steps / t_vec if t_vec > 0 else 0

        print(f"{n:>6}  {t_scalar:>11.4f}  {t_vec:>14.4f}  {speedup:>7.2f}x  {steps_per_sec:>13.0f}")

    print()
    print(f"({n_steps} timesteps per run)")


if __name__ == '__main__':
    main()
