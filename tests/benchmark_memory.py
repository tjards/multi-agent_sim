#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory profiling for the simulation.

Measures peak resident memory (RSS) during a short simulation at varying
agent counts, broken down by component. Reports the n^2 storage cost
directly by computing the size of the dense History arrays.
"""

import sys
import os
import copy
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config.config as cfg


def estimate_history_memory(n_agents, Tf, Ts):
    """Estimate memory for History arrays without actually allocating them."""
    nSteps = int(Tf / Ts + 1)
    bytes_per_float = 8

    # per-timestep arrays that scale with n (O(n) per step)
    linear = {
        'states_all':       nSteps * 6 * n_agents,
        'cmds_all':         nSteps * 3 * n_agents,
        'targets_all':      nSteps * 6 * n_agents,
        'centroid_all':     nSteps * 3 * 1,
        'lemni_all':        nSteps * 2 * n_agents,
        'metrics_order_all': nSteps * 12,
        't_all':            nSteps,
        'f_all':            nSteps,
    }

    # per-timestep arrays that scale with n^2 (O(n^2) per step)
    quadratic = {
        'connectivity':         nSteps * n_agents * n_agents,
        'pins_all':             nSteps * n_agents * n_agents,
        'lattices':             nSteps * n_agents * n_agents,
        'lattice_violations':   nSteps * n_agents * n_agents,
    }

    linear_bytes = sum(linear.values()) * bytes_per_float
    quadratic_bytes = sum(quadratic.values()) * bytes_per_float

    return nSteps, linear, quadratic, linear_bytes, quadratic_bytes


def measure_actual_memory(config_data, n_agents, Tf):
    """Run a sim and measure actual peak memory using History object sizes."""
    import tracemalloc

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

    temp_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_mem_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_config_path)
        np.random.seed(config.random_seed)

        import agents.agents as agents_mod
        agents_mod.agents_config = config_data['agents']
        agents_mod.simulation_config = config_data['simulation']
        from agents.agents import Agents
        AgentsObj = Agents()

        import targets.targets as targets_mod
        TargetsObj = targets_mod.Targets(config.nAgents, config.dimens)

        from planner.trajectory import Trajectory
        TrajectoryObj = Trajectory(config.strategy, TargetsObj.targets, config.nAgents)

        import obstacles.obstacles as obstacles_mod
        ObstaclesObj = obstacles_mod.Obstacles(TargetsObj.targets)

        import learner.conductor
        Learners = learner.conductor.initialize(AgentsObj, config.strategy, config.learning_ctrl, config.Ts, config._data)

        import orchestrator
        ControllerObj = orchestrator.Controller(config, AgentsObj.state)
        ControllerObj.learning_agents(config.strategy, Learners)
        TrajectoryObj.load_planners(ControllerObj.planners)

        # now allocate History and measure
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        from data import data_manager
        Database = data_manager.History(AgentsObj, TargetsObj, ObstaclesObj, ControllerObj,
                                        TrajectoryObj, config.Ts, config.Tf, config.Ti, config.f)

        snapshot_after = tracemalloc.take_snapshot()

        # compute diff
        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        history_alloc = sum(s.size_diff for s in stats if s.size_diff > 0)

        # also measure individual array sizes
        array_sizes = {}
        for attr_name in ['connectivity', 'pins_all', 'lattices', 'lattice_violations',
                          'states_all', 'cmds_all', 'targets_all', 'centroid_all',
                          'metrics_order_all', 't_all']:
            val = getattr(Database, attr_name, None)
            if isinstance(val, np.ndarray):
                array_sizes[attr_name] = val.nbytes
            elif isinstance(val, list):
                array_sizes[attr_name] = sys.getsizeof(val)

        tracemalloc.stop()

    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    return history_alloc, array_sizes


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    Tf = 10.0  # 10 seconds of sim time
    Ts = 0.02

    print(f"Memory analysis (Tf={Tf}s, Ts={Ts})")
    print()

    # theoretical estimates across a range of agent counts
    print("=== Theoretical estimates (History allocation only) ===")
    print(f"{'n':>6}  {'O(n) arrays':>12}  {'O(n²) arrays':>13}  {'Total':>10}  {'n² fraction':>11}")
    print("-" * 62)

    for n in [13, 50, 100, 200, 500, 1000, 5000, 10000]:
        nSteps, _, _, lin_b, quad_b = estimate_history_memory(n, Tf, Ts)
        total = lin_b + quad_b
        frac = quad_b / total * 100 if total > 0 else 0
        print(f"{n:>6}  {fmt_bytes(lin_b):>12}  {fmt_bytes(quad_b):>13}  {fmt_bytes(total):>10}  {frac:>10.1f}%")

    print()

    # actual measurement for feasible sizes
    print("=== Actual measured allocation (tracemalloc) ===")
    print(f"{'n':>6}  {'History alloc':>14}  {'connectivity':>13}  {'pins_all':>13}  {'lattices':>13}  {'states_all':>12}")
    print("-" * 80)

    for n in [13, 50, 100]:
        try:
            alloc, sizes = measure_actual_memory(base_config, n, Tf)
            print(f"{n:>6}  {fmt_bytes(alloc):>14}  "
                  f"{fmt_bytes(sizes.get('connectivity', 0)):>13}  "
                  f"{fmt_bytes(sizes.get('pins_all', 0)):>13}  "
                  f"{fmt_bytes(sizes.get('lattices', 0)):>13}  "
                  f"{fmt_bytes(sizes.get('states_all', 0)):>12}")
        except Exception as e:
            print(f"{n:>6}  ERROR: {e}")


if __name__ == '__main__':
    main()
