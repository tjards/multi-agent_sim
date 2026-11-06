#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-length simulation benchmark.

Runs complete 300-second simulations (15,000 timesteps) at n=100 and n=200
with both scalar and optimized paths. Reports actual wall-clock time.
"""

import sys
import os
import time
import copy
import json
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config.config as cfg


def run_full_sim(config_data, n_agents, Tf, use_vectorized=True):
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

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_fullsim_config.json')
    with open(temp_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_path)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        import agents.agents as agents_mod
        agents_mod.agents_config = config_data['agents']
        agents_mod.simulation_config = config_data['simulation']
        from agents.agents import Agents
        A = Agents()

        import targets.targets as t_mod
        T = t_mod.Targets(config.nAgents, config.dimens)

        from planner.trajectory import Trajectory
        Tr = Trajectory(config.strategy, T.targets, config.nAgents)

        import obstacles.obstacles as o_mod
        O = o_mod.Obstacles(T.targets)

        import learner.conductor
        L = learner.conductor.initialize(A, config.strategy, config.learning_ctrl, config.Ts, config._data)

        import orchestrator
        C = orchestrator.Controller(config, A.state)
        C.learning_agents(config.strategy, L)
        Tr.load_planners(C.planners)

        if not use_vectorized:
            C.planners[config.strategy].compute_cmd_vectorized = lambda *a, **kw: None

        t = config.Ti
        n_steps = 0
        t0 = time.perf_counter()
        last_report = t0

        while round(t, 3) < config.Tf:
            kwargs = {}
            T.evolve(t)
            O.evolve(T.targets, A.state, config.nAgents)
            A.evolve(C.cmd, C.pin_matrix, t, config.Ts)
            t += config.Ts
            n_steps += 1
            kwargs = learner.conductor.update_args(A, C, config.strategy, kwargs)
            kwargs['sorted_neighs'] = Tr.sorted_neighs
            kwargs['i'] = 0
            kwargs['t'] = t
            Tr.update(config.strategy, A.state, T.targets, **kwargs)
            C.commands(A.state, config.strategy, A.centroid, T.targets,
                       O.obstacles_plus, O.walls, Tr.trajectory, config.dynamics, **kwargs)

            # progress report every 30 seconds of sim time
            now = time.perf_counter()
            if now - last_report > 10:
                elapsed = now - t0
                pct = t / config.Tf * 100
                sps = n_steps / elapsed
                eta = (config.Tf - t) / config.Ts / sps if sps > 0 else 0
                print(f"    {pct:5.1f}% ({n_steps} steps, {elapsed:.1f}s elapsed, {sps:.0f} steps/s, ETA {eta:.0f}s)")
                last_report = now

        elapsed = time.perf_counter() - t0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return elapsed, n_steps


def fmt_time(s):
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        return f"{s/60:.1f} min"
    else:
        return f"{s/3600:.1f} hr"


def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    Tf = 300.0  # full 300-second simulation

    print("=" * 70)
    print("FULL SIMULATION BENCHMARK (Tf=300s, Ts=0.02, 15000 timesteps)")
    print("=" * 70)
    print()

    results = []

    for n in [100, 200]:
        print(f"--- n={n} agents, OPTIMIZED ---")
        t_opt, steps = run_full_sim(base_config, n, Tf, use_vectorized=True)
        sps_opt = steps / t_opt
        print(f"  Result: {fmt_time(t_opt)} ({steps} steps, {sps_opt:.0f} steps/s)")
        print()

        # only run scalar for n=100 (n=200 scalar would take ~15+ min)
        if n <= 100:
            print(f"--- n={n} agents, SCALAR (original) ---")
            t_scalar, steps = run_full_sim(base_config, n, Tf, use_vectorized=False)
            sps_scalar = steps / t_scalar
            speedup = t_scalar / t_opt
            print(f"  Result: {fmt_time(t_scalar)} ({steps} steps, {sps_scalar:.0f} steps/s)")
            print(f"  Speedup: {speedup:.1f}x")
            results.append((n, t_scalar, t_opt, speedup))
        else:
            # project scalar from n=100 scaling
            if results:
                ref_n, ref_scalar, _, _ = results[0]
                t_scalar_est = ref_scalar * (n / ref_n) ** 1.79
                speedup_est = t_scalar_est / t_opt
                print(f"  Scalar estimate: {fmt_time(t_scalar_est)} (projected O(n^1.79))")
                print(f"  Speedup estimate: {speedup_est:.1f}x")
                results.append((n, t_scalar_est, t_opt, speedup_est))
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'n':>6}  {'Scalar':>12}  {'Optimized':>12}  {'Speedup':>8}")
    print("-" * 44)
    for n, t_s, t_o, sp in results:
        note = "" if n <= 100 else " (est)"
        print(f"{n:>6}  {fmt_time(t_s):>12}{note}  {fmt_time(t_o):>12}  {sp:>7.1f}x")


if __name__ == '__main__':
    main()
