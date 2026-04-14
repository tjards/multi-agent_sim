#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended benchmark: actual optimized timing at n=500 and n=1000.

Scalar path is not run at these sizes (would take minutes per data point).
Instead, scalar timing and memory are projected from measured O(n²) scaling
at smaller n, and the projected improvement is reported.
"""

import sys
import os
import time
import copy
import json
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config.config as cfg


def run_sim_loop(config_data, n_agents, Tf, use_vectorized=True):
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

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_ext_config.json')
    with open(temp_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_path)
        np.random.seed(config.random_seed)

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
        t0 = time.perf_counter()
        while round(t, 3) < config.Tf:
            kwargs = {}
            T.evolve(t)
            O.evolve(T.targets, A.state, config.nAgents)
            A.evolve(C.cmd, C.pin_matrix, t, config.Ts)
            t += config.Ts
            kwargs = learner.conductor.update_args(A, C, config.strategy, kwargs)
            kwargs['sorted_neighs'] = Tr.sorted_neighs
            kwargs['i'] = 0
            kwargs['t'] = t
            Tr.update(config.strategy, A.state, T.targets, **kwargs)
            C.commands(A.state, config.strategy, A.centroid, T.targets,
                       O.obstacles_plus, O.walls, Tr.trajectory, config.dynamics, **kwargs)
        elapsed = time.perf_counter() - t0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return elapsed


def measure_history_memory(config_data, n_agents, Tf):
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

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_ext_mem_config.json')
    with open(temp_path, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(temp_path)
        np.random.seed(config.random_seed)

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

        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        from data import data_manager
        Database = data_manager.History(A, T, O, C, Tr, config.Ts, config.Tf, config.Ti, config.f)
        snap_after = tracemalloc.take_snapshot()
        stats = snap_after.compare_to(snap_before, 'lineno')
        alloc = sum(s.size_diff for s in stats if s.size_diff > 0)
        tracemalloc.stop()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return alloc


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


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

    Tf = 2.0
    Tf_mem = 10.0
    n_steps = int(Tf / 0.02)

    # reference points from earlier benchmarks (measured on this machine)
    # used to project scalar O(n²) scaling
    scalar_ref_n = 200
    scalar_ref_time = None  # will measure

    print("=" * 80)
    print("EXTENDED BENCHMARK: ACTUAL OPTIMIZED + PROJECTED SCALAR AT LARGE n")
    print("=" * 80)
    print()

    # --- first, measure reference points at n=100 and n=200 (both paths) ---
    print("Phase 1: Measuring reference points (scalar + optimized, n=100 and n=200)")
    print("-" * 80)

    ref_data = {}
    for n in [100, 200]:
        t_s = run_sim_loop(base_config, n, Tf, use_vectorized=False)
        t_v = run_sim_loop(base_config, n, Tf, use_vectorized=True)
        ref_data[n] = {'scalar': t_s, 'optimized': t_v}
        print(f"  n={n:>4}: scalar={t_s:.4f}s  optimized={t_v:.4f}s  speedup={t_s/t_v:.1f}x")

    # compute scalar scaling exponent from reference points
    scalar_exp = np.log(ref_data[200]['scalar'] / ref_data[100]['scalar']) / np.log(200 / 100)
    opt_exp = np.log(ref_data[200]['optimized'] / ref_data[100]['optimized']) / np.log(200 / 100)

    print(f"\n  Observed scaling exponents (n=100 -> n=200):")
    print(f"    Scalar:    O(n^{scalar_exp:.2f})")
    print(f"    Optimized: O(n^{opt_exp:.2f})")

    # --- now run optimized-only at large n ---
    print()
    print("Phase 2: Actual optimized timing at large n (scalar NOT run)")
    print("-" * 80)
    print(f"{'n':>6}  {'optimized':>11}  {'scalar (est)':>13}  {'speedup (est)':>14}  {'steps/s':>9}  {'note'}")
    print("-" * 80)

    for n in [100, 200, 500, 1000]:
        # run optimized
        t_v = run_sim_loop(base_config, n, Tf, use_vectorized=True)

        # scalar: actual for reference sizes, projected for large
        if n in ref_data:
            t_s = ref_data[n]['scalar']
            note = "measured"
        else:
            # project scalar from n=200 reference using observed O(n^exp)
            t_s = ref_data[200]['scalar'] * (n / 200) ** scalar_exp
            note = f"projected O(n^{scalar_exp:.2f})"

        speedup = t_s / t_v if t_v > 0 else 0
        sps = n_steps / t_v if t_v > 0 else 0

        print(f"{n:>6}  {t_v:>10.4f}s  {t_s:>12.2f}s  {speedup:>13.1f}x  {sps:>9.0f}  {note}")

    # --- memory ---
    print()
    print("Phase 3: Memory (actual sparse vs projected dense)")
    print("-" * 80)
    print(f"{'n':>6}  {'sparse (actual)':>16}  {'dense (projected)':>18}  {'reduction':>10}")
    print("-" * 60)

    for n in [100, 200, 500, 1000]:
        alloc = measure_history_memory(base_config, n, Tf_mem)
        nSteps = int(Tf_mem / 0.02 + 1)
        dense_est = 4 * nSteps * n * n * 8 + nSteps * 17 * n * 8
        reduction = dense_est / alloc if alloc > 0 else 0

        print(f"{n:>6}  {fmt_bytes(alloc):>16}  {fmt_bytes(dense_est):>18}  {reduction:>9.0f}x")

    # --- full sim projections ---
    print()
    print("Phase 4: Projected full simulation (Tf=300s)")
    print("-" * 80)
    print(f"{'n':>6}  {'optimized (proj)':>17}  {'scalar (proj)':>14}  {'speedup':>8}  {'memory (sparse)':>16}  {'memory (dense)':>15}")
    print("-" * 80)

    for n in [100, 200, 500, 1000, 5000]:
        # optimized projection from n=200 measured
        t_v_2s = ref_data[200]['optimized'] * (n / 200) ** opt_exp
        t_v_300 = t_v_2s * (300.0 / Tf)

        # scalar projection
        t_s_2s = ref_data[200]['scalar'] * (n / 200) ** scalar_exp
        t_s_300 = t_s_2s * (300.0 / Tf)

        speedup = t_s_300 / t_v_300 if t_v_300 > 0 else 0

        # memory projections (Tf=300)
        nSteps_300 = int(300 / 0.02 + 1)
        dense_mem = 4 * nSteps_300 * n * n * 8 + nSteps_300 * 17 * n * 8
        # sparse: roughly linear with n (from measured data, ~70 bytes/agent/step)
        sparse_mem = nSteps_300 * n * 70 + nSteps_300 * 17 * n * 8

        print(f"{n:>6}  {fmt_time(t_v_300):>17}  {fmt_time(t_s_300):>14}  {speedup:>7.0f}x  {fmt_bytes(sparse_mem):>16}  {fmt_bytes(dense_mem):>15}")


if __name__ == '__main__':
    main()
