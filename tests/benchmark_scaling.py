#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive scaling benchmark for the performance refactoring.

Measures three dimensions across agent counts:
1. End-to-end simulation wall-clock time (vectorized vs scalar)
2. Graph operations (update_A + components + connectivity)
3. History memory allocation

Computes observed scaling exponents between successive data points.
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
from utils.spatial import SpatialIndex
from utils.swarmgraph import Swarmgraph, compute_local_connectivity, convert_A_to_D


# ===========================================================================
# helpers
# ===========================================================================

def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.2f} GB"


def scaling_exp(t1, t2, n1, n2):
    """Compute local scaling exponent: log(t2/t1) / log(n2/n1)."""
    if t1 <= 0 or t2 <= 0 or n1 <= 0 or n2 <= 0 or n1 == n2:
        return ""
    return f"{np.log(t2/t1) / np.log(n2/n1):.2f}"


def time_fn(fn, *args, repeats=2):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


# ===========================================================================
# 1. graph operations benchmark
# ===========================================================================

def brute_force_graph(data, r, n):
    """Original O(n^2) graph operations."""
    r_matrix = r * np.ones((n, n))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(data[0:3, j] - data[0:3, i])
                if dist < r + 0.2:
                    A[i, j] = 1
    D = convert_A_to_D(A)
    degrees = np.sum(A, axis=1).astype(int)
    local_k = {i: int(degrees[i]) for i in range(n)}
    return A


def optimized_graph(data, r, n):
    """New spatial-index backed graph operations."""
    state = np.zeros((6, n))
    state[0:3, :] = data[0:3, :]
    r_matrix = r * np.ones((n, n))
    graph = Swarmgraph(state)
    graph.update_A(data[0:3, :], r_matrix)
    graph.find_connected_components()
    graph.local_k_connectivity = compute_local_connectivity(graph.A)
    return graph.A


# ===========================================================================
# 2. end-to-end sim benchmark
# ===========================================================================

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

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_bench_config.json')
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


# ===========================================================================
# 3. memory benchmark
# ===========================================================================

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

    temp_path = os.path.join(os.path.dirname(__file__), '..', 'config', '_mem2_config.json')
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


# ===========================================================================
# main
# ===========================================================================

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    rng = np.random.default_rng(42)
    radius = 15.0
    spread = 40.0
    Tf_sim = 2.0
    Tf_mem = 10.0
    n_steps = int(Tf_sim / 0.02)

    agent_counts = [10, 50, 100, 200, 500, 1000]

    # -----------------------------------------------------------------------
    print("=" * 78)
    print("MULTI-AGENT SIM PERFORMANCE BENCHMARK")
    print("=" * 78)
    print()

    # -----------------------------------------------------------------------
    print("1. GRAPH OPERATIONS (update_A + components + connectivity)")
    print(f"{'n':>6}  {'old (s)':>10}  {'new (s)':>10}  {'speedup':>8}  {'old O':>7}  {'new O':>7}")
    print("-" * 62)

    old_graph_times = []
    new_graph_times = []

    for n in agent_counts:
        data = rng.uniform(-spread, spread, size=(6, n))

        if n <= 500:
            t_old = time_fn(brute_force_graph, data, radius, n, repeats=2)
        else:
            t_old = float('nan')

        t_new = time_fn(optimized_graph, data, radius, n, repeats=3)

        old_graph_times.append(t_old)
        new_graph_times.append(t_new)

        speedup = t_old / t_new if not np.isnan(t_old) and t_new > 0 else float('nan')
        old_exp = scaling_exp(old_graph_times[-2], t_old, agent_counts[len(old_graph_times)-2], n) if len(old_graph_times) >= 2 else ""
        new_exp = scaling_exp(new_graph_times[-2], t_new, agent_counts[len(new_graph_times)-2], n) if len(new_graph_times) >= 2 else ""

        speedup_str = f"{speedup:.1f}x" if not np.isnan(speedup) else "n/a"
        t_old_str = f"{t_old:.5f}" if not np.isnan(t_old) else "skipped"
        print(f"{n:>6}  {t_old_str:>10}  {t_new:>10.5f}  {speedup_str:>8}  {old_exp:>7}  {new_exp:>7}")

    # -----------------------------------------------------------------------
    print()
    print("2. END-TO-END SIMULATION (Tf=2.0s, Ts=0.02, flocking_saber)")
    print(f"{'n':>6}  {'scalar (s)':>11}  {'optimized (s)':>14}  {'speedup':>8}  {'steps/s':>9}")
    print("-" * 58)

    sim_counts = [n for n in agent_counts if n <= 200]
    scalar_times = []
    vec_times = []

    for n in sim_counts:
        try:
            t_s = run_sim_loop(base_config, n, Tf_sim, use_vectorized=False)
        except Exception:
            t_s = float('nan')
        try:
            t_v = run_sim_loop(base_config, n, Tf_sim, use_vectorized=True)
        except Exception:
            t_v = float('nan')

        scalar_times.append(t_s)
        vec_times.append(t_v)

        speedup = t_s / t_v if t_v > 0 and not np.isnan(t_s) else float('nan')
        sps = n_steps / t_v if t_v > 0 else 0

        print(f"{n:>6}  {t_s:>11.4f}  {t_v:>14.4f}  {speedup:>7.2f}x  {sps:>9.0f}")

    # -----------------------------------------------------------------------
    print()
    print("3. MEMORY (History allocation, Tf=10.0s)")
    print(f"{'n':>6}  {'allocated':>12}  {'would-be dense':>15}  {'reduction':>10}")
    print("-" * 50)

    mem_counts = [n for n in agent_counts if n <= 500]

    for n in mem_counts:
        try:
            alloc = measure_history_memory(base_config, n, Tf_mem)
        except Exception:
            alloc = 0

        nSteps = int(Tf_mem / 0.02 + 1)
        dense_estimate = 4 * nSteps * n * n * 8 + nSteps * 17 * n * 8  # 4 n^2 arrays + linear arrays
        reduction = dense_estimate / alloc if alloc > 0 else 0

        print(f"{n:>6}  {fmt_bytes(alloc):>12}  {fmt_bytes(dense_estimate):>15}  {reduction:>9.0f}x")

    # -----------------------------------------------------------------------
    print()
    print("SCALING EXPONENT KEY: ~1.0=O(n)  ~1.3=O(n log n)  ~2.0=O(n²)")
    print()

    # project full sim times
    if len(vec_times) >= 2 and vec_times[-1] > 0:
        exp = np.log(vec_times[-1] / vec_times[-2]) / np.log(sim_counts[-1] / sim_counts[-2])
        print(f"Projected full sim (Tf=300s, optimized):")
        for n_proj in [100, 500, 1000, 5000]:
            ratio = (n_proj / sim_counts[-1]) ** exp
            t_proj = vec_times[-1] * ratio * (300.0 / Tf_sim)
            if t_proj < 60:
                print(f"  n={n_proj:>5}:  {t_proj:.0f}s")
            elif t_proj < 3600:
                print(f"  n={n_proj:>5}:  {t_proj/60:.1f} min")
            else:
                print(f"  n={n_proj:>5}:  {t_proj/3600:.1f} hr")


if __name__ == '__main__':
    main()
