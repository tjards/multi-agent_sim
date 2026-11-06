#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diverse agent count benchmark.

Runs 60-second simulations (3000 timesteps) at a range of agent counts
with the optimized path. Scalar baseline run at n=100 only for reference.
Reports wall-clock time, steps/s, and observed scaling exponent.
"""

import sys, os, time, copy, json
import numpy as np, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config.config as cfg


def run_sim(config_data, n_agents, Tf, use_vectorized=True):
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

    tmp = os.path.join(os.path.dirname(__file__), '..', 'config', '_diverse_config.json')
    with open(tmp, 'w') as f:
        json.dump(config_data, f)

    try:
        config = cfg.Config(tmp)
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
        steps = 0
        t0 = time.perf_counter()

        while round(t, 3) < config.Tf:
            kwargs = {}
            T.evolve(t)
            O.evolve(T.targets, A.state, config.nAgents)
            A.evolve(C.cmd, C.pin_matrix, t, config.Ts)
            t += config.Ts
            steps += 1
            kwargs = learner.conductor.update_args(A, C, config.strategy, kwargs)
            kwargs['sorted_neighs'] = Tr.sorted_neighs
            kwargs['i'] = 0
            kwargs['t'] = t
            Tr.update(config.strategy, A.state, T.targets, **kwargs)
            C.commands(A.state, config.strategy, A.centroid, T.targets,
                       O.obstacles_plus, O.walls, Tr.trajectory, config.dynamics, **kwargs)

        elapsed = time.perf_counter() - t0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return elapsed, steps


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
        base = json.load(f)

    Tf = 60.0  # 60 seconds = 3000 timesteps
    n_steps = int(Tf / 0.02)

    agent_counts = [50, 100, 200, 300, 500, 750, 1000]

    print("=" * 72)
    print(f"DIVERSE AGENT BENCHMARK (Tf={Tf}s, {n_steps} timesteps, flocking_saber)")
    print("=" * 72)
    print()

    # scalar baseline at n=100
    print("Scalar baseline (n=100)...")
    t_scalar, _ = run_sim(base, 100, Tf, use_vectorized=False)
    print(f"  n=100 scalar: {fmt_time(t_scalar)} ({n_steps/t_scalar:.0f} steps/s)")
    print()

    print(f"{'n':>6}  {'optimized':>11}  {'steps/s':>9}  {'scaling':>8}  {'vs scalar 100':>14}")
    print("-" * 58)

    prev_n = None
    prev_t = None

    for n in agent_counts:
        t_opt, steps = run_sim(base, n, Tf, use_vectorized=True)
        sps = steps / t_opt

        # scaling exponent from previous point
        if prev_n is not None and prev_t is not None and prev_t > 0 and t_opt > 0:
            exp = np.log(t_opt / prev_t) / np.log(n / prev_n)
            exp_str = f"n^{exp:.2f}"
        else:
            exp_str = ""

        # compare to scalar n=100: how many times faster than scalar at n=100
        scalar_ratio = t_scalar / t_opt if t_opt > 0 else 0

        print(f"{n:>6}  {fmt_time(t_opt):>11}  {sps:>9.0f}  {exp_str:>8}  {scalar_ratio:>13.1f}x")

        prev_n = n
        prev_t = t_opt

    # project 300s from 60s data
    print()
    print("Projected full sim (Tf=300s) from 60s measurements:")
    for n in agent_counts:
        t_60, _ = run_sim(base, n, Tf, use_vectorized=True)
        t_300 = t_60 * 5  # linear extrapolation (same per-step cost)
        print(f"  n={n:>5}: {fmt_time(t_300)}")


if __name__ == '__main__':
    main()
