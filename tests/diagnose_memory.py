#!/usr/bin/env python3
"""
Memory leak diagnostic.

Runs a SINGLE short sim (n=200, 500 steps) and tracks memory growth
per timestep using tracemalloc. Identifies the top allocations.

Run this in isolation (not after other sims) for clean measurements.
"""

import sys, os, time, copy, json, gc, tracemalloc
import numpy as np, random, psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config.config as cfg

def get_rss_mb():
    return psutil.Process().memory_info().rss / 1024**2

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    with open(config_path, 'r') as f:
        base = json.load(f)

    n = 200
    Tf = 10.0  # 500 steps only

    base['simulation']['strategy'] = 'flocking_saber'
    base['simulation']['Tf'] = Tf
    base['simulation']['Ts'] = 0.02
    base['simulation']['Ti'] = 0
    base['simulation']['verbose'] = 0
    base['simulation']['dimens'] = 3
    base['simulation']['experimental_save'] = False
    base['agents']['nAgents'] = n
    base['agents']['dynamics'] = 'double integrator'
    base['agents']['iSpread'] = 40
    base['agents']['init_conditions'] = 'random'

    tmp = os.path.join(os.path.dirname(__file__), '..', 'config', '_diag_config.json')
    with open(tmp, 'w') as f:
        json.dump(base, f)

    try:
        config = cfg.Config(tmp)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        import agents.agents as agents_mod
        agents_mod.agents_config = base['agents']
        agents_mod.simulation_config = base['simulation']
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

        gc.collect()
        rss_start = get_rss_mb()
        tracemalloc.start()

        print(f"n={n}, Tf={Tf}s ({int(Tf/0.02)} steps)")
        print(f"RSS at start: {rss_start:.1f} MB")
        print(f"{'step':>6}  {'RSS(MB)':>8}  {'traced(MB)':>10}  {'delta(MB)':>9}  {'steps/s':>8}")
        print("-" * 50)

        t = config.Ti
        step = 0
        t0 = time.perf_counter()
        last_rss = rss_start

        while round(t, 3) < config.Tf:
            kwargs = {}
            T.evolve(t)
            O.evolve(T.targets, A.state, config.nAgents)
            A.evolve(C.cmd, C.pin_matrix, t, config.Ts)
            t += config.Ts
            step += 1
            kwargs = learner.conductor.update_args(A, C, config.strategy, kwargs)
            kwargs['sorted_neighs'] = Tr.sorted_neighs
            kwargs['i'] = 0
            kwargs['t'] = t
            Tr.update(config.strategy, A.state, T.targets, **kwargs)
            C.commands(A.state, config.strategy, A.centroid, T.targets,
                       O.obstacles_plus, O.walls, Tr.trajectory, config.dynamics, **kwargs)

            if step % 100 == 0:
                gc.collect()
                rss = get_rss_mb()
                current, _ = tracemalloc.get_traced_memory()
                elapsed = time.perf_counter() - t0
                sps = step / elapsed
                delta = rss - last_rss
                print(f"{step:>6}  {rss:>8.1f}  {current/1024**2:>10.1f}  {delta:>+8.1f}  {sps:>8.0f}")
                last_rss = rss

        # final snapshot
        gc.collect()
        rss_end = get_rss_mb()
        snap = tracemalloc.take_snapshot()
        print(f"\nFinal RSS: {rss_end:.1f} MB (grew {rss_end - rss_start:.1f} MB)")
        print("\nTop 10 allocations by size:")
        for s in snap.statistics('lineno')[:10]:
            print(f"  {s}")

        tracemalloc.stop()

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == '__main__':
    main()
