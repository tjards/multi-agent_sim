#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: brute-force O(n^2) vs spatial-index Swarmgraph.

Measures wall-clock time for update_A + find_connected_components +
local_k_connectivity at increasing agent counts. Prints a table and
computes observed scaling exponents.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.swarmgraph import (
    Swarmgraph, convert_A_to_D, compute_local_connectivity,
    is_point_in_aperature_range
)
from utils.spatial import SpatialIndex


# ---- brute-force reference (original code) ----

def brute_force_full(data, r_matrix, nNodes, slack=0.2):
    """Original O(n^2) update_A + BFS components + exponential k-connectivity."""
    A = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            if i != j:
                dist = np.linalg.norm(data[0:3, j] - data[0:3, i])
                r = r_matrix[i, j] + slack
                if dist < r:
                    A[i, j] = 1
    D = convert_A_to_D(A)

    # BFS connected components
    all_components = []
    visited = []
    for node in range(nNodes):
        if node not in visited:
            component = []
            candidates = np.nonzero(A[node, :].ravel() == 1)[0].tolist()
            component.append(node)
            visited.append(node)
            candidates = list(set(candidates) - set(visited))
            while candidates:
                candidate = candidates.pop(0)
                visited.append(candidate)
                subcandidates = np.nonzero(A[:, candidate].ravel() == 1)[0].tolist()
                component.append(candidate)
                candidates.extend(list(set(subcandidates) - set(candidates) - set(visited)))
            all_components.append(component)

    # node-degree connectivity (matches compute_local_connectivity)
    degrees = np.sum(A, axis=1).astype(int)
    local_k = {i: int(degrees[i]) for i in range(nNodes)}

    return A, D, all_components, local_k


def new_impl_full(data, r_matrix, nNodes):
    """New spatial-index backed implementation."""
    state = np.zeros((6, nNodes))
    state[0:3, :] = data[0:3, :]
    graph = Swarmgraph(state)
    graph.update_A(data[0:3, :], r_matrix)
    graph.find_connected_components()
    graph.local_k_connectivity = compute_local_connectivity(graph.A)
    return graph.A, graph.D, graph.components, graph.local_k_connectivity


# ---- benchmark ----

def time_fn(fn, *args, repeats=3):
    """Return minimum wall-clock time over repeats."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)


def main():
    agent_counts = [10, 20, 50, 100, 200, 500, 1000, 2000]
    radius = 15.0
    spread = 40.0
    rng = np.random.default_rng(42)

    print(f"{'n':>6}  {'old (s)':>10}  {'new (s)':>10}  {'speedup':>8}  {'old O':>7}  {'new O':>7}")
    print("-" * 62)

    old_times = []
    new_times = []

    for n in agent_counts:
        data = rng.uniform(-spread, spread, size=(6, n))
        r_matrix = radius * np.ones((n, n))

        # skip brute-force for very large n (would take too long)
        if n <= 1000:
            t_old = time_fn(brute_force_full, data, r_matrix, n, repeats=2)
        else:
            t_old = float('nan')

        t_new = time_fn(new_impl_full, data, r_matrix, n, repeats=3)

        old_times.append(t_old)
        new_times.append(t_new)

        speedup = t_old / t_new if not np.isnan(t_old) and t_new > 0 else float('nan')

        # compute local scaling exponents (ratio of log(t)/log(n) between successive points)
        old_exp = ""
        new_exp = ""
        if len(old_times) >= 2:
            if not np.isnan(old_times[-1]) and not np.isnan(old_times[-2]) and old_times[-2] > 0:
                old_exp = f"{np.log(old_times[-1]/old_times[-2]) / np.log(agent_counts[-1]/agent_counts[-2]):.2f}"
            if new_times[-2] > 0:
                new_exp = f"{np.log(new_times[-1]/new_times[-2]) / np.log(agent_counts[-1]/agent_counts[-2]):.2f}"

        print(f"{n:>6}  {t_old:>10.5f}  {t_new:>10.5f}  {speedup:>7.1f}x  {old_exp:>7}  {new_exp:>7}")

    print()
    print("Scaling exponent interpretation:")
    print("  ~1.0 = O(n),  ~1.3 = O(n log n),  ~2.0 = O(n^2),  ~3.0 = O(n^3)")


if __name__ == '__main__':
    main()
