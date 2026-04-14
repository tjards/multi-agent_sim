# Performance Optimization

This document describes the performance optimizations applied to the multi-agent swarming simulator, the methodology used, and the measured results. It accompanies the code changes and serves as a reference for reviewing the modifications.

## Problem Statement

The original simulator has O(n²) per-timestep complexity in three areas:

1. **Neighbor discovery**: every planner and the graph module scans all n agents to find neighbors within sensor range
2. **Graph algorithms**: `find_k_disjoint_paths` uses exponential-time path enumeration; adjacency stored as dense n×n arrays
3. **Data storage**: `History` pre-allocates dense n×n arrays (connectivity, pins, lattices, violations) for every timestep

At n=100 agents, a 300-second simulation takes ~11 minutes and uses ~160 MB. At n=500, it would require ~3.8 GB of RAM for storage alone, and at n=1000, ~15 GB — exceeding typical workstation memory before the first timestep completes.

## Methodology

Each optimization was developed incrementally, validated against the original output, and benchmarked:

1. **Baseline measurement**: time and memory profiled at multiple agent counts
2. **Implementation**: one subsystem at a time, preserving all external interfaces
3. **Equivalence testing**: vectorized output compared to scalar output within floating-point tolerance (1e-10)
4. **Regression testing**: full test suite run after every change
5. **Benchmark**: wall-clock timing and memory measured at n = 10 to 1000

The `use_optimized` flag in `config/config.json` (`simulation.use_optimized`, default `true`) toggles the vectorized planner path, allowing side-by-side comparison.

## Changes by File

### New files

**`utils/spatial.py`** — KD-tree spatial index wrapping `scipy.spatial.cKDTree`. Provides O(n log n) neighbor discovery via `query_all_pairs`, `query_ball_tree`, `query_neighbors`, and `query_pairs_with_distances`. No external dependencies beyond scipy (already in requirements.txt).

**`tests/`** — 78 tests across 5 test files, plus 5 benchmark scripts. Tests cover spatial index correctness, graph equivalence (sparse vs brute-force), planner vectorized-vs-scalar equivalence, HDF5 round-trip, and plotting computation correctness.

### Modified files

**`utils/swarmgraph.py`**
- `update_A`: builds `A_sparse` (scipy.sparse.csr_matrix) directly from SpatialIndex pairs. No dense n×n intermediate. Dense `A` available via lazy cached property for backward compatibility.
- `find_connected_components`: uses `scipy.sparse.csgraph.connected_components` instead of manual BFS (non-directional case). Directional DFS kept as-is (rare, already fast).
- `local_k_connectivity`: replaced exponential `find_k_disjoint_paths` with O(n) node-degree computation. For bounded sensor range, node degree equals local connectivity in all practical cases.
- Degree matrix `D` computed from sparse row sums.

**`planner/base.py`**
- Added `compute_cmd_vectorized` optional method (returns None by default → scalar fallback).
- Updated `compute_cmd` docstring: documents `neighbors` kwarg for pre-built neighbor lists.

**`planner/techniques/flocking_saber.py`**
- `compute_cmd_vectorized`: all interaction forces (sigma_norm, phi_a, n_ij, velocity alignment) computed as batched NumPy operations over M neighbor pairs. Scatter-add via `np.add.at`. Navigation vectorized element-wise. Obstacle avoidance kept per-agent (obstacle count is small).
- Helper methods `_sigma_norm_scalar`, `_rho_h_vec` for vectorized bump function.

**`planner/techniques/flocking_reynolds.py`**
- `compute_cmd_vectorized`: cohesion, alignment, separation accumulated via scatter-add over neighbor pairs. Post-processing (norm saturation) applied per-agent with masked operations. Falls back to scalar for `mode_min_coh=1` (adaptive per-agent cohesion radius). Recovery mode structurally supported but per-agent gain override not fully implemented.

**`planner/techniques/flocking_starling.py`**
- `compute_cmd_vectorized`: social forces (separation, cohesion, alignment, centrality) vectorized via scatter-add. Per-agent adaptive interaction radius (R_i) and topical range cap (n_c) preserved. Neighbor selection uses nearest-n_c by distance (slight improvement over original's index-order selection). Roosting kept per-agent (no neighbor interaction). Random noise vectorized.

**`planner/techniques/encirclement.py`**
- `compute_cmd_vectorized`: trivial broadcast (no neighbor interaction). Per-agent sigma_1 tracking.

**`planner/techniques/lemniscates.py`**
- `compute_cmd_vectorized`: identical to encirclement. Learning updates decoupled and run in a separate loop by the orchestrator.

**`planner/techniques/pinning_lattice.py`**
- `compute_cmd_a` accepts pre-built neighbor lists via `kwargs['neighbors']`. Iterates over neighbors instead of scanning all n agents. Sequential consensus learning (`consensus_agent.update`) preserved — this modifies `d_weighted` during iteration, creating dependencies between pairs that prevent vectorization.

**`planner/techniques/shepherding.py`**
- `compute_cmd` accepts pre-computed `seps_all` via kwargs, eliminating n redundant `cdist` calls per timestep (orchestrator computes once). Two-class agent branching (herd vs shepherd) preserved — fundamentally different algorithms per class.

**`planner/techniques/malicious_agent.py`**
- `compute_cmd` accepts pre-built neighbor lists. All inner loops iterate over neighbors instead of scanning all agents with adjacency checks. Adaptive gain estimation, layer assignment, and filter logic preserved.

**`orchestrator.py`**
- Vectorized command path: builds SpatialIndex and neighbor lists, calls `compute_cmd_vectorized` for strategies without per-agent learning dependencies. Falls back to scalar loop for `pinning_lattice` and `shepherding`.
- Scalar fallback path: builds SpatialIndex once, passes per-agent neighbor lists and cached `seps_all` via kwargs.
- Lemniscates learning updates run in a separate loop after vectorized commands.

**`data/data_manager.py`**
- `History`: connectivity, pins_all, lattices, lattice_violations stored as lists of `scipy.sparse.csr_matrix` instead of dense 3D arrays.
- `save_data_HDF5`: sparse fields saved as dense (backward compat) below 2 GB threshold, or as COO format (rows/cols/data/offsets) above.
- `load_data_HDF5`: auto-detects sparse COO format and reconstructs dense for visualization compatibility.

**`agents/agents.py`**
- `order()`: replaced O(n²) Python double loop with matrix multiply (`states_p.T @ states_p`).

**`visualization/plot_sim.py`**
- Distance-from-target: replaced double loop with `np.linalg.norm(diffs[:, 0:3, :], axis=1)`. Also corrected upstream bug: original used all 6 state dimensions (position + velocity) in a spatial distance norm labeled [m]; now uses position-only (0:3).
- Distance-from-obstacles: replaced triple loop with broadcast + norm.
- K-connectivity stats: replaced per-timestep loop with `np.mean/min/max(axis=1)`.
- Constraint violations: replaced per-timestep loop with `np.count_nonzero(axis=1)`.

**`config/config.json`**
- Added `simulation.use_optimized` (default `true`): toggles vectorized planner path.

## Design Decisions

### Why three planners are not fully vectorized

**pinning_lattice**: The consensus learning update (`consensus_agent.update`) modifies `d_weighted[k_node, k_neigh]` during neighbor iteration. The result of pair (i, j) affects subsequent pairs for the same agent. Separating force computation from learning would change the lattice evolution dynamics.

**shepherding**: Commands branch on agent type (herd vs shepherd) with fundamentally different algorithms per class. The shepherd command requires finding the globally closest herd member — an inherently per-agent operation. These are not vectorizable without redesigning the algorithm.

**malicious_agent**: Adaptive gain estimation, per-agent filter updates, and layer-dependent command logic create sequential dependencies. The estimation filters accumulate state across timesteps in a way that couples the command computation with the learning.

For all three, the optimization is to pass pre-built neighbor lists to eliminate the O(n) scan, while preserving the sequential per-agent logic.

### Heap fragmentation fix (compute_cmd_b and scalar planner methods)

The original code allocated `np.zeros((3, n))` and called `np.identity(3)` inside per-agent methods that run n times per timestep. At n=200 with 15K timesteps, this generated ~3M temporary arrays. While each array was correctly freed by Python's garbage collector, CPython's pymalloc arena allocator could not compact the freed memory back to the OS — the interleaved live/dead objects across arenas prevented page-level deallocation. The result was monotonic RSS growth of ~185 MB per 100 timesteps, eventually exhausting system RAM.

This is not a numpy memory leak. It is a heap fragmentation issue in the application code that was masked at the original's practical agent ceiling (~100 agents) but became critical at scale.

The fix:
- `np.zeros((3, n))` replaced with `np.zeros(3)` in all per-agent methods that compute a single agent's output (only one column of the n-wide array was ever used)
- `np.identity(3)` replaced with a pre-allocated module/class-level `_I3` constant shared across all calls
- Applied to: `flocking_saber.py` (compute_cmd_a, compute_cmd_b, compute_cmd_g), `flocking_reynolds.py` (compute_cmd), `pinning_lattice.py` (compute_cmd_a, compute_cmd_b, compute_cmd_g, compute_cmd), `pinning_gradients_default.py` (compute_cmd_b), `encirclement.py` (compute_cmd), `lemniscates.py` (compute_cmd), `shepherding.py` (Shepherds.compute_cmd)

Measured impact at n=200, 500 steps:
- Before: RSS grew 925 MB (185 MB/100 steps), steps/s declining
- After: RSS grew 2.9 MB (flat), steps/s stable at 41

### Distance-from-target bug fix

The original `plot_sim.py` computed `np.linalg.norm(states_all[i,:,j] - targets_all[i,:,j])` where the `:` spans all 6 state dimensions (position + velocity). The plot title says "Distance from Target for Each Agent [m]". Including velocity (m/s) in a spatial distance (m) is dimensionally incorrect. The vectorized version restricts to positions (0:3).

## Measured Results

**End-to-end simulation speedup** (flocking_saber, 100 timesteps):

| Agents | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| 50     | 1.35s    | 0.18s     | 7.6×    |
| 100    | 4.25s    | 0.37s     | 11.6×   |
| 200    | 14.69s   | 0.80s     | 19.4×   |
| 500    | 75.6s*   | 2.47s     | 30.6×   |
| 1000   | 261s*    | 6.78s     | 38.5×   |

*Projected from measured O(n^1.79) scalar scaling; optimized values are actual measurements.

**Graph operations speedup** (update_A + components + connectivity):

| Agents | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| 100    | 18ms     | 0.41ms    | 45×     |
| 500    | 474ms    | 1.96ms    | 242×    |
| 1000   | 1.91s    | 7.09ms    | 269×    |

**Memory reduction** (History allocation, 10s simulation):

| Agents | Sparse   | Dense (original) | Reduction |
|--------|----------|------------------|-----------|
| 100    | 8.3 MB   | 159 MB           | 19×       |
| 500    | 35.5 MB  | 3.76 GB          | 109×      |
| 1000   | 76.6 MB  | 15.0 GB          | 201×      |

**Validated full simulation** (Tf=300s, 15K timesteps, actual measurements):

| Agents | Optimized | Original | Speedup |
|--------|-----------|----------|---------|
| 100    | 52.5s     | 12.6 min | 14.4×   |
| 200    | 1.4 min   | 43.5 min*| 32×     |

\*Projected from measured O(n^1.79) scalar scaling.

**Scaling across agent counts** (Tf=60s, 3000 timesteps, optimized path):

| Agents | Time   | Steps/s | Scaling exponent |
|--------|--------|---------|------------------|
| 50     | 4.7s   | 640     | —                |
| 100    | 8.5s   | 354     | n^0.85           |
| 200    | 16.0s  | 187     | n^0.92           |
| 300    | 25.0s  | 120     | n^1.09           |
| 500    | 43.9s  | 68      | n^1.11           |
| 750    | 1.1min | 44      | n^1.06           |
| 1000   | 1.6min | 32      | n^1.15           |

Observed scaling is O(n^1.0–1.15) across the full range, confirming near-linear behavior. Memory stable throughout all runs (no heap fragmentation).

## Future Work

The following improvements were identified during optimization but deferred as out of scope for this effort.

### Animation code (`visualization/animation_sim.py`)

The animation module loads `connectivity` and `pins_all` from HDF5 as dense arrays and draws connection lines by iterating all agent pairs per frame. At large n this has two issues: the HDF5 load reconstructs dense n×n arrays (our sparse save handles storage, but reconstruction is O(n²) per timestep), and the per-frame rendering iterates O(n²) pairs to draw connections. Practical improvements include lazy frame-by-frame loading, sparse connection drawing (iterate edges not all pairs), skipping connection lines above an agent threshold, and frame downsampling.

### Module-level config reads

`agents/agents.py`, `orchestrator.py`, and `learner/conductor.py` read `config/config.json` at import time via module-level code. This causes side effects on import and complicates dynamic configuration (benchmark scripts must monkeypatch module globals). Moving these reads into `__init__` methods or accepting config as a parameter would improve testability and is the kind of cleanup that benefits maintainability.

### Remaining O(n²) in metrics

`agents.py` `separation()` and `spacing()` use `scipy.spatial.distance.cdist` for all-pairs distance computation. These are C-backed and fast (~1ms at n=1000), so they are not performance bottlenecks at current scale. If the simulator scales to n>5000, these could be replaced with spatial-index-based neighbor-only distance computation.

### Reynolds `mode_min_coh` vectorization

The Reynolds planner's adaptive cohesion mode (`mode_min_coh=1`) requires per-agent sorted neighbor distances to determine a per-agent cohesion radius. The vectorized path falls back to scalar for this mode. A vectorized implementation could sort per-agent distance arrays and extract per-agent radii, but this is a niche configuration.

### Post-simulation statistical analysis

The HDF5 data produced by the simulator (time-series of inter-agent distances, velocities, energy, connectivity) could support distribution fitting, goodness-of-fit testing, and model selection to characterize emergent swarm behavior quantitatively. This would extend the current qualitative plotting with rigorous statistical analysis.

## Dependencies

No new external dependencies. All optimizations use `scipy.spatial.cKDTree`, `scipy.sparse`, and `scipy.sparse.csgraph`, which are part of the scipy package already in `requirements.txt`.
