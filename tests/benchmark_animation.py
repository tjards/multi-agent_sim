#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animation lattice-rendering benchmark.

Runs a short pinning_lattice simulation at large n, then compares the
legacy per-pair Line2D lattice rendering (one artist per (j, k) pair,
O(n^2) per frame) against the LineCollection path now used by
animation_sim (one collection, O(edges) per frame): artist creation,
per-frame update, and full canvas draw. Finishes with a real
animateMe() end-to-end run.

Reference results (n=500, 2D, up to ~77k edges/frame):
    artist creation : old ~67 s (250k artists) | new <1 ms
    update per frame: old ~1.2 s               | new ~0.12 s
    full canvas draw: old ~12.6 s              | new ~0.28 s
    animateMe end-to-end: ~30 s for a 10-frame gif
"""

import sys
import os
import time
import json
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')

N_AGENTS = 500
TF = 4
NUMFRAMES = 20                 # matches config_animation.json
CONNECTION_THRESH = 10.1
UPDATED_CONNECTIONS = 1
COLOR_LATTICE = ['grey', 'blue']
OLD_FRAMES = 2                 # legacy path is slow; keep it bounded


def run_sim():
    """Run a short n=N_AGENTS sim (plots/animation off) to produce data.h5."""
    cfg_path = os.path.join(REPO_ROOT, 'config', 'config.json')
    backup = cfg_path + '.benchbak'
    shutil.copyfile(cfg_path, backup)
    try:
        with open(cfg_path) as f:
            c = json.load(f)
        c['simulation']['Tf'] = TF
        c['agents']['nAgents'] = N_AGENTS
        c['visualization']['plot_results'] = False
        c['visualization']['animate_results'] = False
        with open(cfg_path, 'w') as f:
            json.dump(c, f, indent=4)
        import main
        t0 = time.perf_counter()
        main.run_simulation()
        print(f'[sim] n={N_AGENTS} Tf={TF}s wall: {time.perf_counter()-t0:.1f}s')
    finally:
        shutil.copyfile(backup, cfg_path)
        os.remove(backup)


# ---- legacy implementation (pre-LineCollection), logic preserved ----

def old_create(ax, nVeh):
    lattices = []
    for j in range(nVeh):
        row = []
        for k in range(nVeh):
            line, = ax.plot([], [], '--', lw=1, color=COLOR_LATTICE[0], alpha=0.3)
            row.append(line)
        lattices.append(row)
    return lattices


def old_update(i, pos, lattices, nVeh, lattices_connections, connectivity):
    for j in range(nVeh):
        for k in range(nVeh):
            line = lattices[j][k]
            if j == k:
                line.set_data([], [])
                continue
            if connectivity[i*NUMFRAMES, j, k] > 0:
                dist = np.linalg.norm(pos[:, j] - pos[:, k])
                thresh = (lattices_connections[i*NUMFRAMES, j, k] + 0.5) if UPDATED_CONNECTIONS == 1 else CONNECTION_THRESH
                if dist <= thresh:
                    line.set_color(COLOR_LATTICE[1]); line.set_alpha(0.6)
                else:
                    line.set_color(COLOR_LATTICE[0]); line.set_alpha(0.3)
                line.set_data([pos[0, j], pos[0, k]], [pos[1, j], pos[1, k]])
            else:
                line.set_data([], [])


# ---- current implementation (animation_sim), logic preserved, 2D ----

def new_update(i, pos, coll, lattices_connections, connectivity):
    A = np.asarray(connectivity[i*NUMFRAMES])
    rows, cols = np.nonzero(A)
    if rows.size == 0:
        coll.set_segments([])
        return
    p = pos[0:2, :].T
    segs = np.stack([p[rows], p[cols]], axis=1)
    dists = np.linalg.norm(pos[:, rows] - pos[:, cols], axis=0)
    if UPDATED_CONNECTIONS == 1:
        thresh = np.asarray(lattices_connections[i*NUMFRAMES])[rows, cols] + 0.5
    else:
        thresh = CONNECTION_THRESH
    connected = dists <= thresh
    colors = np.where(connected[:, None],
                      np.array(mcolors.to_rgba(COLOR_LATTICE[1], 0.6)),
                      np.array(mcolors.to_rgba(COLOR_LATTICE[0], 0.3)))
    coll.set_segments(segs)
    coll.set_color(colors)


def bench_lattice_rendering():
    from data import data_manager
    path = os.path.join(REPO_ROOT, 'data', 'data', 'data.h5')
    _, states_all = data_manager.load_data_HDF5('History', 'states_all', path)
    _, connectivity = data_manager.load_data_HDF5('History', 'connectivity', path)
    _, lattices_connections = data_manager.load_data_HDF5('History', 'lattices', path)
    nVeh = states_all.shape[2]
    n_anim_frames = states_all.shape[0] // NUMFRAMES
    edges = [int(np.count_nonzero(np.asarray(connectivity[i*NUMFRAMES]))) for i in range(n_anim_frames)]
    print(f'[bench] {states_all.shape[0]} steps, n={nVeh}, {n_anim_frames} anim frames, '
          f'edges/frame min..max: {min(edges)}..{max(edges)}')

    # legacy path
    fig, ax = plt.subplots()
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    t0 = time.perf_counter()
    lat_old = old_create(ax, nVeh)
    t_create_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(OLD_FRAMES):
        pos = states_all[i*NUMFRAMES, 0:3, :]
        old_update(i, pos, lat_old, nVeh, lattices_connections, connectivity)
    t_update_old = (time.perf_counter() - t0) / OLD_FRAMES

    t0 = time.perf_counter()
    fig.canvas.draw()
    t_draw_old = time.perf_counter() - t0
    plt.close(fig)

    # collection path
    fig, ax = plt.subplots()
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    t0 = time.perf_counter()
    coll = LineCollection([], linestyles='--', linewidths=1)
    ax.add_collection(coll)
    t_create_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_anim_frames):
        pos = states_all[i*NUMFRAMES, 0:3, :]
        new_update(i, pos, coll, lattices_connections, connectivity)
    t_update_new = (time.perf_counter() - t0) / n_anim_frames

    t0 = time.perf_counter()
    fig.canvas.draw()
    t_draw_new = time.perf_counter() - t0
    plt.close(fig)

    print(f'[bench] artist creation : old {t_create_old:8.3f}s ({nVeh*nVeh} artists) | new {t_create_new*1000:7.2f}ms (1 collection)')
    print(f'[bench] update per frame: old {t_update_old:8.3f}s ({OLD_FRAMES} frames)   | new {t_update_new*1000:7.2f}ms ({n_anim_frames} frames)')
    print(f'[bench] full canvas draw: old {t_draw_old:8.3f}s               | new {t_draw_new:7.3f}s')


def bench_animateMe():
    import visualization.animation_sim as animation_sim
    path = os.path.join(REPO_ROOT, 'data', 'data', 'data.h5')
    t0 = time.perf_counter()
    animation_sim.animateMe(path, 0.02, 2, 'pinning_lattice')
    wall = time.perf_counter() - t0
    gif = os.path.join(REPO_ROOT, 'visualization', 'animations', 'animation.gif')
    print(f'[e2e] animateMe at n={N_AGENTS}: {wall:.1f}s, gif size '
          f'{os.path.getsize(gif)/1e6:.1f} MB')


def main():
    run_sim()
    bench_lattice_rendering()
    bench_animateMe()


if __name__ == '__main__':
    main()
