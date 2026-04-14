#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial index for efficient neighbor discovery.

@author: OldCrow

Wraps scipy.spatial.cKDTree to replace O(n^2) brute-force pairwise scans
with O(n log n) construction and O(k log n) per-agent range queries.

Usage:
    index = SpatialIndex(positions)       # positions shape: (3, n)
    pairs = index.query_all_pairs(r)      # (M, 2) array of (i, j) pairs within radius r
    neighs = index.query_ball_tree(r)     # list of neighbor lists per agent
    neighs_i = index.query_neighbors(i,r) # neighbors of agent i within radius r
"""

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    """KD-tree backed spatial index for agent positions.

    Parameters
    ----------
    positions : ndarray, shape (3, n)
        Agent positions in 3D (or 2D with z=0). Columns are agents.
    """

    def __init__(self, positions):
        # cKDTree expects (n, 3)
        self._points = np.ascontiguousarray(positions.T)
        self._tree = cKDTree(self._points)
        self.n_agents = self._points.shape[0]

    def query_neighbors(self, index, radius):
        """Return neighbor indices for a single agent.

        Parameters
        ----------
        index : int
            Agent index.
        radius : float
            Search radius.

        Returns
        -------
        list[int]
            Neighbor indices (excludes self).
        """
        neighbors = self._tree.query_ball_point(self._points[index], radius)
        neighbors = [j for j in neighbors if j != index]
        return neighbors

    def query_all_pairs(self, radius):
        """Return all neighbor pairs within radius.

        Parameters
        ----------
        radius : float
            Search radius.

        Returns
        -------
        ndarray, shape (M, 2)
            Each row is an (i, j) pair with i < j and ||pos_i - pos_j|| < radius.
            Returns empty (0, 2) array if no pairs found.
        """
        pairs = self._tree.query_pairs(radius, output_type='ndarray')
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.intp)
        return pairs

    def query_ball_tree(self, radius):
        """Return per-agent neighbor lists.

        Parameters
        ----------
        radius : float
            Search radius.

        Returns
        -------
        list[list[int]]
            neighbors[i] is the list of neighbor indices for agent i (excludes self).
        """
        raw = self._tree.query_ball_tree(self._tree, radius)
        # remove self from each list
        return [[j for j in raw[i] if j != i] for i in range(self.n_agents)]

    def query_pairs_with_distances(self, radius):
        """Return all neighbor pairs and their distances.

        Parameters
        ----------
        radius : float
            Search radius.

        Returns
        -------
        pairs : ndarray, shape (M, 2)
            Each row is (i, j) with i < j.
        distances : ndarray, shape (M,)
            Euclidean distance for each pair.
        """
        pairs = self.query_all_pairs(radius)
        if pairs.shape[0] == 0:
            return pairs, np.empty(0)
        diffs = self._points[pairs[:, 1]] - self._points[pairs[:, 0]]
        distances = np.linalg.norm(diffs, axis=1)
        return pairs, distances
