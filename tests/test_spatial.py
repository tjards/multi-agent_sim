#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for utils.spatial.SpatialIndex.

Validates that the KD-tree backed spatial index produces identical
neighbor results to brute-force cdist, across random configurations.
"""

import sys
import os
import numpy as np
import pytest
from scipy.spatial.distance import cdist

# ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.spatial import SpatialIndex


# ---- helpers ----

def brute_force_pairs(positions, radius):
    """Reference: all (i, j) pairs with i < j and dist < radius."""
    pts = positions.T  # (n, 3)
    dists = cdist(pts, pts)
    n = pts.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < radius:
                pairs.append((i, j))
    return sorted(pairs)


def brute_force_neighbors(positions, index, radius):
    """Reference: neighbors of agent `index` within radius (excludes self)."""
    pts = positions.T  # (n, 3)
    dists = np.linalg.norm(pts - pts[index], axis=1)
    return sorted([j for j in range(pts.shape[0]) if j != index and dists[j] < radius])


# ---- fixtures ----

@pytest.fixture
def random_positions_3d():
    """50 agents scattered in a 100x100x100 volume."""
    rng = np.random.default_rng(42)
    return rng.uniform(-50, 50, size=(3, 50))


@pytest.fixture
def random_positions_2d():
    """30 agents in 2D (z=0)."""
    rng = np.random.default_rng(99)
    pos = rng.uniform(-20, 20, size=(3, 30))
    pos[2, :] = 0.0
    return pos


@pytest.fixture
def clustered_positions():
    """Two tight clusters far apart."""
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(loc=0, scale=1, size=(3, 20))
    cluster_b = rng.normal(loc=100, scale=1, size=(3, 20))
    return np.hstack([cluster_a, cluster_b])


# ---- tests: query_all_pairs ----

class TestQueryAllPairs:

    def test_matches_brute_force_3d(self, random_positions_3d):
        radius = 25.0
        idx = SpatialIndex(random_positions_3d)
        pairs = idx.query_all_pairs(radius)
        pairs_list = sorted([tuple(row) for row in pairs.tolist()])
        expected = brute_force_pairs(random_positions_3d, radius)
        assert pairs_list == expected

    def test_matches_brute_force_2d(self, random_positions_2d):
        radius = 10.0
        idx = SpatialIndex(random_positions_2d)
        pairs = idx.query_all_pairs(radius)
        pairs_list = sorted([tuple(row) for row in pairs.tolist()])
        expected = brute_force_pairs(random_positions_2d, radius)
        assert pairs_list == expected

    def test_no_pairs_at_tiny_radius(self, random_positions_3d):
        idx = SpatialIndex(random_positions_3d)
        pairs = idx.query_all_pairs(0.0001)
        assert pairs.shape == (0, 2)

    def test_all_pairs_at_huge_radius(self, random_positions_3d):
        n = random_positions_3d.shape[1]
        idx = SpatialIndex(random_positions_3d)
        pairs = idx.query_all_pairs(1e6)
        expected_count = n * (n - 1) // 2
        assert pairs.shape[0] == expected_count

    def test_clusters_isolated(self, clustered_positions):
        """With radius smaller than cluster separation, no cross-cluster pairs."""
        idx = SpatialIndex(clustered_positions)
        pairs = idx.query_all_pairs(10.0)
        # all pairs should be within-cluster (both < 20 or both >= 20)
        for i, j in pairs:
            same_cluster = (i < 20 and j < 20) or (i >= 20 and j >= 20)
            assert same_cluster, f"Cross-cluster pair found: ({i}, {j})"


# ---- tests: query_neighbors ----

class TestQueryNeighbors:

    def test_matches_brute_force(self, random_positions_3d):
        radius = 25.0
        idx = SpatialIndex(random_positions_3d)
        for i in range(random_positions_3d.shape[1]):
            result = sorted(idx.query_neighbors(i, radius))
            expected = brute_force_neighbors(random_positions_3d, i, radius)
            assert result == expected, f"Mismatch at agent {i}"

    def test_excludes_self(self, random_positions_3d):
        idx = SpatialIndex(random_positions_3d)
        for i in range(random_positions_3d.shape[1]):
            neighbors = idx.query_neighbors(i, 1e6)
            assert i not in neighbors


# ---- tests: query_ball_tree ----

class TestQueryBallTree:

    def test_matches_per_agent_query(self, random_positions_3d):
        radius = 25.0
        idx = SpatialIndex(random_positions_3d)
        all_neighs = idx.query_ball_tree(radius)
        for i in range(random_positions_3d.shape[1]):
            assert sorted(all_neighs[i]) == sorted(idx.query_neighbors(i, radius))

    def test_length_equals_n_agents(self, random_positions_3d):
        idx = SpatialIndex(random_positions_3d)
        all_neighs = idx.query_ball_tree(25.0)
        assert len(all_neighs) == random_positions_3d.shape[1]


# ---- tests: query_pairs_with_distances ----

class TestQueryPairsWithDistances:

    def test_distances_match_manual(self, random_positions_3d):
        radius = 25.0
        idx = SpatialIndex(random_positions_3d)
        pairs, dists = idx.query_pairs_with_distances(radius)
        pts = random_positions_3d.T
        for k in range(pairs.shape[0]):
            i, j = pairs[k]
            expected_dist = np.linalg.norm(pts[j] - pts[i])
            assert abs(dists[k] - expected_dist) < 1e-12

    def test_all_distances_within_radius(self, random_positions_3d):
        radius = 25.0
        idx = SpatialIndex(random_positions_3d)
        _, dists = idx.query_pairs_with_distances(radius)
        assert np.all(dists < radius)

    def test_empty_at_tiny_radius(self, random_positions_3d):
        idx = SpatialIndex(random_positions_3d)
        pairs, dists = idx.query_pairs_with_distances(0.0001)
        assert pairs.shape == (0, 2)
        assert dists.shape == (0,)


# ---- tests: edge cases ----

class TestEdgeCases:

    def test_single_agent(self):
        pos = np.array([[1.0], [2.0], [3.0]])
        idx = SpatialIndex(pos)
        assert idx.n_agents == 1
        assert idx.query_neighbors(0, 100.0) == []
        assert idx.query_all_pairs(100.0).shape == (0, 2)

    def test_two_agents_within_range(self):
        pos = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        idx = SpatialIndex(pos)
        pairs = idx.query_all_pairs(2.0)
        assert pairs.shape[0] == 1
        assert tuple(pairs[0]) == (0, 1)

    def test_two_agents_outside_range(self):
        pos = np.array([[0.0, 100.0], [0.0, 0.0], [0.0, 0.0]])
        idx = SpatialIndex(pos)
        pairs = idx.query_all_pairs(2.0)
        assert pairs.shape[0] == 0

    def test_coincident_agents(self):
        """Two agents at the exact same position."""
        pos = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        idx = SpatialIndex(pos)
        pairs = idx.query_all_pairs(1.0)
        assert pairs.shape[0] == 1
        neighbors = idx.query_neighbors(0, 1.0)
        assert neighbors == [1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
