#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for vectorized plotting computations in plot_sim.py.

Compares vectorized numpy operations against the original scalar loop
implementations to verify identical results.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---- original (scalar) reference implementations ----

def ref_distance_from_target(states_all, targets_all):
    """Reference distance-from-target using position dimensions only (0:3).

    The upstream original used all 6 state dimensions (position + velocity)
    in the norm, which is dimensionally incorrect for a spatial distance [m].
    The vectorized version corrects this by restricting to dims 0:3.
    """
    nSteps = states_all.shape[0]
    nAgents = states_all.shape[2]
    radii = np.zeros([nAgents, nSteps])
    for i in range(nSteps):
        for j in range(nAgents):
            radii[j, i] = np.linalg.norm(states_all[i, 0:3, j] - targets_all[i, 0:3, j])
    return radii


def ref_distance_from_obstacles(states_all, obstacles_all):
    """Original triple-loop distance from obstacles."""
    nSteps = states_all.shape[0]
    nAgents = states_all.shape[2]
    nObs = obstacles_all.shape[2]
    radii_o = np.zeros([nAgents, nSteps, nObs])
    radii_o_means = np.zeros([nAgents, nSteps])
    radii_o_means2 = np.zeros([nSteps])

    for i in range(nSteps):
        for j in range(nAgents):
            for k in range(nObs):
                radii_o[j, i, k] = np.linalg.norm(states_all[i, 0:3, j] - obstacles_all[i, 0:3, k])
            radii_o_means[j, i] = np.mean(radii_o[j, i, :])
        radii_o_means2[i] = np.mean(radii_o_means[:, i])

    return radii_o, radii_o_means2


def ref_k_connectivity_stats(local_k_connectivity):
    """Original per-timestep loop for mean/min/max."""
    nSteps = local_k_connectivity.shape[0]
    temp_means = np.zeros(nSteps)
    temp_mins = np.zeros(nSteps)
    temp_maxs = np.zeros(nSteps)
    for i in range(nSteps):
        temp_means[i] = np.mean(local_k_connectivity[i, :].ravel())
        temp_mins[i] = np.min(local_k_connectivity[i, :].ravel())
        temp_maxs[i] = np.max(local_k_connectivity[i, :].ravel())
    return temp_means, temp_mins, temp_maxs


def ref_count_violations(lattice_violations):
    """Original per-timestep loop for counting nonzeros."""
    nSteps = lattice_violations.shape[0]
    count_violations = np.zeros((nSteps, 1))
    for i in range(nSteps):
        count_violations[i, :] = np.count_nonzero(lattice_violations[i, :, :])
    return count_violations


# ---- vectorized implementations (matching plot_sim.py) ----

def vec_distance_from_target(states_all, targets_all):
    diffs = states_all - targets_all
    radii = np.linalg.norm(diffs[:, 0:3, :], axis=1).T
    return radii


def vec_distance_from_obstacles(states_all, obstacles_all):
    agent_pos = states_all[:, 0:3, :, np.newaxis]
    obs_pos = obstacles_all[:, 0:3, np.newaxis, :]
    radii_o = np.linalg.norm(agent_pos - obs_pos, axis=1)  # (nSteps, nAgents, nObs)
    radii_o_means = np.mean(radii_o, axis=2)
    radii_o_means2 = np.mean(radii_o_means, axis=1)
    return radii_o.transpose(1, 0, 2), radii_o_means2  # transpose to (nAgents, nSteps, nObs)


def vec_k_connectivity_stats(local_k_connectivity):
    temp_means = np.mean(local_k_connectivity, axis=1)
    temp_mins = np.min(local_k_connectivity, axis=1)
    temp_maxs = np.max(local_k_connectivity, axis=1)
    return temp_means, temp_mins, temp_maxs


def vec_count_violations(lattice_violations):
    nSteps = lattice_violations.shape[0]
    return np.count_nonzero(
        lattice_violations.reshape(nSteps, -1), axis=1).reshape(-1, 1)


# ---- fixtures ----

@pytest.fixture
def sim_data():
    """Synthetic simulation data for plot testing."""
    rng = np.random.default_rng(42)
    nSteps = 50
    nAgents = 15
    nObs = 3

    states_all = rng.uniform(-20, 20, size=(nSteps, 6, nAgents))
    targets_all = rng.uniform(-5, 5, size=(nSteps, 6, nAgents))
    obstacles_all = rng.uniform(-10, 10, size=(nSteps, 4, nObs))

    local_k_connectivity = rng.integers(0, 8, size=(nSteps, nAgents)).astype(float)

    lattice_violations = np.zeros((nSteps, nAgents, nAgents))
    # sprinkle some violations
    for i in range(nSteps):
        n_viol = rng.integers(0, 10)
        rows = rng.integers(0, nAgents, size=n_viol)
        cols = rng.integers(0, nAgents, size=n_viol)
        lattice_violations[i, rows, cols] = rng.uniform(-1, 1, size=n_viol)

    return states_all, targets_all, obstacles_all, local_k_connectivity, lattice_violations


# ---- tests ----

class TestDistanceFromTarget:

    def test_matches_reference(self, sim_data):
        states_all, targets_all, _, _, _ = sim_data
        ref = ref_distance_from_target(states_all, targets_all)
        vec = vec_distance_from_target(states_all, targets_all)
        np.testing.assert_allclose(vec, ref, atol=1e-12)

    def test_shape(self, sim_data):
        states_all, targets_all, _, _, _ = sim_data
        result = vec_distance_from_target(states_all, targets_all)
        assert result.shape == (states_all.shape[2], states_all.shape[0])

    def test_non_negative(self, sim_data):
        states_all, targets_all, _, _, _ = sim_data
        result = vec_distance_from_target(states_all, targets_all)
        assert np.all(result >= 0)


class TestDistanceFromObstacles:

    def test_means_match_reference(self, sim_data):
        states_all, _, obstacles_all, _, _ = sim_data
        _, ref_means2 = ref_distance_from_obstacles(states_all, obstacles_all)
        _, vec_means2 = vec_distance_from_obstacles(states_all, obstacles_all)
        np.testing.assert_allclose(vec_means2, ref_means2, atol=1e-12)

    def test_per_agent_distances_match(self, sim_data):
        states_all, _, obstacles_all, _, _ = sim_data
        ref_radii, _ = ref_distance_from_obstacles(states_all, obstacles_all)
        vec_radii, _ = vec_distance_from_obstacles(states_all, obstacles_all)
        np.testing.assert_allclose(vec_radii, ref_radii, atol=1e-12)

    def test_shape(self, sim_data):
        states_all, _, obstacles_all, _, _ = sim_data
        _, means2 = vec_distance_from_obstacles(states_all, obstacles_all)
        assert means2.shape == (states_all.shape[0],)


class TestKConnectivityStats:

    def test_matches_reference(self, sim_data):
        _, _, _, local_k, _ = sim_data
        ref_means, ref_mins, ref_maxs = ref_k_connectivity_stats(local_k)
        vec_means, vec_mins, vec_maxs = vec_k_connectivity_stats(local_k)
        np.testing.assert_array_equal(vec_means, ref_means)
        np.testing.assert_array_equal(vec_mins, ref_mins)
        np.testing.assert_array_equal(vec_maxs, ref_maxs)

    def test_min_le_mean_le_max(self, sim_data):
        _, _, _, local_k, _ = sim_data
        means, mins, maxs = vec_k_connectivity_stats(local_k)
        assert np.all(mins <= means)
        assert np.all(means <= maxs)


class TestCountViolations:

    def test_matches_reference(self, sim_data):
        _, _, _, _, violations = sim_data
        ref = ref_count_violations(violations)
        vec = vec_count_violations(violations)
        np.testing.assert_array_equal(vec, ref)

    def test_zero_violations(self):
        zeros = np.zeros((10, 5, 5))
        result = vec_count_violations(zeros)
        assert np.all(result == 0)

    def test_shape(self, sim_data):
        _, _, _, _, violations = sim_data
        result = vec_count_violations(violations)
        assert result.shape == (violations.shape[0], 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
