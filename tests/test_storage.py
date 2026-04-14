#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for sparse History storage and HDF5 round-trip.

Verifies that:
1. History uses sparse lists instead of dense 3D arrays
2. HDF5 save/load produces identical data
3. Memory usage scales with O(edges) not O(n^2)
"""

import sys
import os
import tempfile
import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_manager import (
    History, save_data_HDF5, load_data_HDF5,
    _reconstruct_dense_3d, _save_sparse_hdf5, _load_sparse_hdf5,
    _SPARSE_FIELDS,
)


# ---- helpers ----

class MockAgents:
    def __init__(self, n):
        self.nAgents = n
        self.state = np.random.rand(6, n)
        self.centroid = np.mean(self.state[0:3, :], axis=1, keepdims=True)
        self.centroid_v = np.mean(self.state[3:6, :], axis=1, keepdims=True)
        self.dynamics_type = 'double integrator'

class MockTargets:
    def __init__(self, n):
        self.targets = np.random.rand(6, n)

class MockObstacles:
    def __init__(self):
        self.obstacles = np.random.rand(4, 1)
        self.nObs = 1
        self.walls_plots = np.zeros((6, 0))

class MockTrajectory:
    def __init__(self, n):
        self.lemni = np.zeros((2, n))
        self.sorted_neighs = list(range(n))

class MockGraphs:
    def __init__(self, n):
        # sparse adjacency: ring topology
        self.A = np.zeros((n, n))
        for i in range(n):
            j = (i + 1) % n
            self.A[i, j] = 1
            self.A[j, i] = 1
        self.local_k_connectivity = {i: 2 for i in range(n)}
        self.directional_graph = False

class MockController:
    def __init__(self, n):
        self.cmd = np.random.rand(3, n)
        self.pin_matrix = np.zeros((n, n))
        self.pin_matrix[0, 0] = 1  # one pin
        self.Graphs = MockGraphs(n)
        self.Graphs_connectivity = MockGraphs(n)
        self.lattice = 10.0 * np.ones((n, n))
        self.Learners = {}


# ---- fixtures ----

@pytest.fixture
def history_13():
    """History object with 13 agents, 0.5s sim."""
    n = 13
    np.random.seed(42)
    agents = MockAgents(n)
    targets = MockTargets(n)
    obstacles = MockObstacles()
    controller = MockController(n)
    trajectory = MockTrajectory(n)
    return History(agents, targets, obstacles, controller, trajectory,
                   Ts=0.02, Tf=0.5, Ti=0, f=0)


@pytest.fixture
def history_50():
    """History object with 50 agents, 0.2s sim."""
    n = 50
    np.random.seed(42)
    agents = MockAgents(n)
    targets = MockTargets(n)
    obstacles = MockObstacles()
    controller = MockController(n)
    trajectory = MockTrajectory(n)
    return History(agents, targets, obstacles, controller, trajectory,
                   Ts=0.02, Tf=0.2, Ti=0, f=0)


# ---- tests: sparse allocation ----

class TestSparseAllocation:

    def test_connectivity_is_list(self, history_13):
        assert isinstance(history_13.connectivity, list)

    def test_pins_is_list(self, history_13):
        assert isinstance(history_13.pins_all, list)

    def test_lattices_is_list(self, history_13):
        assert isinstance(history_13.lattices, list)

    def test_violations_is_list(self, history_13):
        assert isinstance(history_13.lattice_violations, list)

    def test_initial_entry_is_sparse(self, history_13):
        assert sparse.issparse(history_13.connectivity[0])
        assert sparse.issparse(history_13.pins_all[0])

    def test_states_still_dense(self, history_13):
        """O(n) arrays remain dense numpy arrays."""
        assert isinstance(history_13.states_all, np.ndarray)
        assert isinstance(history_13.t_all, np.ndarray)

    def test_memory_smaller_than_dense(self, history_50):
        """Sparse lists should use much less memory than dense 3D arrays would."""
        n = 50
        nSteps = len(history_50.connectivity)
        dense_bytes = nSteps * n * n * 8  # what one dense 3D array would cost

        # measure actual sparse memory
        sparse_bytes = sum(
            m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
            for m in history_50.connectivity if m is not None and sparse.issparse(m)
        )

        # sparse should be significantly smaller (ring topology: 2n edges vs n^2)
        assert sparse_bytes < dense_bytes * 0.5, \
            f"Sparse ({sparse_bytes}) not much smaller than dense ({dense_bytes})"


# ---- tests: HDF5 round-trip ----

class TestHDF5RoundTrip:

    def test_save_load_connectivity(self, history_13):
        """Save and reload connectivity, verify data matches."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            save_data_HDF5(history_13, path)
            _, loaded = load_data_HDF5('History', 'connectivity', path)

            # reconstruct expected dense
            expected = _reconstruct_dense_3d(history_13.connectivity, 13)
            np.testing.assert_array_almost_equal(loaded, expected)
        finally:
            os.unlink(path)

    def test_save_load_pins(self, history_13):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            save_data_HDF5(history_13, path)
            _, loaded = load_data_HDF5('History', 'pins_all', path)

            expected = _reconstruct_dense_3d(history_13.pins_all, 13)
            np.testing.assert_array_almost_equal(loaded, expected)
        finally:
            os.unlink(path)

    def test_save_load_lattices(self, history_13):
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            save_data_HDF5(history_13, path)
            _, loaded = load_data_HDF5('History', 'lattices', path)

            expected = _reconstruct_dense_3d(history_13.lattices, 13)
            np.testing.assert_array_almost_equal(loaded, expected)
        finally:
            os.unlink(path)

    def test_save_load_linear_arrays(self, history_13):
        """O(n) arrays should round-trip unchanged."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        try:
            save_data_HDF5(history_13, path)
            _, t_loaded = load_data_HDF5('History', 't_all', path)
            _, states_loaded = load_data_HDF5('History', 'states_all', path)

            np.testing.assert_array_equal(t_loaded, history_13.t_all)
            np.testing.assert_array_equal(states_loaded, history_13.states_all)
        finally:
            os.unlink(path)


# ---- tests: dense reconstruction helper ----

class TestDenseReconstruction:

    def test_round_trip_identity(self):
        """Sparse -> dense -> verify matches original dense."""
        n = 10
        nSteps = 5
        original = [sparse.random(n, n, density=0.3, format='csr') for _ in range(nSteps)]

        dense = _reconstruct_dense_3d(original, n)
        assert dense.shape == (nSteps, n, n)

        for i in range(nSteps):
            np.testing.assert_array_almost_equal(dense[i], original[i].toarray())

    def test_handles_none_entries(self):
        sparse_list = [None, sparse.eye(5, format='csr'), None]
        dense = _reconstruct_dense_3d(sparse_list, 5)
        assert dense.shape == (3, 5, 5)
        np.testing.assert_array_equal(dense[0], np.zeros((5, 5)))
        np.testing.assert_array_equal(dense[1], np.eye(5))
        np.testing.assert_array_equal(dense[2], np.zeros((5, 5)))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
