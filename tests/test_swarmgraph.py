#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for utils.swarmgraph refactored Swarmgraph.

Compares the KD-tree + scipy backed implementation against a brute-force
reference to verify identical adjacency matrices, connected components,
local connectivity, and pin assignments.
"""

import sys
import os
import numpy as np
import pytest

# ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.swarmgraph import Swarmgraph, convert_A_to_D, compute_local_connectivity


# ---- brute-force reference implementation ----

def brute_force_update_A(data, r_matrix, nNodes, slack=0.2):
    """Reference O(n^2) adjacency construction matching original logic."""
    A = np.zeros((nNodes, nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            if i != j:
                dist = np.linalg.norm(data[0:3, j] - data[0:3, i])
                r = r_matrix[i, j] + slack
                if dist < r:
                    A[i, j] = 1
    return A


def brute_force_connected_components(A):
    """Reference BFS connected components matching original logic."""
    all_components = []
    visited = []
    for node in range(A.shape[1]):
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
    return all_components


# ---- fixtures ----

@pytest.fixture
def swarm_state_clustered():
    """Two clusters of agents with known connectivity."""
    rng = np.random.default_rng(42)
    # cluster A: 8 agents near origin
    cluster_a = rng.normal(loc=0, scale=2, size=(6, 8))
    # cluster B: 5 agents far away
    cluster_b = rng.normal(loc=50, scale=2, size=(6, 5))
    return np.hstack([cluster_a, cluster_b])


@pytest.fixture
def swarm_state_random():
    """Random swarm state with 20 agents."""
    rng = np.random.default_rng(99)
    return rng.uniform(-30, 30, size=(6, 20))


@pytest.fixture
def swarm_state_dense():
    """Dense cluster where all agents are within range of each other."""
    rng = np.random.default_rng(7)
    return rng.normal(loc=0, scale=1, size=(6, 10))


# ---- tests: adjacency matrix ----

class TestUpdateA:

    def test_uniform_radius_matches_brute_force(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r = 15.0
        r_matrix = r * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        expected = brute_force_update_A(swarm_state_random, r_matrix, n)
        np.testing.assert_array_equal(graph.A, expected)

    def test_nonuniform_radius_matches_brute_force(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        rng = np.random.default_rng(123)
        r_matrix = rng.uniform(5, 20, size=(n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        expected = brute_force_update_A(swarm_state_random, r_matrix, n)
        np.testing.assert_array_equal(graph.A, expected)

    def test_clusters_produce_block_diagonal(self, swarm_state_clustered):
        n = swarm_state_clustered.shape[1]
        r = 10.0
        r_matrix = r * np.ones((n, n))

        graph = Swarmgraph(swarm_state_clustered)
        graph.update_A(swarm_state_clustered[0:3, :], r_matrix)

        # no cross-cluster edges (cluster A: indices 0-7, cluster B: 8-12)
        assert np.sum(graph.A[0:8, 8:13]) == 0
        assert np.sum(graph.A[8:13, 0:8]) == 0

    def test_degree_matrix_consistent(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        expected_D = convert_A_to_D(graph.A)
        np.testing.assert_array_equal(graph.D, expected_D)

    def test_zero_radius_produces_empty_graph(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = np.zeros((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        assert np.sum(graph.A) == 0

    def test_huge_radius_produces_complete_graph(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 1e6 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        expected = np.ones((n, n)) - np.eye(n)
        np.testing.assert_array_equal(graph.A, expected)


# ---- tests: connected components ----

class TestConnectedComponents:

    def test_matches_brute_force(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)
        graph.find_connected_components()

        expected = brute_force_connected_components(graph.A)

        # compare as sets-of-sets (order within components may differ)
        result_sets = {frozenset(c) for c in graph.components}
        expected_sets = {frozenset(c) for c in expected}
        assert result_sets == expected_sets

    def test_two_clusters(self, swarm_state_clustered):
        n = swarm_state_clustered.shape[1]
        r_matrix = 10.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_clustered)
        graph.update_A(swarm_state_clustered[0:3, :], r_matrix)
        graph.find_connected_components()

        # should have exactly 2 components
        assert len(graph.components) == 2

    def test_fully_connected(self, swarm_state_dense):
        n = swarm_state_dense.shape[1]
        r_matrix = 100.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_dense)
        graph.update_A(swarm_state_dense[0:3, :], r_matrix)
        graph.find_connected_components()

        assert len(graph.components) == 1
        assert len(graph.components[0]) == n

    def test_fully_disconnected(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = np.zeros((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)
        graph.find_connected_components()

        # each agent is its own component
        assert len(graph.components) == n


# ---- tests: local connectivity ----

class TestLocalConnectivity:

    def test_matches_node_degree(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        connectivity = compute_local_connectivity(graph.A)

        # should match node degrees
        degrees = np.sum(graph.A, axis=1).astype(int)
        for i in range(n):
            assert connectivity[i] == degrees[i]

    def test_disconnected_has_zero_connectivity(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = np.zeros((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        connectivity = compute_local_connectivity(graph.A)
        for i in range(n):
            assert connectivity[i] == 0

    def test_pick_single_node(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_A(swarm_state_random[0:3, :], r_matrix)

        result = compute_local_connectivity(graph.A, pick=5)
        assert len(result) == 1
        assert 5 in result


# ---- tests: pin assignment ----

class TestUpdatePins:

    def test_degree_method(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_pins(swarm_state_random[0:3, :], r_matrix, 'degree')

        # at least one pin per component
        for component in graph.components:
            has_pin = any(graph.pin_matrix[i, i] == 1 for i in component)
            assert has_pin, f"Component {component} has no pin"

    def test_nopins_method(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_pins(swarm_state_random[0:3, :], r_matrix, 'nopins')

        assert np.sum(graph.pin_matrix) == 0

    def test_allpins_method(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_pins(swarm_state_random[0:3, :], r_matrix, 'allpins')

        expected = np.ones((n, n))
        np.testing.assert_array_equal(graph.pin_matrix, expected)

    def test_degree_leafs_method(self, swarm_state_random):
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_pins(swarm_state_random[0:3, :], r_matrix, 'degree_leafs')

        # leaf nodes (degree 1) should be pinned
        D_elements = np.diag(graph.D)
        leaf_nodes = np.where(D_elements == 1)[0]
        for leaf in leaf_nodes:
            assert graph.pin_matrix[leaf, leaf] == 1

    def test_pins_have_local_k_connectivity(self, swarm_state_random):
        """After update_pins, local_k_connectivity should be populated."""
        n = swarm_state_random.shape[1]
        r_matrix = 15.0 * np.ones((n, n))

        graph = Swarmgraph(swarm_state_random)
        graph.update_pins(swarm_state_random[0:3, :], r_matrix, 'degree')

        assert len(graph.local_k_connectivity) == n


# ---- tests: edge cases ----

class TestEdgeCases:

    def test_single_agent(self):
        state = np.zeros((6, 1))
        r_matrix = np.ones((1, 1)) * 10
        graph = Swarmgraph(state)
        graph.update_A(state[0:3, :], r_matrix)
        graph.find_connected_components()

        assert np.sum(graph.A) == 0
        assert len(graph.components) == 1
        assert graph.components[0] == [0]

    def test_two_agents_connected(self):
        state = np.zeros((6, 2))
        state[0, 1] = 5.0  # 5 units apart in x
        r_matrix = 10.0 * np.ones((2, 2))

        graph = Swarmgraph(state)
        graph.update_A(state[0:3, :], r_matrix)
        graph.find_connected_components()

        assert graph.A[0, 1] == 1
        assert graph.A[1, 0] == 1
        assert len(graph.components) == 1

    def test_two_agents_disconnected(self):
        state = np.zeros((6, 2))
        state[0, 1] = 100.0  # far apart
        r_matrix = 10.0 * np.ones((2, 2))

        graph = Swarmgraph(state)
        graph.update_A(state[0:3, :], r_matrix)
        graph.find_connected_components()

        assert graph.A[0, 1] == 0
        assert graph.A[1, 0] == 0
        assert len(graph.components) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
