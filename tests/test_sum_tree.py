"""Mathematical Invariant and SumTree Priority Queue Property Tests for PER."""

import pytest
import numpy as np


class SumTree:
    """Complete array-backed binary tree where parent node = left child + right child."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def add(self, priority: float):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

    def get_leaf(self, value: float):
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child
        return leaf_idx, self.tree[leaf_idx]


def test_sum_tree_root_invariant():
    """Verify tree[0] equals the exact sum of all leaf priorities."""
    tree = SumTree(capacity=16)
    priorities = [1.2, 3.4, 0.5, 9.1, 4.3, 2.2, 0.1, 7.8, 5.5, 6.1, 3.0, 1.9, 0.8, 4.4, 2.7, 8.0]

    for p in priorities:
        tree.add(p)

    expected_sum = sum(priorities)
    assert abs(tree.total_priority - expected_sum) < 1e-5, f"Root sum {tree.total_priority} != {expected_sum}"


def test_sum_tree_dynamic_update():
    """Verify that updating a single leaf dynamically updates the root sum."""
    tree = SumTree(capacity=8)
    for p in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        tree.add(p)

    initial_root = tree.total_priority
    assert abs(initial_root - 36.0) < 1e-5

    # Update leaf index 7 (first leaf is at 8 - 1 = 7)
    tree.update(7, 10.0)  # was 1.0, now 10.0 (+9.0)
    assert abs(tree.total_priority - 45.0) < 1e-5


def test_sum_tree_sampling_segment():
    """Verify that retrieval maps uniformly across stratified cumulative segments."""
    tree = SumTree(capacity=4)
    tree.add(10.0)
    tree.add(20.0)
    tree.add(30.0)
    tree.add(40.0)

    # Total = 100
    idx, p = tree.get_leaf(5.0)
    assert p == 10.0

    idx, p = tree.get_leaf(15.0)
    assert p == 20.0

    idx, p = tree.get_leaf(50.0)
    assert p == 30.0

    idx, p = tree.get_leaf(85.0)
    assert p == 40.0
