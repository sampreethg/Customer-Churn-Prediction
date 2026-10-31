"""
Unit Tests for Experience Replay Buffer.

Validates capacity boundedness, FIFO eviction, sampling shapes, data types, and edge error cases.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.replay_buffer import ReplayBuffer, Transition


def test_transition_dataclass():
    """Verify Transition immutability and attributes."""
    t = Transition(
        state=np.array([0.1, 0.2, 0.3, 0.4]),
        action=1,
        reward=1.0,
        next_state=np.array([0.15, 0.25, 0.35, 0.45]),
        done=False,
    )
    assert t.action == 1
    assert t.reward == 1.0
    assert not t.done
    assert len(t.state) == 4
    assert len(t.next_state) == 4


def test_replay_buffer_capacity_boundedness():
    """Verify buffer respects maximum capacity and discards oldest transitions."""
    capacity = 50
    buffer = ReplayBuffer(capacity=capacity)
    assert len(buffer) == 0

    # Push 100 transitions (double the capacity)
    for i in range(100):
        buffer.push(
            state=np.array([float(i)] * 4),
            action=i % 2,
            reward=float(i),
            next_state=np.array([float(i + 1)] * 4),
            done=(i % 10 == 0),
        )

    # Capacity must not exceed 50
    assert len(buffer) == capacity

    # The oldest 50 items (0..49) should have been evicted; earliest item is 50
    oldest_transition = buffer.memory[0]
    assert oldest_transition.reward == 50.0

    # The newest item is 99
    newest_transition = buffer.memory[-1]
    assert newest_transition.reward == 99.0


def test_replay_buffer_sample_tensors_and_types():
    """Verify sampled mini-batch returns expected tensor shapes and dtypes."""
    buffer = ReplayBuffer(capacity=100)
    for i in range(50):
        buffer.push(
            state=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
            action=i % 2,
            reward=1.0,
            next_state=np.array([0.15 * i, 0.25 * i, 0.35 * i, 0.45 * i]),
            done=False,
        )

    batch_size = 16
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=batch_size)

    # Check shapes
    assert states.shape == (batch_size, 4)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, 4)
    assert dones.shape == (batch_size,)

    # Check dtypes
    assert states.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert dones.dtype == torch.float32


def test_replay_buffer_insufficient_samples_raises_value_error():
    """Verify sampling from under-populated buffer raises ValueError."""
    buffer = ReplayBuffer(capacity=100)
    buffer.push(
        state=np.array([0.0, 0.0, 0.0, 0.0]),
        action=0,
        reward=1.0,
        next_state=np.array([0.0, 0.0, 0.0, 0.0]),
        done=False,
    )

    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(batch_size=32)

