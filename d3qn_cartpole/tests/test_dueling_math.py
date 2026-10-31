"""
Unit Tests for Dueling Q-Network Aggregation Mathematics.

Verifies:
1. Mean-centering identifiability: 1/|A| * sum_a [Q(s, a) - V(s)] == 0.
2. Tensor shape preservation from [B, 4] input to [B, 2] output.
3. Separation of Value stream [B, 1] and Advantage stream [B, 2].
"""

import sys
from pathlib import Path

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import DuelingDQN


def test_dueling_shape_preservation():
    """Verify input shape (B, 4) produces exact output shape (B, 2)."""
    model = DuelingDQN(state_dim=4, action_dim=2)
    model.eval()

    for batch_size in [1, 8, 32, 64]:
        state = torch.randn(batch_size, 4)
        with torch.no_grad():
            q_values = model(state)
        assert q_values.shape == (batch_size, 2)
        assert torch.isfinite(q_values).all()


def test_dueling_streams_separation():
    """Verify feature trunk feeds into distinct value [B, 1] and advantage [B, 2] streams."""
    model = DuelingDQN(state_dim=4, action_dim=2)
    model.eval()

    state = torch.randn(16, 4)
    with torch.no_grad():
        feat = model.feature_network(state)
        val = model.value_stream(feat)
        adv = model.advantage_stream(feat)

    assert val.shape == (16, 1)
    assert adv.shape == (16, 2)


def test_dueling_mean_centering_mathematical_property():
    """
    Verify the fundamental mathematical theorem of Dueling DQN:
    For any state s, the mean across actions of (Q(s, a) - V(s)) is identically zero:
    1 / |A| * sum_{a in A} [ Q(s, a) - V(s) ] = 0.
    """
    model = DuelingDQN(state_dim=4, action_dim=2)
    model.eval()

    states = torch.randn(50, 4)

    with torch.no_grad():
        feat = model.feature_network(states)
        val = model.value_stream(feat) # Shape: (50, 1)
        q_vals = model(states) # Shape: (50, 2)

    # Difference between Q and V
    diff = q_vals - val # Shape: (50, 2)

    # Mean across action dimension
    mean_diff = diff.mean(dim=-1) # Shape: (50,)

    # Must be zero within floating point epsilon (1e-6)
    assert torch.allclose(
        mean_diff,
        torch.zeros_like(mean_diff),
        atol=1e-6,
    ), f"Mean-centering violated: max deviation = {torch.max(torch.abs(mean_diff)).item()}"

