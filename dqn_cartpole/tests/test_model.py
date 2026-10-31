"""
Unit Tests for Deep Q-Network PyTorch Architecture.

Validates forward pass dimensions, parameter initialization, gradient propagation, and backward updates.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import DQN


def test_dqn_output_dimensions():
    """Verify DQN maps arbitrary batches of 4D states to 2D action-values."""
    model = DQN(state_dim=4, action_dim=2)
    model.eval()

    # Single observation
    single_state = torch.randn(1, 4)
    with torch.no_grad():
        q_single = model(single_state)
    assert q_single.shape == (1, 2)

    # Mini-batch of 64 observations
    batch_states = torch.randn(64, 4)
    with torch.no_grad():
        q_batch = model(batch_states)
    assert q_batch.shape == (64, 2)


def test_dqn_initialization_finiteness():
    """Verify all weights and biases are finite and initialized properly."""
    model = DQN(state_dim=4, action_dim=2)

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Non-finite values detected in {name}"
        if "bias" in name:
            assert torch.all(param == 0.0), f"Bias {name} was not zero-initialized"


def test_dqn_gradient_propagation():
    """Verify backpropagation produces valid non-zero gradients across all network parameters."""
    model = DQN(state_dim=4, action_dim=2)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()

    states = torch.randn(32, 4)
    target_q = torch.randn(32, 2)

    predicted_q = model(states)
    loss = criterion(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()

    # Check that gradients exist and are non-zero for all layers
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient contains non-finite values in {name}"
        assert torch.count_nonzero(param.grad) > 0, f"Gradient is completely zero for {name}"

    optimizer.step()

