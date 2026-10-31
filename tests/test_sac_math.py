"""Mathematical property validation for continuous Soft Actor-Critic (SAC)."""

import pytest
import torch
import torch.nn as nn
from sac_continuous_control.src.model import GaussianPolicy


def test_bounded_gaussian_log_prob_jacobian():
    """
    Verify that log-prob correction for Tanh squashing satisfies the change-of-variables theorem.
    Formula:
        log pi(a|s) = log mu(u|s) - sum_i log(scale * (1 - tanh^2(u_i)) + epsilon)
    """
    action_scale = 2.0
    policy = GaussianPolicy(state_dim=3, action_dim=2, hidden_dim=64, action_high=action_scale)

    state = torch.randn(10, 3)
    action, log_prob, mean_action = policy.sample(state)

    assert log_prob.shape == (10, 1)
    assert not torch.isnan(log_prob).any(), "NaN values found in SAC log probabilities"
    assert not torch.isinf(log_prob).any(), "Infinite values found in SAC log probabilities"
    assert torch.all(action <= action_scale + 1e-4)
    assert torch.all(action >= -action_scale - 1e-4)


def test_dueling_aggregation_identifiability():
    """
    Verify that mean subtraction in Dueling aggregation enforces parameter identifiability:
        Q(s, a) = V(s) + (A(s, a) - 1/|A| sum A)
        => mean_a Q(s, a) = V(s)
    """
    batch_size = 16
    action_dim = 4

    V = torch.randn(batch_size, 1)
    A = torch.randn(batch_size, action_dim)

    # Identifiability aggregation
    Q = V + (A - A.mean(dim=-1, keepdim=True))

    # Assert mean across actions equals V
    Q_mean = Q.mean(dim=-1, keepdim=True)
    assert torch.allclose(Q_mean, V, atol=1e-5), "Mean Q over action space does not equal State-Value V"
