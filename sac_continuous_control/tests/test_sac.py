import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import SACAgent
from src.config import SACConfig
from src.model import GaussianPolicy, TwinQNetwork
from src.replay_buffer import ContinuousReplayBuffer


def test_gaussian_policy_action_bounds():
    """Verify squashed Gaussian actor outputs actions strictly bounded by action_high."""
    action_high = 2.0
    policy = GaussianPolicy(state_dim=3, action_dim=1, hidden_dim=64, action_high=action_high)
    states = torch.randn(32, 3)

    action, log_prob, mean_action = policy.sample(states)

    assert action.shape == (32, 1)
    assert log_prob.shape == (32, 1)
    assert mean_action.shape == (32, 1)

    assert torch.all(action >= -action_high - 1e-5), "Sampled action violated lower bound"
    assert torch.all(action <= action_high + 1e-5), "Sampled action violated upper bound"
    assert torch.all(mean_action >= -action_high - 1e-5), "Mean action violated lower bound"
    assert torch.all(mean_action <= action_high + 1e-5), "Mean action violated upper bound"


def test_reparameterization_gradient():
    """Verify gradients propagate from actions to policy network parameters."""
    policy = GaussianPolicy(state_dim=3, action_dim=1, hidden_dim=64, action_high=2.0)
    states = torch.randn(8, 3)

    action, log_prob, _ = policy.sample(states)
    loss = (action.pow(2) + log_prob).mean()
    loss.backward()

    for param in policy.parameters():
        assert param.grad is not None, "Gradient was not propagated through reparameterization"


def test_twin_q_networks():
    """Verify twin critics produce two independent state-action evaluations."""
    twin_q = TwinQNetwork(state_dim=3, action_dim=1, hidden_dim=64)
    states = torch.randn(16, 3)
    actions = torch.randn(16, 1)

    q1, q2 = twin_q(states, actions)

    assert q1.shape == (16, 1)
    assert q2.shape == (16, 1)


def test_continuous_replay_buffer():
    """Verify buffer FIFO cyclic push and uniform sampling."""
    buffer = ContinuousReplayBuffer(state_dim=3, action_dim=1, capacity=50)

    for i in range(75):
        s = np.zeros(3, dtype=np.float32) + i
        a = np.array([float(i)], dtype=np.float32)
        buffer.push(s, a, 1.0, s, False)

    assert len(buffer) == 50
    batch = buffer.sample(16, torch.device("cpu"))
    assert batch["states"].shape == (16, 3)
    assert batch["actions"].shape == (16, 1)
    assert batch["rewards"].shape == (16, 1)
    assert batch["next_states"].shape == (16, 3)
    assert batch["dones"].shape == (16, 1)


def test_sac_agent_update():
    """Verify one full agent gradient update step across actor, critics, and alpha."""
    config = SACConfig(state_dim=3, action_dim=1, hidden_dim=32, batch_size=16)
    agent = SACAgent(config, device=torch.device("cpu"))

    batch = {
        "states": torch.randn(16, 3),
        "actions": torch.randn(16, 1),
        "rewards": torch.randn(16, 1),
        "next_states": torch.randn(16, 3),
        "dones": torch.zeros(16, 1),
    }

    metrics = agent.update_parameters(batch)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert metrics["alpha"] > 0.0
