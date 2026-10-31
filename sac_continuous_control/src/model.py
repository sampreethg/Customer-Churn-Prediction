"""Neural network architectures for Soft Actor-Critic (SAC).

Includes:
1. Reparameterized Squashed Gaussian Policy (Actor) with exact Jacobian log-prob correction.
2. Twin Q-Networks (Critics) to mitigate continuous-action value overestimation.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6


class GaussianPolicy(nn.Module):
    """Squashed Gaussian Actor mapping continuous states to continuous torque control."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_high: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.action_scale = float(action_high)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action using reparameterization trick and computes exact log-prob
        accounting for tanh squashing via change of variables.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization: u = mean + std * eps
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample() # for backpropagation through stochastic nodes
        y = torch.tanh(u)
        action = y * self.action_scale

        # Enforcing Action Bounds & Jacobian log-prob correction
        log_prob = normal.log_prob(u) - torch.log(self.action_scale * (1.0 - y.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean_action


class TwinQNetwork(nn.Module):
    """Twin Q-Critic networks parameterized independently for continuous state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 Architecture
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 Architecture
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)
