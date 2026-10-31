"""
Dueling Deep Q-Network (Dueling DQN) Neural Architecture.

Decomposes Q-value representations into decoupled state-value V(s) and action-advantage A(s, a)
streams, combined via mean-centered identifiability aggregation:
Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum_{a'} A(s, a'))
"""

import torch
import torch.nn as nn

from .config import ACTION_DIM, STATE_DIM


class DuelingDQN(nn.Module):
    """
    Dueling Q-Network architecture splitting representation into state-value and advantage streams.

    Architecture:
    - Shared Feature Trunk: Linear(state_dim, 128) -> ReLU()
    - State-Value Stream V(s): Linear(128, 64) -> ReLU() -> Linear(64, 1)
    - Action-Advantage Stream A(s, a): Linear(128, 64) -> ReLU() -> Linear(64, action_dim)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared representation trunk
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        # Scalar state-value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action-advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply He normal initialization for ReLU layers and Xavier uniform for linear heads.
        """
        for module in [self.feature_network, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    if layer.out_features in (1, self.action_dim):
                        nn.init.xavier_uniform_(layer.weight)
                    else:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-centered action-values Q(s, a).

        Parameters
        ----------
        state : torch.Tensor
            Batch state tensor of shape (B, state_dim).

        Returns
        -------
        torch.Tensor
            Q-values of shape (B, action_dim).
        """
        features = self.feature_network(state)
        val = self.value_stream(features) # Shape: (B, 1)
        adv = self.advantage_stream(features) # Shape: (B, action_dim)

        # Enforce parameter identifiability via mean subtraction:
        # Q(s, a) = V(s) + (A(s, a) - mean_{a'} A(s, a'))
        adv_centered = adv - adv.mean(dim=-1, keepdim=True)
        q_values = val + adv_centered
        return q_values

