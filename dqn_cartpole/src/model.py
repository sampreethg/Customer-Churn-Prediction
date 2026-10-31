"""
PyTorch Deep Q-Network (DQN) Multi-Layer Perceptron (MLP) Architecture.

Maps continuous state observation vectors to discrete action-value Q-logits: R^4 -> R^2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ACTION_DIM, STATE_DIM


class DQN(nn.Module):
    """
    Multi-Layer Perceptron Q-Network for approximating optimal action-value functions.

    Architecture:
    Linear(state_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, action_dim)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply He (Kaiming) normal initialization for ReLU hidden layers
        and Xavier uniform initialization for the linear output layer.
        """
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(self.fc2.bias, 0.0)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted Q-values across all discrete actions for a batch of states.

        Parameters
        ----------
        state : torch.Tensor
            State observation tensor of shape (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Action-value Q logits of shape (batch_size, action_dim).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

