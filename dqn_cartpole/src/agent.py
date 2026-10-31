"""
Deep Q-Network (DQN) Agent Implementation.

Handles epsilon-greedy exploration, experience replay storage, Bellman loss computation,
gradient clipping, and policy/target network synchronization.
"""

from pathlib import Path
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    ACTION_DIM,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    GAMMA,
    LEARNING_RATE,
    STATE_DIM,
    TAU,
)
from .model import DQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Reinforcement learning agent coordinating policy and target networks with experience replay.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        gamma: float = GAMMA,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        buffer_capacity: int = BUFFER_CAPACITY,
        tau: float = TAU,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate actively-updated policy network and frozen target network
        self.policy_net = DQN(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_net = DQN(state_dim=state_dim, action_dim=action_dim).to(self.device)

        # Synchronize target network weights to policy network initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer and Huber Loss (Smooth L1)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        # Experience Replay Memory
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action according to epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            4D continuous state vector.
        epsilon : float, default=0.0
            Exploration probability in [0.0, 1.0].

        Returns
        -------
        int
            Selected discrete action in {0, 1}.
        """
        if random.random() < epsilon:
            # Explore: sample action uniformly at random
            return random.randrange(self.action_dim)

        # Exploit: select action maximizing predicted Q-value
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = int(q_values.argmax(dim=1).item())
        self.policy_net.train()
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Record interaction tuple into experience replay memory.
        """
        self.memory.push(state, action, reward, next_state, done)

    def update_policy(self) -> Optional[float]:
        """
        Sample mini-batch from experience replay and perform single Bellman gradient update.

        Returns
        -------
        Optional[float]
            Loss value, or None if replay memory contains fewer samples than BATCH_SIZE.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size=self.batch_size,
            device=self.device,
        )

        # Compute Q(s, a) predicted by active policy network
        # Shape: (batch_size, 1) -> (batch_size,)
        q_predicted = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Bellman target values using frozen target network:
        # y = r + (1 - done) * gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_state_q_values = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + (1.0 - dones) * self.gamma * next_state_q_values

        # Compute Huber / Smooth L1 loss
        loss = self.criterion(q_predicted, q_target)

        # Gradient Backpropagation & Optimization
        self.optimizer.zero_grad()
        loss.backward()
        # Clamp gradients to curb explosion from large temporal difference errors
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self, soft: bool = False, tau: Optional[float] = None) -> None:
        """
        Synchronize target network parameters from policy network.

        Parameters
        ----------
        soft : bool, default=False
            If True, applies Polyak exponential moving average parameter update:
            theta_target <- tau * theta_policy + (1 - tau) * theta_target.
            If False, performs periodic hard parameter copy.
        tau : Optional[float]
            Interpolation parameter for soft update (defaults to self.tau).
        """
        if soft:
            tau_val = tau if tau is not None else self.tau
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    tau_val * policy_param.data + (1.0 - tau_val) * target_param.data
                )
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: Path) -> None:
        """
        Serialize policy network weights to disk.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: Path) -> None:
        """
        Load policy network weights from disk and synchronize target network.
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()

