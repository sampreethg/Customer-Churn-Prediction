"""
Dueling Double Deep Q-Network (D3QN) Agent.

Combines Dueling network function approximation with decoupled Double DQN target valuation
and Polyak target network exponential smoothing.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    ACTION_DIM,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    GAMMA,
    LR,
    STATE_DIM,
    TAU,
)
from .model import DuelingDQN
from .replay_buffer import ReplayBuffer


class D3QNAgent:
    """
    D3QN Agent coordinating Dueling policy & target networks, Double Q target computation,
    and Polyak soft updates.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        gamma: float = GAMMA,
        lr: float = LR,
        tau: float = TAU,
        batch_size: int = BATCH_SIZE,
        buffer_capacity: int = BUFFER_CAPACITY,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize online policy network and offline target network
        self.policy_net = DuelingDQN(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim=state_dim, action_dim=action_dim).to(self.device)

        # Synchronize target network weights to policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer: AdamW with amsgrad
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        # Replay memory
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : np.ndarray
            Current continuous 4D state observation.
        epsilon : float, default=0.0
            Exploration probability.

        Returns
        -------
        int
            Selected discrete action in {0, 1}.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = int(q_values.argmax(dim=-1).item())
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
        Store transition into experience replay memory.
        """
        self.memory.push(state, action, reward, next_state, done)

    def update_parameters(self) -> Optional[float]:
        """
        Perform a Double DQN parameter update on a mini-batch sampled from replay buffer.

        Returns
        -------
        Optional[float]
            Computed Huber loss value, or None if buffer has insufficient samples.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(batch_size=self.batch_size, device=self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Current predicted Q-values for chosen actions: Q(s, a; theta)
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN Target Computation:
        # 1. Action selection using online policy_net: a* = argmax_a' Q(s', a'; theta)
        # 2. Action evaluation using offline target_net: Q(s', a*; theta^-)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        # Huber Loss (Smooth L1)
        loss = F.smooth_l1_loss(current_q, targets)

        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        # Polyak soft update of target network parameters
        self.soft_update_target_network()

        return float(loss.item())

    def soft_update_target_network(self) -> None:
        """
        Polyak exponential smoothing parameter update:
        theta^- <- tau * theta + (1 - tau) * theta^-
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters(), strict=False
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: Path) -> None:
        """
        Save policy network state dict to disk.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: Path) -> None:
        """
        Load policy network state dict and synchronize target network.
        """
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()

