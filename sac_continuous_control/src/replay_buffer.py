"""High-throughput Experience Replay Buffer for Continuous-Action Reinforcement Learning."""

from typing import Dict

import numpy as np
import torch


class ContinuousReplayBuffer:
    """Contiguous numpy-backed cyclic experience replay buffer."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Append a single transition tuple."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Uniformly sample a mini-batch of transitions transferred to compute device."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.from_numpy(self.states[indices]).to(device),
            "actions": torch.from_numpy(self.actions[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(device),
            "dones": torch.from_numpy(self.dones[indices]).to(device),
        }

    def __len__(self) -> int:
        return self.size
