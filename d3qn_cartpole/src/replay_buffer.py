"""
Bounded Experience Replay Buffer for D3QN.

Maintains a cyclic memory of state transitions and prepares batched PyTorch tensors
for off-policy temporal difference updates.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import BUFFER_CAPACITY


@dataclass(frozen=True)
class Transition:
    """
    Single environment step interaction tuple.
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Cyclic memory buffer bounded by fixed capacity.
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.capacity = capacity
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Append transition to cyclic memory.
        """
        self.memory.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample uniform mini-batch and return PyTorch tensors mapped to device:
        - states: FloatTensor [B, 4]
        - actions: LongTensor [B, 1]
        - rewards: FloatTensor [B, 1]
        - next_states: FloatTensor [B, 4]
        - dones: FloatTensor [B, 1]
        """
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Replay buffer contains {len(self.memory)} transitions, fewer than batch_size {batch_size}."
            )

        batch: List[Transition] = random.sample(self.memory, batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [[t.action] for t in batch],
            dtype=torch.int64,
            device=device,
        )
        rewards = torch.tensor(
            [[t.reward] for t in batch],
            dtype=torch.float32,
            device=device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [[float(t.done)] for t in batch],
            dtype=torch.float32,
            device=device,
        )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self.memory)

