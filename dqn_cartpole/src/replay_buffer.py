"""
Experience Replay Buffer and Transition Dataclass.

Stores chronological agent-environment interaction tuples and breaks temporal correlation
by uniformly sampling independent mini-batches for off-policy Q-learning gradient updates.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import BUFFER_CAPACITY


@dataclass(frozen=True)
class Transition:
    """
    Immutable single-step environment interaction tuple: (s, a, r, s', done).
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-capacity cyclic memory buffer implemented via collections.deque.
    Oldest transitions are automatically discarded once capacity is saturated.
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
        Record a new transition tuple into the cyclic buffer.
        """
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.memory.append(transition)

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniformly sample an independent mini-batch of transitions and convert to PyTorch tensors.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : Optional[torch.device]
            Target compute device (CPU or CUDA).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (states, actions, rewards, next_states, dones) tensors.
        """
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer containing {len(self.memory)} items."
            )

        batch: List[Transition] = random.sample(self.memory, batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.int64,
            device=device,
        )
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [float(t.done) for t in batch],
            dtype=torch.float32,
            device=device,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)

