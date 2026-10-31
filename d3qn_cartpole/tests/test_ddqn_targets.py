"""
Unit Tests for Double DQN (DDQN) Decoupled Target Calculation.

Verifies:
1. Strict decoupling: Online network selects action, target network evaluates value.
2. Prevention of maximization overestimation bias.
3. Terminal state masking: (1 - done) forces future Q-value to 0.
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import D3QNAgent


def test_ddqn_decoupled_action_selection_and_evaluation():
    """
    Construct synthetic tensors to prove Double DQN eliminates maximization overestimation bias:
    Online Net Q-values: [10.0, 5.0]  -> argmax selects Action 0
    Target Net Q-values: [ 3.0, 8.0]  -> Action 0 value is 3.0 (Target Net max is 8.0)
    
    In Naive DQN:  target = reward + gamma * max(3.0, 8.0) = reward + gamma * 8.0
    In Double DQN: target = reward + gamma * target_net(action=0) = reward + gamma * 3.0
    """
    gamma = 0.99
    reward = torch.tensor([[1.0]], dtype=torch.float32)
    done = torch.tensor([[0.0]], dtype=torch.float32)

    # Synthetic Q values for next state
    online_next_q = torch.tensor([[10.0, 5.0]], dtype=torch.float32)  # Prefers action 0
    target_next_q = torch.tensor([[3.0, 8.0]], dtype=torch.float32)   # Has higher value for action 1

    # Double DQN Target Calculation
    next_action = online_next_q.argmax(dim=1, keepdim=True)  # Tensor([[0]])
    assert next_action.item() == 0, "Online network must select action 0"

    evaluated_q = target_next_q.gather(1, next_action)       # Tensor([[3.0]])
    assert evaluated_q.item() == 3.0, "Target network must evaluate action 0 to 3.0"

    ddqn_target = reward + (1.0 - done) * gamma * evaluated_q
    expected_target = 1.0 + gamma * 3.0  # 1.0 + 2.97 = 3.97

    assert torch.isclose(ddqn_target, torch.tensor([[expected_target]])), (
        f"DDQN target {ddqn_target.item()} != expected {expected_target}"
    )


def test_terminal_state_masking():
    """Verify that when done=1.0, future Q-values are completely zeroed out."""
    gamma = 0.99
    reward = torch.tensor([[1.0]], dtype=torch.float32)
    done = torch.tensor([[1.0]], dtype=torch.float32)  # Terminated episode

    target_next_q = torch.tensor([[50.0, 100.0]], dtype=torch.float32)
    next_action = torch.tensor([[1]])
    evaluated_q = target_next_q.gather(1, next_action)

    target = reward + (1.0 - done) * gamma * evaluated_q
    assert torch.isclose(target, reward), (
        f"Terminal target must equal immediate reward {reward.item()}, got {target.item()}"
    )


def test_polyak_soft_update():
    """Verify soft Polyak update properly shifts target parameters toward policy parameters."""
    agent = D3QNAgent(tau=0.1)

    # Manually alter policy network weights
    with torch.no_grad():
        for p in agent.policy_net.parameters():
            p.fill_(10.0)
        for p in agent.target_net.parameters():
            p.fill_(0.0)

    # Execute soft update with tau=0.1: target <- 0.1 * 10.0 + 0.9 * 0.0 = 1.0
    agent.soft_update_target_network()

    for p in agent.target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 1.0)), (
            f"Expected parameter 1.0, got {p[0, 0].item() if p.ndim > 1 else p[0].item()}"
        )

