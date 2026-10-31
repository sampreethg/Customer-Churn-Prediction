"""
DQN Training Loop, Target Network Synchronization, and Convergence Tracking.

Orchestrates training episodes, step-level policy updates, exponential epsilon decay,
100-episode rolling reward monitoring, and automatic checkpointing upon reaching solved criteria.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import torch

from .agent import DQNAgent
from .config import (
    ENV_NAME,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    EVAL_WINDOW,
    MAX_EPISODES,
    MODEL_SAVE_PATH,
    RANDOM_SEED,
    SOLVED_REWARD,
    TARGET_UPDATE_FREQ,
    TAU,
)


def train_dqn(
    agent: Optional[DQNAgent] = None,
    env_name: str = ENV_NAME,
    max_episodes: int = MAX_EPISODES,
    solved_reward: float = SOLVED_REWARD,
    eval_window: int = EVAL_WINDOW,
    checkpoint_path: Path = MODEL_SAVE_PATH,
    use_soft_update: bool = True,
    seed: int = RANDOM_SEED,
) -> Tuple[DQNAgent, Dict[str, List[float]]]:
    """
    Train DQN agent on the continuous CartPole balancing environment.

    Parameters
    ----------
    agent : Optional[DQNAgent]
        Pre-instantiated agent. If None, initializes a new DQNAgent instance.
    env_name : str, default="CartPole-v1"
        Gymnasium environment ID.
    max_episodes : int, default=600
        Maximum training episodes.
    solved_reward : float, default=475.0
        Rolling mean reward threshold indicating environment solution.
    eval_window : int, default=100
        Number of consecutive episodes over which rolling mean is computed.
    checkpoint_path : Path
        Filepath to persist model weights when solved threshold is satisfied.
    use_soft_update : bool, default=True
        Whether to apply soft Polyak target updates per step (True) or periodic hard updates (False).
    seed : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[DQNAgent, Dict[str, List[float]]]
        Trained agent and history dictionary containing episode rewards and moving averages.
    """
    env = gym.make(env_name)
    if agent is None:
        agent = DQNAgent()

    epsilon = EPSILON_START
    episode_rewards: List[float] = []
    moving_averages: List[float] = []
    epsilons: List[float] = []
    recent_rewards: deque[float] = deque(maxlen=eval_window)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    is_solved = False

    print("\n" + "=" * 65)
    print(f"        STARTING DQN TRAINING: {env_name}")
    print("=" * 65)
    print(f"Max Episodes:      {max_episodes}")
    print(f"Solved Threshold:  {solved_reward} (over {eval_window} consecutive episodes)")
    print(f"Replay Capacity:   {agent.memory.capacity}")
    print(f"Batch Size:        {agent.batch_size}")
    print(f"Discount (Gamma):  {agent.gamma}")
    print(f"Target Sync Mode:  {'Soft Polyak (tau=0.005)' if use_soft_update else f'Hard (every {TARGET_UPDATE_FREQ} eps)'}")
    print(f"Compute Device:    {agent.device}\n")

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition: flag termination only if cart left track or pole tipped over
            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
            )

            # Optimization step
            agent.update_policy()

            # Soft target network update at each step if enabled
            if use_soft_update:
                agent.update_target_network(soft=True, tau=TAU)

            state = next_state
            episode_reward += reward

        # Periodic hard target update if soft update is disabled
        if not use_soft_update and episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network(soft=False)

        # Decay exploration rate
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Record metrics
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        epsilons.append(epsilon)

        current_moving_avg = float(np.mean(recent_rewards))
        moving_averages.append(current_moving_avg)

        # Progress reporting
        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:03d}/{max_episodes:03d} | "
                f"Reward: {episode_reward:5.1f} | "
                f"Rolling Avg ({len(recent_rewards):03d} eps): {current_moving_avg:5.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # Check solved criterion
        if len(recent_rewards) == eval_window and current_moving_avg >= solved_reward:
            print("\n" + "*" * 65)
            print(f" [SOLVED] Environment SOLVED in {episode} episodes!")
            print(f" Rolling Average Reward: {current_moving_avg:.2f} >= {solved_reward}")
            print(f" Saving optimal checkpoint to: {checkpoint_path.resolve()}")
            print("*" * 65 + "\n")
            agent.save(checkpoint_path)
            is_solved = True
            break

    env.close()

    if not is_solved:
        print(f"\n[Notice] Max episodes reached. Best rolling average: {max(moving_averages):.2f}")
        # Always persist checkpoint for evaluation
        agent.save(checkpoint_path)

    history = {
        "episodes": list(range(1, len(episode_rewards) + 1)),
        "rewards": episode_rewards,
        "moving_averages": moving_averages,
        "epsilons": epsilons,
    }

    return agent, history

