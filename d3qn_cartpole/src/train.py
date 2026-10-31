"""
D3QN Training Loop and Convergence Verification.

Executes episodic interactions, step-level parameter optimization, epsilon decay,
rolling performance tracking, and automated checkpoint serialization when average reward >= 490.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
import torch

from .agent import D3QNAgent
from .config import (
    ENV_ID,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    EVAL_WINDOW,
    MAX_EPISODES,
    MODEL_SAVE_PATH,
    SEED,
    SOLVED_THRESHOLD,
)


def train_d3qn(
    agent: Optional[D3QNAgent] = None,
    env_id: str = ENV_ID,
    max_episodes: int = MAX_EPISODES,
    solved_threshold: float = SOLVED_THRESHOLD,
    eval_window: int = EVAL_WINDOW,
    checkpoint_path: Path = MODEL_SAVE_PATH,
    seed: int = SEED,
) -> Tuple[D3QNAgent, Dict[str, List[float]]]:
    """
    Train D3QN agent to optimal CartPole-v1 balance.

    Parameters
    ----------
    agent : Optional[D3QNAgent]
        Pre-configured agent instance.
    env_id : str
        Gymnasium environment ID.
    max_episodes : int
        Maximum training episodes.
    solved_threshold : float
        Benchmark reward target over 100 consecutive episodes (490.0).
    eval_window : int
        Rolling window length (100).
    checkpoint_path : Path
        Filepath for saving solved model checkpoint.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[D3QNAgent, Dict[str, List[float]]]
        Trained agent and history metrics dictionary.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    if agent is None:
        agent = D3QNAgent()

    epsilon = EPSILON_START
    episode_rewards: List[float] = []
    moving_averages: List[float] = []
    epsilons: List[float] = []
    recent_rewards: deque[float] = deque(maxlen=eval_window)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    is_solved = False

    print("\n" + "=" * 65)
    print(f"        STARTING D3QN TRAINING: {env_id}")
    print("=" * 65)
    print(f"Max Episodes:      {max_episodes}")
    print(f"Solved Threshold:  {solved_threshold} (over {eval_window} consecutive episodes)")
    print(f"Replay Capacity:   {agent.memory.capacity}")
    print(f"Batch Size:        {agent.batch_size}")
    print(f"Discount (Gamma):  {agent.gamma}")
    print(f"Target Update:     Polyak Soft (Tau={agent.tau})")
    print(f"Compute Device:    {agent.device}\n")

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer (terminated=True only if dropped or left track)
            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated,
            )

            # Perform Bellman gradient step and soft target update
            agent.update_parameters()

            state = next_state
            episode_reward += reward

        # Multiplicative epsilon decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        epsilons.append(epsilon)

        rolling_avg = float(np.mean(recent_rewards))
        moving_averages.append(rolling_avg)

        # Periodic logging
        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:03d}/{max_episodes:03d} | "
                f"Reward: {episode_reward:5.1f} | "
                f"Rolling Avg ({len(recent_rewards):03d} eps): {rolling_avg:5.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

        # Check solved criterion
        if len(recent_rewards) == eval_window and rolling_avg >= solved_threshold:
            print("\n" + "*" * 65)
            print(f" [SOLVED] D3QN Solved CartPole-v1 in {episode} episodes!")
            print(f" Rolling Average Reward: {rolling_avg:.2f} >= {solved_threshold}")
            print(f" Persisting optimal model checkpoint to: {checkpoint_path.resolve()}")
            print("*" * 65 + "\n")
            agent.save(checkpoint_path)
            is_solved = True
            break

    env.close()

    if not is_solved:
        print(f"\n[Completed] Reached episode {max_episodes}. Peak rolling avg: {max(moving_averages):.2f}")
        agent.save(checkpoint_path)

    history = {
        "episodes": list(range(1, len(episode_rewards) + 1)),
        "rewards": episode_rewards,
        "moving_averages": moving_averages,
        "epsilons": epsilons,
    }

    return agent, history

