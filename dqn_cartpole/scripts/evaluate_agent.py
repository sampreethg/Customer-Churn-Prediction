"""
Evaluation Script for Trained DQN CartPole-v1 Agent.

Loads saved checkpoint weights, executes deterministic policy trials (epsilon = 0.0),
evaluates pole balance stability, and prints per-trial metrics.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import DQNAgent
from src.config import ENV_NAME, MAX_STEPS_PER_EPISODE, MODEL_SAVE_PATH


def evaluate_agent(
    model_path: Path = MODEL_SAVE_PATH,
    env_name: str = ENV_NAME,
    num_episodes: int = 10,
    render_mode: str | None = None,
) -> float:
    """
    Run deterministic policy trials on the environment.

    Parameters
    ----------
    model_path : Path
        Location of the serialized PyTorch policy network state dict.
    env_name : str
        Gymnasium environment name.
    num_episodes : int
        Number of test evaluation episodes.
    render_mode : str | None
        Optional rendering mode ("human", "rgb_array", or None).

    Returns
    -------
    float
        Mean reward across evaluation episodes.
    """
    print("\n" + "=" * 65)
    print(f" EVALUATING TRAINED DQN POLICY: {env_name}")
    print("=" * 65)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {model_path.resolve()}.\n"
            "Run 'python main.py' to train the agent before evaluation."
        )

    agent = DQNAgent()
    agent.load(model_path)
    print(f"Loaded model weights from: {model_path.resolve()}")
    print(f"Executing {num_episodes} evaluation episodes with greedy policy (epsilon = 0.0)...\n")

    env = gym.make(env_name, render_mode=render_mode)
    rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=1000 + episode)
        total_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # Deterministic greedy action selection
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        status = "[PERFECT 500]" if total_reward >= 500.0 else f"[{total_reward:.0f} steps]"
        print(f" Trial {episode:02d}/{num_episodes:02d} | Reward: {total_reward:5.1f} / {MAX_STEPS_PER_EPISODE} {status}")

    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print("\n" + "-" * 65)
    print(f"Evaluation Summary ({num_episodes} Trials):")
    print(f" Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f" Max Reward: {max(rewards):.2f}")
    print(f" Min Reward: {min(rewards):.2f}")
    print(f" Success Rate (500): {sum(1 for r in rewards if r >= 500.0) / num_episodes * 100:.1f}%")
    print("-" * 65 + "\n")

    return mean_reward


if __name__ == "__main__":
    evaluate_agent()

