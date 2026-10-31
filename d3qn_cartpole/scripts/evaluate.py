"""
Evaluation Script for Trained D3QN CartPole-v1 Agent.

Executes deterministic greedy evaluation episodes (epsilon = 0.0) without exploration,
evaluates policy stability, and logs step counts per trial.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import D3QNAgent
from src.config import ENV_ID, MAX_STEPS_PER_EPISODE, MODEL_SAVE_PATH


def evaluate_d3qn_agent(
    model_path: Path = MODEL_SAVE_PATH,
    env_id: str = ENV_ID,
    num_episodes: int = 10,
    render_mode: str | None = None,
) -> float:
    """
    Run deterministic evaluation trials using the saved D3QN model.

    Parameters
    ----------
    model_path : Path
        Path to saved PyTorch policy network state dict.
    env_id : str
        Gymnasium environment ID.
    num_episodes : int
        Number of trials to execute.
    render_mode : str | None
        Optional render mode.

    Returns
    -------
    float
        Average reward score across evaluation trials.
    """
    print("\n" + "=" * 65)
    print(f" D3QN DETERMINISTIC POLICY EVALUATION: {env_id}")
    print("=" * 65)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {model_path.resolve()}.\n"
            "Train the agent by running 'python main.py' first."
        )

    agent = D3QNAgent()
    agent.load(model_path)
    print(f"Loaded Dueling policy weights from: {model_path.resolve()}")
    print(f"Executing {num_episodes} evaluation episodes with zero exploration (epsilon = 0.0)...\n")

    env = gym.make(env_id, render_mode=render_mode)
    rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=2000 + episode)
        total_reward = 0.0
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step_count += 1

        rewards.append(total_reward)
        tag = "[PERFECT 500]" if total_reward >= 500.0 else f"[{total_reward:.0f} steps]"
        print(f" Trial {episode:02d}/{num_episodes:02d} | Reward: {total_reward:5.1f} / {MAX_STEPS_PER_EPISODE} {tag}")

    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print("\n" + "-" * 65)
    print(f"D3QN Evaluation Summary ({num_episodes} Trials):")
    print(f" Mean Score: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f" Max Score: {max(rewards):.2f}")
    print(f" Min Score: {min(rewards):.2f}")
    print(f" Success Rate (500): {sum(1 for r in rewards if r >= 500.0) / num_episodes * 100:.1f}%")
    print("-" * 65 + "\n")

    return mean_reward


if __name__ == "__main__":
    evaluate_d3qn_agent()

