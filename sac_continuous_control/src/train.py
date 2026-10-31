"""Training Pipeline for Soft Actor-Critic (SAC) on Continuous Gymnasium Control."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Ensure path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent import SACAgent
from src.config import SACConfig
from src.replay_buffer import ContinuousReplayBuffer


def train_sac(config: SACConfig = None):
    config = config or SACConfig()
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)

    # Set deterministic seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create Environments
    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)

    agent = SACAgent(config)
    buffer = ContinuousReplayBuffer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        capacity=config.buffer_capacity,
    )

    state, _ = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_idx = 0

    best_eval_return = -float("inf")
    return_history = []

    print("=" * 60)
    print(f"Starting SAC Training on {config.env_id}")
    print(f"State Dim: {config.state_dim} | Action Dim: {config.action_dim} | Action High: {config.action_high}")
    print("=" * 60)

    for total_step in range(1, config.total_steps + 1):
        if total_step <= config.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        episode_steps += 1

        if total_step > config.warmup_steps:
            batch = buffer.sample(config.batch_size, agent.device)
            agent.update_parameters(batch)

        if done:
            episode_idx += 1
            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        # Periodic Evaluation
        if total_step % config.eval_interval == 0:
            eval_returns = []
            for _ in range(config.eval_episodes):
                e_state, _ = eval_env.reset()
                e_done = False
                e_ret = 0.0
                while not e_done:
                    e_act = agent.select_action(e_state, evaluate=True)
                    e_state, e_rew, e_term, e_trunc, _ = eval_env.step(e_act)
                    e_ret += e_rew
                    e_done = e_term or e_trunc
                eval_returns.append(e_ret)

            mean_eval = float(np.mean(eval_returns))
            return_history.append((total_step, mean_eval))
            print(
                f"[Step {total_step:6d}/{config.total_steps}] "
                f"Eval Return: {mean_eval:7.2f} | Alpha (Temp): {agent.alpha:.4f} | Buffer Size: {len(buffer)}"
            )

            if mean_eval > best_eval_return:
                best_eval_return = mean_eval
                checkpoint_path = config.models_dir / "best_sac_agent.pt"
                agent.save_checkpoint(str(checkpoint_path))

    env.close()
    eval_env.close()

    print("=" * 60)
    print(f"SAC Training Complete! Best Evaluation Return: {best_eval_return:.2f}")
    print("=" * 60)
    return agent, return_history


if __name__ == "__main__":
    train_sac()
