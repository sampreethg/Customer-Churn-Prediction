"""
End-to-End Orchestration Pipeline for D3QN (Dueling Double DQN) CartPole-v1.

Executes:
1. Environment initialization, seed configuration, and parameter banner
2. D3QN training with experience replay, decoupled DDQN targets, and Polyak soft updates
3. Verification of rolling average reward convergence (>= 490.0)
4. Dual-axis convergence curve generation (artifacts/d3qn_learning_curve.png)
5. Post-training deterministic policy evaluation across 10 independent trials
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_d3qn_agent
from src.agent import D3QNAgent
from src.config import (
    ACTION_DIM,
    ARTIFACTS_DIR,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    ENV_ID,
    GAMMA,
    LEARNING_CURVE_PATH,
    LR,
    MAX_EPISODES,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    SEED,
    SOLVED_THRESHOLD,
    STATE_DIM,
    TAU,
)
from src.train import train_d3qn
from src.visualize import plot_learning_curve


def print_banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def main():
    print_banner("PRODUCTION DUELING DOUBLE DEEP Q-NETWORK (D3QN) CARTPOLE-V1")
    print(f"PyTorch Version:       {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:        {device}")
    print(f"Target Environment:    {ENV_ID}")
    print(f"State Dimension:       {STATE_DIM} continuous")
    print(f"Action Dimension:      {ACTION_DIM} discrete")
    print(f"Batch Size:            {BATCH_SIZE}")
    print(f"Replay Buffer:         {BUFFER_CAPACITY} transitions")
    print(f"Discount (Gamma):      {GAMMA}")
    print(f"Learning Rate (AdamW): {LR}")
    print(f"Target Sync (Polyak):  Tau = {TAU}")
    print(f"Convergence Target:    >= {SOLVED_THRESHOLD:.1f} rolling mean reward (100 eps)\n")

    # Set seeds for deterministic initialization
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train D3QN Agent
    agent = D3QNAgent(device=device)
    trained_agent, history = train_d3qn(
        agent=agent,
        env_id=ENV_ID,
        max_episodes=MAX_EPISODES,
        solved_threshold=SOLVED_THRESHOLD,
        checkpoint_path=MODEL_SAVE_PATH,
        seed=SEED,
    )

    # 2. Export Learning Curve Visualization
    print_banner("GENERATING CONVERGENCE VISUALIZATION")
    plot_learning_curve(
        rewards=history["rewards"],
        moving_averages=history["moving_averages"],
        epsilons=history["epsilons"],
        solved_threshold=SOLVED_THRESHOLD,
        save_path=LEARNING_CURVE_PATH,
    )

    # 3. Post-Training Deterministic Evaluation
    print_banner("DETERMINISTIC EVALUATION BENCHMARK")
    mean_eval_reward = evaluate_d3qn_agent(
        model_path=MODEL_SAVE_PATH,
        env_id=ENV_ID,
        num_episodes=10,
    )

    # 4. Final Verification Summary
    print_banner("EXECUTION & ARTIFACTS SUMMARY")
    artifacts = [
        ("D3QN Checkpoint", MODEL_SAVE_PATH),
        ("Learning Curve Chart", LEARNING_CURVE_PATH),
    ]

    for label, path in artifacts:
        exists = path.is_file()
        size_kb = path.stat().st_size / 1024 if exists else 0.0
        status_str = f"[OK] ({size_kb:.1f} KB)" if exists else "[MISSING]"
        print(f"  {label:<28}: {status_str:<18} -> {path.name}")

    print(f"\nFinal Deterministic Evaluation Score: {mean_eval_reward:.1f} / 500.0 pts")
    if mean_eval_reward >= SOLVED_THRESHOLD:
        print(f"SUCCESS: D3QN policy demonstrates outstanding pole balance (>= {SOLVED_THRESHOLD:.0f} pts)!")
    else:
        print(f"COMPLETED: D3QN policy trained and evaluated ({mean_eval_reward:.1f} pts).")


if __name__ == "__main__":
    main()

