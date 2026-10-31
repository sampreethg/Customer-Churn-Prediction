"""
End-to-End Orchestration Pipeline for Deep Q-Network (DQN) CartPole-v1.

Executes:
1. Environment initialization and hyperparameter banner
2. Training loop with experience replay and soft Polyak target synchronization
3. Verification of rolling average reward convergence (>= 475.0)
4. Learning curve visualization generation (artifacts/learning_curve.png)
5. Post-training deterministic policy evaluation across 10 independent trials
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.evaluate_agent import evaluate_agent

from src.agent import DQNAgent
from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    ENV_NAME,
    GAMMA,
    LEARNING_CURVE_PATH,
    LEARNING_RATE,
    MAX_EPISODES,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    SOLVED_REWARD,
    TAU,
)
from src.train import train_dqn
from src.visualize import plot_learning_curve


def print_banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def main():
    print_banner("PRODUCTION DEEP Q-NETWORK (DQN) CARTPOLE-V1 SYSTEM")
    print(f"PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    print(f"Target Environment: {ENV_NAME}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Replay Buffer: {BUFFER_CAPACITY} transitions")
    print(f"Discount (Gamma): {GAMMA}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Target Sync: Polyak Soft Update (Tau={TAU})")
    print(f"Solved Goal: >= {SOLVED_REWARD} rolling reward\n")

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train Agent
    agent = DQNAgent(device=device)
    trained_agent, history = train_dqn(
        agent=agent,
        env_name=ENV_NAME,
        max_episodes=MAX_EPISODES,
        solved_reward=SOLVED_REWARD,
        checkpoint_path=MODEL_SAVE_PATH,
        use_soft_update=True,
    )

    # 2. Export Diagnostic Learning Curves
    print_banner("GENERATING CONVERGENCE VISUALIZATION")
    plot_learning_curve(
        rewards=history["rewards"],
        moving_averages=history["moving_averages"],
        epsilons=history["epsilons"],
        solved_reward=SOLVED_REWARD,
        save_path=LEARNING_CURVE_PATH,
    )

    # 3. Post-Training Policy Evaluation
    print_banner("DETERMINISTIC EVALUATION BENCHMARK")
    mean_eval_reward = evaluate_agent(
        model_path=MODEL_SAVE_PATH,
        env_name=ENV_NAME,
        num_episodes=10,
    )

    # 4. Final Verification Summary
    print_banner("EXECUTION & ARTIFACTS SUMMARY")
    artifacts = [
        ("Trained Policy Checkpoint", MODEL_SAVE_PATH),
        ("Learning Curve Chart", LEARNING_CURVE_PATH),
    ]

    for label, path in artifacts:
        exists = path.is_file()
        size_kb = path.stat().st_size / 1024 if exists else 0.0
        status_str = f"[OK] ({size_kb:.1f} KB)" if exists else "[MISSING]"
        print(f" {label:<28}: {status_str:<18} -> {path.name}")

    print(f"\nFinal Deterministic Evaluation Score: {mean_eval_reward:.1f} / 500.0 pts")
    if mean_eval_reward >= SOLVED_REWARD:
        print(f"SUCCESS: Policy achieves optimal CartPole-v1 balance (>= {SOLVED_REWARD:.0f} pts)!")
    else:
        print(f"COMPLETED: Policy trained and evaluated ({mean_eval_reward:.1f} pts).")


if __name__ == "__main__":
    main()

