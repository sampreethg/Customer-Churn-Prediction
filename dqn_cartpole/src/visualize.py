"""
Training Convergence and Learning Curve Visualization Module.

Generates high-resolution performance plots depicting per-episode raw reward scores,
rolling moving averages, exploration epsilon decay curves, and the 475-point solved benchmark.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import LEARNING_CURVE_PATH, SOLVED_REWARD


def plot_learning_curve(
    rewards: List[float],
    moving_averages: List[float],
    epsilons: Optional[List[float]] = None,
    solved_reward: float = SOLVED_REWARD,
    save_path: Path = LEARNING_CURVE_PATH,
) -> None:
    """
    Render and persist the DQN learning curve.

    Parameters
    ----------
    rewards : List[float]
        Raw total reward accumulated in each episode.
    moving_averages : List[float]
        100-episode rolling mean reward.
    epsilons : Optional[List[float]]
        Epsilon decay trajectory across episodes.
    solved_reward : float, default=475.0
        Benchmark goal line.
    save_path : Path
        Filesystem destination for exported PNG graphic.
    """
    episodes = range(1, len(rewards) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#f8fafc")

    # Raw Episode Rewards (Light Cyan/Blue)
    ax1.plot(
        episodes,
        rewards,
        label="Raw Episode Reward",
        color="#93c5fd",
        alpha=0.6,
        linewidth=1.2,
    )

    # Rolling Moving Average (Bold Royal Blue)
    ax1.plot(
        episodes,
        moving_averages,
        label="100-Episode Rolling Mean",
        color="#1d4ed8",
        linewidth=2.5,
    )

    # Benchmark Solved Threshold Line (Red Dashed)
    ax1.axhline(
        y=solved_reward,
        color="#dc2626",
        linestyle="--",
        linewidth=2.0,
        label=f"Solved Threshold ({solved_reward:.0f} pts)",
    )

    ax1.set_title("Deep Q-Network (DQN) CartPole-v1 Convergence Trajectory", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax1.set_ylabel("Total Episode Reward (Balance Steps)", fontsize=11, labelpad=8)
    ax1.set_ylim(-10, 520)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")

    # Optional secondary y-axis for Epsilon exploration decay
    if epsilons is not None and len(epsilons) == len(rewards):
        ax2 = ax1.twinx()
        ax2.plot(
            episodes,
            epsilons,
            label="Epsilon (Exploration Rate)",
            color="#d97706",
            linestyle=":",
            linewidth=1.8,
        )
        ax2.set_ylabel("Exploration Epsilon ($\epsilon$)", fontsize=11, color="#d97706", labelpad=8)
        ax2.tick_params(axis="y", labelcolor="#d97706")
        ax2.set_ylim(0.0, 1.05)

        # Merge legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", frameon=True, facecolor="#ffffff")
    else:
        ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Learning curve visualization exported to: {save_path.resolve()}")

