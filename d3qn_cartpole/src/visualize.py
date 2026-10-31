"""
D3QN Visual Performance & Learning Curve Generator.

Renders publication-ready convergence plots featuring raw episode scores,
100-episode moving averages, exploration epsilon decay curves, and the 490-point benchmark.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import LEARNING_CURVE_PATH, SOLVED_THRESHOLD


def plot_learning_curve(
    rewards: List[float],
    moving_averages: List[float],
    epsilons: Optional[List[float]] = None,
    solved_threshold: float = SOLVED_THRESHOLD,
    save_path: Path = LEARNING_CURVE_PATH,
) -> None:
    """
    Render and save dual-axis D3QN learning curve chart.

    Parameters
    ----------
    rewards : List[float]
        Raw per-episode rewards.
    moving_averages : List[float]
        100-episode moving average trajectory.
    epsilons : Optional[List[float]]
        Epsilon exploration progression.
    solved_threshold : float
        Benchmark threshold line (490.0).
    save_path : Path
        PNG save path.
    """
    episodes = range(1, len(rewards) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#f8fafc")

    # Raw Episode Rewards (Light Cyan)
    ax1.plot(
        episodes,
        rewards,
        label="Raw Episode Reward",
        color="#38bdf8",
        alpha=0.55,
        linewidth=1.2,
    )

    # Rolling Moving Average (Bold Navy)
    ax1.plot(
        episodes,
        moving_averages,
        label="100-Episode Rolling Mean",
        color="#0369a1",
        linewidth=2.6,
    )

    # 490 Solved Benchmark Line (Emerald Green)
    ax1.axhline(
        y=solved_threshold,
        color="#059669",
        linestyle="--",
        linewidth=2.2,
        label=f"D3QN Solved Benchmark ({solved_threshold:.0f} pts)",
    )

    ax1.set_title("Dueling Double DQN (D3QN) CartPole-v1 Convergence Trajectory", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax1.set_ylabel("Total Episode Reward (Timesteps Balanced)", fontsize=11, labelpad=8)
    ax1.set_ylim(-10, 520)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")

    # Right axis for Epsilon
    if epsilons is not None and len(epsilons) == len(rewards):
        ax2 = ax1.twinx()
        ax2.plot(
            episodes,
            epsilons,
            label="Epsilon Exploration",
            color="#ea580c",
            linestyle=":",
            linewidth=1.8,
        )
        ax2.set_ylabel(r"Exploration Epsilon ($\epsilon$)", fontsize=11, color="#ea580c", labelpad=8)
        ax2.tick_params(axis="y", labelcolor="#ea580c")
        ax2.set_ylim(0.0, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True, facecolor="#ffffff")
    else:
        ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"D3QN learning curve exported to: {save_path.resolve()}")

