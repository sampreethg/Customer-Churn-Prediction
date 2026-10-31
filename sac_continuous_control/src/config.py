"""Configuration dataclass for Soft Actor-Critic (SAC) on continuous control."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"
    state_dim: int = 3
    action_dim: int = 1
    action_high: float = 2.0
    action_low: float = -2.0
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2 # Initial entropy temperature
    auto_tune_alpha: bool = True
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    hidden_dim: int = 256
    buffer_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 1_000
    total_steps: int = 20_000
    eval_interval: int = 1_000
    eval_episodes: int = 5
    seed: int = 42

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = base_dir / "models"
    plots_dir: Path = base_dir / "artifacts"
