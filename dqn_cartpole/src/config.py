"""
Hyperparameters and Configuration Constants for DQN CartPole Balancing.

Defines environment specifications, network dimensions, exploration parameters,
replay buffer memory bounds, optimization rates, and persistence filepaths.
"""

from pathlib import Path

# Environment Constants
ENV_NAME: str = "CartPole-v1"
STATE_DIM: int = 4 # [cart_pos, cart_velocity, pole_angle, pole_angular_velocity]
ACTION_DIM: int = 2 # [0: push cart left, 1: push cart right]
SOLVED_REWARD: float = 475.0 # Gymnasium CartPole-v1 threshold over 100 consecutive episodes
MAX_STEPS_PER_EPISODE: int = 500

# Reinforcement Learning Hyperparameters
BATCH_SIZE: int = 64
GAMMA: float = 0.99 # Bellman discount factor
EPSILON_START: float = 1.0 # Initial exploration rate
EPSILON_END: float = 0.05 # Minimum exploration rate
EPSILON_DECAY: float = 0.995 # Multiplicative decay per episode
LEARNING_RATE: float = 1e-3 # AdamW learning rate
BUFFER_CAPACITY: int = 10000 # Replay buffer maximum transitions
TARGET_UPDATE_FREQ: int = 10 # Episodes between hard target network weight synchronization
TAU: float = 0.005 # Soft Polyak target network update interpolation factor
MAX_EPISODES: int = 600 # Maximum training episodes
EVAL_WINDOW: int = 100 # Rolling evaluation window size
RANDOM_SEED: int = 42 # Deterministic seed for reproducibility

# File System Directories & Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

MODEL_SAVE_PATH: Path = MODELS_DIR / "best_dqn_cartpole.pt"
LEARNING_CURVE_PATH: Path = ARTIFACTS_DIR / "learning_curve.png"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

