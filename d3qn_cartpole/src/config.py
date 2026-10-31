"""
Hyperparameters and Configuration Constants for D3QN CartPole-v1.

Defines environment specifications, Dueling network dimensions, Double DQN learning rates,
experience replay buffer bounds, exploration schedules, and persistence paths.
"""

from pathlib import Path

# Environment Constants
ENV_ID: str = "CartPole-v1"
STATE_DIM: int = 4        # [cart_pos, cart_velocity, pole_angle, pole_angular_velocity]
ACTION_DIM: int = 2       # [0: push cart left, 1: push cart right]
SOLVED_THRESHOLD: float = 490.0  # D3QN benchmark criteria over 100 consecutive episodes
MAX_STEPS_PER_EPISODE: int = 500

# D3QN Hyperparameters
BATCH_SIZE: int = 64
GAMMA: float = 0.99               # Bellman temporal discount factor
TAU: float = 0.005                # Soft Polyak target network update interpolation rate
LR: float = 1e-3                  # AdamW learning rate
BUFFER_CAPACITY: int = 20000      # Bounded experience replay memory capacity
EPSILON_START: float = 1.0        # Initial exploration probability
EPSILON_END: float = 0.02         # Exploration floor
EPSILON_DECAY: float = 0.995      # Multiplicative exploration decay per episode
MAX_EPISODES: int = 500           # Maximum training episodes
EVAL_WINDOW: int = 100            # Rolling evaluation window size
SEED: int = 42                    # Deterministic random seed

# File System Directories & Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

MODEL_SAVE_PATH: Path = MODELS_DIR / "best_d3qn_cartpole.pt"
LEARNING_CURVE_PATH: Path = ARTIFACTS_DIR / "d3qn_learning_curve.png"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

