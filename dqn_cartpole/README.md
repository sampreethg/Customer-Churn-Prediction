# Production Deep Q-Network (DQN) for CartPole-v1

A production-grade Reinforcement Learning system implemented in Python using **PyTorch** and **Gymnasium** to solve the continuous-state `CartPole-v1` environment.

---

## 1. Mathematical Formulation

### The Bellman Optimality Equation
Q-learning estimates the action-value function $Q^*(s, a)$, representing the expected cumulative discounted future reward obtained by executing action $a$ in state $s$ and following the optimal policy thereafter:

$$Q^*(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a \right]$$

where:
- $s \in \mathbb{R}^4$ is the continuous state vector `[cart_pos, cart_velocity, pole_angle, pole_angular_velocity]`
- $a \in \{0, 1\}$ is the discrete action (`0: push cart left`, `1: push cart right`)
- $r \in \mathbb{R}$ is the immediate scalar reward ($+1$ for every sustained balance timestep)
- $\gamma \in [0, 1)$ is the temporal discount factor ($\gamma = 0.99$)
- $s'$ is the successor state observed after transition.

### Tabular Space Limitations & Function Approximation
The CartPole observation space is infinite and continuous ($\mathbb{R}^4$). Tabular Q-learning cannot generalize across continuous state manifolds. A deep multi-layer neural network parameterized by $\theta$ acts as a non-linear function approximator:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

### Mitigating Instability: Experience Replay
In standard online reinforcement learning, consecutive samples $(s_t, a_t, r_t, s_{t+1})$ are strongly correlated across time, violating the Independent and Identically Distributed (i.i.d.) assumption of gradient descent. Furthermore, policy updates shift the data distribution (non-stationarity).

An **Experience Replay Buffer** $\mathcal{D} = \{e_1, \dots, e_N\}$ of capacity $N = 10,000$ stores historical transitions $e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$. Mini-batches of size $B = 64$ are sampled uniformly at random:

$$\mathcal{B} \sim U(\mathcal{D})$$

This decorrelates training samples and stabilizes gradient updates across a broader empirical state distribution.

### Eliminating Moving Targets: Target Network
In naive Q-learning, the target value depends on the same parameters $\theta$ being updated:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta)$$

Updating $\theta$ simultaneously shifts both predictions and targets, causing feedback loops, oscillations, and divergence. A separate frozen **Target Network** $Q(s, a; \theta^-)$ is maintained. The target is defined as:

$$y_j = r_j + (1 - d_j) \cdot \gamma \cdot \max_{a'} Q(s'_j, a'; \theta^-)$$

### Target Synchronization Methods
1. **Periodic Hard Update**: Every $C = 10$ episodes, parameters are copied directly:
   $$\theta^- \leftarrow \theta$$
2. **Polyak Soft Update (Exponential Smoothing)**: After every gradient step, parameters are smoothly interpolated:
   $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \quad (\tau = 0.005)$$

### Loss Function & Huber Loss Optimization
To prevent gradient explosion from large temporal difference (TD) errors $\delta = Q(s, a; \theta) - y$, we use the **Huber Loss (Smooth L1)**:

$$L(\delta) = \begin{cases} 
\frac{1}{2}\delta^2 & \text{for } |\delta| \le 1 \\
|\delta| - \frac{1}{2} & \text{otherwise}
\end{cases}$$

Optimization is performed using `torch.optim.AdamW(lr=1e-3, amsgrad=True)` with gradient norm clipping clamped at 100.

---

## 2. Neural Architecture

```
State Input s in R^4 [x, x_dot, theta, theta_dot]
                
                

 Linear (4 -> 128) -> He Normal Init 
 Activation: ReLU 

                
                

 Linear (128 -> 128) -> He Normal Init 
 Activation: ReLU 

                
                

 Linear (128 -> 2) -> Xavier Uniform Init 

                
                
Action Values: [ Q(s, a=0), Q(s, a=1) ]
```

---

## 3. Project Structure

```
dqn_cartpole/
 src/
 __init__.py # Package initializer
 config.py # Hyperparameters, environment dimensions, filepaths
 model.py # PyTorch DQN neural network architecture
 replay_buffer.py # Transition dataclass and bounded Experience Replay Buffer
 agent.py # DQNAgent class (epsilon-greedy, replay, Bellman updates)
 train.py # Training loop, target sync, reward tracking, checkpointing
 visualize.py # Learning curve visualization and 100-episode moving average
 scripts/
 evaluate_agent.py # Deterministic evaluation trials with performance summaries
 tests/
 __init__.py # Test package initializer
 test_replay_buffer.py # Capacity boundaries, FIFO eviction, sampling tensor types
 test_model.py # Forward pass shapes, gradient propagation, and initialization
 models/
 best_dqn_cartpole.pt # Saved optimal PyTorch policy network weights
 artifacts/
 learning_curve.png # 600-dpi convergence plot against 475-point benchmark
 main.py # Orchestration pipeline (Train -> Verify Solved -> Evaluate)
 requirements.txt # Production dependencies
 README.md # Mathematical formulation and operational instructions
```

---

## 4. Hyperparameter Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `ENV_NAME` | `CartPole-v1` | Continuous 4D observation, discrete 2D action space |
| `BATCH_SIZE` | `64` | Mini-batch sample size from replay buffer |
| `GAMMA` ($\gamma$) | `0.99` | Temporal discount factor |
| `EPSILON_START` | `1.0` | Initial random exploration probability |
| `EPSILON_END` | `0.05` | Minimum exploration probability floor |
| `EPSILON_DECAY` | `0.995` | Multiplicative decay per episode ($\epsilon \leftarrow \epsilon \cdot 0.995$) |
| `LEARNING_RATE` | `1e-3` | AdamW optimizer learning rate |
| `BUFFER_CAPACITY` | `10,000` | Maximum transitions in cyclic replay memory |
| `TARGET_UPDATE_FREQ` | `10` | Episodes between hard target syncs |
| `TAU` ($\tau$) | `0.005` | Polyak soft target network update rate |
| `SOLVED_REWARD` | `475.0` | 100-consecutive-episode rolling mean threshold |
| `MAX_STEPS` | `500` | Maximum achievable steps per CartPole-v1 episode |

---

## 5. Execution Instructions

### A. Environment Installation
```bash
cd dqn_cartpole
pip install -r requirements.txt
```

### B. Run Unit Test Suite
```bash
pytest tests/ -v
```
Validates experience replay capacity limits, FIFO replacement, tensor batch shapes, and network backpropagation gradient flow.

### C. End-to-End Training & Verification
```bash
python main.py
```
This trains the DQN agent, tracks convergence, persists `models/best_dqn_cartpole.pt` when the rolling average reaches $\ge 475$, exports `artifacts/learning_curve.png`, and runs 10 evaluation trials.

### D. Evaluate Saved Policy
```bash
python scripts/evaluate_agent.py
```
Executes 10 evaluation trials using the deterministic greedy policy ($\epsilon = 0.0$) and prints step survival counts and success rates.

