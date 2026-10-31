# Dueling Double Deep Q-Network (D3QN) for CartPole-v1

A production-grade Reinforcement Learning platform in PyTorch and Gymnasium implementing the **Dueling Double Deep Q-Network (D3QN)** algorithm on the continuous-state `CartPole-v1` environment.

---

## 1. Algorithmic Innovations & Mathematical Foundations

Standard Deep Q-Networks (DQN) suffer from two core theoretical limitations:
1. **Maximization Overestimation Bias**: In standard Bellman target updates, taking the maximum over noisy Q-value estimates systematically overestimates action values: $\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$.
2. **Coupled Representation**: In many states, action choice is irrelevant to survival, yet standard DQN is forced to learn action-values $Q(s, a)$ separately for all actions, leading to slow and inefficient representation learning.

D3QN resolves both challenges by synthesizing two foundational breakthroughs:

### A. Double DQN (Decoupling Action Selection from Evaluation)
Double DQN decouples greedy action selection from target evaluation using separate online ($\theta$) and target ($\theta^-$) networks:

$$y_j^{\text{DDQN}} = r_j + (1 - d_j) \cdot \gamma \cdot Q\left(s_j', \arg\max_{a'} Q(s_j', a'; \theta); \, \theta^-\right)$$

- **Selection**: The actively trained online policy network $\theta$ chooses the optimal greedy action: $a^* = \arg\max_{a'} Q(s_j', a'; \theta)$.
- **Evaluation**: The slowly-moving target network $\theta^-$ estimates the value of that chosen action: $Q(s_j', a^*; \theta^-)$.
- This eliminates the maximization operator over identical parameters, provably extinguishing upward estimation bias.

### B. Dueling Network Architecture (Identifiable Advantage Aggregation)
The network decomposes Q-values into a scalar state-value $V(s)$ and an action-advantage vector $A(s, a)$:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

- **Identifiability Constraint**: Simply adding $V(s) + A(s, a)$ is unidentifiable (infinitely many combinations of $V$ and $A$ produce identical $Q$). Subtracting the mean advantage forces the advantage stream to have zero mean across actions:
  $$\frac{1}{|\mathcal{A}|} \sum_{a} [Q(s, a) - V(s)] \equiv 0$$
- This allows $V(s)$ to learn the intrinsic value of the state independently of actions, accelerating feature discovery across the continuous state space.

### C. Continuous Polyak Soft Target Updates
Rather than abrupt periodic target replacements, target weights evolve smoothly at every step:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \quad (\tau = 0.005)$$

---

## 2. Dueling Architecture Layout

```
Continuous State Observation s in R^4 [x, x_dot, theta, theta_dot]
                        
                        

 Shared Feature Trunk: Linear (4 -> 128) -> ReLU() 

             
             
 
 State-Value Stream V(s) Advantage Stream A(s, a) 
 Linear(128 -> 64) -> ReLU() Linear(128 -> 64) -> ReLU
 Linear(64 -> 1) Linear(64 -> 2) 
 
             [B, 1] [B, 2]
            
                                

 Identifiability Aggregation Layer: 
 Q(s, a) = V(s) + [ A(s, a) - 1/|A| * sum_{a'} A(s, a') ] 

                                
                                
                  Action-Values Q(s, ·) in R^2
```

---

## 3. Project Structure

```
d3qn_cartpole/
 src/
 __init__.py # Package initialization
 config.py # Hyperparameters (lr, gamma, tau, batch_size, buffer, epsilon)
 model.py # DuelingDQN architecture with mean-centered aggregation
 replay_buffer.py # Bounded experience replay buffer returning batched tensors
 agent.py # D3QNAgent (epsilon-greedy, Polyak soft update, DDQN target logic)
 train.py # Training loop, evaluation window tracker, checkpointing
 visualize.py # Dual-axis learning curves (smoothed rewards, moving avg, epsilon)
 scripts/
 evaluate.py # Deterministic policy evaluation running rendered test episodes
 tests/
 __init__.py # Test package initializer
 test_dueling_math.py # Unit tests verifying advantage mean-centering and stream shapes
 test_ddqn_targets.py # Unit tests confirming selection/evaluation separation
 models/
 best_d3qn_cartpole.pt # Checkpointed best PyTorch weights
 artifacts/
 d3qn_learning_curve.png # High-resolution convergence plot against 490-point benchmark
 main.py # Orchestration entry point (train -> assert solved -> export plots)
 requirements.txt # Production dependencies
 README.md # Complete mathematical formulation and run commands
```

---

## 4. Hyperparameter Specifications

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `ENV_ID` | `CartPole-v1` | Gymnasium CartPole environment |
| `BATCH_SIZE` | `64` | Mini-batch sample size from replay memory |
| `GAMMA` ($\gamma$) | `0.99` | Temporal discount factor |
| `TAU` ($\tau$) | `0.005` | Polyak soft target network update coefficient |
| `LR` | `1e-3` | AdamW optimizer learning rate (`amsgrad=True`) |
| `BUFFER_CAPACITY` | `20,000` | Replay buffer cyclic capacity |
| `EPSILON_START` | `1.0` | Initial exploration probability |
| `EPSILON_END` | `0.02` | Minimum exploration probability floor |
| `EPSILON_DECAY` | `0.995` | Multiplicative decay per episode ($\epsilon \leftarrow \epsilon \cdot 0.995$) |
| `MAX_EPISODES` | `500` | Maximum allowed training episodes |
| `SOLVED_THRESHOLD` | `490.0` | 100-consecutive-episode rolling mean benchmark |

---

## 5. Execution Instructions

### A. Environment Installation
```bash
cd d3qn_cartpole
pip install -r requirements.txt
```

### B. Run Unit Test Suite
```bash
pytest tests/ -v
```
Validates advantage mean-centering ($\frac{1}{|\mathcal{A}|}\sum [Q - V] = 0$), output dimensions, and strict decoupling of action selection and valuation in Double DQN targets.

### C. End-to-End Training & Verification
```bash
python main.py
```
Trains D3QN agent until the rolling average reward reaches $\ge 490.0$, saves `models/best_d3qn_cartpole.pt`, exports `artifacts/d3qn_learning_curve.png`, and runs 10 evaluation trials.

### D. Evaluate Saved Policy
```bash
python scripts/evaluate.py
```
Runs deterministic evaluation trials ($\epsilon = 0.0$) and reports step counts and balance stability.

