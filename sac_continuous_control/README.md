# Soft Actor-Critic (SAC) Continuous Control System

Production-grade PyTorch and Gymnasium implementation of the Soft Actor-Critic (SAC) algorithm for continuous torque and robotic control, benchmarked on `Pendulum-v1`.

---

## Key Algorithmic Pillars

1. **Maximum Entropy Reinforcement Learning**:
   Augments standard expected discounted cumulative returns with a policy entropy penalty:
   $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
   where $\alpha$ is the entropy temperature determining the trade-off between reward maximization and omnidirectional policy exploration.

2. **Twin Q-Critics ($Q_1, Q_2$)**:
   Mitigates positive maximization bias in continuous action evaluation by maintaining two independent parameterized critics and evaluating Bellman targets using the minimum estimate:
   $$y = r + \gamma (1 - d) \left( \min_{i \in \{1, 2\}} Q_{\text{target}, i}(s', \tilde{a}') - \alpha \log \pi(\tilde{a}' | s') \right)$$

3. **Reparameterized Squashed Gaussian Policy**:
   Outputs state-conditioned action distributions using the pathwise reparameterization trick:
   $$\tilde{a} = a_{\text{scale}} \cdot \tanh(\mu(s) + \sigma(s) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$
   With exact Jacobian correction for invertible nonlinear squashing:
   $$\log \pi(\tilde{a} | s) = \log \mu(u | s) - \sum_{i=1}^D \log \left( a_{\text{scale}} (1 - \tanh^2(u_i)) + \epsilon \right)$$

4. **Automatic Entropy Temperature Tuning**:
   Updates $\log \alpha$ online using gradient descent to maintain a predefined target entropy $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A})$:
   $$J(\alpha) = \mathbb{E}_{a_t \sim \pi_t} \left[ -\alpha \log \pi_t(a_t | s_t) - \alpha \bar{\mathcal{H}} \right]$$

---

## Directory Layout

```
sac_continuous_control/
 src/
 __init__.py
 config.py # Hyperparameters & path configurations
 model.py # GaussianPolicy & TwinQNetwork
 agent.py # SACAgent coordinating updates
 replay_buffer.py # Continuous numpy experience buffer
 train.py # Training and evaluation runner
 tests/
 __init__.py
 test_sac.py # Unit tests for policy bounds & gradients
 requirements.txt
 README.md
```

---

## Testing

Execute the test suite:
```bash
pytest sac_continuous_control/tests/ -v
```
