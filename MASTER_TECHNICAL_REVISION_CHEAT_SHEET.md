# Master Technical Revision & Exam Cheat Sheet: Full AI Curriculum

A rigorous, consolidated technical compendium covering mathematical formulations, loss functions, statistical proofs, evaluation metric trade-offs, and production systems architecture across Machine Learning, Computer Vision, Transformers, Reinforcement Learning, and MLOps.

---

## 1. Core Loss Functions & Mathematical Formulations

### 1.1 Binary Cross-Entropy (BCE) Loss
Used for binary classification (e.g., Customer Churn):
$$\mathcal{L}_{\text{BCE}}(y, \hat{p}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
- **Gradient with respect to logit $z$ (where $\hat{p} = \sigma(z)$)**:
  $$\frac{\partial \mathcal{L}}{\partial z} = \hat{p} - y$$
  *Significance:* The derivative is linear in the residual error, preventing vanishing gradients when $\hat{p}$ is saturated.

### 1.2 Categorical Cross-Entropy Loss
Used for multi-class classification (e.g., CIFAR-10 10-class image classification):
$$\mathcal{L}_{\text{CCE}}(y, \hat{p}) = -\sum_{c=1}^C y_c \ln(\hat{p}_c) \quad \text{where} \quad \hat{p}_c = \frac{\exp(z_c)}{\sum_{j=1}^C \exp(z_j)}$$
- **Softmax Temperature Scaling**:
  $$\hat{p}_c(T) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$$
  - As $T \to 0$: distribution collapses to $\arg\max$ one-hot argmax (hard decision).
  - As $T \to \infty$: distribution approaches uniform entropy $\frac{1}{C}$ (maximum exploration).

### 1.3 Smooth L1 / Huber Loss
Used for value estimation in DQN to prevent gradient explosion from outliers:
$$\mathcal{L}_{\delta}(a) = \begin{cases} 
\frac{1}{2} a^2 & \text{for } |a| \le \delta \\
\delta \left( |a| - \frac{1}{2} \delta \right) & \text{otherwise}
\end{cases}$$
- For small errors ($|a| \le \delta$): behaves like MSE ($L_2$), providing smooth convergence around the local optimum.
- For large errors ($|a| > \delta$): behaves like MAE ($L_1$), clamping gradient magnitude to $\pm \delta$ to ensure numerical stability.

### 1.4 Mean Squared Bellman Error (MSBE)
Used in Deep Q-Networks (DQN) for policy evaluation:
$$\mathcal{L}_{\text{DQN}}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
where $\theta$ denotes active policy parameters and $\theta^-$ denotes frozen target network parameters.

### 1.5 Maximum Entropy Objective (Soft Actor-Critic)
Augments standard cumulative reward with policy Shannon entropy $\mathcal{H}(\pi)$:
$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
where $\mathcal{H}(\pi(\cdot | s_t)) = -\mathbb{E}_{a \sim \pi} [\ln \pi(a | s_t)]$ and $\alpha$ is the temperature parameter.

---

## 2. Evaluation Metrics, Proofs & Trade-Offs

### 2.1 Confusion Matrix & Derived Metrics
Given True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), and False Negatives ($FN$):

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}$$

$$\text{F}_\beta\text{-Score} = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$
- $\beta = 1.0$: Harmonic mean (balanced).
- $\beta = 2.0$: Emphasizes **Recall** twice as much as Precision (critical in churn/medical diagnosis).
- $\beta = 0.5$: Emphasizes **Precision** (spam filtering).

### 2.2 ROC-AUC vs. PR-AUC Under Severe Class Imbalance
| Metric | X-Axis | Y-Axis | Behavior Under Heavy Imbalance (e.g. 98% Negative, 2% Positive) |
|---|---|---|---|
| **ROC Curve** | False Positive Rate ($\frac{FP}{FP+TN}$) | True Positive Rate ($\frac{TP}{TP+FN}$) | **Optimistically Biased:** Large $TN$ denominator artificially suppresses FPR, making poor models appear to have high AUC (~0.90+). |
| **PR Curve** | Recall ($\frac{TP}{TP+FN}$) | Precision ($\frac{TP}{TP+FP}$) | **Honest & Informative:** Completely ignores $TN$. Tracks positive class fidelity directly. Uninformative baseline is horizontal line at prevalence $\frac{P}{P+N}$. |

### 2.3 Cost-Benefit Bayes Decision Threshold Optimization ($\tau^*$)
Given utility matrix $U = [B_{TP}, -C_{FP}, -C_{FN}, B_{TN}]$ and calibrated predicted churn probability $p(x)$:
The expected value of intervening is:
$$\mathbb{E}[V | \text{Intervene}] = p(x) B_{TP} + (1 - p(x)) (-C_{FP})$$
$$\mathbb{E}[V | \text{Do Nothing}] = p(x) (-C_{FN}) + (1 - p(x)) B_{TN}$$

Setting $\mathbb{E}[V | \text{Intervene}] \ge \mathbb{E}[V | \text{Do Nothing}]$ yields the optimal decision threshold $\tau^*$:
$$\tau^* = \frac{C_{FP} + B_{TN}}{(B_{TP} + C_{FN}) + (C_{FP} + B_{TN})}$$
*Key Insight:* When $C_{FN} \gg C_{FP}$ (cost of losing a high-value customer dwarfs the cost of an email discount), $\tau^*$ drops significantly below 0.50 (e.g., $\tau^* \approx 0.15 - 0.25$).

---

## 3. Deep Learning & Computer Vision Foundations

### 3.1 Convolutional Layer Spatial Output Dimension
For an input of spatial size $W$, kernel size $K$, padding $P$, and stride $S$:
$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

### 3.2 Parameter Count in Convolutional Layers
For $C_{\text{in}}$ input channels, $C_{\text{out}}$ output filters, kernel spatial dimensions $K_h \times K_w$:
$$\text{Parameters} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
*(+1 represents the learnable bias per filter)*.

### 3.3 Batch Normalization (BN) vs. Layer Normalization (LN)
- **Batch Normalization**: Computes mean $\mu_B$ and variance $\sigma_B^2$ across the mini-batch dimension $N$ for each channel $C$ independently. Dependent on batch size; problematic in distributed micro-batches and autoregressive sequence models.
- **Layer Normalization**: Computes mean $\mu_L$ and variance $\sigma_L^2$ across all channels/hidden features $(C, H, W)$ for each sample $i$ independently. Independent of batch size; ideal for Transformers and RNNs.

### 3.4 Grad-CAM (Gradient-Weighted Class Activation Mapping)
Identifies spatial importance map for class $c$ in convolutional feature map $A^k$:
1. **Compute Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
2. **Weighted Linear Combination & ReLU Saliency**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   *Why ReLU?* Only features that have a positive influence on the target class logit $y^c$ should be visualized; negative activations contribute to other competing categories.

---

## 4. Transformers & Contextual Natural Language Processing

### 4.1 Scaled Dot-Product Self-Attention
Given query $Q$, key $K$, and value $V$ matrices of dimension $d_k$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
- **Why divide by $\sqrt{d_k}$?**
  For large $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients ($\frac{\partial \text{softmax}}{\partial z} \approx 0$). The scaling factor maintains variance at 1.0.

### 4.2 Multi-Head Attention (MHA)
$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
Allows the model to jointly attend to information from different representation subspaces at different positions.

### 4.3 DistilBERT Knowledge Distillation
DistilBERT compresses BERT-base by 40% while retaining 97% of language understanding:
$$\mathcal{L}_{\text{total}} = \alpha_{\text{KD}} \cdot T^2 \cdot D_{\text{KL}}\left( p_{\text{student}}^{\tau} \parallel p_{\text{teacher}}^{\tau} \right) + \alpha_{\text{CE}} \cdot \mathcal{L}_{\text{BCE}} + \alpha_{\cos} \cdot (1 - \cos(h_s, h_t))$$
- Reduces transformer layers from 12 to 6 (initialized by taking 1 layer out of every 2 from the teacher).
- Eliminates token-type embeddings and the pooler layer.

---

## 5. Reinforcement Learning Master Blueprint

### 5.1 The Deadly Triad in RL
Catastrophic divergence occurs when combining:
1. **Function Approximation** (Neural networks instead of tabular lookups)
2. **Bootstrapping** (Updating estimates from other estimates: $r + \gamma \max Q$)
3. **Off-Policy Learning** (Training on experiences collected from an exploratory behavior policy $\beta \ne \pi$)

### 5.2 DQN vs. Double DQN vs. Dueling D3QN

```
      Standard DQN              Double DQN (DDQN)             Dueling DQN
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────────┐
│ Target:               │   │ Target:               │   │ Q(s, a) = V(s) +          │
│ r + γ max Q(s', a'; θ⁻)│  │ r + γ Q(s', a*; θ⁻)   │   │  (A(s, a) - 1/|A| Σ A)    │
│                       │   │ where:                │   │                           │
│ (Suffers from positive│   │ a* = argmax Q(s', a; θ)│  │ Separates State-Value     │
│ overestimation bias)  │   │ (Decouples selection  │   │ from Action-Advantage     │
│                       │   │  from evaluation)     │   │ identifiability           │
└───────────────────────┘   └───────────────────────┘   └───────────────────────────┘
```

### 5.3 Prioritized Experience Replay (PER)
Samples transitions according to Temporal Difference (TD) error magnitude $|\delta_i|$:
$$p_i = |\delta_i| + \epsilon, \quad P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$
- **Importance Sampling (IS) Weight Correction** to counter non-uniform sampling bias:
  $$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta \Big/ \max_k w_k$$
  where $\beta$ is annealed linearly from $\beta_0 \to 1.0$ across training.

### 5.4 Continuous Control: Soft Actor-Critic (SAC)
1. **Twin Critics**: $y = r + \gamma(1 - d) (\min(Q_{t1}(s', \tilde{a}'), Q_{t2}(s', \tilde{a}')) - \alpha \ln \pi(\tilde{a}' | s'))$
2. **Reparameterization Trick**: $\tilde{a} = a_{\text{scale}} \tanh(\mu(s) + \sigma(s) \odot \epsilon)$
3. **Jacobian Log-Prob Correction**:
   $$\ln \pi(\tilde{a} | s) = \ln \mu(u | s) - \sum_{i=1}^D \ln\left( a_{\text{scale}} (1 - \tanh^2(u_i)) + \epsilon \right)$$
4. **Auto-Tuning Temperature ($\alpha$)**: $\mathcal{L}(\alpha) = -\alpha (\ln \pi(a | s) + \bar{\mathcal{H}})$, with $\bar{\mathcal{H}} = -\text{dim}(\mathcal{A})$.

---

## 6. Production MLOps, Data Drift & Inference Optimization

### 6.1 Statistical Covariate Shift Detection
1. **Two-Sample Kolmogorov-Smirnov (K-S) Test**:
   $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$
   Rejects null hypothesis of distribution equivalence when $p < 0.05$.

2. **Population Stability Index (PSI)**:
   $$\text{PSI} = \sum_{i=1}^k (P_i - Q_i) \ln\left( \frac{P_i}{Q_i} \right)$$
   - $\text{PSI} < 0.10$: Stable baseline.
   - $0.10 \le \text{PSI} < 0.20$: Moderate drift; warning flagged.
   - $\text{PSI} \ge 0.20$: Severe covariate shift; triggers automated model retraining.

### 6.2 Inference Acceleration via Dynamic INT8 Quantization
Linear mapping from 32-bit floating point $x \in [\alpha, \beta]$ to 8-bit integer $q \in [-128, 127]$:
$$q = \text{round}\left( \frac{x}{S} \right) + Z, \quad S = \frac{\beta - \alpha}{255}, \quad Z = \text{round}\left( -\frac{\alpha}{S} \right) - 128$$
- **Latency & Footprint Gains**: ~75% disk and RAM compression, $2\times - 4\times$ CPU inference speedup via SIMD vectorization (AVX-512 / VNNI).
