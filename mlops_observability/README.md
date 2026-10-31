# Production MLOps Data Drift & Observability Engine

Statistical monitoring and reliability guardrails for post-deployment ML reliability.

---

## Statistical Methodologies

1. **Kolmogorov-Smirnov (K-S) Two-Sample Test**:
   Non-parametric test comparing the cumulative distribution functions (CDFs) of reference $F_{\text{ref}}(x)$ and production $F_{\text{cur}}(x)$ data:
   $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$
   Rejects the null hypothesis $H_0$ (identical distributions) when $p\text{-value} < \alpha = 0.05$.

2. **Population Stability Index (PSI)**:
   Measures divergence in binned probability masses between training baseline and live inference:
   $$\text{PSI} = \sum_{i=1}^k \left( P_i - Q_i \right) \ln\left( \frac{P_i}{Q_i} \right)$$
   - $\text{PSI} < 0.10$: Distribution is stable; no action required.
   - $0.10 \le \text{PSI} < 0.20$: Moderate drift; warning flag raised.
   - $\text{PSI} \ge 0.20$: Severe covariate shift; triggers automated model retraining.

3. **1D Wasserstein Distance (Earth Mover's Distance)**:
   Quantifies the minimum transport cost between normalized empirical distributions:
   $$W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx$$

---

## Automated Retraining Trigger Rule

A global dataset drift event is declared if:
$$\frac{\sum \mathbb{I}(\text{Feature Drift Detected})}{\text{Total Features}} \ge 0.33$$
Upon breach, an incident report is dispatched with retrain recommendation flags.
