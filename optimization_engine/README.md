# Latency & Inference Optimization Engine

Accelerates production model deployments via Dynamic Post-Training Quantization (PTQ) and percentile latency profiling.

---

## Technical Highlights

1. **Dynamic INT8 Quantization**:
   Quantizes weights from Float32 to Int8 while dynamically quantizing activations during inference:
   $$q = \text{round}\left( \frac{x}{S} \right) + Z$$
   where $S$ is the scale factor and $Z$ is the zero-point offset.

2. **Empirical Latency Profiling**:
   Benchmarks throughput (requests/sec) and latency distributions across P50 (median), P95, and P99 tail latencies to guarantee strict SLA compliance.
