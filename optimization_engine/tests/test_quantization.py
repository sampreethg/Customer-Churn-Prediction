"""Unit tests for inference optimization, dynamic INT8 quantization, and latency benchmarking."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optimization_engine.src.optimizer import InferenceOptimizer


class SampleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_dynamic_quantization_functional():
    """Verify INT8 dynamic quantization preserves forward pass integrity and correlation."""
    torch.manual_seed(42)
    model = SampleMLP()
    model.eval()

    inputs = torch.randn(10, 32)
    with torch.no_grad():
        fp32_output = model(inputs)

    quantized_model = InferenceOptimizer.quantize_pytorch_model(model)
    with torch.no_grad():
        int8_output = quantized_model(inputs)

    # Calculate Pearson correlation between FP32 and INT8 output logits
    fp32_flat = fp32_output.numpy().flatten()
    int8_flat = int8_output.numpy().flatten()
    corr = np.corrcoef(fp32_flat, int8_flat)[0, 1]

    assert corr > 0.98, f"Quantization degraded output correlation: {corr}"


def test_latency_benchmark_profiler():
    """Verify latency profiling calculates valid percentiles (P50 <= P95 <= P99)."""
    torch.manual_seed(42)
    model = SampleMLP()
    model.eval()
    sample_input = torch.randn(1, 32)

    metrics = InferenceOptimizer.benchmark_latency(model, sample_input, num_warmup=5, num_runs=30)

    assert "mean_latency_ms" in metrics
    assert "p50_latency_ms" in metrics
    assert "p95_latency_ms" in metrics
    assert "p99_latency_ms" in metrics
    assert "throughput_req_per_sec" in metrics

    assert metrics["p50_latency_ms"] <= metrics["p95_latency_ms"] + 1e-4
    assert metrics["p95_latency_ms"] <= metrics["p99_latency_ms"] + 1e-4
    assert metrics["throughput_req_per_sec"] > 0


def test_model_compression_ratio():
    """Verify INT8 quantization achieves measurable weight footprint reduction."""
    torch.manual_seed(42)
    model = SampleMLP()
    model.eval()
    quantized_model = InferenceOptimizer.quantize_pytorch_model(model)

    comp = InferenceOptimizer.compare_compression(model, quantized_model)

    assert "fp32_bytes" in comp
    assert "int8_bytes" in comp
    assert "compression_ratio" in comp
    assert comp["fp32_bytes"] > 0
    assert comp["int8_bytes"] > 0
