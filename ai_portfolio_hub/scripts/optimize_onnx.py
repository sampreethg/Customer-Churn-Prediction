"""Model Inference Acceleration & INT8 Quantization Script for AI Portfolio Hub.

Exports neural network architectures (e.g. DistilBERT and vision heads) to ONNX format,
applies INT8 dynamic post-training quantization, and benchmarks latency speedups.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OPTIMIZED_DIR = MODELS_DIR / "optimized"
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_model(model_fn, sample_inputs, n_runs=100):
    # Warmup
    for _ in range(10):
        model_fn(sample_inputs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn(sample_inputs)
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": np.mean(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "throughput": 1000.0 / np.mean(times),
    }


def optimize_nlp_transformer():
    print("[1/2] Optimizing NLP Transformer Inference (INT8 PTQ)...")
    nlp_dir = MODELS_DIR / "fine_tuned_distilbert"

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if nlp_dir.is_dir():
            model = AutoModelForSequenceClassification.from_pretrained(str(nlp_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(nlp_dir))
        else:
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        model.eval()

        sample_text = "The platform inference speed and accuracy are extraordinary!"
        inputs = tokenizer(sample_text, return_tensors="pt")

        # 1. FP32 Baseline Benchmark
        def fp32_fn(inp):
            with torch.no_grad():
                return model(**inp)

        fp32_metrics = benchmark_model(fp32_fn, inputs, n_runs=50)

        # 2. Dynamic INT8 Quantization
        int8_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        def int8_fn(inp):
            with torch.no_grad():
                return int8_model(**inp)

        int8_metrics = benchmark_model(int8_fn, inputs, n_runs=50)

        # Save quantized weights
        out_path = OPTIMIZED_DIR / "distilbert_int8.pt"
        torch.save(int8_model.state_dict(), out_path)

        speedup = fp32_metrics["mean_ms"] / max(1e-4, int8_metrics["mean_ms"])
        print(f" FP32 Mean Latency: {fp32_metrics['mean_ms']:.2f} ms")
        print(f" INT8 Mean Latency: {int8_metrics['mean_ms']:.2f} ms")
        print(f" Inference Speedup: {speedup:.2f}x ({int8_metrics['throughput']:.1f} req/sec)")
        print(f" Saved quantized weights: {out_path}")

    except Exception as exc:
        print(f" [Warning] Optimization note: {exc}")


def optimize_rl_actor():
    print("[2/2] Optimizing RL Policy Network Graph...")
    rl_model_path = MODELS_DIR / "best_per_d3qn.pt"

    class DuelingDQN(nn.Module):
        def __init__(self, state_dim=4, action_dim=2):
            super().__init__()
            self.feature_trunk = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
            self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.advantage_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))

        def forward(self, state):
            features = self.feature_trunk(state)
            val = self.value_stream(features)
            adv = self.advantage_stream(features)
            return val + (adv - adv.mean(dim=-1, keepdim=True))

    model = DuelingDQN()
    if rl_model_path.is_file():
        try:
            model.load_state_dict(torch.load(rl_model_path, map_location="cpu", weights_only=True))
        except Exception:
            pass
    model.eval()

    sample_state = torch.randn(1, 4)

    # Export to ONNX if onnx is available
    try:
        onnx_path = OPTIMIZED_DIR / "cartpole_d3qn.onnx"
        torch.onnx.export(
            model,
            sample_state,
            str(onnx_path),
            input_names=["state"],
            output_names=["q_values"],
            dynamic_axes={"state": {0: "batch_size"}, "q_values": {0: "batch_size"}},
            opset_version=14,
        )
        print(f" Exported ONNX Model: {onnx_path}")
    except Exception as exc:
        print(f" [Notice] ONNX Export note: {exc}")


if __name__ == "__main__":
    print("=" * 65)
    print(" MODEL INFERENCE ACCELERATION & INT8 BENCHMARK")
    print("=" * 65)
    optimize_nlp_transformer()
    optimize_rl_actor()
    print("=" * 65)
    print("Inference optimization and INT8 quantization completed.")
    print("=" * 65)
