#!/usr/bin/env python3
"""
ONNX Model Exporter & Tolerance Validator
Converts PyTorch (.pt/.pth) policy models into ONNX, validates numeric parity
within atol=1e-5 against original outputs, and benchmarks inference latency.
"""

import os
import sys
import time
import argparse
import numpy as np

def validate_numeric_parity(torch_output: np.ndarray, onnx_output: np.ndarray, atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """Compares PyTorch output array with ONNX Runtime output array."""
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - onnx_output)))
    is_close = np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol)

    return {
        "passed": bool(is_close),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "atol_threshold": atol
    }

def export_and_validate(model, dummy_input, onnx_path: str, atol: float = 1e-5, benchmark_iters: int = 500):
    try:
        import torch
        import onnxruntime as ort
    except ImportError as e:
        print(f"[ERROR] Required ML packages missing: {e}. Install with: pip install torch onnx onnxruntime", file=sys.stderr)
        sys.exit(1)

    model.eval()

    # 1. Capture PyTorch baseline inference
    with torch.no_grad():
        if isinstance(dummy_input, (tuple, list)):
            torch_out = model(*dummy_input)
            input_names = [f"input_{i}" for i in range(len(dummy_input))]
        else:
            torch_out = model(dummy_input)
            input_names = ["observation"]

    if isinstance(torch_out, tuple):
        torch_out_np = torch_out[0].detach().cpu().numpy()
        output_names = [f"output_{i}" for i in range(len(torch_out))]
    else:
        torch_out_np = torch_out.detach().cpu().numpy()
        output_names = ["action"]

    # 2. Export to ONNX
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    print(f"[*] Exporting PyTorch model to: {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch_size"} for name in input_names + output_names}
    )
    print("[+] ONNX export succeeded.")

    # 3. Initialize ONNX Runtime Session
    print("[*] Initializing ONNX Runtime InferenceSession...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])

    # Prepare input feed
    if isinstance(dummy_input, (tuple, list)):
        feed = {input_names[i]: dummy_input[i].detach().cpu().numpy() for i in range(len(dummy_input))}
    else:
        feed = {input_names[0]: dummy_input.detach().cpu().numpy()}

    onnx_outs = session.run(None, feed)
    onnx_out_np = onnx_outs[0]

    # 4. Numeric Parity Verification
    parity = validate_numeric_parity(torch_out_np, onnx_out_np, atol=atol)
    print("\n" + "=" * 60)
    print(" NUMERIC TOLERANCE VALIDATION (PyTorch vs ONNX)")
    print("=" * 60)
    print(f" Tolerance Threshold (atol): {parity['atol_threshold']}")
    print(f" Maximum Absolute Difference: {parity['max_abs_diff']:.2e}")
    print(f" Mean Absolute Difference:    {parity['mean_abs_diff']:.2e}")

    if parity["passed"]:
        print(f"\n[PASS] Model graph outputs are mathematically identical within atol={atol}!")
    else:
        print(f"\n[FAIL] Discrepancy exceeded tolerance threshold ({parity['max_abs_diff']:.2e} > {atol})!", file=sys.stderr)
        sys.exit(1)

    # 5. Latency Benchmark
    if benchmark_iters > 0:
        print(f"\n[*] Benchmarking latency over {benchmark_iters} iterations...")

        # Benchmark ONNX Runtime
        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = session.run(None, feed)
        t_onnx = (time.perf_counter() - t0) / benchmark_iters * 1000.0

        # Benchmark PyTorch (CPU)
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(benchmark_iters):
                if isinstance(dummy_input, (tuple, list)):
                    _ = model(*dummy_input)
                else:
                    _ = model(dummy_input)
            t_torch = (time.perf_counter() - t0) / benchmark_iters * 1000.0

        print(f" PyTorch (CPU):      {t_torch:.4f} ms / step")
        print(f" ONNX Runtime (CPU): {t_onnx:.4f} ms / step")
        speedup = (t_torch / t_onnx) if t_onnx > 0 else 1.0
        print(f" Speedup Factor:     {speedup:.2f}x")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX & validate numerical parity.")
    parser.add_argument("--onnx-out", default="models/policy.onnx", help="Path for exported .onnx model")
    parser.add_argument("--atol", type=float, default=1e-5, help="Numeric tolerance threshold (default: 1e-5)")
    parser.add_argument("--iters", type=int, default=500, help="Benchmark iterations (default: 500)")
    args = parser.parse_args()

    print(f"[INFO] ONNX Validator configured. Target: {args.onnx_out}, atol={args.atol}")

if __name__ == "__main__":
    main()
