---
name: onnx-validator
description: Automatically exports PyTorch policy checkpoints (.pt/.pth) to ONNX models, benchmarks latency against NumPy/PyTorch, and validates numeric parity within atol=10^-5 tolerance without PyTorch runtime dependencies.
---

# ONNX Runtime Model Exporter & Validator Skill (`onnx-validator`)

This skill converts PyTorch reinforcement learning policy checkpoints into production-ready ONNX models, benchmarks stepping latency, and verifies mathematical equivalence within a strict numeric tolerance ($\text{atol} = 10^{-5}$).

---

## 1. When to Use This Skill

Activate this skill when:
1. **Preparing Models for Production / Deployment:** Deploying an RL policy into a game engine (Godot, Unity), C++ runtime, or mobile client where PyTorch cannot be installed.
2. **Checking Graph Correctness:** Ensuring tensor operations, custom activations, or layer dimensions were preserved identically across model formats.
3. **Latency Benchmarking:** Comparing CPU inference latency between PyTorch and ONNX Runtime.

---

## 2. Validation Workflow

### Step 1: Export & Verify Parity
Run the validation script on the target model or checkpoint:
```bash
python ~/.gemini/config/skills/onnx-validator/scripts/export_and_validate.py --onnx-out models/policy.onnx --atol 1e-5
```

### Step 2: In-Code Integration
```python
from scripts.export_and_validate import export_and_validate
export_and_validate(policy_model, dummy_obs, "models/champion_policy.onnx", atol=1e-5)
```

The tool asserts:
$$\max |Y_{\text{torch}} - Y_{\text{onnx}}| \le 10^{-5}$$
And reports benchmark speedup over 500 iterations.

---

## 3. Reference Files & Guides

* [Export & Validation Tool](./scripts/export_and_validate.py)
* [Sample PPO Policy Export Script](./examples/sample_policy_export.py)
* [Deployment Best Practices](./references/onnx_best_practices.md)
