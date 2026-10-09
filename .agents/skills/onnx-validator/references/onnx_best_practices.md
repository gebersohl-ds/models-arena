# ONNX Export & Deployment Best Practices for RL Agents

Exporting PyTorch models to ONNX enables zero-dependency deployment in mobile environments (Android / iOS), Godot / Unity games via C# or GDExtension, and high-performance C++ backends.

---

## 1. Golden Rules for RL Policy Export

1. **Strict Numerical Equivalence:** Always test exported graphs against PyTorch output with:
   $$\text{atol} \le 10^{-5}$$
2. **Fixed vs Dynamic Batch Axes:** Explicitly configure dynamic axes:
   ```python
   dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}}
   ```
   This allows the single exported model to run single-step evaluation (`batch_size=1`) and batch vector evaluation (`batch_size=64`).
3. **Opset Version:** Use opset version 17 or higher for compatibility with modern ONNX Runtime builds.
4. **Deterministic Evaluation Mode:** Always call `model.eval()` before export to freeze Dropout, BatchNorm, and stochastic heads.

---

## 2. In-Game Runtime Inference (Zero PyTorch Dependency)

In Python, the policy can be evaluated with pure NumPy + ONNX Runtime:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/policy.onnx")
action = session.run(None, {"observation": obs_np})[0]
```
No PyTorch install or GPU drivers are needed in production!
