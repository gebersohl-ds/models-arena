---
name: perf-profiler
description: Profiles Python and RL physics environments using cProfile or py-spy, filtering execution bottlenecks taking >5% total time to identify raycasting, collision, or loop performance degradations.
---

# Performance Profiler & Benchmark Skill (`perf-profiler`)

This skill enables Antigravity to run automated profiling on Python physics environments, simulation loops, and reinforcement learning environments, isolating functions that consume $>5\%$ of total execution time.

---

## 1. When to Use This Skill

Activate this skill when:
1. **Physics / Stepping Throughput Drops:** Stepping rate drops below target (e.g. from 50k steps/sec to <2k steps/sec).
2. **Benchmarking New Environments:** Verifying that a new gym environment or simulation step function is vector-accelerated.
3. **Investigating CPU Spikes:** Diagnosing nested `for`-loops in collision detection, line-of-sight raycasting, or reward computation.

---

## 2. Running the Profiler

### Automated Filter Runner (Recommended)
Run the bundled profiler script, pointing it to your benchmark script:

```bash
# Filter functions consuming >= 5% total execution time
python ~/.gemini/config/skills/perf-profiler/scripts/profile_runner.py scripts/benchmark_env.py

# Custom threshold (e.g., 2% threshold)
python ~/.gemini/config/skills/perf-profiler/scripts/profile_runner.py scripts/benchmark_env.py --threshold 0.02
```

On Windows PowerShell:
```powershell
python "C:\Users\xSyl0\.gemini\config\skills\perf-profiler\scripts\profile_runner.py" scripts/benchmark_env.py
```

### Direct CLI cProfile Execution
If running natively via shell:

```bash
python -m cProfile -s tottime scripts/benchmark_env.py | head -n 30
```

### Py-Spy Sampling (Zero Overhead)
For live running simulations or stuck processes without instrumenting code:
```bash
py-spy top -- python scripts/benchmark_env.py
# Or generate a flamegraph:
py-spy record -o profile_flamegraph.svg -- python scripts/benchmark_env.py
```

---

## 3. Interpreting Output & Remediation

When a function takes $>5\%$ `tottime`, consult the [Optimization Guide](./references/optimization_guide.md):

* **Collision detection functions consuming high %:** Replace $O(N^2)$ checks with a spatial grid or bounding-volume hierarchy (BVH).
* **`np.array` / memory allocations in `step()`:** Pre-allocate observation and buffer arrays during initialization.
* **Raycasting:** Vectorize line-intersection math with NumPy or compiled C/Cython kernels.
