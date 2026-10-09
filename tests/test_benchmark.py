"""Benchmark test verifying simulation throughput > 50,000 steps/sec per CPU core."""

import time
import numpy as np
from src.env.vector_env import VectorArenaEnv


def test_simulation_speed_exceeds_50k_steps_per_sec():
    num_envs = 128
    env = VectorArenaEnv(num_envs=num_envs, map_name="Open")

    # Warmup
    actions = np.zeros((num_envs, 2, 3), dtype=np.float32)
    actions[:, :, 0] = 0.8  # forward thrust
    for _ in range(50):
        env.step(actions)

    # Benchmark run
    num_steps = 1000
    t0 = time.perf_counter()
    for _ in range(num_steps):
        # Steer and thrust command
        env.step(actions)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_env_steps = num_steps * num_envs
    steps_per_sec = total_env_steps / elapsed

    print(f"\n[BENCHMARK] Stepped {total_env_steps} env steps in {elapsed:.3f}s: {steps_per_sec:,.0f} steps/sec")
    # Verify throughput meets the >50,000 steps/sec constraint
    assert steps_per_sec >= 50_000, f"Expected >= 50,000 steps/sec, got {steps_per_sec:.0f}"
