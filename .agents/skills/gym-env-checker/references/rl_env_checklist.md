# RL Environment Compliance & Silent Bug Checklist

Reinforcement learning algorithms frequently fail silently when environments deviate from standard Gym/PettingZoo API contracts.

---

## 1. Top Silent RL Bugs

### A. Non-Deterministic Resets
* **Symptom:** Two calls to `env.reset(seed=42)` produce different initial states.
* **Impact:** Random seeds fail to reproduce bugs or training curves.
* **Fix:** Seed all internal RNGs (e.g. `self.np_random, _ = seeding.np_random(seed)`).

### B. Confusing Truncation vs. Termination
* **`terminated`:** The episode ended due to natural task completion or failure (absorbing state). Value of next state $V(s') = 0$.
* **`truncated`:** The episode ended due to time limits (max step horizon). Value of next state $V(s') \ne 0$ (must bootstrap!).
* **Impact:** Treating truncation as termination causes severe value estimation bias near the time horizon.

### C. Observation Space Bound Leaks
* **Symptom:** Observation arrays contain values outside `[low, high]` or have wrong dtype (`float64` instead of `float32`).
* **Fix:** Use `np.clip(obs, self.observation_space.low, self.observation_space.high, dtype=np.float32)`.

### D. Multi-Agent Agent List Desync (PettingZoo)
* **Symptom:** Dead/terminated agents remain in `env.agents`, causing action key errors on subsequent steps.
* **Fix:** Remove terminated agents from `self.agents` upon episode end.
