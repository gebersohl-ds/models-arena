# Physics & Loop Optimization Guide

When developing RL environments, physics simulations, or raycasters in Python, performance bottlenecks frequently stem from common anti-patterns that collapse stepping throughput from >50,000 steps/sec down to <1,500 steps/sec.

---

## 1. Top Culprits for >5% Bottlenecks

### A. Nested Python Loops in Collision Detection
* **Anti-pattern:** $O(N \times M)$ pairwise checks in raw Python:
  ```python
  for agent in agents:
      for bullet in bullets:
          if math.hypot(agent.x - bullet.x, agent.y - bullet.y) < radius:
              ...
  ```
* **Solution:** 
  1. **Spatial Hashing / Uniform Grid:** Hash entity positions into a 2D grid `grid[(cell_x, cell_y)]`. Only check entities sharing adjacent grid buckets ($O(N)$ expected).
  2. **Vectorized NumPy Distance:**
     ```python
     diff = agent_pos[:, None, :] - bullet_pos[None, :, :]
     dist_sq = np.sum(diff ** 2, axis=-1)
     hits = np.where(dist_sq < radius_sq)
     ```

### B. Raycasting in Pure Python
* **Anti-pattern:** Ray march step-by-step with `while d < max_dist:` in Python.
* **Solution:** Pre-calculate line-segment intersections using cross-products in NumPy or batch raycast with vectorized geometry.

### C. Repeated NumPy Array Allocations in Step Loops
* **Anti-pattern:** Allocating new arrays every step:
  ```python
  def step(self, action):
      obs = np.zeros((10, 10))  # Memory allocation overhead in every step
  ```
* **Solution:** Pre-allocate static buffers during `__init__` and overwrite in-place:
  ```python
  self.obs_buffer[...] = 0.0
  ```

### D. Accidental Object Conversions
* Converting Python lists to NumPy arrays repeatedly (`np.array(my_list)`) inside the critical loop.
* Converting NumPy scalars to Python `float(x)` or `int(x)` repeatedly.
