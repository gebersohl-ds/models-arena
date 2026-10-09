---
name: headless-recorder
description: Captures headless Pygame and RL simulation environments via pygame.surfarray and renders optimized .gif or .mp4 recordings using imageio or ffmpeg for visual behavior validation and README embedding.
---

# Headless GIF & Video Recorder Skill (`headless-recorder`)

This skill allows Antigravity to run headless Pygame and reinforcement learning simulations, capture frames in memory via `pygame.surfarray`, and render lightweight `.gif` or `.mp4` recordings for visual validation loops and documentation.

---

## 1. When to Use This Skill

Activate this skill when:
1. **RL Policy Visual Degeneracy:** Checking whether an agent's policy is degenerating (e.g. spinning in circles, jittering) despite high reward metrics.
2. **Visual Smoke Testing:** Validating physics engines, particle effects, or procedural generation without opening interactive desktop windows.
3. **Automated Documentation:** Recording a 10-second gameplay match and embedding the resulting GIF directly into the repo's `README.md` or PR description.

---

## 2. Standard Workflow

### Step 1: Headless Virtual Driver
Ensure the SDL dummy driver is configured:
```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
```

### Step 2: In-Memory Frame Extraction
Capture frames directly from the Pygame display surface:
```python
import pygame, numpy as np
raw = pygame.surfarray.array3d(screen)
frame = np.transpose(raw, (1, 0, 2))
frames.append(frame)
```

### Step 3: Export Optimized GIF / Video
Using `imageio`:
```python
import imageio
imageio.mimsave("artifacts/match_recording.gif", frames, duration=1.0/30, loop=0)
```

---

## 3. Reference Files & Snippets

* [Reusable Recorder Script](./scripts/record_env.py)
* [Drop-In Pygame Loop Snippet](./examples/recorder_snippet.py)
* [Driver & Display Reference](./references/headless_setup.md)
