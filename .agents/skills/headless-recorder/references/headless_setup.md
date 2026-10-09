# Headless Pygame & RL Rendering Reference

Capturing visual simulations headlessly enables automated validation pipelines in headless CI/CD, remote servers, and background coding agent runs.

---

## 1. Headless Virtual Drivers

### Linux / Headless Servers
Set the dummy SDL video driver before calling `pygame.init()`:
```python
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
```
Or run with `xvfb-run`:
```bash
xvfb-run -a -s "-screen 0 1024x768x24" python scripts/record_env.py
```

### Windows Environments
On Windows, setting `os.environ["SDL_VIDEODRIVER"] = "dummy"` works out of the box for surface allocations.

---

## 2. Fast Frame Capture

Pygame stores surfaces column-major:
```python
raw = pygame.surfarray.array3d(surface) # shape: (width, height, 3)
frame = np.transpose(raw, (1, 0, 2))    # shape: (height, width, 3)
```

## 3. Embedding Recordings into README.md

In your markdown documentation or PR summary:
```markdown
### Visual Match Validation
![10-second Agent Run](./artifacts/match_recording.gif)
```
Or for MP4:
```markdown
<video src="./artifacts/match_recording.mp4" controls autoplay loop width="640"></video>
```
