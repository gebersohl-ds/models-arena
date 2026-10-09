"""
Drop-in snippet for recording headless Pygame / Gymnasium simulation loops.
"""

import os
# Set headless dummy video driver before pygame is initialized
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import numpy as np

# 1. Initialize Pygame & create a display/virtual surface
pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))

# 2. Recording buffer
frames = []
fps = 30
max_seconds = 10.0
max_frames = int(fps * max_seconds)

# 3. Game/Simulation Loop
clock = pygame.time.Clock()
running = True

while running and len(frames) < max_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform updates & rendering
    screen.fill((15, 18, 25))
    # ... draw entities ...

    # Capture frame: surfarray returns (W, H, 3) -> transpose to (H, W, 3)
    frame = pygame.surfarray.array3d(screen)
    frames.append(np.transpose(frame, (1, 0, 2)))

    clock.tick(fps)

pygame.quit()

# 4. Save to optimized GIF or MP4
import imageio
imageio.mimsave("artifacts/headless_run.gif", frames, duration=1.0/fps, loop=0)
print(f"Recorded {len(frames)} frames to artifacts/headless_run.gif")
