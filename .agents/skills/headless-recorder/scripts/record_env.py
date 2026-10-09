#!/usr/bin/env python3
"""
Headless Pygame & RL Environment Frame Recorder
Captures frames from a Pygame display/surface headless using pygame.surfarray
and saves optimized GIF or MP4 recordings.
"""

import os
import sys
import argparse
import numpy as np

# Force dummy driver if headless requested before pygame init
def enable_headless_driver():
    if "SDL_VIDEODRIVER" not in os.environ and sys.platform != "win32":
        os.environ["SDL_VIDEODRIVER"] = "dummy"

class FrameRecorder:
    def __init__(self, fps: int = 30, max_frames: int = 300):
        self.fps = fps
        self.max_frames = max_frames
        self.frames = []

    def capture_surface(self, surface):
        """Captures a Pygame Surface using pygame.surfarray."""
        import pygame
        # pygame.surfarray returns (width, height, 3). Convert to standard (height, width, 3)
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        self.frames.append(frame)

    def capture_window(self):
        """Captures the active Pygame display surface."""
        import pygame
        screen = pygame.display.get_surface()
        if screen is not None:
            self.capture_surface(screen)

    def is_full(self) -> bool:
        return len(self.frames) >= self.max_frames

    def save(self, output_path: str, optimize: bool = True):
        """Saves captured frames as GIF or MP4."""
        if not self.frames:
            print("[WARN] No frames were captured to save.", file=sys.stderr)
            return

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()

        if ext == ".gif":
            self._save_gif(output_path, optimize=optimize)
        elif ext in (".mp4", ".mkv", ".webm"):
            self._save_video(output_path)
        else:
            raise ValueError(f"Unsupported format: {ext} (use .gif or .mp4)")

        print(f"[SUCCESS] Saved {len(self.frames)} frames ({len(self.frames)/self.fps:.1f}s) to: {output_path}")

    def _save_gif(self, output_path: str, optimize: bool = True):
        duration_per_frame = 1.0 / self.fps
        try:
            import imageio
            imageio.mimsave(output_path, self.frames, duration=duration_per_frame, loop=0)
        except ImportError:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in self.frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(duration_per_frame * 1000),
                loop=0,
                optimize=optimize
            )

    def _save_video(self, output_path: str):
        try:
            import imageio
            writer = imageio.get_writer(output_path, fps=self.fps, codec='libx264', quality=8)
            for f in self.frames:
                writer.append_data(f)
            writer.close()
        except Exception:
            # Fallback to ffmpeg CLI if available
            import subprocess
            import tempfile
            from PIL import Image

            with tempfile.TemporaryDirectory() as tmpdir:
                for idx, f in enumerate(self.frames):
                    Image.fromarray(f).save(os.path.join(tmpdir, f"frame_{idx:05d}.png"))
                cmd = [
                    "ffmpeg", "-y", "-framerate", str(self.fps),
                    "-i", os.path.join(tmpdir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser(description="Headless Frame Recorder Utility")
    parser.add_argument("--output", default="artifacts/match_recording.gif", help="Output path (.gif or .mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate (default 30)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Max recording duration in seconds (default 10.0)")
    args = parser.parse_args()

    max_frames = int(args.fps * args.seconds)
    print(f"FrameRecorder initialized: {args.fps} FPS, max {max_frames} frames ({args.seconds}s)")
    print(f"Target output: {args.output}")

if __name__ == "__main__":
    main()
