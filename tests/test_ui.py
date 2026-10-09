"""Test headless rendering and UI frame generation."""

import numpy as np
from src.env.arena_env import KineticSumoEnv
from src.ui.renderer import SumoRenderer
from src.maps import get_map


def test_headless_renderer_rgb_array():
    env = KineticSumoEnv(map_name="Open", render_mode="rgb_array")
    obs, _ = env.reset()

    # Step and render
    actions = {
        "agent_0": np.array([1.0, 0.2, 0.8], dtype=np.float32),  # Dash
        "agent_1": np.array([0.5, -0.4, -0.8], dtype=np.float32),  # Anchor
    }
    env.step(actions)
    frame = env.render()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (900, 900, 3)
    assert frame.dtype == np.uint8

    env.close()
