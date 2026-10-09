"""Compliance and functional tests for KineticSumo PettingZoo ParallelEnv."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from src.env.arena_env import KineticSumoEnv
from src.maps import MAP_REGISTRY


def test_pettingzoo_parallel_api_compliance():
    """Verify that KineticSumoEnv fully complies with PettingZoo ParallelEnv API."""
    env = KineticSumoEnv(map_name="Open")
    parallel_api_test(env, num_cycles=200)
    env.close()


@pytest.mark.parametrize("map_name", list(MAP_REGISTRY.keys()))
def test_all_map_presets_step_and_reset(map_name):
    """Ensure all geometric arena presets (Open, Sanctuary, Pinball, Atolls) work."""
    env = KineticSumoEnv(map_name=map_name)
    obs, infos = env.reset(seed=42)

    assert "agent_0" in obs
    assert "agent_1" in obs
    assert obs["agent_0"].shape == (46,)
    assert obs["agent_1"].shape == (46,)
    assert obs["agent_0"].dtype == np.float32

    # Step with dummy actions
    actions = {
        "agent_0": np.array([0.5, 0.2, 0.0], dtype=np.float32),
        "agent_1": np.array([0.8, -0.3, 0.6], dtype=np.float32),
    }
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert "agent_0" in next_obs
    assert "agent_1" in next_obs
    assert "agent_0" in rewards
    assert "agent_1" in rewards
    assert isinstance(rewards["agent_0"], float)
    env.close()


def test_reward_terminal_signals():
    """Verify win/loss terminal rewards."""
    env = KineticSumoEnv(map_name="Open")
    env.reset(seed=123)

    # Force agent_0 out of bounds by placing it beyond radius
    env.agent_objs["agent_0"].pos = np.array([12.0, 0.0], dtype=np.float32)

    actions = {
        "agent_0": np.zeros(3, dtype=np.float32),
        "agent_1": np.zeros(3, dtype=np.float32),
    }
    obs, rewards, terms, truncs, infos = env.step(actions)

    assert terms["agent_0"] is True
    assert terms["agent_1"] is True
    assert infos["agent_0"]["winner"] == "agent_1"
    # Agent 1 (winner) should receive positive reward, agent 0 (loser) negative
    assert rewards["agent_1"] > rewards["agent_0"]
    env.close()


def test_draw_timeout_penalty():
    """Verify timeout draw assigns -1.0 cold-war penalty."""
    # Set short max_duration
    env = KineticSumoEnv(map_name="Open", max_duration=0.1, dt=0.05)
    env.reset()

    actions = {
        "agent_0": np.zeros(3, dtype=np.float32),
        "agent_1": np.zeros(3, dtype=np.float32),
    }
    # Step 1
    env.step(actions)
    # Step 2 -> reaches 0.1s -> timeout
    obs, rews, terms, truncs, infos = env.step(actions)

    assert truncs["agent_0"] is True
    assert infos["agent_0"]["is_draw"] is True
    # Draw penalty is -1.0 terminal
    assert rews["agent_0"] < -0.5
    assert rews["agent_1"] < -0.5
    env.close()
