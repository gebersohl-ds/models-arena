"""Unit tests for Next-Level features: Tactical Bots, 4-Way FFA Env, Deterministic Maps, and Audio."""

import numpy as np
import pytest

from src.agents.tactical import (
    BulldozerAgent,
    ShinobiAgent,
    GuardianAgent,
    TunableAgent,
    BotConfig,
)
from src.maps import get_map, MAP_REGISTRY
from src.env.arena_4way_env import KineticSumo4WayEnv
from src.ui.audio import SoundEngine


def test_tactical_bots_action_space():
    """Verify all tactical agents produce valid actions in [-1, 1]^3."""
    dummy_obs = np.zeros(46, dtype=np.float32)
    # Set mock heading to positive X
    dummy_obs[4] = 1.0  # cos_th
    # Opponent at (2.0, 0.0)
    dummy_obs[7] = 0.1  # dx
    dummy_obs[12] = 1.0  # energy

    bots = [
        BulldozerAgent(),
        ShinobiAgent(),
        GuardianAgent(),
        TunableAgent(BotConfig(aggression=0.9, dash_frequency=0.8)),
    ]

    for bot in bots:
        bot.reset()
        action = bot.act(dummy_obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_new_deterministic_maps():
    """Verify Colosseum and Crossfire arena presets."""
    assert "Colosseum" in MAP_REGISTRY
    assert "Crossfire" in MAP_REGISTRY

    colosseum = get_map("Colosseum")
    assert len(colosseum.obstacles) == 4
    # Check 4-way spawn positions
    spawns_4w = colosseum.get_4way_spawn_transforms()
    assert len(spawns_4w) == 4
    for pos, angle in spawns_4w:
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)

    crossfire = get_map("Crossfire")
    assert len(crossfire.obstacles) == 6
    for obs in crossfire.obstacles:
        assert obs.is_bumper is True
        assert obs.radius > 0.0


def test_4way_environment_lifecycle():
    """Test full reset and step lifecycle for KineticSumo4WayEnv."""
    env = KineticSumo4WayEnv(map_name="Colosseum", max_duration=10.0)
    obs, infos = env.reset(seed=42)

    assert len(obs) == 4
    for a_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
        assert a_id in obs
        assert obs[a_id].shape == (46,)
        assert obs[a_id].dtype == np.float32

    # Step with dummy forward actions
    actions = {a_id: np.array([1.0, 0.0, 0.0], dtype=np.float32) for a_id in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)

    assert len(next_obs) == 4
    assert len(rewards) == 4
    assert len(terms) == 4
    assert len(truncs) == 4

    env.close()


def test_procedural_sound_engine():
    """Verify sound engine initializes, handles waveforms, and toggles mute."""
    audio = SoundEngine(enabled=False)  # Test safe headless mode
    assert audio.is_muted is True
    audio.toggle_mute()
    assert audio.is_muted is False

    # Calling play methods must not throw
    audio.play_clang(is_flank=False, intensity=0.5)
    audio.play_clang(is_flank=True, intensity=1.0)
    audio.play_dash()
    audio.play_anchor()
    audio.play_ringout()
    audio.play_countdown()
