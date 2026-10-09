"""Tests for engine parity (VectorArenaEnv vs KineticSumoEnv), league tracking, and match rules."""

import numpy as np
import pytest
from src.env.arena_env import KineticSumoEnv
from src.env.vector_env import VectorArenaEnv
from src.engine.physics import PhysicsEngine
from src.engine.entities import DiscAgent
from src.agents.base import BaseAgent
from src.training.league import LeagueManager
from src.eval.tournament import play_match, play_single_round
from src.maps import MAP_REGISTRY


class PredeterminedAgent(BaseAgent):
    """Agent that returns a predetermined action sequence or constant action."""

    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.action.copy()


@pytest.mark.parametrize("map_name", list(MAP_REGISTRY.keys()))
def test_env_observation_parity_at_reset(map_name):
    """Verify that KineticSumoEnv and VectorArenaEnv produce identical observations across all maps."""
    k_env = KineticSumoEnv(map_name=map_name)
    v_env = VectorArenaEnv(num_envs=1, map_name=map_name)

    k_obs, _ = k_env.reset(seed=42)
    v_obs = v_env.reset()

    # Compare agent 0
    diff_0 = np.max(np.abs(k_obs["agent_0"] - v_obs[0, 0]))
    assert diff_0 < 1e-4, f"Mismatch on {map_name} agent_0: max diff = {diff_0}"

    # Compare agent 1
    diff_1 = np.max(np.abs(k_obs["agent_1"] - v_obs[0, 1]))
    assert diff_1 < 1e-4, f"Mismatch on {map_name} agent_1: max diff = {diff_1}"

    k_env.close()


def test_lidar_single_opponent_hit_at_initial_spawn():
    """Verify that at initial spawn distance (10m), only 1 forward ray hits the opponent (not 7)."""
    v_env = VectorArenaEnv(num_envs=1, map_name="Open")
    obs = v_env.get_observations()

    # Surface types are indices 30 to 46 for agent 0
    surface_types = obs[0, 0, 30:46]
    opponent_hits = np.where(surface_types == 1.0)[0]

    # Only ray 0 points along facing direction (+X) toward opponent at (+5, 0)
    assert len(opponent_hits) == 1, f"Expected exactly 1 ray hitting opponent, got {len(opponent_hits)}: {opponent_hits}"
    assert opponent_hits[0] == 0


def test_vector_env_anchor_collision_physics():
    """Verify that VectorArenaEnv applies anchor displacement priority and impulse absorption in parity with PhysicsEngine."""
    # Run in PhysicsEngine
    pe = PhysicsEngine()
    a0 = DiscAgent(initial_pos=[0.0, 0.0], initial_angle=0.0)
    a1 = DiscAgent(initial_pos=[1.5, 0.0], initial_angle=np.pi)
    a0.is_anchored = True
    a1.vel = np.array([-6.0, 0.0], dtype=np.float32)
    pe_res = pe.resolve_agent_collision(a0, a1)

    # Run in VectorArenaEnv
    v_env = VectorArenaEnv(num_envs=1, map_name="Open")
    v_env.reset()
    v_env.pos[0, 0] = [0.0, 0.0]
    v_env.pos[0, 1] = [1.5, 0.0]
    v_env.vel[0, 0] = [0.0, 0.0]
    v_env.vel[0, 1] = [-6.0, 0.0]

    actions = np.zeros((1, 2, 3), dtype=np.float32)
    actions[0, 0, 2] = -1.0  # Anchor agent 0

    obs, rewards, dones, infos = v_env.step(actions)

    # Agent 0 was anchored:
    # 1. Positional resolution parity: Agent 0 remains near origin, agent 1 displaced outward
    assert v_env.pos[0, 0, 0] <= 0.05
    assert v_env.pos[0, 1, 0] >= 1.55

    # 2. Velocity change parity between PhysicsEngine and VectorArenaEnv
    # (Allowing slight difference due to single-step linear drag integration in VectorArenaEnv)
    assert np.isclose(v_env.vel[0, 0, 0], a0.vel[0], atol=0.05)
    assert np.isclose(v_env.vel[0, 1, 0], a1.vel[0], atol=0.05)

    # 3. Anchor impulse absorption: Agent 0 experiences 50% impulse of Agent 1
    delta_v0 = abs(v_env.vel[0, 0, 0])
    delta_v1 = abs(v_env.vel[0, 1, 0] - (-6.0))
    assert delta_v0 < delta_v1 * 0.6


def test_vector_env_auto_reset_observation_integrity():
    """Verify that auto-reset in step() preserves terminal_observation and returns valid reset observation."""
    v_env = VectorArenaEnv(num_envs=2, map_name="Open")
    v_env.reset()

    # Place env 0 agent 0 out of bounds (dist > 10.0)
    v_env.pos[0, 0] = [15.0, 0.0]

    actions = np.zeros((2, 2, 3), dtype=np.float32)
    obs, rewards, dones, infos = v_env.step(actions)

    assert bool(dones[0]) is True
    assert bool(dones[1]) is False
    assert "terminal_observation" in infos[0]

    # Terminal observation should record the out-of-bounds position
    term_pos_x = infos[0]["terminal_observation"][0, 0] * 10.0
    assert term_pos_x >= 14.0

    # The returned observation obs[0] must be the NEW episode reset spawn position (-5.0, 0.0)
    reset_pos_x = obs[0, 0, 0] * 10.0
    assert np.isclose(reset_pos_x, -5.0, atol=0.1)


def test_league_manager_record_result_updates_winrates():
    """Verify that LeagueManager records match results and updates sampling weights."""
    league = LeagueManager()
    agent = PredeterminedAgent(np.zeros(3))
    cp = league.add_snapshot(agent, iteration=1)

    assert cp.win_rate == 0.5
    assert cp.total_matches == 0

    # Record 5 wins for the snapshot opponent
    cat = f"historical_{cp.id}"
    for _ in range(5):
        league.record_result(cat, won=True, draw=False)

    assert cp.wins == 5
    assert cp.total_matches == 5
    assert cp.win_rate == 1.0


def test_tournament_first_to_3_rounds_wins():
    """Verify that play_match adheres to 'First to 3 rounds wins'."""
    # Agent 0 always drives forward with dash
    agent_0 = PredeterminedAgent(np.array([1.0, 0.0, 1.0]))
    # Agent 1 drives in reverse out of the ring
    agent_1 = PredeterminedAgent(np.array([-1.0, 0.0, 0.0]))

    score = play_match(agent_0, agent_1, map_name="Open", rounds_to_win=3)
    # Agent 0 should win 3-0, receiving final score 1.0
    assert score == 1.0
