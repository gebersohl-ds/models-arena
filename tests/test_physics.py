"""Unit tests for Kinetic Sumo physics engine and agent kinematics."""

import numpy as np
import pytest
from src.engine.entities import DiscAgent, AgentConfig
from src.engine.physics import PhysicsEngine, Obstacle


def test_agent_initialization_and_reset():
    agent = DiscAgent(initial_pos=np.array([2.0, 3.0]), initial_angle=0.5)
    assert np.allclose(agent.pos, [2.0, 3.0])
    assert np.isclose(agent.angle, 0.5)
    assert agent.energy == agent.config.max_energy
    assert agent.alive is True

    agent.reset(np.array([0.0, 0.0]), angle=np.pi)
    assert np.allclose(agent.pos, [0.0, 0.0])
    assert np.isclose(agent.angle, np.pi)
    assert np.allclose(agent.vel, [0.0, 0.0])


def test_bumper_arc_detection():
    # Agent facing +X (angle = 0)
    agent = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)

    # Directly in front: (1, 0) -> angle 0 <= 60 deg -> in bumper
    assert agent.is_in_bumper_arc(np.array([1.0, 0.0])) is True
    # At 45 deg: in bumper (45 <= 60)
    assert agent.is_in_bumper_arc(np.array([1.0, 1.0])) is True
    # At 59 deg: in bumper
    ang59 = np.deg2rad(59.0)
    assert agent.is_in_bumper_arc(np.array([np.cos(ang59), np.sin(ang59)])) is True
    # At 70 deg: flank (outside bumper)
    ang70 = np.deg2rad(70.0)
    assert agent.is_in_bumper_arc(np.array([np.cos(ang70), np.sin(ang70)])) is False
    # Directly behind: (-1, 0) -> rear (outside bumper)
    assert agent.is_in_bumper_arc(np.array([-1.0, 0.0])) is False


def test_thrust_and_linear_drag():
    engine = PhysicsEngine(dt=0.02)
    agent = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)

    # Apply full forward thrust for 10 steps
    for _ in range(10):
        engine.apply_agent_controls(agent, np.array([1.0, 0.0, 0.0]))

    # Agent should move forward along +X
    assert agent.vel[0] > 0.0
    assert agent.pos[0] > 0.0
    assert np.isclose(agent.vel[1], 0.0, atol=1e-5)

    # Now let it coast (thrust = 0) - drag should slow it down
    v_prev = agent.vel[0]
    for _ in range(5):
        engine.apply_agent_controls(agent, np.array([0.0, 0.0, 0.0]))
        assert agent.vel[0] < v_prev
        v_prev = agent.vel[0]


def test_nonlinear_rotational_damping():
    engine = PhysicsEngine(dt=0.02)
    agent = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)
    agent.angular_vel = 7.0  # high initial spin

    # Without steering input, non-linear damping tau_drag = -mu * omega * |omega| should rapidly damp spin
    omega_history = []
    for _ in range(25):
        engine.apply_agent_controls(agent, np.array([0.0, 0.0, 0.0]))
        omega_history.append(abs(agent.angular_vel))

    # Omega must monotonically decrease towards 0
    assert omega_history[-1] < omega_history[0] * 0.5
    for i in range(len(omega_history) - 1):
        assert omega_history[i + 1] <= omega_history[i]


def test_dash_mechanic():
    engine = PhysicsEngine(dt=0.02)
    agent = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)

    initial_energy = agent.energy
    # Trigger dash: ability > 0.5
    engine.apply_agent_controls(agent, np.array([1.0, 0.5, 0.8]))

    assert agent.is_dashing is True
    assert agent.energy == pytest.approx(initial_energy - agent.config.dash_cost, abs=1e-3)
    assert agent.dash_cooldown > 0.0

    # During dash, steering must be locked (even if steer input is applied)
    # Check angular acceleration is zero/damped and angle stays near 0
    assert abs(agent.angular_vel) < 1e-4

    # Dash ends after dash_duration (0.25s / 0.02 = 13 steps)
    for _ in range(15):
        engine.apply_agent_controls(agent, np.array([1.0, 0.0, 0.0]))
    assert agent.is_dashing is False


def test_anchor_mechanic():
    engine = PhysicsEngine(dt=0.02)
    agent = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)
    agent.vel = np.array([5.0, 0.0], dtype=np.float32)

    # Trigger anchor: ability < -0.5
    # Forward thrust should be zeroed and friction multiplied by 4
    engine.apply_agent_controls(agent, np.array([1.0, 0.0, -0.8]))
    assert agent.is_anchored is True

    # High deceleration compared to unanchored
    agent_free = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)
    agent_free.vel = np.array([5.0, 0.0], dtype=np.float32)
    engine.apply_agent_controls(agent_free, np.array([0.0, 0.0, 0.0]))

    # Anchored agent should lose more velocity than free gliding agent
    assert agent.vel[0] < agent_free.vel[0]


def test_combat_flank_hit_recoil_cancellation():
    engine = PhysicsEngine(dt=0.02)

    # Agent A facing +X, moving +X toward B: hits with front bumper!
    agent_a = DiscAgent(initial_pos=np.array([0.0, 0.0]), initial_angle=0.0)
    agent_a.vel = np.array([5.0, 0.0], dtype=np.float32)

    # Agent B at (1.2, 0.0) facing +Y: contact normal from A is (+1, 0).
    # For B, contact is (-1, 0) while B faces +Y (0, 1) -> angle is 90 deg -> B is hit on FLANK!
    agent_b = DiscAgent(initial_pos=np.array([1.2, 0.0]), initial_angle=np.pi / 2.0)
    agent_b.vel = np.array([0.0, 0.0], dtype=np.float32)

    initial_vel_a = agent_a.vel.copy()
    res = engine.resolve_agent_collision(agent_a, agent_b)

    assert res.occurred is True
    assert res.flank_hit_on_b is True
    assert res.flank_hit_on_a is False
    # Attacker recoil must be cancelled: A keeps its velocity
    assert np.allclose(agent_a.vel, initial_vel_a)
    # Target B receives boosted impulse (2.5x)
    assert agent_b.vel[0] > 0.0
    assert res.impulse_b > res.impulse_a


def test_combat_head_on_bumper_collision():
    engine = PhysicsEngine(dt=0.02)

    # Agent A at (-0.6, 0.0) facing +X (angle 0), moving +X
    agent_a = DiscAgent(initial_pos=np.array([-0.6, 0.0]), initial_angle=0.0)
    agent_a.vel = np.array([4.0, 0.0], dtype=np.float32)

    # Agent B at (+0.6, 0.0) facing -X (angle pi), moving -X
    agent_b = DiscAgent(initial_pos=np.array([0.6, 0.0]), initial_angle=np.pi)
    agent_b.vel = np.array([-4.0, 0.0], dtype=np.float32)

    res = engine.resolve_agent_collision(agent_a, agent_b)

    assert res.occurred is True
    assert res.flank_hit_on_a is False
    assert res.flank_hit_on_b is False
    # Symmetric bounce: both experience recoil
    assert agent_a.vel[0] < 0.0
    assert agent_b.vel[0] > 0.0
    assert np.isclose(res.impulse_a, res.impulse_b)


def test_arena_shrink_and_out_of_bounds():
    engine = PhysicsEngine(initial_radius=10.0, min_radius=4.0, shrink_start_time=20.0, round_duration=45.0)

    # Before 20s: radius is 10.0
    assert engine.compute_arena_radius(10.0) == 10.0
    assert engine.compute_arena_radius(20.0) == 10.0

    # At 32.5s (midway between 20 and 45): radius is 7.0
    assert engine.compute_arena_radius(32.5) == pytest.approx(7.0, abs=1e-3)

    # At 45s: radius is min_radius (4.0)
    assert engine.compute_arena_radius(45.0) == pytest.approx(4.0, abs=1e-3)

    # Test out of bounds instant death
    agent = DiscAgent(initial_pos=np.array([8.0, 0.0]))
    # Inside radius 10.0 -> alive
    assert engine.check_out_of_bounds(agent, 10.0) is False
    assert agent.alive is True

    # At radius 6.0 -> 8.0 > 6.0 -> dead
    assert engine.check_out_of_bounds(agent, 6.0) is True
    assert agent.alive is False


def test_obstacle_collision():
    engine = PhysicsEngine()
    obs = Obstacle(x=0.0, y=0.0, radius=1.0, restitution=1.0)
    agent = DiscAgent(initial_pos=np.array([1.5, 0.0]))  # radius 0.8 + 1.0 = 1.8 -> overlap
    agent.vel = np.array([-3.0, 0.0], dtype=np.float32)

    collided = engine.resolve_obstacle_collision(agent, obs)
    assert collided is True
    # Agent should bounce backward away from obstacle
    assert agent.vel[0] > 0.0
    assert agent.pos[0] >= 1.8 - 1e-5
