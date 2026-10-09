"""Unit tests for the vectorized 16-ray Lidar intersection engine."""

import numpy as np
import pytest
from src.engine.raycast import RaycastEngine
from src.engine.physics import Obstacle


def test_raycast_output_shapes_and_types():
    engine = RaycastEngine(num_rays=16, max_range=25.0)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    opp_pos = np.array([5.0, 0.0], dtype=np.float32)

    dist, types = engine.cast_lidar(
        origin=origin,
        heading_angle=0.0,
        arena_radius=10.0,
        opponent_pos=opp_pos,
        opponent_radius=0.8,
    )

    assert dist.shape == (16,)
    assert types.shape == (16,)
    assert dist.dtype == np.float32
    assert types.dtype == np.float32

    lidar_arr = engine.get_lidar_array(
        origin=origin,
        heading_angle=0.0,
        arena_radius=10.0,
        opponent_pos=opp_pos,
        opponent_radius=0.8,
    )
    assert lidar_arr.shape == (32,)
    assert lidar_arr.dtype == np.float32


def test_raycast_boundary_at_origin():
    engine = RaycastEngine(num_rays=16, max_range=25.0)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    # Opponent far outside arena
    opp_pos = np.array([100.0, 100.0], dtype=np.float32)

    dist, types = engine.cast_lidar(
        origin=origin,
        heading_angle=0.0,
        arena_radius=10.0,
        opponent_pos=opp_pos,
        opponent_radius=0.8,
    )

    # All rays should hit the circular boundary at radius 10.0
    expected_norm_dist = 10.0 / 25.0
    assert np.allclose(dist, expected_norm_dist, atol=1e-4)
    # All hit types must be -1.0 (void edge)
    assert np.allclose(types, -1.0)


def test_raycast_opponent_hit():
    engine = RaycastEngine(num_rays=16, max_range=25.0)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    # Opponent at (4.0, 0.0) with radius 0.8, agent facing 0 rad (+X)
    opp_pos = np.array([4.0, 0.0], dtype=np.float32)

    dist, types = engine.cast_lidar(
        origin=origin,
        heading_angle=0.0,
        arena_radius=10.0,
        opponent_pos=opp_pos,
        opponent_radius=0.8,
    )

    # Ray 0 points directly along +X -> should hit opponent
    # Distance to surface = 4.0 - 0.8 = 3.2
    assert types[0] == 1.0
    assert dist[0] == pytest.approx(3.2 / 25.0, abs=1e-3)

    # Ray pointing opposite (Ray 8 at 180 degrees) points toward -X -> hits arena boundary at 10.0
    assert types[8] == -1.0
    assert dist[8] == pytest.approx(10.0 / 25.0, abs=1e-3)


def test_raycast_obstacle_hit_and_occlusion():
    engine = RaycastEngine(num_rays=16, max_range=25.0)
    origin = np.array([0.0, 0.0], dtype=np.float32)
    # Obstacle at (2.0, 0.0) with radius 0.5
    obs = Obstacle(x=2.0, y=0.0, radius=0.5)
    # Opponent behind obstacle at (5.0, 0.0)
    opp_pos = np.array([5.0, 0.0], dtype=np.float32)

    dist, types = engine.cast_lidar(
        origin=origin,
        heading_angle=0.0,
        arena_radius=10.0,
        opponent_pos=opp_pos,
        opponent_radius=0.8,
        obstacles=[obs],
    )

    # Ray 0 should hit obstacle at 2.0 - 0.5 = 1.5, occluding the opponent behind it
    assert types[0] == 0.5
    assert dist[0] == pytest.approx(1.5 / 25.0, abs=1e-3)
