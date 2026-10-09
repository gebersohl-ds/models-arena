"""Vectorized 16-ray Lidar intersection engine for Kinetic Sumo."""

import numpy as np
from typing import Sequence
from src.engine.physics import Obstacle


class RaycastEngine:
    """Vectorized 2D raycasting engine for disc agents and obstacles."""

    def __init__(self, num_rays: int = 16, max_range: float = 25.0):
        self.num_rays = int(num_rays)
        self.max_range = float(max_range)
        # 16 radial rays uniformly spaced around the circle [0, 2pi)
        self.ray_offsets = np.linspace(0.0, 2.0 * np.pi, num=self.num_rays, endpoint=False, dtype=np.float32)

    def cast_lidar(
        self,
        origin: np.ndarray,
        heading_angle: float,
        arena_radius: float,
        opponent_pos: np.ndarray,
        opponent_radius: float,
        obstacles: Sequence[Obstacle] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform vectorized raycasting for an agent.

        Args:
            origin: Agent position (2,) float32.
            heading_angle: Agent heading in radians.
            arena_radius: Current radius of the outer death circle.
            opponent_pos: Opponent center position (2,) float32.
            opponent_radius: Opponent radius.
            obstacles: Optional list of Obstacle objects.

        Returns:
            Tuple of (distances, hit_types):
                distances: (16,) float32 in [0.0, 1.0] (normalized by max_range).
                hit_types: (16,) float32 with values:
                    -1.0: void edge (arena boundary)
                     0.0: none (no hit within max_range)
                     0.5: wall / bumper / obstacle
                     1.0: opponent disc
        """
        origin = np.asarray(origin, dtype=np.float32)
        opponent_pos = np.asarray(opponent_pos, dtype=np.float32)

        # 1. Compute ray directions: shape (16, 2)
        ray_angles = heading_angle + self.ray_offsets
        directions = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)  # (16, 2)

        # Initialize hits with max_range and 0.0 (none)
        best_dist = np.full(self.num_rays, self.max_range, dtype=np.float32)
        hit_type = np.zeros(self.num_rays, dtype=np.float32)

        # 2. Check outer arena boundary (void edge: -1.0)
        # Agent is inside circle centered at (0, 0) with radius R.
        # Ray: origin + s * dir.
        # |origin + s * dir|^2 = R^2 => s^2 + 2*(origin . dir)*s + (|origin|^2 - R^2) = 0
        h_boundary = np.sum(origin * directions, axis=-1)  # (16,)
        c_boundary = float(np.sum(origin**2) - arena_radius**2)
        disc_boundary = h_boundary**2 - c_boundary  # c_boundary < 0 when inside arena, so disc > 0
        disc_boundary = np.maximum(disc_boundary, 0.0)
        # Positive root is -h + sqrt(disc)
        s_boundary = -h_boundary + np.sqrt(disc_boundary)
        valid_b = (s_boundary > 1e-4) & (s_boundary <= self.max_range)
        mask_b = valid_b & (s_boundary < best_dist)
        best_dist[mask_b] = s_boundary[mask_b]
        hit_type[mask_b] = -1.0

        # 3. Check obstacles (surface type: 0.5)
        if obstacles:
            for obs in obstacles:
                obs_pos = obs.pos
                r_obs = obs.radius
                delta = origin - obs_pos  # (2,)
                h_obs = np.sum(delta * directions, axis=-1)  # (16,)
                c_obs = float(np.sum(delta**2) - r_obs**2)
                disc_obs = h_obs**2 - c_obs  # (16,)
                valid_obs = disc_obs >= 0.0

                s_obs = np.full(self.num_rays, np.inf, dtype=np.float32)
                if np.any(valid_obs):
                    sqrt_disc = np.sqrt(disc_obs[valid_obs])
                    s1 = -h_obs[valid_obs] - sqrt_disc
                    s2 = -h_obs[valid_obs] + sqrt_disc
                    # We want smallest positive distance
                    s_cand = np.where(s1 > 1e-4, s1, np.where(s2 > 1e-4, s2, np.inf))
                    s_obs[valid_obs] = s_cand

                mask_obs = (s_obs > 1e-4) & (s_obs < best_dist)
                best_dist[mask_obs] = s_obs[mask_obs]
                hit_type[mask_obs] = 0.5

        # 4. Check opponent disc (surface type: 1.0)
        delta_opp = origin - opponent_pos  # (2,)
        h_opp = np.sum(delta_opp * directions, axis=-1)  # (16,)
        c_opp = float(np.sum(delta_opp**2) - opponent_radius**2)
        disc_opp = h_opp**2 - c_opp
        valid_opp = disc_opp >= 0.0

        s_opp = np.full(self.num_rays, np.inf, dtype=np.float32)
        if np.any(valid_opp):
            sqrt_disc = np.sqrt(disc_opp[valid_opp])
            s1 = -h_opp[valid_opp] - sqrt_disc
            s2 = -h_opp[valid_opp] + sqrt_disc
            s_cand = np.where(s1 > 1e-4, s1, np.where(s2 > 1e-4, s2, np.inf))
            s_opp[valid_opp] = s_cand

        mask_opp = (s_opp > 1e-4) & (s_opp < best_dist)
        best_dist[mask_opp] = s_opp[mask_opp]
        hit_type[mask_opp] = 1.0

        # Normalized distances: [0, 1]
        norm_dist = np.clip(best_dist / self.max_range, 0.0, 1.0).astype(np.float32)
        return norm_dist, hit_type.astype(np.float32)

    def cast_multi_lidar(
        self,
        origin: np.ndarray,
        heading_angle: float,
        arena_radius: float,
        opponent_positions: Sequence[tuple[np.ndarray, float]],
        obstacles: Sequence[Obstacle] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform vectorized raycasting against multiple opponents and obstacles."""
        origin = np.asarray(origin, dtype=np.float32)

        # 1. Compute ray directions: shape (16, 2)
        ray_angles = heading_angle + self.ray_offsets
        directions = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

        best_dist = np.full(self.num_rays, self.max_range, dtype=np.float32)
        hit_type = np.zeros(self.num_rays, dtype=np.float32)

        # 2. Check outer arena boundary (void edge: -1.0)
        h_boundary = np.sum(origin * directions, axis=-1)
        c_boundary = float(np.sum(origin**2) - arena_radius**2)
        disc_boundary = h_boundary**2 - c_boundary
        disc_boundary = np.maximum(disc_boundary, 0.0)
        s_boundary = -h_boundary + np.sqrt(disc_boundary)
        valid_b = (s_boundary > 1e-4) & (s_boundary <= self.max_range)
        mask_b = valid_b & (s_boundary < best_dist)
        best_dist[mask_b] = s_boundary[mask_b]
        hit_type[mask_b] = -1.0

        # 3. Check obstacles (surface type: 0.5)
        if obstacles:
            for obs in obstacles:
                obs_pos = obs.pos
                r_obs = obs.radius
                delta = origin - obs_pos
                h_obs = np.sum(delta * directions, axis=-1)
                c_obs = float(np.sum(delta**2) - r_obs**2)
                disc_obs = h_obs**2 - c_obs
                valid_obs = disc_obs >= 0.0

                s_obs = np.full(self.num_rays, np.inf, dtype=np.float32)
                if np.any(valid_obs):
                    sqrt_disc = np.sqrt(disc_obs[valid_obs])
                    s1 = -h_obs[valid_obs] - sqrt_disc
                    s2 = -h_obs[valid_obs] + sqrt_disc
                    s_cand = np.where(s1 > 1e-4, s1, np.where(s2 > 1e-4, s2, np.inf))
                    s_obs[valid_obs] = s_cand

                mask_obs = (s_obs > 1e-4) & (s_obs < best_dist)
                best_dist[mask_obs] = s_obs[mask_obs]
                hit_type[mask_obs] = 0.5

        # 4. Check all opponents (surface type: 1.0)
        for opp_pos, opp_rad in opponent_positions:
            delta_opp = origin - np.asarray(opp_pos, dtype=np.float32)
            h_opp = np.sum(delta_opp * directions, axis=-1)
            c_opp = float(np.sum(delta_opp**2) - opp_rad**2)
            disc_opp = h_opp**2 - c_opp
            valid_opp = disc_opp >= 0.0

            s_opp = np.full(self.num_rays, np.inf, dtype=np.float32)
            if np.any(valid_opp):
                sqrt_disc = np.sqrt(disc_opp[valid_opp])
                s1 = -h_opp[valid_opp] - sqrt_disc
                s2 = -h_opp[valid_opp] + sqrt_disc
                s_cand = np.where(s1 > 1e-4, s1, np.where(s2 > 1e-4, s2, np.inf))
                s_opp[valid_opp] = s_cand

            mask_opp = (s_opp > 1e-4) & (s_opp < best_dist)
            best_dist[mask_opp] = s_opp[mask_opp]
            hit_type[mask_opp] = 1.0

        norm_dist = np.clip(best_dist / self.max_range, 0.0, 1.0).astype(np.float32)
        return norm_dist, hit_type.astype(np.float32)

    def get_lidar_array(
        self,
        origin: np.ndarray,
        heading_angle: float,
        arena_radius: float,
        opponent_pos: np.ndarray,
        opponent_radius: float,
        obstacles: Sequence[Obstacle] | None = None,
    ) -> np.ndarray:
        """Return 1D float32 array of shape (32,) combining distances and hit types."""
        dist, types = self.cast_lidar(
            origin, heading_angle, arena_radius, opponent_pos, opponent_radius, obstacles
        )
        return np.concatenate([dist, types], axis=0).astype(np.float32)

    def get_multi_agent_lidar_array(
        self,
        origin: np.ndarray,
        heading_angle: float,
        arena_radius: float,
        opponent_positions: Sequence[tuple[np.ndarray, float]],
        obstacles: Sequence[Obstacle] | None = None,
    ) -> np.ndarray:
        """Return 1D float32 array of shape (32,) for multi-agent arena."""
        dist, types = self.cast_multi_lidar(
            origin, heading_angle, arena_radius, opponent_positions, obstacles
        )
        return np.concatenate([dist, types], axis=0).astype(np.float32)

