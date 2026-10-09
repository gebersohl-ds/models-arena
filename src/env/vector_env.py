"""Vectorized pure NumPy simulation environment for high-throughput RL training (>50,000 steps/sec)."""

import numpy as np
from src.maps import ArenaMap, get_map


class VectorArenaEnv:
    """Vectorized batch simulation of Kinetic Sumo duels in pure NumPy."""

    def __init__(
        self,
        num_envs: int = 64,
        map_name: str = "Open",
        max_duration: float = 45.0,
        dt: float = 0.02,
        gamma: float = 0.99,
    ):
        self.num_envs = int(num_envs)
        self.map_preset: ArenaMap = get_map(map_name)
        self.max_duration = float(max_duration)
        self.dt = float(dt)
        self.gamma = float(gamma)

        # Agent constants
        self.radius = 0.8
        self.mass = 10.0
        self.inertia = 3.2
        self.bumper_half_arc = np.pi / 3.0  # 60 degrees
        self.max_thrust = 120.0
        self.max_brake = 60.0
        self.max_torque = 40.0
        self.max_speed = 15.0
        self.max_angular_vel = 8.0
        self.linear_drag = 0.8
        self.angular_drag_mu = 1.2
        self.max_energy = 100.0
        self.energy_recharge = 25.0
        self.dash_cost = 40.0
        self.dash_duration = 0.25
        self.dash_cooldown_max = 0.75
        self.dash_mult = 2.0
        self.anchor_friction_mult = 4.0
        self.restitution = 0.8
        self.flank_mult = 2.5

        # Lidar ray offsets
        self.num_rays = 16
        self.max_lidar_range = self.map_preset.initial_radius * 2.5
        self.ray_offsets = np.linspace(0.0, 2.0 * np.pi, num=self.num_rays, endpoint=False, dtype=np.float32)

        # Obstacles
        self.has_obstacles = len(self.map_preset.obstacles) > 0
        if self.has_obstacles:
            self.obs_positions = np.array([[o.x, o.y] for o in self.map_preset.obstacles], dtype=np.float32)
            self.obs_radii = np.array([o.radius for o in self.map_preset.obstacles], dtype=np.float32)
            self.obs_restitutions = np.array([o.restitution for o in self.map_preset.obstacles], dtype=np.float32)
        else:
            self.obs_positions = np.empty((0, 2), dtype=np.float32)
            self.obs_radii = np.empty((0,), dtype=np.float32)
            self.obs_restitutions = np.empty((0,), dtype=np.float32)

        # State tensors: shape (num_envs, 2, ...)
        # Agent 0 is index 0, Agent 1 is index 1
        self.pos = np.zeros((self.num_envs, 2, 2), dtype=np.float32)
        self.vel = np.zeros((self.num_envs, 2, 2), dtype=np.float32)
        self.angle = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.angular_vel = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.energy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.dash_timer = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.dash_cooldown = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.elapsed_time = np.zeros(self.num_envs, dtype=np.float32)

        # Caches for reward calculation
        self.prev_potentials = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_envs, 2, 3), dtype=np.float32)

        self.reset()

    def reset(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Reset specified environments (or all if None). Returns observations (num_envs, 2, 46)."""
        if indices is None:
            idx = np.arange(self.num_envs)
        else:
            idx = np.asarray(indices)

        n = len(idx)
        if n == 0:
            return self.get_observations()

        (p0, a0), (p1, a1) = self.map_preset.get_spawn_transforms()

        self.pos[idx, 0] = p0
        self.pos[idx, 1] = p1
        self.vel[idx] = 0.0
        self.angle[idx, 0] = a0
        self.angle[idx, 1] = a1
        self.angular_vel[idx] = 0.0
        self.energy[idx] = self.max_energy
        self.dash_timer[idx] = 0.0
        self.dash_cooldown[idx] = 0.0
        self.elapsed_time[idx] = 0.0
        self.prev_actions[idx] = 0.0

        # Initial potentials
        current_r = self._compute_arena_radii(self.elapsed_time[idx])
        self.prev_potentials[idx] = self._compute_potentials(idx, current_r)

        return self.get_observations()

    def _compute_arena_radii(self, elapsed: np.ndarray) -> np.ndarray:
        """Compute outer ring radius for an array of elapsed times."""
        t_shrink = self.map_preset.shrink_start_time
        r_init = self.map_preset.initial_radius
        r_min = self.map_preset.min_radius
        window = max(1e-5, self.max_duration - t_shrink)

        progress = np.clip((elapsed - t_shrink) / window, 0.0, 1.0)
        radii = np.where(elapsed <= t_shrink, r_init, r_init - progress * (r_init - r_min))
        return radii.astype(np.float32)

    def _compute_potentials(self, idx: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Vectorized potential Phi(s) = 0.4 * Center_Safety + 0.6 * Closing_Vel for envs in idx."""
        # pos shape (n, 2, 2)
        pos = self.pos[idx]
        vel = self.vel[idx]
        radii_col = radii[:, None]  # (n, 1)

        # Center safety: (n, 2)
        dist_center = np.linalg.norm(pos, axis=-1)
        center_safety = np.clip(1.0 - dist_center / np.maximum(1e-4, radii_col), 0.0, 1.0)

        # Closing velocity: agent 0 towards 1, agent 1 towards 0
        delta = np.zeros_like(pos)
        delta[:, 0] = pos[:, 1] - pos[:, 0]  # opp is 1 for 0
        delta[:, 1] = pos[:, 0] - pos[:, 1]  # opp is 0 for 1

        dist_opp = np.linalg.norm(delta, axis=-1, keepdims=True)  # (n, 2, 1)
        safe_dist = np.maximum(1e-6, dist_opp)
        unit_to_opp = delta / safe_dist  # (n, 2, 2)

        rel_vel = np.zeros_like(vel)
        rel_vel[:, 0] = vel[:, 0] - vel[:, 1]
        rel_vel[:, 1] = vel[:, 1] - vel[:, 0]

        closing_speed = np.sum(rel_vel * unit_to_opp, axis=-1)  # (n, 2)
        closing_norm = np.clip(closing_speed / self.max_speed, -1.0, 1.0)

        return (0.4 * center_safety + 0.6 * closing_norm).astype(np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Advance all environments by dt.

        Args:
            actions: (num_envs, 2, 3) float32 in [-1.0, 1.0].

        Returns:
            obs: (num_envs, 2, 46) float32
            rewards: (num_envs, 2) float32
            dones: (num_envs,) bool (terminated or truncated)
            infos: list of dict per env
        """
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        thrust_cmd = actions[:, :, 0]
        steer_cmd = actions[:, :, 1]
        ability_cmd = actions[:, :, 2]

        dt = self.dt

        # 1. Update cooldowns and timers
        self.dash_cooldown = np.maximum(0.0, self.dash_cooldown - dt)
        self.dash_timer = np.maximum(0.0, self.dash_timer - dt)
        is_dashing = self.dash_timer > 0.0

        # Recharge energy when not dashing
        self.energy = np.where(
            ~is_dashing,
            np.minimum(self.max_energy, self.energy + self.energy_recharge * dt),
            self.energy,
        )

        # Ability trigger checks
        dash_requested = ability_cmd > 0.5
        can_dash = dash_requested & (~is_dashing) & (self.dash_cooldown <= 0.0) & (self.energy >= self.dash_cost)

        self.energy = np.where(can_dash, self.energy - self.dash_cost, self.energy)
        self.dash_timer = np.where(can_dash, self.dash_duration, self.dash_timer)
        self.dash_cooldown = np.where(can_dash, self.dash_cooldown_max, self.dash_cooldown)
        is_dashing = self.dash_timer > 0.0
        is_anchored = ability_cmd < -0.5

        # 2. Heading vectors
        cos_ang = np.cos(self.angle)
        sin_ang = np.sin(self.angle)
        heading = np.stack([cos_ang, sin_ang], axis=-1)  # (num_envs, 2, 2)

        # 3. Linear forces
        # Forward [0, 1] or reverse brake [-1, -0.3] with neutral deadband in (-0.3, 0.0]
        brake_scale = np.clip((-thrust_cmd - 0.3) / 0.7, 0.0, 1.0)
        thrust_force = np.where(
            thrust_cmd > 0.0,
            thrust_cmd * self.max_thrust,
            np.where(thrust_cmd <= -0.3, -brake_scale * self.max_brake, 0.0),
        )
        # Dash multiplier
        thrust_force = np.where(is_dashing, self.max_thrust * self.dash_mult, thrust_force)
        # Anchored disables forward/brake thrust
        thrust_force = np.where(is_anchored, 0.0, thrust_force)

        linear_force = heading * thrust_force[:, :, None]

        # Drag force
        drag_mult = np.where(is_anchored, self.anchor_friction_mult, 1.0)
        drag_force = -self.linear_drag * drag_mult[:, :, None] * self.vel

        accel = (linear_force + drag_force) / self.mass
        self.vel += accel * dt

        # Cap speed
        speed = np.linalg.norm(self.vel, axis=-1, keepdims=True)
        speed_mask = speed > self.max_speed
        self.vel = np.where(speed_mask, (self.vel / np.maximum(1e-6, speed)) * self.max_speed, self.vel)

        # 4. Rotational torque & damping
        steer_torque = np.where(is_dashing, 0.0, steer_cmd * self.max_torque)
        ang_drag = -(self.angular_drag_mu * drag_mult) * self.angular_vel * np.abs(self.angular_vel)
        ang_accel = (steer_torque + ang_drag) / self.inertia
        self.angular_vel += ang_accel * dt
        self.angular_vel = np.clip(self.angular_vel, -self.max_angular_vel, self.max_angular_vel)

        # Integrate positions and angles
        self.pos += self.vel * dt
        self.angle += self.angular_vel * dt
        self.angle = (self.angle + np.pi) % (2.0 * np.pi) - np.pi

        # 5. Inter-agent collisions (Vectorized across all envs)
        # delta pointing from 0 to 1: (num_envs, 2)
        delta_p = self.pos[:, 1] - self.pos[:, 0]
        dist_d = np.linalg.norm(delta_p, axis=-1)
        min_dist = 2.0 * self.radius

        collided = (dist_d < min_dist) & (dist_d > 1e-6)
        c_idx = np.where(collided)[0]

        impulse_0 = np.zeros(self.num_envs, dtype=np.float32)
        impulse_1 = np.zeros(self.num_envs, dtype=np.float32)
        flank_hit_on_0 = np.zeros(self.num_envs, dtype=bool)
        flank_hit_on_1 = np.zeros(self.num_envs, dtype=bool)

        if len(c_idx) > 0:
            normals = delta_p[c_idx] / dist_d[c_idx, None]  # normal 0 -> 1
            pens = (min_dist - dist_d[c_idx])[:, None]

            # Separate positions respecting anchor states
            anc_0 = is_anchored[c_idx, 0]
            anc_1 = is_anchored[c_idx, 1]

            case_0_anc = anc_0 & (~anc_1)
            case_1_anc = anc_1 & (~anc_0)
            case_both = ~(case_0_anc | case_1_anc)

            self.pos[c_idx, 0] -= normals * (
                np.where(case_1_anc[:, None], pens, np.where(case_both[:, None], pens * 0.5, 0.0))
            )
            self.pos[c_idx, 1] += normals * (
                np.where(case_0_anc[:, None], pens, np.where(case_both[:, None], pens * 0.5, 0.0))
            )

            # Rel vel
            rel_v = self.vel[c_idx, 1] - self.vel[c_idx, 0]
            v_rel_n = np.sum(rel_v * normals, axis=-1)
            approach = v_rel_n < 0.0

            if np.any(approach):
                app_idx = c_idx[approach]
                app_normals = normals[approach]
                app_vrel = v_rel_n[approach]
                anc_app_0 = is_anchored[app_idx, 0]
                anc_app_1 = is_anchored[app_idx, 1]

                # Bumper checks
                head_0 = heading[app_idx, 0]
                head_1 = heading[app_idx, 1]
                dot_0 = np.sum(head_0 * app_normals, axis=-1)
                dot_1 = np.sum(head_1 * (-app_normals), axis=-1)
                bumper_0 = dot_0 >= np.cos(self.bumper_half_arc)
                bumper_1 = dot_1 >= np.cos(self.bumper_half_arc)

                m0 = self.mass * np.where(anc_app_0, self.anchor_friction_mult, 1.0)
                m1 = self.mass * np.where(anc_app_1, self.anchor_friction_mult, 1.0)
                j_base = -(1.0 + self.restitution) * app_vrel / (1.0 / m0 + 1.0 / m1)

                j0 = j_base.copy()
                j1 = j_base.copy()

                # Flank on 1 (0 attacked)
                flank_1 = bumper_0 & (~bumper_1)
                j1[flank_1] = j_base[flank_1] * self.flank_mult
                j0[flank_1] = 0.0  # attacker recoil cancelled

                # Flank on 0 (1 attacked)
                flank_0 = bumper_1 & (~bumper_0)
                j0[flank_0] = j_base[flank_0] * self.flank_mult
                j1[flank_0] = 0.0

                # Anchor absorption: if target is anchored, reduce applied impulse
                j0 = np.where(anc_app_0, j0 * 0.5, j0)
                j1 = np.where(anc_app_1, j1 * 0.5, j1)

                self.vel[app_idx, 0] -= (j0[:, None] / self.mass) * app_normals
                self.vel[app_idx, 1] += (j1[:, None] / self.mass) * app_normals

                impulse_0[app_idx] = j0
                impulse_1[app_idx] = j1
                flank_hit_on_0[app_idx] = flank_0
                flank_hit_on_1[app_idx] = flank_1

        # 6. Obstacle collisions
        if self.has_obstacles:
            for obs_p, obs_r, obs_rest in zip(self.obs_positions, self.obs_radii, self.obs_restitutions):
                for a_idx in (0, 1):
                    d_obs = self.pos[:, a_idx] - obs_p
                    dist_o = np.linalg.norm(d_obs, axis=-1)
                    min_o = self.radius + obs_r
                    coll_o = (dist_o < min_o) & (dist_o > 1e-6)
                    if np.any(coll_o):
                        norm_o = d_obs[coll_o] / dist_o[coll_o, None]
                        pen_o = (min_o - dist_o[coll_o])[:, None]
                        self.pos[coll_o, a_idx] += norm_o * pen_o
                        v_dot_n = np.sum(self.vel[coll_o, a_idx] * norm_o, axis=-1)
                        bounce_mask = v_dot_n < 0.0
                        if np.any(bounce_mask):
                            sub_idx = np.where(coll_o)[0][bounce_mask]
                            sub_n = norm_o[bounce_mask]
                            sub_v = v_dot_n[bounce_mask, None]
                            self.vel[sub_idx, a_idx] -= (1.0 + obs_rest) * sub_v * sub_n

        # 7. Time and boundaries
        self.elapsed_time += dt
        curr_radii = self._compute_arena_radii(self.elapsed_time)

        dist_c0 = np.linalg.norm(self.pos[:, 0], axis=-1)
        dist_c1 = np.linalg.norm(self.pos[:, 1], axis=-1)
        oob_0 = dist_c0 > curr_radii
        oob_1 = dist_c1 > curr_radii
        timeout = self.elapsed_time >= self.max_duration

        dones = oob_0 | oob_1 | timeout

        # 8. Compute Rewards
        curr_potentials = self._compute_potentials(np.arange(self.num_envs), curr_radii)
        # If done, terminal potential is 0
        pbrs_0 = np.where(dones, 0.0, self.gamma * curr_potentials[:, 0]) - self.prev_potentials[:, 0]
        pbrs_1 = np.where(dones, 0.0, self.gamma * curr_potentials[:, 1]) - self.prev_potentials[:, 1]
        self.prev_potentials = curr_potentials.copy()

        # Terminal rewards
        term_rew_0 = np.zeros(self.num_envs, dtype=np.float32)
        term_rew_1 = np.zeros(self.num_envs, dtype=np.float32)
        # Case both out or timeout
        draw_mask = (oob_0 & oob_1) | timeout
        term_rew_0[draw_mask] = -1.0
        term_rew_1[draw_mask] = -1.0
        # Case 0 wins
        win_0 = (~oob_0) & oob_1
        term_rew_0[win_0] = 1.0
        term_rew_1[win_0] = -1.0
        # Case 1 wins
        win_1 = oob_0 & (~oob_1)
        term_rew_0[win_1] = -1.0
        term_rew_1[win_1] = 1.0

        # Contact rewards
        cont_0 = (
            0.4 * (impulse_1 / self.max_speed)
            - 0.1 * (impulse_0 / self.max_speed)
            + 0.3 * flank_hit_on_1.astype(np.float32)
        )
        cont_1 = (
            0.4 * (impulse_0 / self.max_speed)
            - 0.1 * (impulse_1 / self.max_speed)
            + 0.3 * flank_hit_on_0.astype(np.float32)
        )

        # Penalties: spin, jitter, inactivity
        spin_excess = np.maximum(0.0, np.abs(self.angular_vel) - 1.0)
        p_spin = 0.05 * (spin_excess**2)

        act_diff = actions - self.prev_actions
        p_jitter = 0.01 * np.sum(act_diff**2, axis=-1)
        self.prev_actions = actions.copy()

        p_inact = 0.002
        penalties = p_spin + p_jitter + p_inact

        rewards_0 = np.where(dones, term_rew_0, 0.0) + pbrs_0 + cont_0 - penalties[:, 0]
        rewards_1 = np.where(dones, term_rew_1, 0.0) + pbrs_1 + cont_1 - penalties[:, 1]
        rewards = np.stack([rewards_0, rewards_1], axis=-1).astype(np.float32)

        infos = [{"done": bool(d), "winner": "agent_0" if w0 else ("agent_1" if w1 else None)} for d, w0, w1 in zip(dones, win_0, win_1)]

        # Collect terminal observations for completed environments before reset
        terminal_obs = self.get_observations()

        # Auto-reset completed environments
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            for d_idx in done_indices:
                infos[d_idx]["terminal_observation"] = terminal_obs[d_idx]
            self.reset(done_indices)

        # Observations for continuing execution (reset envs show fresh spawn observations)
        obs = self.get_observations()

        return obs, rewards, dones, infos

    def get_observations(self) -> np.ndarray:
        """Construct full (num_envs, 2, 46) observation matrix in pure vectorized NumPy."""
        r0 = self.map_preset.initial_radius
        curr_radii = self._compute_arena_radii(self.elapsed_time)

        # (num_envs, 2, 7) Self Kinematics
        cos_ang = np.cos(self.angle)
        sin_ang = np.sin(self.angle)
        self_kin = np.stack(
            [
                self.pos[:, :, 0] / r0,
                self.pos[:, :, 1] / r0,
                self.vel[:, :, 0] / self.max_speed,
                self.vel[:, :, 1] / self.max_speed,
                cos_ang,
                sin_ang,
                self.angular_vel / self.max_angular_vel,
            ],
            axis=-1,
        )  # (num_envs, 2, 7)

        # (num_envs, 2, 5) Opponent Relative
        # For agent 0: opp is 1. For agent 1: opp is 0.
        dx0 = self.pos[:, 1, 0] - self.pos[:, 0, 0]
        dy0 = self.pos[:, 1, 1] - self.pos[:, 0, 1]
        dvx0 = self.vel[:, 1, 0] - self.vel[:, 0, 0]
        dvy0 = self.vel[:, 1, 1] - self.vel[:, 0, 1]

        dx1 = -dx0
        dy1 = -dy0
        dvx1 = -dvx0
        dvy1 = -dvy0

        dx = np.stack([dx0, dx1], axis=-1)  # (n, 2)
        dy = np.stack([dy0, dy1], axis=-1)
        dvx = np.stack([dvx0, dvx1], axis=-1)
        dvy = np.stack([dvy0, dvy1], axis=-1)

        ang_to_opp = np.arctan2(dy, dx)
        ang_diff = (ang_to_opp - self.angle + np.pi) % (2.0 * np.pi) - np.pi
        rel_ang_norm = ang_diff / np.pi

        opp_rel = np.stack(
            [
                dx / (2.0 * r0),
                dy / (2.0 * r0),
                dvx / (2.0 * self.max_speed),
                dvy / (2.0 * self.max_speed),
                rel_ang_norm,
            ],
            axis=-1,
        )  # (num_envs, 2, 5)

        # (num_envs, 2, 2) Resources
        cd_norm = self.dash_cooldown / self.dash_cooldown_max
        resources = np.stack([self.energy / self.max_energy, cd_norm], axis=-1)

        # (num_envs, 2, 32) Lidar Raycasting Vectorized
        # Flatten (num_envs, 2) into (2N,)
        flat_pos = self.pos.reshape(-1, 2)  # (2N, 2)
        flat_angle = self.angle.reshape(-1)  # (2N,)
        flat_radii = np.repeat(curr_radii, 2)  # (2N,)

        opp_p0 = self.pos[:, 1]
        opp_p1 = self.pos[:, 0]
        flat_opp_pos = np.empty_like(flat_pos)
        flat_opp_pos[0::2] = opp_p0
        flat_opp_pos[1::2] = opp_p1

        # Ray directions: (2N, 16, 2)
        ray_angles = flat_angle[:, None] + self.ray_offsets[None, :]  # (2N, 16)
        dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)  # (2N, 16, 2)

        best_dist = np.full((len(flat_pos), self.num_rays), self.max_lidar_range, dtype=np.float32)
        hit_types = np.zeros((len(flat_pos), self.num_rays), dtype=np.float32)

        # Arena boundary intersection: (2N, 16)
        h_b = np.sum(flat_pos[:, None, :] * dirs, axis=-1)
        c_b = (np.sum(flat_pos**2, axis=-1) - flat_radii**2)[:, None]
        disc_b = np.maximum(0.0, h_b**2 - c_b)
        s_b = -h_b + np.sqrt(disc_b)
        mask_b = (s_b > 1e-4) & (s_b <= self.max_lidar_range) & (s_b < best_dist)
        best_dist[mask_b] = s_b[mask_b]
        hit_types[mask_b] = -1.0

        # Opponent disc intersection: (2N, 16)
        d_opp = flat_pos - flat_opp_pos  # (2N, 2)
        h_o = np.sum(d_opp[:, None, :] * dirs, axis=-1)
        c_o = (np.sum(d_opp**2, axis=-1) - self.radius**2)[:, None]
        disc_o = h_o**2 - c_o
        valid_o = disc_o >= 0.0

        s_opp = np.full_like(best_dist, np.inf)
        if np.any(valid_o):
            sq_disc = np.sqrt(np.maximum(0.0, disc_o))
            s1 = -h_o - sq_disc
            s2 = -h_o + sq_disc
            s_cand = np.where(s1 > 1e-4, s1, np.where(s2 > 1e-4, s2, np.inf))
            s_opp = np.where(valid_o, s_cand, np.inf)

        mask_opp = valid_o & (s_opp > 1e-4) & (s_opp < best_dist)
        best_dist[mask_opp] = s_opp[mask_opp]
        hit_types[mask_opp] = 1.0

        # Obstacles intersection
        if self.has_obstacles:
            for obs_p, obs_r in zip(self.obs_positions, self.obs_radii):
                d_obs = flat_pos - obs_p
                h_obs = np.sum(d_obs[:, None, :] * dirs, axis=-1)
                c_obs = (np.sum(d_obs**2, axis=-1) - obs_r**2)[:, None]
                disc_obs = h_obs**2 - c_obs
                sq_obs = np.sqrt(np.maximum(0.0, disc_obs))
                s1_o = -h_obs - sq_obs
                s2_o = -h_obs + sq_obs
                s_cand_o = np.where(s1_o > 1e-4, s1_o, np.where(s2_o > 1e-4, s2_o, np.inf))
                mask_obs = (disc_obs >= 0.0) & (s_cand_o > 1e-4) & (s_cand_o < best_dist)
                best_dist[mask_obs] = s_cand_o[mask_obs]
                hit_types[mask_obs] = 0.5

        norm_dist = np.clip(best_dist / self.max_lidar_range, 0.0, 1.0)
        lidar_flat = np.concatenate([norm_dist, hit_types], axis=-1)  # (2N, 32)
        lidar = lidar_flat.reshape(self.num_envs, 2, 32)

        obs = np.concatenate([self_kin, opp_rel, resources, lidar], axis=-1).astype(np.float32)
        return obs
