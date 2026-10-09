"""PettingZoo ParallelEnv implementation of the Kinetic Sumo 1v1 hover duel."""

import copy
from typing import Any
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.engine.entities import DiscAgent, AgentConfig
from src.engine.physics import PhysicsEngine, CollisionResult
from src.engine.raycast import RaycastEngine
from src.env.rewards import RewardCalculator
from src.maps import ArenaMap, get_map


class KineticSumoEnv(ParallelEnv):
    """PettingZoo ParallelEnv for 1v1 2D physics-driven hover arena duel."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "kinetic_sumo_v0",
    }

    def __init__(
        self,
        map_name: str = "Open",
        arena_map: ArenaMap | None = None,
        render_mode: str | None = None,
        max_duration: float = 45.0,
        dt: float = 0.02,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.map_preset = arena_map if arena_map is not None else get_map(map_name)
        self.render_mode = render_mode
        self.max_duration = float(max_duration)
        self.dt = float(dt)
        self.gamma = float(gamma)

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = copy.copy(self.possible_agents)

        # Physics and sensory engines
        self.physics = PhysicsEngine(
            initial_radius=self.map_preset.initial_radius,
            min_radius=self.map_preset.min_radius,
            shrink_start_time=self.map_preset.shrink_start_time,
            round_duration=self.max_duration,
            dt=self.dt,
            surface_friction=getattr(self.map_preset, "surface_friction", 1.0),
            shrink_mode=getattr(self.map_preset, "shrink_mode", "continuous"),
            step_interval=getattr(self.map_preset, "step_interval", 5.0),
            step_drop_amount=getattr(self.map_preset, "step_drop_amount", 1.5),
        )
        self.raycast = RaycastEngine(num_rays=16, max_range=self.map_preset.initial_radius * 2.5)
        self.reward_calc = RewardCalculator(gamma=self.gamma, max_speed=15.0)

        # Agent state objects
        self.agent_objs: dict[str, DiscAgent] = {
            "agent_0": DiscAgent(AgentConfig()),
            "agent_1": DiscAgent(AgentConfig()),
        }

        # Observation space: 46 floats per agent
        # 7 self kinematics + 5 opp relative + 2 resources + 32 lidar
        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(46,),
            dtype=np.float32,
        )
        # Action space: continuous 3-vector in [-1, 1]
        self._act_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

        # Episode state variables
        self.elapsed_time = 0.0
        self.step_count = 0
        self.last_collision = CollisionResult()
        self.winner: str | None = None
        self.is_draw = False

        # Lazy renderer for offscreen modes, immediate for human window
        self._renderer = None
        if self.render_mode == "human":
            from src.ui.renderer import SumoRenderer
            self._renderer = SumoRenderer(arena_map=self.map_preset, mode=self.render_mode)

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset environment to initial state."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.agents = copy.copy(self.possible_agents)
        self.elapsed_time = 0.0
        self.step_count = 0
        self.winner = None
        self.is_draw = False
        self.last_collision = CollisionResult()

        # Spawns from map preset
        (p0, a0), (p1, a1) = self.map_preset.get_spawn_transforms()

        # Optional random heading noise on reset if seeded / desired
        if seed is not None:
            np.random.seed(seed)

        self.agent_objs["agent_0"].reset(p0, a0)
        self.agent_objs["agent_1"].reset(p1, a1)

        current_r = self.physics.compute_arena_radius(self.elapsed_time)
        self.reward_calc.reset(self.agent_objs, current_r)

        obs = {a: self._compute_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, infos

    def _compute_obs(self, agent_id: str) -> np.ndarray:
        """Construct 1D float32 array of shape (46,) for given agent."""
        self_obj = self.agent_objs[agent_id]
        opp_id = "agent_1" if agent_id == "agent_0" else "agent_0"
        opp_obj = self.agent_objs[opp_id]

        r0 = self.map_preset.initial_radius
        v_max = self_obj.config.max_speed
        w_max = self_obj.config.max_angular_vel
        e_max = self_obj.config.max_energy
        cd_max = self_obj.config.dash_cooldown

        # 1. Self Kinematics (7): [x/R, y/R, vx/Vmax, vy/Vmax, cos(theta), sin(theta), omega/omega_max]
        self_kin = np.array(
            [
                self_obj.pos[0] / r0,
                self_obj.pos[1] / r0,
                self_obj.vel[0] / v_max,
                self_obj.vel[1] / v_max,
                np.cos(self_obj.angle),
                np.sin(self_obj.angle),
                self_obj.angular_vel / w_max,
            ],
            dtype=np.float32,
        )

        # 2. Opponent Relative (5): [dx/(2R), dy/(2R), dvx/(2Vmax), dvy/(2Vmax), (angle_to_opp - theta)/pi]
        dx = opp_obj.pos[0] - self_obj.pos[0]
        dy = opp_obj.pos[1] - self_obj.pos[1]
        dvx = opp_obj.vel[0] - self_obj.vel[0]
        dvy = opp_obj.vel[1] - self_obj.vel[1]

        angle_to_opp = np.arctan2(dy, dx)
        angle_diff = (angle_to_opp - self_obj.angle + np.pi) % (2.0 * np.pi) - np.pi
        rel_angle_norm = angle_diff / np.pi

        opp_rel = np.array(
            [
                dx / (2.0 * r0),
                dy / (2.0 * r0),
                dvx / (2.0 * v_max),
                dvy / (2.0 * v_max),
                rel_angle_norm,
            ],
            dtype=np.float32,
        )

        # 3. Resources (2): [energy / Emax, dash_cooldown / CDmax]
        res = np.array(
            [
                self_obj.energy / e_max,
                (self_obj.dash_cooldown / cd_max) if cd_max > 0 else 0.0,
            ],
            dtype=np.float32,
        )

        # 4. 1D Lidar Array (32): 16 distances + 16 hit surface types
        current_r = self.physics.compute_arena_radius(self.elapsed_time)
        lidar = self.raycast.get_lidar_array(
            origin=self_obj.pos,
            heading_angle=self_obj.angle,
            arena_radius=current_r,
            opponent_pos=opp_obj.pos,
            opponent_radius=opp_obj.config.radius,
            obstacles=self.map_preset.obstacles,
        )

        return np.concatenate([self_kin, opp_rel, res, lidar], axis=0).astype(np.float32)

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Step the environment forward by dt with given agent actions."""
        if not self.agents:
            return {}, {}, {}, {}, {}

        # 1. Apply controls to both agents
        for agent_id in self.agents:
            act = actions.get(agent_id, np.zeros(3, dtype=np.float32))
            self.physics.apply_agent_controls(self.agent_objs[agent_id], act)

        # 2. Resolve inter-agent collision
        self.last_collision = self.physics.resolve_agent_collision(
            self.agent_objs["agent_0"], self.agent_objs["agent_1"]
        )

        # 3. Resolve obstacle collisions
        for obstacle in self.map_preset.obstacles:
            for agent_id in self.agents:
                self.physics.resolve_obstacle_collision(self.agent_objs[agent_id], obstacle)

        # 4. Advance time and compute arena radius
        self.elapsed_time += self.dt
        self.step_count += 1
        current_r = self.physics.compute_arena_radius(self.elapsed_time)

        # 5. Check out of bounds / ring-out
        oob_0 = self.physics.check_out_of_bounds(self.agent_objs["agent_0"], current_r)
        oob_1 = self.physics.check_out_of_bounds(self.agent_objs["agent_1"], current_r)

        # 6. Check termination & truncation conditions
        is_terminated = False
        is_truncated = False
        self.winner = None
        self.is_draw = False

        if oob_0 and oob_1:
            # Both fell off simultaneously
            is_terminated = True
            self.is_draw = True
        elif oob_0:
            is_terminated = True
            self.winner = "agent_1"
        elif oob_1:
            is_terminated = True
            self.winner = "agent_0"
        elif self.elapsed_time >= self.max_duration:
            # Match timeout (cold-war stall draw)
            is_truncated = True
            self.is_draw = True

        is_match_over = is_terminated or is_truncated

        # 7. Compute rewards
        rewards = {}
        for agent_id in self.agents:
            opp_id = "agent_1" if agent_id == "agent_0" else "agent_0"
            act = actions.get(agent_id, np.zeros(3, dtype=np.float32))
            rew = self.reward_calc.compute_step_reward(
                agent_id=agent_id,
                agent=self.agent_objs[agent_id],
                opponent=self.agent_objs[opp_id],
                action=act,
                collision=self.last_collision,
                current_arena_radius=current_r,
                is_terminal=is_match_over,
                winner_id=self.winner,
                is_draw=self.is_draw,
            )
            rewards[agent_id] = float(rew)

        terminations = {a: is_terminated for a in self.agents}
        truncations = {a: is_truncated for a in self.agents}
        observations = {a: self._compute_obs(a) for a in self.agents}
        infos = {
            a: {
                "winner": self.winner,
                "is_draw": self.is_draw,
                "arena_radius": current_r,
                "elapsed_time": self.elapsed_time,
            }
            for a in self.agents
        }

        # Clear agents list if match concluded
        if is_match_over:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(
        self,
        round_scores: tuple[int, int] = (0, 0),
        rounds_to_win: int = 3,
        agent_names: tuple[str, str] = ("AGENT 0", "AGENT 1"),
        subtitle: str | None = None,
        fps: int | None = None,
    ) -> np.ndarray | None:
        """Render the environment using Pygame-CE renderer."""
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from src.ui.renderer import SumoRenderer
            self._renderer = SumoRenderer(arena_map=self.map_preset, mode=self.render_mode)

        current_r = self.physics.compute_arena_radius(self.elapsed_time)
        return self._renderer.render_frame(
            agents=self.agent_objs,
            arena_radius=current_r,
            elapsed_time=self.elapsed_time,
            last_collision=self.last_collision,
            winner=self.winner,
            is_draw=self.is_draw,
            round_scores=round_scores,
            rounds_to_win=rounds_to_win,
            agent_names=agent_names,
            subtitle=subtitle,
            fps=fps,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
