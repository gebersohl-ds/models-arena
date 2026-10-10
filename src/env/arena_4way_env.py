"""4-Way Battle Royale Free-For-All (FFA) environment for Kinetic Sumo."""

import copy
from typing import Any
import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.engine.entities import DiscAgent, AgentConfig
from src.engine.physics import PhysicsEngine, CollisionResult
from src.engine.raycast import RaycastEngine
from src.maps import ArenaMap, get_map


class KineticSumo4WayEnv(ParallelEnv):
    """PettingZoo ParallelEnv for 4-Way 2D physics-driven hover arena battle royale."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "kinetic_sumo_4way_v0",
    }

    def __init__(
        self,
        map_name: str = "Open",
        arena_map: ArenaMap | None = None,
        render_mode: str | None = None,
        max_duration: float = 60.0,
        dt: float = 0.02,
    ):
        super().__init__()
        self.map_preset = arena_map if arena_map is not None else get_map(map_name)
        self.render_mode = render_mode
        self.max_duration = float(max_duration)
        self.dt = float(dt)

        self.possible_agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        self.agents = copy.copy(self.possible_agents)

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

        # 4 disc agents
        self.agent_objs: dict[str, DiscAgent] = {
            a: DiscAgent(AgentConfig()) for a in self.possible_agents
        }

        # Observation space: 46 floats per agent (compatible with 1v1 heuristics)
        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(46,),
            dtype=np.float32,
        )
        self._act_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

        self.elapsed_time = 0.0
        self.step_count = 0
        self.eliminations: list[str] = []  # Ordered list of eliminated agent names
        self.winner: str | None = None
        self.is_draw = False
        self.recent_collisions: list[CollisionResult] = []
        self.last_actions: dict[str, np.ndarray] = {
            a: np.zeros(3, dtype=np.float32) for a in self.possible_agents
        }

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
        """Reset 4-way environment to initial state."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.agents = copy.copy(self.possible_agents)
        self.elapsed_time = 0.0
        self.step_count = 0
        self.eliminations = []
        self.winner = None
        self.is_draw = False
        self.recent_collisions = []
        self.last_actions = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        # 4-way spawn transforms
        spawns = self.map_preset.get_4way_spawn_transforms()
        for i, a_id in enumerate(self.possible_agents):
            pos, angle = spawns[i]
            self.agent_objs[a_id].reset(pos, angle)

        obs = {a: self._compute_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _find_nearest_active_rival(self, agent_id: str) -> str | None:
        """Find the closest alive opponent to agent_id."""
        self_obj = self.agent_objs[agent_id]
        nearest_id = None
        min_dist = float("inf")

        for other_id in self.possible_agents:
            if other_id == agent_id:
                continue
            other_obj = self.agent_objs[other_id]
            if not other_obj.alive:
                continue
            dist = float(np.linalg.norm(other_obj.pos - self_obj.pos))
            if dist < min_dist:
                min_dist = dist
                nearest_id = other_id

        return nearest_id

    def _compute_observation(self, agent_id: str) -> np.ndarray:
        """Compute 46-dimensional observation vector for agent_id."""
        self_obj = self.agent_objs[agent_id]
        r0 = self.map_preset.initial_radius
        v_max = self_obj.config.max_speed
        w_max = self_obj.config.max_angular_vel

        # 1. Self Kinematics (7)
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

        # 2. Nearest Active Opponent Relative (5)
        rival_id = self._find_nearest_active_rival(agent_id)
        if rival_id is not None:
            rival_obj = self.agent_objs[rival_id]
            dx = (rival_obj.pos[0] - self_obj.pos[0]) / (2.0 * r0)
            dy = (rival_obj.pos[1] - self_obj.pos[1]) / (2.0 * r0)
            dvx = (rival_obj.vel[0] - self_obj.vel[0]) / (2.0 * v_max)
            dvy = (rival_obj.vel[1] - self_obj.vel[1]) / (2.0 * v_max)
            rel_ang = (rival_obj.angle - self_obj.angle + np.pi) % (2.0 * np.pi) - np.pi
            norm_rel_ang = rel_ang / np.pi
            opp_rel = np.array([dx, dy, dvx, dvy, norm_rel_ang], dtype=np.float32)
        else:
            opp_rel = np.zeros(5, dtype=np.float32)

        # 3. Resources (2)
        resources = np.array(
            [
                self_obj.energy / self_obj.config.max_energy,
                self_obj.dash_cooldown / self_obj.config.dash_cooldown,
            ],
            dtype=np.float32,
        )

        # 4. 1D Lidar Array (32): 16 distance + 16 surface type
        other_alive_opponents = [
            (self.agent_objs[a].pos, self.agent_objs[a].config.radius)
            for a in self.possible_agents
            if a != agent_id and self.agent_objs[a].alive
        ]
        current_r = self.physics.compute_arena_radius(self.elapsed_time)
        lidar = self.raycast.get_multi_agent_lidar_array(
            origin=self_obj.pos,
            heading_angle=self_obj.angle,
            arena_radius=current_r,
            opponent_positions=other_alive_opponents,
            obstacles=self.map_preset.obstacles,
        )

        return np.concatenate([self_kin, opp_rel, resources, lidar]).astype(np.float32)


    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance simulation by dt."""
        self.elapsed_time += self.dt
        self.step_count += 1
        current_r = self.physics.compute_arena_radius(self.elapsed_time)

        # 1. Apply actions and kinematics for alive agents
        for a_id in list(self.agents):
            if not self.agent_objs[a_id].alive:
                continue
            act = np.asarray(actions.get(a_id, np.zeros(3, dtype=np.float32)), dtype=np.float32)
            self.last_actions[a_id] = act.copy()
            self.physics.apply_agent_controls(self.agent_objs[a_id], act)


        # 2. Pairwise collisions between alive agents
        self.recent_collisions = []
        alive_agents = [a for a in self.possible_agents if self.agent_objs[a].alive]
        for i in range(len(alive_agents)):
            for j in range(i + 1, len(alive_agents)):
                a_i = alive_agents[i]
                a_j = alive_agents[j]
                c_res = self.physics.resolve_agent_collision(self.agent_objs[a_i], self.agent_objs[a_j])
                if c_res.occurred:
                    self.recent_collisions.append(c_res)

        # 3. Obstacle collisions
        for a_id in alive_agents:
            agent = self.agent_objs[a_id]
            for obs in self.map_preset.obstacles:
                self.physics.resolve_obstacle_collision(agent, obs)

        # 4. Out of bounds check & eliminations
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}

        for a_id in alive_agents:
            if self.physics.check_out_of_bounds(self.agent_objs[a_id], current_r):
                self.eliminations.append(a_id)
                terminations[a_id] = True
                # Ring-out penalty
                rewards[a_id] = -1.0

        # Survival check
        still_alive = [a for a in self.possible_agents if self.agent_objs[a].alive]
        if len(still_alive) == 1:
            # 1 survivor wins!
            self.winner = still_alive[0]
            rewards[self.winner] = 3.0  # 1st place bonus
            for a in self.agents:
                terminations[a] = True
        elif len(still_alive) == 0:
            # All eliminated
            self.is_draw = True
            for a in self.agents:
                terminations[a] = True

        # Timeout truncation
        if self.elapsed_time >= self.max_duration and not any(terminations.values()):
            for a in self.agents:
                truncations[a] = True
            # Highest remaining distance to ring edge or tie
            self.is_draw = True

        # Filter active agents for next step
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        observations = {a: self._compute_observation(a) for a in self.possible_agents}
        infos = {
            a: {
                "alive": self.agent_objs[a].alive,
                "winner": self.winner,
                "eliminations": list(self.eliminations),
            }
            for a in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos

    def render(
        self,
        agent_names: tuple[str, str, str, str] | None = None,
        subtitle: str | None = None,
        fps: int | None = None,
        xray_info: dict[str, dict[str, Any]] | None = None,
        sim_status: str | None = None,
    ) -> np.ndarray | None:
        """Render frame using SumoRenderer."""
        if self._renderer is None:
            from src.ui.renderer import SumoRenderer
            self._renderer = SumoRenderer(arena_map=self.map_preset, mode=self.render_mode or "human")

        current_r = self.physics.compute_arena_radius(self.elapsed_time)
        kwargs: dict[str, Any] = {
            "agents": self.agent_objs,
            "arena_radius": current_r,
            "elapsed_time": self.elapsed_time,
            "recent_collisions": self.recent_collisions,
            "winner": self.winner,
            "is_draw": self.is_draw,
            "last_actions": self.last_actions,
            "xray_info": xray_info,
            "sim_status": sim_status,
        }
        if agent_names is not None:
            kwargs["agent_names"] = agent_names
        if subtitle is not None:
            kwargs["subtitle"] = subtitle
        if fps is not None:
            kwargs["fps"] = fps
        return self._renderer.render_4way_frame(**kwargs)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
