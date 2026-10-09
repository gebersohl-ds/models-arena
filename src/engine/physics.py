"""Pure NumPy 2D physics engine: elastic collisions, drag, boundaries, and bumper mechanics."""

from dataclasses import dataclass
import numpy as np
from src.engine.entities import DiscAgent, AgentConfig


@dataclass
class CollisionResult:
    """Detailed result of a collision between two agents or an agent and an obstacle."""

    occurred: bool = False
    impulse_a: float = 0.0
    impulse_b: float = 0.0
    flank_hit_on_a: bool = False
    flank_hit_on_b: bool = False
    contact_pos: np.ndarray | None = None
    contact_normal: np.ndarray | None = None


@dataclass
class Obstacle:
    """Circular arena obstacle or bumper."""

    x: float
    y: float
    radius: float
    restitution: float = 1.0
    is_bumper: bool = True

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class PhysicsEngine:
    """Headless 2D physics engine handling kinematics, collisions, and arena shrink."""

    def __init__(
        self,
        initial_radius: float = 10.0,
        min_radius: float = 4.0,
        shrink_start_time: float = 20.0,
        round_duration: float = 45.0,
        dt: float = 0.02,
        restitution: float = 0.8,
        flank_impulse_multiplier: float = 2.5,
        surface_friction: float = 1.0,
        shrink_mode: str = "continuous",
        step_interval: float = 5.0,
        step_drop_amount: float = 1.5,
    ):
        self.initial_radius = float(initial_radius)
        self.min_radius = float(min_radius)
        self.shrink_start_time = float(shrink_start_time)
        self.round_duration = float(round_duration)
        self.dt = float(dt)
        self.restitution = float(restitution)
        self.flank_impulse_multiplier = float(flank_impulse_multiplier)
        self.surface_friction = float(surface_friction)
        self.shrink_mode = str(shrink_mode)
        self.step_interval = float(step_interval)
        self.step_drop_amount = float(step_drop_amount)

    def compute_arena_radius(self, match_time: float) -> float:
        """Calculate dynamic outer ring radius based on match elapsed time."""
        if match_time <= self.shrink_start_time:
            return self.initial_radius

        if self.shrink_mode == "stepped":
            elapsed_shrink = match_time - self.shrink_start_time
            num_drops = int(elapsed_shrink // self.step_interval) + 1
            dropped_r = self.initial_radius - num_drops * self.step_drop_amount
            return float(max(self.min_radius, dropped_r))

        shrink_window = max(1e-5, self.round_duration - self.shrink_start_time)
        progress = np.clip((match_time - self.shrink_start_time) / shrink_window, 0.0, 1.0)
        return float(self.initial_radius - progress * (self.initial_radius - self.min_radius))

    def apply_agent_controls(self, agent: DiscAgent, action: np.ndarray) -> None:
        """Apply continuous action [thrust, steer, ability] to an agent.

        thrust: Forward drive [0, 1] or reverse brake [-1, -0.3]
        steer: Rotational torque mapped to angular velocity
        ability: > 0.5 triggers Dash; < -0.5 triggers Anchor
        """
        thrust_cmd = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))
        ability_cmd = float(np.clip(action[2], -1.0, 1.0))

        # Update energy and ability triggers
        agent.update_resources(self.dt, ability_cmd)

        cfg = agent.config
        heading = agent.heading_vector

        # 1. Linear forces
        linear_force = np.zeros(2, dtype=np.float32)
        if agent.is_anchored:
            # Anchor absorbs movement: zero forward thrust
            linear_force = np.zeros(2, dtype=np.float32)
        elif agent.is_dashing:
            # Dash gives 2x forward burst thrust
            linear_force = heading * (cfg.max_thrust * cfg.dash_multiplier)
        else:
            if thrust_cmd > 0.0:
                linear_force = heading * (thrust_cmd * cfg.max_thrust)
            elif thrust_cmd <= -0.3:
                # Reverse brake [-1, -0.3] scaled from 0 at -0.3 to full at -1.0
                brake_scale = float(np.clip((-thrust_cmd - 0.3) / 0.7, 0.0, 1.0))
                linear_force = -heading * (brake_scale * cfg.max_brake)
            else:
                # Neutral coasting deadzone in (-0.3, 0.0]
                linear_force = np.zeros(2, dtype=np.float32)

        # Linear drag force (anchored multiplies friction by 4x, ice surface reduces drag)
        drag_coeff = cfg.linear_drag * (cfg.anchor_friction_mult if agent.is_anchored else self.surface_friction)
        drag_force = -drag_coeff * agent.vel
        net_force = linear_force + drag_force

        # Acceleration and velocity update
        accel = net_force / cfg.mass
        agent.vel += accel * self.dt

        # Cap max speed
        speed = float(np.linalg.norm(agent.vel))
        if speed > cfg.max_speed:
            agent.vel = (agent.vel / speed) * cfg.max_speed

        # 2. Rotational torque & damping
        # When dashing, steering is locked for 0.25s
        if agent.is_dashing:
            steer_torque = 0.0
        else:
            steer_torque = steer_cmd * cfg.max_torque

        # Non-linear rotational damping: tau_drag = -mu * omega * |omega|
        ang_drag_mu = cfg.angular_drag * (cfg.anchor_friction_mult if agent.is_anchored else self.surface_friction)
        tau_drag = -ang_drag_mu * agent.angular_vel * abs(agent.angular_vel)
        net_torque = steer_torque + tau_drag

        # Angular acceleration and velocity update
        ang_accel = net_torque / cfg.moment_of_inertia
        agent.angular_vel += ang_accel * self.dt

        # Numerical stability clamp for angular velocity
        if abs(agent.angular_vel) > cfg.max_angular_vel:
            agent.angular_vel = float(np.sign(agent.angular_vel) * cfg.max_angular_vel)

        # 3. Position and heading integration
        agent.pos += agent.vel * self.dt
        agent.angle += agent.angular_vel * self.dt

        # Keep angle in [-pi, pi]
        agent.angle = float((agent.angle + np.pi) % (2.0 * np.pi) - np.pi)

    def resolve_agent_collision(self, agent_a: DiscAgent, agent_b: DiscAgent) -> CollisionResult:
        """Resolve combat collision between two disc agents with front bumper and flank mechanics."""
        delta = agent_b.pos - agent_a.pos
        dist = float(np.linalg.norm(delta))
        min_dist = agent_a.config.radius + agent_b.config.radius

        if dist >= min_dist or dist < 1e-8:
            return CollisionResult(occurred=False)

        # Contact normal pointing from A to B
        normal = delta / dist
        penetration = min_dist - dist

        # Positional resolution (separate discs so they don't overlap)
        if agent_a.is_anchored and not agent_b.is_anchored:
            agent_b.pos += normal * penetration
        elif agent_b.is_anchored and not agent_a.is_anchored:
            agent_a.pos -= normal * penetration
        else:
            agent_a.pos -= normal * (penetration * 0.5)
            agent_b.pos += normal * (penetration * 0.5)

        # Relative velocity along normal
        rel_vel = agent_b.vel - agent_a.vel
        v_rel_normal = float(np.dot(rel_vel, normal))

        # Check bumper alignments:
        # For A, contact is in direction +normal
        # For B, contact is in direction -normal
        bumper_a = agent_a.is_in_bumper_arc(normal)
        bumper_b = agent_b.is_in_bumper_arc(-normal)

        # If separating already, no impulse needed
        if v_rel_normal >= 0:
            return CollisionResult(
                occurred=True,
                impulse_a=0.0,
                impulse_b=0.0,
                flank_hit_on_a=not bumper_a,
                flank_hit_on_b=not bumper_b,
                contact_pos=agent_a.pos + normal * agent_a.config.radius,
                contact_normal=normal,
            )

        # Base 1D elastic impulse
        m_a = agent_a.config.mass * (agent_a.config.anchor_friction_mult if agent_a.is_anchored else 1.0)
        m_b = agent_b.config.mass * (agent_b.config.anchor_friction_mult if agent_b.is_anchored else 1.0)
        e = self.restitution
        j_base = -(1.0 + e) * v_rel_normal / (1.0 / m_a + 1.0 / m_b)

        impulse_a = j_base
        impulse_b = j_base
        flank_hit_on_a = False
        flank_hit_on_b = False

        # Flank / Rear hit combat mechanics:
        # If A attacks with front bumper into B's flank/rear:
        # B takes 2.5x impulse, A's recoil is cancelled.
        if bumper_a and not bumper_b:
            flank_hit_on_b = True
            impulse_b = j_base * self.flank_impulse_multiplier
            impulse_a = 0.0  # attacker recoil cancelled!
        # If B attacks with front bumper into A's flank/rear:
        # A takes 2.5x impulse, B's recoil is cancelled.
        elif bumper_b and not bumper_a:
            flank_hit_on_a = True
            impulse_a = j_base * self.flank_impulse_multiplier
            impulse_b = 0.0  # attacker recoil cancelled!
        # Otherwise: bumper-to-bumper or flank-to-flank: symmetric standard recoil
        else:
            impulse_a = j_base
            impulse_b = j_base

        # Anchor absorption: if target is anchored, reduce applied impulse
        if agent_a.is_anchored and impulse_a > 0:
            impulse_a *= 0.5
        if agent_b.is_anchored and impulse_b > 0:
            impulse_b *= 0.5

        # Apply velocity changes
        agent_a.vel -= (impulse_a / agent_a.config.mass) * normal
        agent_b.vel += (impulse_b / agent_b.config.mass) * normal

        contact_point = agent_a.pos + normal * agent_a.config.radius

        return CollisionResult(
            occurred=True,
            impulse_a=float(impulse_a),
            impulse_b=float(impulse_b),
            flank_hit_on_a=flank_hit_on_a,
            flank_hit_on_b=flank_hit_on_b,
            contact_pos=contact_point,
            contact_normal=normal,
        )

    def resolve_obstacle_collision(self, agent: DiscAgent, obstacle: Obstacle) -> bool:
        """Resolve collision between an agent and a static obstacle."""
        delta = agent.pos - obstacle.pos
        dist = float(np.linalg.norm(delta))
        min_dist = agent.config.radius + obstacle.radius

        if dist >= min_dist or dist < 1e-8:
            return False

        normal = delta / dist
        penetration = min_dist - dist
        agent.pos += normal * penetration

        # Reflect velocity along normal
        v_dot_n = float(np.dot(agent.vel, normal))
        if v_dot_n < 0:
            agent.vel -= (1.0 + obstacle.restitution) * v_dot_n * normal
        return True

    def check_out_of_bounds(self, agent: DiscAgent, current_arena_radius: float) -> bool:
        """Check if an agent has fallen off the arena ring (instant death)."""
        dist_from_center = float(np.linalg.norm(agent.pos))
        if dist_from_center > current_arena_radius:
            agent.alive = False
            return True
        return False
