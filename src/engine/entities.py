"""Disc agent kinematics, physical properties, and bumper arc definition."""

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentConfig:
    """Physical and kinematic parameters for a sumo disc agent."""

    radius: float = 0.8
    mass: float = 10.0
    moment_of_inertia: float = 3.2  # 0.5 * m * r^2 = 0.5 * 10 * 0.64 = 3.2
    bumper_arc: float = 2.0 * np.pi / 3.0  # 120 degrees total ([-pi/3, +pi/3] from heading)
    max_thrust: float = 120.0
    max_brake: float = 60.0
    max_torque: float = 40.0
    max_speed: float = 15.0
    max_angular_vel: float = 8.0
    linear_drag: float = 0.8
    angular_drag: float = 1.2  # mu in tau_drag = -mu * omega * |omega|
    max_energy: float = 100.0
    energy_recharge: float = 25.0  # units per second
    dash_cost: float = 40.0  # 40% of max energy
    dash_duration: float = 0.25  # seconds
    dash_cooldown: float = 0.75  # seconds
    dash_multiplier: float = 2.0
    anchor_friction_mult: float = 4.0


class DiscAgent:
    """Hovering disc agent with linear momentum, rotational inertia, and bumper arcs."""

    def __init__(self, config: AgentConfig | None = None, initial_pos: np.ndarray | None = None, initial_angle: float = 0.0):
        self.config = config or AgentConfig()
        self.pos = np.zeros(2, dtype=np.float32) if initial_pos is None else np.asarray(initial_pos, dtype=np.float32).copy()
        self.vel = np.zeros(2, dtype=np.float32)
        self.angle = float(initial_angle)
        self.angular_vel = 0.0
        self.energy = float(self.config.max_energy)
        self.dash_timer = 0.0
        self.dash_cooldown = 0.0
        self.is_dashing = False
        self.is_anchored = False
        self.alive = True

    def reset(self, pos: np.ndarray, angle: float) -> None:
        """Reset agent to given state."""
        self.pos = np.asarray(pos, dtype=np.float32).copy()
        self.vel = np.zeros(2, dtype=np.float32)
        self.angle = float(angle)
        self.angular_vel = 0.0
        self.energy = float(self.config.max_energy)
        self.dash_timer = 0.0
        self.dash_cooldown = 0.0
        self.is_dashing = False
        self.is_anchored = False
        self.alive = True

    @property
    def heading_vector(self) -> np.ndarray:
        """Normalized unit vector pointing in the agent's facing direction."""
        return np.array([np.cos(self.angle), np.sin(self.angle)], dtype=np.float32)

    def is_in_bumper_arc(self, normal_from_agent: np.ndarray) -> bool:
        """Check if a direction vector from the agent lies within its front 120-degree bumper.

        Args:
            normal_from_agent: 2D vector pointing outward from agent center toward contact point.

        Returns:
            True if within +/- 60 degrees of the heading vector, False otherwise (flank/rear).
        """
        norm = np.linalg.norm(normal_from_agent)
        if norm < 1e-8:
            return True
        unit_contact = normal_from_agent / norm
        heading = self.heading_vector
        dot = np.clip(np.dot(heading, unit_contact), -1.0, 1.0)
        angle_diff = float(np.arccos(dot))
        half_arc = float(self.config.bumper_arc / 2.0)  # pi / 3 = 60 degrees
        return bool(angle_diff <= half_arc)

    def update_resources(self, dt: float, ability_cmd: float) -> None:
        """Update energy, cooldowns, and ability state triggers."""
        # Update cooldowns
        if self.dash_cooldown > 0.0:
            self.dash_cooldown = max(0.0, self.dash_cooldown - dt)

        # Update active dash
        if self.dash_timer > 0.0:
            self.dash_timer = max(0.0, self.dash_timer - dt)
            self.is_dashing = self.dash_timer > 0.0
        else:
            self.is_dashing = False

        # Recharge energy when not dashing
        if not self.is_dashing:
            self.energy = min(self.config.max_energy, self.energy + self.config.energy_recharge * dt)

        # Ability trigger logic:
        # ability > 0.5 triggers Dash (costs 40% energy, locked steering for 0.25s)
        # ability < -0.5 triggers Anchor (4x ground friction, absorbs hits, zero forward thrust)
        if ability_cmd > 0.5:
            self.is_anchored = False
            if (not self.is_dashing) and (self.dash_cooldown <= 0.0) and (self.energy >= self.config.dash_cost):
                self.energy -= self.config.dash_cost
                self.dash_timer = self.config.dash_duration
                self.dash_cooldown = self.config.dash_cooldown
                self.is_dashing = True
        elif ability_cmd < -0.5:
            self.is_anchored = True
        else:
            self.is_anchored = False

    def copy_state(self) -> dict:
        """Return a snapshot dictionary of the current state."""
        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "angle": self.angle,
            "angular_vel": self.angular_vel,
            "energy": self.energy,
            "dash_timer": self.dash_timer,
            "dash_cooldown": self.dash_cooldown,
            "is_dashing": self.is_dashing,
            "is_anchored": self.is_anchored,
            "alive": self.alive,
        }
