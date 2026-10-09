"""Tactical AI Agent Archetypes for Kinetic Sumo: Bulldozer, Shinobi, Guardian, and TunableBot."""

from dataclasses import dataclass
import numpy as np
from src.agents.base import BaseAgent


class BulldozerAgent(BaseAgent):
    """Hyper-aggressive rammer bot.

    Maintains continuous maximum forward throttle, aggressive closing velocity,
    triggers dash bursts early, and aims to forcefully eject opponents.
    """

    def __init__(self, kp: float = 2.8, kd: float = 0.25):
        self.kp = float(kp)
        self.kd = float(kd)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        # Deconstruct observation
        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        cos_th = obs[4]
        sin_th = obs[5]
        heading = np.arctan2(sin_th, cos_th)
        omega = obs[6] * 8.0

        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0

        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Extreme perimeter safety only (Bulldozer risks edge fights)
        if dist_from_center > 8.6:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            thrust = 1.0 if abs(ang_err) < np.pi / 2.0 else 0.2
            ability = 1.0 if (abs(ang_err) < 0.2 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            return np.array([thrust, steer, ability], dtype=np.float32)

        # 2. Ramming target tracking: lead target intercept
        t_lead = np.clip(dist_opp / 14.0, 0.05, 0.35)
        pred_x = dx + dvx * t_lead
        pred_y = dy + dvy * t_lead

        target_angle = np.arctan2(pred_y, pred_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)

        # Maximum throttle always
        thrust = 1.0 if abs(ang_err) < 0.6 else 0.4

        # Dash attack when lined up
        ability = 0.0
        if dist_opp < 5.5 and abs(ang_err) < 0.2 and energy >= 40.0 and dash_cooldown <= 0.0:
            ability = 1.0

        return np.array([thrust, steer, ability], dtype=np.float32)


class ShinobiAgent(BaseAgent):
    """Evasive flanker bot.

    Circles opponents to avoid head-on bumper collisions, seeking vulnerable
    flank and rear angles to deliver devastating 2.5x recoil-cancelling strikes.
    """

    def __init__(self, kp: float = 3.0, kd: float = 0.3):
        self.kp = float(kp)
        self.kd = float(kd)
        self.flank_side = 1.0  # +1 for CCW flanking, -1 for CW

    def reset(self) -> None:
        self.flank_side = float(np.random.choice([1.0, -1.0]))

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        cos_th = obs[4]
        sin_th = obs[5]
        heading = np.arctan2(sin_th, cos_th)
        omega = obs[6] * 8.0

        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0
        opp_rel_heading = obs[11] * np.pi  # opponent angle relative to self

        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Perimeter safety
        if dist_from_center > 7.0:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            return np.array([0.9, steer, 0.0], dtype=np.float32)

        # 2. Check if opponent is charging directly head-on at close range
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi

        if dist_opp < 2.5 and rel_speed_towards_self > 4.0 and abs(ang_err_opp) < 0.4:
            # Side dodge maneuver or anchor brace
            steer = float(np.clip(self.flank_side * 1.0, -1.0, 1.0))
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 3. Flanking target calculation
        # If far, close in with an offset angle. If near, strike the side/rear!
        opp_facing = heading + opp_rel_heading
        # Opponent rear vector
        rear_x = dx - np.cos(opp_facing) * 1.5
        rear_y = dy - np.sin(opp_facing) * 1.5

        if dist_opp > 3.5:
            # Circle orbit: aim at tangential offset around opponent
            tangent_angle = ang_to_opp + self.flank_side * (np.pi / 3.0)
            target_x = dx + np.cos(tangent_angle) * 2.0
            target_y = dy + np.sin(tangent_angle) * 2.0
        else:
            # Inside striking range: plunge towards opponent rear/flank
            target_x = rear_x
            target_y = rear_y

        target_angle = np.arctan2(target_y, target_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
        thrust = 1.0 if abs(ang_err) < 0.5 else 0.3

        # Strike with Dash when positioned behind or beside opponent
        ability = 0.0
        is_flank_angle = abs((ang_to_opp - opp_facing + np.pi) % (2.0 * np.pi) - np.pi) > (np.pi / 3.0)
        if dist_opp < 4.0 and abs(ang_err) < 0.25 and is_flank_angle and energy >= 40.0 and dash_cooldown <= 0.0:
            ability = 1.0  # Flank assassination dash!

        return np.array([thrust, steer, ability], dtype=np.float32)


class GuardianAgent(BaseAgent):
    """Center-control sumo guardian.

    Prioritizes ring-center positioning, uses Anchor to absorb oncoming rushes,
    and counter-pushes bounced opponents out of the arena.
    """

    def __init__(self, kp: float = 3.2, kd: float = 0.4):
        self.kp = float(kp)
        self.kd = float(kd)
        self.counter_push_timer = 0.0

    def reset(self) -> None:
        self.counter_push_timer = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        cos_th = obs[4]
        sin_th = obs[5]
        heading = np.arctan2(sin_th, cos_th)
        omega = obs[6] * 8.0

        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0

        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi

        # 1. Counter-push phase after absorbing a blow
        if self.counter_push_timer > 0.0:
            self.counter_push_timer -= 0.02
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            dash = 1.0 if (dist_opp < 3.0 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            return np.array([1.0, steer, dash], dtype=np.float32)

        # 2. Defensive Anchor Brace: incoming high speed impact
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        if dist_opp < 2.2 and rel_speed_towards_self > 2.5:
            # Face directly toward opponent to absorb on front bumper with Anchor
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            self.counter_push_timer = 0.4  # Trigger counter-push right after impact
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 3. Center Control: If displaced from center, return to center while facing opponent
        if dist_from_center > 2.2:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            thrust = 0.8 if abs(ang_err) < 0.6 else 0.2
            return np.array([thrust, steer, 0.0], dtype=np.float32)

        # 4. In center: Pivot in place to keep front bumper aimed at opponent
        steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
        thrust = 0.2 if abs(ang_err_opp) < 0.2 else 0.0
        return np.array([thrust, steer, 0.0], dtype=np.float32)


@dataclass
class BotConfig:
    """Configurable bot parameters for the Tunable Bot."""

    name: str = "Custom Bot"
    aggression: float = 0.7  # [0.0, 1.0]: low=defensive, high=relentless throttle
    dash_frequency: float = 0.6  # [0.0, 1.0]: threshold for dash triggering
    anchor_bias: float = 0.4  # [0.0, 1.0]: readiness to brace with anchor
    steer_speed: float = 2.5  # [1.0, 4.0]: proportional steering gain


class TunableAgent(BaseAgent):
    """Customizable bot whose combat traits can be adjusted dynamically."""

    def __init__(self, config: BotConfig | None = None):
        self.config = config or BotConfig()
        self.kp = self.config.steer_speed
        self.kd = 0.3

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        cos_th = obs[4]
        sin_th = obs[5]
        heading = np.arctan2(sin_th, cos_th)
        omega = obs[6] * 8.0

        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0

        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi

        # 1. Perimeter boundary safety (scaled inversely by aggression)
        danger_dist = 6.5 + 2.0 * (self.config.aggression)
        if dist_from_center > danger_dist:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            return np.array([0.9, steer, 0.0], dtype=np.float32)

        # 2. Anchor defense
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        anchor_threshold = 5.0 - 3.5 * self.config.anchor_bias
        if dist_opp < 2.2 and rel_speed_towards_self > anchor_threshold and self.config.anchor_bias > 0.1:
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 3. Intercept and steering
        steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
        thrust = float(np.clip(self.config.aggression * (1.0 if abs(ang_err_opp) < 0.5 else 0.3), -1.0, 1.0))

        # 4. Dash ability
        ability = 0.0
        dash_trigger_dist = 2.0 + 4.0 * self.config.dash_frequency
        if (
            dist_opp < dash_trigger_dist
            and abs(ang_err_opp) < 0.25
            and energy >= 40.0
            and dash_cooldown <= 0.0
            and self.config.dash_frequency > 0.1
        ):
            ability = 1.0

        return np.array([thrust, steer, ability], dtype=np.float32)
