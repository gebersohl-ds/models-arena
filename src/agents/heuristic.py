"""Rule-based PD steering & state machine bot baseline for Kinetic Sumo."""

import numpy as np
from src.agents.base import BaseAgent


class HeuristicAgent(BaseAgent):
    """Deterministic state machine bot with PD steering tracking predicted target intercepts,

    perimeter avoidance, anchor bracing, and conditional dash execution.
    """

    def __init__(self, kp: float = 2.4, kd: float = 0.35):
        self.kp = float(kp)
        self.kd = float(kd)
        self.prev_angle_error = 0.0

    def reset(self) -> None:
        self.prev_angle_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Process 46-dim observation and compute [thrust, steer, ability] action."""
        obs = np.asarray(obs, dtype=np.float32)

        # Deconstruct observation
        # Self Kinematics (7)
        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        vel_x = obs[2] * 15.0
        vel_y = obs[3] * 15.0
        cos_th = obs[4]
        sin_th = obs[5]
        heading = np.arctan2(sin_th, cos_th)
        omega = obs[6] * 8.0

        # Opponent Relative (5)
        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0

        # Resources (2)
        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        # Distance calculations
        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. PERIMETER AVOIDANCE STATE (Emergency safety when near boundary)
        # Danger radius threshold: if within danger zone (> 6.8m from center)
        if dist_from_center > 6.8:
            # Steer directly back toward arena center (0, 0)
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)

            # If facing somewhat toward center, drive hard
            if abs(ang_err) < np.pi / 2.0:
                thrust = 1.0
                # Emergency dash back towards center if aligned
                ability = 1.0 if (abs(ang_err) < 0.25 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            else:
                thrust = 0.1  # Slow down to turn quickly
                ability = 0.0

            return np.array([thrust, steer, ability], dtype=np.float32)

        # 2. ANCHOR BRACING STATE (Defensive response)
        # If opponent is very close and closing in fast while self is facing away (risk of flank hit)
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi

        if dist_opp < 2.0 and abs(ang_err_opp) > np.pi / 3.0 and rel_speed_towards_self > 3.0:
            # Brace with Anchor to absorb incoming blow!
            self.prev_angle_error = float(ang_err_opp)
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # 3. INTERCEPT & OFFENSE STATE
        # Lead target prediction: estimate future position based on relative velocity
        t_lead = np.clip(dist_opp / 12.0, 0.05, 0.4)
        pred_target_x = dx + dvx * t_lead
        pred_target_y = dy + dvy * t_lead

        target_angle = np.arctan2(pred_target_y, pred_target_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

        # PD steering controller
        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
        self.prev_angle_error = float(ang_err)

        # Thrust and dash decision
        if abs(ang_err) < 0.35:
            # Aligned with intercept trajectory: full throttle
            thrust = 1.0
            # Conditional dash: execute dash if locked on within effective attack range
            if 1.5 < dist_opp < 6.5 and abs(ang_err) < 0.18 and energy >= 40.0 and dash_cooldown <= 0.0:
                ability = 1.0  # Dash burst!
            else:
                ability = 0.0
        elif abs(ang_err) < np.pi / 2.0:
            # Moderate alignment: moderate thrust
            thrust = 0.6
            ability = 0.0
        else:
            # Off target: brake/reverse slightly to rotate on axis
            thrust = -0.3
            ability = 0.0

        return np.array([thrust, steer, ability], dtype=np.float32)
