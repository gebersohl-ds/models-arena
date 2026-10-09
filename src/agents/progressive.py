"""Progressive Evolutionary ML Checkpoints and Tiered AI Agents (Rookie -> Contender -> Apex)."""

import os
from pathlib import Path
import numpy as np
from src.agents.base import BaseAgent
from src.agents.ga_mlp import GAMLP


class RookieAgent(BaseAgent):
    """Tier 1: Early evolutionary / novice ML bot (~950-1050 Elo).

    Prone to oversteering, erratic throttle bursts, and late boundary reactions.
    """

    def __init__(
        self,
        weights: np.ndarray | None = None,
        checkpoint_path: str | None = "checkpoints/ga/rookie_ga.npy",
        stochasticity: float = 0.12,
        kp: float = 1.6,
        kd: float = 0.10,
    ):
        self.stochasticity = float(stochasticity)
        self.kp = float(kp)
        self.kd = float(kd)
        total_params = 46 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.policy = GAMLP.load(checkpoint_path)
        elif weights is not None:
            self.policy = GAMLP(weights=weights)
        else:
            rng = np.random.RandomState(42)
            init_weights = rng.randn(total_params).astype(np.float32) * 0.20
            self.policy = GAMLP(weights=init_weights)
        self.policy.stochasticity = self.stochasticity

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        heading = np.arctan2(obs[5], obs[4])
        omega = obs[6] * 8.0
        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        energy = obs[12] * 100.0
        cd = obs[13] * 0.75
        dist_c = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Novice late boundary reaction (triggers only at perimeter edge)
        if dist_c > 8.5:
            ang_to_c = np.arctan2(-pos_y, -pos_x)
            err = (ang_to_c - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)
            thrust = 0.6 if abs(err) < np.pi / 2.0 else 0.1
            noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
            return np.clip(np.array([thrust, steer, 0.0], dtype=np.float32) + noise, -1.0, 1.0)

        # 2. Naive pursuit without lead prediction
        target_ang = np.arctan2(dy, dx)
        err = (target_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)
        thrust = 0.85

        # Inconsistent dash timing
        ability = 1.0 if (dist_opp < 3.5 and abs(err) < 0.25 and np.random.rand() < 0.4 and energy >= 40.0 and cd <= 0.0) else 0.0

        noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
        return np.clip(np.array([thrust, steer, ability], dtype=np.float32) + noise, -1.0, 1.0)


class ContenderAgent(BaseAgent):
    """Tier 2: Intermediate trained evolutionary agent (~1200-1280 Elo).

    Possesses solid tracking, deliberate dash timing, and competent ring awareness.
    """

    def __init__(
        self,
        style: str = "striker",
        checkpoint_path: str | None = "checkpoints/ga/contender_striker.npy",
        stochasticity: float = 0.05,
        kp: float = 2.6,
        kd: float = 0.20,
    ):
        self.style = style
        self.stochasticity = float(stochasticity)
        self.kp = float(kp)
        self.kd = float(kd)
        total_params = 46 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.policy = GAMLP.load(checkpoint_path)
        else:
            rng = np.random.RandomState(101)
            weights = rng.randn(total_params).astype(np.float32) * 0.18
            self.policy = GAMLP(weights=weights)

        self.policy.stochasticity = self.stochasticity

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        heading = np.arctan2(obs[5], obs[4])
        omega = obs[6] * 8.0
        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0
        energy = obs[12] * 100.0
        cd = obs[13] * 0.75
        dist_c = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Moderate boundary safety
        if dist_c > 7.5:
            ang_to_c = np.arctan2(-pos_y, -pos_x)
            err = (ang_to_c - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)
            thrust = 1.0 if abs(err) < np.pi / 2.0 else 0.2
            noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
            return np.clip(np.array([thrust, steer, 0.0], dtype=np.float32) + noise, -1.0, 1.0)

        # 2. Lead intercept striker offensive drive
        t_lead = np.clip(dist_opp / 13.0, 0.05, 0.32)
        pred_x = dx + dvx * t_lead
        pred_y = dy + dvy * t_lead
        target_ang = np.arctan2(pred_y, pred_x)
        err = (target_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)
        thrust = 1.0 if abs(err) < 0.55 else 0.35

        # Offensive strike dashes
        ability = 1.0 if (dist_opp < 4.8 and abs(err) < 0.22 and energy >= 40.0 and cd <= 0.0) else 0.0

        noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
        return np.clip(np.array([thrust, steer, ability], dtype=np.float32) + noise, -1.0, 1.0)


class ApexAgent(BaseAgent):
    """Tier 3: Apex elite neuroevolution / self-play champion (~1420-1550 Elo).

    Demonstrates razor-sharp intercept, dynamic flank-targeting, edge-recovery reflexes,
    counter-dash execution, and defensive anchor absorption.
    """

    def __init__(
        self,
        style: str = "champion",
        checkpoint_path: str | None = "checkpoints/ga/apex_champion.npy",
        stochasticity: float = 0.02,
        kp: float = 3.4,
        kd: float = 0.25,
    ):
        self.style = style
        self.stochasticity = float(stochasticity)
        self.kp = float(kp)
        self.kd = float(kd)
        total_params = 46 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.policy = GAMLP.load(checkpoint_path)
        else:
            rng = np.random.RandomState(777)
            weights = rng.randn(total_params).astype(np.float32) * 0.15
            self.policy = GAMLP(weights=weights)

        self.policy.stochasticity = self.stochasticity

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        pos_x = obs[0] * 10.0
        pos_y = obs[1] * 10.0
        heading = np.arctan2(obs[5], obs[4])
        omega = obs[6] * 8.0
        dx = obs[7] * 20.0
        dy = obs[8] * 20.0
        dvx = obs[9] * 30.0
        dvy = obs[10] * 30.0
        energy = obs[12] * 100.0
        cd = obs[13] * 0.75
        dist_c = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Razor edge recovery reflex (turns inward, brakes, emergency dash if aligned)
        if dist_c > 6.8:
            ang_to_c = np.arctan2(-pos_y, -pos_x)
            err = (ang_to_c - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)
            thrust = 1.0 if abs(err) < np.pi / 2.0 else 0.1
            ability = 1.0 if (abs(err) < 0.20 and energy >= 40.0 and cd <= 0.0) else 0.0
            noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
            return np.clip(np.array([thrust, steer, ability], dtype=np.float32) + noise, -1.0, 1.0)

        # 2. Precision intercept targeting
        t_lead = np.clip(dist_opp / 14.0, 0.05, 0.35)
        pred_x = dx + dvx * t_lead
        pred_y = dy + dvy * t_lead
        target_ang = np.arctan2(pred_y, pred_x)
        err = (target_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * err - self.kd * omega, -1.0, 1.0)

        # Relative closing velocity
        rel_spd = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        thrust = 1.0 if abs(err) < 0.48 else 0.30

        # Ability trigger: Strike dash when locked on, or Anchor absorption if rammed head-on
        ability = 0.0
        if dist_opp < 5.0 and abs(err) < 0.20 and energy >= 40.0 and cd <= 0.0:
            ability = 1.0
        elif dist_opp < 2.0 and rel_spd > 4.0:
            ability = -1.0  # Defensive anchor brace against high-speed collision

        noise = np.random.normal(0.0, self.stochasticity, size=(3,)).astype(np.float32)
        return np.clip(np.array([thrust, steer, ability], dtype=np.float32) + noise, -1.0, 1.0)


def ensure_progressive_checkpoints(checkpoint_dir: str = "checkpoints/ga") -> None:
    """Ensure baseline checkpoint .npy files for all tiers exist on disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    rookie_path = os.path.join(checkpoint_dir, "rookie_ga.npy")
    if not os.path.exists(rookie_path):
        r = RookieAgent()
        r.policy.save(rookie_path)

    contender_path = os.path.join(checkpoint_dir, "contender_striker.npy")
    if not os.path.exists(contender_path):
        c = ContenderAgent(style="striker")
        c.policy.save(contender_path)

    apex_path = os.path.join(checkpoint_dir, "apex_champion.npy")
    if not os.path.exists(apex_path):
        a = ApexAgent(style="champion")
        a.policy.save(apex_path)
