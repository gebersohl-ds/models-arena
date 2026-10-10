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
        self.last_intent: str = "RAM CHARGE"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.last_intent = "RAM CHARGE"
        self.target_vec = None

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
            self.last_intent = "EDGE SAVE"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([thrust, steer, ability], dtype=np.float32)

        # 2. Ramming target tracking: lead target intercept
        t_lead = np.clip(dist_opp / 14.0, 0.05, 0.35)
        pred_x = dx + dvx * t_lead
        pred_y = dy + dvy * t_lead
        self.target_vec = np.array([pred_x, pred_y], dtype=np.float32)

        target_angle = np.arctan2(pred_y, pred_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)

        # Maximum throttle always
        thrust = 1.0 if abs(ang_err) < 0.6 else 0.4

        # Dash attack when lined up
        ability = 0.0
        if dist_opp < 5.5 and abs(ang_err) < 0.2 and energy >= 40.0 and dash_cooldown <= 0.0:
            ability = 1.0
            self.last_intent = "BULLDOZE DASH"
        else:
            self.last_intent = "RAM CHARGE"

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
        self.last_intent: str = "FLANK ORBIT"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.flank_side = float(np.random.choice([1.0, -1.0]))
        self.last_intent = "FLANK ORBIT"
        self.target_vec = None

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
            self.last_intent = "SHADOW RETREAT"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([0.9, steer, 0.0], dtype=np.float32)

        # 2. Check if opponent is charging directly head-on at close range
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi

        if dist_opp < 2.5 and rel_speed_towards_self > 4.0 and abs(ang_err_opp) < 0.4:
            # Side dodge maneuver or anchor brace
            steer = float(np.clip(self.flank_side * 1.0, -1.0, 1.0))
            self.last_intent = "EVADE BRACE"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
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
            self.last_intent = "FLANK ORBIT"
        else:
            # Inside striking range: plunge towards opponent rear/flank
            target_x = rear_x
            target_y = rear_y
            self.last_intent = "REAR TARGET"

        self.target_vec = np.array([target_x, target_y], dtype=np.float32)
        target_angle = np.arctan2(target_y, target_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi

        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
        thrust = 1.0 if abs(ang_err) < 0.5 else 0.3

        # Strike with Dash when positioned behind or beside opponent
        ability = 0.0
        is_flank_angle = abs((ang_to_opp - opp_facing + np.pi) % (2.0 * np.pi) - np.pi) > (np.pi / 3.0)
        if dist_opp < 4.0 and abs(ang_err) < 0.25 and is_flank_angle and energy >= 40.0 and dash_cooldown <= 0.0:
            ability = 1.0  # Flank assassination dash!
            self.last_intent = "BACKSTAB 2.5x"

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
        self.last_intent: str = "CENTER HOLD"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.counter_push_timer = 0.0
        self.last_intent = "CENTER HOLD"
        self.target_vec = None

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
            self.last_intent = "COUNTER PUSH"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([1.0, steer, dash], dtype=np.float32)

        # 2. Defensive Anchor Brace: incoming high speed impact
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        if dist_opp < 2.2 and rel_speed_towards_self > 2.5:
            # Face directly toward opponent to absorb on front bumper with Anchor
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            self.counter_push_timer = 0.4  # Trigger counter-push right after impact
            self.last_intent = "ANCHOR FORTRESS"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 3. Center Control: If displaced from center, return to center while facing opponent
        if dist_from_center > 2.2:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            thrust = 0.8 if abs(ang_err) < 0.6 else 0.2
            self.last_intent = "RETURN CENTER"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([thrust, steer, 0.0], dtype=np.float32)

        # 4. In center: Pivot in place to keep front bumper aimed at opponent
        steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
        thrust = 0.2 if abs(ang_err_opp) < 0.2 else 0.0
        self.last_intent = "CENTER SENTRY"
        self.target_vec = np.array([dx, dy], dtype=np.float32)
        return np.array([thrust, steer, 0.0], dtype=np.float32)


class MatadorAgent(BaseAgent):
    """Edge-baiter / Bullfighter archetype.

    Positions near the outer ring perimeter to bait aggressive opponents into charging,
    then executes a timed tangential juke/dash sidestep so the attacker's momentum
    carries them straight out of the ring.
    """

    def __init__(self, kp: float = 3.1, kd: float = 0.3):
        self.kp = float(kp)
        self.kd = float(kd)
        self.juke_dir = 1.0
        self.juke_timer = 0.0
        self.last_intent: str = "CAPE BAIT"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.juke_dir = 1.0
        self.juke_timer = 0.0
        self.last_intent = "CAPE BAIT"
        self.target_vec = None

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
        opp_x = pos_x + dx
        opp_y = pos_y + dy
        opp_dist_center = np.hypot(opp_x, opp_y)

        # 1. Critical boundary save if too close to void (> 7.9m)
        if dist_from_center > 7.9:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            thrust = 1.0 if abs(ang_err) < 0.7 else 0.3
            self.last_intent = "RING ESCAPE"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([thrust, steer, 0.0], dtype=np.float32)

        # 2. Detect opponent high-speed charge -> Execute tangential Juke Sidestep!
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        if self.juke_timer > 0.0:
            self.juke_timer -= 0.02
            # Tangent direction around arena center
            radial_ang = np.arctan2(pos_y, pos_x)
            juke_ang = radial_ang + self.juke_dir * (np.pi / 2.0 + 0.35)
            ang_err = (juke_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            ability = 1.0 if (abs(ang_err) < 0.45 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            self.last_intent = "OLE! SIDESTEP"
            self.target_vec = np.array([np.cos(juke_ang) * 3.5, np.sin(juke_ang) * 3.5], dtype=np.float32)
            return np.array([1.0, steer, ability], dtype=np.float32)

        if dist_opp < 3.6 and rel_speed_towards_self > 3.2:
            # Choose tangent side away from opponent approach line
            cross = dx * dvy - dy * dvx
            self.juke_dir = 1.0 if cross >= 0.0 else -1.0
            self.juke_timer = 0.36
            radial_ang = np.arctan2(pos_y, pos_x)
            juke_ang = radial_ang + self.juke_dir * (np.pi / 2.0 + 0.35)
            ang_err = (juke_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            ability = 1.0 if (abs(ang_err) < 0.45 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            self.last_intent = "OLE! SIDESTEP"
            self.target_vec = np.array([np.cos(juke_ang) * 3.5, np.sin(juke_ang) * 3.5], dtype=np.float32)
            return np.array([1.0, steer, ability], dtype=np.float32)

        # 3. If opponent is stranded near the edge, lunge in for the finish!
        if opp_dist_center > 5.8 and opp_dist_center > dist_from_center + 0.6:
            ang_to_opp = np.arctan2(dy, dx)
            ang_err = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            ability = 1.0 if (dist_opp < 4.5 and abs(ang_err) < 0.22 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            self.last_intent = "BULLFIGHT FINISH"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([1.0, steer, ability], dtype=np.float32)

        # 4. Patrol the "Bait Ring" (~5.2m radius) on a tangent ready to sidestep
        radial_ang = np.arctan2(pos_y, pos_x) if dist_from_center > 0.3 else np.arctan2(-dy, -dx)
        bait_x = np.cos(radial_ang) * 5.2 - pos_x
        bait_y = np.sin(radial_ang) * 5.2 - pos_y
        if np.hypot(bait_x, bait_y) > 1.4:
            target_angle = np.arctan2(bait_y, bait_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            self.last_intent = "BAIT POSITION"
            self.target_vec = np.array([bait_x, bait_y], dtype=np.float32)
            return np.array([0.75, steer, 0.0], dtype=np.float32)

        # Hold tangential stance so we can dash sideways instantaneously
        tangent_ang = radial_ang + self.juke_dir * (np.pi / 2.0)
        ang_err = (tangent_ang - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
        self.last_intent = "CAPE TAUNT"
        self.target_vec = np.array([dx, dy], dtype=np.float32)
        return np.array([0.25, steer, 0.0], dtype=np.float32)


class RicochetAgent(BaseAgent):
    """Lidar-aware hazard & obstacle specialist.

    Actively parses the 16-ray Lidar array (obs[14:46]) to weave around arena pillars
    without getting pinned, while angling strikes to herd opponents into 2.0x bumpers
    and ring boundaries.
    """

    def __init__(self, kp: float = 3.0, kd: float = 0.32):
        self.kp = float(kp)
        self.kd = float(kd)
        self.last_intent: str = "LIDAR SCAN"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.last_intent = "LIDAR SCAN"
        self.target_vec = None

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

        # Parse 16-Ray Lidar: obs[14:30] distances in [0, 1], obs[30:46] surface types
        lidar_dists = obs[14:30] * 25.0  # max_range = 25m
        lidar_types = obs[30:46]

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)

        # 1. Perimeter safety
        if dist_from_center > 7.2:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            self.last_intent = "VOID RETREAT"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([0.95, steer, 0.0], dtype=np.float32)

        # 2. Compute Lidar obstacle repulsive & tangential weave vector
        avoid_x = 0.0
        avoid_y = 0.0
        obstacle_nearby = False
        for i in range(16):
            if np.isclose(lidar_types[i], 0.5, atol=0.15) and lidar_dists[i] < 3.6:
                obstacle_nearby = True
                ray_ang = heading + i * (2.0 * np.pi / 16.0)
                weight = (3.6 - float(lidar_dists[i])) / 3.6
                # Repel away from obstacle + slight tangential swirl
                repel_ang = ray_ang + np.pi * 0.75
                avoid_x += np.cos(repel_ang) * weight * 3.5
                avoid_y += np.sin(repel_ang) * weight * 3.5

        # 3. Combine lead intercept with Lidar obstacle weave
        t_lead = np.clip(dist_opp / 13.0, 0.05, 0.35)
        aim_x = dx + dvx * t_lead + avoid_x
        aim_y = dy + dvy * t_lead + avoid_y
        self.target_vec = np.array([aim_x, aim_y], dtype=np.float32)

        target_angle = np.arctan2(aim_y, aim_x)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)

        # Check if front ray (ray 0) sees an obstacle directly ahead before dashing!
        front_blocked = np.isclose(lidar_types[0], 0.5, atol=0.15) and lidar_dists[0] < 3.5

        thrust = 1.0 if abs(ang_err) < 0.55 else 0.45
        ability = 0.0
        if not front_blocked and dist_opp < 5.0 and abs(ang_err) < 0.22 and energy >= 40.0 and dash_cooldown <= 0.0:
            ability = 1.0
            self.last_intent = "RICOCHET STRIKE"
        elif obstacle_nearby:
            self.last_intent = "LIDAR WEAVE"
        else:
            self.last_intent = "BUMPER HERD"

        return np.array([thrust, steer, ability], dtype=np.float32)


class ChameleonAgent(BaseAgent):
    """Adaptive Counter-Stance Fighter.

    Dynamically switches between 3 distinct combat stances in real time:
    - [GUARDIAN]: When opponent charges at high closing speed, braces with Anchor and counter-pushes.
    - [BULLDOZER]: When opponent is closer to the outer ring edge, goes full-throttle ram + Dash.
    - [SHINOBI]: In neutral mid-ring states, orbits outside the opponent's 120° bumper for 2.5x flank hits.
    """

    def __init__(self, kp: float = 3.1, kd: float = 0.32):
        self.kp = float(kp)
        self.kd = float(kd)
        self.counter_timer = 0.0
        self.orbit_dir = 1.0
        self.last_intent: str = "STANCE: SHINOBI"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.counter_timer = 0.0
        self.orbit_dir = 1.0
        self.last_intent = "STANCE: SHINOBI"
        self.target_vec = None

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
        opp_rel_heading = obs[11] * np.pi

        energy = obs[12] * 100.0
        dash_cooldown = obs[13] * 0.75

        dist_from_center = np.hypot(pos_x, pos_y)
        dist_opp = np.hypot(dx, dy)
        opp_dist_center = np.hypot(pos_x + dx, pos_y + dy)
        ang_to_opp = np.arctan2(dy, dx)
        ang_err_opp = (ang_to_opp - heading + np.pi) % (2.0 * np.pi) - np.pi
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)

        # 0. Emergency boundary recovery
        if dist_from_center > 7.4:
            target_angle = np.arctan2(-pos_y, -pos_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            self.last_intent = "ADAPT: EDGE SAVE"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([0.95, steer, 0.0], dtype=np.float32)

        # 1. [GUARDIAN STANCE]: Incoming charge defense & counter-push
        if self.counter_timer > 0.0:
            self.counter_timer -= 0.02
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            ability = 1.0 if (dist_opp < 3.4 and abs(ang_err_opp) < 0.25 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            self.last_intent = "STANCE: COUNTER"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([1.0, steer, ability], dtype=np.float32)

        if dist_opp < 2.5 and rel_speed_towards_self > 2.8:
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            self.counter_timer = 0.36
            self.last_intent = "STANCE: GUARDIAN"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 2. [BULLDOZER STANCE]: Opponent is near boundary or vulnerable -> Finish them!
        if opp_dist_center > 5.5 and opp_dist_center > dist_from_center + 0.5:
            t_lead = np.clip(dist_opp / 14.0, 0.05, 0.3)
            pred_x = dx + dvx * t_lead
            pred_y = dy + dvy * t_lead
            self.target_vec = np.array([pred_x, pred_y], dtype=np.float32)
            target_angle = np.arctan2(pred_y, pred_x)
            ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
            steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
            ability = 1.0 if (dist_opp < 5.0 and abs(ang_err) < 0.22 and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
            self.last_intent = "STANCE: BULLDOZER"
            return np.array([1.0, steer, ability], dtype=np.float32)

        # 3. [SHINOBI STANCE]: Neutral mid-ring -> Flank outside opponent's 120° bumper
        opp_facing = heading + opp_rel_heading
        if dist_opp > 3.2:
            tangent_angle = ang_to_opp + self.orbit_dir * (np.pi / 3.2)
            tx = dx + np.cos(tangent_angle) * 1.8
            ty = dy + np.sin(tangent_angle) * 1.8
        else:
            tx = dx - np.cos(opp_facing) * 1.4
            ty = dy - np.sin(opp_facing) * 1.4

        self.target_vec = np.array([tx, ty], dtype=np.float32)
        target_angle = np.arctan2(ty, tx)
        ang_err = (target_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
        steer = np.clip(self.kp * ang_err - self.kd * omega, -1.0, 1.0)
        thrust = 1.0 if abs(ang_err) < 0.55 else 0.35

        is_flank = abs((ang_to_opp - opp_facing + np.pi) % (2.0 * np.pi) - np.pi) > (np.pi / 3.0)
        ability = 1.0 if (dist_opp < 3.8 and abs(ang_err) < 0.24 and is_flank and energy >= 40.0 and dash_cooldown <= 0.0) else 0.0
        self.last_intent = "STANCE: SHINOBI"
        return np.array([thrust, steer, ability], dtype=np.float32)


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
        self.last_intent: str = "TUNABLE TRACK"
        self.target_vec: np.ndarray | None = None

    def reset(self) -> None:
        self.last_intent = "TUNABLE TRACK"
        self.target_vec = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        self.kp = self.config.steer_speed

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
            self.last_intent = "TUNABLE RETREAT"
            self.target_vec = np.array([-pos_x, -pos_y], dtype=np.float32)
            return np.array([0.9, steer, 0.0], dtype=np.float32)

        # 2. Anchor defense
        rel_speed_towards_self = -(dx * dvx + dy * dvy) / max(1.0, dist_opp)
        anchor_threshold = 5.0 - 3.5 * self.config.anchor_bias
        if dist_opp < 2.2 and rel_speed_towards_self > anchor_threshold and self.config.anchor_bias > 0.1:
            steer = np.clip(self.kp * ang_err_opp - self.kd * omega, -1.0, 1.0)
            self.last_intent = "TUNABLE ANCHOR"
            self.target_vec = np.array([dx, dy], dtype=np.float32)
            return np.array([0.0, steer, -1.0], dtype=np.float32)

        # 3. Intercept and steering
        self.target_vec = np.array([dx, dy], dtype=np.float32)
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
            self.last_intent = "TUNABLE DASH"
        else:
            self.last_intent = f"TUNABLE (AGG {self.config.aggression:.1f})"

        return np.array([thrust, steer, ability], dtype=np.float32)
