"""Potential-Based Reward Shaping (PBRS), kinetic contact exchange, and anti-spin penalties."""

import numpy as np
from src.engine.entities import DiscAgent
from src.engine.physics import CollisionResult


class RewardCalculator:
    """Computes dense and sparse rewards according to Kinetic Sumo specification."""

    def __init__(self, gamma: float = 0.99, max_speed: float = 15.0):
        self.gamma = float(gamma)
        self.max_speed = float(max_speed)
        # Cache previous potentials and previous actions per agent
        self.prev_potentials: dict[str, float] = {}
        self.prev_actions: dict[str, np.ndarray] = {}

    def reset(self, agents: dict[str, DiscAgent], arena_radius: float) -> None:
        """Reset cached potentials and actions at the start of an episode."""
        self.prev_potentials.clear()
        self.prev_actions.clear()
        for agent_id, agent in agents.items():
            opp_id = "agent_1" if agent_id == "agent_0" else "agent_0"
            opp = agents[opp_id]
            self.prev_potentials[agent_id] = self.compute_potential(agent, opp, arena_radius)
            self.prev_actions[agent_id] = np.zeros(3, dtype=np.float32)

    def compute_potential(self, agent: DiscAgent, opponent: DiscAgent, current_arena_radius: float) -> float:
        """Calculate Phi(s) = 0.4 * Center_Safety + 0.6 * Closing_Velocity_Vector.

        Center_Safety: 1.0 - (||pos|| / R(t)) clamped to [0, 1].
        Closing_Velocity_Vector: projection of (v_self - v_opp) onto direction towards opponent / V_max.
        """
        # Center safety
        dist_from_center = float(np.linalg.norm(agent.pos))
        center_safety = np.clip(1.0 - dist_from_center / max(1e-4, current_arena_radius), 0.0, 1.0)

        # Closing velocity vector
        delta_p = opponent.pos - agent.pos
        dist_opp = float(np.linalg.norm(delta_p))
        if dist_opp > 1e-6:
            unit_to_opp = delta_p / dist_opp
            rel_vel = agent.vel - opponent.vel
            closing_speed = float(np.dot(rel_vel, unit_to_opp))
            closing_vel_norm = np.clip(closing_speed / self.max_speed, -1.0, 1.0)
        else:
            closing_vel_norm = 0.0

        return float(0.4 * center_safety + 0.6 * closing_vel_norm)

    def compute_step_reward(
        self,
        agent_id: str,
        agent: DiscAgent,
        opponent: DiscAgent,
        action: np.ndarray,
        collision: CollisionResult,
        current_arena_radius: float,
        is_terminal: bool,
        winner_id: str | None,
        is_draw: bool,
    ) -> float:
        """Compute full step reward:

        R_t = R_terminal + (gamma * Phi(s_{t+1}) - Phi(s_t)) + R_contact - R_penalties
        """
        # 1. Terminal reward
        r_terminal = 0.0
        if is_terminal:
            if is_draw:
                r_terminal = -1.0  # Cold-war timeout draw penalty
            elif winner_id == agent_id:
                r_terminal = 1.0
            else:
                r_terminal = -1.0

        # 2. PBRS Potential difference: gamma * Phi(s') - Phi(s)
        phi_prev = self.prev_potentials.get(agent_id, 0.0)
        if is_terminal:
            # By standard PBRS theory, terminal state potential is 0
            phi_curr = 0.0
        else:
            phi_curr = self.compute_potential(agent, opponent, current_arena_radius)
        r_pbrs = (self.gamma * phi_curr) - phi_prev
        self.prev_potentials[agent_id] = phi_curr

        # 3. Contact reward
        r_contact = 0.0
        if collision.occurred:
            is_agent_0 = (agent_id == "agent_0")
            self_impulse = collision.impulse_a if is_agent_0 else collision.impulse_b
            opp_impulse = collision.impulse_b if is_agent_0 else collision.impulse_a
            flank_hit_delivered = collision.flank_hit_on_b if is_agent_0 else collision.flank_hit_on_a

            flank_bonus = 1.0 if flank_hit_delivered else 0.0
            r_contact = (
                0.4 * (opp_impulse / self.max_speed)
                - 0.1 * (self_impulse / self.max_speed)
                + 0.3 * flank_bonus
            )

        # 4. Penalties
        # Spin suppression: -0.05 * max(0, |omega| - 1.0)^2
        spin_excess = max(0.0, abs(agent.angular_vel) - 1.0)
        p_spin = 0.05 * (spin_excess**2)

        # Actuator jitter: -0.01 * ||a_t - a_{t-1}||^2
        prev_act = self.prev_actions.get(agent_id, np.zeros(3, dtype=np.float32))
        act_diff = action - prev_act
        p_jitter = 0.01 * float(np.sum(act_diff**2))
        self.prev_actions[agent_id] = np.asarray(action, dtype=np.float32).copy()

        # Inactivity step penalty
        p_inactivity = 0.002

        r_penalties = p_spin + p_jitter + p_inactivity

        return float(r_terminal + r_pbrs + r_contact - r_penalties)
