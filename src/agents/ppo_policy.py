"""Compact PyTorch Actor-Critic policy (CleanRL style) for Kinetic Sumo."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.agents.base import BaseAgent


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization following CleanRL convention."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module, BaseAgent):
    """Actor-Critic MLP policy for continuous action space [-1.0, 1.0]^3."""

    def __init__(self, obs_dim: int = 46, act_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Critic network (evaluates state value V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor network (computes action mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )

        # State-independent learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.hidden_dim = hidden_dim
        self.last_intent: str = "PPO ACTOR-CRITIC"
        self.last_value: float = 0.0
        self.target_vec: np.ndarray | None = None
        self.last_obs: np.ndarray = np.zeros(obs_dim, dtype=np.float32)
        self.last_h1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.last_h2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.last_out: np.ndarray = np.zeros(act_dim, dtype=np.float32)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log probability, distribution entropy, and critic value."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            raw_action = probs.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Inference wrapper fulfilling BaseAgent interface.

        Args:
            obs: 1D float32 numpy array of shape (46,).
            deterministic: If True, uses mean action without noise.

        Returns:
            1D float32 numpy array of shape (3,).
        """
        self.eval()
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            h1_t = torch.tanh(self.actor_mean[0](x))
            h2_t = torch.tanh(self.actor_mean[2](h1_t))
            mean = self.actor_mean[4](h2_t)
            val = self.critic(x)
            self.last_value = float(val.item())
            self.last_h1 = h1_t.squeeze(0).cpu().numpy().astype(np.float32)
            self.last_h2 = h2_t.squeeze(0).cpu().numpy().astype(np.float32)
            if deterministic:
                action = torch.clamp(mean, -1.0, 1.0)
            else:
                std = torch.exp(self.actor_logstd.expand_as(mean))
                dist = Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
            out = action.squeeze(0).cpu().numpy().astype(np.float32)

        self.last_out = out.copy()
        self.last_intent = f"PPO V(s)={self.last_value:+.2f}"
        if len(obs) >= 9:
            cos_th, sin_th = float(obs[4]), float(obs[5])
            heading = np.arctan2(sin_th, cos_th)
            aim_ang = heading + float(out[1]) * 0.8
            mag = max(1.5, abs(float(out[0])) * 4.0)
            self.target_vec = np.array([np.cos(aim_ang) * mag, np.sin(aim_ang) * mag], dtype=np.float32)
        return out

    def get_causal_attribution(self, obs: np.ndarray | None = None) -> list[tuple[str, float, str]]:
        """Compute Top-3 sensory drivers for the PPO Actor network."""
        from src.agents.ga_mlp import SENSORY_CLUSTER_SPECS

        if obs is not None:
            self.act(obs, deterministic=True)
        w1 = self.actor_mean[0].weight.detach().cpu().numpy().T  # (46, H)
        w2 = self.actor_mean[2].weight.detach().cpu().numpy().T  # (H, H)
        w3 = self.actor_mean[4].weight.detach().cpu().numpy().T  # (H, 3)

        h2_contrib = np.abs(self.last_h2)[:, None] * np.abs(w3) + 1e-4
        h1_contrib = np.abs(self.last_h1)[:, None] * (np.abs(w2) @ h2_contrib)
        in_mag = np.maximum(np.abs(self.last_obs), 0.05)[:, None]
        in_to_out = in_mag * (np.abs(w1) @ h1_contrib)

        actuator_names = ["THR", "STR", "ABL"]
        raw_entries: list[tuple[str, float, str]] = []
        total_drive = 0.0
        for c_name, indices in SENSORY_CLUSTER_SPECS:
            valid_idx = [i for i in indices if i < len(in_to_out)]
            if not valid_idx:
                continue
            c_vec = np.mean(in_to_out[valid_idx], axis=0)
            c_score = float(np.sum(c_vec))
            dom_act = actuator_names[int(np.argmax(c_vec))]
            raw_entries.append((c_name, c_score, dom_act))
            total_drive += c_score

        denom = max(1e-6, total_drive)
        drivers = [(c_name, c_score / denom, dom_act) for c_name, c_score, dom_act in raw_entries]
        drivers.sort(key=lambda item: item[1], reverse=True)
        return drivers[:3]

    def get_brain_snapshot(self, obs: np.ndarray | None = None) -> dict:
        """Return neural activation & synaptic pathway snapshot for PPO Actor network."""
        from src.agents.ga_mlp import SENSORY_CLUSTER_SPECS, compute_sensory_clusters

        if obs is not None:
            self.act(obs, deterministic=True)

        w1 = self.actor_mean[0].weight.detach().cpu().numpy().T  # (46, H)
        w2 = self.actor_mean[2].weight.detach().cpu().numpy().T  # (H, H)
        w3 = self.actor_mean[4].weight.detach().cpu().numpy().T  # (H, 3)

        clusters = compute_sensory_clusters(self.last_obs)
        cluster_w1 = np.zeros((len(SENSORY_CLUSTER_SPECS), self.hidden_dim), dtype=np.float32)
        for c_idx, (_, indices) in enumerate(SENSORY_CLUSTER_SPECS):
            valid_idx = [i for i in indices if i < self.obs_dim]
            if valid_idx:
                cluster_w1[c_idx] = np.mean(w1[valid_idx], axis=0)

        return {
            "arch": f"PPO {self.obs_dim}->{self.hidden_dim}->{self.hidden_dim}->{self.act_dim}",
            "clusters": clusters,
            "h1": self.last_h1.copy(),
            "h2": self.last_h2.copy(),
            "out": self.last_out.copy(),
            "cluster_w1": cluster_w1,
            "w2": w2,
            "w3": w3,
            "top_drivers": self.get_causal_attribution(),
            "intent": self.last_intent,
        }


    def save(self, filepath: str) -> None:
        """Save policy weights to file."""
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(cls, filepath: str, obs_dim: int = 46, act_dim: int = 3) -> "PPOPolicy":
        """Load policy weights from file."""
        policy = cls(obs_dim=obs_dim, act_dim=act_dim)
        state_dict = torch.load(filepath, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy
