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
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean = self.actor_mean(x)
            if deterministic:
                action = torch.clamp(mean, -1.0, 1.0)
            else:
                std = torch.exp(self.actor_logstd.expand_as(mean))
                dist = Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
            return action.squeeze(0).cpu().numpy().astype(np.float32)

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
