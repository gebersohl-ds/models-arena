"""
Example: End-to-end PPO Policy Export to ONNX with Numeric Validation
"""

import torch
import torch.nn as nn
import numpy as np

# 1. Define standard RL MLP Policy (Actor)
class ActorPolicy(nn.Module):
    def __init__(self, obs_dim: int = 16, act_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

if __name__ == "__main__":
    obs_dim = 16
    act_dim = 4
    model = ActorPolicy(obs_dim, act_dim)
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    # Import validation script
    from scripts.export_and_validate import export_and_validate
    export_and_validate(model, dummy_input, "models/actor_policy.onnx", atol=1e-5)
