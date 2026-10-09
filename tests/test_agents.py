"""Unit tests for agent policies: Heuristic, GA MLP, and PPO Actor-Critic."""

import os
import tempfile
import numpy as np
import pytest
import torch
from src.agents.heuristic import HeuristicAgent
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy


def test_heuristic_agent_behavior():
    agent = HeuristicAgent()

    # 1. Normal observation inside arena facing opponent
    obs = np.zeros(46, dtype=np.float32)
    obs[0] = 0.0  # pos x = 0
    obs[1] = 0.0  # pos y = 0
    obs[4] = 1.0  # facing +X
    obs[7] = 0.2  # opp dx = 4.0 m in front
    obs[12] = 1.0  # full energy

    action = agent.act(obs)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    # When aligned with opponent at medium range with full energy, should charge / dash
    assert action[0] > 0.5  # positive forward thrust

    # 2. Perimeter danger state: agent at x=8.0 (near edge > 6.8m)
    obs_danger = np.zeros(46, dtype=np.float32)
    obs_danger[0] = 0.8  # pos x = 8.0 m
    obs_danger[1] = 0.0
    obs_danger[4] = 1.0  # facing +X (toward edge!)

    action_danger = agent.act(obs_danger)
    # Agent must steer away from edge back towards center (steer towards -X)
    assert abs(action_danger[1]) > 0.0  # steering corrective torque applied


def test_ga_mlp_policy():
    mlp = GAMLP(input_dim=46, hidden_dims=(32, 32), output_dim=3)
    assert mlp.total_params == 2659

    # Weight flattening roundtrip
    weights = mlp.get_weights()
    assert weights.shape == (2659,)
    mlp.set_weights(weights)
    assert np.allclose(mlp.get_weights(), weights)

    # Single act
    obs = np.random.randn(46).astype(np.float32)
    act = mlp.act(obs)
    assert act.shape == (3,)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)

    # Batch act
    obs_batch = np.random.randn(10, 46).astype(np.float32)
    acts = mlp.batch_act(obs_batch)
    assert acts.shape == (10, 3)
    assert np.all(acts >= -1.0) and np.all(acts <= 1.0)

    # Mutation
    mutated = mlp.mutate(mutation_rate=0.1, sigma=0.2)
    assert mutated.total_params == 2659
    assert not np.allclose(mutated.get_weights(), weights)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        path = tmp.name
    try:
        mlp.save(path)
        loaded = GAMLP.load(path)
        assert np.allclose(loaded.get_weights(), weights)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_ppo_policy():
    policy = PPOPolicy(obs_dim=46, act_dim=3, hidden_dim=64)

    # Tensor input test
    x = torch.randn(4, 46)
    action, log_prob, entropy, value = policy.get_action_and_value(x)

    assert action.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)

    # BaseAgent act inference test
    obs_np = np.random.randn(46).astype(np.float32)
    act_np = policy.act(obs_np, deterministic=True)
    assert act_np.shape == (3,)
    assert np.all(act_np >= -1.0) and np.all(act_np <= 1.0)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        policy.save(path)
        loaded = PPOPolicy.load(path)
        act_loaded = loaded.act(obs_np, deterministic=True)
        assert np.allclose(act_np, act_loaded, atol=1e-5)
    finally:
        if os.path.exists(path):
            os.remove(path)
