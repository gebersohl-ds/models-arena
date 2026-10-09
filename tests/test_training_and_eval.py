"""Unit tests for League management, training smoke tests, and tournament evaluation."""

import os
import shutil
import tempfile
import pytest
from src.agents.heuristic import HeuristicAgent
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy
from src.training.league import LeagueManager
from src.training.train_ga import train_ga
from src.training.train_ppo import train_ppo
from src.eval.tournament import run_tournament, Competitor, RandomAgent


def test_league_manager():
    with tempfile.TemporaryDirectory() as tmp_dir:
        league = LeagueManager(checkpoint_dir=tmp_dir)
        policy = PPOPolicy()
        league.set_best_agent(policy)
        cp = league.add_snapshot(policy, iteration=1)

        assert len(league.snapshots) == 1
        assert cp.win_rate == 0.5

        # Sample multiple opponents and check valid types
        categories = set()
        for _ in range(30):
            agent, cat = league.sample_opponent(policy)
            assert agent is not None
            categories.add(cat)

        # Should sample across categories
        assert "heuristic" in categories or "best" in categories or any("historical" in c for c in categories)


def test_train_ga_smoke():
    with tempfile.TemporaryDirectory() as tmp_dir:
        best_agent = train_ga(
            pop_size=8,
            generations=2,
            save_dir=tmp_dir,
            map_name="Open",
        )
        assert isinstance(best_agent, GAMLP)
        assert os.path.exists(os.path.join(tmp_dir, "best_ga_agent.npy"))


def test_train_ppo_smoke():
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy = train_ppo(
            total_timesteps=512,
            num_envs=4,
            num_steps=32,
            update_epochs=2,
            num_minibatches=2,
            snapshot_interval=2,
            save_dir=tmp_dir,
            map_name="Open",
        )
        assert isinstance(policy, PPOPolicy)
        assert os.path.exists(os.path.join(tmp_dir, "best_ppo_policy.pt"))


def test_tournament_ladder():
    comps = [
        Competitor("Heuristic", HeuristicAgent()),
        Competitor("Random", RandomAgent()),
        Competitor("GA", GAMLP()),
    ]
    ladder = run_tournament(comps, matches_per_pair=2, map_name="Open", k_factor=32.0)
    assert len(ladder) == 3
    # Check that Elo values changed from initial 1200
    assert any(c.elo != 1200.0 for c in ladder)
    # Check leaderboard is sorted descending by Elo
    for i in range(len(ladder) - 1):
        assert ladder[i].elo >= ladder[i + 1].elo
