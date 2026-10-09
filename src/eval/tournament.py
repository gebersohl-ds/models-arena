"""Automated round-robin Elo ladder generator (K=32) for Kinetic Sumo agents."""

import os
import argparse
import numpy as np
from dataclasses import dataclass
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy
from src.env.arena_env import KineticSumoEnv


class RandomAgent(BaseAgent):
    """Uniform random continuous action baseline."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)


@dataclass
class Competitor:
    name: str
    agent: BaseAgent
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games


def update_elo(comp_a: Competitor, comp_b: Competitor, score_a: float, k: float = 32.0) -> None:
    """Update Elo ratings for two competitors given match score (1.0 win, 0.5 draw, 0.0 loss)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((comp_b.elo - comp_a.elo) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    comp_a.elo += k * (score_a - expected_a)
    comp_b.elo += k * (score_b - expected_b)


def play_match(
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    map_name: str = "Open",
    max_duration: float = 45.0,
) -> float:
    """Run a single duel between two agents.

    Returns:
        Score for agent_0: 1.0 if agent_0 wins, 0.0 if agent_1 wins, 0.5 if draw.
    """
    env = KineticSumoEnv(map_name=map_name, max_duration=max_duration)
    obs, _ = env.reset()
    agent_0.reset()
    agent_1.reset()

    while True:
        act_0 = agent_0.act(obs["agent_0"])
        act_1 = agent_1.act(obs["agent_1"])

        obs, rewards, terminations, truncations, infos = env.step(
            {"agent_0": act_0, "agent_1": act_1}
        )

        if terminations.get("agent_0", False) or truncations.get("agent_0", False):
            winner = infos["agent_0"].get("winner")
            is_draw = infos["agent_0"].get("is_draw", False)
            env.close()

            if is_draw or winner is None:
                return 0.5
            elif winner == "agent_0":
                return 1.0
            else:
                return 0.0


def run_tournament(
    competitors: list[Competitor],
    matches_per_pair: int = 10,
    map_name: str = "Open",
    k_factor: float = 32.0,
) -> list[Competitor]:
    """Execute a round-robin tournament computing true Elo ratings."""
    n = len(competitors)
    print(f"\n================ KINETIC SUMO TOURNAMENT ================")
    print(f"Competitors: {n} | Matches per Pair: {matches_per_pair} | Map: {map_name} | K-factor: {k_factor}")
    print(f"=========================================================\n")

    for i in range(n):
        for j in range(i + 1, n):
            comp_a = competitors[i]
            comp_b = competitors[j]

            # Play pairs, alternating starting sides
            for match_idx in range(matches_per_pair):
                if match_idx % 2 == 0:
                    # comp_a is agent_0, comp_b is agent_1
                    score_a = play_match(comp_a.agent, comp_b.agent, map_name=map_name)
                else:
                    # comp_b is agent_0, comp_a is agent_1
                    score_b = play_match(comp_b.agent, comp_a.agent, map_name=map_name)
                    score_a = 1.0 - score_b

                # Update stats
                if score_a == 1.0:
                    comp_a.wins += 1
                    comp_b.losses += 1
                elif score_a == 0.0:
                    comp_b.wins += 1
                    comp_a.losses += 1
                else:
                    comp_a.draws += 1
                    comp_b.draws += 1

                update_elo(comp_a, comp_b, score_a, k=k_factor)

    # Sort leaderboard by Elo descending
    sorted_ladder = sorted(competitors, key=lambda c: c.elo, reverse=True)

    print(f"{'Rank':<5} | {'Competitor':<20} | {'Elo':<8} | {'W-L-D':<12} | {'Win Rate':<10}")
    print("-" * 65)
    for rank, comp in enumerate(sorted_ladder, start=1):
        record = f"{comp.wins}-{comp.losses}-{comp.draws}"
        print(
            f"{rank:<5} | {comp.name:<20} | {comp.elo:8.1f} | {record:<12} | {comp.win_rate * 100:6.1f}%"
        )
    print("=" * 65 + "\n")

    return sorted_ladder


def main():
    parser = argparse.ArgumentParser(description="Run Kinetic Sumo Elo Tournament")
    parser.add_argument("--matches-per-pair", type=int, default=10, help="Matches per matchup")
    parser.add_argument("--map", type=str, default="Open", help="Arena map preset")
    parser.add_argument("--ga-checkpoint", type=str, default="checkpoints/ga/best_ga_agent.npy")
    parser.add_argument("--ppo-checkpoint", type=str, default="checkpoints/ppo/best_ppo_policy.pt")
    args = parser.parse_args()

    # Load agents
    pool = [
        Competitor("Heuristic_Bot", HeuristicAgent()),
        Competitor("Random_Baseline", RandomAgent()),
    ]

    if os.path.exists(args.ga_checkpoint):
        ga_agent = GAMLP.load(args.ga_checkpoint)
        pool.append(Competitor("Neuroevolution_GA", ga_agent))
    else:
        # Default initialized GA
        pool.append(Competitor("Untrained_GA", GAMLP()))

    if os.path.exists(args.ppo_checkpoint):
        ppo_agent = PPOPolicy.load(args.ppo_checkpoint)
        pool.append(Competitor("SelfPlay_PPO", ppo_agent))
    else:
        # Default initialized PPO
        pool.append(Competitor("Untrained_PPO", PPOPolicy()))

    run_tournament(pool, matches_per_pair=args.matches_per_pair, map_name=args.map)


if __name__ == "__main__":
    main()
