"""Automated round-robin Elo ladder generator (K=32) for Kinetic Sumo agents."""

import os
import sys
from pathlib import Path
import argparse
import time
import numpy as np
import pygame
from dataclasses import dataclass


class TournamentInterrupted(Exception):
    """Raised when tournament visual playback is cancelled by the user."""


# Ensure repository root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy
from src.env.arena_env import KineticSumoEnv


class RandomAgent(BaseAgent):
    """Uniform random continuous action baseline."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)


from dataclasses import dataclass, field


@dataclass
class Competitor:
    name: str
    agent: BaseAgent
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    ring_outs_inflicted: int = 0
    ring_outs_taken: int = 0
    elo_history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games

    def __getitem__(self, item: str):
        if item in ("agent", "name"):
            return self.name
        elif item == "elo":
            return self.elo
        elif hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def __contains__(self, item: str):
        return hasattr(self, item) or item in ("agent", "name", "elo")

    def get(self, item: str, default=None):
        try:
            return self[item]
        except KeyError:
            return default


def update_elo(comp_a: Competitor, comp_b: Competitor, score_a: float, k: float = 32.0) -> None:
    """Update Elo ratings for two competitors given match score (1.0 win, 0.5 draw, 0.0 loss)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((comp_b.elo - comp_a.elo) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    comp_a.elo += k * (score_a - expected_a)
    comp_b.elo += k * (score_b - expected_b)


def play_single_round(
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    map_name: str = "Open",
    max_duration: float = 45.0,
    render: bool = False,
    env: KineticSumoEnv | None = None,
    agent_names: tuple[str, str] = ("agent_0", "agent_1"),
    round_scores: tuple[int, int] = (0, 0),
    rounds_to_win: int = 3,
    subtitle: str | None = None,
    fps: int = 50,
    return_details: bool = False,
) -> float | tuple[float, str | None, bool]:
    """Run a single 45-second round between two agents.

    Returns:
        Score for agent_0: 1.0 if agent_0 wins, 0.0 if agent_1 wins, 0.5 if draw.
        If return_details is True, returns (score, winner_name, is_ring_out).
    """
    close_env = False
    if env is None:
        close_env = True
        env = KineticSumoEnv(map_name=map_name, max_duration=max_duration, render_mode="human" if render else None)

    obs, _ = env.reset()
    agent_0.reset()
    agent_1.reset()

    clock = None
    if render:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if env._renderer is not None and getattr(env._renderer, "fps", None):
            fps = env._renderer.fps
        clock = pygame.time.Clock()
        env.render(
            round_scores=round_scores,
            rounds_to_win=rounds_to_win,
            agent_names=agent_names,
            subtitle=subtitle,
            fps=fps,
        )

    paused = False
    step_once = False
    interrupted = False

    def _build_xray() -> dict[str, dict]:
        return {
            "agent_0": {
                "intent": getattr(agent_0, "last_intent", agent_0.__class__.__name__.replace("Agent", "").upper()),
                "target_vec": getattr(agent_0, "target_vec", None),
            },
            "agent_1": {
                "intent": getattr(agent_1, "last_intent", agent_1.__class__.__name__.replace("Agent", "").upper()),
                "target_vec": getattr(agent_1, "target_vec", None),
            },
        }

    while True:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    interrupted = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        interrupted = True
                        break
                    elif event.key in (pygame.K_SPACE, pygame.K_p):
                        paused = not paused
                    elif event.key in (pygame.K_PERIOD, pygame.K_n):
                        paused = True
                        step_once = True
                    elif event.key == pygame.K_l:
                        if env._renderer is not None:
                            env._renderer.show_lidar_overlay = not env._renderer.show_lidar_overlay
                    elif event.key == pygame.K_i:
                        if env._renderer is not None:
                            env._renderer.show_xray_overlay = not env._renderer.show_xray_overlay
                    elif event.key in (pygame.K_f, pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET):
                        fps = min(200, fps + 25)
                    elif event.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_LEFTBRACKET):
                        fps = max(15, fps - 25)

            if interrupted:
                break

            if paused and not step_once:
                env.render(
                    round_scores=round_scores,
                    rounds_to_win=rounds_to_win,
                    agent_names=agent_names,
                    subtitle=f"{subtitle} [PAUSED - P/Space to resume, . to step]" if subtitle else "[PAUSED - P/Space to resume, . to step]",
                    fps=fps,
                    xray_info=_build_xray(),
                    sim_status=f"PAUSED ({fps / 50.0:.2f}x)",
                )
                if clock:
                    clock.tick(20)
                continue

            step_once = False

        act_0 = agent_0.act(obs["agent_0"])
        act_1 = agent_1.act(obs["agent_1"])

        obs, rewards, terminations, truncations, infos = env.step(
            {"agent_0": act_0, "agent_1": act_1}
        )

        if render:
            env.render(
                round_scores=round_scores,
                rounds_to_win=rounds_to_win,
                agent_names=agent_names,
                subtitle=subtitle,
                fps=fps,
                xray_info=_build_xray(),
                sim_status=f"SPEED:{fps / 50.0:.2f}x",
            )

        if terminations.get("agent_0", False) or truncations.get("agent_0", False):
            winner = infos["agent_0"].get("winner")
            is_draw = infos["agent_0"].get("is_draw", False)

            if render:
                if winner == "agent_0":
                    display_scores = (round_scores[0] + 1, round_scores[1])
                elif winner == "agent_1":
                    display_scores = (round_scores[0], round_scores[1] + 1)
                else:
                    display_scores = round_scores

                end_time = time.time() + 1.0
                while time.time() < end_time:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            interrupted = True
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_l and env._renderer is not None:
                                env._renderer.show_lidar_overlay = not env._renderer.show_lidar_overlay
                            elif event.key == pygame.K_i and env._renderer is not None:
                                env._renderer.show_xray_overlay = not env._renderer.show_xray_overlay
                            elif event.key in (pygame.K_f, pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                                fps = min(200, fps + 25)
                            elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                                fps = max(15, fps - 25)
                    if interrupted:
                        break
                    env.render(
                        round_scores=display_scores,
                        rounds_to_win=rounds_to_win,
                        agent_names=agent_names,
                        subtitle=subtitle,
                        fps=fps,
                        xray_info=_build_xray(),
                        sim_status=f"SPEED:{fps / 50.0:.2f}x",
                    )

            if close_env:
                env.close()

            if interrupted:
                raise TournamentInterrupted("Tournament cancelled by user.")

            if is_draw or winner is None:
                r_score = 0.5
            elif winner == "agent_0":
                r_score = 1.0
            else:
                r_score = 0.0

            if return_details:
                is_ring_out = bool(terminations.get("agent_0", False) and not is_draw and winner is not None)
                return r_score, winner, is_ring_out
            return r_score

    if interrupted:
        if close_env:
            env.close()
        raise TournamentInterrupted("Tournament cancelled by user.")


def play_match(
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    map_name: str = "Open",
    max_duration: float = 45.0,
    rounds_to_win: int = 3,
    render: bool = False,
    env: KineticSumoEnv | None = None,
    agent_names: tuple[str, str] = ("agent_0", "agent_1"),
    match_info: str | None = None,
    fps: int = 50,
    return_details: bool = False,
) -> float | tuple[float, int, int]:
    """Run a match adhering to the rule: First to N rounds wins.

    Returns:
        Score for agent_0: 1.0 if agent_0 wins the match, 0.0 if agent_1 wins, 0.5 if draw.
        If return_details is True, returns (score, ring_outs_0, ring_outs_1).
    """
    wins_0 = 0
    wins_1 = 0
    ring_outs_0 = 0
    ring_outs_1 = 0
    max_rounds = max(rounds_to_win * 3, 9)
    close_env_at_end = False
    if env is None:
        close_env_at_end = True
        env = KineticSumoEnv(map_name=map_name, max_duration=max_duration, render_mode="human" if render else None)

    try:
        for _ in range(max_rounds):
            curr_fps = env._renderer.fps if (env is not None and env._renderer is not None and getattr(env._renderer, "fps", None)) else fps
            round_score, r_winner, r_is_ro = play_single_round(
                agent_0=agent_0,
                agent_1=agent_1,
                map_name=map_name,
                max_duration=max_duration,
                render=render,
                env=env,
                agent_names=agent_names,
                round_scores=(wins_0, wins_1),
                rounds_to_win=rounds_to_win,
                subtitle=match_info,
                fps=curr_fps,
                return_details=True,
            )
            if round_score == 1.0:
                wins_0 += 1
                if r_is_ro:
                    ring_outs_0 += 1
            elif round_score == 0.0:
                wins_1 += 1
                if r_is_ro:
                    ring_outs_1 += 1

            if wins_0 >= rounds_to_win or wins_1 >= rounds_to_win:
                break

        if render:
            if wins_0 > wins_1:
                match_result_str = f"{agent_names[0]} WINS MATCH ({wins_0}-{wins_1})!"
            elif wins_1 > wins_0:
                match_result_str = f"{agent_names[1]} WINS MATCH ({wins_0}-{wins_1})!"
            else:
                match_result_str = f"MATCH TIED ({wins_0}-{wins_1})!"

            curr_fps = env._renderer.fps if (env is not None and env._renderer is not None and getattr(env._renderer, "fps", None)) else fps
            end_time = time.time() + 1.2
            while time.time() < end_time:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        raise TournamentInterrupted("Tournament cancelled by user.")
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_l and env._renderer is not None:
                            env._renderer.show_lidar_overlay = not env._renderer.show_lidar_overlay
                        elif event.key in (pygame.K_f, pygame.K_UP):
                            curr_fps = min(200, curr_fps + 25)
                        elif event.key == pygame.K_DOWN:
                            curr_fps = max(25, curr_fps - 25)
                env.render(
                    round_scores=(wins_0, wins_1),
                    rounds_to_win=rounds_to_win,
                    agent_names=agent_names,
                    subtitle=match_result_str,
                    fps=curr_fps,
                )

        if wins_0 > wins_1:
            match_score = 1.0
        elif wins_1 > wins_0:
            match_score = 0.0
        else:
            match_score = 0.5

        if return_details:
            return match_score, ring_outs_0, ring_outs_1
        return match_score
    finally:
        if close_env_at_end:
            env.close()


class TournamentLadder(list):
    """Enriched list of Competitor objects with tournament statistics and metadata."""

    def __init__(self, competitors: list[Competitor], stats: dict | None = None):
        super().__init__(competitors)
        self.stats = stats or {}
        self.head_to_head = self.stats.get("head_to_head", {})
        self.elo_history = self.stats.get("elo_history", {})
        self.total_matches = self.stats.get("total_matches", 0)
        self.map_name = self.stats.get("map_name", "Open")


LAST_TOURNAMENT_STATS: dict = {}


def get_last_tournament_stats() -> dict:
    """Retrieve statistics from the most recent tournament simulation."""
    global LAST_TOURNAMENT_STATS
    return LAST_TOURNAMENT_STATS


def get_default_competitors() -> list[Competitor]:
    """Build default tournament competitor pool across tiers and archetypes."""
    from src.agents.tactical import BulldozerAgent, ShinobiAgent, GuardianAgent
    from src.agents.progressive import ApexAgent, ContenderAgent, RookieAgent

    pool = [
        Competitor("Apex_Champion", ApexAgent(style="champion"), elo=1420.0),
        Competitor("Bulldozer", BulldozerAgent(), elo=1280.0),
        Competitor("Contender_Bot", ContenderAgent(style="striker"), elo=1250.0),
        Competitor("Shinobi", ShinobiAgent(), elo=1240.0),
        Competitor("Guardian", GuardianAgent(), elo=1220.0),
        Competitor("Heuristic", HeuristicAgent(), elo=1200.0),
        Competitor("Rookie_Bot", RookieAgent(), elo=980.0),
        Competitor("Random", RandomAgent(), elo=800.0),
    ]
    return pool


def run_tournament(
    competitors: list[Competitor] | None = None,
    matches_per_pair: int = 10,
    map_name: str = "Open",
    k_factor: float = 32.0,
    rounds_to_win: int = 3,
    render: bool = False,
    fps: int = 50,
    quiet: bool = False,
    **kwargs,
) -> TournamentLadder:
    """Execute a round-robin tournament computing true Elo ratings."""
    global LAST_TOURNAMENT_STATS

    if competitors is None:
        competitors = get_default_competitors()

    n = len(competitors)
    total_matchups = (n * (n - 1)) // 2
    total_matches = total_matchups * matches_per_pair

    if not quiet:
        print(f"\n================ KINETIC SUMO TOURNAMENT ================")
        print(f"Competitors: {n} | Matches per Pair: {matches_per_pair} | Total Matches: {total_matches} | Map: {map_name} | K-factor: {k_factor} | First to {rounds_to_win} rounds")
        if render:
            print("Visual Mode: ON (Controls: SPACE=Pause, L=Toggle Lidar, UP/DOWN=Speed, ESC=Stop)")
        print(f"=========================================================\n")

    # Initialize tracking
    h2h: dict[str, dict[str, dict[str, int]]] = {
        c.name: {o.name: {"wins": 0, "losses": 0, "draws": 0} for o in competitors}
        for c in competitors
    }
    elo_hist: dict[str, list[tuple[int, float]]] = {
        c.name: [(0, float(c.elo))] for c in competitors
    }

    env = None
    if render:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()
        env = KineticSumoEnv(map_name=map_name, render_mode="human")
    else:
        env = KineticSumoEnv(map_name=map_name, render_mode=None)

    match_count = 0
    interrupted = False

    try:
        for i in range(n):
            for j in range(i + 1, n):
                comp_a = competitors[i]
                comp_b = competitors[j]

                # Play pairs, alternating starting sides
                for match_idx in range(matches_per_pair):
                    match_count += 1
                    match_label = f"Match {match_count}/{total_matches}: {comp_a.name} vs {comp_b.name}"

                    if match_idx % 2 == 0:
                        # comp_a is agent_0, comp_b is agent_1
                        score_a, ro_a, ro_b = play_match(
                            comp_a.agent,
                            comp_b.agent,
                            map_name=map_name,
                            rounds_to_win=rounds_to_win,
                            render=render,
                            env=env,
                            agent_names=(comp_a.name, comp_b.name),
                            match_info=match_label,
                            fps=fps,
                            return_details=True,
                        )
                    else:
                        # comp_b is agent_0, comp_a is agent_1
                        score_b, ro_b, ro_a = play_match(
                            comp_b.agent,
                            comp_a.agent,
                            map_name=map_name,
                            rounds_to_win=rounds_to_win,
                            render=render,
                            env=env,
                            agent_names=(comp_b.name, comp_a.name),
                            match_info=f"Match {match_count}/{total_matches}: {comp_b.name} vs {comp_a.name}",
                            fps=fps,
                            return_details=True,
                        )
                        score_a = 1.0 - score_b

                    # Update real ring-out stats
                    comp_a.ring_outs_inflicted += ro_a
                    comp_a.ring_outs_taken += ro_b
                    comp_b.ring_outs_inflicted += ro_b
                    comp_b.ring_outs_taken += ro_a

                    # Update match records
                    if score_a == 1.0:
                        comp_a.wins += 1
                        comp_b.losses += 1
                        h2h[comp_a.name][comp_b.name]["wins"] += 1
                        h2h[comp_b.name][comp_a.name]["losses"] += 1
                    elif score_a == 0.0:
                        comp_b.wins += 1
                        comp_a.losses += 1
                        h2h[comp_a.name][comp_b.name]["losses"] += 1
                        h2h[comp_b.name][comp_a.name]["wins"] += 1
                    else:
                        comp_a.draws += 1
                        comp_b.draws += 1
                        h2h[comp_a.name][comp_b.name]["draws"] += 1
                        h2h[comp_b.name][comp_a.name]["draws"] += 1

                    update_elo(comp_a, comp_b, score_a, k=k_factor)
                    elo_hist[comp_a.name].append((match_count, float(comp_a.elo)))
                    elo_hist[comp_b.name].append((match_count, float(comp_b.elo)))

    except TournamentInterrupted:
        interrupted = True
        if not quiet:
            print(f"\n[!] Tournament playback stopped by user at match {match_count}/{total_matches}.")
    finally:
        if env is not None:
            env.close()

    # Sort leaderboard by Elo descending
    sorted_ladder = sorted(competitors, key=lambda c: c.elo, reverse=True)

    if not quiet:
        header = "CURRENT STANDINGS (INTERRUPTED)" if interrupted else "FINAL TOURNAMENT STANDINGS"
        print(f"\n{header}:")
        print(f"{'Rank':<5} | {'Competitor':<20} | {'Elo':<8} | {'W-L-D':<12} | {'Win Rate':<10}")
        print("-" * 65)
        for rank, comp in enumerate(sorted_ladder, start=1):
            record = f"{comp.wins}-{comp.losses}-{comp.draws}"
            print(
                f"{rank:<5} | {comp.name:<20} | {comp.elo:8.1f} | {record:<12} | {comp.win_rate * 100:6.1f}%"
            )
        print("=" * 65 + "\n")

    stats = {
        "head_to_head": h2h,
        "elo_history": elo_hist,
        "total_matches": match_count,
        "map_name": map_name,
        "interrupted": interrupted,
        "ring_outs": {
            c.name: {
                "inflicted": c.ring_outs_inflicted,
                "taken": c.ring_outs_taken,
            }
            for c in competitors
        },
    }
    LAST_TOURNAMENT_STATS = stats

    return TournamentLadder(sorted_ladder, stats=stats)


def main():
    parser = argparse.ArgumentParser(description="Run Kinetic Sumo Elo Tournament")
    parser.add_argument("--matches-per-pair", type=int, default=10, help="Matches per matchup")
    parser.add_argument("--map", type=str, default="Open", help="Arena map preset")
    parser.add_argument("--ga-checkpoint", type=str, default="checkpoints/ga/best_ga_agent.npy")
    parser.add_argument("--ppo-checkpoint", type=str, default="checkpoints/ppo/best_ppo_policy.pt")
    parser.add_argument("--rounds-to-win", type=int, default=3, help="First to N rounds to win match")
    parser.add_argument("--render", action="store_true", help="Watch tournament matches live in Pygame window")
    parser.add_argument("--fps", type=int, default=50, help="Target FPS when visual rendering is enabled")
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

    run_tournament(
        pool,
        matches_per_pair=args.matches_per_pair,
        map_name=args.map,
        rounds_to_win=args.rounds_to_win,
        render=args.render,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
