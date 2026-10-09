"""League checkpoint pool and Prioritized Fictitious Self-Play (PFSP) matchmaking."""

import os
import copy
import numpy as np
from typing import Any
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent


class LeagueCheckpoint:
    """Historical checkpoint with metadata and winrate records."""

    def __init__(self, checkpoint_id: str, agent: BaseAgent, iteration: int):
        self.id = checkpoint_id
        self.agent = agent
        self.iteration = iteration
        self.wins = 0
        self.losses = 0
        self.draws = 0

    @property
    def total_matches(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_matches == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.total_matches


class LeagueManager:
    """Prioritized Fictitious Self-Play (PFSP) manager.

    Matchmaking distribution:
    - 50% vs current best policy
    - 35% vs historical league snapshots
    - 15% vs heuristic bot
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/league",
        prob_best: float = 0.50,
        prob_historical: float = 0.35,
        prob_heuristic: float = 0.15,
        max_snapshots: int = 50,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.prob_best = prob_best
        self.prob_historical = prob_historical
        self.prob_heuristic = prob_heuristic
        self.max_snapshots = max_snapshots

        self.snapshots: list[LeagueCheckpoint] = []
        self.heuristic_agent = HeuristicAgent()
        self.best_agent: BaseAgent | None = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_best_agent(self, agent: BaseAgent) -> None:
        """Update current best policy."""
        self.best_agent = agent

    def add_snapshot(self, agent: BaseAgent, iteration: int) -> LeagueCheckpoint:
        """Add a policy snapshot to the historical league pool."""
        cp_id = f"snapshot_iter_{iteration:05d}"
        # Make a deep clone of the agent
        if hasattr(agent, "clone"):
            agent_clone = agent.clone()
        elif hasattr(agent, "state_dict"):
            agent_clone = copy.deepcopy(agent)
        else:
            agent_clone = copy.copy(agent)

        checkpoint = LeagueCheckpoint(cp_id, agent_clone, iteration)
        self.snapshots.append(checkpoint)

        # Evict oldest if exceeding max size
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        return checkpoint

    def sample_opponent(self, current_agent: BaseAgent) -> tuple[BaseAgent, str]:
        """Sample an opponent according to the PFSP distribution.

        Returns (opponent_agent, opponent_category_str).
        """
        # If no snapshots exist yet, fallback to heuristic or self
        has_snapshots = len(self.snapshots) > 0
        has_best = self.best_agent is not None

        rand = np.random.rand()

        if rand < self.prob_heuristic or (not has_snapshots and not has_best):
            return self.heuristic_agent, "heuristic"

        if rand < (self.prob_heuristic + self.prob_historical) and has_snapshots:
            # Prioritized sampling: sample historical snapshots
            # Weight opponents with winrate closer to 0.5 higher (fictitious self-play focus)
            weights = np.array(
                [1.0 / (0.1 + abs(s.win_rate - 0.5)) for s in self.snapshots],
                dtype=np.float32,
            )
            weights /= np.sum(weights)
            idx = np.random.choice(len(self.snapshots), p=weights)
            return self.snapshots[idx].agent, f"historical_{self.snapshots[idx].id}"

        # Otherwise 50% vs current best (or current agent copy)
        best = self.best_agent if has_best else current_agent
        return best, "best"
