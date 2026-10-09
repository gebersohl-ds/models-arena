"""Training pipelines and league management."""

from src.training.league import LeagueManager
from src.training.train_ga import train_ga
from src.training.train_ppo import train_ppo

__all__ = ["LeagueManager", "train_ga", "train_ppo"]
