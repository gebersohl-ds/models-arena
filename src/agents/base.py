"""Abstract Agent interface for Kinetic Sumo policies."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for Kinetic Sumo agents."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Produce an action continuous 3-vector [thrust, steer, ability] in [-1.0, 1.0].

        Args:
            obs: 1D float32 array of shape (46,).

        Returns:
            1D float32 array of shape (3,).
        """
        pass

    def reset(self) -> None:
        """Reset internal state/memory between episodes."""
        pass
