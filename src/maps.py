"""Geometric arena presets: Open, Sanctuary, Pinball, and Atolls."""

from dataclasses import dataclass, field
import numpy as np
from src.engine.physics import Obstacle


@dataclass
class ArenaMap:
    """Arena geometry and configuration."""

    name: str
    initial_radius: float = 10.0
    min_radius: float = 4.0
    shrink_start_time: float = 20.0
    round_duration: float = 45.0
    obstacles: list[Obstacle] = field(default_factory=list)
    spawn_distance: float = 5.0

    def get_spawn_transforms(self) -> tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]]:
        """Return initial (pos, angle) for agent_0 and agent_1.

        Agent 0 starts at (-spawn_distance, 0) facing +X (angle 0).
        Agent 1 starts at (+spawn_distance, 0) facing -X (angle pi).
        """
        p0 = np.array([-self.spawn_distance, 0.0], dtype=np.float32)
        ang0 = 0.0
        p1 = np.array([self.spawn_distance, 0.0], dtype=np.float32)
        ang1 = float(np.pi)
        return (p0, ang0), (p1, ang1)


def make_open_map() -> ArenaMap:
    """Pure circular sumo ring with no interior obstacles."""
    return ArenaMap(name="Open")


def make_sanctuary_map() -> ArenaMap:
    """Sanctuary arena with a central defensive bumper pillar absorbing recoil."""
    center_pillar = Obstacle(x=0.0, y=0.0, radius=1.3, restitution=0.6, is_bumper=True)
    return ArenaMap(
        name="Sanctuary",
        obstacles=[center_pillar],
        spawn_distance=5.5,
    )


def make_pinball_map() -> ArenaMap:
    """Pinball arena with 4 high-restitution bouncy bumpers arranged symmetrically."""
    dist = 4.2
    bumpers = [
        Obstacle(x=dist * np.cos(ang), y=dist * np.sin(ang), radius=0.9, restitution=1.5, is_bumper=True)
        for ang in [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
    ]
    return ArenaMap(
        name="Pinball",
        obstacles=bumpers,
        spawn_distance=5.2,
    )


def make_atolls_map() -> ArenaMap:
    """Atolls arena with stepping platform obstacles creating chokepoints."""
    pillars = [
        Obstacle(x=0.0, y=3.2, radius=1.0, restitution=0.9, is_bumper=True),
        Obstacle(x=0.0, y=-3.2, radius=1.0, restitution=0.9, is_bumper=True),
    ]
    return ArenaMap(
        name="Atolls",
        obstacles=pillars,
        spawn_distance=5.0,
    )


MAP_REGISTRY = {
    "Open": make_open_map,
    "Sanctuary": make_sanctuary_map,
    "Pinball": make_pinball_map,
    "Atolls": make_atolls_map,
}


def get_map(name: str = "Open") -> ArenaMap:
    """Retrieve map preset by name."""
    factory = MAP_REGISTRY.get(name)
    if factory is None:
        valid = list(MAP_REGISTRY.keys())
        raise ValueError(f"Unknown map preset '{name}'. Valid choices: {valid}")
    return factory()
