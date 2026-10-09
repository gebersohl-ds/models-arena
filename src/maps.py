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
    surface_friction: float = 1.0
    shrink_mode: str = "continuous"
    step_interval: float = 5.0
    step_drop_amount: float = 1.5
    hazard_type: str = "standard"

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

    def get_4way_spawn_transforms(self) -> list[tuple[np.ndarray, float]]:

        """Return initial (pos, angle) for 4 agents in 4-Way FFA.

        Agent 0: (-spawn_distance, 0) facing 0 (East)
        Agent 1: (+spawn_distance, 0) facing pi (West)
        Agent 2: (0, +spawn_distance) facing -pi/2 (South)
        Agent 3: (0, -spawn_distance) facing +pi/2 (North)
        """
        d = float(self.spawn_distance)
        return [
            (np.array([-d, 0.0], dtype=np.float32), 0.0),
            (np.array([d, 0.0], dtype=np.float32), float(np.pi)),
            (np.array([0.0, d], dtype=np.float32), float(-np.pi / 2.0)),
            (np.array([0.0, -d], dtype=np.float32), float(np.pi / 2.0)),
        ]


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


def make_colosseum_map() -> ArenaMap:
    """Colosseum arena with 4 symmetrical bastion pillars forming tactical combat quadrants."""
    dist = 4.6
    bastions = [
        Obstacle(x=dist * np.cos(ang), y=dist * np.sin(ang), radius=1.1, restitution=0.85, is_bumper=True)
        for ang in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    ]
    return ArenaMap(
        name="Colosseum",
        obstacles=bastions,
        spawn_distance=5.5,
    )


def make_crossfire_map() -> ArenaMap:
    """Crossfire arena with perpendicular defensive barriers creating tight cross-lanes."""
    barriers = [
        # Outer quadrant bumpers
        Obstacle(x=3.8, y=3.8, radius=0.8, restitution=1.2, is_bumper=True),
        Obstacle(x=-3.8, y=3.8, radius=0.8, restitution=1.2, is_bumper=True),
        Obstacle(x=-3.8, y=-3.8, radius=0.8, restitution=1.2, is_bumper=True),
        Obstacle(x=3.8, y=-3.8, radius=0.8, restitution=1.2, is_bumper=True),
        # Inner center chokepoint barriers
        Obstacle(x=0.0, y=1.8, radius=0.7, restitution=0.7, is_bumper=True),
        Obstacle(x=0.0, y=-1.8, radius=0.7, restitution=0.7, is_bumper=True),
    ]
    return ArenaMap(
        name="Crossfire",
        obstacles=barriers,
        spawn_distance=5.2,
    )


def make_bumper_coliseum_map() -> ArenaMap:
    """Pinball-style rebound bumpers with high restitution (2.0x bounce impulse)."""
    dist_outer = 4.5
    outer_bumpers = [
        Obstacle(
            x=dist_outer * np.cos(ang),
            y=dist_outer * np.sin(ang),
            radius=1.0,
            restitution=2.0,
            is_bumper=True,
        )
        for ang in [np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0]
    ]
    inner_bumpers = [
        Obstacle(x=0.0, y=2.5, radius=0.85, restitution=2.0, is_bumper=True),
        Obstacle(x=0.0, y=-2.5, radius=0.85, restitution=2.0, is_bumper=True),
        Obstacle(x=2.6, y=0.0, radius=0.75, restitution=2.0, is_bumper=True),
        Obstacle(x=-2.6, y=0.0, radius=0.75, restitution=2.0, is_bumper=True),
    ]
    return ArenaMap(
        name="Bumper_Coliseum",
        obstacles=outer_bumpers + inner_bumpers,
        spawn_distance=5.2,
        hazard_type="bumper",
    )


def make_frost_summit_map() -> ArenaMap:
    """Low-friction ice surface (momentum drift requiring throttle/counter-steer control)."""
    ice_pillars = [
        Obstacle(x=0.0, y=3.5, radius=0.9, restitution=0.4, is_bumper=False),
        Obstacle(x=0.0, y=-3.5, radius=0.9, restitution=0.4, is_bumper=False),
    ]
    return ArenaMap(
        name="Frost_Summit",
        obstacles=ice_pillars,
        spawn_distance=5.0,
        surface_friction=0.22,
        hazard_type="ice",
    )


def make_sinking_pillars_map() -> ArenaMap:
    """Concentric collapsing outer ring that drops in alarming stepped intervals."""
    dist = 6.2
    collapsing_pillars = [
        Obstacle(
            x=dist * np.cos(ang),
            y=dist * np.sin(ang),
            radius=1.0,
            restitution=1.0,
            is_bumper=True,
        )
        for ang in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    ]
    return ArenaMap(
        name="Sinking_Pillars",
        obstacles=collapsing_pillars,
        spawn_distance=4.8,
        shrink_start_time=15.0,
        step_interval=6.0,
        step_drop_amount=1.5,
        shrink_mode="stepped",
        hazard_type="sinking",
    )


MAP_REGISTRY = {
    "Open": make_open_map,
    "Sanctuary": make_sanctuary_map,
    "Pinball": make_pinball_map,
    "Atolls": make_atolls_map,
    "Colosseum": make_colosseum_map,
    "Crossfire": make_crossfire_map,
    "Bumper_Coliseum": make_bumper_coliseum_map,
    "Frost_Summit": make_frost_summit_map,
    "Sinking_Pillars": make_sinking_pillars_map,
}


def get_map(name: str = "Open") -> ArenaMap:
    """Retrieve map preset by name."""
    factory = MAP_REGISTRY.get(name)
    if factory is None:
        valid = list(MAP_REGISTRY.keys())
        raise ValueError(f"Unknown map preset '{name}'. Valid choices: {valid}")
    return factory()


get_map_by_name = get_map


