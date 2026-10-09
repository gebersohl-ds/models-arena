"""Re-export arena maps from src.maps for compatibility."""

from src.maps import (
    ArenaMap,
    MAP_REGISTRY,
    make_open_map,
    make_sanctuary_map,
    make_pinball_map,
    make_atolls_map,
    make_colosseum_map,
    make_crossfire_map,
    make_bumper_coliseum_map,
    make_frost_summit_map,
    make_sinking_pillars_map,
    get_map,
    get_map_by_name,
)

__all__ = [
    "ArenaMap",
    "MAP_REGISTRY",
    "make_open_map",
    "make_sanctuary_map",
    "make_pinball_map",
    "make_atolls_map",
    "make_colosseum_map",
    "make_crossfire_map",
    "make_bumper_coliseum_map",
    "make_frost_summit_map",
    "make_sinking_pillars_map",
    "get_map",
    "get_map_by_name",
]
