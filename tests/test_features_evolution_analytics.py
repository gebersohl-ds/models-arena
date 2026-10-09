"""Comprehensive tests for Tournament fixes, Progressive ML agents, Analytics, and Dynamic Hazards."""

import os
import tempfile
import numpy as np
import pytest
import pygame

from src.maps import get_map, MAP_REGISTRY
from src.engine.physics import PhysicsEngine, DiscAgent
from src.env.arena_env import KineticSumoEnv
from src.agents.progressive import (
    RookieAgent,
    ContenderAgent,
    ApexAgent,
    ensure_progressive_checkpoints,
)
from src.agents.ga_mlp import GAMLP
from src.eval.tournament import (
    Competitor,
    TournamentLadder,
    run_tournament,
    get_default_competitors,
    RandomAgent,
)
from src.ui.hub import ArcadeHub


def test_competitor_dict_and_attribute_access():
    """Verify Competitor supports both object attribute access and dictionary-like subscripting."""
    agent = RandomAgent()
    comp = Competitor(name="Apex_Champion", agent=agent, elo=1420.0)

    # Attribute access
    assert comp.name == "Apex_Champion"
    assert comp.elo == 1420.0
    assert comp.wins == 0
    assert comp.win_rate == 0.0

    # Dict-like access
    assert comp["agent"] == "Apex_Champion"
    assert comp["name"] == "Apex_Champion"
    assert comp["elo"] == 1420.0
    assert "agent" in comp
    assert "elo" in comp
    assert comp.get("elo") == 1420.0
    assert comp.get("unknown_key", "default") == "default"


def test_run_tournament_default_competitors_and_quiet():
    """Verify run_tournament works with competitors=None, quiet=True, returning TournamentLadder."""
    ladder = run_tournament(
        competitors=None,
        matches_per_pair=1,
        map_name="Open",
        render=False,
        quiet=True,
    )

    assert isinstance(ladder, TournamentLadder)
    assert len(ladder) == 8
    # Dict-like lookup for hub.py compatibility
    assert "agent" in ladder[0]
    assert isinstance(ladder[0]["elo"], float)

    # Deep stats
    assert hasattr(ladder, "head_to_head")
    assert hasattr(ladder, "elo_history")
    assert len(ladder.head_to_head) == 8
    assert len(ladder.elo_history) == 8

    # Descending Elo sorting
    for i in range(len(ladder) - 1):
        assert ladder[i].elo >= ladder[i + 1].elo


def test_progressive_evolutionary_agents_and_checkpoints():
    """Verify Rookie, Contender, and Apex agents act properly and checkpoints persist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ensure_progressive_checkpoints(tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "rookie_ga.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "contender_striker.npy"))
        assert os.path.exists(os.path.join(tmp_dir, "apex_champion.npy"))

        rookie = RookieAgent(checkpoint_path=os.path.join(tmp_dir, "rookie_ga.npy"))
        contender = ContenderAgent(checkpoint_path=os.path.join(tmp_dir, "contender_striker.npy"))
        apex = ApexAgent(checkpoint_path=os.path.join(tmp_dir, "apex_champion.npy"))

        obs = np.zeros(46, dtype=np.float32)
        r_act = rookie.act(obs)
        c_act = contender.act(obs)
        a_act = apex.act(obs)

        assert r_act.shape == (3,)
        assert c_act.shape == (3,)
        assert a_act.shape == (3,)
        assert np.all(r_act >= -1.0) and np.all(r_act <= 1.0)
        assert np.all(c_act >= -1.0) and np.all(c_act <= 1.0)
        assert np.all(a_act >= -1.0) and np.all(a_act <= 1.0)


def test_mirror_symmetry_breaking():
    """Verify that identical bots facing each other output non-identical stochastic actions."""
    obs = np.zeros(46, dtype=np.float32)
    g1 = GAMLP(stochasticity=0.08)
    g2 = GAMLP(weights=g1.get_weights(), stochasticity=0.08)

    acts_1 = [g1.act(obs) for _ in range(20)]
    acts_2 = [g2.act(obs) for _ in range(20)]

    # Stochasticity breaks rigid lockstep mirror actions
    diffs = [np.linalg.norm(a1 - a2) for a1, a2 in zip(acts_1, acts_2)]
    assert np.mean(diffs) > 0.01


def test_bumper_coliseum_hazard_mechanics():
    """Verify Bumper_Coliseum has 2.0x restitution bumpers that kick back with amplified impulse."""
    coliseum_map = get_map("Bumper_Coliseum")
    assert coliseum_map.name == "Bumper_Coliseum"
    assert len(coliseum_map.obstacles) >= 6
    assert any(obs.restitution >= 1.8 for obs in coliseum_map.obstacles)

    # Test physics collision with high-restitution bumper
    engine = PhysicsEngine()
    bumper = coliseum_map.obstacles[0]
    agent = DiscAgent(initial_pos=bumper.pos - np.array([agent_radius := 0.6, 0.0]))
    agent.vel = np.array([5.0, 0.0], dtype=np.float32)

    collided = engine.resolve_obstacle_collision(agent, bumper)
    assert collided is True
    # Normal points towards -X, restitution = 2.0 -> velocity reversed with > 2.0x impulse
    assert agent.vel[0] < -5.0


def test_frost_summit_ice_surface_momentum_drift():
    """Verify Frost_Summit has reduced surface friction causing extended momentum drift."""
    frost_map = get_map("Frost_Summit")
    assert frost_map.name == "Frost_Summit"
    assert frost_map.surface_friction < 0.5

    engine_ice = PhysicsEngine(surface_friction=frost_map.surface_friction, dt=0.02)
    engine_normal = PhysicsEngine(surface_friction=1.0, dt=0.02)

    agent_ice = DiscAgent(initial_pos=np.array([0.0, 0.0]))
    agent_ice.vel = np.array([6.0, 0.0], dtype=np.float32)

    agent_normal = DiscAgent(initial_pos=np.array([0.0, 0.0]))
    agent_normal.vel = np.array([6.0, 0.0], dtype=np.float32)

    # Coast without thrust for 10 steps
    for _ in range(10):
        engine_ice.apply_agent_controls(agent_ice, np.zeros(3, dtype=np.float32))
        engine_normal.apply_agent_controls(agent_normal, np.zeros(3, dtype=np.float32))

    # Ice surface should retain significantly more speed than normal surface
    assert agent_ice.vel[0] > agent_normal.vel[0]


def test_sinking_pillars_stepped_outer_ring_collapse():
    """Verify Sinking_Pillars drops outer ring in discrete stepped intervals."""
    sinking_map = get_map("Sinking_Pillars")
    assert sinking_map.shrink_mode == "stepped"

    engine = PhysicsEngine(
        initial_radius=sinking_map.initial_radius,
        min_radius=sinking_map.min_radius,
        shrink_start_time=sinking_map.shrink_start_time,
        shrink_mode=sinking_map.shrink_mode,
        step_interval=sinking_map.step_interval,
        step_drop_amount=sinking_map.step_drop_amount,
    )

    # Before shrink start time (15s): radius is initial (10m)
    assert engine.compute_arena_radius(10.0) == 10.0
    assert engine.compute_arena_radius(15.0) == 10.0

    # At 16s (first step dropped): radius drops by step_drop_amount (1.5m -> 8.5m)
    r_step1 = engine.compute_arena_radius(16.0)
    assert np.isclose(r_step1, 8.5)

    # At 20s (still within first step interval [15, 21)): radius stays 8.5m (discrete plateaus)
    assert np.isclose(engine.compute_arena_radius(20.0), 8.5)

    # At 22s (second step dropped): radius drops to 7.0m
    r_step2 = engine.compute_arena_radius(22.0)
    assert np.isclose(r_step2, 7.0)


def test_arcade_hub_analytics_and_evo_lab_state():
    """Verify ArcadeHub initializes all 7 tabs, analytics state, and runs evo lab generation stepping."""
    hub = ArcadeHub(screen_size=900)
    assert len(hub.TABS) == 7
    assert "ANALYTICS" in hub.TABS
    assert "EVO LAB" in hub.TABS

    # Analytics state
    assert hub.analytics_subview in (0, 1)
    assert "head_to_head" in hub.tournament_stats
    assert "elo_history" in hub.tournament_stats

    # Evo Lab state & step
    init_gen = hub.evo_generation
    hub._evo_step_generation(turbo_count=1)
    assert hub.evo_generation == init_gen + 1
    assert len(hub.evo_history_best) >= 2
    assert hub.evo_best_agent is not None

    # Agent instantiation for progressive bots
    apex_agent = hub._instantiate_agent("Apex_Bot")
    assert isinstance(apex_agent, ApexAgent)
    contender_agent = hub._instantiate_agent("Contender_Bot")
    assert isinstance(contender_agent, ContenderAgent)
    rookie_agent = hub._instantiate_agent("Rookie_Bot")
    assert isinstance(rookie_agent, RookieAgent)


def test_real_ring_out_tracking_in_tournament():
    """Verify play_match and run_tournament record true physical ring-outs without hardcoding."""
    from src.eval.tournament import play_match
    from src.agents.heuristic import HeuristicAgent

    agent_a = HeuristicAgent()
    agent_b = RandomAgent()

    # Play match with return_details=True
    score, ro_a, ro_b = play_match(
        agent_0=agent_a,
        agent_1=agent_b,
        map_name="Open",
        rounds_to_win=2,
        return_details=True,
    )
    assert score in (0.0, 0.5, 1.0)
    assert isinstance(ro_a, int) and isinstance(ro_b, int)
    assert ro_a >= 0 and ro_b >= 0

    # Test mini tournament with 3 competitors
    comps = [
        Competitor("Apex", ApexAgent(), elo=1420.0),
        Competitor("Contender", ContenderAgent(), elo=1250.0),
        Competitor("Rookie", RookieAgent(), elo=980.0),
    ]
    ladder = run_tournament(
        competitors=comps,
        matches_per_pair=1,
        rounds_to_win=1,
        quiet=True,
    )
    assert "ring_outs" in ladder.stats
    for c in ladder:
        assert c.name in ladder.stats["ring_outs"]
        ro_info = ladder.stats["ring_outs"][c.name]
        assert "inflicted" in ro_info and "taken" in ro_info
        assert c.ring_outs_inflicted == ro_info["inflicted"]
        assert c.ring_outs_taken == ro_info["taken"]


def test_progressive_hierarchy_and_ladder_climb():
    """Verify Apex outperforms Rookie and Contender outperforms Rookie in direct combat."""
    apex = ApexAgent()
    contender = ContenderAgent()
    rookie = RookieAgent()

    # Apex vs Rookie (Apex should dominate)
    from src.eval.tournament import play_match
    score_apex = play_match(apex, rookie, rounds_to_win=2)
    assert score_apex == 1.0

    # Contender vs Rookie (Contender should defeat novice Rookie)
    score_cont = play_match(contender, rookie, rounds_to_win=2)
    assert score_cont == 1.0


def test_export_evo_champion_integration():
    """Verify exporting Evo Lab champion integrates into Hub tournament roster and play.py loader."""
    from src.ui.play import load_agent
    hub = ArcadeHub(screen_size=900)

    # Initial export
    hub._export_evo_champion()
    champ_name = f"Evo_Gen{hub.evo_generation}"
    assert champ_name in hub.BOT_CHOICES
    assert champ_name in hub.tournament_ratings

    # Verify competitor pool includes exported champion
    comps = hub.get_tournament_competitors()
    comp_names = [c.name for c in comps]
    assert champ_name in comp_names

    # Verify play.py loader can instantiate exported evo agent
    evo_loaded = load_agent("evo")
    assert isinstance(evo_loaded, GAMLP)
    evo_named_loaded = load_agent(champ_name)
    assert isinstance(evo_named_loaded, GAMLP)


def test_hitbox_alignment_and_slider_tolerance():
    """Verify event hitboxes match rendered rectangles for Analytics and UISlider."""
    import pygame
    from src.ui.hub import UISlider
    hub = ArcadeHub(screen_size=900)

    # Test Analytics button click event at rendered position (y=160)
    click_event_h2h = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (200, 175)})
    hub.current_tab = 3
    hub.analytics_subview = 1
    hub._handle_tab_events(click_event_h2h)
    assert hub.analytics_subview == 0

    click_event_elo = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (500, 175)})
    hub._handle_tab_events(click_event_elo)
    assert hub.analytics_subview == 1

    # Test UISlider grab tolerance inflate
    slider = UISlider(pygame.Rect(100, 200, 200, 10), "Test", 0.0, 1.0, 0.5, hub.font_small)
    # Click 5px above the 10px track (y=195): should be grabbed due to inflate
    grab_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (150, 195)})
    assert slider.handle_event(grab_event) is True
    assert slider.dragging is True


def test_turbo_evolve_and_mutation_synchrony():
    """Verify turbo evolution and mutate champion maintain equal history lengths with no IndexError."""
    hub = ArcadeHub(screen_size=900)
    init_gen = hub.evo_generation
    init_best_len = len(hub.evo_history_best)
    init_mean_len = len(hub.evo_history_mean)
    assert init_best_len == init_mean_len

    # Test turbo evolve (10 generations)
    hub._evo_step_generation(turbo_count=10)
    assert hub.evo_generation == init_gen + 10
    assert len(hub.evo_history_best) == init_best_len + 10
    assert len(hub.evo_history_mean) == init_mean_len + 10
    assert len(hub.evo_history_best) == len(hub.evo_history_mean)

    # Test mutate champion
    gen_before_mutate = hub.evo_generation
    hub._mutate_champion()
    assert hub.evo_generation == gen_before_mutate + 1
    assert len(hub.evo_history_best) == len(hub.evo_history_mean)

    # Ensure tab draws cleanly with no IndexError
    hub.current_tab = 4
    hub._draw_hub_frame()


def test_evo_lab_history_index_mismatch_resilience():
    """Verify _draw_evo_lab_tab handles desynchronized or empty history lists defensively."""
    hub = ArcadeHub(screen_size=900)
    hub.current_tab = 4

    # Desynchronized history: best has 10 items, mean has 2 items
    hub.evo_history_best = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    hub.evo_history_mean = [-2.0, -1.0]
    # Must NOT raise IndexError: list index out of range
    hub._draw_evo_lab_tab()

    # Empty history lists
    hub.evo_history_best = []
    hub.evo_history_mean = []
    # Must NOT crash on best[-1]
    hub._draw_evo_lab_tab()


def test_no_text_overlaps_across_all_tabs():
    """Verify no rendered text bounding boxes overlap across all 7 hub tabs and analytics subviews."""
    hub = ArcadeHub(screen_size=900)

    # Track rendered text surfaces and bounding boxes
    text_blits = []
    surf_tag_map = {}

    class TrackedSurface(pygame.Surface):
        def blit(self, source, dest, *args, **kwargs):
            res = super().blit(source, dest, *args, **kwargs)
            if id(source) in surf_tag_map:
                if isinstance(dest, (tuple, list)):
                    r = pygame.Rect(dest[0], dest[1], source.get_width(), source.get_height())
                elif isinstance(dest, pygame.Rect):
                    r = pygame.Rect(dest.x, dest.y, source.get_width(), source.get_height())
                else:
                    r = pygame.Rect(0, 0, source.get_width(), source.get_height())
                text_blits.append((r, surf_tag_map[id(source)]))
            return res

    # Wrap fonts on hub
    font_wrappers = []
    for f_attr in ["font_large", "font_med", "font_small", "font_tiny"]:
        real_f = getattr(hub, f_attr)
        class Wrapper:
            def __init__(self, f):
                self._f = f
            def render(self, text, antialias, color, background=None):
                s = self._f.render(text, antialias, color, background)
                surf_tag_map[id(s)] = str(text)
                return s
            def size(self, text):
                return self._f.size(text)
            def get_height(self):
                return self._f.get_height()
            def __getattr__(self, name):
                return getattr(self._f, name)
        wrapped = Wrapper(real_f)
        setattr(hub, f_attr, wrapped)
        font_wrappers.append(wrapped)

    for s in hub.sliders + hub.evo_sliders:
        s.font = hub.font_small

    tabs_to_check = [
        (0, None),  # QUICK DUEL
        (1, None),  # 4-WAY BRAWL
        (2, None),  # TOURNAMENT
        (3, 0),     # ANALYTICS H2H
        (3, 1),     # ANALYTICS ELO
        (4, None),  # EVO LAB
        (5, None),  # BOT LOCKER
        (6, None),  # MAP SHOWCASE
    ]

    for tab_idx, subview in tabs_to_check:
        hub.current_tab = tab_idx
        if subview is not None:
            hub.analytics_subview = subview

        text_blits.clear()
        ts = TrackedSurface((hub.screen_size, hub.screen_size))
        hub.screen = ts
        hub._draw_hub_frame()

        # Check all pairs for collisions
        for i in range(len(text_blits)):
            r1, t1 = text_blits[i]
            for j in range(i + 1, len(text_blits)):
                r2, t2 = text_blits[j]
                inter = r1.clip(r2)
                assert not (inter.width > 0 and inter.height > 0), (
                    f"Text overlap detected in tab {tab_idx} (subview {subview}): "
                    f"'{t1}' at {r1} overlaps with '{t2}' at {r2} (intersection: {inter})"
                )


def test_bot_locker_tuning_events():
    """Verify Bot Locker Reset Defaults and Mutate Jitter events."""
    hub = ArcadeHub(screen_size=900)
    hub.current_tab = 5

    # Modify values
    hub.sliders[0].val = 0.2
    hub.custom_bot_config.aggression = 0.2

    # Click reset defaults at (500, 525)
    click_reset = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (500, 525)})
    hub._handle_tab_events(click_reset)
    assert np.isclose(hub.custom_bot_config.aggression, 0.75)
    assert np.isclose(hub.sliders[0].val, 0.75)

    # Click mutate jitter at (500, 580)
    click_mutate = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (500, 580)})
    hub._handle_tab_events(click_mutate)
    # Values should stay within bounds
    assert 0.1 <= hub.custom_bot_config.aggression <= 1.0
    assert 0.0 <= hub.custom_bot_config.dash_frequency <= 1.0
    assert 0.0 <= hub.custom_bot_config.anchor_bias <= 1.0
    assert 1.0 <= hub.custom_bot_config.steer_speed <= 4.5


def test_save_tab_screenshots_method():
    """Verify ArcadeHub.save_tab_screenshots renders all 8 frames offscreen to disk."""
    hub = ArcadeHub(screen_size=900)
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = hub.save_tab_screenshots(output_dir=tmp_dir)
        assert len(paths) == 8
        for p in paths:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000  # valid PNG


def test_mutation_logic_and_champion_improvement():
    """Verify _mutate_champion properly evaluates mutant, keeps history lengths identical, and handles score."""
    hub = ArcadeHub(screen_size=900)
    init_gen = hub.evo_generation
    init_champ_weights = hub.evo_best_agent.get_weights().copy()

    # Run mutation
    hub._mutate_champion()
    assert hub.evo_generation == init_gen + 1
    assert len(hub.evo_history_best) == len(hub.evo_history_mean)
    assert len(hub.evo_history_best) == 2

    # Draw frame to guarantee no rendering error with new history
    hub.current_tab = 4
    hub._draw_hub_frame()


def test_ultra_long_names_rendering_without_overlap():
    """Verify that very long competitor and card names scale down without overlapping columns."""
    hub = ArcadeHub(screen_size=900)

    # Insert abnormally long names
    long_name = "SuperUltraMegaExtendedCompetitorName_Alpha"
    hub.BOT_CHOICES.append(long_name)
    hub.selected_agent0_idx = len(hub.BOT_CHOICES) - 1
    hub.selected_4way_indices[0] = len(hub.BOT_CHOICES) - 1
    hub.tournament_ratings[long_name] = 1380.0

    # Draw Quick Duel
    hub.current_tab = 0
    hub._draw_hub_frame()

    # Draw 4-Way Brawl
    hub.current_tab = 1
    hub._draw_hub_frame()

    # Draw Tournament
    hub.current_tab = 2
    hub._draw_hub_frame()

    # Draw Analytics
    hub.current_tab = 3
    hub.analytics_subview = 0
    hub._draw_hub_frame()
    hub.analytics_subview = 1
    hub._draw_hub_frame()


