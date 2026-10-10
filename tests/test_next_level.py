"""Unit tests for Next-Level features: Tactical Bots, 4-Way FFA Env, Deterministic Maps, Audio, Dropdowns, AI X-Ray, and Parameter Helpers."""

import numpy as np
import pygame
import pytest

from src.agents.tactical import (
    BulldozerAgent,
    ShinobiAgent,
    GuardianAgent,
    MatadorAgent,
    RicochetAgent,
    ChameleonAgent,
    TunableAgent,
    BotConfig,
)
from src.agents.ga_mlp import GAMLP
from src.maps import get_map, MAP_REGISTRY
from src.env.arena_env import KineticSumoEnv
from src.env.arena_4way_env import KineticSumo4WayEnv
from src.ui.audio import SoundEngine
from src.ui.hub import ArcadeHub
from src.ui.play import load_agent, extract_xray_info


def test_tactical_bots_action_space():
    """Verify all 7 tactical agents produce valid deterministic actions in [-1, 1]^3 and expose X-Ray intent."""
    dummy_obs = np.zeros(46, dtype=np.float32)
    # Set mock heading to positive X
    dummy_obs[4] = 1.0  # cos_th
    # Opponent at (2.0, 0.0)
    dummy_obs[7] = 0.1  # dx
    dummy_obs[12] = 1.0  # energy

    bots = [
        BulldozerAgent(),
        ShinobiAgent(),
        GuardianAgent(),
        MatadorAgent(),
        RicochetAgent(),
        ChameleonAgent(),
        TunableAgent(BotConfig(aggression=0.9, dash_frequency=0.8)),
    ]

    for bot in bots:
        bot.reset()
        action = bot.act(dummy_obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        assert isinstance(bot.last_intent, str) and len(bot.last_intent) > 0
        assert isinstance(bot.target_vec, np.ndarray) and bot.target_vec.shape == (2,)
        # Verify strict determinism
        action2 = bot.act(dummy_obs)
        assert np.allclose(action, action2)


def test_matador_ricochet_chameleon_distinct_behaviors():
    """Verify Matador, Ricochet, and Chameleon trigger their signature tactical modes."""
    # 1. Matador near boundary with charging opponent -> triggers OLE! SIDESTEP
    obs_edge = np.zeros(46, dtype=np.float32)
    obs_edge[0] = 0.70   # near right edge (r_self = 7.0m)
    obs_edge[4] = -0.34  # facing tangent juke direction (pi/2 + 0.35)
    obs_edge[5] = 0.94
    obs_edge[7] = -0.15  # opponent 3.0m away
    obs_edge[9] = 0.45   # opponent charging toward Matador
    obs_edge[12] = 1.0   # full energy
    matador = MatadorAgent()
    act_m = matador.act(obs_edge)
    assert matador.last_intent == "OLE! SIDESTEP"
    assert act_m[2] == 1.0  # triggers juke dash

    # 2. Ricochet with close Lidar obstacle -> triggers LIDAR WEAVE
    obs_lidar = np.zeros(46, dtype=np.float32)
    obs_lidar[4] = 1.0
    obs_lidar[7] = 0.25
    obs_lidar[12] = 1.0
    obs_lidar[14:30] = 1.0
    # Ray 0 has an obstacle (type 0.5 at index 30) very close (dist 0.08 at index 14)
    obs_lidar[14] = 0.08
    obs_lidar[30] = 0.5
    ricochet = RicochetAgent()
    act_r = ricochet.act(obs_lidar)
    assert ricochet.last_intent == "LIDAR WEAVE"
    assert act_r.shape == (3,)

    # 3. Chameleon switches stances based on combat state
    chameleon = ChameleonAgent()
    # Opponent near edge -> STANCE: BULLDOZER
    obs_opp_edge = np.zeros(46, dtype=np.float32)
    obs_opp_edge[0] = 0.1
    obs_opp_edge[4] = 1.0
    obs_opp_edge[7] = 0.35  # opp_pos = 0.1*10 + 0.35*20 = 8.0m (> 0.68 * 10m)
    obs_opp_edge[12] = 1.0
    chameleon.act(obs_opp_edge)
    assert "BULLDOZER" in chameleon.last_intent

    # Opponent rushing head-on -> STANCE: GUARDIAN
    obs_rush = np.zeros(46, dtype=np.float32)
    obs_rush[4] = 1.0
    obs_rush[7] = 0.12  # 2.4m away
    obs_rush[9] = -0.35  # closing fast at 3.5 m/s
    obs_rush[12] = 1.0
    chameleon.act(obs_rush)
    assert "GUARDIAN" in chameleon.last_intent


def test_gamlp_sensory_importance_and_salience():
    """Verify GAMLP computes normalized sensory group importance and 46-channel salience."""
    ga = GAMLP()
    imp = ga.get_sensory_importance()
    assert set(imp.keys()) == {"Self Kinematics", "Opponent Rel", "Resources", "16-Ray Lidar"}
    assert np.isclose(sum(imp.values()), 1.0, atol=1e-4)

    sal = ga.get_input_salience()
    assert sal.shape == (46,)
    assert np.all(sal >= 0.0) and np.all(sal <= 1.0)


def test_ai_xray_overlay_and_reward_breakdown_rendering():
    """Verify KineticSumoEnv tracks reward breakdowns and renders AI X-Ray overlay cleanly."""
    env = KineticSumoEnv(map_name="Bumper_Coliseum", render_mode="rgb_array")
    obs, _ = env.reset(seed=7)

    bot0 = MatadorAgent()
    bot1 = ChameleonAgent()
    a0 = bot0.act(obs["agent_0"])
    a1 = bot1.act(obs["agent_1"])
    obs, rews, _, _, _ = env.step({"agent_0": a0, "agent_1": a1})

    assert "agent_0" in env.reward_calc.last_breakdown
    bd0 = env.reward_calc.last_breakdown["agent_0"]
    assert all(k in bd0 for k in ("total", "pbrs", "contact", "penalties", "terminal"))

    xray = extract_xray_info({"agent_0": bot0, "agent_1": bot1})
    frame = env.render(
        agent_names=("Matador", "Chameleon"),
        subtitle="X-Ray Test",
        xray_info=xray,
        sim_status={"paused": True, "speed_mult": 0.5},
    )
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (900, 900, 3)
    env.close()


def test_dropdowns_presets_and_parameter_helpers_in_hub():
    """Verify UIDropdown selection, Bot Locker preset saving, and Evo/Locker parameter helpers."""
    hub = ArcadeHub(screen_size=900)

    # 1. Test opening and selecting an option in Quick Duel Player 1 Dropdown
    hub.current_tab = 0
    assert hub.dd_duel_p1.is_open is False
    open_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (200, 220)})
    hub._handle_tab_events(open_click)
    assert hub.dd_duel_p1.is_open is True
    # Render with open dropdown overlay
    hub._draw_hub_frame()

    # Click second visible item in dropdown menu
    m_rect = hub.dd_duel_p1.get_menu_rect(hub.screen_size)
    select_click = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        {"button": 1, "pos": (m_rect.centerx, m_rect.y + 4 + int(1.5 * hub.dd_duel_p1.row_height))},
    )
    hub._handle_tab_events(select_click)
    assert hub.dd_duel_p1.is_open is False
    assert hub.selected_agent0_idx == 1

    # 2. Test saving a custom preset in Bot Locker
    hub.current_tab = 5
    hub.sliders[0].val = 0.92
    hub.custom_bot_config.aggression = 0.92
    preset_name = hub._save_custom_bot_preset()
    assert preset_name == "Custom_Preset_1"
    assert preset_name in hub.BOT_CHOICES
    assert preset_name in hub.tournament_ratings
    inst = hub._instantiate_agent(preset_name)
    assert isinstance(inst, TunableAgent)
    assert np.isclose(inst.config.aggression, 0.92)

    # 3. Verify all 7 Evo Lab parameter helper states and 4 Bot Locker helper states render cleanly
    hub.current_tab = 4
    for idx in range(len(hub.evo_sliders)):
        hub.active_evo_slider_idx = idx
        hub._draw_hub_frame()

    hub.current_tab = 5
    for idx in range(len(hub.sliders)):
        hub.active_locker_slider_idx = idx
        hub._draw_hub_frame()


def test_new_deterministic_maps():
    """Verify Colosseum and Crossfire arena presets."""
    assert "Colosseum" in MAP_REGISTRY
    assert "Crossfire" in MAP_REGISTRY

    colosseum = get_map("Colosseum")
    assert len(colosseum.obstacles) == 4
    spawns_4w = colosseum.get_4way_spawn_transforms()
    assert len(spawns_4w) == 4
    for pos, angle in spawns_4w:
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)

    crossfire = get_map("Crossfire")
    assert len(crossfire.obstacles) == 6
    for obs in crossfire.obstacles:
        assert obs.is_bumper is True
        assert obs.radius > 0.0


def test_4way_environment_lifecycle():
    """Test full reset and step lifecycle for KineticSumo4WayEnv."""
    env = KineticSumo4WayEnv(map_name="Colosseum", max_duration=10.0)
    obs, infos = env.reset(seed=42)

    assert len(obs) == 4
    for a_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
        assert a_id in obs
        assert obs[a_id].shape == (46,)
        assert obs[a_id].dtype == np.float32

    actions = {a_id: np.array([1.0, 0.0, 0.0], dtype=np.float32) for a_id in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)

    assert len(next_obs) == 4
    assert len(rewards) == 4
    assert len(terms) == 4
    assert len(truncs) == 4

    env.close()


def test_procedural_sound_engine():
    """Verify sound engine initializes, handles waveforms, and toggles mute."""
    audio = SoundEngine(enabled=False)
    assert audio.is_muted is True
    audio.toggle_mute()
    assert audio.is_muted is False

    audio.play_clang(is_flank=False, intensity=0.5)
    audio.play_clang(is_flank=True, intensity=1.0)
    audio.play_dash()
    audio.play_anchor()
    audio.play_ringout()
    audio.play_countdown()


def test_neural_brain_snapshot_causal_attribution_and_crossover():
    """Verify GAMLP/PPOPolicy/Progressive brain snapshots, Top-3 causal attribution, and 2-parent crossover."""
    from src.agents.ga_mlp import GAMLP, synthesize_tactical_brain_snapshot
    from src.agents.ppo_policy import PPOPolicy
    from src.agents.progressive import ApexAgent

    obs = np.random.uniform(-1.0, 1.0, size=(46,)).astype(np.float32)

    # 1. GAMLP Brain Snapshot & Causal Attribution
    ga_a = GAMLP()
    ga_b = GAMLP()
    out_a = ga_a.act(obs)
    assert out_a.shape == (3,)
    assert ga_a.last_h1.shape == (32,)
    assert ga_a.last_h2.shape == (32,)

    snap_ga = ga_a.get_brain_snapshot()
    assert snap_ga["arch"] == "46->32->32->3"
    assert len(snap_ga["clusters"]) == 8
    assert snap_ga["cluster_w1"].shape == (8, 32)
    assert len(snap_ga["top_drivers"]) == 3
    for c_name, share, act_name in snap_ga["top_drivers"]:
        assert isinstance(c_name, str)
        assert 0.0 <= share <= 1.0
        assert act_name in ("THR", "STR", "ABL")

    # 2. 2-Parent Synaptic Crossover
    child = ga_a.crossover(ga_b, crossover_rate=0.5)
    assert isinstance(child, GAMLP)
    assert child.get_weights().shape == ga_a.get_weights().shape
    assert not np.allclose(child.get_weights(), ga_a.get_weights())
    assert not np.allclose(child.get_weights(), ga_b.get_weights())

    # 3. PPOPolicy Brain Snapshot
    ppo = PPOPolicy()
    ppo.act(obs)
    snap_ppo = ppo.get_brain_snapshot()
    assert "PPO" in snap_ppo["arch"]
    assert len(snap_ppo["top_drivers"]) == 3

    # 4. Progressive Agent & Tactical Cortex Fallback
    apex = ApexAgent()
    apex.act(obs)
    snap_apex = apex.get_brain_snapshot()
    assert len(snap_apex["clusters"]) == 8
    snap_tac = synthesize_tactical_brain_snapshot(obs, out_a, intent="MATADOR OLE")
    assert snap_tac["intent"] == "MATADOR OLE"
    assert len(snap_tac["top_drivers"]) == 3


def test_live_match_brain_overlay_and_evo_brain_probe():
    """Verify [B] Brain View overlay in renderer and Interactive Brain Probe + Gauntlet Curriculum in Evo Lab."""
    env = KineticSumoEnv(map_name="Open", render_mode="rgb_array")
    obs, _ = env.reset(seed=7)
    b0 = GAMLP()
    b1 = MatadorAgent()
    a0 = b0.act(obs["agent_0"])
    a1 = b1.act(obs["agent_1"])
    env.step({"agent_0": a0, "agent_1": a1})

    xray = extract_xray_info({"agent_0": b0, "agent_1": b1}, obs, {"agent_0": a0, "agent_1": a1})
    assert "brain" in xray["agent_0"]
    assert "brain" in xray["agent_1"]

    # Enable show_brain_overlay on renderer
    frame_init = env.render(xray_info=xray)
    assert env._renderer is not None
    env._renderer.show_brain_overlay = True
    frame_brain = env.render(xray_info=xray, sim_status="SPEED:1.0x")
    assert isinstance(frame_brain, np.ndarray)
    assert frame_brain.shape == (900, 900, 3)
    env.close()

    # Verify Evo Lab Brain Topology & Scenario Probe subview and Gauntlet Curriculum
    hub = ArcadeHub(screen_size=900)
    hub.current_tab = 4
    assert "Gauntlet_Curriculum" in hub.evo_sparring_choices

    # Click BRAIN TOPOLOGY & PROBE toggle button
    click_brain_view = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (680, 175)})
    hub._handle_tab_events(click_brain_view)
    assert hub.evo_subview == 1
    hub._draw_hub_frame()

    # Click each of the 4 Scenario Probe buttons
    probe_xs = [450, 560, 670, 780]
    for expected_idx, px in enumerate(probe_xs):
        click_probe = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (px, 208)})
        hub._handle_tab_events(click_probe)
        assert hub.evo_probe_idx == expected_idx
        hub._draw_hub_frame()

    # Run 1 generation against Gauntlet_Curriculum using 2-parent crossover
    hub.evo_sparring_idx = hub.evo_sparring_choices.index("Gauntlet_Curriculum")
    hub.evo_pop_size = 6
    hub.evo_population = hub.evo_population[:6]
    gen_before = hub.evo_generation
    hub._evo_step_generation(turbo_count=1)
    assert hub.evo_generation == gen_before + 1
    assert "Crossover+Mut" in hub.evo_status_text

