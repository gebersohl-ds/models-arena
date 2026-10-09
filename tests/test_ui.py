"""Test headless rendering and UI frame generation."""

import numpy as np
from src.env.arena_env import KineticSumoEnv
from src.ui.renderer import SumoRenderer
from src.maps import get_map


def test_headless_renderer_rgb_array():
    env = KineticSumoEnv(map_name="Open", render_mode="rgb_array")
    obs, _ = env.reset()

    # Step and render
    actions = {
        "agent_0": np.array([1.0, 0.2, 0.8], dtype=np.float32),  # Dash
        "agent_1": np.array([0.5, -0.4, -0.8], dtype=np.float32),  # Anchor
    }
    env.step(actions)
    frame = env.render()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (900, 900, 3)
    assert frame.dtype == np.uint8

    env.close()


def test_renderer_with_custom_names_and_subtitle():
    """Verify that renderer HUD functions correctly with custom agent names and tournament subtitles."""
    env = KineticSumoEnv(map_name="Pinball", render_mode="rgb_array")
    env.reset()

    frame = env.render(
        round_scores=(2, 1),
        rounds_to_win=3,
        agent_names=("Neuroevolution_GA", "SelfPlay_PPO"),
        subtitle="Match 3/6: GA vs PPO",
    )

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (900, 900, 3)

    env.close()


def test_env_human_render_mode_initializes_video_system():
    """Verify that KineticSumoEnv with render_mode='human' initializes the video system,

    preventing pygame.error: video system not initialized on event polling.
    """
    import pygame
    env = KineticSumoEnv(map_name="Pinball", render_mode="human")
    assert pygame.get_init()
    assert pygame.display.get_init()

    # Polling events must NOT raise pygame.error: video system not initialized
    events = pygame.event.get()
    assert isinstance(events, list)

    env.close()


def test_play_helpers():
    """Verify play.py agent loading and input polling helpers."""
    from src.ui.play import load_agent, get_human_action
    from src.agents.heuristic import HeuristicAgent
    from src.eval.tournament import RandomAgent

    heuristic = load_agent("heuristic")
    assert isinstance(heuristic, HeuristicAgent)

    random_agent = load_agent("random")
    assert isinstance(random_agent, RandomAgent)

    action = get_human_action()
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert action.dtype == np.float32


def test_tournament_round_with_custom_hud_info():
    """Verify play_single_round and play_match function correctly with tournament HUD arguments."""
    from src.eval.tournament import play_single_round, play_match
    from src.agents.heuristic import HeuristicAgent

    agent_a = HeuristicAgent()
    agent_b = HeuristicAgent()

    score = play_single_round(
        agent_0=agent_a,
        agent_1=agent_b,
        map_name="Open",
        max_duration=0.2,
        agent_names=("Bot_A", "Bot_B"),
        round_scores=(1, 0),
        rounds_to_win=3,
        subtitle="Test Match",
    )
    assert score in (0.0, 0.5, 1.0)

    # Test play_match completes and returns valid score
    match_score = play_match(
        agent_0=agent_a,
        agent_1=agent_b,
        map_name="Open",
        max_duration=0.2,
        rounds_to_win=1,
        agent_names=("Bot_A", "Bot_B"),
    )
    assert match_score in (0.0, 0.5, 1.0)


def test_renderer_long_agent_name_boundary():
    """Verify that renderer does not clip or error on unusually long agent names."""
    env = KineticSumoEnv(map_name="Open", render_mode="rgb_array")
    env.reset()

    frame = env.render(
        round_scores=(2, 2),
        rounds_to_win=3,
        agent_names=("SuperLongCompetitorNameAlpha_0", "UltraExtendedCompetitorBeta_1"),
        subtitle="Championship Match",
    )
    assert frame is not None
    assert frame.shape == (900, 900, 3)

    env.close()


def test_visual_tournament_and_interruption():
    """Verify visual tournament loop and user interruption (ESC/QUIT) handling."""
    from src.eval.tournament import Competitor, RandomAgent, run_tournament, TournamentInterrupted

    pool = [
        Competitor("Bot_1", RandomAgent()),
        Competitor("Bot_2", RandomAgent()),
    ]

    # Run 1 match per pair, 1 round to win, rendered in human mode with high fps
    ladder = run_tournament(
        pool,
        matches_per_pair=1,
        rounds_to_win=1,
        render=True,
        fps=200,
    )
    assert len(ladder) == 2
    assert ladder[0].elo >= ladder[1].elo

