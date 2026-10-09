"""Interactive launcher for Human vs AI and AI vs AI duels in Kinetic Sumo."""

import os
import argparse
import numpy as np
import pygame

from src.engine.entities import DiscAgent
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy
from src.maps import get_map, MAP_REGISTRY
from src.ui.renderer import SumoRenderer
from src.env.arena_env import KineticSumoEnv


def get_human_action() -> np.ndarray:
    """Read keyboard input for human player.

    W/Up: Forward thrust (+1.0)
    S/Down: Reverse brake (-0.8)
    A/Left: Steer counter-clockwise (+1.0)
    D/Right: Steer clockwise (-1.0)
    Space: Dash (+1.0)
    Left Shift: Anchor (-1.0)
    """
    keys = pygame.key.get_pressed()

    thrust = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        thrust = 1.0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        thrust = -0.8

    steer = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer = 1.0  # counter-clockwise
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer = -1.0  # clockwise

    ability = 0.0
    if keys[pygame.K_SPACE]:
        ability = 1.0  # Dash
    elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        ability = -1.0  # Anchor

    return np.array([thrust, steer, ability], dtype=np.float32)


def load_agent(agent_type: str, checkpoint_path: str | None = None) -> BaseAgent:
    """Factory helper to instantiate chosen agent type."""
    agent_type = agent_type.lower()
    if agent_type == "heuristic":
        return HeuristicAgent()
    elif agent_type == "ga":
        if checkpoint_path and os.path.exists(checkpoint_path):
            return GAMLP.load(checkpoint_path)
        return GAMLP()
    elif agent_type == "ppo":
        if checkpoint_path and os.path.exists(checkpoint_path):
            return PPOPolicy.load(checkpoint_path)
        return PPOPolicy()
    elif agent_type == "random":
        from src.eval.tournament import RandomAgent
        return RandomAgent()
    else:
        raise ValueError(f"Unknown agent type '{agent_type}'. Choices: heuristic, ga, ppo, random")


def run_interactive_match(
    agent0_type: str = "human",
    agent1_type: str = "heuristic",
    map_name: str = "Open",
    ga_path: str = "checkpoints/ga/best_ga_agent.npy",
    ppo_path: str = "checkpoints/ppo/best_ppo_policy.pt",
) -> None:
    """Run interactive Pygame match session."""
    agent0 = None if agent0_type == "human" else load_agent(agent0_type, ga_path if agent0_type == "ga" else ppo_path)
    agent1 = load_agent(agent1_type, ga_path if agent1_type == "ga" else ppo_path)

    env = KineticSumoEnv(map_name=map_name, render_mode="human")
    obs, _ = env.reset()

    if agent0:
        agent0.reset()
    agent1.reset()

    running = True
    print("\n--- Kinetic Sumo Interactive Match ---")
    print("Controls:")
    print("  W / S : Forward / Brake")
    print("  A / D : Steer CCW / CW")
    print("  SPACE : Dash (2x speed burst, steering locked)")
    print("  LSHIFT: Anchor (4x friction, absorbs hits)")
    print("  L     : Toggle Lidar Debug Overlay")
    print("  R     : Reset Match")
    print("  ESC   : Quit\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    if agent0:
                        agent0.reset()
                    agent1.reset()
                elif event.key == pygame.K_l:
                    if env._renderer is not None:
                        env._renderer.show_lidar_overlay = not env._renderer.show_lidar_overlay

        # Action selection
        if not env.agents:
            # Match finished - continue rendering end state until reset or keypress
            env.render()
            pygame.time.delay(20)
            continue

        act_0 = get_human_action() if agent0_type == "human" else agent0.act(obs["agent_0"])
        act_1 = agent1.act(obs["agent_1"])

        obs, rewards, terms, truncs, infos = env.step({"agent_0": act_0, "agent_1": act_1})
        env.render()

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Kinetic Sumo Interactive Play Launcher")
    parser.add_argument("--agent0", type=str, default="human", help="human, heuristic, ga, ppo")
    parser.add_argument("--agent1", type=str, default="heuristic", help="heuristic, ga, ppo, random")
    parser.add_argument("--map", type=str, default="Open", choices=list(MAP_REGISTRY.keys()))
    parser.add_argument("--ga-path", type=str, default="checkpoints/ga/best_ga_agent.npy")
    parser.add_argument("--ppo-path", type=str, default="checkpoints/ppo/best_ppo_policy.pt")
    args = parser.parse_args()

    run_interactive_match(
        agent0_type=args.agent0,
        agent1_type=args.agent1,
        map_name=args.map,
        ga_path=args.ga_path,
        ppo_path=args.ppo_path,
    )


if __name__ == "__main__":
    main()
