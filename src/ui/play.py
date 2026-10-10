"""Interactive launcher for Human vs AI, 4-Way Battle Royale, and Arcade Hub in Kinetic Sumo."""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pygame

# Ensure repository root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine.entities import DiscAgent
from src.agents.base import BaseAgent
from src.agents.heuristic import HeuristicAgent
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
from src.agents.ga_mlp import GAMLP, synthesize_tactical_brain_snapshot
from src.agents.ppo_policy import PPOPolicy
from src.maps import get_map, MAP_REGISTRY
from src.ui.renderer import SumoRenderer
from src.env.arena_env import KineticSumoEnv
from src.env.arena_4way_env import KineticSumo4WayEnv


SPEED_LEVELS = [0.25, 0.5, 1.0, 2.0, 4.0]


def extract_xray_info(
    agents_map: dict[str, BaseAgent | None],
    last_obs: dict[str, np.ndarray] | None = None,
    last_actions: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Extract live behind-the-scenes intent, target vector, and neural brain snapshot from agent instances."""
    info: dict[str, dict] = {}
    for a_id, bot in agents_map.items():
        obs_i = (last_obs or {}).get(a_id)
        act_i = (last_actions or {}).get(a_id)
        if bot is None:
            info[a_id] = {
                "intent": "HUMAN PILOT",
                "target_vec": None,
                "brain": synthesize_tactical_brain_snapshot(obs_i, act_i, intent="HUMAN PILOT"),
            }
        else:
            intent = getattr(bot, "last_intent", bot.__class__.__name__.replace("Agent", "").upper())
            if hasattr(bot, "get_brain_snapshot"):
                brain_snap = bot.get_brain_snapshot()
            else:
                brain_snap = synthesize_tactical_brain_snapshot(obs_i, act_i, intent=intent)
            info[a_id] = {
                "intent": intent,
                "target_vec": getattr(bot, "target_vec", None),
                "brain": brain_snap,
            }
    return info


def get_human_action(player_idx: int = 0) -> np.ndarray:
    """Read keyboard input for human player.

    Player 0 (P1):
      W / S: Forward thrust (+1.0) / Reverse brake (-0.8)
      A / D: Steer CCW (+1.0) / Steer CW (-1.0)
      Space: Dash (+1.0)
      Left Shift: Anchor (-1.0)

    Player 1 (P2):
      Up / Down: Forward thrust (+1.0) / Reverse brake (-0.8)
      Left / Right: Steer CCW (+1.0) / Steer CW (-1.0)
      Return / Right Ctrl: Dash (+1.0)
      Right Shift: Anchor (-1.0)
    """
    if not pygame.get_init():
        return np.zeros(3, dtype=np.float32)
    keys = pygame.key.get_pressed()

    thrust = 0.0
    steer = 0.0
    ability = 0.0

    if player_idx == 0:
        # P1 controls
        if keys[pygame.K_w]:
            thrust = 1.0
        elif keys[pygame.K_s]:
            thrust = -0.8

        if keys[pygame.K_a]:
            steer = 1.0
        elif keys[pygame.K_d]:
            steer = -1.0

        if keys[pygame.K_SPACE]:
            ability = 1.0
        elif keys[pygame.K_LSHIFT]:
            ability = -1.0

    else:
        # P2 controls
        if keys[pygame.K_UP]:
            thrust = 1.0
        elif keys[pygame.K_DOWN]:
            thrust = -0.8

        if keys[pygame.K_LEFT]:
            steer = 1.0
        elif keys[pygame.K_RIGHT]:
            steer = -1.0

        if keys[pygame.K_RETURN] or keys[pygame.K_RCTRL] or keys[pygame.K_KP_ENTER]:
            ability = 1.0
        elif keys[pygame.K_RSHIFT]:
            ability = -1.0

    return np.array([thrust, steer, ability], dtype=np.float32)


def load_agent(
    agent_type: str,
    checkpoint_path: str | None = None,
    custom_bot_config: BotConfig | None = None,
    preset_configs: dict[str, BotConfig] | None = None,
) -> BaseAgent:
    """Factory helper to instantiate chosen agent type."""
    raw_name = agent_type.strip()
    if preset_configs and raw_name in preset_configs:
        return TunableAgent(preset_configs[raw_name])
    agent_type = raw_name.lower()
    if preset_configs:
        for k, cfg in preset_configs.items():
            if k.lower() == agent_type:
                return TunableAgent(cfg)

    if agent_type == "heuristic":
        return HeuristicAgent()
    elif agent_type == "bulldozer":
        return BulldozerAgent()
    elif agent_type == "shinobi":
        return ShinobiAgent()
    elif agent_type == "guardian":
        return GuardianAgent()
    elif agent_type == "matador":
        return MatadorAgent()
    elif agent_type == "ricochet":
        return RicochetAgent()
    elif agent_type == "chameleon":
        return ChameleonAgent()
    elif agent_type == "tunable" or agent_type.startswith("preset_"):
        return TunableAgent(custom_bot_config)
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
    elif "apex" in agent_type:
        from src.agents.progressive import ApexAgent
        return ApexAgent(checkpoint_path="checkpoints/ga/apex_champion.npy")
    elif "contender" in agent_type:
        from src.agents.progressive import ContenderAgent
        return ContenderAgent(checkpoint_path="checkpoints/ga/contender_striker.npy")
    elif "rookie" in agent_type:
        from src.agents.progressive import RookieAgent
        return RookieAgent(checkpoint_path="checkpoints/ga/rookie_ga.npy")
    elif "evo" in agent_type:
        evo_path = "checkpoints/ga/evo_champion.npy"
        if os.path.exists(evo_path):
            return GAMLP.load(evo_path)
        return GAMLP()
    else:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. Choices: heuristic, bulldozer, shinobi, guardian, matador, ricochet, chameleon, tunable, ga, ppo, random, apex, contender, rookie, evo"
        )


def run_interactive_match(
    agent0_type: str = "human",
    agent1_type: str = "heuristic",
    map_name: str = "Open",
    ga_path: str = "checkpoints/ga/best_ga_agent.npy",
    ppo_path: str = "checkpoints/ppo/best_ppo_policy.pt",
    rounds_to_win: int = 3,
    return_to_hub: bool = False,
    custom_bot_config: BotConfig | None = None,
    preset_configs: dict[str, BotConfig] | None = None,
) -> None:
    """Run interactive 1v1 Pygame match session."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    if not pygame.display.get_init():
        pygame.display.init()

    is_p0_human = "human" in agent0_type.lower()
    p0_idx = 1 if "p2" in agent0_type.lower() else 0
    is_p1_human = "human" in agent1_type.lower()
    p1_idx = 1 if "p2" in agent1_type.lower() else 0

    agent0 = (
        None
        if is_p0_human
        else load_agent(
            agent0_type,
            ga_path if agent0_type == "ga" else ppo_path,
            custom_bot_config=custom_bot_config,
            preset_configs=preset_configs,
        )
    )
    agent1 = (
        None
        if is_p1_human
        else load_agent(
            agent1_type,
            ga_path if agent1_type == "ga" else ppo_path,
            custom_bot_config=custom_bot_config,
            preset_configs=preset_configs,
        )
    )

    agent0_label = f"Human (P{p0_idx + 1})" if is_p0_human else f"AI ({agent0_type.upper()})"
    agent1_label = f"Human (P{p1_idx + 1})" if is_p1_human else f"AI ({agent1_type.upper()})"
    agent_names = (agent0_label, agent1_label)

    env = KineticSumoEnv(map_name=map_name, render_mode="human")
    obs, _ = env.reset()

    if agent0:
        agent0.reset()
    if agent1:
        agent1.reset()

    running = True
    paused = False
    step_once = False
    speed_idx = 2  # 1.0x default
    wins_0 = 0
    wins_1 = 0
    round_over_timer = 0.0

    print("\n--- Kinetic Sumo 1v1 Interactive Match ---")
    print(f"Match Rule: First to {rounds_to_win} rounds wins!")
    print("Controls:")
    print("  P1: W/S (drive/brake), A/D (steer), Space (dash), LShift (anchor)")
    print("  P2: Up/Down (drive/brake), Left/Right (steer), Enter (dash), RShift (anchor)")
    print("  I: Toggle AI X-Ray  |  B: Toggle Brain View  |  L: Toggle Lidar  |  P: Pause  |  .: Step  |  -/+: Speed\n")

    agents_map = {"agent_0": agent0, "agent_1": agent1}
    env.render(
        round_scores=(wins_0, wins_1),
        rounds_to_win=rounds_to_win,
        agent_names=agent_names,
        xray_info=extract_xray_info(agents_map, obs, getattr(env, "_last_actions", None)),
        sim_status=f"SPEED:{SPEED_LEVELS[speed_idx]}x",
    )

    clock = pygame.time.Clock()

    while running:
        speed_mult = SPEED_LEVELS[speed_idx]
        target_fps = max(12, int(50 * speed_mult))
        dt = clock.tick(target_fps if not paused else 30) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                if not return_to_hub:
                    sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    if env._renderer is not None:
                        env._renderer.audio.toggle_mute()
                elif event.key == pygame.K_l:
                    if env._renderer is not None:
                        env._renderer.show_lidar_overlay = not env._renderer.show_lidar_overlay
                elif event.key == pygame.K_i:
                    if env._renderer is not None:
                        env._renderer.show_xray_overlay = not env._renderer.show_xray_overlay
                elif event.key == pygame.K_b:
                    if env._renderer is not None:
                        env._renderer.show_brain_overlay = not env._renderer.show_brain_overlay
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key in (pygame.K_PERIOD, pygame.K_n):
                    paused = True
                    step_once = True
                elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET, pygame.K_KP_MINUS):
                    speed_idx = max(0, speed_idx - 1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET, pygame.K_KP_PLUS):
                    speed_idx = min(len(SPEED_LEVELS) - 1, speed_idx + 1)
                elif event.key == pygame.K_r:
                    wins_0 = 0
                    wins_1 = 0
                    round_over_timer = 0.0
                    obs, _ = env.reset()
                    if agent0:
                        agent0.reset()
                    if agent1:
                        agent1.reset()

        if not running:
            break

        sim_status = f"PAUSED ({SPEED_LEVELS[speed_idx]}x)" if paused else f"SPEED:{SPEED_LEVELS[speed_idx]}x"

        if paused and not step_once:
            env.render(
                round_scores=(wins_0, wins_1),
                rounds_to_win=rounds_to_win,
                agent_names=agent_names,
                subtitle="[PAUSED — Press P to Resume or . to Step 1 Frame]",
                fps=30,
                xray_info=extract_xray_info(agents_map, obs, getattr(env, "_last_actions", None)),
                sim_status=sim_status,
            )
            continue

        step_once = False

        # Between-rounds handling
        if not env.agents:
            match_won = (wins_0 >= rounds_to_win) or (wins_1 >= rounds_to_win)
            subtitle = None
            if match_won:
                final_winner = agent_names[0] if wins_0 >= rounds_to_win else agent_names[1]
                subtitle = f"MATCH OVER: {final_winner} WINS ({wins_0}-{wins_1})! [R] Restart  [ESC] Menu"

            env.render(
                round_scores=(wins_0, wins_1),
                rounds_to_win=rounds_to_win,
                agent_names=agent_names,
                subtitle=subtitle,
                fps=target_fps,
                xray_info=extract_xray_info(agents_map, obs, getattr(env, "_last_actions", None)),
                sim_status=sim_status,
            )
            round_over_timer += dt

            if not match_won and round_over_timer >= 1.5:
                round_over_timer = 0.0
                obs, _ = env.reset()
                if agent0:
                    agent0.reset()
                if agent1:
                    agent1.reset()
            continue

        act_0 = get_human_action(p0_idx) if is_p0_human else agent0.act(obs["agent_0"])
        act_1 = get_human_action(p1_idx) if is_p1_human else agent1.act(obs["agent_1"])

        obs, rewards, terms, truncs, infos = env.step({"agent_0": act_0, "agent_1": act_1})

        if not env.agents:
            winner = infos.get("agent_0", {}).get("winner")
            if winner == "agent_0":
                wins_0 += 1
            elif winner == "agent_1":
                wins_1 += 1
            round_over_timer = 0.0

        match_won = (wins_0 >= rounds_to_win) or (wins_1 >= rounds_to_win)
        subtitle = None
        if match_won:
            final_winner = agent_names[0] if wins_0 >= rounds_to_win else agent_names[1]
            subtitle = f"MATCH OVER: {final_winner} WINS ({wins_0}-{wins_1})! [R] Restart  [ESC] Menu"

        env.render(
            round_scores=(wins_0, wins_1),
            rounds_to_win=rounds_to_win,
            agent_names=agent_names,
            subtitle=subtitle,
            fps=target_fps,
            xray_info=extract_xray_info(agents_map, obs, getattr(env, "_last_actions", None)),
            sim_status=sim_status,
        )

    if not return_to_hub:
        env.close()


def run_interactive_4way_match(
    agent_types: list[str] | None = None,
    map_name: str = "Colosseum",
    custom_bot_config: BotConfig | None = None,
    return_to_hub: bool = False,
    preset_configs: dict[str, BotConfig] | None = None,
) -> None:
    """Run interactive 4-Way Battle Royale session."""
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    if not pygame.display.get_init():
        pygame.display.init()

    agent_types = agent_types or ["Human P1", "Bulldozer", "Shinobi", "Guardian"]
    env = KineticSumo4WayEnv(map_name=map_name, render_mode="human")
    obs, _ = env.reset()

    # Instantiate agents
    agent_instances: dict[str, BaseAgent | None] = {}
    agent_labels: list[str] = []
    agent_keys = ["agent_0", "agent_1", "agent_2", "agent_3"]

    for idx, atype in enumerate(agent_types):
        k = agent_keys[idx]
        name_lower = atype.lower()
        if "human p1" in name_lower:
            agent_instances[k] = None
            agent_labels.append("Human (P1)")
        elif "human p2" in name_lower:
            agent_instances[k] = None
            agent_labels.append("Human (P2)")
        else:
            bot = load_agent(atype, custom_bot_config=custom_bot_config, preset_configs=preset_configs)
            bot.reset()
            agent_instances[k] = bot
            agent_labels.append(f"AI ({atype.upper()})")

    running = True
    paused = False
    step_once = False
    speed_idx = 2
    clock = pygame.time.Clock()

    print("\n--- Kinetic Sumo 4-Way Battle Royale ---")
    print(f"Arena Map: {map_name} | Elimination: Last Disc Standing")
    print("Controls: P1 (WASD/Space/Shift), P2 (Arrows/Enter/RShift), I (X-Ray), B (Brain View), P (Pause), . (Step), -/+ (Speed), M (Mute), ESC (Menu)\n")

    env.render(
        agent_names=tuple(agent_labels),
        xray_info=extract_xray_info(agent_instances, obs, getattr(env, "_last_actions", None)),
        sim_status=f"SPEED:{SPEED_LEVELS[speed_idx]}x",
    )

    while running:
        speed_mult = SPEED_LEVELS[speed_idx]
        target_fps = max(12, int(50 * speed_mult))
        clock.tick(target_fps if not paused else 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                if not return_to_hub:
                    sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    if env._renderer is not None:
                        env._renderer.audio.toggle_mute()
                elif event.key == pygame.K_i:
                    if env._renderer is not None:
                        env._renderer.show_xray_overlay = not env._renderer.show_xray_overlay
                elif event.key == pygame.K_b:
                    if env._renderer is not None:
                        env._renderer.show_brain_overlay = not env._renderer.show_brain_overlay
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key in (pygame.K_PERIOD, pygame.K_n):
                    paused = True
                    step_once = True
                elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET, pygame.K_KP_MINUS):
                    speed_idx = max(0, speed_idx - 1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET, pygame.K_KP_PLUS):
                    speed_idx = min(len(SPEED_LEVELS) - 1, speed_idx + 1)
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    for b in agent_instances.values():
                        if b:
                            b.reset()

        if not running:
            break

        sim_status = f"PAUSED ({SPEED_LEVELS[speed_idx]}x)" if paused else f"SPEED:{SPEED_LEVELS[speed_idx]}x"
        if paused and not step_once:
            env.render(
                agent_names=tuple(agent_labels),
                subtitle="[PAUSED — Press P to Resume or . to Step 1 Frame]",
                fps=30,
                xray_info=extract_xray_info(agent_instances, obs, getattr(env, "_last_actions", None)),
                sim_status=sim_status,
            )
            continue

        step_once = False

        # Compute actions
        actions = {}
        for idx, k in enumerate(agent_keys):
            if not env.agent_objs[k].alive:
                continue
            inst = agent_instances[k]
            if inst is None:
                p_idx = 1 if "p2" in agent_types[idx].lower() else 0
                actions[k] = get_human_action(p_idx)
            else:
                actions[k] = inst.act(obs[k])

        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render(
            agent_names=tuple(agent_labels),
            fps=target_fps,
            xray_info=extract_xray_info(agent_instances, obs, getattr(env, "_last_actions", None)),
            sim_status=sim_status,
        )

        # If round ends (all eliminated or 1 winner), allow pause before restart
        if not env.agents:
            pygame.time.delay(1200)
            obs, _ = env.reset()
            for b in agent_instances.values():
                if b:
                    b.reset()

    if not return_to_hub:
        env.close()


def run_visual_tournament(
    map_name: str = "Open",
    matches_per_pair: int = 2,
    ga_path: str = "checkpoints/ga/best_ga_agent.npy",
    ppo_path: str = "checkpoints/ppo/best_ppo_policy.pt",
    rounds_to_win: int = 3,
    fps: int = 50,
) -> None:
    """Launch visual tournament between available AI models."""
    from src.eval.tournament import Competitor, RandomAgent, run_tournament

    pool = [
        Competitor("Heuristic_Bot", HeuristicAgent()),
        Competitor("Bulldozer_Bot", BulldozerAgent()),
        Competitor("Shinobi_Bot", ShinobiAgent()),
        Competitor("Guardian_Bot", GuardianAgent()),
        Competitor("Random_Baseline", RandomAgent()),
    ]
    if os.path.exists(ga_path):
        pool.append(Competitor("Neuroevolution_GA", GAMLP.load(ga_path)))
    else:
        pool.append(Competitor("Untrained_GA", GAMLP()))

    if os.path.exists(ppo_path):
        pool.append(Competitor("SelfPlay_PPO", PPOPolicy.load(ppo_path)))
    else:
        pool.append(Competitor("Untrained_PPO", PPOPolicy()))

    run_tournament(
        competitors=pool,
        matches_per_pair=matches_per_pair,
        map_name=map_name,
        rounds_to_win=rounds_to_win,
        render=True,
        fps=fps,
    )


def main():
    # If launched with zero arguments, default to Arcade Hub GUI!
    if len(sys.argv) == 1:
        from src.ui.hub import launch_hub
        launch_hub()
        return

    parser = argparse.ArgumentParser(description="Kinetic Sumo Interactive Play Launcher")
    parser.add_argument("--hub", action="store_true", help="Launch interactive Arcade Hub GUI")
    parser.add_argument("--agent0", type=str, default="human", help="human, heuristic, bulldozer, shinobi, guardian, tunable, ga, ppo, random")
    parser.add_argument("--agent1", type=str, default="heuristic", help="human, heuristic, bulldozer, shinobi, guardian, tunable, ga, ppo, random")
    parser.add_argument("--map", type=str, default="Open", choices=list(MAP_REGISTRY.keys()))
    parser.add_argument("--4way", action="store_true", dest="is_4way", help="Launch 4-Way Battle Royale FFA")
    parser.add_argument("--ga-path", type=str, default="checkpoints/ga/best_ga_agent.npy")
    parser.add_argument("--ppo-path", type=str, default="checkpoints/ppo/best_ppo_policy.pt")
    parser.add_argument("--rounds-to-win", type=int, default=3, help="First to N rounds to win match")
    parser.add_argument("--tournament", action="store_true", help="Launch visual round-robin AI tournament")
    parser.add_argument("--matches-per-pair", type=int, default=2, help="Matches per pair in tournament mode")
    parser.add_argument("--fps", type=int, default=50, help="Target FPS for rendering (e.g. 50 or 100)")
    args = parser.parse_args()

    if args.hub:
        from src.ui.hub import launch_hub
        launch_hub()
    elif args.tournament:
        run_visual_tournament(
            map_name=args.map,
            matches_per_pair=args.matches_per_pair,
            ga_path=args.ga_path,
            ppo_path=args.ppo_path,
            rounds_to_win=args.rounds_to_win,
            fps=args.fps,
        )
    elif args.is_4way:
        run_interactive_4way_match(
            map_name=args.map,
        )
    else:
        run_interactive_match(
            agent0_type=args.agent0,
            agent1_type=args.agent1,
            map_name=args.map,
            ga_path=args.ga_path,
            ppo_path=args.ppo_path,
            rounds_to_win=args.rounds_to_win,
        )


if __name__ == "__main__":
    main()
