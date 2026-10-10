"""Pygame-CE renderer with particle sparks, thruster trails, and Lidar debug overlay."""

import os
import numpy as np
import pygame
from src.engine.entities import DiscAgent
from src.engine.physics import CollisionResult, Obstacle
from src.maps import ArenaMap
from src.ui.audio import SoundEngine



class Particle:
    """Particle for collision sparks and thruster trails."""

    def __init__(self, pos: np.ndarray, vel: np.ndarray, color: tuple[int, int, int], lifetime: float, radius: float = 3.0):
        self.pos = pos.copy().astype(np.float32)
        self.vel = vel.copy().astype(np.float32)
        self.color = color
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.radius = radius

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        self.pos += self.vel * dt
        self.vel *= 0.92  # friction
        return self.lifetime > 0.0


class SumoRenderer:
    """Renderer supporting interactive window display and headless rgb_array capture."""

    def __init__(
        self,
        arena_map: ArenaMap,
        screen_size: int = 900,
        mode: str = "human",
        fps: int = 50,
    ):
        self.arena_map = arena_map
        self.screen_size = screen_size
        self.mode = mode
        self.fps = fps
        self.show_lidar_overlay = True
        self.show_xray_overlay = True

        # Scale: arena radius (10m) maps to ~400 pixels
        self.scale = (self.screen_size / 2.0 - 40.0) / self.arena_map.initial_radius
        self.center = np.array([self.screen_size / 2.0, self.screen_size / 2.0], dtype=np.float32)

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()

        if self.mode == "human":
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Kinetic Sumo - 1v1 Hover Arena Duel")
        else:
            # Headless / offscreen
            self.screen = pygame.Surface((self.screen_size, self.screen_size))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.title_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.small_font = pygame.font.SysFont("Consolas", 12)
        self.tiny_font = pygame.font.SysFont("Consolas", 11)

        self.particles: list[Particle] = []
        self.audio = SoundEngine(enabled=(self.mode == "human"))
        self._prev_winner: str | None = None
        self._prev_dashing: dict[str, bool] = {}
        self._prev_anchored: dict[str, bool] = {}


    def world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        """Convert world coordinates (meters) to screen coordinates (pixels)."""
        screen_x = self.center[0] + pos[0] * self.scale
        screen_y = self.center[1] - pos[1] * self.scale  # invert Y for screen coords
        return int(screen_x), int(screen_y)

    def add_sparks(self, pos: np.ndarray, count: int = 25) -> None:
        """Spawn collision spark particles."""
        for _ in range(count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2.0, 10.0)
            vel = np.array([np.cos(angle), np.sin(angle)]) * speed
            color = (255, np.random.randint(180, 255), np.random.randint(50, 150))
            lifetime = np.random.uniform(0.15, 0.45)
            self.particles.append(Particle(pos, vel, color, lifetime, radius=np.random.uniform(2, 4)))

    def _draw_xray_intent_vectors(
        self,
        agents: dict[str, DiscAgent],
        agent_colors: dict[str, tuple[int, int, int]],
        last_actions: dict[str, np.ndarray] | None = None,
        xray_info: dict[str, dict] | None = None,
    ) -> None:
        """Draw AI X-Ray target aim / intercept vectors and reticles on the arena floor."""
        if not self.show_xray_overlay:
            return
        for a_id, agent in agents.items():
            if not agent.alive:
                continue
            clr = agent_colors.get(a_id, (0, 210, 255))
            info = (xray_info or {}).get(a_id, {})
            t_vec = info.get("target_vec")
            if t_vec is None and last_actions and a_id in last_actions:
                act = last_actions[a_id]
                aim_ang = agent.angle + float(act[1]) * 0.65
                mag = max(1.2, abs(float(act[0])) * 3.2)
                t_vec = np.array([np.cos(aim_ang) * mag, np.sin(aim_ang) * mag], dtype=np.float32)

            if t_vec is not None:
                t_vec_arr = np.asarray(t_vec, dtype=np.float32)
                norm = float(np.linalg.norm(t_vec_arr))
                if norm > 0.2:
                    # Clamp visual vector length to 4.5m so it stays clean inside the arena
                    if norm > 4.5:
                        t_vec_arr = t_vec_arr * (4.5 / norm)
                    ax, ay = self.world_to_screen(agent.pos)
                    tx, ty = self.world_to_screen(agent.pos + t_vec_arr)
                    pygame.draw.line(self.screen, clr, (ax, ay), (tx, ty), 2)
                    pygame.draw.circle(self.screen, clr, (tx, ty), 5, 1)

    def _draw_agent_xray_panel(
        self,
        agent_id: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        last_actions: dict[str, np.ndarray] | None = None,
        reward_breakdowns: dict[str, dict[str, float]] | None = None,
        xray_info: dict[str, dict] | None = None,
    ) -> None:
        """Render compact behind-the-scenes AI X-Ray actuator gauges, decision state, and step rewards."""
        if not self.show_xray_overlay:
            return

        panel_w = 200
        panel_h = 74
        panel_rect = pygame.Rect(x, y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (18, 22, 32), panel_rect, border_radius=5)
        pygame.draw.rect(self.screen, (45, 55, 78), panel_rect, 1, border_radius=5)

        act = (last_actions or {}).get(agent_id, np.zeros(3, dtype=np.float32))
        info = (xray_info or {}).get(agent_id, {})
        intent = info.get("intent")
        if not intent:
            if float(act[2]) > 0.5:
                intent = "ACT: DASH SURGE"
            elif float(act[2]) < -0.5:
                intent = "ACT: ANCHOR BRACE"
            elif float(act[0]) > 0.4:
                intent = "ACT: FORWARD DRIVE"
            elif float(act[0]) < -0.2:
                intent = "ACT: REVERSE BRAKE"
            else:
                intent = "ACT: IDLE / COAST"

        intent_surf = self.tiny_font.render(f"AI: {str(intent)[:22]}", True, color)
        self.screen.blit(intent_surf, (x + 6, y + 4))

        # 3 Bidirectional Actuator Gauges: THR, STR, ABL
        labels = [("THR", float(act[0]), (50, 220, 130)), ("STR", float(act[1]), (0, 200, 255)), ("ABL", float(act[2]), (255, 195, 50))]
        for idx, (lbl_txt, val, bar_col) in enumerate(labels):
            by = y + 20 + idx * 12
            l_surf = self.tiny_font.render(lbl_txt, True, (150, 165, 190))
            self.screen.blit(l_surf, (x + 6, by - 1))

            track_x = x + 34
            track_w = 122
            track_h = 7
            mid_x = track_x + track_w // 2
            pygame.draw.rect(self.screen, (30, 36, 48), (track_x, by + 2, track_w, track_h), border_radius=2)

            v_clamped = float(np.clip(val, -1.0, 1.0))
            fill_len = int(abs(v_clamped) * (track_w // 2))
            if v_clamped >= 0:
                pygame.draw.rect(self.screen, bar_col, (mid_x, by + 2, fill_len, track_h), border_radius=2)
            else:
                neg_col = (255, 95, 95) if lbl_txt != "ABL" else (180, 130, 255)
                pygame.draw.rect(self.screen, neg_col, (mid_x - fill_len, by + 2, fill_len, track_h), border_radius=2)

            # Center zero tick
            pygame.draw.line(self.screen, (200, 210, 230), (mid_x, by + 1), (mid_x, by + track_h + 2), 1)
            # Threshold ticks for ABL (-0.5 Anchor, +0.5 Dash)
            if lbl_txt == "ABL":
                t_neg = mid_x - track_w // 4
                t_pos = mid_x + track_w // 4
                pygame.draw.line(self.screen, (140, 140, 165), (t_neg, by + 1), (t_neg, by + track_h + 2), 1)
                pygame.draw.line(self.screen, (140, 140, 165), (t_pos, by + 1), (t_pos, by + track_h + 2), 1)

            v_surf = self.tiny_font.render(f"{v_clamped:+.2f}", True, (210, 220, 240))
            self.screen.blit(v_surf, (track_x + track_w + 4, by - 1))

        # Step reward breakdown line
        rb = (reward_breakdowns or {}).get(agent_id)
        if rb:
            r_txt = f"REW:{rb.get('total', 0.0):+.2f} P:{rb.get('pbrs', 0.0):+.2f} C:{rb.get('contact', 0.0):+.2f}"
            r_surf = self.tiny_font.render(r_txt, True, (165, 185, 215))
            self.screen.blit(r_surf, (x + 6, y + 57))

    def render_frame(
        self,
        agents: dict[str, DiscAgent],
        arena_radius: float,
        elapsed_time: float,
        last_collision: CollisionResult | None = None,
        winner: str | None = None,
        is_draw: bool = False,
        round_scores: tuple[int, int] = (0, 0),
        rounds_to_win: int = 3,
        agent_names: tuple[str, str] = ("AGENT 0", "AGENT 1"),
        subtitle: str | None = None,
        fps: int | None = None,
        last_actions: dict[str, np.ndarray] | None = None,
        reward_breakdowns: dict[str, dict[str, float]] | None = None,
        xray_info: dict[str, dict] | None = None,
        sim_status: str | None = None,
    ) -> np.ndarray | None:
        """Render a single frame."""
        if fps is not None and fps > 0:
            self.fps = fps
        dt = 1.0 / self.fps

        # 1. Update particles and trigger audio
        if last_collision and last_collision.occurred:
            if last_collision.contact_pos is not None:
                self.add_sparks(last_collision.contact_pos, count=20)
            impulse = max(last_collision.impulse_a, last_collision.impulse_b)
            self.audio.play_clang(
                is_flank=last_collision.flank_hit_on_a or last_collision.flank_hit_on_b,
                intensity=float(np.clip(impulse / 15.0, 0.3, 1.0)),
            )

        for a_id, ag in agents.items():
            if ag.is_dashing and not self._prev_dashing.get(a_id, False):
                self.audio.play_dash()
            if ag.is_anchored and not self._prev_anchored.get(a_id, False):
                self.audio.play_anchor()
            self._prev_dashing[a_id] = ag.is_dashing
            self._prev_anchored[a_id] = ag.is_anchored

        if winner and winner != self._prev_winner:
            self.audio.play_ringout()
        self._prev_winner = winner

        # Thruster exhaust particles
        for a_id, ag in agents.items():
            if ag.alive:
                speed = np.linalg.norm(ag.vel)
                if speed > 0.5 or ag.is_dashing:
                    rear_offset = -ag.heading_vector * ag.config.radius
                    emit_pos = ag.pos + rear_offset
                    color = (0, 230, 255) if a_id == "agent_0" else (255, 120, 0)
                    if ag.is_dashing:
                        color = (255, 255, 255)
                    self.particles.append(
                        Particle(
                            emit_pos,
                            -ag.heading_vector * np.random.uniform(2.0, 6.0),
                            color,
                            lifetime=0.2,
                            radius=3.0,
                        )
                    )


        self.particles = [p for p in self.particles if p.update(dt)]

        # 2. Draw Background & Outer Void
        self.screen.fill((12, 14, 20))  # Deep abyss

        # Outer ring initial guide
        r_init_px = int(self.arena_map.initial_radius * self.scale)
        pygame.draw.circle(self.screen, (25, 30, 45), (int(self.center[0]), int(self.center[1])), r_init_px, 1)

        # Active Arena ring
        r_curr_px = max(1, int(arena_radius * self.scale))
        hazard = getattr(self.arena_map, "hazard_type", "standard")
        if hazard == "ice":
            floor_color = (16, 28, 44)
            border_color = (130, 220, 255)
        elif hazard == "bumper":
            floor_color = (25, 20, 36)
            border_color = (255, 100, 200)
        else:
            floor_color = (22, 26, 38)
            border_color = (80, 160, 255)

        is_shrinking = elapsed_time > self.arena_map.shrink_start_time
        if is_shrinking:
            border_color = (255, 75, 75)

        # Inner arena floor
        pygame.draw.circle(self.screen, floor_color, (int(self.center[0]), int(self.center[1])), r_curr_px)

        # Concentric stepped rings for Sinking_Pillars
        if getattr(self.arena_map, "shrink_mode", "continuous") == "stepped":
            for step_r in [8.5, 7.0, 5.5, 4.0]:
                if step_r <= arena_radius:
                    s_px = int(step_r * self.scale)
                    pygame.draw.circle(self.screen, (40, 55, 80), (int(self.center[0]), int(self.center[1])), s_px, 1)

        pygame.draw.circle(self.screen, border_color, (int(self.center[0]), int(self.center[1])), r_curr_px, 4)

        # Subtle arena center crosshair
        c_x, c_y = int(self.center[0]), int(self.center[1])
        pygame.draw.line(self.screen, (40, 48, 68), (c_x - 15, c_y), (c_x + 15, c_y), 1)
        pygame.draw.line(self.screen, (40, 48, 68), (c_x, c_y - 15), (c_x, c_y + 15), 1)

        # 3. Draw Obstacles
        for obs in self.arena_map.obstacles:
            ox, oy = self.world_to_screen(obs.pos)
            or_px = int(obs.radius * self.scale)
            if obs.restitution >= 1.8:
                # High restitution pinball bumper
                pygame.draw.circle(self.screen, (55, 20, 45), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (255, 70, 180), (ox, oy), or_px, 3)
                pygame.draw.circle(self.screen, (255, 220, 50), (ox, oy), max(3, or_px // 2))
                pygame.draw.circle(self.screen, (255, 255, 255), (ox, oy), max(2, or_px // 4))
            elif hazard == "ice":
                # Ice pillar
                pygame.draw.circle(self.screen, (30, 50, 75), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (120, 200, 255), (ox, oy), or_px, 2)
                pygame.draw.circle(self.screen, (220, 245, 255), (ox, oy), max(2, or_px // 3))
            else:
                # Metallic bumper
                pygame.draw.circle(self.screen, (45, 52, 70), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (100, 140, 180), (ox, oy), or_px, 3)
                pygame.draw.circle(self.screen, (180, 220, 255), (ox, oy), max(2, or_px // 3))

        # 4. Draw Particles (Sparks & Thruster trails)
        for p in self.particles:
            px, py = self.world_to_screen(p.pos)
            alpha_ratio = max(0.0, p.lifetime / p.max_lifetime)
            cur_r = max(1, int(p.radius * alpha_ratio))
            pygame.draw.circle(self.screen, p.color, (px, py), cur_r)

        # 5. Draw Agents
        agent_colors = {
            "agent_0": (0, 210, 255),   # Neon Cyan
            "agent_1": (255, 87, 34),   # Bright Orange
        }

        self._draw_xray_intent_vectors(agents, agent_colors, last_actions, xray_info)

        for a_id, agent in agents.items():
            if not agent.alive:
                continue

            ax, ay = self.world_to_screen(agent.pos)
            ar_px = int(agent.config.radius * self.scale)

            # Anchor effect aura
            if agent.is_anchored:
                pygame.draw.circle(self.screen, (150, 150, 150), (ax, ay), ar_px + 8, 2)

            # Dash effect glow
            if agent.is_dashing:
                pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), ar_px + 6, 2)

            # Main disc body
            pygame.draw.circle(self.screen, (30, 34, 48), (ax, ay), ar_px)
            pygame.draw.circle(self.screen, agent_colors[a_id], (ax, ay), ar_px, 3)

            # Front 120-degree reinforced bumper arc
            # Arc covers [-pi/3, +pi/3] from agent.angle
            # Pygame arc angles are in radians measured counter-clockwise from 3 o'clock
            half_arc = agent.config.bumper_arc / 2.0
            # For screen coords (inverted Y), screen angle is -agent.angle
            base_screen_ang = -agent.angle
            rect = pygame.Rect(ax - ar_px, ay - ar_px, ar_px * 2, ar_px * 2)
            pygame.draw.arc(
                self.screen,
                (255, 255, 255),
                rect,
                base_screen_ang - half_arc,
                base_screen_ang + half_arc,
                5,
            )

            # Directional nose indicator
            head = agent.heading_vector
            tip_pos = agent.pos + head * (agent.config.radius * 1.1)
            tx, ty = self.world_to_screen(tip_pos)
            pygame.draw.line(self.screen, (255, 255, 255), (ax, ay), (tx, ty), 3)

        # 6. Lidar Debug Overlay (if enabled)
        if self.show_lidar_overlay:
            for a_id, agent in agents.items():
                if not agent.alive:
                    continue
                # Cast lidar rays for visualization
                opp_id = "agent_1" if a_id == "agent_0" else "agent_0"
                opp = agents[opp_id]
                from src.engine.raycast import RaycastEngine
                ray_engine = RaycastEngine(num_rays=16, max_range=self.arena_map.initial_radius * 2.5)
                dists, hit_types = ray_engine.cast_lidar(
                    origin=agent.pos,
                    heading_angle=agent.angle,
                    arena_radius=arena_radius,
                    opponent_pos=opp.pos,
                    opponent_radius=opp.config.radius,
                    obstacles=self.arena_map.obstacles,
                )

                ax, ay = self.world_to_screen(agent.pos)
                angles = agent.angle + ray_engine.ray_offsets
                for i in range(16):
                    r_len = dists[i] * ray_engine.max_range
                    target_w = agent.pos + np.array([np.cos(angles[i]), np.sin(angles[i])]) * r_len
                    rx, ry = self.world_to_screen(target_w)

                    # Color coding: -1.0 void (red), 0.5 wall (yellow), 1.0 opp (green), 0.0 none (faint grey)
                    htype = hit_types[i]
                    if htype == -1.0:
                        lcolor = (180, 50, 50, 100)
                    elif htype == 0.5:
                        lcolor = (200, 200, 50, 100)
                    elif htype == 1.0:
                        lcolor = (50, 230, 100, 180)
                    else:
                        lcolor = (60, 60, 80, 60)

                    pygame.draw.line(self.screen, lcolor, (ax, ay), (rx, ry), 1)

        # 7. HUD & Telemetry
        self._render_hud(
            agents,
            arena_radius,
            elapsed_time,
            winner,
            is_draw,
            round_scores,
            rounds_to_win,
            agent_names,
            subtitle,
            last_actions=last_actions,
            reward_breakdowns=reward_breakdowns,
            xray_info=xray_info,
            sim_status=sim_status,
        )

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        else:
            # Return RGB array (H, W, 3)
            img = pygame.surfarray.array3d(self.screen)
            return np.transpose(img, (1, 0, 2))

    def _render_hud(
        self,
        agents: dict[str, DiscAgent],
        arena_radius: float,
        elapsed_time: float,
        winner: str | None,
        is_draw: bool,
        round_scores: tuple[int, int] = (0, 0),
        rounds_to_win: int = 3,
        agent_names: tuple[str, str] = ("AGENT 0", "AGENT 1"),
        subtitle: str | None = None,
        last_actions: dict[str, np.ndarray] | None = None,
        reward_breakdowns: dict[str, dict[str, float]] | None = None,
        xray_info: dict[str, dict] | None = None,
        sim_status: str | None = None,
    ) -> None:
        # Match timer
        time_left = max(0.0, self.arena_map.round_duration - elapsed_time)
        time_text = self.title_font.render(f"TIME: {time_left:4.1f}s", True, (240, 240, 255))
        self.screen.blit(time_text, (self.screen_size // 2 - time_text.get_width() // 2, 16))

        # Rounds score (First to 3)
        score_txt = self.font.render(
            f"ROUNDS: {round_scores[0]} - {round_scores[1]} (First to {rounds_to_win})",
            True,
            (220, 220, 240),
        )
        self.screen.blit(score_txt, (self.screen_size // 2 - score_txt.get_width() // 2, 44))

        # Subtitle (e.g. tournament matchup info)
        if subtitle:
            sub_txt = self.font.render(subtitle, True, (255, 215, 0))
            self.screen.blit(sub_txt, (self.screen_size // 2 - sub_txt.get_width() // 2, 68))
            rad_y = 90
        else:
            rad_y = 68

        # Arena radius status
        rad_color = (255, 100, 100) if elapsed_time > self.arena_map.shrink_start_time else (150, 200, 255)
        rad_text = self.font.render(f"RING RADIUS: {arena_radius:.1f}m", True, rad_color)
        self.screen.blit(rad_text, (self.screen_size // 2 - rad_text.get_width() // 2, rad_y))

        # Agent 0 Bar (Left)
        ag0 = agents.get("agent_0")
        if ag0:
            name0 = agent_names[0] if len(agent_names) > 0 else "AGENT 0"
            self._draw_agent_bars(ag0, 30, 20, f"{name0} (Cyan)", (0, 210, 255))
            self._draw_agent_xray_panel("agent_0", 30, 88, (0, 210, 255), last_actions, reward_breakdowns, xray_info)

        # Agent 1 Bar (Right)
        ag1 = agents.get("agent_1")
        if ag1:
            name1 = agent_names[1] if len(agent_names) > 1 else "AGENT 1"
            self._draw_agent_bars(ag1, self.screen_size - 240, 20, f"{name1} (Orange)", (255, 87, 34))
            self._draw_agent_xray_panel("agent_1", self.screen_size - 240, 88, (255, 87, 34), last_actions, reward_breakdowns, xray_info)

        # Bottom Controls & Simulation Status Strip
        lidar_state = "ON" if self.show_lidar_overlay else "OFF"
        xray_state = "ON" if self.show_xray_overlay else "OFF"
        status_suffix = f"  |  {sim_status}" if sim_status else ""
        ctrl_str = f"[L] Lidar:{lidar_state}  [I] AI X-Ray:{xray_state}  [P] Pause  [.] Step  [-/+] Speed{status_suffix}"
        ctrl_col = (255, 215, 70) if sim_status and ("PAUSED" in sim_status or "x" in sim_status and "1.0x" not in sim_status) else (130, 145, 175)
        ctrl_surf = self.tiny_font.render(ctrl_str, True, ctrl_col)
        self.screen.blit(ctrl_surf, (self.screen_size // 2 - ctrl_surf.get_width() // 2, self.screen_size - 22))

        # Winner / Game Over Banner
        if winner or is_draw:
            if is_draw:
                banner_txt = "TIMEOUT DRAW / COLD-WAR STALL"
                banner_color = (255, 200, 50)
            elif winner == "agent_0":
                name0 = agent_names[0] if len(agent_names) > 0 else "AGENT 0"
                banner_txt = f"{name0} WINS BY RING OUT!"
                banner_color = (0, 210, 255)
            else:
                name1 = agent_names[1] if len(agent_names) > 1 else "AGENT 1"
                banner_txt = f"{name1} WINS BY RING OUT!"
                banner_color = (255, 87, 34)

            b_surface = self.title_font.render(banner_txt, True, banner_color)
            self.screen.blit(
                b_surface,
                (self.screen_size // 2 - b_surface.get_width() // 2, self.screen_size - 60),
            )

    def _draw_agent_bars(self, agent: DiscAgent, x: int, y: int, label: str, color: tuple) -> None:
        name_txt = self.font.render(label, True, color)
        draw_x = x
        if x > self.screen_size // 2:
            max_right = self.screen_size - 20
            draw_x = min(x, max_right - name_txt.get_width())
        self.screen.blit(name_txt, (draw_x, y))

        # Energy Bar
        bar_w = 200
        bar_h = 10
        e_ratio = float(np.clip(agent.energy / agent.config.max_energy, 0.0, 1.0))
        pygame.draw.rect(self.screen, (35, 38, 48), (x, y + 24, bar_w, bar_h))
        pygame.draw.rect(self.screen, (50, 220, 120), (x, y + 24, int(bar_w * e_ratio), bar_h))
        pygame.draw.rect(self.screen, (100, 110, 130), (x, y + 24, bar_w, bar_h), 1)

        # Dash Cooldown Bar
        cd_ratio = 1.0 - float(np.clip(agent.dash_cooldown / agent.config.dash_cooldown, 0.0, 1.0))
        cd_color = (0, 200, 255) if cd_ratio >= 1.0 else (120, 140, 180)
        pygame.draw.rect(self.screen, (35, 38, 48), (x, y + 37, bar_w, 6))
        pygame.draw.rect(self.screen, cd_color, (x, y + 37, int(bar_w * cd_ratio), 6))

        # Combat Telemetry metrics
        speed = float(np.linalg.norm(agent.vel))
        omega = abs(float(agent.angular_vel))
        state_str = "DASH" if agent.is_dashing else ("ANCHOR" if agent.is_anchored else "GLIDE")
        if not agent.alive:
            state_str = "ELIMINATED"
        telemetry_txt = self.small_font.render(
            f"SPD:{speed:4.1f}m/s ANG:{omega:4.1f}r/s [{state_str}]",
            True,
            (170, 185, 210),
        )
        self.screen.blit(telemetry_txt, (x, y + 47))

    def render_4way_frame(
        self,
        agents: dict[str, DiscAgent],
        arena_radius: float,
        elapsed_time: float,
        recent_collisions: list[CollisionResult] | None = None,
        winner: str | None = None,
        is_draw: bool = False,
        agent_names: tuple[str, str, str, str] = ("P1 (CYAN)", "P2 (ORANGE)", "P3 (MAGENTA)", "P4 (EMERALD)"),
        subtitle: str | None = None,
        fps: int | None = None,
        last_actions: dict[str, np.ndarray] | None = None,
        xray_info: dict[str, dict] | None = None,
        sim_status: str | None = None,
    ) -> np.ndarray | None:
        """Render a single frame for 4-Way Battle Royale."""
        if fps is not None and fps > 0:
            self.fps = fps
        dt = 1.0 / self.fps

        # Audio triggers on collisions
        if recent_collisions:
            for c_res in recent_collisions:
                if c_res.occurred:
                    if c_res.contact_pos is not None:
                        self.add_sparks(c_res.contact_pos, count=20)
                    imp = max(c_res.impulse_a, c_res.impulse_b)
                    self.audio.play_clang(
                        is_flank=c_res.flank_hit_on_a or c_res.flank_hit_on_b,
                        intensity=float(np.clip(imp / 15.0, 0.3, 1.0)),
                    )

        # Agent state sound triggers
        for a_id, ag in agents.items():
            if ag.is_dashing and not self._prev_dashing.get(a_id, False):
                self.audio.play_dash()
            if ag.is_anchored and not self._prev_anchored.get(a_id, False):
                self.audio.play_anchor()
            self._prev_dashing[a_id] = ag.is_dashing
            self._prev_anchored[a_id] = ag.is_anchored

        if winner and winner != self._prev_winner:
            self.audio.play_ringout()
        self._prev_winner = winner

        # Thruster exhaust particles
        colors_4way = {
            "agent_0": (0, 210, 255),
            "agent_1": (255, 87, 34),
            "agent_2": (255, 0, 180),
            "agent_3": (0, 230, 120),
        }

        for a_id, ag in agents.items():
            if ag.alive:
                speed = np.linalg.norm(ag.vel)
                if speed > 0.5 or ag.is_dashing:
                    rear_offset = -ag.heading_vector * ag.config.radius
                    emit_pos = ag.pos + rear_offset
                    clr = (255, 255, 255) if ag.is_dashing else colors_4way.get(a_id, (200, 200, 200))
                    self.particles.append(
                        Particle(
                            emit_pos,
                            -ag.heading_vector * np.random.uniform(2.0, 6.0),
                            clr,
                            lifetime=0.2,
                            radius=3.0,
                        )
                    )

        self.particles = [p for p in self.particles if p.update(dt)]

        # Background
        self.screen.fill((12, 14, 20))

        # Initial guide ring
        r_init_px = int(self.arena_map.initial_radius * self.scale)
        pygame.draw.circle(self.screen, (25, 30, 45), (int(self.center[0]), int(self.center[1])), r_init_px, 1)

        # Active Arena floor
        r_curr_px = max(1, int(arena_radius * self.scale))
        hazard = getattr(self.arena_map, "hazard_type", "standard")
        if hazard == "ice":
            floor_color = (16, 28, 44)
            border_color = (130, 220, 255)
        elif hazard == "bumper":
            floor_color = (25, 20, 36)
            border_color = (255, 100, 200)
        else:
            floor_color = (22, 26, 38)
            border_color = (80, 160, 255)

        is_shrinking = elapsed_time > self.arena_map.shrink_start_time
        if is_shrinking:
            border_color = (255, 75, 75)

        pygame.draw.circle(self.screen, floor_color, (int(self.center[0]), int(self.center[1])), r_curr_px)

        # Concentric stepped rings for Sinking_Pillars
        if getattr(self.arena_map, "shrink_mode", "continuous") == "stepped":
            for step_r in [8.5, 7.0, 5.5, 4.0]:
                if step_r <= arena_radius:
                    s_px = int(step_r * self.scale)
                    pygame.draw.circle(self.screen, (40, 55, 80), (int(self.center[0]), int(self.center[1])), s_px, 1)

        pygame.draw.circle(self.screen, border_color, (int(self.center[0]), int(self.center[1])), r_curr_px, 4)

        # Center crosshair
        c_x, c_y = int(self.center[0]), int(self.center[1])
        pygame.draw.line(self.screen, (40, 48, 68), (c_x - 15, c_y), (c_x + 15, c_y), 1)
        pygame.draw.line(self.screen, (40, 48, 68), (c_x, c_y - 15), (c_x, c_y + 15), 1)

        # Obstacles
        for obs in self.arena_map.obstacles:
            ox, oy = self.world_to_screen(obs.pos)
            or_px = int(obs.radius * self.scale)
            if obs.restitution >= 1.8:
                # High restitution pinball bumper
                pygame.draw.circle(self.screen, (55, 20, 45), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (255, 70, 180), (ox, oy), or_px, 3)
                pygame.draw.circle(self.screen, (255, 220, 50), (ox, oy), max(3, or_px // 2))
                pygame.draw.circle(self.screen, (255, 255, 255), (ox, oy), max(2, or_px // 4))
            elif hazard == "ice":
                # Ice pillar
                pygame.draw.circle(self.screen, (30, 50, 75), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (120, 200, 255), (ox, oy), or_px, 2)
                pygame.draw.circle(self.screen, (220, 245, 255), (ox, oy), max(2, or_px // 3))
            else:
                # Metallic bumper
                pygame.draw.circle(self.screen, (45, 52, 70), (ox, oy), or_px)
                pygame.draw.circle(self.screen, (100, 140, 180), (ox, oy), or_px, 3)
                pygame.draw.circle(self.screen, (180, 220, 255), (ox, oy), max(2, or_px // 3))

        # Particles
        for p in self.particles:
            px, py = self.world_to_screen(p.pos)
            alpha_ratio = max(0.0, p.lifetime / p.max_lifetime)
            cur_r = max(1, int(p.radius * alpha_ratio))
            pygame.draw.circle(self.screen, p.color, (px, py), cur_r)

        self._draw_xray_intent_vectors(agents, colors_4way, last_actions, xray_info)

        # Agents
        for a_id, agent in agents.items():
            if not agent.alive:
                continue

            ax, ay = self.world_to_screen(agent.pos)
            ar_px = int(agent.config.radius * self.scale)

            if agent.is_anchored:
                pygame.draw.circle(self.screen, (150, 150, 150), (ax, ay), ar_px + 8, 2)
            if agent.is_dashing:
                pygame.draw.circle(self.screen, (255, 255, 255), (ax, ay), ar_px + 6, 2)

            pygame.draw.circle(self.screen, (30, 34, 48), (ax, ay), ar_px)
            pygame.draw.circle(self.screen, colors_4way[a_id], (ax, ay), ar_px, 3)

            half_arc = agent.config.bumper_arc / 2.0
            base_screen_ang = -agent.angle
            rect = pygame.Rect(ax - ar_px, ay - ar_px, ar_px * 2, ar_px * 2)
            pygame.draw.arc(
                self.screen,
                (255, 255, 255),
                rect,
                base_screen_ang - half_arc,
                base_screen_ang + half_arc,
                5,
            )

            head = agent.heading_vector
            tip_pos = agent.pos + head * (agent.config.radius * 1.1)
            tx, ty = self.world_to_screen(tip_pos)
            pygame.draw.line(self.screen, (255, 255, 255), (ax, ay), (tx, ty), 3)

        # 4-Way HUD
        time_left = max(0.0, self.arena_map.round_duration - elapsed_time)
        t_surf = self.title_font.render(f"4-WAY FFA: {time_left:4.1f}s", True, (240, 240, 255))
        self.screen.blit(t_surf, (self.screen_size // 2 - t_surf.get_width() // 2, 16))

        rad_color = (255, 100, 100) if elapsed_time > self.arena_map.shrink_start_time else (150, 200, 255)
        rad_surf = self.font.render(f"RING RADIUS: {arena_radius:.1f}m", True, rad_color)
        self.screen.blit(rad_surf, (self.screen_size // 2 - rad_surf.get_width() // 2, 46))

        # 4 Player Cards
        corner_coords = [
            (25, 20),                                # agent_0: top-left
            (self.screen_size - 225, 20),           # agent_1: top-right
            (25, self.screen_size - 95),            # agent_2: bottom-left
            (self.screen_size - 225, self.screen_size - 95),  # agent_3: bottom-right
        ]
        agent_keys = ["agent_0", "agent_1", "agent_2", "agent_3"]
        for idx, ak in enumerate(agent_keys):
            if ak in agents:
                cx, cy = corner_coords[idx]
                lbl = agent_names[idx] if idx < len(agent_names) else ak.upper()
                self._draw_agent_bars(agents[ak], cx, cy, lbl, colors_4way[ak])
                if self.show_xray_overlay and idx < 2:
                    self._draw_agent_xray_panel(ak, cx, cy + 68, colors_4way[ak], last_actions, None, xray_info)

        xray_state = "ON" if self.show_xray_overlay else "OFF"
        status_suffix = f"  |  {sim_status}" if sim_status else ""
        ctrl_str = f"[I] AI X-Ray:{xray_state}  [P] Pause  [.] Step  [-/+] Speed{status_suffix}"
        ctrl_surf = self.tiny_font.render(ctrl_str, True, (140, 155, 185))
        self.screen.blit(ctrl_surf, (self.screen_size // 2 - ctrl_surf.get_width() // 2, self.screen_size - 20))

        # Elimination / Winner banner
        if winner or is_draw:
            if is_draw:
                banner_txt = "4-WAY BATTLE DRAW"
                banner_color = (255, 200, 50)
            else:
                w_idx = agent_keys.index(winner) if winner in agent_keys else 0
                w_name = agent_names[w_idx] if w_idx < len(agent_names) else winner.upper()
                banner_txt = f"{w_name} IS THE LAST DISC STANDING!"
                banner_color = colors_4way.get(winner, (255, 255, 255))

            b_surface = self.title_font.render(banner_txt, True, banner_color)
            self.screen.blit(
                b_surface,
                (self.screen_size // 2 - b_surface.get_width() // 2, self.screen_size - 110),
            )

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        else:
            img = pygame.surfarray.array3d(self.screen)
            return np.transpose(img, (1, 0, 2))

    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()

