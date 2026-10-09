"""Pygame-CE renderer with particle sparks, thruster trails, and Lidar debug overlay."""

import os
import numpy as np
import pygame
from src.engine.entities import DiscAgent
from src.engine.physics import CollisionResult, Obstacle
from src.maps import ArenaMap


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

        # Scale: arena radius (10m) maps to ~400 pixels
        self.scale = (self.screen_size / 2.0 - 40.0) / self.arena_map.initial_radius
        self.center = np.array([self.screen_size / 2.0, self.screen_size / 2.0], dtype=np.float32)

        if not pygame.get_init():
            pygame.init()
            pygame.font.init()

        if self.mode == "human":
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Kinetic Sumo - 1v1 Hover Arena Duel")
        else:
            # Headless / offscreen
            self.screen = pygame.Surface((self.screen_size, self.screen_size))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.title_font = pygame.font.SysFont("Consolas", 24, bold=True)

        self.particles: list[Particle] = []

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

    def render_frame(
        self,
        agents: dict[str, DiscAgent],
        arena_radius: float,
        elapsed_time: float,
        last_collision: CollisionResult | None = None,
        winner: str | None = None,
        is_draw: bool = False,
    ) -> np.ndarray | None:
        """Render a single frame."""
        dt = 1.0 / self.fps

        # 1. Update particles
        if last_collision and last_collision.occurred and last_collision.contact_pos is not None:
            self.add_sparks(last_collision.contact_pos, count=20)

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
        # Inner arena floor
        pygame.draw.circle(self.screen, (22, 26, 38), (int(self.center[0]), int(self.center[1])), r_curr_px)

        # Ring border with shrink warning
        is_shrinking = elapsed_time > self.arena_map.shrink_start_time
        border_color = (255, 75, 75) if is_shrinking else (80, 160, 255)
        pygame.draw.circle(self.screen, border_color, (int(self.center[0]), int(self.center[1])), r_curr_px, 4)

        # Subtle arena center crosshair
        c_x, c_y = int(self.center[0]), int(self.center[1])
        pygame.draw.line(self.screen, (40, 48, 68), (c_x - 15, c_y), (c_x + 15, c_y), 1)
        pygame.draw.line(self.screen, (40, 48, 68), (c_x, c_y - 15), (c_x, c_y + 15), 1)

        # 3. Draw Obstacles
        for obs in self.arena_map.obstacles:
            ox, oy = self.world_to_screen(obs.pos)
            or_px = int(obs.radius * self.scale)
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
        self._render_hud(agents, arena_radius, elapsed_time, winner, is_draw)

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
    ) -> None:
        # Match timer
        time_left = max(0.0, self.arena_map.round_duration - elapsed_time)
        time_text = self.title_font.render(f"TIME: {time_left:4.1f}s", True, (240, 240, 255))
        self.screen.blit(time_text, (self.screen_size // 2 - time_text.get_width() // 2, 20))

        # Arena radius status
        rad_color = (255, 100, 100) if elapsed_time > self.arena_map.shrink_start_time else (150, 200, 255)
        rad_text = self.font.render(f"RING RADIUS: {arena_radius:.1f}m", True, rad_color)
        self.screen.blit(rad_text, (self.screen_size // 2 - rad_text.get_width() // 2, 55))

        # Agent 0 Bar (Left)
        ag0 = agents.get("agent_0")
        if ag0:
            self._draw_agent_bars(ag0, 30, 20, "AGENT 0 (Cyan)", (0, 210, 255))

        # Agent 1 Bar (Right)
        ag1 = agents.get("agent_1")
        if ag1:
            self._draw_agent_bars(ag1, self.screen_size - 230, 20, "AGENT 1 (Orange)", (255, 87, 34))

        # Winner / Game Over Banner
        if winner or is_draw:
            if is_draw:
                banner_txt = "TIMEOUT DRAW / COLD-WAR STALL"
                banner_color = (255, 200, 50)
            elif winner == "agent_0":
                banner_txt = "AGENT 0 WINS BY RING OUT!"
                banner_color = (0, 210, 255)
            else:
                banner_txt = "AGENT 1 WINS BY RING OUT!"
                banner_color = (255, 87, 34)

            b_surface = self.title_font.render(banner_txt, True, banner_color)
            self.screen.blit(
                b_surface,
                (self.screen_size // 2 - b_surface.get_width() // 2, self.screen_size - 60),
            )

    def _draw_agent_bars(self, agent: DiscAgent, x: int, y: int, label: str, color: tuple) -> None:
        name_txt = self.font.render(label, True, color)
        self.screen.blit(name_txt, (x, y))

        # Energy Bar
        bar_w = 200
        bar_h = 12
        e_ratio = np.clip(agent.energy / agent.config.max_energy, 0.0, 1.0)
        pygame.draw.rect(self.screen, (40, 40, 50), (x, y + 25, bar_w, bar_h))
        pygame.draw.rect(self.screen, (50, 220, 120), (x, y + 25, int(bar_w * e_ratio), bar_h))
        pygame.draw.rect(self.screen, (150, 150, 180), (x, y + 25, bar_w, bar_h), 1)

        # Status text
        state_str = "DASHING" if agent.is_dashing else ("ANCHORED" if agent.is_anchored else "GLIDING")
        st_txt = self.font.render(f"State: {state_str}", True, (180, 190, 210))
        self.screen.blit(st_txt, (x, y + 42))

    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()
