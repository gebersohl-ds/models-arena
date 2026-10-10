"""Kinetic Sumo Arcade Hub: Main Menu, Matchmaker, Bot Locker, Tournament GUI, Analytics, and Evo Lab."""

import os
import sys
from pathlib import Path
import numpy as np
import pygame

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
from src.agents.progressive import (
    RookieAgent,
    ContenderAgent,
    ApexAgent,
    ensure_progressive_checkpoints,
)
from src.agents.ga_mlp import GAMLP
from src.agents.ppo_policy import PPOPolicy
from src.maps import get_map, MAP_REGISTRY, ArenaMap
from src.ui.renderer import SumoRenderer
from src.ui.audio import SoundEngine
from src.env.arena_env import KineticSumoEnv
from src.env.arena_4way_env import KineticSumo4WayEnv


# UI Colors
BG_DARK = (12, 14, 20)
PANEL_BG = (20, 24, 34)
PANEL_BORDER = (45, 52, 72)
TEXT_WHITE = (240, 244, 255)
TEXT_MUTED = (140, 150, 175)
CYAN_ACCENT = (0, 210, 255)
ORANGE_ACCENT = (255, 87, 34)
GOLD_ACCENT = (255, 205, 50)
GREEN_ACCENT = (0, 230, 120)
MAGENTA_ACCENT = (255, 0, 180)
PURPLE_ACCENT = (180, 100, 255)
RED_ACCENT = (255, 75, 75)


def get_bot_category_badge(name: str) -> tuple[str, tuple[int, int, int]]:
    """Return a concise category label and color for a fighter/map option."""
    low = name.lower()
    if "human" in low:
        return "HUMAN", CYAN_ACCENT
    if low in ("bulldozer", "shinobi", "guardian", "matador", "ricochet", "chameleon", "heuristic"):
        return "TACTICAL", ORANGE_ACCENT
    if "preset" in low or "tunable" in low or "custom" in low:
        return "CUSTOM", GOLD_ACCENT
    if any(k in low for k in ("apex", "contender", "rookie", "ppo", "ga", "evo")):
        return "NEURAL ML", GREEN_ACCENT
    if "random" in low:
        return "BASELINE", TEXT_MUTED
    return "SELECT", CYAN_ACCENT


class UIButton:
    """Clickable arcade button with hover and active states."""

    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font, color=CYAN_ACCENT, bg=PANEL_BG):
        self.rect = rect
        self.text = text
        self.font = font
        self.color = color
        self.bg = bg
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        bg_col = (40, 48, 68) if self.is_hovered else self.bg
        border_col = (255, 255, 255) if self.is_hovered else self.color

        pygame.draw.rect(surface, bg_col, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_col, self.rect, width=2, border_radius=6)

        txt_col = TEXT_WHITE if self.is_hovered else self.color
        txt_surf = self.font.render(self.text, True, txt_col)
        surface.blit(
            txt_surf,
            (
                self.rect.x + (self.rect.width - txt_surf.get_width()) // 2,
                self.rect.y + (self.rect.height - txt_surf.get_height()) // 2,
            ),
        )


class UISlider:
    """Draggable parameter tuning slider with hover detection for live helper cards."""

    def __init__(self, rect: pygame.Rect, label: str, min_val: float, max_val: float, val: float, font: pygame.font.Font):
        self.rect = rect
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.val = val
        self.font = font
        self.dragging = False
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        grab_rect = self.rect.inflate(10, 18)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if grab_rect.collidepoint(event.pos):
                self.dragging = True
                self.is_hovered = True
                self._update_val(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            self.is_hovered = grab_rect.collidepoint(event.pos)
            if self.dragging:
                self._update_val(event.pos[0])
                return True
        return False

    def _update_val(self, mouse_x: int) -> None:
        rel = np.clip((mouse_x - self.rect.x) / self.rect.width, 0.0, 1.0)
        self.val = float(self.min_val + rel * (self.max_val - self.min_val))

    def draw(self, surface: pygame.Surface, accent_color=CYAN_ACCENT) -> None:
        val_str = f"{int(self.val)}" if self.label == "Pop Size" else f"{self.val:.2f}"
        lbl_col = GOLD_ACCENT if (self.is_hovered or self.dragging) else TEXT_WHITE
        lbl_surf = self.font.render(f"{self.label}: {val_str}", True, lbl_col)
        surface.blit(lbl_surf, (self.rect.x, self.rect.y - 20))

        # Track
        pygame.draw.rect(surface, (40, 48, 68), self.rect, border_radius=4)
        # Filled portion
        ratio = (self.val - self.min_val) / max(1e-5, (self.max_val - self.min_val))
        fill_w = int(self.rect.width * ratio)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
        bar_col = GOLD_ACCENT if (self.is_hovered or self.dragging) else accent_color
        pygame.draw.rect(surface, bar_col, fill_rect, border_radius=4)

        # Handle
        handle_x = self.rect.x + fill_w
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.rect.centery), 7)


class UIDropdown:
    """Interactive dropdown menu supporting direct item selection, scrolling, and category badges."""

    def __init__(
        self,
        rect: pygame.Rect,
        options_getter,
        idx_getter,
        idx_setter,
        color=CYAN_ACCENT,
        max_visible: int = 8,
        row_height: int = 28,
    ):
        self.rect = rect
        self.options_getter = options_getter
        self.idx_getter = idx_getter
        self.idx_setter = idx_setter
        self.color = color
        self.max_visible = max_visible
        self.row_height = row_height
        self.is_open = False
        self.scroll_offset = 0
        self.hovered_row = -1
        self.is_hovered = False

    @property
    def options(self) -> list[str]:
        opts = self.options_getter()
        return list(opts) if opts else ["None"]

    @property
    def selected_idx(self) -> int:
        opts = self.options
        idx = int(self.idx_getter())
        return max(0, min(idx, len(opts) - 1))

    @property
    def selected_text(self) -> str:
        opts = self.options
        return opts[self.selected_idx]

    def get_menu_rect(self, screen_height: int = 900) -> pygame.Rect:
        opts = self.options
        vis_rows = min(len(opts), self.max_visible)
        menu_h = vis_rows * self.row_height + 8
        menu_w = max(self.rect.width, 220)
        menu_x = self.rect.x + (self.rect.width - menu_w) // 2
        menu_x = max(20, min(menu_x, 900 - menu_w - 20))
        # Open downwards unless near bottom of screen
        if self.rect.bottom + 4 + menu_h <= screen_height - 15:
            menu_y = self.rect.bottom + 4
        else:
            menu_y = max(75, self.rect.top - menu_h - 4)
        return pygame.Rect(menu_x, menu_y, menu_w, menu_h)

    def handle_event(self, event: pygame.event.Event, screen_height: int = 900) -> bool:
        """Process mouse events. Returns True if the event was consumed by this dropdown."""
        opts = self.options
        max_scroll = max(0, len(opts) - self.max_visible)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.is_open:
                m_rect = self.get_menu_rect(screen_height)
                if m_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - (m_rect.y + 4)
                    row = rel_y // self.row_height
                    vis_rows = min(len(opts), self.max_visible)
                    if 0 <= row < vis_rows:
                        self.hovered_row = self.scroll_offset + int(row)
                    else:
                        self.hovered_row = -1
                else:
                    self.hovered_row = -1
            return False

        if event.type == pygame.MOUSEWHEEL and self.is_open:
            self.scroll_offset = max(0, min(max_scroll, self.scroll_offset - event.y))
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_open:
                if event.button == 4:  # Wheel up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                    return True
                if event.button == 5:  # Wheel down
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
                    return True
                if event.button == 1:
                    m_rect = self.get_menu_rect(screen_height)
                    if m_rect.collidepoint(event.pos):
                        rel_y = event.pos[1] - (m_rect.y + 4)
                        row = int(rel_y // self.row_height)
                        vis_rows = min(len(opts), self.max_visible)
                        if 0 <= row < vis_rows:
                            chosen = self.scroll_offset + row
                            if 0 <= chosen < len(opts):
                                self.idx_setter(chosen)
                        self.is_open = False
                        return True
                    elif self.rect.collidepoint(event.pos):
                        self.is_open = False
                        return True
                    else:
                        self.is_open = False
                        return False
            else:
                if event.button == 1 and self.rect.collidepoint(event.pos):
                    self.is_open = True
                    # Center scroll around currently selected item
                    cur = self.selected_idx
                    self.scroll_offset = max(0, min(max_scroll, cur - self.max_visible // 2))
                    return True

        return False

    def draw_overlay(self, surface: pygame.Surface, font_small: pygame.font.Font, font_tiny: pygame.font.Font) -> None:
        """Render the expanded dropdown list on top of other UI elements."""
        if not self.is_open:
            return

        opts = self.options
        vis_rows = min(len(opts), self.max_visible)
        m_rect = self.get_menu_rect(surface.get_height())

        # Drop shadow & background
        shadow_rect = m_rect.move(4, 4)
        pygame.draw.rect(surface, (5, 7, 12), shadow_rect, border_radius=8)
        pygame.draw.rect(surface, (16, 20, 30), m_rect, border_radius=8)
        pygame.draw.rect(surface, self.color, m_rect, width=2, border_radius=8)

        cur_idx = self.selected_idx
        for r in range(vis_rows):
            opt_idx = self.scroll_offset + r
            if opt_idx >= len(opts):
                break
            opt_name = str(opts[opt_idx])
            row_rect = pygame.Rect(m_rect.x + 4, m_rect.y + 4 + r * self.row_height, m_rect.width - 14, self.row_height - 2)

            if opt_idx == self.hovered_row:
                pygame.draw.rect(surface, (42, 54, 78), row_rect, border_radius=4)
            elif opt_idx == cur_idx:
                pygame.draw.rect(surface, (28, 38, 56), row_rect, border_radius=4)

            txt_col = GOLD_ACCENT if opt_idx == cur_idx else (TEXT_WHITE if opt_idx == self.hovered_row else TEXT_MUTED)
            display_str = opt_name
            max_txt_w = row_rect.width - 78
            if font_small.size(display_str)[0] > max_txt_w:
                while len(display_str) > 3 and font_small.size(display_str + "..")[0] > max_txt_w:
                    display_str = display_str[:-1]
                display_str += ".."

            t_surf = font_small.render(display_str, True, txt_col)
            surface.blit(t_surf, (row_rect.x + 8, row_rect.y + (row_rect.height - t_surf.get_height()) // 2))

            badge_txt, badge_col = get_bot_category_badge(opt_name)
            b_surf = font_tiny.render(badge_txt, True, badge_col)
            surface.blit(b_surf, (row_rect.right - b_surf.get_width() - 6, row_rect.y + (row_rect.height - b_surf.get_height()) // 2))

        # Scrollbar indicator if needed
        if len(opts) > self.max_visible:
            track_h = m_rect.height - 10
            thumb_h = max(18, int(track_h * (self.max_visible / len(opts))))
            max_scroll = max(1, len(opts) - self.max_visible)
            thumb_y = m_rect.y + 5 + int((track_h - thumb_h) * (self.scroll_offset / max_scroll))
            pygame.draw.rect(surface, (35, 42, 58), pygame.Rect(m_rect.right - 8, m_rect.y + 5, 4, track_h), border_radius=2)
            pygame.draw.rect(surface, self.color, pygame.Rect(m_rect.right - 8, thumb_y, 4, thumb_h), border_radius=2)


class ArcadeHub:
    """Interactive Arcade Hub Application."""

    TABS = ["QUICK DUEL", "4-WAY BRAWL", "TOURNAMENT", "ANALYTICS", "EVO LAB", "BOT LOCKER", "MAP SHOWCASE"]
    BOT_CHOICES = [
        "Human P1",
        "Human P2",
        "Apex_Bot",
        "Bulldozer",
        "Contender_Bot",
        "Shinobi",
        "Guardian",
        "Matador",
        "Ricochet",
        "Chameleon",
        "Heuristic",
        "Rookie_Bot",
        "Tunable",
        "PPO",
        "GA",
        "Random",
    ]

    def __init__(self, screen_size: int = 900):
        self.screen_size = screen_size
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()

        ensure_progressive_checkpoints("checkpoints/ga")

        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Kinetic Sumo - Arcade Hub & Battle Arena")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Consolas", 26, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 17, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 13)
        self.font_tiny = pygame.font.SysFont("Consolas", 11)

        self.current_tab = 0
        self.analytics_subview = 0  # 0: H2H Matrix, 1: Elo Trajectory
        self.audio = SoundEngine(enabled=True)

        # Match configuration state
        self.selected_agent0_idx = 0  # Human P1
        self.selected_agent1_idx = 2  # Apex_Bot
        self.selected_4way_indices = [0, 2, 3, 7]  # Human P1, Apex_Bot, Bulldozer, Matador
        self.map_list = list(MAP_REGISTRY.keys())
        self.selected_map_idx = 0  # Open
        self.rounds_to_win = 3
        self.round_choices = ["1 Round", "2 Rounds", "3 Rounds", "4 Rounds", "5 Rounds"]

        # Tunable Bot Custom Sliders & Savable Presets
        self.custom_bot_config = BotConfig(
            name="Custom Bot", aggression=0.75, dash_frequency=0.6, anchor_bias=0.4, steer_speed=2.8
        )
        self.saved_bot_presets: dict[str, BotConfig] = {}
        self.active_locker_slider_idx = 0
        self.locker_status_text = "Changes apply live to Tunable Bot & saved presets!"

        self.sliders = [
            UISlider(pygame.Rect(480, 240, 320, 10), "Aggression (Throttle)", 0.1, 1.0, self.custom_bot_config.aggression, self.font_small),
            UISlider(pygame.Rect(480, 300, 320, 10), "Dash Frequency", 0.0, 1.0, self.custom_bot_config.dash_frequency, self.font_small),
            UISlider(pygame.Rect(480, 360, 320, 10), "Anchor Readiness", 0.0, 1.0, self.custom_bot_config.anchor_bias, self.font_small),
            UISlider(pygame.Rect(480, 420, 320, 10), "Steering Rate", 1.0, 4.5, self.custom_bot_config.steer_speed, self.font_small),
        ]

        # Tournament State
        self.tournament_ratings: dict[str, float] = {
            "Apex_Champion": 1420.0,
            "Bulldozer": 1280.0,
            "Chameleon": 1265.0,
            "Contender_Bot": 1250.0,
            "Matador": 1245.0,
            "Shinobi": 1240.0,
            "Ricochet": 1230.0,
            "Guardian": 1220.0,
            "Heuristic": 1200.0,
            "PPO": 1180.0,
            "GA": 1100.0,
            "Rookie_Bot": 980.0,
            "Random": 800.0,
        }
        self.tournament_status_text = "Ready to simulate or spectate round-robin ladder."
        self.tournament_stats = self._build_initial_tournament_stats()

        # Evolution Lab State & Reward Shaping Coefficients
        self.evo_generation = 1
        self.evo_pop_size = 16
        self.evo_mutation_rate = 0.08
        self.evo_mutation_sigma = 0.12
        self.evo_elite_ratio = 0.25
        self.evo_aggression_coeff = 1.00
        self.evo_center_coeff = 0.40
        self.evo_win_coeff = 1.00
        self.evo_sparring_choices = [
            "Heuristic",
            "Bulldozer",
            "Shinobi",
            "Guardian",
            "Matador",
            "Ricochet",
            "Chameleon",
            "Contender_Bot",
            "Apex_Bot",
        ]
        self.evo_sparring_idx = 0
        self.active_evo_slider_idx = 0

        self.evo_population = [GAMLP(stochasticity=0.04) for _ in range(self.evo_pop_size)]
        self.evo_best_agent = self.evo_population[0].clone()
        self.evo_history_best = [3.4]
        self.evo_history_mean = [-4.8]
        self.evo_status_text = "Gen 1 ready: adjust sliders or reward shaping and breed live!"
        self.evo_sliders = [
            UISlider(pygame.Rect(65, 240, 290, 8), "Mutation Rate", 0.01, 0.30, self.evo_mutation_rate, self.font_small),
            UISlider(pygame.Rect(65, 278, 290, 8), "Mutation Sigma", 0.02, 0.40, self.evo_mutation_sigma, self.font_small),
            UISlider(pygame.Rect(65, 316, 290, 8), "Elite Ratio", 0.10, 0.50, self.evo_elite_ratio, self.font_small),
            UISlider(pygame.Rect(65, 354, 290, 8), "Pop Size", 8.0, 32.0, float(self.evo_pop_size), self.font_small),
            UISlider(pygame.Rect(65, 392, 290, 8), "Aggression Coeff", 0.00, 2.50, self.evo_aggression_coeff, self.font_small),
            UISlider(pygame.Rect(65, 430, 290, 8), "Center Safety", 0.00, 2.00, self.evo_center_coeff, self.font_small),
            UISlider(pygame.Rect(65, 468, 290, 8), "KO Win Bonus", 0.20, 2.50, self.evo_win_coeff, self.font_small),
        ]

        # Initialize Dropdown Menus across tabs
        self._init_dropdowns()

        self.running = True

    def _init_dropdowns(self) -> None:
        """Create interactive dropdown selectors for Quick Duel, 4-Way Brawl, Tournament, and Evo Lab."""
        # Tab 0: Quick Duel
        self.dd_duel_p1 = UIDropdown(
            pygame.Rect(100, 160, 280, 160),
            lambda: self.BOT_CHOICES,
            lambda: self.selected_agent0_idx,
            lambda v: setattr(self, "selected_agent0_idx", v),
            color=CYAN_ACCENT,
        )
        self.dd_duel_p2 = UIDropdown(
            pygame.Rect(520, 160, 280, 160),
            lambda: self.BOT_CHOICES,
            lambda: self.selected_agent1_idx,
            lambda v: setattr(self, "selected_agent1_idx", v),
            color=ORANGE_ACCENT,
        )
        self.dd_duel_map = UIDropdown(
            pygame.Rect(310, 345, 280, 105),
            lambda: self.map_list,
            lambda: self.selected_map_idx,
            lambda v: setattr(self, "selected_map_idx", v),
            color=GOLD_ACCENT,
        )
        self.dd_duel_rounds = UIDropdown(
            pygame.Rect(310, 470, 280, 65),
            lambda: self.round_choices,
            lambda: self.rounds_to_win - 1,
            lambda v: setattr(self, "rounds_to_win", v + 1),
            color=CYAN_ACCENT,
        )

        # Tab 1: 4-Way Brawl
        slot_rects = [
            pygame.Rect(80, 165, 165, 115),
            pygame.Rect(275, 165, 165, 115),
            pygame.Rect(470, 165, 165, 115),
            pygame.Rect(665, 165, 165, 115),
        ]
        slot_colors = [CYAN_ACCENT, ORANGE_ACCENT, MAGENTA_ACCENT, GREEN_ACCENT]
        self.dd_4way_slots = []
        for i in range(4):
            idx = i
            self.dd_4way_slots.append(
                UIDropdown(
                    slot_rects[idx],
                    lambda: self.BOT_CHOICES,
                    lambda slot=idx: self.selected_4way_indices[slot],
                    lambda val, slot=idx: self.selected_4way_indices.__setitem__(slot, val),
                    color=slot_colors[idx],
                )
            )
        self.dd_4way_map = UIDropdown(
            pygame.Rect(310, 305, 280, 105),
            lambda: self.map_list,
            lambda: self.selected_map_idx,
            lambda v: setattr(self, "selected_map_idx", v),
            color=GOLD_ACCENT,
        )

        # Tab 2: Tournament Map Selector
        self.dd_tourney_map = UIDropdown(
            pygame.Rect(310, 692, 280, 48),
            lambda: self.map_list,
            lambda: self.selected_map_idx,
            lambda v: setattr(self, "selected_map_idx", v),
            color=GOLD_ACCENT,
        )

        # Tab 4: Evo Lab Sparring Partner Selector
        self.dd_evo_sparring = UIDropdown(
            pygame.Rect(148, 191, 207, 22),
            lambda: self.evo_sparring_choices,
            lambda: self.evo_sparring_idx,
            lambda v: setattr(self, "evo_sparring_idx", v),
            color=ORANGE_ACCENT,
        )

    def _get_tab_dropdowns(self, tab_idx: int) -> list[UIDropdown]:
        if tab_idx == 0:
            return [self.dd_duel_p1, self.dd_duel_p2, self.dd_duel_map, self.dd_duel_rounds]
        elif tab_idx == 1:
            return [*self.dd_4way_slots, self.dd_4way_map]
        elif tab_idx == 2:
            return [self.dd_tourney_map]
        elif tab_idx == 4:
            return [self.dd_evo_sparring]
        return []

    def _close_all_dropdowns(self) -> None:
        for t in range(len(self.TABS)):
            for dd in self._get_tab_dropdowns(t):
                dd.is_open = False

    def _build_initial_tournament_stats(self) -> dict:
        """Seed realistic tournament analytics for immediate visualization."""
        bots = ["Apex_Champion", "Bulldozer", "Contender_Bot", "Shinobi", "Guardian", "Heuristic", "Rookie_Bot", "Random"]
        h2h = {a: {b: {"wins": 0, "losses": 0, "draws": 0} for b in bots} for a in bots}
        base_elo = {
            "Apex_Champion": 1420.0,
            "Bulldozer": 1280.0,
            "Contender_Bot": 1250.0,
            "Shinobi": 1240.0,
            "Guardian": 1220.0,
            "Heuristic": 1200.0,
            "Rookie_Bot": 980.0,
            "Random": 800.0,
        }

        # Plausible win-loss records based on Elo differences
        for i, a in enumerate(bots):
            for j, b in enumerate(bots):
                if i == j:
                    continue
                diff = base_elo[a] - base_elo[b]
                p_a = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
                total_games = 10
                w_a = int(round(p_a * total_games))
                w_b = total_games - w_a
                h2h[a][b] = {"wins": w_a, "losses": w_b, "draws": 0}

        # Seed trajectory over 40 matches
        elo_hist = {}
        for b in bots:
            start = 1200.0 if b != "Random" else 1000.0
            target = base_elo[b]
            steps = 40
            pts = []
            for m in range(steps + 1):
                prog = m / float(steps)
                noise = np.sin(m * 0.7 + hash(b) % 7) * 15.0
                curr = start + prog * (target - start) + noise * (1.0 - prog * 0.5)
                pts.append((m, float(curr)))
            elo_hist[b] = pts

        return {
            "head_to_head": h2h,
            "elo_history": elo_hist,
            "total_matches": 40,
            "map_name": "Open",
            "ring_outs": {b: int(base_elo[b] * 0.08) for b in bots},
        }

    def run(self) -> None:
        """Main Hub application loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        any_open = any(dd.is_open for dd in self._get_tab_dropdowns(self.current_tab))
                        if any_open:
                            self._close_all_dropdowns()
                            continue
                        self.running = False
                        return
                    elif event.key == pygame.K_m:
                        self.audio.toggle_mute()
                    elif event.key == pygame.K_TAB:
                        self._close_all_dropdowns()
                        self.current_tab = (self.current_tab + 1) % len(self.TABS)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if self.current_tab == 0:
                            self._launch_1v1_match()
                        elif self.current_tab == 1:
                            self._launch_4way_match()

                # Tab click handling
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    my = event.pos[1]
                    mx = event.pos[0]
                    if 70 <= my <= 110:
                        tab_w = self.screen_size // len(self.TABS)
                        clicked_tab = mx // tab_w
                        if 0 <= clicked_tab < len(self.TABS):
                            self._close_all_dropdowns()
                            self.current_tab = clicked_tab
                            continue

                # Handle tab specific UI events
                self._handle_tab_events(event)

            self._draw_hub_frame()
            pygame.display.flip()
            self.clock.tick(50)

    def _handle_tab_events(self, event: pygame.event.Event) -> None:
        """Process tab-specific dropdowns, clicks, and sliders."""
        # 1. If any dropdown on the current tab is open, let it handle the event first
        tab_dropdowns = self._get_tab_dropdowns(self.current_tab)
        open_dds = [dd for dd in tab_dropdowns if dd.is_open]
        if open_dds:
            for dd in open_dds:
                if dd.handle_event(event, self.screen_size):
                    return
            # If user clicked outside the open dropdown, close it
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for dd in open_dds:
                    dd.is_open = False

        # 2. Check closed dropdowns for opening or hover
        for dd in tab_dropdowns:
            if dd.handle_event(event, self.screen_size):
                # Close other dropdowns
                for other in tab_dropdowns:
                    if other is not dd:
                        other.is_open = False
                return

        if self.current_tab == 0:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(300, 560, 300, 55).collidepoint(event.pos):
                    self._launch_1v1_match()

        elif self.current_tab == 1:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(300, 480, 300, 55).collidepoint(event.pos):
                    self._launch_4way_match()

        elif self.current_tab == 2:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(180, 620, 240, 55).collidepoint(event.pos):
                    self._run_fast_tournament_simulation()
                elif pygame.Rect(480, 620, 240, 55).collidepoint(event.pos):
                    self._run_live_tournament_spectator()

        elif self.current_tab == 3:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(180, 155, 250, 34).collidepoint(event.pos):
                    self.analytics_subview = 0
                elif pygame.Rect(470, 155, 250, 34).collidepoint(event.pos):
                    self.analytics_subview = 1

        elif self.current_tab == 4:
            for idx, s in enumerate(self.evo_sliders):
                changed = s.handle_event(event)
                if s.is_hovered or s.dragging:
                    self.active_evo_slider_idx = idx
                if changed:
                    self.evo_mutation_rate = self.evo_sliders[0].val
                    self.evo_mutation_sigma = self.evo_sliders[1].val
                    self.evo_elite_ratio = self.evo_sliders[2].val
                    self.evo_pop_size = int(self.evo_sliders[3].val)
                    if len(self.evo_sliders) >= 7:
                        self.evo_aggression_coeff = self.evo_sliders[4].val
                        self.evo_center_coeff = self.evo_sliders[5].val
                        self.evo_win_coeff = self.evo_sliders[6].val

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(65, 582, 140, 32).collidepoint(event.pos):
                    self._evo_step_generation(turbo_count=1)
                elif pygame.Rect(215, 582, 140, 32).collidepoint(event.pos):
                    self._evo_step_generation(turbo_count=10)
                elif pygame.Rect(65, 620, 140, 32).collidepoint(event.pos):
                    self._mutate_champion()
                elif pygame.Rect(215, 620, 140, 32).collidepoint(event.pos):
                    self._watch_evo_duel()
                elif pygame.Rect(65, 658, 290, 32).collidepoint(event.pos):
                    self._export_evo_champion()

        elif self.current_tab == 5:
            for idx, s in enumerate(self.sliders):
                changed = s.handle_event(event)
                if s.is_hovered or s.dragging:
                    self.active_locker_slider_idx = idx
                if changed:
                    self.custom_bot_config.aggression = self.sliders[0].val
                    self.custom_bot_config.dash_frequency = self.sliders[1].val
                    self.custom_bot_config.anchor_bias = self.sliders[2].val
                    self.custom_bot_config.steer_speed = self.sliders[3].val

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Reset Defaults (contains 500, 525)
                if pygame.Rect(480, 510, 165, 38).collidepoint(event.pos):
                    self.custom_bot_config.aggression = 0.75
                    self.custom_bot_config.dash_frequency = 0.6
                    self.custom_bot_config.anchor_bias = 0.4
                    self.custom_bot_config.steer_speed = 2.8
                    self.sliders[0].val = 0.75
                    self.sliders[1].val = 0.6
                    self.sliders[2].val = 0.4
                    self.sliders[3].val = 2.8
                    self.locker_status_text = "Reset Tunable Bot parameters to balanced defaults."
                # Save Preset (right of Reset Defaults)
                elif pygame.Rect(655, 510, 165, 38).collidepoint(event.pos):
                    self._save_custom_bot_preset()
                # Mutate Jitter (contains 500, 580)
                elif pygame.Rect(480, 565, 165, 38).collidepoint(event.pos):
                    self.custom_bot_config.aggression = float(np.clip(self.sliders[0].val + np.random.normal(0, 0.1), 0.1, 1.0))
                    self.custom_bot_config.dash_frequency = float(np.clip(self.sliders[1].val + np.random.normal(0, 0.1), 0.0, 1.0))
                    self.custom_bot_config.anchor_bias = float(np.clip(self.sliders[2].val + np.random.normal(0, 0.1), 0.0, 1.0))
                    self.custom_bot_config.steer_speed = float(np.clip(self.sliders[3].val + np.random.normal(0, 0.3), 1.0, 4.5))
                    self.sliders[0].val = self.custom_bot_config.aggression
                    self.sliders[1].val = self.custom_bot_config.dash_frequency
                    self.sliders[2].val = self.custom_bot_config.anchor_bias
                    self.sliders[3].val = self.custom_bot_config.steer_speed
                    self.locker_status_text = "Applied stochastic jitter to Tunable Bot sliders!"
                # 1-Click Test in Arena (right of Mutate Jitter)
                elif pygame.Rect(655, 565, 165, 38).collidepoint(event.pos):
                    self._test_custom_bot_in_arena()

        elif self.current_tab == 6:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i in range(len(self.map_list)):
                    tx = 40 + i * 92
                    ty = 610
                    thumb_rect = pygame.Rect(tx, ty, 85, 58)
                    if thumb_rect.collidepoint(event.pos):
                        self.selected_map_idx = i

    def _save_custom_bot_preset(self) -> str:
        """Save current Bot Locker slider configuration as a named preset available across all dropdowns."""
        preset_idx = len(self.saved_bot_presets) + 1
        preset_name = f"Custom_Preset_{preset_idx}"
        cfg = BotConfig(
            name=preset_name,
            aggression=float(self.custom_bot_config.aggression),
            dash_frequency=float(self.custom_bot_config.dash_frequency),
            anchor_bias=float(self.custom_bot_config.anchor_bias),
            steer_speed=float(self.custom_bot_config.steer_speed),
        )
        self.saved_bot_presets[preset_name] = cfg
        if preset_name not in self.BOT_CHOICES:
            self.BOT_CHOICES.append(preset_name)
        self.tournament_ratings[preset_name] = 1210.0
        self.locker_status_text = f"Saved '{preset_name}' to Fighter Dropdowns & Tournament!"
        return preset_name

    def _test_custom_bot_in_arena(self) -> None:
        """Launch an immediate 1-click 1v1 test match in the arena with the currently tuned bot."""
        from src.ui.play import run_interactive_match
        map_name = self.map_list[self.selected_map_idx]
        opp_name = self.BOT_CHOICES[self.selected_agent1_idx]
        if "human" in opp_name.lower() or "tunable" in opp_name.lower():
            opp_type = "heuristic"
        else:
            opp_type = opp_name.lower()

        run_interactive_match(
            agent0_type="tunable",
            agent1_type=opp_type,
            map_name=map_name,
            rounds_to_win=2,
            custom_bot_config=self.custom_bot_config,
            return_to_hub=True,
        )
        self._restore_hub_display()

    def _restore_hub_display(self) -> None:
        """Restore Pygame display, fonts, and window caption after returning from a child match or spectator view."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if not pygame.display.get_init():
            pygame.display.init()

        surf = pygame.display.get_surface()
        try:
            if surf is not None and surf.get_size() == (self.screen_size, self.screen_size):
                self.screen = surf
            else:
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        except pygame.error:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))

        pygame.display.set_caption("Kinetic Sumo - Arcade Hub & Battle Arena")

        self.font_large = pygame.font.SysFont("Consolas", 26, bold=True)
        self.font_med = pygame.font.SysFont("Consolas", 17, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 13)
        self.font_tiny = pygame.font.SysFont("Consolas", 11)
        for s in self.sliders:
            s.font = self.font_small
        for s in self.evo_sliders:
            s.font = self.font_small

    def _draw_hub_frame(self) -> None:
        """Render active Hub screen."""
        try:
            self.screen.fill(BG_DARK)
        except pygame.error:
            self._restore_hub_display()
            self.screen.fill(BG_DARK)

        # Title Header
        title_surf = self.font_large.render("KINETIC SUMO : ARCADE ARENA & LAB", True, CYAN_ACCENT)
        self.screen.blit(title_surf, (self.screen_size // 2 - title_surf.get_width() // 2, 20))

        # Audio indicator
        mute_txt = "AUDIO: OFF (M)" if self.audio.is_muted else "AUDIO: ON (M)"
        mute_col = (255, 80, 80) if self.audio.is_muted else GREEN_ACCENT
        m_surf = self.font_small.render(mute_txt, True, mute_col)
        self.screen.blit(m_surf, (self.screen_size - m_surf.get_width() - 25, 25))

        # Tab Navigation Bar (7 Tabs)
        tab_w = self.screen_size // len(self.TABS)
        for i, tab_name in enumerate(self.TABS):
            tab_rect = pygame.Rect(i * tab_w, 70, tab_w, 40)
            is_active = i == self.current_tab
            bg_col = (30, 36, 50) if is_active else PANEL_BG
            pygame.draw.rect(self.screen, bg_col, tab_rect)
            pygame.draw.rect(self.screen, PANEL_BORDER, tab_rect, 1)
            if is_active:
                pygame.draw.line(self.screen, CYAN_ACCENT, (tab_rect.left, tab_rect.bottom - 2), (tab_rect.right, tab_rect.bottom - 2), 3)

            col = CYAN_ACCENT if is_active else TEXT_MUTED
            t_surf = self.font_small.render(tab_name, True, col)
            self.screen.blit(
                t_surf,
                (tab_rect.x + (tab_w - t_surf.get_width()) // 2, tab_rect.y + 12),
            )

        # Render Active Tab Content
        if self.current_tab == 0:
            self._draw_quick_duel_tab()
        elif self.current_tab == 1:
            self._draw_4way_tab()
        elif self.current_tab == 2:
            self._draw_tournament_tab()
        elif self.current_tab == 3:
            self._draw_analytics_tab()
        elif self.current_tab == 4:
            self._draw_evo_lab_tab()
        elif self.current_tab == 5:
            self._draw_bot_locker_tab()
        elif self.current_tab == 6:
            self._draw_map_showcase_tab()

        # Render any open Dropdown Menu Overlay on top of the active tab
        for dd in self._get_tab_dropdowns(self.current_tab):
            if dd.is_open:
                dd.draw_overlay(self.screen, self.font_small, self.font_tiny)

    def _draw_quick_duel_tab(self) -> None:
        title = self.font_med.render("QUICK DUEL : 1v1 CUSTOM MATCHMAKER", True, GOLD_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        vs_surf = self.font_large.render("VS", True, GOLD_ACCENT)
        self.screen.blit(vs_surf, (self.screen_size // 2 - vs_surf.get_width() // 2, 230))

        self.selected_agent0_idx = max(0, min(self.selected_agent0_idx, len(self.BOT_CHOICES) - 1))
        a0_name = self.BOT_CHOICES[self.selected_agent0_idx]
        self._draw_card(
            100, 160, 280, 160,
            "PLAYER 1 (CYAN)",
            a0_name,
            CYAN_ACCENT,
            "v Click to open fighter dropdown",
            is_open=self.dd_duel_p1.is_open,
        )

        self.selected_agent1_idx = max(0, min(self.selected_agent1_idx, len(self.BOT_CHOICES) - 1))
        a1_name = self.BOT_CHOICES[self.selected_agent1_idx]
        self._draw_card(
            520, 160, 280, 160,
            "PLAYER 2 (ORANGE)",
            a1_name,
            ORANGE_ACCENT,
            "v Click to open fighter dropdown",
            is_open=self.dd_duel_p2.is_open,
        )

        map_name = self.map_list[self.selected_map_idx]
        self._draw_card(
            310, 345, 280, 105,
            "ARENA MAP",
            map_name.upper(),
            GOLD_ACCENT,
            "v Click to open map dropdown",
            is_open=self.dd_duel_map.is_open,
        )

        # Rounds to win dropdown card
        round_rect = pygame.Rect(310, 470, 280, 65)
        pygame.draw.rect(self.screen, PANEL_BG, round_rect, border_radius=6)
        pygame.draw.rect(self.screen, CYAN_ACCENT if self.dd_duel_rounds.is_open else PANEL_BORDER, round_rect, 1, border_radius=6)
        r_text = self.font_med.render(f"ROUNDS TO WIN: {self.rounds_to_win}", True, TEXT_WHITE)
        self.screen.blit(r_text, (round_rect.centerx - r_text.get_width() // 2, round_rect.y + 12))
        hint_surf = self.font_tiny.render("v Click to select match length (1 - 5)", True, TEXT_MUTED)
        self.screen.blit(hint_surf, (round_rect.centerx - hint_surf.get_width() // 2, round_rect.y + 38))

        launch_btn = UIButton(pygame.Rect(300, 560, 300, 55), "LAUNCH 1v1 MATCH", self.font_med, CYAN_ACCENT)
        launch_btn.draw(self.screen)

        controls_surf = self.font_small.render("P1: WASD + Space/Shift  |  P2: Arrows + Enter/RShift  |  ESC: Menu", True, TEXT_MUTED)
        self.screen.blit(controls_surf, (self.screen_size // 2 - controls_surf.get_width() // 2, 640))

        xray_hint = self.font_tiny.render(
            "In-Match Controls: [I] AI X-Ray Overlay  |  [L] Lidar  |  [P] Pause  |  [.] Step 1 Frame  |  [-/+] Speed",
            True,
            CYAN_ACCENT,
        )
        self.screen.blit(xray_hint, (self.screen_size // 2 - xray_hint.get_width() // 2, 668))

    def _draw_4way_tab(self) -> None:
        title = self.font_med.render("4-WAY BATTLE ROYALE : LAST DISC STANDING", True, GOLD_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        colors = [CYAN_ACCENT, ORANGE_ACCENT, MAGENTA_ACCENT, GREEN_ACCENT]
        labels = ["SLOT 1 (Cyan)", "SLOT 2 (Orange)", "SLOT 3 (Magenta)", "SLOT 4 (Emerald)"]
        card_positions = [
            (80, 165, 165, 115),
            (275, 165, 165, 115),
            (470, 165, 165, 115),
            (665, 165, 165, 115),
        ]

        for i in range(4):
            x, y, w, h = card_positions[i]
            self.selected_4way_indices[i] = max(0, min(self.selected_4way_indices[i], len(self.BOT_CHOICES) - 1))
            bot_name = self.BOT_CHOICES[self.selected_4way_indices[i]]
            self._draw_card(
                x, y, w, h,
                labels[i],
                bot_name,
                colors[i],
                "v Select fighter",
                is_open=self.dd_4way_slots[i].is_open,
            )

        map_name = self.map_list[self.selected_map_idx]
        self._draw_card(
            310, 305, 280, 105,
            "ARENA MAP",
            map_name.upper(),
            GOLD_ACCENT,
            "v Click to open map dropdown",
            is_open=self.dd_4way_map.is_open,
        )

        rules_surf = self.font_small.render("Rules: 4-Way Pairwise Physics | Instant Ring-Outs | Dynamic Ring Shrink", True, TEXT_WHITE)
        self.screen.blit(rules_surf, (self.screen_size // 2 - rules_surf.get_width() // 2, 435))

        launch_btn = UIButton(pygame.Rect(300, 480, 300, 55), "LAUNCH 4-WAY BRAWL", self.font_med, GOLD_ACCENT)
        launch_btn.draw(self.screen)

        controls_surf = self.font_small.render("P1: WASD+Space  |  P2: Arrows+Enter  |  AI: Autonomous MARL  |  ESC: Menu", True, TEXT_MUTED)
        self.screen.blit(controls_surf, (self.screen_size // 2 - controls_surf.get_width() // 2, 558))

        xray_hint = self.font_tiny.render(
            "In-Match Controls: [I] AI X-Ray Overlay  |  [L] Lidar  |  [P] Pause  |  [.] Step 1 Frame  |  [-/+] Speed",
            True,
            CYAN_ACCENT,
        )
        self.screen.blit(xray_hint, (self.screen_size // 2 - xray_hint.get_width() // 2, 584))

    def _draw_tournament_tab(self) -> None:
        title = self.font_med.render("AUTONOMOUS LEAGUE & ELO RATING LADDER", True, GOLD_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        table_rect = pygame.Rect(160, 160, 580, 410)
        pygame.draw.rect(self.screen, PANEL_BG, table_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, table_rect, 2, border_radius=8)

        # Header
        self.screen.blit(self.font_med.render("RANK", True, TEXT_MUTED), (195, 175))
        self.screen.blit(self.font_med.render("COMPETITOR", True, TEXT_MUTED), (280, 175))
        self.screen.blit(self.font_med.render("ELO RATING", True, TEXT_MUTED), (505, 175))
        self.screen.blit(self.font_med.render("TIER", True, TEXT_MUTED), (645, 175))
        pygame.draw.line(self.screen, PANEL_BORDER, (180, 205), (720, 205), 1)

        sorted_ratings = sorted(self.tournament_ratings.items(), key=lambda x: x[1], reverse=True)
        for i, (bot, elo) in enumerate(sorted_ratings[:8]):
            row_y = 220 + i * 38
            rank_col = GOLD_ACCENT if i == 0 else (TEXT_WHITE if i < 3 else TEXT_MUTED)
            self.screen.blit(self.font_med.render(f"#{i+1}", True, rank_col), (195, row_y))
            b_font = self.font_med
            if b_font.size(bot)[0] > 210:
                b_font = self.font_small
            if b_font.size(bot)[0] > 210:
                b_font = self.font_tiny
            b_y = row_y + (2 if b_font != self.font_med else 0)
            self.screen.blit(b_font.render(bot, True, TEXT_WHITE), (280, b_y))
            self.screen.blit(self.font_med.render(f"{elo:.1f}", True, CYAN_ACCENT), (505, row_y))

            tier_txt = "APEX" if elo >= 1350 else ("CONTENDER" if elo >= 1200 else "ROOKIE")
            tier_col = GOLD_ACCENT if elo >= 1350 else (GREEN_ACCENT if elo >= 1200 else ORANGE_ACCENT)
            self.screen.blit(self.font_tiny.render(tier_txt, True, tier_col), (645, row_y + 4))

        st_surf = self.font_small.render(self.tournament_status_text, True, TEXT_MUTED)
        self.screen.blit(st_surf, (self.screen_size // 2 - st_surf.get_width() // 2, 585))

        # Simulation Buttons (Hitbox exact: 180, 620 and 480, 620)
        fast_btn = UIButton(pygame.Rect(180, 620, 240, 55), "FAST SIMULATE (2s)", self.font_med, CYAN_ACCENT)
        fast_btn.draw(self.screen)

        live_btn = UIButton(pygame.Rect(480, 620, 240, 55), "WATCH LIVE MATCHES", self.font_med, GOLD_ACCENT)
        live_btn.draw(self.screen)

        # Tournament Arena Map Dropdown Card
        map_name = self.map_list[self.selected_map_idx]
        m_rect = self.dd_tourney_map.rect
        pygame.draw.rect(self.screen, PANEL_BG, m_rect, border_radius=6)
        pygame.draw.rect(self.screen, GOLD_ACCENT if self.dd_tourney_map.is_open else PANEL_BORDER, m_rect, 1, border_radius=6)
        m_lbl = self.font_small.render(f"TOURNAMENT ARENA: {map_name.upper()} v", True, GOLD_ACCENT)
        self.screen.blit(m_lbl, (m_rect.centerx - m_lbl.get_width() // 2, m_rect.y + 8))
        m_hint = self.font_tiny.render("Click to select arena map for League evaluation", True, TEXT_MUTED)
        self.screen.blit(m_hint, (m_rect.centerx - m_hint.get_width() // 2, m_rect.y + 28))

    def _draw_analytics_tab(self) -> None:
        """Render dedicated deep statistics: Confusion Matrix and Elo-over-time Chart."""
        title = self.font_med.render("TOURNAMENT ANALYTICS & TELEMETRY", True, GOLD_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        # View Toggle Buttons (y=155, h=34)
        btn0_col = CYAN_ACCENT if self.analytics_subview == 0 else TEXT_MUTED
        btn1_col = CYAN_ACCENT if self.analytics_subview == 1 else TEXT_MUTED
        btn0 = UIButton(pygame.Rect(180, 155, 250, 34), "H2H MATCHUP MATRIX", self.font_small, btn0_col)
        btn1 = UIButton(pygame.Rect(470, 155, 250, 34), "ELO TRAJECTORY PLOT", self.font_small, btn1_col)
        btn0.draw(self.screen)
        btn1.draw(self.screen)

        if self.analytics_subview == 0:
            self._draw_h2h_confusion_matrix()
        else:
            self._draw_elo_trajectory_chart()

        # Bottom: Per-Bot Stats & Ring-Out Tendencies
        self._draw_per_bot_metrics_table()

    def _draw_h2h_confusion_matrix(self) -> None:
        """Render N x N matchup win/loss confusion matrix."""
        h2h = self.tournament_stats.get("head_to_head", {})
        if h2h:
            key_bots = list(h2h.keys())[:8]
        else:
            key_bots = list(self.tournament_ratings.keys())[:8]

        short_names = [
            b.replace("Champion", "").replace("Agent", "").replace("Bot", "").replace("_", "")[:4].upper()
            for b in key_bots
        ]

        n = max(1, len(key_bots))
        gx = 130
        gy = 222
        total_avail_w = self.screen_size - gx - 90
        cw = min(92, total_avail_w // n)
        ch = min(36, 280 // n)

        # Header Columns
        for j, s_name in enumerate(short_names):
            hdr = self.font_tiny.render(s_name, True, CYAN_ACCENT)
            self.screen.blit(hdr, (gx + 70 + j * cw + (cw - hdr.get_width()) // 2, gy - 18))

        # Rows
        for i, b_a in enumerate(key_bots):
            rh = self.font_tiny.render(short_names[i], True, ORANGE_ACCENT)
            self.screen.blit(rh, (gx + 10, gy + i * ch + max(2, (ch - 14) // 2)))

            for j, b_b in enumerate(key_bots):
                c_rect = pygame.Rect(gx + 70 + j * cw, gy + i * ch, cw - 4, ch - 4)

                if i == j:
                    pygame.draw.rect(self.screen, (25, 30, 42), c_rect, border_radius=4)
                    diag_txt = self.font_tiny.render("---", True, (80, 90, 110))
                    self.screen.blit(diag_txt, (c_rect.centerx - diag_txt.get_width() // 2, c_rect.centery - 8))
                else:
                    rec = h2h.get(b_a, {}).get(b_b, {"wins": 0, "losses": 0, "draws": 0})
                    w = rec.get("wins", 0)
                    l = rec.get("losses", 0)
                    total = max(1, w + l + rec.get("draws", 0))
                    wr = w / total

                    if wr >= 0.60:
                        bg = (18, 55, 36)
                        txt_col = GREEN_ACCENT
                    elif wr <= 0.40:
                        bg = (55, 20, 26)
                        txt_col = RED_ACCENT
                    else:
                        bg = (35, 42, 58)
                        txt_col = TEXT_WHITE

                    pygame.draw.rect(self.screen, bg, c_rect, border_radius=4)
                    pygame.draw.rect(self.screen, PANEL_BORDER, c_rect, width=1, border_radius=4)

                    w_txt = self.font_tiny.render(f"{w}W-{l}L", True, txt_col)
                    pct_txt = self.font_tiny.render(f"{int(wr*100)}%", True, TEXT_MUTED)
                    self.screen.blit(w_txt, (c_rect.centerx - w_txt.get_width() // 2, c_rect.y + 2))
                    self.screen.blit(pct_txt, (c_rect.centerx - pct_txt.get_width() // 2, c_rect.y + 17))

        note = self.font_tiny.render(
            "Matrix shows Row vs Column: Green = Dominant (>60%), Red = Disadvantaged (<40%).",
            True,
            TEXT_MUTED,
        )
        self.screen.blit(note, (self.screen_size // 2 - note.get_width() // 2, gy + len(key_bots) * ch + 8))

    def _draw_elo_trajectory_chart(self) -> None:
        """Render multi-line Elo rating trajectory across match series."""
        chart_rect = pygame.Rect(120, 205, 660, 275)
        pygame.draw.rect(self.screen, PANEL_BG, chart_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, chart_rect, 1, border_radius=8)

        elo_hist = self.tournament_stats.get("elo_history", {})
        colors = {
            "Apex_Champion": GOLD_ACCENT,
            "Bulldozer": ORANGE_ACCENT,
            "Contender_Bot": CYAN_ACCENT,
            "Shinobi": MAGENTA_ACCENT,
            "Guardian": GREEN_ACCENT,
            "Heuristic": (160, 180, 255),
            "Rookie_Bot": (255, 140, 140),
            "Random": (120, 120, 120),
        }
        for k in elo_hist.keys():
            if k not in colors:
                colors[k] = (255, 100, 220)

        min_elo = 750.0
        max_elo = 1600.0

        for elo_val in [800, 1000, 1200, 1400, 1600]:
            rel_y = (elo_val - min_elo) / (max_elo - min_elo)
            y_px = int(chart_rect.bottom - rel_y * chart_rect.height)
            pygame.draw.line(self.screen, (32, 38, 54), (chart_rect.left, y_px), (chart_rect.right, y_px), 1)
            lbl = self.font_tiny.render(f"{elo_val}", True, TEXT_MUTED)
            self.screen.blit(lbl, (chart_rect.left - 42, y_px - 8))

        max_match = 1
        for pts in elo_hist.values():
            if pts:
                max_match = max(max_match, max(p[0] for p in pts))

        for bot, pts in elo_hist.items():
            if len(pts) < 2:
                continue
            clr = colors.get(bot, TEXT_WHITE)
            poly_points = []
            for m_idx, elo_val in pts:
                x_rel = m_idx / max_match
                y_rel = (elo_val - min_elo) / (max_elo - min_elo)
                px = int(chart_rect.left + x_rel * chart_rect.width)
                py = int(chart_rect.bottom - y_rel * chart_rect.height)
                poly_points.append((px, py))

            if len(poly_points) >= 2:
                pygame.draw.lines(self.screen, clr, False, poly_points, 2)
                pygame.draw.circle(self.screen, clr, poly_points[-1], 4)

        bots_to_show = list(elo_hist.keys())[:8]
        leg_x = chart_rect.left + 10
        leg_y = chart_rect.bottom + 10
        for i, bot in enumerate(bots_to_show):
            clr = colors.get(bot, TEXT_WHITE)
            col_idx = i % 4
            row_idx = i // 4
            item_x = leg_x + col_idx * 160
            item_y = leg_y + row_idx * 16
            pygame.draw.rect(self.screen, clr, pygame.Rect(item_x, item_y + 2, 10, 10))
            self.screen.blit(self.font_tiny.render(bot.replace("_", " ")[:16], True, TEXT_WHITE), (item_x + 14, item_y))

    def _draw_per_bot_metrics_table(self) -> None:
        """Render Per-Bot Win Rate, Games, and Ring-Out Tendencies card table."""
        card_rect = pygame.Rect(120, 540, 660, 195)
        pygame.draw.rect(self.screen, PANEL_BG, card_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, card_rect, 1, border_radius=8)

        self.screen.blit(self.font_tiny.render("AGENT", True, TEXT_MUTED), (140, 552))
        self.screen.blit(self.font_tiny.render("ELO", True, TEXT_MUTED), (290, 552))
        self.screen.blit(self.font_tiny.render("WIN %", True, TEXT_MUTED), (370, 552))
        self.screen.blit(self.font_tiny.render("GAMES", True, TEXT_MUTED), (460, 552))
        self.screen.blit(self.font_tiny.render("RING-OUT KO RATIO", True, TEXT_MUTED), (550, 552))
        self.screen.blit(self.font_tiny.render("TIER", True, TEXT_MUTED), (695, 552))
        pygame.draw.line(self.screen, PANEL_BORDER, (135, 572), (765, 572), 1)

        ro_stats = self.tournament_stats.get("ring_outs", {})
        sorted_ratings = sorted(self.tournament_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
        for idx, (b_name, elo) in enumerate(sorted_ratings):
            ry = 582 + idx * 27
            h2h = self.tournament_stats.get("head_to_head", {}).get(b_name, {})
            tot_w = sum(v.get("wins", 0) for v in h2h.values())
            tot_l = sum(v.get("losses", 0) for v in h2h.values())
            tot_g = max(1, tot_w + tot_l)
            wr = tot_w / float(tot_g)

            bot_ro = ro_stats.get(b_name, {})
            if isinstance(bot_ro, dict):
                ko_inflicted = bot_ro.get("inflicted", 0)
                ko_taken = bot_ro.get("taken", 0)
            else:
                ko_inflicted = int(bot_ro) if bot_ro else 0
                ko_taken = max(1, tot_l)

            ko_taken_denom = max(1, ko_taken)
            ko_ratio = ko_inflicted / float(ko_taken_denom)

            tier = "APEX" if elo >= 1350 else ("CONTENDER" if elo >= 1200 else "ROOKIE")
            t_col = GOLD_ACCENT if elo >= 1350 else (GREEN_ACCENT if elo >= 1200 else ORANGE_ACCENT)

            b_display = b_name
            if self.font_tiny.size(b_display)[0] > 140:
                while len(b_display) > 3 and self.font_tiny.size(b_display + "...")[0] > 140:
                    b_display = b_display[:-1]
                b_display += "..."
            self.screen.blit(self.font_tiny.render(b_display, True, TEXT_WHITE), (140, ry))
            self.screen.blit(self.font_tiny.render(f"{elo:.1f}", True, CYAN_ACCENT), (290, ry))
            self.screen.blit(self.font_tiny.render(f"{wr*100:.1f}%", True, GREEN_ACCENT if wr >= 0.55 else TEXT_WHITE), (370, ry))
            self.screen.blit(self.font_tiny.render(f"{tot_g}", True, TEXT_MUTED), (460, ry))
            self.screen.blit(self.font_tiny.render(f"{ko_ratio:.2f}x ({ko_inflicted}/{ko_taken})", True, TEXT_MUTED), (550, ry))
            self.screen.blit(self.font_tiny.render(tier, True, t_col), (695, ry))

    def _draw_evo_lab_tab(self) -> None:
        """Render live Neuroevolution breeding, reward shaping sliders, parameter helper card, and genome heatmap."""
        title = self.font_med.render("NEUROEVOLUTION BREEDING & GENETICS LAB", True, CYAN_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        # Left Column: Sliders, Live Parameter Helper, & Controls
        left_box = pygame.Rect(50, 155, 320, 600)
        pygame.draw.rect(self.screen, PANEL_BG, left_box, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, left_box, 1, border_radius=8)

        lbl = self.font_med.render("EVO & REWARD SHAPING", True, GOLD_ACCENT)
        self.screen.blit(lbl, (65, 165))

        # Sparring partner dropdown selector row (y=191..213)
        sp_lbl = self.font_tiny.render("SPARRING FOE:", True, TEXT_MUTED)
        self.screen.blit(sp_lbl, (65, 195))
        sp_rect = self.dd_evo_sparring.rect
        pygame.draw.rect(self.screen, (28, 34, 48), sp_rect, border_radius=4)
        pygame.draw.rect(self.screen, ORANGE_ACCENT if self.dd_evo_sparring.is_open else PANEL_BORDER, sp_rect, 1, border_radius=4)
        sp_name = self.evo_sparring_choices[self.evo_sparring_idx]
        sp_txt = self.font_tiny.render(f"{sp_name} v", True, ORANGE_ACCENT)
        self.screen.blit(sp_txt, (sp_rect.x + 8, sp_rect.y + 4))

        for idx, s in enumerate(self.evo_sliders):
            accent = GREEN_ACCENT if idx >= 4 else CYAN_ACCENT
            s.draw(self.screen, accent_color=accent)

        # Live "Under the Hood" Parameter Helper Inspector Card (y=484..574)
        self._draw_evo_parameter_helper_card(pygame.Rect(65, 484, 290, 90))

        # Compact 2-Column Action Buttons (y=582..690)
        btn_step = UIButton(pygame.Rect(65, 582, 140, 32), "STEP 1 GEN", self.font_small, CYAN_ACCENT)
        btn_turbo = UIButton(pygame.Rect(215, 582, 140, 32), "TURBO (10 GENS)", self.font_small, GOLD_ACCENT)
        btn_mutate = UIButton(pygame.Rect(65, 620, 140, 32), "MUTATE CHAMP", self.font_small, (255, 120, 200))
        btn_duel = UIButton(pygame.Rect(215, 620, 140, 32), "WATCH DUEL", self.font_small, GREEN_ACCENT)
        btn_export = UIButton(pygame.Rect(65, 658, 290, 32), "EXPORT CHAMPION TO LADDER", self.font_small, MAGENTA_ACCENT)

        btn_step.draw(self.screen)
        btn_turbo.draw(self.screen)
        btn_mutate.draw(self.screen)
        btn_duel.draw(self.screen)
        btn_export.draw(self.screen)

        # Status text with clean word wrapping
        st_lines = self._wrap_text(self.evo_status_text, self.font_tiny, 290)
        for s_idx, s_line in enumerate(st_lines[:2]):
            st = self.font_tiny.render(s_line, True, TEXT_MUTED)
            self.screen.blit(st, (65, 698 + s_idx * 16))

        # Right Column: Fitness Curves & Champion Genome Sensory/Weight Heatmap
        right_box = pygame.Rect(390, 155, 460, 600)
        pygame.draw.rect(self.screen, PANEL_BG, right_box, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, right_box, 1, border_radius=8)

        c_title = self.font_med.render(f"GENERATION FITNESS CURVES (GEN {self.evo_generation})", True, GREEN_ACCENT)
        self.screen.blit(c_title, (410, 168))

        # Curve plot area
        plot_rect = pygame.Rect(410, 195, 420, 195)
        pygame.draw.rect(self.screen, (16, 20, 28), plot_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, plot_rect, 1, border_radius=6)

        for y_step in [0.25, 0.5, 0.75]:
            py = int(plot_rect.top + y_step * plot_rect.height)
            pygame.draw.line(self.screen, (28, 34, 48), (plot_rect.left, py), (plot_rect.right, py), 1)

        gens = min(len(self.evo_history_best), len(self.evo_history_mean))
        all_vals = self.evo_history_best[:gens] + self.evo_history_mean[:gens]
        if not all_vals:
            all_vals = [0.0]
        min_fit = min(-10.0, min(all_vals))
        max_fit = max(15.0, max(all_vals) + 2.0)
        denom = max(1e-5, max_fit - min_fit)

        lbl_max = self.font_tiny.render(f"Max: {max_fit:+.1f}", True, (90, 105, 130))
        lbl_min = self.font_tiny.render(f"Min: {min_fit:+.1f}", True, (90, 105, 130))
        self.screen.blit(lbl_max, (plot_rect.left + 8, plot_rect.top + 6))
        self.screen.blit(lbl_min, (plot_rect.left + 8, plot_rect.bottom - 18))

        if gens >= 2:
            pts_best = []
            pts_mean = []
            for g_idx in range(gens):
                x_rel = g_idx / float(max(1, gens - 1))
                y_best = (self.evo_history_best[g_idx] - min_fit) / denom
                y_mean = (self.evo_history_mean[g_idx] - min_fit) / denom
                px = int(plot_rect.left + x_rel * plot_rect.width)
                pts_best.append((px, int(plot_rect.bottom - y_best * plot_rect.height)))
                pts_mean.append((px, int(plot_rect.bottom - y_mean * plot_rect.height)))

            pygame.draw.lines(self.screen, GREEN_ACCENT, False, pts_best, 2)
            pygame.draw.lines(self.screen, CYAN_ACCENT, False, pts_mean, 2)
            pygame.draw.circle(self.screen, GREEN_ACCENT, pts_best[-1], 4)
            pygame.draw.circle(self.screen, CYAN_ACCENT, pts_mean[-1], 4)
        elif gens == 1:
            y_best = (self.evo_history_best[0] - min_fit) / denom
            y_mean = (self.evo_history_mean[0] - min_fit) / denom
            px = int(plot_rect.left + 20)
            py_b = int(plot_rect.bottom - y_best * plot_rect.height)
            py_m = int(plot_rect.bottom - y_mean * plot_rect.height)
            pygame.draw.circle(self.screen, GREEN_ACCENT, (px, py_b), 5)
            pygame.draw.circle(self.screen, CYAN_ACCENT, (px, py_m), 4)

        # Legend
        pygame.draw.rect(self.screen, GREEN_ACCENT, pygame.Rect(420, 400, 10, 10))
        self.screen.blit(self.font_tiny.render("Best Elite Fitness", True, GREEN_ACCENT), (436, 398))
        pygame.draw.rect(self.screen, CYAN_ACCENT, pygame.Rect(580, 400, 10, 10))
        self.screen.blit(self.font_tiny.render("Population Mean", True, CYAN_ACCENT), (596, 398))

        # Champion Genome & Sensory Heatmap Inspector Card (y=420..735)
        self._draw_evo_genome_heatmap_card(pygame.Rect(410, 420, 420, 318))

    def _draw_evo_parameter_helper_card(self, rect: pygame.Rect) -> None:
        """Render the live Under-the-Hood parameter explanation & visual preview bar in Evo Lab."""
        pygame.draw.rect(self.screen, (15, 19, 28), rect, border_radius=6)
        pygame.draw.rect(self.screen, GOLD_ACCENT, rect, width=1, border_radius=6)

        idx = self.active_evo_slider_idx
        total_w = len(self.evo_best_agent.get_weights())  # 2659
        if idx == 0:
            cnt = int(round(self.evo_mutation_rate * total_w))
            title = f"HELPER: MUTATION RATE ({self.evo_mutation_rate*100:.1f}%)"
            l1 = f"Mutates ~{cnt:,} / {total_w:,} synapses per child."
            l2 = "Low (<5%) refines; High (>18%) explores."
            ratio = min(1.0, self.evo_mutation_rate / 0.30)
            bar_col = CYAN_ACCENT
        elif idx == 1:
            title = f"HELPER: MUTATION SIGMA (std={self.evo_mutation_sigma:.2f})"
            l1 = f"Adds N(0, {self.evo_mutation_sigma:.2f}) Gaussian weight noise."
            l2 = "Controls step size on the fitness surface."
            ratio = min(1.0, self.evo_mutation_sigma / 0.40)
            bar_col = CYAN_ACCENT
        elif idx == 2:
            elites = max(2, int(self.evo_pop_size * self.evo_elite_ratio))
            title = f"HELPER: ELITE RATIO ({elites}/{self.evo_pop_size} Survive)"
            l1 = f"Keeps top {elites} agents unmutated each gen."
            l2 = "Prevents forgetting of winning tactics."
            ratio = min(1.0, self.evo_elite_ratio / 0.50)
            bar_col = GOLD_ACCENT
        elif idx == 3:
            elites = max(2, int(self.evo_pop_size * self.evo_elite_ratio))
            offspring = max(0, self.evo_pop_size - elites)
            title = f"HELPER: POPULATION SIZE ({self.evo_pop_size} Agents)"
            l1 = f"Runs {elites} elites + {offspring} children per gen."
            l2 = "Larger pops explore more; smaller run fast."
            ratio = (self.evo_pop_size - 8.0) / 24.0
            bar_col = CYAN_ACCENT
        elif idx == 4:
            title = f"REWARD: AGGRESSION COEFF ({self.evo_aggression_coeff:.2f}x)"
            l1 = "Scales closing-speed + collision impulse."
            l2 = "Higher values breed relentless strikers."
            ratio = min(1.0, self.evo_aggression_coeff / 2.50)
            bar_col = ORANGE_ACCENT
        elif idx == 5:
            title = f"REWARD: CENTER SAFETY ({self.evo_center_coeff:.2f}x)"
            l1 = "Scales ring-center potential Phi_center."
            l2 = "Higher values breed patient sumo guardians."
            ratio = min(1.0, self.evo_center_coeff / 2.00)
            bar_col = GREEN_ACCENT
        else:
            pts = 15.0 * self.evo_win_coeff
            title = f"REWARD: KO WIN BONUS ({self.evo_win_coeff:.2f}x -> +{pts:.1f})"
            l1 = "Scales terminal ring-out KO bonus reward."
            l2 = "Prioritizes fast decisive knockouts."
            ratio = min(1.0, self.evo_win_coeff / 2.50)
            bar_col = GOLD_ACCENT

        self.screen.blit(self.font_tiny.render(title, True, GOLD_ACCENT), (rect.x + 8, rect.y + 6))
        self.screen.blit(self.font_tiny.render(l1, True, TEXT_WHITE), (rect.x + 8, rect.y + 24))
        self.screen.blit(self.font_tiny.render(l2, True, TEXT_MUTED), (rect.x + 8, rect.y + 42))

        # Mini visual preview bar
        bar_bg = pygame.Rect(rect.x + 8, rect.y + 64, rect.width - 16, 12)
        pygame.draw.rect(self.screen, (30, 36, 52), bar_bg, border_radius=4)
        fill_w = int(bar_bg.width * np.clip(ratio, 0.0, 1.0))
        if fill_w > 0:
            pygame.draw.rect(self.screen, bar_col, pygame.Rect(bar_bg.x, bar_bg.y, fill_w, bar_bg.height), border_radius=4)

    def _draw_evo_genome_heatmap_card(self, rect: pygame.Rect) -> None:
        """Render Champion Genome Sensory Group Importance & 46-Input Synapse Weight Heatmap."""
        pygame.draw.rect(self.screen, (24, 28, 40), rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, 1, border_radius=6)

        best_score_str = f"{self.evo_history_best[-1]:+.2f}" if self.evo_history_best else "N/A"
        self.screen.blit(self.font_med.render("CHAMPION GENOME & SENSORY X-RAY", True, GOLD_ACCENT), (rect.x + 15, rect.y + 10))

        meta_str = f"Gen: {self.evo_generation} | Pop: {self.evo_pop_size} | Best Fit: {best_score_str} | 2,659 Synapses"
        self.screen.blit(self.font_small.render(meta_str, True, TEXT_WHITE), (rect.x + 15, rect.y + 34))

        # 1. Sensory Group Importance Bars
        self.screen.blit(
            self.font_tiny.render("SENSORY GROUP IMPORTANCE (MEAN L1 SYNAPSE MAGNITUDE):", True, CYAN_ACCENT),
            (rect.x + 15, rect.y + 58),
        )
        importance = self.evo_best_agent.get_sensory_importance()
        bar_colors = {
            "Self Kinematics": CYAN_ACCENT,
            "Opponent Rel": ORANGE_ACCENT,
            "Resources": GOLD_ACCENT,
            "16-Ray Lidar": GREEN_ACCENT,
        }
        for i, (group_name, share) in enumerate(importance.items()):
            by = rect.y + 78 + i * 20
            self.screen.blit(self.font_tiny.render(group_name, True, TEXT_WHITE), (rect.x + 15, by))
            track_rect = pygame.Rect(rect.x + 145, by + 2, 205, 10)
            pygame.draw.rect(self.screen, (14, 18, 26), track_rect, border_radius=3)
            fill_w = int(track_rect.width * np.clip(share * 1.8, 0.05, 1.0))
            pygame.draw.rect(
                self.screen,
                bar_colors.get(group_name, CYAN_ACCENT),
                pygame.Rect(track_rect.x, track_rect.y, fill_w, track_rect.height),
                border_radius=3,
            )
            pct_surf = self.font_tiny.render(f"{share*100:.1f}%", True, TEXT_MUTED)
            self.screen.blit(pct_surf, (rect.x + 358, by))

        # 2. 46-Channel Input Synapse Heatmap Strip (W1: 46x32)
        self.screen.blit(
            self.font_tiny.render("46-INPUT SYNAPSE SALIENCE HEATMAP (W1 -> 32 HIDDEN NEURONS):", True, GOLD_ACCENT),
            (rect.x + 15, rect.y + 164),
        )

        salience = self.evo_best_agent.get_input_salience()
        hm_x = rect.x + 15
        hm_y = rect.y + 182
        cell_w = 8
        cell_h = 34
        for ch_idx, sal_val in enumerate(salience):
            cx = hm_x + ch_idx * cell_w
            v = float(np.clip(sal_val, 0.0, 1.0))
            # Color gradient: deep navy -> cyan -> gold/hot orange
            if v < 0.5:
                t = v * 2.0
                c_rgb = (int(18 + 0 * t), int(28 + 180 * t), int(55 + 200 * t))
            else:
                t = (v - 0.5) * 2.0
                c_rgb = (int(0 + 255 * t), int(208 - 20 * t), int(255 - 205 * t))
            pygame.draw.rect(self.screen, c_rgb, pygame.Rect(cx, hm_y, cell_w - 1, cell_h))

        # Domain dividers at channels 7 (Opponent), 12 (Resources), 14 (Lidar)
        for div_idx in (7, 12, 14):
            dx = hm_x + div_idx * cell_w - 1
            pygame.draw.line(self.screen, TEXT_WHITE, (dx, hm_y - 2), (dx, hm_y + cell_h + 2), 1)

        # Domain Labels below heatmap (y = rect.y + 220)
        ly = rect.y + 220
        self.screen.blit(self.font_tiny.render("Self[0:7]", True, CYAN_ACCENT), (hm_x, ly))
        self.screen.blit(self.font_tiny.render("Opp[7:12]", True, ORANGE_ACCENT), (hm_x + 68, ly))
        self.screen.blit(self.font_tiny.render("Res", True, GOLD_ACCENT), (hm_x + 138, ly))
        self.screen.blit(self.font_tiny.render("16-Ray Lidar Channels [14:46]", True, GREEN_ACCENT), (hm_x + 178, ly))

        sparring_name = self.evo_sparring_choices[self.evo_sparring_idx]
        cfg_line = f"Sparring: {sparring_name} | Shaping: Agg {self.evo_aggression_coeff:.1f}x, Ctr {self.evo_center_coeff:.1f}x, KO {self.evo_win_coeff:.1f}x"
        self.screen.blit(self.font_tiny.render(cfg_line, True, TEXT_MUTED), (rect.x + 15, rect.y + 246))
        self.screen.blit(
            self.font_tiny.render("Click 'EXPORT CHAMPION' to deploy bot to Tournament ladder!", True, CYAN_ACCENT),
            (rect.x + 15, rect.y + 268),
        )
        self.screen.blit(
            self.font_tiny.render("Tip: Hover any slider on the left to inspect its live math & effect.", True, GREEN_ACCENT),
            (rect.x + 15, rect.y + 288),
        )

    def _evo_step_generation(self, turbo_count: int = 1) -> None:
        """Run neuroevolution generation(s) headless with active sparring partner and reward shaping."""
        sparring_name = self.evo_sparring_choices[self.evo_sparring_idx]
        sparring_partner = self._instantiate_agent(sparring_name) or HeuristicAgent()
        eval_env = KineticSumoEnv(map_name="Open", max_duration=12.0)

        # Apply pre-training reward shaping coefficients from Evo Lab sliders
        eval_env.reward_calc.center_weight = float(self.evo_center_coeff)
        eval_env.reward_calc.closing_weight = float(0.6 * self.evo_aggression_coeff)
        eval_env.reward_calc.contact_weight = float(self.evo_aggression_coeff)
        eval_env.reward_calc.win_bonus = float(self.evo_win_coeff)

        try:
            for _ in range(turbo_count):
                scores = []
                for ind in self.evo_population:
                    obs, _ = eval_env.reset()
                    sparring_partner.reset()
                    score = 0.0
                    for _ in range(120):
                        a0 = ind.act(obs["agent_0"])
                        a1 = sparring_partner.act(obs["agent_1"])
                        obs, rews, terms, truncs, infos = eval_env.step({"agent_0": a0, "agent_1": a1})
                        score += float(rews["agent_0"])
                        if terms["agent_0"] or truncs["agent_0"]:
                            if infos["agent_0"].get("winner") == "agent_0":
                                score += 15.0 * float(self.evo_win_coeff)
                            elif infos["agent_0"].get("winner") == "agent_1":
                                score -= 10.0 * float(self.evo_win_coeff)
                            break
                    scores.append(score)

                sorted_indices = np.argsort(scores)[::-1]
                best_fit = float(scores[sorted_indices[0]])
                mean_fit = float(np.mean(scores))

                self.evo_history_best.append(best_fit)
                self.evo_history_mean.append(mean_fit)
                self.evo_best_agent = self.evo_population[sorted_indices[0]].clone()

                # Selection and elitism
                num_elites = max(2, int(len(self.evo_population) * self.evo_elite_ratio))
                elites = [self.evo_population[i].clone() for i in sorted_indices[:num_elites]]

                # Mutated offspring
                next_pop = [e.clone() for e in elites]
                while len(next_pop) < self.evo_pop_size:
                    parent = elites[np.random.randint(0, len(elites))]
                    child = parent.mutate(mutation_rate=self.evo_mutation_rate, sigma=self.evo_mutation_sigma)
                    next_pop.append(child)

                self.evo_population = next_pop
                self.evo_generation += 1
        finally:
            eval_env.close()

        best_str = f"{self.evo_history_best[-1]:+.2f}" if self.evo_history_best else "N/A"
        self.evo_status_text = f"Evolved to Gen {self.evo_generation} vs {sparring_name}! Best fitness: {best_str}"

    def _mutate_champion(self) -> None:
        """Mutate current champion directly, evaluate fitness and append consistently to history."""
        mutated = self.evo_best_agent.mutate(mutation_rate=self.evo_mutation_rate, sigma=self.evo_mutation_sigma)
        sparring_name = self.evo_sparring_choices[self.evo_sparring_idx]
        sparring_partner = self._instantiate_agent(sparring_name) or HeuristicAgent()
        eval_env = KineticSumoEnv(map_name="Open", max_duration=12.0)
        eval_env.reward_calc.center_weight = float(self.evo_center_coeff)
        eval_env.reward_calc.closing_weight = float(0.6 * self.evo_aggression_coeff)
        eval_env.reward_calc.contact_weight = float(self.evo_aggression_coeff)
        eval_env.reward_calc.win_bonus = float(self.evo_win_coeff)

        score = 0.0
        try:
            obs, _ = eval_env.reset()
            sparring_partner.reset()
            for _ in range(120):
                a0 = mutated.act(obs["agent_0"])
                a1 = sparring_partner.act(obs["agent_1"])
                obs, rews, terms, truncs, infos = eval_env.step({"agent_0": a0, "agent_1": a1})
                score += float(rews["agent_0"])
                if terms["agent_0"] or truncs["agent_0"]:
                    if infos["agent_0"].get("winner") == "agent_0":
                        score += 15.0 * float(self.evo_win_coeff)
                    elif infos["agent_0"].get("winner") == "agent_1":
                        score -= 10.0 * float(self.evo_win_coeff)
                    break
        finally:
            eval_env.close()

        best_prev = self.evo_history_best[-1] if self.evo_history_best else score
        is_improved = score >= best_prev
        self.evo_generation += 1
        if is_improved:
            self.evo_best_agent = mutated
            self.evo_population[0] = mutated.clone()
            best_to_record = score
            self.evo_status_text = f"Mutation improved Champion (Gen {self.evo_generation})! Fitness: {score:+.2f}"
        else:
            if len(self.evo_population) > 1:
                self.evo_population[1] = mutated.clone()
            best_to_record = best_prev
            self.evo_status_text = f"Mutated Champion (Gen {self.evo_generation}): {score:+.2f} (retained best: {best_prev:+.2f})"

        self.evo_history_best.append(best_to_record)
        self.evo_history_mean.append(score)

    def _watch_evo_duel(self) -> None:
        """Watch champion bot duel the selected Sparring Partner in real time."""
        from src.eval.tournament import play_match
        map_name = self.map_list[self.selected_map_idx]
        sparring_name = self.evo_sparring_choices[self.evo_sparring_idx]
        sparring_agent = self._instantiate_agent(sparring_name) or HeuristicAgent()
        play_match(
            agent_0=self.evo_best_agent,
            agent_1=sparring_agent,
            map_name=map_name,
            rounds_to_win=2,
            render=True,
            agent_names=(f"Evo_Gen{self.evo_generation}", f"{sparring_name}_Spar"),
            match_info=f"EVO LAB LIVE DUEL: Gen {self.evo_generation} Champion vs {sparring_name}",
            fps=50,
        )
        self._restore_hub_display()

    def _export_evo_champion(self) -> None:
        """Export champion bot weights and add to Tournament ladder."""
        save_path = "checkpoints/ga/evo_champion.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.evo_best_agent.save(save_path)

        champ_name = f"Evo_Gen{self.evo_generation}"
        if champ_name not in self.BOT_CHOICES:
            self.BOT_CHOICES.append(champ_name)

        new_elo = float(1200.0 + min(320.0, self.evo_generation * 14.0))
        self.tournament_ratings[champ_name] = new_elo
        self.evo_status_text = f"Exported {champ_name} (Elo {new_elo:.0f}) to Tournament roster!"

    def _draw_bot_locker_tab(self) -> None:
        title = self.font_med.render("BOT LOCKER : ARCHETYPES, PRESETS & LIVE TUNING", True, CYAN_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        left_panel = pygame.Rect(60, 155, 380, 625)
        pygame.draw.rect(self.screen, PANEL_BG, left_panel, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, left_panel, 2, border_radius=8)

        self.screen.blit(self.font_med.render("FIGHTER ARCHETYPE ROSTER (11)", True, GOLD_ACCENT), (78, 165))

        archetypes = [
            ("APEX CHAMPION", "Elite neuroevolution policy; razor edge-recovery & counters.", GOLD_ACCENT),
            ("BULLDOZER", "Relentless forward rammer; dash attack at ring edge.", ORANGE_ACCENT),
            ("SHINOBI", "Evasive flanker; circles outside bumper for 2.5x rear hit.", MAGENTA_ACCENT),
            ("GUARDIAN", "Center-control sumo; anchors to absorb charge then shoves.", GREEN_ACCENT),
            ("MATADOR [NEW]", "Edge-baiter; lures charges to boundary & juke-dashes aside.", (255, 190, 60)),
            ("RICOCHET [NEW]", "16-Ray Lidar hazard weaver; herds foes into 2.0x bumpers.", (0, 240, 210)),
            ("CHAMELEON [NEW]", "Adaptive counter-stance; shifts Guardian/Bulldozer/Shinobi.", PURPLE_ACCENT),
            ("CONTENDER STRIKER", "Intermediate offensive GA striker with momentum charges.", CYAN_ACCENT),
            ("HEURISTIC", "Balanced PD intercept baseline with perimeter retreat.", (160, 180, 255)),
            ("PPO POLICY", "Deep RL CleanRL Actor-Critic continuous control MLP.", (120, 220, 180)),
            ("ROOKIE BOT", "Novice ML policy; high stochastic jitter near boundary.", (255, 140, 140)),
        ]

        for i, (name, desc, color) in enumerate(archetypes):
            ay = 194 + i * 52
            self.screen.blit(self.font_small.render(name, True, color), (78, ay))
            self.screen.blit(self.font_tiny.render(desc, True, TEXT_MUTED), (78, ay + 18))
            if i < len(archetypes) - 1:
                pygame.draw.line(self.screen, (30, 36, 50), (78, ay + 42), (420, ay + 42), 1)

        right_panel = pygame.Rect(460, 155, 380, 625)
        pygame.draw.rect(self.screen, PANEL_BG, right_panel, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, right_panel, 2, border_radius=8)

        t_header = self.font_med.render("TUNABLE BOT & PRESET LAB", True, GOLD_ACCENT)
        self.screen.blit(t_header, (480, 168))

        preset_info = self.font_tiny.render(
            f"Saved Presets in Dropdowns: {len(self.saved_bot_presets)}  |  Active: {self.custom_bot_config.name}",
            True,
            CYAN_ACCENT,
        )
        self.screen.blit(preset_info, (480, 194))

        for s in self.sliders:
            s.draw(self.screen)

        note_surf = self.font_small.render(self.locker_status_text, True, GREEN_ACCENT)
        self.screen.blit(note_surf, (480, 460))

        # 2x2 Action Button Grid (preserving Reset @ 500,525 and Mutate @ 500,580)
        btn_reset = UIButton(pygame.Rect(480, 510, 165, 38), "RESET DEFAULTS", self.font_small, CYAN_ACCENT)
        btn_save = UIButton(pygame.Rect(655, 510, 165, 38), "SAVE AS PRESET", self.font_small, GOLD_ACCENT)
        btn_mutate = UIButton(pygame.Rect(480, 565, 165, 38), "MUTATE JITTER", self.font_small, (255, 120, 200))
        btn_test = UIButton(pygame.Rect(655, 565, 165, 38), "TEST IN ARENA", self.font_small, GREEN_ACCENT)

        btn_reset.draw(self.screen)
        btn_save.draw(self.screen)
        btn_mutate.draw(self.screen)
        btn_test.draw(self.screen)

        # Live Under-the-Hood Coefficient Helper Card (y=618..765)
        self._draw_locker_parameter_helper_card(pygame.Rect(480, 618, 340, 146))

    def _draw_locker_parameter_helper_card(self, rect: pygame.Rect) -> None:
        """Render live physical threshold math & explanation for Bot Locker sliders."""
        pygame.draw.rect(self.screen, (15, 19, 28), rect, border_radius=6)
        pygame.draw.rect(self.screen, CYAN_ACCENT, rect, width=1, border_radius=6)

        agg = self.custom_bot_config.aggression
        dash_f = self.custom_bot_config.dash_frequency
        anc_b = self.custom_bot_config.anchor_bias
        steer_s = self.custom_bot_config.steer_speed

        retreat_r = (0.88 - 0.15 * agg) * 10.0
        base_thr = (0.25 + 0.75 * agg) * 100.0
        dash_dist = 1.2 + 3.5 * dash_f
        anc_spd = 2.5 - 1.5 * anc_b

        idx = self.active_locker_slider_idx
        if idx == 0:
            hdr = f"INSPECTOR: AGGRESSION ({agg:.2f})"
            desc1 = f"Base Forward Thrust: {base_thr:.0f}% max actuator power."
            desc2 = f"Edge Retreat Trigger: r > {retreat_r:.2f}m (on 10m ring)."
            ratio = (agg - 0.1) / 0.9
            col = CYAN_ACCENT
        elif idx == 1:
            hdr = f"INSPECTOR: DASH FREQUENCY ({dash_f:.2f})"
            desc1 = f"Fires +1.0 Dash Impulse when foe is within {dash_dist:.2f}m"
            desc2 = "and heading alignment error is < 0.45 rad (25.8 deg)."
            ratio = dash_f
            col = GOLD_ACCENT
        elif idx == 2:
            hdr = f"INSPECTOR: ANCHOR READINESS ({anc_b:.2f})"
            desc1 = f"Deploys -1.0 Anchor (3.5x mass, 4x traction) when foe"
            desc2 = f"is < 3.0m and closing faster than {anc_spd:.2f} m/s."
            ratio = anc_b
            col = GREEN_ACCENT
        else:
            hdr = f"INSPECTOR: STEERING PD GAIN (Kp = {steer_s:.2f})"
            desc1 = f"Torque cmd = clip({steer_s:.2f} * angle_err - 0.15 * omega)."
            desc2 = "High Kp snaps aim fast; lower Kp prevents oscillation."
            ratio = (steer_s - 1.0) / 3.5
            col = ORANGE_ACCENT

        self.screen.blit(self.font_tiny.render(hdr, True, GOLD_ACCENT), (rect.x + 10, rect.y + 8))
        self.screen.blit(self.font_tiny.render(desc1, True, TEXT_WHITE), (rect.x + 10, rect.y + 28))
        self.screen.blit(self.font_tiny.render(desc2, True, TEXT_MUTED), (rect.x + 10, rect.y + 46))

        # Threshold summary strip
        summary = f"Active: Thr {base_thr:.0f}% | Dash <{dash_dist:.1f}m | Anc >{anc_spd:.1f}m/s | Kp {steer_s:.1f}"
        self.screen.blit(self.font_tiny.render(summary, True, CYAN_ACCENT), (rect.x + 10, rect.y + 72))

        # Visual gauge bar
        bar_bg = pygame.Rect(rect.x + 10, rect.y + 96, rect.width - 20, 12)
        pygame.draw.rect(self.screen, (30, 36, 52), bar_bg, border_radius=4)
        fill_w = int(bar_bg.width * np.clip(ratio, 0.0, 1.0))
        if fill_w > 0:
            pygame.draw.rect(self.screen, col, pygame.Rect(bar_bg.x, bar_bg.y, fill_w, bar_bg.height), border_radius=4)

        tip = self.font_tiny.render("Hover any slider above to inspect its physics math.", True, TEXT_MUTED)
        self.screen.blit(tip, (rect.x + 10, rect.y + 118))

    def _draw_map_showcase_tab(self) -> None:
        title = self.font_med.render("DETERMINISTIC ARENA MAPS & DYNAMIC HAZARDS", True, GOLD_ACCENT)
        self.screen.blit(title, (self.screen_size // 2 - title.get_width() // 2, 125))

        active_map_name = self.map_list[self.selected_map_idx]
        active_map = get_map(active_map_name)

        preview_rect = pygame.Rect(260, 155, 380, 355)
        pygame.draw.rect(self.screen, (22, 26, 38), preview_rect, border_radius=8)
        pygame.draw.rect(self.screen, (45, 52, 72), preview_rect, 2, border_radius=8)

        pv_center = preview_rect.center
        scale = 15.0
        r_px = int(active_map.initial_radius * scale)
        pygame.draw.circle(self.screen, (80, 160, 255), pv_center, r_px, 2)
        r_min_px = int(active_map.min_radius * scale)
        pygame.draw.circle(self.screen, (255, 75, 75), pv_center, r_min_px, 1)

        for obs in active_map.obstacles:
            ox = int(pv_center[0] + obs.x * scale)
            oy = int(pv_center[1] - obs.y * scale)
            orad = max(3, int(obs.radius * scale))
            if obs.restitution >= 1.8:
                pygame.draw.circle(self.screen, (255, 70, 180), (ox, oy), orad)
                pygame.draw.circle(self.screen, (255, 220, 50), (ox, oy), max(2, orad // 2))
            else:
                pygame.draw.circle(self.screen, (100, 140, 180), (ox, oy), orad)
                pygame.draw.circle(self.screen, (200, 230, 255), (ox, oy), max(2, orad // 3))

        info_y = 525
        hazard_label = ""
        if active_map_name == "Bumper_Coliseum":
            hazard_label = " [2.0x PINBALL BUMPERS]"
        elif active_map_name == "Frost_Summit":
            hazard_label = " [LOW-FRICTION ICE DRIFT]"
        elif active_map_name == "Sinking_Pillars":
            hazard_label = " [COLLAPSING OUTER TIER]"

        name_surf = self.font_med.render(f"ARENA: {active_map_name.upper()}{hazard_label}", True, CYAN_ACCENT)
        self.screen.blit(name_surf, (self.screen_size // 2 - name_surf.get_width() // 2, info_y))

        stats_str = f"Obstacles: {len(active_map.obstacles)}  |  Spawn Distance: {active_map.spawn_distance:.1f}m  |  Shrink: {active_map.shrink_start_time:.0f}s to {active_map.round_duration:.0f}s"
        stats_surf = self.font_small.render(stats_str, True, TEXT_WHITE)
        self.screen.blit(stats_surf, (self.screen_size // 2 - stats_surf.get_width() // 2, info_y + 30))

        for i, m_name in enumerate(self.map_list):
            tx = 40 + i * 92
            ty = 610
            t_rect = pygame.Rect(tx, ty, 85, 58)
            is_cur = i == self.selected_map_idx
            bg_col = (40, 48, 68) if is_cur else PANEL_BG
            pygame.draw.rect(self.screen, bg_col, t_rect, border_radius=6)
            pygame.draw.rect(self.screen, (CYAN_ACCENT if is_cur else PANEL_BORDER), t_rect, width=(2 if is_cur else 1), border_radius=6)

            parts = m_name.split("_")
            if len(parts) == 2:
                t1 = self.font_tiny.render(parts[0], True, (TEXT_WHITE if is_cur else TEXT_MUTED))
                t2 = self.font_tiny.render(parts[1], True, (TEXT_WHITE if is_cur else TEXT_MUTED))
                self.screen.blit(t1, (tx + (85 - t1.get_width()) // 2, ty + 12))
                self.screen.blit(t2, (tx + (85 - t2.get_width()) // 2, ty + 28))
            else:
                t1 = self.font_tiny.render(m_name, True, (TEXT_WHITE if is_cur else TEXT_MUTED))
                self.screen.blit(t1, (tx + (85 - t1.get_width()) // 2, ty + 20))

        guide_surf = self.font_tiny.render("Click arena thumbnail to preview obstacles, spawn positions, and ring hazards.", True, TEXT_MUTED)
        self.screen.blit(guide_surf, (self.screen_size // 2 - guide_surf.get_width() // 2, 685))

    def _draw_card(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        val: str,
        color: tuple,
        hint: str,
        is_open: bool = False,
    ) -> None:
        rect = pygame.Rect(x, y, w, h)
        bg = (30, 38, 56) if is_open else PANEL_BG
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        pygame.draw.rect(self.screen, TEXT_WHITE if is_open else color, rect, width=2, border_radius=8)

        t_surf = self.font_small.render(title, True, TEXT_MUTED)
        self.screen.blit(t_surf, (x + (w - t_surf.get_width()) // 2, y + 14))

        # Dynamically scale font size if value text exceeds card width
        v_font = self.font_med
        if v_font.size(val)[0] > w - 16:
            v_font = self.font_small
        if v_font.size(val)[0] > w - 16:
            v_font = self.font_tiny

        display_val = val
        if v_font.size(display_val)[0] > w - 12:
            while len(display_val) > 3 and v_font.size(display_val + "...")[0] > w - 12:
                display_val = display_val[:-1]
            display_val += "..."

        v_surf = v_font.render(display_val, True, color)
        v_y = y + (h - v_surf.get_height()) // 2
        self.screen.blit(v_surf, (x + (w - v_surf.get_width()) // 2, v_y))

        h_surf = self.font_tiny.render(hint, True, (115, 130, 155))
        self.screen.blit(h_surf, (x + (w - h_surf.get_width()) // 2, y + h - 22))

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = text.split(" ")
        lines = []
        current_line = ""
        for w in words:
            test = (current_line + " " + w).strip()
            if font.size(test)[0] <= max_width:
                current_line = test
            else:
                lines.append(current_line)
                current_line = w
        if current_line:
            lines.append(current_line)
        return lines

    def _instantiate_agent(self, agent_name: str) -> BaseAgent | None:
        """Helper to create agent instance based on name or saved preset."""
        if agent_name in self.saved_bot_presets:
            return TunableAgent(self.saved_bot_presets[agent_name])

        name = agent_name.lower()
        if "human" in name:
            return None
        elif "apex" in name:
            return ApexAgent(checkpoint_path="checkpoints/ga/apex_champion.npy")
        elif "contender" in name:
            return ContenderAgent(checkpoint_path="checkpoints/ga/contender_striker.npy")
        elif "rookie" in name:
            return RookieAgent(checkpoint_path="checkpoints/ga/rookie_ga.npy")
        elif "heuristic" in name:
            return HeuristicAgent()
        elif "bulldozer" in name:
            return BulldozerAgent()
        elif "shinobi" in name:
            return ShinobiAgent()
        elif "guardian" in name:
            return GuardianAgent()
        elif "matador" in name:
            return MatadorAgent()
        elif "ricochet" in name:
            return RicochetAgent()
        elif "chameleon" in name:
            return ChameleonAgent()
        elif "tunable" in name or "preset" in name:
            return TunableAgent(self.custom_bot_config)
        elif "ga" in name:
            ga_path = "checkpoints/ga/best_ga_agent.npy"
            return GAMLP.load(ga_path) if os.path.exists(ga_path) else GAMLP()
        elif "ppo" in name:
            ppo_path = "checkpoints/ppo/best_ppo_policy.pt"
            return PPOPolicy.load(ppo_path) if os.path.exists(ppo_path) else PPOPolicy()
        elif "evo" in name:
            return self.evo_best_agent.clone()
        elif "random" in name:
            from src.eval.tournament import RandomAgent
            return RandomAgent()
        return HeuristicAgent()

    def _launch_1v1_match(self) -> None:
        """Launch 1v1 match loop with seamless ESC return to Hub."""
        from src.ui.play import run_interactive_match
        name0 = self.BOT_CHOICES[self.selected_agent0_idx]
        name1 = self.BOT_CHOICES[self.selected_agent1_idx]
        map_name = self.map_list[self.selected_map_idx]

        type0 = "human" if "Human P1" in name0 else ("human_p2" if "Human P2" in name0 else name0.lower())
        type1 = "human_p2" if "Human P2" in name1 else ("human" if "Human P1" in name1 else name1.lower())

        active_cfg = self.saved_bot_presets.get(name0, self.saved_bot_presets.get(name1, self.custom_bot_config))

        run_interactive_match(
            agent0_type=type0,
            agent1_type=type1,
            map_name=map_name,
            rounds_to_win=self.rounds_to_win,
            custom_bot_config=active_cfg,
            return_to_hub=True,
        )
        self._restore_hub_display()

    def _launch_4way_match(self) -> None:
        """Launch 4-Way Battle Royale with seamless ESC return to Hub."""
        from src.ui.play import run_interactive_4way_match
        map_name = self.map_list[self.selected_map_idx]
        f_types = [self.BOT_CHOICES[idx] for idx in self.selected_4way_indices]

        run_interactive_4way_match(
            agent_types=f_types,
            map_name=map_name,
            custom_bot_config=self.custom_bot_config,
            return_to_hub=True,
        )
        self._restore_hub_display()

    def get_tournament_competitors(self) -> list:
        """Build competitor pool for tournament simulation from current roster."""
        from src.eval.tournament import Competitor
        comps = []
        for name, elo in self.tournament_ratings.items():
            agent = self._instantiate_agent(name)
            if agent is not None:
                comps.append(Competitor(name=name, agent=agent, elo=float(elo)))
        return comps

    def _run_fast_tournament_simulation(self) -> None:
        """Run fast headless tournament simulation and update Elo ratings."""
        self.tournament_status_text = "Simulating round-robin matches at >100k steps/sec..."
        self._draw_hub_frame()
        pygame.display.flip()

        from src.eval.tournament import run_tournament
        competitors = self.get_tournament_competitors()
        map_name = self.map_list[self.selected_map_idx]
        ladder = run_tournament(
            competitors=competitors,
            matches_per_pair=2,
            rounds_to_win=2,
            map_name=map_name,
            render=False,
            quiet=True,
        )

        for entry in ladder:
            agent_name = entry["agent"].replace("Agent", "").replace("Policy", "").strip()
            self.tournament_ratings[agent_name] = float(entry["elo"])

        if hasattr(ladder, "stats") and ladder.stats:
            self.tournament_stats = ladder.stats

        self.tournament_status_text = f"Simulation complete on '{map_name}' arena! Elo ladder and analytics updated."

    def _run_live_tournament_spectator(self) -> None:
        """Run visual live tournament spectator match-by-match."""
        from src.eval.tournament import run_tournament
        competitors = self.get_tournament_competitors()
        map_name = self.map_list[self.selected_map_idx]
        ladder = run_tournament(
            competitors=competitors,
            matches_per_pair=2,
            map_name=map_name,
            render=True,
            fps=50,
            quiet=False,
        )
        self._restore_hub_display()
        if hasattr(ladder, "stats") and ladder.stats:
            self.tournament_stats = ladder.stats
        for entry in ladder:
            agent_name = entry["agent"].replace("Agent", "").replace("Policy", "").strip()
            self.tournament_ratings[agent_name] = float(entry["elo"])

    def save_tab_screenshots(self, output_dir: str = "tab_screenshots") -> list[str]:
        """Render and save clean offscreen PNG screenshots of all hub tabs for visual inspection."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        tabs_to_render = [
            (0, None, "tab_0_quick_duel.png"),
            (1, None, "tab_1_4-way_brawl.png"),
            (2, None, "tab_2_tournament.png"),
            (3, 0, "tab_3_analytics_h2h.png"),
            (3, 1, "tab_3_analytics_elo.png"),
            (4, None, "tab_4_evo_lab.png"),
            (5, None, "tab_5_bot_locker.png"),
            (6, None, "tab_6_map_showcase.png"),
        ]
        prev_tab = self.current_tab
        prev_subview = self.analytics_subview
        self._close_all_dropdowns()
        for tab_idx, subview, fname in tabs_to_render:
            self.current_tab = tab_idx
            if subview is not None:
                self.analytics_subview = subview
            self._draw_hub_frame()
            out_path = os.path.join(output_dir, fname)
            pygame.image.save(self.screen, out_path)
            saved_paths.append(out_path)
        self.current_tab = prev_tab
        self.analytics_subview = prev_subview
        return saved_paths


def launch_hub() -> None:
    """Entry point to launch the Arcade Hub."""
    hub = ArcadeHub()
    hub.run()


if __name__ == "__main__":
    launch_hub()
