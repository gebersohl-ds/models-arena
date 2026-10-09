"""Procedural waveform audio synthesis for Kinetic Sumo using pure NumPy and Pygame mixer.

Synthesizes responsive arcade sound effects entirely in memory without requiring
any external audio asset files. Supports instant muting and headless fallback.
"""

import numpy as np
import pygame


class SoundEngine:
    """Procedural audio engine with synthesized sound effects."""

    def __init__(self, sample_rate: int = 44100, enabled: bool = True):
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.is_muted = not enabled
        self._initialized = False

        self.snd_clang: pygame.mixer.Sound | None = None
        self.snd_flank_clang: pygame.mixer.Sound | None = None
        self.snd_dash: pygame.mixer.Sound | None = None
        self.snd_anchor: pygame.mixer.Sound | None = None
        self.snd_ringout: pygame.mixer.Sound | None = None
        self.snd_countdown: pygame.mixer.Sound | None = None

        if self.enabled:
            self._init_mixer()

    def _init_mixer(self) -> None:
        """Initialize pygame mixer and synthesize audio clips."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
            self._initialized = True
            self._synthesize_all()
        except Exception:
            # Fallback for headless environments or systems without audio devices
            self._initialized = False
            self.enabled = False

    def _synth_to_sound(self, waveform: np.ndarray) -> pygame.mixer.Sound | None:
        """Convert float [-1, 1] array to a 16-bit pygame Sound."""
        if not self._initialized:
            return None
        waveform = np.clip(waveform, -1.0, 1.0)
        int16_wave = (waveform * 32767.0).astype(np.int16)
        return pygame.sndarray.make_sound(int16_wave)

    def _synthesize_all(self) -> None:
        """Generate all procedural sound waveforms."""
        # 1. Standard metallic bumper clang (0.25s)
        t_clang = np.linspace(0, 0.25, int(self.sample_rate * 0.25), endpoint=False)
        f_clang = 750.0 - 450.0 * (t_clang / 0.25)
        wave_clang = np.exp(-14.0 * t_clang) * (
            0.6 * np.sin(2 * np.pi * f_clang * t_clang)
            + 0.3 * np.sin(2 * np.pi * (f_clang * 2.2) * t_clang)
            + 0.1 * np.random.uniform(-1.0, 1.0, len(t_clang))
        )
        self.snd_clang = self._synth_to_sound(wave_clang)

        # 2. Heavy flank/rear impact clang (0.35s) - deeper, more punch
        t_flank = np.linspace(0, 0.35, int(self.sample_rate * 0.35), endpoint=False)
        wave_flank = np.exp(-10.0 * t_flank) * (
            0.7 * np.sin(2 * np.pi * (320.0 - 180.0 * t_flank / 0.35) * t_flank)
            + 0.3 * np.sin(2 * np.pi * 640.0 * t_flank)
            + 0.2 * np.random.uniform(-1.0, 1.0, len(t_flank))
        )
        self.snd_flank_clang = self._synth_to_sound(wave_flank)

        # 3. Dash sonic burst (0.22s)
        t_dash = np.linspace(0, 0.22, int(self.sample_rate * 0.22), endpoint=False)
        f_dash = 380.0 - 240.0 * (t_dash / 0.22)
        wave_dash = np.exp(-7.0 * t_dash) * (
            0.6 * np.sin(2 * np.pi * f_dash * t_dash)
            + 0.4 * np.random.uniform(-1.0, 1.0, len(t_dash))
        )
        self.snd_dash = self._synth_to_sound(wave_dash)

        # 4. Anchor magnetic lock (0.18s)
        t_anchor = np.linspace(0, 0.18, int(self.sample_rate * 0.18), endpoint=False)
        wave_anchor = np.exp(-15.0 * t_anchor) * (
            0.8 * np.sin(2 * np.pi * 210.0 * t_anchor)
            + 0.2 * np.sin(2 * np.pi * 420.0 * t_anchor)
        )
        self.snd_anchor = self._synth_to_sound(wave_anchor)

        # 5. Ring-out elimination alarm / buzzer (0.5s)
        t_ro = np.linspace(0, 0.5, int(self.sample_rate * 0.5), endpoint=False)
        wave_ro = np.exp(-4.0 * t_ro) * (
            0.5 * np.sign(np.sin(2 * np.pi * 220.0 * t_ro))
            + 0.5 * np.sign(np.sin(2 * np.pi * 233.0 * t_ro))
        )
        self.snd_ringout = self._synth_to_sound(wave_ro)

        # 6. Countdown beep (0.12s)
        t_cd = np.linspace(0, 0.12, int(self.sample_rate * 0.12), endpoint=False)
        wave_cd = np.exp(-12.0 * t_cd) * np.sin(2 * np.pi * 880.0 * t_cd)
        self.snd_countdown = self._synth_to_sound(wave_cd)

    def toggle_mute(self) -> bool:
        """Toggle audio mute state. Returns current muted state."""
        self.is_muted = not self.is_muted
        return self.is_muted

    def play_clang(self, is_flank: bool = False, intensity: float = 1.0) -> None:
        """Play bumper impact clang."""
        if not self._initialized or self.is_muted:
            return
        sound = self.snd_flank_clang if is_flank else self.snd_clang
        if sound:
            volume = float(np.clip(intensity, 0.2, 1.0))
            sound.set_volume(volume)
            sound.play()

    def play_dash(self) -> None:
        """Play dash burst audio."""
        if not self._initialized or self.is_muted or not self.snd_dash:
            return
        self.snd_dash.set_volume(0.8)
        self.snd_dash.play()

    def play_anchor(self) -> None:
        """Play anchor engagement audio."""
        if not self._initialized or self.is_muted or not self.snd_anchor:
            return
        self.snd_anchor.set_volume(0.7)
        self.snd_anchor.play()

    def play_ringout(self) -> None:
        """Play ring-out elimination buzzer."""
        if not self._initialized or self.is_muted or not self.snd_ringout:
            return
        self.snd_ringout.set_volume(0.9)
        self.snd_ringout.play()

    def play_countdown(self, high_pitch: bool = False) -> None:
        """Play match countdown tick."""
        if not self._initialized or self.is_muted or not self.snd_countdown:
            return
        self.snd_countdown.set_volume(0.6)
        self.snd_countdown.play()
