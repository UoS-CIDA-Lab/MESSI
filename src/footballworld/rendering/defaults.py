"""Numerical-stack-free defaults for the FootballWorld replay renderer."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

DEFAULT_RENDER_FPS = 20.0
_ENCODER_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a non-boolean real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RenderStyle:
    """Static presentation and encoder settings.

    The renderer intentionally has one presentation path. These values tune
    that path; they do not select between rendering modes.
    """

    width_px: int = 1920
    height_px: int = 1080
    dpi: int = 150
    # One spherical team-coloured marker is deliberately presentation-only.
    # This is the largest marker that stays legible without overrunning the
    # nearby intent/FOV cues.
    player_size: float = 185.0
    crf: int = 20
    encoder_preset: str = "veryfast"
    encoder_threads: int = 4
    # Optional renderer controls follow the required positional fields.
    camera_azimuth_degrees: float = -90.0
    camera_elevation_degrees: float = 33.0
    camera_distance_m: float = 150.0
    camera_focal_length: float = 820.0
    adjudication_seconds: float = 2.5
    goal_hold_seconds: float = 2.0
    # Full-view observations have no physical visibility aperture. This
    # host-only width changes only the visual gaze cue; partial observations
    # always render the environment's authoritative horizontal FOV.
    gaze_cue_degrees: float = 160.0

    def __post_init__(self) -> None:
        width_px = _integer("width_px", self.width_px, minimum=320)
        height_px = _integer("height_px", self.height_px, minimum=240)
        _integer("dpi", self.dpi, minimum=50)
        crf = _integer("crf", self.crf, minimum=0)
        _integer("encoder_threads", self.encoder_threads, minimum=1)
        if width_px < 320 or height_px < 240:
            raise ValueError("render dimensions must be at least 320 x 240")
        if width_px % 2 or height_px % 2:
            raise ValueError("render dimensions must be even for yuv420p")
        player_size = _finite_real("player_size", self.player_size)
        if player_size <= 0.0:
            raise ValueError("player_size must be positive")
        camera_azimuth = _finite_real(
            "camera_azimuth_degrees", self.camera_azimuth_degrees
        )
        camera_elevation = _finite_real(
            "camera_elevation_degrees", self.camera_elevation_degrees
        )
        camera_distance = _finite_real("camera_distance_m", self.camera_distance_m)
        camera_focal = _finite_real("camera_focal_length", self.camera_focal_length)
        adjudication_seconds = _finite_real(
            "adjudication_seconds", self.adjudication_seconds
        )
        goal_hold_seconds = _finite_real("goal_hold_seconds", self.goal_hold_seconds)
        gaze_cue_degrees = _finite_real("gaze_cue_degrees", self.gaze_cue_degrees)
        if not -90.0 <= camera_azimuth <= 90.0:
            raise ValueError("camera_azimuth_degrees must be in [-90, 90]")
        if not 0.0 < camera_elevation < 90.0:
            raise ValueError("camera_elevation_degrees must be in (0, 90)")
        if camera_distance <= 0.0:
            raise ValueError("camera_distance_m must be positive")
        if camera_focal <= 0.0:
            raise ValueError("camera_focal_length must be positive")
        if adjudication_seconds <= 0.0:
            raise ValueError("adjudication_seconds must be positive")
        if goal_hold_seconds <= 0.0:
            raise ValueError("goal_hold_seconds must be positive")
        if not 0.0 < gaze_cue_degrees <= 360.0:
            raise ValueError("gaze_cue_degrees must be in (0, 360]")
        if crf > 51:
            raise ValueError("crf must be in [0, 51]")
        if type(self.encoder_preset) is not str:
            raise TypeError("encoder_preset must be a string")
        if self.encoder_preset not in _ENCODER_PRESETS:
            raise ValueError(
                "encoder_preset must be one of " + ", ".join(sorted(_ENCODER_PRESETS))
            )

    @property
    def figsize(self) -> tuple[float, float]:
        return self.width_px / self.dpi, self.height_px / self.dpi


__all__ = ["DEFAULT_RENDER_FPS", "RenderStyle"]
