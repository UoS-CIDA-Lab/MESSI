"""Episode-level player-profile sampling configuration."""

import math
from dataclasses import dataclass

from footballworld.core.numeric import require_float32_representable


@dataclass(frozen=True, slots=True)
class RosterSampling:
    """Clipped-Gaussian design priors around each supplied player profile.

    These defaults are modelling choices, not fitted population statistics.
    A reset key activates sampling; omitting the key preserves the supplied
    profiles exactly.  ``height`` and the non-negative reach margin
    ``reach_height - height`` are sampled instead of two independent heights.
    """

    enabled: bool = True
    clip_standard_deviations: float = 2.5

    max_speed_std_mps: float = 0.35
    min_max_speed_mps: float = 3.0
    max_max_speed_mps: float = 11.0

    height_std_m: float = 0.05
    min_height_m: float = 1.45
    max_height_m: float = 2.20

    reach_margin_std_m: float = 0.08
    min_reach_margin_m: float = 0.0
    max_reach_margin_m: float = 1.40

    ball_control_std: float = 0.08
    min_ball_control: float = 0.0
    max_ball_control: float = 1.0

    endurance_factor_std: float = 0.08
    min_endurance_factor: float = 0.50
    max_endurance_factor: float = 1.50

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        names = (
            "clip_standard_deviations",
            "max_speed_std_mps",
            "min_max_speed_mps",
            "max_max_speed_mps",
            "height_std_m",
            "min_height_m",
            "max_height_m",
            "reach_margin_std_m",
            "min_reach_margin_m",
            "max_reach_margin_m",
            "ball_control_std",
            "min_ball_control",
            "max_ball_control",
            "endurance_factor_std",
            "min_endurance_factor",
            "max_endurance_factor",
        )
        values = {name: getattr(self, name) for name in names}
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values.values()
        ):
            raise ValueError("roster sampling values must be finite real numbers")
        for name, value in values.items():
            require_float32_representable(name, value)
        if self.clip_standard_deviations <= 0.0:
            raise ValueError("clip_standard_deviations must be positive")
        for name in (
            "max_speed_std_mps",
            "height_std_m",
            "reach_margin_std_m",
            "ball_control_std",
            "endurance_factor_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        bounds = (
            ("max_speed", self.min_max_speed_mps, self.max_max_speed_mps),
            ("height", self.min_height_m, self.max_height_m),
            (
                "reach_margin",
                self.min_reach_margin_m,
                self.max_reach_margin_m,
            ),
            ("ball_control", self.min_ball_control, self.max_ball_control),
            (
                "endurance_factor",
                self.min_endurance_factor,
                self.max_endurance_factor,
            ),
        )
        for name, lower, upper in bounds:
            if lower >= upper:
                raise ValueError(f"{name} sampling bounds must increase")
        if self.min_max_speed_mps <= 0.0:
            raise ValueError("min_max_speed_mps must be positive")
        if self.min_height_m <= 0.0:
            raise ValueError("min_height_m must be positive")
        if self.min_reach_margin_m < 0.0:
            raise ValueError("min_reach_margin_m must be non-negative")
        if not 0.0 <= self.min_ball_control < self.max_ball_control <= 1.0:
            raise ValueError("ball_control sampling bounds must lie in [0, 1]")
        if self.min_endurance_factor <= 0.0:
            raise ValueError("min_endurance_factor must be positive")


__all__ = ["RosterSampling"]
