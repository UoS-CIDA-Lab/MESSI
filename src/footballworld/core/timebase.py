"""Canonical physical and control clock definitions for FootballWorld."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

DEFAULT_MATCH_DURATION_SECONDS = 90.0 * 60.0
"""Default regulation duration in real seconds."""

MAX_PHYSICS_DT_S = 1.0 / 30.0
"""Largest supported outer interval for bounded physics/event integration.

This is not a continuous-collision-detection guarantee. Player locomotion may
use smaller internal collision microsteps when its active speed support
requires them.
"""

_RoundingMode = Literal["nearest", "floor", "ceil"]


def _is_finite_real(value: numbers.Real) -> bool:
    """Return whether a real scalar fits the finite binary64 clock domain."""

    try:
        return bool(math.isfinite(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _exact_real_fraction(value: numbers.Real) -> Fraction:
    """Return the exact value represented by a validated real scalar."""

    if isinstance(value, numbers.Rational):
        return Fraction(value)
    try:
        numerator, denominator = value.as_integer_ratio()
    except (AttributeError, OverflowError, TypeError, ValueError):
        numerator, denominator = float(value).as_integer_ratio()
    return Fraction(int(numerator), int(denominator))


def duration_to_ticks(
    seconds: float,
    tick_seconds: float,
    *,
    rounding: _RoundingMode = "nearest",
    minimum: int = 0,
) -> int:
    """Convert a real-time duration to an integer number of ticks.

    Nearest is appropriate for ordinary durations. Floor and ceil are exact,
    directional contracts used for physical and rule boundaries.
    """

    if not isinstance(seconds, numbers.Real) or isinstance(seconds, bool):
        raise TypeError(f"seconds must be a real non-boolean scalar, got {seconds!r}")
    if not isinstance(tick_seconds, numbers.Real) or isinstance(tick_seconds, bool):
        raise TypeError(
            f"tick_seconds must be a real non-boolean scalar, got {tick_seconds!r}"
        )
    if not _is_finite_real(seconds) or seconds < 0.0:
        raise ValueError(f"seconds must be finite and non-negative, got {seconds!r}")
    if not _is_finite_real(tick_seconds) or tick_seconds <= 0.0:
        raise ValueError(
            f"tick_seconds must be finite and positive, got {tick_seconds!r}"
        )
    if (
        not isinstance(minimum, numbers.Integral)
        or isinstance(minimum, bool)
        or minimum < 0
    ):
        raise ValueError(f"minimum must be a non-negative integer, got {minimum!r}")
    minimum = int(minimum)

    try:
        raw_ticks = seconds / tick_seconds
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "seconds / tick_seconds is outside the finite float timebase range"
        ) from exc
    if not _is_finite_real(raw_ticks):
        raise ValueError(
            "seconds / tick_seconds must remain finite in the float timebase "
            f"range, got {seconds!r} / {tick_seconds!r}"
        )

    if rounding == "nearest":
        ticks = round(raw_ticks)
    elif rounding == "floor":
        exact = _exact_real_fraction(seconds) / _exact_real_fraction(tick_seconds)
        ticks = exact.numerator // exact.denominator
        candidate = ticks + 1
        try:
            on_represented_boundary = candidate * tick_seconds == seconds
        except (OverflowError, TypeError, ValueError):
            on_represented_boundary = False
        if on_represented_boundary:
            ticks = candidate
    elif rounding == "ceil":
        exact = _exact_real_fraction(seconds) / _exact_real_fraction(tick_seconds)
        ticks = -(-exact.numerator // exact.denominator)
        candidate = ticks - 1
        try:
            on_represented_boundary = candidate * tick_seconds == seconds
        except (OverflowError, TypeError, ValueError):
            on_represented_boundary = False
        if on_represented_boundary:
            ticks = candidate
    else:
        raise ValueError(f"unsupported tick rounding mode: {rounding!r}")

    return int(max(minimum, ticks))


@dataclass(frozen=True)
class Timebase:
    """Static relationship between physics ticks and policy control frames.

    The host validates this configuration once before JAX tracing. Rollout
    kernels receive the derived scalar values and fixed integer decimation.
    """

    dt_phys: float
    control_fps: float
    decimation: int = field(init=False)
    control_dt: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dt_phys, numbers.Real) or isinstance(self.dt_phys, bool):
            raise TypeError(
                f"dt_phys must be a real non-boolean scalar, got {self.dt_phys!r}"
            )
        if not isinstance(self.control_fps, numbers.Real) or isinstance(
            self.control_fps, bool
        ):
            raise TypeError(
                "control_fps must be a real non-boolean scalar, "
                f"got {self.control_fps!r}"
            )
        if not _is_finite_real(self.dt_phys) or self.dt_phys <= 0.0:
            raise ValueError(
                f"dt_phys must be finite and positive, got {self.dt_phys!r}"
            )
        if self.dt_phys > MAX_PHYSICS_DT_S:
            raise ValueError(
                "dt_phys must not exceed the supported 1/30 s outer-physics bound, "
                f"got {self.dt_phys!r}"
            )
        if not math.isfinite(1.0 / float(self.dt_phys)):
            raise ValueError(
                "dt_phys is too small to have a finite physics_fps, "
                f"got {self.dt_phys!r}"
            )
        if not _is_finite_real(self.control_fps) or self.control_fps <= 0.0:
            raise ValueError(
                f"control_fps must be finite and positive, got {self.control_fps!r}"
            )

        control_tick_product = self.control_fps * self.dt_phys
        if control_tick_product == 0.0 or not _is_finite_real(control_tick_product):
            raise ValueError(
                "control_fps * dt_phys is outside the finite float timebase range"
            )
        substeps_exact = 1.0 / control_tick_product
        if not _is_finite_real(substeps_exact):
            raise ValueError("physics substeps per control frame must be finite")
        decimation = round(substeps_exact)
        integer_tolerance = max(
            1.0e-9,
            4.0 * math.ulp(substeps_exact),
        )
        if decimation < 1 or abs(substeps_exact - decimation) > integer_tolerance:
            physics_fps = 1.0 / self.dt_phys
            raise ValueError(
                f"control_fps={self.control_fps!r} is not exactly "
                f"representable with dt_phys={self.dt_phys}; supported rates "
                f"are {physics_fps:g} / positive_integer Hz"
            )
        if decimation > 2**31 - 1:
            raise ValueError(
                "physics substeps per control frame exceed the int32 counter "
                f"contract: {decimation}"
            )

        dt_phys = float(self.dt_phys)
        control_fps = float(self.control_fps)
        control_dt = decimation * dt_phys
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError(
                "derived control_dt is outside the finite float timebase range"
            )

        object.__setattr__(self, "dt_phys", dt_phys)
        object.__setattr__(self, "control_fps", control_fps)
        object.__setattr__(self, "decimation", decimation)
        object.__setattr__(self, "control_dt", control_dt)

    @property
    def physics_fps(self) -> float:
        """Number of physics ticks per real second."""

        return 1.0 / self.dt_phys

    def control_steps_for(self, seconds: float, *, minimum: int = 0) -> int:
        """Convert seconds to control frames."""

        return duration_to_ticks(
            seconds,
            self.control_dt,
            rounding="nearest",
            minimum=minimum,
        )

    def seconds_for_control_steps(self, steps: int) -> float:
        """Convert a non-negative control-frame count to real seconds."""

        if (
            not isinstance(steps, numbers.Integral)
            or isinstance(steps, bool)
            or steps < 0
        ):
            raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
        try:
            seconds = int(steps) * self.control_dt
        except OverflowError as exc:
            raise ValueError(
                "steps * control_dt is outside the finite float timebase range"
            ) from exc
        if not math.isfinite(seconds):
            raise ValueError(
                "steps * control_dt is outside the finite float timebase range"
            )
        return seconds

    def physics_steps_for(
        self,
        seconds: float,
        *,
        rounding: _RoundingMode = "nearest",
        minimum: int = 0,
    ) -> int:
        """Convert seconds to physics ticks."""

        return duration_to_ticks(
            seconds,
            self.dt_phys,
            rounding=rounding,
            minimum=minimum,
        )


def _exact_render_grid(
    timebase: Timebase,
    fps: numbers.Real,
    *,
    context: str,
) -> tuple[float, int, tuple[int, ...]]:
    """Return a uniform end-of-substep grid for exact-event rendering."""

    if (
        not isinstance(fps, numbers.Real)
        or isinstance(fps, bool)
        or not _is_finite_real(fps)
        or fps <= 0.0
    ):
        raise ValueError(f"{context} fps must be a finite positive real scalar")
    render_fps = float(fps)
    ratio = render_fps / timebase.control_fps
    samples_per_control = round(ratio)
    if (
        samples_per_control < 1
        or samples_per_control > timebase.decimation
        or abs(ratio - samples_per_control) > 1.0e-9
        or timebase.decimation % samples_per_control != 0
    ):
        raise ValueError(
            f"{context} fps must be an integer multiple of control_fps "
            "whose samples divide the physics decimation exactly"
        )
    stride = timebase.decimation // samples_per_control
    indices = tuple(index * stride - 1 for index in range(1, samples_per_control + 1))
    return render_fps, samples_per_control, indices


DEFAULT_TIMEBASE = Timebase(dt_phys=1.0 / 80.0, control_fps=10.0)
"""Default 80 Hz physics and 10 Hz policy-control clock."""


__all__ = [
    "DEFAULT_MATCH_DURATION_SECONDS",
    "DEFAULT_TIMEBASE",
    "MAX_PHYSICS_DT_S",
    "Timebase",
    "duration_to_ticks",
]
