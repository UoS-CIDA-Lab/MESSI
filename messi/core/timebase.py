"""물리 tick과 policy control frame의 표준 시간 contract."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np

from messi.core.numeric import require_float32_representable

DEFAULT_MATCH_DURATION_SECONDS = 90.0 * 60.0
MAX_PHYSICS_DT_S = 1.0 / 30.0
MAX_INT32_TICKS = (1 << 31) - 1

_RoundingMode = Literal["nearest", "floor", "ceil"]


def _is_finite_real(value: numbers.Real) -> bool:
    try:
        return bool(math.isfinite(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _exact_real_fraction(value: numbers.Real) -> Fraction:
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
    maximum: int | None = None,
) -> int:
    """실제 시간을 지정한 방향의 정수 tick 수로 변환한다."""

    if not isinstance(seconds, numbers.Real) or isinstance(seconds, bool):
        raise TypeError(f"seconds는 bool이 아닌 실수여야 한다: {seconds!r}")
    if not isinstance(tick_seconds, numbers.Real) or isinstance(tick_seconds, bool):
        raise TypeError(f"tick_seconds는 bool이 아닌 실수여야 한다: {tick_seconds!r}")
    if not _is_finite_real(seconds) or seconds < 0.0:
        raise ValueError(f"seconds는 유한한 음이 아닌 값이어야 한다: {seconds!r}")
    if not _is_finite_real(tick_seconds) or tick_seconds <= 0.0:
        raise ValueError(f"tick_seconds는 유한한 양수여야 한다: {tick_seconds!r}")
    if (
        not isinstance(minimum, numbers.Integral)
        or isinstance(minimum, bool)
        or minimum < 0
    ):
        raise ValueError(f"minimum은 음이 아닌 정수여야 한다: {minimum!r}")
    minimum = int(minimum)
    if maximum is not None:
        if (
            not isinstance(maximum, numbers.Integral)
            or isinstance(maximum, bool)
            or maximum < minimum
        ):
            raise ValueError(f"maximum은 minimum 이상의 정수여야 한다: {maximum!r}")
        maximum = int(maximum)

    try:
        raw_ticks = seconds / tick_seconds
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "seconds / tick_seconds가 유한한 float 시간 범위를 벗어났다"
        ) from exc
    if not _is_finite_real(raw_ticks):
        raise ValueError(
            "seconds / tick_seconds는 유한한 float 시간 범위에 있어야 한다: "
            f"{seconds!r} / {tick_seconds!r}"
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
        raise ValueError(f"지원하지 않는 tick rounding mode: {rounding!r}")

    result = int(max(minimum, ticks))
    if maximum is not None and result > maximum:
        raise ValueError(f"tick 수 {result}가 지원 상한 {maximum}을 넘는다")
    return result


@dataclass(frozen=True)
class Timebase:
    """physics tick과 policy control frame 사이의 static 관계."""

    dt_phys: float
    control_fps: float
    decimation: int = field(init=False)
    control_dt: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dt_phys, numbers.Real) or isinstance(self.dt_phys, bool):
            raise TypeError(f"dt_phys는 bool이 아닌 실수여야 한다: {self.dt_phys!r}")
        if not isinstance(self.control_fps, numbers.Real) or isinstance(
            self.control_fps, bool
        ):
            raise TypeError(
                f"control_fps는 bool이 아닌 실수여야 한다: {self.control_fps!r}"
            )
        if not _is_finite_real(self.dt_phys) or self.dt_phys <= 0.0:
            raise ValueError(f"dt_phys는 유한한 양수여야 한다: {self.dt_phys!r}")
        represented_dt_phys = np.float32(
            require_float32_representable("dt_phys", self.dt_phys)
        )
        if represented_dt_phys < np.finfo(np.float32).tiny:
            raise ValueError("dt_phys는 float32 최소 positive normal 이상이어야 한다")
        if self.dt_phys > MAX_PHYSICS_DT_S:
            raise ValueError(
                "dt_phys는 지원되는 1/30초 outer physics 한도를 넘으면 안 된다: "
                f"{self.dt_phys!r}"
            )
        if not math.isfinite(1.0 / float(self.dt_phys)):
            raise ValueError(
                f"dt_phys가 너무 작아 유한한 physics_fps를 만들 수 없다: {self.dt_phys!r}"
            )
        if not _is_finite_real(self.control_fps) or self.control_fps <= 0.0:
            raise ValueError(
                f"control_fps는 유한한 양수여야 한다: {self.control_fps!r}"
            )
        represented_control_fps = np.float32(
            require_float32_representable("control_fps", self.control_fps)
        )

        control_tick_product = self.control_fps * self.dt_phys
        if control_tick_product == 0.0 or not _is_finite_real(control_tick_product):
            raise ValueError(
                "control_fps * dt_phys가 유한한 float 시간 범위를 벗어났다"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            represented_control_product = represented_control_fps * represented_dt_phys
        if (
            not np.isfinite(represented_control_product)
            or represented_control_product <= 0.0
        ):
            raise ValueError("control_fps * dt_phys는 float32에서 유한한 양수여야 한다")
        substeps_exact = 1.0 / control_tick_product
        if not _is_finite_real(substeps_exact):
            raise ValueError("control frame당 physics substep 수는 유한해야 한다")
        decimation = round(substeps_exact)
        integer_tolerance = max(1.0e-9, 4.0 * math.ulp(substeps_exact))
        if decimation < 1 or abs(substeps_exact - decimation) > integer_tolerance:
            physics_fps = 1.0 / self.dt_phys
            raise ValueError(
                f"control_fps={self.control_fps!r}는 dt_phys={self.dt_phys}로 "
                f"정확히 표현할 수 없다. 지원 rate는 {physics_fps:g} / 양의 정수 Hz다"
            )
        if decimation > MAX_INT32_TICKS:
            raise ValueError(
                f"control frame당 physics substep 수가 int32 contract를 넘는다: {decimation}"
            )

        dt_phys = float(self.dt_phys)
        control_fps = float(self.control_fps)
        control_dt = decimation * dt_phys
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("계산된 control_dt가 유한한 float 시간 범위를 벗어났다")
        represented_control_dt = np.float32(
            require_float32_representable("control_dt", control_dt)
        )
        if represented_control_dt < np.finfo(np.float32).tiny:
            raise ValueError(
                "control_dt는 float32 최소 positive normal 이상이어야 한다"
            )

        object.__setattr__(self, "dt_phys", dt_phys)
        object.__setattr__(self, "control_fps", control_fps)
        object.__setattr__(self, "decimation", decimation)
        object.__setattr__(self, "control_dt", control_dt)

    @property
    def physics_fps(self) -> float:
        return 1.0 / self.dt_phys

    def control_steps_for(
        self,
        seconds: float,
        *,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int:
        return duration_to_ticks(
            seconds,
            self.control_dt,
            rounding="nearest",
            minimum=minimum,
            maximum=maximum,
        )

    def seconds_for_control_steps(self, steps: int) -> float:
        if (
            not isinstance(steps, numbers.Integral)
            or isinstance(steps, bool)
            or steps < 0
        ):
            raise ValueError(f"steps는 음이 아닌 정수여야 한다: {steps!r}")
        try:
            seconds = int(steps) * self.control_dt
        except OverflowError as exc:
            raise ValueError(
                "steps * control_dt가 유한한 float 범위를 벗어났다"
            ) from exc
        if not math.isfinite(seconds):
            raise ValueError("steps * control_dt가 유한한 float 범위를 벗어났다")
        return seconds

    def physics_steps_for(
        self,
        seconds: float,
        *,
        rounding: _RoundingMode = "nearest",
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int:
        return duration_to_ticks(
            seconds,
            self.dt_phys,
            rounding=rounding,
            minimum=minimum,
            maximum=maximum,
        )


def _exact_render_grid(
    timebase: Timebase,
    fps: numbers.Real,
    *,
    context: str,
) -> tuple[float, int, tuple[int, ...]]:
    """physics substep 끝점에 맞춘 균일 render grid를 반환한다."""

    if (
        not isinstance(fps, numbers.Real)
        or isinstance(fps, bool)
        or not _is_finite_real(fps)
        or fps <= 0.0
    ):
        raise ValueError(f"{context} fps는 유한한 양의 실수여야 한다")
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
            f"{context} fps는 control_fps의 정수배이고 sample 수가 physics "
            "decimation을 정확히 나눠야 한다"
        )
    stride = timebase.decimation // samples_per_control
    indices = tuple(index * stride - 1 for index in range(1, samples_per_control + 1))
    return render_fps, samples_per_control, indices


DEFAULT_TIMEBASE = Timebase(dt_phys=1.0 / 80.0, control_fps=10.0)

__all__ = [
    "DEFAULT_MATCH_DURATION_SECONDS",
    "DEFAULT_TIMEBASE",
    "MAX_INT32_TICKS",
    "MAX_PHYSICS_DT_S",
    "Timebase",
    "duration_to_ticks",
]
