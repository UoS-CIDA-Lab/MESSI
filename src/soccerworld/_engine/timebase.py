"""환경의 물리·제어 시간축과 초↔tick 변환의 단일 진실원천."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
import numbers
from typing import Literal


DEFAULT_MATCH_DURATION_SECONDS = 90.0 * 60.0
"""정규 경기의 기본 실시간 길이(s). 프레임 수는 :class:`Timebase`가 파생한다."""


def _is_finite_real(value: numbers.Real) -> bool:
    """``math.isfinite`` without leaking overflow from unbounded Real kinds."""

    # Fraction and third-party Real implementations can be mathematically
    # finite yet too large to convert to the binary64 clock used here.
    try:
        return bool(math.isfinite(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _exact_real_fraction(value: numbers.Real) -> Fraction:
    """Return the exact value represented by a validated real scalar.

    ``floor``/``ceil`` are directional contracts, so computing them from an
    already-rounded floating quotient is not sufficient near an integer.  In
    particular, moving that quotient by one ULP can cross a *real* boundary.
    Python and NumPy binary scalars expose ``as_integer_ratio``; rational kinds
    are already exact.  The float fallback covers uncommon ``Real`` wrappers
    after the caller has established that conversion is finite.
    """

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
    rounding: Literal["nearest", "floor", "ceil"] = "nearest",
    minimum: int = 0,
) -> int:
    """실시간 길이를 tick 수로 바꾼다.

    지속시간 창은 ``nearest``, 실측 하한을 넘으면 안 되는 쿨다운은 ``floor``를 쓴다.
    모든 물리 카운터가 이 함수를 공유하므로 물리 tick 크기를 바꾸면 함께 재파생된다.
    """
    # ``bool`` is an ``int`` subclass.  Without an explicit scalar-kind
    # contract, ``seconds=True`` silently means one second and
    # ``tick_seconds=True`` means a one-second clock.  Time is a public
    # physics input, so reject coercible non-numeric values before arithmetic.
    if (
        not isinstance(seconds, numbers.Real)
        or isinstance(seconds, bool)
    ):
        raise TypeError(
            f"seconds must be a real non-boolean scalar, got {seconds!r}"
        )
    if (
        not isinstance(tick_seconds, numbers.Real)
        or isinstance(tick_seconds, bool)
    ):
        raise TypeError(
            "tick_seconds must be a real non-boolean scalar, "
            f"got {tick_seconds!r}"
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
    # Integral includes NumPy scalar integers.  Canonicalise at this public
    # boundary so a minimum-dominated result still honours the annotated
    # Python ``int`` return contract and remains directly JSON serialisable.
    minimum = int(minimum)

    try:
        raw_ticks = seconds / tick_seconds
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "seconds / tick_seconds is outside the finite float timebase range"
        ) from exc
    # Both public operands can be finite while their ratio overflows (for
    # example ``1e308 / 1e-308``).  Letting ``round(inf)``/``floor(inf)`` run
    # leaks an implementation-specific ``OverflowError`` instead of rejecting
    # a time interval that this clock cannot represent.
    if not _is_finite_real(raw_ticks):
        raise ValueError(
            "seconds / tick_seconds must remain finite in the float timebase "
            f"range, got {seconds!r} / {tick_seconds!r}"
        )
    if rounding == "nearest":
        ticks = int(round(raw_ticks))
    elif rounding == "floor":
        exact = _exact_real_fraction(seconds) / _exact_real_fraction(tick_seconds)
        ticks = exact.numerator // exact.denominator
        # Engineering clocks commonly spell one intended boundary with two
        # independently rounded floats (1/15 s and 1/90 s, or 8e9 * 1e-9).
        # Recover that boundary only when multiplying the candidate tick back
        # in the caller's scalar arithmetic reproduces ``seconds`` exactly.
        # Unlike an unconditional nextafter(), this cannot turn a genuine
        # nextafter(1, 0) duration into a full one-second tick.
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
    """물리 tick과 정책 제어 frame의 결합 계약.

    ``dt_phys``와 ``control_fps``만 입력값이다. 정수 물리 서브스텝 수, 제어 간격과
    초↔frame 변환은 모두 여기서 파생하며 표현 불가능한 조합은 즉시 거부한다.
    """

    dt_phys: float
    control_fps: float
    decimation: int = field(init=False)
    control_dt: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dt_phys, numbers.Real)
            or isinstance(self.dt_phys, bool)
        ):
            raise TypeError(
                "dt_phys must be a real non-boolean scalar, "
                f"got {self.dt_phys!r}"
            )
        if (
            not isinstance(self.control_fps, numbers.Real)
            or isinstance(self.control_fps, bool)
        ):
            raise TypeError(
                "control_fps must be a real non-boolean scalar, "
                f"got {self.control_fps!r}"
            )
        if not _is_finite_real(self.dt_phys) or self.dt_phys <= 0.0:
            raise ValueError(
                f"dt_phys must be finite and positive, got {self.dt_phys!r}"
            )
        # 유한하고 양수라도 **역수가 무한**일 수 있다(subnormal). 그러면
        # ``physics_fps``가 inf가 되어, 그 값으로 나누는 모든 파생 시간이 0이 된다.
        # 시간축의 두 방향이 모두 유한해야 시간축이다.
        if not math.isfinite(1.0 / float(self.dt_phys)):
            raise ValueError(
                "dt_phys is too small to have a finite physics_fps, "
                f"got {self.dt_phys!r}"
            )
        if not _is_finite_real(self.control_fps) or self.control_fps <= 0.0:
            raise ValueError(
                "control_fps must be finite and positive, "
                f"got {self.control_fps!r}"
            )

        control_tick_product = self.control_fps * self.dt_phys
        if control_tick_product == 0.0 or not _is_finite_real(control_tick_product):
            raise ValueError(
                "control_fps * dt_phys is outside the finite float timebase range"
            )
        substeps_exact = 1.0 / control_tick_product
        if not _is_finite_real(substeps_exact):
            raise ValueError("physics substeps per control frame must be finite")
        decimation = int(round(substeps_exact))
        integer_tolerance = max(1.0e-9, 4.0 * math.ulp(substeps_exact))
        if decimation < 1 or abs(substeps_exact - decimation) > integer_tolerance:
            physics_fps = 1.0 / self.dt_phys
            raise ValueError(
                f"control_fps={self.control_fps!r} is not exactly representable "
                f"with dt_phys={self.dt_phys}; supported rates are "
                f"{physics_fps:g} / positive_integer Hz"
            )

        if decimation > 2**31 - 1:
            raise ValueError(
                "physics substeps per control frame exceed the int32 counter "
                f"contract: {decimation}"
            )

        # Public annotations say float.  Canonicalise integral/Fraction/NumPy
        # Real inputs after validation so numerically identical clocks have the
        # same metadata and weak scalar kinds cannot alter JAX promotion.
        dt_phys = float(self.dt_phys)
        control_fps = float(self.control_fps)
        control_dt = decimation * dt_phys
        # A finite input pair and a finite integer decimation do not imply a
        # finite derived control period: ``decimation * dt_phys`` can overflow.
        # Such a Timebase cannot round-trip seconds or produce finite metadata.
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
        return 1.0 / self.dt_phys

    def control_steps_for(self, seconds: float, *, minimum: int = 0) -> int:
        return duration_to_ticks(
            seconds,
            self.control_dt,
            rounding="nearest",
            minimum=minimum,
        )

    def seconds_for_control_steps(self, steps: int) -> float:
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
        rounding: Literal["nearest", "floor", "ceil"] = "nearest",
        minimum: int = 0,
    ) -> int:
        return duration_to_ticks(
            seconds,
            self.dt_phys,
            rounding=rounding,
            minimum=minimum,
        )


DEFAULT_TIMEBASE = Timebase(dt_phys=1.0 / 90.0, control_fps=15.0)
"""기본 90 Hz 물리 / 15 Hz 제어 시간축."""
