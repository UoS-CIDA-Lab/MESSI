"""physics module이 공유하는 immutable geometry 설정."""

from __future__ import annotations

import math
from dataclasses import dataclass

from messi.core.numeric import require_float32_representable

_MIN_BALL_RADIUS_M = 0.68 / (2.0 * math.pi)
_MAX_BALL_RADIUS_M = 0.70 / (2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class BallGeometry:
    """IFAB Law 2 circumference 범위 안의 spherical ball geometry."""

    radius_m: float = 0.11

    def __post_init__(self) -> None:
        value = require_float32_representable("radius_m", self.radius_m)
        if not _MIN_BALL_RADIUS_M <= value <= _MAX_BALL_RADIUS_M:
            raise ValueError(
                f"radius_m은 [{_MIN_BALL_RADIUS_M}, {_MAX_BALL_RADIUS_M}] 범위여야 한다"
            )


__all__ = ["BallGeometry"]
