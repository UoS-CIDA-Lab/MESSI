"""공개 float32 설정값에 사용하는 host-only 수치 검증."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

_MAX_INT32 = (1 << 31) - 1


def require_bounded_integer(
    name: str,
    value: Integral,
    *,
    minimum: int,
    maximum: int = _MAX_INT32,
) -> int:
    """bool이 아닌 정수를 inclusive host 범위로 검증하고 canonicalize한다."""

    if isinstance(minimum, bool) or not isinstance(minimum, Integral):
        raise TypeError("minimum은 bool이 아닌 정수여야 한다")
    if isinstance(maximum, bool) or not isinstance(maximum, Integral):
        raise TypeError("maximum은 bool이 아닌 정수여야 한다")
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum > maximum:
        raise ValueError("minimum은 maximum 이하여야 한다")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name}은 bool이 아닌 정수여야 한다")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name}은 [{minimum}, {maximum}] 범위여야 한다")
    return result


def require_float32_representable(name: str, value: Real) -> float:
    """유한한 실수이며 float32에서 infinity나 0으로 변하지 않는지 검증한다."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name}은 실수여야 한다")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}은 유한해야 한다")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        represented = np.float32(result)
    if not np.isfinite(represented):
        raise ValueError(f"{name}은 float32로 표현해도 유한해야 한다")
    if result != 0.0 and float(represented) == 0.0:
        raise ValueError(f"{name}은 float32에서 0으로 underflow하면 안 된다")
    return result


def require_float32_field_half_extent(
    value: object,
    *,
    name: str = "field_half_extent",
) -> tuple[float, float]:
    """Static pitch extent를 physics와 observation에 안전한 float32 값으로 만든다."""

    try:
        extent_input = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name}는 실수 두 개여야 한다") from error
    if extent_input.shape != (2,):
        raise ValueError(f"{name}는 (2,) 값이어야 한다")
    half_x = require_float32_representable(f"{name}[0]", extent_input[0])
    half_y = require_float32_representable(f"{name}[1]", extent_input[1])
    if half_x <= 0.0 or half_y <= 0.0:
        raise ValueError(f"{name}는 양수여야 한다")

    extent = np.asarray((half_x, half_y), dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        standard_scale = extent / np.asarray((52.5, 34.0), dtype=np.float32)
        relative_extent = np.float32(2.0) * extent
        maximum_distance_squared = np.sum(
            relative_extent * relative_extent, dtype=np.float32
        )
    if not np.all(np.isfinite(standard_scale)) or np.any(standard_scale <= 0.0):
        raise ValueError(
            f"{name}의 표준 pitch 대비 float32 scale은 유한한 양수여야 한다"
        )
    if not np.isfinite(maximum_distance_squared) or maximum_distance_squared <= 0.0:
        raise ValueError(f"{name}의 float32 거리 제곱은 유한한 양수여야 한다")
    return float(extent[0]), float(extent[1])


__all__ = [
    "require_bounded_integer",
    "require_float32_field_half_extent",
    "require_float32_representable",
]
