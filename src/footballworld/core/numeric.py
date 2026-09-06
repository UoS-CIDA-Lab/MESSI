"""Host-only numeric validation shared by public float32-facing configs."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np


def require_float32_representable(name: str, value: Real) -> float:
    """Return ``value`` as float after preserving its runtime value category.

    JAX runs FootballWorld's transition and built-in policies in float32.
    Python-finite configuration values must therefore not become infinity or
    signed zero when captured as float32 constants. This helper intentionally
    remains at the host validation boundary and is never part of a JAX step.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        represented = np.float32(result)
    if not np.isfinite(represented):
        raise ValueError(f"{name} must be finite when represented as float32")
    if result != 0.0 and float(represented) == 0.0:
        raise ValueError(f"{name} must not underflow to zero in float32")
    return result


__all__ = ["require_float32_representable"]
