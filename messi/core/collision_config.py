"""Player-player body collision의 static configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from messi.core.numeric import require_float32_representable


@dataclass(frozen=True, slots=True)
class BodyCollision:
    """FootballWorld body 치수에서 면적 등가 원을 만드는 collision envelope."""

    shoulder_width_m: float = 0.50
    torso_depth_m: float = 0.20

    @property
    def effective_diameter_m(self) -> float:
        """Capsule footprint와 면적이 같은 원의 지름."""

        core_length = self.shoulder_width_m - self.torso_depth_m
        area = (
            core_length * self.torso_depth_m + math.pi * (0.5 * self.torso_depth_m) ** 2
        )
        return 2.0 * math.sqrt(area / math.pi)

    def __post_init__(self) -> None:
        shoulder = require_float32_representable(
            "shoulder_width_m", self.shoulder_width_m
        )
        depth = require_float32_representable("torso_depth_m", self.torso_depth_m)
        if not 0.10 <= shoulder <= 0.80:
            raise ValueError("shoulder_width_m은 [0.10, 0.80] 범위여야 한다")
        if not 0.05 <= depth <= shoulder:
            raise ValueError("torso_depth_m은 [0.05, shoulder_width_m] 범위여야 한다")


@dataclass(frozen=True, slots=True)
class CollisionPhysics:
    """충돌 impulse와 bounded separation 설정."""

    normal_restitution: float = 0.05
    separation_iterations: int = 1

    def __post_init__(self) -> None:
        restitution = require_float32_representable(
            "normal_restitution", self.normal_restitution
        )
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("normal_restitution은 [0, 1] 범위여야 한다")
        if (
            not isinstance(self.separation_iterations, int)
            or isinstance(self.separation_iterations, bool)
            or not 1 <= self.separation_iterations <= 8
        ):
            raise ValueError("separation_iterations는 [1, 8] 정수여야 한다")


__all__ = ["BodyCollision", "CollisionPhysics"]
