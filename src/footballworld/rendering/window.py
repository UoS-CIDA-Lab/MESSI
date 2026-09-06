"""Time windows for bounded-memory replay production."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReplayWindow:
    """One half-open match interval selected on the control-frame clock."""

    name: str
    start_s: float = 0.0
    end_s: float | None = None

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("window name must be one non-empty path component")
        if not math.isfinite(self.start_s) or self.start_s < 0.0:
            raise ValueError("window start_s must be finite and non-negative")
        if self.end_s is not None and (
            not math.isfinite(self.end_s) or self.end_s <= self.start_s
        ):
            raise ValueError("window end_s must be finite and greater than start_s")

    def step_bounds(self, control_fps: float) -> tuple[int, int | None]:
        """Return half-open relative control-step bounds."""

        start = round(self.start_s * control_fps)
        end = None if self.end_s is None else round(self.end_s * control_fps)
        if end is not None and end <= start:
            raise ValueError("window is empty on the environment control grid")
        return start, end


__all__ = ["ReplayWindow"]
