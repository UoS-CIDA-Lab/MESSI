"""Chunk 경계를 보존하는 저빈도 경기 timeline 집계."""

from __future__ import annotations

import numpy as np


class PossessionTimelineAccumulator:
    def __init__(self, *, control_dt: float, interval_s: float = 30.0) -> None:
        self._dt = float(control_dt)
        interval_s = float(interval_s)
        if not np.isfinite(self._dt) or self._dt <= 0.0:
            raise ValueError("control_dt는 유한한 양수여야 한다")
        if not np.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("interval_s는 유한한 양수여야 한다")
        self._interval_frames = max(1, round(interval_s / self._dt))
        self._frames = 0
        self._controlled = np.zeros(2, dtype=np.int64)
        self._points: list[dict[str, object]] = []

    def add(self, owner_team) -> None:
        for team in np.asarray(owner_team, dtype=np.int32):
            if team in (0, 1):
                self._controlled[team] += 1
            self._frames += 1
            if self._frames % self._interval_frames == 0:
                self._append()

    def _append(self) -> None:
        total = max(int(np.sum(self._controlled)), 1)
        self._points.append(
            {
                "frame": self._frames,
                "time_s": self._frames * self._dt,
                "controlled_frames": (
                    int(self._controlled[0]),
                    int(self._controlled[1]),
                ),
                "share": (
                    float(self._controlled[0]) / total,
                    float(self._controlled[1]) / total,
                ),
            }
        )

    def finish(self) -> tuple[dict[str, object], ...]:
        if self._frames and (
            not self._points or self._points[-1]["frame"] != self._frames
        ):
            self._append()
        return tuple(self._points)
