"""고정 player slot의 roster 교체 세대를 host에서 추적한다."""

from __future__ import annotations

import jax
import numpy as np

from messi.analysis.causal import is_shot_attempt


class PlayerGenerationTracker:
    def __init__(self, roster) -> None:
        player_id, team, is_goalkeeper = jax.device_get(
            (roster.player_id, roster.team_id, roster.is_goalkeeper)
        )
        self._player_id = np.asarray(player_id, dtype=np.int32).copy()
        self._team = np.asarray(team, dtype=np.int32).copy()
        self._is_goalkeeper = np.asarray(is_goalkeeper, dtype=bool).copy()
        self._generation = np.zeros(self._player_id.shape, dtype=np.int32)
        self._start = np.zeros(self._player_id.shape, dtype=np.int64)
        self._closed: list[dict[str, int]] = []

    def update(self, roster, *, frame: int) -> None:
        player_id, team, is_goalkeeper = jax.device_get(
            (roster.player_id, roster.team_id, roster.is_goalkeeper)
        )
        player_id = np.asarray(player_id, dtype=np.int32)
        team = np.asarray(team, dtype=np.int32)
        is_goalkeeper = np.asarray(is_goalkeeper, dtype=bool)
        if (
            player_id.shape != self._player_id.shape
            or team.shape != self._team.shape
            or is_goalkeeper.shape != self._is_goalkeeper.shape
        ):
            raise ValueError("roster player축은 recording 중 바뀔 수 없다")
        changed = (
            (player_id != self._player_id)
            | (team != self._team)
            | (is_goalkeeper != self._is_goalkeeper)
        )
        for slot in np.flatnonzero(changed):
            self._closed.append(self._row(int(slot), int(frame)))
            self._generation[slot] += 1
            self._start[slot] = frame
            self._player_id[slot] = player_id[slot]
            self._team[slot] = team[slot]
            self._is_goalkeeper[slot] = is_goalkeeper[slot]

    def _row(self, slot: int, end_frame: int) -> dict[str, object]:
        return {
            "slot": slot,
            "generation": int(self._generation[slot]),
            "player_id": int(self._player_id[slot]),
            "team": int(self._team[slot]),
            "is_goalkeeper": bool(self._is_goalkeeper[slot]),
            "start_frame": int(self._start[slot]),
            "end_frame": int(end_frame),
        }

    def finish(self, frame: int) -> tuple[dict[str, object], ...]:
        rows = list(self._closed)
        rows.extend(self._row(slot, frame) for slot in range(self._player_id.size))
        return tuple(rows)


def build_player_action_credit(
    causal,
    generations,
    *,
    distance_by_generation_m: dict[tuple[int, int], float] | None = None,
    spatial_by_generation: dict[tuple[int, int], dict[str, object]] | None = None,
    goal_line_x: float = 52.5,
) -> tuple[dict[str, object], ...]:
    """Causal action과 이동거리를 해당 slot generation에 귀속한다."""

    distances = {} if distance_by_generation_m is None else distance_by_generation_m
    spatial = {} if spatial_by_generation is None else spatial_by_generation
    credits: dict[tuple[int, int], dict[str, object]] = {}
    for generation in generations:
        key = (generation["slot"], generation["generation"])
        credits[key] = {
            "slot": generation["slot"],
            "generation": generation["generation"],
            "player_id": generation["player_id"],
            "team": generation["team"],
            "is_goalkeeper": bool(generation.get("is_goalkeeper", False)),
            "pass_attempts": 0,
            "pass_completions": 0,
            "shot_attempts": 0,
            "goals": 0,
            "distance_m": float(distances.get(key, 0.0)),
            **spatial.get(
                key,
                {
                    "active_seconds": 0.0,
                    "controlled_possession_seconds": 0.0,
                    "mean_absolute_position_m": None,
                    "mean_in_possession_position_m": None,
                    "mean_out_of_possession_position_m": None,
                    "mean_teammate_distance_m": None,
                    "mean_nearest_teammate_distance_m": None,
                },
            ),
        }

    def locate(actor: int, frame: int):
        return next(
            (
                row
                for row in generations
                if row["slot"] == actor
                and row["start_frame"] <= frame < row["end_frame"]
            ),
            None,
        )

    for row in causal.pass_map:
        generation = locate(int(row["actor"]), int(row["source_frame"]))
        if generation is not None:
            credit = credits[(generation["slot"], generation["generation"])]
            credit["pass_attempts"] += 1
            credit["pass_completions"] += int(bool(row["completed"]))
    for row in causal.shot_map:
        if not is_shot_attempt(row, goal_line_x=goal_line_x):
            continue
        generation = locate(int(row["actor"]), int(row["source_frame"]))
        if generation is not None:
            credit = credits[(generation["slot"], generation["generation"])]
            credit["shot_attempts"] += 1
            credit["goals"] += int(row["outcome"] == "goal")
    return tuple(credits[key] for key in sorted(credits))
