"""Recording chunk를 고정 크기 공간 점유 통계로 축약한다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
SPATIAL_X_BINS = 35
SPATIAL_Y_BINS = 23


@dataclass(frozen=True, slots=True)
class SpatialSummary:
    x_edges_m: tuple[float, ...]
    y_edges_m: tuple[float, ...]
    ball_density_seconds: tuple[tuple[float, ...], ...]
    team_occupancy_seconds: tuple[
        tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]
    ]
    ball_out_of_pitch_seconds: float
    player_occupancy_seconds: tuple[
        tuple[tuple[float, ...], ...], ...
    ]
    player_active_seconds: tuple[float, ...]


class SpatialAccumulator:
    """Chunk 사이에 fixed histogram만 유지하는 host-side 집계기."""

    def __init__(
        self,
        team_id,
        *,
        control_dt: float,
        field_half_extent: tuple[float, float] = (
            0.5 * PITCH_LENGTH_M,
            0.5 * PITCH_WIDTH_M,
        ),
    ) -> None:
        self._team = np.asarray(team_id, dtype=np.int32)
        self._dt = float(control_dt)
        if not np.isfinite(self._dt) or self._dt <= 0.0:
            raise ValueError("control_dt는 유한한 양수여야 한다")
        extent = np.asarray(field_half_extent, dtype=np.float64)
        if extent.shape != (2,) or not np.all(np.isfinite(extent)):
            raise ValueError("field_half_extent는 유한한 (2,) 값이어야 한다")
        if np.any(extent <= 0.0):
            raise ValueError("field_half_extent는 양수여야 한다")
        self._x_edges = np.linspace(
            -extent[0],
            extent[0],
            SPATIAL_X_BINS + 1,
            dtype=np.float64,
        )
        self._y_edges = np.linspace(
            -extent[1],
            extent[1],
            SPATIAL_Y_BINS + 1,
            dtype=np.float64,
        )
        self._ball = np.zeros((SPATIAL_Y_BINS, SPATIAL_X_BINS), dtype=np.float64)
        self._team_occupancy = np.zeros(
            (2, SPATIAL_Y_BINS, SPATIAL_X_BINS), dtype=np.float64
        )
        self._player_occupancy = np.zeros(
            (self._team.size, SPATIAL_Y_BINS, SPATIAL_X_BINS), dtype=np.float64
        )
        self._player_active_seconds = np.zeros(self._team.size, dtype=np.float64)
        self._ball_out_of_pitch = 0.0

    def _indices(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = position[..., 0]
        y = position[..., 1]
        inside = (
            (x >= self._x_edges[0])
            & (x <= self._x_edges[-1])
            & (y >= self._y_edges[0])
            & (y <= self._y_edges[-1])
        )
        x_index = np.clip(
            np.searchsorted(self._x_edges, x, side="right") - 1,
            0,
            SPATIAL_X_BINS - 1,
        )
        y_index = np.clip(
            np.searchsorted(self._y_edges, y, side="right") - 1,
            0,
            SPATIAL_Y_BINS - 1,
        )
        return inside, y_index, x_index

    def add(self, states) -> None:
        ball_position = np.asarray(states.ball.position, dtype=np.float64)[..., :2]
        ball_live = np.asarray(states.ball.live, dtype=bool)
        inside, y_index, x_index = self._indices(ball_position)
        included = ball_live & inside
        np.add.at(self._ball, (y_index[included], x_index[included]), self._dt)
        self._ball_out_of_pitch += float(np.count_nonzero(ball_live & (~inside))) * self._dt

        position = np.asarray(states.players.position, dtype=np.float64)
        active = np.asarray(states.players.active, dtype=bool)
        inside, y_index, x_index = self._indices(position)
        included_player = active & inside
        player_index = np.broadcast_to(
            np.arange(self._team.size, dtype=np.int32), active.shape
        )
        flat_index = (
            player_index * (SPATIAL_Y_BINS * SPATIAL_X_BINS)
            + y_index * SPATIAL_X_BINS
            + x_index
        )
        self._player_occupancy += np.bincount(
            flat_index[included_player],
            minlength=self._player_occupancy.size,
        ).reshape(self._player_occupancy.shape) * self._dt
        self._player_active_seconds += np.count_nonzero(active, axis=0) * self._dt
        for team in range(2):
            included = active & inside & (self._team[None, :] == team)
            np.add.at(
                self._team_occupancy[team],
                (y_index[included], x_index[included]),
                self._dt,
            )

    def finish(self) -> SpatialSummary:
        matrix = lambda value: tuple(
            tuple(float(cell) for cell in row) for row in value
        )
        return SpatialSummary(
            x_edges_m=tuple(float(value) for value in self._x_edges),
            y_edges_m=tuple(float(value) for value in self._y_edges),
            ball_density_seconds=matrix(self._ball),
            team_occupancy_seconds=(
                matrix(self._team_occupancy[0]),
                matrix(self._team_occupancy[1]),
            ),
            ball_out_of_pitch_seconds=self._ball_out_of_pitch,
            player_occupancy_seconds=tuple(
                matrix(value) for value in self._player_occupancy
            ),
            player_active_seconds=tuple(
                float(value) for value in self._player_active_seconds
            ),
        )


class PlayerSpatialAccumulator:
    """선수 generation별 위치·점유·동료 간격을 bounded memory로 누적한다."""

    def __init__(self, team_id, *, control_dt: float) -> None:
        self._team = np.asarray(team_id, dtype=np.int32)
        self._dt = float(control_dt)
        if not np.isfinite(self._dt) or self._dt <= 0.0:
            raise ValueError("control_dt는 유한한 양수여야 한다")
        self._rows: dict[tuple[int, int], dict[str, object]] = {}

    def add(self, states, generation=None) -> None:
        position = np.asarray(states.players.position, dtype=np.float64)
        active = np.asarray(states.players.active, dtype=bool)
        owner = np.asarray(states.possession.player, dtype=np.int32)
        if position.ndim != 3 or position.shape[1:] != (self._team.size, 2):
            raise ValueError("player position은 (frame, player, 2)여야 한다")
        if active.shape != position.shape[:2] or owner.shape != position.shape[:1]:
            raise ValueError("player active 또는 possession frame축이 position과 다르다")
        generation = (
            np.zeros(self._team.shape, dtype=np.int32)
            if generation is None
            else np.asarray(generation, dtype=np.int32)
        )
        if generation.shape != self._team.shape:
            raise ValueError("player generation은 player축과 같아야 한다")

        finite = np.all(np.isfinite(position), axis=-1)
        valid_active = active & finite
        safe_owner = np.clip(owner, 0, self._team.size - 1)
        owner_valid = (owner >= 0) & (owner < self._team.size)
        owner_team = np.where(owner_valid, self._team[safe_owner], -1)
        for slot in range(self._team.size):
            key = (slot, int(generation[slot]))
            row = self._rows.setdefault(
                key,
                {
                    "active_frames": 0,
                    "possession_frames": 0,
                    "position_sum": np.zeros(2, dtype=np.float64),
                    "in_possession_position_sum": np.zeros(2, dtype=np.float64),
                    "in_possession_frames": 0,
                    "out_of_possession_position_sum": np.zeros(2, dtype=np.float64),
                    "out_of_possession_frames": 0,
                    "teammate_distance_sum": 0.0,
                    "teammate_distance_count": 0,
                    "nearest_teammate_sum": 0.0,
                    "nearest_teammate_count": 0,
                },
            )
            selected = valid_active[:, slot]
            row["active_frames"] += int(np.count_nonzero(selected))
            row["possession_frames"] += int(
                np.count_nonzero(selected & (owner == slot))
            )
            row["position_sum"] += np.sum(position[selected, slot], axis=0)
            in_possession = selected & (owner_team == self._team[slot])
            out_of_possession = selected & (owner_team >= 0) & (
                owner_team != self._team[slot]
            )
            row["in_possession_frames"] += int(np.count_nonzero(in_possession))
            row["out_of_possession_frames"] += int(
                np.count_nonzero(out_of_possession)
            )
            row["in_possession_position_sum"] += np.sum(
                position[in_possession, slot], axis=0
            )
            row["out_of_possession_position_sum"] += np.sum(
                position[out_of_possession, slot], axis=0
            )

            team_slots = np.flatnonzero(self._team == self._team[slot])
            teammate = valid_active[:, team_slots].copy()
            teammate[:, np.searchsorted(team_slots, slot)] = False
            pair_valid = selected[:, None] & teammate
            distance = np.linalg.norm(
                position[:, team_slots] - position[:, slot, None, :], axis=-1
            )
            row["teammate_distance_sum"] += float(np.sum(distance[pair_valid]))
            row["teammate_distance_count"] += int(np.count_nonzero(pair_valid))
            has_teammate = np.any(pair_valid, axis=1)
            if np.any(has_teammate):
                nearest = np.min(np.where(pair_valid, distance, np.inf), axis=1)
                row["nearest_teammate_sum"] += float(np.sum(nearest[has_teammate]))
                row["nearest_teammate_count"] += int(np.count_nonzero(has_teammate))

    def finish(self) -> dict[tuple[int, int], dict[str, object]]:
        result = {}
        for key, row in self._rows.items():
            active_frames = int(row["active_frames"])
            teammate_count = int(row["teammate_distance_count"])
            nearest_count = int(row["nearest_teammate_count"])
            position_sum = np.asarray(row["position_sum"], dtype=np.float64)
            in_frames = int(row["in_possession_frames"])
            out_frames = int(row["out_of_possession_frames"])
            result[key] = {
                "active_seconds": active_frames * self._dt,
                "controlled_possession_seconds": int(row["possession_frames"])
                * self._dt,
                "mean_absolute_position_m": (
                    tuple(float(value) for value in position_sum / active_frames)
                    if active_frames
                    else None
                ),
                "mean_in_possession_position_m": (
                    tuple(
                        float(value)
                        for value in np.asarray(
                            row["in_possession_position_sum"], dtype=np.float64
                        )
                        / in_frames
                    )
                    if in_frames
                    else None
                ),
                "mean_out_of_possession_position_m": (
                    tuple(
                        float(value)
                        for value in np.asarray(
                            row["out_of_possession_position_sum"], dtype=np.float64
                        )
                        / out_frames
                    )
                    if out_frames
                    else None
                ),
                "mean_teammate_distance_m": (
                    float(row["teammate_distance_sum"]) / teammate_count
                    if teammate_count
                    else None
                ),
                "mean_nearest_teammate_distance_m": (
                    float(row["nearest_teammate_sum"]) / nearest_count
                    if nearest_count
                    else None
                ),
            }
        return result


class TeamShapeAccumulator:
    """팀별 폭·깊이·중심과 pairwise 간격을 frame 평균으로 누적한다."""

    _DISTANCE_BIN_M = 0.5
    _DISTANCE_MAX_M = 130.0

    def __init__(self, team_id) -> None:
        self._team = np.asarray(team_id, dtype=np.int32)
        self._rows = [self._empty_row(), self._empty_row()]

    @staticmethod
    def _empty_row() -> dict[str, object]:
        return {
            "frames": 0,
            "width_sum": 0.0,
            "depth_sum": 0.0,
            "pairwise_sum": 0.0,
            "pairwise_histogram": np.zeros(260, dtype=np.int64),
            "centroid_sum": np.zeros(2, dtype=np.float64),
            "stretch_sum": 0.0,
            "longitudinal_max_gap_sum": 0.0,
            "opponent_centroid_distance_sum": 0.0,
            "opponent_centroid_distance_count": 0,
            "ball_centroid_distance_sum": 0.0,
            "ball_centroid_distance_count": 0,
            "in_frames": 0,
            "in_width_sum": 0.0,
            "in_depth_sum": 0.0,
            "out_frames": 0,
            "out_width_sum": 0.0,
            "out_depth_sum": 0.0,
        }

    def add(self, states) -> None:
        position = np.asarray(states.players.position, dtype=np.float64)
        active = np.asarray(states.players.active, dtype=bool)
        owner = np.asarray(states.possession.player, dtype=np.int32)
        ball_position = np.asarray(states.ball.position, dtype=np.float64)[..., :2]
        ball_live = np.asarray(states.ball.live, dtype=bool)
        safe_owner = np.clip(owner, 0, self._team.size - 1)
        owner_team = np.where(
            (owner >= 0) & (owner < self._team.size), self._team[safe_owner], -1
        )
        finite = np.all(np.isfinite(position), axis=-1)
        ball_valid = ball_live & np.all(np.isfinite(ball_position), axis=-1)
        centroids: list[np.ndarray | None] = [None, None]
        selected_frames: list[np.ndarray | None] = [None, None]
        for team in range(2):
            slots = np.flatnonzero(self._team == team)
            points = position[:, slots]
            valid = active[:, slots] & finite[:, slots]
            count = np.count_nonzero(valid, axis=1)
            selected = count >= 2
            if not np.any(selected):
                continue
            x = points[..., 0]
            y = points[..., 1]
            depth = np.max(np.where(valid, x, -np.inf), axis=1) - np.min(
                np.where(valid, x, np.inf), axis=1
            )
            width = np.max(np.where(valid, y, -np.inf), axis=1) - np.min(
                np.where(valid, y, np.inf), axis=1
            )
            centroid = np.sum(np.where(valid[..., None], points, 0.0), axis=1)
            centroid /= np.maximum(count, 1)[:, None]
            centroids[team] = centroid
            selected_frames[team] = selected
            squared_radius = np.sum((points - centroid[:, None, :]) ** 2, axis=-1)
            stretch = np.sqrt(
                np.sum(np.where(valid, squared_radius, 0.0), axis=1)
                / np.maximum(count, 1)
            )
            sorted_x = np.sort(np.where(valid, x, np.inf), axis=1)
            with np.errstate(invalid="ignore"):
                longitudinal_gap = np.diff(sorted_x, axis=1)
            longitudinal_gap = np.max(
                np.where(np.isfinite(longitudinal_gap), longitudinal_gap, -np.inf),
                axis=1,
            )
            pair_sum = np.zeros(position.shape[0], dtype=np.float64)
            pair_count = np.zeros(position.shape[0], dtype=np.int32)
            for left in range(slots.size):
                for right in range(left + 1, slots.size):
                    pair_valid = valid[:, left] & valid[:, right]
                    pair_distance = np.linalg.norm(
                        points[pair_valid, left] - points[pair_valid, right], axis=-1
                    )
                    pair_sum[pair_valid] += pair_distance
                    pair_count[pair_valid] += 1
                    row = self._rows[team]
                    row["pairwise_histogram"] += np.histogram(
                        pair_distance,
                        bins=row["pairwise_histogram"].size,
                        range=(0.0, self._DISTANCE_MAX_M),
                    )[0]
            pairwise = pair_sum / np.maximum(pair_count, 1)
            row = self._rows[team]
            row["frames"] += int(np.count_nonzero(selected))
            row["width_sum"] += float(np.sum(width[selected]))
            row["depth_sum"] += float(np.sum(depth[selected]))
            row["pairwise_sum"] += float(np.sum(pairwise[selected]))
            row["centroid_sum"] += np.sum(centroid[selected], axis=0)
            row["stretch_sum"] += float(np.sum(stretch[selected]))
            row["longitudinal_max_gap_sum"] += float(
                np.sum(longitudinal_gap[selected])
            )
            ball_selected = selected & ball_valid
            row["ball_centroid_distance_sum"] += float(
                np.sum(
                    np.linalg.norm(
                        ball_position[ball_selected] - centroid[ball_selected], axis=-1
                    )
                )
            )
            row["ball_centroid_distance_count"] += int(
                np.count_nonzero(ball_selected)
            )
            in_possession = selected & (owner_team == team)
            out_of_possession = selected & (owner_team >= 0) & (owner_team != team)
            row["in_frames"] += int(np.count_nonzero(in_possession))
            row["in_width_sum"] += float(np.sum(width[in_possession]))
            row["in_depth_sum"] += float(np.sum(depth[in_possession]))
            row["out_frames"] += int(np.count_nonzero(out_of_possession))
            row["out_width_sum"] += float(np.sum(width[out_of_possession]))
            row["out_depth_sum"] += float(np.sum(depth[out_of_possession]))

        if centroids[0] is not None and centroids[1] is not None:
            paired = selected_frames[0] & selected_frames[1]
            separation = np.linalg.norm(centroids[0] - centroids[1], axis=-1)
            for row in self._rows:
                row["opponent_centroid_distance_sum"] += float(
                    np.sum(separation[paired])
                )
                row["opponent_centroid_distance_count"] += int(
                    np.count_nonzero(paired)
                )

    def finish(self) -> tuple[dict[str, object], dict[str, object]]:
        def quantile(histogram: np.ndarray, probability: float) -> float | None:
            total = int(np.sum(histogram))
            if total == 0:
                return None
            index = int(
                np.searchsorted(np.cumsum(histogram), probability * total, side="left")
            )
            return (index + 0.5) * self._DISTANCE_BIN_M

        summaries = []
        for row in self._rows:
            frames = max(int(row["frames"]), 1)
            in_frames = int(row["in_frames"])
            out_frames = int(row["out_frames"])
            opponent_count = int(row["opponent_centroid_distance_count"])
            ball_count = int(row["ball_centroid_distance_count"])
            histogram = np.asarray(row["pairwise_histogram"], dtype=np.int64)
            summaries.append(
                {
                    "sampled_frames": int(row["frames"]),
                    "mean_width_m": float(row["width_sum"]) / frames,
                    "mean_depth_m": float(row["depth_sum"]) / frames,
                    "mean_pairwise_distance_m": float(row["pairwise_sum"]) / frames,
                    "pairwise_distance_p10_m": quantile(histogram, 0.10),
                    "pairwise_distance_p50_m": quantile(histogram, 0.50),
                    "pairwise_distance_p90_m": quantile(histogram, 0.90),
                    "mean_stretch_index_m": float(row["stretch_sum"]) / frames,
                    "mean_longitudinal_max_gap_m": float(
                        row["longitudinal_max_gap_sum"]
                    )
                    / frames,
                    "mean_opponent_centroid_distance_m": (
                        float(row["opponent_centroid_distance_sum"])
                        / opponent_count
                        if opponent_count
                        else None
                    ),
                    "mean_ball_centroid_distance_m": (
                        float(row["ball_centroid_distance_sum"]) / ball_count
                        if ball_count
                        else None
                    ),
                    "mean_centroid_m": tuple(
                        float(value)
                        for value in np.asarray(row["centroid_sum"]) / frames
                    ),
                    "in_possession_mean_width_m": (
                        float(row["in_width_sum"]) / in_frames
                        if in_frames
                        else None
                    ),
                    "in_possession_mean_depth_m": (
                        float(row["in_depth_sum"]) / in_frames
                        if in_frames
                        else None
                    ),
                    "out_of_possession_mean_width_m": (
                        float(row["out_width_sum"]) / out_frames
                        if out_frames
                        else None
                    ),
                    "out_of_possession_mean_depth_m": (
                        float(row["out_depth_sum"]) / out_frames
                        if out_frames
                        else None
                    ),
                }
            )
        return summaries[0], summaries[1]
