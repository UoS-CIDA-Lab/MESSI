"""FootballWorld presentation contract를 따르는 host-side MP4 renderer."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from messi.core.constants import BALL_EVENT_GOAL
from messi.core.constants import MAX_SUPPORTED_PLAYER_SPEED_MPS
from messi.core.contact import ContactPhysics
from messi.core.geometry import BallGeometry
from messi.rendering import _video_backend as _backend
from messi.rendering._poisson_win_probability import WIN_PROBABILITY_RATE_MODEL
from messi.rendering._video_backend import (
    RenderStyle,
    _abort_ffmpeg,
    _display_seconds,
    _goal_visible_mask,
    _interpolated_visual_state,
    _output_frame_sample,
    _start_ffmpeg,
    _validated_field_half_extent,
)


def _intent_ring_segments(
    camera,
    position,
    intent,
    goalkeeper,
    active,
    *,
    contact_physics=ContactPhysics(),
    ball_geometry=BallGeometry(),
):
    return _backend._intent_ring_segments(
        camera,
        position,
        intent,
        goalkeeper,
        active,
        contact_physics=contact_physics,
        ball_geometry=ball_geometry,
    )


class _ReplayRenderer(_backend._ReplayRenderer):
    def __init__(
        self,
        style,
        player_count,
        contact_physics=ContactPhysics(),
        ball_geometry=BallGeometry(),
        field_half_extent=(52.5, 34.0),
        player_id=None,
        analysis=None,
        mode="analysis",
    ):
        super().__init__(
            style,
            player_count,
            contact_physics,
            ball_geometry,
            field_half_extent,
            player_id=player_id,
            analysis=analysis,
            mode=mode,
        )


def _render_segment(job):
    return _backend._render_segment(
        job, renderer_factory=_ReplayRenderer, encoder_factory=_start_ffmpeg
    )


def _trajectory_history(values, start: int, stop: int, current) -> np.ndarray:
    """현재 위치로 끝나는 짧은 host trajectory 구간을 만든다."""

    history = np.asarray(values[start : stop + 1]).copy()
    if history.shape[0]:
        history[-1] = current
    return history


def _build_video_analysis(match_report, host, team, *, source_start_frame: int):
    """기존 report와 전달 trajectory에서 영상 하단 요약을 축약한다."""

    dt = float(match_report.control_dt_s)
    credits = {
        (int(row["slot"]), int(row["generation"])): row
        for row in match_report.player_action_credit
    }
    generation_rows = []
    position = np.asarray(host.states.players.position, dtype=np.float64)
    active = np.asarray(host.states.players.active, dtype=bool)
    for generation in match_report.player_generations:
        slot = int(generation["slot"])
        key = (slot, int(generation["generation"]))
        credit = credits.get(key, {})
        start = max(int(generation["start_frame"]), source_start_frame)
        stop = min(int(generation["end_frame"]), source_start_frame + position.shape[0])
        start_local = start - source_start_frame
        stop_local = stop - source_start_frame
        top_speed = 0.0
        if 0 <= slot < position.shape[1] and stop_local - start_local >= 2:
            displacement = np.diff(position[start_local:stop_local, slot], axis=0)
            observed = active[start_local + 1:stop_local, slot] & active[start_local:stop_local - 1, slot]
            speed = np.linalg.norm(displacement, axis=-1) / dt * 3.6
            # Half-time/restart placement can move an active slot discontinuously.
            # Such one-frame jumps are not player speed samples.
            observed &= speed <= MAX_SUPPORTED_PLAYER_SPEED_MPS * 3.6
            if np.any(observed):
                top_speed = float(np.max(speed[observed]))
        generation_rows.append({
            "team": int(generation["team"]),
            "player_id": int(generation["player_id"]),
            "distance_km": float(credit.get("distance_m", 0.0)) / 1000.0,
            "top_speed_kmh": top_speed,
            "possession_min": float(credit.get("controlled_possession_seconds", 0.0)) / 60.0,
        })

    leaders = {}
    for metric in ("distance_km", "top_speed_kmh", "possession_min"):
        leaders[metric] = [
            row
            for team_index in range(2)
            for row in sorted(
                (item for item in generation_rows if item["team"] == team_index),
                key=lambda item: item[metric],
                reverse=True,
            )[:5]
        ]

    substitutions = []
    by_slot = {}
    for generation in match_report.player_generations:
        by_slot.setdefault(int(generation["slot"]), []).append(generation)
    for rows in by_slot.values():
        rows.sort(key=lambda row: int(row["start_frame"]))
        for outgoing, incoming in zip(rows, rows[1:]):
            if int(outgoing["player_id"]) == int(incoming["player_id"]):
                continue
            substitutions.append({
                "team": int(incoming["team"]),
                "out_player_id": int(outgoing["player_id"]),
                "in_player_id": int(incoming["player_id"]),
                "frame": int(incoming["start_frame"]),
                "minute": int(int(incoming["start_frame"]) * dt // 60.0),
            })
    substitutions.sort(key=lambda row: row["frame"])

    owner = np.asarray(host.states.possession_player, dtype=np.int32).reshape(-1)
    valid = (owner >= 0) & (owner < len(team))
    owner_team = np.full(owner.shape, -1, dtype=np.int32)
    owner_team[valid] = np.asarray(team, dtype=np.int32)[owner[valid]]
    prefix = tuple(
        np.concatenate(([0], np.cumsum(owner_team == team_index, dtype=np.int64)))
        for team_index in range(2)
    )
    window = max(1, round(300.0 / dt))
    sample = max(1, round(30.0 / dt))
    ends = list(range(sample, owner.size + 1, sample))
    if owner.size and (not ends or ends[-1] != owner.size):
        ends.append(owner.size)
    momentum = []
    possession_history = [{
        "time_min": source_start_frame * dt / 60.0,
        "team_share": (0.0, 0.0),
    }]
    for end in ends:
        begin = max(0, end - window)
        counts = tuple(int(item[end] - item[begin]) for item in prefix)
        controlled = sum(counts)
        cumulative_counts = tuple(int(item[end]) for item in prefix)
        cumulative_controlled = sum(cumulative_counts)
        elapsed = (source_start_frame + end) * dt / 60.0
        momentum.append({
            "time_min": elapsed,
            "balance": (counts[0] - counts[1]) / max(controlled, 1),
        })
        possession_history.append({
            "time_min": elapsed,
            "team_share": tuple(
                count / max(cumulative_controlled, 1) for count in cumulative_counts
            ),
        })

    final_active = np.asarray(host.states.players.final_active, dtype=bool)
    final_long = np.asarray(host.states.players.final_stamina_long, dtype=np.float64)
    final_short = np.asarray(host.states.players.final_stamina_short, dtype=np.float64)
    team_array = np.asarray(team, dtype=np.int32)
    latest_generation = {}
    for generation in match_report.player_generations:
        slot = int(generation["slot"])
        previous = latest_generation.get(slot)
        if previous is None or int(generation["end_frame"]) > int(previous["end_frame"]):
            latest_generation[slot] = generation
    final_stamina_players = []
    for slot, generation in latest_generation.items():
        if slot < 0 or slot >= len(final_active) or not final_active[slot]:
            continue
        final_stamina_players.append({
            "team": int(generation["team"]),
            "player_id": int(generation["player_id"]),
            "long": float(final_long[slot]),
            "short": float(final_short[slot]),
        })
    final_stamina = []
    for team_index in range(2):
        selected = (team_array == team_index) & final_active
        if np.any(selected):
            final_stamina.append({
                "long_mean": float(np.mean(final_long[selected])),
                "long_min": float(np.min(final_long[selected])),
                "short_mean": float(np.mean(final_short[selected])),
                "short_min": float(np.min(final_short[selected])),
            })
        else:
            final_stamina.append({
                "long_mean": 0.0, "long_min": 0.0,
                "short_mean": 0.0, "short_min": 0.0,
            })

    metric_history = _build_live_metric_history(
        match_report,
        host,
        team_array,
        source_start_frame=source_start_frame,
        dt=dt,
        position=position,
    )

    return {
        "leaders": leaders,
        "possession_share": tuple(float(value) for value in match_report.possession_share),
        "uncontrolled_share": max(
            0.0,
            1.0 - sum(int(value) for value in match_report.possession_frames)
            / max(int(match_report.frames), 1),
        ),
        "substitutions": tuple(substitutions),
        "momentum": tuple(momentum),
        "possession_history": tuple(possession_history),
        "final_stamina": tuple(final_stamina),
        "final_stamina_players": tuple(final_stamina_players),
        "metric_history": tuple(metric_history),
        "win_probability_rate_model": WIN_PROBABILITY_RATE_MODEL,
    }


def _build_live_metric_history(
    match_report,
    host,
    team: np.ndarray,
    *,
    source_start_frame: int,
    dt: float,
    position: np.ndarray,
) -> list[dict[str, object]]:
    """30초 prefix별 선수 이동·소유·스태미나 요약을 만든다."""

    position = np.asarray(position, dtype=np.float64)
    active = np.asarray(host.states.players.active, dtype=bool)
    owner = np.asarray(host.states.possession_player, dtype=np.int32).reshape(-1)
    generations = tuple(match_report.player_generations)
    count = position.shape[0]
    sample_frames = max(1, round(30.0 / dt))
    sample_indices = [0] if count else []
    sample_indices.extend(end - 1 for end in range(sample_frames, count + 1, sample_frames))
    if count and sample_indices[-1] != count - 1:
        sample_indices.append(count - 1)
    sample_indices = sorted(set(sample_indices))

    stamina_indices = np.asarray(
        host.states.players.stamina_sample_local_frames, dtype=np.int64
    )
    stamina_long = np.asarray(host.states.players.stamina_long_samples, dtype=np.float64)
    stamina_short = np.asarray(host.states.players.stamina_short_samples, dtype=np.float64)
    stamina_lookup = {
        int(frame): (stamina_long[index], stamina_short[index])
        for index, frame in enumerate(stamina_indices)
    }
    layout_ready = np.asarray(host.states.restart_layout_ready, dtype=bool)
    first_half_end = np.asarray(host.states.first_half_wall_end_tick, dtype=np.int64)
    control_tick = np.asarray(host.states.control_tick, dtype=np.int64)
    distance = {
        (int(row["slot"]), int(row["generation"])): 0.0 for row in generations
    }
    top_speed = {key: 0.0 for key in distance}
    possession_frames = {key: 0 for key in distance}
    history = []
    previous_sample = 0
    possession_cursor = 0

    for sample_number, current in enumerate(sample_indices):
        if sample_number:
            if current > previous_sample:
                step_distance = np.linalg.norm(
                    np.diff(position[previous_sample : current + 1], axis=0), axis=-1
                )
                speed = step_distance / dt * 3.6
                step_current = np.arange(previous_sample + 1, current + 1)
                reset = (
                    layout_ready[step_current] & ~layout_ready[step_current - 1]
                ) | (
                    (first_half_end[step_current] >= 0)
                    & (first_half_end[step_current - 1] < 0)
                )
                active_pair = (
                    active[step_current - 1] & active[step_current] & ~reset[:, None]
                )
                global_current = source_start_frame + step_current
                for row in generations:
                    slot = int(row["slot"])
                    key = (slot, int(row["generation"]))
                    valid_step = (
                        (global_current > int(row["start_frame"]))
                        & (global_current < int(row["end_frame"]))
                        & active_pair[:, slot]
                    )
                    if not np.any(valid_step):
                        continue
                    distances = step_distance[:, slot]
                    velocities = speed[:, slot]
                    distance[key] += float(np.sum(distances[valid_step]))
                    speed_valid = valid_step & (
                        velocities <= MAX_SUPPORTED_PLAYER_SPEED_MPS * 3.6
                    )
                    if np.any(speed_valid):
                        top_speed[key] = max(
                            top_speed[key], float(np.max(velocities[speed_valid]))
                        )

            possession_stop = current + 1
            if possession_cursor < possession_stop:
                possession_frames_in_window = owner[possession_cursor:possession_stop]
                global_frames = source_start_frame + np.arange(
                    possession_cursor, possession_stop
                )
                for row in generations:
                    slot = int(row["slot"])
                    key = (slot, int(row["generation"]))
                    in_generation = (
                        (global_frames >= int(row["start_frame"]))
                        & (global_frames < int(row["end_frame"]))
                    )
                    possession_frames[key] += int(
                        np.count_nonzero(
                            in_generation & (possession_frames_in_window == slot)
                        )
                    )
                possession_cursor = possession_stop

        global_frame = source_start_frame + current
        metric_generations = [
            row
            for row in generations
            if int(row["start_frame"]) <= global_frame
        ]
        active_generations = [
            row
            for row in metric_generations
            if global_frame < int(row["end_frame"])
            and 0 <= int(row["slot"]) < active.shape[1]
            and active[current, int(row["slot"])]
        ]
        current_rows = [
            {
                "team": int(row["team"]),
                "player_id": int(row["player_id"]),
                "distance_km": distance[(int(row["slot"]), int(row["generation"]))] / 1000.0,
                "top_speed_kmh": top_speed[(int(row["slot"]), int(row["generation"]))],
                "possession_min": possession_frames[
                    (int(row["slot"]), int(row["generation"]))
                ] * dt / 60.0,
            }
            for row in metric_generations
        ]
        leaders = {
            metric: [
                row
                for team_index in range(2)
                for row in sorted(
                    (entry for entry in current_rows if entry["team"] == team_index),
                    key=lambda entry: entry[metric],
                    reverse=True,
                )[:5]
            ]
            for metric in ("distance_km", "top_speed_kmh", "possession_min")
        }
        sampled_long, sampled_short = stamina_lookup[current]
        current_stamina = []
        for row in active_generations:
            slot = int(row["slot"])
            current_stamina.append({
                "team": int(row["team"]),
                "player_id": int(row["player_id"]),
                "long": float(sampled_long[slot]),
                "short": float(sampled_short[slot]),
            })
        stamina_by_team = []
        for team_index in range(2):
            members = [row for row in current_stamina if row["team"] == team_index]
            stamina_by_team.append({
                "long_mean": float(np.mean([row["long"] for row in members])) if members else 0.0,
                "short_mean": float(np.mean([row["short"] for row in members])) if members else 0.0,
                "players": sorted(members, key=lambda row: row["long"])[:3],
            })
        history.append({
            "time_min": float(control_tick[current]) * dt / 60.0,
            "leaders": leaders,
            "stamina": stamina_by_team,
        })
        previous_sample = current

    return history


def _validated_render_frame_rates(
    fps: object, control_fps: object
) -> tuple[int, float]:
    """Output FPS가 trajectory control frame 시각을 정확히 보존하는지 검증한다."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps는 양의 정수여야 한다")
    try:
        trajectory_fps = float(control_fps)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("control_fps는 finite positive여야 한다") from error
    if (
        isinstance(control_fps, bool)
        or not math.isfinite(trajectory_fps)
        or trajectory_fps <= 0.0
    ):
        raise ValueError("control_fps는 finite positive여야 한다")
    if fps < trajectory_fps and not math.isclose(
        float(fps), trajectory_fps, rel_tol=1.0e-9, abs_tol=0.0
    ):
        raise ValueError(
            "fps는 trajectory control_fps 이상이어야 한다: "
            f"fps={fps}, control_fps={trajectory_fps:g}"
        )
    ratio = float(fps) / trajectory_fps
    nearest_multiple = round(ratio)
    if nearest_multiple < 1 or not math.isclose(
        ratio, float(nearest_multiple), rel_tol=1.0e-9, abs_tol=1.0e-9
    ):
        raise ValueError(
            "fps는 trajectory control_fps의 양의 정수배여야 한다: "
            f"fps={fps}, control_fps={trajectory_fps:g}, ratio={ratio:g}"
        )
    return fps, trajectory_fps


def _resolve_ffmpeg_executable() -> str:
    """설치된 imageio-ffmpeg가 선택한 encoder executable을 반환한다."""

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MP4 rendering에는 imageio-ffmpeg dependency가 필요하다"
        ) from error
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except RuntimeError as error:
        raise RuntimeError("FFmpeg MP4 encoder를 찾을 수 없다") from error


def _goal_frame_mask(evaluation, frame_count: int) -> np.ndarray:
    """Post-transition state와 exact receipt에서 goal source frame을 찾는다."""

    scores = np.asarray(evaluation.states.score)
    if scores.shape != (frame_count, 2):
        raise ValueError("rendering score trajectory shape은 (frame, 2)여야 한다")
    goals = np.zeros(frame_count, dtype=bool)
    if frame_count > 1:
        goals[1:] = np.any(scores[1:] > scores[:-1], axis=-1)

    receipts = getattr(evaluation, "receipts", None)
    boundary = getattr(receipts, "boundary", None)
    occurred = getattr(boundary, "occurred", None)
    kind = getattr(boundary, "kind", None)
    if occurred is None and kind is None:
        return goals
    if occurred is None or kind is None:
        raise ValueError(
            "rendering boundary receipt에는 occurred와 kind가 모두 필요하다"
        )
    occurred = np.asarray(occurred, dtype=bool)
    kind = np.asarray(kind, dtype=np.int32)
    if occurred.shape != (frame_count,) or kind.shape != (frame_count,):
        raise ValueError("rendering boundary receipt의 time축은 state와 같아야 한다")
    return goals | (occurred & (kind == BALL_EVENT_GOAL))


def _copy_render_fields_to_host(
    evaluation,
    roster,
    source_slice=slice(None),
    *,
    include_analysis=False,
    analysis_sample_frames=300,
):
    """MP4에 필요한 trajectory leaf만 host로 옮긴다.

    Event Capture receipt 전체와 physics ``State`` 전체를 ``device_get``하면 긴 경기의
    rendering에서 불필요한 host transfer와 peak memory가 커진다. Goal overlay에 필요한
    boundary scalar가 있으면 그 둘만 보존한다.
    """

    states = evaluation.states
    actions = evaluation.actions
    first_half_wall_end_tick = getattr(states, "first_half_wall_end_tick", None)
    if first_half_wall_end_tick is None:
        first_half_wall_end_tick = np.full(
            states.control_tick[source_slice].shape, -1, dtype=np.int32
        )
    else:
        first_half_wall_end_tick = first_half_wall_end_tick[source_slice]
    receipts = getattr(evaluation, "receipts", None)
    boundary = getattr(receipts, "boundary", None)
    occurred = getattr(boundary, "occurred", None)
    kind = getattr(boundary, "kind", None)
    possession_player = getattr(
        getattr(states, "possession", None), "player", None
    )
    if include_analysis and possession_player is None:
        raise ValueError("MATCH MOMENTUM 분석에는 possession.player trajectory가 필요하다")
    stamina_long = getattr(states.players, "stamina_long", None)
    stamina_short = getattr(states.players, "stamina_short", None)
    if include_analysis and (stamina_long is None or stamina_short is None):
        raise ValueError("FINAL STAMINA 분석에는 long·short stamina trajectory가 필요하다")
    payload = (
        states.control_tick[source_slice],
        first_half_wall_end_tick,
        states.score[source_slice],
        states.players.position[source_slice],
        states.players.active[source_slice],
        states.ball.position[source_slice],
        actions.intent[source_slice],
        None if occurred is None else occurred[source_slice],
        None if kind is None else kind[source_slice],
        roster.team_id,
        roster.is_goalkeeper,
        roster.player_id,
    )
    if include_analysis:
        source_start = 0 if source_slice.start is None else int(source_slice.start)
        source_stop = states.control_tick.shape[0] if source_slice.stop is None else int(source_slice.stop)
        source_length = max(0, source_stop - source_start)
        sample_local = [0] if source_length else []
        sample_local.extend(
            end - 1
            for end in range(analysis_sample_frames, source_length + 1, analysis_sample_frames)
        )
        if source_length and sample_local[-1] != source_length - 1:
            sample_local.append(source_length - 1)
        sample_local = np.asarray(sorted(set(sample_local)), dtype=np.int32)
        sample_global = sample_local + source_start
        restart_layout_ready = getattr(states, "restart_layout_ready", None)
        if restart_layout_ready is None:
            restart_layout_ready = np.zeros(states.control_tick.shape, dtype=bool)
        has_lookahead = (
            isinstance(source_slice, slice)
            and source_slice.stop is not None
            and source_slice.stop < states.control_tick.shape[0]
        )
        stamina_end_offset = -2 if has_lookahead else -1
        payload += (
            possession_player[source_slice],
            stamina_long[source_slice][stamina_end_offset],
            stamina_short[source_slice][stamina_end_offset],
            states.players.active[source_slice][stamina_end_offset],
            sample_global,
            stamina_long[sample_global],
            stamina_short[sample_global],
            restart_layout_ready[source_slice],
        )
    copied = jax.device_get(payload)
    (
        control_tick,
        first_half_wall_end_tick,
        score,
        player_position,
        player_active,
        ball_position,
        intent,
        boundary_occurred,
        boundary_kind,
        team,
        goalkeeper,
        player_id,
    ) = copied[:12]
    host_possession_player = copied[12] if include_analysis else None
    final_stamina_long = copied[13] if include_analysis else None
    final_stamina_short = copied[14] if include_analysis else None
    final_active = copied[15] if include_analysis else None
    sample_global_frames = copied[16] if include_analysis else None
    stamina_long_samples = copied[17] if include_analysis else None
    stamina_short_samples = copied[18] if include_analysis else None
    restart_layout = copied[19] if include_analysis else None
    host_boundary = (
        None
        if boundary_occurred is None and boundary_kind is None
        else SimpleNamespace(occurred=boundary_occurred, kind=boundary_kind)
    )
    host = SimpleNamespace(
        states=SimpleNamespace(
            control_tick=control_tick,
            first_half_wall_end_tick=first_half_wall_end_tick,
            score=score,
            players=SimpleNamespace(
                position=player_position,
                active=player_active,
                final_stamina_long=final_stamina_long,
                final_stamina_short=final_stamina_short,
                final_active=final_active,
                stamina_sample_local_frames=(
                    None
                    if sample_global_frames is None
                    else np.asarray(sample_global_frames, dtype=np.int32)
                    - (0 if source_slice.start is None else int(source_slice.start))
                ),
                stamina_long_samples=stamina_long_samples,
                stamina_short_samples=stamina_short_samples,
            ),
            possession_player=host_possession_player,
            ball=SimpleNamespace(position=ball_position),
            restart_layout_ready=restart_layout,
        ),
        actions=SimpleNamespace(intent=intent),
        receipts=(
            None if host_boundary is None else SimpleNamespace(boundary=host_boundary)
        ),
    )
    return (
        host,
        np.asarray(team, dtype=np.int32),
        np.asarray(goalkeeper, dtype=bool),
        np.asarray(player_id, dtype=np.int32),
    )


def _validate_render_shapes(
    host,
    team: np.ndarray,
    goalkeeper: np.ndarray,
    player_id: np.ndarray,
    *,
    validate_values=True,
) -> int:
    """Encoder process를 열기 전에 dense rendering 축 contract를 검증한다."""

    ticks = host.states.control_tick
    if ticks.ndim != 1 or ticks.shape[0] <= 0:
        raise ValueError("rendering control_tick은 비어 있지 않은 (frame,)이어야 한다")
    frame_count = int(ticks.shape[0])
    position = host.states.players.position
    if position.ndim != 3 or position.shape[0] != frame_count or position.shape[2] != 2:
        raise ValueError(
            "rendering player position shape은 (frame, player, 2)여야 한다"
        )
    player_count = int(position.shape[1])
    expected_shapes = (
        ("score", np.shape(host.states.score), (frame_count, 2)),
        (
            "first-half boundary",
            np.shape(getattr(host.states, "first_half_wall_end_tick", ticks)),
            (frame_count,),
        ),
        (
            "player active",
            np.shape(host.states.players.active),
            (frame_count, player_count),
        ),
        (
            "ball position",
            np.shape(host.states.ball.position),
            (frame_count, 3),
        ),
        (
            "action intent",
            np.shape(host.actions.intent),
            (frame_count, player_count),
        ),
        ("roster team", team.shape, (player_count,)),
        ("roster goalkeeper", goalkeeper.shape, (player_count,)),
        ("roster player_id", player_id.shape, (player_count,)),
    )
    for name, actual, expected in expected_shapes:
        if actual != expected:
            raise ValueError(f"rendering {name} shape은 {expected}여야 한다: {actual}")
    boundary = getattr(getattr(host, "receipts", None), "boundary", None)
    if boundary is not None:
        occurred, kind = (
            getattr(boundary, "occurred", None),
            getattr(boundary, "kind", None),
        )
        if (occurred is None) != (kind is None):
            raise ValueError(
                "rendering boundary receipt에는 occurred와 kind가 모두 필요하다"
            )
        if occurred is not None and (
            np.shape(occurred) != (frame_count,) or np.shape(kind) != (frame_count,)
        ):
            raise ValueError(
                "rendering boundary receipt의 time축은 state와 같아야 한다"
            )
    if validate_values and np.any((team < 0) | (team > 1)):
        raise ValueError("rendering roster team_id는 0 또는 1이어야 한다")
    return frame_count


def _render_parallel(
    host,
    destination: Path,
    *,
    team,
    goalkeeper,
    player_id,
    style: RenderStyle,
    contact_physics,
    ball_geometry,
    field_half_extent,
    fps: int,
    control_fps: float,
    frame_count: int,
    output_count: int,
    goal_frames,
    goal_hold_frames: int,
    halftime_ticks: int,
    ffmpeg: str,
    workers: int,
    analysis=None,
    mode="analysis",
    output_start: int = 0,
) -> Path:
    """구간을 나눠 동시에 encode한 뒤 ffmpeg concat으로 이어 붙인다."""

    probability_history = ()
    if analysis is not None and analysis.get("win_probability_rate_model") is not None:
        probability_history = _backend._build_win_probability_history(
            host.states.control_tick,
            host.states.first_half_wall_end_tick,
            host.states.score,
            control_fps=control_fps,
            halftime_ticks=halftime_ticks,
            rate_model=analysis["win_probability_rate_model"],
        )
    visible = _goal_visible_mask(
        goal_frames,
        output_count,
        control_fps=control_fps,
        fps=fps,
        hold_frames=goal_hold_frames,
    )
    segments = min(workers, output_count - output_start)
    bounds = [
        output_start + round(index * (output_count - output_start) / segments)
        for index in range(segments + 1)
    ]
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        ignore_cleanup_errors=True,
    ) as temporary_directory:
        directory = Path(temporary_directory)
        jobs = []
        for index in range(segments):
            begin, finish = bounds[index], bounds[index + 1]
            if finish <= begin:
                continue
            first, _, _ = _output_frame_sample(
                begin,
                control_fps=control_fps,
                fps=fps,
                frame_count=frame_count,
            )
            _, last, _ = _output_frame_sample(
                finish - 1,
                control_fps=control_fps,
                fps=fps,
                frame_count=frame_count,
            )
            trail_frames = max(
                math.ceil(style.player_trail_seconds * control_fps),
                math.ceil(style.ball_trail_seconds * control_fps),
            )
            window_start = max(0, first - trail_frames)
            window = slice(window_start, last + 1)
            job_analysis = analysis
            if analysis is not None and probability_history:
                job_analysis = dict(analysis)
                job_analysis["win_probability_history"] = tuple(
                    point[1:] for point in probability_history if point[0] <= first
                )
            jobs.append(
                {
                    "start": begin,
                    "stop": finish,
                    "path": str(directory / f"segment-{index:04d}.mp4"),
                    "source_offset": window_start,
                    "frame_count": frame_count,
                    "position": np.asarray(host.states.players.position[window]),
                    "active": np.asarray(host.states.players.active[window]),
                    "ball": np.asarray(host.states.ball.position[window]),
                    "score": np.asarray(host.states.score[window]),
                    "intent": np.asarray(host.actions.intent[window]),
                    "control_tick": np.asarray(host.states.control_tick[window]),
                    "first_half_end": np.asarray(
                        host.states.first_half_wall_end_tick[window]
                    ),
                    "goal_visible": visible[begin:finish],
                    "team": team,
                    "goalkeeper": goalkeeper,
                    "player_id": player_id,
                    "analysis": job_analysis,
                    "mode": mode,
                    "style": style,
                    "contact_physics": contact_physics,
                    "ball_geometry": ball_geometry,
                    "field_half_extent": field_half_extent,
                    "fps": fps,
                    "control_fps": control_fps,
                    "halftime_ticks": halftime_ticks,
                    "ffmpeg": ffmpeg,
                }
            )
        # Package import 없이 standalone backend를 실행해 JAX 초기화를 피한다.
        children = []
        paths = []
        environment = os.environ.copy()
        environment["JAX_PLATFORMS"] = "cpu"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        try:
            for index, job in enumerate(jobs):
                arrays = {
                    key: value
                    for key, value in job.items()
                    if isinstance(value, np.ndarray)
                }
                metadata = {
                    key: value for key, value in job.items() if key not in arrays
                }
                metadata["style"] = {
                    name: value.item() if isinstance(value, np.generic) else value
                    for name, value in asdict(style).items()
                }
                metadata["contact_physics"] = {
                    name: float(getattr(contact_physics, name))
                    for name in (
                        "carry_radius_m",
                        "challenge_radius_m",
                        "goalkeeper_radius_m",
                    )
                }
                metadata["ball_geometry"] = {"radius_m": float(ball_geometry.radius_m)}
                payload = directory / f"input-{index}.npz"
                config = directory / f"input-{index}.json"
                np.savez(payload, **arrays)
                config.write_text(json.dumps(metadata), encoding="utf-8")
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(_backend.__file__).resolve()),
                            str(payload),
                            str(config),
                        ],
                        env=environment,
                        start_new_session=os.name == "posix",
                    )
                )
                paths.append(job["path"])
            for child in children:
                if child.wait() != 0:
                    raise RuntimeError("MP4 구간 renderer 실행에 실패했다")
        except BaseException:
            for child in children:
                with suppress(BaseException):
                    if os.name == "posix":
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGTERM)
                    elif child.poll() is None:
                        child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(child.pid, signal.SIGKILL)
                        else:
                            child.kill()
                        child.wait()
            raise
        listing = directory / "segments.txt"
        listing.write_text(
            "".join(f"file '{path}'\n" for path in paths), encoding="utf-8"
        )
        staging = directory / destination.name
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(staging),
            ],
            check=False,
            capture_output=True,
        )
        if (
            completed.returncode != 0
            or not staging.is_file()
            or staging.stat().st_size == 0
        ):
            raise RuntimeError(
                "FFmpeg segment concat에 실패했다: "
                f"{completed.stderr.decode('utf-8', 'replace')[:200]}"
            )
        staging.replace(destination)
    return destination


def render_match_video(
    evaluation,
    roster,
    path: str | Path,
    *,
    fps: int = 20,
    control_fps: float = 10.0,
    halftime_seconds: float = 45.0 * 60.0,
    style: RenderStyle | None = None,
    contact_physics: ContactPhysics = ContactPhysics(),
    ball_geometry: BallGeometry = BallGeometry(),
    field_half_extent=(52.5, 34.0),
    workers: int = 1,
    start_frame: int = 0,
    stop_frame: int | None = None,
    match_report=None,
    team_names: tuple[str, str] | None = None,
    mode: str = "analysis",
) -> Path:
    """Recording trajectory를 perspective H.264 MP4로 encode한다.

    Frame 생성은 output index의 함수이므로 ``workers``를 1보다 크게 주면 구간을
    나눠 별도 process에서 동시에 encode하고 ffmpeg concat으로 이어 붙인다.
    ``workers``는 상한이며 짧은 구간은 초기화 비용을 줄이기 위해 단일 process를 쓴다.
    Worker는 NumPy·Agg 전용 script를 실행하므로 caller의 main을 다시 import하지 않는다.

    ``[start_frame, stop_frame)``은 원본 control frame index 구간이다.
    생략하면 전체 trajectory를 출력하며 원본 시계와 overlay를 보존한다.
    ``team_names``를 전달하면 HOME/AWAY 기본 표기를 실제 팀명으로 바꾼다.
    ``mode``는 ``"analysis"``(기본) 또는 ``"rollout"``이며, rollout은 분석 패널을
    숨기고 경기장·미니맵·intent 표현을 화면 전체에 배치한다. rollout에서는
    ``match_report``를 전달할 수 없다.
    """

    fps, control_fps = _validated_render_frame_rates(fps, control_fps)
    if not isinstance(mode, str) or mode not in ("analysis", "rollout"):
        raise ValueError("mode는 'analysis' 또는 'rollout'이어야 한다")
    if mode == "rollout" and match_report is not None:
        raise ValueError("rollout mode에서는 match_report를 전달할 수 없다")
    if match_report is not None:
        try:
            report_control_dt = float(match_report.control_dt_s)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("match_report에 유효한 control_dt_s가 없다") from error
        if not np.isclose(report_control_dt, 1.0 / control_fps, rtol=1.0e-9, atol=0.0):
            raise ValueError("match_report의 control_dt_s가 control_fps와 다르다")
    if not math.isfinite(float(halftime_seconds)) or halftime_seconds <= 0.0:
        raise ValueError("halftime_seconds는 finite positive여야 한다")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers는 양의 정수여야 한다")
    style = RenderStyle() if style is None else style
    if type(style) is not RenderStyle:
        raise TypeError("style은 RenderStyle이어야 한다")
    if type(contact_physics) is not ContactPhysics:
        raise TypeError("contact_physics는 ContactPhysics여야 한다")
    if type(ball_geometry) is not BallGeometry:
        raise TypeError("ball_geometry는 BallGeometry여야 한다")
    if team_names is None:
        display_team_names = ("HOME", "AWAY")
    elif (
        not isinstance(team_names, (tuple, list))
        or len(team_names) != 2
        or any(
            not isinstance(name, str) or not name.strip() or len(name.strip()) > 12
            for name in team_names
        )
    ):
        raise ValueError("team_names는 비어 있지 않은 팀명 2개여야 한다 (각 12자 이하)")
    else:
        display_team_names = tuple(name.strip() for name in team_names)
    field_half_extent = _validated_field_half_extent(field_half_extent)
    frame_count = _validate_render_shapes(
        evaluation,
        roster.team_id,
        roster.is_goalkeeper,
        roster.player_id,
        validate_values=False,
    )
    for name, value in (("start_frame", start_frame), ("stop_frame", stop_frame)):
        if not (name == "stop_frame" and value is None) and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(f"{name}은 정수여야 한다")
    stop_frame = frame_count if stop_frame is None else stop_frame
    if not 0 <= start_frame < stop_frame <= frame_count:
        raise ValueError("구간은 0 <= start_frame < stop_frame <= frame 수여야 한다")
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("render_match_video output은 .mp4여야 한다")
    # GOAL hold 직전 score와 마지막 subframe의 lookahead만 함께 회수한다.
    source_start = max(
        0, start_frame - math.ceil(style.goal_hold_seconds * control_fps) - 1
    )
    source_stop = min(frame_count, stop_frame + 1)
    host, team, goalkeeper, player_id = _copy_render_fields_to_host(
        evaluation, roster, slice(source_start, source_stop),
        include_analysis=match_report is not None,
        analysis_sample_frames=max(1, round(30.0 * control_fps)),
    )
    analysis = (
        {"team_names": display_team_names}
        if match_report is None
        else _build_video_analysis(
            match_report, host, team, source_start_frame=source_start
        )
    )
    analysis["team_names"] = display_team_names
    if analysis is not None and (
        not math.isclose(control_fps, 10.0, rel_tol=0.0, abs_tol=1.0e-9)
        or not math.isclose(halftime_seconds, 45.0 * 60.0, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        analysis.pop("win_probability_rate_model", None)
    frame_count = _validate_render_shapes(host, team, goalkeeper, player_id)
    start_frame -= source_start
    stop_frame -= source_start
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_start = round(start_frame / control_fps * fps)
    output_count = max(1, round(stop_frame / control_fps * fps))
    goal_frames = _goal_frame_mask(host, frame_count)
    goal_hold_frames = max(1, round(style.goal_hold_seconds * fps))
    ffmpeg_executable = _resolve_ffmpeg_executable()
    workers = min(workers, max(1, (output_count - output_start) // 128))
    if workers > 1:
        return _render_parallel(
            host,
            destination,
            team=team,
            goalkeeper=goalkeeper,
            player_id=player_id,
            style=style,
            contact_physics=contact_physics,
            ball_geometry=ball_geometry,
            field_half_extent=field_half_extent,
            fps=fps,
            control_fps=control_fps,
            frame_count=frame_count,
            output_count=output_count,
            goal_frames=goal_frames,
            goal_hold_frames=goal_hold_frames,
            halftime_ticks=round(halftime_seconds * control_fps),
            ffmpeg=ffmpeg_executable,
            workers=workers,
            output_start=output_start,
            analysis=analysis,
            mode=mode,
        )
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        ignore_cleanup_errors=True,
    ) as temporary_directory:
        staging = Path(temporary_directory) / destination.name
        renderer = None
        process = None
        try:
            renderer = _ReplayRenderer(
                style,
                team.shape[0],
                contact_physics,
                ball_geometry,
                field_half_extent,
                player_id=player_id,
                analysis=analysis,
                mode=mode,
            )
            process = _start_ffmpeg(ffmpeg_executable, staging, fps=fps, style=style)
            stream = process.stdin
            if stream is None:
                raise RuntimeError("FFmpeg MP4 encoder의 input pipe가 없다")
            goal_visible = _goal_visible_mask(
                goal_frames,
                output_count,
                control_fps=control_fps,
                fps=fps,
                hold_frames=goal_hold_frames,
            )
            for output_index in range(output_start, output_count):
                source_index, upper_index, alpha = _output_frame_sample(
                    output_index,
                    control_fps=control_fps,
                    fps=fps,
                    frame_count=frame_count,
                )
                position, ball_position = _interpolated_visual_state(
                    host, source_index, upper_index, alpha
                )
                player_trail_frames = math.ceil(style.player_trail_seconds * control_fps)
                ball_trail_frames = math.ceil(style.ball_trail_seconds * control_fps)
                player_trail_start = max(0, source_index - player_trail_frames)
                ball_trail_start = max(0, source_index - ball_trail_frames)
                current_score = np.asarray(host.states.score[source_index])
                first_half_end = int(host.states.first_half_wall_end_tick[source_index])
                seconds = _display_seconds(
                    host.states.control_tick[source_index],
                    first_half_end,
                    halftime_ticks=round(halftime_seconds * control_fps),
                    control_fps=control_fps,
                )
                second_half = (
                    first_half_end >= 0
                    and int(host.states.control_tick[source_index]) >= first_half_end
                )
                frame = renderer.draw(
                    position=position,
                    active=np.asarray(host.states.players.active[source_index]),
                    ball=ball_position,
                    score=current_score,
                    intent=np.asarray(host.actions.intent[source_index]),
                    team=team,
                    goalkeeper=goalkeeper,
                    seconds=seconds,
                    second_half=second_half,
                    elapsed_minutes=(
                        float(host.states.control_tick[source_index])
                        / control_fps
                        / 60.0
                    ),
                    goal_visible=bool(goal_visible[output_index]),
                    player_trail=_trajectory_history(
                        host.states.players.position, player_trail_start, source_index, position
                    ),
                    player_active_trail=np.asarray(
                        host.states.players.active[player_trail_start : source_index + 1]
                    ),
                    ball_trail=_trajectory_history(
                        host.states.ball.position, ball_trail_start, source_index, ball_position
                    ),
                )
                try:
                    stream.write(frame)
                except OSError as error:
                    raise RuntimeError("FFmpeg MP4 frame 전달에 실패했다") from error
            try:
                stream.close()
            except OSError as error:
                raise RuntimeError(
                    "FFmpeg MP4 frame stream을 종료하지 못했다"
                ) from error
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg MP4 encoding에 실패했다 (exit code {return_code})"
                )
            if not staging.is_file() or staging.stat().st_size <= 0:
                raise RuntimeError("FFmpeg MP4 encoder가 유효한 output을 만들지 않았다")
            renderer.close()
            renderer = None
            staging.replace(destination)
        except BaseException:
            if process is not None:
                _abort_ffmpeg(process)
            if renderer is not None:
                with suppress(BaseException):
                    renderer.close()
            raise
    return destination


__all__ = ["RenderStyle", "render_match_video"]
