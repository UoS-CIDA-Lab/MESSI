"""Lossless host sidecars for replay renders."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from footballworld.core.contact import INTENT_SOURCE_NAMES, INTENT_SOURCE_SCHEMA
from footballworld.rendering.events import EVENTS_SCHEMA
from footballworld.rendering.sparse_events import (
    SPARSE_ACTION_ENCODING,
    SPARSE_EVENT_ENCODING,
    sparse_action_trace,
    sparse_frame_events,
)
from footballworld.rendering.tracking import (
    TRACKING_COMPRESSION_LEVEL,
    TRACKING_FILENAME,
    TRACKING_SCHEMA,
    TRACKING_STORAGE_SCHEMA,
    open_tracking,
    open_tracking_jsonl,
    tracking_storage_receipt,
    write_tracking_archive,
)
from footballworld.rendering.transfer import HostFrame, to_jsonable

METADATA_SCHEMA = "footballworld.replay-metadata/9"


def _clock(frame: HostFrame, control_fps: float) -> float:
    return frame.control_tick / control_fps


def _tracking_fps(frames: list[HostFrame], control_fps: float) -> float | None:
    """Return a row rate only when control-tick spacing is uniform."""

    if len(frames) < 2:
        return None
    ticks = np.asarray([frame.control_tick for frame in frames], dtype=np.int64)
    deltas = np.diff(ticks)
    if deltas[0] <= 0 or np.any(deltas != deltas[0]):
        return None
    return control_fps / int(deltas[0])


def _public_clock(
    frame: HostFrame,
    control_fps: float,
    halftime_seconds: float,
    fulltime_seconds: float,
    halftime_enabled: bool,
) -> tuple[int, float, float]:
    """Return period, broadcast clock, and elapsed added time."""

    second_half = halftime_enabled and frame.first_half_wall_end_tick >= 0
    first_half_added_ticks = (
        frame.first_half_wall_end_tick - round(halftime_seconds * control_fps)
        if second_half
        else 0
    )
    display_ticks = frame.control_tick - max(first_half_added_ticks, 0)
    period = 2 if second_half else 1
    nominal_seconds = (
        fulltime_seconds if second_half or not halftime_enabled else halftime_seconds
    )
    display_seconds = display_ticks / control_fps
    return period, display_seconds, max(0.0, display_seconds - nominal_seconds)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            record,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        stream.write("\n")


def _missing_spec(missing: list[int], count: int) -> list[int] | str:
    """Compact the common all-missing case without losing its extent."""

    if not missing:
        return []
    if len(missing) == count:
        return "all"
    return missing


def _unknown_boundary_last_contact() -> dict[str, Any]:
    """Return a fail-closed last-touch attribution record."""

    return {
        "known": False,
        "slot": None,
        "player_id": None,
        "slot_generation": None,
        "team": None,
        "intent": None,
        "intent_source": None,
        "mechanism": None,
        "outcome": None,
        "kick_applied": None,
    }


def _transition_identity_arrays(
    frame: HostFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the identities that executed this frame's physics transition."""

    source = frame.pre_management_identity
    if source is None:
        arrays = (
            frame.player_id,
            frame.slot_generation,
            frame.team_id,
            frame.is_goalkeeper,
        )
    else:
        try:
            arrays = tuple(
                np.asarray(source[name])
                for name in (
                    "player_id",
                    "slot_generation",
                    "team_id",
                    "is_goalkeeper",
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("pre-management identity is incomplete") from exc

    player_count = int(frame.player_id.shape[0])
    if any(value.shape != (player_count,) for value in arrays):
        raise ValueError("transition identity arrays must share one player axis")
    return arrays


def _initial_roster_records(frame: HostFrame) -> list[dict[str, Any]]:
    """Preserve both sides of the first captured management transition.

    The sidecar cannot recover identities last seen before the captured window.
    It records the first post-management state plus distinct identities that
    executed the first transition, with transition attributes winning when an
    identity appears on both sides.
    """

    transition = _transition_identity_arrays(frame)
    post_management = (
        frame.player_id,
        frame.slot_generation,
        frame.team_id,
        frame.is_goalkeeper,
    )
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for arrays in (post_management, transition):
        player_id, generation, team_id, is_goalkeeper = arrays
        for slot in range(int(frame.player_id.shape[0])):
            identity = (int(player_id[slot]), int(generation[slot]))
            records[identity] = {
                "slot": slot,
                "player_id": identity[0],
                "slot_generation": identity[1],
                "team": int(team_id[slot]),
                "goalkeeper": bool(is_goalkeeper[slot]),
            }
    return list(records.values())


def _pre_management_identity_records(
    frame: HostFrame,
) -> list[dict[str, Any]] | None:
    """Return sparse pre-command overrides for slots changed by management."""

    if frame.pre_management_identity is None:
        return None
    player_id, generation, team_id, is_goalkeeper = _transition_identity_arrays(frame)
    changed = (
        (player_id != frame.player_id)
        | (generation != frame.slot_generation)
        | (team_id != frame.team_id)
        | (is_goalkeeper != frame.is_goalkeeper)
    )
    return [
        {
            "slot": int(slot),
            "player_id": int(player_id[slot]),
            "slot_generation": int(generation[slot]),
            "team": int(team_id[slot]),
            "goalkeeper": bool(is_goalkeeper[slot]),
        }
        for slot in np.flatnonzero(changed)
    ]


def _last_contact_at_boundary(frame: HostFrame) -> dict[str, Any]:
    """Identify the post-frame last touch without inventing an actor.

    This is last-touch attribution only. It is not a persistent causal shot ID
    and cannot by itself define a goals-per-shot conversion.
    """

    contact = frame.last_contact
    if contact is None:
        return _unknown_boundary_last_contact()

    def scalar(name: str) -> Any:
        try:
            source = (
                contact[name]
                if isinstance(contact, Mapping)
                else getattr(contact, name)
            )
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"last_contact is missing {name!r}") from exc
        value = np.asarray(source)
        if value.shape != ():
            raise ValueError(f"last_contact.{name} must be scalar")
        return value.item()

    actor = int(scalar("actor"))
    player_count = int(frame.player_id.shape[0])
    if actor < 0 or actor >= player_count:
        return _unknown_boundary_last_contact()
    player_id, slot_generation, team_id, _ = _transition_identity_arrays(frame)

    return {
        "known": True,
        "slot": actor,
        "player_id": int(player_id[actor]),
        "slot_generation": int(slot_generation[actor]),
        "team": int(team_id[actor]),
        "intent": int(scalar("intent")),
        "intent_source": int(scalar("intent_source")),
        "mechanism": int(scalar("mechanism")),
        "outcome": int(scalar("outcome")),
        "kick_applied": bool(scalar("kick_applied")),
    }


def _annotate_boundary_last_contact(
    frame: HostFrame, exact_event: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Attach one derived attribution record to each exact boundary row."""

    if exact_event is None:
        return None
    attribution = None
    rows = []
    for row in exact_event["events"]:
        if row["type"] != "boundary":
            rows.append(row)
            continue
        if attribution is None:
            attribution = _last_contact_at_boundary(frame)
        rows.append({**row, "last_contact_at_boundary": dict(attribution)})
    return {**exact_event, "events": rows}


def _substitution_records(value: Any) -> list[dict[str, int]] | None:
    """Return only committed substitution cells in deterministic row order."""

    if value is None:
        return None

    def field(name: str) -> np.ndarray:
        source = value[name] if isinstance(value, Mapping) else getattr(value, name)
        return np.asarray(source)

    occurred = field("occurred").astype(bool, copy=False)
    names = (
        "team",
        "player_slot",
        "outgoing_player_id",
        "incoming_player_id",
        "slot_generation",
        "control_tick",
    )
    fields = {name: field(name) for name in names}
    if any(item.shape != occurred.shape for item in fields.values()):
        raise ValueError("substitution event fields must share one fixed shape")
    if occurred.ndim == 0:
        indices = [()] if bool(occurred) else []
    else:
        indices = [tuple(index) for index in np.argwhere(occurred)]
    return [
        {
            "type": "substitution",
            "team": int(fields["team"][index]),
            "slot": int(fields["player_slot"][index]),
            "player_out": int(fields["outgoing_player_id"][index]),
            "player_in": int(fields["incoming_player_id"][index]),
            "slot_generation": int(fields["slot_generation"][index]),
            "control_tick": int(fields["control_tick"][index]),
        }
        for index in indices
    ]


def _validate_substitution_records(
    frame: HostFrame, records: list[dict[str, int]] | None
) -> None:
    """Reject sidecars that contradict the selected post-command frame."""

    if records is None:
        return
    player_count = int(frame.player_id.shape[0])
    for event in records:
        if event["control_tick"] != frame.control_tick:
            raise ValueError(
                "substitution event control_tick must match its render frame"
            )
        slot = event["slot"]
        if not 0 <= slot < player_count:
            raise ValueError("substitution event slot is outside the player array")
        if event["team"] != int(frame.team_id[slot]):
            raise ValueError("substitution event team contradicts its render frame")
        if event["player_in"] != int(frame.player_id[slot]):
            raise ValueError(
                "substitution incoming player contradicts its render frame"
            )
        if event["slot_generation"] != int(frame.slot_generation[slot]):
            raise ValueError("substitution generation contradicts its render frame")


def _acting_goalkeeper_records(value: Any) -> list[dict[str, Any]] | None:
    """Return only committed role assignments in deterministic team order."""

    if value is None:
        return None

    def field(name: str) -> np.ndarray:
        source = value[name] if isinstance(value, Mapping) else getattr(value, name)
        return np.asarray(source)

    occurred = field("occurred").astype(bool, copy=False)
    names = (
        "team",
        "player_slot",
        "player_id",
        "slot_generation",
        "environment_forced",
        "control_tick",
    )
    fields = {name: field(name) for name in names}
    if any(item.shape != occurred.shape for item in fields.values()):
        raise ValueError("acting goalkeeper event fields must share one fixed shape")
    if occurred.ndim == 0:
        indices = [()] if bool(occurred) else []
    else:
        indices = [tuple(index) for index in np.argwhere(occurred)]
    return [
        {
            "type": "acting_goalkeeper",
            "team": int(fields["team"][index]),
            "slot": int(fields["player_slot"][index]),
            "player_id": int(fields["player_id"][index]),
            "slot_generation": int(fields["slot_generation"][index]),
            "environment_forced": bool(fields["environment_forced"][index]),
            "control_tick": int(fields["control_tick"][index]),
        }
        for index in indices
    ]


def _formation_records(
    requested: Any,
    layout_index: Any,
    applied: Any,
    tactical_epoch: Any = None,
    changed_control_tick: Any = None,
) -> list[dict[str, Any]] | None:
    """Return one authoritative receipt for each requested team layout."""

    if requested is None and layout_index is None and applied is None:
        return None
    if requested is None or layout_index is None or applied is None:
        raise ValueError("formation receipt fields must be provided together")
    requested = np.asarray(requested, dtype=bool)
    layout_index = np.asarray(layout_index)
    applied = np.asarray(applied, dtype=bool)
    if requested.shape != (2,) or layout_index.shape != (2,) or applied.shape != (2,):
        raise ValueError("formation receipt fields must have shape [2]")
    if np.any(applied & ~requested):
        raise ValueError("an unrequested formation cannot be recorded as applied")
    if (tactical_epoch is None) != (changed_control_tick is None):
        raise ValueError("formation epoch and changed tick must be provided together")
    if tactical_epoch is not None:
        tactical_epoch = np.asarray(tactical_epoch)
        changed_control_tick = np.asarray(changed_control_tick)
        if tactical_epoch.shape != (2,) or changed_control_tick.shape != (2,):
            raise ValueError("formation epoch fields must have shape [2]")
    records = []
    for team in range(2):
        if not bool(requested[team]):
            continue
        record = {
            "type": "formation",
            "team": team,
            "layout_index": int(layout_index[team]),
            "applied": bool(applied[team]),
        }
        if bool(applied[team]) and tactical_epoch is not None:
            record.update(
                tactical_epoch=int(tactical_epoch[team]),
                formation_changed_control_tick=int(changed_control_tick[team]),
            )
        records.append(record)
    return records


def _validate_acting_goalkeeper_records(
    frame: HostFrame, records: list[dict[str, Any]] | None
) -> None:
    """Reject role sidecars that contradict the selected post-command frame."""

    if records is None:
        return
    player_count = int(frame.player_id.shape[0])
    for event in records:
        if event["control_tick"] != frame.control_tick:
            raise ValueError(
                "acting goalkeeper event control_tick must match its render frame"
            )
        slot = event["slot"]
        if not 0 <= slot < player_count:
            raise ValueError("acting goalkeeper event slot is outside the player array")
        if event["team"] != int(frame.team_id[slot]):
            raise ValueError(
                "acting goalkeeper event team contradicts its render frame"
            )
        if event["player_id"] != int(frame.player_id[slot]):
            raise ValueError("acting goalkeeper player_id contradicts its render frame")
        if event["slot_generation"] != int(frame.slot_generation[slot]):
            raise ValueError(
                "acting goalkeeper generation contradicts its render frame"
            )
        if not bool(frame.active[slot]) or not bool(frame.is_goalkeeper[slot]):
            raise ValueError(
                "acting goalkeeper event must identify the active post-command goalkeeper"
            )


def write_replay_sidecars(
    frames: list[HostFrame],
    video_path: str | Path,
    *,
    control_fps: float,
    video_sample_fps: float,
    sample_every: int,
    video_sample_frame_count: int,
    video_frame_count: int,
    video_fps: float,
    match_index: int,
    stadium: Any,
    halftime_seconds: float = 45.0 * 60.0,
    fulltime_seconds: float = 90.0 * 60.0,
    halftime_enabled: bool = True,
    metadata: Any = None,
    render_metadata: Any = None,
    match_manifest: Any = None,
    collect_exact_events: bool = True,
    frame_offset: int = 0,
    include_all_action_controls: bool = False,
) -> tuple[Path, Path, Path, list[Any]]:
    """Stream source-grid sidecars and record the distinct video time axis."""

    if not frames:
        raise ValueError("frames must not be empty")
    if video_sample_frame_count < 1 or video_frame_count < 1:
        raise ValueError("video sample and encoded frame counts must be positive")
    if not np.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError("video_fps must be finite and positive")
    if (
        not isinstance(frame_offset, int)
        or isinstance(frame_offset, bool)
        or frame_offset < 0
    ):
        raise ValueError("frame_offset must be a non-negative integer")
    if type(include_all_action_controls) is not bool:
        raise TypeError("include_all_action_controls must be bool")

    video_path = Path(video_path)
    event_path = video_path.parent / "event.json"
    tracking_path = video_path.parent / TRACKING_FILENAME
    metadata_path = video_path.parent / "metadata.json"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    missing_substitution_frames = [
        index for index, frame in enumerate(frames) if frame.substitution_events is None
    ]
    missing_acting_goalkeeper_frames = [
        index
        for index, frame in enumerate(frames)
        if frame.acting_goalkeeper_events is None
    ]
    missing_frames = [
        index for index, frame in enumerate(frames) if frame.frame_events is None
    ]
    missing_action_control_frames = [
        index
        for index, frame in enumerate(frames)
        if frame.action_trace is None or frame.submitted_action is None
    ]
    event_header = {
        "schema": EVENTS_SCHEMA,
        "match_index": match_index,
        "available": not missing_frames,
        "frame_event_encoding": SPARSE_EVENT_ENCODING,
        "omitted_event_slots": "canonical_empty_sentinels",
        "clock_source": "control_tick/control_fps",
        "intent_source_schema": INTENT_SOURCE_SCHEMA,
        "intent_source_names": list(INTENT_SOURCE_NAMES),
        "action_encoding": SPARSE_ACTION_ENCODING,
        "action_scope": (
            "all_players"
            if include_all_action_controls
            else "non_move_or_forced_or_realized_contact"
        ),
        "action_categorical_source": "authoritative_action_trace",
        "action_continuous_source": "submitted_policy_action",
        "intended_receiver_semantics": {
            "field": "intended_receiver_player_id",
            "scope": "retained_pass_action_rows_only",
            "identity": "registered_player_id",
            "source": "submitted_player_policy_plan",
            "unknown": None,
            "completion_claim": False,
        },
        "complete_action_reconstruction": (
            include_all_action_controls and not missing_action_control_frames
        ),
        "omitted_move_controls": ("none" if include_all_action_controls else "unknown"),
        "frame_order": {
            "action": "pre_management_transition",
            "frame_events": "pre_management_transition",
            "transition": "pre_management_transition",
            "substitutions": "post_transition_management_transaction",
            "acting_goalkeepers": "post_transition_management_transaction",
            "formations": "post_transition_management_transaction",
            "set_piece_taker_changes": "post_transition_management_transaction",
            "tracking": "post_management_state",
        },
        "transition_identity_semantics": {
            "default_source": "same_control_tick_tracking.players",
            "override_field": "pre_management_identity",
            "override_key": "slot",
            "lookup": "override_by_slot_then_tracking",
            "scope": "action_and_frame_events",
            "absent_override": "transition_identity_equals_tracking_identity",
        },
        "action_controls_available": not missing_action_control_frames,
        "missing_action_control_frames": _missing_spec(
            missing_action_control_frames,
            len(frames),
        ),
        "missing_frames": _missing_spec(missing_frames, len(frames)),
        "boundary_last_contact_semantics": {
            "record": "last_contact_at_boundary",
            "source": "post_frame.last_contact",
            "scope": "last_touch_attribution_only",
            "persistent_causal_shot_id": False,
            "goals_per_shot_conversion": False,
            "absent_boundary": "no_record",
        },
        "substitution_events_available": not missing_substitution_frames,
        "missing_substitution_frames": _missing_spec(
            missing_substitution_frames, len(frames)
        ),
        "acting_goalkeeper_events_available": not missing_acting_goalkeeper_frames,
        "missing_acting_goalkeeper_frames": _missing_spec(
            missing_acting_goalkeeper_frames, len(frames)
        ),
    }
    exact_events: list[Any] = []
    substitution_records = [
        _substitution_records(frame.substitution_events) for frame in frames
    ]
    acting_goalkeeper_records = [
        _acting_goalkeeper_records(frame.acting_goalkeeper_events) for frame in frames
    ]
    for frame, substitutions, acting_goalkeepers in zip(
        frames,
        substitution_records,
        acting_goalkeeper_records,
        strict=True,
    ):
        _validate_substitution_records(frame, substitutions)
        _validate_acting_goalkeeper_records(frame, acting_goalkeepers)

    common_rows: list[dict[str, Any]] = []
    with event_path.open("w", encoding="utf-8") as event_stream:
        event_stream.write(
            json.dumps(
                event_header,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )[:-1]
        )
        event_stream.write(',"frames":[')
        for index, frame in enumerate(frames):
            exact_event = _annotate_boundary_last_contact(
                frame,
                sparse_frame_events(frame.frame_events),
            )
            substitutions = substitution_records[index]
            acting_goalkeepers = acting_goalkeeper_records[index]
            pre_management_identity = _pre_management_identity_records(frame)
            if collect_exact_events:
                exact_events.append(exact_event)
            period, display_clock_s, added_time_s = _public_clock(
                frame,
                control_fps,
                halftime_seconds,
                fulltime_seconds,
                halftime_enabled,
            )
            common = {
                "frame": frame_offset + index,
                "control_tick": frame.control_tick,
                "clock_s": _clock(frame, control_fps),
                "period": period,
                "display_clock_s": display_clock_s,
                "added_time_s": added_time_s,
                "dead_ball_s": frame.dead_ball_control_ticks / control_fps,
                "first_half_live_extension_s": (
                    frame.first_half_live_extension_ticks / control_fps
                ),
            }
            common_rows.append(common)
            event_record = {
                "frame": frame_offset + index,
                "control_tick": frame.control_tick,
                "clock_s": _clock(frame, control_fps),
                "available": exact_event is not None,
                "action": sparse_action_trace(
                    frame.action_trace,
                    frame.submitted_action,
                    frame.frame_events,
                    frame.intended_receiver_ids,
                    include_move=include_all_action_controls,
                ),
                "frame_events": exact_event,
                "substitutions": substitutions,
                "acting_goalkeepers": acting_goalkeepers,
                "transition": to_jsonable(frame.telemetry),
            }
            if pre_management_identity:
                event_record["pre_management_identity"] = pre_management_identity
            if index:
                event_stream.write(",")
            json.dump(
                event_record,
                event_stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

        event_stream.write("]}\n")

    tracking_index = write_tracking_archive(tracking_path, frames, common_rows)

    first = frames[0]
    tracking_fps = _tracking_fps(frames, control_fps)
    tracking_span_s = (frames[-1].control_tick - frames[0].control_tick) / control_fps
    metadata_record = {
        "schema": METADATA_SCHEMA,
        "match_index": match_index,
        "source_frame_count": len(frames),
        "tracking_frame_count": len(frames),
        "video_sample_frame_count": video_sample_frame_count,
        "video_frame_count": video_frame_count,
        "control_fps": control_fps,
        "tracking_fps": tracking_fps,
        "video_sample_fps": video_sample_fps,
        "video_fps": video_fps,
        "sample_every": sample_every,
        "time_axis": {
            "tracking_clock": "absolute_control_tick/control_fps",
            "video_clock": "absolute_video_time_s",
            "video_origin_control_tick": int(first.control_tick),
            "video_origin_time_s": first.control_tick / control_fps,
            "video_time_step_s": 1.0 / video_fps,
            "video_sample_semantics": (
                "causal_hold_latest_source_at_or_before_relative_sample_time"
            ),
            "tracking_rows": "source_frames",
            "tracking_uniform": tracking_fps is not None,
            "tracking_span_s": tracking_span_s,
            "video_sampling": "causal_hold_on_control_tick",
            "video_decimation": "retain_each_sample_every_frame_and_preserve_duration",
            "video_duration_s": video_frame_count / video_fps,
        },
        "clock": {
            "halftime_seconds": halftime_seconds,
            "fulltime_seconds": fulltime_seconds,
            "halftime_enabled": halftime_enabled,
            "added_time_rule": (
                "out_of_play_restart_control_frames_plus_"
                "period_boundary_penalty_completion"
            ),
            "goalkeeper_hold_counts_as_dead_ball": False,
        },
        "stadium": {
            key: to_jsonable(getattr(stadium, key))
            for key in (
                "length",
                "width",
                "goal_width",
                "goal_height",
                "penalty_area_length",
                "penalty_area_width",
                "goal_area_length",
                "goal_area_width",
                "center_circle_radius",
                "penalty_arc_radius",
                "corner_arc_radius",
            )
        },
        "identity": {
            "slot_generation_available": bool(
                all(np.all(frame.slot_generation >= 0) for frame in frames)
            ),
            "untracked_slot_generation": -1,
        },
        "tracking_storage": tracking_storage_receipt(tracking_path, tracking_index),
        "match_manifest": to_jsonable(match_manifest),
        "roster": _initial_roster_records(first),
        "render": to_jsonable(render_metadata),
        "user_metadata": to_jsonable(metadata),
    }
    _write_json(metadata_path, metadata_record)
    return event_path, tracking_path, metadata_path, exact_events


__all__ = [
    "EVENTS_SCHEMA",
    "METADATA_SCHEMA",
    "TRACKING_COMPRESSION_LEVEL",
    "TRACKING_FILENAME",
    "TRACKING_SCHEMA",
    "TRACKING_STORAGE_SCHEMA",
    "open_tracking",
    "open_tracking_jsonl",
    "write_replay_sidecars",
]
