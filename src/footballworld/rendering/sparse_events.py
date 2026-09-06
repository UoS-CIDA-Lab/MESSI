"""Sparse, lossless host encoding for exact fixed-shape frame events."""

from __future__ import annotations

from typing import Any

import numpy as np

from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_POLICY,
    MECHANISM_NONE,
    OUTCOME_NONE,
)
from footballworld.rendering.transfer import to_jsonable

SPARSE_EVENT_ENCODING = "footballworld.occurred-events/2"
SPARSE_ACTION_ENCODING = "footballworld.contact-actions/2"

_GROUP_ORDER = {
    "deliberate_contact": 0,
    "contest": 1,
    "contact": 2,
    "foul": 3,
    "retouch": 4,
    "offside": 5,
    "boundary": 6,
    "goalkeeper_holding": 7,
    "nutmeg": 8,
    "woodwork": 9,
}


def _mask_for(name: str, value: Any) -> np.ndarray:
    """Return rows whose fields cannot be reconstructed as empty sentinels."""

    if name == "deliberate_contact":
        return (
            (np.asarray(value.actor) >= 0)
            | (np.asarray(value.mechanism) != MECHANISM_NONE)
            | (np.asarray(value.outcome) != OUTCOME_NONE)
            | np.asarray(value.kick_applied, dtype=bool)
        )
    if name == "contest":
        return (
            np.asarray(value.selected, dtype=bool)
            | np.asarray(value.occurred, dtype=bool)
            | (~np.asarray(value.override_valid, dtype=bool))
        )
    if name == "goalkeeper_holding":
        return np.asarray(value.opened, dtype=bool) | np.asarray(
            value.expired, dtype=bool
        )
    return np.asarray(value.occurred, dtype=bool)


def _group_rows(name: str, value: Any) -> list[dict[str, Any]]:
    mask = _mask_for(name, value)
    if mask.ndim not in (1, 2):
        raise ValueError(f"{name} occurrence mask must have one or two axes")
    arrays = {field: np.asarray(getattr(value, field)) for field in value._fields}
    for field, array in arrays.items():
        if array.shape[: mask.ndim] != mask.shape:
            raise ValueError(f"{name}.{field} does not share its occurrence prefix")
    rows = []
    for raw_index in np.argwhere(mask):
        index = tuple(int(part) for part in raw_index)
        row = {
            "type": name,
            "substep": index[0],
            "fields": {
                field: to_jsonable(array[index]) for field, array in arrays.items()
            },
        }
        if len(index) == 2:
            row["slot"] = index[1]
        rows.append(row)
    return rows


def sparse_frame_events(frame_events: Any) -> dict[str, Any] | None:
    """Encode only occurred or otherwise meaningful exact event rows.

    Omitted fixed slots are the canonical empty values represented by the
    decoder template. Every retained row carries every source field, so this
    changes storage only and never reclassifies an event.
    """

    if frame_events is None:
        return None
    rows: list[dict[str, Any]] = []
    for name in (
        "deliberate_contact",
        "contest",
        "contacts",
        "foul",
        "retouch",
        "offside",
        "boundary",
        "goalkeeper_holding",
        "nutmeg",
    ):
        output_name = "contact" if name == "contacts" else name
        rows.extend(_group_rows(output_name, getattr(frame_events, name)))

    occurred = np.asarray(frame_events.woodwork_occurred, dtype=bool)
    kind = np.asarray(frame_events.woodwork_kind)
    if occurred.shape != kind.shape or occurred.ndim != 2:
        raise ValueError("woodwork event arrays must share [substep, slot]")
    for substep, slot in np.argwhere(occurred):
        rows.append(
            {
                "type": "woodwork",
                "substep": int(substep),
                "slot": int(slot),
                "fields": {
                    "occurred": True,
                    "kind": to_jsonable(kind[substep, slot]),
                },
            }
        )

    rows.sort(
        key=lambda row: (
            row["substep"],
            _GROUP_ORDER[row["type"]],
            row.get("slot", -1),
        )
    )
    return {"encoding": SPARSE_EVENT_ENCODING, "events": rows}


def _realized_contact_mask(frame_events: Any, player_count: int) -> np.ndarray:
    """Return actors whose submitted controls matter to contact auditing."""

    realized = np.zeros(player_count, dtype=bool)
    if frame_events is None:
        return realized
    actors = np.asarray(frame_events.deliberate_contact.actor, dtype=np.int64)
    valid = actors[(actors >= 0) & (actors < player_count)]
    realized[valid] = True
    return realized


def sparse_action_trace(
    action_trace: Any,
    submitted_action: Any = None,
    frame_events: Any = None,
    *,
    include_move: bool = False,
) -> dict[str, Any] | None:
    """Retain categorical exceptions and contact-relevant submitted controls.

    Omitted MOVE rows can contain non-zero movement controls and are therefore
    deliberately unknown. This is a bounded contact-audit trace, not a full
    action-replay representation. Realized contact intent/source remains in
    ``frame_events``; these continuous values are the submitted policy action.
    """

    if action_trace is None:
        return None
    requested = np.asarray(action_trace.requested_intent, dtype=np.int32)
    source = np.asarray(action_trace.intent_source, dtype=np.int32)
    if requested.ndim != 1 or source.shape != requested.shape:
        raise ValueError("action trace intent/source must share one player axis")
    if type(include_move) is not bool:
        raise TypeError("include_move must be bool")
    retain = (
        np.ones(requested.shape, dtype=bool)
        if include_move
        else (
            (requested != INTENT_MOVE)
            | (source != INTENT_SOURCE_POLICY)
            | _realized_contact_mask(frame_events, requested.size)
        )
    )
    controls = None
    if submitted_action is not None:
        submitted_intent = np.asarray(submitted_action.intent, dtype=np.int32)
        move = np.asarray(submitted_action.move, dtype=np.float32)
        force_to_ball = np.asarray(submitted_action.force_to_ball, dtype=np.float32)
        launch = np.asarray(submitted_action.launch, dtype=np.float32)
        spin = np.asarray(submitted_action.spin, dtype=np.float32)
        gaze_center = np.asarray(submitted_action.gaze_center, dtype=np.float32)
        expected_vector = (requested.size, 2)
        if (
            submitted_intent.shape != requested.shape
            or move.shape != expected_vector
            or force_to_ball.shape != expected_vector
            or launch.shape != requested.shape
            or spin.shape != expected_vector
            or gaze_center.shape != requested.shape
        ):
            raise ValueError("submitted action fields do not share one player axis")
        if not np.array_equal(submitted_intent, requested):
            raise ValueError("submitted action intent disagrees with action trace")
        controls = (move, force_to_ball, launch, spin, gaze_center)

    rows = []
    for player in np.flatnonzero(retain):
        row = {
            "player": int(player),
            "intent": int(requested[player]),
            "source": int(source[player]),
        }
        if controls is not None:
            move, force_to_ball, launch, spin, gaze_center = controls
            row.update(
                {
                    "move": to_jsonable(move[player]),
                    "force_to_ball": to_jsonable(force_to_ball[player]),
                    "launch": to_jsonable(launch[player]),
                    "spin": to_jsonable(spin[player]),
                    "gaze_center": to_jsonable(gaze_center[player]),
                }
            )
        rows.append(row)
    return {
        "executed": bool(np.asarray(action_trace.executed)),
        "player_count": int(requested.size),
        "default_intent": INTENT_MOVE,
        "default_source": INTENT_SOURCE_POLICY,
        "rows": rows,
    }


def restore_sparse_action_trace(record: dict[str, Any], template: Any) -> Any:
    """Restore an exact action trace from its categorical default encoding."""

    requested_template = np.asarray(template.requested_intent)
    source_template = np.asarray(template.intent_source)
    if (
        requested_template.ndim != 1
        or source_template.shape != requested_template.shape
    ):
        raise ValueError("action trace template must share one player axis")
    player_count = int(record["player_count"])
    if player_count != requested_template.size:
        raise ValueError("sparse action player_count does not match template")
    requested = np.full(
        requested_template.shape,
        int(record["default_intent"]),
        dtype=requested_template.dtype,
    )
    source = np.full(
        source_template.shape,
        int(record["default_source"]),
        dtype=source_template.dtype,
    )
    seen: set[int] = set()
    for row in record["rows"]:
        player = int(row["player"])
        if not 0 <= player < player_count:
            raise ValueError("sparse action player index is outside the template")
        if player in seen:
            raise ValueError("sparse action player index occurs more than once")
        seen.add(player)
        requested[player] = row["intent"]
        source[player] = row["source"]
    executed_template = np.asarray(template.executed)
    if executed_template.shape != ():
        raise ValueError("action trace executed template must be scalar")
    return type(template)(
        requested_intent=requested,
        intent_source=source,
        executed=np.asarray(record["executed"], dtype=executed_template.dtype),
    )


def restore_sparse_frame_events(record: dict[str, Any], template: Any) -> Any:
    """Restore the exact event tree using one canonical empty-shape template."""

    if record.get("encoding") != SPARSE_EVENT_ENCODING:
        raise ValueError("unsupported sparse frame-event encoding")
    groups = {
        "deliberate_contact": "deliberate_contact",
        "contest": "contest",
        "contact": "contacts",
        "foul": "foul",
        "retouch": "retouch",
        "offside": "offside",
        "boundary": "boundary",
        "goalkeeper_holding": "goalkeeper_holding",
        "nutmeg": "nutmeg",
    }
    mutable = {
        name: {
            field: np.array(getattr(getattr(template, attribute), field), copy=True)
            for field in getattr(template, attribute)._fields
        }
        for name, attribute in groups.items()
    }
    woodwork_occurred = np.array(template.woodwork_occurred, copy=True)
    woodwork_kind = np.array(template.woodwork_kind, copy=True)

    for row in record["events"]:
        name = row["type"]
        index = (int(row["substep"]),)
        if "slot" in row:
            index += (int(row["slot"]),)
        if name == "woodwork":
            if set(row.get("fields", {})) != {"occurred", "kind"}:
                raise ValueError("woodwork sparse fields do not match encoding")
            woodwork_occurred[index] = row["fields"]["occurred"]
            woodwork_kind[index] = row["fields"]["kind"]
            continue
        if name not in mutable:
            raise ValueError(f"unknown sparse event type: {name!r}")
        fields = row.get("fields", {})
        if set(fields) != set(mutable[name]):
            raise ValueError(f"{name} sparse fields do not match encoding")
        for field, value in fields.items():
            mutable[name][field][index] = value

    rebuilt = {}
    for name, attribute in groups.items():
        original = getattr(template, attribute)
        rebuilt[attribute] = type(original)(
            *(mutable[name][field] for field in original._fields)
        )
    return type(template)(
        deliberate_contact=rebuilt["deliberate_contact"],
        contest=rebuilt["contest"],
        contacts=rebuilt["contacts"],
        foul=rebuilt["foul"],
        retouch=rebuilt["retouch"],
        offside=rebuilt["offside"],
        boundary=rebuilt["boundary"],
        goalkeeper_holding=rebuilt["goalkeeper_holding"],
        nutmeg=rebuilt["nutmeg"],
        woodwork_occurred=woodwork_occurred,
        woodwork_kind=woodwork_kind,
    )


__all__ = [
    "SPARSE_ACTION_ENCODING",
    "SPARSE_EVENT_ENCODING",
    "restore_sparse_action_trace",
    "restore_sparse_frame_events",
    "sparse_action_trace",
    "sparse_frame_events",
]
