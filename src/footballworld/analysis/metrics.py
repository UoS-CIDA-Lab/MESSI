"""Versioned metrics derived from replay source facts."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Any

import numpy as np

from footballworld.analysis.dataset import MatchDataset

REPORT_SCHEMA = "footballworld.match-report/9"
METRICS_VERSION = "footballworld.match-metrics/12"
INTENT_PASS = 2
INTENT_SHOT = 3
INTENT_CLEAR = 4
INTENT_CHALLENGE = 5
INTENT_CONTROL = 1
RESTART_NONE = 0
RESTART_FREE_KICK = 5
RESTART_PENALTY = 6
RESTART_NAMES = {
    1: "kickoff",
    2: "throw_in",
    3: "goal_kick",
    4: "corner",
    5: "free_kick",
    6: "penalty",
    7: "offside",
    8: "goalkeeper_hold",
}
ANALYSIS_WINDOW_S = 30.0
SPEED_OVERFLOW_BIN_MPS = 12
BALL_TERRITORY_BINS = 6
BALL_DENSITY_TARGET_CELL_M = 2.0
BALL_DENSITY_WINDOW_S = 30.0
# Host-only diagnostic priors. These identify simulation pathologies for review;
# they are not presented as measured football constants.
POLICY_AUDIT_STATIONARY_BALL_SPEED_MPS = 0.05
POLICY_AUDIT_STATIONARY_LOOSE_WARN_S = 5.0
POLICY_AUDIT_LOOSE_BALL_WARN_S = 15.0
POLICY_AUDIT_DENSITY_WARN_S = 60.0
POLICY_AUDIT_DENSITY_WARN_SHARE = 0.02
POLICY_AUDIT_EVENT_REPEAT_WARN_S = 30.0
POLICY_AUDIT_DISMISSAL_WARN_COUNT = 3
POLICY_AUDIT_PASS_WINDOW_S = 15.0 * 60.0
POLICY_AUDIT_PASS_WINDOW_MIN_ATTEMPTS = 10
POLICY_AUDIT_PASS_COMPLETION_TREND_DROP = 0.05
POLICY_AUDIT_PASS_ACTIVITY_MAX_ATTEMPTS = 5
POLICY_AUDIT_PASS_ACTIVITY_NEIGHBOR_MIN_ATTEMPTS = 20
POLICY_AUDIT_ATTACKING_THIRD_MIN_PASSES = 20
POLICY_AUDIT_ATTACKING_THIRD_BACKWARD_SHARE = 0.65
PLAYER_POSITION_WINDOW_S = 15.0
PLAYER_POSITION_TARGET_CELL_M = 4.0
SPACE_OCCUPANCY_WINDOW_S = 30.0
SPACE_OCCUPANCY_TARGET_CELL_M = 4.0
SHOT_CONTEXT_RESTART_MAX_S = 10.0
SHOT_CONTEXT_QUICK_REGAIN_MAX_S = 5.0
SHOT_CONTEXT_COUNTERATTACK_MAX_S = 12.0
SHOT_CONTEXT_COUNTERATTACK_MIN_PROGRESS_M = 20.0
SHOT_CONTEXT_SUSTAINED_MIN_S = 12.0
SHOT_ROUTE_MAX_PRIOR_NODES = 7
SHOT_ROUTE_MAX_TRACK_PATH_POINTS = 2048
SHOT_ROUTE_FALLBACK_LOOKBACK_S = 15.0
CROSS_SIGNATURE_SIDE_SPIN = 0.18
CROSS_SIGNATURE_BACK_SPIN = 0.35
CROSS_SIGNATURE_ATOL = 1.0e-6
DEFENSIVE_LINE_BREAK_MIN_PROGRESS_M = 5.0
DEFENSIVE_LINE_BREAK_MARGIN_M = 0.5
SHOT_CONTEXT_CATEGORIES = (
    ("penalty_kick", "Penalty kick"),
    ("free_kick", "Free kick"),
    ("restart_attack", "Restart attack"),
    ("counterattack", "Counterattack"),
    ("quick_after_regain", "Quick after regain"),
    ("sustained_buildup", "Sustained buildup"),
    ("open_play_buildup", "Open-play buildup"),
    ("unclassified", "Unclassified"),
)
INTENT_EVENT_NAMES = {
    INTENT_CONTROL: "control_touch",
    INTENT_PASS: "pass",
    INTENT_SHOT: "shot",
    INTENT_CLEAR: "clear",
    INTENT_CHALLENGE: "challenge",
}
LAW11_DELIBERATE_SAVE_NO_RESET = 3
LAW11_DEFLECTION_NO_RESET = 2
BALL_EVENT_GOAL = 1
BALL_EVENT_NAMES = {
    1: "goal",
    2: "corner",
    3: "goal_kick",
    4: "throw_in",
}


def _metric(
    metric_id: str,
    value: Any,
    unit: str,
    definition: str,
    *,
    quality: str = "derived",
) -> dict[str, Any]:
    return {
        "id": metric_id,
        "version": 1,
        "value": value,
        "unit": unit,
        "definition": definition,
        "quality": quality,
    }


def _clock_label(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _attack_normalized_position(
    position: tuple[float, float] | list[float] | np.ndarray,
    direction: float,
) -> tuple[float, float]:
    """Rotate a world position into a team's positive-x attacking frame.

    Changing ends is a 180-degree rotation, not a reflection across the y axis.
    Both coordinates therefore change sign for a team attacking toward negative x.
    """

    return direction * float(position[0]), direction * float(position[1])


def _tracking_lookback_sample(
    rows: Iterable[tuple[int, int, float, float, float]],
    *,
    period: int,
    shot_clock_s: float,
    direction: float,
) -> dict[str, Any] | None:
    """Build a positioned pre-shot route solely from retained tracking rows."""

    path: list[list[float]] = []
    start_clock_s: float | None = None
    for row_period, tick, clock_s, x_m, y_m in rows:
        if (
            row_period != period
            or clock_s < shot_clock_s - SHOT_ROUTE_FALLBACK_LOOKBACK_S
        ):
            continue
        path_x_m, path_y_m = _attack_normalized_position((x_m, y_m), direction)
        if not all(math.isfinite(value) for value in (path_x_m, path_y_m)):
            continue
        start_clock_s = clock_s if start_clock_s is None else start_clock_s
        path.append([float(tick), round(path_x_m, 3), round(path_y_m, 3)])
    if not path or start_clock_s is None:
        return None
    return {
        "origin": "tracking_lookback",
        "route_basis": "bounded_tracking_lookback",
        "start_clock_s": start_clock_s,
        "start_control_tick": int(path[0][0]),
        "elapsed_s": max(0.0, shot_clock_s - start_clock_s),
        "start_x_m": float(path[0][1]),
        "tracking_path": path,
    }


def _action_direction_unit(row: dict[str, Any]) -> list[float] | None:
    """Decode a retained team-relative force vector for report arrows."""

    raw = row.get("force_to_ball")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        return None
    magnitude = float(np.linalg.norm(vector))
    if not np.isfinite(magnitude) or magnitude <= 1.0e-9:
        return None
    unit = vector / magnitude
    return [round(float(unit[0]), 6), round(float(unit[1]), 6)]


def _action_spin(row: dict[str, Any]) -> list[float] | None:
    """Decode the retained submitted spin without inferring a kick type."""

    raw = row.get("spin")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        spin = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if spin.shape != (2,) or not np.all(np.isfinite(spin)):
        return None
    return [float(spin[0]), float(spin[1])]


def _is_rule_policy_cross_signature(spin: list[float] | None) -> bool:
    """Match the shipped rule policy's cross controls, not a provider label."""

    if spin is None:
        return False
    return bool(
        np.isclose(
            abs(spin[0]),
            CROSS_SIGNATURE_SIDE_SPIN,
            rtol=0.0,
            atol=CROSS_SIGNATURE_ATOL,
        )
        and np.isclose(
            spin[1],
            CROSS_SIGNATURE_BACK_SPIN,
            rtol=0.0,
            atol=CROSS_SIGNATURE_ATOL,
        )
    )


def _direction_family(direction_unit: list[float] | None) -> str:
    """Classify an attack-normalized applied action direction."""

    if direction_unit is None:
        return "unknown"
    cosine = float(direction_unit[0])
    if not np.isfinite(cosine):
        return "unknown"
    return (
        "backward" if cosine <= -0.34 else ("lateral" if cosine < 0.34 else "forward")
    )


def _source_context(normalized_x: float, half_length: float) -> tuple[str, str]:
    """Return source half and equal-length pitch third in the attack frame."""

    source_half = "own_half" if normalized_x < 0.0 else "opposition_half"
    third_edge = max(float(half_length), 1.0e-9) / 3.0
    source_third = (
        "defensive_third"
        if normalized_x < -third_edge
        else ("middle_third" if normalized_x < third_edge else "attacking_third")
    )
    return source_half, source_third


def _timeline_clock(
    clock: float,
    public_clock: tuple[int, float, float] | None,
) -> dict[str, Any]:
    if public_clock is None:
        return {"clock_s": clock, "clock_label": _clock_label(clock)}
    period, display_clock_s, added_time_s = public_clock
    label = _clock_label(display_clock_s)
    if added_time_s > 0.0:
        nominal_s = 45.0 * 60.0 * period
        label = f"{_clock_label(nominal_s)} +{_clock_label(added_time_s)}"
    return {
        "clock_s": clock,
        "clock_label": label,
        "period": period,
        "display_clock_s": display_clock_s,
        "added_time_s": added_time_s,
    }


def _next_boundary(
    boundaries: list[tuple[int, int, float]],
    time_key: tuple[int, int, float],
) -> tuple[int, int, float] | None:
    index = bisect_left(boundaries, time_key)
    return boundaries[index] if index < len(boundaries) else None


def _segment_intersects_box(
    start: tuple[float, float],
    end: tuple[float, float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> bool:
    """Return whether a closed 2D segment intersects an axis-aligned box."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for p, q in (
        (-dx, start[0] - x_bounds[0]),
        (dx, x_bounds[1] - start[0]),
        (-dy, start[1] - y_bounds[0]),
        (dy, y_bounds[1] - start[1]),
    ):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def _event_label(kind: str, fields: dict[str, Any], team: Any) -> str:
    if kind == "boundary" and fields.get("scoring_team") in (0, 1):
        return f"Goal, Team {fields['scoring_team']}"
    if kind == "foul":
        return f"Foul{'' if team is None else f', Team {team}'}"
    return kind.replace("_", " ").title()


def _initial_identities(dataset: MatchDataset) -> dict[int, dict[str, int]]:
    identities: dict[int, dict[str, int]] = {}
    for row in dataset.metadata.get("roster", []):
        if isinstance(row, dict) and isinstance(row.get("slot"), int):
            identities[int(row["slot"])] = {
                "player_id": int(row.get("player_id", -1)),
                "slot_generation": int(row.get("slot_generation", -1)),
                "team": int(row.get("team", -1)),
            }
    return identities


def _event_header(dataset: MatchDataset) -> dict[str, Any]:
    header = getattr(dataset, "event_header", None)
    return header if isinstance(header, dict) else dataset.events


def _iter_event_frames(dataset: MatchDataset) -> Iterator[dict[str, Any]]:
    opener = getattr(dataset, "open_events", None)
    if callable(opener):
        with opener() as stream:
            yield from stream
        return
    yield from dataset.events["frames"]


def _integer_event_field(fields: dict[str, Any], name: str) -> int:
    value = fields.get(name)
    return (
        int(value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        else -1
    )


def _collect_policy_contact_facts(
    events: dict[str, Any],
    initial_identities: dict[int, dict[str, int]],
    frames: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collect exact open-play service contacts without inventing outcomes."""

    identities = {slot: dict(value) for slot, value in initial_identities.items()}
    contacts: list[dict[str, Any]] = []
    boundaries: list[tuple[int, int, float]] = []
    pass_sources: list[dict[str, Any]] = []
    applied_sources: list[dict[str, Any]] = []
    control_sources: list[dict[str, Any]] = []
    intent_sources: list[dict[str, Any]] = []
    shot_sources: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    woodwork_events: list[dict[str, Any]] = []
    shot_contacts = 0
    discontinuity_ticks: set[int] = set()
    timeline_ticks: set[int] = set()

    for frame in events["frames"] if frames is None else frames:
        tick = int(frame["control_tick"])
        formations = frame.get("formations", []) or []
        if any(
            isinstance(event, dict) and event.get("applied") is True
            for event in formations
        ):
            discontinuity_ticks.add(tick)
        exact_events = (
            (frame.get("frame_events") or {}).get("events", [])
            if isinstance(frame.get("frame_events"), dict)
            else []
        )
        if (
            formations
            or frame.get("substitutions")
            or frame.get("set_piece_taker_changes")
            or any(
                isinstance(event, dict) and event.get("type") in {"foul", "boundary"}
                for event in exact_events
            )
        ):
            timeline_ticks.add(tick)
        overrides = {
            int(item["slot"]): item
            for item in frame.get("pre_management_identity", [])
            if isinstance(item, dict) and "slot" in item
        }

        action = frame.get("action")
        pass_actions: dict[int, dict[str, Any]] = {}
        if isinstance(action, dict):
            pass_actions = {
                int(row["player"]): {
                    "intended_receiver_player_id": row.get(
                        "intended_receiver_player_id"
                    ),
                    "applied_direction_unit": _action_direction_unit(row),
                    "submitted_spin": _action_spin(row),
                }
                for row in action.get("rows", [])
                if isinstance(row, dict)
                and int(row.get("intent", -1)) == INTENT_PASS
                and isinstance(row.get("player"), int)
            }

        exact = frame.get("frame_events")
        rows = exact.get("events", []) if isinstance(exact, dict) else []
        local_contacts: list[dict[str, Any]] = []
        local_sources: list[dict[str, Any]] = []
        for order, event in enumerate(rows):
            kind = str(event.get("type", ""))
            fields = event.get("fields", {})
            if not isinstance(fields, dict):
                continue
            substep = int(event.get("substep", 0))
            fraction = float(fields.get("time_fraction", 0.0))
            time_key = (tick, substep, fraction)
            if kind == "contact" and fields.get("occurred") is True:
                actor = int(fields.get("actor", -1))
                raw_position = fields.get("position")
                position = (
                    tuple(float(value) for value in raw_position[:3])
                    if isinstance(raw_position, list) and len(raw_position) >= 2
                    else None
                )
                identity = overrides.get(actor, identities.get(actor, {}))
                row = {
                    "time_key": time_key,
                    "order": order,
                    "slot": int(event.get("slot", -1)),
                    "tick": tick,
                    "substep": substep,
                    "actor": actor,
                    "team": identity.get("team"),
                    "player_id": identity.get("player_id"),
                    "slot_generation": identity.get("slot_generation", 0),
                    "mechanism": fields.get("mechanism"),
                    "law11_effect": fields.get("law11_effect"),
                    "position": position,
                }
                contacts.append(row)
                local_contacts.append(row)
            elif kind == "boundary" and fields.get("occurred") is True:
                boundaries.append(time_key)
                boundary_events.append(
                    {
                        "time_key": time_key,
                        "tick": tick,
                        "substep": substep,
                        "kind": int(fields.get("kind", 0)),
                        "team": fields.get("team"),
                        "scoring_team": fields.get("scoring_team"),
                        "position": fields.get("position"),
                    }
                )
            elif kind == "woodwork" and fields.get("occurred") is True:
                woodwork_key = (
                    tick,
                    substep,
                    float(fields.get("time_fraction", 1.0)),
                )
                woodwork_events.append(
                    {
                        "time_key": woodwork_key,
                        "tick": tick,
                        "substep": substep,
                        "kind": fields.get("kind"),
                    }
                )
            elif kind == "deliberate_contact":
                intent = int(fields.get("intent", -1))
                actor = int(fields.get("actor", -1))
                identity = overrides.get(actor, identities.get(actor, {}))
                kick_applied = fields.get("kick_applied") is True
                if intent == INTENT_CONTROL or (
                    kick_applied
                    and intent
                    in (INTENT_PASS, INTENT_SHOT, INTENT_CLEAR, INTENT_CHALLENGE)
                ):
                    pass_action = pass_actions.get(actor, {})
                    source = {
                        "tick": tick,
                        "substep": substep,
                        "order": order,
                        "actor": actor,
                        "team": identity.get("team"),
                        "player_id": identity.get("player_id"),
                        "slot_generation": identity.get("slot_generation", 0),
                        "intent": intent,
                        "intended_receiver_player_id": pass_action.get(
                            "intended_receiver_player_id"
                        ),
                        "applied_direction_unit": pass_action.get(
                            "applied_direction_unit"
                        ),
                        "submitted_spin": pass_action.get("submitted_spin"),
                        "mechanism": fields.get("mechanism"),
                        "law11_effect": fields.get("law11_effect"),
                        "restart_kind": int(fields.get("restart_kind", -1)),
                        "kick_applied": kick_applied,
                        "contact": None,
                    }
                    local_sources.append(source)
                    if intent == INTENT_CONTROL:
                        control_sources.append(source)
                        continue
                    applied_sources.append(source)
                    if intent == INTENT_SHOT:
                        shot_sources.append(source)
                        timeline_ticks.add(tick)
                    if source["restart_kind"] == RESTART_NONE:
                        intent_sources.append(source)
                        if intent == INTENT_PASS:
                            pass_sources.append(source)
                        elif intent == INTENT_SHOT:
                            shot_contacts += 1

        for source in local_sources:
            candidates = [
                contact
                for contact in local_contacts
                if contact["substep"] == source["substep"]
                and contact["actor"] == source["actor"]
                and contact["mechanism"] == source["mechanism"]
                and contact["law11_effect"] == source["law11_effect"]
            ]
            if candidates:
                source["contact"] = min(
                    candidates,
                    key=lambda contact: (contact["time_key"], contact["slot"]),
                )

        for substitution in frame.get("substitutions", []) or []:
            slot = int(substitution["slot"])
            identities[slot] = {
                "player_id": int(substitution["player_in"]),
                "slot_generation": int(substitution["slot_generation"]),
                "team": int(substitution["team"]),
            }

    contacts.sort(
        key=lambda contact: (
            contact["time_key"],
            contact["order"],
            contact["slot"],
        )
    )
    boundaries.sort()
    boundary_events.sort(key=lambda event: event["time_key"])
    woodwork_events.sort(key=lambda event: event["time_key"])
    return {
        "contacts": contacts,
        "boundaries": boundaries,
        "pass_sources": pass_sources,
        "applied_sources": applied_sources,
        "control_sources": control_sources,
        "intent_sources": intent_sources,
        "shot_sources": shot_sources,
        "boundary_events": boundary_events,
        "woodwork_events": woodwork_events,
        "shot_contacts": shot_contacts,
        "attack_direction_ticks": {
            int(source["tick"]) for source in chain(applied_sources, control_sources)
        },
        "discontinuity_ticks": discontinuity_ticks,
        "timeline_ticks": timeline_ticks,
    }


def _policy_alignment_metrics(
    facts: dict[str, Any],
    attack_direction_by_tick: dict[int, tuple[float, float]],
    live_s: float,
    half_length: float,
) -> dict[str, Any]:
    """Build DFL-neighbor policy metrics from exact FootballWorld facts."""

    contacts = facts["contacts"]
    boundaries = facts["boundaries"]
    contact_rank = {id(contact): index for index, contact in enumerate(contacts)}
    geometry = {
        name: {"attempts": 0, "completed": 0}
        for name in ("backward", "lateral", "forward", "unknown")
    }
    applied_geometry = {
        name: {
            "attempts": 0,
            "resolved": 0,
            "same_team": 0,
            "intended_receiver": 0,
        }
        for name in ("backward", "lateral", "forward", "unknown")
    }
    source_context = {
        name: {
            "attempts": 0,
            "resolved": 0,
            "same_team": 0,
            "intended_receiver": 0,
            "applied_direction": {
                family: {"attempts": 0, "resolved": 0, "same_team": 0}
                for family in ("backward", "lateral", "forward", "unknown")
            },
        }
        for name in (
            "own_half",
            "opposition_half",
            "defensive_third",
            "middle_third",
            "attacking_third",
        )
    }
    matched_sources = 0
    observed_next_contacts = 0
    same_team_receipts = 0
    ended_by_boundary = 0
    right_censored = 0
    skipped_same_actor = 0

    for source in facts["pass_sources"]:
        contact = source["contact"]
        if contact is None:
            continue
        matched_sources += 1
        applied_family = _direction_family(source.get("applied_direction_unit"))
        applied_geometry[applied_family]["attempts"] += 1
        source_context_names: tuple[str, ...] = ()
        source_position = contact["position"]
        team = source["team"]
        directions = attack_direction_by_tick.get(int(source["tick"]))
        if source_position is not None and team in (0, 1) and directions is not None:
            normalized_source = _attack_normalized_position(
                source_position, float(directions[int(team)])
            )
            source_half, source_third = _source_context(
                normalized_source[0], half_length
            )
            source_context_names = (source_half, source_third)
            for context_name in source_context_names:
                context_row = source_context[context_name]
                context_row["attempts"] += 1
                context_row["applied_direction"][applied_family]["attempts"] += 1
        rank = contact_rank[id(contact)]
        next_rank = rank + 1
        while (
            next_rank < len(contacts)
            and contacts[next_rank]["actor"] == source["actor"]
        ):
            skipped_same_actor += 1
            next_rank += 1
        next_contact = contacts[next_rank] if next_rank < len(contacts) else None
        next_boundary = _next_boundary(boundaries, contact["time_key"])
        if next_boundary is not None and (
            next_contact is None or next_boundary <= next_contact["time_key"]
        ):
            ended_by_boundary += 1
            applied_geometry[applied_family]["resolved"] += 1
            for context_name in source_context_names:
                context_row = source_context[context_name]
                context_row["resolved"] += 1
                context_row["applied_direction"][applied_family]["resolved"] += 1
            continue
        if next_contact is None:
            right_censored += 1
            continue

        receiver_position = next_contact["position"]
        applied_geometry[applied_family]["resolved"] += 1
        for context_name in source_context_names:
            context_row = source_context[context_name]
            context_row["resolved"] += 1
            context_row["applied_direction"][applied_family]["resolved"] += 1
        same_team = team in (0, 1) and next_contact["team"] == team
        intended_receiver = source.get("intended_receiver_player_id")
        intended_match = (
            same_team
            and isinstance(intended_receiver, int)
            and next_contact.get("player_id") == intended_receiver
        )
        if same_team:
            applied_geometry[applied_family]["same_team"] += 1
        if intended_match:
            applied_geometry[applied_family]["intended_receiver"] += 1
        for context_name in source_context_names:
            context_row = source_context[context_name]
            if same_team:
                context_row["same_team"] += 1
                context_row["applied_direction"][applied_family]["same_team"] += 1
            if intended_match:
                context_row["intended_receiver"] += 1
        if (
            source_position is None
            or receiver_position is None
            or team not in (0, 1)
            or next_contact["team"] not in (0, 1)
            or directions is None
        ):
            geometry["unknown"]["attempts"] += 1
            continue
        delta_x = receiver_position[0] - source_position[0]
        delta_y = receiver_position[1] - source_position[1]
        distance = float(np.hypot(delta_x, delta_y))
        if not np.isfinite(distance) or distance <= 1.0e-9:
            family = "unknown"
        else:
            cosine = float(directions[int(team)]) * delta_x / distance
            family = (
                "backward"
                if cosine <= -0.34
                else ("lateral" if cosine < 0.34 else "forward")
            )
        geometry[family]["attempts"] += 1
        observed_next_contacts += 1
        if int(next_contact["team"]) == int(team):
            geometry[family]["completed"] += 1
            same_team_receipts += 1

    known_attempts = sum(
        geometry[name]["attempts"] for name in ("backward", "lateral", "forward")
    )
    known_applied_attempts = sum(
        applied_geometry[name]["attempts"]
        for name in ("backward", "lateral", "forward")
    )
    applied_resolved = sum(row["resolved"] for row in applied_geometry.values())
    applied_same_team = sum(row["same_team"] for row in applied_geometry.values())
    applied_intended = sum(
        row["intended_receiver"] for row in applied_geometry.values()
    )
    team_live_minutes = 2.0 * live_s / 60.0
    pass_contacts = len(facts["pass_sources"])
    shot_contacts = int(facts["shot_contacts"])

    def divide(numerator: float, denominator: float) -> float | None:
        return None if denominator <= 0.0 else round(numerator / denominator, 6)

    metrics = {
        "open_play.pass_contacts_per_team_live_minute": {
            "value": divide(pass_contacts, team_live_minutes),
            "unit": "contacts/team-live-minute",
            "quality": "proxy",
        },
        "open_play.shot_contacts_per_team_live_minute": {
            "value": divide(shot_contacts, team_live_minutes),
            "unit": "contacts/team-live-minute",
            "quality": "proxy",
        },
        "open_play.pass_geometry.backward_share": {
            "value": divide(geometry["backward"]["attempts"], known_attempts),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.pass_geometry.lateral_share": {
            "value": divide(geometry["lateral"]["attempts"], known_attempts),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.pass_geometry.forward_share": {
            "value": divide(geometry["forward"]["attempts"], known_attempts),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.applied_pass_direction.backward_share": {
            "value": divide(
                applied_geometry["backward"]["attempts"], known_applied_attempts
            ),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.applied_pass_direction.lateral_share": {
            "value": divide(
                applied_geometry["lateral"]["attempts"], known_applied_attempts
            ),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.applied_pass_direction.forward_share": {
            "value": divide(
                applied_geometry["forward"]["attempts"], known_applied_attempts
            ),
            "unit": "ratio",
            "quality": "closest_comparable",
        },
        "open_play.forward_pass_contacts_per_team_live_minute": {
            "value": divide(geometry["forward"]["attempts"], team_live_minutes),
            "unit": "contacts/team-live-minute",
            "quality": "proxy",
        },
        "open_play.pass_receipt.forward": {
            "value": divide(
                geometry["forward"]["completed"],
                geometry["forward"]["attempts"],
            ),
            "unit": "ratio",
            "quality": "diagnostic_neighbor",
        },
        "open_play.pass_receipt.overall": {
            "value": divide(same_team_receipts, observed_next_contacts),
            "unit": "ratio",
            "quality": "diagnostic_neighbor",
        },
        "open_play.applied_pass_receipt.overall": {
            "value": divide(applied_same_team, applied_resolved),
            "unit": "ratio",
            "quality": "diagnostic_neighbor",
        },
        "open_play.intended_receiver_receipt.overall": {
            "value": divide(applied_intended, applied_resolved),
            "unit": "ratio",
            "quality": "diagnostic_neighbor",
        },
    }
    for context_name in ("own_half", "opposition_half"):
        context_row = source_context[context_name]
        metrics[f"open_play.applied_pass_receipt.{context_name}"] = {
            "value": divide(context_row["same_team"], context_row["resolved"]),
            "unit": "ratio",
            "quality": "closest_comparable",
        }
        metrics[f"open_play.intended_receiver_receipt.{context_name}"] = {
            "value": divide(context_row["intended_receiver"], context_row["resolved"]),
            "unit": "ratio",
            "quality": "diagnostic_neighbor",
        }
        known_context_attempts = sum(
            context_row["applied_direction"][family]["attempts"]
            for family in ("backward", "lateral", "forward")
        )
        for family in ("backward", "lateral", "forward"):
            metrics[
                f"open_play.applied_pass_direction.{context_name}.{family}_share"
            ] = {
                "value": divide(
                    context_row["applied_direction"][family]["attempts"],
                    known_context_attempts,
                ),
                "unit": "ratio",
                "quality": "closest_comparable",
            }
    return {
        "schema": "footballworld.policy-alignment-metrics/2",
        "denominator": {
            "live_ball_s": round(live_s, 3),
            "team_live_minutes": round(team_live_minutes, 6),
            "definition": "Two teams multiplied by live tracking interval minutes.",
        },
        "open_play_counts": {
            "pass_contacts": pass_contacts,
            "shot_contacts": shot_contacts,
        },
        "pass_sequence": {
            "matched_source_contacts": matched_sources,
            "observed_next_distinct_actor_contacts": observed_next_contacts,
            "same_team_next_contact": same_team_receipts,
            "same_actor_contacts_skipped": skipped_same_actor,
            "ended_by_boundary": ended_by_boundary,
            "right_censored_at_capture_end": right_censored,
            "geometry": geometry,
            "applied_kick_geometry": applied_geometry,
            "source_context": source_context,
        },
        "metrics": metrics,
        "cautions": [
            "FootballWorld live time is not identical to provider alive time.",
            "Same-team next contact is a receipt proxy, not provider pass completion.",
            "Applied direction comes from the submitted force vector only when the exact deliberate-contact event confirms that the kick was applied.",
            "A single match or partial capture is descriptive and not a policy calibration verdict.",
        ],
    }


def _pass_map_rows(
    facts: dict[str, Any],
    attack_direction_by_tick: dict[int, tuple[float, float]],
    player_positions_by_tick: dict[int, dict[int, tuple[float, float]]],
    player_snapshots_by_tick: dict[int, list[dict[str, Any]]],
    half_length: float,
) -> list[dict[str, Any]]:
    """Describe realized open-play pass contacts without claiming completion."""

    contacts = facts["contacts"]
    boundaries = facts["boundaries"]
    contact_rank = {id(contact): index for index, contact in enumerate(contacts)}
    rows: list[dict[str, Any]] = []
    for source in facts["pass_sources"]:
        contact = source["contact"]
        team = source["team"]
        directions = attack_direction_by_tick.get(int(source["tick"]))
        if contact is None or contact["position"] is None or team not in (0, 1):
            continue
        if directions is None:
            continue
        direction = float(directions[int(team)])
        start = contact["position"]
        if not np.isfinite(start[0]) or not np.isfinite(start[1]):
            continue

        rank = contact_rank[id(contact)]
        next_rank = rank + 1
        while (
            next_rank < len(contacts)
            and contacts[next_rank]["actor"] == source["actor"]
        ):
            next_rank += 1
        next_contact = contacts[next_rank] if next_rank < len(contacts) else None
        next_boundary = _next_boundary(boundaries, contact["time_key"])
        if next_boundary is not None and (
            next_contact is None or next_boundary <= next_contact["time_key"]
        ):
            outcome = "ended_by_boundary"
            next_contact = None
        elif next_contact is None:
            outcome = "right_censored"
        elif next_contact["team"] == team:
            outcome = "same_team_next_contact"
        else:
            outcome = "opponent_next_contact"

        end = None
        family = "unknown"
        distance_m = None
        if next_contact is not None and next_contact["position"] is not None:
            raw_end = next_contact["position"]
            if np.isfinite(raw_end[0]) and np.isfinite(raw_end[1]):
                normalized_end = _attack_normalized_position(raw_end, direction)
                end = [
                    round(normalized_end[0], 3),
                    round(normalized_end[1], 3),
                ]
                delta_x = float(raw_end[0]) - float(start[0])
                delta_y = float(raw_end[1]) - float(start[1])
                distance = float(np.hypot(delta_x, delta_y))
                if np.isfinite(distance) and distance > 1.0e-9:
                    cosine = direction * delta_x / distance
                    family = (
                        "backward"
                        if cosine <= -0.34
                        else ("lateral" if cosine < 0.34 else "forward")
                    )
                    distance_m = round(distance, 3)

        intended_receiver = source.get("intended_receiver_player_id")
        intended_position = None
        if isinstance(intended_receiver, int):
            raw_target = player_positions_by_tick.get(int(source["tick"]), {}).get(
                intended_receiver
            )
            if raw_target is not None:
                normalized_target = _attack_normalized_position(raw_target, direction)
                intended_position = [
                    round(normalized_target[0], 3),
                    round(normalized_target[1], 3),
                ]
        actual_receiver = (
            None if next_contact is None else next_contact.get("player_id")
        )
        intended_receiver_match = (
            None
            if not isinstance(intended_receiver, int) or actual_receiver is None
            else int(actual_receiver) == intended_receiver
        )
        normalized_start = _attack_normalized_position(start, direction)
        source_half, source_third = _source_context(normalized_start[0], half_length)
        applied_direction_unit = source.get("applied_direction_unit")
        cross_signature = _is_rule_policy_cross_signature(source.get("submitted_spin"))
        opponent_x = sorted(
            (
                direction * float(player["position"][0])
                for player in player_snapshots_by_tick.get(int(source["tick"]), [])
                if player.get("team") == 1 - int(team)
                and isinstance(player.get("position"), tuple)
                and len(player["position"]) >= 2
                and np.isfinite(player["position"][0])
            ),
            reverse=True,
        )
        defensive_line_x_m = opponent_x[1] if len(opponent_x) >= 2 else None
        progress_m = None if end is None else float(end[0]) - float(normalized_start[0])
        defensive_line_break = bool(
            defensive_line_x_m is not None
            and end is not None
            and progress_m is not None
            and progress_m >= DEFENSIVE_LINE_BREAK_MIN_PROGRESS_M
            and normalized_start[0] <= defensive_line_x_m
            and float(end[0]) >= defensive_line_x_m + DEFENSIVE_LINE_BREAK_MARGIN_M
        )
        rows.append(
            {
                "control_tick": int(source["tick"]),
                "team": int(team),
                "passer_player_id": source.get("player_id"),
                "passer_slot_generation": source.get("slot_generation", 0),
                "intended_receiver_player_id": intended_receiver,
                "actual_next_player_id": actual_receiver,
                "actual_next_slot_generation": (
                    None
                    if next_contact is None
                    else next_contact.get("slot_generation", 0)
                ),
                "intended_receiver_match": intended_receiver_match,
                "start_m": [
                    round(normalized_start[0], 3),
                    round(normalized_start[1], 3),
                ],
                "intended_target_m": intended_position,
                "actual_end_m": end,
                "distance_m": distance_m,
                "direction_family": family,
                "receipt_geometry_family": family,
                "applied_direction_unit": applied_direction_unit,
                "applied_direction_family": _direction_family(applied_direction_unit),
                "rule_policy_cross_control_signature": cross_signature,
                "defensive_line_x_m_at_source": (
                    None
                    if defensive_line_x_m is None
                    else round(float(defensive_line_x_m), 3)
                ),
                "defensive_line_breaking_pass_proxy": defensive_line_break,
                "source_half": source_half,
                "source_third": source_third,
                "outcome": outcome,
            }
        )
    return rows


def _shot_map_rows(
    facts: dict[str, Any],
    attack_direction_by_tick: dict[int, tuple[float, float]],
    public_clocks: dict[int, tuple[int, float, float]],
    dataset: MatchDataset,
) -> list[dict[str, Any]]:
    """Resolve applied SHOT contacts from exact causal replay facts.

    A goal boundary wins over any earlier woodwork in the same flight. A later
    opponent contact is on target only when physics labelled that contact a
    deliberate save for Law 11. Everything else is an explicitly resolved
    off-target outcome; a capture ending first remains unresolved.
    """

    contacts = facts["contacts"]
    boundary_events = facts["boundary_events"]
    woodwork_events = facts["woodwork_events"]
    contact_rank = {id(contact): index for index, contact in enumerate(contacts)}
    control_fps = float(dataset.metadata["control_fps"])
    rows: list[dict[str, Any]] = []

    for source in facts["shot_sources"]:
        contact = source.get("contact")
        team = source.get("team")
        tick = int(source["tick"])
        directions = attack_direction_by_tick.get(tick)
        if (
            contact is None
            or contact.get("position") is None
            or team not in (0, 1)
            or directions is None
        ):
            continue
        position = contact["position"]
        if not np.isfinite(position[0]) or not np.isfinite(position[1]):
            continue
        source_key = contact["time_key"]
        rank = contact_rank[id(contact)]
        deflections: list[dict[str, Any]] = []
        next_rank = rank + 1
        while (
            next_rank < len(contacts)
            and contacts[next_rank].get("law11_effect") == LAW11_DEFLECTION_NO_RESET
        ):
            deflections.append(contacts[next_rank])
            next_rank += 1
        next_contact = contacts[next_rank] if next_rank < len(contacts) else None
        opponent_deflection = next(
            (
                event
                for event in deflections
                if event.get("team") in (0, 1) and event.get("team") != team
            ),
            None,
        )
        next_boundary = next(
            (
                event
                for event in boundary_events
                if event["time_key"] >= source_key
                and (
                    next_contact is None
                    or event["time_key"] <= next_contact["time_key"]
                )
            ),
            None,
        )
        terminal_key = (
            next_boundary["time_key"]
            if next_boundary is not None
            else (next_contact["time_key"] if next_contact is not None else None)
        )
        woodwork = next(
            (
                event
                for event in woodwork_events
                if event["time_key"] >= source_key
                and (terminal_key is None or event["time_key"] <= terminal_key)
            ),
            None,
        )

        category = "unresolved"
        resolution = "capture_end"
        terminal_tick = None
        if (
            next_boundary is not None
            and next_boundary["kind"] == BALL_EVENT_GOAL
            and next_boundary.get("scoring_team") == team
        ):
            category = "goal"
            resolution = "goal"
            terminal_tick = int(next_boundary["tick"])
        elif next_boundary is not None and next_boundary["kind"] == BALL_EVENT_GOAL:
            category = "off_target"
            resolution = "own_goal"
            terminal_tick = int(next_boundary["tick"])
        elif woodwork is not None:
            category = "off_target"
            resolution = "woodwork"
            terminal_tick = int(woodwork["tick"])
        elif opponent_deflection is not None:
            category = "off_target"
            resolution = "opponent_block"
            terminal_tick = int(opponent_deflection["tick"])
        elif next_contact is not None:
            terminal_tick = int(next_contact["tick"])
            if (
                next_contact.get("team") != team
                and next_contact.get("law11_effect") == LAW11_DELIBERATE_SAVE_NO_RESET
            ):
                category = "on_target"
                resolution = "saved"
            else:
                category = "off_target"
                if next_contact.get("actor") == source.get("actor"):
                    resolution = "shooter_retouch"
                elif next_contact.get("team") == team:
                    resolution = "own_team_rebound"
                elif next_contact.get("team") in (0, 1):
                    resolution = "opponent_block"
                else:
                    resolution = "other_contact"
        elif next_boundary is not None:
            category = "off_target"
            resolution = BALL_EVENT_NAMES.get(
                int(next_boundary["kind"]),
                f"boundary_{int(next_boundary['kind'])}",
            )
            terminal_tick = int(next_boundary["tick"])

        direction = float(directions[int(team)])
        normalized = _attack_normalized_position(position, direction)
        clock = tick / control_fps
        rows.append(
            {
                "control_tick": tick,
                **_timeline_clock(clock, public_clocks.get(tick)),
                "video_time_s": dataset.video_time_s(tick),
                "team": int(team),
                "player_id": source.get("player_id"),
                "slot_generation": source.get("slot_generation", 0),
                "position_m": [
                    round(normalized[0], 3),
                    round(normalized[1], 3),
                ],
                "category": category,
                "resolution": resolution,
                "restart_kind": int(source.get("restart_kind", RESTART_NONE)),
                "terminal_control_tick": terminal_tick,
            }
        )
    return rows


def _classify_shot_context(
    shot: dict[str, Any], sample: dict[str, Any] | None
) -> dict[str, Any]:
    """Classify a pre-shot attacking sequence using declared report heuristics."""

    restart_kind = int(shot.get("restart_kind", RESTART_NONE))
    elapsed_s = None if sample is None else float(sample["elapsed_s"])
    progress_m = None
    origin = "unavailable" if sample is None else str(sample["origin"])
    if sample is not None:
        progress_m = float(shot["position_m"][0]) - float(sample["start_x_m"])

    if restart_kind == RESTART_PENALTY:
        category = "penalty_kick"
    elif restart_kind == RESTART_FREE_KICK:
        category = "free_kick"
    elif restart_kind != RESTART_NONE or (
        origin == "restart"
        and elapsed_s is not None
        and elapsed_s <= SHOT_CONTEXT_RESTART_MAX_S
    ):
        category = "restart_attack"
    elif sample is None:
        category = "unclassified"
    elif (
        origin == "opponent_regain"
        and elapsed_s <= SHOT_CONTEXT_COUNTERATTACK_MAX_S
        and progress_m is not None
        and progress_m >= SHOT_CONTEXT_COUNTERATTACK_MIN_PROGRESS_M
    ):
        category = "counterattack"
    elif origin == "opponent_regain" and elapsed_s <= SHOT_CONTEXT_QUICK_REGAIN_MAX_S:
        category = "quick_after_regain"
    elif elapsed_s >= SHOT_CONTEXT_SUSTAINED_MIN_S:
        category = "sustained_buildup"
    else:
        category = "open_play_buildup"

    labels = dict(SHOT_CONTEXT_CATEGORIES)
    return {
        "category": category,
        "label": labels[category],
        "sequence_origin": origin,
        "seconds_since_sequence_start": (
            None if elapsed_s is None else round(elapsed_s, 3)
        ),
        "forward_progress_m": (None if progress_m is None else round(progress_m, 3)),
    }


def _attach_shot_history_and_context(
    shot_rows: list[dict[str, Any]],
    facts: dict[str, Any],
    timeline: list[dict[str, Any]],
    sequence_samples: dict[tuple[int, int], dict[str, Any]],
    public_clocks: dict[int, tuple[int, float, float]],
    attack_direction_by_tick: dict[int, tuple[float, float]],
    dataset: MatchDataset,
) -> dict[str, Any]:
    """Attach positioned same-attack events and an explicit context receipt."""

    control_fps = float(dataset.metadata["control_fps"])
    causal_events: list[tuple[tuple[int, int, int, str], dict[str, Any]]] = []
    for source in chain(
        facts.get("applied_sources", []),
        facts.get("control_sources", []),
    ):
        team = source.get("team")
        tick = int(source["tick"])
        intent = int(source.get("intent", -1))
        event_type = INTENT_EVENT_NAMES.get(intent)
        if event_type is None:
            continue
        restart_kind = int(source.get("restart_kind", RESTART_NONE))
        restart_name = RESTART_NAMES.get(restart_kind)
        label = event_type.replace("_", " ").title()
        if restart_name is not None:
            label = f"{restart_name.replace('_', ' ').title()} {label}"
        contact = source.get("contact")
        raw_position = contact.get("position") if isinstance(contact, dict) else None
        absolute_position = None
        if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
            candidate = np.asarray(raw_position[:2], dtype=np.float64)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                absolute_position = [
                    round(float(candidate[0]), 3),
                    round(float(candidate[1]), 3),
                ]
        causal_events.append(
            (
                (
                    tick,
                    int(source.get("substep", 0)),
                    int(source.get("order", 0)),
                    event_type,
                ),
                {
                    "type": event_type,
                    "control_tick": tick,
                    **_timeline_clock(tick / control_fps, public_clocks.get(tick)),
                    "team": team,
                    "player_id": source.get("player_id"),
                    "label": label,
                    "absolute_position_m": absolute_position,
                },
            )
        )
    for item in timeline:
        tick = int(item["control_tick"])
        event_type = str(item["type"])
        causal_events.append(
            (
                (tick, -1, -1, event_type),
                {
                    "type": event_type,
                    "control_tick": tick,
                    "clock_s": float(item["clock_s"]),
                    "clock_label": str(item["clock_label"]),
                    "team": item.get("team"),
                    "player_id": None,
                    "label": str(item.get("label", event_type)),
                },
            )
        )
    causal_events.sort(key=lambda pair: pair[0])

    category_counts = {category: [0, 0] for category, _ in SHOT_CONTEXT_CATEGORIES}
    team_counts = {
        category: [[0, 0], [0, 0]] for category, _ in SHOT_CONTEXT_CATEGORIES
    }
    for shot in shot_rows:
        tick = int(shot["control_tick"])
        team = int(shot["team"])
        sample = sequence_samples.get((tick, team))
        sequence_start_tick = (
            tick if sample is None else int(sample.get("start_control_tick", tick))
        )
        directions = attack_direction_by_tick.get(tick)
        direction = None if directions is None else float(directions[team])
        positioned_events: list[dict[str, Any]] = []
        if direction in (-1.0, 1.0):
            for key, event in causal_events:
                if key[0] < sequence_start_tick or key[0] >= tick:
                    continue
                if event.get("team") != team:
                    continue
                raw_position = event.get("absolute_position_m")
                if not isinstance(raw_position, list) or len(raw_position) != 2:
                    continue
                normalized = _attack_normalized_position(raw_position, direction)
                positioned_events.append(
                    {
                        name: value
                        for name, value in event.items()
                        if name != "absolute_position_m"
                    }
                    | {
                        "position_m": [
                            round(normalized[0], 3),
                            round(normalized[1], 3),
                        ]
                    }
                )
        compacted_events: list[dict[str, Any]] = []
        for event in positioned_events:
            if (
                event.get("type") == "control_touch"
                and compacted_events
                and compacted_events[-1].get("type") == "control_touch"
                and compacted_events[-1].get("player_id") == event.get("player_id")
            ):
                compacted_events[-1] = event
            else:
                compacted_events.append(event)
        positioned_events = compacted_events
        route_basis = (
            "unavailable"
            if sample is None
            else str(sample.get("route_basis", "continuous_attacking_sequence"))
        )
        if route_basis == "bounded_tracking_lookback":
            # A lookback can cross a dead-ball boundary. Its exact tracking
            # positions remain useful, but earlier-phase contacts must not be
            # presented as actions in the current attacking sequence.
            positioned_events = []
        sequence_start = None
        if sample is not None:
            raw_start = sample.get("start_position_m")
            raw_shot = shot.get("position_m")
            if (
                sequence_start_tick < tick
                and isinstance(raw_start, list)
                and len(raw_start) == 2
                and isinstance(raw_shot, list)
                and len(raw_shot) == 2
            ):
                try:
                    start_position = [float(raw_start[0]), float(raw_start[1])]
                    shot_position = [float(raw_shot[0]), float(raw_shot[1])]
                except (TypeError, ValueError):
                    start_position = []
                    shot_position = []
                if (
                    len(start_position) == 2
                    and len(shot_position) == 2
                    and np.all(np.isfinite(start_position))
                    and np.all(np.isfinite(shot_position))
                    and float(
                        np.hypot(
                            start_position[0] - shot_position[0],
                            start_position[1] - shot_position[1],
                        )
                    )
                    > 1.0e-9
                ):
                    origin = str(sample.get("origin", "unavailable"))
                    start_labels = {
                        "opponent_regain": "Regain",
                        "restart": "Restart",
                        "loose_recovery": "Loose recovery",
                    }
                    if origin in start_labels:
                        sequence_start = {
                            "type": "sequence_start",
                            "control_tick": sequence_start_tick,
                            **_timeline_clock(
                                float(
                                    sample.get(
                                        "start_clock_s",
                                        sequence_start_tick / control_fps,
                                    )
                                ),
                                public_clocks.get(sequence_start_tick),
                            ),
                            "team": team,
                            "player_id": None,
                            "label": start_labels[origin],
                            "position_m": [
                                round(start_position[0], 3),
                                round(start_position[1], 3),
                            ],
                        }
        tracking_path: list[list[float]] = []
        if sample is not None:
            for item in sample.get("tracking_path", []):
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    continue
                try:
                    path_tick = int(item[0])
                    path_x_m = float(item[1])
                    path_y_m = float(item[2])
                except (TypeError, ValueError, OverflowError):
                    continue
                if (
                    sequence_start_tick <= path_tick < tick
                    and math.isfinite(path_x_m)
                    and math.isfinite(path_y_m)
                ):
                    tracking_path.append(
                        [float(path_tick), round(path_x_m, 3), round(path_y_m, 3)]
                    )
        shot["tracking_path"] = tracking_path
        shot["tracking_path_basis"] = route_basis
        if sequence_start is not None:
            positioned_events = [
                event
                for event in positioned_events
                if not (
                    int(event["control_tick"]) == sequence_start_tick
                    and float(
                        np.hypot(
                            float(event["position_m"][0])
                            - float(sequence_start["position_m"][0]),
                            float(event["position_m"][1])
                            - float(sequence_start["position_m"][1]),
                        )
                    )
                    <= 1.0e-9
                )
            ]
            shot["preceding_events"] = [sequence_start] + positioned_events[
                -(SHOT_ROUTE_MAX_PRIOR_NODES - 1) :
            ]
        else:
            shot["preceding_events"] = positioned_events[-SHOT_ROUTE_MAX_PRIOR_NODES:]
        context = _classify_shot_context(shot, sample)
        shot["pre_shot_context"] = context
        category = str(context["category"])
        is_goal = int(shot.get("category") == "goal")
        category_counts[category][0] += 1
        category_counts[category][1] += is_goal
        team_counts[category][team][0] += 1
        team_counts[category][team][1] += is_goal

    total_shots = len(shot_rows)
    total_goals = sum(row.get("category") == "goal" for row in shot_rows)
    return {
        "scope": "realized_shots",
        "classification": "mutually_exclusive_deterministic_report_heuristic",
        "sequence_semantics": "continuous live attacking sequence; intervening loose-ball frames do not end the last controlling team's sequence",
        "causal_history_semantics": "the route line uses actual attack-normalized ball positions from tracking through the frame strictly before the shot; it starts at the continuous live attacking-sequence boundary when available; when that boundary is unavailable, a dotted fallback shows the preceding 15 seconds of tracking position history, which may include dead-ball placement and is not labelled a continuous physical trajectory; fallback paths do not attach earlier-phase event nodes; normal event nodes are the optional sequence start plus up to six latest same-team exact deliberate-control or kick-applied pass/shot/clear/challenge contact locations; consecutive controls by the same player are collapsed to their latest contact; no intermediate position is inferred",
        "thresholds": {
            "restart_attack_max_s": SHOT_CONTEXT_RESTART_MAX_S,
            "quick_after_regain_max_s": SHOT_CONTEXT_QUICK_REGAIN_MAX_S,
            "counterattack_max_s": SHOT_CONTEXT_COUNTERATTACK_MAX_S,
            "counterattack_min_forward_progress_m": SHOT_CONTEXT_COUNTERATTACK_MIN_PROGRESS_M,
            "sustained_buildup_min_s": SHOT_CONTEXT_SUSTAINED_MIN_S,
            "route_max_prior_nodes_including_start": SHOT_ROUTE_MAX_PRIOR_NODES,
            "route_max_tracking_path_points": SHOT_ROUTE_MAX_TRACK_PATH_POINTS,
            "route_fallback_tracking_lookback_s": SHOT_ROUTE_FALLBACK_LOOKBACK_S,
        },
        "category_priority": [category for category, _ in SHOT_CONTEXT_CATEGORIES],
        "rows": [
            {
                "category": category,
                "label": label,
                "shots": category_counts[category][0],
                "goals": category_counts[category][1],
                "shot_share": (
                    0.0
                    if total_shots == 0
                    else round(category_counts[category][0] / total_shots, 6)
                ),
                "goal_share": (
                    0.0
                    if total_goals == 0
                    else round(category_counts[category][1] / total_goals, 6)
                ),
                "goal_conversion": (
                    None
                    if category_counts[category][0] == 0
                    else round(
                        category_counts[category][1] / category_counts[category][0],
                        6,
                    )
                ),
                "team_shots": [team_counts[category][team][0] for team in (0, 1)],
                "team_goals": [team_counts[category][team][1] for team in (0, 1)],
            }
            for category, label in SHOT_CONTEXT_CATEGORIES
        ],
    }


def build_match_report(dataset: MatchDataset) -> dict[str, Any]:
    """Recompute a compact report from verified tracking and event facts."""

    event_header = _event_header(dataset)
    action_controls_available = bool(
        event_header.get("action_controls_available", True)
    )
    pre_frame_management = event_header.get("pre_frame_management", []) or []
    initial_identities = _initial_identities(dataset)
    for batch in pre_frame_management:
        for substitution in batch.get("substitutions", []) or []:
            initial_identities[int(substitution["slot"])] = {
                "player_id": int(substitution["player_in"]),
                "slot_generation": int(substitution["slot_generation"]),
                "team": int(substitution["team"]),
            }
    contact_facts = _collect_policy_contact_facts(
        event_header, initial_identities, _iter_event_frames(dataset)
    )
    discontinuity_ticks = set(contact_facts["discontinuity_ticks"])
    timeline_ticks = set(contact_facts["timeline_ticks"])
    for batch in pre_frame_management:
        tick = int(batch["control_tick"])
        if any(
            isinstance(event, dict) and event.get("applied") is True
            for event in (batch.get("formations", []) or [])
        ):
            discontinuity_ticks.add(tick)
        if (
            batch.get("formations")
            or batch.get("substitutions")
            or batch.get("set_piece_taker_changes")
        ):
            timeline_ticks.add(tick)
    public_clocks: dict[int, tuple[int, float, float]] = {}
    attack_direction_by_tick: dict[int, tuple[float, float]] = {}
    player_positions_by_tick: dict[int, dict[int, tuple[float, float]]] = {}
    player_snapshots_by_tick: dict[int, list[dict[str, Any]]] = {}
    stadium = dataset.metadata["stadium"]
    length = float(stadium["length"])
    legacy_stadium_width = "width" not in stadium
    width = float(stadium.get("width", 68.0))
    penalty_length = float(stadium["penalty_area_length"])
    penalty_width = float(stadium["penalty_area_width"])
    player_stats: dict[tuple[int, int, int], dict[str, Any]] = {}
    possession_s = np.zeros(2, dtype=np.float64)
    loose_s = 0.0
    penalty_area_entries = np.zeros(2, dtype=np.int64)
    final_score = [0, 0]
    frame_count = 0
    first_clock: float | None = None
    last_clock = 0.0
    previous: dict[str, Any] | None = None
    shot_teams_by_tick: defaultdict[int, set[int]] = defaultdict(set)
    for source in contact_facts["shot_sources"]:
        if source.get("team") in (0, 1):
            shot_teams_by_tick[int(source["tick"])].add(int(source["team"]))
    player_position_ticks = {
        int(source["tick"]) for source in contact_facts["pass_sources"]
    }
    shot_sequence_samples: dict[tuple[int, int], dict[str, Any]] = {}
    recent_ball_tracking: deque[tuple[int, int, float, float, float]] = deque()
    sequence_team: int | None = None
    sequence_start_clock_s = 0.0
    sequence_start_tick = 0
    sequence_start_x_m = 0.0
    sequence_start_y_m = 0.0
    sequence_origin = "unavailable"
    sequence_period: int | None = None
    sequence_ball_path: list[list[float]] = []
    sequence_ball_path_stride = 1
    live_phase = False
    phase_waiting_for_first_control = False
    last_controlled_team: int | None = None
    chart_windows: list[dict[str, Any]] = []
    speed_histogram = np.zeros((2, SPEED_OVERFLOW_BIN_MPS + 1), dtype=np.int64)
    ball_territory = np.zeros(BALL_TERRITORY_BINS, dtype=np.int64)
    density_x_bins = max(1, int(np.ceil(length / BALL_DENSITY_TARGET_CELL_M)))
    density_y_bins = max(1, int(np.ceil(width / BALL_DENSITY_TARGET_CELL_M)))
    density_x_edges = np.linspace(-length / 2.0, length / 2.0, density_x_bins + 1)
    density_y_edges = np.linspace(-width / 2.0, width / 2.0, density_y_bins + 1)
    ball_density_s = np.zeros((density_y_bins, density_x_bins), dtype=np.float64)
    team_ball_density_s = np.zeros(
        (2, density_y_bins, density_x_bins), dtype=np.float64
    )
    loose_ball_density_s = np.zeros((density_y_bins, density_x_bins), dtype=np.float64)
    density_out_of_pitch_s = 0.0
    ball_density_window_s: defaultdict[tuple[int, int, int], float] = defaultdict(float)
    ball_density_windows: dict[int, dict[str, Any]] = {}
    current_loose_run: dict[str, Any] | None = None
    longest_loose_run: dict[str, Any] | None = None
    current_stationary_loose_run: dict[str, Any] | None = None
    longest_stationary_loose_run: dict[str, Any] | None = None
    player_density_x_bins = max(1, int(np.ceil(length / PLAYER_POSITION_TARGET_CELL_M)))
    player_density_y_bins = max(1, int(np.ceil(width / PLAYER_POSITION_TARGET_CELL_M)))
    player_density_x_edges = np.linspace(
        -length / 2.0, length / 2.0, player_density_x_bins + 1
    )
    player_density_y_edges = np.linspace(
        -width / 2.0, width / 2.0, player_density_y_bins + 1
    )
    player_occupancy_s: defaultdict[tuple[int, int, int, int, int, int], float] = (
        defaultdict(float)
    )
    player_window_end_tick: dict[tuple[int, int, int, int], int] = {}
    player_activity: dict[tuple[int, int, int], dict[str, Any]] = {}
    space_x_bins = max(1, int(np.ceil(length / SPACE_OCCUPANCY_TARGET_CELL_M)))
    space_y_bins = max(1, int(np.ceil(width / SPACE_OCCUPANCY_TARGET_CELL_M)))
    space_x_edges = np.linspace(-length / 2.0, length / 2.0, space_x_bins + 1)
    space_y_edges = np.linspace(-width / 2.0, width / 2.0, space_y_bins + 1)
    team_space_occupancy_s: defaultdict[tuple[int, int, int, int], float] = defaultdict(
        float
    )
    space_windows: dict[int, dict[str, Any]] = {}
    dismissal_records: list[dict[str, Any]] = []

    def chart_window(index: int) -> dict[str, Any]:
        while len(chart_windows) <= index:
            chart_windows.append(
                {
                    "team_possession_s": np.zeros(2, dtype=np.float64),
                    "loose_ball_s": 0.0,
                }
            )
        return chart_windows[index]

    for row in dataset.iter_tracking_rows():
        frame_count += 1
        clock = float(row["clock_s"])
        first_clock = clock if first_clock is None else first_clock
        last_clock = clock
        final_score = [int(value) for value in row["score"]]
        tick = int(row["control_tick"])
        current_period = int(row.get("period", 1))
        while (
            recent_ball_tracking
            and recent_ball_tracking[0][2] < clock - SHOT_ROUTE_FALLBACK_LOOKBACK_S
        ):
            recent_ball_tracking.popleft()
        for shot_team in shot_teams_by_tick.get(tick, ()):
            if (
                live_phase
                and sequence_team == shot_team
                and previous is not None
                and bool(previous["ball"]["live"])
            ):
                shot_sequence_samples[(tick, shot_team)] = {
                    "origin": sequence_origin,
                    "start_clock_s": sequence_start_clock_s,
                    "start_control_tick": sequence_start_tick,
                    "elapsed_s": max(0.0, clock - sequence_start_clock_s),
                    "start_x_m": sequence_start_x_m,
                    "start_position_m": [
                        round(sequence_start_x_m, 3),
                        round(sequence_start_y_m, 3),
                    ],
                    "tracking_path": [list(item) for item in sequence_ball_path],
                    "route_basis": "continuous_attacking_sequence",
                }
            else:
                fallback_sample = _tracking_lookback_sample(
                    recent_ball_tracking,
                    period=current_period,
                    shot_clock_s=clock,
                    direction=float(row["attack_direction"][shot_team]),
                )
                if fallback_sample is not None:
                    shot_sequence_samples[(tick, shot_team)] = fallback_sample
        if tick in timeline_ticks or tick in contact_facts["attack_direction_ticks"]:
            public_clocks[tick] = (
                int(row.get("period", 1)),
                float(row.get("display_clock_s", clock)),
                float(row.get("added_time_s", 0.0)),
            )
        players = row["players"]
        if tick in contact_facts["attack_direction_ticks"]:
            attack_direction_by_tick[tick] = tuple(
                float(value) for value in row["attack_direction"]
            )
        if tick in player_position_ticks:
            player_positions_by_tick[tick] = {
                int(player["player_id"]): (
                    float(player["position"][0]),
                    float(player["position"][1]),
                )
                for player in players
                if player["active"] and player["on_pitch"] and not player["sent_off"]
            }
            if previous is not None and int(previous["control_tick"]) == tick - 1:
                player_snapshots_by_tick[tick] = [
                    {
                        "team": int(player["team"]),
                        "player_id": int(player["player_id"]),
                        "position": (
                            float(player["position"][0]),
                            float(player["position"][1]),
                        ),
                    }
                    for player in previous["players"]
                    if player["active"]
                    and player["on_pitch"]
                    and not player["sent_off"]
                ]
        elapsed_s = max(0.0, clock - first_clock)
        window_index = int(elapsed_s // ANALYSIS_WINDOW_S)
        chart_window(window_index)

        for player in players:
            if not player["active"] or not player["on_pitch"]:
                continue
            key = (
                int(player["team"]),
                int(player["player_id"]),
                int(player["slot_generation"]),
            )
            stat = player_stats.setdefault(
                key,
                {
                    "team": key[0],
                    "player_id": key[1],
                    "slot_generation": key[2],
                    "active_s": 0.0,
                    "distance_m": 0.0,
                    "max_speed_mps": 0.0,
                },
            )
            speed = float(np.linalg.norm(np.asarray(player["velocity"], dtype=float)))
            stat["max_speed_mps"] = max(stat["max_speed_mps"], speed)
            speed_bin = min(int(max(0.0, speed)), SPEED_OVERFLOW_BIN_MPS)
            speed_histogram[key[0], speed_bin] += 1

        if bool(row["ball"]["live"]):
            ball_x = float(row["ball"]["position"][0])
            normalized_x = (ball_x + length / 2.0) / length
            territory_bin = min(
                BALL_TERRITORY_BINS - 1,
                max(0, int(normalized_x * BALL_TERRITORY_BINS)),
            )
            ball_territory[territory_bin] += 1

        if previous is not None:
            dt = clock - float(previous["clock_s"])
            continuous_interval = (
                dt > 0.0
                and int(previous.get("period", 1)) == int(row.get("period", 1))
                and tick not in discontinuity_ticks
            )
            interval_is_live_loose = (
                continuous_interval
                and bool(previous["ball"]["live"])
                and int(previous["possession"]["team"]) not in (0, 1)
            )
            if interval_is_live_loose:
                ball_position_3d = np.asarray(previous["ball"]["position"], dtype=float)
                ball_velocity_3d = np.asarray(
                    previous["ball"].get("velocity", [0.0, 0.0, 0.0]), dtype=float
                )
                if current_loose_run is None:
                    current_loose_run = {
                        "start_clock_s": float(previous["clock_s"]),
                        "start_control_tick": int(previous["control_tick"]),
                        "duration_s": 0.0,
                    }
                current_loose_run["duration_s"] += dt
                current_loose_run["end_clock_s"] = clock
                current_loose_run["end_control_tick"] = tick
                current_loose_run["position_m"] = [
                    round(float(value), 3) for value in ball_position_3d
                ]
                if (
                    longest_loose_run is None
                    or current_loose_run["duration_s"] > longest_loose_run["duration_s"]
                ):
                    longest_loose_run = dict(current_loose_run)

                stationary = bool(
                    np.all(np.isfinite(ball_velocity_3d))
                    and np.linalg.norm(ball_velocity_3d)
                    <= POLICY_AUDIT_STATIONARY_BALL_SPEED_MPS
                )
                if stationary:
                    if current_stationary_loose_run is None:
                        current_stationary_loose_run = {
                            "start_clock_s": float(previous["clock_s"]),
                            "start_control_tick": int(previous["control_tick"]),
                            "duration_s": 0.0,
                        }
                    current_stationary_loose_run["duration_s"] += dt
                    current_stationary_loose_run["end_clock_s"] = clock
                    current_stationary_loose_run["end_control_tick"] = tick
                    current_stationary_loose_run["position_m"] = [
                        round(float(value), 3) for value in ball_position_3d
                    ]
                    if (
                        longest_stationary_loose_run is None
                        or current_stationary_loose_run["duration_s"]
                        > longest_stationary_loose_run["duration_s"]
                    ):
                        longest_stationary_loose_run = dict(
                            current_stationary_loose_run
                        )
                else:
                    current_stationary_loose_run = None
            else:
                current_loose_run = None
                current_stationary_loose_run = None
            if continuous_interval:
                previous_team = int(previous["possession"]["team"])
                if bool(previous["ball"]["live"]):
                    previous_elapsed_s = max(
                        0.0, float(previous["clock_s"]) - float(first_clock)
                    )
                    density_window_index = int(
                        previous_elapsed_s // BALL_DENSITY_WINDOW_S
                    )
                    density_receipt = ball_density_windows.setdefault(
                        density_window_index,
                        {
                            "index": density_window_index,
                            "start_clock_s": float(previous["clock_s"]),
                            "end_clock_s": clock,
                            "end_control_tick": tick,
                            "observed_live_seconds": 0.0,
                            "excluded_out_of_pitch_seconds": 0.0,
                        },
                    )
                    ball_position = np.asarray(
                        previous["ball"]["position"][:2], dtype=float
                    )
                    if (
                        np.all(np.isfinite(ball_position))
                        and density_x_edges[0]
                        <= ball_position[0]
                        <= density_x_edges[-1]
                        and density_y_edges[0]
                        <= ball_position[1]
                        <= density_y_edges[-1]
                    ):
                        x_index = min(
                            density_x_bins - 1,
                            int(
                                np.searchsorted(
                                    density_x_edges, ball_position[0], side="right"
                                )
                                - 1
                            ),
                        )
                        y_index = min(
                            density_y_bins - 1,
                            int(
                                np.searchsorted(
                                    density_y_edges, ball_position[1], side="right"
                                )
                                - 1
                            ),
                        )
                        ball_density_s[y_index, x_index] += dt
                        ball_density_window_s[
                            (density_window_index, y_index, x_index)
                        ] += dt
                        density_receipt["observed_live_seconds"] += dt
                        if previous_team in (0, 1):
                            team_ball_density_s[previous_team, y_index, x_index] += dt
                        else:
                            loose_ball_density_s[y_index, x_index] += dt
                    else:
                        density_out_of_pitch_s += dt
                        density_receipt["excluded_out_of_pitch_seconds"] += dt
                    density_receipt["start_clock_s"] = min(
                        float(density_receipt["start_clock_s"]),
                        float(previous["clock_s"]),
                    )
                    density_receipt["end_clock_s"] = max(
                        float(density_receipt["end_clock_s"]), clock
                    )
                    density_receipt["end_control_tick"] = max(
                        int(density_receipt["end_control_tick"]), tick
                    )
                    possession_window = chart_window(
                        int(previous_elapsed_s // ANALYSIS_WINDOW_S)
                    )
                    if previous_team in (0, 1):
                        possession_s[previous_team] += dt
                        possession_window["team_possession_s"][previous_team] += dt
                    else:
                        loose_s += dt
                        possession_window["loose_ball_s"] += dt
                previous_players = {int(p["slot"]): p for p in previous["players"]}
                for player in players:
                    prior = previous_players.get(int(player["slot"]))
                    if prior is None:
                        continue
                    identity = (
                        int(player["team"]),
                        int(player["player_id"]),
                        int(player["slot_generation"]),
                    )
                    same_identity = identity == (
                        int(prior["team"]),
                        int(prior["player_id"]),
                        int(prior["slot_generation"]),
                    )
                    if (
                        same_identity
                        and player["active"]
                        and player["on_pitch"]
                        and prior["active"]
                        and prior["on_pitch"]
                    ):
                        stat = player_stats[identity]
                        current_position = np.asarray(player["position"], dtype=float)
                        prior_position = np.asarray(prior["position"], dtype=float)
                        stat["active_s"] += dt
                        stat["distance_m"] += float(
                            np.linalg.norm(current_position - prior_position)
                        )
                        if np.all(np.isfinite(current_position)) and np.all(
                            np.isfinite(prior_position)
                        ):
                            direction = float(previous["attack_direction"][identity[0]])
                            normalized_prior = _attack_normalized_position(
                                prior_position, direction
                            )
                            normalized_current = _attack_normalized_position(
                                current_position, direction
                            )
                            midpoint = 0.5 * (
                                np.asarray(normalized_prior)
                                + np.asarray(normalized_current)
                            )
                            activity = player_activity.setdefault(
                                identity,
                                {
                                    "team": identity[0],
                                    "player_id": identity[1],
                                    "slot_generation": identity[2],
                                    "first_tick": int(previous["control_tick"]),
                                    "last_tick": tick,
                                    "initial_position_m": [
                                        round(normalized_prior[0], 3),
                                        round(normalized_prior[1], 3),
                                    ],
                                },
                            )
                            activity["last_tick"] = tick
                            if (
                                player_density_x_edges[0]
                                <= midpoint[0]
                                <= player_density_x_edges[-1]
                                and player_density_y_edges[0]
                                <= midpoint[1]
                                <= player_density_y_edges[-1]
                            ):
                                x_index = min(
                                    player_density_x_bins - 1,
                                    int(
                                        np.searchsorted(
                                            player_density_x_edges,
                                            midpoint[0],
                                            side="right",
                                        )
                                        - 1
                                    ),
                                )
                                y_index = min(
                                    player_density_y_bins - 1,
                                    int(
                                        np.searchsorted(
                                            player_density_y_edges,
                                            midpoint[1],
                                            side="right",
                                        )
                                        - 1
                                    ),
                                )
                                previous_elapsed_s = max(
                                    0.0, float(previous["clock_s"]) - first_clock
                                )
                                position_window = int(
                                    previous_elapsed_s // PLAYER_POSITION_WINDOW_S
                                )
                                player_occupancy_s[
                                    (position_window, *identity, y_index, x_index)
                                ] += dt
                                window_identity = (position_window, *identity)
                                player_window_end_tick[window_identity] = max(
                                    tick,
                                    player_window_end_tick.get(window_identity, tick),
                                )
                            if bool(previous["ball"]["live"]):
                                team0_direction = float(previous["attack_direction"][0])
                                shared_prior = _attack_normalized_position(
                                    prior_position, team0_direction
                                )
                                shared_current = _attack_normalized_position(
                                    current_position, team0_direction
                                )
                                shared_midpoint = 0.5 * (
                                    np.asarray(shared_prior)
                                    + np.asarray(shared_current)
                                )
                                if (
                                    space_x_edges[0]
                                    <= shared_midpoint[0]
                                    <= space_x_edges[-1]
                                    and space_y_edges[0]
                                    <= shared_midpoint[1]
                                    <= space_y_edges[-1]
                                ):
                                    space_x_index = min(
                                        space_x_bins - 1,
                                        int(
                                            np.searchsorted(
                                                space_x_edges,
                                                shared_midpoint[0],
                                                side="right",
                                            )
                                            - 1
                                        ),
                                    )
                                    space_y_index = min(
                                        space_y_bins - 1,
                                        int(
                                            np.searchsorted(
                                                space_y_edges,
                                                shared_midpoint[1],
                                                side="right",
                                            )
                                            - 1
                                        ),
                                    )
                                    space_window = int(
                                        previous_elapsed_s // SPACE_OCCUPANCY_WINDOW_S
                                    )
                                    team_space_occupancy_s[
                                        (
                                            space_window,
                                            identity[0],
                                            space_y_index,
                                            space_x_index,
                                        )
                                    ] += dt
                                    receipt = space_windows.setdefault(
                                        space_window,
                                        {
                                            "index": space_window,
                                            "start_clock_s": float(previous["clock_s"]),
                                            "end_clock_s": clock,
                                            "end_control_tick": tick,
                                        },
                                    )
                                    receipt["start_clock_s"] = min(
                                        float(receipt["start_clock_s"]),
                                        float(previous["clock_s"]),
                                    )
                                    receipt["end_clock_s"] = max(
                                        float(receipt["end_clock_s"]), clock
                                    )
                                    receipt["end_control_tick"] = max(
                                        int(receipt["end_control_tick"]), tick
                                    )

                team = int(row["possession"]["team"])
                if (
                    team in (0, 1)
                    and team == int(previous["possession"]["team"])
                    and bool(row["ball"]["live"])
                    and bool(previous["ball"]["live"])
                ):
                    direction = float(row["attack_direction"][team])
                    x_now = float(row["ball"]["position"][0]) * direction
                    x_prev = float(previous["ball"]["position"][0]) * direction
                    y_prev = abs(float(previous["ball"]["position"][1]))
                    penalty_x = length / 2.0 - penalty_length
                    previous_inside = (
                        x_prev >= penalty_x and y_prev <= penalty_width / 2.0
                    )
                    crosses_penalty_area = _segment_intersects_box(
                        (x_prev, float(previous["ball"]["position"][1])),
                        (x_now, float(row["ball"]["position"][1])),
                        (penalty_x, length / 2.0),
                        (-penalty_width / 2.0, penalty_width / 2.0),
                    )
                    if not previous_inside and crosses_penalty_area:
                        penalty_area_entries[team] += 1

            previous_players_by_slot = {
                int(player["slot"]): player for player in previous["players"]
            }
            for player in players:
                prior = previous_players_by_slot.get(int(player["slot"]))
                if (
                    prior is not None
                    and player["sent_off"]
                    and not prior["sent_off"]
                    and int(player["player_id"]) == int(prior["player_id"])
                    and int(player["slot_generation"]) == int(prior["slot_generation"])
                ):
                    dismissal_records.append(
                        {
                            "team": int(player["team"]),
                            "player_id": int(player["player_id"]),
                            "slot_generation": int(player["slot_generation"]),
                            "clock_s": round(clock, 3),
                            "control_tick": tick,
                        }
                    )

        period = current_period
        current_live = bool(row["ball"]["live"])
        period_changed = sequence_period is not None and period != sequence_period
        if not current_live or period_changed:
            sequence_team = None
            sequence_origin = "unavailable"
            live_phase = False
            phase_waiting_for_first_control = False
            last_controlled_team = None
            sequence_ball_path = []
            sequence_ball_path_stride = 1
        if current_live:
            if not live_phase:
                live_phase = True
                phase_waiting_for_first_control = True
            current_team = int(row["possession"]["team"])
            if current_team in (0, 1):
                if sequence_team != current_team:
                    regained_from_opponent = last_controlled_team == 1 - current_team
                    sequence_origin = (
                        "opponent_regain"
                        if regained_from_opponent
                        else (
                            "restart"
                            if phase_waiting_for_first_control
                            else "loose_recovery"
                        )
                    )
                    sequence_team = current_team
                    sequence_start_clock_s = clock
                    sequence_start_tick = tick
                    sequence_start_x_m, sequence_start_y_m = (
                        _attack_normalized_position(
                            row["ball"]["position"][:2],
                            float(row["attack_direction"][current_team]),
                        )
                    )
                    sequence_ball_path = []
                    sequence_ball_path_stride = 1
                    public_clocks.setdefault(
                        tick,
                        (
                            period,
                            float(row.get("display_clock_s", clock)),
                            float(row.get("added_time_s", 0.0)),
                        ),
                    )
                last_controlled_team = current_team
                phase_waiting_for_first_control = False
        if current_live and sequence_team in (0, 1):
            path_x_m, path_y_m = _attack_normalized_position(
                row["ball"]["position"][:2],
                float(row["attack_direction"][sequence_team]),
            )
            if not sequence_ball_path or tick % sequence_ball_path_stride == 0:
                sequence_ball_path.append(
                    [float(tick), round(path_x_m, 3), round(path_y_m, 3)]
                )
            if len(sequence_ball_path) > SHOT_ROUTE_MAX_TRACK_PATH_POINTS:
                sequence_ball_path_stride *= 2
                first_path_point = sequence_ball_path[0]
                sequence_ball_path = [first_path_point] + [
                    item
                    for item in sequence_ball_path[1:]
                    if int(item[0]) % sequence_ball_path_stride == 0
                ]
        recent_ball_tracking.append(
            (
                period,
                tick,
                clock,
                float(row["ball"]["position"][0]),
                float(row["ball"]["position"][1]),
            )
        )
        sequence_period = period
        previous = row

    captured_duration = (
        0.0
        if first_clock is None
        else last_clock - first_clock + 1.0 / float(dataset.metadata["control_fps"])
    )
    live_s = float(possession_s.sum() + loose_s)
    observed_density_s = float(ball_density_s.sum())
    dominant_density: dict[str, Any] | None = None
    if observed_density_s > 0.0:
        dominant_flat_index = int(np.argmax(ball_density_s))
        dominant_y_index, dominant_x_index = np.unravel_index(
            dominant_flat_index, ball_density_s.shape
        )
        dominant_seconds = float(ball_density_s[dominant_y_index, dominant_x_index])
        dominant_density = {
            "seconds": round(dominant_seconds, 3),
            "share": round(dominant_seconds / observed_density_s, 6),
            "x_index": int(dominant_x_index),
            "y_index": int(dominant_y_index),
            "x_bounds_m": [
                round(float(density_x_edges[dominant_x_index]), 3),
                round(float(density_x_edges[dominant_x_index + 1]), 3),
            ],
            "y_bounds_m": [
                round(float(density_y_edges[dominant_y_index]), 3),
                round(float(density_y_edges[dominant_y_index + 1]), 3),
            ],
        }

    policy_anomalies: list[dict[str, Any]] = []
    if (
        longest_stationary_loose_run is not None
        and float(longest_stationary_loose_run["duration_s"])
        >= POLICY_AUDIT_STATIONARY_LOOSE_WARN_S
    ):
        policy_anomalies.append(
            {
                "code": "stationary_live_loose_ball",
                "severity": "high",
                "message": "Live loose ball remained nearly stationary long enough to suggest a policy recovery failure.",
                "threshold": {
                    "duration_s": POLICY_AUDIT_STATIONARY_LOOSE_WARN_S,
                    "maximum_speed_mps": POLICY_AUDIT_STATIONARY_BALL_SPEED_MPS,
                },
                "observed": longest_stationary_loose_run,
            }
        )
    if (
        longest_loose_run is not None
        and float(longest_loose_run["duration_s"]) >= POLICY_AUDIT_LOOSE_BALL_WARN_S
    ):
        policy_anomalies.append(
            {
                "code": "prolonged_live_loose_ball",
                "severity": "medium",
                "message": "Live ball remained uncontrolled long enough to require policy review.",
                "threshold": {"duration_s": POLICY_AUDIT_LOOSE_BALL_WARN_S},
                "observed": longest_loose_run,
            }
        )
    if (
        dominant_density is not None
        and float(dominant_density["seconds"]) >= POLICY_AUDIT_DENSITY_WARN_S
        and float(dominant_density["share"]) >= POLICY_AUDIT_DENSITY_WARN_SHARE
    ):
        policy_anomalies.append(
            {
                "code": "ball_position_overconcentration",
                "severity": "medium",
                "message": "One 2 m ball-position cell contains an unusually large share of live time.",
                "threshold": {
                    "seconds": POLICY_AUDIT_DENSITY_WARN_S,
                    "share": POLICY_AUDIT_DENSITY_WARN_SHARE,
                },
                "observed": dominant_density,
            }
        )
    possession_share = [
        (float(possession_s[team] / live_s) if live_s else None) for team in (0, 1)
    ]
    controlled_s = float(possession_s.sum())
    controlled_possession_share = [
        (float(possession_s[team] / controlled_s) if controlled_s else None)
        for team in (0, 1)
    ]
    policy_alignment = _policy_alignment_metrics(
        contact_facts, attack_direction_by_tick, live_s, length / 2.0
    )
    pass_map_rows = _pass_map_rows(
        contact_facts,
        attack_direction_by_tick,
        player_positions_by_tick,
        player_snapshots_by_tick,
        length / 2.0,
    )
    shot_map_rows = _shot_map_rows(
        contact_facts,
        attack_direction_by_tick,
        public_clocks,
        dataset,
    )

    event_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    team_action_counts = [Counter(), Counter()]
    team_event_counts = [Counter(), Counter()]
    pass_target_known = 0
    intent_action_rows: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    substitution_records: list[dict[str, Any]] = []
    repeated_event_runs: dict[tuple[Any, ...], dict[str, Any]] = {}
    longest_repeated_event: dict[str, Any] | None = None
    identities = _initial_identities(dataset)
    for frame in chain(pre_frame_management, _iter_event_frames(dataset)):
        tick = int(frame["control_tick"])
        clock = float(frame.get("clock_s", tick / dataset.metadata["control_fps"]))
        overrides = {
            int(item["slot"]): item
            for item in frame.get("pre_management_identity", [])
            if isinstance(item, dict) and "slot" in item
        }
        transition_outcome = (frame.get("transition") or {}).get("outcome", {})
        if transition_outcome.get("restart_opened") is True:
            restart_kind = int(transition_outcome.get("restart_kind", -1))
            restart_name = f"restart_{RESTART_NAMES.get(restart_kind, 'unknown')}"
            event_counts[restart_name] += 1
            restart_team = int(transition_outcome.get("restart_team", -1))
            if restart_team in (0, 1):
                team_event_counts[restart_team][restart_name] += 1
        action = frame.get("action")
        if isinstance(action, dict):
            for row in action.get("rows", []):
                intent = int(row.get("intent", -1))
                actor = row.get("player")
                identity = overrides.get(actor, identities.get(actor, {}))
                team = identity.get("team") if isinstance(identity, dict) else None
                player_id = (
                    identity.get("player_id") if isinstance(identity, dict) else None
                )
                raw_position = None
                directions = None
                if isinstance(player_id, int):
                    for nearby_tick in (
                        tick,
                        tick + 1,
                        tick - 1,
                        tick + 2,
                        tick - 2,
                        tick + 3,
                        tick - 3,
                        tick + 4,
                        tick - 4,
                        tick + 5,
                        tick - 5,
                    ):
                        raw_position = player_positions_by_tick.get(
                            nearby_tick, {}
                        ).get(player_id)
                        directions = attack_direction_by_tick.get(nearby_tick)
                        if raw_position is not None and directions is not None:
                            break
                if (
                    intent in (INTENT_PASS, INTENT_SHOT, INTENT_CLEAR, INTENT_CHALLENGE)
                    and team in (0, 1)
                    and raw_position is not None
                    and directions is not None
                ):
                    normalized_position = _attack_normalized_position(
                        raw_position, float(directions[int(team)])
                    )
                    intent_action_rows.append(
                        {
                            "control_tick": tick,
                            "team": int(team),
                            "player_id": player_id,
                            "slot_generation": identity.get("slot_generation", 0),
                            "intent": intent,
                            "position_m": [
                                round(normalized_position[0], 3),
                                round(normalized_position[1], 3),
                            ],
                            "direction_unit": _action_direction_unit(row),
                        }
                    )
                if intent == INTENT_PASS:
                    action_counts["submitted_pass"] += 1
                    if team in (0, 1):
                        team_action_counts[int(team)]["submitted_pass"] += 1
                    if row.get("intended_receiver_player_id") is not None:
                        pass_target_known += 1
                elif intent == INTENT_SHOT:
                    action_counts["submitted_shot"] += 1
                    if team in (0, 1):
                        team_action_counts[int(team)]["submitted_shot"] += 1
        exact = frame.get("frame_events")
        if isinstance(exact, dict):
            for event in exact.get("events", []):
                kind = str(event.get("type", "unknown"))
                fields = event.get("fields", {})
                if not isinstance(fields, dict):
                    fields = {}
                if kind == "contest" and fields.get("occurred") is not True:
                    continue
                event_counts[kind] += 1
                if kind in {"contact", "deliberate_contact"} and (
                    kind != "contact" or fields.get("occurred") is True
                ):
                    signature = (
                        kind,
                        _integer_event_field(fields, "actor"),
                        _integer_event_field(fields, "intent"),
                        _integer_event_field(fields, "mechanism"),
                        _integer_event_field(fields, "law11_effect"),
                        bool(fields.get("kick_applied", False)),
                    )
                    prior_run = repeated_event_runs.get(signature)
                    if (
                        prior_run is None
                        or tick - int(prior_run["last_control_tick"]) > 1
                    ):
                        prior_run = {
                            "event_type": kind,
                            "actor_slot": signature[1],
                            "intent": signature[2],
                            "mechanism": signature[3],
                            "law11_effect": signature[4],
                            "kick_applied": signature[5],
                            "start_clock_s": round(clock, 3),
                            "start_control_tick": tick,
                            "occurrences": 0,
                        }
                    prior_run["occurrences"] += 1
                    prior_run["last_clock_s"] = round(clock, 3)
                    prior_run["last_control_tick"] = tick
                    prior_run["duration_s"] = round(
                        max(0.0, clock - float(prior_run["start_clock_s"])), 3
                    )
                    repeated_event_runs[signature] = prior_run
                    if longest_repeated_event is None or float(
                        prior_run["duration_s"]
                    ) > float(longest_repeated_event["duration_s"]):
                        longest_repeated_event = dict(prior_run)
                actor_slot = (
                    fields.get("offender", event.get("slot"))
                    if kind == "foul"
                    else fields.get("actor", event.get("slot"))
                )
                identity = overrides.get(actor_slot, identities.get(actor_slot, {}))
                team = identity.get("team") if isinstance(identity, dict) else None
                if kind == "foul" and fields.get("offender_team") in (0, 1):
                    team = int(fields["offender_team"])
                elif kind == "offside" and fields.get("team") in (0, 1):
                    team = int(fields["team"])
                is_goal = kind == "boundary" and fields.get("scoring_team") in (0, 1)
                if kind == "foul" and team in (0, 1):
                    team_event_counts[int(team)]["fouls_committed"] += 1
                elif kind == "offside" and team in (0, 1):
                    team_event_counts[int(team)]["offsides"] += 1
                if is_goal:
                    scoring_team = int(fields["scoring_team"])
                    team_event_counts[scoring_team]["goals"] += 1
                if kind == "foul" or is_goal:
                    timeline.append(
                        {
                            "type": "goal" if is_goal else kind,
                            "control_tick": tick,
                            **_timeline_clock(clock, public_clocks.get(tick)),
                            "video_time_s": dataset.video_time_s(tick),
                            "team": fields.get("scoring_team", team),
                            "label": _event_label(kind, fields, team),
                        }
                    )
        for key, kind in (
            ("substitutions", "substitution"),
            ("acting_goalkeepers", "acting_goalkeeper"),
            ("formations", "formation"),
            ("set_piece_taker_changes", "set_piece_taker_changed"),
        ):
            for event in frame.get(key, []) or []:
                if kind == "formation" and event.get("applied") is not True:
                    continue
                event_counts[kind] += 1
                if kind in {"substitution", "formation", "set_piece_taker_changed"}:
                    timeline.append(
                        {
                            "type": kind,
                            "control_tick": tick,
                            **_timeline_clock(clock, public_clocks.get(tick)),
                            "video_time_s": dataset.video_time_s(tick),
                            "team": event.get("team"),
                            "label": _event_label(kind, event, event.get("team")),
                            "details": event,
                        }
                    )
                if kind == "substitution":
                    slot = int(event["slot"])
                    prior_identity = identities.get(slot, {})
                    substitution_records.append(
                        {
                            "team": int(event["team"]),
                            "player_out": int(event["player_out"]),
                            "player_out_slot_generation": int(
                                prior_identity.get(
                                    "slot_generation",
                                    int(event["slot_generation"]) - 1,
                                )
                            ),
                            "player_in": int(event["player_in"]),
                            "player_in_slot_generation": int(event["slot_generation"]),
                            "control_tick": tick,
                            **_timeline_clock(clock, public_clocks.get(tick)),
                        }
                    )
                    identities[slot] = {
                        "player_id": int(event["player_in"]),
                        "slot_generation": int(event["slot_generation"]),
                        "team": int(event["team"]),
                    }

    timeline.sort(key=lambda item: (item["control_tick"], item["type"]))
    shot_context_summary = _attach_shot_history_and_context(
        shot_map_rows,
        contact_facts,
        timeline,
        shot_sequence_samples,
        public_clocks,
        attack_direction_by_tick,
        dataset,
    )

    existing_intent_keys = {
        (int(row["control_tick"]), int(row["player_id"]), int(row["intent"]))
        for row in intent_action_rows
        if isinstance(row.get("player_id"), int)
    }
    for source in contact_facts["intent_sources"]:
        contact = source.get("contact")
        team = source.get("team")
        source_tick = int(source["tick"])
        directions = next(
            (
                attack_direction_by_tick.get(nearby_tick)
                for nearby_tick in (
                    source_tick,
                    source_tick + 1,
                    source_tick - 1,
                    source_tick + 2,
                    source_tick - 2,
                    source_tick + 3,
                    source_tick - 3,
                    source_tick + 4,
                    source_tick - 4,
                    source_tick + 5,
                    source_tick - 5,
                )
                if attack_direction_by_tick.get(nearby_tick) is not None
            ),
            None,
        )
        if (
            contact is None
            or contact.get("position") is None
            or team not in (0, 1)
            or directions is None
        ):
            continue
        key = (int(source["tick"]), int(source["player_id"]), int(source["intent"]))
        if key in existing_intent_keys:
            continue
        raw_position = contact["position"]
        normalized_position = _attack_normalized_position(
            raw_position, float(directions[int(team)])
        )
        intent_action_rows.append(
            {
                "control_tick": int(source["tick"]),
                "team": int(team),
                "player_id": source.get("player_id"),
                "slot_generation": source.get("slot_generation", 0),
                "intent": int(source["intent"]),
                "position_m": [
                    round(normalized_position[0], 3),
                    round(normalized_position[1], 3),
                ],
                "realized_contact": True,
            }
        )

    team_realized_passes = np.zeros(2, dtype=np.int64)
    team_completed_passes = np.zeros(2, dtype=np.int64)
    team_cross_signatures = np.zeros(2, dtype=np.int64)
    team_completed_cross_signatures = np.zeros(2, dtype=np.int64)
    team_line_break_proxies = np.zeros(2, dtype=np.int64)
    team_completed_line_break_proxies = np.zeros(2, dtype=np.int64)
    team_forward_passes = np.zeros(2, dtype=np.int64)
    team_completed_forward_passes = np.zeros(2, dtype=np.int64)
    team_attacking_third_passes = np.zeros(2, dtype=np.int64)
    team_completed_attacking_third_passes = np.zeros(2, dtype=np.int64)
    team_attacking_third_backward_passes = np.zeros(2, dtype=np.int64)
    pass_time_bins = np.zeros((2, 6, 2), dtype=np.int64)
    control_fps = float(dataset.metadata["control_fps"])
    for row in pass_map_rows:
        team = row.get("team")
        if team not in (0, 1):
            continue
        team_realized_passes[int(team)] += 1
        completed = row.get("outcome") == "same_team_next_contact"
        pass_window = min(
            pass_time_bins.shape[1] - 1,
            max(
                0,
                int(
                    (float(row["control_tick"]) / control_fps)
                    // POLICY_AUDIT_PASS_WINDOW_S
                ),
            ),
        )
        pass_time_bins[int(team), pass_window, 0] += 1
        pass_time_bins[int(team), pass_window, 1] += int(completed)
        cross_signature = row.get("rule_policy_cross_control_signature") is True
        line_break = row.get("defensive_line_breaking_pass_proxy") is True
        forward = row.get("applied_direction_family") == "forward"
        attacking_third = row.get("source_third") == "attacking_third"
        team_cross_signatures[int(team)] += int(cross_signature)
        team_line_break_proxies[int(team)] += int(line_break)
        team_forward_passes[int(team)] += int(forward)
        team_attacking_third_passes[int(team)] += int(attacking_third)
        team_attacking_third_backward_passes[int(team)] += int(
            attacking_third and row.get("applied_direction_family") == "backward"
        )
        if completed:
            team_completed_passes[int(team)] += 1
            team_completed_cross_signatures[int(team)] += int(cross_signature)
            team_completed_line_break_proxies[int(team)] += int(line_break)
            team_completed_forward_passes[int(team)] += int(forward)
            team_completed_attacking_third_passes[int(team)] += int(attacking_third)

    pass_completion_by_team: list[list[dict[str, Any]]] = []
    for team in (0, 1):
        time_rows: list[dict[str, Any]] = []
        for window in range(pass_time_bins.shape[1]):
            attempts = int(pass_time_bins[team, window, 0])
            completed = int(pass_time_bins[team, window, 1])
            time_rows.append(
                {
                    "start_minute": 15 * window,
                    "end_minute": 15 * (window + 1),
                    "attempts": attempts,
                    "completed": completed,
                    "completion": (
                        None if attempts == 0 else round(completed / attempts, 6)
                    ),
                }
            )
        pass_completion_by_team.append(time_rows)
        valid_windows = [
            window
            for window in range(pass_time_bins.shape[1])
            if pass_time_bins[team, window, 0] >= POLICY_AUDIT_PASS_WINDOW_MIN_ATTEMPTS
        ]
        if len(valid_windows) >= 4:
            x_minutes = np.asarray(
                [15.0 * window + 7.5 for window in valid_windows], dtype=float
            )
            weights = np.asarray(
                [pass_time_bins[team, window, 0] for window in valid_windows],
                dtype=float,
            )
            rates = np.asarray(
                [
                    pass_time_bins[team, window, 1] / pass_time_bins[team, window, 0]
                    for window in valid_windows
                ],
                dtype=float,
            )
            mean_x = float(np.average(x_minutes, weights=weights))
            mean_rate = float(np.average(rates, weights=weights))
            denominator = float(np.sum(weights * np.square(x_minutes - mean_x)))
            slope_per_minute = (
                0.0
                if denominator <= 0.0
                else float(
                    np.sum(weights * (x_minutes - mean_x) * (rates - mean_rate))
                    / denominator
                )
            )
            fitted_change = slope_per_minute * float(x_minutes[-1] - x_minutes[0])
            if -fitted_change >= POLICY_AUDIT_PASS_COMPLETION_TREND_DROP:
                policy_anomalies.append(
                    {
                        "code": "pass_completion_time_trend_drop",
                        "severity": "medium",
                        "message": "The weighted 15-minute pass-receipt trend fell beyond the configured late-match degradation allowance.",
                        "threshold": {
                            "minimum_attempts_per_window": POLICY_AUDIT_PASS_WINDOW_MIN_ATTEMPTS,
                            "minimum_fitted_drop": POLICY_AUDIT_PASS_COMPLETION_TREND_DROP,
                        },
                        "observed": {
                            "team": team,
                            "valid_window_indices": valid_windows,
                            "slope_per_minute": round(slope_per_minute, 8),
                            "fitted_first_to_last_change": round(fitted_change, 6),
                            "windows": time_rows,
                        },
                    }
                )

    combined_pass_attempts = pass_time_bins[:, :, 0].sum(axis=0)
    for window in range(1, pass_time_bins.shape[1] - 1):
        if (window + 1) * POLICY_AUDIT_PASS_WINDOW_S > captured_duration + 1.0e-6:
            continue
        attempts = int(combined_pass_attempts[window])
        prior_attempts = int(combined_pass_attempts[window - 1])
        next_attempts = int(combined_pass_attempts[window + 1])
        if (
            attempts <= POLICY_AUDIT_PASS_ACTIVITY_MAX_ATTEMPTS
            and prior_attempts >= POLICY_AUDIT_PASS_ACTIVITY_NEIGHBOR_MIN_ATTEMPTS
            and next_attempts >= POLICY_AUDIT_PASS_ACTIVITY_NEIGHBOR_MIN_ATTEMPTS
        ):
            policy_anomalies.append(
                {
                    "code": "mid_match_pass_activity_collapse",
                    "severity": "high",
                    "message": "Realized passing nearly vanished for a full 15-minute window despite active neighboring windows.",
                    "threshold": {
                        "maximum_combined_attempts": POLICY_AUDIT_PASS_ACTIVITY_MAX_ATTEMPTS,
                        "minimum_neighbor_attempts": POLICY_AUDIT_PASS_ACTIVITY_NEIGHBOR_MIN_ATTEMPTS,
                    },
                    "observed": {
                        "start_minute": 15 * window,
                        "end_minute": 15 * (window + 1),
                        "combined_attempts": attempts,
                        "previous_window_attempts": prior_attempts,
                        "next_window_attempts": next_attempts,
                    },
                }
            )

    for team in (0, 1):
        attacking_third_attempts = int(team_attacking_third_passes[team])
        backward_attempts = int(team_attacking_third_backward_passes[team])
        backward_share = (
            0.0
            if attacking_third_attempts == 0
            else backward_attempts / attacking_third_attempts
        )
        if (
            attacking_third_attempts >= POLICY_AUDIT_ATTACKING_THIRD_MIN_PASSES
            and backward_share >= POLICY_AUDIT_ATTACKING_THIRD_BACKWARD_SHARE
        ):
            policy_anomalies.append(
                {
                    "code": "attacking_third_backward_pass_concentration",
                    "severity": "medium",
                    "message": "Backward releases dominate this team's attacking-third passing and may indicate stalled chance creation.",
                    "threshold": {
                        "minimum_attempts": POLICY_AUDIT_ATTACKING_THIRD_MIN_PASSES,
                        "minimum_backward_share": POLICY_AUDIT_ATTACKING_THIRD_BACKWARD_SHARE,
                    },
                    "observed": {
                        "team": team,
                        "attempts": attacking_third_attempts,
                        "backward_attempts": backward_attempts,
                        "backward_share": round(backward_share, 6),
                    },
                }
            )

    dismissal_counts = Counter(record["team"] for record in dismissal_records)
    for team in (0, 1):
        if dismissal_counts[team] >= POLICY_AUDIT_DISMISSAL_WARN_COUNT:
            policy_anomalies.append(
                {
                    "code": "excessive_dismissals",
                    "severity": "high",
                    "message": "A team accumulated enough dismissals to suggest discipline or challenge-policy instability.",
                    "threshold": {"dismissals": POLICY_AUDIT_DISMISSAL_WARN_COUNT},
                    "observed": {
                        "team": team,
                        "dismissals": int(dismissal_counts[team]),
                        "records": [
                            row for row in dismissal_records if row["team"] == team
                        ],
                    },
                }
            )
    if (
        longest_repeated_event is not None
        and float(longest_repeated_event["duration_s"])
        >= POLICY_AUDIT_EVENT_REPEAT_WARN_S
    ):
        policy_anomalies.append(
            {
                "code": "repeated_event_pattern",
                "severity": "medium",
                "message": "The same actor/event signature repeated on consecutive control ticks for an unusually long interval.",
                "threshold": {"duration_s": POLICY_AUDIT_EVENT_REPEAT_WARN_S},
                "observed": longest_repeated_event,
            }
        )
    team_shot_outcomes = [Counter(), Counter()]
    team_shot_distances: list[list[float]] = [[], []]
    team_shots_inside_penalty_area = np.zeros(2, dtype=np.int64)
    half_length = 0.5 * length
    for row in shot_map_rows:
        team = row.get("team")
        if team in (0, 1):
            team_index = int(team)
            team_shot_outcomes[team_index][
                str(row.get("category", "unresolved"))
            ] += 1
            shot_x, shot_y = (float(value) for value in row["position_m"])
            team_shot_distances[team_index].append(
                float(np.hypot(half_length - shot_x, shot_y))
            )
            team_shots_inside_penalty_area[team_index] += int(
                shot_x >= half_length - penalty_length
                and abs(shot_y) <= 0.5 * penalty_width
            )

    teams = []
    metric_receipts = []
    for team in (0, 1):
        realized_passes = int(team_realized_passes[team])
        completed_passes = int(team_completed_passes[team])
        cross_signatures = int(team_cross_signatures[team])
        completed_cross_signatures = int(team_completed_cross_signatures[team])
        line_break_proxies = int(team_line_break_proxies[team])
        completed_line_break_proxies = int(team_completed_line_break_proxies[team])
        shot_outcomes = team_shot_outcomes[team]
        realized_shots = int(sum(shot_outcomes.values()))
        shot_goals = int(shot_outcomes["goal"])
        saved_on_target = int(shot_outcomes["on_target"])
        shot_distances = team_shot_distances[team]
        team_row = {
            "team": team,
            "possession_s": round(float(possession_s[team]), 3),
            "live_time_possession_share": (
                None
                if possession_share[team] is None
                else round(possession_share[team], 6)
            ),
            "controlled_possession_share": (
                None
                if controlled_possession_share[team] is None
                else round(controlled_possession_share[team], 6)
            ),
            "goals": int(final_score[team]),
            "submitted_shots": int(team_action_counts[team]["submitted_shot"]),
            "realized_shots": realized_shots,
            "shots_on_target": shot_goals + saved_on_target,
            "shot_goals": shot_goals,
            "saved_on_target_shots": saved_on_target,
            "off_target_shots": int(shot_outcomes["off_target"]),
            "unresolved_shots": int(shot_outcomes["unresolved"]),
            "mean_shot_distance_m": (
                None
                if not shot_distances
                else round(float(np.mean(shot_distances)), 6)
            ),
            "shots_inside_penalty_area": int(team_shots_inside_penalty_area[team]),
            "open_play_pass_attempts": realized_passes,
            "open_play_completed_passes": completed_passes,
            "open_play_pass_completion": (
                None
                if not realized_passes
                else round(completed_passes / realized_passes, 6)
            ),
            "pass_completion_by_15m": pass_completion_by_team[team],
            "rule_policy_cross_control_signatures": (
                cross_signatures if action_controls_available else None
            ),
            "completed_rule_policy_cross_control_signatures": (
                completed_cross_signatures if action_controls_available else None
            ),
            "defensive_line_breaking_pass_proxies": line_break_proxies,
            "completed_defensive_line_breaking_pass_proxies": completed_line_break_proxies,
            "forward_pass_attempts": int(team_forward_passes[team]),
            "completed_forward_passes": int(team_completed_forward_passes[team]),
            "attacking_third_pass_attempts": int(team_attacking_third_passes[team]),
            "completed_attacking_third_passes": int(
                team_completed_attacking_third_passes[team]
            ),
            "attacking_third_backward_pass_attempts": int(
                team_attacking_third_backward_passes[team]
            ),
            "penalty_area_entries": int(penalty_area_entries[team]),
            "corners": int(team_event_counts[team]["restart_corner"]),
            "fouls_committed": int(team_event_counts[team]["fouls_committed"]),
            "offsides": int(team_event_counts[team]["offsides"]),
        }
        teams.append(team_row)
        metric_receipts.extend(
            (
                _metric(
                    f"team.{team}.possession_seconds",
                    team_row["possession_s"],
                    "s",
                    "Live tracking intervals attributed to this team.",
                ),
                _metric(
                    f"team.{team}.live_time_possession_share",
                    team_row["live_time_possession_share"],
                    "ratio_of_all_live_time",
                    "Team-attributed live intervals divided by all live intervals, including loose-ball time.",
                ),
                _metric(
                    f"team.{team}.controlled_possession_share",
                    team_row["controlled_possession_share"],
                    "ratio_of_team_controlled_time",
                    "Full-capture team-attributed live intervals divided by intervals attributed to either team; loose-ball time is excluded.",
                ),
                _metric(
                    f"team.{team}.submitted_shots",
                    team_row["submitted_shots"],
                    "actions",
                    "Submitted SHOT actions attributed to this team; on-target status is not inferred.",
                    quality="submitted_fact_count",
                ),
                _metric(
                    f"team.{team}.realized_shots",
                    team_row["realized_shots"],
                    "shots",
                    "Applied SHOT contacts with an exact source location and known attack direction.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.shots_on_target",
                    team_row["shots_on_target"],
                    "shots",
                    "Realized shots ending in a confirmed goal for the shooting team or an opponent contact labelled by physics as a deliberate save.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.mean_shot_distance_m",
                    team_row["mean_shot_distance_m"],
                    "m",
                    "Euclidean distance from each normalized realized-shot contact to the attacking goal centre.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.shots_inside_penalty_area",
                    team_row["shots_inside_penalty_area"],
                    "shots",
                    "Realized-shot contacts inside the configured attacking penalty-area rectangle.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.off_target_shots",
                    team_row["off_target_shots"],
                    "shots",
                    "Resolved realized shots that were neither a shooting-team goal nor stopped by a deliberate-save contact; blocks and woodwork remain identified in the shot map.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.open_play_completed_passes",
                    team_row["open_play_completed_passes"],
                    "passes",
                    "Realized open-play pass contacts whose next distinct actor contact belongs to the same team.",
                    quality="receipt_proxy",
                ),
                _metric(
                    f"team.{team}.open_play_pass_completion",
                    team_row["open_play_pass_completion"],
                    "ratio",
                    "Same-team next-contact receipts divided by realized open-play pass contacts.",
                    quality="receipt_proxy",
                ),
                _metric(
                    f"team.{team}.rule_policy_cross_control_signatures",
                    team_row["rule_policy_cross_control_signatures"],
                    "passes",
                    "Realized open-play PASS contacts whose retained submitted spin exactly matches the shipped rule policy cross-control signature (absolute side spin 0.18, back spin 0.35, tolerance 1e-6).",
                    quality="policy_specific_control_signature",
                ),
                _metric(
                    f"team.{team}.completed_rule_policy_cross_control_signatures",
                    team_row["completed_rule_policy_cross_control_signatures"],
                    "passes",
                    "Rule-policy cross-control signatures followed by a same-team next distinct-actor contact before a boundary.",
                    quality="policy_specific_receipt_proxy",
                ),
                _metric(
                    f"team.{team}.defensive_line_breaking_pass_proxies",
                    team_row["defensive_line_breaking_pass_proxies"],
                    "passes",
                    "Realized open-play passes progressing at least 5 m whose next distinct-actor contact lies at least 0.5 m beyond the source-time second-last opponent; this is a through-pass proxy, not an event label.",
                    quality="geometry_proxy",
                ),
                _metric(
                    f"team.{team}.completed_defensive_line_breaking_pass_proxies",
                    team_row["completed_defensive_line_breaking_pass_proxies"],
                    "passes",
                    "Defensive-line-breaking pass proxies followed by a same-team next distinct-actor contact before a boundary.",
                    quality="geometry_receipt_proxy",
                ),
                _metric(
                    f"team.{team}.forward_pass_attempts",
                    team_row["forward_pass_attempts"],
                    "passes",
                    "Realized open-play passes whose submitted direction is forward in the attacking frame.",
                    quality="submitted_direction_fact",
                ),
                _metric(
                    f"team.{team}.attacking_third_pass_attempts",
                    team_row["attacking_third_pass_attempts"],
                    "passes",
                    "Realized open-play passes released from the attacking third.",
                    quality="derived_from_exact_events",
                ),
                _metric(
                    f"team.{team}.penalty_area_entries",
                    team_row["penalty_area_entries"],
                    "crossings",
                    "Live-ball attacking penalty-area front-line crossings inside its lateral bounds while retaining possession.",
                ),
                _metric(
                    f"team.{team}.corners",
                    team_row["corners"],
                    "restarts_awarded",
                    "Confirmed corner restarts awarded to this team.",
                ),
                _metric(
                    f"team.{team}.fouls_committed",
                    team_row["fouls_committed"],
                    "fouls",
                    "Confirmed foul events attributed to this team as offender.",
                ),
                _metric(
                    f"team.{team}.offsides",
                    team_row["offsides"],
                    "offside_events",
                    "Confirmed offside events attributed to this team.",
                ),
            )
        )

    substitution_by_identity: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in substitution_records:
        timing = {
            "control_tick": record["control_tick"],
            "clock_s": record["clock_s"],
            "clock_label": record["clock_label"],
        }
        outgoing = (
            record["team"],
            record["player_out"],
            record["player_out_slot_generation"],
        )
        incoming = (
            record["team"],
            record["player_in"],
            record["player_in_slot_generation"],
        )
        substitution_by_identity.setdefault(outgoing, {})["exited"] = timing
        substitution_by_identity.setdefault(incoming, {})["entered"] = timing

    players = []
    for stat in sorted(
        player_stats.values(),
        key=lambda x: (x["team"], x["player_id"], x["slot_generation"]),
    ):
        player = {
            **stat,
            "active_s": round(float(stat["active_s"]), 3),
            "distance_m": round(float(stat["distance_m"]), 3),
            "max_speed_mps": round(float(stat["max_speed_mps"]), 3),
        }
        substitution = substitution_by_identity.get(
            (player["team"], player["player_id"], player["slot_generation"])
        )
        if substitution:
            player["substitution"] = substitution
        players.append(player)
        prefix = (
            f"player.{player['team']}.{player['player_id']}."
            f"generation_{player['slot_generation']}"
        )
        metric_receipts.extend(
            (
                _metric(
                    f"{prefix}.active_seconds",
                    player["active_s"],
                    "s",
                    "Adjacent intervals with continuous player ID, slot generation, team, active state, and on-pitch state.",
                ),
                _metric(
                    f"{prefix}.distance",
                    player["distance_m"],
                    "m",
                    "Sum of adjacent tracking displacement over identity-continuous active on-pitch intervals.",
                ),
                _metric(
                    f"{prefix}.maximum_speed",
                    player["max_speed_mps"],
                    "m/s",
                    "Maximum tracking velocity norm while active and on-pitch.",
                ),
            )
        )

    pass_attempts = int(action_counts["submitted_pass"])
    receiver_identity_available = (
        event_header.get("schema") != "footballworld.events/14"
    )
    receiver_coverage = (
        None
        if not receiver_identity_available or not pass_attempts
        else round(pass_target_known / pass_attempts, 6)
    )
    windows = []
    cumulative_team_possession = np.zeros(2, dtype=np.float64)
    for index, raw_window in enumerate(chart_windows):
        team_possession = raw_window["team_possession_s"]
        cumulative_team_possession += team_possession
        cumulative_controlled_s = float(cumulative_team_possession.sum())
        windows.append(
            {
                "start_s": round(index * ANALYSIS_WINDOW_S, 3),
                "end_s": round(
                    min(captured_duration, (index + 1) * ANALYSIS_WINDOW_S), 3
                ),
                "team_possession_s": [
                    round(float(value), 3) for value in team_possession
                ],
                "cumulative_team_possession_s": [
                    round(float(value), 3) for value in cumulative_team_possession
                ],
                "loose_ball_s": round(float(raw_window["loose_ball_s"]), 3),
                "controlled_possession_share": [
                    (
                        None
                        if not cumulative_controlled_s
                        else round(
                            float(
                                cumulative_team_possession[team]
                                / cumulative_controlled_s
                            ),
                            6,
                        )
                    )
                    for team in (0, 1)
                ],
            }
        )

    speed_counts = speed_histogram.tolist()
    speed_shares = []
    for team in (0, 1):
        sample_count = int(speed_histogram[team].sum())
        speed_shares.append(
            [
                0.0 if not sample_count else round(int(value) / sample_count, 6)
                for value in speed_histogram[team]
            ]
        )
    territory_total = int(ball_territory.sum())
    territory_edges = np.linspace(-length / 2.0, length / 2.0, BALL_TERRITORY_BINS + 1)
    ball_density_rows = [
        [
            window_index,
            x_index,
            y_index,
            round(float(seconds), 3),
        ]
        for (
            window_index,
            y_index,
            x_index,
        ), seconds in sorted(ball_density_window_s.items())
        if seconds > 0.0
    ]
    ball_density_window_receipts = [
        {
            "index": int(receipt["index"]),
            "start_clock_s": round(float(receipt["start_clock_s"]), 3),
            "end_clock_s": round(float(receipt["end_clock_s"]), 3),
            "end_control_tick": int(receipt["end_control_tick"]),
            "observed_live_seconds": round(float(receipt["observed_live_seconds"]), 3),
            "excluded_out_of_pitch_seconds": round(
                float(receipt["excluded_out_of_pitch_seconds"]), 3
            ),
        }
        for _, receipt in sorted(ball_density_windows.items())
    ]
    player_position_rows = [
        [
            player_window_end_tick[(window_index, team, player_id, generation)],
            team,
            player_id,
            generation,
            x_index,
            y_index,
            round(float(seconds), 3),
        ]
        for (
            window_index,
            team,
            player_id,
            generation,
            y_index,
            x_index,
        ), seconds in sorted(player_occupancy_s.items())
        if seconds > 0.0
    ]
    player_activity_rows = [player_activity[key] for key in sorted(player_activity)]
    team_space_rows = [
        [
            window_index,
            team,
            x_index,
            y_index,
            round(float(seconds), 3),
        ]
        for (
            window_index,
            team,
            y_index,
            x_index,
        ), seconds in sorted(team_space_occupancy_s.items())
        if seconds > 0.0
    ]
    team_space_windows = [
        {
            "index": int(receipt["index"]),
            "start_clock_s": round(float(receipt["start_clock_s"]), 3),
            "end_clock_s": round(float(receipt["end_clock_s"]), 3),
            "end_control_tick": int(receipt["end_control_tick"]),
        }
        for _, receipt in sorted(space_windows.items())
    ]
    visualizations = {
        "window_s": ANALYSIS_WINDOW_S,
        "windows": windows,
        "speed_histogram": {
            "bin_starts_mps": list(range(SPEED_OVERFLOW_BIN_MPS + 1)),
            "overflow_from_mps": SPEED_OVERFLOW_BIN_MPS,
            "sample_counts": speed_counts,
            "sample_shares": speed_shares,
        },
        "ball_territory": {
            "x_edges_m": [round(float(value), 3) for value in territory_edges],
            "live_frame_counts": ball_territory.tolist(),
            "live_frame_shares": [
                0.0 if not territory_total else round(int(value) / territory_total, 6)
                for value in ball_territory
            ],
        },
        "ball_density_2d": {
            "coordinate_frame": "absolute_pitch",
            "weighting": "preceding_live_tracking_interval_seconds",
            "target_cell_m": BALL_DENSITY_TARGET_CELL_M,
            "window_s": BALL_DENSITY_WINDOW_S,
            "row_semantics": "incremental windows prefix-summed by the report from match start",
            "windows": ball_density_window_receipts,
            "columns": [
                "window_index",
                "x_index",
                "y_index",
                "seconds",
            ],
            "rows": ball_density_rows,
            "x_edges_m": [round(float(value), 3) for value in density_x_edges],
            "y_edges_m": [round(float(value), 3) for value in density_y_edges],
            "live_seconds": [
                [round(float(value), 3) for value in row] for row in ball_density_s
            ],
            "team_controlled_seconds": [
                [
                    [round(float(value), 3) for value in row]
                    for row in team_ball_density_s[team]
                ]
                for team in (0, 1)
            ],
            "loose_seconds": [
                [round(float(value), 3) for value in row]
                for row in loose_ball_density_s
            ],
            "observed_live_seconds": round(float(ball_density_s.sum()), 3),
            "excluded_out_of_pitch_seconds": round(density_out_of_pitch_s, 3),
            "display_transform": "square_root_relative_to_maximum_cell",
        },
        "player_position_occupancy": {
            "coordinate_frame": "team_attack_normalized",
            "transform": "rotate_180_when_attack_direction_is_negative",
            "weighting": "identity_continuous_active_on_pitch_seconds",
            "window_s": PLAYER_POSITION_WINDOW_S,
            "target_cell_m": PLAYER_POSITION_TARGET_CELL_M,
            "x_edges_m": [round(float(value), 3) for value in player_density_x_edges],
            "y_edges_m": [round(float(value), 3) for value in player_density_y_edges],
            "columns": [
                "window_end_control_tick",
                "team",
                "player_id",
                "slot_generation",
                "x_index",
                "y_index",
                "seconds",
            ],
            "rows": player_position_rows,
            "activity": player_activity_rows,
        },
        "team_space_occupancy": {
            "coordinate_frame": "team_0_attack_normalized",
            "transform": "rotate_180_when_team_0_attack_direction_is_negative",
            "weighting": "identity_continuous_active_on_pitch_live_ball_player_seconds",
            "comparison": "match-to-date cumulative per-team normalized player-position density difference",
            "row_semantics": "incremental windows prefix-summed by the report from match start",
            "interpretation": "spatial occupancy, not modeled territorial control or possession probability",
            "window_s": SPACE_OCCUPANCY_WINDOW_S,
            "target_cell_m": SPACE_OCCUPANCY_TARGET_CELL_M,
            "x_edges_m": [round(float(value), 3) for value in space_x_edges],
            "y_edges_m": [round(float(value), 3) for value in space_y_edges],
            "windows": team_space_windows,
            "columns": [
                "window_index",
                "team",
                "x_index",
                "y_index",
                "player_seconds",
            ],
            "rows": team_space_rows,
        },
        "intent_actions": {
            "coordinate_frame": "team_attack_normalized",
            "attack_direction": "positive_x",
            "scope": "submitted_policy_actions",
            "direction_semantics": "L2-unit submitted force_to_ball in team attack-normalized xy; null when unavailable or zero",
            "intent_names": {"2": "pass", "3": "shot", "4": "clear", "5": "challenge"},
            "rows": intent_action_rows,
        },
        "pass_map": {
            "coordinate_frame": "team_attack_normalized",
            "attack_direction": "positive_x",
            "scope": "realized_open_play_pass_contacts",
            "outcome_semantics": "next_contact_by_distinct_actor_before_boundary",
            "applied_direction_semantics": "L2-unit submitted force_to_ball in the team attack-normalized frame, retained only when the deliberate PASS kick was applied",
            "receipt_geometry_semantics": "direction from exact source contact to the next distinct-actor contact before a boundary; this is not kick direction",
            "cross_control_signature_semantics": "policy-specific exact match of retained submitted spin: absolute side spin 0.18 and back spin 0.35 within absolute tolerance 1e-6; this identifies the shipped rule policy control signature and is not a generic or provider cross label",
            "defensive_line_breaking_proxy_semantics": "source-to-next-distinct-contact progress of at least 5 m, starting at or behind and ending at least 0.5 m beyond the second-last active opponent's source-time attack-normalized x; this geometry proxy is not a through-pass event label",
            "rows": pass_map_rows,
        },
        "shot_map": {
            "coordinate_frame": "team_attack_normalized",
            "attack_direction": "positive_x",
            "scope": "applied_shot_contacts_with_exact_source_position",
            "on_target_semantics": "shooting-team goal boundary or opponent contact labelled LAW11_DELIBERATE_SAVE_NO_RESET",
            "off_target_semantics": "resolved shot flight not classified as goal or deliberate save; resolution preserves block, woodwork, rebound, retouch, own-goal, and boundary detail",
            "unresolved_semantics": "no later contact or boundary was captured; never coerced to off target",
            "preceding_event_semantics": "for a known continuous attack, up to seven positioned nodes from strictly earlier control ticks: the optional sequence start plus the latest six exact same-team contacts; current-shot and same-tick ordering are excluded fail-closed; a bounded tracking-lookback fallback has no event nodes",
            "rows": shot_map_rows,
        },
        "shot_context": shot_context_summary,
    }
    metric_receipts.extend(
        (
            _metric(
                "match.captured_duration",
                round(captured_duration, 3),
                "s",
                "Inclusive duration of the captured tracking control-frame interval.",
            ),
            _metric(
                "match.live_ball_seconds",
                round(live_s, 3),
                "s",
                "Adjacent intervals whose preceding tracking frame marks the ball live.",
            ),
            _metric(
                "match.loose_ball_seconds",
                round(loose_s, 3),
                "s",
                "Live intervals with no team possession attribution.",
            ),
            _metric(
                "actions.submitted_passes",
                pass_attempts,
                "actions",
                "Retained action rows whose submitted categorical intent is PASS; realization is not implied.",
                quality="submitted_fact_count",
            ),
            _metric(
                "actions.intended_receiver_coverage",
                receiver_coverage,
                "ratio_of_submitted_passes",
                "Submitted PASS rows with a non-null registered intended receiver ID.",
            ),
            _metric(
                "series.controlled_possession_share",
                [row["controlled_possession_share"] for row in windows],
                "cumulative_ratio_sampled_every_30_seconds",
                "Full-match-to-date team-controlled possession share sampled at fixed capture-relative checkpoints; loose-ball time is excluded.",
            ),
            _metric(
                "distribution.player_speed_samples",
                speed_counts,
                "active_on_pitch_tracking_samples",
                "Counts of active on-pitch player speed samples in one-metre-per-second bins, with the final bin containing all samples at or above 12 m/s.",
            ),
            _metric(
                "distribution.ball_longitudinal_territory",
                ball_territory.tolist(),
                "live_tracking_frames",
                "Counts of live-ball tracking frames in six equal absolute-pitch longitudinal zones from negative to positive x.",
            ),
            _metric(
                "distribution.ball_position_density.observed_seconds",
                round(float(ball_density_s.sum()), 3),
                "s",
                "Sum of preceding-frame live tracking intervals assigned to in-pitch two-dimensional ball-position cells.",
            ),
            _metric(
                "distribution.team_space_occupancy.observed_player_seconds",
                round(float(sum(team_space_occupancy_s.values())), 3),
                "live_ball_player_seconds",
                "Identity-continuous active on-pitch player time assigned to team-0-attack-normalized spatial cells in fixed capture-relative windows.",
            ),
            _metric(
                "actions.realized_open_play_pass_map_records",
                len(pass_map_rows),
                "passes",
                "Realized deliberate open-play PASS contacts with an exact source position and known team attack direction; next-contact outcomes remain receipt proxies.",
            ),
            _metric(
                "actions.realized_shot_map_records",
                len(shot_map_rows),
                "shots",
                "Applied SHOT contacts with exact source positions and causal goal, deliberate-save, other resolved, or unresolved outcomes.",
                quality="derived_from_exact_events",
            ),
        )
    )
    quality_warnings = list(dataset.warnings)
    if legacy_stadium_width:
        quality_warnings.append(
            "Legacy replay metadata omitted stadium width; assuming 68 metres."
        )
    if not action_controls_available:
        quality_warnings.append(
            "Submitted action controls are unavailable; submitted action counts are not observable."
        )
    for anomaly in policy_anomalies:
        observed = anomaly["observed"]
        if anomaly["code"] == "stationary_live_loose_ball":
            quality_warnings.append(
                "Policy anomaly [high]: stationary live loose ball for "
                f"{float(observed['duration_s']):.1f}s from match clock "
                f"{float(observed['start_clock_s']):.1f}s to "
                f"{float(observed['end_clock_s']):.1f}s near "
                f"{observed['position_m']}."
            )
        elif anomaly["code"] == "prolonged_live_loose_ball":
            quality_warnings.append(
                "Policy anomaly [medium]: live ball remained loose for "
                f"{float(observed['duration_s']):.1f}s from match clock "
                f"{float(observed['start_clock_s']):.1f}s to "
                f"{float(observed['end_clock_s']):.1f}s."
            )
        elif anomaly["code"] == "ball_position_overconcentration":
            quality_warnings.append(
                "Policy anomaly [medium]: one 2 m ball-position cell "
                f"({observed['x_bounds_m']} x {observed['y_bounds_m']}) contains "
                f"{float(observed['seconds']):.1f}s "
                f"({100.0 * float(observed['share']):.1f}%) of observed live time."
            )
        elif anomaly["code"] == "pass_completion_time_trend_drop":
            quality_warnings.append(
                "Policy anomaly [medium]: Team "
                f"{observed['team']} weighted 15-minute pass receipt trend implies "
                f"a {100.0 * -float(observed['fitted_first_to_last_change']):.1f}%p "
                "fall across the observed match windows."
            )
        elif anomaly["code"] == "excessive_dismissals":
            quality_warnings.append(
                "Policy anomaly [high]: Team "
                f"{observed['team']} accumulated {observed['dismissals']} dismissals."
            )
        elif anomaly["code"] == "repeated_event_pattern":
            quality_warnings.append(
                "Policy anomaly [medium]: repeated "
                f"{observed['event_type']} signature persisted for "
                f"{float(observed['duration_s']):.1f}s "
                f"({observed['occurrences']} occurrences)."
            )
        elif anomaly["code"] == "mid_match_pass_activity_collapse":
            quality_warnings.append(
                "Policy anomaly [high]: only "
                f"{observed['combined_attempts']} realized passes occurred from "
                f"minute {observed['start_minute']} to {observed['end_minute']}, "
                "despite active neighboring 15-minute windows."
            )
    report = {
        "schema": REPORT_SCHEMA,
        "metrics_schema": METRICS_VERSION,
        "source": {
            "replay_root": str(dataset.replay.root),
            "video": dataset.replay.video.name,
            "event_schema": event_header["schema"],
            "tracking_schema": dataset.metadata.get("tracking_storage", {}).get(
                "semantic_schema"
            ),
            "metadata_schema": dataset.metadata.get("schema"),
            "artifact_receipts": dataset.replay.output.get("artifacts"),
            "policies": (
                dataset.metadata.get("user_metadata", {})
                .get("provenance", {})
                .get("policies")
            ),
            "capture_source": dataset.replay.completion.get("production_source"),
            "git_revision": (
                dataset.replay.completion.get("publication_guard", {})
                .get("git_start", {})
                .get("revision")
            ),
        },
        "quality": {
            "authoritative": dataset.authoritative,
            "hashes_verified": dataset.hashes_verified,
            "full_duration_complete": bool(
                dataset.replay.completion.get("full_duration_complete")
            ),
            "event_budget_exhausted_count": int(
                dataset.replay.completion.get("event_budget_exhausted_count", 0)
            ),
            "policy_audit": {
                "basis": "host-only diagnostic priors, not measured football constants",
                "anomalies": policy_anomalies,
                "longest_live_loose_ball_run": longest_loose_run,
                "longest_stationary_live_loose_ball_run": (
                    longest_stationary_loose_run
                ),
                "dominant_ball_density_cell": dominant_density,
                "dismissals": dismissal_records,
                "longest_repeated_event_run": longest_repeated_event,
                "pass_completion_by_15m": pass_completion_by_team,
            },
            "warnings": quality_warnings,
        },
        "summary": {
            "score": final_score,
            "captured_duration_s": round(captured_duration, 3),
            "tracking_frames": frame_count,
            "live_ball_s": round(live_s, 3),
            "loose_ball_s": round(loose_s, 3),
        },
        "teams": teams,
        "players": players,
        "events": {
            "exact_event_counts": dict(sorted(event_counts.items())),
            "submitted_action_counts": dict(sorted(action_counts.items())),
            "submitted_action_counts_available": bool(action_controls_available),
            "intended_receiver_known": pass_target_known,
            "intended_receiver_identity_available": receiver_identity_available,
            "intended_receiver_coverage": receiver_coverage,
        },
        "visualizations": visualizations,
        "policy_alignment": policy_alignment,
        "timeline": timeline,
        "metric_receipts": metric_receipts,
        "match_manifest": dataset.match_manifest,
    }
    return report


__all__ = ["METRICS_VERSION", "REPORT_SCHEMA", "build_match_report"]
