"""Single-transfer host boundary for rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.environment.management import (
    ActingGoalkeeperEvent,
    SubstitutionEvent,
)


@dataclass(slots=True)
class HostFrame:
    """One selected match frame represented entirely by NumPy values."""

    control_tick: int
    dead_ball_control_ticks: int
    first_half_wall_end_tick: int
    first_half_live_extension_ticks: int
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_spin: np.ndarray
    ball_live: bool
    player_position: np.ndarray
    player_velocity: np.ndarray
    player_body_forward: np.ndarray
    player_gaze_yaw: np.ndarray
    player_height: np.ndarray
    aerial_recovery_substeps: np.ndarray
    team_id: np.ndarray
    player_id: np.ndarray
    slot_generation: np.ndarray
    active: np.ndarray
    on_pitch: np.ndarray
    sent_off: np.ndarray
    is_goalkeeper: np.ndarray
    stamina_long: np.ndarray
    stamina_short: np.ndarray
    yellow_cards: np.ndarray
    attack_direction: np.ndarray
    kickoff_team: int
    possession_team: int
    possession_player: int
    possession_previous_team: int
    possession_control_ticks: int
    last_contact: Any
    restart_kind: int
    restart_team: int
    restart_substeps_remaining: int
    restart_taker: int
    restart_indirect: bool
    score: np.ndarray
    offside_flagged: np.ndarray
    submitted_action: Any
    action_trace: Any
    frame_events: Any
    substitution_events: Any
    acting_goalkeeper_events: Any
    observation: Any
    telemetry: Any
    # Host-only causal identity retained when a rare manager transaction replaces
    # this frame's post-transition roster state. Ordinary frames keep ``None``;
    # this field never enters a JAX payload or compiled transition.
    pre_management_identity: Any = None
    # Exact-event rendering may sample more frequently than the control grid.
    # Keep this host-only timestamp separate from the authoritative integer
    # control tick used by tracking and event sidecars.
    video_time_s: float | None = None


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/NamedTuple payloads without losing fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {key: to_jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def _state_of(value: Any) -> Any:
    if hasattr(value, "rollout"):
        return value.rollout.state
    if hasattr(value, "state") and hasattr(value, "offside"):
        return value.state
    return value


def _offside_of(value: Any) -> Any | None:
    if hasattr(value, "rollout"):
        return getattr(value.rollout, "offside", None)
    return getattr(value, "offside", None)


def _telemetry_of(value: Any) -> Any | None:
    if not hasattr(value, "rollout"):
        return None
    names = (
        "outcome",
        "contact_attempted",
        "kick_applied",
        "restart_opened",
        "event_budget_exhausted",
        "contest_override_valid",
        "terminated",
        "truncated",
        "done",
        "terminal_frozen",
        "halftime_reset",
    )
    return {name: getattr(value, name) for name in names if hasattr(value, name)}


def _is_snapshot(value: Any) -> bool:
    state = _state_of(value)
    return hasattr(state, "ball") and hasattr(state, "players")


def _stack_rows(rows: list[Any]) -> Any:
    if rows[0] is None:
        if not all(row is None for row in rows):
            raise ValueError("sidecar rows must be consistently present")
        return None
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def _stacked_input(states: Any) -> tuple[Any, bool, int]:
    if _is_snapshot(states):
        state = _state_of(states)
        rank = state.players.position.ndim
        if rank == 2:
            return jax.tree_util.tree_map(lambda x: x[None], states), False, 1
        if rank == 3:
            return states, False, int(state.players.position.shape[0])
        if rank == 4:
            return states, True, int(state.players.position.shape[0])
        raise ValueError("positions require [N,2], [T,N,2], or [T,B,N,2]")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise TypeError("states must be State/Rollout/StepResult or a sequence")
    rows = list(states)
    if not rows or not all(_is_snapshot(row) for row in rows):
        raise ValueError("states must contain rollout frames")
    rank = _state_of(rows[0]).players.position.ndim
    if rank not in (2, 3):
        raise ValueError("sequence positions require [N,2] or [B,N,2]")
    if any(_state_of(row).players.position.ndim != rank for row in rows):
        raise ValueError("all frames must use the same batch rank")
    return _stack_rows(rows), rank == 3, len(rows)


def _stack_sidecar(value: Any, count: int) -> Any:
    if value is None:
        return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and not hasattr(value, "_fields")
    ):
        rows = list(value)
        if len(rows) != count:
            raise ValueError("sidecar frame count does not match states")
        return _stack_rows(rows)
    leaves = jax.tree_util.tree_leaves(value)
    lengths = [int(x.shape[0]) for x in leaves if hasattr(x, "ndim") and x.ndim]
    if not lengths or any(length != count for length in lengths):
        raise ValueError("stacked sidecar requires the same [T,...] axis")
    return value


def _submitted_action_sidecar(value: Any, count: int) -> Any:
    """Normalize one IntentAction or its time-major stacked PyTree."""

    if value is None:
        return None
    if count == 1 and hasattr(value, "intent") and value.intent.ndim == 1:
        value = jax.tree_util.tree_map(lambda leaf: leaf[None], value)
    return _stack_sidecar(value, count)


def _slot_generation_sidecar(value: Any, count: int) -> Any:
    """Extract only generation leaves from raw arrays or manager states."""

    if value is None:
        return None
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and not hasattr(value, "_fields")
    ):
        value = [getattr(row, "slot_generation", row) for row in value]
    else:
        value = getattr(value, "slot_generation", value)
    if count == 1 and hasattr(value, "ndim") and value.ndim == 1:
        value = value[None]
    return _stack_sidecar(value, count)


def _as_substitution_event(value: Any) -> SubstitutionEvent:
    """Normalize exact event mappings and result payloads to one PyTree."""

    value = getattr(
        value,
        "substitution_events",
        getattr(value, "substitution_event", value),
    )
    if isinstance(value, Mapping):
        try:
            return SubstitutionEvent(
                *(value[name] for name in SubstitutionEvent._fields)
            )
        except KeyError as exc:
            raise ValueError(
                f"substitution event mapping is missing {exc.args[0]!r}"
            ) from exc
    if not isinstance(value, SubstitutionEvent):
        raise TypeError("substitution events must use SubstitutionEvent fields")
    return value


def _substitution_event_sidecar(value: Any, count: int) -> Any:
    """Stack exact events while allowing None on frames without a command."""

    if value is None:
        return None

    def event_of(row: Any) -> Any:
        if row is None:
            return None
        return _as_substitution_event(row)

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and not hasattr(value, "_fields")
    ):
        rows = [event_of(row) for row in value]
        if len(rows) != count:
            raise ValueError("substitution event frame count does not match states")
        template = next((row for row in rows if row is not None), None)
        if template is None:
            return None
        shape = tuple(jnp.shape(template.occurred))
        empty = SubstitutionEvent.empty(shape)
        return _stack_rows([empty if row is None else row for row in rows])
    event = event_of(value)
    if count == 1 and event.occurred.ndim in (0, 2):
        event = jax.tree_util.tree_map(lambda x: x[None], event)
    return _stack_sidecar(event, count)


def _as_acting_goalkeeper_event(value: Any) -> ActingGoalkeeperEvent:
    """Normalize exact event mappings and result payloads to one PyTree."""

    value = getattr(
        value,
        "acting_goalkeeper_events",
        getattr(value, "acting_goalkeeper_event", value),
    )
    if isinstance(value, Mapping):
        try:
            return ActingGoalkeeperEvent(
                *(value[name] for name in ActingGoalkeeperEvent._fields)
            )
        except KeyError as exc:
            raise ValueError(
                f"acting goalkeeper event mapping is missing {exc.args[0]!r}"
            ) from exc
    if not isinstance(value, ActingGoalkeeperEvent):
        raise TypeError(
            "acting goalkeeper events must use ActingGoalkeeperEvent fields"
        )
    return value


def _acting_goalkeeper_event_sidecar(value: Any, count: int) -> Any:
    """Stack exact events while allowing None on frames without a command."""

    if value is None:
        return None

    def event_of(row: Any) -> Any:
        if row is None:
            return None
        return _as_acting_goalkeeper_event(row)

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, np.ndarray))
        and not hasattr(value, "_fields")
    ):
        rows = [event_of(row) for row in value]
        if len(rows) != count:
            raise ValueError(
                "acting goalkeeper event frame count does not match states"
            )
        template = next((row for row in rows if row is not None), None)
        if template is None:
            return None
        shape = tuple(jnp.shape(template.occurred))
        empty = ActingGoalkeeperEvent.empty(shape)
        return _stack_rows([empty if row is None else row for row in rows])
    event = event_of(value)
    if count == 1 and event.occurred.ndim in (0, 1):
        event = jax.tree_util.tree_map(lambda x: x[None], event)
    return _stack_sidecar(event, count)


def _validate_substitution_event_axes(
    value: SubstitutionEvent | None,
    *,
    count: int,
    batched: bool,
    batch_size: int,
) -> None:
    """Reject ambiguous team/batch axes before selecting one match."""

    if value is None:
        return
    occurred = value.occurred
    expected_ranks = (2, 4) if batched else (1, 3)
    if occurred.ndim not in expected_ranks:
        layout = "[T,B] or [T,B,2,K]" if batched else "[T] or [T,2,K]"
        raise ValueError(f"substitution occurred must use {layout}")
    if occurred.shape[0] != count:
        raise ValueError("substitution event frame axis does not match states")
    if batched and occurred.shape[1] != batch_size:
        raise ValueError("substitution event batch axis does not match states")
    team_axis = 2 if batched else 1
    command_rank = 4 if batched else 3
    if occurred.ndim == command_rank and occurred.shape[team_axis] != 2:
        raise ValueError("substitution command events require exactly two teams")
    if any(leaf.shape != occurred.shape for leaf in jax.tree_util.tree_leaves(value)):
        raise ValueError("substitution event fields must share one fixed shape")


def _validate_acting_goalkeeper_event_axes(
    value: ActingGoalkeeperEvent | None,
    *,
    count: int,
    batched: bool,
    batch_size: int,
) -> None:
    """Reject ambiguous team/batch axes before selecting one match."""

    if value is None:
        return
    occurred = value.occurred
    expected_ranks = (2, 3) if batched else (1, 2)
    if occurred.ndim not in expected_ranks:
        layout = "[T,B] or [T,B,2]" if batched else "[T] or [T,2]"
        raise ValueError(f"acting goalkeeper occurred must use {layout}")
    if occurred.shape[0] != count:
        raise ValueError("acting goalkeeper event frame axis does not match states")
    if batched and occurred.shape[1] != batch_size:
        raise ValueError("acting goalkeeper event batch axis does not match states")
    team_axis = 2 if batched else 1
    command_rank = 3 if batched else 2
    if occurred.ndim == command_rank and occurred.shape[team_axis] != 2:
        raise ValueError("acting goalkeeper command events require exactly two teams")
    if any(leaf.shape != occurred.shape for leaf in jax.tree_util.tree_leaves(value)):
        raise ValueError("acting goalkeeper event fields must share one fixed shape")


def _select(tree: Any, match_index: int, batched: bool) -> Any:
    if tree is None or not batched:
        return tree

    def take(value: Any) -> Any:
        if not hasattr(value, "ndim"):
            return value
        if value.ndim < 2 or not 0 <= match_index < value.shape[1]:
            raise ValueError("batched sidecars must match state [T,B,...] axes")
        return value[:, match_index]

    return jax.tree_util.tree_map(take, tree)


def _payload(
    stacked: Any,
    events: Any,
    observations: Any,
    slot_generations: Any,
    substitution_events: Any,
    acting_goalkeeper_events: Any,
    submitted_actions: Any,
) -> dict[str, Any]:
    if events is None:
        events = getattr(stacked, "frame_events", getattr(stacked, "events", None))
    if observations is None:
        observations = getattr(stacked, "observation", None)
    action_trace = getattr(stacked, "action_trace", None)
    state = _state_of(stacked)
    offside = _offside_of(stacked)
    p = state.players
    s = state.possession
    r = state.restart
    count, players = p.position.shape[:2]
    if slot_generations is None:
        slot_generations = np.full(p.player_id.shape, -1, dtype=np.int32)
    else:
        if slot_generations.shape != p.player_id.shape:
            raise ValueError("slot_generations must have shape [T, N]")
        if slot_generations.dtype != jnp.int32:
            raise TypeError("slot_generations must have int32 dtype")
    return {
        "control_tick": state.control_tick,
        "dead_ball_control_ticks": state.dead_ball_control_ticks,
        "first_half_wall_end_tick": state.first_half_wall_end_tick,
        "first_half_live_extension_ticks": (state.first_half_live_extension_ticks),
        "ball_position": state.ball.position,
        "ball_velocity": state.ball.velocity,
        "ball_spin": state.ball.spin,
        "ball_live": state.ball.live,
        "player_position": p.position,
        "player_velocity": p.velocity,
        "player_body_forward": p.body_forward,
        "player_gaze_yaw": p.gaze_yaw,
        "player_height": p.height,
        "aerial_recovery_substeps": p.aerial_recovery_substeps,
        "team_id": p.team_id,
        "player_id": p.player_id,
        "slot_generation": slot_generations,
        "active": p.active,
        "on_pitch": p.on_pitch,
        "sent_off": p.sent_off,
        "is_goalkeeper": p.is_goalkeeper,
        "stamina_long": p.stamina_long,
        "stamina_short": p.stamina_short,
        "yellow_cards": p.yellow_cards,
        "attack_direction": state.attack_direction,
        "kickoff_team": state.kickoff_team,
        "possession_team": s.team,
        "possession_player": s.player,
        "possession_previous_team": s.previous_team,
        "possession_control_ticks": s.control_ticks,
        "last_contact": s.last_contact,
        "restart_kind": r.kind,
        "restart_team": r.team,
        "restart_substeps_remaining": r.substeps_remaining,
        "restart_taker": r.taker,
        "restart_indirect": r.indirect,
        "score": state.score,
        "offside_flagged": offside.flagged
        if offside is not None
        else np.zeros((count, players), bool),
        "submitted_action": submitted_actions,
        "action_trace": action_trace,
        "frame_events": events,
        "substitution_events": substitution_events,
        "acting_goalkeeper_events": acting_goalkeeper_events,
        "observation": observations,
        "telemetry": _telemetry_of(stacked),
    }


def _at(tree: Any, index: int) -> Any:
    if tree is None:
        return None
    return jax.tree_util.tree_map(
        lambda x: x[index] if hasattr(x, "ndim") and x.ndim else x, tree
    )


def prepare_host_frames(
    states: Any,
    *,
    frame_events: Any = None,
    observations: Any = None,
    slot_generations: Any = None,
    substitution_events: Any = None,
    acting_goalkeeper_events: Any = None,
    submitted_actions: Any = None,
    match_index: int = 0,
) -> list[HostFrame]:
    """Select one match once, device_get once, then split on the host."""
    if isinstance(match_index, bool) or not isinstance(match_index, int):
        raise TypeError("match_index must be an integer")
    leaves = jax.tree_util.tree_leaves(
        (
            states,
            frame_events,
            observations,
            slot_generations,
            substitution_events,
            acting_goalkeeper_events,
            submitted_actions,
        )
    )
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        raise RuntimeError("render_mp4 is host-only; call it outside jax.jit/vmap/scan")
    stacked, batched, count = _stacked_input(states)
    if not batched and match_index != 0:
        raise IndexError("unbatched rendering only has match_index=0")
    events = _stack_sidecar(frame_events, count)
    observations = _stack_sidecar(observations, count)
    slot_generations = _slot_generation_sidecar(slot_generations, count)
    substitution_events = _substitution_event_sidecar(substitution_events, count)
    acting_goalkeeper_events = _acting_goalkeeper_event_sidecar(
        acting_goalkeeper_events, count
    )
    submitted_actions = _submitted_action_sidecar(submitted_actions, count)
    state = _state_of(stacked)
    batch_size = int(state.players.position.shape[1]) if batched else 1
    _validate_substitution_event_axes(
        substitution_events,
        count=count,
        batched=batched,
        batch_size=batch_size,
    )
    _validate_acting_goalkeeper_event_axes(
        acting_goalkeeper_events,
        count=count,
        batched=batched,
        batch_size=batch_size,
    )
    selected = _select(stacked, match_index, batched)
    events = _select(events, match_index, batched)
    observations = _select(observations, match_index, batched)
    slot_generations = _select(slot_generations, match_index, batched)
    substitution_events = _select(substitution_events, match_index, batched)
    acting_goalkeeper_events = _select(acting_goalkeeper_events, match_index, batched)
    submitted_actions = _select(submitted_actions, match_index, batched)
    host = jax.device_get(
        _payload(
            selected,
            events,
            observations,
            slot_generations,
            substitution_events,
            acting_goalkeeper_events,
            submitted_actions,
        )
    )
    result = []
    for i in range(count):
        x = {key: _at(value, i) for key, value in host.items()}
        result.append(
            HostFrame(
                control_tick=int(np.asarray(x["control_tick"])),
                dead_ball_control_ticks=int(np.asarray(x["dead_ball_control_ticks"])),
                first_half_wall_end_tick=int(np.asarray(x["first_half_wall_end_tick"])),
                first_half_live_extension_ticks=int(
                    np.asarray(x["first_half_live_extension_ticks"])
                ),
                ball_position=np.asarray(x["ball_position"], np.float32),
                ball_velocity=np.asarray(x["ball_velocity"], np.float32),
                ball_spin=np.asarray(x["ball_spin"], np.float32),
                ball_live=bool(np.asarray(x["ball_live"])),
                player_position=np.asarray(x["player_position"], np.float32),
                player_velocity=np.asarray(x["player_velocity"], np.float32),
                player_body_forward=np.asarray(x["player_body_forward"], np.float32),
                player_gaze_yaw=np.asarray(x["player_gaze_yaw"], np.float32),
                player_height=np.asarray(x["player_height"], np.float32),
                aerial_recovery_substeps=np.asarray(
                    x["aerial_recovery_substeps"], np.int32
                ),
                team_id=np.asarray(x["team_id"], np.int32),
                player_id=np.asarray(x["player_id"], np.int64),
                slot_generation=np.asarray(x["slot_generation"], np.int32),
                active=np.asarray(x["active"], bool),
                on_pitch=np.asarray(x["on_pitch"], bool),
                sent_off=np.asarray(x["sent_off"], bool),
                is_goalkeeper=np.asarray(x["is_goalkeeper"], bool),
                stamina_long=np.asarray(x["stamina_long"], np.float32),
                stamina_short=np.asarray(x["stamina_short"], np.float32),
                yellow_cards=np.asarray(x["yellow_cards"], np.int32),
                attack_direction=np.asarray(x["attack_direction"], np.float32),
                kickoff_team=int(np.asarray(x["kickoff_team"])),
                possession_team=int(np.asarray(x["possession_team"])),
                possession_player=int(np.asarray(x["possession_player"])),
                possession_previous_team=int(np.asarray(x["possession_previous_team"])),
                possession_control_ticks=int(np.asarray(x["possession_control_ticks"])),
                last_contact=x["last_contact"],
                restart_kind=int(np.asarray(x["restart_kind"])),
                restart_team=int(np.asarray(x["restart_team"])),
                restart_substeps_remaining=int(
                    np.asarray(x["restart_substeps_remaining"])
                ),
                restart_taker=int(np.asarray(x["restart_taker"])),
                restart_indirect=bool(np.asarray(x["restart_indirect"])),
                score=np.asarray(x["score"], np.int32),
                offside_flagged=np.asarray(x["offside_flagged"], bool),
                submitted_action=x["submitted_action"],
                action_trace=x["action_trace"],
                frame_events=x["frame_events"],
                substitution_events=x["substitution_events"],
                acting_goalkeeper_events=x["acting_goalkeeper_events"],
                observation=x["observation"],
                telemetry=x["telemetry"],
            )
        )
    return result


__all__ = ["HostFrame", "prepare_host_frames", "to_jsonable"]
