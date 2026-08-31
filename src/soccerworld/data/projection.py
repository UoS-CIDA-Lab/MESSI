"""Pure JAX projections from engine outputs to public training records."""

from __future__ import annotations

import jax.numpy as jnp

from soccerworld._engine.constants import (
    ACTION_MAX,
    ACTION_MIN,
    FORMATION_DECISION_APPLIED,
    FORMATION_DECISION_OUT_OF_RANGE,
    FORMATION_DECISION_TERMINAL,
    FORMATION_DECISION_UNCHANGED,
    SUB_DECISION_APPLIED,
    SUB_DECISION_BALL_LIVE,
    SUB_DECISION_BENCH_EMPTY,
    SUB_DECISION_BENCH_RANGE,
    SUB_DECISION_GK_ROLE,
    SUB_DECISION_NO_CARD,
    SUB_DECISION_NOT_REQUESTED,
    SUB_DECISION_PLACEMENT,
    SUB_DECISION_SLOT_ALREADY_CHANGED,
    SUB_DECISION_SLOT_INACTIVE,
    SUB_DECISION_SLOT_RANGE,
    SUB_DECISION_TERMINAL,
    SUB_DECISION_WINDOW_BUDGET,
    SUB_DECISION_WRONG_TEAM,
)
from soccerworld.core.commands import StepCommand
from soccerworld.core.results import (
    ActionResult,
    CommandReason,
    CommandResult,
    DecisionSource,
    FormationResult,
    SetPieceTakerResult,
    SubstitutionResult,
)
from soccerworld.data.capture import CaptureSpec
from soccerworld.data.events import EventBatch, events_from_state
from soccerworld.data.records import (
    ActionProvenance,
    FrameIdentity,
    ManagerProvenance,
    TransitionRecord,
)

_SUBSTITUTION_REASON = jnp.asarray(
    [
        CommandReason.APPLIED,
        CommandReason.NOT_REQUESTED,
        CommandReason.NOT_DEAD_BALL,
        CommandReason.TERMINAL,
        CommandReason.INVALID_SLOT,
        CommandReason.BENCH_UNAVAILABLE,
        CommandReason.BENCH_UNAVAILABLE,
        CommandReason.WRONG_TEAM,
        CommandReason.INACTIVE_PLAYER,
        CommandReason.ALREADY_CHANGED,
        CommandReason.GOALKEEPER_CONSTRAINT,
        CommandReason.SUBSTITUTION_LIMIT_EXHAUSTED,
        CommandReason.SUBSTITUTION_WINDOW_EXHAUSTED,
        CommandReason.PLACEMENT_FAILED,
    ],
    dtype=jnp.int32,
)

_FORMATION_REASON = jnp.asarray(
    [
        CommandReason.APPLIED,
        CommandReason.UNCHANGED,
        CommandReason.INVALID_FORMATION,
        CommandReason.TERMINAL,
    ],
    dtype=jnp.int32,
)

assert SUB_DECISION_APPLIED == 0
assert SUB_DECISION_NOT_REQUESTED == 1
assert SUB_DECISION_BALL_LIVE == 2
assert SUB_DECISION_TERMINAL == 3
assert SUB_DECISION_SLOT_RANGE == 4
assert SUB_DECISION_BENCH_RANGE == 5
assert SUB_DECISION_BENCH_EMPTY == 6
assert SUB_DECISION_WRONG_TEAM == 7
assert SUB_DECISION_SLOT_INACTIVE == 8
assert SUB_DECISION_SLOT_ALREADY_CHANGED == 9
assert SUB_DECISION_GK_ROLE == 10
assert SUB_DECISION_NO_CARD == 11
assert SUB_DECISION_WINDOW_BUDGET == 12
assert SUB_DECISION_PLACEMENT == 13
assert FORMATION_DECISION_APPLIED == 0
assert FORMATION_DECISION_UNCHANGED == 1
assert FORMATION_DECISION_OUT_OF_RANGE == 2
assert FORMATION_DECISION_TERMINAL == 3


def _source(shape, external: bool):
    value = DecisionSource.EXTERNAL if external else DecisionSource.INTERNAL
    return jnp.full(shape, value, dtype=jnp.int32)


def _decision_source(requested, external: bool):
    if not external:
        return _source(requested.shape, False)
    return jnp.where(
        requested,
        jnp.int32(DecisionSource.EXTERNAL),
        jnp.int32(DecisionSource.INTERNAL),
    )


def _mapped_reason(code, table):
    safe = jnp.clip(code, 0, table.shape[0] - 1)
    return table[safe]


def command_result_from_info(
    command: StepCommand,
    info,
    *,
    external_substitutions: bool,
    external_formations: bool,
    external_takers: bool,
) -> CommandResult:
    """Build stable public provenance from one engine transition."""

    submitted = jnp.asarray(command.player_actions)
    normalized = jnp.clip(
        jnp.where(jnp.isfinite(submitted), submitted, 0.0),
        ACTION_MIN,
        ACTION_MAX,
    )
    agency = info.get("bc_action_mask", jnp.ones_like(submitted, dtype=jnp.bool_))
    parameter_consumed = info.get("parameter_consumed_mask", agency[:, 3:])
    kick_forced = info.get(
        "kick_forced", jnp.zeros((submitted.shape[0],), dtype=jnp.bool_)
    )
    consumed_mask = agency.at[:, 0].set(agency[:, 0] & (~kick_forced))
    consumed_mask = consumed_mask.at[:, 3:].set(parameter_consumed)
    action_reason = jnp.where(
        consumed_mask,
        jnp.int32(CommandReason.APPLIED),
        jnp.int32(CommandReason.ACTION_MASKED),
    )
    actions = ActionResult(
        source=_source((submitted.shape[0],), True),
        submitted=submitted,
        normalized=normalized,
        applied=jnp.where(
            consumed_mask, normalized, jnp.zeros_like(normalized)
        ),
        agency=agency,
        consumed=consumed_mask,
        reason=action_reason,
    )

    sub_code = info.get(
        "substitution_decision_code",
        jnp.full_like(command.substitutions.out_slot, SUB_DECISION_NOT_REQUESTED),
    )
    sub_out = info.get("substitution_proposed_out_slot", command.substitutions.out_slot)
    sub_bench = info.get(
        "substitution_proposed_bench_index", command.substitutions.bench_index
    )
    sub_source = _decision_source(
        command.substitutions.requested, external_substitutions
    )
    sub_requested = (sub_source == DecisionSource.EXTERNAL) | (sub_out >= 0)
    sub_applied = info.get("substitution_applied", jnp.zeros_like(sub_requested))
    substitutions = SubstitutionResult(
        source=sub_source,
        requested=sub_requested,
        accepted=sub_applied,
        applied=sub_applied,
        out_slot=sub_out,
        bench_index=sub_bench,
        reason=jnp.where(
            sub_requested,
            _mapped_reason(sub_code, _SUBSTITUTION_REASON),
            jnp.int32(CommandReason.NOT_REQUESTED),
        ),
    )

    form_code = info.get(
        "formation_decision_code",
        jnp.full_like(command.formations.layout_index, FORMATION_DECISION_UNCHANGED),
    )
    form_layout = info.get("formation_proposed_layout", command.formations.layout_index)
    form_source = _decision_source(command.formations.requested, external_formations)
    manager_called = info.get("manager_called", jnp.bool_(False))
    form_requested = (form_source == DecisionSource.EXTERNAL) | (
        manager_called & (form_layout >= 0)
    )
    form_applied = info.get("formation_applied", jnp.zeros_like(form_requested))
    formations = FormationResult(
        source=form_source,
        requested=form_requested,
        accepted=form_applied,
        applied=form_applied,
        layout_index=form_layout,
        reason=jnp.where(
            form_requested,
            _mapped_reason(form_code, _FORMATION_REASON),
            jnp.int32(CommandReason.NOT_REQUESTED),
        ),
    )

    taker_requested = info.get(
        "set_piece_taker_requested", command.set_piece_takers.requested
    )
    taker_slot = info.get(
        "set_piece_taker_proposed_player_slot", command.set_piece_takers.player_slot
    )
    taker_accepted = info.get(
        "set_piece_taker_accepted", jnp.zeros_like(taker_requested)
    )
    taker_applied = info.get("set_piece_taker_applied", jnp.zeros_like(taker_requested))
    default_taker_reason = jnp.where(
        taker_requested,
        jnp.int32(CommandReason.NO_MATCHING_RESTART),
        jnp.int32(CommandReason.NOT_REQUESTED),
    )
    taker_reason = info.get("set_piece_taker_decision_code", default_taker_reason)
    set_piece_takers = SetPieceTakerResult(
        source=_decision_source(command.set_piece_takers.requested, external_takers),
        requested=taker_requested,
        accepted=taker_accepted,
        applied=taker_applied,
        player_slot=taker_slot,
        reason=taker_reason.astype(jnp.int32),
    )
    return CommandResult(actions, substitutions, formations, set_piece_takers)


def frame_identity(state, *, include_roster: bool) -> FrameIdentity:
    if include_roster:
        player_id = state.player_id
        slot_generation = state.slot_generation
    else:
        player_id = jnp.empty((0,), dtype=jnp.int32)
        slot_generation = jnp.empty((0,), dtype=jnp.int32)
    return FrameIdentity(
        episode_seed=state.episode_seed,
        control_tick=state.t,
        player_id=player_id,
        slot_generation=slot_generation,
    )


def capture_has_record(capture: CaptureSpec) -> bool:
    return any(
        (
            capture.observation,
            capture.reward,
            capture.events,
            capture.action_provenance,
            capture.manager_provenance,
            capture.roster_identity,
            capture.substep_telemetry,
        )
    )


def transition_record(
    *,
    pre_state,
    pre_observation,
    command: StepCommand,
    command_result: CommandResult,
    reward,
    terminated,
    truncated,
    post_state,
    info,
    capture: CaptureSpec,
) -> TransitionRecord:
    """Create an on-device causal record according to a static capture profile."""

    observation = (
        pre_observation
        if capture.observation
        else jnp.empty((pre_state.player_id.shape[0], 0), dtype=pre_state.ball_pos.dtype)
    )
    captured_reward = reward if capture.reward else jnp.empty((0,), dtype=reward.dtype)
    events = (
        events_from_state(post_state, pre_state)
        if capture.events
        else EventBatch.empty(post_state.ball_pos.dtype)
    )
    action_provenance = (
        ActionProvenance(
            parameter_consumed=info["parameter_consumed_mask"],
            kick_applied=info["kick_applied"],
            move_forced=info["move_forced"],
            kick_gated=info["kick_gated"],
            kick_forced=info["kick_forced"],
            halftime_reset=info["halftime_reset"],
        )
        if capture.action_provenance
        else ActionProvenance.empty()
    )
    manager_provenance = (
        ManagerProvenance(
            called=info["manager_called"],
            view=info["manager_decision_trace"]["raw_view"],
        )
        if capture.manager_provenance
        else ManagerProvenance.empty()
    )
    substeps = info["substeps"] if capture.substep_telemetry else None
    return TransitionRecord(
        pre=frame_identity(pre_state, include_roster=capture.roster_identity),
        observation=observation,
        command=command,
        command_result=command_result,
        reward=captured_reward,
        terminated=terminated,
        truncated=truncated,
        events=events,
        action_provenance=action_provenance,
        manager_provenance=manager_provenance,
        substeps=substeps,
        post=frame_identity(post_state, include_roster=capture.roster_identity),
    )


__all__ = [
    "capture_has_record",
    "command_result_from_info",
    "frame_identity",
    "transition_record",
]
