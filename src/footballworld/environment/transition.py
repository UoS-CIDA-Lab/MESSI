"""One fixed-duration policy-control transition over physics substeps.

The public action is chosen once and held for every physics interval in the
frame. Only fixed-shape state and compact causal outputs cross this boundary;
per-substep trajectories belong in an optional diagnostics wrapper, not the
deployment rollout kernel.
"""

import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.body_foul import BodyFoul
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.config.perception import Perception
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.reach import Reach
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.action import IntentAction
from footballworld.core.constants import (
    BALL_EVENT_NONE,
    DISCIPLINE_NONE,
    GEOMETRY_EPS,
    INTENT_ACTION_CONTINUOUS_DIM,
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_NONE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
    WOODWORK_NONE,
)
from footballworld.core.contact import (
    INTENT_CLEAR,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    INTENT_SOURCE_ENVIRONMENT_FORCED,
    INTENT_SOURCE_NONE,
    LAW11_NONE,
    MECHANISM_CHEST,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_HEAD,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_NONE,
    ContactOccurrence,
    ContactResult,
)
from footballworld.core.state import State
from footballworld.core.timebase import (
    DEFAULT_MATCH_DURATION_SECONDS,
    DEFAULT_TIMEBASE,
    Timebase,
    _exact_render_grid,
)
from footballworld.dynamics.action import (
    ACTION_FLAG_ACTIVE_CONTACT,
    ACTION_FLAG_BODY_ONLY_DEFLECTION,
    ACTION_FLAG_CONTACT_ATTEMPTED,
    ACTION_FLAG_ENVIRONMENT_OVERWRITE,
    ACTION_FLAG_FORCED_RELEASE,
    ACTION_FLAG_INTENT_AVAILABLE,
    ACTION_FLAG_KICK_APPLIED,
    ACTION_FLAG_NAMES,
    ACTION_FLAG_PARAMETERS_APPLIED,
    ACTION_FLAG_PASSIVE_CONTACT,
    ACTION_FLAG_REFEREE_PROJECTION,
    ACTION_REASON_CONTACT_ATTEMPTED_NO_REALIZATION,
    ACTION_REASON_CONTACT_NOT_REACHED,
    ACTION_REASON_CONTACT_REALIZED,
    ACTION_REASON_ENVIRONMENT_OVERWRITE,
    ACTION_REASON_FORCED_RELEASE,
    ACTION_REASON_INPUT_SANITIZED,
    ACTION_REASON_INTENT_UNAVAILABLE,
    ACTION_REASON_MOVE_CONSUMED,
    ACTION_REASON_NAMES,
    ACTION_REASON_NONE,
    ACTION_REASON_PARAMETERS_APPLIED,
    ACTION_REASON_PASSIVE_CONTACT,
    ACTION_REASON_REFEREE_PROJECTION,
    ACTION_RECEIPT_SCHEMA,
    DISPLACEMENT_COLLISION_VELOCITY,
    DISPLACEMENT_REFEREE_PROJECTION,
    DISPLACEMENT_RESTART_APPROACH,
    DISPLACEMENT_SELF_MOTION,
    DISPLACEMENT_SEPARATION_POSITION,
    DISPLACEMENT_SOURCE_NAMES,
    ELIGIBILITY_HEIGHT,
    ELIGIBILITY_HORIZONTAL_REACH,
    ELIGIBILITY_NAMES,
    ELIGIBILITY_PHASE_RULE,
    ELIGIBILITY_PHYSICAL_CANDIDATE,
    ELIGIBILITY_RECOVERY,
    ELIGIBILITY_SPEED,
    PARAMETER_FORCE_DIRECTION,
    PARAMETER_FORCE_POWER,
    PARAMETER_GAZE,
    PARAMETER_LAUNCH,
    PARAMETER_MOVE,
    PARAMETER_NAMES,
    PARAMETER_SPIN,
    ActionReceipt,
    ActionTrace,
    decode_physics_action,
    trace_action,
    trace_action_receipt,
)
from footballworld.dynamics.contact_predicates import (
    evaluate_contact_predicates,
    fresh_trap_control_grace,
    verified_controlled_carrier,
)
from footballworld.dynamics.contest import (
    SAMPLE_CONTEST,
    ContestOverride,
    ContestResult,
)
from footballworld.dynamics.orientation import step_gaze
from footballworld.dynamics.passive_contact import detect_between_legs_passage
from footballworld.dynamics.substep import step_physics_substep
from footballworld.environment.clock import regulation_elapsed_ticks
from footballworld.rules.ball_boundary import BoundaryEvent
from footballworld.rules.foul import (
    OFFENCE_NONE,
    SEVERITY_NONE,
    TACTICAL_NONE,
    FoulEvent,
)
from footballworld.rules.gk_holding import (
    GoalkeeperHoldingEvent,
    holding_limit_substeps,
    step_goalkeeper_holding,
)
from footballworld.rules.offside import (
    OffsideEvent,
    OffsideState,
    clear_offside_state,
)
from footballworld.rules.restart_legality import (
    RestartFrameGuard,
    begin_restart_frame,
    mark_restart_opened,
    restart_release_actor_mask,
    restart_visible_to_action,
)
from footballworld.rules.restart_positioning import (
    restart_positioning_pinned,
    restart_taker_release_pose,
)
from footballworld.rules.restart_retouch import RestartRetouchEvent
from footballworld.rules.restart_timing import (
    advance_restart_release_clock,
    continuous_restart_approach_enabled,
    forced_release_delay_substeps,
    restart_release_due,
)
from footballworld.rules.transition import resolve_rules_transition


class ControlFrame(NamedTuple):
    """Post-frame state and compact action-credit/safety latches."""

    state: State
    offside_state: OffsideState
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    penalty_settling_contact: jax.Array


class FrameEvents(NamedTuple):
    """Exact substep events; every leaf starts with ``[decimation]``.

    Contact occurrences use ``[decimation, 4, ...]`` and woodwork events use
    ``[decimation, 3]``. Every ``time_fraction`` is local to its substep.
    """

    deliberate_contact: ContactResult
    contest: ContestResult
    contacts: ContactOccurrence
    foul: FoulEvent
    retouch: RestartRetouchEvent
    offside: OffsideEvent
    boundary: BoundaryEvent
    goalkeeper_holding: GoalkeeperHoldingEvent
    nutmeg: "NutmegEvent"
    woodwork_occurred: jax.Array
    woodwork_kind: jax.Array


class NutmegEvent(NamedTuple):
    """Attributed clean between-legs passage during one physics substep.

    This event is observational only: it never changes physics, possession,
    rules, or the chronological collision-event budget.
    """

    occurred: jax.Array
    attacker: jax.Array
    defender: jax.Array
    position: jax.Array
    time_fraction: jax.Array


class ControlFrameWithEvents(NamedTuple):
    """A control transition plus its exact fixed-shape event sidecar."""

    state: State
    offside_state: OffsideState
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    penalty_settling_contact: jax.Array
    action_trace: ActionTrace
    action_receipt: ActionReceipt
    events: FrameEvents


class _ControlFrameWithEventsAndRenderSamples(NamedTuple):
    """Eventful frame plus bounded renderer-only physics states."""

    state: State
    offside_state: OffsideState
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    penalty_settling_contact: jax.Array
    action_trace: ActionTrace
    action_receipt: ActionReceipt
    events: FrameEvents
    render_state: State
    render_offside_state: OffsideState


class _ActionSubstepTelemetry(NamedTuple):
    """Small eventful-only facts collapsed across the control frame."""

    effective_intent: jax.Array
    flags: jax.Array
    eligibility_seen: jax.Array
    parameter_consumed: jax.Array
    displacement_source: jax.Array


class _EventfulSubstep(NamedTuple):
    events: FrameEvents
    action: _ActionSubstepTelemetry
    render_state: State | None
    render_offside_state: OffsideState | None


class _ControlCarry(NamedTuple):
    state: State
    offside_state: OffsideState
    restart_guard: RestartFrameGuard
    contact_attempted: jax.Array
    deliberate_blocked: jax.Array
    possession_changed: jax.Array
    kick_applied: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    penalty_settling_contact: jax.Array
    body_impact_consumed: jax.Array


class _MatchControlCarry(NamedTuple):
    control: _ControlCarry
    abandoned: jax.Array


def _empty_substep_events(dtype) -> FrameEvents:
    zero_position = jnp.zeros(3, dtype=dtype)
    contact_slots = 4
    return FrameEvents(
        deliberate_contact=ContactResult(
            actor=jnp.int32(NO_PLAYER),
            mechanism=jnp.int32(MECHANISM_NONE),
            intent=jnp.int32(INTENT_MOVE),
            outcome=jnp.int32(OUTCOME_NONE),
            restart_kind=jnp.int32(RK_NONE),
            law11_effect=jnp.int32(LAW11_NONE),
            kick_applied=jnp.bool_(False),
            intent_source=jnp.int32(INTENT_SOURCE_NONE),
        ),
        contest=ContestResult(
            selected=jnp.bool_(False),
            occurred=jnp.bool_(False),
            actor=jnp.int32(NO_PLAYER),
            challenger=jnp.int32(NO_PLAYER),
            outcome=jnp.int32(OUTCOME_NONE),
            parameters_applied=jnp.bool_(False),
            foul_actor=jnp.int32(NO_PLAYER),
            foul_victim=jnp.int32(NO_PLAYER),
            offence_position=jnp.zeros(2, dtype=dtype),
            discipline=jnp.int32(DISCIPLINE_NONE),
            override_valid=jnp.bool_(True),
        ),
        contacts=ContactOccurrence(
            occurred=jnp.zeros(contact_slots, dtype=jnp.bool_),
            actor=jnp.full(contact_slots, NO_PLAYER, dtype=jnp.int32),
            mechanism=jnp.full(contact_slots, MECHANISM_NONE, dtype=jnp.int32),
            law11_effect=jnp.full(contact_slots, LAW11_NONE, dtype=jnp.int32),
            position=jnp.zeros((contact_slots, 3), dtype=dtype),
            time_fraction=jnp.zeros(contact_slots, dtype=dtype),
        ),
        foul=FoulEvent(
            occurred=jnp.bool_(False),
            contest_source=jnp.bool_(False),
            offender=jnp.int32(NO_PLAYER),
            victim=jnp.int32(NO_PLAYER),
            offender_team=jnp.int32(NO_TEAM),
            offence_type=jnp.int32(OFFENCE_NONE),
            severity=jnp.int32(SEVERITY_NONE),
            tactical_effect=jnp.int32(TACTICAL_NONE),
            position=zero_position,
            time_fraction=jnp.asarray(0.0, dtype=dtype),
            restart_kind=jnp.int32(RK_NONE),
            discipline=jnp.int32(DISCIPLINE_NONE),
            advantage_applied=jnp.bool_(False),
        ),
        retouch=RestartRetouchEvent(
            occurred=jnp.bool_(False),
            offender=jnp.int32(NO_PLAYER),
            source_restart_kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            restart_kind=jnp.int32(RK_NONE),
            indirect=jnp.bool_(False),
            goalkeeper_handling=jnp.bool_(False),
            time_fraction=jnp.asarray(0.0, dtype=dtype),
            contact_position=zero_position,
            restart_position=zero_position,
        ),
        offside=OffsideEvent(
            occurred=jnp.bool_(False),
            actor=jnp.int32(NO_PLAYER),
            team=jnp.int32(NO_TEAM),
            position=zero_position,
            time_fraction=jnp.asarray(0.0, dtype=dtype),
        ),
        boundary=BoundaryEvent(
            occurred=jnp.bool_(False),
            kind=jnp.int32(BALL_EVENT_NONE),
            team=jnp.int32(NO_TEAM),
            scoring_team=jnp.int32(NO_TEAM),
            time_fraction=jnp.asarray(0.0, dtype=dtype),
            position=zero_position,
        ),
        goalkeeper_holding=GoalkeeperHoldingEvent(
            opened=jnp.bool_(False),
            expired=jnp.bool_(False),
            goalkeeper=jnp.int32(NO_PLAYER),
            goalkeeper_team=jnp.int32(NO_TEAM),
            restart_team=jnp.int32(NO_TEAM),
            corner_position=zero_position,
        ),
        nutmeg=NutmegEvent(
            occurred=jnp.bool_(False),
            attacker=jnp.int32(NO_PLAYER),
            defender=jnp.int32(NO_PLAYER),
            position=zero_position,
            time_fraction=jnp.asarray(0.0, dtype=dtype),
        ),
        woodwork_occurred=jnp.zeros(3, dtype=jnp.bool_),
        woodwork_kind=jnp.full(3, WOODWORK_NONE, dtype=jnp.int32),
    )


def _bits(condition: jax.Array, value: int, dtype) -> jax.Array:
    """Return one fixed-width bit for each true row."""

    return jnp.asarray(condition, dtype=dtype) * jnp.asarray(value, dtype=dtype)


def _actor_rows(
    occurred: jax.Array,
    actor: jax.Array,
    player_count: int,
) -> jax.Array:
    """Project scalar or fixed-slot causal actor facts onto player rows."""

    occurred = jnp.ravel(jnp.asarray(occurred, dtype=jnp.bool_))
    actor = jnp.ravel(jnp.asarray(actor, dtype=jnp.int32))
    valid = occurred & (actor >= 0) & (actor < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    return jnp.zeros(player_count, dtype=jnp.bool_).at[safe_actor].max(valid)


def _empty_action_substep(effective_intent: jax.Array) -> _ActionSubstepTelemetry:
    shape = effective_intent.shape
    return _ActionSubstepTelemetry(
        effective_intent=effective_intent.astype(jnp.int32),
        flags=jnp.zeros(shape, dtype=jnp.uint32),
        eligibility_seen=jnp.zeros(shape, dtype=jnp.uint16),
        parameter_consumed=jnp.zeros(shape, dtype=jnp.uint16),
        displacement_source=jnp.zeros(shape, dtype=jnp.uint16),
    )


def _reduce_action_receipt(
    input_receipt: ActionReceipt,
    stacked: _ActionSubstepTelemetry,
    *,
    entry_active: jax.Array,
) -> ActionReceipt:
    """Collapse exact substep facts without exposing a second dense timeline."""

    flags = input_receipt.flags | jnp.bitwise_or.reduce(stacked.flags, axis=0)
    eligibility_seen = jnp.bitwise_or.reduce(stacked.eligibility_seen, axis=0)
    parameter_consumed = jnp.bitwise_or.reduce(
        stacked.parameter_consumed, axis=0
    ) | _bits(entry_active, PARAMETER_GAZE, jnp.uint16)
    displacement_source = jnp.bitwise_or.reduce(stacked.displacement_source, axis=0)

    forced_by_substep = (stacked.flags & jnp.uint32(ACTION_FLAG_FORCED_RELEASE)) != 0
    forced_seen = jnp.any(forced_by_substep, axis=0)
    forced_intent = jnp.max(
        jnp.where(forced_by_substep, stacked.effective_intent, INTENT_MOVE),
        axis=0,
    )
    effective_intent = jnp.where(
        forced_seen, forced_intent, input_receipt.effective_intent
    ).astype(jnp.int32)

    available = (flags & jnp.uint32(ACTION_FLAG_INTENT_AVAILABLE)) != 0
    attempted = (flags & jnp.uint32(ACTION_FLAG_CONTACT_ATTEMPTED)) != 0
    active_contact = (flags & jnp.uint32(ACTION_FLAG_ACTIVE_CONTACT)) != 0
    passive_contact = (flags & jnp.uint32(ACTION_FLAG_PASSIVE_CONTACT)) != 0
    parameters_applied = (flags & jnp.uint32(ACTION_FLAG_PARAMETERS_APPLIED)) != 0
    forced_release = (flags & jnp.uint32(ACTION_FLAG_FORCED_RELEASE)) != 0
    referee_projection = (flags & jnp.uint32(ACTION_FLAG_REFEREE_PROJECTION)) != 0
    environment_overwrite = (flags & jnp.uint32(ACTION_FLAG_ENVIRONMENT_OVERWRITE)) != 0
    requested_contact = effective_intent != INTENT_MOVE
    move_consumed = (parameter_consumed & jnp.uint16(PARAMETER_MOVE)) != 0

    reason = jnp.full(effective_intent.shape, ACTION_REASON_NONE, dtype=jnp.int16)
    reason = jnp.where(move_consumed, ACTION_REASON_MOVE_CONSUMED, reason)
    reason = jnp.where(
        requested_contact & (~available), ACTION_REASON_INTENT_UNAVAILABLE, reason
    )
    reason = jnp.where(
        requested_contact & available & (~attempted),
        ACTION_REASON_CONTACT_NOT_REACHED,
        reason,
    )
    reason = jnp.where(
        attempted & (~active_contact),
        ACTION_REASON_CONTACT_ATTEMPTED_NO_REALIZATION,
        reason,
    )
    reason = jnp.where(passive_contact, ACTION_REASON_PASSIVE_CONTACT, reason)
    reason = jnp.where(active_contact, ACTION_REASON_CONTACT_REALIZED, reason)
    reason = jnp.where(parameters_applied, ACTION_REASON_PARAMETERS_APPLIED, reason)
    reason = jnp.where(forced_release, ACTION_REASON_FORCED_RELEASE, reason)
    reason = jnp.where(referee_projection, ACTION_REASON_REFEREE_PROJECTION, reason)
    reason = jnp.where(
        environment_overwrite, ACTION_REASON_ENVIRONMENT_OVERWRITE, reason
    )
    reason = jnp.where(
        input_receipt.primary_reason == ACTION_REASON_INPUT_SANITIZED,
        ACTION_REASON_INPUT_SANITIZED,
        reason,
    ).astype(jnp.int16)
    return input_receipt._replace(
        effective_intent=effective_intent,
        flags=flags,
        eligibility_seen=eligibility_seen,
        primary_reason=reason,
        parameter_consumed=parameter_consumed,
        displacement_source=displacement_source,
    )


def _not_after_foul(time_fraction: jax.Array, foul: FoulEvent) -> jax.Array:
    """Keep a physical event only when it is no later than the whistle."""

    return (~foul.occurred) | (time_fraction <= foul.time_fraction)


def _contacts_not_after_foul(
    contacts: ContactOccurrence,
    foul: FoulEvent,
    stoppage_time_fraction: jax.Array,
) -> ContactOccurrence:
    """Return clean sentinels for contact slots after the selected stoppage.

    The explicit foul predicate preserves the existing treatment of a foul
    even if a future advantage model emits it without opening a restart.
    """

    kept = (
        contacts.occurred
        & _not_after_foul(contacts.time_fraction, foul)
        & (contacts.time_fraction <= stoppage_time_fraction)
    )
    return ContactOccurrence(
        occurred=kept,
        actor=jnp.where(kept, contacts.actor, NO_PLAYER).astype(jnp.int32),
        mechanism=jnp.where(kept, contacts.mechanism, MECHANISM_NONE).astype(jnp.int32),
        law11_effect=jnp.where(kept, contacts.law11_effect, LAW11_NONE).astype(
            jnp.int32
        ),
        position=jnp.where(
            kept[:, None],
            contacts.position,
            jnp.zeros_like(contacts.position),
        ),
        time_fraction=jnp.where(kept, contacts.time_fraction, 0.0),
    )


def _first_stoppage_time_fraction(physics, rules, holding_event) -> jax.Array:
    """Return the first selected play-stopping instant in one substep.

    Ball-boundary and player-contact scheduling already share an exact local
    timeline. Player-body fouls are a locomotion-microstep-end upper bound.
    Goalkeeper hold opening is owned by its deliberate hand contact, while an
    expiry is a substep-end transition. Advantage fouls have no restart and do
    not stop play. This private cutoff does not add a public event leaf.
    """

    dtype = physics.woodwork_time_fraction.dtype
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    foul_stops = rules.foul_event.occurred & (
        rules.foul_event.restart_kind != jnp.int32(RK_NONE)
    )
    return jnp.min(
        jnp.stack(
            (
                jnp.where(foul_stops, rules.foul_event.time_fraction, infinity),
                jnp.where(
                    rules.retouch_event.occurred,
                    rules.retouch_event.time_fraction,
                    infinity,
                ),
                jnp.where(
                    rules.offside_event.occurred,
                    rules.offside_event.time_fraction,
                    infinity,
                ),
                jnp.where(
                    rules.boundary_event.occurred,
                    rules.boundary_event.time_fraction,
                    infinity,
                ),
                jnp.where(
                    holding_event.opened,
                    physics.deliberate_time_fraction,
                    infinity,
                ),
                jnp.where(
                    holding_event.expired,
                    jnp.asarray(1.0, dtype=dtype),
                    infinity,
                ),
            )
        )
    )


def _woodwork_not_after_stoppage(physics, stoppage_time_fraction):
    """Mask goal-frame records strictly later than play was stopped."""

    occurred = physics.woodwork_occurred & (
        physics.woodwork_time_fraction <= stoppage_time_fraction
    )
    kind = jnp.where(occurred, physics.woodwork_kind, WOODWORK_NONE).astype(jnp.int32)
    return occurred, kind


def _detect_nutmeg_event(
    pre: State,
    physics,
    *,
    ball_geometry: Ball,
    body: BodyContact,
) -> NutmegEvent:
    """Attribute one conservative, non-state-changing clean passage.

    The post-physics chord is trusted only when no player contact, frame hit,
    boundary crossing, or ground rebound can have bent it. This deliberately
    omits ambiguous same-substep kick-to-passage cases instead of inventing a
    false trajectory. A requested but unrealized contact remains eligible.
    """

    player_count = pre.players.position.shape[0]
    dtype = pre.ball.position.dtype
    attacker = pre.possession.last_contact.actor.astype(jnp.int32)
    attacker_valid = (attacker >= 0) & (attacker < player_count)
    safe_attacker = jnp.clip(attacker, 0, player_count - 1)
    last_contact = pre.possession.last_contact
    deliberate_source = (
        (last_contact.mechanism != MECHANISM_NONE)
        & (last_contact.mechanism != MECHANISM_PASSIVE_BODY)
        & (last_contact.intent_source != INTENT_SOURCE_NONE)
    )
    attacker_valid = (
        attacker_valid & pre.players.active[safe_attacker] & deliberate_source
    )
    attacker_team = pre.players.team_id[safe_attacker]
    defender_eligible = (
        pre.players.active
        & (pre.players.team_id != attacker_team)
        & (jnp.arange(player_count, dtype=jnp.int32) != attacker)
    )
    passage_state = pre._replace(
        players=pre.players._replace(
            on_pitch=pre.players.on_pitch & defender_eligible,
            # Passive leg collision uses the post-locomotion chest direction
            # for this substep; telemetry must use the identical frame.
            body_forward=physics.state.players.body_forward,
        )
    )
    path_delta = physics.state.ball.position - pre.ball.position
    passage = detect_between_legs_passage(
        passage_state,
        path_delta,
        ball_geometry=ball_geometry,
        body=body,
        player_start_position=physics.player_path_start,
        player_path_delta=physics.player_path_delta,
    )

    # Every listed event can split or terminate the chord. A downward-to-
    # non-downward velocity change identifies the otherwise-unexposed ground
    # response; supported rolling begins at zero and remains eligible.
    uninterrupted = (
        (~jnp.any(physics.contact_occurrences.occurred))
        & (~jnp.any(physics.woodwork_occurred))
        & (~physics.boundary.occurred)
        & (~physics.event_budget_exhausted)
        & (~((pre.ball.velocity[2] < 0.0) & (physics.state.ball.velocity[2] >= 0.0)))
    )
    defender = passage.actor.astype(jnp.int32)
    defender_valid = (defender >= 0) & (defender < player_count)
    safe_defender = jnp.clip(defender, 0, player_count - 1)
    opposing = pre.players.team_id[safe_defender] != attacker_team
    occurred = (
        passage.occurred
        & pre.ball.live
        & physics.state.ball.live
        & uninterrupted
        & attacker_valid
        & defender_valid
        & opposing
    )
    return NutmegEvent(
        occurred=occurred,
        attacker=jnp.where(occurred, attacker, NO_PLAYER).astype(jnp.int32),
        defender=jnp.where(occurred, defender, NO_PLAYER).astype(jnp.int32),
        position=jnp.where(occurred, passage.position, jnp.zeros(3, dtype=dtype)),
        time_fraction=jnp.where(occurred, passage.time_fraction, 0.0).astype(dtype),
    )


def _empty_frame_events(decimation: int, dtype) -> FrameEvents:
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (decimation,) + value.shape),
        _empty_substep_events(dtype),
    )


def sample_control_contest_overrides(decimation: int) -> ContestOverride:
    """Return ordinary stochastic-contest sentinels for one control frame."""

    return ContestOverride(
        winner=jnp.full(decimation, SAMPLE_CONTEST, dtype=jnp.int32),
        outcome=jnp.full(decimation, SAMPLE_CONTEST, dtype=jnp.int32),
    )


def _possession_identity_changed(before: State, after: State) -> jax.Array:
    return (before.possession.team != after.possession.team) | (
        before.possession.player != after.possession.player
    )


def _regulation_elapsed_fraction(state: State, *, timebase: Timebase) -> jax.Array:
    """Return causal 90-minute progress, excluding accumulated dead balls."""

    reference_ticks = max(
        1, round(DEFAULT_MATCH_DURATION_SECONDS * timebase.control_fps)
    )
    return jnp.clip(
        regulation_elapsed_ticks(state).astype(jnp.float32)
        / jnp.float32(reference_ticks),
        0.0,
        1.0,
    )


def _finish_control_ticks(
    entry: State,
    final: State,
    possession_changed: jax.Array,
) -> State:
    final_valid = (final.possession.team != NO_TEAM) & (
        final.possession.player != NO_PLAYER
    )
    continuous = (
        final_valid
        & (~possession_changed)
        & (final.possession.team == entry.possession.team)
        & (final.possession.player == entry.possession.player)
    )
    control_ticks = jnp.where(
        continuous,
        entry.possession.control_ticks + jnp.int32(1),
        jnp.where(final_valid, jnp.int32(1), jnp.int32(0)),
    ).astype(jnp.int32)
    return final._replace(
        control_tick=(entry.control_tick + jnp.int32(1)).astype(jnp.int32),
        possession=final.possession._replace(control_ticks=control_ticks),
    )


def step_control_frame(
    state: State,
    offside_state: OffsideState,
    action: IntentAction,
    key: jax.Array,
    contest_overrides: ContestOverride,
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    boundary_margin_m: float = 0.0,
    locomotion_enabled: jax.Array | None = None,
    position_update_enabled: jax.Array | None = None,
    separation_pinned: jax.Array | None = None,
    minimum_team_players: tuple[int, int] | None = None,
    goalkeeper_holding: GoalkeeperHolding = GoalkeeperHolding(),
    restart_timing: RestartTiming = RestartTiming(),
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    action_scale: ActionScale = ActionScale(),
    contact_timing: ContactTiming = ContactTiming(),
    contest_config: Contest = Contest(),
    body_foul_config: BodyFoul = BodyFoul(),
    player_physics: PlayerPhysics = PlayerPhysics(),
    perception: Perception = Perception(),
    body: BodyContact = BodyContact(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
    ball_physics: BallPhysics = BallPhysics(),
    period_boundary_penalty_enforced: jax.Array = False,
    _collect_events: bool = False,
    _render_fps: float | None = None,
) -> ControlFrame | ControlFrameWithEvents | _ControlFrameWithEventsAndRenderSamples:
    """Advance exactly one policy frame with a fixed action and scan length."""

    render_sample_count: int | None = None
    render_sample_indices: jax.Array | None = None
    if _render_fps is not None:
        if not _collect_events:
            raise ValueError("renderer physics samples require event collection")
        if not isinstance(_render_fps, numbers.Real) or isinstance(_render_fps, bool):
            raise TypeError("_render_fps must be a real non-boolean scalar or None")
        _, render_sample_count, sample_indices = _exact_render_grid(
            timebase,
            _render_fps,
            context="event render",
        )
        render_sample_indices = jnp.asarray(sample_indices, dtype=jnp.int32)

    player_count = state.players.position.shape[0]
    if not isinstance(action, IntentAction):
        raise TypeError("action must be IntentAction")
    continuous = action.as_continuous_array()
    expected_continuous_shape = (
        player_count,
        INTENT_ACTION_CONTINUOUS_DIM,
    )
    if continuous.shape != expected_continuous_shape:
        raise ValueError(
            "intent action continuous controls must have shape "
            f"{expected_continuous_shape}, got {continuous.shape}"
        )
    expected_intent_shape = (player_count,)
    if action.intent.shape != expected_intent_shape:
        raise ValueError(
            "intent action categories must have shape "
            f"{expected_intent_shape}, got {action.intent.shape}"
        )
    # Capture repairs before normalizing the authoritative physics action.
    # This branch is Python-static and absent from the lean step graph.
    submitted_action = action
    input_action_receipt = (
        trace_action_receipt(submitted_action) if _collect_events else None
    )
    action = IntentAction.from_array(action.intent, continuous)
    physics_action = decode_physics_action(state, action)
    state = state._replace(
        players=step_gaze(
            state.players,
            physics_action.gaze_center,
            dt_control=timebase.control_dt,
            perception=perception,
        )
    )

    def normalize_player_mask(
        value: jax.Array | None,
        *,
        name: str,
        default: bool,
    ) -> jax.Array:
        if value is None:
            return jnp.full(player_count, default, dtype=jnp.bool_)
        normalized = jnp.asarray(value, dtype=jnp.bool_)
        expected_shape = (player_count,)
        if normalized.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {normalized.shape}"
            )
        return normalized

    locomotion_enabled = normalize_player_mask(
        locomotion_enabled, name="locomotion_enabled", default=True
    )
    position_update_enabled = normalize_player_mask(
        position_update_enabled, name="position_update_enabled", default=True
    )
    separation_pinned = normalize_player_mask(
        separation_pinned, name="separation_pinned", default=False
    )
    period_boundary_penalty_enforced = jnp.asarray(
        period_boundary_penalty_enforced, dtype=jnp.bool_
    )
    expected_override_shape = (timebase.decimation,)
    if (
        contest_overrides.winner.shape != expected_override_shape
        or contest_overrides.outcome.shape != expected_override_shape
    ):
        raise ValueError(
            "contest override leaves must have shape "
            f"{expected_override_shape}, got "
            f"{contest_overrides.winner.shape} and "
            f"{contest_overrides.outcome.shape}"
        )
    if minimum_team_players is not None:
        if (
            not isinstance(minimum_team_players, tuple)
            or len(minimum_team_players) != 2
            or any(
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 0
                for value in minimum_team_players
            )
        ):
            raise ValueError(
                "minimum_team_players must be None or a pair of non-negative integers"
            )
        minimum_team_players = tuple(int(value) for value in minimum_team_players)
    entry_restart_active = (state.restart.kind > RK_NONE) & (
        state.restart.kind < RESTART_COUNT
    )
    entry_restart_ready = entry_restart_active & state.restart_layout_ready
    entry_restart_pinned = entry_restart_ready & restart_positioning_pinned(state)
    entry_continuous_approach = entry_restart_active & (
        continuous_restart_approach_enabled(
            state.restart.kind,
            config=restart_timing,
        )
    )
    prepared_entry = state
    initial = _ControlCarry(
        state=prepared_entry,
        offside_state=offside_state,
        restart_guard=begin_restart_frame(state),
        contact_attempted=jnp.zeros(player_count, dtype=jnp.bool_),
        deliberate_blocked=jnp.bool_(False),
        possession_changed=jnp.bool_(False),
        kick_applied=jnp.zeros(player_count, dtype=jnp.bool_),
        event_budget_exhausted=jnp.bool_(False),
        contest_override_valid=jnp.bool_(True),
        penalty_settling_contact=jnp.bool_(False),
        body_impact_consumed=jnp.bool_(False),
    )
    hold_limit = holding_limit_substeps(timebase=timebase, config=goalkeeper_holding)
    release_delay = forced_release_delay_substeps(
        timebase=timebase, config=restart_timing
    )
    regulation_elapsed_fraction = _regulation_elapsed_fraction(
        prepared_entry, timebase=timebase
    )

    def substep(carry: _ControlCarry, inputs):
        substep_index, contest_override = inputs
        pre = carry.state
        substep_key = jax.random.fold_in(key, substep_index)
        release_visible = restart_visible_to_action(pre, carry.restart_guard)
        release_allowed = (
            restart_release_actor_mask(pre, carry.restart_guard)
            & entry_restart_ready
            & restart_release_due(
                pre,
                delay_substeps=release_delay,
                hold_limit_substeps=hold_limit,
            )
        )
        submitted_kick_intent = (
            (physics_action.requested_intent == jnp.int32(INTENT_PASS))
            | (physics_action.requested_intent == jnp.int32(INTENT_SHOT))
            | (physics_action.requested_intent == jnp.int32(INTENT_CLEAR))
        )
        pass_only_restart = (pre.restart.kind == jnp.int32(RK_THROWIN)) | (
            pre.restart.kind == jnp.int32(RK_GK_HOLD)
        )
        submitted_intent_legal = submitted_kick_intent & (
            (~pass_only_restart)
            | (physics_action.requested_intent == jnp.int32(INTENT_PASS))
        )
        forced_fallback_intent = jnp.where(
            pre.restart.kind == jnp.int32(RK_PENALTY),
            jnp.int32(INTENT_SHOT),
            jnp.int32(INTENT_PASS),
        )
        forced_intent = jnp.where(
            submitted_intent_legal,
            physics_action.requested_intent,
            forced_fallback_intent,
        )
        effective_action = physics_action._replace(
            contact=physics_action.contact | release_allowed,
            requested_intent=jnp.where(
                release_allowed,
                forced_intent,
                physics_action.requested_intent,
            ),
            intent_source=jnp.where(
                release_allowed,
                jnp.int32(INTENT_SOURCE_ENVIRONMENT_FORCED),
                physics_action.intent_source,
            ),
        )
        visible_restart_pinned = (
            release_visible & entry_restart_pinned & (~entry_continuous_approach)
        )
        visible_restart_approach = (
            release_visible & entry_restart_ready & entry_continuous_approach
        )
        if _collect_events:
            # Eventful-only eligibility reuses the authoritative predicate at
            # the causal pre-substep state. Independent bits preserve why a
            # request was or was not physically available without entering the
            # lean step.
            receipt_predicates = evaluate_contact_predicates(
                pre,
                release_allowed,
                requested_intent=effective_action.requested_intent,
                ball_geometry=ball_geometry,
                stadium=stadium,
                reach=reach,
                scale=action_scale,
                body=body,
            )
        attempted = jnp.where(
            carry.deliberate_blocked,
            jnp.ones_like(carry.contact_attempted),
            carry.contact_attempted,
        )
        physics = step_physics_substep(
            pre,
            effective_action,
            substep_key,
            contest_override,
            release_allowed,
            dt=timebase.dt_phys,
            boundary_margin_m=boundary_margin_m,
            contact_attempted=attempted,
            locomotion_enabled=locomotion_enabled,
            position_update_enabled=(
                position_update_enabled & (~visible_restart_pinned)
            ),
            separation_pinned=separation_pinned | visible_restart_pinned,
            restart_positioning_enabled=(release_visible & entry_restart_ready),
            restart_approach_enabled=visible_restart_approach,
            ball_geometry=ball_geometry,
            stadium=stadium,
            reach=reach,
            action_scale=action_scale,
            contact_timing=contact_timing,
            contest_config=contest_config,
            regulation_elapsed_fraction=regulation_elapsed_fraction,
            player_physics=player_physics,
            body=body,
            long_stamina=long_stamina,
            short_stamina=short_stamina,
            ball_physics=ball_physics,
            period_boundary_penalty_enforced=period_boundary_penalty_enforced,
        )
        rules = resolve_rules_transition(
            pre,
            physics,
            carry.offside_state,
            effective_action.requested_intent,
            substep_key,
            ~carry.body_impact_consumed,
            regulation_elapsed_fraction=regulation_elapsed_fraction,
            body_foul_config=body_foul_config,
            contest_config=contest_config,
            stadium=stadium,
            ball_geometry=ball_geometry,
            body=body,
        )
        holding = step_goalkeeper_holding(
            pre,
            rules.state,
            limit_substeps=hold_limit,
            stadium=stadium,
            ball_geometry=ball_geometry,
        )
        deliberate_not_after_foul = _not_after_foul(
            physics.deliberate_time_fraction,
            rules.foul_event,
        )
        stoppage_time_fraction = _first_stoppage_time_fraction(
            physics,
            rules,
            holding.event,
        )
        deliberate_in_time = deliberate_not_after_foul & (
            physics.deliberate_time_fraction <= stoppage_time_fraction
        )
        next_offside = jax.tree_util.tree_map(
            lambda cleared, current: jnp.where(holding.event.expired, cleared, current),
            clear_offside_state(rules.offside_state),
            rules.offside_state,
        )
        restart_opened = (
            (rules.foul_event.occurred & (rules.foul_event.restart_kind != RK_NONE))
            | rules.retouch_event.occurred
            | rules.offside_event.occurred
            | rules.boundary_event.occurred
            | holding.event.opened
            | holding.event.expired
        )
        guard = mark_restart_opened(carry.restart_guard, restart_opened)
        released_entry_restart = (
            release_visible
            & physics.deliberate_occurred
            & deliberate_in_time
            & (physics.deliberate_actor == pre.restart.taker)
            & (physics.deliberate_contact.restart_kind != RK_NONE)
        )
        approach_target, _ = restart_taker_release_pose(
            holding.state,
            stadium=stadium,
            ball=ball_geometry,
            body=body,
        )
        approach_taker = jnp.clip(holding.state.restart.taker, 0, player_count - 1)
        approach_arrived = (
            jnp.linalg.norm(
                holding.state.players.position[approach_taker] - approach_target
            )
            <= GEOMETRY_EPS
        )
        continuous_timer_guard = release_visible & entry_continuous_approach
        approach_countdown_enabled = (~continuous_timer_guard) | (
            visible_restart_approach & approach_arrived
        )
        next_state = advance_restart_release_clock(
            holding.state,
            opened=restart_opened,
            delay_substeps=release_delay,
            countdown_enabled=approach_countdown_enabled,
        )
        next_restart_active = (next_state.restart.kind > RK_NONE) & (
            next_state.restart.kind < RESTART_COUNT
        )
        next_layout_ready = jnp.where(
            restart_opened,
            next_state.restart.kind == RK_GK_HOLD,
            next_state.restart_layout_ready,
        )
        penalty_release = released_entry_restart & (
            pre.restart.kind == jnp.int32(RK_PENALTY)
        )
        next_state = next_state._replace(
            restart_layout_ready=next_restart_active & next_layout_ready,
            penalty_completion_team=jnp.where(
                penalty_release,
                pre.restart.team,
                next_state.penalty_completion_team,
            ).astype(jnp.int32),
        )
        verified_carrier = verified_controlled_carrier(
            next_state,
            ball_geometry,
            reach,
            action_scale,
        )
        possession_claimed = (
            (next_state.possession.team != NO_TEAM)
            | (next_state.possession.player != NO_PLAYER)
            | (next_state.possession.control_ticks > 0)
        )
        live_open_play = next_state.ball.live & (
            next_state.restart.kind == jnp.int32(RK_NONE)
        )
        release_uncontrolled = (
            live_open_play
            & possession_claimed
            & (verified_carrier == jnp.int32(NO_PLAYER))
        )
        # A successful trap must survive until its actor can observe and react
        # to it. Without this one-decision causal grace, a contact near the
        # reach boundary can be erased by later physics substeps in the same
        # frame, producing an unobservable possession and repeated loose-ball
        # pursuit. Physical carrier verification remains authoritative after
        # that decision opportunity and for every tackle predicate.
        fresh_trap_grace = fresh_trap_control_grace(next_state)
        release_uncontrolled = release_uncontrolled & (~fresh_trap_grace)
        old_team = next_state.possession.team
        valid_old_team = (old_team == jnp.int32(TEAM_0)) | (
            old_team == jnp.int32(TEAM_1)
        )
        next_state = next_state._replace(
            possession=next_state.possession._replace(
                team=jnp.where(
                    release_uncontrolled,
                    jnp.int32(NO_TEAM),
                    next_state.possession.team,
                ).astype(jnp.int32),
                player=jnp.where(
                    release_uncontrolled,
                    jnp.int32(NO_PLAYER),
                    next_state.possession.player,
                ).astype(jnp.int32),
                previous_team=jnp.where(
                    release_uncontrolled & valid_old_team,
                    old_team,
                    next_state.possession.previous_team,
                ).astype(jnp.int32),
                control_ticks=jnp.where(
                    release_uncontrolled,
                    jnp.int32(0),
                    next_state.possession.control_ticks,
                ).astype(jnp.int32),
            )
        )
        changed = carry.possession_changed | _possession_identity_changed(
            pre, next_state
        )

        actor = physics.deliberate_actor.astype(jnp.int32)
        actor_valid = (actor >= 0) & (actor < player_count)
        safe_actor = jnp.clip(actor, 0, player_count - 1)
        applied = (
            physics.deliberate_occurred
            & deliberate_in_time
            & actor_valid
            & physics.deliberate_contact.kick_applied
        )
        kick_applied = carry.kick_applied.at[safe_actor].set(
            carry.kick_applied[safe_actor] | applied
        )
        next_carry = _ControlCarry(
            state=next_state,
            offside_state=next_offside,
            restart_guard=guard,
            contact_attempted=jnp.where(
                carry.deliberate_blocked | (~deliberate_in_time),
                carry.contact_attempted,
                physics.contact_attempted,
            ),
            deliberate_blocked=(carry.deliberate_blocked | released_entry_restart),
            possession_changed=changed,
            kick_applied=kick_applied,
            event_budget_exhausted=(
                carry.event_budget_exhausted
                | (physics.event_budget_exhausted & (~rules.foul_event.occurred))
            ),
            contest_override_valid=(
                carry.contest_override_valid
                & jnp.where(
                    deliberate_in_time,
                    physics.contest.override_valid,
                    jnp.bool_(True),
                )
            ),
            penalty_settling_contact=(
                carry.penalty_settling_contact | physics.penalty_settling_contact
            ),
            body_impact_consumed=(
                carry.body_impact_consumed | rules.body_impact_consumed
            ),
        )
        if not _collect_events:
            return next_carry, None
        nutmeg = _detect_nutmeg_event(
            pre,
            physics,
            ball_geometry=ball_geometry,
            body=body,
        )
        empty_events = _empty_substep_events(pre.ball.position.dtype)
        masked_deliberate_contact = jax.tree_util.tree_map(
            lambda current, empty: jnp.where(
                deliberate_in_time,
                current,
                empty,
            ),
            physics.deliberate_contact,
            empty_events.deliberate_contact,
        )
        masked_contest = jax.tree_util.tree_map(
            lambda current, empty: jnp.where(
                deliberate_in_time,
                current,
                empty,
            ),
            physics.contest,
            empty_events.contest,
        )
        masked_contacts = _contacts_not_after_foul(
            physics.contact_occurrences,
            rules.foul_event,
            stoppage_time_fraction,
        )
        nutmeg_not_after_foul = _not_after_foul(
            nutmeg.time_fraction,
            rules.foul_event,
        ) & (nutmeg.time_fraction <= stoppage_time_fraction)
        masked_nutmeg = jax.tree_util.tree_map(
            lambda current, empty: jnp.where(
                nutmeg_not_after_foul,
                current,
                empty,
            ),
            nutmeg,
            empty_events.nutmeg,
        )
        woodwork_occurred, woodwork_kind = _woodwork_not_after_stoppage(
            physics,
            stoppage_time_fraction,
        )
        frame_events = FrameEvents(
            deliberate_contact=masked_deliberate_contact,
            contest=masked_contest,
            contacts=masked_contacts,
            foul=rules.foul_event,
            retouch=rules.retouch_event,
            offside=rules.offside_event,
            boundary=rules.boundary_event,
            goalkeeper_holding=holding.event,
            nutmeg=masked_nutmeg,
            woodwork_occurred=woodwork_occurred,
            woodwork_kind=woodwork_kind,
        )

        intent_available = pre.players.active & (
            (effective_action.requested_intent == INTENT_MOVE)
            | receipt_predicates.intent_allowed
        )
        eligibility_seen = (
            _bits(
                receipt_predicates.phase_allowed,
                ELIGIBILITY_PHASE_RULE,
                jnp.uint16,
            )
            | _bits(
                receipt_predicates.horizontal_reach,
                ELIGIBILITY_HORIZONTAL_REACH,
                jnp.uint16,
            )
            | _bits(
                receipt_predicates.height_allowed,
                ELIGIBILITY_HEIGHT,
                jnp.uint16,
            )
            | _bits(
                receipt_predicates.speed_allowed,
                ELIGIBILITY_SPEED,
                jnp.uint16,
            )
            | _bits(
                receipt_predicates.recovery_ready,
                ELIGIBILITY_RECOVERY,
                jnp.uint16,
            )
            | _bits(
                receipt_predicates.possible_now & receipt_predicates.intent_allowed,
                ELIGIBILITY_PHYSICAL_CANDIDATE,
                jnp.uint16,
            )
        )
        attempted_rows = next_carry.contact_attempted
        active_rows = _actor_rows(
            masked_deliberate_contact.actor != NO_PLAYER,
            masked_deliberate_contact.actor,
            player_count,
        )
        passive_slots = masked_contacts.occurred & (
            masked_contacts.mechanism == MECHANISM_PASSIVE_BODY
        )
        passive_rows = _actor_rows(
            passive_slots,
            masked_contacts.actor,
            player_count,
        )
        parameter_rows = _actor_rows(
            masked_contest.parameters_applied,
            masked_deliberate_contact.actor,
            player_count,
        )
        kick_rows = _actor_rows(
            masked_deliberate_contact.kick_applied,
            masked_deliberate_contact.actor,
            player_count,
        )
        body_only = (
            (masked_deliberate_contact.mechanism == MECHANISM_HEAD)
            | (masked_deliberate_contact.mechanism == MECHANISM_CHEST)
        ) & (~masked_deliberate_contact.kick_applied)
        body_only_rows = _actor_rows(
            body_only,
            masked_deliberate_contact.actor,
            player_count,
        )
        forced_rows = jnp.asarray(release_allowed, dtype=jnp.bool_)

        projection_changed = physics.restart_projection_moved
        approach_rows = physics.restart_approach_moved
        path_changed = jnp.any(
            jnp.abs(physics.player_path_delta) > GEOMETRY_EPS,
            axis=-1,
        )
        movement_parameter_rows = (
            pre.players.active
            & locomotion_enabled
            & position_update_enabled
            & (~visible_restart_pinned)
            & (~approach_rows)
        )
        motion_drive = jnp.any(
            jnp.abs(pre.players.velocity) > GEOMETRY_EPS, axis=-1
        ) | jnp.any(
            jnp.abs(effective_action.desired_velocity) > GEOMETRY_EPS,
            axis=-1,
        )
        self_motion = movement_parameter_rows & motion_drive & path_changed

        impact_rows = _actor_rows(
            physics.player_impact.occurred,
            physics.player_impact.actor,
            player_count,
        ) | _actor_rows(
            physics.player_impact.occurred,
            physics.player_impact.victim,
            player_count,
        )
        movable_impact = impact_rows & (~separation_pinned) & (~visible_restart_pinned)
        collision_velocity = movable_impact
        separation_position = movable_impact & path_changed

        safe_actor = jnp.clip(masked_deliberate_contact.actor, 0, player_count - 1)
        automatic_hand_response = (
            masked_deliberate_contact.mechanism == MECHANISM_GOALKEEPER_HAND
        ) & (effective_action.force_power[safe_actor] <= 0.0)
        directed_parameter_rows = parameter_rows & _actor_rows(
            ~automatic_hand_response,
            masked_deliberate_contact.actor,
            player_count,
        )
        parameter_consumed = (
            _bits(movement_parameter_rows, PARAMETER_MOVE, jnp.uint16)
            | _bits(
                directed_parameter_rows,
                PARAMETER_FORCE_DIRECTION,
                jnp.uint16,
            )
            | _bits(parameter_rows, PARAMETER_FORCE_POWER, jnp.uint16)
            | _bits(directed_parameter_rows, PARAMETER_LAUNCH, jnp.uint16)
            | _bits(kick_rows, PARAMETER_SPIN, jnp.uint16)
        )
        displacement_source = (
            _bits(self_motion, DISPLACEMENT_SELF_MOTION, jnp.uint16)
            | _bits(
                collision_velocity,
                DISPLACEMENT_COLLISION_VELOCITY,
                jnp.uint16,
            )
            | _bits(
                separation_position,
                DISPLACEMENT_SEPARATION_POSITION,
                jnp.uint16,
            )
            | _bits(
                projection_changed,
                DISPLACEMENT_REFEREE_PROJECTION,
                jnp.uint16,
            )
            | _bits(
                approach_rows,
                DISPLACEMENT_RESTART_APPROACH,
                jnp.uint16,
            )
        )
        flags = (
            _bits(intent_available, ACTION_FLAG_INTENT_AVAILABLE, jnp.uint32)
            | _bits(
                attempted_rows,
                ACTION_FLAG_CONTACT_ATTEMPTED,
                jnp.uint32,
            )
            | _bits(active_rows, ACTION_FLAG_ACTIVE_CONTACT, jnp.uint32)
            | _bits(passive_rows, ACTION_FLAG_PASSIVE_CONTACT, jnp.uint32)
            | _bits(
                parameter_rows,
                ACTION_FLAG_PARAMETERS_APPLIED,
                jnp.uint32,
            )
            | _bits(kick_rows, ACTION_FLAG_KICK_APPLIED, jnp.uint32)
            | _bits(
                body_only_rows,
                ACTION_FLAG_BODY_ONLY_DEFLECTION,
                jnp.uint32,
            )
            | _bits(forced_rows, ACTION_FLAG_FORCED_RELEASE, jnp.uint32)
            | _bits(
                projection_changed,
                ACTION_FLAG_REFEREE_PROJECTION,
                jnp.uint32,
            )
            | _bits(
                approach_rows,
                ACTION_FLAG_ENVIRONMENT_OVERWRITE,
                jnp.uint32,
            )
        )
        return next_carry, _EventfulSubstep(
            events=frame_events,
            action=_ActionSubstepTelemetry(
                effective_intent=effective_action.requested_intent,
                flags=flags,
                eligibility_seen=eligibility_seen,
                parameter_consumed=parameter_consumed,
                displacement_source=displacement_source,
            ),
            render_state=next_carry.state if render_sample_count is not None else None,
            render_offside_state=(
                next_carry.offside_state if render_sample_count is not None else None
            ),
        )

    scan_inputs = (
        jnp.arange(timebase.decimation, dtype=jnp.uint32),
        contest_overrides,
    )
    if minimum_team_players is None:
        if not _collect_events:

            def guarded_substep(carry: _ControlCarry, inputs):
                def advance(_):
                    control, _ = substep(carry, inputs)
                    return control

                frozen = carry.restart_guard.opened_this_control | (
                    period_boundary_penalty_enforced & carry.penalty_settling_contact
                )
                return jax.lax.cond(
                    frozen,
                    lambda _: carry,
                    advance,
                    operand=None,
                ), None

            final, events = jax.lax.scan(guarded_substep, initial, scan_inputs)
        else:

            def eventful_guarded_substep(carry: _ControlCarry, inputs):
                frozen = carry.restart_guard.opened_this_control | (
                    period_boundary_penalty_enforced & carry.penalty_settling_contact
                )
                return jax.lax.cond(
                    frozen,
                    lambda _: (
                        carry,
                        _EventfulSubstep(
                            events=_empty_substep_events(
                                carry.state.ball.position.dtype
                            ),
                            action=_empty_action_substep(
                                physics_action.requested_intent
                            ),
                            render_state=(
                                carry.state if render_sample_count is not None else None
                            ),
                            render_offside_state=(
                                carry.offside_state
                                if render_sample_count is not None
                                else None
                            ),
                        ),
                    ),
                    lambda _: substep(carry, inputs),
                    operand=None,
                )

            final, events = jax.lax.scan(eventful_guarded_substep, initial, scan_inputs)
    else:
        minimum = jnp.asarray(minimum_team_players, dtype=jnp.int32)
        match_initial = _MatchControlCarry(
            control=initial,
            abandoned=jnp.bool_(False),
        )
        if not _collect_events:

            def match_substep(carry: _MatchControlCarry, inputs):
                def advance(_):
                    control, _ = substep(carry.control, inputs)
                    active = control.state.players.active
                    active_per_team = jnp.stack(
                        [
                            jnp.sum(active & (control.state.players.team_id == team))
                            for team in range(2)
                        ]
                    ).astype(jnp.int32)
                    return _MatchControlCarry(
                        control=control,
                        abandoned=jnp.any(active_per_team < minimum),
                    )

                return jax.lax.cond(
                    carry.abandoned
                    | carry.control.restart_guard.opened_this_control
                    | (
                        period_boundary_penalty_enforced
                        & carry.control.penalty_settling_contact
                    ),
                    lambda _: carry,
                    advance,
                    operand=None,
                ), None

            match_final, events = jax.lax.scan(
                match_substep, match_initial, scan_inputs
            )
        else:

            def eventful_match_substep(carry: _MatchControlCarry, inputs):
                def advance(_):
                    control, event = substep(carry.control, inputs)
                    active = control.state.players.active
                    active_per_team = jnp.stack(
                        [
                            jnp.sum(active & (control.state.players.team_id == team))
                            for team in range(2)
                        ]
                    ).astype(jnp.int32)
                    return _MatchControlCarry(
                        control=control,
                        abandoned=jnp.any(active_per_team < minimum),
                    ), event

                return jax.lax.cond(
                    carry.abandoned
                    | carry.control.restart_guard.opened_this_control
                    | (
                        period_boundary_penalty_enforced
                        & carry.control.penalty_settling_contact
                    ),
                    lambda _: (
                        carry,
                        _EventfulSubstep(
                            events=_empty_substep_events(
                                carry.control.state.ball.position.dtype
                            ),
                            action=_empty_action_substep(
                                physics_action.requested_intent
                            ),
                            render_state=(
                                carry.control.state
                                if render_sample_count is not None
                                else None
                            ),
                            render_offside_state=(
                                carry.control.offside_state
                                if render_sample_count is not None
                                else None
                            ),
                        ),
                    ),
                    advance,
                    operand=None,
                )

            match_final, events = jax.lax.scan(
                eventful_match_substep, match_initial, scan_inputs
            )
        final = match_final.control
    final_state = _finish_control_ticks(state, final.state, final.possession_changed)
    frame = ControlFrame(
        state=final_state,
        offside_state=final.offside_state,
        contact_attempted=final.contact_attempted,
        kick_applied=final.kick_applied,
        restart_opened=final.restart_guard.opened_this_control,
        event_budget_exhausted=final.event_budget_exhausted,
        contest_override_valid=final.contest_override_valid,
        penalty_settling_contact=final.penalty_settling_contact,
    )
    if not _collect_events:
        return frame
    action_receipt = _reduce_action_receipt(
        input_action_receipt,
        events.action,
        entry_active=state.players.active,
    )
    if render_sample_count is None:
        return ControlFrameWithEvents(
            *frame,
            action_trace=trace_action(submitted_action),
            action_receipt=action_receipt,
            events=events.events,
        )
    assert render_sample_indices is not None
    return _ControlFrameWithEventsAndRenderSamples(
        *frame,
        action_trace=trace_action(submitted_action),
        action_receipt=action_receipt,
        events=events.events,
        render_state=jax.tree.map(
            lambda value: value[render_sample_indices], events.render_state
        ),
        render_offside_state=jax.tree.map(
            lambda value: value[render_sample_indices], events.render_offside_state
        ),
    )


__all__ = [
    "ACTION_FLAG_NAMES",
    "ACTION_REASON_NAMES",
    "ACTION_RECEIPT_SCHEMA",
    "DISPLACEMENT_SOURCE_NAMES",
    "ELIGIBILITY_NAMES",
    "PARAMETER_NAMES",
    "ActionReceipt",
    "ActionTrace",
    "ControlFrame",
    "ControlFrameWithEvents",
    "FrameEvents",
    "NutmegEvent",
    "sample_control_contest_overrides",
    "step_control_frame",
]
