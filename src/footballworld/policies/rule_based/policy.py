"""Recurrent wiring for the seeded observation-only baseline policy.

This module establishes the public policy contract and a lawful baseline.
Passes recompute prospective Law 11 geometry, rank receiver and interception
arrival, and invert the public FootballWorld rolling/contact physics only for
the selected target.  Off-ball actors retain formation shape, with one visible
outfielder assigned to a loose ball, while goalkeeper cover uses observed ball
kinematics.  Recurrent tactical ages and every decision remain observation-only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.action import IntentAction
from footballworld.core.action_mapping import linf_radial_encode
from footballworld.core.constants import (
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_NONE,
    SAFE_NORM_EPS,
    STATIONARY_SPEED_EPS,
)
from footballworld.core.contact import (
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_DEFLECTION,
    OUTCOME_PARRY,
    OUTCOME_RELEASE,
    OUTCOME_TRAP,
)
from footballworld.dynamics.ball import advance_supported_ground_motion
from footballworld.dynamics.contest import (
    challenge_foul_probability,
    challenge_success_probability,
)
from footballworld.dynamics.stamina import effective_speed_limit
from footballworld.environment.action_availability import intent_availability_hint
from footballworld.environment.management import PlayerTacticalObservation
from footballworld.environment.observation import Observation, RosterMetadata
from footballworld.policies.rule_based.aerial_model import (
    aerial_kick_controls,
    aerial_kick_lookup,
)
from footballworld.policies.rule_based.attack_pattern import (
    AttackPattern,
    progressive_carry_commit_seconds,
    select_attack_pattern,
)
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.contact_timing import (
    dribble_recontact_ready,
)
from footballworld.policies.rule_based.context import build_rule_policy_context
from footballworld.policies.rule_based.deadball_shape import restart_shape_target
from footballworld.policies.rule_based.goalkeeper_contact import (
    select_goalkeeper_contact_intent,
)
from footballworld.policies.rule_based.keeper_aerial import (
    aerial_arrival_score,
    aerial_contest_decision,
    goalkeeper_cover_decision,
    select_goalkeeper_distribution_target,
)
from footballworld.policies.rule_based.opponent_pool import (
    TeamSlotTable,
    gather_opponent_pool,
)
from footballworld.policies.rule_based.pass_model import (
    ground_pass_lookup,
)
from footballworld.policies.rule_based.possession import (
    POSSESSION_CLEAR,
    POSSESSION_DRIBBLE,
    POSSESSION_PASS,
    POSSESSION_SHOT,
    PossessionCandidateTrace,
    decide_possession,
    plan_shot,
)
from footballworld.policies.rule_based.restart import decide_restart
from footballworld.policies.rule_based.shape import shape_movement
from footballworld.policies.rule_based.state import (
    INACTIVE_AGE,
    ROLE_CENTRE_FORWARD,
    ROLE_WIDE_FORWARD,
    RulePolicyState,
    apply_tactical_observation,
    initialize_rule_policy_state,
    update_rule_policy_state,
)
from footballworld.policies.rule_based.tactical_plan import (
    gather_tactical_profile,
    tactical_plan_code,
)
from footballworld.policies.rule_based.tactics import (
    ground_pass_controls,
    lane_completion,
    moving_receiver_target,
    pitch_value,
    pressure,
    prospective_offside,
    reception_evaluation,
)

if TYPE_CHECKING:
    from footballworld.environment.api import FootballWorld


_CONTROL_BOUNDARY_MARGIN_M = 1.0
_CONTROL_TOUCHLINE_GUARD_M = 5.0
_CONTROL_GOAL_LINE_GUARD_M = 5.0


def _touchline_control_risk(
    ball_position: jax.Array,
    ball_velocity: jax.Array,
    half_width: float,
) -> jax.Array:
    """Return rows where an outward-moving ball needs a stronger infield trap."""

    near_touchline = jnp.abs(ball_position[..., 1]) > (
        jnp.float32(half_width) - jnp.float32(_CONTROL_TOUCHLINE_GUARD_M)
    )
    moving_outward = ball_position[..., 1] * ball_velocity[..., 1] > 0.0
    return near_touchline & moving_outward


def _goal_line_control_risk(
    ball_position: jax.Array,
    ball_velocity: jax.Array,
    half_length: float,
) -> jax.Array:
    """Return rows where an outward-moving ball risks crossing a goal line."""

    near_goal_line = jnp.abs(ball_position[..., 0]) > (
        jnp.float32(half_length) - jnp.float32(_CONTROL_GOAL_LINE_GUARD_M)
    )
    moving_outward = ball_position[..., 0] * ball_velocity[..., 0] > 0.0
    return near_goal_line & moving_outward


def _boundary_safe_dribble_control(
    direction: jax.Array,
    power: jax.Array,
    is_dribble: jax.Array,
    touchline_risk: jax.Array,
    inward_touchline: jax.Array,
    touchline_power: float,
) -> tuple[jax.Array, jax.Array]:
    """Keep periodic dribble touches from bypassing touchline-safe control."""

    direction = jnp.where(
        (is_dribble & touchline_risk)[:, None],
        inward_touchline,
        direction,
    )
    power = jnp.where(
        is_dribble & touchline_risk,
        jnp.float32(touchline_power),
        power,
    )
    return direction, power


def _infield_ground_settle_direction(
    ball_relative_xy: jax.Array,
    self_position: jax.Array,
    half_width: float,
) -> jax.Array:
    """Aim a ground trap at the player's feet, clamped inside the touchline."""

    safe_target_y = jnp.clip(
        self_position[..., 1],
        -jnp.float32(half_width) + jnp.float32(_CONTROL_BOUNDARY_MARGIN_M),
        jnp.float32(half_width) - jnp.float32(_CONTROL_BOUNDARY_MARGIN_M),
    )
    safe_target = jnp.stack((self_position[..., 0], safe_target_y), axis=-1)
    ball_position = self_position + ball_relative_xy
    delta = safe_target - ball_position
    return delta / jnp.maximum(
        jnp.linalg.norm(delta, axis=-1, keepdims=True),
        jnp.float32(GEOMETRY_EPS),
    )


class _ActionDecision(NamedTuple):
    """Private action result plus the observer-local loose-ball assignment."""

    action: IntentAction
    loose_chaser: jax.Array
    loose_chase_active: jax.Array
    planned_receiver: jax.Array
    planned_arrival: jax.Array
    planned_eta_ticks: jax.Array
    pass_plan_active: jax.Array
    service_opportunity: jax.Array
    intended_receiver_ids: jax.Array


_ActionForState = Callable[
    [Observation, RosterMetadata, RulePolicyState, jax.Array],
    _ActionDecision,
]
_ActionForStateWithPassDiagnostic = Callable[
    [Observation, RosterMetadata, RulePolicyState, jax.Array],
    tuple[_ActionDecision, PossessionCandidateTrace],
]


_GROUND_LOOSE_HORIZON_S = 2.5
_GROUND_LOOSE_PATH_SAMPLES = 10
_GROUND_LOOSE_STEP_S = _GROUND_LOOSE_HORIZON_S / _GROUND_LOOSE_PATH_SAMPLES

_CARRIER_DECISION_RANDOM_STREAM = 0x43415252
_ATTACK_EPISODE_RANDOM_STREAM = 0x41544550
_RUN_BEHIND_RECEIVER_RANDOM_STREAM = 0x52554E52
_CHALLENGE_RANDOM_STREAM = 0x5441434B
_DEEP_CLEAR_RANDOM_STREAM = 0x434C4541
_OFFSIDE_TIMING_RANDOM_STREAM = 0x4F465344
_RESTART_DECISION_RANDOM_STREAM = 0x52535452
_REBOUND_SHOT_RANDOM_STREAM = 0x52425348
_REBOUND_CHOICE_RANDOM_STREAM = 0x52424348
_QUICK_RELAY_RANDOM_STREAM = 0x51524C59


def _contextual_bernoulli_probability(
    reference: jax.Array,
    context: jax.Array,
    logit_limit: float,
) -> jax.Array:
    """Move a reference Bernoulli probability by bounded context."""

    reference = jnp.asarray(reference, dtype=jnp.float32)
    context = jnp.clip(jnp.asarray(context, dtype=jnp.float32), 0.0, 1.0)
    safe_reference = jnp.clip(reference, 1.0e-6, 1.0 - 1.0e-6)
    reference_logit = jnp.log(safe_reference) - jnp.log1p(-safe_reference)
    contextual = jax.nn.sigmoid(
        reference_logit + jnp.float32(logit_limit) * (2.0 * context - 1.0)
    )
    return jnp.where(
        reference <= 0.0,
        jnp.float32(0.0),
        jnp.where(reference >= 1.0, jnp.float32(1.0), contextual),
    )


def _policy_base_key(
    match_key: jax.Array | None,
    default_seed: int,
) -> jax.Array:
    """Return the immutable key from which causal policy decisions are derived."""

    return jax.random.key(default_seed) if match_key is None else match_key


def _policy_absolute_tick(observations: Observation) -> jax.Array:
    """Return the shared public control tick without depending on one row."""

    return jnp.max(
        observations.match.control_tick,
        initial=jnp.int32(0),
    ).astype(jnp.int32)


def _policy_frame_key(
    observations: Observation,
    base_key: jax.Array,
) -> jax.Array:
    """Fold an immutable match key by absolute tick, independent of chunks."""

    return jax.random.fold_in(
        base_key,
        _policy_absolute_tick(observations).astype(jnp.uint32),
    )


def _stable_decision_key(
    base_key: jax.Array,
    stream: int,
    *components: jax.Array,
) -> jax.Array:
    """Fold one fixed-shape scalar episode/identity fingerprint into a key.

    A word-wise FNV-1a fingerprint keeps the rare-key hot path to one Threefry
    fold rather than compiling one fold for every scalar. The downstream PRNG
    remains authoritative; these constants are hash-domain separators, not
    football coefficients.
    """

    fingerprint = jnp.bitwise_xor(jnp.uint32(2166136261), jnp.uint32(stream))
    for component in components:
        fingerprint = jnp.bitwise_xor(
            fingerprint,
            jnp.asarray(component, dtype=jnp.uint32),
        )
        fingerprint = fingerprint * jnp.uint32(16777619)
    return jax.random.fold_in(base_key, fingerprint)


def _bounded_two_hop_continuation(
    receiver_target,
    support_position,
    support_velocity,
    team_support,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    opponent_max_speed,
    first_arrival_s,
    lookup_distance,
    lookup_time,
    config,
    *,
    defender_acceleration_mps2,
    half_length,
    half_width,
):
    """Score one physically reachable second ground leg per first receiver.

    This inherits SoccerWorld's receiver-to-support continuation principle but
    rejects its full multi-row producer. FootballWorld evaluates one
    receiver-by-support-by-visible-opponent tensor for the sparse carrier row,
    with fixed roster axes and no sampled trajectory or hidden team state.
    The returned values are bounded tactical rankings, not probabilities.
    """

    receiver_target = jnp.asarray(receiver_target, dtype=jnp.float32)
    support_position = jnp.asarray(support_position, dtype=jnp.float32)
    support_velocity = jnp.asarray(support_velocity, dtype=jnp.float32)
    team_support = jnp.asarray(team_support, dtype=jnp.bool_)
    player_count = receiver_target.shape[0]
    lead_s = jnp.minimum(
        jnp.asarray(first_arrival_s, dtype=jnp.float32),
        jnp.float32(config.continuation_support_lead_cap_s),
    )
    support_future = (
        support_position[None, :, :]
        + support_velocity[None, :, :] * lead_s[:, None, None]
    )
    support_inside_pitch = (jnp.abs(support_future[..., 0]) <= half_length - 1.0) & (
        jnp.abs(support_future[..., 1]) <= half_width - 1.0
    )
    support_future = jnp.stack(
        (
            jnp.clip(support_future[..., 0], -half_length + 1.0, half_length - 1.0),
            jnp.clip(support_future[..., 1], -half_width + 1.0, half_width - 1.0),
        ),
        axis=-1,
    )
    second_delta = support_future - receiver_target[:, None, :]
    second_distance = jnp.linalg.norm(second_delta, axis=-1)
    second_time = jnp.interp(second_distance, lookup_distance, lookup_time)
    second_speed = second_distance / jnp.maximum(second_time, jnp.float32(1.0e-3))

    def receiver_lanes(source, targets, speed):
        return lane_completion(
            source,
            targets,
            opponent_position,
            opponent_velocity,
            opponent_visible,
            ball_speed_mps=speed,
            opponent_max_speed_mps=opponent_max_speed,
            defender_acceleration_mps2=defender_acceleration_mps2,
        )

    second_completion = jax.vmap(receiver_lanes)(
        receiver_target,
        support_future,
        second_speed,
    )
    distinct_support = ~jnp.eye(player_count, dtype=jnp.bool_)
    progressive_exit = support_future[..., 0] >= receiver_target[
        :, None, 0
    ] - jnp.float32(config.continuation_backward_tolerance_m)
    eligible = (
        team_support[None, :]
        & distinct_support
        & support_inside_pitch
        & (second_distance <= lookup_distance[-1])
        & (second_distance >= jnp.float32(config.continuation_min_distance_m))
        & (second_distance <= jnp.float32(config.continuation_max_distance_m))
        & progressive_exit
        & (
            second_completion
            >= jnp.float32(config.service_opportunity_completion_floor)
        )
    )
    support_threat = pitch_value(
        support_future.reshape((-1, 2)),
        half_length=half_length,
        half_width=half_width,
    ).reshape((player_count, player_count))
    exit_progress = jnp.clip(
        (
            support_future[..., 0]
            - receiver_target[:, None, 0]
            + jnp.float32(config.continuation_backward_tolerance_m)
        )
        / jnp.float32(config.continuation_max_distance_m),
        0.0,
        1.0,
    )
    value = second_completion * jnp.maximum(support_threat, exit_progress)
    return jnp.max(jnp.where(eligible, value, 0.0), axis=-1).astype(jnp.float32)


def _pass_receiver_chaser_candidates(
    base_candidate,
    last_actor,
    own_live_pass,
):
    """Exclude the original passer when another lawful receiver is visible."""

    base_candidate = jnp.asarray(base_candidate, dtype=jnp.bool_)
    last_actor = jnp.asarray(last_actor, dtype=jnp.bool_)
    own_live_pass = jnp.asarray(own_live_pass, dtype=jnp.bool_)
    if base_candidate.ndim != 2:
        raise ValueError("base_candidate must have observer and roster axes")
    if last_actor.shape != base_candidate.shape:
        raise ValueError("last_actor must match base_candidate")
    if own_live_pass.shape != (base_candidate.shape[0],):
        raise ValueError("own_live_pass must have one value per observer")
    distinct_candidate = base_candidate & (~last_actor)
    has_distinct_candidate = jnp.any(distinct_candidate, axis=-1)
    # Falling back when no distinct receiver is visible keeps the planner
    # total. This does not grant an environment re-touch exemption: actual
    # contact legality remains in the transition gate.
    return jnp.where(
        (own_live_pass & has_distinct_candidate)[:, None],
        distinct_candidate,
        base_candidate,
    )


def _service_decision_due(
    regular_due,
    current_opportunity,
    previous_opportunity,
):
    """Wake on ordinary cadence or one eligible service rising edge."""

    regular_due = jnp.asarray(regular_due, dtype=jnp.bool_)
    current_opportunity = jnp.asarray(current_opportunity, dtype=jnp.bool_)
    previous_opportunity = jnp.asarray(previous_opportunity, dtype=jnp.bool_)
    if current_opportunity.shape != regular_due.shape:
        raise ValueError("current_opportunity must match regular_due")
    if previous_opportunity.shape != regular_due.shape:
        raise ValueError("previous_opportunity must match regular_due")
    return regular_due | (current_opportunity & (~previous_opportunity))


def _intercept_approach_velocity(
    target_delta,
    *,
    contact_radius_m,
    braking_mps2,
):
    """Return target-only arrival velocity under a braking-distance envelope."""

    delta = jnp.asarray(target_delta, dtype=jnp.float32)
    distance = jnp.linalg.norm(delta, axis=-1)
    direction = delta / jnp.maximum(distance[..., None], jnp.float32(GEOMETRY_EPS))
    remaining = jnp.maximum(distance - jnp.float32(contact_radius_m), 0.0)
    closing_speed = jnp.sqrt(2.0 * jnp.float32(braking_mps2) * remaining)
    return (closing_speed[..., None] * direction).astype(jnp.float32)


def _ground_loose_interception(
    context,
    observations,
    roster,
    candidate,
    previous_chaser,
    *,
    ball_radius_m,
    contact_radius_m,
    player_acceleration_mps2,
    player_braking_mps2,
    half_length_m,
    half_width_m,
    physics,
    long_stamina,
    short_stamina,
    arrival_slack_s,
):
    """Forecast one shared turf path and choose the first team arrival.

    Select the runner by a current or future ball/player meeting rather than
    current distance alone. Ten static future times share one observation-level
    ground path; no ball state is advanced per player, and a scan keeps the
    compiled graph bounded. The coarse path reuses the environment's
    supported-ground response, while the environment's smaller physics substeps
    remain authoritative for the actual transition.
    Player ETA includes radial braking before a direction reversal, but does
    not reproduce the full longitudinal/lateral acceleration ellipse. This
    fixed-shape radial approximation ranks pursuit without one locomotion
    simulation per player and candidate time.
    """

    complete_ball_x = jnp.asarray(
        half_length_m + ball_radius_m, dtype=context.ball_position.dtype
    )
    complete_ball_y = jnp.asarray(
        half_width_m + ball_radius_m, dtype=context.ball_position.dtype
    )

    def ground_step(carry, _):
        physical_position_xy, velocity, spin, in_play = carry
        velocity, spin = jax.vmap(
            lambda one_velocity, one_spin: advance_supported_ground_motion(
                one_velocity,
                one_spin,
                dt=_GROUND_LOOSE_STEP_S,
                radius=ball_radius_m,
                physics=physics,
            )
        )(velocity, spin)
        physical_position_xy = (
            physical_position_xy + _GROUND_LOOSE_STEP_S * velocity[:, :2]
        )
        # SoccerWorld clips its receive forecast directly to the pitch. Keep
        # that useful bounded target, but reject the clip-only chronology: an
        # unclipped carry plus the environment's complete-ball extent prevents
        # a sample after an exit from becoming a reachable interception.
        in_play = (
            in_play
            & (jnp.abs(physical_position_xy[:, 0]) <= complete_ball_x)
            & (jnp.abs(physical_position_xy[:, 1]) <= complete_ball_y)
        )
        target_xy = jnp.stack(
            (
                jnp.clip(physical_position_xy[:, 0], -half_length_m, half_length_m),
                jnp.clip(physical_position_xy[:, 1], -half_width_m, half_width_m),
            ),
            axis=-1,
        )
        return (
            physical_position_xy,
            velocity,
            spin,
            in_play,
        ), (target_xy, velocity[:, :2], in_play)

    initial_position_xy = context.ball_position[:, :2]
    initial_velocity = context.ball_velocity.at[:, 2].set(0.0)
    initial_in_play = (
        observations.ball.live
        & (jnp.abs(initial_position_xy[:, 0]) <= complete_ball_x)
        & (jnp.abs(initial_position_xy[:, 1]) <= complete_ball_y)
    )
    (
        (_, _, _, _),
        (
            future_path_time_major,
            future_velocity_time_major,
            future_in_play_time_major,
        ),
    ) = jax.lax.scan(
        ground_step,
        (
            initial_position_xy,
            initial_velocity,
            context.ball_spin,
            initial_in_play,
        ),
        None,
        length=_GROUND_LOOSE_PATH_SAMPLES,
    )
    initial_target = jnp.stack(
        (
            jnp.clip(initial_position_xy[:, 0], -half_length_m, half_length_m),
            jnp.clip(initial_position_xy[:, 1], -half_width_m, half_width_m),
        ),
        axis=-1,
    )
    path = jnp.concatenate(
        (initial_target[:, None, :], jnp.swapaxes(future_path_time_major, 0, 1)),
        axis=1,
    )
    path_velocity = jnp.concatenate(
        (
            initial_velocity[:, None, :2],
            jnp.swapaxes(future_velocity_time_major, 0, 1),
        ),
        axis=1,
    )
    path_in_play = jnp.concatenate(
        (
            initial_in_play[:, None],
            jnp.swapaxes(future_in_play_time_major, 0, 1),
        ),
        axis=1,
    )
    times = jnp.arange(_GROUND_LOOSE_PATH_SAMPLES + 1, dtype=jnp.float32) * jnp.float32(
        _GROUND_LOOSE_STEP_S
    )

    sample_index = jnp.arange(path.shape[1], dtype=jnp.int32)
    last_in_play_index = jnp.max(
        jnp.where(path_in_play, sample_index[None, :], jnp.int32(0)), axis=-1
    )

    delta = path[:, None, :, :] - context.player_position[:, :, None, :]
    center_distance = jnp.sqrt(
        jnp.sum(delta * delta, axis=-1) + jnp.float32(SAFE_NORM_EPS)
    )
    distance = jnp.maximum(center_distance - contact_radius_m, 0.0)
    direction = delta / center_distance[..., None]
    projected_speed = jnp.sum(
        context.player_velocity[:, :, None, :] * direction, axis=-1
    )
    braking = jnp.maximum(
        jnp.asarray(player_braking_mps2, dtype=jnp.float32),
        jnp.float32(1.0e-6),
    )
    moving_away = projected_speed < 0.0
    stop_time = jnp.where(moving_away, -projected_speed / braking, 0.0)
    away_distance = jnp.where(
        moving_away,
        projected_speed * projected_speed / (2.0 * braking),
        0.0,
    )
    travel_distance = distance + away_distance
    initial_speed = jnp.maximum(projected_speed, 0.0)
    maximum_speed = effective_speed_limit(
        roster.max_speed[None, :],
        observations.players.stamina_long,
        observations.players.stamina_short,
        long=long_stamina,
        short=short_stamina,
    )
    maximum_speed = jnp.maximum(maximum_speed, jnp.float32(1.0e-6))
    acceleration = jnp.maximum(
        jnp.asarray(player_acceleration_mps2, dtype=jnp.float32),
        jnp.float32(1.0e-6),
    )
    clipped_initial = jnp.minimum(initial_speed, maximum_speed[:, :, None])
    acceleration_time = (maximum_speed[:, :, None] - clipped_initial) / acceleration
    acceleration_distance = (
        clipped_initial * acceleration_time
        + 0.5 * acceleration * acceleration_time * acceleration_time
    )
    within_acceleration = (
        -clipped_initial
        + jnp.sqrt(
            clipped_initial * clipped_initial + 2.0 * acceleration * travel_distance
        )
    ) / acceleration
    after_acceleration = (
        acceleration_time
        + (travel_distance - acceleration_distance) / maximum_speed[:, :, None]
    )
    travel_eta = jnp.where(
        travel_distance <= acceleration_distance,
        within_acceleration,
        after_acceleration,
    )
    eta = stop_time + travel_eta

    feasible = (
        candidate[:, :, None]
        & path_in_play[:, None, :]
        & (eta <= times[None, None, :] + arrival_slack_s)
    )
    player_has_intercept = jnp.any(feasible, axis=-1)
    first_time_index = jnp.argmax(feasible.astype(jnp.int32), axis=-1)
    observer_index = jnp.arange(path.shape[0], dtype=jnp.int32)[:, None]
    first_target = path[observer_index, first_time_index]
    observer_row = jnp.arange(path.shape[0], dtype=jnp.int32)
    last_in_play_target = path[observer_row, last_in_play_index]
    terminal_target = jnp.broadcast_to(
        last_in_play_target[:, None, :], first_target.shape
    )
    player_target = jnp.where(
        player_has_intercept[:, :, None], first_target, terminal_target
    )

    feasible_cost = jnp.min(
        jnp.where(
            feasible,
            times[None, None, :] + jnp.float32(1.0e-3) * eta,
            jnp.inf,
        ),
        axis=-1,
    )
    team_has_intercept = jnp.any(player_has_intercept & candidate, axis=-1)
    last_in_play_eta = jnp.take_along_axis(
        eta,
        jnp.broadcast_to(last_in_play_index[:, None, None], (*eta.shape[:2], 1)),
        axis=-1,
    )[:, :, 0]
    player_cost = jnp.where(
        team_has_intercept[:, None],
        feasible_cost,
        last_in_play_eta,
    )
    forecast_winner = jnp.argmin(jnp.where(candidate, player_cost, jnp.inf), axis=-1)
    # Keep the preceding assignment while that runner remains a lawful
    # candidate and still reaches the shared forecast. If nobody reaches the
    # finite horizon, retain the candidate rather than changing runners on a
    # sub-sample cost tie. This is hysteresis from feasibility, not a fitted
    # football threshold.
    safe_previous = jnp.clip(previous_chaser, 0, candidate.shape[1] - 1)
    previous_valid = (previous_chaser >= 0) & (previous_chaser < candidate.shape[1])
    previous_candidate = candidate[observer_row, safe_previous]
    previous_reachable = player_has_intercept[observer_row, safe_previous]
    retain_previous = (
        previous_valid
        & previous_candidate
        & (previous_reachable | (~team_has_intercept))
    )
    winner = jnp.where(retain_previous, safe_previous, forecast_winner)
    selected_target = player_target[jnp.arange(path.shape[0], dtype=jnp.int32), winner]
    selected_time_index = first_time_index[observer_row, winner]
    last_in_play_time = times[last_in_play_index]
    selected_time_s = jnp.where(
        team_has_intercept, times[selected_time_index], last_in_play_time
    )
    selected_velocity = path_velocity[observer_row, selected_time_index]
    last_in_play_velocity = path_velocity[observer_row, last_in_play_index]
    selected_velocity = jnp.where(
        team_has_intercept[:, None], selected_velocity, last_in_play_velocity
    )
    return (
        winner.astype(jnp.int32),
        selected_target.astype(jnp.float32),
        selected_velocity.astype(jnp.float32),
        selected_time_s.astype(jnp.float32),
    )


class PolicyStep(NamedTuple):
    """One policy action and the observation-only memory for the next step."""

    action: IntentAction
    state: RulePolicyState


class PolicyEventStep(NamedTuple):
    """Capture-only action result with the submitted pass receiver receipt."""

    action: IntentAction
    state: RulePolicyState
    intended_receiver_ids: jax.Array


class PolicyPassDiagnosticStep(NamedTuple):
    """Explicit diagnostic step; ordinary rollout methods never return this tree."""

    action: IntentAction
    state: RulePolicyState
    intended_receiver_ids: jax.Array
    carrier_slot: jax.Array
    pass_candidates: PossessionCandidateTrace


def _require_si_policy_inputs(
    observations: Observation, roster: RosterMetadata
) -> None:
    if type(observations) is not Observation:
        raise TypeError(
            "RuleBasedPolicy requires SI observations; use env.observe_all_si()"
        )
    if type(roster) is not RosterMetadata:
        raise TypeError(
            "RuleBasedPolicy requires SI roster metadata; use env.roster_metadata_si()"
        )


@dataclass(frozen=True)
class RuleBasedPolicy:
    """Policy definition with explicit, fixed-shape rollout memory.

    Call :meth:`initialize` once, then :meth:`step` for every control frame.
    Direct ``policy(observations, roster)`` calls remain available for
    open-play compatibility, but reset memory on every invocation.
    """

    config: RulePolicyConfig
    team_tactical_plan: tuple[int, int]
    counterpress_window_ticks: int
    secure_control_window_ticks: int
    _action_for_state: _ActionForState
    _action_for_state_with_pass_diagnostic: _ActionForStateWithPassDiagnostic

    def initialize(
        self,
        observations: Observation,
        roster: RosterMetadata,
    ) -> RulePolicyState:
        """Initialize memory from SI observations only."""

        _require_si_policy_inputs(observations, roster)
        return initialize_rule_policy_state(
            observations,
            roster,
            team_tactical_plan=jnp.asarray(self.team_tactical_plan, dtype=jnp.int32),
        )

    @staticmethod
    def apply_tactics(
        tactics: PlayerTacticalObservation,
        roster: RosterMetadata,
        state: RulePolicyState,
    ) -> RulePolicyState:
        """Apply one SI own-team manager observation to movement targets."""

        if type(tactics) is not PlayerTacticalObservation:
            raise TypeError(
                "RuleBasedPolicy requires SI tactics; "
                "use env.observe_player_tactics_si()"
            )
        if type(roster) is not RosterMetadata:
            raise TypeError(
                "RuleBasedPolicy requires SI roster metadata; use env.roster_metadata_si()"
            )
        return apply_tactical_observation(state, tactics, roster)

    def _step_internal(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: RulePolicyState,
        match_key: jax.Array | None = None,
        *,
        with_pass_diagnostic: bool = False,
    ) -> PolicyEventStep | PolicyPassDiagnosticStep:
        """Advance memory through either the lean or explicit diagnostic path."""

        if type(with_pass_diagnostic) is not bool:
            raise TypeError("with_pass_diagnostic must be a static bool")
        _require_si_policy_inputs(observations, roster)
        next_state = update_rule_policy_state(state, observations, roster)
        bounded_counterpress_age = jnp.where(
            (next_state.counterpress_age >= 0)
            & (next_state.counterpress_age < self.counterpress_window_ticks),
            next_state.counterpress_age,
            jnp.int32(INACTIVE_AGE),
        )
        bounded_secure_control_age = jnp.where(
            (next_state.secure_control_age >= 0)
            & (next_state.secure_control_age < self.secure_control_window_ticks),
            next_state.secure_control_age,
            jnp.int32(INACTIVE_AGE),
        )
        next_state = next_state._replace(
            counterpress_age=bounded_counterpress_age,
            secure_control_age=bounded_secure_control_age,
        )
        base_key = _policy_base_key(match_key, self.config.default_seed)
        if with_pass_diagnostic:
            decision, pass_candidates = self._action_for_state_with_pass_diagnostic(
                observations, roster, next_state, base_key
            )
        else:
            decision = self._action_for_state(
                observations, roster, next_state, base_key
            )
        next_state = next_state._replace(
            loose_chaser=jnp.where(
                decision.loose_chase_active,
                decision.loose_chaser,
                jnp.int32(NO_PLAYER),
            ),
            planned_receiver=jnp.where(
                decision.pass_plan_active,
                decision.planned_receiver,
                jnp.int32(NO_PLAYER),
            ),
            planned_receiver_id=jnp.where(
                decision.pass_plan_active,
                roster.player_id[jnp.maximum(decision.planned_receiver, 0)],
                jnp.int32(NO_PLAYER),
            ),
            planned_arrival=jnp.where(
                decision.pass_plan_active[:, None],
                decision.planned_arrival,
                jnp.float32(0.0),
            ),
            planned_eta_ticks=jnp.where(
                decision.pass_plan_active,
                decision.planned_eta_ticks,
                jnp.int32(0),
            ),
            service_opportunity=decision.service_opportunity,
        )
        if with_pass_diagnostic:
            return PolicyPassDiagnosticStep(
                action=decision.action,
                state=next_state,
                intended_receiver_ids=decision.intended_receiver_ids,
                carrier_slot=jnp.asarray(pass_candidates.carrier_slot, dtype=jnp.int32),
                pass_candidates=pass_candidates,
            )
        return PolicyEventStep(
            action=decision.action,
            state=next_state,
            intended_receiver_ids=decision.intended_receiver_ids,
        )

    def _step_with_event_receipt(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: RulePolicyState,
        match_key: jax.Array | None = None,
    ) -> PolicyEventStep:
        result = self._step_internal(
            observations, roster, state, match_key, with_pass_diagnostic=False
        )
        if not isinstance(result, PolicyEventStep):
            raise TypeError("lean policy path returned a diagnostic step")
        return result

    def step(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: RulePolicyState,
        match_key: jax.Array | None = None,
    ) -> PolicyStep:
        """Advance causal memory and emit this control frame's action."""

        result = self._step_with_event_receipt(observations, roster, state, match_key)
        return PolicyStep(action=result.action, state=result.state)

    def step_with_event_receipt(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: RulePolicyState,
        match_key: jax.Array | None = None,
    ) -> PolicyEventStep:
        """Return the ordinary step plus capture-only intended receiver IDs."""

        return self._step_with_event_receipt(observations, roster, state, match_key)

    def step_with_pass_diagnostic(
        self,
        observations: Observation,
        roster: RosterMetadata,
        state: RulePolicyState,
        match_key: jax.Array | None = None,
    ) -> PolicyPassDiagnosticStep:
        """Return fixed candidate telemetry without changing ordinary step output."""

        result = self._step_internal(
            observations, roster, state, match_key, with_pass_diagnostic=True
        )
        if not isinstance(result, PolicyPassDiagnosticStep):
            raise TypeError("diagnostic policy path returned a lean step")
        return result

    def __call__(
        self,
        observations: Observation,
        roster: RosterMetadata,
    ) -> IntentAction:
        """Evaluate one SI frame with freshly initialized memory.

        A direct call made part-way through a restart can only anchor restart
        randomness at the current observed tick: the earlier restart age is
        not present in a freshly initialized policy state. Recurrent
        :meth:`step` and chunk continuation retain the true observed age.
        """

        _require_si_policy_inputs(observations, roster)
        state = self.initialize(observations, roster)
        base_key = _policy_base_key(None, self.config.default_seed)
        return self._action_for_state(observations, roster, state, base_key).action


def _row_gather(values: jax.Array, row: jax.Array, index: jax.Array) -> jax.Array:
    """Gather one roster slot from each observer row."""

    return values[row, index]


def _encode(direction: jax.Array, power: jax.Array) -> jax.Array:
    """Encode a team-relative planar target without a zero-direction bias."""

    return linf_radial_encode(direction, power).astype(jnp.float32)


def _select_action_intent(
    normal_contact: jax.Array,
    shoot: jax.Array,
    pass_ball: jax.Array,
    clear_ball: jax.Array,
    loose_control: jax.Array,
    challenge_contact: jax.Array,
    defensive_clear_contact: jax.Array,
    goalkeeper_hand_claim: jax.Array,
    goalkeeper_foot_pass: jax.Array,
    goalkeeper_foot_clearance: jax.Array,
    restart_release: jax.Array,
    restart_penalty_shot: jax.Array,
    request_contact: jax.Array,
    availability: jax.Array,
) -> jax.Array:
    """Label existing policy branches without changing their controls.

    Later assignments deliberately mirror the physical-control override order:
    restart decisions override goalkeeper decisions, which override ordinary
    open-play decisions.  Rows that do not ultimately request contact remain
    MOVE, including invalid actors and blocked retouches.
    """

    intent = jnp.full(normal_contact.shape, INTENT_MOVE, dtype=jnp.int32)
    intent = jnp.where(loose_control, INTENT_CONTROL, intent)
    intent = jnp.where(challenge_contact, INTENT_CHALLENGE, intent)
    intent = jnp.where(defensive_clear_contact, INTENT_CLEAR, intent)
    normal_intent = jnp.where(
        shoot,
        INTENT_SHOT,
        jnp.where(
            pass_ball,
            INTENT_PASS,
            jnp.where(clear_ball, INTENT_CLEAR, INTENT_CONTROL),
        ),
    )
    intent = jnp.where(normal_contact, normal_intent, intent)
    intent = jnp.where(goalkeeper_hand_claim, INTENT_CONTROL, intent)
    intent = jnp.where(goalkeeper_foot_pass, INTENT_PASS, intent)
    intent = jnp.where(goalkeeper_foot_clearance, INTENT_CLEAR, intent)
    restart_intent = jnp.where(
        restart_penalty_shot,
        INTENT_SHOT,
        INTENT_PASS,
    )
    intent = jnp.where(restart_release, restart_intent, intent)
    intent = jnp.where(request_contact, intent, INTENT_MOVE).astype(jnp.int32)
    selected_available = jnp.take_along_axis(
        availability,
        intent[:, None],
        axis=-1,
    )[:, 0]
    return jnp.where(selected_available, intent, INTENT_MOVE).astype(jnp.int32)


def _finalize_policy_action(
    intent: jax.Array,
    move: jax.Array,
    force_to_ball: jax.Array,
    launch: jax.Array,
    spin: jax.Array,
    gaze_center: jax.Array,
) -> IntentAction:
    """Keep continuous controls independent of the categorical intent.

    Dynamics decide whether a contact is physically and legally realized.
    Keeping these values open for all six intents avoids an action-space mask
    that would unnecessarily couple a model's continuous head to its intent
    head; unused MOVE parameters remain inert rather than being overwritten.
    """

    return IntentAction(
        intent=intent,
        move=move,
        force_to_ball=force_to_ball.astype(jnp.float32),
        launch=launch.astype(jnp.float32),
        spin=spin.astype(jnp.float32),
        gaze_center=gaze_center.astype(jnp.float32),
    )


def make_rule_based_policy(
    env: FootballWorld,
    config: RulePolicyConfig | None = None,
) -> RuleBasedPolicy:
    """Build a small seeded-reproducible baseline over public observations only.

    The returned policy consumes the leading observer axis independently: an
    actor may select among the player slots in its own observation row, but it
    never combines another observer's row or reads a rollout ``State``.  Planar
    outputs stay in the environment's team-attacking coordinate frame.
    """

    config = RulePolicyConfig() if config is None else config
    if not isinstance(config, RulePolicyConfig):
        raise TypeError("config must be RulePolicyConfig or None")
    team_tactical_plan = tuple(
        tactical_plan_code(plan) for plan in config.team_tactical_plans
    )
    half_length = float(env.stadium.half_length)
    half_width = float(env.stadium.half_width)
    goal_width = float(env.stadium.goal_width)
    penalty_area_length = float(env.stadium.penalty_area_length)
    ball_radius = float(env.ball.radius)
    gravity = float(env.ball_physics.g)
    air_force_scale = (
        0.5
        * float(env.ball_physics.air_density_kgpm3)
        * math.pi
        * ball_radius
        * ball_radius
        / float(env.ball_physics.ball_mass_kg)
    )
    control_fps = float(env.timebase.control_fps)
    pass_lookup = ground_pass_lookup(
        float(env.timebase.dt_phys),
        float(env.ball.radius),
        float(env.action_scale.kick_speed_max_mps),
        7.0,
        env.ball_physics,
    )
    cross_lookup = aerial_kick_lookup(
        float(env.timebase.dt_phys),
        ball_radius,
        env.action_scale,
        config.cross_launch,
        config.cross_side_spin,
        config.cross_back_spin,
        env.ball_physics,
    )
    dribble_touch_interval_ticks = max(
        1,
        round(config.dribble_touch_interval_s * control_fps),
    )
    challenge_attempt_interval_ticks = max(
        1,
        round(config.challenge_attempt_interval_s * control_fps),
    )
    counterpress_window_ticks = max(
        1,
        round(config.counterpress_window_s * control_fps),
    )
    quick_relay_window_ticks = max(
        1,
        round(config.quick_relay_window_s * control_fps),
    )
    kickoff_path_window_ticks = max(
        1,
        round(config.kickoff_path_window_s * control_fps),
    )

    control_power = min(
        1.0,
        config.dribble_power
        * env.action_scale.kick_speed_max_mps
        / env.action_scale.control_request_speed_max_mps,
    )
    gaze_limit_radians = math.radians(env.perception.gaze_yaw_limit_degrees)

    def action_for_state(
        observations: Observation,
        roster: RosterMetadata,
        _policy_state: RulePolicyState,
        base_key: jax.Array,
        *,
        with_pass_diagnostic: bool = False,
    ) -> _ActionDecision | tuple[_ActionDecision, PossessionCandidateTrace]:
        absolute_tick = _policy_absolute_tick(observations)
        frame_key = _policy_frame_key(observations, base_key)
        context = build_rule_policy_context(observations, roster)
        opponent_pool = gather_opponent_pool(
            context,
            roster,
            TeamSlotTable(
                index=_policy_state.team_slot_index,
                valid=_policy_state.team_slot_valid,
            ),
        )
        player_count = context.self_index.shape[0]
        if observations.players.relative_position.shape != (
            player_count,
            player_count,
            2,
        ):
            raise ValueError("observations must have leading observer and roster axes")
        if roster.team_id.shape != (player_count,):
            raise ValueError("roster must describe the observation roster axis")

        row = jnp.arange(player_count, dtype=jnp.int32)
        self_index = context.self_index
        self_team = context.self_team
        safe_self_team = jnp.clip(self_team, 0, 1)
        self_stamina_long = _row_gather(
            observations.players.stamina_long, row, self_index
        )
        self_stamina_short = _row_gather(
            observations.players.stamina_short, row, self_index
        )
        self_max_speed = effective_speed_limit(
            roster.max_speed[self_index],
            self_stamina_long,
            self_stamina_short,
            long=env.long_stamina,
            short=env.short_stamina,
        )
        row_tactical = gather_tactical_profile(
            _policy_state.team_tactical_plan[safe_self_team]
        )

        self_goalkeeper = context.self_goalkeeper

        active = context.self_active
        own_visible = _row_gather(context.player_visible, row, self_index)
        own_possessor = _row_gather(observations.players.possessor, row, self_index)
        own_restart_taker = _row_gather(
            observations.players.restart_taker, row, self_index
        )
        own_release_taker = _row_gather(
            observations.players.release_taker, row, self_index
        )
        contact_available = _row_gather(
            observations.players.contact_may_occur_this_frame,
            row,
            self_index,
        )
        athletic_contact_available = (
            _row_gather(
                observations.players.aerial_recovery_substeps,
                row,
                self_index,
            )
            < env.timebase.decimation
        )

        visible = context.player_visible
        participating = context.participating
        same_team = context.same_team
        team_player = same_team & participating & visible
        teammate = context.teammate
        opponent = context.opponent

        relative_players = observations.players.relative_position
        ball_xy = observations.ball.relative_state[:, :2]
        ball_visible = context.ball_visible
        ball_live = observations.ball.live
        self_position = context.self_position
        self_body = jnp.stack(
            (
                _row_gather(observations.players.facing_cos, row, self_index),
                _row_gather(observations.players.facing_sin, row, self_index),
            ),
            axis=-1,
        )
        ball_position = self_position + ball_xy
        goal_direction = jnp.stack(
            (
                jnp.asarray(half_length, dtype=jnp.float32) - ball_position[:, 0],
                -ball_position[:, 1],
            ),
            axis=-1,
        )

        possession_known = observations.possession.known
        own_team_possession = possession_known & (
            observations.possession.team == self_team
        )
        possession_present = observations.possession.team != NO_TEAM
        opponent_possession = (
            possession_known
            & possession_present
            & (observations.possession.team != self_team)
        )

        # A visible possession flag is the only actor target used for pressure.
        visible_opponent_carrier = observations.players.possessor & opponent
        has_visible_carrier = jnp.any(visible_opponent_carrier, axis=-1)
        carrier_index = jnp.argmax(visible_opponent_carrier.astype(jnp.int32), axis=-1)
        carrier_relative = _row_gather(relative_players, row, carrier_index)

        pressure_candidate = team_player & (~roster.is_goalkeeper[None, :])
        pressure_candidate = jnp.where(
            jnp.any(pressure_candidate, axis=-1, keepdims=True),
            pressure_candidate,
            team_player,
        )
        teammate_to_carrier = jnp.linalg.norm(
            relative_players - carrier_relative[:, None, :], axis=-1
        )
        pressure_index = jnp.argmin(
            jnp.where(pressure_candidate, teammate_to_carrier, jnp.inf),
            axis=-1,
        )
        nearest_defender = has_visible_carrier & (self_index == pressure_index)

        carrier_distance = jnp.linalg.norm(carrier_relative, axis=-1)
        pressure_power = jnp.where(
            nearest_defender & (carrier_distance <= config.pressure_distance_m),
            config.pressure_power,
            config.support_power,
        )
        counterpress_power_gain = jnp.where(
            _policy_state.counterpress_age >= 0,
            row_tactical.counterpress_gain,
            1.0,
        )
        pressure_power = jnp.clip(pressure_power * counterpress_power_gain, 0.0, 1.0)

        # Resolve contact geometry before selecting the one sparse planning
        # row. A loose ball that is physically at a goalkeeper's foot can then
        # reuse the ordinary receiver, lane, Law 11, and release-physics graph
        # below instead of compiling a second P-by-opponent planning branch.
        ball_distance = jnp.linalg.norm(ball_xy, axis=-1)
        ball_height = observations.ball.relative_state[:, 2]
        ball_horizontal_velocity = (
            observations.ball.relative_state[:, 3:5] + observations.self_state.velocity
        )
        ball_horizontal_speed = jnp.linalg.norm(ball_horizontal_velocity, axis=-1)
        relative_ball_horizontal_velocity = observations.ball.relative_state[:, 3:5]
        closing_dot = jnp.sum(ball_xy * relative_ball_horizontal_velocity, axis=-1)
        relative_speed_squared = jnp.sum(
            relative_ball_horizontal_velocity * relative_ball_horizontal_velocity,
            axis=-1,
        )
        ball_approaching = (closing_dot < 0.0) & (relative_speed_squared > GEOMETRY_EPS)
        closest_time = jnp.clip(
            -closing_dot / jnp.maximum(relative_speed_squared, GEOMETRY_EPS),
            0.0,
            env.timebase.control_dt,
        )
        closest_relative_xy = (
            ball_xy + relative_ball_horizontal_velocity * closest_time[:, None]
        )
        closest_ball_height = jnp.maximum(
            ball_radius,
            ball_height
            + observations.ball.relative_state[:, 5] * closest_time
            - 0.5 * env.ball_physics.g * closest_time * closest_time,
        )
        closest_ball_position = (
            context.ball_position[:, :2]
            + ball_horizontal_velocity * closest_time[:, None]
        )
        foot_contact_reachable = (
            (ball_distance <= env.reach.carry_radius_m + env.ball.radius)
            & (
                ball_height
                <= roster.height[self_index] * env.action_scale.pelvis_height_factor
                + env.ball.radius
            )
            & (
                ball_horizontal_speed
                + env.reach.height_speed_penalty_mps_per_m * ball_height
                <= env.reach.block_speed_limit_mps
            )
        )
        current_hand_contact_reachable = (
            ball_distance <= env.reach.goalkeeper_radius_m + env.ball.radius
        ) & (ball_height <= roster.reach_height[self_index] + env.ball.radius)
        closest_hand_contact_reachable = (
            jnp.linalg.norm(closest_relative_xy, axis=-1)
            <= env.reach.goalkeeper_radius_m + env.ball.radius
        ) & (closest_ball_height <= roster.reach_height[self_index] + env.ball.radius)
        mechanism_contact_available = contact_available & athletic_contact_available
        restart_active = observations.restart.kind != RK_NONE
        goalkeeper_contact_eligible = (
            self_goalkeeper
            & (~restart_active)
            & ball_visible
            & ball_live
            & (~(possession_known & own_team_possession))
            & contact_available
        )
        own_penalty_depth = half_length + context.ball_position[:, 0]
        ball_in_own_penalty_area = (
            (own_penalty_depth >= 0.0)
            & (own_penalty_depth <= env.stadium.penalty_area_length)
            & (
                jnp.abs(context.ball_position[:, 1])
                <= 0.5 * env.stadium.penalty_area_width
            )
        )
        closest_penalty_depth = half_length + closest_ball_position[:, 0]
        closest_ball_in_own_penalty_area = (
            (closest_penalty_depth >= 0.0)
            & (closest_penalty_depth <= env.stadium.penalty_area_length)
            & (
                jnp.abs(closest_ball_position[:, 1])
                <= 0.5 * env.stadium.penalty_area_width
            )
        )
        predicted_hand_claim = (
            ball_approaching
            & closest_hand_contact_reachable
            & closest_ball_in_own_penalty_area
        )
        hand_contact_reachable = athletic_contact_available & (
            (current_hand_contact_reachable & ball_in_own_penalty_area)
            | predicted_hand_claim
        )
        ball_in_own_penalty_area = ball_in_own_penalty_area | predicted_hand_claim
        handling_restricted = (
            observations.match.gk_handling_restricted_team != NO_TEAM
        ) & (observations.match.gk_handling_restricted_team == self_team)
        goalkeeper_foot_required = (
            goalkeeper_contact_eligible
            & foot_contact_reachable
            & (handling_restricted | (~ball_in_own_penalty_area))
        )

        # One sparse planning row bounds receiver scoring. During open play it
        # is the unique possessor; during a restart it is the unique taker. A
        # loose, currently reachable goalkeeper foot contact is the final
        # fallback. No two non-overlapping players can reach the same ball at
        # once, so one argmax row is sufficient without a second lax.map.
        has_restart_taker = jnp.any(own_restart_taker)
        restart_row = jnp.argmax(own_restart_taker.astype(jnp.int32))
        has_possessor = jnp.any(own_possessor)
        possessor_row = jnp.argmax(own_possessor.astype(jnp.int32))
        has_goalkeeper_foot = jnp.any(goalkeeper_foot_required)
        goalkeeper_foot_row = jnp.argmax(goalkeeper_foot_required.astype(jnp.int32))
        decision_row = jnp.where(
            has_restart_taker,
            restart_row,
            jnp.where(
                has_possessor,
                possessor_row,
                jnp.where(has_goalkeeper_foot, goalkeeper_foot_row, possessor_row),
            ),
        ).astype(
            jnp.int32,
        )
        # Recompute Law 11 eligibility only for the sparse planning row. The
        # public players.offside field is a latch from the previous relevant
        # touch and is therefore not a valid candidate gate for a new pass.
        # Every downstream receiver/restart decision consumes only this row;
        # evaluating all observer rows first was an unused O(P²) policy cost.
        # Carrier-control draws belong to one observed player-control segment
        # and one dribble-decision bucket, not to a render/control frame. Folding both
        # physical slot and public player id prevents a substituted identity
        # from inheriting the outgoing players stream without growing state.
        carrier_control_ticks = observations.possession.control_ticks[decision_row]
        carrier_slot = self_index[decision_row]
        carrier_episode_start_tick = jnp.maximum(
            absolute_tick - jnp.maximum(carrier_control_ticks - 1, 0),
            0,
        )
        carrier_decision_bucket = jnp.maximum(carrier_control_ticks, 0) // jnp.int32(
            dribble_touch_interval_ticks
        )
        carrier_episode_key = _stable_decision_key(
            base_key,
            _CARRIER_DECISION_RANDOM_STREAM,
            carrier_episode_start_tick,
            carrier_slot,
            roster.player_id[carrier_slot],
        )
        carrier_decision_key = jax.random.fold_in(
            carrier_episode_key,
            carrier_decision_bucket.astype(jnp.uint32),
        )
        # Every observer derives its plan only from its own known possession
        # episode. This preserves decentralized partial-observation semantics:
        # another actor seeing the ball cannot change this row's movement.
        # During a publicly proven same-team PASS flight, state.py preserves
        # that row's age and team. Unknown, opposing, and restart rows receive
        # the -1 fail-closed code rather than inventing an attack episode.
        active_attack_episode = (
            possession_known
            & (_policy_state.possession_age >= 0)
            & (_policy_state.attack_phase >= 0)
            & (_policy_state.possession_team != NO_TEAM)
            & (_policy_state.possession_team == self_team)
            & observations.ball.live
            & (observations.restart.kind == RK_NONE)
        )
        attack_episode_age = jnp.maximum(_policy_state.possession_age, 0)
        attack_episode_start_tick = jnp.maximum(absolute_tick - attack_episode_age, 0)
        carrier_team = jnp.clip(self_team[decision_row], 0, 1)
        attack_team_by_row = jnp.clip(_policy_state.possession_team, 0, 1)

        def row_attack_pattern(start_tick, team, active_episode):
            episode_key = _stable_decision_key(
                base_key,
                _ATTACK_EPISODE_RANDOM_STREAM,
                start_tick,
                team,
            )
            pattern = select_attack_pattern(
                episode_key,
                _policy_state.team_tactical_plan[team],
            )
            return jnp.where(active_episode, pattern, jnp.int32(-1))

        attack_pattern_by_row = jax.vmap(row_attack_pattern)(
            attack_episode_start_tick,
            attack_team_by_row,
            active_attack_episode,
        )
        has_attack_episode = active_attack_episode[decision_row]
        attack_pattern = attack_pattern_by_row[decision_row]
        attack_phase_by_row = jnp.where(
            active_attack_episode,
            _policy_state.attack_phase,
            jnp.int32(-1),
        )
        attack_phase = attack_phase_by_row[decision_row]
        attack_episode_key = _stable_decision_key(
            base_key,
            _ATTACK_EPISODE_RANDOM_STREAM,
            attack_episode_start_tick[decision_row],
            carrier_team,
        )
        progressive_commit_s = progressive_carry_commit_seconds(
            carrier_episode_key,
            minimum_s=config.progressive_carry_min_commit_s,
            maximum_s=config.solo_carry_soft_limit_s,
        )
        roster_index = jnp.arange(player_count, dtype=jnp.int32)
        episode_actor = _policy_state.current_possessor[decision_row]
        runner_candidate = (
            (roster.team_id == carrier_team)
            & context.participating[decision_row]
            & (
                (_policy_state.role == ROLE_CENTRE_FORWARD)
                | (_policy_state.role == ROLE_WIDE_FORWARD)
            )
            & (roster_index != episode_actor)
        )
        has_runner = has_attack_episode & jnp.any(runner_candidate)
        safe_runner_candidate = jnp.where(
            has_runner,
            runner_candidate,
            roster_index == 0,
        )
        runner_key = jax.random.fold_in(
            attack_episode_key, _RUN_BEHIND_RECEIVER_RANDOM_STREAM
        )
        run_behind_receiver = jax.random.categorical(
            runner_key,
            jnp.where(safe_runner_candidate, 0.0, -jnp.inf),
        ).astype(jnp.int32)
        run_behind_receiver = jnp.where(
            has_runner,
            run_behind_receiver,
            jnp.int32(-1),
        )
        safe_runner_index = jnp.clip(run_behind_receiver, 0, player_count - 1)
        runner_speed = jnp.linalg.norm(
            context.player_velocity[decision_row, safe_runner_index]
        )
        runner_speed_fraction = jnp.clip(
            runner_speed
            / jnp.maximum(
                roster.max_speed[safe_runner_index], jnp.float32(GEOMETRY_EPS)
            ),
            0.0,
            1.0,
        )
        timing_error_probability = _contextual_bernoulli_probability(
            jnp.float32(config.offside_timing_error_probability),
            runner_speed_fraction,
            config.offside_timing_context_logit_limit,
        )
        timing_error = (
            has_attack_episode
            & has_runner
            & (
                jax.random.uniform(
                    jax.random.fold_in(
                        attack_episode_key, _OFFSIDE_TIMING_RANDOM_STREAM
                    )
                )
                < timing_error_probability
            )
        ) & (attack_pattern == jnp.int32(AttackPattern.RUN_BEHIND_DIRECT))
        safe_run_behind_receiver = jnp.maximum(run_behind_receiver, 0)
        prospective_tolerance = (
            jnp.full((player_count,), jnp.float32(0.10), dtype=jnp.float32)
            .at[safe_run_behind_receiver]
            .set(
                jnp.where(
                    has_runner & timing_error,
                    jnp.float32(config.offside_timing_error_margin_m),
                    jnp.float32(0.10),
                )
            )
        )
        projected_offside_row = prospective_offside(
            context.player_position[decision_row],
            context.player_velocity[decision_row],
            teammate[decision_row],
            context.player_position[decision_row],
            context.player_velocity[decision_row],
            opponent[decision_row],
            context.ball_position[decision_row],
            context.ball_velocity[decision_row],
            tolerance_m=prospective_tolerance,
        )
        carrier_tactical = gather_tactical_profile(
            _policy_state.team_tactical_plan[carrier_team]
        )
        carrier_regular_decision_due = (carrier_control_ticks > 0) & (
            carrier_control_ticks % dribble_touch_interval_ticks == 0
        )

        # restart_age is recurrent observation-only memory. Subtracting it
        # from the public clock reconstructs a chunk-invariant episode anchor;
        # a fresh mid-restart initialization necessarily anchors "now" instead.
        restart_age = jnp.maximum(_policy_state.restart_age[decision_row], 0)
        restart_episode_start_tick = jnp.maximum(absolute_tick - restart_age, 0)
        restart_decision_key = _stable_decision_key(
            base_key,
            _RESTART_DECISION_RANDOM_STREAM,
            restart_episode_start_tick,
            observations.restart.kind[decision_row],
            carrier_slot,
            roster.player_id[carrier_slot],
        )

        pass_target = moving_receiver_target(
            ball_position[decision_row],
            context.player_position[decision_row],
            context.player_velocity[decision_row],
            teammate[decision_row],
            half_length=half_length,
            half_width=half_width,
        )
        pass_delta = pass_target - ball_position[decision_row]
        distance = jnp.linalg.norm(pass_delta, axis=-1)
        pass_candidate = teammate[decision_row] & (~projected_offside_row)

        # Rank candidates with the same factory-calibrated rolling model used
        # for the selected pass.  Interpolation is cheap, while a fixed 18 m/s
        # estimate systematically gave long passes unrealistically early
        # defender-arrival windows.
        lookup_distance = jnp.asarray(pass_lookup.distance_m, dtype=jnp.float32)
        lookup_time = jnp.asarray(pass_lookup.travel_time_s, dtype=jnp.float32)
        pass_candidate = pass_candidate & (distance <= lookup_distance[-1])
        nominal_arrival_time = jnp.interp(distance, lookup_distance, lookup_time)
        candidate_ball_speed = distance / jnp.maximum(
            nominal_arrival_time, jnp.float32(1.0e-3)
        )
        # Arrival races must use the same stamina-limited speed that the
        # locomotion transition can actually realize.  The previous ground
        # pass path used raw roster maxima (and the aerial opponent path did
        # likewise), making tired receivers and defenders up to 43% too fast
        # in the policy ETA model.  Compute one roster vector and reuse it for
        # both service families; padding in the compact opponent axis is
        # explicitly zeroed rather than inheriting roster slot zero.
        effective_roster_speed = effective_speed_limit(
            roster.max_speed,
            observations.players.stamina_long[decision_row],
            observations.players.stamina_short[decision_row],
            long=env.long_stamina,
            short=env.short_stamina,
        )
        opponent_speed_index = opponent_pool.roster_index[decision_row]
        effective_opponent_speed = jnp.where(
            opponent_pool.valid[decision_row],
            effective_roster_speed[opponent_speed_index],
            jnp.float32(0.0),
        )
        pass_lane_row = lane_completion(
            ball_position[decision_row],
            pass_target,
            opponent_pool.position[decision_row],
            opponent_pool.velocity[decision_row],
            opponent_pool.available[decision_row],
            ball_speed_mps=candidate_ball_speed,
            opponent_max_speed_mps=effective_opponent_speed,
            defender_acceleration_mps2=(env.player_physics.forward_acceleration_mps2),
            opponent_reduction_index=opponent_pool.roster_index[decision_row],
            opponent_reduction_size=player_count,
        )
        pass_reception_row = reception_evaluation(
            pass_target,
            context.player_position[decision_row],
            context.player_velocity[decision_row],
            opponent_pool.position[decision_row],
            opponent_pool.velocity[decision_row],
            opponent_pool.available[decision_row],
            nominal_arrival_time,
            receiver_max_speed_mps=effective_roster_speed,
            opponent_max_speed_mps=effective_opponent_speed,
            receiver_acceleration_mps2=(env.player_physics.forward_acceleration_mps2),
            opponent_acceleration_mps2=(env.player_physics.forward_acceleration_mps2),
        )
        pass_safety_row = pass_lane_row * pass_reception_row.completion
        # Cross candidates are selected as aerial services before PASS is
        # labelled. The fixed table comes from this environment's free-ball
        # physics; runtime work is two 1-D interpolations plus one P-by-11
        # arrival comparison, never a trajectory solve.
        cross_distance_knots = jnp.asarray(cross_lookup.distance_m, dtype=jnp.float32)
        cross_time_knots = jnp.asarray(cross_lookup.travel_time_s, dtype=jnp.float32)
        cross_player_delta = (
            context.player_position[decision_row] - ball_position[decision_row]
        )
        cross_player_distance = jnp.linalg.norm(cross_player_delta, axis=-1)
        cross_flight_time = jnp.interp(
            cross_player_distance, cross_distance_knots, cross_time_knots
        )
        cross_target = (
            context.player_position[decision_row]
            + context.player_velocity[decision_row] * cross_flight_time[:, None]
        )
        cross_target = jnp.stack(
            (
                jnp.clip(cross_target[:, 0], -half_length + 0.5, half_length - 0.5),
                jnp.clip(cross_target[:, 1], -half_width + 0.5, half_width - 0.5),
            ),
            axis=-1,
        )
        cross_distance = jnp.linalg.norm(
            cross_target - ball_position[decision_row], axis=-1
        )
        cross_flight_time = jnp.interp(
            cross_distance, cross_distance_knots, cross_time_knots
        )
        # One extra fixed-point refinement makes the moving receiver's target
        # and the flight time agree. A longer iterative targeting path is not
        # carried into this deployment policy:
        # this fixed-shape pass adds no loop or dynamic trajectory solve.
        cross_target = (
            context.player_position[decision_row]
            + context.player_velocity[decision_row] * cross_flight_time[:, None]
        )
        cross_target = jnp.stack(
            (
                jnp.clip(cross_target[:, 0], -half_length + 0.5, half_length - 0.5),
                jnp.clip(cross_target[:, 1], -half_width + 0.5, half_width - 0.5),
            ),
            axis=-1,
        )
        cross_distance = jnp.linalg.norm(
            cross_target - ball_position[decision_row], axis=-1
        )
        cross_arrival = aerial_arrival_score(
            cross_target[None, :, :],
            context.player_position[decision_row][None, :, :],
            effective_roster_speed[None, :],
            cross_flight_time[None, :],
            opponent_pool.position[decision_row][None, :, :],
            opponent_pool.velocity[decision_row][None, :, :],
            effective_opponent_speed[None, :],
            opponent_pool.available[decision_row][None, :],
        ).score[0]
        cross_origin = (
            ball_position[decision_row, 0] >= config.cross_start_fraction * half_length
        ) & (
            jnp.abs(ball_position[decision_row, 1])
            >= config.cross_wide_fraction * half_width
        )
        cross_candidate = (
            pass_candidate
            & (~roster.is_goalkeeper)
            & cross_origin
            # The attacking target may sit behind a byline carrier: that is a
            # cutback, not an ineligible cross. The target-zone gate below
            # still excludes backward services into non-attacking areas.
            & (cross_target[:, 0] > 0.45 * half_length)
            & (
                jnp.abs(cross_target[:, 1])
                <= config.cross_target_central_fraction * half_width
            )
            & (cross_distance >= cross_distance_knots[0])
            & (cross_distance <= cross_distance_knots[-1])
        )
        progressive_ground_opportunity = jnp.any(
            pass_candidate
            & (pass_delta[:, 0] >= jnp.float32(config.pass_min_progress_m))
            & (
                pass_safety_row
                >= jnp.float32(config.service_opportunity_completion_floor)
            )
        )
        cross_opportunity = jnp.any(
            cross_candidate
            & (
                cross_arrival
                >= jnp.float32(config.service_opportunity_completion_floor)
            )
        )
        current_service_opportunity = (
            has_possessor
            & (~has_restart_taker)
            & ball_visible[decision_row]
            & mechanism_contact_available[decision_row]
            & foot_contact_reachable[decision_row]
            & (progressive_ground_opportunity | cross_opportunity)
        )
        # Keep the ordinary touch cadence, but wake once on the rising edge of
        # a newly completion-qualified progressive pass or cross. The latch is
        # observer-local and current-frame causal; a persistent option cannot
        # turn this into a release on every control step.
        carrier_decision_due = _service_decision_due(
            carrier_regular_decision_due,
            current_service_opportunity,
            _policy_state.service_opportunity[decision_row],
        )
        service_opportunity_by_row = (
            jnp.zeros((player_count,), dtype=jnp.bool_)
            .at[decision_row]
            .set(current_service_opportunity)
        )

        # SoccerWorld's useful two-hop principle is retained, while its dense
        # all-observer producer is rejected. The helper excludes each receiver
        # itself and boundary-invalid exits, but permits the original passer
        # (a physical wall pass). Law 11 applies at the later second kick, not
        # at the first release.
        continuation_support = team_player[decision_row]
        pass_continuation = jax.lax.cond(
            carrier_decision_due & has_possessor & (~has_restart_taker),
            lambda _: _bounded_two_hop_continuation(
                pass_target,
                context.player_position[decision_row],
                context.player_velocity[decision_row],
                continuation_support,
                opponent_pool.position[decision_row],
                opponent_pool.velocity[decision_row],
                opponent_pool.available[decision_row],
                effective_opponent_speed,
                nominal_arrival_time,
                lookup_distance,
                lookup_time,
                config,
                defender_acceleration_mps2=(
                    env.player_physics.forward_acceleration_mps2
                ),
                half_length=half_length,
                half_width=half_width,
            ),
            lambda _: jnp.zeros((player_count,), dtype=jnp.float32),
            operand=None,
        )
        # A completed reception is not automatically a 0.4 s one-touch pass.
        # The earlier cadence made 92% of observed different-teammate
        # receptions relay within two seconds in a 60 s policy diagnostic,
        # with 17/23 delays pinned to exactly 0.4 s. SoccerWorld gates quick
        # relays by context; adapt that principle to FootballWorld without
        # weakening its intentionally difficult 1v1 carry/contest physics.
        last_contact = observations.possession.last_contact
        carrier_received_control = (
            last_contact.known[decision_row]
            & (last_contact.intent[decision_row] == jnp.int32(INTENT_CONTROL))
            & (last_contact.outcome[decision_row] == jnp.int32(OUTCOME_TRAP))
            & observations.players.last_actor[decision_row, carrier_slot]
        )
        early_relay_window = (
            carrier_received_control
            & (carrier_control_ticks > 0)
            & (carrier_control_ticks <= jnp.int32(quick_relay_window_ticks))
        )
        source_pressure = pressure(
            ball_position[decision_row],
            context.player_position[decision_row],
            context.player_velocity[decision_row],
            opponent[decision_row],
            distance_scale_m=config.pressure_distance_m,
        )
        pass_direction_row = pass_delta / jnp.maximum(
            jnp.linalg.norm(pass_delta, axis=-1)[:, None],
            jnp.float32(GEOMETRY_EPS),
        )
        relay_alignment = jnp.sum(
            pass_direction_row * self_body[decision_row][None, :], axis=-1
        )
        relay_candidate = (
            pass_candidate
            & (pass_safety_row >= jnp.float32(config.quick_relay_completion_floor))
            & (pass_continuation >= jnp.float32(config.quick_relay_continuation_floor))
            & (relay_alignment >= jnp.float32(config.quick_relay_alignment_floor))
        )
        relay_key = jax.random.fold_in(
            carrier_episode_key, jnp.uint32(_QUICK_RELAY_RANDOM_STREAM)
        )
        relay_progress = 0.5 + 0.5 * jnp.tanh(
            pass_delta[:, 0]
            / jnp.maximum(
                jnp.float32(2.0 * config.pass_min_progress_m),
                jnp.float32(GEOMETRY_EPS),
            )
        )
        relay_quality = (
            0.25 * jnp.clip(source_pressure, 0.0, 1.0)
            + 0.20 * jnp.max(jnp.where(relay_candidate, pass_safety_row, 0.0))
            + 0.20 * jnp.max(jnp.where(relay_candidate, pass_continuation, 0.0))
            + 0.15
            * jnp.max(
                jnp.where(
                    relay_candidate,
                    0.5 + 0.5 * relay_alignment,
                    0.0,
                )
            )
            + 0.20 * jnp.max(jnp.where(relay_candidate, relay_progress, 0.0))
        )
        relay_probability = _contextual_bernoulli_probability(
            jnp.float32(config.quick_relay_probability),
            relay_quality,
            config.quick_relay_context_logit_limit,
        )
        relay_draw_allowed = jax.random.uniform(relay_key) < relay_probability
        relay_context_allowed = relay_draw_allowed & (
            source_pressure >= jnp.float32(config.quick_relay_pressure_floor)
        )
        effective_pass_candidate = jnp.where(
            early_relay_window,
            pass_candidate & relay_candidate & relay_context_allowed,
            pass_candidate,
        )
        effective_cross_candidate = jnp.where(
            early_relay_window,
            jnp.zeros_like(cross_candidate),
            cross_candidate,
        )
        restart_ground_candidate_row = teammate[decision_row] & (
            distance <= lookup_distance[-1]
        )
        restart_aerial_candidate_row = (
            teammate[decision_row]
            & (~roster.is_goalkeeper)
            & (cross_distance >= cross_distance_knots[0])
            & (cross_distance <= cross_distance_knots[-1])
        )

        def one_observer_row(value):
            return jax.lax.dynamic_slice_in_dim(value, decision_row, 1, axis=0)

        restart_context = jax.tree_util.tree_map(one_observer_row, context)
        restart_observations = jax.tree_util.tree_map(one_observer_row, observations)
        restart_decision_row = decide_restart(
            restart_context,
            restart_observations,
            projected_offside_row[None, :],
            config,
            decision_key=restart_decision_key,
            player_is_goalkeeper=roster.is_goalkeeper,
            ground_target_xy=pass_target[None, :, :],
            ground_completion=pass_safety_row[None, :],
            ground_candidate=restart_ground_candidate_row[None, :],
            aerial_target_xy=cross_target[None, :, :],
            aerial_completion=cross_arrival[None, :],
            aerial_candidate=restart_aerial_candidate_row[None, :],
            half_length=half_length,
            half_width=half_width,
            goal_width=goal_width,
        )
        restart_decision = jax.tree_util.tree_map(
            lambda value: value[0], restart_decision_row
        )
        carrier_context = jax.tree_util.tree_map(
            lambda value: value[decision_row],
            context,
        )
        possession_result = decide_possession(
            carrier_context,
            projected_offside_row,
            roster.is_goalkeeper,
            config,
            half_length=half_length,
            half_width=half_width,
            goal_width=goal_width,
            pass_target_xy=pass_target,
            pass_completion=pass_safety_row,
            pass_candidate=effective_pass_candidate,
            pass_continuation=pass_continuation,
            cross_target_xy=cross_target,
            cross_completion=cross_arrival,
            cross_candidate=effective_cross_candidate,
            possession_seconds=carrier_control_ticks / control_fps,
            possession_episode_seconds=(
                jnp.maximum(_policy_state.possession_age[decision_row], 0) / control_fps
            ),
            formation_anchor_y=_policy_state.formation_anchor[carrier_slot, 1],
            previous_actor=(
                jnp.arange(player_count)
                == _policy_state.previous_possessor[decision_row]
            ),
            decision_key=carrier_decision_key,
            decision_due=carrier_decision_due,
            tactical=carrier_tactical,
            attack_pattern=attack_pattern,
            attack_phase=attack_phase,
            run_behind_receiver=run_behind_receiver,
            progressive_carry_commit_s=progressive_commit_s,
            shot_launch_radians_per_action_unit=(
                0.5
                * (
                    env.action_scale.launch_max_radians
                    + env.action_scale.ground_launch_down_max_radians
                )
            ),
            with_candidate_trace=with_pass_diagnostic,
        )
        if with_pass_diagnostic:
            possession_decision, pass_candidates = possession_result
        else:
            possession_decision = possession_result
        pass_index = jnp.where(
            possession_decision.target >= 0,
            possession_decision.target,
            jnp.int32(0),
        )
        best_pass_target = jnp.where(
            possession_decision.cross,
            cross_target[pass_index],
            pass_target[pass_index],
        )

        # Match FootballWorld's additive contact impulse and rolling table for
        # the one selected receiver.  This removes fixed pass power/launch and
        # also compensates for currently observed ball velocity.
        selected_pass = ground_pass_controls(
            ball_position[decision_row],
            best_pass_target,
            context.ball_velocity[decision_row],
            ball_height_m=context.ball_position[decision_row, 2],
            kick_speed_max_mps=env.action_scale.kick_speed_max_mps,
            ball_radius_m=env.ball.radius,
            launch_max_radians=env.action_scale.launch_max_radians,
            ground_launch_down_max_radians=(
                env.action_scale.ground_launch_down_max_radians
            ),
            ground_launch_down_reference_height_m=(
                env.action_scale.ground_launch_down_reference_height_m
            ),
            roll_speed_knots_mps=env.ball_physics.roll_v_knots,
            roll_deceleration_knots_mps2=env.ball_physics.roll_d_knots,
            distance_lookup_m=pass_lookup.distance_m,
            launch_speed_lookup_mps=pass_lookup.launch_speed_mps,
            travel_time_lookup_s=pass_lookup.travel_time_s,
        )
        selected_cross = aerial_kick_controls(
            ball_position[decision_row],
            best_pass_target,
            context.ball_velocity[decision_row],
            context.ball_position[decision_row, 2],
            ball_radius,
            possession_decision.spin[0],
            cross_lookup,
            env.action_scale,
        )
        selected_pass_direction = jnp.where(
            possession_decision.cross,
            selected_cross.direction,
            selected_pass.direction,
        )
        selected_pass_power = jnp.where(
            possession_decision.cross, selected_cross.power, selected_pass.power
        )
        selected_pass_launch = jnp.where(
            possession_decision.cross, selected_cross.launch, selected_pass.launch
        )
        selected_pass_reachable = jnp.where(
            possession_decision.cross, selected_cross.reachable, selected_pass.reachable
        )
        pass_direction = (
            jnp.zeros((player_count, 2), dtype=jnp.float32)
            .at[decision_row]
            .set(selected_pass_direction)
        )
        pass_power = (
            jnp.zeros(player_count, dtype=jnp.float32)
            .at[decision_row]
            .set(selected_pass_power)
        )
        pass_launch = (
            jnp.full(player_count, -1.0, dtype=jnp.float32)
            .at[decision_row]
            .set(selected_pass_launch)
        )
        pass_reachable = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(selected_pass_reachable)
        )
        restart_target = ball_position[decision_row] + restart_decision.direction
        selected_restart_ground = ground_pass_controls(
            ball_position[decision_row],
            restart_target,
            context.ball_velocity[decision_row],
            ball_height_m=context.ball_position[decision_row, 2],
            kick_speed_max_mps=env.action_scale.kick_speed_max_mps,
            ball_radius_m=env.ball.radius,
            launch_max_radians=env.action_scale.launch_max_radians,
            ground_launch_down_max_radians=(
                env.action_scale.ground_launch_down_max_radians
            ),
            ground_launch_down_reference_height_m=(
                env.action_scale.ground_launch_down_reference_height_m
            ),
            roll_speed_knots_mps=env.ball_physics.roll_v_knots,
            roll_deceleration_knots_mps2=env.ball_physics.roll_d_knots,
            distance_lookup_m=pass_lookup.distance_m,
            launch_speed_lookup_mps=pass_lookup.launch_speed_mps,
            travel_time_lookup_s=pass_lookup.travel_time_s,
        )
        selected_restart_aerial = aerial_kick_controls(
            ball_position[decision_row],
            restart_target,
            context.ball_velocity[decision_row],
            context.ball_position[decision_row, 2],
            ball_radius,
            -jnp.where(ball_position[decision_row, 1] >= 0.0, 1.0, -1.0)
            * config.cross_side_spin,
            cross_lookup,
            env.action_scale,
        )
        restart_is_shot = restart_decision.shot
        restart_is_aerial = restart_decision.aerial_service
        selected_restart_direction = jnp.where(
            restart_is_aerial,
            selected_restart_aerial.direction,
            selected_restart_ground.direction,
        )
        selected_restart_power = jnp.where(
            restart_is_aerial,
            selected_restart_aerial.power,
            selected_restart_ground.power,
        )
        selected_restart_launch = jnp.where(
            restart_is_aerial,
            selected_restart_aerial.launch,
            selected_restart_ground.launch,
        )
        selected_restart_reachable = jnp.where(
            restart_is_aerial,
            selected_restart_aerial.reachable,
            selected_restart_ground.reachable,
        )
        planned_restart_direction = (
            jnp.zeros((player_count, 2), dtype=jnp.float32)
            .at[decision_row]
            .set(
                jnp.where(
                    restart_is_shot,
                    restart_decision.direction,
                    selected_restart_direction,
                )
            )
        )
        planned_restart_power = (
            jnp.zeros(player_count, dtype=jnp.float32)
            .at[decision_row]
            .set(jnp.where(restart_is_shot, config.shoot_power, selected_restart_power))
        )
        planned_restart_launch = (
            jnp.full(player_count, -1.0, dtype=jnp.float32)
            .at[decision_row]
            .set(
                jnp.where(restart_is_shot, config.shoot_launch, selected_restart_launch)
            )
        )
        planned_restart_valid = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(
                restart_decision.valid & (restart_is_shot | selected_restart_reachable)
            )
        )
        planned_restart_aerial = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(restart_is_aerial)
        )
        planned_restart_shot = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(restart_is_shot)
        )
        # GK_HOLD has one designated goalkeeper, so evaluate only that observer
        # row. This avoids materializing P-by-P distribution matrices.
        distribution_target = pass_target[None, :, :]
        distribution_receiver = teammate[decision_row] & (~projected_offside_row)
        distribution_short_score = jnp.where(
            distribution_receiver, pass_safety_row, -jnp.inf
        )[None, :]
        # GK_HOLD release height differs from ordinary ground kicks. Until a
        # height-conditioned table is shipped, fail closed on aerial punts.
        distribution_long_score = jnp.full(
            (1, player_count), -jnp.inf, dtype=jnp.float32
        )
        distribution_key = jax.random.fold_in(restart_decision_key, 0x474B4449)
        goalkeeper_distribution = select_goalkeeper_distribution_target(
            restart_context,
            restart_observations,
            roster,
            distribution_target,
            distribution_short_score,
            distribution_target,
            distribution_long_score,
            jnp.zeros((1,), dtype=jnp.bool_),
            decision_key=distribution_key,
            temperature=config.receiver_choice_temperature,
        )
        goalkeeper_hold_receiver = goalkeeper_distribution.receiver[0]
        selected_distribution = ground_pass_controls(
            ball_position[decision_row],
            goalkeeper_distribution.target[0],
            context.ball_velocity[decision_row],
            ball_height_m=context.ball_position[decision_row, 2],
            kick_speed_max_mps=env.action_scale.kick_speed_max_mps,
            ball_radius_m=env.ball.radius,
            launch_max_radians=env.action_scale.launch_max_radians,
            ground_launch_down_max_radians=(
                env.action_scale.ground_launch_down_max_radians
            ),
            ground_launch_down_reference_height_m=(
                env.action_scale.ground_launch_down_reference_height_m
            ),
            roll_speed_knots_mps=env.ball_physics.roll_v_knots,
            roll_deceleration_knots_mps2=env.ball_physics.roll_d_knots,
            distance_lookup_m=pass_lookup.distance_m,
            launch_speed_lookup_mps=pass_lookup.launch_speed_mps,
            travel_time_lookup_s=pass_lookup.travel_time_s,
        )
        selected_distribution_direction = selected_distribution.direction
        selected_distribution_power = selected_distribution.power
        selected_distribution_launch = selected_distribution.launch
        selected_distribution_reachable = selected_distribution.reachable
        distribution_direction = (
            jnp.zeros((player_count, 2), dtype=jnp.float32)
            .at[decision_row]
            .set(selected_distribution_direction)
        )
        distribution_power = (
            jnp.zeros(player_count, dtype=jnp.float32)
            .at[decision_row]
            .set(selected_distribution_power)
        )
        distribution_launch = (
            jnp.full(player_count, -1.0, dtype=jnp.float32)
            .at[decision_row]
            .set(selected_distribution_launch)
        )
        distribution_reachable = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(selected_distribution_reachable)
        )
        distribution_valid = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(goalkeeper_distribution.valid[0])
        )
        distribution_use_long = (
            jnp.zeros(player_count, dtype=jnp.bool_)
            .at[decision_row]
            .set(goalkeeper_distribution.use_long[0])
        )
        goalkeeper_hold_distribution = (
            (observations.restart.kind == RK_GK_HOLD)
            & distribution_valid
            & distribution_reachable
        )
        restart_direction = jnp.where(
            goalkeeper_hold_distribution[:, None],
            distribution_direction,
            planned_restart_direction,
        )
        restart_power = jnp.where(
            goalkeeper_hold_distribution,
            distribution_power,
            planned_restart_power,
        )
        restart_launch = jnp.where(
            goalkeeper_hold_distribution,
            distribution_launch,
            planned_restart_launch,
        )

        carrier_kind = (
            jnp.full(player_count, POSSESSION_DRIBBLE, dtype=jnp.int32)
            .at[decision_row]
            .set(possession_decision.kind)
        )
        # Macro/receiver scores may still be evaluated between decision ticks,
        # but they cannot introduce frame-wise movement jitter: until the next
        # decision the carrier continues along its observed velocity, falling
        # back to public body facing only when stationary.
        carrier_velocity = context.self_velocity[decision_row]
        carrier_speed = jnp.linalg.norm(carrier_velocity)
        carrier_motion_direction = carrier_velocity / jnp.maximum(
            carrier_speed,
            jnp.float32(GEOMETRY_EPS),
        )
        carrier_preserved_direction = jnp.where(
            carrier_speed > STATIONARY_SPEED_EPS,
            carrier_motion_direction,
            self_body[decision_row],
        )
        carrier_frame_direction = jnp.where(
            carrier_decision_due,
            possession_decision.direction,
            carrier_preserved_direction,
        )
        carrier_direction = (
            jnp.zeros((player_count, 2), dtype=jnp.float32)
            .at[decision_row]
            .set(carrier_frame_direction)
        )
        carrier_decision_power = (
            jnp.zeros(player_count, dtype=jnp.float32)
            .at[decision_row]
            .set(possession_decision.power)
        )
        carrier_decision_launch = (
            jnp.full(player_count, -1.0, dtype=jnp.float32)
            .at[decision_row]
            .set(possession_decision.launch)
        )
        carrier_decision_spin = (
            jnp.zeros((player_count, 2), dtype=jnp.float32)
            .at[decision_row]
            .set(possession_decision.spin)
        )
        pass_unreachable = (
            own_possessor & (carrier_kind == POSSESSION_PASS) & (~pass_reachable)
        )
        carrier_kind = jnp.where(
            pass_unreachable,
            jnp.int32(POSSESSION_DRIBBLE),
            carrier_kind,
        )
        secure_follow = (
            own_possessor
            & ball_visible
            & (observations.restart.kind == RK_NONE)
            & (_policy_state.secure_control_age >= 0)
        )
        carrier_kind = jnp.where(
            secure_follow,
            jnp.int32(POSSESSION_DRIBBLE),
            carrier_kind,
        )
        carrier_direction = jnp.where(
            pass_unreachable[:, None],
            goal_direction,
            carrier_direction,
        )
        carrier_decision_spin = jnp.where(
            pass_unreachable[:, None], 0.0, carrier_decision_spin
        )
        shoot = own_possessor & (carrier_kind == POSSESSION_SHOT)
        pass_ball = own_possessor & (carrier_kind == POSSESSION_PASS) & pass_reachable
        clear_ball = own_possessor & (carrier_kind == POSSESSION_CLEAR)
        run_behind_release = (
            (attack_pattern == jnp.int32(AttackPattern.RUN_BEHIND_DIRECT))
            & (possession_decision.kind == POSSESSION_PASS)
            & (possession_decision.target == run_behind_receiver)
        )

        carrier_goal_line_risk = _goal_line_control_risk(
            ball_position,
            context.ball_velocity,
            half_length,
        )
        carrier_touchline_risk = _touchline_control_risk(
            ball_position,
            context.ball_velocity,
            half_width,
        )
        carrier_inward_touchline = jnp.stack(
            (
                jnp.zeros_like(ball_position[:, 1]),
                -jnp.sign(ball_position[:, 1]),
            ),
            axis=-1,
        )
        carrier_inward_goal_line = jnp.stack(
            (
                -jnp.sign(ball_position[:, 0]),
                jnp.zeros_like(ball_position[:, 0]),
            ),
            axis=-1,
        )
        boundary_safe_dribble = carrier_kind == POSSESSION_DRIBBLE
        strike_direction = jnp.where(
            pass_ball[:, None], pass_direction, carrier_direction
        )
        # A newly secured planned reception cushions along the receiver's
        # current momentum (or the incoming ball direction from rest). This
        # carries SoccerWorld's first-touch stabilization principle without a
        # hidden shared receive plan or an extra fitted speed coefficient.
        reception_velocity = context.self_velocity
        reception_speed = jnp.linalg.norm(reception_velocity, axis=-1)
        incoming_velocity = context.ball_velocity[:, :2]
        incoming_speed = jnp.linalg.norm(incoming_velocity, axis=-1)
        reception_direction = reception_velocity / jnp.maximum(
            reception_speed[:, None], jnp.float32(GEOMETRY_EPS)
        )
        reception_direction = jnp.where(
            (reception_speed > GEOMETRY_EPS)[:, None],
            reception_direction,
            incoming_velocity
            / jnp.maximum(incoming_speed[:, None], jnp.float32(GEOMETRY_EPS)),
        )
        reception_direction = jnp.where(
            (incoming_speed > GEOMETRY_EPS)[:, None]
            | (reception_speed > GEOMETRY_EPS)[:, None],
            reception_direction,
            goal_direction,
        )
        strike_direction = jnp.where(
            secure_follow[:, None], reception_direction, strike_direction
        )

        strike_power = jnp.where(
            pass_ball,
            pass_power,
            jnp.where(
                carrier_kind == POSSESSION_DRIBBLE,
                control_power,
                carrier_decision_power,
            ),
        )
        strike_direction, strike_power = _boundary_safe_dribble_control(
            strike_direction,
            strike_power,
            boundary_safe_dribble,
            carrier_touchline_risk,
            carrier_inward_touchline,
            config.touchline_dribble_control_power,
        )
        strike_launch = jnp.where(
            pass_ball,
            pass_launch,
            jnp.where(
                carrier_kind == POSSESSION_DRIBBLE, -1.0, carrier_decision_launch
            ),
        )

        carrier_move = _encode(carrier_direction, config.carrier_power)
        secure_ball_velocity = context.ball_velocity[:, :2]
        secure_ball_speed = jnp.linalg.norm(secure_ball_velocity, axis=-1)
        secure_move_power = jnp.minimum(
            secure_ball_speed / jnp.maximum(self_max_speed, GEOMETRY_EPS),
            jnp.float32(config.carrier_power),
        )
        secure_move = _encode(secure_ball_velocity, secure_move_power)
        carrier_move = jnp.where(secure_follow[:, None], secure_move, carrier_move)
        carrier_force = _encode(strike_direction, strike_power)

        restart_force = _encode(restart_direction, restart_power)

        restart_free = observations.restart.kind == RK_NONE
        counterpress_active = _policy_state.counterpress_age >= 0
        active_counterpress = counterpress_active & restart_free
        active_pressure = nearest_defender & restart_free
        reliable_pass_flight_attack = (
            possession_known
            & (observations.possession.team == NO_TEAM)
            & (_policy_state.possession_age >= 0)
            & (_policy_state.possession_team == self_team)
        )
        # A restart still has an attacking and defending team even though the
        # physical possession observation may be deliberately conservative.
        # SoccerWorld carries that phase into its set-piece positioning.  Keep
        # FootballWorld's lean formation field, but preserve the same causal
        # phase continuity instead of leaving the role choice undefined.
        own_restart_phase = restart_active & (observations.restart.team == self_team)
        opponent_restart_phase = restart_active & (
            observations.restart.team != self_team
        )
        shape_own_possession = (
            own_team_possession | reliable_pass_flight_attack | own_restart_phase
        )
        kickoff_path_active = (
            observations.ball.live
            & restart_free
            & (absolute_tick < kickoff_path_window_ticks)
        )
        kickoff_path_phase = jnp.where(
            kickoff_path_active,
            (absolute_tick + jnp.int32(1)) / jnp.float32(kickoff_path_window_ticks),
            jnp.float32(0.0),
        )

        formation_move = shape_movement(
            context,
            _policy_state,
            half_length=half_length,
            half_width=half_width,
            penalty_area_length=penalty_area_length,
            penalty_area_width=env.stadium.penalty_area_width,
            goal_width=goal_width,
            cross_start_fraction=config.cross_start_fraction,
            cross_wide_fraction=config.cross_wide_fraction,
            cross_target_central_fraction=config.cross_target_central_fraction,
            box_mark_lead_s=config.box_mark_lead_s,
            box_mark_runner_margin_m=config.box_mark_runner_margin_m,
            box_mark_ball_margin_m=config.box_mark_ball_margin_m,
            box_mark_goal_side_distance_m=config.box_mark_goal_side_distance_m,
            own_possession=shape_own_possession,
            opponent_possession=(opponent_possession & restart_free)
            | opponent_restart_phase,
            # A goalkeeper hold is live play, but an opponent cannot challenge
            # the ball in the keeper's hands. Keep the block moving back into
            # shape instead of parking the selected presser beside the release
            # point. Ordinary open play retains the one-player press.
            nearest_pressure=active_pressure,
            counterpress_active=active_counterpress,
            tactical=row_tactical,
            attack_pattern=attack_pattern_by_row,
            attack_phase=attack_phase_by_row,
            attack_pattern_shape_shift_m=config.attack_pattern_shape_shift_m,
            forward_pocket_shift_m=config.forward_pocket_shift_m,
            run_behind_receiver=run_behind_receiver,
            run_behind_release=run_behind_release,
            run_behind_timing_error=timing_error,
            offside_line_error_m=jnp.float32(config.offside_timing_error_margin_m),
            forward_run_min_gap_m=config.pass_min_progress_m,
            kickoff_path_phase=kickoff_path_phase,
            kickoff_path_lateral_shift_m=config.kickoff_path_lateral_shift_m,
        )
        # The distance arrival taper applies only to ordinary formation actors.
        # Dedicated pressure, loose-ball, aerial, carrier,
        # and goalkeeper branches keep their own urgency. The values remain a
        # transfer prior: FootballWorld does not alter stamina coefficients to
        # hide a policy that continually commands 0.55 power near its target.
        offball_ramp = jnp.clip(
            formation_move.distance / config.offball_arrival_radius_m,
            0.0,
            1.0,
        )
        offball_surge = jnp.clip(
            (formation_move.distance - config.offball_arrival_radius_m)
            / (config.offball_surge_span_radii * config.offball_arrival_radius_m),
            0.0,
            1.0,
        )
        offball_power = (
            config.offball_walk_power
            + (config.offball_cruise_power - config.offball_walk_power) * offball_ramp
            + (config.offball_surge_cap - config.offball_cruise_power) * offball_surge
        )
        offball_power = jnp.where(
            formation_move.urgent,
            jnp.maximum(offball_power, config.offball_surge_cap),
            offball_power,
        )
        role_power_scale = jnp.asarray(
            (
                1.0,
                config.offball_cb_power_scale,
                config.offball_fb_power_scale,
                config.offball_cm_power_scale,
                config.offball_wm_power_scale,
                config.offball_cf_power_scale,
                config.offball_wf_power_scale,
            ),
            dtype=jnp.float32,
        )[_policy_state.role]
        offball_power = jnp.clip(offball_power * role_power_scale, 0.0, 1.0)
        formation_power = jnp.where(active_pressure, pressure_power, offball_power)
        shape_move = _encode(formation_move.direction, formation_power)

        measured_restart_target, measured_restart_valid = restart_shape_target(
            context,
            _policy_state,
            observations.restart.kind,
            own_restart_phase,
            half_length=half_length,
            half_width=half_width,
        )
        measured_restart_delta = measured_restart_target - self_position
        measured_restart_distance = jnp.linalg.norm(measured_restart_delta, axis=-1)
        measured_restart_direction = measured_restart_delta / jnp.maximum(
            measured_restart_distance[:, None], jnp.float32(GEOMETRY_EPS)
        )
        measured_restart_power = jnp.clip(
            measured_restart_distance / jnp.float32(4.0), 0.0, 1.0
        )
        measured_restart_move = _encode(
            measured_restart_direction, measured_restart_power
        )
        shape_move = jnp.where(
            (measured_restart_valid & (~own_restart_taker))[:, None],
            measured_restart_move,
            shape_move,
        )

        # One first-landing forecast is shared by every outfield candidate in
        # the observer row.  The vertical root is ballistic; horizontal travel
        # uses FootballWorld's current spin-aware drag coefficient from
        # "Soccer ball lift coefficients via trajectory analysis", frozen at
        # the observed state. This is a policy-only lightweight approximation,
        # retaining O(O*P) assignment without a per-player trajectory rollout
        # or a sampled time axis; the environment integrator remains authoritative.
        ball_velocity = context.ball_velocity
        ball_speed_squared = jnp.sum(ball_velocity * ball_velocity, axis=-1)
        ball_speed = jnp.sqrt(ball_speed_squared + jnp.float32(1.0e-12))
        velocity_direction = ball_velocity / ball_speed[:, None]
        perpendicular_spin = (
            context.ball_spin
            - jnp.sum(context.ball_spin * velocity_direction, axis=-1, keepdims=True)
            * velocity_direction
        )
        perpendicular_spin_speed = jnp.sqrt(
            jnp.sum(perpendicular_spin * perpendicular_spin, axis=-1)
            + jnp.float32(1.0e-12)
        )
        spin_parameter = jnp.where(
            ball_speed_squared > jnp.float32(1.0e-12),
            ball_radius * perpendicular_spin_speed / ball_speed,
            0.0,
        )
        crisis_fraction = jax.nn.sigmoid(
            (env.ball_physics.drag_crisis_speed_mps - ball_speed)
            / env.ball_physics.drag_crisis_width_mps
        )
        base_drag = env.ball_physics.drag_coefficient_high_re + (
            env.ball_physics.drag_crisis_drop * crisis_fraction
        )
        spinning_drag = env.ball_physics.spin_drag_scale * jnp.power(
            jnp.maximum(spin_parameter, jnp.float32(1.0e-6)),
            env.ball_physics.spin_drag_exponent,
        )
        spin_interval = env.ball_physics.spin_drag_min_parameter
        spin_fraction = jnp.clip(
            (spin_parameter - spin_interval) / spin_interval, 0.0, 1.0
        )
        spin_fraction = spin_fraction * spin_fraction * (3.0 - 2.0 * spin_fraction)
        drag_coefficient = base_drag + (1.0 - crisis_fraction) * spin_fraction * (
            spinning_drag - base_drag
        )
        drag_rate = air_force_scale * drag_coefficient * ball_speed
        height_above_ground = jnp.maximum(
            context.ball_position[:, 2] - ball_radius, 0.0
        )
        vertical_speed = ball_velocity[:, 2]
        landing_time = (
            vertical_speed
            + jnp.sqrt(
                vertical_speed * vertical_speed + 2.0 * gravity * height_above_ground
            )
        ) / gravity
        travel_scale = jnp.log1p(drag_rate * landing_time) / jnp.maximum(
            drag_rate, jnp.float32(1.0e-6)
        )
        landing_xy = context.ball_position[:, :2] + (
            ball_velocity[:, :2] * travel_scale[:, None]
        )
        landing_xy = jnp.stack(
            (
                jnp.clip(landing_xy[:, 0], -half_length, half_length),
                jnp.clip(landing_xy[:, 1], -half_width, half_width),
            ),
            axis=-1,
        )
        aerial = aerial_contest_decision(
            context,
            observations,
            roster,
            landing_xy,
            landing_time,
            half_length_m=half_length,
            excluded_player=(
                observations.restart_release.active[:, None]
                & observations.restart_release.untouched[:, None]
                & observations.players.release_taker
            ),
            long_stamina_vmax_floor=env.long_stamina.vmax_floor,
            short_stamina_vmax_floor=env.short_stamina.vmax_floor,
            short_stamina_headroom_knee=env.short_stamina.headroom_knee,
        )
        aerial_power = jnp.where(
            aerial.cover_runner, config.support_power, config.approach_power
        )
        aerial_move = _encode(aerial.target - self_position, aerial_power)

        # Use angle cover and a bounded urgent-rush classification derived from
        # the public ball trajectory.  Catch/parry legality remains in physics.
        # Only the one active goalkeeper per team needs the twelve-point path.
        # Select those two observer rows dynamically so substitutions and an
        # acting-goalkeeper assignment follow refreshed roster metadata.  A
        # missing goalkeeper uses row zero only as a shape-static gather dummy;
        # ``goalkeeper_available`` removes it before the two rows are expanded.
        goalkeeper_team = jnp.arange(2, dtype=jnp.int32)[:, None]
        goalkeeper_candidate = (
            (context.self_team[None, :] == goalkeeper_team)
            & context.self_goalkeeper[None, :]
            & context.self_active[None, :]
        )
        goalkeeper_observer = jnp.argmax(goalkeeper_candidate, axis=-1).astype(
            jnp.int32
        )
        goalkeeper_available = jnp.any(goalkeeper_candidate, axis=-1)
        goalkeeper_context = jax.tree_util.tree_map(
            lambda value: value[goalkeeper_observer], context
        )
        goalkeeper_observations = jax.tree_util.tree_map(
            lambda value: value[goalkeeper_observer], observations
        )
        goalkeeper_row = jnp.arange(2, dtype=jnp.int32)
        goalkeeper_self_index = goalkeeper_context.self_index
        goalkeeper_speed = effective_speed_limit(
            roster.max_speed[goalkeeper_self_index],
            goalkeeper_observations.players.stamina_long[
                goalkeeper_row, goalkeeper_self_index
            ],
            goalkeeper_observations.players.stamina_short[
                goalkeeper_row, goalkeeper_self_index
            ],
            long=env.long_stamina,
            short=env.short_stamina,
        )
        goalkeeper_cover = goalkeeper_cover_decision(
            goalkeeper_context,
            goalkeeper_observations,
            half_length_m=half_length,
            goal_width_m=goal_width,
            penalty_area_length_m=env.stadium.penalty_area_length,
            penalty_area_width_m=env.stadium.penalty_area_width,
            ball_radius_m=env.ball.radius,
            gravity_mps2=env.ball_physics.g,
            air_drag_rate_per_s=drag_rate[goalkeeper_observer],
            goalkeeper_speed_mps=goalkeeper_speed,
            goalkeeper_reach_height_m=roster.reach_height[goalkeeper_self_index],
        )
        goalkeeper_direction = (
            goalkeeper_cover.target - goalkeeper_context.self_position
        )
        goalkeeper_cover_power = config.goalkeeper_power * jnp.clip(
            jnp.linalg.norm(goalkeeper_direction, axis=-1)
            / config.approach_slow_radius_m,
            0.0,
            1.0,
        )
        goalkeeper_power = jnp.where(
            goalkeeper_cover.urgent_rush | goalkeeper_cover.sweeper_claim,
            config.pressure_power,
            goalkeeper_cover_power,
        )
        goalkeeper_team_move = _encode(goalkeeper_direction, goalkeeper_power)
        goalkeeper_selector = (
            jnp.arange(player_count, dtype=jnp.int32)[None, :]
            == goalkeeper_observer[:, None]
        ) & goalkeeper_available[:, None]
        goalkeeper_move = jnp.sum(
            jnp.where(
                goalkeeper_selector[:, :, None],
                goalkeeper_team_move[:, None, :],
                jnp.float32(0.0),
            ),
            axis=0,
        ).astype(jnp.float32)

        loose_ball = ball_visible & ball_live & possession_known & (~possession_present)
        # Preserve exact nearest-player selection for a stationary loose ball.
        # A moving supported ball gets one shared turf forecast; all player
        # candidates compare arrival time against that same fixed-size path.
        retouch_runner_blocked = (
            observations.restart_release.active[:, None]
            & observations.restart_release.untouched[:, None]
            & observations.players.release_taker
        )
        base_outfield_chaser_candidate = (
            team_player & (~roster.is_goalkeeper[None, :]) & (~retouch_runner_blocked)
        )
        outfield_chaser_candidate = _pass_receiver_chaser_candidates(
            base_outfield_chaser_candidate,
            observations.players.last_actor,
            reliable_pass_flight_attack,
        )

        player_to_ball = jnp.linalg.norm(
            relative_players - ball_xy[:, None, :], axis=-1
        )
        has_loose_chaser = jnp.any(outfield_chaser_candidate, axis=-1)
        current_chaser_index = jnp.argmin(
            jnp.where(outfield_chaser_candidate, player_to_ball, jnp.inf),
            axis=-1,
        )
        ground_ball = context.ball_position[:, 2] <= ball_radius + 0.08
        supported_ground_ball = (
            (context.ball_position[:, 2] <= ball_radius + GEOMETRY_EPS)
            & (context.ball_velocity[:, 2] <= 0.0)
            & (context.ball_velocity[:, 2] >= -env.ball_physics.ground_settle_vz)
        )
        surface_velocity = ball_radius * jnp.stack(
            (-context.ball_spin[:, 1], context.ball_spin[:, 0]),
            axis=-1,
        )
        ground_motion_speed = jnp.maximum(
            jnp.linalg.norm(context.ball_velocity[:, :2], axis=-1),
            jnp.linalg.norm(
                context.ball_velocity[:, :2] + surface_velocity,
                axis=-1,
            ),
        )
        moving_ground_loose = (
            loose_ball
            & supported_ground_ball
            & has_loose_chaser
            & (ground_motion_speed > STATIONARY_SPEED_EPS)
        )

        # Each observer reconstructs the planned receiver from public PASS
        # release provenance. A still-valid planned receiver is used only as
        # feasibility hysteresis; the forecast remains independently derived
        # from this observer's visible fixed-shape inputs.
        previous_ground_receiver = jnp.where(
            reliable_pass_flight_attack,
            _policy_state.planned_receiver,
            _policy_state.loose_chaser,
        )

        def moving_plan(_):
            return _ground_loose_interception(
                context,
                observations,
                roster,
                outfield_chaser_candidate,
                previous_ground_receiver,
                ball_radius_m=ball_radius,
                # SoccerWorld sends its primary claimant to the receive point.
                # Homing on the ball centre lets the swept contact solver, not
                # a tangent-distance approximation, decide the first entry.
                contact_radius_m=0.0,
                player_acceleration_mps2=(env.player_physics.forward_acceleration_mps2),
                player_braking_mps2=env.player_physics.braking_deceleration_mps2,
                half_length_m=half_length,
                half_width_m=half_width,
                physics=env.ball_physics,
                long_stamina=env.long_stamina,
                short_stamina=env.short_stamina,
                arrival_slack_s=1.0 / control_fps,
            )

        (
            predicted_chaser_index,
            predicted_target,
            _predicted_ball_velocity,
            predicted_eta_s,
        ) = jax.lax.cond(
            jnp.any(moving_ground_loose),
            moving_plan,
            lambda _: (
                current_chaser_index,
                context.ball_position[:, :2],
                context.ball_velocity[:, :2],
                jnp.zeros((player_count,), dtype=jnp.float32),
            ),
            operand=None,
        )
        # A live deliberate pass has one explicit intended receiver.  Keep
        # that lawful runner until the first later contact rather than
        # reassigning at the meeting point.  The former feasibility-only
        # hysteresis could switch one frame before arrival: the designated
        # receiver then resumed formation movement and met the ball only as a
        # passive body deflection.  SoccerWorld likewise holds one
        # ``receive_runner`` for an own live pass.  FootballWorld retains its
        # observer-local public plan and the environment remains authoritative
        # for the actual swept contact.
        observer_row = jnp.arange(player_count, dtype=jnp.int32)
        safe_planned_receiver = jnp.clip(
            _policy_state.planned_receiver, 0, player_count - 1
        )
        planned_receiver_valid = (
            reliable_pass_flight_attack
            & (_policy_state.planned_receiver >= 0)
            & (_policy_state.planned_receiver < player_count)
            & outfield_chaser_candidate[observer_row, safe_planned_receiver]
        )
        predicted_chaser_index = jnp.where(
            planned_receiver_valid,
            safe_planned_receiver,
            predicted_chaser_index,
        )
        arrival_difference = jnp.linalg.norm(
            _policy_state.planned_arrival - predicted_target, axis=-1
        )
        arrival_tolerance = jnp.float32(
            env.reach.carry_radius_m + ball_radius
        ) + ground_motion_speed / jnp.float32(control_fps)
        retain_planned_arrival = (
            planned_receiver_valid
            & (_policy_state.planned_receiver == predicted_chaser_index)
            & (
                (_policy_state.planned_eta_ticks > 0)
                | (arrival_difference <= arrival_tolerance)
            )
        )
        # Retain a still-physical arrival/ETA instead of moving the receiver's
        # target every control frame. A forecast disagreement larger than one
        # contact radius plus one observed ball step refreshes the plan.
        predicted_target = jnp.where(
            retain_planned_arrival[:, None],
            _policy_state.planned_arrival,
            predicted_target,
        )
        loose_chaser_index = jnp.where(
            moving_ground_loose,
            predicted_chaser_index,
            current_chaser_index,
        )
        loose_target = jnp.where(
            moving_ground_loose[:, None],
            predicted_target,
            context.ball_position[:, :2],
        )
        loose_delta = loose_target - self_position
        own_ground_pass_flight = moving_ground_loose & reliable_pass_flight_attack
        fresh_eta_ticks = jnp.maximum(
            jnp.ceil(predicted_eta_s * jnp.float32(control_fps)).astype(jnp.int32),
            jnp.int32(1),
        )
        planned_eta_ticks = jnp.where(
            retain_planned_arrival,
            _policy_state.planned_eta_ticks,
            fresh_eta_ticks,
        )
        # The forecast target is the meeting point, not a moving waypoint.
        # Adding the ball's future velocity here made a receiver in front of
        # an incoming pass reverse before reaching that point: the incoming
        # velocity eventually outweighed the closing term.  Approach the
        # intercept itself with a braking-distance speed profile; CONTROL
        # owns the first-touch impulse once the ball enters physical reach.
        loose_desired_velocity = _intercept_approach_velocity(
            loose_delta,
            contact_radius_m=0.0,
            braking_mps2=env.player_physics.braking_deceleration_mps2,
        )
        loose_desired_speed = jnp.linalg.norm(loose_desired_velocity, axis=-1)
        loose_speed_limit = jnp.maximum(
            jnp.float32(config.approach_power) * self_max_speed,
            jnp.float32(GEOMETRY_EPS),
        )
        loose_command_speed = jnp.minimum(loose_desired_speed, loose_speed_limit)
        loose_approach_power = jnp.clip(
            loose_command_speed / jnp.maximum(self_max_speed, GEOMETRY_EPS),
            0.0,
            1.0,
        )
        loose_approach_move = _encode(
            loose_desired_velocity,
            loose_approach_power,
        )
        loose_chaser = (
            loose_ball
            & has_loose_chaser
            & (self_index == loose_chaser_index)
            & (~self_goalkeeper)
            & ground_ball
        )

        move = jnp.zeros((player_count, 2), dtype=jnp.float32)
        shape_context = possession_known & (~own_possessor)
        move = jnp.where(shape_context[:, None], shape_move, move)
        move = jnp.where(loose_chaser[:, None], loose_approach_move, move)
        move = jnp.where(
            (aerial.direct_runner | aerial.cover_runner)[:, None],
            aerial_move,
            move,
        )
        move = jnp.where(own_possessor[:, None], carrier_move, move)
        move = jnp.where(self_goalkeeper[:, None], goalkeeper_move, move)

        # At a corner, contact reach can become true while the taker is still
        # approaching from the field side. Arming there sends the ball back
        # through the taker's solid moving capsule and triggers a lawful
        # same-actor retouch. SoccerWorld waits for its executable kicker pose;
        # FootballWorld keeps the exact statutory corner point and instead
        # fails closed until the observed ball lies on the actual kick side of
        # the taker. This is a sign-only geometry gate, not a fitted margin.
        ball_relative_direction = ball_xy / jnp.maximum(
            jnp.linalg.norm(ball_xy, axis=-1)[:, None],
            jnp.float32(GEOMETRY_EPS),
        )
        corner_side = jnp.where(
            jnp.abs(context.ball_position[:, :2]) > jnp.float32(GEOMETRY_EPS),
            jnp.sign(context.ball_position[:, :2]),
            1.0,
        )
        signed_corner_direction = restart_direction * corner_side
        corner_field_inward = jnp.all(
            signed_corner_direction <= jnp.float32(GEOMETRY_EPS), axis=-1
        ) & jnp.any(signed_corner_direction < -jnp.float32(GEOMETRY_EPS), axis=-1)
        corner_release_aligned = (observations.restart.kind != RK_CORNER) | (
            (jnp.sum(ball_relative_direction * restart_direction, axis=-1) > 0.0)
            & corner_field_inward
        )
        restart_release = (
            restart_active
            & own_restart_taker
            & own_visible
            & ball_visible
            & contact_available
            & (planned_restart_valid | goalkeeper_hold_distribution)
            & corner_release_aligned
        )
        last_contact_intent = observations.possession.last_contact.intent
        last_contact_deliberate = (
            observations.possession.last_contact.known
            & (observations.possession.last_contact.outcome == OUTCOME_RELEASE)
            & (
                (last_contact_intent == INTENT_PASS)
                | (last_contact_intent == INTENT_SHOT)
                | (last_contact_intent == INTENT_CLEAR)
            )
        )
        opponent_aerial_service = last_contact_deliberate & jnp.any(
            observations.players.last_actor & opponent,
            axis=-1,
        )
        aerial_horizontal_reach = ball_distance <= jnp.where(
            opponent_aerial_service,
            env.reach.challenge_radius_m + ball_radius,
            env.reach.carry_radius_m + ball_radius,
        )
        aerial_height_reach = (
            ball_height <= roster.reach_height[self_index] + ball_radius
        )
        aerial_speed_reach = (
            ball_horizontal_speed
            + env.reach.height_speed_penalty_mps_per_m * ball_height
            <= env.reach.block_speed_limit_mps
        )
        aerial_contact = (
            (~restart_active)
            & aerial.direct_runner
            & aerial_horizontal_reach
            & aerial_height_reach
            & aerial_speed_reach
            & mechanism_contact_available
        )
        # This is a deliberately broad, policy-level deep-clear safety prior,
        # not a measured pressure/no-outlet rule.
        deep_clear_zone = (
            context.ball_position[:, 0]
            <= -config.defensive_clear_depth_fraction * half_length
        )
        aerial_control = aerial_contact & (~opponent_aerial_service)
        defensive_aerial_clear = (
            aerial_contact
            & opponent_aerial_service
            & deep_clear_zone
            & (ball_distance <= env.reach.carry_radius_m + ball_radius)
        )
        aerial_challenge = (
            aerial_contact & opponent_aerial_service & (~defensive_aerial_clear)
        )
        loose_control = (
            (~restart_active)
            & loose_chaser
            & foot_contact_reachable
            & contact_available
        ) | aerial_control
        previous_team_known = observations.possession.previous_team != NO_TEAM
        opponent_loose_challenge = (
            loose_control
            & previous_team_known
            & (observations.possession.previous_team != self_team)
        )
        defensive_loose_clear = opponent_loose_challenge & deep_clear_zone
        opponent_loose_challenge = opponent_loose_challenge & (~defensive_loose_clear)
        loose_control = loose_control & (~opponent_loose_challenge)
        loose_control = loose_control & (~defensive_loose_clear)

        # The last-contact fact is part of each public observation row and is
        # visibility-masked by the observation builder. A hidden goalkeeper
        # therefore fails closed to ordinary loose control; no engine state is
        # reconstructed here. Only the selected attacking loose-ball chaser
        # may convert a hand parry or passive save into a first-time shot.
        availability = intent_availability_hint(observations)
        last_contact = observations.possession.last_contact
        last_actor_opponent_goalkeeper = jnp.any(
            observations.players.last_actor & opponent & roster.is_goalkeeper[None, :],
            axis=-1,
        )
        goalkeeper_save = last_contact.known & (
            (
                (last_contact.mechanism == MECHANISM_GOALKEEPER_HAND)
                & (last_contact.outcome == OUTCOME_PARRY)
            )
            | (
                (last_contact.mechanism == MECHANISM_PASSIVE_BODY)
                & (last_contact.outcome == OUTCOME_DEFLECTION)
            )
        )
        goalkeeper_rebound = (
            loose_control & last_actor_opponent_goalkeeper & goalkeeper_save
        )
        # loose_chaser has one stable winner; selecting only that row avoids
        # vmapping the shot graph over every observer for a rare causal event.
        # Row zero is evaluated when the mask is empty but can never escape
        # the goalkeeper_rebound predicate below.
        rebound_row = jnp.argmax(goalkeeper_rebound.astype(jnp.int32))
        rebound_opponent = opponent[rebound_row]
        rebound_goalkeeper = rebound_opponent & roster.is_goalkeeper
        rebound_pressure = pressure(
            ball_position[rebound_row, :2],
            context.player_position[rebound_row],
            context.player_velocity[rebound_row],
            rebound_opponent,
            distance_scale_m=config.pressure_distance_m,
        )
        rebound_plan_key = jax.random.fold_in(frame_key, _REBOUND_SHOT_RANDOM_STREAM)
        rebound_plan = plan_shot(
            ball_position[rebound_row, :2],
            context.player_position[rebound_row],
            rebound_opponent,
            rebound_goalkeeper,
            config,
            half_length=half_length,
            goal_width=goal_width,
            current_pressure=rebound_pressure,
            decision_key=rebound_plan_key,
            shot_launch_radians_per_action_unit=(
                0.5
                * (
                    env.action_scale.launch_max_radians
                    + env.action_scale.ground_launch_down_max_radians
                )
            ),
        )
        rebound_values = jnp.stack((rebound_plan.value, 1.0 - rebound_plan.value))
        rebound_logits = (
            jnp.log(jnp.maximum(rebound_values, jnp.float32(1.0e-4)))
            / config.macro_choice_temperature
        )
        rebound_choose_shot = (
            jax.random.categorical(
                jax.random.fold_in(frame_key, _REBOUND_CHOICE_RANDOM_STREAM),
                rebound_logits,
            )
            == 0
        )
        # Bind availability before every categorical or continuous override.
        # If SHOT is unavailable, the complete baseline CONTROL action survives.
        rebound_shot = (
            goalkeeper_rebound & rebound_choose_shot & availability[:, INTENT_SHOT]
        )
        defensive_clear_contact = defensive_aerial_clear | defensive_loose_clear
        challenge_due = (observations.possession.control_ticks > 0) & (
            (observations.possession.control_ticks - 1)
            % challenge_attempt_interval_ticks
            == 0
        )
        challenge_key = jax.random.fold_in(frame_key, _CHALLENGE_RANDOM_STREAM)
        challenge_draw = jax.random.uniform(challenge_key, shape=(player_count,))
        challenge_proximity = jnp.clip(
            1.0 - carrier_distance / config.pressure_distance_m, 0.0, 1.0
        )
        safe_carrier_index = jnp.clip(carrier_index, 0, player_count - 1)
        carrier_velocity = _row_gather(context.player_velocity, row, safe_carrier_index)
        approach_direction = carrier_relative / jnp.maximum(
            carrier_distance[:, None], jnp.float32(GEOMETRY_EPS)
        )
        relative_velocity = observations.self_state.velocity - carrier_velocity
        challenge_closing_fraction = jnp.clip(
            jnp.maximum(jnp.sum(relative_velocity * approach_direction, axis=-1), 0.0)
            / jnp.maximum(
                self_max_speed + roster.max_speed[safe_carrier_index],
                jnp.float32(GEOMETRY_EPS),
            ),
            0.0,
            1.0,
        )
        carrier_facing = jnp.stack(
            (
                _row_gather(observations.players.facing_cos, row, safe_carrier_index),
                _row_gather(observations.players.facing_sin, row, safe_carrier_index),
            ),
            axis=-1,
        )
        challenge_behind_fraction = jnp.clip(
            jnp.sum(approach_direction * carrier_facing, axis=-1), 0.0, 1.0
        )
        expected_challenge_success = challenge_success_probability(
            challenge_closing_fraction,
            challenge_behind_fraction,
            jnp.full_like(challenge_proximity, jnp.float32(config.dribble_power)),
            roster.ball_control[self_index] - roster.ball_control[safe_carrier_index],
            config=env.contest,
        )
        success_opportunity = expected_challenge_success - jnp.float32(
            env.contest.tackle_success_probability
        )
        expected_challenge_foul = challenge_foul_probability(
            challenge_closing_fraction,
            challenge_behind_fraction,
            jnp.full_like(challenge_proximity, jnp.float32(config.dribble_power)),
            config=env.contest,
        )
        foul_excess = jnp.maximum(
            expected_challenge_foul - jnp.float32(env.contest.tackle_foul_probability),
            0.0,
        )
        challenge_probability = jnp.clip(
            (
                config.challenge_attempt_probability
                + config.challenge_proximity_gain * challenge_proximity
                + config.challenge_success_opportunity_gain * success_opportunity
                - config.challenge_foul_avoidance_gain * foul_excess
            )
            * jnp.where(counterpress_active, row_tactical.counterpress_gain, 1.0),
            0.0,
            1.0,
        )
        challenge_selected = challenge_draw < challenge_probability
        self_booked = (
            _row_gather(observations.players.yellow_cards, row, self_index) > 0
        )
        challenge_edge = env.reach.challenge_radius_m + ball_radius
        pelvis_contact_height = (
            roster.height[self_index] * env.action_scale.pelvis_height_factor
            + ball_radius
        )
        current_challenge_reachable = (
            (ball_distance <= challenge_edge)
            & (ball_height <= pelvis_contact_height)
            & (
                ball_horizontal_speed
                + env.reach.height_speed_penalty_mps_per_m * ball_height
                <= env.reach.block_speed_limit_mps
            )
        )
        swept_challenge_reachable = (
            ball_approaching
            & (jnp.linalg.norm(closest_relative_xy, axis=-1) <= challenge_edge)
            & (closest_ball_height <= pelvis_contact_height)
            & (
                ball_horizontal_speed
                + env.reach.height_speed_penalty_mps_per_m * closest_ball_height
                <= env.reach.block_speed_limit_mps
            )
        )
        controlled_challenge_contact = (
            (~restart_active)
            & opponent_possession
            & has_visible_carrier
            & nearest_defender
            & (~self_goalkeeper)
            & ball_visible
            & ball_live
            & mechanism_contact_available
            & (current_challenge_reachable | swept_challenge_reachable)
            & challenge_due
            & challenge_selected
        )
        challenge_contact = (
            controlled_challenge_contact | aerial_challenge | opponent_loose_challenge
        ) & (~self_booked)
        # The environment owns the ordinary restart taker's legal approach and
        # projects every player's minimum IFAB separation.  It does not own
        # tactical positioning.  SoccerWorld likewise keeps non-takers moving
        # in an attack/defence phase during a restart; FootballWorld inherits
        # that separation of responsibility through its existing fixed-shape
        # formation field rather than importing SoccerWorld's K-League-fitted
        # set-piece table as if it were DFL evidence.  A goalkeeper hold remains
        # live play and continues to use the ordinary formation branches.
        ordinary_restart = restart_active & (observations.restart.kind != RK_GK_HOLD)
        move = jnp.where(
            (ordinary_restart & own_restart_taker)[:, None],
            0.0,
            move,
        )
        restart_egress_move = _encode(
            -restart_direction,
            jnp.full(player_count, config.dribble_power, dtype=jnp.float32),
        )
        move = jnp.where(
            (ordinary_restart & restart_release)[:, None],
            restart_egress_move,
            move,
        )

        # Deep emergency clearances use the same three feasible, seeded lanes
        # for outfield and goalkeeper foot play. Invalid outward lanes near a
        # touchline fall back to the straight-ahead lane; this adds no subtype
        # or data-dependent graph.
        deep_clear_directions = jnp.asarray(
            (
                (1.0, 0.0),
                (1.0, config.deep_clear_lateral_ratio),
                (1.0, -config.deep_clear_lateral_ratio),
            ),
            dtype=jnp.float32,
        )
        deep_clear_directions = deep_clear_directions / jnp.linalg.norm(
            deep_clear_directions,
            axis=-1,
            keepdims=True,
        )
        deep_clear_key = jax.random.fold_in(
            frame_key,
            _DEEP_CLEAR_RANDOM_STREAM,
        )
        proposed_deep_clear_lane = jax.random.randint(
            deep_clear_key,
            (player_count,),
            0,
            deep_clear_directions.shape[0],
        ).astype(jnp.int32)
        proposed_deep_clear_direction = deep_clear_directions[proposed_deep_clear_lane]
        proposed_deep_clear_target_y = (
            ball_position[:, 1]
            + jnp.float32(config.deep_clear_distance_m)
            * proposed_deep_clear_direction[:, 1]
        )
        deep_clear_lane = jnp.where(
            jnp.abs(proposed_deep_clear_target_y)
            <= jnp.float32(half_width - config.deep_clear_touchline_margin_m),
            proposed_deep_clear_lane,
            jnp.int32(0),
        )
        deep_clear_direction = deep_clear_directions[deep_clear_lane]

        # The ordinary possession planner deliberately preserves its seeded
        # best receiver even when PASS loses the macro draw. On the rare loose
        # goalkeeper-foot row selected above, reuse that receiver, its exact
        # prospective Law 11 gate, lane/reception score, and selected additive
        # release controls. This adds no second P-by-opponent graph.
        goalkeeper_complete_view = observations.valid[decision_row] & jnp.all(
            (~context.participating[decision_row])
            | context.player_visible[decision_row]
        )
        goalkeeper_source_pressure = pressure(
            ball_position[decision_row, :2],
            opponent_pool.position[decision_row],
            opponent_pool.velocity[decision_row],
            opponent_pool.available[decision_row],
            distance_scale_m=config.pressure_distance_m,
        )
        # This product is an explicit FootballWorld design prior, not a fitted
        # completion probability. It closes lane_completion's deliberate
        # 0.5 m source blind spot without adding a new numerical coefficient.
        goalkeeper_short_safety = jnp.clip(
            pass_safety_row * (1.0 - goalkeeper_source_pressure), 0.0, 1.0
        )
        safe_goalkeeper_receiver = jnp.maximum(possession_decision.target, 0)
        goalkeeper_selected_receiver = (
            jnp.arange(player_count, dtype=jnp.int32) == safe_goalkeeper_receiver
        ) & (possession_decision.target >= 0)
        goalkeeper_receiver_eligible = (
            goalkeeper_complete_view
            & goalkeeper_selected_receiver
            & pass_candidate
            & (~possession_decision.cross)
        )
        goalkeeper_distribution = select_goalkeeper_distribution_target(
            restart_context,
            restart_observations,
            roster,
            pass_target[None, :, :],
            goalkeeper_short_safety[None, :],
            pass_target[None, :, :],
            jnp.full((1, player_count), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.bool_),
            receiver_eligible=goalkeeper_receiver_eligible[None, :],
        )
        goalkeeper_planning_row = (
            has_goalkeeper_foot & (~has_restart_taker) & (~has_possessor)
        )
        goalkeeper_foot_pass_available = (
            goalkeeper_planning_row
            & goalkeeper_distribution.valid[0]
            & selected_pass.reachable
        )
        goalkeeper_foot_pass_actor = (
            jnp.arange(player_count, dtype=jnp.int32) == decision_row
        ) & goalkeeper_foot_pass_available
        goalkeeper_foot_pass = (
            goalkeeper_foot_required
            & goalkeeper_foot_pass_actor
            & availability[:, INTENT_PASS]
        )
        goalkeeper_foot_clearance = (
            goalkeeper_foot_required
            & (~goalkeeper_foot_pass)
            & availability[:, INTENT_CLEAR]
        )
        goalkeeper_foot_request = goalkeeper_foot_pass | goalkeeper_foot_clearance
        goalkeeper_foot_direction = jnp.where(
            goalkeeper_foot_pass[:, None],
            selected_pass.direction,
            deep_clear_direction,
        )
        goalkeeper_foot_power = jnp.where(
            goalkeeper_foot_pass,
            selected_pass.power,
            jnp.float32(config.clear_power),
        )
        goalkeeper_foot_launch = jnp.where(
            goalkeeper_foot_pass,
            selected_pass.launch,
            jnp.float32(config.clear_launch),
        )
        goalkeeper_foot_direction = jnp.where(
            goalkeeper_foot_request[:, None], goalkeeper_foot_direction, 0.0
        )
        goalkeeper_foot_power = jnp.where(
            goalkeeper_foot_request, goalkeeper_foot_power, 0.0
        )
        goalkeeper_contact = select_goalkeeper_contact_intent(
            goalkeeper_contact_eligible,
            foot_contact_reachable,
            hand_contact_reachable,
            ball_in_own_penalty_area,
            handling_restricted,
            goalkeeper_foot_request,
            goalkeeper_foot_direction,
            goalkeeper_foot_power,
            scale=env.action_scale,
        )
        goalkeeper_foot_play = (
            goalkeeper_contact.foot_control | goalkeeper_contact.foot_clearance
        )
        goalkeeper_foot_pass = goalkeeper_foot_pass & goalkeeper_foot_play
        goalkeeper_foot_clearance = goalkeeper_foot_clearance & goalkeeper_foot_play
        self_touched_last = _row_gather(
            observations.players.last_actor, row, self_index
        )
        recontact_ready = dribble_recontact_ready(
            ball_xy,
            relative_ball_horizontal_velocity,
            ball_horizontal_speed,
            self_touched_last,
            observations.possession.last_contact.known,
            observations.possession.last_contact.intent,
            observations.possession.last_contact.outcome,
        )
        dribble_touch_due = (
            (observations.possession.control_ticks > 0)
            & (
                observations.possession.control_ticks % dribble_touch_interval_ticks
                == 0
            )
            & recontact_ready
            # A newly secured receiver first gets a decision opportunity to
            # follow the cushioned ball. Re-kicking it during this stabilization
            # window recreated the loose state the trap had just resolved.
            & (~secure_follow)
        )
        normal_contact = (
            own_possessor
            & possession_known
            & ball_visible
            & ball_live
            & mechanism_contact_available
            & (shoot | pass_ball | clear_ball | dribble_touch_due)
        )
        retouch_blocked = (
            observations.restart_release.active
            & observations.restart_release.untouched
            & own_release_taker
        )
        open_play_contact = (
            normal_contact
            | loose_control
            | challenge_contact
            | defensive_clear_contact
            | goalkeeper_contact.automatic_hand_claim
            | goalkeeper_contact.foot_control
            | goalkeeper_contact.foot_clearance
        ) & (~retouch_blocked)
        request_contact = restart_release | open_play_contact

        # A ground CONTROL is a settling touch, not a miniature forward kick.
        # Contact usually begins near the carry-radius boundary; propelling the
        # ball outward at the carried-dribble speed can therefore invalidate
        # possession on the very next physics substep and create an endless
        # loose-ball CONTROL loop.  Direct the first touch toward the player's
        # feet at the smaller native control scale.  Aerial cushioning retains
        # the momentum-aligned behavior above.
        ground_settle_direction = _infield_ground_settle_direction(
            ball_xy,
            self_position,
            half_width,
        )
        goal_line_control_risk = carrier_goal_line_risk
        touchline_control_risk = carrier_touchline_risk
        inward_touchline_direction = carrier_inward_touchline
        ground_settle_direction = jnp.where(
            touchline_control_risk[:, None],
            inward_touchline_direction,
            ground_settle_direction,
        )
        inward_goal_line_direction = carrier_inward_goal_line
        ground_settle_direction = jnp.where(
            goal_line_control_risk[:, None],
            inward_goal_line_direction,
            ground_settle_direction,
        )
        ground_loose_control = loose_control & foot_contact_reachable
        first_touch_direction = jnp.where(
            ground_loose_control[:, None],
            ground_settle_direction,
            reception_direction,
        )
        first_touch_power = jnp.where(
            ground_loose_control & goal_line_control_risk,
            jnp.float32(config.goal_line_control_power),
            jnp.where(
                ground_loose_control & touchline_control_risk,
                jnp.float32(config.touchline_control_power),
                jnp.where(
                    ground_loose_control,
                    jnp.float32(config.dribble_power),
                    jnp.float32(control_power),
                ),
            ),
        )
        control_force = _encode(first_touch_direction, first_touch_power)
        challenge_force = _encode(goal_direction, config.dribble_power)
        defensive_clear_force = _encode(
            deep_clear_direction,
            config.clear_power,
        )
        force_to_ball = jnp.zeros((player_count, 2), dtype=jnp.float32)
        force_to_ball = jnp.where(
            loose_control[:, None],
            control_force,
            force_to_ball,
        )
        force_to_ball = jnp.where(
            challenge_contact[:, None],
            challenge_force,
            force_to_ball,
        )
        force_to_ball = jnp.where(
            defensive_clear_contact[:, None],
            defensive_clear_force,
            force_to_ball,
        )
        force_to_ball = jnp.where(
            rebound_shot[:, None],
            _encode(rebound_plan.direction, rebound_plan.power)[None, :],
            force_to_ball,
        )
        force_to_ball = jnp.where(normal_contact[:, None], carrier_force, force_to_ball)
        force_to_ball = jnp.where(
            goalkeeper_foot_play[:, None],
            _encode(goalkeeper_foot_direction, goalkeeper_foot_power),
            force_to_ball,
        )
        force_to_ball = jnp.where(
            restart_release[:, None], restart_force, force_to_ball
        )
        launch = jnp.where(normal_contact, strike_launch, -1.0)
        launch = jnp.where(
            defensive_clear_contact,
            config.clear_launch,
            launch,
        )
        launch = jnp.where(rebound_shot, rebound_plan.launch, launch)
        launch = jnp.where(
            goalkeeper_foot_play,
            goalkeeper_foot_launch,
            launch,
        )
        launch = jnp.where(restart_release, restart_launch, launch).astype(jnp.float32)
        spin = jnp.where(normal_contact[:, None], carrier_decision_spin, 0.0).astype(
            jnp.float32
        )
        spin = jnp.where(
            rebound_shot[:, None], rebound_plan.spin[None, :], spin
        ).astype(jnp.float32)
        spin = jnp.where(goalkeeper_foot_play[:, None], 0.0, spin).astype(jnp.float32)
        restart_aerial = planned_restart_aerial | (
            goalkeeper_hold_distribution & distribution_use_long
        )
        restart_spin_side = -jnp.where(ball_position[:, 1] >= 0.0, 1.0, -1.0)
        restart_aerial_spin = jnp.stack(
            (
                restart_spin_side * config.cross_side_spin,
                jnp.full(player_count, config.cross_back_spin, dtype=jnp.float32),
            ),
            axis=-1,
        )
        spin = jnp.where(
            (restart_release & restart_aerial)[:, None], restart_aerial_spin, spin
        ).astype(jnp.float32)

        valid_actor = active & own_visible
        move = jnp.where(valid_actor[:, None], move, 0.0)
        force_to_ball = jnp.where(valid_actor[:, None], force_to_ball, 0.0)
        request_contact = valid_actor & request_contact
        launch = jnp.where(valid_actor, launch, -1.0)
        spin = jnp.where(valid_actor[:, None], spin, 0.0)
        intent = _select_action_intent(
            normal_contact,
            shoot,
            pass_ball,
            clear_ball,
            loose_control,
            challenge_contact,
            defensive_clear_contact,
            goalkeeper_contact.automatic_hand_claim,
            goalkeeper_foot_pass,
            goalkeeper_foot_clearance,
            restart_release,
            planned_restart_shot,
            request_contact,
            availability,
        )
        intent = jnp.where(rebound_shot, INTENT_SHOT, intent).astype(jnp.int32)

        # MOVE leaves the force controls physically inert, so the baseline
        # reuses their planar direction as its torso target. A defending team
        # faces the visible ball and accepts the locomotion model's backward
        # speed limit. In possession, an off-ball runner instead turns into
        # the run and uses gaze to keep the ball in view; otherwise the new
        # body contract would make every overlap and forward support run a
        # capped back-pedal. Contact intents retain their full impulse
        # direction, including back-heels. When the selected direction is
        # zero, locomotion remains the decoder's fallback body target.
        move_norm = jnp.linalg.norm(move, axis=-1)
        attacking_run = (
            own_team_possession
            & (~own_possessor)
            & ball_visible
            & (move_norm > GEOMETRY_EPS)
        )
        body_target_delta = jnp.where(
            attacking_run[:, None],
            move,
            ball_xy,
        )
        body_aim = _encode(
            body_target_delta,
            jnp.where(ball_visible, 1.0, 0.0).astype(jnp.float32),
        )
        force_to_ball = jnp.where(
            ((intent == INTENT_MOVE) & ball_visible & valid_actor)[:, None],
            body_aim,
            force_to_ball,
        )

        # Gaze is a torso-relative bounded target. Off-ball runners look back
        # to the ball while their torso follows the run. A carrier executing a
        # pass, shot, or clearance looks along that intent; while carrying, it
        # scans the already selected visible service option when one exists.
        # No hidden teammate row is consulted. atan2(cross, dot) measures the
        # shortest signed separation without exposing a 0/2pi seam.
        default_look_delta = jnp.where(
            ball_visible[:, None],
            ball_xy,
            jnp.where((move_norm > GEOMETRY_EPS)[:, None], move, self_body),
        )
        carrier_scan = (
            own_possessor
            & (carrier_kind == POSSESSION_DRIBBLE)
            & (possession_decision.target >= 0)
        )
        carrier_scan_delta = best_pass_target - self_position
        carrier_look_delta = jnp.where(
            carrier_scan[:, None], carrier_scan_delta, carrier_direction
        )
        look_delta = jnp.where(
            own_possessor[:, None], carrier_look_delta, default_look_delta
        )
        look_norm = jnp.linalg.norm(look_delta, axis=-1)
        look_direction = look_delta / jnp.maximum(
            look_norm[:, None], jnp.float32(GEOMETRY_EPS)
        )
        look_direction = jnp.where(
            (look_norm > GEOMETRY_EPS)[:, None], look_direction, self_body
        )
        gaze_cross = (
            self_body[:, 0] * look_direction[:, 1]
            - self_body[:, 1] * look_direction[:, 0]
        )
        gaze_dot = jnp.sum(self_body * look_direction, axis=-1)
        gaze_center = jnp.clip(
            jnp.arctan2(gaze_cross, gaze_dot) / gaze_limit_radians,
            -1.0,
            1.0,
        )
        gaze_center = jnp.where(valid_actor, gaze_center, 0.0).astype(jnp.float32)

        action = _finalize_policy_action(
            intent, move, force_to_ball, launch, spin, gaze_center
        )
        intended_receiver = jnp.where(
            pass_ball,
            possession_decision.target,
            jnp.int32(NO_PLAYER),
        )
        intended_receiver = jnp.where(
            goalkeeper_foot_pass,
            goalkeeper_distribution.receiver[0],
            intended_receiver,
        )
        restart_receiver = jnp.where(
            goalkeeper_hold_distribution,
            goalkeeper_hold_receiver,
            restart_decision.receiver,
        )
        intended_receiver = jnp.where(
            restart_release & (intent == INTENT_PASS),
            restart_receiver,
            intended_receiver,
        )
        intended_receiver = jnp.where(
            (intent == INTENT_PASS)
            & (intended_receiver >= 0)
            & (intended_receiver < player_count),
            intended_receiver,
            jnp.int32(NO_PLAYER),
        )
        safe_intended_receiver = jnp.maximum(intended_receiver, 0)
        intended_receiver_ids = jnp.where(
            intended_receiver >= 0,
            roster.player_id[safe_intended_receiver],
            jnp.int32(NO_PLAYER),
        ).astype(jnp.int32)
        action_decision = _ActionDecision(
            action=action,
            loose_chaser=loose_chaser_index,
            loose_chase_active=loose_ball & has_loose_chaser & ground_ball,
            planned_receiver=predicted_chaser_index,
            planned_arrival=predicted_target,
            planned_eta_ticks=planned_eta_ticks,
            pass_plan_active=own_ground_pass_flight,
            service_opportunity=service_opportunity_by_row,
            intended_receiver_ids=intended_receiver_ids,
        )
        if with_pass_diagnostic:
            pass_candidates = pass_candidates._replace(
                carrier_slot=decision_row.astype(jnp.int32)
            )
            return action_decision, pass_candidates
        return action_decision

    def lean_action_for_state(observations, roster, state, base_key):
        return action_for_state(observations, roster, state, base_key)

    def traced_action_for_state(observations, roster, state, base_key):
        return action_for_state(
            observations,
            roster,
            state,
            base_key,
            with_pass_diagnostic=True,
        )

    return RuleBasedPolicy(
        config=config,
        team_tactical_plan=team_tactical_plan,
        counterpress_window_ticks=counterpress_window_ticks,
        secure_control_window_ticks=dribble_touch_interval_ticks,
        _action_for_state=lean_action_for_state,
        _action_for_state_with_pass_diagnostic=traced_action_for_state,
    )


__all__ = [
    "PolicyEventStep",
    "PolicyPassDiagnosticStep",
    "PolicyStep",
    "RuleBasedPolicy",
    "make_rule_based_policy",
]
