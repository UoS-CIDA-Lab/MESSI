"""Bounded goalkeeper and aerial decisions over public observations.

This module is deliberately a set of *decision helpers*, not a second policy.
The rule-policy integrator is expected to build a
:class:`~footballworld.policies.rule_based.context.RulePolicyContext`, derive
candidate pass targets with ``tactics.moving_receiver_target``, and then use:

``aerial_arrival_score``
    Score fixed-shape candidate arrival and opponent contest geometry.
``aerial_contest_decision``
    Give one visible outfielder the direct aerial run, or a late defender a
    goal-side cover run, without combining different observers' rows.
``goalkeeper_cover_decision``
    Return the goalkeeper's bounded angle-cover, earliest reachable
    interception, or conservative sweeper-claim target.
``select_goalkeeper_distribution_target``
    Select among caller-supplied short and long targets.  No-candidate is an
    explicit result so the integrating policy retains ownership of fallback
    clearances and action encoding.

Coefficient provenance
----------------------
The numeric defaults are transfer priors recorded in the coefficient audit.
The 0.20 second duel window, 3 metre cover offset, 6--28 metre short envelope,
and long gates (22 metre distance, 6 metre progress, 0.36 arrival) form one
coupled decision model. They are not claimed as independently fitted
probabilities. In particular, arrival ``score`` is a bounded ranking currency.
Keeping the values together makes a later held-out calibration an auditable
replacement.

All array work is pure JAX and shape-static.  Helpers fail closed for invisible
balls/players and consume no rollout ``State`` or environment affordance.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER, RK_NONE
from footballworld.core.contact import (
    INTENT_CLEAR,
    INTENT_PASS,
    INTENT_SHOT,
    OUTCOME_RELEASE,
)
from footballworld.environment.observation import (
    Observation,
    RosterMetadata,
)
from footballworld.policies.rule_based.context import RulePolicyContext

# See the module docstring before changing one value in isolation: several
# values form coupled decision boundaries.
_ARRIVAL_SLACK_S = 0.35
_ARRIVAL_SOFTNESS_S = 0.45
_CONTEST_SOFTNESS_S = 0.65
_OPPONENT_LEAD_FRACTION = 0.35
_OPPONENT_LEAD_CAP_S = 0.60
_AERIAL_DUEL_WINDOW_S = 0.20
_AERIAL_COVER_DISTANCE_M = 3.0

_SHORT_MIN_DISTANCE_M = 6.0
_SHORT_MAX_DISTANCE_M = 28.0
_SHORT_COMPLETION_FLOOR = 0.58
_LONG_MIN_DISTANCE_M = 22.0
_LONG_MIN_PROGRESS_M = 6.0
_LONG_ARRIVAL_FLOOR = 0.36

_GK_CROSSING_HORIZON_S = 3.0
_GK_MIN_BALL_X_SPEED_MPS = 0.30
_GK_HEADING_SPEED_MPS = 0.50
_GK_MIN_OFFSET_M = 0.60
_GK_MAX_OFFSET_M = 4.0
_GK_POSITION_MAX_OFFSET_M = 4.5
_GK_RUSH_DISTANCE_M = 7.0
_GK_HEADING_RUSH_DISTANCE_M = 12.0
_GK_CATCH_COMPATIBILITY_SPEED_MPS = 21.3 * 0.90
_GK_INTERCEPT_START_S = 0.08
_GK_INTERCEPT_HORIZON_S = 2.20
_GK_INTERCEPT_SAMPLES = 12
_GK_PREDICTION_DECELERATION_MPS2 = 5.0
_GK_INTERCEPT_REACH_MARGIN_M = 0.50
_GK_SWEEP_OPPONENT_MARGIN_M = 1.50
_GK_SWEEP_BALL_HEIGHT_M = 1.70
_GK_SWEEP_FIELD_FRACTION = 0.30
_GK_SWEEP_GOAL_DISTANCE_M = 24.0
_GK_CLAIM_BOX_MARGIN_M = 1.0

_EPS = jnp.float32(1.0e-6)


class AerialArrivalScore(NamedTuple):
    """Per-candidate arrival and contest ranking arrays.

    Every field has shape ``(observers, candidates)``.  ``score`` lies in
    ``[0, 1]`` but is not a calibrated completion probability.
    """

    score: jax.Array
    receiver_eta_s: jax.Array
    nearest_opponent_eta_s: jax.Array


class AerialContestDecision(NamedTuple):
    """One actor's direct-duel or second-ball-cover assignment."""

    target: jax.Array
    direct_runner: jax.Array
    cover_runner: jax.Array
    team_best_slot: jax.Array
    team_best_eta_s: jax.Array
    opponent_best_eta_s: jax.Array
    arrival_score: jax.Array


class GoalkeeperCoverDecision(NamedTuple):
    """Angle-cover target and bounded interception classifications."""

    target: jax.Array
    angle_target: jax.Array
    intercept_target: jax.Array
    sweep_target: jax.Array
    heading_toward_goal: jax.Array
    predicted_goal_y: jax.Array
    reachable_intercept: jax.Array
    sweeper_claim: jax.Array
    urgent_rush: jax.Array
    active: jax.Array


class GoalkeeperDistributionDecision(NamedTuple):
    """Selected receiver target without inventing a no-option clearance."""

    target: jax.Array
    receiver: jax.Array
    valid: jax.Array
    use_long: jax.Array
    has_short: jax.Array
    has_long: jax.Array


def _safe_norm(value: jax.Array, *, axis: int = -1) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(value * value, axis=axis) + _EPS * _EPS)


def _safe_unit(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value, dtype=jnp.float32)
    squared = jnp.sum(value * value, axis=-1, keepdims=True)
    norm = jnp.sqrt(squared + jnp.where(squared > 0.0, 0.0, 1.0))
    return jnp.where(squared > 0.0, value / norm, 0.0).astype(jnp.float32)


def _sigmoid(value: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(jnp.clip(value, -30.0, 30.0))


def _validate_candidate_arrays(
    target_xy: jax.Array,
    receiver_position_xy: jax.Array,
    receiver_max_speed_mps: jax.Array,
    flight_time_s: jax.Array,
    player_position_xy: jax.Array,
    player_velocity_xy: jax.Array,
    player_max_speed_mps: jax.Array,
    opponent_mask: jax.Array,
) -> tuple[int, int]:
    if target_xy.ndim != 3 or target_xy.shape[-1] != 2:
        raise ValueError("target_xy must have shape (observers, candidates, 2)")
    observers, candidates = target_xy.shape[:2]
    candidate_shape = (observers, candidates)
    player_shape = player_position_xy.shape[:2]
    if receiver_position_xy.shape != target_xy.shape:
        raise ValueError("receiver_position_xy must match target_xy")
    if receiver_max_speed_mps.shape != candidate_shape:
        raise ValueError(
            "receiver_max_speed_mps must have shape (observers, candidates)"
        )
    if flight_time_s.shape != candidate_shape:
        raise ValueError("flight_time_s must have shape (observers, candidates)")
    if player_position_xy.ndim != 3 or player_position_xy.shape[-1] != 2:
        raise ValueError("player_position_xy must have shape (observers, players, 2)")
    if player_velocity_xy.shape != player_position_xy.shape:
        raise ValueError("player_velocity_xy must match player_position_xy")
    if player_shape[0] != observers:
        raise ValueError("player_position_xy must share the observer axis")
    if player_max_speed_mps.shape != player_shape:
        raise ValueError("player_max_speed_mps must have shape (observers, players)")
    if opponent_mask.shape != player_shape:
        raise ValueError("opponent_mask must have shape (observers, players)")
    return observers, candidates


def aerial_arrival_score(
    target_xy: jax.Array,
    receiver_position_xy: jax.Array,
    receiver_max_speed_mps: jax.Array,
    flight_time_s: jax.Array,
    player_position_xy: jax.Array,
    player_velocity_xy: jax.Array,
    player_max_speed_mps: jax.Array,
    opponent_mask: jax.Array,
) -> AerialArrivalScore:
    """Score fixed-shape receiver arrival and the nearest visible contest.

    Inputs use shapes ``(O, C, 2)`` for receiver targets/positions,
    ``(O, C)`` for receiver speed and flight time, and ``(O, P, ...)`` for
    the visible player pool.  An integrating policy normally sets ``C=P``.
    The caller owns ball trajectory prediction; this helper intentionally does
    not duplicate FootballWorld's spin-dependent flight physics.
    """

    target_xy = jnp.asarray(target_xy, dtype=jnp.float32)
    receiver_position_xy = jnp.asarray(receiver_position_xy, dtype=jnp.float32)
    receiver_max_speed_mps = jnp.asarray(receiver_max_speed_mps, dtype=jnp.float32)
    flight_time_s = jnp.asarray(flight_time_s, dtype=jnp.float32)
    player_position_xy = jnp.asarray(player_position_xy, dtype=jnp.float32)
    player_velocity_xy = jnp.asarray(player_velocity_xy, dtype=jnp.float32)
    player_max_speed_mps = jnp.asarray(player_max_speed_mps, dtype=jnp.float32)
    opponent_mask = jnp.asarray(opponent_mask, dtype=jnp.bool_)
    _validate_candidate_arrays(
        target_xy,
        receiver_position_xy,
        receiver_max_speed_mps,
        flight_time_s,
        player_position_xy,
        player_velocity_xy,
        player_max_speed_mps,
        opponent_mask,
    )

    safe_receiver_speed = jnp.maximum(receiver_max_speed_mps, _EPS)
    receiver_eta = _safe_norm(target_xy - receiver_position_xy) / safe_receiver_speed
    bounded_flight = jnp.maximum(flight_time_s, 0.0)
    opponent_lead = jnp.clip(
        _OPPONENT_LEAD_FRACTION * bounded_flight,
        0.0,
        _OPPONENT_LEAD_CAP_S,
    )
    opponent_future = (
        player_position_xy[:, None, :, :]
        + player_velocity_xy[:, None, :, :] * opponent_lead[:, :, None, None]
    )
    opponent_distance = _safe_norm(
        target_xy[:, :, None, :] - opponent_future,
        axis=-1,
    )
    opponent_eta = opponent_distance / jnp.maximum(
        player_max_speed_mps[:, None, :], _EPS
    )
    nearest_opponent_eta = jnp.min(
        jnp.where(opponent_mask[:, None, :], opponent_eta, jnp.inf),
        axis=-1,
    )
    arrival = _sigmoid(
        (bounded_flight + _ARRIVAL_SLACK_S - receiver_eta) / _ARRIVAL_SOFTNESS_S
    )
    contest = _sigmoid((nearest_opponent_eta - receiver_eta) / _CONTEST_SOFTNESS_S)
    score = arrival * (0.35 + 0.65 * contest)
    score = jnp.where(jnp.isfinite(score), score, 0.0)
    return AerialArrivalScore(
        score=jnp.clip(score, 0.0, 1.0).astype(jnp.float32),
        receiver_eta_s=receiver_eta.astype(jnp.float32),
        nearest_opponent_eta_s=nearest_opponent_eta.astype(jnp.float32),
    )


def _policy_axes(
    context: RulePolicyContext,
    observations: Observation,
    roster: RosterMetadata,
) -> tuple[int, jax.Array, jax.Array]:
    observers = context.self_index.shape[0]
    if context.player_position.shape != (observers, observers, 2):
        raise ValueError("context must have matching observer and roster axes")
    if observations.players.offside.shape != (observers, observers):
        raise ValueError("observations must have matching observer and roster axes")
    if roster.team_id.shape != (observers,):
        raise ValueError("roster must match the context roster axis")
    if roster.is_goalkeeper.shape != (observers,):
        raise ValueError("roster goalkeeper flags must match the roster axis")
    if roster.max_speed.shape != (observers,):
        raise ValueError("roster maximum speed must match the roster axis")
    row = jnp.arange(observers, dtype=jnp.int32)
    slot = jnp.arange(observers, dtype=jnp.int32)
    return observers, row, slot


def aerial_contest_decision(
    context: RulePolicyContext,
    observations: Observation,
    roster: RosterMetadata,
    contact_target_xy: jax.Array,
    flight_time_s: jax.Array,
    *,
    half_length_m: float | jax.Array,
    excluded_player: jax.Array | None = None,
    long_stamina_vmax_floor: float | jax.Array = 0.99,
    short_stamina_vmax_floor: float | jax.Array = 0.70,
    short_stamina_headroom_knee: float | jax.Array = 0.25,
) -> AerialContestDecision:
    """Assign a direct aerial runner or goal-side cover runner per observer.

    ``contact_target_xy`` and ``flight_time_s`` are one predicted contact point
    and remaining flight time per observer row.  They must be derived from that
    same row's public ball observation.  A known deliberate last kick supplies
    service ownership; otherwise both teams retain a symmetric contest.
    Goalkeepers are excluded because their cover/claim branch is separate.

    The three stamina keywords are public static environment coefficients.  Their
    defaults equal FootballWorld's default stamina configuration; a policy built
    for an overridden environment should close over that environment's values.
    """

    observers, row, _ = _policy_axes(context, observations, roster)
    contact_target_xy = jnp.asarray(contact_target_xy, dtype=jnp.float32)
    flight_time_s = jnp.asarray(flight_time_s, dtype=jnp.float32)
    if contact_target_xy.shape != (observers, 2):
        raise ValueError("contact_target_xy must have shape (observers, 2)")
    if flight_time_s.shape != (observers,):
        raise ValueError("flight_time_s must have shape (observers,)")

    participating = context.participating & context.player_visible
    excluded = (
        jnp.zeros_like(participating)
        if excluded_player is None
        else jnp.asarray(excluded_player, dtype=jnp.bool_)
    )
    if excluded.shape != participating.shape:
        raise ValueError("excluded_player must match observer and roster axes")
    outfielder = participating & (~roster.is_goalkeeper[None, :]) & (~excluded)
    own = outfielder & context.same_team
    opponent = outfielder & (~context.same_team)
    long_floor = jnp.asarray(long_stamina_vmax_floor, dtype=jnp.float32)
    short_floor = jnp.asarray(short_stamina_vmax_floor, dtype=jnp.float32)
    short_knee = jnp.maximum(
        jnp.asarray(short_stamina_headroom_knee, dtype=jnp.float32), _EPS
    )
    sustained_fraction = long_floor + (1.0 - long_floor) * jnp.clip(
        observations.players.stamina_long, 0.0, 1.0
    )
    short_x = jnp.clip(observations.players.stamina_short / short_knee, 0.0, 1.0)
    short_headroom = short_x * short_x * (3.0 - 2.0 * short_x)
    speed = (
        roster.max_speed[None, :]
        * sustained_fraction
        * (short_floor + (1.0 - short_floor) * short_headroom)
    )
    # Every candidate in this decision runs to the same forecast contact point.
    # Calling ``aerial_arrival_score`` with that point repeated on a candidate
    # axis would materialize an unnecessary observer x player x player contest
    # tensor. Keep the identical ranking semantics with one O(O*P) distance
    # table instead.
    bounded_flight = jnp.maximum(flight_time_s, 0.0)
    target = contact_target_xy[:, None, :]
    eta = _safe_norm(target - context.player_position) / jnp.maximum(speed, _EPS)
    opponent_lead = jnp.clip(
        _OPPONENT_LEAD_FRACTION * bounded_flight,
        0.0,
        _OPPONENT_LEAD_CAP_S,
    )
    opponent_future = (
        context.player_position + context.player_velocity * opponent_lead[:, None, None]
    )
    opponent_eta_at_contact = _safe_norm(target - opponent_future) / jnp.maximum(
        speed, _EPS
    )
    nearest_opponent_eta = jnp.min(
        jnp.where(opponent, opponent_eta_at_contact, jnp.inf),
        axis=-1,
    )
    own_cost = jnp.where(own, eta, jnp.inf)
    opponent_cost = jnp.where(opponent, eta, jnp.inf)
    own_best_slot = jnp.argmin(own_cost, axis=-1).astype(jnp.int32)
    own_best_eta = jnp.min(own_cost, axis=-1)
    opponent_best_eta = jnp.min(opponent_cost, axis=-1)
    self_is_best = context.self_index == own_best_slot

    last_contact = observations.possession.last_contact
    deliberate = (
        last_contact.known
        & (last_contact.outcome == OUTCOME_RELEASE)
        & (
            (last_contact.intent == INTENT_PASS)
            | (last_contact.intent == INTENT_SHOT)
            | (last_contact.intent == INTENT_CLEAR)
        )
    )
    own_last_actor = jnp.any(
        observations.players.last_actor & context.same_team,
        axis=-1,
    )
    opponent_last_actor = jnp.any(
        observations.players.last_actor & (~context.same_team),
        axis=-1,
    )
    own_service = deliberate & own_last_actor
    opponent_service = deliberate & opponent_last_actor
    unknown_service = (~own_service) & (~opponent_service)

    active_aerial = (
        context.self_active
        & context.ball_visible
        & observations.ball.live
        & (observations.restart.kind == RK_NONE)
        & (context.ball_position[:, 2] > jnp.float32(0.19))
        & jnp.isfinite(own_best_eta)
    )
    defender_can_contest = jnp.isfinite(opponent_best_eta) & (
        own_best_eta <= opponent_best_eta + _AERIAL_DUEL_WINDOW_S
    )
    direct = (
        active_aerial
        & self_is_best
        & (own_service | unknown_service | (opponent_service & defender_can_contest))
    )
    cover = active_aerial & self_is_best & opponent_service & (~defender_can_contest)

    own_goal = jnp.stack(
        (
            -jnp.broadcast_to(
                jnp.asarray(half_length_m, dtype=jnp.float32), (observers,)
            ),
            jnp.zeros(observers, dtype=jnp.float32),
        ),
        axis=-1,
    )
    cover_target = contact_target_xy + _safe_unit(own_goal - contact_target_xy) * (
        _AERIAL_COVER_DISTANCE_M
    )
    movement_target = jnp.where(
        direct[:, None],
        contact_target_xy,
        jnp.where(cover[:, None], cover_target, context.self_position),
    )
    self_eta = eta[row, context.self_index]
    self_arrival = _sigmoid(
        (bounded_flight + _ARRIVAL_SLACK_S - self_eta) / _ARRIVAL_SOFTNESS_S
    )
    self_contest = _sigmoid((nearest_opponent_eta - self_eta) / _CONTEST_SOFTNESS_S)
    self_score = jnp.clip(
        self_arrival * (0.35 + 0.65 * self_contest),
        0.0,
        1.0,
    )
    return AerialContestDecision(
        target=movement_target.astype(jnp.float32),
        direct_runner=direct,
        cover_runner=cover,
        team_best_slot=jnp.where(
            jnp.isfinite(own_best_eta), own_best_slot, jnp.int32(NO_PLAYER)
        ),
        team_best_eta_s=own_best_eta.astype(jnp.float32),
        opponent_best_eta_s=opponent_best_eta.astype(jnp.float32),
        arrival_score=self_score.astype(jnp.float32),
    )


def goalkeeper_cover_decision(
    context: RulePolicyContext,
    observations: Observation,
    *,
    half_length_m: float | jax.Array,
    goal_width_m: float | jax.Array,
    penalty_area_length_m: float | jax.Array,
    penalty_area_width_m: float | jax.Array,
    ball_radius_m: float | jax.Array,
    gravity_mps2: float | jax.Array,
    air_drag_rate_per_s: jax.Array,
    goalkeeper_speed_mps: jax.Array,
    goalkeeper_reach_height_m: jax.Array,
    own_team_controlled: jax.Array,
) -> GoalkeeperCoverDecision:
    """Return goalkeeper cover from public ball kinematics.

    The twelve-point, 2.2 second interception horizon and its supported-ball
    five m/s^2 deceleration value are transfer priors. An airborne row uses the
    observed-state drag rate and ballistic height, and a candidate is reachable
    only below the goalkeeper's public reach height and before the scoring
    plane. A goalkeeper sweeps only when the first reachable point is low,
    catch-speed-compatible, and the keeper beats a *visible* opponent by at
    least 1.5 metres. Requiring a visible opponent keeps view-limited decisions
    fail-closed. Very close threats use the same earliest reachable point
    instead of chasing the ball's current position.

    ``own_team_controlled`` includes observed possession and a caller-proven
    same-actor CONTROL/TRAP continuation, but not an own PASS in flight. A
    goalkeeper therefore leaves a teammate settling touch alone while still
    acting as the receiver of a genuine back-pass. A controlled ball that is
    projected into the goal mouth remains an urgent threat.

    The helper does not decide hand legality or catch/parry; those remain
    environment rules and stochastic contest outcomes. ``goalkeeper_speed_mps``
    must be the public roster speed after the observed stamina limits. All
    forecast arrays have the fixed shape ``(observers, 12, 2)``.
    """

    observers = context.self_index.shape[0]
    if observations.restart.kind.shape != (observers,):
        raise ValueError("restart observations must have an observer axis")
    goalkeeper_speed = jnp.asarray(goalkeeper_speed_mps, dtype=jnp.float32)
    goalkeeper_reach_height = jnp.asarray(goalkeeper_reach_height_m, dtype=jnp.float32)
    air_drag_rate = jnp.asarray(air_drag_rate_per_s, dtype=jnp.float32)
    own_team_controlled = jnp.asarray(own_team_controlled, dtype=jnp.bool_)
    if goalkeeper_speed.shape != (observers,):
        raise ValueError("goalkeeper_speed_mps must have shape (observers,)")
    if goalkeeper_reach_height.shape != (observers,):
        raise ValueError("goalkeeper_reach_height_m must have shape (observers,)")
    if air_drag_rate.shape != (observers,):
        raise ValueError("air_drag_rate_per_s must have shape (observers,)")
    if own_team_controlled.shape != (observers,):
        raise ValueError("own_team_controlled must have shape (observers,)")
    half_length = jnp.asarray(half_length_m, dtype=jnp.float32)
    half_goal_width = jnp.asarray(goal_width_m, dtype=jnp.float32) * 0.5
    penalty_length = jnp.asarray(penalty_area_length_m, dtype=jnp.float32)
    half_penalty_width = jnp.asarray(penalty_area_width_m, dtype=jnp.float32) * 0.5
    ball_radius = jnp.asarray(ball_radius_m, dtype=jnp.float32)
    gravity = jnp.asarray(gravity_mps2, dtype=jnp.float32)
    own_goal = jnp.stack(
        (
            -jnp.broadcast_to(half_length, (observers,)),
            jnp.zeros(observers, dtype=jnp.float32),
        ),
        axis=-1,
    )
    ball = context.ball_position[:, :2]
    velocity = context.ball_velocity[:, :2]
    delta = ball - own_goal
    distance = _safe_norm(delta)
    toward_speed = -velocity[:, 0]
    heading = (toward_speed > _GK_HEADING_SPEED_MPS) & (distance < 32.0)

    vx = velocity[:, 0]
    safe_vx = jnp.where(
        jnp.abs(vx) < _GK_MIN_BALL_X_SPEED_MPS,
        -jnp.float32(_GK_MIN_BALL_X_SPEED_MPS),
        vx,
    )
    crossing_time = jnp.clip(
        (own_goal[:, 0] - ball[:, 0]) / safe_vx,
        0.0,
        _GK_CROSSING_HORIZON_S,
    )
    raw_crossing_y = ball[:, 1] + velocity[:, 1] * crossing_time
    # A ball heading toward the goal line but projected outside the mouth does
    # not justify abandoning angle cover for an urgent rush.
    heading = heading & (jnp.abs(raw_crossing_y) <= half_goal_width)
    predicted_y = jnp.clip(
        raw_crossing_y,
        -half_goal_width,
        half_goal_width,
    )
    offset = jnp.clip(
        _GK_MIN_OFFSET_M + distance * 0.045,
        _GK_MIN_OFFSET_M,
        _GK_MAX_OFFSET_M,
    )
    angle_position = own_goal + _safe_unit(delta) * offset[:, None]
    target_y = jnp.where(
        heading,
        predicted_y,
        jnp.clip(angle_position[:, 1], -half_goal_width, half_goal_width),
    )
    angle_target = jnp.stack(
        (
            jnp.clip(
                angle_position[:, 0],
                own_goal[:, 0] + 0.3,
                own_goal[:, 0] + _GK_POSITION_MAX_OFFSET_M,
            ),
            target_y,
        ),
        axis=-1,
    )

    # Approximate the horizontal path with bounded mean deceleration, then
    # select the earliest sample the
    # observed goalkeeper can reach. ``argmax`` is used only after preserving
    # the explicit any-reachable mask, so an all-false row cannot silently turn
    # the first sample into a rush target.
    intercept_times = jnp.linspace(
        _GK_INTERCEPT_START_S,
        _GK_INTERCEPT_HORIZON_S,
        _GK_INTERCEPT_SAMPLES,
        dtype=jnp.float32,
    )
    ball_speed = _safe_norm(velocity)
    ball_direction = _safe_unit(velocity)
    stop_distance = (
        0.5 * ball_speed * ball_speed / jnp.float32(_GK_PREDICTION_DECELERATION_MPS2)
    )
    travel_distance = jnp.minimum(
        jnp.maximum(
            ball_speed[:, None] * intercept_times[None, :]
            - 0.5
            * jnp.float32(_GK_PREDICTION_DECELERATION_MPS2)
            * intercept_times[None, :]
            * intercept_times[None, :],
            0.0,
        ),
        stop_distance[:, None],
    )
    ground_future_ball = (
        ball[:, None, :] + ball_direction[:, None, :] * travel_distance[:, :, None]
    )

    height_above_ground = jnp.maximum(context.ball_position[:, 2] - ball_radius, 0.0)
    vertical_speed = context.ball_velocity[:, 2]
    landing_time = (
        vertical_speed
        + jnp.sqrt(
            jnp.maximum(
                vertical_speed * vertical_speed + 2.0 * gravity * height_above_ground,
                0.0,
            )
        )
    ) / jnp.maximum(gravity, _EPS)
    air_time = jnp.minimum(intercept_times[None, :], landing_time[:, None])
    safe_drag_rate = jnp.maximum(air_drag_rate, _EPS)
    air_travel_scale = jnp.where(
        air_drag_rate[:, None] > _EPS,
        jnp.log1p(safe_drag_rate[:, None] * air_time) / safe_drag_rate[:, None],
        air_time,
    )
    airborne_future_ball = (
        ball[:, None, :] + velocity[:, None, :] * air_travel_scale[:, :, None]
    )
    airborne = (context.ball_position[:, 2] > ball_radius + jnp.float32(1.0e-4)) | (
        vertical_speed > 0.0
    )
    future_ball = jnp.where(
        airborne[:, None, None], airborne_future_ball, ground_future_ball
    )
    future_height = jnp.maximum(
        ball_radius,
        context.ball_position[:, 2:3]
        + vertical_speed[:, None] * intercept_times[None, :]
        - 0.5 * gravity * intercept_times[None, :] * intercept_times[None, :],
    )
    goalkeeper_distance = _safe_norm(
        future_ball - context.self_position[:, None, :], axis=-1
    )
    height_reachable = future_height <= (
        jnp.maximum(goalkeeper_reach_height, 0.0)[:, None] + ball_radius
    )
    before_scoring_plane = future_ball[:, :, 0] >= (own_goal[:, 0:1] - ball_radius)
    reachable = (
        (
            goalkeeper_distance
            <= (
                jnp.maximum(goalkeeper_speed, 0.0)[:, None] * intercept_times[None, :]
                + jnp.float32(_GK_INTERCEPT_REACH_MARGIN_M)
            )
        )
        & height_reachable
        & before_scoring_plane
    )
    has_reachable_intercept = jnp.any(reachable, axis=-1)
    first_reachable = jnp.argmax(reachable, axis=-1).astype(jnp.int32)
    intercept = future_ball[jnp.arange(observers, dtype=jnp.int32), first_reachable]

    intercept_distance = _safe_norm(
        context.player_position - intercept[:, None, :], axis=-1
    )
    has_visible_opponent = jnp.any(context.opponent, axis=-1)
    nearest_opponent_distance = jnp.min(
        jnp.where(context.opponent, intercept_distance, jnp.inf), axis=-1
    )
    goalkeeper_intercept_distance = _safe_norm(intercept - context.self_position)
    keeper_wins = has_visible_opponent & (
        goalkeeper_intercept_distance
        <= nearest_opponent_distance - jnp.float32(_GK_SWEEP_OPPONENT_MARGIN_M)
    )
    # A visible lawful teammate who is already closer owns an ordinary loose
    # recovery. Exclude the last actor just as the outfield receiver planner
    # does, so a true back-pass can still nominate the goalkeeper rather than
    # being blocked by the passer who may not immediately retouch it.
    teammate_candidate = context.teammate & (~observations.players.last_actor)
    has_visible_teammate = jnp.any(teammate_candidate, axis=-1)
    nearest_teammate_distance = jnp.min(
        jnp.where(teammate_candidate, intercept_distance, jnp.inf), axis=-1
    )
    goalkeeper_is_primary = (~has_visible_teammate) | (
        goalkeeper_intercept_distance < nearest_teammate_distance
    )

    claimable_compatibility = ball_speed < _GK_CATCH_COMPATIBILITY_SPEED_MPS
    # Keep a goalkeeper/outfielder role split, but reject the broad
    # toward-goal-or-not-physically-possessed rule. That rule makes the goalkeeper and a teammate converge on a
    # settling CONTROL touch. Public causal lineage rejects that false threat
    # without hiding a real goal-mouth trajectory or an unpossessed back-pass.
    threat = heading | (~own_team_controlled)
    active = (
        context.self_active
        & context.self_goalkeeper
        & context.ball_visible
        & observations.ball.live
        & (observations.restart.kind == RK_NONE)
    )
    sweeper_claim = (
        active
        & has_reachable_intercept
        & (context.ball_position[:, 2] < _GK_SWEEP_BALL_HEIGHT_M)
        & claimable_compatibility
        & keeper_wins
        & goalkeeper_is_primary
        & (ball[:, 0] < -jnp.float32(_GK_SWEEP_FIELD_FRACTION) * half_length)
        & threat
        & (distance < _GK_SWEEP_GOAL_DISTANCE_M)
    )
    urgent_rush = (
        active
        & has_reachable_intercept
        & (
            (
                (distance < _GK_RUSH_DISTANCE_M)
                & (~own_team_controlled)
                & goalkeeper_is_primary
            )
            | (
                heading
                & (distance < _GK_HEADING_RUSH_DISTANCE_M)
                & claimable_compatibility
            )
        )
    )
    claim_y_limit = jnp.maximum(
        half_penalty_width - jnp.float32(_GK_CLAIM_BOX_MARGIN_M), 0.0
    )
    sweep_target = jnp.stack(
        (
            jnp.clip(
                intercept[:, 0],
                own_goal[:, 0] + 0.3,
                own_goal[:, 0]
                + jnp.maximum(
                    penalty_length - jnp.float32(_GK_CLAIM_BOX_MARGIN_M),
                    jnp.float32(0.3),
                ),
            ),
            jnp.clip(intercept[:, 1], -claim_y_limit, claim_y_limit),
        ),
        axis=-1,
    )
    target = jnp.where(
        active[:, None],
        jnp.where(
            urgent_rush[:, None],
            intercept,
            jnp.where(sweeper_claim[:, None], sweep_target, angle_target),
        ),
        context.self_position,
    )
    return GoalkeeperCoverDecision(
        target=target.astype(jnp.float32),
        angle_target=angle_target.astype(jnp.float32),
        intercept_target=intercept.astype(jnp.float32),
        sweep_target=sweep_target.astype(jnp.float32),
        heading_toward_goal=heading & active,
        predicted_goal_y=predicted_y.astype(jnp.float32),
        reachable_intercept=has_reachable_intercept & active,
        sweeper_claim=sweeper_claim,
        urgent_rush=urgent_rush,
        active=active,
    )


def select_goalkeeper_distribution_target(
    context: RulePolicyContext,
    observations: Observation,
    roster: RosterMetadata,
    short_target_xy: jax.Array,
    short_completion: jax.Array,
    long_target_xy: jax.Array,
    long_arrival: jax.Array,
    prefer_long: jax.Array,
    *,
    receiver_eligible: jax.Array | None = None,
    decision_key: jax.Array | None = None,
    temperature: float = 0.20,
) -> GoalkeeperDistributionDecision:
    """Select a visible legal short or long goalkeeper receiver.

    Candidate arrays have shape ``(O, P, ...)``.  ``short_completion`` should
    come from the existing lane helper and ``long_arrival`` from
    :func:`aerial_arrival_score`; both are ranking scores in ``[0, 1]``.
    ``prefer_long`` has shape ``(O,)`` and is policy/style intent, not hidden
    environment state. ``receiver_eligible`` may supply a prospective Law 11
    and visibility mask for the current kick. When omitted, the restart path
    retains its public offside-latch gate. A long target is chosen
    only when explicitly preferred or no viable short outlet exists. If neither
    exists, ``valid`` is false and ``target`` remains the observed ball position.
    """

    observers = context.self_index.shape[0]
    if context.player_position.ndim != 3:
        raise ValueError("context.player_position must have observer and roster axes")
    players = context.player_position.shape[1]
    if context.teammate.shape != (observers, players):
        raise ValueError("context.teammate must match observer and roster axes")
    if observations.players.offside.shape != (observers, players):
        raise ValueError("offside mask must match observer and roster axes")
    if roster.is_goalkeeper.shape != (players,):
        raise ValueError("roster goalkeeper flags must match the roster axis")
    row = jnp.arange(observers, dtype=jnp.int32)
    expected_points = (observers, players, 2)
    expected_scores = (observers, players)
    short_target_xy = jnp.asarray(short_target_xy, dtype=jnp.float32)
    short_completion = jnp.asarray(short_completion, dtype=jnp.float32)
    long_target_xy = jnp.asarray(long_target_xy, dtype=jnp.float32)
    long_arrival = jnp.asarray(long_arrival, dtype=jnp.float32)
    prefer_long = jnp.asarray(prefer_long, dtype=jnp.bool_)
    if short_target_xy.shape != expected_points:
        raise ValueError("short_target_xy must have shape (observers, players, 2)")
    if long_target_xy.shape != expected_points:
        raise ValueError("long_target_xy must have shape (observers, players, 2)")
    if short_completion.shape != expected_scores:
        raise ValueError("short_completion must have shape (observers, players)")
    if long_arrival.shape != expected_scores:
        raise ValueError("long_arrival must have shape (observers, players)")
    if prefer_long.shape != (observers,):
        raise ValueError("prefer_long must have shape (observers,)")

    if receiver_eligible is None:
        eligible = ~observations.players.offside
    else:
        eligible = jnp.asarray(receiver_eligible, dtype=jnp.bool_)
        if eligible.shape != expected_scores:
            raise ValueError("receiver_eligible must have shape (observers, players)")
    receiver = context.teammate & (~roster.is_goalkeeper[None, :]) & eligible
    ball_xy = context.ball_position[:, :2]
    short_distance = _safe_norm(short_target_xy - ball_xy[:, None, :])
    long_distance = _safe_norm(long_target_xy - ball_xy[:, None, :])
    long_progress = long_target_xy[:, :, 0] - ball_xy[:, None, 0]
    short_candidate = (
        receiver
        & jnp.isfinite(short_completion)
        & (short_distance >= _SHORT_MIN_DISTANCE_M)
        & (short_distance <= _SHORT_MAX_DISTANCE_M)
        & (short_completion >= _SHORT_COMPLETION_FLOOR)
    )
    long_candidate = (
        receiver
        & jnp.isfinite(long_arrival)
        & (long_distance >= _LONG_MIN_DISTANCE_M)
        & (long_progress >= _LONG_MIN_PROGRESS_M)
        & (long_arrival >= _LONG_ARRIVAL_FLOOR)
    )
    short_score = jnp.where(short_candidate, short_completion, -jnp.inf)
    long_score = jnp.where(long_candidate, long_arrival, -jnp.inf)
    if decision_key is None:
        short_rank = short_score
        long_rank = long_score
    else:
        short_key, long_key = jax.random.split(decision_key)
        safe_temperature = jnp.maximum(jnp.float32(temperature), _EPS)
        short_rank = jnp.where(
            short_candidate,
            jnp.log(jnp.maximum(short_completion, 1.0e-4)) / safe_temperature
            + jax.random.gumbel(short_key, short_score.shape, dtype=jnp.float32),
            -jnp.inf,
        )
        long_rank = jnp.where(
            long_candidate,
            jnp.log(jnp.maximum(long_arrival, 1.0e-4)) / safe_temperature
            + jax.random.gumbel(long_key, long_score.shape, dtype=jnp.float32),
            -jnp.inf,
        )
    short_slot = jnp.argmax(short_rank, axis=-1).astype(jnp.int32)
    long_slot = jnp.argmax(long_rank, axis=-1).astype(jnp.int32)
    has_short = jnp.any(short_candidate, axis=-1)
    has_long = jnp.any(long_candidate, axis=-1)
    use_long = has_long & (prefer_long | (~has_short))
    chosen_slot = jnp.where(use_long, long_slot, short_slot)
    short_target = short_target_xy[row, chosen_slot]
    long_target = long_target_xy[row, chosen_slot]
    target = jnp.where(use_long[:, None], long_target, short_target)
    active_goalkeeper = (
        context.self_active & context.self_goalkeeper & context.ball_visible
    )
    valid = active_goalkeeper & (has_short | has_long)
    target = jnp.where(valid[:, None], target, ball_xy)
    return GoalkeeperDistributionDecision(
        target=target.astype(jnp.float32),
        receiver=jnp.where(valid, chosen_slot, jnp.int32(NO_PLAYER)),
        valid=valid,
        use_long=use_long & valid,
        has_short=has_short & active_goalkeeper,
        has_long=has_long & active_goalkeeper,
    )


__all__ = [
    "AerialArrivalScore",
    "AerialContestDecision",
    "GoalkeeperCoverDecision",
    "GoalkeeperDistributionDecision",
    "aerial_arrival_score",
    "aerial_contest_decision",
    "goalkeeper_cover_decision",
    "select_goalkeeper_distribution_target",
]
