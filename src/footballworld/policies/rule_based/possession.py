"""Observation-only possession scoring for one carrier row.

The carrier compares shots, passes, dribbles, and emergency clearances in one
bounded ranking currency.  These scores are deterministic tactical rankings,
not calibrated probabilities or expected-goals estimates.  Every input comes
from one :class:`RulePolicyContext` row plus public offside and roster flags;
hidden environment state is deliberately out of scope.  The integrating policy
may supply its physics-aware pass targets and completion scores so candidate
geometry is not evaluated twice.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER
from footballworld.policies.rule_based.attack_pattern import AttackPattern
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.context import RulePolicyContext
from footballworld.policies.rule_based.tactical_plan import (
    TacticalProfile,
    tactical_profile,
)
from footballworld.policies.rule_based.tactics import (
    deterministic_masked_argmax,
    lane_completion,
    moving_receiver_target,
    openness,
    pitch_value,
    pressure,
    shot_quality,
)

POSSESSION_SHOT = 0
POSSESSION_PASS = 1
POSSESSION_DRIBBLE = 2
POSSESSION_CLEAR = 3

# These are uncalibrated policy design priors, not measured football constants.
# A six-metre pass excludes a
# teammate standing inside the carrier's immediate control pocket.  The
# clearance gate (deep 72% of the own half and pressure >= 0.78) keeps a long
# release as an emergency action instead of an ordinary build-up shortcut.
_MIN_PASS_DISTANCE_M = 6.0
_SAFE_PASS_COMPLETION = 0.58
_CLEARANCE_DEPTH_FRACTION = 0.72
# The reference 0.78 design prior was expressed on an unbounded raw pressure
# sum. This policy exposes ``1 - exp(-raw)`` instead, so use the exact
# coordinate transform rather than treating 0.78 as a measured bounded value.
_REFERENCE_RAW_CLEARANCE_PRESSURE_PRIOR = 0.78
_CLEARANCE_PRESSURE = 1.0 - math.exp(-_REFERENCE_RAW_CLEARANCE_PRESSURE_PRIOR)

# Candidate horizons are compact tactical look-aheads, not ball-trajectory
# integration.  The openness cap matches ``tactics.openness`` and therefore
# converts its metre output into the same [0, 1] currency as the other terms.
_OPENNESS_CAP_M = 15.0
_DRIBBLE_STEP_M = 4.0
_CLEARANCE_DISTANCE_M = 30.0
_TARGET_BOUNDARY_MARGIN_M = 1.0

# Shots target a point 64% of the way from goal centre to the post opposite a
# visible goalkeeper (or opposite the carrier's side as a stable fallback).
# The margin leaves room for the ball radius and model error.
_SHOT_TARGET_HALF_WIDTH_FRACTION = 0.64
_SHOT_RANGE_SOFTNESS_FRACTION = 0.25
_SHOT_RANGE_MIN_SOFTNESS_M = 4.0
_SHOT_PRESSURE_PREFERENCE_GAIN = 0.18
_SHOT_GK_COVERAGE_GAIN = 0.30
_SHOT_GK_COVERAGE_WIDTH_FRACTION = 0.22

_EPS = 1.0e-6
_RECEIVER_RANDOM_STREAM = 0x52454356
_MACRO_RANDOM_STREAM = 0x4D414352
_DRIBBLE_DIRECTION_RANDOM_STREAM = 0x44524942
_CLEAR_DIRECTION_RANDOM_STREAM = 0x434C5244
_SHOT_PORTION_RANDOM_STREAM = 0x53484F54
_CURVE_RANDOM_STREAM = 0x43555256
_SHOT_DIRECTION_RANDOM_STREAM = 0x53484452
_SHOT_LAUNCH_RANDOM_STREAM = 0x53484C4E


class PossessionDecision(NamedTuple):
    """Compact carrier command before encoding continuous contact controls.

    ``direction`` is a unit vector in the observer's attacking frame.  Power
    and launch use the public normalized action convention.  ``target`` keeps
    the best eligible receiver slot even when another macro action wins, and is
    ``NO_PLAYER`` only when no pass candidate exists.  This lets the caller
    reuse the same ranking for goalkeeper distribution without scoring twice.
    """

    direction: jax.Array
    power: jax.Array
    launch: jax.Array
    spin: jax.Array
    cross: jax.Array
    kind: jax.Array
    target: jax.Array


class PossessionCandidateTrace(NamedTuple):
    """Fixed-shape receiver candidates emitted only by diagnostic calls."""

    carrier_slot: jax.Array
    receiver_slot: jax.Array
    is_cross: jax.Array
    eligible: jax.Array
    selected: jax.Array
    target_xy: jax.Array
    completion: jax.Array
    selection_value: jax.Array
    macro_value: jax.Array
    progression_contribution: jax.Array
    relative_depth_m: jax.Array
    receiver_forward_velocity_mps: jax.Array
    source_xy: jax.Array
    decision_due: jax.Array


class ShotPlan(NamedTuple):
    """One fixed-shape, observation-only shot proposal.

    Value is the bounded tactical preference used by the macro selector; it
    is not a calibrated scoring probability. Keeping geometry, execution
    noise and controls together ensures that established possession and
    first-time goalkeeper rebounds use identical shooting semantics.
    """

    direction: jax.Array
    power: jax.Array
    launch: jax.Array
    spin: jax.Array
    value: jax.Array
    quality: jax.Array


def _safe_unit(vector: jax.Array) -> jax.Array:
    """Return an exact zero for a zero vector and a unit vector otherwise."""

    vector = jnp.asarray(vector, dtype=jnp.float32)
    squared = jnp.sum(vector * vector, axis=-1, keepdims=True)
    norm = jnp.sqrt(squared + (squared == 0.0).astype(jnp.float32))
    return jnp.where(squared > 0.0, vector / norm, 0.0).astype(jnp.float32)


def _masked_categorical(
    values: jax.Array,
    mask: jax.Array,
    key: jax.Array | None,
    temperature: float,
) -> tuple[jax.Array, jax.Array]:
    """Sample one valid bounded score, with deterministic standalone fallback."""

    fallback, available = deterministic_masked_argmax(values, mask)
    if key is None:
        return fallback, available
    logits = jnp.where(
        mask,
        jnp.log(jnp.maximum(values, jnp.float32(1.0e-4))) / temperature,
        -jnp.inf,
    )
    sampled = jax.random.categorical(key, logits.astype(jnp.float32)).astype(jnp.int32)
    return jnp.where(available, sampled, fallback), available


def _validate_row(
    context: RulePolicyContext,
    player_offside: jax.Array,
    player_is_goalkeeper: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Validate static one-observer axes while tracing, without runtime checks."""

    if context.self_position.shape != (2,):
        raise ValueError("context must be one observer row")
    if context.player_position.ndim != 2 or context.player_position.shape[1] != 2:
        raise ValueError("context player_position must have shape (players, 2)")
    player_count = context.player_position.shape[0]
    if context.player_velocity.shape != (player_count, 2):
        raise ValueError("context player_velocity must match player_position")
    if context.teammate.shape != (player_count,):
        raise ValueError("context teammate mask must match the roster axis")
    if context.opponent.shape != (player_count,):
        raise ValueError("context opponent mask must match the roster axis")

    offside = jnp.asarray(player_offside, dtype=jnp.bool_)
    goalkeeper = jnp.asarray(player_is_goalkeeper, dtype=jnp.bool_)
    if offside.shape != (player_count,):
        raise ValueError("player_offside must have shape (players,)")
    if goalkeeper.shape != (player_count,):
        raise ValueError("player_is_goalkeeper must have shape (players,)")
    return offside, goalkeeper


def plan_shot(
    source_xy: jax.Array,
    player_position: jax.Array,
    opponent: jax.Array,
    opponent_goalkeeper: jax.Array,
    config: RulePolicyConfig,
    *,
    half_length: float,
    goal_width: float,
    current_pressure: jax.Array,
    decision_key: jax.Array | None = None,
    shot_launch_radians_per_action_unit: float = 1.0,
) -> ShotPlan:
    """Build the shared fixed-shape shot value and execution controls.

    All inputs are one public observation row. The helper deliberately keeps
    FootballWorld's three-zone goalkeeper-avoidance target, curl, launch and
    execution-noise model in one place so a rebound cannot silently use a
    simpler or more accurate shooting model than established possession.
    """

    source = jnp.asarray(source_xy, dtype=jnp.float32)
    positions = jnp.asarray(player_position, dtype=jnp.float32)
    opponents = jnp.asarray(opponent, dtype=jnp.bool_)
    goalkeepers = jnp.asarray(opponent_goalkeeper, dtype=jnp.bool_)
    if source.shape != (2,):
        raise ValueError("source_xy must have shape (2,)")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("player_position must have shape (players, 2)")
    if opponents.shape != positions.shape[:1]:
        raise ValueError("opponent must have shape (players,)")
    if goalkeepers.shape != positions.shape[:1]:
        raise ValueError("opponent_goalkeeper must have shape (players,)")

    hx = jnp.maximum(jnp.asarray(half_length, dtype=jnp.float32), _EPS)
    width = jnp.maximum(jnp.asarray(goal_width, dtype=jnp.float32), _EPS)
    pressure_value = jnp.clip(
        jnp.asarray(current_pressure, dtype=jnp.float32), 0.0, 1.0
    )
    launch_radians_per_unit = jnp.maximum(
        jnp.asarray(
            shot_launch_radians_per_action_unit,
            dtype=jnp.float32,
        ),
        _EPS,
    )

    visible_goalkeeper_count = jnp.sum(goalkeepers.astype(jnp.float32))
    goalkeeper_y = jnp.sum(jnp.where(goalkeepers, positions[:, 1], 0.0)) / jnp.maximum(
        visible_goalkeeper_count, 1.0
    )
    reference_y = jnp.where(visible_goalkeeper_count > 0.0, goalkeeper_y, source[1])
    goal_portions = (
        jnp.asarray((-1.0, 0.0, 1.0), dtype=jnp.float32)
        * 0.5
        * width
        * _SHOT_TARGET_HALF_WIDTH_FRACTION
    )
    portion_value = jnp.where(
        visible_goalkeeper_count > 0.0,
        0.20 + 0.80 * jnp.clip(jnp.abs(goal_portions - goalkeeper_y) / width, 0.0, 1.0),
        jnp.asarray((1.0, 0.82, 1.0), dtype=jnp.float32),
    )
    portion_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _SHOT_PORTION_RANDOM_STREAM)
    )
    if portion_key is None:
        portion_index = jnp.argmax(portion_value)
    else:
        portion_logits = (
            jnp.log(jnp.maximum(portion_value, 1e-4)) / config.shot_portion_temperature
        )
        portion_index = jax.random.categorical(
            portion_key, portion_logits.astype(jnp.float32)
        )
    desired_goal_y = goal_portions[portion_index]
    target_side = jnp.where(
        jnp.abs(desired_goal_y) > _EPS,
        jnp.sign(desired_goal_y),
        -jnp.where(reference_y >= 0.0, 1.0, -1.0),
    )
    distance_to_goal = jnp.linalg.norm(
        jnp.stack((hx - source[0], desired_goal_y - source[1]))
    )
    curve_fraction = jnp.clip(
        (distance_to_goal - config.shoot_curl_start_m)
        / jnp.maximum(config.shoot_distance_m - config.shoot_curl_start_m, _EPS),
        0.0,
        1.0,
    )
    curve_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _CURVE_RANDOM_STREAM)
    )
    curve_strength = (
        jnp.float32(1.0)
        if curve_key is None
        else jnp.float32(0.72)
        + jnp.float32(0.28) * jax.random.uniform(curve_key, dtype=jnp.float32)
    )
    curve_control = config.shoot_curl_spin * curve_fraction * curve_strength
    aim_y = desired_goal_y * (
        1.0 - config.shoot_curl_aim_compensation * curve_fraction * curve_strength
    )
    shot_target = jnp.stack((hx, aim_y))
    direction = _safe_unit(shot_target - source)
    spin = jnp.stack((target_side * curve_control, 0.0)).astype(jnp.float32)
    quality = shot_quality(
        source,
        positions,
        opponents,
        goalkeepers,
        half_length=hx,
        goal_width=width,
    )
    range_softness = jnp.maximum(
        jnp.asarray(config.shoot_distance_m, jnp.float32)
        * _SHOT_RANGE_SOFTNESS_FRACTION,
        _SHOT_RANGE_MIN_SOFTNESS_M,
    )
    range_preference = jax.nn.sigmoid(
        (jnp.asarray(config.shoot_distance_m, jnp.float32) - distance_to_goal)
        / range_softness
    )
    goalkeeper_separation = jnp.abs(goalkeeper_y - desired_goal_y)
    goalkeeper_coverage = jnp.where(
        visible_goalkeeper_count > 0.0,
        jnp.exp(
            -goalkeeper_separation
            / jnp.maximum(width * _SHOT_GK_COVERAGE_WIDTH_FRACTION, _EPS)
        ),
        0.0,
    )
    goalkeeper_factor = 1.0 - _SHOT_GK_COVERAGE_GAIN * goalkeeper_coverage
    pressure_preference = (
        1.0
        - _SHOT_PRESSURE_PREFERENCE_GAIN
        + _SHOT_PRESSURE_PREFERENCE_GAIN * pressure_value
    )
    value = jnp.clip(
        config.shot_value_gain
        * quality
        * (
            1.0
            + config.shot_quality_selectivity_gain * jnp.square(jnp.square(quality))
        )
        * (0.06 + 0.94 * range_preference)
        * goalkeeper_factor
        * pressure_preference,
        0.0,
        1.0,
    )
    shot_noise = (
        config.shot_noise_base_rad
        + config.shot_noise_quality_rad * (1.0 - quality)
        + config.shot_noise_pressure_rad * pressure_value
    )
    if decision_key is not None:
        horizontal_error = (
            jax.random.normal(
                jax.random.fold_in(decision_key, _SHOT_DIRECTION_RANDOM_STREAM),
                dtype=jnp.float32,
            )
            * shot_noise
        )
        cos_error = jnp.cos(horizontal_error)
        sin_error = jnp.sin(horizontal_error)
        direction = jnp.stack(
            (
                cos_error * direction[0] - sin_error * direction[1],
                sin_error * direction[0] + cos_error * direction[1],
            )
        ).astype(jnp.float32)
        vertical_error = (
            jax.random.normal(
                jax.random.fold_in(decision_key, _SHOT_LAUNCH_RANDOM_STREAM),
                dtype=jnp.float32,
            )
            * shot_noise
        )
        launch = jnp.clip(
            config.shoot_launch + vertical_error / launch_radians_per_unit,
            -1.0,
            1.0,
        )
    else:
        launch = jnp.asarray(config.shoot_launch, dtype=jnp.float32)

    return ShotPlan(
        direction=direction.astype(jnp.float32),
        power=jnp.asarray(config.shoot_power, dtype=jnp.float32),
        launch=launch.astype(jnp.float32),
        spin=spin,
        value=value.astype(jnp.float32),
        quality=quality.astype(jnp.float32),
    )


def _progressive_pass_gain(
    signed_progress: jax.Array,
    pass_lane: jax.Array,
    tactical_gain: float | jax.Array,
    policy_gain: float | jax.Array,
) -> jax.Array:
    """Bound an independent reward to positive attack-axis progress only."""

    progress = jnp.maximum(jnp.asarray(signed_progress, jnp.float32), 0.0)
    completion = jnp.clip(jnp.asarray(pass_lane, jnp.float32), 0.0, 1.0)
    inherited = jnp.asarray(tactical_gain, jnp.float32) * progress
    calibrated = jnp.asarray(policy_gain, jnp.float32) * completion * progress
    return jnp.clip(inherited + calibrated, 0.0, 1.0)


def _backward_pass_cost(
    signed_progress: jax.Array,
    current_pressure: jax.Array,
    build_up_depth: jax.Array,
    base_penalty: float | jax.Array,
    advanced_gain: float | jax.Array,
) -> jax.Array:
    """Penalize safe recycling more after midfield, fading under pressure."""

    advanced_depth = jnp.clip(
        jnp.float32(2.0) * jnp.asarray(build_up_depth, jnp.float32) - jnp.float32(1.0),
        0.0,
        1.0,
    )
    penalty = (
        jnp.asarray(base_penalty, jnp.float32)
        + jnp.asarray(advanced_gain, jnp.float32) * advanced_depth
    )
    return (
        penalty
        * jnp.maximum(-jnp.asarray(signed_progress, jnp.float32), 0.0)
        * (1.0 - jnp.clip(jnp.asarray(current_pressure, jnp.float32), 0.0, 1.0))
    )


def _pass_width_context(source_y, target_y, completion, half_width):
    """Return wide-source, switch-lane and contextual lateral-cost arrays."""

    source_y = jnp.asarray(source_y, dtype=jnp.float32)
    target_y = jnp.asarray(target_y, dtype=jnp.float32)
    completion = jnp.asarray(completion, dtype=jnp.float32)
    hy = jnp.maximum(jnp.asarray(half_width, dtype=jnp.float32), _EPS)
    source_width = jnp.clip(jnp.abs(source_y) / hy, 0.0, 1.0)
    target_width = jnp.clip(jnp.abs(target_y) / hy, 0.0, 1.0)
    source_is_wide = source_width >= jnp.float32(0.18)
    switch_lane = (
        source_is_wide
        & (source_y * target_y < 0.0)
        & (jnp.abs(target_y - source_y) >= jnp.float32(0.35) * hy)
    )
    width_expansion = jnp.maximum(target_width - source_width, 0.0)
    productive_width = jnp.maximum(width_expansion, switch_lane.astype(jnp.float32))
    lateral_cost_scale = 1.0 - jnp.clip(completion * productive_width, 0.0, 1.0)
    return source_is_wide, switch_lane, lateral_cost_scale


def decide_possession(
    context: RulePolicyContext,
    player_offside: jax.Array,
    player_is_goalkeeper: jax.Array,
    config: RulePolicyConfig,
    *,
    half_length: float,
    half_width: float,
    goal_width: float,
    pass_target_xy: jax.Array | None = None,
    pass_completion: jax.Array | None = None,
    pass_candidate: jax.Array | None = None,
    pass_continuation: jax.Array | None = None,
    cross_target_xy: jax.Array | None = None,
    cross_completion: jax.Array | None = None,
    cross_candidate: jax.Array | None = None,
    possession_seconds: jax.Array = jnp.float32(0.0),
    possession_episode_seconds: jax.Array | None = None,
    formation_anchor_y: jax.Array = jnp.float32(0.0),
    previous_actor: jax.Array | None = None,
    decision_key: jax.Array | None = None,
    tactical: TacticalProfile | None = None,
    decision_due: jax.Array = jnp.bool_(True),
    attack_pattern: jax.Array = jnp.int32(-1),
    attack_phase: jax.Array = jnp.int32(-1),
    run_behind_receiver: jax.Array = jnp.int32(NO_PLAYER),
    progressive_carry_commit_s: jax.Array = jnp.float32(0.0),
    shot_launch_radians_per_action_unit: float = 1.0,
    with_candidate_trace: bool = False,
) -> PossessionDecision | tuple[PossessionDecision, PossessionCandidateTrace]:
    """Choose one carrier action from one observer's public information.

    A teammate is eligible only when that slot is active, visible, on the same
    team, and onside for the prospective touch supplied by the caller.  Forward
    progress is a soft preference rather than a gate, so a sufficiently safer
    lateral or backward outlet remains selectable.

    ``pass_target_xy``, ``pass_completion``, and ``pass_candidate`` are an
    all-or-none override.  The baseline policy supplies them from its rolling
    ball model, moving-receiver ETA and interception calculation.  The three
    ``cross_*`` inputs form an independent all-or-none aerial service override.
    Omitting either group retains the lightweight standalone fallback.
    """

    if type(with_candidate_trace) is not bool:
        raise TypeError("with_candidate_trace must be a static bool")
    offside, goalkeeper = _validate_row(context, player_offside, player_is_goalkeeper)
    hx = jnp.maximum(jnp.asarray(half_length, jnp.float32), _EPS)
    hy = jnp.maximum(jnp.asarray(half_width, jnp.float32), _EPS)
    goal_width = jnp.maximum(jnp.asarray(goal_width, jnp.float32), _EPS)

    tactical = (
        tactical_profile(config.team_tactical_plans[0])
        if tactical is None
        else tactical
    )
    possession_seconds = jnp.maximum(
        jnp.asarray(possession_seconds, dtype=jnp.float32), 0.0
    )
    possession_episode_seconds = jnp.maximum(
        possession_seconds
        if possession_episode_seconds is None
        else jnp.asarray(possession_episode_seconds, dtype=jnp.float32),
        0.0,
    )
    formation_anchor_y = jnp.asarray(formation_anchor_y, dtype=jnp.float32)
    attack_pattern = jnp.asarray(attack_pattern, dtype=jnp.int32)
    attack_phase = jnp.asarray(attack_phase, dtype=jnp.int32)
    run_behind_receiver = jnp.asarray(run_behind_receiver, dtype=jnp.int32)
    progressive_carry_commit_s = jnp.maximum(
        jnp.asarray(progressive_carry_commit_s, dtype=jnp.float32), 0.0
    )
    for name, value in (
        ("possession_episode_seconds", possession_episode_seconds),
        ("formation_anchor_y", formation_anchor_y),
        ("attack_pattern", attack_pattern),
        ("attack_phase", attack_phase),
        ("run_behind_receiver", run_behind_receiver),
        ("progressive_carry_commit_s", progressive_carry_commit_s),
    ):
        if value.shape != ():
            raise ValueError(f"{name} must be scalar")
    launch_radians_per_unit = jnp.maximum(
        jnp.asarray(shot_launch_radians_per_action_unit, dtype=jnp.float32),
        _EPS,
    )
    source = jnp.where(
        context.ball_visible,
        context.ball_position[:2],
        context.self_position,
    ).astype(jnp.float32)
    player_position = jnp.asarray(context.player_position, jnp.float32)
    player_velocity = jnp.asarray(context.player_velocity, jnp.float32)
    opponent = jnp.asarray(context.opponent, jnp.bool_)
    onside_teammate = jnp.asarray(context.teammate, jnp.bool_) & (~offside)
    opponent_goalkeeper = opponent & goalkeeper
    previous_actor = (
        jnp.zeros_like(onside_teammate)
        if previous_actor is None
        else jnp.asarray(previous_actor, dtype=jnp.bool_)
    )
    if previous_actor.shape != onside_teammate.shape:
        raise ValueError("previous_actor must have shape (players,)")
    roster_slot = jnp.arange(player_position.shape[0], dtype=jnp.int32)
    active_same_team_actor = (
        jnp.asarray(context.same_team, dtype=jnp.bool_)
        & jnp.asarray(context.participating, dtype=jnp.bool_)
        & (roster_slot != jnp.asarray(context.self_index, dtype=jnp.int32))
    )
    has_previous_actor = jnp.any(previous_actor & active_same_team_actor)
    # possession_seconds is the caller's observer-causal player carry age,
    # which bridges same-player dribble recontacts. Team episode age remains a
    # fallback only when no preceding teammate exists, so a normal reception
    # never inherits the whole build-up as personal tenure.
    solo_tenure_seconds = jnp.maximum(
        possession_seconds,
        jnp.where(has_previous_actor, 0.0, possession_episode_seconds),
    )
    carry_fraction = jnp.clip(
        solo_tenure_seconds / config.solo_carry_soft_limit_s, 0.0, 1.0
    )

    current_pressure = pressure(
        source,
        player_position,
        player_velocity,
        opponent,
        distance_scale_m=config.pressure_distance_m,
    )
    # Passes aim at the receiver's short-horizon arrival point.  Completion,
    # target threat, pressure relief, openness, progress, distance, and width
    # are all bounded before comparison with the other macro actions.
    supplied_pass = pass_target_xy is not None
    if supplied_pass != (pass_completion is not None) or supplied_pass != (
        pass_candidate is not None
    ):
        raise ValueError(
            "pass_target_xy, pass_completion, and pass_candidate must be supplied "
            "together"
        )
    supplied_cross = cross_target_xy is not None
    if supplied_cross != (cross_completion is not None) or supplied_cross != (
        cross_candidate is not None
    ):
        raise ValueError(
            "cross_target_xy, cross_completion, and cross_candidate must be "
            "supplied together"
        )
    expected_points = player_position.shape
    expected_scores = player_position.shape[:1]
    if pass_continuation is None:
        prepared_continuation = jnp.zeros(expected_scores, dtype=jnp.float32)
    else:
        prepared_continuation = jnp.asarray(pass_continuation, dtype=jnp.float32)
        if prepared_continuation.shape != expected_scores:
            raise ValueError("pass_continuation must have shape (players,)")
    if supplied_pass:
        receiver_target = jnp.asarray(pass_target_xy, dtype=jnp.float32)
        prepared_completion = jnp.asarray(pass_completion, dtype=jnp.float32)
        prepared_candidate = jnp.asarray(pass_candidate, dtype=jnp.bool_)
        if receiver_target.shape != expected_points:
            raise ValueError("pass_target_xy must have shape (players, 2)")
        if prepared_completion.shape != expected_scores:
            raise ValueError("pass_completion must have shape (players,)")
        if prepared_candidate.shape != expected_scores:
            raise ValueError("pass_candidate must have shape (players,)")
    else:
        receiver_target = moving_receiver_target(
            source,
            player_position,
            player_velocity,
            onside_teammate,
            half_length=hx,
            half_width=hy,
            boundary_margin_m=_TARGET_BOUNDARY_MARGIN_M,
        )
    pass_delta = receiver_target - source
    pass_distance = jnp.linalg.norm(pass_delta, axis=-1)
    if supplied_pass:
        pass_lane = jnp.clip(prepared_completion, 0.0, 1.0)
    else:
        pass_lane = lane_completion(
            source,
            receiver_target,
            player_position,
            player_velocity,
            opponent,
        )
    if supplied_cross:
        prepared_cross_target = jnp.asarray(cross_target_xy, dtype=jnp.float32)
        prepared_cross_completion = jnp.asarray(cross_completion, dtype=jnp.float32)
        prepared_cross_candidate = jnp.asarray(cross_candidate, dtype=jnp.bool_)
        if prepared_cross_target.shape != expected_points:
            raise ValueError("cross_target_xy must have shape (players, 2)")
        if prepared_cross_completion.shape != expected_scores:
            raise ValueError("cross_completion must have shape (players,)")
        if prepared_cross_candidate.shape != expected_scores:
            raise ValueError("cross_candidate must have shape (players,)")
    else:
        prepared_cross_target = receiver_target
        prepared_cross_completion = jnp.zeros(expected_scores, dtype=jnp.float32)
        prepared_cross_candidate = jnp.zeros(expected_scores, dtype=jnp.bool_)
    receiver_pressure = pressure(
        receiver_target,
        player_position,
        player_velocity,
        opponent,
        distance_scale_m=config.pressure_distance_m,
    )
    receiver_open = jnp.clip(
        openness(
            receiver_target,
            player_position,
            player_velocity,
            opponent,
            cap_m=_OPENNESS_CAP_M,
        )
        / _OPENNESS_CAP_M,
        0.0,
        1.0,
    )
    receiver_shot = shot_quality(
        receiver_target,
        player_position,
        opponent,
        opponent_goalkeeper,
        half_length=hx,
        goal_width=goal_width,
    )
    receiver_pitch = pitch_value(receiver_target, half_length=hx, half_width=hy)
    receiver_threat = jnp.maximum(receiver_shot, receiver_pitch)
    receiver_security = 0.5 * (1.0 - receiver_pressure) + 0.5 * receiver_open

    # ``pass_min_progress_m`` is retained as a smooth scale.  It no longer
    # excludes lateral/backward outlets as the first-stage policy did.
    progress_scale = jnp.maximum(
        jnp.asarray(config.pass_min_progress_m, jnp.float32), _EPS
    )
    signed_progress = jnp.tanh(pass_delta[:, 0] / (2.0 * progress_scale))
    progress_value = 0.5 + 0.5 * signed_progress
    # The tactical progression gain previously divided longitudinal metres by
    # the full 105 m pitch. That made its nominal 0.10 range nearly inert for
    # ordinary 4--20 m passes. Reuse the already-bounded pass-progress scale:
    # backward/lateral options receive no bonus and forward options approach
    # the configured gain smoothly without changing candidate eligibility.
    progressive_preference = jnp.maximum(signed_progress, 0.0)
    pass_base = (
        0.45 * receiver_threat + 0.35 * receiver_security + 0.20 * progress_value
    )
    distance_retention = jnp.exp(
        -jnp.asarray(config.pass_distance_penalty_per_m, jnp.float32) * pass_distance
    )
    lateral_fraction = jnp.clip(jnp.abs(pass_delta[:, 1]) / (2.0 * hy), 0.0, 1.0)
    target_width = jnp.clip(jnp.abs(receiver_target[:, 1]) / hy, 0.0, 1.0)
    build_up_depth = jnp.clip((source[0] + hx) / (2.0 * hx), 0.25, 1.0)
    # A completed pass that creates usable width or switches the point of
    # attack is not the sterile lateral circulation penalized below. Unsafe
    # or merely sideways candidates retain the full cost.
    source_is_wide, switch_lane, lateral_cost_scale = _pass_width_context(
        source[1], receiver_target[:, 1], pass_lane, hy
    )
    immediate_return = previous_actor & (possession_seconds < jnp.float32(1.0))
    # Progression/width preference is conditional on a completion-qualified
    # pass. Reference numerical floors are not copied because this policy's
    # lane-and-arrival score has a different contract. Keep
    # the same safety principle continuously so a zero-completion lane cannot
    # become attractive from tactical preference alone.
    pass_progression_gain = _progressive_pass_gain(
        progressive_preference,
        pass_lane,
        tactical.progressive_pass_gain,
        config.progressive_pass_value_gain,
    )
    tactical_gain = (
        pass_progression_gain + tactical.wide_pass_gain * target_width * build_up_depth
    )
    return_cost = (
        config.immediate_return_penalty
        * immediate_return.astype(jnp.float32)
        * (1.0 - current_pressure)
    )
    # The long-run diagnostic found that ordinary completed passes moved the
    # ball backwards on average even though crosses progressed. Penalize only
    # the backward component, and fade that cost under pressure so a safe
    # reset remains available. This is a policy prior, never an eligibility
    # gate, and therefore cannot suppress the only reachable teammate.
    backward_cost = _backward_pass_cost(
        signed_progress,
        current_pressure,
        build_up_depth,
        config.backward_pass_penalty,
        config.advanced_backward_pass_penalty_gain,
    )
    pass_completion_weight = jnp.square(pass_lane)
    pass_value = jnp.clip(
        pass_lane * (pass_base * distance_retention + tactical_gain)
        - config.pass_lateral_penalty * lateral_fraction * lateral_cost_scale
        - return_cost
        - backward_cost,
        0.0,
        1.0,
    )
    # Episode patterns are completion-conditioned score priors only.  They do
    # not open a teammate, offside, distance, or caller-supplied eligibility
    # gate. THIRD_MAN means an A→B→C preference based on the observed previous
    # actor; it is deliberately not claimed as a guaranteed physical two-hop.
    candidate_index = jnp.arange(player_position.shape[0], dtype=jnp.int32)
    setup_phase = attack_phase == jnp.int32(0)
    execution_phase = attack_phase == jnp.int32(1)
    # A third-player combination needs an actual first leg.  In setup, favour
    # a secure connecting receiver; after the observed handoff, exclude that
    # connector and look for the progressing third player.  The phase machine
    # ends after the next handoff, so this cannot become an A-B-A loop.
    third_man_setup_value = setup_phase.astype(jnp.float32) * (
        0.70 * receiver_security + 0.30 * progress_value
    )
    third_man_execution_value = (
        has_previous_actor.astype(jnp.float32)
        * execution_phase.astype(jnp.float32)
        * (~previous_actor).astype(jnp.float32)
        * (0.55 * receiver_security + 0.45 * progress_value)
    )
    third_man_value = jnp.where(
        setup_phase,
        third_man_setup_value,
        third_man_execution_value,
    )
    same_flank = jnp.where(
        source_is_wide,
        (source[1] * receiver_target[:, 1] >= 0.0).astype(jnp.float32),
        target_width,
    )
    wide_active = (attack_phase >= 0) & (attack_phase <= 1)
    wide_value = (
        wide_active.astype(jnp.float32)
        * same_flank
        * (0.55 * target_width + 0.45 * progress_value)
    )
    switch_setup_value = same_flank * (0.70 * receiver_security + 0.30 * progress_value)
    switch_release_value = switch_lane.astype(jnp.float32) * (
        0.55 * receiver_security + 0.45 * progress_value
    )
    switch_value = jnp.where(
        setup_phase,
        switch_setup_value,
        jnp.where(execution_phase, switch_release_value, jnp.float32(0.0)),
    )
    receiver_forward_speed = jnp.clip(player_velocity[:, 0] / 7.0, 0.0, 1.0)
    run_value = (
        execution_phase.astype(jnp.float32)
        * (candidate_index == run_behind_receiver).astype(jnp.float32)
        * (
            0.50 * progressive_preference
            + 0.30 * receiver_forward_speed
            + 0.20 * receiver_security
        )
    )
    pattern_value = jnp.where(
        attack_pattern == jnp.int32(AttackPattern.THIRD_MAN),
        third_man_value,
        jnp.where(
            attack_pattern == jnp.int32(AttackPattern.WIDE_OVERLOAD),
            wide_value,
            jnp.where(
                attack_pattern == jnp.int32(AttackPattern.SWITCH_PLAY),
                switch_value,
                jnp.where(
                    attack_pattern == jnp.int32(AttackPattern.RUN_BEHIND_DIRECT),
                    run_value,
                    jnp.float32(0.0),
                ),
            ),
        ),
    )
    # The same episode-stable forward that occupies the movement pocket gets
    # a bounded completion-conditioned preference. This coordinates space
    # creation with pass intent without opening a marginal or illegal lane.
    designated_forward_value = (
        (attack_phase >= 0).astype(jnp.float32)
        * (candidate_index == run_behind_receiver).astype(jnp.float32)
        * (0.55 * progressive_preference + 0.45 * receiver_security)
    )
    pass_value = jnp.clip(
        pass_value
        # Pattern identity may rank only already credible lanes. Squaring the
        # completion score prevents a long-plan bias from rescuing a marginal
        # pass merely because it fits the intended choreography.
        + config.attack_pattern_receiver_gain * pass_completion_weight * pattern_value
        + config.forward_pocket_receiver_gain
        * pass_completion_weight
        * designated_forward_value
        # Credit the first leg only for a bounded, completion-qualified second
        # ground leg. The caller computes this physical continuation on the
        # sparse carrier row; it remains a tactical score, not a probability.
        + config.continuation_value_gain
        * pass_completion_weight
        * jnp.clip(prepared_continuation, 0.0, 1.0),
        0.0,
        1.0,
    )
    eligible_pass = onside_teammate & (pass_distance >= _MIN_PASS_DISTANCE_M)
    if supplied_pass:
        eligible_pass = eligible_pass & prepared_candidate
    cross_delta = prepared_cross_target - source
    cross_distance = jnp.linalg.norm(cross_delta, axis=-1)
    cross_centrality = jnp.clip(
        1.0 - jnp.abs(prepared_cross_target[:, 1]) / hy, 0.0, 1.0
    )
    cross_forward = jnp.clip(cross_delta[:, 0] / (2.0 * hx), 0.0, 1.0)
    cross_progressive_preference = jnp.maximum(
        jnp.tanh(cross_delta[:, 0] / (2.0 * progress_scale)),
        0.0,
    )
    cross_threat = pitch_value(prepared_cross_target, half_length=hx, half_width=hy)
    cross_completion = jnp.clip(prepared_cross_completion, 0.0, 1.0)
    cross_tactical_gain = (
        tactical.progressive_pass_gain * cross_progressive_preference
        + tactical.wide_pass_gain * cross_centrality * build_up_depth
    )
    cross_base_value = jnp.clip(
        cross_completion
        * (
            (0.45 * cross_threat + 0.35 * cross_centrality + 0.20 * cross_forward)
            * jnp.exp(
                -0.5
                * jnp.asarray(config.pass_distance_penalty_per_m, jnp.float32)
                * cross_distance
            )
            + cross_tactical_gain
        ),
        0.0,
        1.0,
    )
    wide_cross_pattern = (
        attack_pattern == jnp.int32(AttackPattern.WIDE_OVERLOAD)
    ).astype(jnp.float32) * execution_phase.astype(jnp.float32)
    cross_base_value = jnp.clip(
        cross_base_value
        + config.attack_pattern_receiver_gain
        * wide_cross_pattern
        * jnp.square(cross_completion)
        * cross_centrality,
        0.0,
        1.0,
    )
    # Bias which PASS service is selected, but do not reuse that bias in the
    # later shot/pass/dribble/clear categorical decision.
    cross_value = jnp.clip(
        config.cross_value_gain * cross_base_value,
        0.0,
        1.0,
    )
    eligible_cross = onside_teammate & prepared_cross_candidate
    service_value = jnp.concatenate((pass_value, cross_value), axis=0)
    service_eligible = jnp.concatenate((eligible_pass, eligible_cross), axis=0)
    receiver_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _RECEIVER_RANDOM_STREAM)
    )
    best_service, has_pass = _masked_categorical(
        service_value,
        service_eligible,
        receiver_key,
        config.receiver_choice_temperature,
    )
    player_count = player_position.shape[0]
    selected_cross = best_service >= player_count
    best_pass = jnp.mod(best_service, player_count).astype(jnp.int32)
    # Score PASS from the service that would actually be executed.  Using the
    # family maximum here lets a mediocre sampled outlet borrow a different
    # candidate's value and creates low-purpose passes.  Keep the unboosted
    # base value so ``cross_value_gain`` chooses a service without inflating
    # PASS relative to SHOT/DRIBBLE/CLEAR.
    service_base_value = jnp.concatenate((pass_value, cross_base_value), axis=0)
    best_pass_value = jnp.where(
        has_pass,
        service_base_value[best_service],
        jnp.float32(-1.0),
    )
    service_completion = jnp.concatenate((pass_lane, cross_completion), axis=0)
    best_pass_lane = jnp.where(has_pass, service_completion[best_service], 0.0)
    selected_service_target = jnp.where(
        selected_cross, prepared_cross_target[best_pass], receiver_target[best_pass]
    )
    pass_direction = _safe_unit(selected_service_target - source)

    # A small fixed set of escape directions is cheaper to compile than a
    # runtime optimiser and still lets the carrier evade laterally or retreat.
    dribble_direction_candidates = _safe_unit(
        jnp.asarray(
            (
                (1.0, 0.0),
                (1.0, 1.0),
                (1.0, -1.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
                (-1.0, 0.0),
            ),
            dtype=jnp.float32,
        )
    )
    dribble_target = source + _DRIBBLE_STEP_M * dribble_direction_candidates
    dribble_target = jnp.stack(
        (
            jnp.clip(
                dribble_target[:, 0],
                -hx + _TARGET_BOUNDARY_MARGIN_M,
                hx - _TARGET_BOUNDARY_MARGIN_M,
            ),
            jnp.clip(
                dribble_target[:, 1],
                -hy + _TARGET_BOUNDARY_MARGIN_M,
                hy - _TARGET_BOUNDARY_MARGIN_M,
            ),
        ),
        axis=-1,
    )
    dribble_pressure = pressure(
        dribble_target,
        player_position,
        player_velocity,
        opponent,
        distance_scale_m=config.pressure_distance_m,
    )
    dribble_open = jnp.clip(
        openness(
            dribble_target,
            player_position,
            player_velocity,
            opponent,
            cap_m=_OPENNESS_CAP_M,
        )
        / _OPENNESS_CAP_M,
        0.0,
        1.0,
    )
    dribble_shot = shot_quality(
        dribble_target,
        player_position,
        opponent,
        opponent_goalkeeper,
        half_length=hx,
        goal_width=goal_width,
    )
    dribble_pitch = pitch_value(dribble_target, half_length=hx, half_width=hy)
    dribble_threat = jnp.maximum(dribble_shot, dribble_pitch)
    dribble_keep = jnp.clip(1.0 - 0.45 * current_pressure, 0.25, 1.0)
    dribble_value = jnp.clip(
        dribble_keep
        * (
            0.65 * dribble_threat
            + 0.20 * dribble_open
            + 0.15 * (1.0 - dribble_pressure)
        ),
        0.0,
        1.0,
    )
    progressive_pattern = attack_pattern == jnp.int32(AttackPattern.PROGRESSIVE_CARRY)
    progressive_lane = jnp.clip(dribble_direction_candidates[:, 0], 0.0, 1.0) * (
        1.0 - dribble_pressure
    )
    dribble_value = jnp.clip(
        dribble_value
        + config.attack_pattern_receiver_gain
        * progressive_pattern.astype(jnp.float32)
        * progressive_lane,
        0.0,
        1.0,
    )
    # Keep ordinary carries near the player's formation-side lane. This is a
    # soft common-currency cost, not a direction ban, and fades under pressure.
    shape_drift = jnp.clip(
        jnp.abs(dribble_target[:, 1] - formation_anchor_y) / hy,
        0.0,
        1.0,
    )
    dribble_value = jnp.clip(
        dribble_value
        - config.dribble_shape_drift_penalty
        * carry_fraction
        * (1.0 - current_pressure)
        * shape_drift,
        0.0,
        1.0,
    )
    best_dribble_value = jnp.max(dribble_value)
    safe_release = has_pass & (best_pass_lane >= _SAFE_PASS_COMPLETION)
    progressive_dribble_value = jnp.max(
        jnp.where(dribble_direction_candidates[:, 0] >= 0.5, dribble_value, -1.0)
    )
    high_quality_breakthrough = (
        progressive_dribble_value >= jnp.float32(_SAFE_PASS_COMPLETION)
    ) & (progressive_dribble_value > best_pass_value)
    # Episode age bridges momentary loose touches for the same solo carrier.
    # A preceding teammate prevents a receiver inheriting the whole build-up.
    release_urgency = (safe_release & (~high_quality_breakthrough)).astype(
        jnp.float32
    ) * jnp.clip(
        solo_tenure_seconds / config.solo_carry_soft_limit_s - 1.0,
        0.0,
        1.0,
    )
    best_dribble_value = (
        best_dribble_value
        * (
            1.0
            - config.solo_carry_value_decay
            * carry_fraction
            * safe_release.astype(jnp.float32)
        )
        * (1.0 - config.solo_carry_value_decay * release_urgency)
    )
    best_pass_value = jnp.clip(
        best_pass_value
        + config.solo_carry_value_decay * release_urgency * (1.0 - best_pass_value),
        0.0,
        1.0,
    )
    # Receiver and service selection above retain their completion-aware
    # ordering. Scale only the macro utility so abundant safe outlets do not
    # suppress every viable shot or carry in settled possession.
    best_pass_value = best_pass_value * jnp.float32(config.pass_macro_value_scale)
    dribble_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _DRIBBLE_DIRECTION_RANDOM_STREAM)
    )
    dribble_lane, _ = _masked_categorical(
        dribble_value,
        jnp.ones_like(dribble_value, dtype=jnp.bool_),
        dribble_key,
        config.receiver_choice_temperature,
    )
    dribble_direction = _safe_unit(dribble_target[dribble_lane] - source)

    shot_plan = plan_shot(
        source,
        player_position,
        opponent,
        opponent_goalkeeper,
        config,
        half_length=hx,
        goal_width=goal_width,
        current_pressure=current_pressure,
        decision_key=decision_key,
        shot_launch_radians_per_action_unit=launch_radians_per_unit,
    )
    # A newly won possession has no observed previous teammate. Discount only
    # low-quality shots during a short settle window; clear chances stay live.
    settled_fraction = jnp.clip(
        possession_episode_seconds / config.turnover_shot_settle_s,
        0.0,
        1.0,
    )
    fresh_unlinked_possession = (
        (~has_previous_actor)
        & (possession_episode_seconds < config.turnover_shot_settle_s)
        & (shot_plan.quality < jnp.float32(_SAFE_PASS_COMPLETION))
    )
    transition_shot_scale = (
        config.turnover_shot_value_scale
        + (1.0 - config.turnover_shot_value_scale) * settled_fraction
    )
    shot_macro_value = shot_plan.value * jnp.where(
        fresh_unlinked_possession, transition_shot_scale, 1.0
    )

    # Evaluate three downfield destinations and open clearance only in the
    # narrow deep/pressured/no-safe-outlet situation retained from the reference.
    clear_directions = _safe_unit(
        jnp.asarray(
            ((1.0, 0.0), (1.0, 0.72), (1.0, -0.72)),
            dtype=jnp.float32,
        )
    )
    raw_clear_target = source + _CLEARANCE_DISTANCE_M * clear_directions
    clear_lane_available = jnp.abs(raw_clear_target[:, 1]) <= hy - jnp.float32(
        _TARGET_BOUNDARY_MARGIN_M
    )
    clear_lane_available = clear_lane_available.at[0].set(True)
    clear_target = raw_clear_target
    clear_target = jnp.stack(
        (
            jnp.clip(
                clear_target[:, 0],
                -hx + _TARGET_BOUNDARY_MARGIN_M,
                hx - _TARGET_BOUNDARY_MARGIN_M,
            ),
            jnp.clip(
                clear_target[:, 1],
                -hy + _TARGET_BOUNDARY_MARGIN_M,
                hy - _TARGET_BOUNDARY_MARGIN_M,
            ),
        ),
        axis=-1,
    )
    clear_pressure = pressure(
        clear_target,
        player_position,
        player_velocity,
        opponent,
        distance_scale_m=config.pressure_distance_m,
    )
    clear_open = jnp.clip(
        openness(
            clear_target,
            player_position,
            player_velocity,
            opponent,
            cap_m=_OPENNESS_CAP_M,
        )
        / _OPENNESS_CAP_M,
        0.0,
        1.0,
    )
    clear_pitch = pitch_value(clear_target, half_length=hx, half_width=hy)
    clear_destination_value = jnp.clip(
        0.45 * clear_open + 0.35 * (1.0 - clear_pressure) + 0.20 * clear_pitch,
        0.0,
        1.0,
    )
    best_clear_value = jnp.max(
        jnp.where(clear_lane_available, clear_destination_value, -1.0)
    )
    clear_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _CLEAR_DIRECTION_RANDOM_STREAM)
    )
    clear_lane, _ = _masked_categorical(
        clear_destination_value,
        clear_lane_available,
        clear_key,
        config.receiver_choice_temperature,
    )
    clear_direction = _safe_unit(clear_target[clear_lane] - source)
    safe_outlet = has_pass & (best_pass_lane >= _SAFE_PASS_COMPLETION)
    clear_available = (
        (source[0] <= -hx * _CLEARANCE_DEPTH_FRACTION)
        & (current_pressure >= _CLEARANCE_PRESSURE)
        & (~safe_outlet)
    )
    clear_value = jnp.where(
        clear_available,
        jnp.clip(
            0.45 + 0.30 * current_pressure + 0.25 * best_clear_value,
            0.0,
            1.0,
        ),
        -1.0,
    )

    # A selected cross remains a PASS and retains the chosen teammate target.

    cross_side = -jnp.where(source[1] >= 0.0, 1.0, -1.0)
    cross_spin = jnp.stack(
        (
            cross_side * config.cross_side_spin,
            jnp.float32(config.cross_back_spin),
        )
    )
    # Utilities become relative categorical weights. Log-space temperature
    # preserves their ratios, keeping the smooth long-shot tail small.
    values = jnp.stack(
        (shot_macro_value, best_pass_value, best_dribble_value, clear_value)
    )
    macro_available = (values >= 0.0).at[POSSESSION_PASS].set(has_pass)
    fallback_kind = jnp.argmax(values).astype(jnp.int32)
    macro_key = (
        None
        if decision_key is None
        else jax.random.fold_in(decision_key, _MACRO_RANDOM_STREAM)
    )
    if macro_key is None:
        kind = fallback_kind
    else:
        macro_logits = (
            jnp.log(jnp.maximum(values, 1e-4)) / config.macro_choice_temperature
        )
        masked_logits = jnp.where(macro_available, macro_logits, -jnp.inf)
        kind = jax.random.categorical(
            macro_key, masked_logits.astype(jnp.float32)
        ).astype(jnp.int32)
    progressive_commit = (
        progressive_pattern
        & setup_phase
        & (possession_seconds < progressive_carry_commit_s)
        & (~safe_release)
        & (current_pressure < jnp.float32(_CLEARANCE_PRESSURE))
        & (shot_macro_value < jnp.float32(_SAFE_PASS_COMPLETION))
        & (~clear_available)
    )
    kind = jnp.where(progressive_commit, jnp.int32(POSSESSION_DRIBBLE), kind)
    release_due = (
        decision_due
        & safe_release
        & (solo_tenure_seconds >= jnp.float32(config.solo_carry_soft_limit_s))
    )
    release_kind = jnp.where(
        shot_plan.value > best_pass_value,
        jnp.int32(POSSESSION_SHOT),
        jnp.int32(POSSESSION_PASS),
    )
    kind = jnp.where(release_due, release_kind, kind)
    kind = jnp.where(decision_due, kind, jnp.int32(POSSESSION_DRIBBLE))
    direction = jnp.where(
        kind == POSSESSION_SHOT,
        shot_plan.direction,
        jnp.where(
            kind == POSSESSION_PASS,
            pass_direction,
            jnp.where(
                kind == POSSESSION_DRIBBLE,
                dribble_direction,
                clear_direction,
            ),
        ),
    ).astype(jnp.float32)
    power = jnp.where(
        kind == POSSESSION_SHOT,
        shot_plan.power,
        jnp.where(
            kind == POSSESSION_PASS,
            jnp.where(selected_cross, config.cross_power, config.pass_power),
            jnp.where(
                kind == POSSESSION_DRIBBLE,
                config.dribble_power,
                config.clear_power,
            ),
        ),
    ).astype(jnp.float32)
    launch = jnp.where(
        kind == POSSESSION_SHOT,
        shot_plan.launch,
        jnp.where(
            kind == POSSESSION_PASS,
            jnp.where(selected_cross, config.cross_launch, config.pass_launch),
            jnp.where(kind == POSSESSION_DRIBBLE, -1.0, config.clear_launch),
        ),
    ).astype(jnp.float32)
    cross_applied = (kind == POSSESSION_PASS) & selected_cross
    spin = jnp.where(
        kind == POSSESSION_SHOT,
        shot_plan.spin,
        jnp.where(cross_applied, cross_spin, jnp.zeros(2, dtype=jnp.float32)),
    ).astype(jnp.float32)

    target = jnp.where(has_pass, best_pass, jnp.int32(NO_PLAYER)).astype(jnp.int32)

    decision = PossessionDecision(
        direction=direction,
        power=power,
        launch=launch,
        kind=kind,
        target=target,
        spin=spin,
        cross=cross_applied,
    )
    if not with_candidate_trace:
        return decision

    receiver_slot = jnp.tile(jnp.arange(player_count, dtype=jnp.int32), 2)
    is_cross = jnp.concatenate(
        (
            jnp.zeros(player_count, dtype=jnp.bool_),
            jnp.ones(player_count, dtype=jnp.bool_),
        ),
        axis=0,
    )
    target_xy = jnp.concatenate((receiver_target, prepared_cross_target), axis=0)
    progression_contribution = jnp.concatenate(
        (
            pass_lane * pass_progression_gain,
            cross_completion
            * tactical.progressive_pass_gain
            * cross_progressive_preference,
        ),
        axis=0,
    )
    receiver_forward_velocity = jnp.concatenate(
        (player_velocity[:, 0], player_velocity[:, 0]), axis=0
    )
    return decision, PossessionCandidateTrace(
        carrier_slot=jnp.int32(NO_PLAYER),
        receiver_slot=receiver_slot,
        is_cross=is_cross,
        eligible=service_eligible,
        selected=(
            has_pass & (jnp.arange(2 * player_count, dtype=jnp.int32) == best_service)
        ),
        target_xy=target_xy.astype(jnp.float32),
        completion=service_completion.astype(jnp.float32),
        selection_value=service_value.astype(jnp.float32),
        macro_value=service_base_value.astype(jnp.float32),
        progression_contribution=progression_contribution.astype(jnp.float32),
        relative_depth_m=(target_xy[:, 0] - source[0]).astype(jnp.float32),
        receiver_forward_velocity_mps=receiver_forward_velocity.astype(jnp.float32),
        source_xy=source.astype(jnp.float32),
        decision_due=jnp.asarray(decision_due, dtype=jnp.bool_),
    )


__all__ = [
    "POSSESSION_CLEAR",
    "POSSESSION_DRIBBLE",
    "POSSESSION_PASS",
    "POSSESSION_SHOT",
    "PossessionCandidateTrace",
    "PossessionDecision",
    "ShotPlan",
    "decide_possession",
    "plan_shot",
]
