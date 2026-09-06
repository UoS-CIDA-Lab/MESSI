"""Observation-only restart decisions for the seeded stochastic rule policy.

The helpers in this module select a lawful restart target but do not request
physical contact.  The integrating policy remains responsible for combining
``RestartDecision.valid`` with the public contact/timing observations before
emitting a non-MOVE intent. All geometry is reconstructed from one observer row
of :class:`RulePolicyContext`; no rollout ``State`` or affordance view is used.

The prospective-offside mask belongs to the prospective kick being planned.
It is deliberately ignored only for the three Law 11 direct-receive exemptions
(goal kick, throw-in, and corner kick), and enforced for all other receiver
selections, including free kicks, offside restarts, goalkeeper releases, and
the generic kickoff path. Penalties select a seeded visible-goal lane. Direct
free kicks may choose a shot from observed distance, goal angle, and goalkeeper
position; indirect free kicks remain pass-only. If partial observation exposes
no receiver, a short release into visible field space preserves liveness.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    GEOMETRY_EPS,
    NO_PLAYER,
    RK_CORNER,
    RK_FREEKICK,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
)
from footballworld.environment.observation import Observation
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.context import RulePolicyContext


class RestartDecision(NamedTuple):
    """One restart taker's observation-only tactical decision.

    Array shapes are ``(observers, 2)`` for ``direction`` and ``(observers,)``
    for every other field.  ``direction`` is an unencoded vector in the
    observer team's attacking frame; ``power`` and ``launch`` already use the
    public normalized action convention.  ``receiver`` is a roster slot only
    when ``receiver_valid`` is true.

    ``valid`` means that the helper found a lawful tactical target.  It does
    not mean contact may occur during the current control frame.  Invalid rows
    are an exact fail-closed action intent: zero direction/power, launch -1,
    and ``NO_PLAYER``.
    """

    direction: jax.Array
    power: jax.Array
    launch: jax.Array
    receiver: jax.Array
    receiver_valid: jax.Array
    valid: jax.Array
    shot: jax.Array
    aerial_service: jax.Array
    law11_direct_exempt: jax.Array


def _validate_inputs(
    context: RulePolicyContext,
    observations: Observation,
    prospective_offside: jax.Array,
) -> tuple[int, int, jax.Array]:
    """Validate static observer/roster axes before JAX tracing."""

    if context.self_index.ndim != 1:
        raise ValueError("context must have one leading observer axis")
    observers = context.self_index.shape[0]
    if context.player_position.ndim != 3 or context.player_position.shape != (
        observers,
        context.player_position.shape[1],
        2,
    ):
        raise ValueError(
            "context.player_position must have shape (observers, players, 2)"
        )
    players = context.player_position.shape[1]
    if context.teammate.shape != (observers, players):
        raise ValueError("context.teammate must match observer and roster axes")
    if context.ball_position.shape != (observers, 3):
        raise ValueError("context.ball_position must have shape (observers, 3)")
    if observations.restart.kind.shape != (observers,):
        raise ValueError("restart observations must have an observer axis")
    if observations.restart.team.shape != (observers,):
        raise ValueError("restart team must have an observer axis")
    if observations.restart.indirect.shape != (observers,):
        raise ValueError("restart indirect flag must have an observer axis")
    if observations.players.restart_taker.shape != (observers, players):
        raise ValueError("restart taker flags must match observer and roster axes")
    mask = jnp.asarray(prospective_offside, dtype=jnp.bool_)
    if mask.shape != (observers, players):
        raise ValueError("prospective_offside must have shape (observers, players)")
    return observers, players, mask


def _masked_choice(
    score: jax.Array,
    candidate: jax.Array,
    key: jax.Array | None,
    temperature: float,
) -> tuple[jax.Array, jax.Array]:
    """Seeded fixed-shape categorical with deterministic no-key fallback."""

    value = jnp.asarray(score, dtype=jnp.float32)
    available = jnp.asarray(candidate, dtype=jnp.bool_)
    fallback = jnp.argmax(jnp.where(available, value, -jnp.inf), axis=-1)
    if key is None:
        selected = fallback
    else:
        logits = jnp.log(jnp.maximum(value, 1.0e-4)) / jnp.float32(temperature)
        selected = jnp.argmax(
            jnp.where(
                available, logits + jax.random.gumbel(key, value.shape), -jnp.inf
            ),
            axis=-1,
        )
    return selected.astype(jnp.int32), jnp.any(available, axis=-1)


def decide_restart(
    context: RulePolicyContext,
    observations: Observation,
    prospective_offside: jax.Array,
    config: RulePolicyConfig,
    *,
    decision_key: jax.Array | None = None,
    player_is_goalkeeper: jax.Array | None = None,
    ground_target_xy: jax.Array | None = None,
    ground_completion: jax.Array | None = None,
    ground_candidate: jax.Array | None = None,
    aerial_target_xy: jax.Array | None = None,
    aerial_completion: jax.Array | None = None,
    aerial_candidate: jax.Array | None = None,
    half_length: float = 52.5,
    half_width: float = 34.0,
    goal_width: float = 7.32,
) -> RestartDecision:
    """Choose a type-aware restart shot or receiver for every observer row.

    The caller supplies the prospective Law 11 mask plus optional rolling and
    aerial arrival evaluations for the planned kick. A candidate receiver must
    already be a visible, participating teammate in ``context.teammate``.
    Selection is seeded when ``decision_key`` is supplied and deterministic
    otherwise. Penalties and eligible direct free kicks choose among three
    visible-goal lanes; indirect restarts remain pass-only. Rows without a
    legal observable receiver use a bounded field-space release.
    """

    observers, _, prospective_offside = _validate_inputs(
        context, observations, prospective_offside
    )
    row = jnp.arange(observers, dtype=jnp.int32)
    self_index = jnp.asarray(context.self_index, dtype=jnp.int32)
    kind = jnp.asarray(observations.restart.kind, dtype=jnp.int32)

    restart_active = kind != jnp.int32(RK_NONE)
    own_restart = observations.restart.team == context.self_team
    self_is_taker = observations.players.restart_taker[row, self_index]
    eligible_actor = (
        restart_active
        & own_restart
        & context.self_active
        & context.ball_visible
        & self_is_taker
    )

    players = context.player_position.shape[1]
    expected_points = (observers, players, 2)
    expected_scores = (observers, players)
    ball_xy = context.ball_position[:, None, :2]
    receiver_delta = context.player_position - ball_xy
    receiver_distance = jnp.linalg.norm(receiver_delta, axis=-1)

    if ground_target_xy is None:
        ground_target = context.player_position
        ground_score = jnp.exp(
            -jnp.float32(config.pass_distance_penalty_per_m) * receiver_distance
        )
        supplied_ground_candidate = context.teammate
    else:
        if ground_completion is None or ground_candidate is None:
            raise ValueError("ground restart inputs must be supplied together")
        ground_target = jnp.asarray(ground_target_xy, dtype=jnp.float32)
        ground_score = jnp.asarray(ground_completion, dtype=jnp.float32)
        supplied_ground_candidate = jnp.asarray(ground_candidate, dtype=jnp.bool_)
        if ground_target.shape != expected_points:
            raise ValueError("ground_target_xy must have shape (observers, players, 2)")
        if ground_score.shape != expected_scores:
            raise ValueError("ground_completion must have shape (observers, players)")
        if supplied_ground_candidate.shape != expected_scores:
            raise ValueError("ground_candidate must have shape (observers, players)")

    if aerial_target_xy is None:
        aerial_target = ground_target
        aerial_score = jnp.zeros(expected_scores, dtype=jnp.float32)
        supplied_aerial_candidate = jnp.zeros(expected_scores, dtype=jnp.bool_)
    else:
        if aerial_completion is None or aerial_candidate is None:
            raise ValueError("aerial restart inputs must be supplied together")
        aerial_target = jnp.asarray(aerial_target_xy, dtype=jnp.float32)
        aerial_score = jnp.asarray(aerial_completion, dtype=jnp.float32)
        supplied_aerial_candidate = jnp.asarray(aerial_candidate, dtype=jnp.bool_)
        if aerial_target.shape != expected_points:
            raise ValueError("aerial_target_xy must have shape (observers, players, 2)")
        if aerial_score.shape != expected_scores:
            raise ValueError("aerial_completion must have shape (observers, players)")
        if supplied_aerial_candidate.shape != expected_scores:
            raise ValueError("aerial_candidate must have shape (observers, players)")

    goalkeeper = (
        jnp.zeros((players,), dtype=jnp.bool_)
        if player_is_goalkeeper is None
        else jnp.asarray(player_is_goalkeeper, dtype=jnp.bool_)
    )
    if goalkeeper.shape != (players,):
        raise ValueError("player_is_goalkeeper must have shape (players,)")

    penalty = kind == jnp.int32(RK_PENALTY)
    throw_in = kind == jnp.int32(RK_THROWIN)
    goal_kick = kind == jnp.int32(RK_GOALKICK)
    corner = kind == jnp.int32(RK_CORNER)
    free_kick = kind == jnp.int32(RK_FREEKICK)
    offside_restart = kind == jnp.int32(RK_OFFSIDE)
    kickoff = kind == jnp.int32(RK_KICKOFF)
    guarded_restart = kickoff | free_kick | offside_restart
    receiver_service = free_kick | offside_restart
    direct_free_kick = free_kick & (
        ~jnp.asarray(observations.restart.indirect, dtype=jnp.bool_)
    )
    direct_exempt = goal_kick | throw_in | corner
    legal_receiver = context.teammate & (
        direct_exempt[:, None] | (~prospective_offside)
    )
    # Kick-off, free-kick, and offside-restart takers are placed immediately
    # beside the stationary ball. A target on the taker's side sends the
    # released ball back through the taker's own solid body on the following
    # physics substep. Keep the environment expressive for arbitrary actions,
    # but make the built-in policy choose only targets that clear the taker's
    # body.
    taker_to_ball = context.ball_position[:, :2] - context.self_position
    taker_to_ball_norm = jnp.linalg.norm(taker_to_ball, axis=-1, keepdims=True)
    escape_axis = jnp.where(
        taker_to_ball_norm > jnp.float32(GEOMETRY_EPS),
        taker_to_ball / jnp.maximum(taker_to_ball_norm, jnp.float32(GEOMETRY_EPS)),
        jnp.asarray((1.0, 0.0), dtype=jnp.float32),
    )
    ground_clears_taker = jnp.sum(
        (ground_target - context.ball_position[:, None, :2]) * escape_axis[:, None, :],
        axis=-1,
    ) > jnp.float32(GEOMETRY_EPS)
    aerial_clears_taker = jnp.sum(
        (aerial_target - context.ball_position[:, None, :2]) * escape_axis[:, None, :],
        axis=-1,
    ) > jnp.float32(GEOMETRY_EPS)
    ground_candidate_mask = (
        legal_receiver
        & supplied_ground_candidate
        & ((~guarded_restart)[:, None] | ground_clears_taker)
    )
    aerial_candidate_mask = (
        legal_receiver
        & supplied_aerial_candidate
        & ((~guarded_restart)[:, None] | aerial_clears_taker)
    )
    short_throw_candidate = ground_candidate_mask & (
        receiver_distance <= jnp.float32(config.shoot_distance_m)
    )
    has_short_throw = jnp.any(short_throw_candidate, axis=-1)
    ground_candidate_mask = jnp.where(
        throw_in[:, None] & has_short_throw[:, None],
        short_throw_candidate,
        ground_candidate_mask,
    )

    # Corners seek central penalty-area runners; goal kicks seek a wider or
    # progressive long outlet. These are design priors over caller-supplied
    # physics-reachable targets, not fitted event probabilities.
    central_box = (aerial_target[:, :, 0] > jnp.float32(0.45 * half_length)) & (
        jnp.abs(aerial_target[:, :, 1])
        <= jnp.float32(config.cross_target_central_fraction * half_width)
    )
    long_goal_kick = (receiver_distance >= jnp.float32(config.shoot_distance_m)) & (
        (aerial_target[:, :, 0] - ball_xy[:, :, 0] > 0.0)
        | (jnp.abs(aerial_target[:, :, 1]) > jnp.float32(0.35 * half_width))
    )
    aerial_candidate_mask = aerial_candidate_mask & jnp.where(
        corner[:, None],
        central_box,
        jnp.where(
            goal_kick[:, None],
            long_goal_kick,
            receiver_service[:, None],
        ),
    )
    ground_value = jnp.clip(ground_score, 0.0, 1.0) * jnp.exp(
        -jnp.float32(config.pass_distance_penalty_per_m) * receiver_distance
    )
    aerial_width = jnp.clip(jnp.abs(aerial_target[:, :, 1]) / half_width, 0.0, 1.0)
    aerial_progress = jnp.clip(
        (aerial_target[:, :, 0] - ball_xy[:, :, 0]) / (2.0 * half_length), 0.0, 1.0
    )
    aerial_value = jnp.clip(aerial_score, 0.0, 1.0) * (
        0.70 + 0.15 * aerial_width + 0.15 * aerial_progress
    )
    ground_key = (
        None if decision_key is None else jax.random.fold_in(decision_key, 0x47524E44)
    )
    aerial_key = (
        None if decision_key is None else jax.random.fold_in(decision_key, 0x4145524C)
    )
    ground_index, has_ground = _masked_choice(
        ground_value,
        ground_candidate_mask,
        ground_key,
        config.receiver_choice_temperature,
    )
    aerial_index, has_aerial = _masked_choice(
        aerial_value,
        aerial_candidate_mask,
        aerial_key,
        config.receiver_choice_temperature,
    )

    service_value = jnp.concatenate((ground_value, aerial_value), axis=-1)
    service_candidate = jnp.concatenate(
        (ground_candidate_mask, aerial_candidate_mask), axis=-1
    )
    service_key = (
        None if decision_key is None else jax.random.fold_in(decision_key, 0x474B4943)
    )
    service_index, has_service = _masked_choice(
        service_value,
        service_candidate,
        service_key,
        config.receiver_choice_temperature,
    )
    service_aerial = service_index >= players
    service_receiver = jnp.mod(service_index, players).astype(jnp.int32)

    corner_aerial = corner & has_aerial
    combined_service = goal_kick | receiver_service
    use_aerial = corner_aerial | (combined_service & has_service & service_aerial)
    receiver_index = jnp.where(
        combined_service,
        service_receiver,
        jnp.where(corner_aerial, aerial_index, ground_index),
    ).astype(jnp.int32)
    has_receiver = jnp.where(
        combined_service,
        has_service,
        jnp.where(corner, has_aerial | has_ground, has_ground),
    )
    selected_ground = ground_target[row, receiver_index]
    selected_aerial = aerial_target[row, receiver_index]
    selected_receiver = jnp.where(use_aerial[:, None], selected_aerial, selected_ground)
    # A view-limited taker may observe no teammate. A short attacking-frame
    # release into the field preserves restart liveness without consulting a
    # hidden player or turning a remote direct free kick into a forced shot.
    fallback_step = jnp.float32(config.pass_min_progress_m)
    fallback_forward_x = context.ball_position[:, 0] + fallback_step
    fallback_x = jnp.where(
        fallback_forward_x <= jnp.float32(half_length - 0.5),
        fallback_forward_x,
        context.ball_position[:, 0] - fallback_step,
    )
    bounded_fallback_target = jnp.stack(
        (
            jnp.clip(
                fallback_x,
                jnp.float32(-half_length + 0.5),
                jnp.float32(half_length - 0.5),
            ),
            jnp.clip(
                context.ball_position[:, 1],
                jnp.float32(-half_width + 0.5),
                jnp.float32(half_width - 0.5),
            ),
        ),
        axis=-1,
    )
    guarded_restart_fallback_target = (
        context.ball_position[:, :2] + fallback_step * escape_axis
    )
    fallback_target = jnp.where(
        guarded_restart[:, None],
        guarded_restart_fallback_target,
        bounded_fallback_target,
    )
    selected_position = jnp.where(
        has_receiver[:, None], selected_receiver, fallback_target
    )

    # Use a three-lane target set for direct restarts. The seeded choice is
    # weighted away from a visible goalkeeper; without one, all lanes are
    # equiprobable. These lane weights are explicit design priors.
    goal_y = jnp.float32(goal_width) * jnp.asarray((-0.30, 0.0, 0.30), jnp.float32)
    goal_y = jnp.broadcast_to(goal_y, (observers, 3))
    opponent_goalkeeper = context.opponent & goalkeeper[None, :]
    goalkeeper_visible = jnp.any(opponent_goalkeeper, axis=-1)
    goalkeeper_y = jnp.sum(
        jnp.where(opponent_goalkeeper, context.player_position[:, :, 1], 0.0), axis=-1
    ) / jnp.maximum(jnp.sum(opponent_goalkeeper, axis=-1), 1)
    lane_score = jnp.where(
        goalkeeper_visible[:, None],
        0.25
        + 0.75
        * jnp.clip(jnp.abs(goal_y - goalkeeper_y[:, None]) / goal_width, 0.0, 1.0),
        1.0,
    )
    lane_key = (
        None if decision_key is None else jax.random.fold_in(decision_key, 0x474F414C)
    )
    lane_index, _ = _masked_choice(
        lane_score,
        jnp.ones_like(lane_score, dtype=jnp.bool_),
        lane_key,
        config.shot_portion_temperature,
    )
    selected_goal_y = goal_y[row, lane_index]
    goal_target = jnp.stack(
        (jnp.full((observers,), jnp.float32(half_length)), selected_goal_y), axis=-1
    )
    goal_vector = goal_target - context.ball_position[:, :2]
    goal_distance = jnp.linalg.norm(goal_vector, axis=-1)
    x_to_goal = jnp.maximum(
        jnp.float32(half_length) - context.ball_position[:, 0], 1.0e-3
    )
    upper_angle = jnp.arctan2(
        jnp.float32(0.5 * goal_width) - context.ball_position[:, 1], x_to_goal
    )
    lower_angle = jnp.arctan2(
        -jnp.float32(0.5 * goal_width) - context.ball_position[:, 1], x_to_goal
    )
    aperture = jnp.abs(upper_angle - lower_angle)
    reference_aperture = 2.0 * jnp.arctan2(
        jnp.float32(0.5 * goal_width), jnp.float32(config.shoot_distance_m)
    )
    angle_quality = jnp.clip(aperture / reference_aperture, 0.0, 1.0)
    distance_quality = jax.nn.sigmoid(
        (jnp.float32(config.shoot_distance_m) - goal_distance)
        / jnp.float32(0.25 * config.shoot_distance_m)
    )
    goalkeeper_factor = jnp.where(
        goalkeeper_visible,
        0.85 + 0.15 * jnp.clip(jnp.abs(goalkeeper_y) / (0.5 * goal_width), 0.0, 1.0),
        1.0,
    )
    shot_probability = jnp.clip(
        distance_quality * (0.35 + 0.65 * angle_quality) * goalkeeper_factor,
        0.0,
        0.90,
    )
    shot_key = (
        None if decision_key is None else jax.random.fold_in(decision_key, 0x53484F54)
    )
    shot_draw = (
        jnp.full((observers,), 0.5, dtype=jnp.float32)
        if shot_key is None
        else jax.random.uniform(shot_key, shape=(observers,))
    )
    goal_clears_taker = jnp.sum(goal_vector * escape_axis, axis=-1) > jnp.float32(
        GEOMETRY_EPS
    )
    restart_shot = eligible_actor & (
        penalty
        | (
            direct_free_kick
            & goal_clears_taker
            & has_receiver
            & (shot_draw < shot_probability)
        )
    )
    pass_release = eligible_actor & (~restart_shot)
    receiver_valid = pass_release & has_receiver
    valid = restart_shot | pass_release
    pass_direction = selected_position - context.ball_position[:, :2]
    direction = jnp.where(
        restart_shot[:, None],
        goal_vector,
        jnp.where(pass_release[:, None], pass_direction, 0.0),
    ).astype(jnp.float32)
    power = jnp.where(
        restart_shot,
        jnp.float32(config.shoot_power),
        jnp.where(pass_release, jnp.float32(config.restart_power), jnp.float32(0.0)),
    ).astype(jnp.float32)
    launch = jnp.where(
        restart_shot,
        jnp.float32(config.shoot_launch),
        jnp.where(pass_release, jnp.float32(config.restart_launch), jnp.float32(-1.0)),
    ).astype(jnp.float32)
    receiver = jnp.where(receiver_valid, receiver_index, jnp.int32(NO_PLAYER)).astype(
        jnp.int32
    )

    # ``observations.restart.indirect`` is intentionally consumed as a public
    # contract assertion: an indirect row can release a pass, never a shot.
    indirect = jnp.asarray(observations.restart.indirect, dtype=jnp.bool_)
    valid = valid & ((~indirect) | (~restart_shot))
    direction = jnp.where(valid[:, None], direction, 0.0).astype(jnp.float32)
    power = jnp.where(valid, power, jnp.float32(0.0)).astype(jnp.float32)
    launch = jnp.where(valid, launch, jnp.float32(-1.0)).astype(jnp.float32)
    receiver_valid = receiver_valid & valid
    receiver = jnp.where(receiver_valid, receiver, jnp.int32(NO_PLAYER)).astype(
        jnp.int32
    )
    restart_shot = restart_shot & valid

    return RestartDecision(
        direction=direction,
        power=power,
        launch=launch,
        receiver=receiver,
        receiver_valid=receiver_valid,
        valid=valid,
        shot=restart_shot,
        aerial_service=use_aerial & receiver_valid,
        law11_direct_exempt=direct_exempt & eligible_actor,
    )


__all__ = ["RestartDecision", "decide_restart"]
