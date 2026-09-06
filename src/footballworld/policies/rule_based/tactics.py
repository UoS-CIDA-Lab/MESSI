"""Compact tactical metrics for one observer's public observation row.

All coordinates use the observer's attacking frame: ``+x`` points towards the
opponent goal and positions are measured in metres.  These functions consume
only one observer's visible player row.  They do not read environment state,
infer hidden slots, or solve a ball trajectory at policy runtime.

The defaults are bounded transfer or design priors, not fitted probabilities.
In particular, :func:`shot_quality` is a ranking score and must not be
reported as expected goals.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

_EPS = 1.0e-6

# Pass-decision priors are deterministic ranking and arrival-model choices,
# not fitted
# probabilities: 7 m/s controlled arrival, 0.18 s defender reaction, and
# 5.5 m/s^2 post-reaction acceleration.  The rolling tables and contact limits
# below are the public BallPhysics/ActionScale defaults. Callers
# using a non-default environment must pass that environment's values.
_PASS_ARRIVAL_SPEED_MPS = 7.0
_PASS_REACTION_TIME_S = 0.18
_PASS_DEFENDER_ACCELERATION_MPS2 = 5.5
_PLAYER_ACCELERATION_MPS2 = 7.95
_PLAYER_MAX_SPEED_MPS = 8.0
_KICK_SPEED_MAX_MPS = 34.76
_BALL_RADIUS_M = 0.11
_LAUNCH_MAX_RADIANS = 1.0
_GROUND_LAUNCH_DOWN_MAX_RADIANS = 0.21
_GROUND_LAUNCH_DOWN_REFERENCE_HEIGHT_M = 1.0
_ROLL_SPEED_KNOTS_MPS = (
    0.0,
    2.0,
    4.0,
    6.0,
    9.0,
    12.0,
    16.0,
    20.0,
    26.0,
    40.0,
)
_ROLL_DECELERATION_KNOTS_MPS2 = (
    0.70,
    0.95,
    1.09,
    1.70,
    6.04,
    7.85,
    9.72,
    13.29,
    17.39,
    26.96,
)


class ReceptionEvaluation(NamedTuple):
    """Receiver arrival and opponent-contest terms for pass candidates."""

    completion: jnp.ndarray
    receiver_eta_s: jnp.ndarray
    nearest_opponent_eta_s: jnp.ndarray


class GroundPassControls(NamedTuple):
    """Physics-aware normalized action controls and their arrival estimate."""

    direction: jnp.ndarray
    power: jnp.ndarray
    launch: jnp.ndarray
    distance_m: jnp.ndarray
    launch_speed_mps: jnp.ndarray
    arrival_speed_mps: jnp.ndarray
    average_speed_mps: jnp.ndarray
    arrival_time_s: jnp.ndarray
    reachable: jnp.ndarray


def _sigmoid(value):
    """Return a numerically bounded logistic transform."""

    value = jnp.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + jnp.exp(-value))


def _safe_norm(value, *, axis=-1):
    """Euclidean norm with a finite derivative at the origin."""

    return jnp.sqrt(jnp.sum(value * value, axis=axis) + _EPS * _EPS)


def _candidate_points(name, value):
    """Normalize one point or a fixed candidate list to ``(K, 2)``."""

    value = jnp.asarray(value, dtype=jnp.float32)
    if value.shape == (2,):
        return value[None, :], True
    if value.ndim == 2 and value.shape[1] == 2:
        return value, False
    raise ValueError(f"{name} must have shape (2,) or (K, 2), got {value.shape}")


def _visible_players(position, velocity, mask):
    """Validate and normalize one observer's visible-player arrays."""

    position = jnp.asarray(position, dtype=jnp.float32)
    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(
            f"player position must have shape (P, 2), got {position.shape}"
        )
    if velocity.shape != position.shape:
        raise ValueError(
            "player velocity must match position shape, "
            f"got {velocity.shape} and {position.shape}"
        )
    if mask.shape != position.shape[:1]:
        raise ValueError(
            f"player mask must have shape (P,), got {mask.shape} for {position.shape}"
        )
    return position, velocity, mask


def observable_offside_line(
    opponent_position,
    opponent_velocity,
    opponent_visible,
    ball_position,
    ball_velocity,
    *,
    strike_delay_s=0.0,
):
    """Return the observable Law 11 line at the anticipated kick instant.

    The line is the more advanced of the ball and second-last visible opponent
    in the observer's attacking frame.  With fewer than two visible opponents,
    the ball is the conservative line: an ahead receiver is rejected instead
    of being guessed onside from hidden positions.
    """

    opponents, velocity, visible = _visible_players(
        opponent_position, opponent_velocity, opponent_visible
    )
    ball_position = jnp.asarray(ball_position, dtype=jnp.float32)
    ball_velocity = jnp.asarray(ball_velocity, dtype=jnp.float32)
    if ball_position.shape not in {(2,), (3,)}:
        raise ValueError(
            f"ball_position must have shape (2,) or (3,), got {ball_position.shape}"
        )
    if ball_velocity.shape != ball_position.shape:
        raise ValueError("ball_velocity must match ball_position")
    delay = jnp.clip(jnp.asarray(strike_delay_s, jnp.float32), 0.0, 1.0)
    future_x = opponents[:, 0] + velocity[:, 0] * delay
    first = jnp.argmax(jnp.where(visible, future_x, -jnp.inf))
    index = jnp.arange(opponents.shape[0], dtype=jnp.int32)
    second = jnp.max(jnp.where(visible & (index != first), future_x, -jnp.inf))
    ball_x = ball_position[0] + ball_velocity[0] * delay
    enough_defenders = jnp.sum(visible.astype(jnp.int32)) >= 2
    defender_line = jnp.where(enough_defenders, second, -jnp.inf)
    return jnp.maximum(ball_x, defender_line).astype(jnp.float32)


def prospective_offside(
    receiver_position,
    receiver_velocity,
    receiver_visible,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    ball_position,
    ball_velocity,
    *,
    strike_delay_s=0.0,
    tolerance_m=0.10,
):
    """Flag receivers projected offside at this prospective pass contact.

    This recomputes position from the current observation and deliberately
    does not consume players.offside, which is a latch from an earlier touch.
    Law 11 is tested at the kick instant, not at the later arrival target, so a
    currently onside runner may legally be led beyond the line.  Centre
    coordinates are used because playable-body front support is not fully
    observable; tolerance_m is a small body-orientation uncertainty allowance.
    """

    receivers, velocity, visible = _visible_players(
        receiver_position, receiver_velocity, receiver_visible
    )
    delay = jnp.clip(jnp.asarray(strike_delay_s, jnp.float32), 0.0, 1.0)
    line = observable_offside_line(
        opponent_position,
        opponent_velocity,
        opponent_visible,
        ball_position,
        ball_velocity,
        strike_delay_s=delay,
    )
    future_x = receivers[:, 0] + velocity[:, 0] * delay
    tolerance = jnp.maximum(jnp.asarray(tolerance_m, jnp.float32), 0.0)
    return visible & (future_x > 0.0) & (future_x > line + tolerance)


def _earliest_arrival_eta(
    distance,
    initial_speed_toward_target,
    maximum_speed,
    acceleration,
):
    """Minimum constant-heading ETA under a bounded acceleration envelope."""

    distance = jnp.maximum(jnp.asarray(distance, jnp.float32), 0.0)
    maximum_speed = jnp.maximum(jnp.asarray(maximum_speed, jnp.float32), _EPS)
    acceleration = jnp.maximum(jnp.asarray(acceleration, jnp.float32), _EPS)
    initial = jnp.clip(
        jnp.asarray(initial_speed_toward_target, jnp.float32),
        0.0,
        maximum_speed,
    )
    acceleration_time = (maximum_speed - initial) / acceleration
    acceleration_distance = (
        initial * acceleration_time
        + 0.5 * acceleration * acceleration_time * acceleration_time
    )
    within_acceleration = (
        -initial + jnp.sqrt(initial * initial + 2.0 * acceleration * distance)
    ) / acceleration
    after_acceleration = (
        acceleration_time + (distance - acceleration_distance) / maximum_speed
    )
    return jnp.where(
        distance <= acceleration_distance,
        within_acceleration,
        after_acceleration,
    )


def reception_evaluation(
    target,
    receiver_position,
    receiver_velocity,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    ball_arrival_time_s,
    *,
    receiver_max_speed_mps=_PLAYER_MAX_SPEED_MPS,
    opponent_max_speed_mps=_PLAYER_MAX_SPEED_MPS,
    receiver_acceleration_mps2=_PLAYER_ACCELERATION_MPS2,
    opponent_acceleration_mps2=_PASS_DEFENDER_ACCELERATION_MPS2,
    opponent_reaction_time_s=_PASS_REACTION_TIME_S,
    arrival_slack_s=0.18,
    arrival_softness_s=0.35,
    contest_softness_s=0.48,
):
    """Evaluate whether the intended receiver reaches the pass before cover.

    Receiver arrays correspond one-to-one with target candidates.  Opponents
    are a shared visible roster row.  Current velocity toward each target,
    public maximum speed, acceleration, and defender reaction time all enter
    the ETA.  Completion is a tactical score, not a calibrated probability.
    """

    targets, single = _candidate_points("target", target)
    receivers = jnp.asarray(receiver_position, dtype=jnp.float32)
    receiver_velocity = jnp.asarray(receiver_velocity, dtype=jnp.float32)
    if receivers.shape == (2,):
        receivers = receivers[None, :]
    if receiver_velocity.shape == (2,):
        receiver_velocity = receiver_velocity[None, :]
    if receivers.shape != targets.shape or receiver_velocity.shape != targets.shape:
        raise ValueError(
            "receiver_position and receiver_velocity must match target shape"
        )
    opponents, opponent_velocity, opponent_visible = _visible_players(
        opponent_position, opponent_velocity, opponent_visible
    )

    receiver_delta = targets - receivers
    receiver_distance = _safe_norm(receiver_delta)
    receiver_direction = receiver_delta / receiver_distance[:, None]
    receiver_initial = jnp.maximum(
        jnp.sum(receiver_velocity * receiver_direction, axis=-1), 0.0
    )
    receiver_eta = _earliest_arrival_eta(
        receiver_distance,
        receiver_initial,
        receiver_max_speed_mps,
        receiver_acceleration_mps2,
    )

    opponent_delta = targets[:, None, :] - opponents[None, :, :]
    opponent_distance = _safe_norm(opponent_delta)
    opponent_direction = opponent_delta / opponent_distance[:, :, None]
    opponent_initial = jnp.maximum(
        jnp.sum(
            opponent_velocity[None, :, :] * opponent_direction,
            axis=-1,
        ),
        0.0,
    )
    reaction = jnp.maximum(jnp.asarray(opponent_reaction_time_s, jnp.float32), 0.0)
    distance_after_reaction = jnp.maximum(
        opponent_distance - opponent_initial * reaction, 0.0
    )
    opponent_eta = reaction + _earliest_arrival_eta(
        distance_after_reaction,
        opponent_initial,
        jnp.asarray(opponent_max_speed_mps, jnp.float32),
        opponent_acceleration_mps2,
    )
    nearest_opponent_eta = jnp.min(
        jnp.where(opponent_visible[None, :], opponent_eta, jnp.inf),
        axis=-1,
    )

    ball_arrival = jnp.asarray(ball_arrival_time_s, jnp.float32)
    if ball_arrival.ndim == 0:
        ball_arrival = jnp.broadcast_to(ball_arrival, receiver_eta.shape)
    if ball_arrival.shape != receiver_eta.shape:
        raise ValueError("ball_arrival_time_s must be scalar or shape (K,)")
    arrival_softness = jnp.maximum(jnp.asarray(arrival_softness_s, jnp.float32), 0.05)
    contest_softness = jnp.maximum(jnp.asarray(contest_softness_s, jnp.float32), 0.05)
    arrival = _sigmoid(
        (ball_arrival + arrival_slack_s - receiver_eta) / arrival_softness
    )
    contest = _sigmoid((nearest_opponent_eta - receiver_eta) / contest_softness)
    completion = arrival * (0.25 + 0.75 * contest)
    result = ReceptionEvaluation(
        completion=completion.astype(jnp.float32),
        receiver_eta_s=receiver_eta.astype(jnp.float32),
        nearest_opponent_eta_s=nearest_opponent_eta.astype(jnp.float32),
    )
    if single:
        return ReceptionEvaluation(*(value[0] for value in result))
    return result


def _rolling_deceleration(speed, speed_knots, deceleration_knots):
    return jnp.interp(
        jnp.maximum(speed, 0.0),
        speed_knots,
        deceleration_knots,
    )


def ground_pass_controls(
    source,
    target,
    incoming_ball_velocity,
    *,
    ball_height_m=_BALL_RADIUS_M,
    desired_arrival_speed_mps=_PASS_ARRIVAL_SPEED_MPS,
    kick_speed_max_mps=_KICK_SPEED_MAX_MPS,
    ball_radius_m=_BALL_RADIUS_M,
    launch_max_radians=_LAUNCH_MAX_RADIANS,
    ground_launch_down_max_radians=_GROUND_LAUNCH_DOWN_MAX_RADIANS,
    ground_launch_down_reference_height_m=(_GROUND_LAUNCH_DOWN_REFERENCE_HEIGHT_M),
    roll_speed_knots_mps=_ROLL_SPEED_KNOTS_MPS,
    roll_deceleration_knots_mps2=_ROLL_DECELERATION_KNOTS_MPS2,
    distance_lookup_m=None,
    launch_speed_lookup_mps=None,
    travel_time_lookup_s=None,
):
    """Choose ground-pass direction, power, and launch from public ball physics.

    The FootballWorld rolling-deceleration table is inverted with a fixed
    eight-step work-energy iteration to find the release speed that reaches
    the target near desired_arrival_speed_mps.  The requested impulse is then
    corrected for the observed incoming three-dimensional ball velocity,
    matching contact dynamics' additive impulse convention.  The physical
    impulse angle is mapped through the public height-dependent launch
    envelope.  No environment state or runtime trajectory solver is read.
    """

    source = jnp.asarray(source, dtype=jnp.float32)
    if source.shape != (2,):
        raise ValueError(f"source must have shape (2,), got {source.shape}")
    targets, single = _candidate_points("target", target)
    incoming = jnp.asarray(incoming_ball_velocity, dtype=jnp.float32)
    if incoming.shape == (2,):
        incoming = jnp.concatenate((incoming, jnp.zeros((1,), dtype=jnp.float32)))
    if incoming.shape != (3,):
        raise ValueError(
            f"incoming_ball_velocity must have shape (2,) or (3,), got {incoming.shape}"
        )
    speed_knots = jnp.asarray(roll_speed_knots_mps, dtype=jnp.float32)
    deceleration_knots = jnp.asarray(roll_deceleration_knots_mps2, dtype=jnp.float32)
    if (
        speed_knots.ndim != 1
        or deceleration_knots.shape != speed_knots.shape
        or speed_knots.shape[0] < 2
    ):
        raise ValueError(
            "rolling speed/deceleration knots must share one axis of length >= 2"
        )

    delta = targets - source
    distance = _safe_norm(delta)
    target_direction = delta / distance[:, None]
    desired_arrival = jnp.maximum(
        jnp.asarray(desired_arrival_speed_mps, jnp.float32), 0.0
    )
    use_lookup = distance_lookup_m is not None
    if use_lookup:
        lookup_distance = jnp.asarray(distance_lookup_m, jnp.float32)
        lookup_speed = jnp.asarray(launch_speed_lookup_mps, jnp.float32)
        lookup_time = jnp.asarray(travel_time_lookup_s, jnp.float32)
        if (
            lookup_distance.ndim != 1
            or lookup_speed.shape != lookup_distance.shape
            or lookup_time.shape != lookup_distance.shape
            or lookup_distance.shape[0] < 2
        ):
            raise ValueError(
                "ground-pass lookup arrays must share one axis of length >= 2"
            )
        required_release = jnp.interp(distance, lookup_distance, lookup_speed)
        lookup_in_range = distance <= lookup_distance[-1]
    else:
        required_release = jnp.sqrt(
            desired_arrival * desired_arrival
            + 2.0
            * _rolling_deceleration(desired_arrival, speed_knots, deceleration_knots)
            * distance
        )

        # A loop primitive keeps the fallback solve compact in compiled HLO.
        def refine_required(_, estimate):
            mean_speed = 0.5 * (estimate + desired_arrival)
            next_release = jnp.sqrt(
                desired_arrival * desired_arrival
                + 2.0
                * _rolling_deceleration(mean_speed, speed_knots, deceleration_knots)
                * distance
            )
            return 0.5 * (estimate + next_release)

        required_release = jax.lax.fori_loop(0, 8, refine_required, required_release)
        lookup_in_range = jnp.ones_like(distance, dtype=jnp.bool_)

    desired_velocity = jnp.concatenate(
        (
            required_release[:, None] * target_direction,
            jnp.zeros((targets.shape[0], 1), dtype=jnp.float32),
        ),
        axis=-1,
    )
    requested_impulse = desired_velocity - incoming[None, :]
    requested_impulse_norm = _safe_norm(requested_impulse)
    kick_limit = jnp.maximum(jnp.asarray(kick_speed_max_mps, jnp.float32), _EPS)
    impulse_direction = requested_impulse / requested_impulse_norm[:, None]
    applied_impulse = (
        jnp.minimum(requested_impulse_norm, kick_limit)[:, None] * impulse_direction
    )
    candidate_velocity = incoming[None, :] + applied_impulse
    candidate_speed = _safe_norm(candidate_velocity)
    release_limit = jnp.maximum(_safe_norm(incoming), kick_limit)
    actual_velocity = (
        candidate_velocity * jnp.minimum(1.0, release_limit / candidate_speed)[:, None]
    )
    actual_release_speed = _safe_norm(actual_velocity[:, :2])

    if use_lookup:
        speed_fraction = jnp.clip(
            actual_release_speed / jnp.maximum(required_release, _EPS),
            0.0,
            1.0,
        )
        predicted_arrival = desired_arrival * speed_fraction
        nominal_time = jnp.interp(distance, lookup_distance, lookup_time)
        arrival_time = nominal_time / jnp.maximum(speed_fraction, 0.1)
        average_speed = jnp.where(arrival_time > _EPS, distance / arrival_time, 0.0)
    else:
        predicted_arrival = jnp.minimum(desired_arrival, actual_release_speed)

        def refine_arrival(_, estimate):
            mean_speed = 0.5 * (actual_release_speed + estimate)
            next_arrival = jnp.sqrt(
                jnp.maximum(
                    actual_release_speed * actual_release_speed
                    - 2.0
                    * _rolling_deceleration(mean_speed, speed_knots, deceleration_knots)
                    * distance,
                    0.0,
                )
            )
            return 0.5 * (estimate + next_arrival)

        predicted_arrival = jax.lax.fori_loop(0, 8, refine_arrival, predicted_arrival)
        average_speed = jnp.maximum(
            0.5 * (actual_release_speed + predicted_arrival), _EPS
        )
        arrival_time = jnp.where(distance > 1.0e-3, distance / average_speed, 0.0)

    horizontal_impulse = _safe_norm(requested_impulse[:, :2])
    direction = jnp.where(
        (horizontal_impulse > _EPS)[:, None],
        requested_impulse[:, :2] / horizontal_impulse[:, None],
        target_direction,
    )
    physical_launch = jnp.arctan2(requested_impulse[:, 2], horizontal_impulse)
    ball_height = jnp.asarray(ball_height_m, jnp.float32)
    radius = jnp.asarray(ball_radius_m, jnp.float32)
    reference_height = jnp.maximum(
        jnp.asarray(ground_launch_down_reference_height_m, jnp.float32),
        radius + _EPS,
    )
    height_fraction = jnp.clip(
        (ball_height - radius) / (reference_height - radius), 0.0, 1.0
    )
    maximum_launch = jnp.asarray(launch_max_radians, jnp.float32)
    ground_down = jnp.asarray(ground_launch_down_max_radians, jnp.float32)
    launch_floor = -(ground_down + (maximum_launch - ground_down) * height_fraction)
    launch_fraction = jnp.clip(
        (physical_launch - launch_floor)
        / jnp.maximum(maximum_launch - launch_floor, _EPS),
        0.0,
        1.0,
    )
    signed_launch = 2.0 * launch_fraction - 1.0
    reachable = (
        (distance > 1.0e-3)
        & lookup_in_range
        & (required_release <= release_limit + 1.0e-4)
        & (requested_impulse_norm <= kick_limit + 1.0e-4)
    )
    result = GroundPassControls(
        direction=direction.astype(jnp.float32),
        power=jnp.clip(requested_impulse_norm / kick_limit, 0.0, 1.0).astype(
            jnp.float32
        ),
        launch=signed_launch.astype(jnp.float32),
        distance_m=distance.astype(jnp.float32),
        launch_speed_mps=actual_release_speed.astype(jnp.float32),
        arrival_speed_mps=predicted_arrival.astype(jnp.float32),
        average_speed_mps=average_speed.astype(jnp.float32),
        arrival_time_s=arrival_time.astype(jnp.float32),
        reachable=reachable,
    )
    if single:
        return GroundPassControls(*(value[0] for value in result))
    return result


def pressure(
    query,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    *,
    lead_time_s=0.35,
    distance_scale_m=5.0,
):
    """Return bounded aggregate pressure at one point or candidate points.

    Visible opponents are advanced by ``lead_time_s`` before their distance is
    measured.  Individual contributions decay exponentially and the aggregate
    is folded into ``[0, 1)`` so its scale does not grow with roster size.
    """

    queries, single = _candidate_points("query", query)
    opponents, velocity, visible = _visible_players(
        opponent_position, opponent_velocity, opponent_visible
    )
    scale = jnp.maximum(jnp.asarray(distance_scale_m, jnp.float32), _EPS)
    lead = jnp.clip(jnp.asarray(lead_time_s, jnp.float32), 0.0, 2.0)
    future = opponents + velocity * lead
    distance = _safe_norm(queries[:, None, :] - future[None, :, :])
    raw = jnp.sum(
        jnp.where(visible[None, :], jnp.exp(-distance / scale), 0.0),
        axis=-1,
    )
    bounded = 1.0 - jnp.exp(-raw)
    return bounded[0] if single else bounded


def openness(
    query,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    *,
    lead_time_s=0.30,
    cap_m=15.0,
):
    """Return distance to the nearest anticipated visible opponent in metres."""

    queries, single = _candidate_points("query", query)
    opponents, velocity, visible = _visible_players(
        opponent_position, opponent_velocity, opponent_visible
    )
    lead = jnp.clip(jnp.asarray(lead_time_s, jnp.float32), 0.0, 2.0)
    cap = jnp.maximum(jnp.asarray(cap_m, jnp.float32), _EPS)
    future = opponents + velocity * lead
    distance = _safe_norm(queries[:, None, :] - future[None, :, :])
    nearest = jnp.min(jnp.where(visible[None, :], distance, jnp.inf), axis=-1)
    result = jnp.where(jnp.isfinite(nearest), jnp.minimum(nearest, cap), cap)
    return result[0] if single else result


def lane_completion(
    source,
    target,
    opponent_position,
    opponent_velocity,
    opponent_visible,
    *,
    ball_speed_mps=18.0,
    intercept_radius_m=1.25,
    softness_m=0.55,
    reaction_time_s=_PASS_REACTION_TIME_S,
    maximum_lead_time_s=1.25,
    opponent_max_speed_mps=None,
    defender_acceleration_mps2=_PASS_DEFENDER_ACCELERATION_MPS2,
    opponent_reduction_index=None,
    opponent_reduction_size=None,
):
    """Estimate ground-pass lane completion from defender reachability.

    Ball arrival time uses each candidate's physical average speed.  For every
    visible defender, only velocity toward the lane closes the perpendicular
    gap.  With public maximum speeds, a stationary defender can accelerate
    after reaction_time_s and total reach is capped by that maximum speed.
    A fixed-width padded opponent axis may be reduced into a smaller roster
    axis; out-of-domain padding indices contribute no blocking probability.
    The output is a tactical score rather than a calibrated probability.
    """

    source = jnp.asarray(source, dtype=jnp.float32)
    if source.shape != (2,):
        raise ValueError(f"source must have shape (2,), got {source.shape}")
    targets, single = _candidate_points("target", target)
    opponents, velocity, visible = _visible_players(
        opponent_position, opponent_velocity, opponent_visible
    )

    segment = targets - source
    length = _safe_norm(segment)
    direction = segment / length[:, None]
    relative = opponents[None, :, :] - source
    along = jnp.sum(relative * direction[:, None, :], axis=-1)
    clipped_along = jnp.clip(along, 0.0, length[:, None])
    lane_point = source + clipped_along[:, :, None] * direction[:, None, :]
    gap_vector = opponents[None, :, :] - lane_point
    perpendicular_gap = _safe_norm(gap_vector)
    perpendicular_direction = gap_vector / perpendicular_gap[:, :, None]
    closing_speed = jnp.maximum(
        -jnp.sum(
            velocity[None, :, :] * perpendicular_direction,
            axis=-1,
        ),
        0.0,
    )

    speed = jnp.asarray(ball_speed_mps, jnp.float32)
    if speed.ndim == 0:
        speed = jnp.broadcast_to(speed, (targets.shape[0],))
    if speed.shape == (targets.shape[0], 1):
        speed = speed[:, 0]
    if speed.shape != (targets.shape[0],):
        raise ValueError("ball_speed_mps must be scalar, shape (K,), or shape (K, 1)")
    speed = jnp.maximum(speed, 1.0)
    reaction = jnp.clip(jnp.asarray(reaction_time_s, jnp.float32), 0.0, 1.0)
    maximum_lead = jnp.clip(jnp.asarray(maximum_lead_time_s, jnp.float32), 0.0, 2.0)
    ball_arrival = clipped_along / speed[:, None]
    response_time = jnp.clip(ball_arrival - reaction, 0.0, maximum_lead)
    reachable = closing_speed * response_time
    if opponent_max_speed_mps is not None:
        maximum_speed = jnp.asarray(opponent_max_speed_mps, dtype=jnp.float32)
        if maximum_speed.ndim == 0:
            maximum_speed = jnp.broadcast_to(maximum_speed, (opponents.shape[0],))
        if maximum_speed.shape != (opponents.shape[0],):
            raise ValueError("opponent_max_speed_mps must be scalar or shape (P,)")
        accelerated_reach = (
            reachable
            + 0.5
            * jnp.maximum(
                jnp.asarray(defender_acceleration_mps2, jnp.float32),
                0.0,
            )
            * response_time
            * response_time
        )
        reachable = jnp.minimum(
            accelerated_reach,
            jnp.maximum(maximum_speed, 0.0)[None, :] * response_time,
        )
    anticipated_gap = perpendicular_gap - reachable

    between = (
        visible[None, :]
        & (along > 0.5)
        & (along < length[:, None] - 0.3)
        & (length[:, None] > 1.0)
    )
    radius = jnp.maximum(jnp.asarray(intercept_radius_m, jnp.float32), 0.0)
    softness = jnp.maximum(jnp.asarray(softness_m, jnp.float32), 0.05)
    block = _sigmoid((radius - anticipated_gap) / softness)
    block = jnp.where(between, jnp.clip(block, 0.0, 0.999), 0.0)
    if opponent_reduction_index is not None:
        reduction_index = jnp.asarray(opponent_reduction_index, dtype=jnp.int32)
        if reduction_index.shape != (opponents.shape[0],):
            raise ValueError("opponent_reduction_index must match the opponent axis")
        if (
            not isinstance(opponent_reduction_size, int)
            or isinstance(opponent_reduction_size, bool)
            or opponent_reduction_size < 1
        ):
            raise ValueError("opponent_reduction_size must be a positive integer")
        index_in_range = (reduction_index >= 0) & (
            reduction_index < opponent_reduction_size
        )
        safe_reduction_index = jnp.clip(
            reduction_index,
            0,
            opponent_reduction_size - 1,
        )
        block = jnp.where(index_in_range[None, :], block, 0.0)
        block = (
            jnp.zeros((targets.shape[0], opponent_reduction_size), dtype=block.dtype)
            .at[:, safe_reduction_index]
            .add(block)
        )
    elif opponent_reduction_size is not None:
        raise ValueError("opponent_reduction_size requires opponent_reduction_index")
    completion = jnp.exp(jnp.sum(jnp.log1p(-block), axis=-1))
    completion = jnp.where(length > 1.0, completion, 0.0)
    return completion[0] if single else completion


def moving_receiver_target(
    source,
    receiver_position,
    receiver_velocity,
    receiver_visible,
    *,
    half_length,
    half_width,
    ball_speed_mps=18.0,
    velocity_weight=0.72,
    lead_time_cap_s=0.90,
    lead_distance_cap_m=6.5,
    boundary_margin_m=1.0,
):
    """Aim at visible receivers' bounded arrival positions, not their feet.

    The lead time is the straight-line flight estimate capped at a short pass
    horizon.  The displacement follows observed receiver velocity and is
    bounded independently, preventing a malformed or very fast observation
    from producing a target outside the pitch.  Invisible slots remain at
    their supplied position and must still be excluded by the caller's mask.
    """

    source = jnp.asarray(source, dtype=jnp.float32)
    if source.shape != (2,):
        raise ValueError(f"source must have shape (2,), got {source.shape}")
    receivers, velocity, visible = _visible_players(
        receiver_position, receiver_velocity, receiver_visible
    )
    speed = jnp.maximum(jnp.asarray(ball_speed_mps, jnp.float32), 1.0)
    time_cap = jnp.clip(jnp.asarray(lead_time_cap_s, jnp.float32), 0.0, 2.0)
    weight = jnp.clip(jnp.asarray(velocity_weight, jnp.float32), 0.0, 1.5)
    lead_cap = jnp.maximum(jnp.asarray(lead_distance_cap_m, jnp.float32), 0.0)

    flight_time = jnp.clip(_safe_norm(receivers - source) / speed, 0.0, time_cap)
    displacement = velocity * (flight_time * weight)[:, None]
    displacement_norm = _safe_norm(displacement)
    displacement = (
        displacement * jnp.minimum(1.0, lead_cap / displacement_norm)[:, None]
    )
    target = receivers + displacement

    margin = jnp.maximum(jnp.asarray(boundary_margin_m, jnp.float32), 0.0)
    x_limit = jnp.maximum(jnp.asarray(half_length, jnp.float32) - margin, 0.0)
    y_limit = jnp.maximum(jnp.asarray(half_width, jnp.float32) - margin, 0.0)
    target = jnp.stack(
        (
            jnp.clip(target[:, 0], -x_limit, x_limit),
            jnp.clip(target[:, 1], -y_limit, y_limit),
        ),
        axis=-1,
    )
    return jnp.where(visible[:, None], target, receivers)


def pitch_value(position, *, half_length, half_width):
    """Return bounded static field-position value in the attacking frame."""

    points, single = _candidate_points("position", position)
    hx = jnp.maximum(jnp.asarray(half_length, jnp.float32), _EPS)
    hy = jnp.maximum(jnp.asarray(half_width, jnp.float32), _EPS)
    advance = jnp.clip((points[:, 0] + hx) / (2.0 * hx), 0.0, 1.0)
    centrality = 1.0 - jnp.clip(jnp.abs(points[:, 1]) / hy, 0.0, 1.0)
    base = jnp.power(advance, 1.35) * (0.72 + 0.28 * centrality)
    final_third = jnp.clip((advance - 0.72) / 0.28, 0.0, 1.0)
    value = jnp.clip(
        0.08 + 0.72 * base + 0.20 * final_third * (0.5 + 0.5 * centrality),
        0.0,
        1.0,
    )
    return value[0] if single else value


def shot_quality(
    source,
    opponent_position,
    opponent_visible,
    goalkeeper_mask,
    *,
    half_length,
    goal_width,
    intercept_width_m=0.65,
):
    """Rank a shot location by distance, goal angle, and visible obstruction.

    This deliberately returns a ``quality`` score rather than ``xG``: its
    bounded coefficients preserve useful ordering from the old policy but have
    not been fitted to FootballWorld rollouts or event data.
    """

    sources, single = _candidate_points("source", source)
    opponents = jnp.asarray(opponent_position, dtype=jnp.float32)
    visible = jnp.asarray(opponent_visible, dtype=jnp.bool_)
    goalkeeper = jnp.asarray(goalkeeper_mask, dtype=jnp.bool_)
    if opponents.ndim != 2 or opponents.shape[1] != 2:
        raise ValueError(
            f"opponent_position must have shape (P, 2), got {opponents.shape}"
        )
    if visible.shape != opponents.shape[:1] or goalkeeper.shape != visible.shape:
        raise ValueError("opponent_visible and goalkeeper_mask must have shape (P,)")

    hx = jnp.asarray(half_length, dtype=jnp.float32)
    width = jnp.maximum(jnp.asarray(goal_width, jnp.float32), _EPS)
    goal = jnp.stack((hx, jnp.asarray(0.0, jnp.float32)))
    upper_post = jnp.stack((hx, 0.5 * width))
    lower_post = jnp.stack((hx, -0.5 * width))
    to_goal = goal - sources
    distance = _safe_norm(to_goal)
    upper = upper_post - sources
    lower = lower_post - sources
    cross = upper[:, 0] * lower[:, 1] - upper[:, 1] * lower[:, 0]
    dot = jnp.sum(upper * lower, axis=-1)
    goal_angle = jnp.abs(jnp.arctan2(cross, dot))

    direction = to_goal / distance[:, None]
    relative = opponents[None, :, :] - sources[:, None, :]
    along = jnp.sum(relative * direction[:, None, :], axis=-1)
    lateral_vector = relative - along[:, :, None] * direction[:, None, :]
    lateral = _safe_norm(lateral_vector)
    cone_half_width = 0.5 * width * jnp.clip(along / distance[:, None], 0.0, 1.0)
    in_cone = (
        visible[None, :]
        & (along > 0.3)
        & (along < distance[:, None])
        & (lateral < cone_half_width + intercept_width_m)
    )
    near_weight = jnp.clip(1.0 - along / distance[:, None], 0.0, 1.0)
    outfield_obstruction = jnp.sum(
        jnp.where(in_cone & (~goalkeeper[None, :]), near_weight, 0.0),
        axis=-1,
    )
    goalkeeper_obstruction = jnp.sum(
        jnp.where(in_cone & goalkeeper[None, :], near_weight, 0.0),
        axis=-1,
    )
    logit = (
        1.6
        + 1.7 * goal_angle
        - 0.09 * distance
        - 1.3 * outfield_obstruction
        - 1.1 * goalkeeper_obstruction
    )
    quality = _sigmoid(logit)
    return quality[0] if single else quality


def deterministic_masked_argmax(score, mask):
    """Return ``(index, found)`` with stable lowest-slot tie breaking.

    Non-finite scores are ineligible.  If no candidate is valid, ``index`` is
    zero and ``found`` is false, allowing the caller to select an explicit
    fallback without an out-of-range gather.
    """

    score = jnp.asarray(score, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if score.ndim != 1 or mask.shape != score.shape:
        raise ValueError(
            f"score and mask must share shape (K,), got {score.shape} and {mask.shape}"
        )
    valid = mask & jnp.isfinite(score)
    found = jnp.any(valid)
    index = jnp.argmax(jnp.where(valid, score, -jnp.inf)).astype(jnp.int32)
    index = jnp.where(found, index, jnp.int32(0))
    return index, found


__all__ = [
    "GroundPassControls",
    "ReceptionEvaluation",
    "deterministic_masked_argmax",
    "ground_pass_controls",
    "lane_completion",
    "moving_receiver_target",
    "observable_offside_line",
    "openness",
    "pitch_value",
    "pressure",
    "prospective_offside",
    "reception_evaluation",
    "shot_quality",
]
