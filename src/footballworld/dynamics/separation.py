"""Vectorized separation of oriented player body capsules."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.player_physics import PlayerPhysics
from footballworld.core.constants import (
    COINCIDENT_DISTANCE_EPS,
    DIV_EPS,
    GEOMETRY_EPS,
    GOLDEN_ANGLE,
    NO_PLAYER,
    SQUARED_EPS,
)
from footballworld.core.state import PlayerState


class PlayerImpactFacts(NamedTuple):
    """Strongest cross-team body impact in one physics substep.

    ``contact_normal`` points from actor to victim. ``impact_score`` is the
    mass-free closing-speed proxy in metres per second; it is intentionally
    not presented as energy or force while players have no identified mass.
    ``time_fraction`` is the end of the locomotion microstep that exposed the
    impact, expressed on the enclosing physics interval. It is a conservative
    upper bound, not a continuous collision time of impact.
    ``velocity_alignment`` is the symmetric clipped cosine in ``[0, 1]``
    between the two players' collision-pre-response velocities. It preserves
    the impact-time shoulder-alignment input without carrying velocity vectors.
    ``actor_attribution_decisive`` distinguishes a kinematic attribution from
    the deterministic pair-order tie break used only to keep shapes static.
    """

    occurred: jax.Array
    actor: jax.Array
    victim: jax.Array
    contact_normal: jax.Array
    contact_position: jax.Array
    actor_attribution_decisive: jax.Array
    impact_score: jax.Array
    velocity_alignment: jax.Array
    time_fraction: jax.Array


class SeparationStep(NamedTuple):
    """Separated player state and fixed-shape transient impact facts."""

    players: PlayerState
    impact: PlayerImpactFacts


def _cross(left: jax.Array, right: jax.Array) -> jax.Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _point_segment_difference(
    point: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    segment = end - start
    fraction = jnp.clip(
        jnp.sum((point - start) * segment, axis=-1)
        / (jnp.sum(segment * segment, axis=-1) + SQUARED_EPS),
        0.0,
        1.0,
    )
    return point - (start + fraction[..., None] * segment)


def _tie_directions(team_id: jax.Array, attack_direction: jax.Array) -> jax.Array:
    indices = jnp.arange(team_id.shape[0])
    same_team = team_id[:, None] == team_id[None, :]
    earlier_or_self = indices[None, :] <= indices[:, None]
    within_team_rank = jnp.sum(same_team & earlier_or_self, axis=1) - 1
    angle = within_team_rank.astype(jnp.float32) * GOLDEN_ANGLE
    direction = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    return direction * attack_direction[team_id, None]


def _capsule_axes(body_forward: jax.Array, body: BodyContact):
    forward = body_forward
    shoulder = jnp.stack([-forward[:, 1], forward[:, 0]], axis=-1)
    half_core = 0.5 * (body.shoulder_width_m - body.torso_depth_m)
    return shoulder, forward, half_core


def _capsule_core_geometry(
    position: jax.Array,
    shoulder: jax.Array,
    half_core: float,
):
    return (
        position - half_core * shoulder,
        position + half_core * shoulder,
    )


def _capsule_pair_direction(
    position: jax.Array,
    core_start: jax.Array,
    core_end: jax.Array,
    forward: jax.Array,
    tie: jax.Array,
):
    a_start = core_start[:, None, :]
    a_end = core_end[:, None, :]
    b_start = core_start[None, :, :]
    b_end = core_end[None, :, :]

    difference_a_start = _point_segment_difference(a_start, b_start, b_end)
    difference_a_end = _point_segment_difference(a_end, b_start, b_end)
    difference_b_start = -_point_segment_difference(b_start, a_start, a_end)
    difference_b_end = -_point_segment_difference(b_end, a_start, a_end)
    differences = jnp.stack(
        [
            difference_a_start,
            difference_a_end,
            difference_b_start,
            difference_b_end,
        ],
        axis=-2,
    )
    distance_squared = jnp.sum(differences * differences, axis=-1)
    nearest_index = jnp.argmin(distance_squared, axis=-1)
    nearest = jnp.take_along_axis(differences, nearest_index[..., None, None], axis=-2)[
        ..., 0, :
    ]
    minimum_distance_squared = jnp.min(distance_squared, axis=-1)
    nearest_distance = jnp.sqrt(minimum_distance_squared + SQUARED_EPS)

    segment_a = a_end - a_start
    segment_b = b_end - b_start
    relative_start = b_start - a_start
    denominator = _cross(segment_a, segment_b)
    safe_denominator = jnp.where(jnp.abs(denominator) > GEOMETRY_EPS, denominator, 1.0)
    fraction_a = _cross(relative_start, segment_b) / safe_denominator
    fraction_b = _cross(relative_start, segment_a) / safe_denominator
    crossing = (
        (jnp.abs(denominator) > GEOMETRY_EPS)
        & (fraction_a >= 0.0)
        & (fraction_a <= 1.0)
        & (fraction_b >= 0.0)
        & (fraction_b <= 1.0)
    )

    segment_length_squared = jnp.sum(segment_a * segment_a, axis=-1)
    projection_0 = jnp.sum(relative_start * segment_a, axis=-1) / (
        segment_length_squared + SQUARED_EPS
    )
    projection_1 = projection_0 + jnp.sum(segment_b * segment_a, axis=-1) / (
        segment_length_squared + SQUARED_EPS
    )
    collinear = (
        (jnp.abs(denominator) <= GEOMETRY_EPS)
        & (jnp.abs(_cross(relative_start, segment_a)) <= GEOMETRY_EPS)
        & (
            jnp.maximum(jnp.minimum(projection_0, projection_1), 0.0)
            <= jnp.minimum(jnp.maximum(projection_0, projection_1), 1.0)
        )
    )
    cores_intersect = crossing | collinear
    forward_a = forward[:, None, :]
    forward_b = forward[None, :, :]
    aligned_b = jnp.where(
        (jnp.sum(forward_a * forward_b, axis=-1) >= 0.0)[..., None],
        forward_b,
        -forward_b,
    )
    separating_axis = forward_a + aligned_b
    separating_axis = separating_axis / (
        jnp.sqrt(
            jnp.sum(separating_axis * separating_axis, axis=-1, keepdims=True)
            + SQUARED_EPS
        )
        + DIV_EPS
    )
    center_difference = position[:, None, :] - position[None, :, :]
    center_projection = jnp.sum(center_difference * separating_axis, axis=-1)
    tie_projection = jnp.sum(
        (tie[:, None, :] - tie[None, :, :]) * separating_axis, axis=-1
    )
    orientation = jnp.where(
        jnp.abs(center_projection) > GEOMETRY_EPS,
        jnp.sign(center_projection),
        jnp.where(
            jnp.abs(tie_projection) > GEOMETRY_EPS,
            jnp.sign(tie_projection),
            1.0,
        ),
    )
    intersection_direction = separating_axis * orientation[..., None]

    direction = nearest / (nearest_distance[..., None] + DIV_EPS)
    direction = jnp.where(
        (cores_intersect | (nearest_distance < COINCIDENT_DISTANCE_EPS))[..., None],
        intersection_direction,
        direction,
    )
    core_distance = jnp.where(cores_intersect, 0.0, nearest_distance)
    return direction, core_distance


def _circle_pair_direction(
    position: jax.Array,
    tie: jax.Array,
):
    """Return antisymmetric centre directions for a circular torso."""

    difference = position[:, None, :] - position[None, :, :]
    distance = jnp.sqrt(jnp.sum(difference * difference, axis=-1) + SQUARED_EPS)
    tie_difference = tie[:, None, :] - tie[None, :, :]
    tie_distance = jnp.sqrt(
        jnp.sum(tie_difference * tie_difference, axis=-1, keepdims=True) + SQUARED_EPS
    )
    fallback = tie_difference / (tie_distance + DIV_EPS)
    direction = jnp.where(
        (distance < COINCIDENT_DISTANCE_EPS)[..., None],
        fallback,
        difference / (distance[..., None] + DIV_EPS),
    )
    return direction, distance


def _capsule_pair_contact_position(
    position: jax.Array,
    core_start: jax.Array,
    core_end: jax.Array,
    first: jax.Array,
    second: jax.Array,
) -> jax.Array:
    """Reconstruct the exact contact point for one selected capsule pair."""

    a_start = core_start[first]
    a_end = core_end[first]
    b_start = core_start[second]
    b_end = core_end[second]
    difference_a_start = _point_segment_difference(a_start, b_start, b_end)
    difference_a_end = _point_segment_difference(a_end, b_start, b_end)
    difference_b_start = -_point_segment_difference(b_start, a_start, a_end)
    difference_b_end = -_point_segment_difference(b_end, a_start, a_end)
    differences = jnp.stack(
        [
            difference_a_start,
            difference_a_end,
            difference_b_start,
            difference_b_end,
        ]
    )
    midpoints = jnp.stack(
        [
            a_start - 0.5 * difference_a_start,
            a_end - 0.5 * difference_a_end,
            b_start + 0.5 * difference_b_start,
            b_end + 0.5 * difference_b_end,
        ]
    )
    distance_squared = jnp.sum(differences * differences, axis=-1)
    nearest_index = jnp.argmin(distance_squared)
    minimum_distance_squared = jnp.min(distance_squared)
    ambiguous_nearest = (
        jnp.sum(jnp.abs(distance_squared - minimum_distance_squared) <= GEOMETRY_EPS)
        > 1
    )
    center_midpoint = 0.5 * (position[first] + position[second])
    nearest_midpoint = jnp.where(
        ambiguous_nearest,
        center_midpoint,
        midpoints[nearest_index],
    )

    segment_a = a_end - a_start
    segment_b = b_end - b_start
    relative_start = b_start - a_start
    denominator = _cross(segment_a, segment_b)
    safe_denominator = jnp.where(jnp.abs(denominator) > GEOMETRY_EPS, denominator, 1.0)
    fraction_a = _cross(relative_start, segment_b) / safe_denominator
    fraction_b = _cross(relative_start, segment_a) / safe_denominator
    crossing = (
        (jnp.abs(denominator) > GEOMETRY_EPS)
        & (fraction_a >= 0.0)
        & (fraction_a <= 1.0)
        & (fraction_b >= 0.0)
        & (fraction_b <= 1.0)
    )
    segment_length_squared = jnp.sum(segment_a * segment_a)
    projection_0 = jnp.sum(relative_start * segment_a) / (
        segment_length_squared + SQUARED_EPS
    )
    projection_1 = projection_0 + jnp.sum(segment_b * segment_a) / (
        segment_length_squared + SQUARED_EPS
    )
    collinear = (
        (jnp.abs(denominator) <= GEOMETRY_EPS)
        & (jnp.abs(_cross(relative_start, segment_a)) <= GEOMETRY_EPS)
        & (
            jnp.maximum(jnp.minimum(projection_0, projection_1), 0.0)
            <= jnp.minimum(jnp.maximum(projection_0, projection_1), 1.0)
        )
    )
    crossing_position = a_start + fraction_a * segment_a
    intersect_position = jnp.where(crossing, crossing_position, center_midpoint)
    return jnp.where(crossing | collinear, intersect_position, nearest_midpoint)


def separate_players(
    players: PlayerState,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    *,
    boundary_margin_m: float,
    pinned: jax.Array | None = None,
    body: BodyContact = BodyContact(),
    physics: PlayerPhysics = PlayerPhysics(),
) -> SeparationStep:
    """Resolve active torso overlap and one near-inelastic impact response.

    The dense capsule geometry is evaluated once for the initial response and
    then reused as the first configured Jacobi separation pass. Every enabled
    pair contributes to velocity response; only the strongest cross-team pair
    is retained as a fixed-shape transient fact for later adjudication.
    """

    position = players.position
    active = players.active
    count = position.shape[0]
    pinned = (
        jnp.zeros(count, dtype=bool)
        if pinned is None
        else jnp.asarray(pinned, dtype=bool)
    )
    movable = active & (~pinned)
    pair_enabled = (
        active[:, None]
        & active[None, :]
        & (~jnp.eye(count, dtype=bool))
        & (~(pinned[:, None] & pinned[None, :]))
    )
    tie = _tie_directions(players.team_id, attack_direction)
    bound = field_half_extent + boundary_margin_m
    shoulder, forward, half_core = _capsule_axes(players.body_forward, body)
    # The segment-crossing denominator scales with core_length**2.  Below its
    # geometry tolerance the capsule solve cannot distinguish orientation, and
    # the sub-millimetre core contributes no resolved shape at this precision.
    # BodyContact is static, so this specializes one graph without lax.cond.
    circular_torso = (2.0 * half_core) ** 2 <= GEOMETRY_EPS

    def pair_geometry(position):
        if circular_torso:
            return _circle_pair_direction(position, tie)
        core_start, core_end = _capsule_core_geometry(
            position,
            shoulder,
            half_core,
        )
        return _capsule_pair_direction(position, core_start, core_end, forward, tie)

    def apply_separation(position, direction, core_distance):
        overlap = jnp.where(
            pair_enabled,
            jnp.clip(
                body.torso_depth_m - core_distance,
                0.0,
                body.torso_depth_m,
            ),
            0.0,
        )

        at_lower = position <= -bound + GEOMETRY_EPS
        at_upper = position >= bound - GEOMETRY_EPS
        blocked = (at_lower[:, None, :] & (direction < 0.0)) | (
            at_upper[:, None, :] & (direction > 0.0)
        )
        feasible_direction = jnp.where(blocked, 0.0, direction)
        effectiveness = jnp.maximum(
            jnp.sum(feasible_direction * direction, axis=-1), 0.0
        )
        capacity = effectiveness * movable[:, None]
        pair_capacity = capacity + capacity.T
        contribution_share = jnp.where(
            pair_capacity > DIV_EPS,
            capacity / (pair_capacity + DIV_EPS),
            0.0,
        )
        movement_distance = jnp.where(
            effectiveness > DIV_EPS,
            overlap * contribution_share / (effectiveness + DIV_EPS),
            0.0,
        )
        correction = jnp.sum(movement_distance[:, :, None] * feasible_direction, axis=1)
        return jnp.clip(position + correction, -bound, bound)

    core_start, core_end = _capsule_core_geometry(position, shoulder, half_core)
    if circular_torso:
        direction, core_distance = _circle_pair_direction(position, tie)
    else:
        direction, core_distance = _capsule_pair_direction(
            position, core_start, core_end, forward, tie
        )
    overlap = jnp.where(
        pair_enabled,
        jnp.clip(
            body.torso_depth_m - core_distance,
            0.0,
            body.torso_depth_m,
        ),
        0.0,
    )

    velocity = players.velocity
    relative_velocity = velocity[:, None, :] - velocity[None, :, :]
    closing_speed = jnp.maximum(-jnp.sum(relative_velocity * direction, axis=-1), 0.0)
    upper_triangle = jnp.triu(jnp.ones((count, count), dtype=bool), k=1)
    responding_pair = (
        upper_triangle
        & pair_enabled
        & (overlap > GEOMETRY_EPS)
        & (closing_speed > GEOMETRY_EPS)
    )

    # Ordinary pairs use the equal-mass impulse. A pinned participant is a
    # positional constraint, so the movable counterpart receives the full
    # normal response and the pinned player's velocity is left unchanged.
    # Simultaneous Jacobi edges are normalized by their maximum incident
    # degree. A degree-one pair is therefore bit-identical to the isolated
    # impulse, while dense stars cannot add that impulse once per neighbour
    # and manufacture kinetic energy.
    first_share = movable[:, None] * jnp.where(pinned[None, :], 1.0, 0.5)
    second_share = movable[None, :] * jnp.where(pinned[:, None], 1.0, 0.5)
    incident_count = jnp.sum(responding_pair, axis=1) + jnp.sum(
        responding_pair,
        axis=0,
    )
    pair_divisor = jnp.maximum(
        jnp.maximum(incident_count[:, None], incident_count[None, :]),
        1,
    ).astype(closing_speed.dtype)
    response = (
        (1.0 + physics.collision_normal_restitution) * closing_speed / pair_divisor
    )
    first_magnitude = jnp.where(responding_pair, response * first_share, 0.0)
    second_magnitude = jnp.where(responding_pair, response * second_share, 0.0)
    velocity_correction = jnp.sum(
        first_magnitude[:, :, None] * direction, axis=1
    ) + jnp.sum(-second_magnitude[:, :, None] * direction, axis=0)
    correction_budget = jnp.sum(first_magnitude, axis=1) + jnp.sum(
        second_magnitude, axis=0
    )
    correction_norm = jnp.sqrt(
        jnp.sum(velocity_correction * velocity_correction, axis=-1) + SQUARED_EPS
    )
    correction_scale = jnp.minimum(correction_budget / (correction_norm + DIV_EPS), 1.0)
    velocity = velocity + velocity_correction * correction_scale[:, None]

    opponent_pair = players.team_id[:, None] != players.team_id[None, :]
    impact_candidate = responding_pair & opponent_pair
    flat_score = jnp.where(impact_candidate, closing_speed, -1.0).reshape(-1)
    strongest_flat = jnp.argmax(flat_score)
    strongest_first = strongest_flat // count
    strongest_second = strongest_flat % count
    impact_occurred = jnp.any(impact_candidate)
    safe_first = jnp.where(impact_occurred, strongest_first, 0)
    safe_second = jnp.where(impact_occurred, strongest_second, 0)
    strongest_normal = direction[safe_first, safe_second]
    if circular_torso:
        strongest_contact = 0.5 * (position[safe_first] + position[safe_second])
    else:
        strongest_contact = _capsule_pair_contact_position(
            position,
            core_start,
            core_end,
            safe_first,
            safe_second,
        )
    strongest_closing = closing_speed[safe_first, safe_second]
    strongest_first_velocity = players.velocity[safe_first]
    strongest_second_velocity = players.velocity[safe_second]
    strongest_first_speed = jnp.linalg.norm(strongest_first_velocity)
    strongest_second_speed = jnp.linalg.norm(strongest_second_velocity)
    strongest_velocity_alignment = jnp.clip(
        jnp.dot(strongest_first_velocity, strongest_second_velocity)
        / jnp.maximum(
            strongest_first_speed * strongest_second_speed,
            jnp.asarray(1.0e-9, dtype=players.velocity.dtype),
        ),
        0.0,
        1.0,
    )

    first_toward_second = jnp.maximum(
        -jnp.sum(players.velocity[safe_first] * strongest_normal), 0.0
    )
    second_toward_first = jnp.maximum(
        jnp.sum(players.velocity[safe_second] * strongest_normal), 0.0
    )
    actor_is_first = first_toward_second >= second_toward_first
    actor_attribution_decisive = (
        jnp.abs(first_toward_second - second_toward_first) > GEOMETRY_EPS
    )
    actor = jnp.where(actor_is_first, safe_first, safe_second).astype(jnp.int32)
    victim = jnp.where(actor_is_first, safe_second, safe_first).astype(jnp.int32)
    actor_to_victim_normal = jnp.where(
        actor_is_first,
        -strongest_normal,
        strongest_normal,
    )
    impact = PlayerImpactFacts(
        occurred=impact_occurred,
        actor=jnp.where(impact_occurred, actor, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(impact_occurred, victim, NO_PLAYER).astype(jnp.int32),
        contact_normal=jnp.where(
            impact_occurred, actor_to_victim_normal, jnp.zeros(2, jnp.float32)
        ).astype(jnp.float32),
        contact_position=jnp.where(
            impact_occurred, strongest_contact, jnp.zeros(2, jnp.float32)
        ).astype(jnp.float32),
        actor_attribution_decisive=(impact_occurred & actor_attribution_decisive),
        impact_score=jnp.where(impact_occurred, strongest_closing, 0.0).astype(
            jnp.float32
        ),
        velocity_alignment=jnp.where(
            impact_occurred, strongest_velocity_alignment, 0.0
        ).astype(jnp.float32),
        # A direct separate_players call observes this interval endpoint.
        # step_players replaces it with the enclosing locomotion-microstep
        # endpoint before reducing impacts across that interval.
        time_fraction=jnp.where(impact_occurred, 1.0, 0.0).astype(jnp.float32),
    )

    if physics.separation_iterations > 0:
        position = apply_separation(position, direction, core_distance)

        def separate_once(_, current_position):
            current_direction, current_distance = pair_geometry(current_position)
            return apply_separation(
                current_position, current_direction, current_distance
            )

        # Stage the fixed-count iteration instead of duplicating the dense
        # N x N capsule solve in HLO once per configured iteration.
        position = jax.lax.fori_loop(
            1,
            physics.separation_iterations,
            separate_once,
            position,
        )

    at_lower = position <= -bound + GEOMETRY_EPS
    at_upper = position >= bound - GEOMETRY_EPS
    outward = (at_lower & (velocity < 0.0)) | (at_upper & (velocity > 0.0))
    velocity = jnp.where(outward, 0.0, velocity)
    velocity = jnp.where(active[:, None], velocity, 0.0)
    position = jnp.where(active[:, None], position, players.position)
    return SeparationStep(
        players=players._replace(position=position, velocity=velocity),
        impact=impact,
    )


__all__ = ["PlayerImpactFacts", "SeparationStep", "separate_players"]
