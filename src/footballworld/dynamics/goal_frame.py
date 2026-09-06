"""Swept goal-post and crossbar collision primitives."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    SQUARED_EPS,
    WOODWORK_CROSSBAR,
    WOODWORK_NONE,
    WOODWORK_POST,
)
from footballworld.core.state import BallState


class GoalFrameEvent(NamedTuple):
    """Earliest frame contact along one fixed-shape path segment."""

    occurred: jax.Array
    kind: jax.Array
    time_fraction: jax.Array
    position: jax.Array
    normal: jax.Array


def _norm(vector: jax.Array, *, axis: int = -1) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=axis) + SQUARED_EPS)


def _capsule_entry(
    position: jax.Array,
    path_delta: jax.Array,
    centers: jax.Array,
    perpendicular_axes: tuple[int, int],
    along_axis: int,
    along_lower: float,
    along_upper: float,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    i, j = perpendicular_axes
    perpendicular_delta = jnp.stack((path_delta[i], path_delta[j]))
    origin = jnp.stack((position[i], position[j]))[None, :] - centers

    quadratic = jnp.sum(perpendicular_delta * perpendicular_delta)
    moving = quadratic > SQUARED_EPS
    linear = 2.0 * jnp.sum(origin * perpendicular_delta[None, :], axis=-1)
    distance_squared = jnp.sum(origin * origin, axis=-1)
    constant = distance_squared - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
    perpendicular_entry = jnp.where(moving, (-linear - root) / denominator, 0.0)
    perpendicular_exit = jnp.where(moving, (-linear + root) / denominator, 1.0)
    perpendicular_valid = jnp.where(
        moving,
        (discriminant > 0.0)
        & (perpendicular_exit >= 0.0)
        & (perpendicular_entry <= 1.0),
        distance_squared < radius * radius,
    )

    along_origin = position[along_axis]
    along_delta = path_delta[along_axis]
    along_moving = jnp.abs(along_delta) > GEOMETRY_EPS
    inverse = 1.0 / jnp.where(along_moving, along_delta, 1.0)
    lower_fraction = (along_lower - along_origin) * inverse
    upper_fraction = (along_upper - along_origin) * inverse
    span_entry = jnp.where(
        along_moving,
        jnp.minimum(lower_fraction, upper_fraction),
        0.0,
    )
    span_exit = jnp.where(
        along_moving,
        jnp.maximum(lower_fraction, upper_fraction),
        1.0,
    )
    along_valid = jnp.where(
        along_moving,
        (span_exit >= 0.0) & (span_entry <= 1.0),
        (along_origin >= along_lower) & (along_origin <= along_upper),
    )

    entry = jnp.maximum(jnp.maximum(perpendicular_entry, span_entry), 0.0)
    exit_ = jnp.minimum(jnp.minimum(perpendicular_exit, span_exit), 1.0)
    hit = perpendicular_valid & along_valid & (entry <= exit_)
    fraction = jnp.clip(entry, 0.0, 1.0)
    offset = origin + fraction[:, None] * perpendicular_delta[None, :]
    return hit, fraction, offset, linear


def _sphere_entry(
    position: jax.Array,
    path_delta: jax.Array,
    centers: jax.Array,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    origin = position[None, :] - centers
    quadratic = jnp.sum(path_delta * path_delta)
    moving = quadratic > SQUARED_EPS
    linear = 2.0 * jnp.sum(origin * path_delta[None, :], axis=-1)
    distance_squared = jnp.sum(origin * origin, axis=-1)
    constant = distance_squared - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
    entry = jnp.where(moving, (-linear - root) / denominator, 0.0)
    exit_ = jnp.where(moving, (-linear + root) / denominator, 1.0)
    valid = jnp.where(
        moving,
        (discriminant > 0.0) & (exit_ >= 0.0) & (entry <= 1.0),
        distance_squared < radius * radius,
    )
    lower = jnp.maximum(entry, 0.0)
    upper = jnp.minimum(exit_, 1.0)
    hit = valid & (lower <= upper)
    fraction = jnp.clip(lower, 0.0, 1.0)
    offset = origin + fraction[:, None] * path_delta[None, :]
    return hit, fraction, offset, linear


def detect_goal_frame_contact(
    position: jax.Array,
    path_delta: jax.Array,
    ball_live: jax.Array,
    *,
    ball: Ball = Ball(),
    stadium: Stadium = Stadium(),
    physics: BallPhysics = BallPhysics(),
) -> GoalFrameEvent:
    """Detect the earliest swept contact with either goal frame."""

    dtype = position.dtype
    frame_radius = physics.goal_frame_radius
    collision_radius = ball.radius + frame_radius
    post_y = 0.5 * stadium.goal_width + frame_radius
    bar_z = stadium.goal_height + frame_radius
    half_length = stadium.half_length

    post_centers = jnp.asarray(
        [
            [-half_length, -post_y],
            [-half_length, post_y],
            [half_length, -post_y],
            [half_length, post_y],
        ],
        dtype=dtype,
    )
    bar_centers = jnp.asarray(
        [[-half_length, bar_z], [half_length, bar_z]],
        dtype=dtype,
    )
    corner_centers = jnp.asarray(
        [
            [-half_length, -post_y, bar_z],
            [-half_length, post_y, bar_z],
            [half_length, -post_y, bar_z],
            [half_length, post_y, bar_z],
        ],
        dtype=dtype,
    )

    post_hit, post_fraction, post_offset, post_radial_motion = _capsule_entry(
        position,
        path_delta,
        post_centers,
        (0, 1),
        2,
        0.0,
        bar_z,
        collision_radius,
    )
    bar_hit, bar_fraction, bar_offset, bar_radial_motion = _capsule_entry(
        position,
        path_delta,
        bar_centers,
        (0, 2),
        1,
        -post_y,
        post_y,
        collision_radius,
    )
    corner_hit, corner_fraction, corner_offset, corner_radial_motion = _sphere_entry(
        position,
        path_delta,
        corner_centers,
        collision_radius,
    )

    post_normal = jnp.concatenate(
        (post_offset, jnp.zeros((post_offset.shape[0], 1), dtype=dtype)),
        axis=-1,
    )
    bar_normal = jnp.stack(
        (
            bar_offset[:, 0],
            jnp.zeros(bar_offset.shape[0], dtype=dtype),
            bar_offset[:, 1],
        ),
        axis=-1,
    )
    hit = jnp.concatenate((post_hit, bar_hit, corner_hit))
    fraction = jnp.concatenate((post_fraction, bar_fraction, corner_fraction))
    normal_offset = jnp.concatenate((post_normal, bar_normal, corner_offset), axis=0)

    # The quadratic entry routines deliberately report a time-zero overlap so
    # reconstructed or stationary states can be projected out.  After a real
    # impact, however, float32 rounding can leave the centre microscopically on
    # the surface.  That is an egress state, not another impact.  Reject only
    # candidates already moving out of their own radial surface; keep other
    # post/bar candidates alive so a genuine corner-to-frame contact is still
    # scheduled on the shared timeline.
    # The entry solvers already form twice this radial derivative as their
    # quadratic linear term, so reusing it avoids another vector reduction.
    radial_motion = jnp.concatenate(
        (post_radial_motion, bar_radial_motion, corner_radial_motion)
    )
    escaping_overlap = (fraction == 0.0) & (radial_motion > 0.0)
    hit = hit & (~escaping_overlap)
    reverse_path = -path_delta
    post_path_fallback = jnp.broadcast_to(
        jnp.asarray([reverse_path[0], reverse_path[1], 0.0], dtype=dtype),
        post_normal.shape,
    )
    bar_path_fallback = jnp.broadcast_to(
        jnp.asarray([reverse_path[0], 0.0, reverse_path[2]], dtype=dtype),
        bar_normal.shape,
    )
    corner_path_fallback = jnp.broadcast_to(reverse_path, corner_offset.shape)
    path_fallback = jnp.concatenate(
        (post_path_fallback, bar_path_fallback, corner_path_fallback), axis=0
    )

    # A reconstructed state may begin exactly on a post/bar axis. A zero normal
    # would make the reported time-zero hit
    # neither reflected nor pushed the ball out and could consume the whole
    # event budget. Prefer the reverse incoming path in the collider's normal
    # plane; a static feature-centre direction closes the zero-path case without
    # introducing a physical coefficient.
    post_feature_fallback = jnp.concatenate(
        (post_centers, jnp.zeros((post_centers.shape[0], 1), dtype=dtype)), axis=-1
    )
    bar_feature_fallback = jnp.stack(
        (
            bar_centers[:, 0],
            jnp.zeros(bar_centers.shape[0], dtype=dtype),
            bar_centers[:, 1],
        ),
        axis=-1,
    )
    feature_fallback = jnp.concatenate(
        (post_feature_fallback, bar_feature_fallback, corner_centers), axis=0
    )
    kind = jnp.concatenate(
        (
            jnp.full(post_hit.shape, WOODWORK_POST, dtype=jnp.int32),
            jnp.full(bar_hit.shape, WOODWORK_CROSSBAR, dtype=jnp.int32),
            jnp.full(corner_hit.shape, WOODWORK_CROSSBAR, dtype=jnp.int32),
        )
    )

    hit = hit & jnp.asarray(ball_live, dtype=bool) & (frame_radius > 0.0)
    winner = jnp.argmin(jnp.where(hit, fraction, jnp.inf))
    occurred = jnp.any(hit)
    time_fraction = jnp.where(occurred, fraction[winner], 0.0)
    selected_offset = normal_offset[winner]
    geometric_offset_length = jnp.sqrt(jnp.sum(selected_offset * selected_offset))
    offset_length = _norm(selected_offset)
    selected_path_fallback = path_fallback[winner]
    path_length = _norm(selected_path_fallback)
    selected_feature_fallback = feature_fallback[winner]
    feature_length = _norm(selected_feature_fallback)
    fallback_normal = jnp.where(
        path_length > GEOMETRY_EPS,
        selected_path_fallback / jnp.maximum(path_length, DIV_EPS),
        selected_feature_fallback / jnp.maximum(feature_length, DIV_EPS),
    )
    normal = jnp.where(
        geometric_offset_length > GEOMETRY_EPS,
        selected_offset / jnp.maximum(offset_length, DIV_EPS),
        fallback_normal,
    )

    contact_position = position + time_fraction * path_delta
    # Four geometry epsilons exceed one float32 coordinate ULP at the default
    # goal line; this is numerical progress for the exact-axis fallback, not an
    # enlarged frame or fitted clearance.
    push_margin = jnp.where(
        geometric_offset_length > GEOMETRY_EPS,
        GEOMETRY_EPS,
        4.0 * GEOMETRY_EPS,
    )
    push_distance = jnp.maximum(
        collision_radius + push_margin - geometric_offset_length, 0.0
    )
    contact_position = contact_position + push_distance * normal
    return GoalFrameEvent(
        occurred=occurred,
        kind=jnp.where(occurred, kind[winner], WOODWORK_NONE).astype(jnp.int32),
        time_fraction=time_fraction,
        position=jnp.where(
            occurred, contact_position, jnp.zeros_like(contact_position)
        ),
        normal=jnp.where(occurred, normal, jnp.zeros_like(normal)),
    )


def apply_goal_frame_contact(
    state: BallState,
    event: GoalFrameEvent,
    *,
    ball: Ball = Ball(),
    physics: BallPhysics = BallPhysics(),
) -> BallState:
    """Apply one frame impulse to a ball already advanced to contact time."""

    velocity = state.velocity
    spin = state.spin
    normal_speed = jnp.dot(velocity, event.normal)
    approaching = event.occurred & (normal_speed < 0.0)
    tangential_velocity = velocity - normal_speed * event.normal
    slip = tangential_velocity - ball.radius * jnp.cross(spin, event.normal)
    slip = slip - jnp.dot(slip, event.normal) * event.normal
    slip_speed = _norm(slip)
    slip_direction = slip / jnp.maximum(slip_speed, DIV_EPS)

    inertia = physics.ball_inertia_ratio
    normal_impulse = (1.0 + physics.goal_frame_e_rest) * jnp.maximum(-normal_speed, 0.0)
    sticking_impulse = (inertia / (1.0 + inertia)) * slip_speed
    tangential_impulse = jnp.minimum(
        sticking_impulse,
        physics.goal_frame_mu * normal_impulse,
    )
    tangential_delta = -tangential_impulse * slip_direction
    candidate_velocity = (
        tangential_velocity
        + tangential_delta
        - physics.goal_frame_e_rest * normal_speed * event.normal
    )
    candidate_spin = spin - jnp.cross(event.normal, tangential_delta) / (
        inertia * ball.radius
    )

    incoming_energy = jnp.sum(velocity * velocity) + (
        inertia * ball.radius * ball.radius * jnp.sum(spin * spin)
    )
    outgoing_energy = jnp.sum(candidate_velocity * candidate_velocity) + (
        inertia * ball.radius * ball.radius * jnp.sum(candidate_spin * candidate_spin)
    )
    energy_scale = jnp.sqrt(
        jnp.minimum(1.0, incoming_energy / (outgoing_energy + DIV_EPS))
    )
    candidate_velocity = candidate_velocity * energy_scale
    candidate_spin = candidate_spin * energy_scale

    return state._replace(
        position=jnp.where(event.occurred, event.position, state.position),
        velocity=jnp.where(approaching, candidate_velocity, velocity),
        spin=jnp.where(approaching, candidate_spin, spin),
    )
