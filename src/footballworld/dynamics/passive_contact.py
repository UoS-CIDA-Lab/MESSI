"""Deterministic swept collision between a live ball and player bodies."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.reach import Reach
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    RK_NONE,
    RK_THROWIN,
    SAFE_NORM_EPS,
    SQUARED_EPS,
)
from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_DEFLECTION,
    ContactResult,
)
from footballworld.core.state import (
    RestartReleaseProvenance,
    State,
)
from footballworld.rules.gk_handling_restriction import (
    clear_backpass_after_passive_contact,
    goalkeeper_hand_restricted_mask,
)


class PassiveContactEvent(NamedTuple):
    """First accidental player-body hit along a frozen ball segment."""

    occurred: jax.Array
    actor: jax.Array
    time_fraction: jax.Array
    normal: jax.Array


class BetweenLegsPassageEvent(NamedTuple):
    """First clean crossing through a player's fixed triangular leg gap."""

    occurred: jax.Array
    actor: jax.Array
    time_fraction: jax.Array
    position: jax.Array


class PassiveContactStep(NamedTuple):
    """State at impact plus fixed-shape timing needed by the ball integrator."""

    state: State
    occurred: jax.Array
    actor: jax.Array
    time_fraction: jax.Array


def _norm(vector: jax.Array, *, axis: int = -1) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=axis) + SAFE_NORM_EPS)


def _slab_interval(
    origin: jax.Array,
    delta: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Intersection interval of a fixed-shape segment with an axis-aligned box."""

    moving = jnp.abs(delta) > GEOMETRY_EPS
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (lower - origin) / safe_delta
    second = (upper - origin) / safe_delta
    axis_entry = jnp.minimum(first, second)
    axis_exit = jnp.maximum(first, second)

    inside = (origin >= lower) & (origin <= upper)
    axis_entry = jnp.where(moving, axis_entry, jnp.where(inside, -jnp.inf, jnp.inf))
    axis_exit = jnp.where(moving, axis_exit, jnp.where(inside, jnp.inf, -jnp.inf))
    entry = jnp.max(axis_entry, axis=-1)
    exit = jnp.min(axis_exit, axis=-1)
    valid = (entry <= exit) & (exit >= 0.0) & (entry <= 1.0)
    return valid, jnp.clip(entry, 0.0, 1.0), jnp.clip(exit, 0.0, 1.0)


def _circle_interval(
    origin: jax.Array,
    delta: jax.Array,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Intersection interval of a fixed-shape segment with circles at the origin."""

    delta = jnp.broadcast_to(delta, origin.shape)
    quadratic = jnp.sum(delta * delta, axis=-1)
    linear = 2.0 * jnp.sum(origin * delta, axis=-1)
    constant = jnp.sum(origin * origin, axis=-1) - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    moving = quadratic > SQUARED_EPS
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(moving, 2.0 * quadratic, 1.0)
    entry = (-linear - root) / denominator
    exit = (-linear + root) / denominator
    inside = constant <= 0.0
    valid = jnp.where(
        moving,
        (discriminant >= 0.0) & (exit >= 0.0) & (entry <= 1.0),
        inside,
    )
    entry = jnp.where(moving, entry, 0.0)
    exit = jnp.where(moving, exit, 1.0)
    return valid, jnp.clip(entry, 0.0, 1.0), jnp.clip(exit, 0.0, 1.0)


def _capsule_interval(
    origin: jax.Array,
    delta: jax.Array,
    half_core: float,
    radius: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Segment interval inside a 2-D capsule whose core lies on local x."""

    rectangle_valid, rectangle_entry, rectangle_exit = _slab_interval(
        origin,
        delta,
        jnp.array([-half_core, -radius], dtype=origin.dtype),
        jnp.array([half_core, radius], dtype=origin.dtype),
    )
    lower_valid, lower_entry, lower_exit = _circle_interval(
        origin - jnp.array([-half_core, 0.0], dtype=origin.dtype),
        delta,
        radius,
    )
    upper_valid, upper_entry, upper_exit = _circle_interval(
        origin - jnp.array([half_core, 0.0], dtype=origin.dtype),
        delta,
        radius,
    )

    valid_parts = jnp.stack((rectangle_valid, lower_valid, upper_valid), axis=-1)
    entries = jnp.stack((rectangle_entry, lower_entry, upper_entry), axis=-1)
    exits = jnp.stack((rectangle_exit, lower_exit, upper_exit), axis=-1)
    valid = jnp.any(valid_parts, axis=-1)
    entry = jnp.min(jnp.where(valid_parts, entries, jnp.inf), axis=-1)
    exit = jnp.max(jnp.where(valid_parts, exits, -jnp.inf), axis=-1)
    return valid, entry, exit


def _segment_capsule_collision(
    origin: jax.Array,
    delta: jax.Array,
    core_start: jax.Array,
    core_end: jax.Array,
    radius: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact swept-point entry and normal for a fixed 3-D capsule.

    Expanding ``radius`` by the ball radius turns this into swept-sphere
    collision without adding a substep loop. The finite cylinder and its two
    spherical caps are solved independently and the earliest valid root wins.
    """

    axis = core_end - core_start
    axis_squared = jnp.sum(axis * axis, axis=-1)
    safe_axis_squared = jnp.maximum(axis_squared, SQUARED_EPS)
    relative_origin = origin - core_start
    origin_axis = jnp.sum(relative_origin * axis, axis=-1)
    delta_axis = jnp.sum(delta * axis, axis=-1)
    perpendicular_origin = (
        relative_origin - (origin_axis / safe_axis_squared)[..., None] * axis
    )
    perpendicular_delta = delta - (delta_axis / safe_axis_squared)[..., None] * axis

    quadratic = jnp.sum(perpendicular_delta * perpendicular_delta, axis=-1)
    linear = 2.0 * jnp.sum(perpendicular_origin * perpendicular_delta, axis=-1)
    constant = (
        jnp.sum(perpendicular_origin * perpendicular_origin, axis=-1) - radius * radius
    )
    discriminant = linear * linear - 4.0 * quadratic * constant
    moving_across_axis = quadratic > SQUARED_EPS
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    denominator = jnp.where(moving_across_axis, 2.0 * quadratic, 1.0)
    cylinder_entry_raw = (-linear - root) / denominator
    cylinder_exit_raw = (-linear + root) / denominator
    inside_cylinder = (
        (constant <= 0.0) & (origin_axis >= 0.0) & (origin_axis <= axis_squared)
    )
    cylinder_entry = jnp.where(inside_cylinder, 0.0, cylinder_entry_raw)
    cylinder_axis = origin_axis + cylinder_entry * delta_axis
    cylinder_valid = inside_cylinder | (
        moving_across_axis
        & (discriminant >= 0.0)
        & (cylinder_exit_raw >= 0.0)
        & (cylinder_entry_raw <= 1.0)
        & (cylinder_axis >= 0.0)
        & (cylinder_axis <= axis_squared)
    )
    cylinder_entry = jnp.clip(cylinder_entry, 0.0, 1.0)

    start_valid, start_entry, _ = _circle_interval(origin - core_start, delta, radius)
    end_valid, end_entry, _ = _circle_interval(origin - core_end, delta, radius)
    valid_parts = jnp.stack((cylinder_valid, start_valid, end_valid), axis=-1)
    entries = jnp.stack((cylinder_entry, start_entry, end_entry), axis=-1)
    valid = jnp.any(valid_parts, axis=-1)
    entry = jnp.min(jnp.where(valid_parts, entries, jnp.inf), axis=-1)
    safe_entry = jnp.where(valid, entry, 0.0)

    point = origin + safe_entry[..., None] * delta
    core_fraction = jnp.clip(
        jnp.sum((point - core_start) * axis, axis=-1) / safe_axis_squared,
        0.0,
        1.0,
    )
    nearest_core = core_start + core_fraction[..., None] * axis
    normal = point - nearest_core
    normal_length = _norm(normal)
    fallback = -delta / _norm(delta)[..., None]
    normal = jnp.where(
        (normal_length > GEOMETRY_EPS)[..., None],
        normal / normal_length[..., None],
        fallback,
    )
    return valid, entry, normal


def _player_local_ball_path(
    state: State,
    path_delta: jax.Array,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Express the ball/player relative sweep in each player-facing frame."""

    forward = state.players.body_forward
    shoulder = jnp.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    relative_xy = state.ball.position[:2] - player_start_position
    relative_path_delta = path_delta[:2] - player_path_delta
    local_origin_xy = jnp.stack(
        (
            jnp.sum(relative_xy * shoulder, axis=-1),
            jnp.sum(relative_xy * forward, axis=-1),
        ),
        axis=-1,
    )
    local_delta_xy = jnp.stack(
        (
            jnp.sum(relative_path_delta * shoulder, axis=-1),
            jnp.sum(relative_path_delta * forward, axis=-1),
        ),
        axis=-1,
    )
    local_origin = jnp.concatenate(
        (
            local_origin_xy,
            jnp.broadcast_to(
                state.ball.position[2], player_start_position[:, :1].shape
            ),
        ),
        axis=-1,
    )
    local_delta = jnp.concatenate(
        (
            local_delta_xy,
            jnp.broadcast_to(path_delta[2], player_path_delta[:, :1].shape),
        ),
        axis=-1,
    )
    return local_origin, local_delta, shoulder, forward


def _leg_collision(
    state: State,
    local_origin: jax.Array,
    local_delta: jax.Array,
    shoulder: jax.Array,
    forward: jax.Array,
    *,
    ball: Ball,
    body: BodyContact,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Earliest hit on either diagonal leg of the fixed isosceles stance."""

    player_count = state.players.position.shape[0]
    dtype = local_origin.dtype
    apex_height = state.players.height * body.leg_apex_height_factor
    lateral_feet = jnp.broadcast_to(
        jnp.asarray(
            [-0.5 * body.shoulder_width_m, 0.5 * body.shoulder_width_m],
            dtype=dtype,
        ),
        (player_count, 2),
    )
    zero = jnp.zeros_like(lateral_feet)
    foot_centres = jnp.stack((lateral_feet, zero, zero), axis=-1)
    apex = jnp.stack(
        (
            zero,
            zero,
            jnp.broadcast_to(apex_height[:, None], lateral_feet.shape),
        ),
        axis=-1,
    )
    leg_valid, leg_entry, leg_local_normal = _segment_capsule_collision(
        local_origin[:, None, :],
        local_delta[:, None, :],
        apex,
        foot_centres,
        body.leg_radius_m + ball.radius,
    )
    first_leg = jnp.argmin(jnp.where(leg_valid, leg_entry, jnp.inf), axis=-1).astype(
        jnp.int32
    )
    selected_entry = jnp.take_along_axis(leg_entry, first_leg[:, None], axis=-1)[:, 0]
    selected_local_normal = jnp.take_along_axis(
        leg_local_normal, first_leg[:, None, None], axis=1
    )[:, 0, :]
    horizontal_normal = (
        selected_local_normal[:, 0, None] * shoulder
        + selected_local_normal[:, 1, None] * forward
    )
    normal = jnp.concatenate(
        (horizontal_normal, selected_local_normal[:, 2, None]), axis=-1
    )
    valid = jnp.any(leg_valid, axis=-1)
    return valid, selected_entry, normal


def _body_collision(
    state: State,
    path_delta: jax.Array,
    *,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    ball: Ball,
    body: BodyContact,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    players = state.players
    local_origin_3d, local_delta_3d, shoulder, forward = _player_local_ball_path(
        state, path_delta, player_start_position, player_path_delta
    )
    local_origin = local_origin_3d[:, :2]
    local_delta = local_delta_3d[:, :2]
    top = body.torso_top_height(players.height)
    leg_apex = players.height * body.leg_apex_height_factor

    def collision_for_shape(width, depth, bottom, shape_top):
        half_core = 0.5 * (width - depth)
        collision_radius = 0.5 * depth + ball.radius
        horizontal_valid, horizontal_entry, horizontal_exit = _capsule_interval(
            local_origin,
            local_delta,
            half_core,
            collision_radius,
        )
        vertical_valid, vertical_entry, vertical_exit = _slab_interval(
            jnp.broadcast_to(state.ball.position[2], top[:, None].shape),
            jnp.broadcast_to(path_delta[2], top[:, None].shape),
            (bottom - ball.radius)[:, None],
            (shape_top + ball.radius)[:, None],
        )
        entry = jnp.maximum(horizontal_entry, vertical_entry)
        exit = jnp.minimum(horizontal_exit, vertical_exit)
        valid = horizontal_valid & vertical_valid & (entry <= exit)

        local_point = local_origin + entry[:, None] * local_delta
        nearest_core = jnp.stack(
            (
                jnp.clip(local_point[:, 0], -half_core, half_core),
                jnp.zeros_like(local_point[:, 1]),
            ),
            axis=-1,
        )
        local_normal = local_point - nearest_core
        local_normal_length = _norm(local_normal)
        horizontal_normal = (
            local_normal[:, 0, None] * shoulder + local_normal[:, 1, None] * forward
        ) / local_normal_length[:, None]
        horizontal_normal = jnp.concatenate(
            (
                horizontal_normal,
                jnp.zeros(
                    (players.position.shape[0], 1),
                    dtype=horizontal_normal.dtype,
                ),
            ),
            axis=-1,
        )

        contact_height = state.ball.position[2] + entry * path_delta[2]
        expanded_bottom = bottom - ball.radius
        expanded_top = shape_top + ball.radius
        vertical_sign = jnp.where(
            jnp.abs(contact_height - expanded_bottom)
            <= jnp.abs(contact_height - expanded_top),
            -1.0,
            1.0,
        )
        vertical_normal = jnp.stack(
            (
                jnp.zeros_like(vertical_sign),
                jnp.zeros_like(vertical_sign),
                vertical_sign,
            ),
            axis=-1,
        )
        vertical_first = vertical_entry > horizontal_entry + GEOMETRY_EPS
        normal = jnp.where(vertical_first[:, None], vertical_normal, horizontal_normal)
        return valid, entry, normal

    torso_valid, torso_entry, torso_normal = collision_for_shape(
        body.shoulder_width_m,
        body.torso_depth_m,
        leg_apex,
        top,
    )
    lower_valid, lower_entry, lower_normal = _leg_collision(
        state,
        local_origin_3d,
        local_delta_3d,
        shoulder,
        forward,
        ball=ball,
        body=body,
    )
    use_lower = lower_valid & ((~torso_valid) | (lower_entry <= torso_entry))
    valid = torso_valid | lower_valid
    entry = jnp.where(use_lower, lower_entry, torso_entry)
    normal = jnp.where(use_lower[:, None], lower_normal, torso_normal)

    velocity = jnp.broadcast_to(state.ball.velocity, normal.shape)
    fallback = -velocity / _norm(velocity)[:, None]
    normal = jnp.where((_norm(normal) > GEOMETRY_EPS)[:, None], normal, fallback)
    return valid, entry, normal


def _head_collision(
    state: State,
    path_delta: jax.Array,
    *,
    player_start_position: jax.Array,
    player_path_delta: jax.Array,
    ball: Ball,
    body: BodyContact,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    head_center = jnp.concatenate(
        (
            player_start_position,
            body.head_center_height(state.players.height)[:, None],
        ),
        axis=-1,
    )
    origin = state.ball.position - head_center
    relative_path_delta = jnp.concatenate(
        (
            path_delta[:2] - player_path_delta,
            jnp.broadcast_to(path_delta[2], player_path_delta[:, :1].shape),
        ),
        axis=-1,
    )
    valid, entry, _ = _circle_interval(
        origin, relative_path_delta, body.head_radius_m + ball.radius
    )
    relative_contact_point = origin + entry[:, None] * relative_path_delta
    normal = relative_contact_point
    velocity = jnp.broadcast_to(state.ball.velocity, normal.shape)
    fallback = -velocity / _norm(velocity)[:, None]
    normal_length = _norm(normal)
    normal = jnp.where(
        (normal_length > GEOMETRY_EPS)[:, None],
        normal / normal_length[:, None],
        fallback,
    )
    return valid, entry, normal


def _field_exit_fraction(
    position: jax.Array,
    path_delta: jax.Array,
    *,
    ball: Ball,
    stadium: Stadium,
) -> jax.Array:
    boundary = jnp.array(
        [stadium.half_length + ball.radius, stadium.half_width + ball.radius],
        dtype=position.dtype,
    )
    start = position[:2]
    delta = path_delta[:2]
    end = start + delta
    positive = (delta > 0.0) & (end > boundary)
    negative = (delta < 0.0) & (end < -boundary)
    positive_time = (boundary - start) / (delta + DIV_EPS)
    negative_time = (-boundary - start) / (delta - DIV_EPS)
    exit_time = jnp.where(
        positive, positive_time, jnp.where(negative, negative_time, 1.0)
    )
    exit_time = jnp.where(jnp.abs(start) > boundary, 0.0, exit_time)
    return jnp.min(jnp.clip(exit_time, 0.0, 1.0))


def detect_between_legs_passage(
    state: State,
    path_delta: jax.Array,
    excluded_actor: int | jax.Array = NO_PLAYER,
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
    *,
    player_start_position: jax.Array | None = None,
    player_path_delta: jax.Array | None = None,
    secondary_excluded_actor: int | jax.Array = NO_PLAYER,
) -> BetweenLegsPassageEvent:
    """Detect the first clean frontal-plane crossing inside a triangular gap.

    This is fixed-shape telemetry: it neither changes ball motion nor assigns
    an attacking player. A consumer can combine it with deliberate-contact
    provenance to classify a nutmeg. Any swept hit on either physical leg
    suppresses the passage, independent of the defender's selected intent.
    """

    player_start_position = (
        state.players.position
        if player_start_position is None
        else player_start_position
    )
    player_path_delta = (
        jnp.zeros_like(state.players.position)
        if player_path_delta is None
        else player_path_delta
    )
    local_origin, local_delta, shoulder, forward = _player_local_ball_path(
        state, path_delta, player_start_position, player_path_delta
    )
    leg_hit, _, _ = _leg_collision(
        state,
        local_origin,
        local_delta,
        shoulder,
        forward,
        ball=ball_geometry,
        body=body,
    )

    forward_start = local_origin[:, 1]
    forward_end = forward_start + local_delta[:, 1]
    crosses_plane = ((forward_start < -GEOMETRY_EPS) & (forward_end >= 0.0)) | (
        (forward_start > GEOMETRY_EPS) & (forward_end <= 0.0)
    )
    safe_forward_delta = jnp.where(
        jnp.abs(local_delta[:, 1]) > GEOMETRY_EPS, local_delta[:, 1], 1.0
    )
    crossing = jnp.clip(-forward_start / safe_forward_delta, 0.0, 1.0)
    local_point = local_origin + crossing[:, None] * local_delta
    apex_height = state.players.height * body.leg_apex_height_factor
    half_core_gap = (
        0.5
        * body.shoulder_width_m
        * (1.0 - local_point[:, 2] / jnp.maximum(apex_height, GEOMETRY_EPS))
    )
    inside_triangle = (
        (local_point[:, 2] >= 0.0)
        & (local_point[:, 2] <= apex_height)
        & (jnp.abs(local_point[:, 0]) < half_core_gap)
    )

    player_index = jnp.arange(state.players.position.shape[0])
    candidate = (
        crosses_plane
        & inside_triangle
        & (~leg_hit)
        & state.players.active
        & (player_index != jnp.asarray(excluded_actor, dtype=jnp.int32))
        & (player_index != jnp.asarray(secondary_excluded_actor, dtype=jnp.int32))
        & state.ball.live
    )
    earliest = jnp.min(jnp.where(candidate, crossing, jnp.inf))
    finalists = candidate & (crossing == earliest)
    winner = jnp.argmin(
        jnp.where(finalists, player_index, player_index.shape[0])
    ).astype(jnp.int32)
    occurred = jnp.any(candidate)
    winner = jnp.where(occurred, winner, 0).astype(jnp.int32)
    time_fraction = jnp.where(occurred, crossing[winner], 0.0)
    position = state.ball.position + time_fraction * path_delta
    return BetweenLegsPassageEvent(
        occurred=occurred,
        actor=jnp.where(occurred, winner, NO_PLAYER).astype(jnp.int32),
        time_fraction=time_fraction,
        position=jnp.where(occurred, position, jnp.zeros_like(position)),
    )


def detect_passive_contact(
    state: State,
    path_delta: jax.Array,
    excluded_actor: int | jax.Array = NO_PLAYER,
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
    *,
    player_start_position: jax.Array | None = None,
    player_path_delta: jax.Array | None = None,
    secondary_excluded_actor: int | jax.Array = NO_PLAYER,
    reach: Reach = Reach(),
) -> PassiveContactEvent:
    """Detect the first accidental hit along ball/player relative paths."""

    player_start_position = (
        state.players.position
        if player_start_position is None
        else player_start_position
    )
    player_path_delta = (
        jnp.zeros_like(state.players.position)
        if player_path_delta is None
        else player_path_delta
    )

    body_valid, body_entry, body_normal = _body_collision(
        state,
        path_delta,
        player_start_position=player_start_position,
        player_path_delta=player_path_delta,
        ball=ball_geometry,
        body=body,
    )
    head_valid, head_entry, head_normal = _head_collision(
        state,
        path_delta,
        player_start_position=player_start_position,
        player_path_delta=player_path_delta,
        ball=ball_geometry,
        body=body,
    )

    relative_path_delta = jnp.concatenate(
        (
            path_delta[:2] - player_path_delta,
            jnp.broadcast_to(path_delta[2], player_path_delta[:, :1].shape),
        ),
        axis=-1,
    )
    body_closing = jnp.sum(relative_path_delta * body_normal, axis=-1)
    head_closing = jnp.sum(relative_path_delta * head_normal, axis=-1)
    body_candidate = body_valid & (body_closing < -GEOMETRY_EPS)
    head_candidate = head_valid & (head_closing < -GEOMETRY_EPS)
    use_head = head_candidate & ((~body_candidate) | (head_entry <= body_entry))
    candidate = body_candidate | head_candidate
    entry = jnp.where(use_head, head_entry, body_entry)
    normal = jnp.where(use_head[:, None], head_normal, body_normal)

    # Keep only the necessary cross-substep egress debounce: the
    # latest active actor may let an outward-moving ball leave its body, while
    # an inward return still collides. Once that release episode ends, MOVE and
    # every other intent must accept an ordinary solid-body hit.
    starts_inside = (body_valid & (body_entry <= GEOMETRY_EPS)) | (
        head_valid & (head_entry <= GEOMETRY_EPS)
    )
    player_index = jnp.arange(state.players.position.shape[0])
    last_contact = state.possession.last_contact
    active_provenance = (last_contact.mechanism != MECHANISM_NONE) & (
        last_contact.mechanism != MECHANISM_PASSIVE_BODY
    )
    same_active_episode = (
        active_provenance
        & (player_index == last_contact.actor)
        & (state.players.contact_lock_substeps > 0)
    )
    radial_offset = state.ball.position[:2] - player_start_position
    relative_path_xy = path_delta[:2] - player_path_delta
    moving_outward = jnp.sum(radial_offset * relative_path_xy, axis=-1) >= -GEOMETRY_EPS
    active_egress = same_active_episode & starts_inside & moving_outward
    release_overlap_egress = (
        state.restart_release.active
        & state.restart_release.untouched
        & (player_index == state.restart_release.taker)
        & starts_inside
        & moving_outward
    )
    candidate = candidate & (~(active_egress | release_overlap_egress))

    candidate = (
        candidate
        & state.players.active
        & (player_index != jnp.asarray(excluded_actor, dtype=jnp.int32))
        & (player_index != jnp.asarray(secondary_excluded_actor, dtype=jnp.int32))
        & state.ball.live
    )

    earliest = jnp.min(jnp.where(candidate, entry, jnp.inf))
    entry_tie = candidate & (entry == earliest)
    contact_point = state.ball.position + entry[:, None] * path_delta
    player_contact_position = player_start_position + entry[:, None] * player_path_delta
    distance_squared = jnp.sum(
        (contact_point[:, :2] - player_contact_position) ** 2, axis=-1
    )
    closest = jnp.min(jnp.where(entry_tie, distance_squared, jnp.inf))
    finalist = entry_tie & (distance_squared == closest)
    winner = jnp.argmin(jnp.where(finalist, player_index, player_index.shape[0]))
    occurred = jnp.any(candidate)
    winner = jnp.where(occurred, winner, 0).astype(jnp.int32)
    time_fraction = jnp.where(occurred, entry[winner], 0.0)

    return PassiveContactEvent(
        occurred=occurred,
        actor=jnp.where(occurred, winner, NO_PLAYER).astype(jnp.int32),
        time_fraction=time_fraction,
        normal=normal[winner],
    )


def apply_passive_contact(
    state: State,
    event: PassiveContactEvent,
    *,
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> State:
    """Apply one detected passive-player impulse at the current position."""

    normal_speed = jnp.minimum(jnp.dot(state.ball.velocity, event.normal), 0.0)
    reflected_velocity = state.ball.velocity - (
        (1.0 + body.restitution) * normal_speed * event.normal
    )
    incoming_speed = _norm(state.ball.velocity)
    reflected_speed = _norm(reflected_velocity)
    reflected_velocity = reflected_velocity * jnp.minimum(
        1.0, incoming_speed / (reflected_speed + DIV_EPS)
    )
    supported_by_turf = state.ball.position[2] <= (
        jnp.asarray(ball_geometry.radius, dtype=state.ball.position.dtype)
        + GEOMETRY_EPS
    )
    reflected_velocity = reflected_velocity.at[2].set(
        jnp.where(
            supported_by_turf,
            jnp.maximum(reflected_velocity[2], 0.0),
            reflected_velocity[2],
        )
    )
    ball = state.ball._replace(
        velocity=jnp.where(event.occurred, reflected_velocity, state.ball.velocity),
        spin=state.ball.spin,
    )

    contact = ContactResult(
        actor=jnp.where(
            event.occurred, event.actor, state.possession.last_contact.actor
        ).astype(jnp.int32),
        mechanism=jnp.where(
            event.occurred,
            MECHANISM_PASSIVE_BODY,
            state.possession.last_contact.mechanism,
        ).astype(jnp.int32),
        intent=jnp.where(
            event.occurred, INTENT_MOVE, state.possession.last_contact.intent
        ).astype(jnp.int32),
        outcome=jnp.where(
            event.occurred,
            OUTCOME_DEFLECTION,
            state.possession.last_contact.outcome,
        ).astype(jnp.int32),
        restart_kind=jnp.where(
            event.occurred, RK_NONE, state.possession.last_contact.restart_kind
        ).astype(jnp.int32),
        law11_effect=jnp.where(
            event.occurred,
            LAW11_DEFLECTION_NO_RESET,
            state.possession.last_contact.law11_effect,
        ).astype(jnp.int32),
        kick_applied=jnp.where(
            event.occurred, False, state.possession.last_contact.kick_applied
        ),
        intent_source=jnp.where(
            event.occurred,
            INTENT_SOURCE_NONE,
            state.possession.last_contact.intent_source,
        ).astype(jnp.int32),
    )
    possession = state.possession._replace(
        team=jnp.where(event.occurred, NO_TEAM, state.possession.team).astype(
            jnp.int32
        ),
        player=jnp.where(event.occurred, NO_PLAYER, state.possession.player).astype(
            jnp.int32
        ),
        previous_team=jnp.where(
            event.occurred & (state.possession.team != NO_TEAM),
            state.possession.team,
            state.possession.previous_team,
        ).astype(jnp.int32),
        control_ticks=jnp.where(
            event.occurred, 0, state.possession.control_ticks
        ).astype(jnp.int32),
        last_contact=contact,
    )
    player_count = state.players.position.shape[0]
    safe_actor = jnp.clip(event.actor, 0, player_count - 1)
    restricted_goalkeeper_throw_receipt = (
        state.restart_release.active
        & (state.restart_release.kind == RK_THROWIN)
        & goalkeeper_hand_restricted_mask(state)[safe_actor]
    )
    other_player_contact = (
        event.occurred
        & state.restart_release.active
        & (event.actor != state.restart_release.taker)
        & (~restricted_goalkeeper_throw_receipt)
    )
    continued_restart_release = state.restart_release._replace(
        untouched=jnp.where(
            event.occurred,
            jnp.bool_(False),
            state.restart_release.untouched,
        )
    )
    cleared_restart_release = RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    restart_release = jax.tree_util.tree_map(
        lambda cleared, current: jnp.where(other_player_contact, cleared, current),
        cleared_restart_release,
        continued_restart_release,
    )
    gk_backpass_team = clear_backpass_after_passive_contact(
        state, event.actor, event.occurred
    )
    return state._replace(
        ball=ball,
        possession=possession,
        restart_release=restart_release,
        gk_backpass_team=gk_backpass_team,
    )


def resolve_passive_contact(
    state: State,
    *,
    dt: float | jax.Array,
    excluded_actor: int | jax.Array = NO_PLAYER,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    body: BodyContact = BodyContact(),
    reach: Reach = Reach(),
) -> PassiveContactStep:
    """Resolve one passive hit on the constant-velocity compatibility path."""

    path_delta = state.ball.velocity * dt
    event = detect_passive_contact(
        state,
        path_delta=path_delta,
        excluded_actor=excluded_actor,
        ball_geometry=ball_geometry,
        body=body,
        reach=reach,
    )
    field_exit = _field_exit_fraction(
        state.ball.position,
        path_delta,
        ball=ball_geometry,
        stadium=stadium,
    )
    within_field = event.occurred & (event.time_fraction <= field_exit + GEOMETRY_EPS)
    event = PassiveContactEvent(
        occurred=within_field,
        actor=jnp.where(within_field, event.actor, NO_PLAYER).astype(jnp.int32),
        time_fraction=jnp.where(within_field, event.time_fraction, 0.0),
        normal=jnp.where(within_field, event.normal, jnp.zeros_like(event.normal)),
    )
    impact_position = state.ball.position + event.time_fraction * path_delta
    ball = state.ball._replace(
        position=jnp.where(event.occurred, impact_position, state.ball.position)
    )
    at_impact = state._replace(ball=ball)
    next_state = apply_passive_contact(
        at_impact,
        event,
        ball_geometry=ball_geometry,
        body=body,
    )
    return PassiveContactStep(
        state=next_state,
        occurred=event.occurred,
        actor=event.actor,
        time_fraction=event.time_fraction,
    )
