"""Pure-JAX positioning while a restart is waiting to be taken.

Statutory exclusions are separate from the taker's release pose. The latter
is an environment contact-geometry contract, not an IFAB requirement.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    GEOMETRY_EPS,
    IFAB_MAX_TEAM_PLAYERS,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.state import State, body_angle_from_forward
from footballworld.rules.restart_legality import restart_actor_mask


@dataclass(frozen=True, slots=True)
class RestartPositionLaw:
    """Statutory opponent-clearance distances represented here."""

    ordinary_clearance_m: float = 9.15
    throwin_clearance_m: float = 2.0


class RestartPositioning(NamedTuple):
    position: jax.Array
    facing: jax.Array
    forced: jax.Array
    pinned: jax.Array
    taker_target: jax.Array
    taker_ready: jax.Array


_KICKOFF_PACKING_LANES = 6


def kickoff_packing_minimum_half_extents(
    *,
    ball: Ball,
    stadium: Stadium,
    body: BodyContact,
) -> tuple[float, float]:
    """Return host-side pitch extents required by the fixed kickoff fallback."""

    spacing = body.shoulder_width_m + 0.5 * body.torso_depth_m
    maximum_support = 0.5 * body.shoulder_width_m
    maximum_layer = (IFAB_MAX_TEAM_PLAYERS - 1) // _KICKOFF_PACKING_LANES
    opponent_depth = (
        stadium.center_circle_radius
        + maximum_support
        + spacing
        + maximum_layer * spacing
        + maximum_support
    )
    restart_team_depth = (
        ball.radius
        + maximum_support
        + body.shoulder_width_m
        + spacing
        + maximum_layer * spacing
        + maximum_support
    )
    lateral_extent = 0.5 * (_KICKOFF_PACKING_LANES - 1) * spacing + maximum_support
    return max(opponent_depth, restart_team_depth), lateral_extent


def restart_positioning_pinned(state: State) -> jax.Array:
    """Return structural restart pins without rebuilding a full layout.

    The designated non-hold taker and the defending penalty goalkeeper are
    the only poses that restart enforcement must freeze.  This O(N) mask is
    intentionally separate from the O(N^2) layout audit so a prepared restart
    can enter another control frame without recompiling or rerunning it.
    """

    actor = restart_actor_mask(state)
    kind = state.restart.kind
    hold = kind == RK_GK_HOLD
    penalty = kind == RK_PENALTY
    supported = (
        (kind == RK_KICKOFF)
        | (kind == RK_THROWIN)
        | (kind == RK_GOALKICK)
        | (kind == RK_CORNER)
        | (kind == RK_FREEKICK)
        | (kind == RK_OFFSIDE)
        | penalty
        | hold
    )
    restart_team = jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    goalkeeper_candidate = (
        state.players.active
        & state.players.is_goalkeeper
        & (state.players.team_id != restart_team)
    )
    goalkeeper_index = jnp.argmax(goalkeeper_candidate.astype(jnp.int32))
    penalty_goalkeeper = goalkeeper_candidate & (
        jnp.arange(state.players.position.shape[0], dtype=jnp.int32) == goalkeeper_index
    )
    enabled = jnp.any(actor) & supported
    return enabled & ((actor & (~hold)) | (penalty & penalty_goalkeeper))


def _safe_unit(vector: jax.Array, fallback: jax.Array) -> jax.Array:
    length = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
    unit = vector / jnp.where(length > GEOMETRY_EPS, length, 1.0)[..., None]
    return jnp.where((length > GEOMETRY_EPS)[..., None], unit, fallback)


def capsule_support(
    direction: jax.Array,
    facing: jax.Array,
    *,
    body: BodyContact = BodyContact(),
) -> jax.Array:
    """Support of the 0.50 x 0.20 m horizontal torso capsule."""

    direction = jnp.asarray(direction)
    forward = jnp.stack((jnp.cos(facing), jnp.sin(facing)), axis=-1)
    shoulder = jnp.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    half_core = 0.5 * (body.shoulder_width_m - body.torso_depth_m)
    cap_radius = 0.5 * body.torso_depth_m
    return cap_radius + half_core * jnp.abs(jnp.sum(direction * shoulder, axis=-1))


def _radial(
    position: jax.Array,
    facing: jax.Array,
    subject: jax.Array,
    centre: jax.Array,
    clearance: jax.Array,
    fallback: jax.Array,
    body: BodyContact,
) -> jax.Array:
    relative = position - centre
    distance = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
    direction = _safe_unit(relative, fallback)
    required = clearance + capsule_support(direction, facing, body=body)
    target = centre + required[:, None] * direction
    return jnp.where((subject & (distance < required))[:, None], target, position)


def _in_penalty_area(
    point: jax.Array, goal_sign: jax.Array, stadium: Stadium
) -> jax.Array:
    goal_coordinate = goal_sign * point[..., 0]
    front = stadium.half_length - stadium.penalty_area_length
    return (
        (goal_coordinate >= front)
        & (goal_coordinate <= stadium.half_length)
        & (jnp.abs(point[..., 1]) <= 0.5 * stadium.penalty_area_width)
    )


def _outside_penalty_area(
    position: jax.Array,
    facing: jax.Array,
    subject: jax.Array,
    goal_sign: jax.Array,
    stadium: Stadium,
    body: BodyContact,
) -> jax.Array:
    """Nearest axis translation outside the capsule-expanded penalty area."""

    dtype = position.dtype
    x_axis = jnp.stack(
        (
            jnp.full(position.shape[0], goal_sign, dtype=dtype),
            jnp.zeros(position.shape[0], dtype=dtype),
        ),
        axis=-1,
    )
    support_x = capsule_support(x_axis, facing, body=body)
    support_y = capsule_support(jnp.asarray([0.0, 1.0], dtype=dtype), facing, body=body)
    u = goal_sign * position[:, 0]
    front = jnp.asarray(stadium.half_length - stadium.penalty_area_length, dtype=dtype)
    half_width = jnp.asarray(0.5 * stadium.penalty_area_width, dtype=dtype)
    front_limit = front - support_x
    side_limit = half_width + support_y
    intersects = (
        (u > front_limit)
        & (u < stadium.half_length + support_x)
        & (jnp.abs(position[:, 1]) < side_limit)
    )
    moves = jnp.stack(
        (
            jnp.maximum(u - front_limit, 0.0),
            jnp.maximum(side_limit - position[:, 1], 0.0),
            jnp.maximum(position[:, 1] + side_limit, 0.0),
        ),
        axis=-1,
    )
    choice = jnp.argmin(moves, axis=-1)
    front_target = position.at[:, 0].set(goal_sign * front_limit)
    positive_target = position.at[:, 1].set(side_limit)
    negative_target = position.at[:, 1].set(-side_limit)
    target = jnp.where(
        (choice == 0)[:, None],
        front_target,
        jnp.where((choice == 1)[:, None], positive_target, negative_target),
    )
    return jnp.where((subject & intersects)[:, None], target, position)


def _inside_field(
    position: jax.Array,
    facing: jax.Array,
    subject: jax.Array,
    stadium: Stadium,
    body: BodyContact,
) -> jax.Array:
    dtype = position.dtype
    sx = capsule_support(jnp.asarray([1.0, 0.0], dtype=dtype), facing, body=body)
    sy = capsule_support(jnp.asarray([0.0, 1.0], dtype=dtype), facing, body=body)
    target = jnp.stack(
        (
            jnp.clip(
                position[:, 0],
                -stadium.half_length + sx,
                stadium.half_length - sx,
            ),
            jnp.clip(
                position[:, 1],
                -stadium.half_width + sy,
                stadium.half_width - sy,
            ),
        ),
        axis=-1,
    )
    return jnp.where(subject[:, None], target, position)


def _release_pose(
    state: State,
    restart_direction: jax.Array,
    stadium: Stadium,
    ball: Ball,
    body: BodyContact,
) -> tuple[jax.Array, jax.Array]:
    """Environment-defined pose that makes the restart physically executable."""

    dtype = state.players.position.dtype
    kind = state.restart.kind
    side_y = jnp.where(state.ball.position[1] >= 0.0, 1.0, -1.0)
    side_x = jnp.where(state.ball.position[0] >= 0.0, 1.0, -1.0)
    ordinary = jnp.asarray([restart_direction, 0.0], dtype=dtype)
    # IFAB Law 8 exempts only the taker from the own-half requirement. Keep
    # every other restart's conventional behind-ball approach, but put the
    # kick-off taker just into the opponents' half facing back toward the
    # centre mark. FootballWorld's solid torso capsule otherwise occupies the
    # entire legal passing half-plane: the rule policy must reject every
    # own-half team-mate and can only release its no-receiver fallback toward
    # the opponents. The opposite release pose retains exact capsule clearance
    # and makes a backward kick leave the taker's body instead of crossing it.
    kickoff = -ordinary
    throw = jnp.asarray([0.0, -side_y], dtype=dtype)
    corner = _safe_unit(jnp.asarray([-side_x, -side_y], dtype=dtype), ordinary)
    forward = jnp.where(
        kind == RK_THROWIN,
        throw,
        jnp.where(
            kind == RK_CORNER,
            corner,
            jnp.where(kind == RK_KICKOFF, kickoff, ordinary),
        ),
    )
    angle = jnp.arctan2(forward[1], forward[0])
    gap = jnp.asarray(ball.radius, dtype=dtype) + capsule_support(
        -forward, angle, body=body
    )
    target = state.ball.position[:2] - gap * forward
    throw_target = jnp.asarray(
        [
            state.ball.position[0],
            side_y * (stadium.half_width + gap),
        ],
        dtype=dtype,
    )
    target = jnp.where(kind == RK_THROWIN, throw_target, target)
    outward = -forward
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    next_direction = jnp.where(outward > 0.0, infinity, -infinity)
    target = jnp.where(
        jnp.abs(outward) > GEOMETRY_EPS,
        jnp.nextafter(target, next_direction),
        target,
    )
    taker = jnp.clip(state.restart.taker, 0, state.players.position.shape[0] - 1)
    target = jnp.where(kind == RK_GK_HOLD, state.players.position[taker], target)
    current_angle = body_angle_from_forward(state.players.body_forward[taker])
    angle = jnp.where(kind == RK_GK_HOLD, current_angle, angle)
    return target, angle


def restart_taker_release_pose(
    state: State,
    *,
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> tuple[jax.Array, jax.Array]:
    """Return the legal target and facing for the designated restart taker."""

    team = jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    return _release_pose(state, state.attack_direction[team], stadium, ball, body)


def _kickoff(
    position: jax.Array,
    facing: jax.Array,
    state: State,
    actor: jax.Array,
    restart_team: jax.Array,
    stadium: Stadium,
    body: BodyContact,
) -> jax.Array:
    active = state.players.active
    team_direction = state.attack_direction[state.players.team_id]
    subject = active & (~actor)
    opponent = subject & (state.players.team_id != restart_team)
    axis = jnp.stack((team_direction, jnp.zeros_like(team_direction)), axis=-1)
    support = capsule_support(axis, facing, body=body)
    limit = -support
    illegal = subject & (position[:, 0] * team_direction > limit)
    half_target = position.at[:, 0].set(team_direction * limit)
    positioned = jnp.where(illegal[:, None], half_target, position)
    fallback = jnp.stack((-team_direction, jnp.zeros_like(team_direction)), axis=-1)
    positioned = _radial(
        positioned,
        facing,
        opponent,
        jnp.zeros(2, dtype=position.dtype),
        jnp.full(position.shape[0], stadium.center_circle_radius, dtype=position.dtype),
        fallback,
        body,
    )
    crossed = opponent & (positioned[:, 0] * team_direction > limit)
    y_sign = jnp.where(
        jnp.abs(position[:, 1]) > GEOMETRY_EPS,
        jnp.sign(position[:, 1]),
        team_direction,
    )
    y_limit = stadium.center_circle_radius + capsule_support(
        jnp.asarray([0.0, 1.0], dtype=position.dtype), facing, body=body
    )
    joint_target = jnp.stack((team_direction * limit, y_sign * y_limit), axis=-1)
    return jnp.where(crossed[:, None], joint_target, positioned)


def _free_kick(
    state: State,
    position: jax.Array,
    facing: jax.Array,
    opponent: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> jax.Array:
    """Law 13 distance, own-area exit, and independent goal-line exception."""

    own_goal_sign = -restart_direction
    own_area = _in_penalty_area(state.ball.position[:2], own_goal_sign, stadium)
    positioned = _outside_penalty_area(
        position,
        facing,
        opponent & own_area,
        own_goal_sign,
        stadium,
        body,
    )
    fallback = jnp.stack(
        (
            state.attack_direction[state.players.team_id],
            jnp.zeros(position.shape[0], dtype=position.dtype),
        ),
        axis=-1,
    )
    ordinary = _radial(
        positioned,
        facing,
        opponent,
        state.ball.position[:2],
        jnp.full(position.shape[0], law.ordinary_clearance_m, dtype=position.dtype),
        fallback,
        body,
    )
    sy = capsule_support(
        jnp.asarray([0.0, 1.0], dtype=position.dtype), facing, body=body
    )
    inner = jnp.maximum(0.5 * stadium.goal_width - sy, 0.0)
    goal_line = jnp.stack(
        (
            jnp.full(
                position.shape[0],
                restart_direction * stadium.half_length,
                dtype=position.dtype,
            ),
            jnp.clip(position[:, 1], -inner, inner),
        ),
        axis=-1,
    )
    regular_cost = jnp.sum((ordinary - position) ** 2, axis=-1)
    exception_cost = jnp.sum((goal_line - position) ** 2, axis=-1)
    use_exception = opponent & (~own_area) & (exception_cost < regular_cost)
    return jnp.where(use_exception[:, None], goal_line, ordinary)


def _penalty(
    state: State,
    position: jax.Array,
    facing: jax.Array,
    actor: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    active = state.players.active
    opponent = active & (state.players.team_id != restart_team)
    candidates = opponent & state.players.is_goalkeeper
    gk_index = jnp.argmax(candidates.astype(jnp.int32))
    goalkeeper = candidates & (
        jnp.arange(position.shape[0], dtype=jnp.int32) == gk_index
    )
    outfield = active & (~actor) & (~goalkeeper)
    positioned = _inside_field(position, facing, outfield, stadium, body)

    axis = jnp.stack(
        (
            jnp.full(position.shape[0], restart_direction, dtype=position.dtype),
            jnp.zeros(position.shape[0], dtype=position.dtype),
        ),
        axis=-1,
    )
    sx = capsule_support(axis, facing, body=body)
    mark_u = restart_direction * state.ball.position[0]
    behind_limit = mark_u - sx
    u = restart_direction * positioned[:, 0]
    behind_target = positioned.at[:, 0].set(restart_direction * behind_limit)
    positioned = jnp.where(
        (outfield & (u > behind_limit))[:, None], behind_target, positioned
    )
    fallback = jnp.stack(
        (
            jnp.full(position.shape[0], -restart_direction, dtype=position.dtype),
            jnp.zeros(position.shape[0], dtype=position.dtype),
        ),
        axis=-1,
    )
    positioned = _radial(
        positioned,
        facing,
        outfield,
        state.ball.position[:2],
        jnp.full(position.shape[0], law.ordinary_clearance_m, dtype=position.dtype),
        fallback,
        body,
    )
    positioned = _outside_penalty_area(
        positioned,
        facing,
        outfield,
        restart_direction,
        stadium,
        body,
    )

    gk_facing = jnp.where(restart_direction >= 0.0, jnp.pi, 0.0)
    gk_sy = capsule_support(
        jnp.asarray([0.0, 1.0], dtype=position.dtype),
        gk_facing,
        body=body,
    )
    gk_inner = jnp.maximum(0.5 * stadium.goal_width - gk_sy, 0.0)
    gk_target = jnp.stack(
        (
            jnp.full(
                position.shape[0],
                restart_direction * stadium.half_length,
                dtype=position.dtype,
            ),
            jnp.clip(position[:, 1], -gk_inner, gk_inner),
        ),
        axis=-1,
    )
    positioned = jnp.where(goalkeeper[:, None], gk_target, positioned)
    next_facing = jnp.where(goalkeeper, gk_facing, facing)
    return positioned, next_facing, goalkeeper


def _point_segment_distance(
    point: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    segment = end - start
    denominator = jnp.sum(segment * segment, axis=-1)
    fraction = jnp.clip(
        jnp.sum((point - start) * segment, axis=-1)
        / jnp.where(denominator > GEOMETRY_EPS**2, denominator, 1.0),
        0.0,
        1.0,
    )
    difference = point - (start + fraction[..., None] * segment)
    return jnp.sqrt(jnp.sum(difference * difference, axis=-1))


def _cross(left: jax.Array, right: jax.Array) -> jax.Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _capsule_clearance(
    point: jax.Array,
    facing: jax.Array,
    other_position: jax.Array,
    other_facing: jax.Array,
    body: BodyContact,
) -> jax.Array:
    """Whether one torso capsule is disjoint from every other capsule."""

    half_core = 0.5 * (body.shoulder_width_m - body.torso_depth_m)
    forward = jnp.asarray([jnp.cos(facing), jnp.sin(facing)])
    shoulder = jnp.asarray([-forward[1], forward[0]])
    other_forward = jnp.stack((jnp.cos(other_facing), jnp.sin(other_facing)), axis=-1)
    other_shoulder = jnp.stack((-other_forward[:, 1], other_forward[:, 0]), axis=-1)
    start = point - half_core * shoulder
    end = point + half_core * shoulder
    other_start = other_position - half_core * other_shoulder
    other_end = other_position + half_core * other_shoulder

    distances = jnp.stack(
        (
            _point_segment_distance(start, other_start, other_end),
            _point_segment_distance(end, other_start, other_end),
            _point_segment_distance(other_start, start, end),
            _point_segment_distance(other_end, start, end),
        ),
        axis=-1,
    )
    segment = end - start
    other_segment = other_end - other_start
    relative = other_start - start
    denominator = _cross(segment, other_segment)
    nonparallel = jnp.abs(denominator) > GEOMETRY_EPS
    safe_denominator = jnp.where(nonparallel, denominator, 1.0)
    fraction = _cross(relative, other_segment) / safe_denominator
    other_fraction = _cross(relative, segment) / safe_denominator
    crossing = (
        nonparallel
        & (fraction >= 0.0)
        & (fraction <= 1.0)
        & (other_fraction >= 0.0)
        & (other_fraction <= 1.0)
    )
    segment_squared = jnp.sum(segment * segment)
    projection_0 = jnp.sum(relative * segment, axis=-1) / jnp.where(
        segment_squared > GEOMETRY_EPS**2, segment_squared, 1.0
    )
    projection_1 = projection_0 + jnp.sum(other_segment * segment, axis=-1) / jnp.where(
        segment_squared > GEOMETRY_EPS**2, segment_squared, 1.0
    )
    collinear = (
        (~nonparallel)
        & (jnp.abs(_cross(relative, segment)) <= GEOMETRY_EPS)
        & (
            jnp.maximum(jnp.minimum(projection_0, projection_1), 0.0)
            <= jnp.minimum(jnp.maximum(projection_0, projection_1), 1.0)
        )
    )
    core_distance = jnp.where(crossing | collinear, 0.0, jnp.min(distances, axis=-1))
    return core_distance >= (
        jnp.asarray(body.torso_depth_m, dtype=point.dtype) - GEOMETRY_EPS
    )


def _radial_one(
    point: jax.Array,
    facing: jax.Array,
    subject: jax.Array,
    centre: jax.Array,
    clearance: jax.Array,
    fallback: jax.Array,
    body: BodyContact,
) -> jax.Array:
    relative = point - centre
    distance = jnp.sqrt(jnp.sum(relative * relative))
    direction = _safe_unit(relative, fallback)
    required = clearance + capsule_support(direction, facing, body=body)
    target = centre + required * direction
    return jnp.where(subject & (distance < required), target, point)


def _outside_penalty_area_one(
    point: jax.Array,
    facing: jax.Array,
    subject: jax.Array,
    goal_sign: jax.Array,
    stadium: Stadium,
    body: BodyContact,
) -> jax.Array:
    dtype = point.dtype
    support_x = capsule_support(
        jnp.asarray([goal_sign, 0.0], dtype=dtype), facing, body=body
    )
    support_y = capsule_support(jnp.asarray([0.0, 1.0], dtype=dtype), facing, body=body)
    u = goal_sign * point[0]
    front_limit = (
        jnp.asarray(stadium.half_length - stadium.penalty_area_length, dtype=dtype)
        - support_x
    )
    side_limit = jnp.asarray(0.5 * stadium.penalty_area_width, dtype=dtype) + support_y
    intersects = (
        (u > front_limit)
        & (u < stadium.half_length + support_x)
        & (jnp.abs(point[1]) < side_limit)
    )
    moves = jnp.asarray(
        (
            jnp.maximum(u - front_limit, 0.0),
            jnp.maximum(side_limit - point[1], 0.0),
            jnp.maximum(point[1] + side_limit, 0.0),
        )
    )
    choice = jnp.argmin(moves)
    front = point.at[0].set(goal_sign * front_limit)
    positive = point.at[1].set(side_limit)
    negative = point.at[1].set(-side_limit)
    target = jnp.where(choice == 0, front, jnp.where(choice == 1, positive, negative))
    return jnp.where(subject & intersects, target, point)


def _project_one(
    point: jax.Array,
    facing: jax.Array,
    index: jax.Array,
    state: State,
    actor: jax.Array,
    penalty_gk: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> jax.Array:
    """Project one movable active player onto all current restart constraints."""

    kind = state.restart.kind
    team = state.players.team_id[index]
    active = state.players.active[index]
    is_actor = actor[index]
    is_penalty_gk = penalty_gk[index]
    opponent = active & (team != restart_team)
    dtype = point.dtype

    # With the thrower/corner taker pinned outside, every movable active centre
    # must stay on or inside the touch/goal lines.
    point = jnp.asarray(
        (
            jnp.clip(point[0], -stadium.half_length, stadium.half_length),
            jnp.clip(point[1], -stadium.half_width, stadium.half_width),
        ),
        dtype=dtype,
    )
    fallback = jnp.asarray([state.attack_direction[team], 0.0], dtype=dtype)

    team_direction = state.attack_direction[team]
    half_support = capsule_support(
        jnp.asarray([team_direction, 0.0], dtype=dtype), facing, body=body
    )
    half_limit = -half_support
    kickoff_point = point.at[0].set(
        jnp.where(
            point[0] * team_direction > half_limit,
            team_direction * half_limit,
            point[0],
        )
    )
    kickoff_point = _radial_one(
        kickoff_point,
        facing,
        opponent,
        jnp.zeros(2, dtype=dtype),
        jnp.asarray(stadium.center_circle_radius, dtype=dtype),
        jnp.asarray([-team_direction, 0.0], dtype=dtype),
        body,
    )
    kickoff_point = kickoff_point.at[0].set(
        jnp.where(
            kickoff_point[0] * team_direction > half_limit,
            team_direction * half_limit,
            kickoff_point[0],
        )
    )

    side_y = jnp.where(state.ball.position[1] >= 0.0, 1.0, -1.0)
    throw_point = jnp.asarray(
        [state.ball.position[0], side_y * stadium.half_width], dtype=dtype
    )
    throw_position = _radial_one(
        point,
        facing,
        opponent,
        throw_point,
        jnp.asarray(law.throwin_clearance_m, dtype=dtype),
        fallback,
        body,
    )
    goal_kick_position = _outside_penalty_area_one(
        point, facing, opponent, -restart_direction, stadium, body
    )

    side_x = jnp.where(state.ball.position[0] >= 0.0, 1.0, -1.0)
    corner_flag = jnp.asarray(
        [side_x * stadium.half_length, side_y * stadium.half_width], dtype=dtype
    )
    corner_position = _radial_one(
        point,
        facing,
        opponent,
        corner_flag,
        jnp.asarray(law.ordinary_clearance_m + stadium.corner_arc_radius, dtype=dtype),
        _safe_unit(-corner_flag, fallback),
        body,
    )

    own_goal_sign = -restart_direction
    own_area = _in_penalty_area(state.ball.position[:2], own_goal_sign, stadium)
    free_position = _outside_penalty_area_one(
        point,
        facing,
        opponent & own_area,
        own_goal_sign,
        stadium,
        body,
    )
    ordinary = _radial_one(
        free_position,
        facing,
        opponent,
        state.ball.position[:2],
        jnp.asarray(law.ordinary_clearance_m, dtype=dtype),
        fallback,
        body,
    )
    sy = capsule_support(jnp.asarray([0.0, 1.0], dtype=dtype), facing, body=body)
    inner = jnp.maximum(0.5 * stadium.goal_width - sy, 0.0)
    goal_line = jnp.asarray(
        [restart_direction * stadium.half_length, jnp.clip(point[1], -inner, inner)],
        dtype=dtype,
    )
    use_exception = (
        opponent
        & (~own_area)
        & (jnp.sum((goal_line - point) ** 2) < jnp.sum((ordinary - point) ** 2))
    )
    free_position = jnp.where(use_exception, goal_line, ordinary)

    penalty_outfield = active & (~is_actor) & (~is_penalty_gk)
    sx = capsule_support(
        jnp.asarray([restart_direction, 0.0], dtype=dtype), facing, body=body
    )
    penalty_position = jnp.asarray(
        (
            jnp.clip(
                point[0],
                -stadium.half_length
                + capsule_support(
                    jnp.asarray([1.0, 0.0], dtype=dtype), facing, body=body
                ),
                stadium.half_length
                - capsule_support(
                    jnp.asarray([1.0, 0.0], dtype=dtype), facing, body=body
                ),
            ),
            jnp.clip(
                point[1],
                -stadium.half_width + sy,
                stadium.half_width - sy,
            ),
        ),
        dtype=dtype,
    )
    mark_u = restart_direction * state.ball.position[0]
    behind_limit = mark_u - sx
    penalty_position = penalty_position.at[0].set(
        jnp.where(
            restart_direction * penalty_position[0] > behind_limit,
            restart_direction * behind_limit,
            penalty_position[0],
        )
    )
    penalty_position = _radial_one(
        penalty_position,
        facing,
        penalty_outfield,
        state.ball.position[:2],
        jnp.asarray(law.ordinary_clearance_m, dtype=dtype),
        jnp.asarray([-restart_direction, 0.0], dtype=dtype),
        body,
    )
    penalty_position = _outside_penalty_area_one(
        penalty_position,
        facing,
        penalty_outfield,
        restart_direction,
        stadium,
        body,
    )

    free_kind = (kind == RK_FREEKICK) | (kind == RK_OFFSIDE)
    selected = jnp.where(
        kind == RK_KICKOFF,
        kickoff_point,
        jnp.where(
            kind == RK_THROWIN,
            throw_position,
            jnp.where(
                kind == RK_GOALKICK,
                goal_kick_position,
                jnp.where(
                    kind == RK_CORNER,
                    corner_position,
                    jnp.where(
                        free_kind,
                        free_position,
                        jnp.where(kind == RK_PENALTY, penalty_position, point),
                    ),
                ),
            ),
        ),
    )
    return jnp.where(active & (~is_actor) & (~is_penalty_gk), selected, point)


def _kickoff_global_layout(
    state: State,
    position: jax.Array,
    actor: jax.Array,
    pinned: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
) -> jax.Array:
    """Pack a whole kickoff deterministically when local projection collides."""

    active = state.players.active
    movable = active & (~pinned)
    same_team = state.players.team_id == restart_team
    restart_subject = movable & same_team
    opponent_subject = movable & (~same_team)
    restart_rank = jnp.cumsum(restart_subject.astype(jnp.int32)) - jnp.int32(1)
    opponent_rank = jnp.cumsum(opponent_subject.astype(jnp.int32)) - jnp.int32(1)
    rank = jnp.where(same_team, restart_rank, opponent_rank)
    slot = jnp.mod(rank, jnp.int32(_KICKOFF_PACKING_LANES)).astype(position.dtype)
    layer = (rank // jnp.int32(_KICKOFF_PACKING_LANES)).astype(position.dtype)

    spacing = jnp.asarray(
        body.shoulder_width_m + 0.5 * body.torso_depth_m,
        dtype=position.dtype,
    )
    maximum_support = jnp.asarray(
        0.5 * body.shoulder_width_m,
        dtype=position.dtype,
    )
    actor_depth = jnp.sum(
        jnp.where(
            actor,
            -restart_direction * position[:, 0],
            jnp.zeros(position.shape[0], dtype=position.dtype),
        )
    )
    restart_base_depth = (
        actor_depth + jnp.asarray(body.shoulder_width_m, dtype=position.dtype) + spacing
    )
    opponent_base_depth = (
        jnp.asarray(stadium.center_circle_radius, dtype=position.dtype)
        + maximum_support
        + spacing
    )
    depth = jnp.where(
        same_team,
        restart_base_depth + layer * spacing,
        opponent_base_depth + layer * spacing,
    )
    team_direction = state.attack_direction[state.players.team_id]
    lateral = (
        slot - 0.5 * jnp.asarray(_KICKOFF_PACKING_LANES - 1, position.dtype)
    ) * spacing
    packed = jnp.stack((-team_direction * depth, lateral), axis=-1)
    return jnp.where(movable[:, None], packed, position)


def _free_kick_emergency_layout(
    state: State,
    position: jax.Array,
    pinned: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> jax.Array:
    """Return one geometry-derived all-movable liveness layout."""

    dtype = position.dtype
    active = state.players.active
    movable = active & (~pinned)
    same_team = state.players.team_id == restart_team
    restart_subject = movable & same_team
    opponent_subject = movable & (~same_team)
    restart_rank = jnp.cumsum(restart_subject.astype(jnp.int32)) - jnp.int32(1)
    opponent_rank = jnp.cumsum(opponent_subject.astype(jnp.int32)) - jnp.int32(1)
    rank = jnp.maximum(jnp.where(same_team, restart_rank, opponent_rank), jnp.int32(0))
    lane = jnp.mod(rank, jnp.int32(_KICKOFF_PACKING_LANES)).astype(dtype)
    layer = (rank // jnp.int32(_KICKOFF_PACKING_LANES)).astype(dtype)
    inward = _safe_unit(
        -state.ball.position[:2],
        jnp.asarray([-restart_direction, 0.0], dtype=dtype),
    )
    tangent = jnp.asarray([-inward[1], inward[0]], dtype=dtype)
    spacing = jnp.asarray(body.shoulder_width_m + 0.5 * body.torso_depth_m, dtype=dtype)
    lateral = (
        lane - 0.5 * jnp.asarray(_KICKOFF_PACKING_LANES - 1, dtype=dtype)
    ) * spacing
    maximum_support = jnp.asarray(0.5 * body.shoulder_width_m, dtype=dtype)
    restart_base = jnp.asarray(_KICKOFF_PACKING_LANES - 1, dtype=dtype) * spacing
    own_area = _in_penalty_area(state.ball.position[:2], -restart_direction, stadium)
    opponent_base = (
        jnp.asarray(law.ordinary_clearance_m, dtype=dtype)
        + maximum_support
        + jnp.asarray(2.0, dtype=dtype) * spacing
        + jnp.where(
            own_area,
            jnp.asarray(stadium.penalty_area_length, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
    )
    depth = jnp.where(
        same_team,
        restart_base + layer * spacing,
        opponent_base + layer * spacing,
    )
    packed = (
        state.ball.position[:2] + depth[:, None] * inward + lateral[:, None] * tangent
    )
    return jnp.where(movable[:, None], packed, position)


def _restart_layout_valid(
    state: State,
    layout: jax.Array,
    facing: jax.Array,
    release_target: jax.Array,
    actor: jax.Array,
    pinned: jax.Array,
    penalty_gk: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> jax.Array:
    """Audit one prepared layout without searching alternative layouts."""

    count = layout.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    active = state.players.active
    solve = state.restart.kind != RK_GK_HOLD
    reprojected = jax.vmap(
        lambda point, face, index: _project_one(
            point,
            face,
            index,
            state,
            actor,
            penalty_gk,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )
    )(layout, facing, indices)
    law_ok = jnp.all(
        (~active)
        | pinned
        | (jnp.max(jnp.abs(reprojected - layout), axis=-1) <= 8.0 * GEOMETRY_EPS)
    )
    field_limit = jnp.asarray(
        [stadium.half_length, stadium.half_width], dtype=layout.dtype
    )
    ball_inside = jnp.all(
        jnp.abs(state.ball.position[:2]) <= field_limit + GEOMETRY_EPS
    )
    canonical_actor_pose = (
        jnp.max(jnp.abs(layout - release_target[None, :]), axis=-1)
        <= 8.0 * GEOMETRY_EPS
    )
    external_kind = (
        (state.restart.kind == RK_THROWIN)
        | (state.restart.kind == RK_CORNER)
        | (state.restart.kind == RK_FREEKICK)
        | (state.restart.kind == RK_OFFSIDE)
    )
    external_taker = actor & external_kind & ball_inside & canonical_actor_pose
    field_ok = jnp.all(
        (~active)
        | external_taker
        | (
            (jnp.abs(layout[:, 0]) <= stadium.half_length + GEOMETRY_EPS)
            & (jnp.abs(layout[:, 1]) <= stadium.half_width + GEOMETRY_EPS)
        )
    )
    pair_active = active[:, None] & active[None, :]
    clearance = jax.vmap(
        lambda point, face: _capsule_clearance(point, face, layout, facing, body)
    )(layout, facing)
    pair_ok = jnp.all(clearance | (~pair_active) | jnp.eye(count, dtype=jnp.bool_))
    return (~solve) | (law_ok & field_ok & pair_ok)


def _resolve_restart_constraints(
    state: State,
    position: jax.Array,
    facing: jax.Array,
    release_target: jax.Array,
    actor: jax.Array,
    pinned: jax.Array,
    penalty_gk: jax.Array,
    restart_team: jax.Array,
    restart_direction: jax.Array,
    stadium: Stadium,
    body: BodyContact,
    law: RestartPositionLaw,
) -> tuple[jax.Array, jax.Array]:
    """Choose one of three fixed vector layouts, then fail closed on audit."""

    count = position.shape[0]
    dtype = position.dtype
    indices = jnp.arange(count, dtype=jnp.int32)
    active = state.players.active
    solve = state.restart.kind != RK_GK_HOLD
    pair_active = active[:, None] & active[None, :]
    initial_clearance = jax.vmap(
        lambda point, face: _capsule_clearance(point, face, position, facing, body)
    )(position, facing)
    colliding = jnp.any(
        (~initial_clearance) & pair_active & (~jnp.eye(count, dtype=jnp.bool_)),
        axis=-1,
    )
    outfield = (jnp.abs(position[:, 0]) > stadium.half_length + GEOMETRY_EPS) | (
        jnp.abs(position[:, 1]) > stadium.half_width + GEOMETRY_EPS
    )
    needs_move = solve & active & (~pinned) & (colliding | outfield)
    rank = jnp.cumsum(needs_move.astype(jnp.int32)) - jnp.int32(1)
    packing_spacing = jnp.asarray(
        body.shoulder_width_m + 0.5 * body.torso_depth_m, dtype=dtype
    )
    packing_radius = jnp.asarray(
        law.ordinary_clearance_m + stadium.corner_arc_radius + body.shoulder_width_m,
        dtype=dtype,
    )
    packing_ring_step = jnp.asarray(
        body.shoulder_width_m + body.torso_depth_m, dtype=dtype
    )

    relative = position - state.ball.position[:2]
    radial = jax.vmap(
        lambda vector: _safe_unit(
            vector, jnp.asarray([-restart_direction, 0.0], dtype=dtype)
        )
    )(relative)
    tangent = jnp.stack((-radial[:, 1], radial[:, 0]), axis=-1)
    local_magnitude = ((rank + 1) // 2).astype(dtype)
    local_sign = jnp.where((rank % 2) == 1, 1.0, -1.0).astype(dtype)
    local_sign = jnp.where(rank == 0, 0.0, local_sign)
    local_raw = (
        position + (packing_spacing * local_magnitude * local_sign)[:, None] * tangent
    )

    safe_rank = jnp.maximum(rank, jnp.int32(0))
    slot = jnp.mod(safe_rank, jnp.int32(16)).astype(dtype)
    ring_index = safe_rank // jnp.int32(16)
    ring = ring_index.astype(dtype)
    needs_count = jnp.sum(needs_move.astype(jnp.int32))
    ring_size = jnp.minimum(
        jnp.maximum(needs_count - jnp.int32(16) * ring_index, jnp.int32(0)),
        jnp.int32(16),
    )
    inward = _safe_unit(
        -state.ball.position[:2],
        jnp.asarray([-restart_direction, 0.0], dtype=dtype),
    )
    inward_angle = jnp.arctan2(inward[1], inward[0])
    centred_slot = slot - 0.5 * (ring_size.astype(dtype) - 1.0)
    slot_angle = inward_angle + centred_slot * (jnp.pi / 16.0)
    ring_direction = jnp.stack((jnp.cos(slot_angle), jnp.sin(slot_angle)), axis=-1)
    ring_raw = (
        state.ball.position[:2]
        + (packing_radius + packing_ring_step * ring)[:, None] * ring_direction
    )

    def project_layout(raw_layout):
        projected_layout = jax.vmap(
            lambda point, face, index: _project_one(
                point,
                face,
                index,
                state,
                actor,
                penalty_gk,
                restart_team,
                restart_direction,
                stadium,
                body,
                law,
            )
        )(raw_layout, facing, indices)
        return jnp.where(needs_move[:, None], projected_layout, position)

    local_layout = project_layout(local_raw)
    ring_layout = project_layout(ring_raw)

    set_piece_subject = active & (~pinned)
    set_piece_rank = jnp.cumsum(set_piece_subject.astype(jnp.int32)) - jnp.int32(1)
    set_piece_count = jnp.sum(set_piece_subject.astype(jnp.int32))
    free_kind = (state.restart.kind == RK_FREEKICK) | (state.restart.kind == RK_OFFSIDE)

    def generic_special(_):
        kickoff_layout = _kickoff_global_layout(
            state,
            position,
            actor,
            pinned,
            restart_team,
            restart_direction,
            stadium,
            body,
        )
        return jnp.where(state.restart.kind == RK_KICKOFF, kickoff_layout, position)

    def penalty_special(_):
        support_x = jax.vmap(
            lambda face: capsule_support(
                jnp.asarray([restart_direction, 0.0], dtype=dtype),
                face,
                body=body,
            )
        )(facing)
        penalty_line = (
            restart_direction * state.ball.position[0]
            - jnp.asarray(law.ordinary_clearance_m, dtype=dtype)
            - support_x
        )
        penalty_y = (
            set_piece_rank.astype(dtype) - 0.5 * (set_piece_count.astype(dtype) - 1.0)
        ) * packing_spacing
        penalty_slots = jnp.stack(
            (restart_direction * penalty_line, penalty_y), axis=-1
        )
        return jnp.where(set_piece_subject[:, None], penalty_slots, position)

    def corner_special(_):
        side_x = jnp.where(state.ball.position[0] >= 0.0, 1.0, -1.0)
        side_y = jnp.where(state.ball.position[1] >= 0.0, 1.0, -1.0)
        corner_flag = jnp.asarray(
            [side_x * stadium.half_length, side_y * stadium.half_width],
            dtype=dtype,
        )
        corner_slot = jnp.mod(set_piece_rank, jnp.int32(8)).astype(dtype)
        corner_ring = (set_piece_rank // jnp.int32(8)).astype(dtype)
        corner_angle = (corner_slot + 0.5) * (0.5 * jnp.pi / 8.0)
        corner_radius = packing_radius + packing_ring_step * corner_ring
        corner_direction = jnp.stack(
            (-side_x * jnp.cos(corner_angle), -side_y * jnp.sin(corner_angle)),
            axis=-1,
        )
        corner_slots = corner_flag + corner_radius[:, None] * corner_direction
        return jnp.where(set_piece_subject[:, None], corner_slots, position)

    def emergency_special(_):
        # The local and centred-ring candidates preserve the match shape and
        # therefore win the displacement-cost comparison whenever either is
        # legal. This all-movable layout is only a bounded liveness fallback:
        # unlike a search loop, it adds no candidate count or recurrent state.
        return _free_kick_emergency_layout(
            state,
            position,
            pinned,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )

    special_kind = jnp.where(
        state.restart.kind == RK_PENALTY,
        1,
        jnp.where(
            state.restart.kind == RK_CORNER,
            2,
            jnp.where(free_kind, 3, 0),
        ),
    ).astype(jnp.int32)
    special_layout = jax.lax.switch(
        special_kind,
        (generic_special, penalty_special, corner_special, emergency_special),
        operand=None,
    )
    layouts = jnp.stack(
        (
            local_layout,
            ring_layout,
            special_layout,
        ),
        axis=0,
    )
    layout_valid = jax.vmap(
        lambda layout: _restart_layout_valid(
            state,
            layout,
            facing,
            release_target,
            actor,
            pinned,
            penalty_gk,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )
    )(layouts)
    cost = jnp.sum((layouts - state.players.position[None, :, :]) ** 2, axis=(1, 2))
    choice = jnp.argmin(jnp.where(layout_valid, cost, jnp.inf))
    valid = jnp.any(layout_valid)
    return jnp.where(valid, layouts[choice], position), valid


def prepare_restart_positioning(
    state: State,
    *,
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
    body: BodyContact = BodyContact(),
    law: RestartPositionLaw = RestartPositionLaw(),
    preserve_taker: jax.Array = jnp.bool_(False),
    _resolve_constraints: bool = True,
    _validate_constraints: bool = False,
) -> RestartPositioning:
    """Prepare and continuously enforce every represented restart kind."""

    preserve_taker = jnp.asarray(preserve_taker, dtype=jnp.bool_)
    if preserve_taker.shape != ():
        raise ValueError("preserve_taker must be scalar")
    position = state.players.position
    facing = body_angle_from_forward(state.players.body_forward)
    actor = restart_actor_mask(state)
    coherent = jnp.any(actor)
    kind = state.restart.kind
    kickoff = kind == RK_KICKOFF
    throwin = kind == RK_THROWIN
    goalkick = kind == RK_GOALKICK
    corner = kind == RK_CORNER
    free_kick = (kind == RK_FREEKICK) | (kind == RK_OFFSIDE)
    penalty = kind == RK_PENALTY
    hold = kind == RK_GK_HOLD
    supported = kickoff | throwin | goalkick | corner | free_kick | penalty | hold
    restart_team = jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    restart_direction = state.attack_direction[restart_team]
    opponent = state.players.active & (state.players.team_id != restart_team)
    enabled = coherent & supported
    count = position.shape[0]

    release_target, release_facing = _release_pose(
        state, restart_direction, stadium, ball, body
    )
    taker_target = jnp.where(enabled, release_target, jnp.zeros_like(release_target))
    release_kind = supported & (~hold)
    prepared = jnp.where((actor & release_kind)[:, None], taker_target, position)
    prepared_facing = jnp.where(actor & release_kind, release_facing, facing)
    fallback = jnp.stack(
        (
            state.attack_direction[state.players.team_id],
            jnp.zeros(count, dtype=position.dtype),
        ),
        axis=-1,
    )
    empty_penalty_gk = jnp.zeros(count, dtype=jnp.bool_)

    def unchanged_layout(_):
        return prepared, prepared_facing, empty_penalty_gk

    def kickoff_layout(_):
        return (
            _kickoff(
                prepared,
                prepared_facing,
                state,
                actor,
                restart_team,
                stadium,
                body,
            ),
            prepared_facing,
            empty_penalty_gk,
        )

    def throwin_layout(_):
        side_y = jnp.where(state.ball.position[1] >= 0.0, 1.0, -1.0)
        throw_point = jnp.asarray(
            [state.ball.position[0], side_y * stadium.half_width],
            dtype=position.dtype,
        )
        return (
            _radial(
                prepared,
                prepared_facing,
                opponent,
                throw_point,
                jnp.full(count, law.throwin_clearance_m, dtype=position.dtype),
                fallback,
                body,
            ),
            prepared_facing,
            empty_penalty_gk,
        )

    def goalkick_layout(_):
        return (
            _outside_penalty_area(
                prepared,
                prepared_facing,
                opponent,
                -restart_direction,
                stadium,
                body,
            ),
            prepared_facing,
            empty_penalty_gk,
        )

    def corner_layout(_):
        side_x = jnp.where(state.ball.position[0] >= 0.0, 1.0, -1.0)
        side_y = jnp.where(state.ball.position[1] >= 0.0, 1.0, -1.0)
        corner_flag = jnp.asarray(
            [side_x * stadium.half_length, side_y * stadium.half_width],
            dtype=position.dtype,
        )
        corner_fallback = jnp.broadcast_to(
            _safe_unit(
                -corner_flag,
                jnp.asarray([restart_direction, 0.0], dtype=position.dtype),
            ),
            position.shape,
        )
        return (
            _radial(
                prepared,
                prepared_facing,
                opponent,
                corner_flag,
                jnp.full(
                    count,
                    law.ordinary_clearance_m + stadium.corner_arc_radius,
                    dtype=position.dtype,
                ),
                corner_fallback,
                body,
            ),
            prepared_facing,
            empty_penalty_gk,
        )

    def free_kick_layout(_):
        return (
            _free_kick(
                state,
                prepared,
                prepared_facing,
                opponent,
                restart_direction,
                stadium,
                body,
                law,
            ),
            prepared_facing,
            empty_penalty_gk,
        )

    def penalty_layout(_):
        return _penalty(
            state,
            prepared,
            prepared_facing,
            actor,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )

    layout_kind = jnp.where(
        supported & (~hold),
        jnp.where(kind == RK_OFFSIDE, RK_FREEKICK, kind),
        RK_NONE,
    )
    selected, selected_facing, penalty_gk = jax.lax.switch(
        layout_kind,
        (
            unchanged_layout,
            kickoff_layout,
            throwin_layout,
            goalkick_layout,
            corner_layout,
            free_kick_layout,
            penalty_layout,
        ),
        operand=None,
    )
    pin_taker = actor & supported & (~hold)
    structural_pinned = enabled & (pin_taker | (penalty & penalty_gk))
    if _resolve_constraints:
        resolved, placement_valid = _resolve_restart_constraints(
            state,
            selected,
            selected_facing,
            release_target,
            actor,
            structural_pinned,
            penalty & penalty_gk,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )
    elif _validate_constraints:
        resolved = selected
        placement_valid = _restart_layout_valid(
            state,
            selected,
            selected_facing,
            release_target,
            actor,
            structural_pinned,
            penalty & penalty_gk,
            restart_team,
            restart_direction,
            stadium,
            body,
            law,
        )
    else:
        resolved, placement_valid = selected, jnp.bool_(True)
    enabled = enabled & placement_valid
    preserve_actor = enabled & preserve_taker & actor
    final_position = jnp.where(enabled, resolved, position)
    final_position = jnp.where(preserve_actor[:, None], position, final_position)
    final_facing = jnp.where(enabled, selected_facing, facing)
    final_facing = jnp.where(preserve_actor, facing, final_facing)
    moved = jnp.any(jnp.abs(final_position - position) > GEOMETRY_EPS, axis=-1)
    old_heading = jnp.stack((jnp.cos(facing), jnp.sin(facing)), axis=-1)
    new_heading = jnp.stack((jnp.cos(final_facing), jnp.sin(final_facing)), axis=-1)
    turned = jnp.any(jnp.abs(new_heading - old_heading) > GEOMETRY_EPS, axis=-1)
    forced = enabled & (moved | turned)
    pinned = enabled & ((pin_taker & (~preserve_actor)) | (penalty & penalty_gk))
    return RestartPositioning(
        position=final_position,
        facing=final_facing,
        forced=forced,
        pinned=pinned,
        taker_target=jnp.where(enabled, taker_target, jnp.zeros_like(taker_target)),
        taker_ready=enabled,
    )
