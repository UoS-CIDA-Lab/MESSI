"""Pure JAX detection and classification of ball boundary crossings."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    GEOMETRY_EPS,
    NO_TEAM,
    TEAM_0,
    TEAM_1,
)

CROSSING_NONE = 0
CROSSING_GOAL_LINE = 1
CROSSING_TOUCH_LINE = 2


class BoundaryCrossing(NamedTuple):
    """First complete-ball boundary crossing along one straight segment."""

    occurred: jax.Array
    axis: jax.Array
    time_fraction: jax.Array
    position: jax.Array
    through_goal: jax.Array


class BoundaryEvent(NamedTuple):
    """A classified goal or out-of-play event without state mutation."""

    occurred: jax.Array
    kind: jax.Array
    team: jax.Array
    scoring_team: jax.Array
    time_fraction: jax.Array
    position: jax.Array


def _crossing_fraction(
    before: jax.Array,
    after: jax.Array,
    boundary: jax.Array,
) -> jax.Array:
    delta = after - before
    target = jnp.where(after >= 0.0, boundary, -boundary)
    raw = (target - before) / jnp.where(
        jnp.abs(delta) > GEOMETRY_EPS,
        delta,
        1.0,
    )
    started_out = jnp.abs(before) > boundary
    return jnp.where(started_out, 0.0, jnp.clip(raw, 0.0, 1.0))


def detect_boundary_crossing(
    position_before: jax.Array,
    position_after: jax.Array,
    ball_live: jax.Array,
    *,
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
) -> BoundaryCrossing:
    """Detect the earliest line crossed by the complete ball.

    The inputs are the endpoints of one collision-free path segment. A goal
    line wins an exact corner tie; otherwise the smaller crossing fraction
    determines which boundary owns the event.
    """

    position_before = jnp.asarray(position_before)
    position_after = jnp.asarray(position_after)
    line_x = jnp.asarray(
        stadium.half_length + ball.radius,
        dtype=position_after.dtype,
    )
    line_y = jnp.asarray(
        stadium.half_width + ball.radius,
        dtype=position_after.dtype,
    )

    x_out = jnp.abs(position_after[0]) > line_x
    y_out = jnp.abs(position_after[1]) > line_y
    x_fraction = _crossing_fraction(position_before[0], position_after[0], line_x)
    y_fraction = _crossing_fraction(position_before[1], position_after[1], line_y)
    goal_line_first = x_out & ((~y_out) | (x_fraction <= y_fraction))
    touch_line_first = y_out & ((~x_out) | (y_fraction < x_fraction))
    occurred = jnp.asarray(ball_live, dtype=bool) & (goal_line_first | touch_line_first)
    fraction = jnp.where(
        goal_line_first,
        x_fraction,
        jnp.where(touch_line_first, y_fraction, 0.0),
    )

    crossing_position = position_before + fraction * (position_after - position_before)
    crossing_position = crossing_position.at[0].set(
        jnp.where(
            goal_line_first,
            jnp.where(position_after[0] >= 0.0, line_x, -line_x),
            crossing_position[0],
        )
    )
    crossing_position = crossing_position.at[1].set(
        jnp.where(
            touch_line_first,
            jnp.where(position_after[1] >= 0.0, line_y, -line_y),
            crossing_position[1],
        )
    )

    goal_half_clear = 0.5 * stadium.goal_width - ball.radius
    crossbar_clear = stadium.goal_height - ball.radius
    through_goal = (
        occurred
        & goal_line_first
        & (jnp.abs(crossing_position[1]) <= goal_half_clear)
        & (crossing_position[2] <= crossbar_clear)
    )
    axis = jnp.where(
        occurred,
        jnp.where(
            goal_line_first,
            CROSSING_GOAL_LINE,
            CROSSING_TOUCH_LINE,
        ),
        CROSSING_NONE,
    ).astype(jnp.int32)
    return BoundaryCrossing(
        occurred=occurred,
        axis=axis,
        time_fraction=jnp.where(occurred, fraction, 0.0),
        position=jnp.where(
            occurred, crossing_position, jnp.zeros_like(crossing_position)
        ),
        through_goal=through_goal,
    )


def classify_boundary_event(
    crossing: BoundaryCrossing,
    last_touch_team: jax.Array,
    possession_team: jax.Array,
    attack_direction: jax.Array,
    kickoff_team: jax.Array,
) -> BoundaryEvent:
    """Classify one crossing using valid or deterministic team provenance."""

    team_plus = jnp.where(attack_direction[TEAM_0] > 0.0, TEAM_0, TEAM_1).astype(
        jnp.int32
    )
    team_minus = (TEAM_1 - team_plus).astype(jnp.int32)
    goal_sign = jnp.where(crossing.position[0] >= 0.0, 1.0, -1.0)
    scoring_team = jnp.where(goal_sign > 0.0, team_plus, team_minus).astype(jnp.int32)
    defending_team = (TEAM_1 - scoring_team).astype(jnp.int32)

    valid_kickoff = (kickoff_team == TEAM_0) | (kickoff_team == TEAM_1)
    centre_fallback = jnp.where(valid_kickoff, kickoff_team, TEAM_0).astype(jnp.int32)
    end_attacker = jnp.where(
        crossing.position[0] > GEOMETRY_EPS,
        team_plus,
        jnp.where(
            crossing.position[0] < -GEOMETRY_EPS,
            team_minus,
            centre_fallback,
        ),
    ).astype(jnp.int32)
    valid_last_touch = (last_touch_team == TEAM_0) | (last_touch_team == TEAM_1)
    valid_possession = (possession_team == TEAM_0) | (possession_team == TEAM_1)
    effective_last_touch = jnp.where(
        valid_last_touch,
        last_touch_team,
        jnp.where(valid_possession, possession_team, end_attacker),
    ).astype(jnp.int32)
    restart_team = (TEAM_1 - effective_last_touch).astype(jnp.int32)

    goal = crossing.occurred & crossing.through_goal
    goal_line_out = crossing.occurred & (crossing.axis == CROSSING_GOAL_LINE) & (~goal)
    touch_line_out = crossing.occurred & (crossing.axis == CROSSING_TOUCH_LINE)
    corner = goal_line_out & (effective_last_touch == defending_team)
    goal_kick = goal_line_out & (~corner)
    kind = jnp.where(
        goal,
        BALL_EVENT_GOAL,
        jnp.where(
            corner,
            BALL_EVENT_CORNER,
            jnp.where(
                goal_kick,
                BALL_EVENT_GOALKICK,
                jnp.where(
                    touch_line_out,
                    BALL_EVENT_THROWIN,
                    BALL_EVENT_NONE,
                ),
            ),
        ),
    ).astype(jnp.int32)
    event_team = jnp.where(
        goal,
        scoring_team,
        jnp.where(corner | goal_kick | touch_line_out, restart_team, NO_TEAM),
    ).astype(jnp.int32)
    return BoundaryEvent(
        occurred=crossing.occurred,
        kind=kind,
        team=event_team,
        scoring_team=jnp.where(goal, scoring_team, NO_TEAM).astype(jnp.int32),
        time_fraction=crossing.time_fraction,
        position=crossing.position,
    )
