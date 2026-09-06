"""Single pure-JAX source of truth for canonical restart ball positions."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    RK_CORNER,
    RK_FREEKICK,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)


@dataclass(frozen=True, slots=True)
class RestartSpotLaw:
    """Only the statutory distance needed to locate a restart ball."""

    penalty_mark_distance_m: float = 11.0


def _indirect_goal_area_position(
    position: jax.Array,
    restart_team: jax.Array,
    attack_direction: jax.Array,
    *,
    stadium: Stadium,
    ball: Ball,
) -> jax.Array:
    """Apply the attacking indirect-FK goal-area line exception."""

    dtype = position.dtype
    safe_team = jnp.clip(restart_team, TEAM_0, TEAM_1)
    direction = attack_direction[safe_team]
    half_length = jnp.asarray(stadium.half_length, dtype=dtype)
    goal_area_length = jnp.asarray(stadium.goal_area_length, dtype=dtype)
    goal_area_half_width = jnp.asarray(0.5 * stadium.goal_area_width, dtype=dtype)
    depth_from_attacking_goal = half_length - direction * position[0]
    inside_opponent_goal_area = (
        (depth_from_attacking_goal >= 0.0)
        & (depth_from_attacking_goal <= goal_area_length)
        & (jnp.abs(position[1]) <= goal_area_half_width)
    )
    goal_area_line_x = direction * (half_length - goal_area_length)
    return (
        position.at[0]
        .set(
            jnp.where(
                inside_opponent_goal_area,
                goal_area_line_x,
                position[0],
            )
        )
        .at[2]
        .set(jnp.asarray(ball.radius, dtype=dtype))
    )


def canonical_restart_spot(
    kind: jax.Array,
    team: jax.Array,
    incident_position: jax.Array,
    attack_direction: jax.Array,
    *,
    indirect: jax.Array = jnp.bool_(False),
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
    law: RestartSpotLaw = RestartSpotLaw(),
) -> jax.Array:
    """Return the deterministic ball centre for every represented restart.

    ``incident_position`` is the actual boundary crossing or offence point.
    Valid restart teams are guaranteed by the separate actor-legality gate;
    clipping here keeps traced gathers safe for reconstructed invalid states.

    The throw-in and corner zero-axis tie convention intentionally remains
    positive, matching FootballWorld's current boundary transition bit for bit.
    """

    incident_position = jnp.asarray(incident_position)
    dtype = incident_position.dtype
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    direction = attack_direction[safe_team]
    half_length = jnp.asarray(stadium.half_length, dtype=dtype)
    half_width = jnp.asarray(stadium.half_width, dtype=dtype)
    radius = jnp.asarray(ball.radius, dtype=dtype)
    x_sign = jnp.where(incident_position[0] >= 0.0, 1.0, -1.0)
    y_sign = jnp.where(incident_position[1] >= 0.0, 1.0, -1.0)

    ordinary = incident_position.at[2].set(radius)
    centre = jnp.asarray([0.0, 0.0, radius], dtype=dtype)
    throw_in = jnp.asarray(
        [
            jnp.clip(incident_position[0], -half_length, half_length),
            y_sign * half_width,
            radius,
        ],
        dtype=dtype,
    )
    corner = jnp.asarray(
        [x_sign * half_length, y_sign * half_width, radius], dtype=dtype
    )
    goal_kick = jnp.asarray(
        [
            -direction * (half_length - 0.5 * stadium.goal_area_length),
            0.0,
            radius,
        ],
        dtype=dtype,
    )
    penalty = jnp.asarray(
        [
            direction * (half_length - law.penalty_mark_distance_m),
            0.0,
            radius,
        ],
        dtype=dtype,
    )
    indirect_free_kick = _indirect_goal_area_position(
        ordinary,
        safe_team,
        attack_direction,
        stadium=stadium,
        ball=ball,
    )
    free_kick = jnp.where(indirect, indirect_free_kick, ordinary)

    return jnp.where(
        kind == RK_KICKOFF,
        centre,
        jnp.where(
            kind == RK_THROWIN,
            throw_in,
            jnp.where(
                kind == RK_GOALKICK,
                goal_kick,
                jnp.where(
                    kind == RK_CORNER,
                    corner,
                    jnp.where(
                        kind == RK_PENALTY,
                        penalty,
                        jnp.where(
                            kind == RK_OFFSIDE,
                            indirect_free_kick,
                            jnp.where(kind == RK_FREEKICK, free_kick, ordinary),
                        ),
                    ),
                ),
            ),
        ),
    )
