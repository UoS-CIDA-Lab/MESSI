"""reset 경계의 fixed-shape 초기 formation profile."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from messi.core.constants import STANDARD_PLAYERS_PER_TEAM
from messi.core.randomness import RandomEvent, event_random_key

ROLE_GOALKEEPER = 0
ROLE_CENTRE_BACK = 1
ROLE_FULL_BACK = 2
ROLE_CENTRE_MIDFIELDER = 3
ROLE_WIDE_MIDFIELDER = 4
ROLE_CENTRE_FORWARD = 5
ROLE_WIDE_FORWARD = 6

FORMATION_433 = 0
FORMATION_442 = 1
FORMATION_352 = 2
FORMATION_4231 = 3
FORMATION_343 = 4
FORMATION_COUNT = 5
DEFAULT_FORMATION_IDS = (
    FORMATION_433,
    FORMATION_442,
    FORMATION_352,
    FORMATION_4231,
    FORMATION_343,
)

_ROLES = jnp.asarray(
    (
        (0, 2, 1, 1, 2, 3, 3, 3, 6, 5, 6),
        (0, 2, 1, 1, 2, 4, 3, 3, 4, 5, 5),
        (0, 1, 1, 1, 4, 3, 3, 3, 4, 5, 5),
        (0, 2, 1, 1, 2, 3, 3, 6, 3, 6, 5),
        (0, 1, 1, 1, 4, 3, 3, 4, 6, 5, 6),
    ),
    dtype=jnp.int32,
)

_ANCHORS = jnp.asarray(
    (
        (
            (-47.5, 0.0),
            (-31.0, -23.0),
            (-34.0, -8.0),
            (-34.0, 8.0),
            (-31.0, 23.0),
            (-10.0, -16.0),
            (-14.0, 0.0),
            (-10.0, 16.0),
            (10.0, -24.0),
            (15.0, 0.0),
            (10.0, 24.0),
        ),
        (
            (-47.5, 0.0),
            (-31.0, -23.0),
            (-34.0, -8.0),
            (-34.0, 8.0),
            (-31.0, 23.0),
            (-10.0, -24.0),
            (-13.0, -8.0),
            (-13.0, 8.0),
            (-10.0, 24.0),
            (13.0, -8.0),
            (13.0, 8.0),
        ),
        (
            (-47.5, 0.0),
            (-34.0, -15.0),
            (-36.0, 0.0),
            (-34.0, 15.0),
            (-10.0, -25.0),
            (-14.0, -11.0),
            (-16.0, 0.0),
            (-14.0, 11.0),
            (-10.0, 25.0),
            (13.0, -8.0),
            (13.0, 8.0),
        ),
        (
            (-47.5, 0.0),
            (-31.0, -23.0),
            (-34.0, -8.0),
            (-34.0, 8.0),
            (-31.0, 23.0),
            (-15.0, -9.0),
            (-15.0, 9.0),
            (2.0, -22.0),
            (0.0, 0.0),
            (2.0, 22.0),
            (15.0, 0.0),
        ),
        (
            (-47.5, 0.0),
            (-34.0, -15.0),
            (-36.0, 0.0),
            (-34.0, 15.0),
            (-9.0, -25.0),
            (-12.0, -8.0),
            (-12.0, 8.0),
            (-9.0, 25.0),
            (10.0, -22.0),
            (15.0, 0.0),
            (10.0, 22.0),
        ),
    ),
    dtype=jnp.float32,
)


def realize_initial_formation(
    players_per_team: int,
    *,
    key: jax.Array | None,
    formation_id: jax.Array | None = None,
    goalkeeper: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """팀별 formation ID, player role과 team-frame anchor를 반환한다."""

    if players_per_team != STANDARD_PLAYERS_PER_TEAM and formation_id is not None:
        raise ValueError("formation_id는 players_per_team=11일 때만 지원한다")

    if players_per_team == STANDARD_PLAYERS_PER_TEAM:
        if formation_id is not None:
            formation_id = jnp.asarray(formation_id, dtype=jnp.int32)
        elif key is None:
            formation_id = jnp.zeros(2, dtype=jnp.int32)
        else:
            formation_index = jax.random.randint(
                event_random_key(key, RandomEvent.OPENING_FORMATION),
                (2,),
                0,
                len(DEFAULT_FORMATION_IDS),
            )
            formation_id = jnp.asarray(DEFAULT_FORMATION_IDS, dtype=jnp.int32)[
                formation_index
            ]
        role = _ROLES[formation_id]
        anchor = _ANCHORS[formation_id]
        if goalkeeper is not None:
            goalkeeper = jnp.asarray(goalkeeper, dtype=jnp.bool_).reshape(2, 11)
            for team in range(2):
                goalkeeper_slot = jnp.argmax(goalkeeper[team]).astype(jnp.int32)
                role_zero = role[team, 0]
                role_goalkeeper = role[team, goalkeeper_slot]
                anchor_zero = anchor[team, 0]
                anchor_goalkeeper = anchor[team, goalkeeper_slot]
                role = role.at[team, 0].set(role_goalkeeper)
                role = role.at[team, goalkeeper_slot].set(role_zero)
                anchor = anchor.at[team, 0].set(anchor_goalkeeper)
                anchor = anchor.at[team, goalkeeper_slot].set(anchor_zero)
        return formation_id, role.reshape(-1), anchor.reshape(-1, 2)

    rank = jnp.arange(players_per_team, dtype=jnp.int32)
    denominator = jnp.maximum(players_per_team - 1, 1)
    lateral = -24.0 + 48.0 * rank.astype(jnp.float32) / denominator
    anchor_team = (
        jnp.stack(
            (
                jnp.full_like(lateral, -20.0),
                jnp.where(players_per_team > 1, lateral, 0.0),
            ),
            axis=-1,
        )
        .at[0]
        .set(jnp.asarray((-47.5, 0.0), jnp.float32))
    )
    role_team = jnp.where(rank == 0, ROLE_GOALKEEPER, ROLE_CENTRE_MIDFIELDER)
    role = jnp.tile(role_team, (2, 1))
    anchor = jnp.tile(anchor_team, (2, 1, 1))
    if goalkeeper is not None:
        goalkeeper = jnp.asarray(goalkeeper, dtype=jnp.bool_).reshape(
            2, players_per_team
        )
        for team in range(2):
            goalkeeper_slot = jnp.argmax(goalkeeper[team]).astype(jnp.int32)
            role_zero = role[team, 0]
            role_goalkeeper = role[team, goalkeeper_slot]
            anchor_zero = anchor[team, 0]
            anchor_goalkeeper = anchor[team, goalkeeper_slot]
            role = role.at[team, 0].set(role_goalkeeper)
            role = role.at[team, goalkeeper_slot].set(role_zero)
            anchor = anchor.at[team, 0].set(anchor_goalkeeper)
            anchor = anchor.at[team, goalkeeper_slot].set(anchor_zero)
    return (
        jnp.full(2, -1, dtype=jnp.int32),
        role.reshape(-1),
        anchor.reshape(-1, 2),
    )


def formation_positions(
    roster,
    attack_direction: jax.Array,
    *,
    field_half_extent: jax.Array = (52.5, 34.0),
) -> jax.Array:
    """표준 pitch team-frame anchor를 요청 pitch의 world-frame 위치로 변환한다."""

    anchor_scale = jnp.asarray(
        field_half_extent, dtype=roster.formation_anchor.dtype
    ) / jnp.asarray((52.5, 34.0), dtype=roster.formation_anchor.dtype)
    scaled_anchor = roster.formation_anchor * anchor_scale
    direction = attack_direction[roster.team_id]
    return scaled_anchor * jnp.stack((direction, jnp.ones_like(direction)), axis=-1)


__all__ = [
    "DEFAULT_FORMATION_IDS",
    "FORMATION_343",
    "FORMATION_352",
    "FORMATION_433",
    "FORMATION_442",
    "FORMATION_4231",
    "FORMATION_COUNT",
    "ROLE_CENTRE_BACK",
    "ROLE_CENTRE_FORWARD",
    "ROLE_CENTRE_MIDFIELDER",
    "ROLE_FULL_BACK",
    "ROLE_GOALKEEPER",
    "ROLE_WIDE_FORWARD",
    "ROLE_WIDE_MIDFIELDER",
    "formation_positions",
    "realize_initial_formation",
]
