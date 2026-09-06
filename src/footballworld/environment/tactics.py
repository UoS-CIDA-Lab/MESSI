"""Shared formation semantics without policy or physics ownership."""

import jax
import jax.numpy as jnp

from footballworld.core.constants import TEAM_0, TEAM_1

ROLE_GOALKEEPER = 0
ROLE_CENTRE_BACK = 1
ROLE_FULL_BACK = 2
ROLE_CENTRE_MIDFIELDER = 3
ROLE_WIDE_MIDFIELDER = 4
ROLE_CENTRE_FORWARD = 5
ROLE_WIDE_FORWARD = 6
ROLE_COUNT = 7

_DEPTH_TOLERANCE_M = 0.5
_WIDTH_TIE_TOLERANCE_M = 1.0e-3
_WIDE_MINIMUM_LINE_SIZE = 3
_WIDE_SLOTS_PER_LINE = 2


def classify_formation_roles(
    formation_anchor: jax.Array,
    team_id: jax.Array,
    is_goalkeeper: jax.Array,
) -> jax.Array:
    """Classify slot roles from attacking-frame anchors for both teams."""

    player_count = formation_anchor.shape[0]
    index = jnp.arange(player_count, dtype=jnp.int32)
    depth = formation_anchor[:, 0]
    width = jnp.abs(formation_anchor[:, 1])
    goalkeeper = jnp.asarray(is_goalkeeper, dtype=jnp.bool_)
    outfield = ~goalkeeper
    band = jnp.zeros(player_count, dtype=jnp.int32)
    wide = jnp.zeros(player_count, dtype=jnp.bool_)
    for team in (TEAM_0, TEAM_1):
        same_team = outfield & (team_id == team)
        same_depth = (
            same_team[:, None]
            & same_team[None, :]
            & (jnp.abs(depth[:, None] - depth[None, :]) <= _DEPTH_TOLERANCE_M)
        )
        has_earlier_peer = jnp.any(
            same_depth & (index[None, :] < index[:, None]), axis=1
        )
        representative = same_team & (~has_earlier_peer)
        deeper_representative = (
            same_team[:, None]
            & representative[None, :]
            & (depth[None, :] < depth[:, None] - _DEPTH_TOLERANCE_M)
        )
        team_band = jnp.clip(jnp.sum(deeper_representative, axis=1), 0, 2).astype(
            jnp.int32
        )
        band = jnp.where(same_team, team_band, band)
        for line in range(3):
            line_member = same_team & (team_band == line)
            wider_count = jnp.sum(
                line_member[:, None]
                & line_member[None, :]
                & (width[None, :] > width[:, None] + _WIDTH_TIE_TOLERANCE_M),
                axis=1,
            )
            wide = wide | (
                line_member
                & (jnp.sum(line_member) >= _WIDE_MINIMUM_LINE_SIZE)
                & (wider_count < _WIDE_SLOTS_PER_LINE)
            )
    role = jnp.where(
        band == 0,
        jnp.where(wide, ROLE_FULL_BACK, ROLE_CENTRE_BACK),
        jnp.where(
            band == 2,
            jnp.where(wide, ROLE_WIDE_FORWARD, ROLE_CENTRE_FORWARD),
            jnp.where(wide, ROLE_WIDE_MIDFIELDER, ROLE_CENTRE_MIDFIELDER),
        ),
    )
    return jnp.where(goalkeeper, ROLE_GOALKEEPER, role).astype(jnp.int32)


__all__ = [
    "ROLE_CENTRE_BACK",
    "ROLE_CENTRE_FORWARD",
    "ROLE_CENTRE_MIDFIELDER",
    "ROLE_COUNT",
    "ROLE_FULL_BACK",
    "ROLE_GOALKEEPER",
    "ROLE_WIDE_FORWARD",
    "ROLE_WIDE_MIDFIELDER",
    "classify_formation_roles",
]
