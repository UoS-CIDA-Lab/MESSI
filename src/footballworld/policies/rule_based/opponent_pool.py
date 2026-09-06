"""Fixed-width opponent gathers for observation-only policy calculations."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    IFAB_MAX_TEAM_PLAYERS,
    TEAM_0,
    TEAM_1,
)
from footballworld.environment.observation import RosterMetadata
from footballworld.policies.rule_based.context import RulePolicyContext


class TeamSlotTable(NamedTuple):
    """Roster indices and padding validity for both fixed-width teams."""

    index: jax.Array
    valid: jax.Array


class OpponentPool(NamedTuple):
    """One compact opponent axis for every independent observer row.

    ``valid`` distinguishes real roster slots from padding. ``available`` also
    applies that observer's visibility, on-pitch, and dismissal masks. The
    remaining arrays have a fixed trailing opponent axis of length eleven.
    """

    roster_index: jax.Array
    valid: jax.Array
    available: jax.Array
    position: jax.Array
    velocity: jax.Array
    max_speed: jax.Array


def build_team_slot_table(team_id: jax.Array) -> TeamSlotTable:
    """Build a reusable ``[2, 11]`` roster-slot table.

    Call this when roster metadata is created or refreshed, rather than in
    every policy frame. Teams with fewer than eleven players retain invalid
    padded columns, so a filled index can never become an accidental duplicate.
    """

    team_id = jnp.asarray(team_id, dtype=jnp.int32)
    if team_id.ndim != 1:
        raise ValueError("team_id must have shape (players,)")
    width = IFAB_MAX_TEAM_PLAYERS
    columns = jnp.arange(width, dtype=jnp.int32)

    def one(team: int) -> tuple[jax.Array, jax.Array]:
        member = team_id == jnp.int32(team)
        count = jnp.sum(member, dtype=jnp.int32)
        index = jnp.nonzero(member, size=width, fill_value=0)[0].astype(jnp.int32)
        return index, columns < count

    index_0, valid_0 = one(TEAM_0)
    index_1, valid_1 = one(TEAM_1)
    return TeamSlotTable(
        index=jnp.stack((index_0, index_1), axis=0),
        valid=jnp.stack((valid_0, valid_1), axis=0),
    )


def gather_opponent_pool(
    context: RulePolicyContext,
    roster: RosterMetadata,
    table: TeamSlotTable,
) -> OpponentPool:
    """Gather at most eleven opponents without combining observer rows.

    The returned ``position``, ``velocity``, ``available``, and ``max_speed``
    arrays can replace the dense opponent arguments to ``lane_completion`` and
    ``reception_evaluation``. Candidate/receiver axes remain unchanged.
    """

    observers = context.self_index.shape[0]
    players = roster.team_id.shape[0]
    width = IFAB_MAX_TEAM_PLAYERS
    if context.player_position.shape != (observers, players, 2):
        raise ValueError("context player positions must have shape (O, P, 2)")
    if context.player_velocity.shape != (observers, players, 2):
        raise ValueError("context player velocities must have shape (O, P, 2)")
    if context.opponent.shape != (observers, players):
        raise ValueError("context opponent mask must have shape (O, P)")
    if roster.max_speed.shape != (players,):
        raise ValueError("roster max_speed must have shape (P,)")
    if table.index.shape != (2, width) or table.valid.shape != (2, width):
        raise ValueError("team slot table must have shape (2, 11)")

    opponent_team = jnp.where(
        context.self_team == TEAM_0,
        jnp.int32(TEAM_1),
        jnp.int32(TEAM_0),
    )
    roster_index = table.index[opponent_team]
    valid = table.valid[opponent_team]
    row = jnp.arange(observers, dtype=jnp.int32)[:, None]
    position = context.player_position[row, roster_index]
    velocity = context.player_velocity[row, roster_index]
    max_speed = roster.max_speed[roster_index]
    available = valid & context.opponent[row, roster_index]

    return OpponentPool(
        roster_index=roster_index,
        valid=valid,
        available=available,
        position=jnp.where(valid[..., None], position, jnp.float32(0.0)),
        velocity=jnp.where(valid[..., None], velocity, jnp.float32(0.0)),
        max_speed=jnp.where(valid, max_speed, jnp.float32(0.0)),
    )


__all__ = [
    "OpponentPool",
    "TeamSlotTable",
    "build_team_slot_table",
    "gather_opponent_pool",
]
