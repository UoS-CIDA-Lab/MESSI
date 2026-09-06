"""Pure-JAX legality gates for actors that may consume a restart.

This staged module deliberately owns no positioning or timing policy.  It
only rejects structurally impossible restart actors and preserves the control
decision boundary: a policy action may consume only the same restart that was
visible when that action was chosen.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_NONE,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.state import State


class RestartFrameGuard(NamedTuple):
    """Fixed-shape restart snapshot carried through one control frame.

    ``opened_this_control`` must be latched by the control-frame orchestrator
    whenever a rules event creates a restart.  The latch is necessary even if
    the new restart has exactly the same kind, team, and taker as the snapshot.
    """

    observed_kind: jax.Array
    observed_team: jax.Array
    observed_taker: jax.Array
    opened_this_control: jax.Array
    entry_restart_active: jax.Array


def _active_restart_kind(kind: jax.Array) -> jax.Array:
    return (kind > RK_NONE) & (kind < RESTART_COUNT)


def restart_actor_mask(state: State) -> jax.Array:
    """Return the sole player allowed to execute a coherent restart.

    The result has fixed shape ``[N]``.  Invalid scalar restart fields yield an
    all-false mask; indexing remains safe under eager execution, JIT, and vmap.
    """

    players = state.players
    player_count = players.team_id.shape[0]
    taker = state.restart.taker
    taker_in_range = (taker >= 0) & (taker < player_count)
    safe_taker = jnp.clip(taker, 0, player_count - 1)
    valid_team = (state.restart.team == TEAM_0) | (state.restart.team == TEAM_1)
    active_taker = players.active[safe_taker]
    same_team = players.team_id[safe_taker] == state.restart.team
    role_eligible = (state.restart.kind != RK_GK_HOLD) | (
        players.is_goalkeeper[safe_taker]
    )
    coherent = (
        _active_restart_kind(state.restart.kind)
        & valid_team
        & taker_in_range
        & active_taker
        & same_team
        & role_eligible
    )
    return coherent & (jnp.arange(player_count, dtype=jnp.int32) == taker)


def goalkeeper_restart_boundary_ready(state: State) -> jax.Array:
    """Whether play may leave a dead restart with both teams represented."""

    players = state.players
    goalkeeper_ready = jnp.all(
        jnp.stack(
            [
                jnp.any(
                    players.active & players.is_goalkeeper & (players.team_id == team)
                )
                for team in (TEAM_0, TEAM_1)
            ]
        )
    )
    return goalkeeper_ready | (state.restart.kind == RK_GK_HOLD)


def begin_restart_frame(state: State) -> RestartFrameGuard:
    """Capture the restart observable when the current action was chosen."""

    return RestartFrameGuard(
        observed_kind=state.restart.kind.astype(jnp.int32),
        observed_team=state.restart.team.astype(jnp.int32),
        observed_taker=state.restart.taker.astype(jnp.int32),
        opened_this_control=jnp.bool_(False),
        entry_restart_active=_active_restart_kind(state.restart.kind),
    )


def mark_restart_opened(
    guard: RestartFrameGuard,
    opened: jax.Array,
) -> RestartFrameGuard:
    """Latch a restart-creation event for the rest of the control frame."""

    return guard._replace(
        opened_this_control=(
            guard.opened_this_control | jnp.asarray(opened, dtype=jnp.bool_)
        )
    )


def restart_visible_to_action(
    state: State,
    guard: RestartFrameGuard,
) -> jax.Array:
    """Whether ``state.restart`` is the restart observed by this action."""

    same_snapshot = (
        (state.restart.kind == guard.observed_kind)
        & (state.restart.team == guard.observed_team)
        & (state.restart.taker == guard.observed_taker)
    )
    return (
        guard.entry_restart_active
        & (~guard.opened_this_control)
        & _active_restart_kind(state.restart.kind)
        & same_snapshot
    )


def restart_release_actor_mask(
    state: State,
    guard: RestartFrameGuard,
) -> jax.Array:
    """Return actors both structurally legal and visible to the action."""

    return (
        restart_actor_mask(state)
        & restart_visible_to_action(state, guard)
        & goalkeeper_restart_boundary_ready(state)
    )
