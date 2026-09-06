"""Structural intent legality for one policy control phase.

This module owns only phase and restart-law constraints.  Contact geometry,
ball speed, and player recovery remain authoritative in the dynamics layer.
"""

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_NONE,
    RK_THROWIN,
)
from footballworld.core.contact import (
    ACTION_INTENT_COUNT,
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
)
from footballworld.core.state import State
from footballworld.rules.restart_legality import restart_actor_mask


def restart_intent_allowed(
    state: State,
    release_allowed: jax.Array,
) -> jax.Array:
    """Return the fixed ``[players, 6]`` structural intent mask.

    ``release_allowed`` is the caller-owned timing/visibility mask for the
    current control frame.  It does not replace the designated-taker check.
    Invalid or inactive slots retain only ``MOVE`` as a fail-closed action.
    """

    player_count = state.players.position.shape[0]
    release_allowed = jnp.asarray(release_allowed, dtype=jnp.bool_)
    if release_allowed.shape != (player_count,):
        raise ValueError(
            "release_allowed must have shape "
            f"({player_count},), got {release_allowed.shape}"
        )

    kind = state.restart.kind
    open_play = kind == RK_NONE
    valid_restart = (kind > RK_NONE) & (kind < RESTART_COUNT)
    active_live = state.players.active & state.ball.live
    open_contact = open_play & active_live

    legal_taker = valid_restart & restart_actor_mask(state) & release_allowed
    pass_restart = legal_taker
    ordinary_kick_restart = legal_taker & (kind != RK_THROWIN) & (kind != RK_GK_HOLD)

    control = open_contact
    pass_intent = open_contact | pass_restart
    shot = open_contact | ordinary_kick_restart
    clear = open_contact | ordinary_kick_restart
    tackle = open_contact

    allowed = jnp.zeros((player_count, ACTION_INTENT_COUNT), dtype=jnp.bool_)
    allowed = allowed.at[:, INTENT_MOVE].set(True)
    allowed = allowed.at[:, INTENT_CONTROL].set(control)
    allowed = allowed.at[:, INTENT_PASS].set(pass_intent)
    allowed = allowed.at[:, INTENT_SHOT].set(shot)
    allowed = allowed.at[:, INTENT_CLEAR].set(clear)
    allowed = allowed.at[:, INTENT_CHALLENGE].set(tackle)
    return allowed


__all__ = ["restart_intent_allowed"]
