"""Observation-side hints for the six public action intents.

The returned mask is deliberately not an engine-legality oracle. Exact
contact depends on continuous controls and substep physics absent from an
``Observation``. Dynamics remain authoritative and fail closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from footballworld.environment.observation import Observation

if TYPE_CHECKING:
    from footballworld.environment.normalization import NormalizedObservation

ACTION_INTENT_NAMES = (
    "MOVE",
    "CONTROL",
    "PASS",
    "SHOT",
    "CLEAR",
    "CHALLENGE",
)


def _observer_slot(value: jax.Array, observer: jax.Array) -> jax.Array:
    """Read the observer's own roster slot for scalar or batched views."""

    player_count = value.shape[-1]
    safe_observer = jnp.clip(observer, 0, player_count - 1)
    return jnp.take_along_axis(value, safe_observer[..., None], axis=-1)[..., 0]


def intent_availability_hint(
    observation: Observation | NormalizedObservation,
) -> jax.Array:
    """Return a policy-side ``[..., 6]`` intent-selection hint.

    Columns are exactly ``MOVE, CONTROL, PASS, SHOT, CLEAR, CHALLENGE``.
    ``False`` means the intent is unavailable to this observation-conditioned
    policy in this frame. ``True`` means only that the policy may attempt it;
    it does not promise contact or success. Exact physics and rules remain
    authoritative.
    """

    observer = jnp.asarray(observation.self_state.player_index, dtype=jnp.int32)
    player_count = observation.players.on_pitch.shape[-1]
    observer_valid = (observer >= 0) & (observer < player_count)
    observer_valid = observer_valid & jnp.asarray(observation.valid, dtype=jnp.bool_)
    contact_may_occur = _observer_slot(
        observation.players.contact_may_occur_this_frame,
        observer,
    )
    restart_taker = _observer_slot(observation.players.restart_taker, observer)
    observer_active = (
        observer_valid
        & _observer_slot(observation.players.on_pitch, observer)
        & (~_observer_slot(observation.players.sent_off, observer))
    )
    observable_candidate = (
        observer_active & contact_may_occur & observation.ball.visible
    )

    kind = observation.restart.kind
    open_play = kind == RK_NONE
    valid_restart = (kind > RK_NONE) & (kind < RESTART_COUNT)
    open_contact = observable_candidate & open_play
    restart_release = observable_candidate & valid_restart & restart_taker
    ordinary_kick_restart = (
        restart_release & (kind != RK_THROWIN) & (kind != RK_GK_HOLD)
    )

    allowed = jnp.zeros(jnp.shape(kind) + (ACTION_INTENT_COUNT,), dtype=jnp.bool_)
    allowed = allowed.at[..., INTENT_MOVE].set(observer_active)
    allowed = allowed.at[..., INTENT_CONTROL].set(open_contact)
    allowed = allowed.at[..., INTENT_PASS].set(open_contact | restart_release)
    allowed = allowed.at[..., INTENT_SHOT].set(open_contact | ordinary_kick_restart)
    allowed = allowed.at[..., INTENT_CLEAR].set(open_contact | ordinary_kick_restart)
    allowed = allowed.at[..., INTENT_CHALLENGE].set(open_contact)
    return allowed


__all__ = ["ACTION_INTENT_NAMES", "intent_availability_hint"]
