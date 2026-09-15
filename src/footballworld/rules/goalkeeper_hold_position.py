"""Shared physical pose for a ball held by a goalkeeper."""

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball
from footballworld.core.state import State


def goalkeeper_held_ball_position(
    state: State,
    goalkeeper: jax.Array,
    *,
    ball: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> jax.Array:
    """Place the ball one radius in front of the holder's torso centre."""

    player_count = state.players.position.shape[0]
    safe_goalkeeper = jnp.clip(goalkeeper, 0, player_count - 1)
    held_xy = state.players.position[safe_goalkeeper] + state.players.body_forward[
        safe_goalkeeper
    ] * jnp.asarray(ball.radius, dtype=state.ball.position.dtype)
    return jnp.concatenate(
        (
            held_xy,
            body.torso_top_height(state.players.height[safe_goalkeeper])[None],
        )
    ).astype(state.ball.position.dtype)


__all__ = ["goalkeeper_held_ball_position"]
