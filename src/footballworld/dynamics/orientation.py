"""Seam-free player body and torso-relative gaze transitions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from footballworld.config.perception import Perception
from footballworld.core.state import PlayerState


def step_gaze(
    players: PlayerState,
    normalized_target: jax.Array,
    *,
    dt_control: float,
    perception: Perception,
) -> PlayerState:
    """Slew bounded relative gaze once per control frame.

    The public action is a target in ``[-1, 1]`` rather than a periodic world
    angle.  Body direction owns full-circle rotation; gaze remains inside a
    torso-relative interval and therefore has no 0/2pi discontinuity.
    """

    limit = jnp.asarray(
        math.radians(perception.gaze_yaw_limit_degrees), dtype=jnp.float32
    )
    maximum_delta = jnp.asarray(
        math.radians(perception.gaze_slew_rate_degrees_s) * dt_control,
        dtype=jnp.float32,
    )
    target = jnp.clip(jnp.asarray(normalized_target, dtype=jnp.float32), -1.0, 1.0)
    target = target * limit
    delta = jnp.clip(target - players.gaze_yaw, -maximum_delta, maximum_delta)
    gaze_yaw = jnp.clip(players.gaze_yaw + delta, -limit, limit)
    gaze_yaw = jnp.where(players.active, gaze_yaw, 0.0)
    return players._replace(gaze_yaw=gaze_yaw.astype(jnp.float32))


def view_forward(players: PlayerState) -> jax.Array:
    """Return each player's world-frame gaze centre as a unit vector."""

    cosine = jnp.cos(players.gaze_yaw)
    sine = jnp.sin(players.gaze_yaw)
    body = players.body_forward
    return jnp.stack(
        (
            cosine * body[:, 0] - sine * body[:, 1],
            sine * body[:, 0] + cosine * body[:, 1],
        ),
        axis=-1,
    ).astype(jnp.float32)


__all__ = ["step_gaze", "view_forward"]
