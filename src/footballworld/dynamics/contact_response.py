"""Continuous response for a deliberate attempt to control the ball.

This module only resolves the velocity change at an already-established
contact.  Reach, timing, contact selection, possession, and event labels stay
with their respective dynamics and rules layers.

The relative ``ball_control`` term belongs to the contest actor score.  A
separate trapping lottery or ability-scaled impulse budget would give that one
coordinate a second, unrelated meaning: once an unopposed physical contact is
established, it must not independently turn an otherwise feasible touch into a
miss.  Kinematic eligibility and the bounded impulse remain authoritative.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import DIV_EPS, SQUARED_EPS


class ControlResponse(NamedTuple):
    """Result of applying a bounded control impulse at physical contact.

    ``success`` means that the requested player-relative exit velocity was
    reached in this contact.  A failed response is still a physical touch:
    ``velocity`` contains the largest admissible change toward the request and
    ``miscontrolled`` reports that a residual remained.
    """

    velocity: jax.Array
    impulse: jax.Array
    relative_speed_before: jax.Array
    relative_speed_after: jax.Array
    required_impulse_speed: jax.Array
    applied_impulse_speed: jax.Array
    completion: jax.Array
    success: jax.Array
    miscontrolled: jax.Array


def resolve_control_response(
    incoming_velocity: jax.Array,
    player_velocity: jax.Array,
    requested_relative_velocity: jax.Array,
    *,
    impulse_speed_limit_mps: jax.Array,
) -> ControlResponse:
    """Move the ball toward a requested exit state with a bounded impulse.

    All velocity arguments use world coordinates and may have arbitrary
    leading batch dimensions, with the final dimension holding xyz.  The
    request is relative to the player so that the response is invariant to a
    shared change of reference velocity.

    The only physical budget is ``impulse_speed_limit_mps``.  A player's
    ``ball_control`` coordinate belongs to stochastic actor selection when
    opponents contest the same ball; it must not make an otherwise identical,
    unopposed physical contact weaker.  Speed, height, timing, and recovery
    eligibility remain upstream in the contact detector, while an excessive
    requested velocity change can still exceed this physical impulse budget.
    """

    incoming_velocity = jnp.asarray(incoming_velocity)
    dtype = incoming_velocity.dtype
    player_velocity = jnp.asarray(player_velocity, dtype=dtype)
    requested_relative_velocity = jnp.asarray(requested_relative_velocity, dtype=dtype)
    impulse_speed_limit_mps = jnp.asarray(impulse_speed_limit_mps, dtype=dtype)

    target_velocity = player_velocity + requested_relative_velocity
    required_impulse = target_velocity - incoming_velocity
    required_squared = jnp.sum(required_impulse * required_impulse, axis=-1)
    required_speed = jnp.sqrt(jnp.maximum(required_squared, 0.0))

    impulse_budget = jnp.maximum(impulse_speed_limit_mps, 0.0)
    completion = jnp.where(
        required_squared > SQUARED_EPS,
        jnp.minimum(1.0, impulse_budget / (required_speed + DIV_EPS)),
        1.0,
    )
    impulse = required_impulse * completion[..., None]
    velocity = incoming_velocity + impulse

    relative_before = incoming_velocity - player_velocity
    relative_after = velocity - player_velocity
    applied_speed = required_speed * completion
    success = required_speed <= impulse_budget

    return ControlResponse(
        velocity=velocity,
        impulse=impulse,
        relative_speed_before=jnp.sqrt(
            jnp.maximum(jnp.sum(relative_before * relative_before, axis=-1), 0.0)
        ),
        relative_speed_after=jnp.sqrt(
            jnp.maximum(jnp.sum(relative_after * relative_after, axis=-1), 0.0)
        ),
        required_impulse_speed=required_speed,
        applied_impulse_speed=applied_speed,
        completion=completion,
        success=success,
        miscontrolled=~success,
    )
