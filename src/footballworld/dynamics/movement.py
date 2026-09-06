"""Pure JAX locomotion for one player-physics substep."""

import jax
import jax.numpy as jnp

from footballworld.config.player_physics import PlayerPhysics
from footballworld.core.action_mapping import linf_radial_decode
from footballworld.core.constants import DIV_EPS, SAFE_NORM_EPS, STATIONARY_SPEED_EPS
from footballworld.core.state import PlayerState


def _norm(vector: jax.Array, *, keepdims: bool = False) -> jax.Array:
    return jnp.sqrt(
        jnp.sum(vector * vector, axis=-1, keepdims=keepdims) + SAFE_NORM_EPS
    )


def decode_desired_velocity(
    move: jax.Array,
    max_speed: jax.Array,
    team_id: jax.Array,
    attack_direction: jax.Array,
) -> jax.Array:
    """Decode a team-relative L-infinity radial command into world velocity."""

    decoded = linf_radial_decode(jnp.clip(jnp.asarray(move), -1.0, 1.0))
    planar = decoded.direction * decoded.power[:, None]
    team_rotation = attack_direction[team_id, None]
    return planar * team_rotation * max_speed[:, None]


def step_velocity(
    velocity: jax.Array,
    desired_velocity: jax.Array,
    speed_limit: jax.Array,
    movement_enabled: jax.Array,
    *,
    dt: float,
    config: PlayerPhysics = PlayerPhysics(),
) -> jax.Array:
    """Move velocity toward a target inside a longitudinal/lateral ellipse.

    ``speed_limit`` constrains self-propelled target velocity. It deliberately
    does not clip the realized velocity: an externally induced velocity above
    the locomotion limit decays through the ordinary braking model instead of
    disappearing at the next physics boundary.
    """

    desired_speed = _norm(desired_velocity, keepdims=True)
    target = desired_velocity * jnp.minimum(
        speed_limit[:, None] / (desired_speed + DIV_EPS), 1.0
    )

    speed = _norm(velocity, keepdims=True)
    target_direction = target / (_norm(target, keepdims=True) + DIV_EPS)
    movement_direction = jnp.where(
        speed > STATIONARY_SPEED_EPS,
        velocity / (speed + DIV_EPS),
        target_direction,
    )

    delta = target - velocity
    longitudinal = jnp.sum(delta * movement_direction, axis=-1, keepdims=True)
    lateral = delta - longitudinal * movement_direction
    lateral_magnitude = _norm(lateral, keepdims=True)

    longitudinal_limit = jnp.where(
        longitudinal >= 0.0,
        config.forward_acceleration_mps2 * dt,
        config.braking_deceleration_mps2 * dt,
    )
    lateral_limit = config.lateral_acceleration_mps2 * dt
    ellipse_radius = jnp.sqrt(
        (longitudinal / (longitudinal_limit + DIV_EPS)) ** 2
        + (lateral_magnitude / (lateral_limit + DIV_EPS)) ** 2
        + SAFE_NORM_EPS
    )
    scale = 1.0 / jnp.maximum(ellipse_radius, 1.0)
    feasible_delta = longitudinal * scale * movement_direction + lateral * scale

    return velocity + jnp.where(movement_enabled[:, None], feasible_delta, 0.0)


def step_player_motion(
    players: PlayerState,
    desired_velocity: jax.Array,
    speed_limit: jax.Array,
    movement_enabled: jax.Array,
    field_half_extent: jax.Array,
    *,
    boundary_margin_m: float,
    dt: float,
    position_update_enabled: jax.Array | None = None,
    config: PlayerPhysics = PlayerPhysics(),
) -> PlayerState:
    """Advance active players and remove outward velocity at field bounds."""

    active = players.active
    position_update_enabled = (
        jnp.ones_like(active)
        if position_update_enabled is None
        else jnp.asarray(position_update_enabled, dtype=bool)
    )
    position_update = active & position_update_enabled
    velocity = step_velocity(
        players.velocity,
        desired_velocity,
        speed_limit,
        movement_enabled & position_update,
        dt=dt,
        config=config,
    )
    velocity = jnp.where(position_update[:, None], velocity, 0.0)

    free_position = players.position + dt * velocity
    bound = field_half_extent + boundary_margin_m
    position = jnp.clip(free_position, -bound, bound)
    velocity = jnp.where(position != free_position, 0.0, velocity)
    position = jnp.where(position_update[:, None], position, players.position)

    return players._replace(position=position, velocity=velocity)
