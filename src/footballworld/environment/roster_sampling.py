"""Pure JAX sampling of fixed-shape episode roster abilities."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.roster_sampling import RosterSampling


class ProfileValues(NamedTuple):
    """Five sampled physical/ability arrays with a shared leading shape."""

    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


def _bounded_draw(
    mean: jax.Array,
    noise: jax.Array,
    standard_deviation: float,
    lower: float,
    upper: float,
    clip_standard_deviations: float,
) -> jax.Array:
    clipped_noise = jnp.clip(
        noise,
        -clip_standard_deviations,
        clip_standard_deviations,
    )
    return jnp.clip(mean + standard_deviation * clipped_noise, lower, upper)


def sample_profile_values(
    values: ProfileValues,
    key: jax.Array,
    config: RosterSampling,
    *,
    valid: jax.Array | None = None,
    minimum_height_m: float | None = None,
) -> ProfileValues:
    """Sample profile locations once without changing shape or identity.

    The supplied values are Gaussian locations. Near a configured bound the
    realised distribution is clipped and its empirical mean may consequently
    differ from that location. ``valid`` leaves padded bench cells untouched.
    """

    if type(config) is not RosterSampling:
        raise TypeError("config must be exactly RosterSampling")
    arrays = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32), values
    )
    shape = arrays.max_speed.shape
    for name, value in zip(ProfileValues._fields, arrays, strict=True):
        if value.shape != shape:
            raise ValueError(f"values.{name} must have shape {shape}")
    if valid is None:
        valid_mask = jnp.ones(shape, dtype=jnp.bool_)
    else:
        valid_mask = jnp.asarray(valid, dtype=jnp.bool_)
        if valid_mask.shape != shape:
            raise ValueError(f"valid must have shape {shape}")

    height_lower = (
        config.min_height_m
        if minimum_height_m is None
        else max(config.min_height_m, minimum_height_m)
    )
    if height_lower >= config.max_height_m:
        raise ValueError("physical minimum height must be below max_height_m")

    noise = jax.random.normal(key, (5, *shape), dtype=jnp.float32)
    z = config.clip_standard_deviations
    max_speed = _bounded_draw(
        arrays.max_speed,
        noise[0],
        config.max_speed_std_mps,
        config.min_max_speed_mps,
        config.max_max_speed_mps,
        z,
    )
    height = _bounded_draw(
        arrays.height,
        noise[1],
        config.height_std_m,
        height_lower,
        config.max_height_m,
        z,
    )
    reach_margin = _bounded_draw(
        arrays.reach_height - arrays.height,
        noise[2],
        config.reach_margin_std_m,
        config.min_reach_margin_m,
        config.max_reach_margin_m,
        z,
    )
    reach_height = height + reach_margin
    ball_control = _bounded_draw(
        arrays.ball_control,
        noise[3],
        config.ball_control_std,
        config.min_ball_control,
        config.max_ball_control,
        z,
    )
    endurance_factor = _bounded_draw(
        arrays.endurance_factor,
        noise[4],
        config.endurance_factor_std,
        config.min_endurance_factor,
        config.max_endurance_factor,
        z,
    )
    sampled = ProfileValues(
        max_speed=max_speed,
        height=height,
        reach_height=reach_height,
        ball_control=ball_control,
        endurance_factor=endurance_factor,
    )
    return jax.tree_util.tree_map(
        lambda candidate, original: jnp.where(valid_mask, candidate, original),
        sampled,
        arrays,
    )


__all__ = ["ProfileValues", "sample_profile_values"]
