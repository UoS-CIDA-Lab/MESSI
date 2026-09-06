"""Reversible mappings for the continuous action box."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RadialControl(NamedTuple):
    """Decoded direction and isotropic magnitude of a two-axis command."""

    direction: jax.Array
    power: jax.Array


def _nonzero_l2(vector: jax.Array) -> jax.Array:
    """Return exact positive L2 norms and a differentiable unit at zero."""

    squared = jnp.sum(vector * vector, axis=-1, keepdims=True)
    return jnp.sqrt(squared + (squared == 0.0).astype(vector.dtype))


def linf_radial_decode(vector: jax.Array) -> RadialControl:
    """Map a square command to unit direction and isotropic power.

    For ``vector`` in ``[-1, 1]^2``, power is its L-infinity norm and
    direction is its Euclidean unit direction. The zero command has zero
    direction and power.
    """

    vector = jnp.asarray(vector, dtype=jnp.float32)
    direction = vector / _nonzero_l2(vector)
    power = jnp.max(jnp.abs(vector), axis=-1)
    return RadialControl(direction, power)


def linf_radial_encode(direction: jax.Array, power: jax.Array) -> jax.Array:
    """Inverse of :func:`linf_radial_decode` for direction and power."""

    direction = jnp.asarray(direction, dtype=jnp.float32)
    power = jnp.asarray(power, dtype=jnp.float32)
    unit_direction = direction / _nonzero_l2(direction)
    linf = jnp.max(jnp.abs(unit_direction), axis=-1, keepdims=True)
    safe_linf = jnp.where(linf > 0.0, linf, 1.0)
    encoded = unit_direction * (jnp.clip(power, 0.0, 1.0)[..., None] / safe_linf)
    return jnp.where(linf > 0.0, encoded, 0.0).astype(jnp.float32)


def signed_to_unit(value: jax.Array) -> jax.Array:
    """Map a normalized signed scalar from ``[-1, 1]`` to ``[0, 1]``."""

    value = jnp.asarray(value, dtype=jnp.float32)
    return (jnp.clip(value, -1.0, 1.0) + 1.0) * 0.5
