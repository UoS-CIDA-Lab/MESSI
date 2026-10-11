"""정규화된 continuous action box의 가역적 mapping."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class RadialControl(NamedTuple):
    """2축 command를 unit direction과 등방성 power로 분리한 값."""

    direction: jax.Array
    power: jax.Array


def _nonzero_l2(vector: jax.Array) -> jax.Array:
    """0에서는 유한한 값을 유지하고 나머지는 정확한 L2 norm을 반환한다."""

    squared = jnp.sum(vector * vector, axis=-1, keepdims=True)
    return jnp.sqrt(squared + (squared == 0.0).astype(vector.dtype))


def linf_radial_decode(vector: jax.Array) -> RadialControl:
    """`[-1, 1]^2` command를 unit direction과 L-infinity power로 변환한다."""

    vector = jnp.asarray(vector, dtype=jnp.float32)
    direction = vector / _nonzero_l2(vector)
    power = jnp.max(jnp.abs(vector), axis=-1)
    return RadialControl(direction=direction, power=power)


def linf_radial_encode(direction: jax.Array, power: jax.Array) -> jax.Array:
    """direction과 power를 `linf_radial_decode`의 square 표현으로 변환한다."""

    direction = jnp.asarray(direction, dtype=jnp.float32)
    power = jnp.asarray(power, dtype=jnp.float32)
    unit_direction = direction / _nonzero_l2(direction)
    linf = jnp.max(jnp.abs(unit_direction), axis=-1, keepdims=True)
    safe_linf = jnp.where(linf > 0.0, linf, 1.0)
    encoded = unit_direction * (jnp.clip(power, 0.0, 1.0)[..., None] / safe_linf)
    return jnp.where(linf > 0.0, encoded, 0.0).astype(jnp.float32)


def signed_to_unit(value: jax.Array) -> jax.Array:
    """정규화된 `[-1, 1]` scalar를 `[0, 1]`로 변환한다."""

    value = jnp.asarray(value, dtype=jnp.float32)
    return (jnp.clip(value, -1.0, 1.0) + 1.0) * 0.5


__all__ = [
    "RadialControl",
    "linf_radial_decode",
    "linf_radial_encode",
    "signed_to_unit",
]
