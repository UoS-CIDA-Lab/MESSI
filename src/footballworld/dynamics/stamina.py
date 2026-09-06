"""Pure JAX transition for long- and short-horizon stamina."""

import math
import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.constants import SAFE_NORM_EPS


class StaminaState(NamedTuple):
    """The two stamina stores carried by rollout state."""

    long: jax.Array
    short: jax.Array


def _norm(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + SAFE_NORM_EPS)


def _long_after_drain(
    stamina: jax.Array,
    nominal_drain: jax.Array,
    config: LongStamina,
) -> jax.Array:
    linear_room = jnp.maximum(stamina - config.tail_knee, 0.0)
    remains_linear = nominal_drain <= linear_room
    linear_next = stamina - nominal_drain

    tail_decay = math.log(config.tail_knee / config.end_frac) / (
        config.tail_knee - config.end_frac
    )
    tail_start = jnp.minimum(stamina, config.tail_knee)
    tail_drain = jnp.maximum(nominal_drain - linear_room, 0.0)
    tail_next = tail_start * jnp.exp(-tail_decay * tail_drain)
    return jnp.clip(jnp.where(remains_linear, linear_next, tail_next), 0.0, 1.0)


def _long_speed_limit(
    max_speed: jax.Array,
    stamina_long: jax.Array,
    config: LongStamina,
) -> jax.Array:
    fraction = config.vmax_floor + (1.0 - config.vmax_floor) * jnp.clip(
        stamina_long, 0.0, 1.0
    )
    return max_speed * fraction


def effective_speed_limit(
    max_speed: jax.Array,
    stamina_long: jax.Array,
    stamina_short: jax.Array,
    *,
    long: LongStamina = LongStamina(),
    short: ShortStamina = ShortStamina(),
) -> jax.Array:
    """Return the instantaneous speed limit set by both stamina stores."""

    sustained = _long_speed_limit(max_speed, stamina_long, long)
    x = jnp.clip(stamina_short / short.headroom_knee, 0.0, 1.0)
    headroom = x * x * (3.0 - 2.0 * x)
    return sustained * (short.vmax_floor + (1.0 - short.vmax_floor) * headroom)


def recover_short_stamina_at_rest(
    stamina_short: jax.Array,
    rest_seconds: float | jax.Array,
    *,
    short: ShortStamina = ShortStamina(),
) -> jax.Array:
    """Apply the exact resting solution of the short-store recovery ODE.

    At rest the configured recovery factor is one, so ``ds/dt=(1-s)/tau``.
    This analytic boundary update avoids simulating a 15-minute interval at
    the 90 Hz physics clock and is exact for the model rather than an Euler
    approximation.
    """

    if not math.isfinite(short.recovery_tau_s) or short.recovery_tau_s <= 0.0:
        raise ValueError("recovery_tau_s must be finite and positive")
    if isinstance(rest_seconds, numbers.Real) and (
        isinstance(rest_seconds, bool)
        or not math.isfinite(rest_seconds)
        or rest_seconds < 0.0
    ):
        raise ValueError("rest_seconds must be finite and non-negative")
    stamina_short = jnp.asarray(stamina_short)
    rest_seconds = jnp.asarray(rest_seconds, dtype=stamina_short.dtype)
    recovered = 1.0 - (1.0 - stamina_short) * jnp.exp(
        -rest_seconds / short.recovery_tau_s
    )
    return jnp.clip(recovered, 0.0, 1.0)


def step_stamina(
    stamina_long: jax.Array,
    stamina_short: jax.Array,
    velocity: jax.Array,
    previous_velocity: jax.Array,
    active: jax.Array,
    locomotion: jax.Array,
    max_speed: jax.Array,
    endurance_factor: jax.Array,
    *,
    dt: float,
    long: LongStamina = LongStamina(),
    short: ShortStamina = ShortStamina(),
) -> StaminaState:
    """Advance both stamina stores by one physics substep."""

    speed = _norm(velocity)
    previous_speed = _norm(previous_velocity)
    acceleration = _norm((velocity - previous_velocity) / dt)
    positive_speed_acceleration = jnp.clip((speed - previous_speed) / dt, 0.0)
    self_driven = active & locomotion

    sprint_excess = jnp.clip((speed - long.sprint_speed) / long.sprint_speed, 0.0, 2.0)
    sprint_load = jnp.where(
        speed > long.sprint_speed,
        (long.sprint_mult - 1.0) * sprint_excess,
        0.0,
    )
    speed_load = long.speed_load * jnp.clip(speed / long.speed_ref, 0.0, 4.0)
    acceleration_load = (
        long.accel_load * jnp.clip(acceleration / long.accel_ref, 0.0, 4.0) ** 2
    )
    workload = jnp.where(
        active,
        long.idle_load
        + jnp.where(
            self_driven,
            sprint_load + speed_load + acceleration_load,
            0.0,
        ),
        0.0,
    )
    long_drain_base = (1.0 - long.end_frac) / (
        long.reference_duration_s * long.reference_workload
    )
    endurance_reciprocal = jnp.reciprocal(endurance_factor)
    stamina_long_next = _long_after_drain(
        stamina_long,
        dt * long_drain_base * workload * endurance_reciprocal,
        long,
    )

    sustained_limit = _long_speed_limit(max_speed, stamina_long, long)
    speed_fraction = speed / jnp.maximum(sustained_limit, 1.0e-6)
    depletion_width = 1.0 - short.depletion_speed_frac
    speed_intensity = (
        jnp.clip(
            (speed_fraction - short.depletion_speed_frac) / depletion_width,
            0.0,
            1.0,
        )
        ** short.speed_exponent
    )
    acceleration_intensity = short.accel_load * jnp.clip(
        positive_speed_acceleration / short.accel_ref, 0.0, 1.0
    )
    short_load = jnp.where(
        self_driven,
        jnp.clip(speed_intensity + acceleration_intensity, 0.0, 2.0),
        0.0,
    )
    drain_rate = short_load / short.depletion_s * endurance_reciprocal

    recovery_speed_fraction = jnp.where(self_driven, speed_fraction, 0.0)
    recovery_factor = (
        jnp.clip(
            1.0 - recovery_speed_fraction / short.recovery_speed_frac,
            0.0,
            1.0,
        )
        ** short.recovery_exponent
    )
    recovery_rate = jnp.where(
        active,
        (1.0 - stamina_short) * recovery_factor / short.recovery_tau_s,
        0.0,
    )
    stamina_short_next = jnp.clip(
        stamina_short + dt * (recovery_rate - drain_rate), 0.0, 1.0
    )

    return StaminaState(long=stamina_long_next, short=stamina_short_next)
