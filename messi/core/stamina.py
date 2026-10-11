"""두 시간척도의 player stamina를 갱신하는 pure JAX kernel."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.numeric import require_float32_representable
from messi.core.timebase import DEFAULT_MATCH_DURATION_SECONDS


@dataclass(frozen=True, slots=True)
class LongStamina:
    """경기 단위 workload와 지속 속도 응답 설정.

    이 구조와 기본 계수는 현재 직접적인 논문 fit이 아니라 결합된 design prior다.
    """

    sprint_speed: float = 5.5
    sprint_mult: float = 3.0
    idle_load: float = 0.12
    speed_ref: float = 2.4
    speed_load: float = 0.90
    accel_ref: float = 4.0
    accel_load: float = 0.10
    vmax_floor: float = 0.99
    end_frac: float = 0.05
    tail_knee: float = 0.20
    reference_duration_s: float = DEFAULT_MATCH_DURATION_SECONDS
    reference_workload: float = 1.85

    def __post_init__(self) -> None:
        _canonicalize_float32_fields(
            self,
            (
                "sprint_speed",
                "sprint_mult",
                "idle_load",
                "speed_ref",
                "speed_load",
                "accel_ref",
                "accel_load",
                "vmax_floor",
                "end_frac",
                "tail_knee",
                "reference_duration_s",
                "reference_workload",
            ),
        )
        _validate_long_stamina(self)


@dataclass(frozen=True, slots=True)
class ShortStamina:
    """sprint 고갈·회복과 순간 속도 headroom 설정.

    이 구조와 기본 계수는 현재 직접적인 논문 fit이 아니라 결합된 design prior다.
    """

    vmax_floor: float = 0.70
    headroom_knee: float = 0.25
    depletion_s: float = 10.0
    depletion_speed_frac: float = 0.70
    speed_exponent: float = 2.0
    accel_ref: float = 6.0
    accel_load: float = 0.20
    recovery_tau_s: float = 60.0
    recovery_speed_frac: float = 0.70
    recovery_exponent: float = 1.0

    def __post_init__(self) -> None:
        _canonicalize_float32_fields(
            self,
            (
                "vmax_floor",
                "headroom_knee",
                "depletion_s",
                "depletion_speed_frac",
                "speed_exponent",
                "accel_ref",
                "accel_load",
                "recovery_tau_s",
                "recovery_speed_frac",
                "recovery_exponent",
            ),
        )
        _validate_short_stamina(self)


class StaminaState(NamedTuple):
    long: jax.Array
    short: jax.Array


class Kinematics(NamedTuple):
    """movement와 stamina가 공유하는 player별 운동량."""

    speed: jax.Array
    previous_speed: jax.Array
    acceleration: jax.Array
    positive_speed_acceleration: jax.Array


def _finite(name: str, value: float) -> float:
    return require_float32_representable(name, value)


def _canonicalize_float32_fields(config, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(
            config, name, float(np.float32(_finite(name, getattr(config, name))))
        )


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name}은 0보다 커야 한다")
    return result


def _unit(name: str, value: float, *, open_interval: bool = False) -> float:
    result = _finite(name, value)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name}은 {interval} 범위여야 한다")
    return result


def _long_drain_base(config: LongStamina) -> float:
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        denominator = np.float32(config.reference_duration_s) * np.float32(
            config.reference_workload
        )
        drain_base = np.float32(
            (1.0 - config.end_frac)
            / (config.reference_duration_s * config.reference_workload)
        )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(
            "reference_duration_s * reference_workload는 float32에서 유한한 양수여야 한다"
        )
    if not np.isfinite(drain_base) or drain_base <= 0.0:
        raise ValueError("long stamina drain_base는 float32에서 유한한 양수여야 한다")
    return float(drain_base)


def _long_tail_decay(config: LongStamina) -> float:
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        tail_decay = np.float32(
            math.log(config.tail_knee / config.end_frac)
            / (config.tail_knee - config.end_frac)
        )
    if not np.isfinite(tail_decay) or tail_decay <= 0.0:
        raise ValueError("long stamina tail decay는 float32에서 유한한 양수여야 한다")
    return float(tail_decay)


def _validate_long_stamina(config: LongStamina) -> None:
    for name in (
        "sprint_speed",
        "sprint_mult",
        "speed_ref",
        "accel_ref",
        "reference_duration_s",
        "reference_workload",
    ):
        _positive(name, getattr(config, name))
    for name in ("idle_load", "speed_load", "accel_load"):
        if _finite(name, getattr(config, name)) < 0.0:
            raise ValueError(f"{name}은 0 이상이어야 한다")
    if config.sprint_mult < 1.0:
        raise ValueError("sprint_mult는 1 이상이어야 한다")
    _unit("vmax_floor", config.vmax_floor, open_interval=True)
    _unit("end_frac", config.end_frac, open_interval=True)
    _unit("tail_knee", config.tail_knee, open_interval=True)
    if config.end_frac >= config.tail_knee:
        raise ValueError("end_frac은 tail_knee보다 작아야 한다")
    _long_drain_base(config)
    _long_tail_decay(config)


def _validate_short_stamina(config: ShortStamina) -> None:
    for name in (
        "depletion_s",
        "speed_exponent",
        "accel_ref",
        "recovery_tau_s",
        "recovery_exponent",
    ):
        _positive(name, getattr(config, name))
    if _finite("accel_load", config.accel_load) < 0.0:
        raise ValueError("accel_load는 0 이상이어야 한다")
    for name in (
        "vmax_floor",
        "headroom_knee",
        "depletion_speed_frac",
        "recovery_speed_frac",
    ):
        _unit(name, getattr(config, name), open_interval=True)


_DEFAULT_LONG_STAMINA = LongStamina()
_DEFAULT_SHORT_STAMINA = ShortStamina()


def compute_kinematics(
    velocity: jax.Array,
    previous_velocity: jax.Array,
    *,
    dt: float,
) -> Kinematics:
    """movement와 stamina가 중복 계산하지 않도록 norm을 한 번 계산한다."""

    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    previous_velocity = jnp.asarray(previous_velocity, dtype=jnp.float32)
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    speed = jnp.linalg.norm(velocity, axis=-1)
    previous_speed = jnp.linalg.norm(previous_velocity, axis=-1)
    acceleration = jnp.linalg.norm((velocity - previous_velocity) / dt_value, axis=-1)
    positive = jnp.maximum((speed - previous_speed) / dt_value, 0.0)
    return Kinematics(speed, previous_speed, acceleration, positive)


def effective_speed_limit(
    max_speed: jax.Array,
    stamina_long: jax.Array,
    stamina_short: jax.Array,
    *,
    long: LongStamina = _DEFAULT_LONG_STAMINA,
    short: ShortStamina = _DEFAULT_SHORT_STAMINA,
) -> jax.Array:
    """두 stamina store가 허용하는 순간 self-propelled 속도를 반환한다."""

    sustained_fraction = long.vmax_floor + (1.0 - long.vmax_floor) * jnp.clip(
        stamina_long, 0.0, 1.0
    )
    x = jnp.clip(stamina_short / short.headroom_knee, 0.0, 1.0)
    headroom = x * x * (3.0 - 2.0 * x)
    return (
        jnp.asarray(max_speed, dtype=jnp.float32)
        * sustained_fraction
        * (short.vmax_floor + (1.0 - short.vmax_floor) * headroom)
    )


def _long_after_drain(
    stamina: jax.Array,
    nominal_drain: jax.Array,
    config: LongStamina,
) -> jax.Array:
    linear_room = jnp.maximum(stamina - config.tail_knee, 0.0)
    tail_decay = _long_tail_decay(config)
    linear_next = stamina - nominal_drain
    tail_start = jnp.minimum(stamina, config.tail_knee)
    tail_drain = jnp.maximum(nominal_drain - linear_room, 0.0)
    tail_next = tail_start * jnp.exp(-tail_decay * tail_drain)
    return jnp.clip(
        jnp.where(nominal_drain <= linear_room, linear_next, tail_next), 0.0, 1.0
    )


def step_stamina_from_kinematics(
    stamina_long: jax.Array,
    stamina_short: jax.Array,
    kinematics: Kinematics,
    active: jax.Array,
    locomotion: jax.Array,
    max_speed: jax.Array,
    endurance_factor: jax.Array,
    *,
    dt: float,
    long: LongStamina = _DEFAULT_LONG_STAMINA,
    short: ShortStamina = _DEFAULT_SHORT_STAMINA,
) -> StaminaState:
    """이미 계산된 운동량으로 두 stamina store를 한 번 갱신한다."""

    speed = kinematics.speed
    self_driven = jnp.asarray(active, dtype=jnp.bool_) & jnp.asarray(
        locomotion, dtype=jnp.bool_
    )
    sprint_excess = jnp.clip((speed - long.sprint_speed) / long.sprint_speed, 0.0, 2.0)
    sprint_load = jnp.where(
        speed > long.sprint_speed,
        (long.sprint_mult - 1.0) * sprint_excess,
        0.0,
    )
    speed_load = long.speed_load * jnp.clip(speed / long.speed_ref, 0.0, 4.0)
    acceleration_load = (
        long.accel_load
        * jnp.clip(kinematics.acceleration / long.accel_ref, 0.0, 4.0) ** 2
    )
    workload = jnp.where(
        active,
        long.idle_load
        + jnp.where(self_driven, sprint_load + speed_load + acceleration_load, 0.0),
        0.0,
    )
    drain_base = _long_drain_base(long)
    endurance_reciprocal = jnp.reciprocal(endurance_factor)
    next_long = _long_after_drain(
        stamina_long,
        jnp.asarray(dt, dtype=jnp.float32)
        * drain_base
        * workload
        * endurance_reciprocal,
        long,
    )

    sustained = max_speed * (
        long.vmax_floor + (1.0 - long.vmax_floor) * jnp.clip(stamina_long, 0.0, 1.0)
    )
    speed_fraction = speed / jnp.maximum(sustained, 1.0e-6)
    speed_intensity = (
        jnp.clip(
            (speed_fraction - short.depletion_speed_frac)
            / (1.0 - short.depletion_speed_frac),
            0.0,
            1.0,
        )
        ** short.speed_exponent
    )
    acceleration_intensity = short.accel_load * jnp.clip(
        kinematics.positive_speed_acceleration / short.accel_ref, 0.0, 1.0
    )
    short_load = jnp.where(
        self_driven, jnp.clip(speed_intensity + acceleration_intensity, 0.0, 2.0), 0.0
    )
    drain_rate = short_load / short.depletion_s * endurance_reciprocal
    recovery_speed_fraction = jnp.where(self_driven, speed_fraction, 0.0)
    recovery_factor = (
        jnp.clip(1.0 - recovery_speed_fraction / short.recovery_speed_frac, 0.0, 1.0)
        ** short.recovery_exponent
    )
    recovery_rate = jnp.where(
        active,
        (1.0 - stamina_short) * recovery_factor / short.recovery_tau_s,
        0.0,
    )
    next_short = jnp.clip(
        stamina_short
        + jnp.asarray(dt, dtype=jnp.float32) * (recovery_rate - drain_rate),
        0.0,
        1.0,
    )
    return StaminaState(next_long, next_short)


def recover_short_stamina_at_rest(
    stamina_short: jax.Array,
    rest_seconds: float | jax.Array,
    *,
    short: ShortStamina = _DEFAULT_SHORT_STAMINA,
) -> jax.Array:
    """정지 회복 ODE의 analytic solution을 boundary update로 적용한다."""

    if isinstance(rest_seconds, numbers.Real) and (
        isinstance(rest_seconds, bool)
        or not math.isfinite(rest_seconds)
        or rest_seconds < 0.0
    ):
        raise ValueError("rest_seconds는 유한한 0 이상의 실수여야 한다")
    stamina_short = jnp.asarray(stamina_short, dtype=jnp.float32)
    seconds = jnp.maximum(jnp.asarray(rest_seconds, dtype=jnp.float32), 0.0)
    recovered = 1.0 - (1.0 - stamina_short) * jnp.exp(-seconds / short.recovery_tau_s)
    return jnp.clip(recovered, 0.0, 1.0)


__all__ = [
    "Kinematics",
    "LongStamina",
    "ShortStamina",
    "StaminaState",
    "compute_kinematics",
    "effective_speed_limit",
    "recover_short_stamina_at_rest",
    "step_stamina_from_kinematics",
]
