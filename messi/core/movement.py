"""선수축 전체를 broadcasting하는 player locomotion kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from messi.core.action_mapping import linf_radial_decode
from messi.core.numeric import require_float32_representable
from messi.core.stamina import (
    LongStamina,
    ShortStamina,
    compute_kinematics,
    effective_speed_limit,
    step_stamina_from_kinematics,
)
from messi.core.state import PlayerState, RosterState

_DIV_EPS = 1.0e-9
_STATIONARY_SPEED_EPS = 1.0e-3


@dataclass(frozen=True, slots=True)
class PlayerPhysics:
    """locomotion의 solver limit.

    tracking의 windowed acceleration과 solver의 순간 vector-change limit은 서로 다른
    estimand이므로 기본 계수는 논문 fit이 아닌 결합된 design prior다.
    """

    forward_acceleration_mps2: float = 7.95
    lateral_acceleration_mps2: float = 8.46
    braking_deceleration_mps2: float = 10.51
    body_turn_rate_max_radps: float = 8.0

    def __post_init__(self) -> None:
        limits = {
            "forward_acceleration_mps2": 12.0,
            "lateral_acceleration_mps2": 12.0,
            "braking_deceleration_mps2": 15.0,
            "body_turn_rate_max_radps": 4.0 * math.pi,
        }
        for name, upper in limits.items():
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 < value <= upper:
                raise ValueError(f"{name}은 (0, {upper}] 범위여야 한다")


_DEFAULT_PLAYER_PHYSICS = PlayerPhysics()
_DEFAULT_LONG_STAMINA = LongStamina()
_DEFAULT_SHORT_STAMINA = ShortStamina()


def decode_desired_velocity(
    move: jax.Array,
    roster: RosterState,
    attack_direction: jax.Array,
) -> jax.Array:
    """control-frame move를 world-frame 목표 속도로 한 번 decode한다."""

    decoded = linf_radial_decode(jnp.clip(jnp.asarray(move), -1.0, 1.0))
    planar = decoded.direction * decoded.power[..., None]
    team_direction = jnp.asarray(attack_direction, dtype=jnp.float32)[roster.team_id]
    team_to_world = jnp.stack(
        (team_direction, jnp.ones_like(team_direction)), axis=-1
    )
    return (
        planar
        * team_to_world
        * jnp.asarray(roster.max_speed, dtype=jnp.float32)[..., None]
    ).astype(jnp.float32)


def _rotate_body_forward(
    current: jax.Array,
    target: jax.Array,
    valid: jax.Array,
    *,
    maximum_turn_radians: jax.Array,
) -> jax.Array:
    current_norm = jnp.linalg.norm(current, axis=-1, keepdims=True)
    target_norm = jnp.linalg.norm(target, axis=-1, keepdims=True)
    current_unit = current / jnp.maximum(current_norm, _DIV_EPS)
    target_unit = target / jnp.maximum(target_norm, _DIV_EPS)
    dot = jnp.clip(jnp.sum(current_unit * target_unit, axis=-1), -1.0, 1.0)
    cross = (
        current_unit[:, 0] * target_unit[:, 1] - current_unit[:, 1] * target_unit[:, 0]
    )
    sign = jnp.where(jnp.abs(cross) > _DIV_EPS, jnp.sign(cross), 1.0)
    turn = jnp.minimum(jnp.maximum(maximum_turn_radians, 0.0), jnp.pi)
    cosine = jnp.cos(turn)
    sine = jnp.sin(turn) * sign
    rotated = jnp.stack(
        (
            cosine * current_unit[:, 0] - sine * current_unit[:, 1],
            sine * current_unit[:, 0] + cosine * current_unit[:, 1],
        ),
        axis=-1,
    )
    updated = jnp.where((dot >= cosine)[:, None], target_unit, rotated)
    use_target = jnp.asarray(valid, dtype=jnp.bool_) & (target_norm[:, 0] > _DIV_EPS)
    return jnp.where(use_target[:, None], updated, current_unit).astype(jnp.float32)


def step_velocity(
    velocity: jax.Array,
    desired_velocity: jax.Array,
    speed_limit: jax.Array,
    movement_enabled: jax.Array,
    *,
    dt: float,
    config: PlayerPhysics = _DEFAULT_PLAYER_PHYSICS,
) -> jax.Array:
    """목표 속도 방향의 longitudinal/lateral acceleration ellipse를 적용한다."""

    desired_speed = jnp.linalg.norm(desired_velocity, axis=-1, keepdims=True)
    target = desired_velocity * jnp.minimum(
        speed_limit[:, None] / jnp.maximum(desired_speed, _DIV_EPS), 1.0
    )
    speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    target_direction = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), _DIV_EPS
    )
    movement_direction = jnp.where(
        speed > _STATIONARY_SPEED_EPS,
        velocity / jnp.maximum(speed, _DIV_EPS),
        target_direction,
    )
    delta = target - velocity
    longitudinal = jnp.sum(delta * movement_direction, axis=-1, keepdims=True)
    lateral = delta - longitudinal * movement_direction
    lateral_magnitude = jnp.linalg.norm(lateral, axis=-1, keepdims=True)
    dt_value = jnp.asarray(dt, dtype=jnp.float32)
    longitudinal_limit = jnp.where(
        longitudinal >= 0.0,
        config.forward_acceleration_mps2 * dt_value,
        config.braking_deceleration_mps2 * dt_value,
    )
    lateral_limit = config.lateral_acceleration_mps2 * dt_value
    ellipse_radius = jnp.sqrt(
        (longitudinal / jnp.maximum(longitudinal_limit, _DIV_EPS)) ** 2
        + (lateral_magnitude / jnp.maximum(lateral_limit, _DIV_EPS)) ** 2
    )
    scale = jnp.reciprocal(jnp.maximum(ellipse_radius, 1.0))
    feasible_delta = longitudinal * scale * movement_direction + lateral * scale
    return velocity + jnp.where(movement_enabled[:, None], feasible_delta, 0.0)


def step_locomotion(
    players: PlayerState,
    roster: RosterState,
    desired_velocity: jax.Array,
    movement_enabled: jax.Array,
    field_half_extent: jax.Array,
    *,
    dt: float,
    boundary_margin_m: float = 0.0,
    physics: PlayerPhysics = _DEFAULT_PLAYER_PHYSICS,
    long_stamina: LongStamina = _DEFAULT_LONG_STAMINA,
    short_stamina: ShortStamina = _DEFAULT_SHORT_STAMINA,
) -> PlayerState:
    """한 physics substep의 body turn, movement와 stamina를 함께 갱신한다."""

    active = players.active
    locomotion = active & jnp.asarray(movement_enabled, dtype=jnp.bool_)
    body_forward = _rotate_body_forward(
        players.body_forward,
        desired_velocity,
        locomotion,
        maximum_turn_radians=jnp.asarray(
            physics.body_turn_rate_max_radps * dt, dtype=jnp.float32
        ),
    )
    speed_limit = effective_speed_limit(
        roster.max_speed,
        players.stamina_long,
        players.stamina_short,
        long=long_stamina,
        short=short_stamina,
    )
    velocity = step_velocity(
        players.velocity,
        desired_velocity,
        speed_limit,
        locomotion,
        dt=dt,
        config=physics,
    )
    velocity = jnp.where(active[:, None], velocity, 0.0)
    free_position = players.position + jnp.asarray(dt, dtype=jnp.float32) * velocity
    bound = jnp.asarray(field_half_extent, dtype=jnp.float32) + jnp.asarray(
        boundary_margin_m, dtype=jnp.float32
    )
    position = jnp.clip(free_position, -bound, bound)
    velocity = jnp.where(position != free_position, 0.0, velocity)
    position = jnp.where(active[:, None], position, players.position)
    kinematics = compute_kinematics(velocity, players.velocity, dt=dt)
    stamina = step_stamina_from_kinematics(
        players.stamina_long,
        players.stamina_short,
        kinematics,
        active,
        locomotion,
        roster.max_speed,
        roster.endurance_factor,
        dt=dt,
        long=long_stamina,
        short=short_stamina,
    )
    return players._replace(
        position=position.astype(jnp.float32),
        velocity=velocity.astype(jnp.float32),
        body_forward=body_forward,
        stamina_long=stamina.long.astype(jnp.float32),
        stamina_short=stamina.short.astype(jnp.float32),
    )


__all__ = [
    "PlayerPhysics",
    "decode_desired_velocity",
    "step_locomotion",
    "step_velocity",
]
