"""공중 비행, turf 충돌과 지면 운동을 처리하는 free-ball JAX kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_representable
from messi.core.state import BallState

_SAFE_NORM_EPS = 1.0e-18
_SQUARED_EPS = 1.0e-12
_DIV_EPS = 1.0e-9
_GEOMETRY_EPS = 1.0e-6


def _finite(name: str, value: float) -> float:
    return require_float32_representable(name, value)


def _bounded(name: str, value: float, lower: float, upper: float) -> float:
    result = _finite(name, value)
    if not lower <= result <= upper:
        raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")
    return result


@dataclass(frozen=True, slots=True)
class BallPhysics:
    """free-ball integrator의 고정 계수.

    Airborne 기본값의 provenance는 module 함수 주석과 ``docs/ball.md``에
    기록한다. Turf와 bounce 계수는 실측 fit으로 주장하지 않는 design prior다.
    """

    gravity_mps2: float = 9.81
    ground_settle_speed_mps: float = 0.5
    mass_kg: float = 0.424
    air_density_kgpm3: float = 1.2
    drag_coefficient_high_re: float = 0.155
    drag_crisis_drop: float = 0.346
    drag_crisis_speed_mps: float = 12.19
    drag_crisis_width_mps: float = 1.309
    spin_drag_scale: float = 0.4127
    spin_drag_exponent: float = 0.3056
    spin_drag_min_parameter: float = 0.05
    lift_coefficient_limit: float = 0.42
    air_spin_decay_per_s: float = 0.075
    ground_curl_coefficient: float = 0.00506
    ground_slide_friction: float = 0.35
    ground_spin_decay_per_s: float = 0.31
    restitution: float = 0.773
    inertia_ratio: float = 0.667
    bounce_tangential_restitution: float = 0.0
    bounce_spin_min_vertical_speed_mps: float = 0.7
    bounce_horizontal_retention: float = 0.732
    rolling_speed_knots_mps: tuple[float, ...] = (
        0.0,
        2.0,
        4.0,
        6.0,
        9.0,
        12.0,
        16.0,
        20.0,
        26.0,
        40.0,
    )
    rolling_deceleration_knots_mps2: tuple[float, ...] = (
        0.70,
        0.95,
        1.09,
        1.70,
        6.04,
        7.85,
        9.72,
        13.29,
        17.39,
        26.96,
    )

    def __post_init__(self) -> None:
        _bounded("gravity_mps2", self.gravity_mps2, 1.0, 20.0)
        _bounded("ground_settle_speed_mps", self.ground_settle_speed_mps, 0.0, 5.0)
        # IFAB Law 2: match 시작 시 410--450 g.
        _bounded("mass_kg", self.mass_kg, 0.410, 0.450)
        _bounded("air_density_kgpm3", self.air_density_kgpm3, 0.5, 2.0)
        for name in (
            "drag_coefficient_high_re",
            "drag_crisis_drop",
            "spin_drag_scale",
            "spin_drag_exponent",
            "air_spin_decay_per_s",
            "ground_curl_coefficient",
            "ground_slide_friction",
            "ground_spin_decay_per_s",
            "bounce_spin_min_vertical_speed_mps",
        ):
            _bounded(name, getattr(self, name), 0.0, 5.0)
        _bounded("spin_drag_min_parameter", self.spin_drag_min_parameter, 1.0e-6, 5.0)
        _bounded("lift_coefficient_limit", self.lift_coefficient_limit, 1.0e-6, 5.0)
        _bounded("drag_crisis_speed_mps", self.drag_crisis_speed_mps, 1.0, 60.0)
        _bounded("drag_crisis_width_mps", self.drag_crisis_width_mps, 0.01, 20.0)
        for name in (
            "restitution",
            "bounce_tangential_restitution",
            "bounce_horizontal_retention",
        ):
            _bounded(name, getattr(self, name), 0.0, 1.0)
        _bounded("inertia_ratio", self.inertia_ratio, 0.01, 2.0)
        speeds = tuple(self.rolling_speed_knots_mps)
        decelerations = tuple(self.rolling_deceleration_knots_mps2)
        if len(speeds) < 2 or len(speeds) != len(decelerations):
            raise ValueError(
                "rolling knot은 같은 길이의 2개 이상 speed/deceleration 쌍이어야 한다"
            )
        checked_speeds = tuple(
            float(
                np.float32(_bounded("rolling_speed_knots_mps", x, 0.0, 60.0))
            )
            for x in speeds
        )
        checked_decelerations = tuple(
            float(
                np.float32(
                    _bounded(
                        "rolling_deceleration_knots_mps2", x, 1.0e-6, 50.0
                    )
                )
            )
            for x in decelerations
        )
        if checked_speeds[0] != 0.0 or any(
            right <= left for left, right in pairwise(checked_speeds)
        ):
            raise ValueError(
                "rolling speed knot은 float32에서 0부터 엄격히 증가해야 한다"
            )
        object.__setattr__(self, "rolling_speed_knots_mps", checked_speeds)
        object.__setattr__(
            self,
            "rolling_deceleration_knots_mps2",
            checked_decelerations,
        )


_DEFAULT_GEOMETRY = BallGeometry()
_DEFAULT_PHYSICS = BallPhysics()


def _norm(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector) + _SAFE_NORM_EPS)


def _rotate(vector: jax.Array, axis: jax.Array, angle: jax.Array) -> jax.Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return (
        vector * cosine
        + jnp.cross(axis, vector) * sine
        + axis * jnp.dot(axis, vector) * (1.0 - cosine)
    )


def _smoothstep01(value: jax.Array) -> jax.Array:
    value = jnp.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _drag_coefficient(
    speed: jax.Array,
    spin_parameter: jax.Array,
    physics: BallPhysics,
) -> jax.Array:
    # "Soccer ball lift coefficients via trajectory analysis" (Goff & Carre,
    # 2010, doi:10.1088/0143-0807/31/4/007), Eq. (3)의 non-spinning fit과
    # Eq. (4)의 post-critical spin fit. 두 측정 구간의 smooth blend는
    # discontinuity를 피하기 위한 구현 선택이며 논문의 별도 식이 아니다.
    crisis_fraction = jax.nn.sigmoid(
        (physics.drag_crisis_speed_mps - speed) / physics.drag_crisis_width_mps
    )
    base = physics.drag_coefficient_high_re + physics.drag_crisis_drop * crisis_fraction
    spinning = physics.spin_drag_scale * jnp.power(
        jnp.maximum(spin_parameter, _DIV_EPS), physics.spin_drag_exponent
    )
    interval = jnp.maximum(physics.spin_drag_min_parameter, _DIV_EPS)
    spin_fraction = _smoothstep01(
        (spin_parameter - physics.spin_drag_min_parameter) / interval
    )
    blend = (1.0 - crisis_fraction) * spin_fraction
    return base + blend * (spinning - base)


def _advance_air_velocity(
    velocity: jax.Array,
    spin: jax.Array,
    dt: jax.Array,
    geometry: BallGeometry,
    physics: BallPhysics,
) -> jax.Array:
    speed_squared = jnp.sum(velocity * velocity)
    speed = jnp.sqrt(speed_squared + _SAFE_NORM_EPS)
    moving = speed_squared > _SQUARED_EPS
    direction = velocity / speed
    perpendicular_spin = spin - jnp.dot(spin, direction) * direction
    perpendicular_spin_squared = jnp.sum(perpendicular_spin * perpendicular_spin)
    perpendicular_spin_speed = jnp.sqrt(perpendicular_spin_squared + _SAFE_NORM_EPS)
    spin_parameter = jnp.where(
        moving, geometry.radius_m * perpendicular_spin_speed / speed, 0.0
    )
    drag_coefficient = _drag_coefficient(speed, spin_parameter, physics)
    area = math.pi * geometry.radius_m * geometry.radius_m
    force_scale = 0.5 * physics.air_density_kgpm3 * area / physics.mass_kg
    drag_rate = force_scale * drag_coefficient * speed
    damped = velocity / (1.0 + drag_rate * dt)

    # "Trajectory analysis of a soccer ball" (Goff & Carre, 2009,
    # doi:10.1119/1.3197187)와 "Investigations into soccer aerodynamics via
    # trajectory analysis and dust experiments" (Goff & Carre, 2012,
    # doi:10.1016/j.proeng.2012.04.028)의 관측 범위를 따른다. tanh shape와
    # 0.42 기본 cap은 bounded surrogate인 design prior다.
    lift_coefficient = physics.lift_coefficient_limit * jnp.tanh(
        spin_parameter / physics.lift_coefficient_limit
    )
    turn_rate = force_scale * lift_coefficient * speed
    axis = perpendicular_spin / perpendicular_spin_speed
    rotated = _rotate(damped, axis, turn_rate * dt)
    lift_active = moving & (perpendicular_spin_squared > _SQUARED_EPS)
    return jnp.where(lift_active, rotated, damped)


def _advance_ground_velocity(
    velocity: jax.Array,
    spin: jax.Array,
    dt: jax.Array,
    geometry: BallGeometry,
    physics: BallPhysics,
) -> tuple[jax.Array, jax.Array]:
    ground_axis = jnp.asarray((0.0, 0.0, 1.0), dtype=velocity.dtype)
    velocity = _rotate(
        velocity.at[2].set(0.0),
        ground_axis,
        physics.ground_curl_coefficient * spin[2] * dt,
    )
    surface_velocity = geometry.radius_m * jnp.asarray(
        (-spin[1], spin[0]), dtype=velocity.dtype
    )
    slip = velocity[:2] + surface_velocity
    sticking_delta = -(physics.inertia_ratio / (1.0 + physics.inertia_ratio)) * slip
    sticking_speed = _norm(sticking_delta)
    maximum_delta = physics.ground_slide_friction * physics.gravity_mps2 * dt
    fraction = jnp.minimum(1.0, maximum_delta / sticking_speed)
    velocity_delta = fraction * sticking_delta
    velocity = velocity.at[:2].add(velocity_delta)
    spin_delta = jnp.asarray(
        (velocity_delta[1], -velocity_delta[0]), dtype=spin.dtype
    ) / (physics.inertia_ratio * geometry.radius_m)
    spin = spin.at[:2].add(spin_delta)
    rolling = sticking_speed <= maximum_delta + _DIV_EPS

    horizontal_speed = _norm(velocity[:2]) + _DIV_EPS
    rolling_deceleration = jnp.interp(
        horizontal_speed,
        jnp.asarray(physics.rolling_speed_knots_mps, dtype=velocity.dtype),
        jnp.asarray(physics.rolling_deceleration_knots_mps2, dtype=velocity.dtype),
    )
    rolling_scale = (
        jnp.maximum(0.0, horizontal_speed - rolling_deceleration * dt)
        / horizontal_speed
    )
    velocity = velocity.at[:2].multiply(jnp.where(rolling, rolling_scale, 1.0))
    spin = spin.at[2].multiply(jnp.exp(-physics.ground_spin_decay_per_s * dt))
    rolling_spin = jnp.asarray(
        (-velocity[1] / geometry.radius_m, velocity[0] / geometry.radius_m),
        dtype=spin.dtype,
    )
    spin = spin.at[:2].set(jnp.where(rolling, rolling_spin, spin[:2]))
    return velocity, spin


def advance_supported_ground_motion(
    velocity: jax.Array,
    spin: jax.Array,
    *,
    dt: float | jax.Array,
    geometry: BallGeometry = _DEFAULT_GEOMETRY,
    physics: BallPhysics = _DEFAULT_PHYSICS,
) -> tuple[jax.Array, jax.Array]:
    """지원된 공의 turf curl, slip-to-roll과 rolling resistance를 전개한다."""

    return _advance_ground_velocity(
        jnp.asarray(velocity),
        jnp.asarray(spin),
        jnp.asarray(dt, dtype=jnp.asarray(velocity).dtype),
        geometry,
        physics,
    )


def advance_ball_smooth(
    state: BallState,
    *,
    dt: float | jax.Array,
    geometry: BallGeometry = _DEFAULT_GEOMETRY,
    physics: BallPhysics = _DEFAULT_PHYSICS,
) -> BallState:
    """새 surface impact를 풀지 않고 한 구간의 smooth force를 적용한다.

    공중·지면 후보를 함께 계산한 뒤 scalar predicate로 선택한다. GPU의 긴
    ``lax.scan``에서 dynamic ``lax.cond``보다 fusion과 실행 속도가 유리하다는 실제
    workload 측정에 따른 선택이다.
    """

    dt_value = jnp.asarray(dt, dtype=state.position.dtype)
    ground_height = jnp.asarray(geometry.radius_m, dtype=state.position.dtype)
    supported = (
        (state.position[2] <= ground_height + _GEOMETRY_EPS)
        & (state.velocity[2] <= 0.0)
        & (state.velocity[2] >= -physics.ground_settle_speed_mps)
    )

    velocity_with_gravity = state.velocity + dt_value * jnp.asarray(
        (0.0, 0.0, jnp.where(supported, 0.0, -physics.gravity_mps2)),
        dtype=state.velocity.dtype,
    )
    air_velocity = _advance_air_velocity(
        velocity_with_gravity, state.spin, dt_value, geometry, physics
    )
    ground_velocity, ground_spin = _advance_ground_velocity(
        velocity_with_gravity, state.spin, dt_value, geometry, physics
    )
    velocity = jnp.where(supported, ground_velocity, air_velocity)
    position = state.position + dt_value * velocity
    position = position.at[2].set(jnp.where(supported, ground_height, position[2]))
    air_spin = state.spin * jnp.exp(-physics.air_spin_decay_per_s * dt_value)
    spin = jnp.where(supported, ground_spin, air_spin)
    stepped = BallState(position, velocity, spin, state.live)
    return jax.tree.map(
        lambda new, old: jnp.where(state.live, new, old), stepped, state
    )


def apply_ground_impact(
    state: BallState,
    *,
    geometry: BallGeometry = _DEFAULT_GEOMETRY,
    physics: BallPhysics = _DEFAULT_PHYSICS,
) -> BallState:
    """현재 위치에서 한 번의 instantaneous turf impact를 적용한다."""

    ground_height = jnp.asarray(geometry.radius_m, dtype=state.position.dtype)
    falling = (
        state.live
        & (state.position[2] <= ground_height + _GEOMETRY_EPS)
        & (state.velocity[2] < 0.0)
    )
    bounce = falling & (state.velocity[2] < -physics.ground_settle_speed_mps)
    spin_coupling = falling & (
        state.velocity[2] < -physics.bounce_spin_min_vertical_speed_mps
    )
    surface_velocity = geometry.radius_m * jnp.asarray(
        (-state.spin[1], state.spin[0]), dtype=state.velocity.dtype
    )
    tangential_delta = (
        -(1.0 + physics.bounce_tangential_restitution)
        * (physics.inertia_ratio / (1.0 + physics.inertia_ratio))
        * surface_velocity
    )
    spin_delta = jnp.asarray(
        (tangential_delta[1], -tangential_delta[0]), dtype=state.spin.dtype
    ) / (physics.inertia_ratio * geometry.radius_m)
    velocity = state.velocity.at[:2].add(
        jnp.where(spin_coupling, tangential_delta, jnp.zeros_like(tangential_delta))
    )
    spin = state.spin.at[:2].add(
        jnp.where(spin_coupling, spin_delta, jnp.zeros_like(spin_delta))
    )
    velocity = velocity.at[:2].multiply(
        jnp.where(bounce, physics.bounce_horizontal_retention, 1.0)
    )
    velocity = velocity.at[2].set(
        jnp.where(
            bounce,
            -physics.restitution * velocity[2],
            jnp.where(falling, 0.0, velocity[2]),
        )
    )
    impacted = BallState(
        state.position.at[2].set(jnp.where(falling, ground_height, state.position[2])),
        velocity,
        spin,
        state.live,
    )
    return jax.tree.map(lambda new, old: jnp.where(falling, new, old), impacted, state)


def step_free_ball(
    state: BallState,
    *,
    dt: float | jax.Array,
    geometry: BallGeometry = _DEFAULT_GEOMETRY,
    physics: BallPhysics = _DEFAULT_PHYSICS,
) -> BallState:
    """player contact가 없는 한 substep과 첫 turf impact를 전개한다."""

    dt_value = jnp.asarray(dt, dtype=state.position.dtype)
    predicted = advance_ball_smooth(
        state, dt=dt_value, geometry=geometry, physics=physics
    )
    ground_height = jnp.asarray(geometry.radius_m, dtype=state.position.dtype)
    immediate = (state.position[2] <= ground_height + _GEOMETRY_EPS) & (
        state.velocity[2] < -physics.ground_settle_speed_mps
    )
    crossed = (
        (state.position[2] > ground_height + _GEOMETRY_EPS)
        & (predicted.position[2] <= ground_height)
        & (predicted.velocity[2] < 0.0)
    )
    # Semi-implicit endpoint seam: ground band에서 위로 출발한 공이 같은 step 안에서
    # 중력에 의해 되돌아와 endpoint가 turf 아래가 되는 경우 impact를 놓치지 않는다.
    depart_reimpact = (
        (state.position[2] <= ground_height + _GEOMETRY_EPS)
        & (state.velocity[2] > 0.0)
        & (predicted.position[2] <= ground_height)
        & (predicted.velocity[2] < 0.0)
    )
    crossing = state.live & (immediate | crossed | depart_reimpact)
    vertical_delta = predicted.position[2] - state.position[2]
    fraction = (ground_height - state.position[2]) / jnp.where(
        jnp.abs(vertical_delta) > _DIV_EPS, vertical_delta, -1.0
    )
    fraction = jnp.where(
        immediate,
        0.0,
        jnp.where(depart_reimpact, 1.0, jnp.clip(fraction, 0.0, 1.0)),
    )

    def impact_branch(_: None) -> BallState:
        before = advance_ball_smooth(
            state, dt=dt_value * fraction, geometry=geometry, physics=physics
        )
        before = before._replace(position=before.position.at[2].set(ground_height))
        impacted = apply_ground_impact(before, geometry=geometry, physics=physics)
        return advance_ball_smooth(
            impacted,
            dt=dt_value * (1.0 - fraction),
            geometry=geometry,
            physics=physics,
        )

    return jax.lax.cond(crossing, impact_branch, lambda _: predicted, operand=None)


__all__ = [
    "BallGeometry",
    "BallPhysics",
    "advance_ball_smooth",
    "advance_supported_ground_motion",
    "apply_ground_impact",
    "step_free_ball",
]
