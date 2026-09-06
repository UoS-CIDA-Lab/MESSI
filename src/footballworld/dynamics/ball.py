"""Deterministic free-ball flight, ground impact, and rolling."""

import math

import jax
import jax.numpy as jnp

from footballworld.config.ball_physics import BallPhysics
from footballworld.config.geometry import Ball
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    SAFE_NORM_EPS,
    SQUARED_EPS,
)
from footballworld.core.state import BallState


def _norm(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector) + SAFE_NORM_EPS)


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
    *,
    physics: BallPhysics,
) -> jax.Array:
    """Blend Eq. (3)/(4) fits from *Soccer ball lift coefficients via
    trajectory analysis*: non-spinning drag and post-critical spin drag.
    """

    crisis_fraction = jax.nn.sigmoid(
        (physics.drag_crisis_speed_mps - speed) / physics.drag_crisis_width_mps
    )
    base = physics.drag_coefficient_high_re + (
        physics.drag_crisis_drop * crisis_fraction
    )
    spinning = physics.spin_drag_scale * jnp.power(
        jnp.maximum(spin_parameter, DIV_EPS), physics.spin_drag_exponent
    )

    # Goff and Carré report the spin fit only for post-critical flow and
    # Sp > 0.05, while explicitly warning that flow transitions are not
    # instantaneous. Reuse the fitted speed-crisis sigmoid and bridge the
    # first measured spin interval [Sp_min, 2 Sp_min] with a C1 smoothstep.
    post_crisis_fraction = 1.0 - crisis_fraction
    spin_interval = jnp.maximum(physics.spin_drag_min_parameter, DIV_EPS)
    spin_fraction = _smoothstep01(
        (spin_parameter - physics.spin_drag_min_parameter) / spin_interval
    )
    blend = post_crisis_fraction * spin_fraction
    return base + blend * (spinning - base)


def _air_velocity(
    velocity: jax.Array,
    spin: jax.Array,
    *,
    dt: float | jax.Array,
    radius: float,
    physics: BallPhysics,
) -> jax.Array:
    speed_squared = jnp.sum(velocity * velocity)
    speed = jnp.sqrt(speed_squared + SAFE_NORM_EPS)
    moving = speed_squared > SQUARED_EPS
    direction = velocity / speed

    perpendicular_spin = spin - jnp.dot(spin, direction) * direction
    perpendicular_spin_squared = jnp.sum(perpendicular_spin * perpendicular_spin)
    perpendicular_spin_speed = jnp.sqrt(perpendicular_spin_squared + SAFE_NORM_EPS)
    spin_parameter = jnp.where(
        moving,
        radius * perpendicular_spin_speed / speed,
        0.0,
    )

    drag_coefficient = _drag_coefficient(
        speed,
        spin_parameter,
        physics=physics,
    )

    area = math.pi * radius * radius
    force_scale = 0.5 * physics.air_density_kgpm3 * area / physics.ball_mass_kg
    drag_rate = force_scale * drag_coefficient * speed
    damped = velocity / (1.0 + drag_rate * dt)

    # "Soccer ball lift coefficients via trajectory analysis" supplies the
    # measured lift regime, while "Investigations into soccer aerodynamics via
    # trajectory analysis and dust experiments" supplies flow context. The
    # tanh shape and 0.42 cap are a bounded model choice, not a paper equation.
    lift_coefficient = physics.lift_coefficient_limit * jnp.tanh(
        spin_parameter / physics.lift_coefficient_limit
    )
    turn_rate = force_scale * lift_coefficient * speed
    lift_active = moving & (perpendicular_spin_squared > SQUARED_EPS)
    axis = perpendicular_spin / perpendicular_spin_speed
    rotated = _rotate(damped, axis, turn_rate * dt)
    return jnp.where(lift_active, rotated, damped)


def _ground_slip_step(
    velocity: jax.Array,
    spin: jax.Array,
    *,
    dt: float | jax.Array,
    radius: float,
    physics: BallPhysics,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    surface_velocity = radius * jnp.asarray([-spin[1], spin[0]], dtype=velocity.dtype)
    slip = velocity[:2] + surface_velocity
    inertia = physics.ball_inertia_ratio
    sticking_delta = -(inertia / (1.0 + inertia)) * slip
    sticking_speed = _norm(sticking_delta)
    maximum_delta = physics.ground_slide_friction * physics.g * dt
    fraction = jnp.minimum(1.0, maximum_delta / sticking_speed)
    velocity_delta = fraction * sticking_delta

    velocity = velocity.at[:2].add(velocity_delta)
    spin_delta = jnp.asarray(
        [velocity_delta[1], -velocity_delta[0]], dtype=spin.dtype
    ) / (inertia * radius)
    spin = spin.at[:2].add(spin_delta)
    rolling = sticking_speed <= maximum_delta + DIV_EPS
    return velocity, spin, rolling


def advance_supported_ground_motion(
    velocity: jax.Array,
    spin: jax.Array,
    *,
    dt: float | jax.Array,
    radius: float,
    physics: BallPhysics,
) -> tuple[jax.Array, jax.Array]:
    """Advance one supported-ball velocity/spin interval.

    This is the single source of truth for turf curl, slip-to-roll coupling,
    rolling resistance, and ground-spin decay. It deliberately excludes
    position and support detection so an observation-only policy can forecast
    one shared ground path without calling the complete hybrid-event solver or
    copying its coefficients.
    """

    ground_axis = jnp.array([0.0, 0.0, 1.0], dtype=velocity.dtype)
    ground_velocity = _rotate(
        velocity.at[2].set(0.0),
        ground_axis,
        physics.c_ground_curl * spin[2] * dt,
    )
    ground_velocity, ground_spin, rolling = _ground_slip_step(
        ground_velocity,
        spin,
        dt=dt,
        radius=radius,
        physics=physics,
    )

    horizontal_speed = _norm(ground_velocity[:2]) + DIV_EPS
    rolling_deceleration = jnp.interp(
        horizontal_speed,
        jnp.asarray(physics.roll_v_knots, dtype=velocity.dtype),
        jnp.asarray(physics.roll_d_knots, dtype=velocity.dtype),
    )
    rolling_scale = (
        jnp.maximum(0.0, horizontal_speed - rolling_deceleration * dt)
        / horizontal_speed
    )
    ground_velocity = ground_velocity.at[:2].multiply(
        jnp.where(rolling, rolling_scale, 1.0)
    )

    ground_spin = ground_spin.at[2].multiply(jnp.exp(-physics.ground_spin_decay * dt))
    rolling_spin = jnp.asarray(
        [
            -ground_velocity[1] / radius,
            ground_velocity[0] / radius,
        ],
        dtype=spin.dtype,
    )
    ground_spin = ground_spin.at[:2].set(
        jnp.where(rolling, rolling_spin, ground_spin[:2])
    )
    return ground_velocity, ground_spin


def advance_smooth(
    state: BallState,
    *,
    dt: float | jax.Array,
    geometry: Ball = Ball(),
    physics: BallPhysics = BallPhysics(),
) -> BallState:
    """Advance smooth forces without resolving a new surface impact.

    A ball already supported by the turf remains constrained to one ball
    radius above the pitch and receives slip, rolling, and ground-spin forces.
    An unsupported ball may cross the pitch plane; callers doing swept
    collision detection use that unconstrained endpoint to choose the first
    event on the segment.
    """

    position = state.position
    velocity = state.velocity
    spin = state.spin
    ground_height = jnp.asarray(geometry.radius, dtype=position.dtype)

    at_ground = position[2] <= ground_height + GEOMETRY_EPS
    supported = (
        at_ground & (velocity[2] <= 0.0) & (velocity[2] >= -physics.ground_settle_vz)
    )

    gravity = jnp.where(supported, 0.0, -physics.g)
    velocity_with_gravity = velocity + dt * jnp.array(
        [0.0, 0.0, gravity], dtype=velocity.dtype
    )
    air_velocity = _air_velocity(
        velocity_with_gravity,
        spin,
        dt=dt,
        radius=geometry.radius,
        physics=physics,
    )

    ground_velocity, ground_spin = advance_supported_ground_motion(
        velocity_with_gravity,
        spin,
        dt=dt,
        radius=geometry.radius,
        physics=physics,
    )

    velocity = jnp.where(supported, ground_velocity, air_velocity)
    position = position + dt * velocity
    position = position.at[2].set(jnp.where(supported, ground_height, position[2]))

    air_spin = spin * jnp.exp(-physics.air_spin_decay * dt)
    spin = jnp.where(supported, ground_spin, air_spin)

    stepped = BallState(
        position=position,
        velocity=velocity,
        spin=spin,
        live=state.live,
    )
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(state.live, new, old), stepped, state
    )


def apply_ground_impact(
    state: BallState,
    *,
    geometry: Ball = Ball(),
    physics: BallPhysics = BallPhysics(),
) -> BallState:
    """Apply one instantaneous pitch impact at the current ball position."""

    position = state.position
    velocity = state.velocity
    spin = state.spin
    ground_height = jnp.asarray(geometry.radius, dtype=position.dtype)
    falling = (
        state.live & (position[2] <= ground_height + GEOMETRY_EPS) & (velocity[2] < 0.0)
    )
    bounce = falling & (velocity[2] < -physics.ground_settle_vz)
    settle = falling & (~bounce)

    spin_coupling = falling & (velocity[2] < -physics.bounce_spin_vmin)
    inertia = physics.ball_inertia_ratio
    surface_velocity = geometry.radius * jnp.asarray(
        [-spin[1], spin[0]], dtype=velocity.dtype
    )
    tangential_delta = (
        -(1.0 + physics.bounce_tangential_e)
        * (inertia / (1.0 + inertia))
        * surface_velocity
    )
    spin_delta = jnp.asarray(
        [tangential_delta[1], -tangential_delta[0]], dtype=spin.dtype
    ) / (inertia * geometry.radius)
    velocity = velocity.at[:2].add(
        jnp.where(spin_coupling, tangential_delta, jnp.zeros_like(velocity[:2]))
    )
    spin = spin.at[:2].add(
        jnp.where(spin_coupling, spin_delta, jnp.zeros_like(spin[:2]))
    )

    velocity = velocity.at[:2].multiply(jnp.where(bounce, physics.bounce_h_keep, 1.0))
    velocity = velocity.at[2].set(
        jnp.where(
            bounce,
            -physics.e_rest * velocity[2],
            jnp.where(settle, 0.0, velocity[2]),
        )
    )
    position = position.at[2].set(jnp.where(falling, ground_height, position[2]))

    impacted = state._replace(
        position=position,
        velocity=velocity,
        spin=spin,
    )
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(falling, new, old), impacted, state
    )


def step_free_ball(
    state: BallState,
    *,
    dt: float | jax.Array,
    geometry: Ball = Ball(),
    physics: BallPhysics = BallPhysics(),
) -> BallState:
    """Advance one player-free substep, including the first turf impact."""

    predicted = advance_smooth(
        state,
        dt=dt,
        geometry=geometry,
        physics=physics,
    )
    ground_height = jnp.asarray(geometry.radius, dtype=state.position.dtype)
    immediate = (state.position[2] <= ground_height + GEOMETRY_EPS) & (
        state.velocity[2] < -physics.ground_settle_vz
    )
    crossed = (
        (state.position[2] > ground_height + GEOMETRY_EPS)
        & (predicted.position[2] <= ground_height)
        & (predicted.velocity[2] < 0.0)
    )
    # Every reachable sample must stay on or above the turf. Preserve the
    # chronological impact path and close only the semi-implicit seam: a ball
    # may start in the ground band with a small upward velocity yet finish the
    # same step below the pitch after gravity reverses it.  The straight endpoint
    # chord starts on the surface, so its algebraic root is zero even though the
    # representable re-impact belongs at this step's endpoint.
    depart_reimpact = (
        (state.position[2] <= ground_height + GEOMETRY_EPS)
        & (state.velocity[2] > 0.0)
        & (predicted.position[2] <= ground_height)
        & (predicted.velocity[2] < 0.0)
    )
    crossing = state.live & (immediate | crossed | depart_reimpact)
    vertical_delta = predicted.position[2] - state.position[2]
    fraction = (ground_height - state.position[2]) / jnp.where(
        jnp.abs(vertical_delta) > DIV_EPS,
        vertical_delta,
        -1.0,
    )
    fraction = jnp.where(
        immediate,
        0.0,
        jnp.where(depart_reimpact, 1.0, jnp.clip(fraction, 0.0, 1.0)),
    )

    def impact_branch(_: None) -> BallState:
        before = advance_smooth(
            state,
            dt=dt * fraction,
            geometry=geometry,
            physics=physics,
        )
        before = before._replace(position=before.position.at[2].set(ground_height))
        impacted = apply_ground_impact(
            before,
            geometry=geometry,
            physics=physics,
        )
        return advance_smooth(
            impacted,
            dt=dt * (1.0 - fraction),
            geometry=geometry,
            physics=physics,
        )

    return jax.lax.cond(
        crossing,
        impact_branch,
        lambda _: predicted,
        operand=None,
    )
