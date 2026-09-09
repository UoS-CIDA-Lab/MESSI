"""Composed player transition for one physics substep."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.constants import (
    DIV_EPS,
    GEOMETRY_EPS,
    MAX_PLAYER_COLLISION_MICROSTEPS,
    NO_PLAYER,
    SAFE_NORM_EPS,
)
from footballworld.core.state import PlayerState
from footballworld.dynamics.movement import step_player_motion
from footballworld.dynamics.separation import (
    PlayerImpactFacts,
    SeparationStep,
    separate_players,
)
from footballworld.dynamics.stamina import effective_speed_limit, step_stamina


class PlayerStep(NamedTuple):
    """Player transition, causal contact path, and strongest opponent impact.

    ``contact_position`` and ``contact_velocity`` are the end of the same
    locomotion transition without player separation or collision impulses.
    They are internal contact-scheduler inputs, not a second authoritative
    player state.
    """

    players: PlayerState
    contact_position: jax.Array
    contact_velocity: jax.Array
    impact: PlayerImpactFacts


class _CollisionMicrostepPlan(NamedTuple):
    """Conservative collision schedule and its broadphase gate."""

    count: jax.Array
    collision_possible: jax.Array


def _empty_player_impact(dtype: jnp.dtype) -> PlayerImpactFacts:
    """Return the fixed-shape identity for strongest-impact reduction."""

    return PlayerImpactFacts(
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        victim=jnp.int32(NO_PLAYER),
        contact_normal=jnp.zeros(2, dtype=dtype),
        contact_position=jnp.zeros(2, dtype=dtype),
        actor_attribution_decisive=jnp.bool_(False),
        impact_score=jnp.asarray(0.0, dtype=dtype),
        velocity_alignment=jnp.asarray(0.0, dtype=dtype),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
    )


def _stronger_impact(
    current: PlayerImpactFacts,
    candidate: PlayerImpactFacts,
) -> PlayerImpactFacts:
    """Keep the earliest impact among equal strongest microstep scores."""

    replace = candidate.occurred & (
        (~current.occurred) | (candidate.impact_score > current.impact_score)
    )
    return jax.tree_util.tree_map(
        lambda old, new: jnp.where(replace, new, old),
        current,
        candidate,
    )


def _rotate_body_forward(
    current: jax.Array,
    target: jax.Array,
    target_valid: jax.Array,
    *,
    maximum_turn_radians: jax.Array,
) -> jax.Array:
    """Turn unit body vectors along the shortest arc without angle wrapping."""

    current_norm = jnp.sqrt(
        jnp.sum(current * current, axis=-1, keepdims=True) + SAFE_NORM_EPS
    )
    current = current / (current_norm + DIV_EPS)
    target_norm = jnp.sqrt(
        jnp.sum(target * target, axis=-1, keepdims=True) + SAFE_NORM_EPS
    )
    target = target / (target_norm + DIV_EPS)

    dot = jnp.clip(jnp.sum(current * target, axis=-1), -1.0, 1.0)
    cross = current[:, 0] * target[:, 1] - current[:, 1] * target[:, 0]
    turn_sign = jnp.where(jnp.abs(cross) > DIV_EPS, jnp.sign(cross), 1.0)
    turn = jnp.minimum(jnp.maximum(maximum_turn_radians, 0.0), jnp.pi)
    cosine = jnp.cos(turn)
    sine = jnp.sin(turn) * turn_sign
    rotated = jnp.stack(
        [
            cosine * current[:, 0] - sine * current[:, 1],
            sine * current[:, 0] + cosine * current[:, 1],
        ],
        axis=-1,
    )
    reached = dot >= cosine
    updated = jnp.where(reached[:, None], target, rotated)
    valid = jnp.asarray(target_valid, dtype=bool) & (target_norm[:, 0] > DIV_EPS)
    return jnp.where(valid[:, None], updated, current).astype(jnp.float32)


def _limit_self_propelled_target(
    desired_velocity: jax.Array,
    speed_limit: jax.Array,
    body_forward: jax.Array,
    *,
    backward_speed_ratio: float,
) -> jax.Array:
    """Continuously limit only the requested target when travelling backward."""

    desired_speed = jnp.sqrt(
        jnp.sum(desired_velocity * desired_velocity, axis=-1) + SAFE_NORM_EPS
    )
    direction = desired_velocity / (desired_speed[:, None] + DIV_EPS)
    backness = jnp.maximum(-jnp.sum(direction * body_forward, axis=-1), 0.0)
    directional_fraction = 1.0 - (1.0 - backward_speed_ratio) * backness
    directional_limit = speed_limit * directional_fraction
    scale = jnp.minimum(directional_limit / (desired_speed + DIV_EPS), 1.0)
    return desired_velocity * scale[:, None]


def _two_largest_speed_sum(speed_support: jax.Array) -> jax.Array:
    """Return the sum of the two largest fixed-roster speed supports."""

    first_index = jnp.argmax(speed_support)
    first = speed_support[first_index]
    indices = jnp.arange(speed_support.shape[0], dtype=first_index.dtype)
    second = jnp.max(jnp.where(indices != first_index, speed_support, 0.0))
    return first + second


def _bounded_microstep_count(
    relative_speed_support: jax.Array,
    *,
    dt: float,
    body: BodyContact,
) -> jax.Array:
    """Convert a relative speed envelope to a bounded loop count."""

    dtype = relative_speed_support.dtype
    safe_torso_depth = jnp.asarray(
        max(body.torso_depth_m - GEOMETRY_EPS, GEOMETRY_EPS),
        dtype=dtype,
    )
    raw_count = jnp.ceil(
        relative_speed_support * jnp.asarray(dt, dtype=dtype) / safe_torso_depth
    )
    bounded_float = jnp.where(
        jnp.isfinite(raw_count),
        jnp.clip(raw_count, 1.0, MAX_PLAYER_COLLISION_MICROSTEPS + 1.0),
        MAX_PLAYER_COLLISION_MICROSTEPS + 1.0,
    )
    return bounded_float.astype(jnp.int32)


def _collision_microstep_plan(
    players: PlayerState,
    desired_velocity: jax.Array,
    movement_enabled: jax.Array,
    position_update_enabled: jax.Array,
    *,
    dt: float,
    body: BodyContact,
    physics: PlayerPhysics,
    locomotion_microstep_count: jax.Array | None = None,
    realised_speed: jax.Array | None = None,
) -> _CollisionMicrostepPlan:
    """Bound player translation by speed reachable during this interval.

    Relative translation is bounded by the two largest active per-player
    supports. ``step_velocity`` is a convex move toward the requested target,
    with its vector change inside the configured acceleration ellipse. The
    largest ellipse axis therefore bounds the speed reachable from the current
    velocity during this interval. This avoids treating every player's profile
    maximum as instantaneously reachable while preserving the relative-path
    anti-tunnelling bound for players that are already moving quickly.

    This is a practical anti-tunnelling guard, not continuous collision
    detection. Dense Jacobi responses are separately degree-normalized so
    simultaneous neighbours do not each add a full isolated-pair impulse.
    """

    if realised_speed is None:
        realised_speed = jnp.sqrt(
            jnp.sum(players.velocity * players.velocity, axis=-1) + SAFE_NORM_EPS
        )
    target_speed = jnp.minimum(
        jnp.sqrt(jnp.sum(desired_velocity * desired_velocity, axis=-1) + SAFE_NORM_EPS),
        players.max_speed,
    )
    maximum_acceleration = max(
        physics.forward_acceleration_mps2,
        physics.lateral_acceleration_mps2,
        physics.braking_deceleration_mps2,
    )
    reachable_target_speed = jnp.minimum(
        target_speed,
        realised_speed
        + jnp.asarray(maximum_acceleration * dt, dtype=players.position.dtype),
    )
    translation_speed = jnp.where(
        movement_enabled,
        jnp.maximum(realised_speed, reachable_target_speed),
        realised_speed,
    )
    reachable_speed_support = jnp.where(
        players.active & position_update_enabled,
        translation_speed,
        0.0,
    )

    reachable_count = _bounded_microstep_count(
        _two_largest_speed_sum(reachable_speed_support),
        dt=dt,
        body=body,
    )
    count = players.position.shape[0]
    pair_enabled = (
        players.active[:, None]
        & players.active[None, :]
        & jnp.triu(jnp.ones((count, count), dtype=jnp.bool_), k=1)
    )
    relative_travel = (
        reachable_speed_support[:, None] + reachable_speed_support[None, :]
    ) * jnp.asarray(dt, dtype=players.position.dtype)
    broadphase_diameter = jnp.asarray(
        max(body.shoulder_width_m, body.torso_depth_m),
        dtype=players.position.dtype,
    )
    broadphase_distance = broadphase_diameter + relative_travel
    difference = players.position[:, None, :] - players.position[None, :, :]
    distance_squared = jnp.sum(difference * difference, axis=-1)
    collision_possible = jnp.any(
        pair_enabled & (distance_squared <= broadphase_distance * broadphase_distance)
    )
    profile_count = (
        _locomotion_microstep_count(
            players,
            dt=dt,
            body=body,
            realised_speed=realised_speed,
        )
        if locomotion_microstep_count is None
        else locomotion_microstep_count
    )
    return _CollisionMicrostepPlan(
        count=jnp.where(collision_possible, profile_count, reachable_count),
        collision_possible=collision_possible,
    )


def _collision_microstep_count(
    players: PlayerState,
    desired_velocity: jax.Array,
    movement_enabled: jax.Array,
    position_update_enabled: jax.Array,
    *,
    dt: float,
    body: BodyContact,
    physics: PlayerPhysics,
    locomotion_microstep_count: jax.Array | None = None,
) -> jax.Array:
    """Preserve the scalar private helper contract for focused callers."""

    return _collision_microstep_plan(
        players,
        desired_velocity,
        movement_enabled,
        position_update_enabled,
        dt=dt,
        body=body,
        physics=physics,
        locomotion_microstep_count=locomotion_microstep_count,
    ).count


def _locomotion_microstep_count(
    players: PlayerState,
    *,
    dt: float,
    body: BodyContact,
    realised_speed: jax.Array | None = None,
) -> jax.Array:
    """Preserve the profile-based integration and fail-closed budget."""

    if realised_speed is None:
        realised_speed = jnp.sqrt(
            jnp.sum(players.velocity * players.velocity, axis=-1) + SAFE_NORM_EPS
        )
    profile_speed_support = jnp.where(
        players.active,
        jnp.maximum(realised_speed, players.max_speed),
        0.0,
    )
    return _bounded_microstep_count(
        _two_largest_speed_sum(profile_speed_support),
        dt=dt,
        body=body,
    )


def _collision_microstep_due(
    microstep_index: jax.Array,
    microstep_count: jax.Array,
    collision_microstep_count: jax.Array,
) -> jax.Array:
    """Schedule checks without exceeding the requested path interval."""

    stride = jnp.maximum(microstep_count // collision_microstep_count, 1)
    completed = microstep_index + 1
    return (completed % stride == 0) | (completed == microstep_count)


def step_players(
    players: PlayerState,
    desired_velocity: jax.Array,
    movement_enabled: jax.Array,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    *,
    boundary_margin_m: float,
    dt: float,
    body_target: jax.Array,
    body_target_valid: jax.Array,
    position_update_enabled: jax.Array | None = None,
    pinned: jax.Array | None = None,
    physics: PlayerPhysics = PlayerPhysics(),
    body: BodyContact = BodyContact(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
) -> PlayerStep:
    """Apply body turn, locomotion, stamina, and adaptive body collision.

    The public physics interval remains the clock used by ball and rule
    transitions. Player locomotion alone is split when the active roster's
    speed support could cross the torso's minimum depth in one interval. The
    existing dense separator is staged once inside a dynamic loop rather than
    adding another pairwise collision graph.
    """

    active = players.active
    position_update_enabled = (
        jnp.ones_like(active)
        if position_update_enabled is None
        else jnp.asarray(position_update_enabled, dtype=bool)
    )
    locomotion = movement_enabled & position_update_enabled & active
    body_forward = _rotate_body_forward(
        players.body_forward,
        body_target,
        body_target_valid & active & position_update_enabled,
        maximum_turn_radians=physics.body_turn_rate_max_radps * dt,
    )
    oriented = players._replace(body_forward=body_forward)

    dtype = players.position.dtype
    realised_speed = jnp.sqrt(
        jnp.sum(players.velocity * players.velocity, axis=-1) + SAFE_NORM_EPS
    )
    requested_microstep_count = _locomotion_microstep_count(
        players,
        dt=dt,
        body=body,
        realised_speed=realised_speed,
    )
    collision_plan = _collision_microstep_plan(
        players,
        desired_velocity,
        movement_enabled,
        position_update_enabled,
        dt=dt,
        body=body,
        physics=physics,
        locomotion_microstep_count=requested_microstep_count,
        realised_speed=realised_speed,
    )
    requested_collision_microstep_count = collision_plan.count
    microstep_budget_valid = (
        requested_microstep_count <= MAX_PLAYER_COLLISION_MICROSTEPS
    )
    microstep_count = jnp.where(
        microstep_budget_valid, requested_microstep_count, jnp.int32(1)
    )
    collision_microstep_count = jnp.minimum(
        requested_collision_microstep_count,
        microstep_count,
    )
    microstep_dt = jnp.asarray(dt, dtype=dtype) / microstep_count.astype(dtype)

    def microstep(microstep_index, carry):
        current, strongest_impact, collision_corrected = carry
        speed_limit = effective_speed_limit(
            current.max_speed,
            current.stamina_long,
            current.stamina_short,
            long=long_stamina,
            short=short_stamina,
        )
        locomotion_target = _limit_self_propelled_target(
            desired_velocity,
            speed_limit,
            body_forward,
            backward_speed_ratio=physics.backward_speed_ratio,
        )
        moved = step_player_motion(
            current,
            locomotion_target,
            speed_limit,
            locomotion,
            field_half_extent,
            boundary_margin_m=boundary_margin_m,
            dt=microstep_dt,
            position_update_enabled=position_update_enabled,
            config=physics,
        )
        collision_due = collision_plan.collision_possible & (
            _collision_microstep_due(
                microstep_index,
                microstep_count,
                collision_microstep_count,
            )
        )

        def resolve_collision(current_players):
            return separate_players(
                current_players,
                attack_direction,
                field_half_extent,
                boundary_margin_m=boundary_margin_m,
                pinned=pinned,
                body=body,
                physics=physics,
            )

        collision = jax.lax.cond(
            collision_due,
            resolve_collision,
            lambda current_players: SeparationStep(
                players=current_players,
                impact=_empty_player_impact(dtype),
            ),
            moved,
        )
        impact = collision.impact._replace(
            time_fraction=(
                jnp.asarray(microstep_index, dtype=dtype)
                + jnp.asarray(1.0, dtype=dtype)
            )
            / microstep_count.astype(dtype)
        )
        stamina = step_stamina(
            current.stamina_long,
            current.stamina_short,
            # Collision impulses are external work. Charge only the locomotion
            # velocity immediately before this microstep's body response.
            moved.velocity,
            current.velocity,
            active,
            locomotion,
            current.max_speed,
            current.endurance_factor,
            dt=microstep_dt,
            long=long_stamina,
            short=short_stamina,
        )
        next_players = collision.players._replace(
            stamina_long=stamina.long,
            stamina_short=stamina.short,
        )
        corrected_now = jnp.any(collision.players.position != moved.position) | jnp.any(
            collision.players.velocity != moved.velocity
        )
        return (
            next_players,
            _stronger_impact(strongest_impact, impact),
            collision_corrected | corrected_now,
        )

    final_players, strongest_impact, collision_corrected = jax.lax.fori_loop(
        0,
        microstep_count,
        microstep,
        (oriented, _empty_player_impact(dtype), jnp.bool_(False)),
    )
    # Invalid reconstructed states fail closed. The host validation makes this
    # select an identity only outside the supported public rollout domain.
    final_players = jax.tree_util.tree_map(
        lambda new, old: jnp.where(microstep_budget_valid, new, old),
        final_players,
        players,
    )
    strongest_impact = jax.tree_util.tree_map(
        lambda new, old: jnp.where(microstep_budget_valid, new, old),
        strongest_impact,
        _empty_player_impact(dtype),
    )
    collision_corrected = collision_corrected & microstep_budget_valid

    def collision_free_contact_path(initial: PlayerState) -> PlayerState:
        """Replay only O(N) locomotion when body response changed the path."""

        def contact_microstep(_, current):
            speed_limit = effective_speed_limit(
                current.max_speed,
                current.stamina_long,
                current.stamina_short,
                long=long_stamina,
                short=short_stamina,
            )
            locomotion_target = _limit_self_propelled_target(
                desired_velocity,
                speed_limit,
                body_forward,
                backward_speed_ratio=physics.backward_speed_ratio,
            )
            moved = step_player_motion(
                current,
                locomotion_target,
                speed_limit,
                locomotion,
                field_half_extent,
                boundary_margin_m=boundary_margin_m,
                dt=microstep_dt,
                position_update_enabled=position_update_enabled,
                config=physics,
            )
            stamina = step_stamina(
                current.stamina_long,
                current.stamina_short,
                moved.velocity,
                current.velocity,
                active,
                locomotion,
                current.max_speed,
                current.endurance_factor,
                dt=microstep_dt,
                long=long_stamina,
                short=short_stamina,
            )
            return moved._replace(
                stamina_long=stamina.long,
                stamina_short=stamina.short,
            )

        return jax.lax.fori_loop(0, microstep_count, contact_microstep, initial)

    contact_players = jax.lax.cond(
        collision_corrected,
        collision_free_contact_path,
        lambda _: final_players,
        oriented,
    )
    return PlayerStep(
        players=final_players,
        contact_position=contact_players.position,
        contact_velocity=contact_players.velocity,
        impact=strongest_impact,
    )


__all__ = ["PlayerStep", "step_players"]
