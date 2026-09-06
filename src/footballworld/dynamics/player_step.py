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
from footballworld.dynamics.separation import PlayerImpactFacts, separate_players
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


def _collision_microstep_count(
    players: PlayerState,
    *,
    dt: float,
    body: BodyContact,
    physics: PlayerPhysics,
) -> jax.Array:
    """Bound player translation by realised or reachable locomotion speed.

    Relative translation is bounded by the two largest active per-player
    supports. ``step_velocity`` is a convex move toward a target capped by the
    player's maximum speed, so ``max(realised_speed, max_speed)`` is a tighter
    transition bound than an unconstrained acceleration increment. This is a
    practical anti-tunnelling guard, not continuous collision detection. Dense
    Jacobi responses are separately degree-normalized so simultaneous
    neighbours do not each add a full isolated-pair impulse.
    """

    dtype = players.position.dtype
    realised_speed = jnp.sqrt(
        jnp.sum(players.velocity * players.velocity, axis=-1) + SAFE_NORM_EPS
    )
    # step_velocity is a convex move toward a target whose magnitude cannot
    # exceed max_speed.  Its result therefore cannot exceed the larger of the
    # realised and target-speed bounds.  Using acceleration * dt here is both
    # looser than that contract and lets otherwise finite custom acceleration
    # coefficients manufacture an unbounded dynamic loop count.
    speed_support = jnp.where(
        players.active,
        jnp.maximum(realised_speed, players.max_speed),
        0.0,
    )
    first_index = jnp.argmax(speed_support)
    first = speed_support[first_index]
    indices = jnp.arange(speed_support.shape[0], dtype=first_index.dtype)
    second = jnp.max(jnp.where(indices != first_index, speed_support, 0.0))
    relative_speed_support = first + second
    safe_torso_depth = jnp.asarray(
        max(body.torso_depth_m - GEOMETRY_EPS, GEOMETRY_EPS),
        dtype=dtype,
    )
    raw_count = jnp.ceil(
        relative_speed_support * jnp.asarray(dt, dtype=dtype) / safe_torso_depth
    )
    # Public configurations are rejected on the host before reaching this
    # bound. Keep one overflow sentinel here for reconstructed traced states;
    # step_players then freezes those players instead of silently tunnelling or
    # executing an effectively unbounded dynamic loop.
    bounded_float = jnp.where(
        jnp.isfinite(raw_count),
        jnp.clip(raw_count, 1.0, MAX_PLAYER_COLLISION_MICROSTEPS + 1.0),
        MAX_PLAYER_COLLISION_MICROSTEPS + 1.0,
    )
    return bounded_float.astype(jnp.int32)


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
    requested_microstep_count = _collision_microstep_count(
        players,
        dt=dt,
        body=body,
        physics=physics,
    )
    microstep_budget_valid = (
        requested_microstep_count <= MAX_PLAYER_COLLISION_MICROSTEPS
    )
    microstep_count = jnp.where(
        microstep_budget_valid, requested_microstep_count, jnp.int32(1)
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
        collision = separate_players(
            moved,
            attack_direction,
            field_half_extent,
            boundary_margin_m=boundary_margin_m,
            pinned=pinned,
            body=body,
            physics=physics,
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
