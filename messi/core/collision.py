"""상삼각 pair로 계산하는 원형 player body collision."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.collision_config import BodyCollision, CollisionPhysics
from messi.core.constants import NO_PLAYER
from messi.core.state import PlayerState

_DIV_EPS = 1.0e-9
_GEOMETRY_EPS = 1.0e-6
_SQUARED_EPS = 1.0e-12
_COINCIDENT_EPS = 1.0e-4
_GOLDEN_ANGLE = 2.399963229728653


class PlayerImpactFacts(NamedTuple):
    """한 physics substep에서 가장 강한 상대 team body impact."""

    occurred: jax.Array
    actor: jax.Array
    victim: jax.Array
    contact_normal: jax.Array
    contact_position: jax.Array
    actor_attribution_decisive: jax.Array
    impact_score: jax.Array
    velocity_alignment: jax.Array
    time_fraction: jax.Array


class SeparationStep(NamedTuple):
    """분리된 선수 상태와 fixed-shape transient impact."""

    players: PlayerState
    impact: PlayerImpactFacts


def _empty_impact(dtype) -> PlayerImpactFacts:
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


def _tie_directions(team_id: jax.Array, attack_direction: jax.Array) -> jax.Array:
    indices = jnp.arange(team_id.shape[0])
    same_team = team_id[:, None] == team_id[None, :]
    rank = jnp.sum(same_team & (indices[None, :] <= indices[:, None]), axis=1) - 1
    angle = rank.astype(jnp.float32) * _GOLDEN_ANGLE
    direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    return direction * attack_direction[team_id, None]


def _pair_sum(first, second, count, value_first, value_second):
    """Pair 기여를 선수축으로 결정적으로 합산한다.

    ``zeros.at[first].add(...)``는 ``first``에 같은 slot이 여러 번 나오므로 GPU에서
    atomic으로 낮아지고, float 덧셈 순서가 실행마다 달라져 결과가 재현되지 않는다.
    긴 rollout은 혼돈계라 마지막 비트 차이가 증폭된다. Pair 좌표는 유일하므로
    dense ``[N, N]`` buffer에 ``set``한 뒤 고정 축으로 reduce하면 순서가 고정된다.
    N이 작아(22) dense buffer가 pair 수와 같은 자릿수다.
    """

    tail = value_first.shape[1:]
    dense_first = jnp.zeros((count, count) + tail, dtype=value_first.dtype)
    dense_first = dense_first.at[first, second].set(value_first)
    dense_second = jnp.zeros((count, count) + tail, dtype=value_second.dtype)
    dense_second = dense_second.at[first, second].set(value_second)
    return jnp.sum(dense_first, axis=1) + jnp.sum(dense_second, axis=0)


def separate_players(
    players: PlayerState,
    team_id: jax.Array,
    attack_direction: jax.Array,
    field_half_extent: jax.Array,
    *,
    boundary_margin_m: float,
    pinned: jax.Array | None = None,
    body: BodyCollision = BodyCollision(),
    physics: CollisionPhysics = CollisionPhysics(),
) -> SeparationStep:
    """면적 등가 원형 body의 overlap, impulse와 bounded separation을 계산한다."""

    position = players.position
    velocity_before = players.velocity
    active = players.active
    count = position.shape[0]
    if count < 2:
        return SeparationStep(
            players=players._replace(
                velocity=jnp.where(active[:, None], velocity_before, 0.0)
            ),
            impact=_empty_impact(position.dtype),
        )

    pinned = (
        jnp.zeros(count, dtype=jnp.bool_)
        if pinned is None
        else jnp.asarray(pinned, dtype=jnp.bool_)
    )
    movable = active & (~pinned)
    first, second = jnp.triu_indices(count, k=1)
    pair_enabled = active[first] & active[second] & (~(pinned[first] & pinned[second]))
    tie = _tie_directions(team_id, attack_direction)
    bound = jnp.asarray(field_half_extent, dtype=position.dtype) + jnp.asarray(
        boundary_margin_m, dtype=position.dtype
    )
    diameter = jnp.asarray(body.effective_diameter_m, dtype=position.dtype)

    def pair_geometry(current_position):
        difference = current_position[first] - current_position[second]
        distance = jnp.sqrt(jnp.sum(difference * difference, axis=-1) + _SQUARED_EPS)
        tie_difference = tie[first] - tie[second]
        tie_norm = jnp.sqrt(
            jnp.sum(tie_difference * tie_difference, axis=-1, keepdims=True)
            + _SQUARED_EPS
        )
        fallback = tie_difference / (tie_norm + _DIV_EPS)
        direction = jnp.where(
            (distance < _COINCIDENT_EPS)[:, None],
            fallback,
            difference / (distance[:, None] + _DIV_EPS),
        )
        return direction, distance

    def apply_separation(current_position, direction, distance):
        overlap = jnp.where(
            pair_enabled, jnp.clip(diameter - distance, 0.0, diameter), 0.0
        )
        direction_first = direction
        direction_second = -direction
        blocked_first = (
            (current_position[first] <= -bound + _GEOMETRY_EPS)
            & (direction_first < 0.0)
        ) | (
            (current_position[first] >= bound - _GEOMETRY_EPS) & (direction_first > 0.0)
        )
        blocked_second = (
            (current_position[second] <= -bound + _GEOMETRY_EPS)
            & (direction_second < 0.0)
        ) | (
            (current_position[second] >= bound - _GEOMETRY_EPS)
            & (direction_second > 0.0)
        )
        feasible_first = jnp.where(blocked_first, 0.0, direction_first)
        feasible_second = jnp.where(blocked_second, 0.0, direction_second)
        effectiveness_first = jnp.maximum(
            jnp.sum(feasible_first * direction_first, axis=-1), 0.0
        )
        effectiveness_second = jnp.maximum(
            jnp.sum(feasible_second * direction_second, axis=-1), 0.0
        )
        capacity_first = effectiveness_first * movable[first]
        capacity_second = effectiveness_second * movable[second]
        capacity = capacity_first + capacity_second
        distance_first = jnp.where(
            effectiveness_first > _DIV_EPS,
            overlap
            * capacity_first
            / (capacity + _DIV_EPS)
            / (effectiveness_first + _DIV_EPS),
            0.0,
        )
        distance_second = jnp.where(
            effectiveness_second > _DIV_EPS,
            overlap
            * capacity_second
            / (capacity + _DIV_EPS)
            / (effectiveness_second + _DIV_EPS),
            0.0,
        )
        correction = _pair_sum(
            first,
            second,
            count,
            distance_first[:, None] * feasible_first,
            distance_second[:, None] * feasible_second,
        )
        return jnp.clip(current_position + correction, -bound, bound)

    direction, distance = pair_geometry(position)
    overlap = jnp.where(pair_enabled, jnp.clip(diameter - distance, 0.0, diameter), 0.0)
    relative_velocity = velocity_before[first] - velocity_before[second]
    closing_speed = jnp.maximum(-jnp.sum(relative_velocity * direction, axis=-1), 0.0)
    responding = (
        pair_enabled & (overlap > _GEOMETRY_EPS) & (closing_speed > _GEOMETRY_EPS)
    )
    responding_count = responding.astype(jnp.int32)
    incident = _pair_sum(first, second, count, responding_count, responding_count)
    divisor = jnp.maximum(jnp.maximum(incident[first], incident[second]), 1).astype(
        position.dtype
    )
    response = (1.0 + physics.normal_restitution) * closing_speed / divisor
    first_magnitude = jnp.where(
        responding,
        response * movable[first] * jnp.where(pinned[second], 1.0, 0.5),
        0.0,
    )
    second_magnitude = jnp.where(
        responding,
        response * movable[second] * jnp.where(pinned[first], 1.0, 0.5),
        0.0,
    )
    velocity_correction = _pair_sum(
        first,
        second,
        count,
        first_magnitude[:, None] * direction,
        -second_magnitude[:, None] * direction,
    )
    budget = _pair_sum(first, second, count, first_magnitude, second_magnitude)
    correction_norm = jnp.sqrt(
        jnp.sum(velocity_correction * velocity_correction, axis=-1) + _SQUARED_EPS
    )
    velocity = (
        velocity_before
        + velocity_correction
        * jnp.minimum(budget / (correction_norm + _DIV_EPS), 1.0)[:, None]
    )

    impact_candidate = responding & (team_id[first] != team_id[second])
    score = jnp.where(impact_candidate, closing_speed, -1.0)
    strongest_pair = jnp.argmax(score)
    occurred = jnp.any(impact_candidate)
    safe_pair = jnp.where(occurred, strongest_pair, 0)
    first_player = first[safe_pair]
    second_player = second[safe_pair]
    normal = direction[safe_pair]
    first_velocity = velocity_before[first_player]
    second_velocity = velocity_before[second_player]
    first_toward = jnp.maximum(-jnp.sum(first_velocity * normal), 0.0)
    second_toward = jnp.maximum(jnp.sum(second_velocity * normal), 0.0)
    actor_is_first = first_toward >= second_toward
    decisive = jnp.abs(first_toward - second_toward) > _GEOMETRY_EPS
    actor = jnp.where(actor_is_first, first_player, second_player).astype(jnp.int32)
    victim = jnp.where(actor_is_first, second_player, first_player).astype(jnp.int32)
    actor_normal = jnp.where(actor_is_first, -normal, normal)
    alignment = jnp.clip(
        jnp.dot(first_velocity, second_velocity)
        / jnp.maximum(
            jnp.linalg.norm(first_velocity) * jnp.linalg.norm(second_velocity),
            jnp.asarray(1.0e-9, dtype=position.dtype),
        ),
        0.0,
        1.0,
    )
    impact = PlayerImpactFacts(
        occurred=occurred,
        actor=jnp.where(occurred, actor, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(occurred, victim, NO_PLAYER).astype(jnp.int32),
        contact_normal=jnp.where(
            occurred, actor_normal, jnp.zeros(2, dtype=position.dtype)
        ).astype(jnp.float32),
        contact_position=jnp.where(
            occurred,
            0.5 * (position[first_player] + position[second_player]),
            jnp.zeros(2, dtype=position.dtype),
        ).astype(jnp.float32),
        actor_attribution_decisive=occurred & decisive,
        impact_score=jnp.where(occurred, closing_speed[safe_pair], 0.0).astype(
            jnp.float32
        ),
        velocity_alignment=jnp.where(occurred, alignment, 0.0).astype(jnp.float32),
        time_fraction=jnp.where(occurred, 1.0, 0.0).astype(jnp.float32),
    )

    position = apply_separation(position, direction, distance)

    def separate_once(_, current_position):
        current_direction, current_distance = pair_geometry(current_position)
        return apply_separation(current_position, current_direction, current_distance)

    position = jax.lax.fori_loop(
        1, physics.separation_iterations, separate_once, position
    )
    at_lower = position <= -bound + _GEOMETRY_EPS
    at_upper = position >= bound - _GEOMETRY_EPS
    outward = (at_lower & (velocity < 0.0)) | (at_upper & (velocity > 0.0))
    velocity = jnp.where(outward, 0.0, velocity)
    velocity = jnp.where(active[:, None], velocity, 0.0)
    position = jnp.where(active[:, None], position, players.position)
    return SeparationStep(
        players=players._replace(
            position=position.astype(jnp.float32),
            velocity=velocity.astype(jnp.float32),
        ),
        impact=impact,
    )


__all__ = ["PlayerImpactFacts", "SeparationStep", "separate_players"]
