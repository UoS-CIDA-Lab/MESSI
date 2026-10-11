"""규칙 정책의 기존 고정-shape feature·좌표·영역 판정 leaf."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.environment.observation import SharedScene
from messi.policies.rule_config import RulePolicyConfig


class _RulePolicyFeatures(NamedTuple):
    """여러 intent 판단이 공유하는 policy 전용 고정-shape feature."""

    ball_distance_squared: jax.Array
    nearest_opponent_distance_squared: jax.Array
    receiver_space: jax.Array


def _policy_features(scene: SharedScene) -> _RulePolicyFeatures:
    opponent_distance_squared = jnp.where(
        scene.opponent, scene.distance_squared, jnp.inf
    )
    nearest_opponent_distance_squared = jnp.min(opponent_distance_squared, axis=-1)
    return _RulePolicyFeatures(
        ball_distance_squared=jnp.sum(
            scene.ball_relative_position * scene.ball_relative_position, axis=-1
        ),
        nearest_opponent_distance_squared=nearest_opponent_distance_squared,
        receiver_space=jnp.sqrt(nearest_opponent_distance_squared + 1.0e-6),
    )


def _target_relative_velocity(scene: SharedScene, target: jax.Array) -> jax.Array:
    """선택된 teammate에 대해서만 team-frame 상대속도를 생성한다."""

    return scene.velocity_team[target] - scene.velocity_team


def _receiver_arrival_eta(
    pass_distance: jax.Array,
    max_speed: jax.Array,
    config: RulePolicyConfig,
) -> jax.Array:
    """패스 거리 비율과 고정 전방 lead로 정한 receiver 이동시간 prior.

    실제 receiver 위치와 planned target 사이의 변위를 측정한 ETA가 아니다.
    기본값은 거리 항을 사용하지 않고 forward lead만 가상 이동거리로 사용한다.
    """

    receiver_travel = jnp.asarray(
        config.receiver_arrival_distance_fraction, dtype=jnp.float32
    ) * pass_distance + jnp.asarray(
        config.planned_receiver_forward_lead_m, dtype=jnp.float32
    )
    return receiver_travel / jnp.maximum(max_speed, jnp.float32(1.0e-3))


def _inside_attacking_penalty_area(
    position_team: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """공격 방향이 +x인 authoritative SI 좌표에서 penalty area를 판정한다."""

    return (
        position_team[:, 0]
        >= jnp.asarray(
            config.field_half_length_m - config.penalty_area_length_m, jnp.float32
        )
    ) & (
        jnp.abs(position_team[:, 1])
        <= jnp.asarray(config.penalty_area_half_width_m, jnp.float32)
    )


def _inside_own_penalty_area(scene: SharedScene, config: RulePolicyConfig) -> jax.Array:
    return (
        scene.position_team[:, 0]
        <= -jnp.float32(config.field_half_length_m - config.penalty_area_length_m)
    ) & (
        jnp.abs(scene.position_team[:, 1])
        <= jnp.float32(config.penalty_area_half_width_m)
    )
