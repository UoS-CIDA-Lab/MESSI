"""규칙 정책의 순수 전술 선호·활성·피로 보간 leaf."""

import jax
import jax.numpy as jnp

from messi.policies.rule_config import (
    RuleTacticalPlan,
    _TACTIC_DIRECT_PASS_GAIN,
    _TACTIC_SHORT_PASS_GAIN,
    _TACTICAL_PROFILE,
    _TACTIC_FATIGUE_FLOOR,
    _TACTIC_FATIGUE_KNEE,
)


def _specialist_pass_bonus(plan, distance, progress, completion, arrival, open_play):
    """[N] receiver 후보의 metre/무차원 입력에서 bounded 선호 점수를 반환한다."""
    short = jnp.clip(distance / 6.0, 0.0, 1.0) * jnp.clip((28.0 - distance) / 12.0, 0.0, 1.0)
    direct = jnp.clip((distance - 18.0) / 16.0, 0.0, 1.0) * jnp.clip(progress / 20.0, 0.0, 1.0)
    safe = (completion >= 0.65) & (arrival >= 0.35)
    value = completion * (
        _TACTIC_SHORT_PASS_GAIN[plan] * short * (progress >= -4.0)
        + _TACTIC_DIRECT_PASS_GAIN[plan] * direct
    )
    return jnp.where(open_play & safe, value, 0.0)


def _specialist_long_service(plan, distance, progress, completion, open_play):
    """Catenaccio의 전진 long PASS만 lofted service로 바꾼다. 슛은 바꾸지 않는다."""
    return (
        (plan == int(RuleTacticalPlan.CATENACCIO)) & open_play
        & (distance >= 25.0) & (progress >= 18.0) & (completion >= 0.65)
    )


def _specialist_macro_scale(direct, box_entry_value):
    """안전한 direct outlet의 macro 배율. 더 좋은 박스 진입은 공통 base 값으로 둔다."""
    return jnp.where(box_entry_value > 0.0, 1.0, jnp.where(direct, 1.25, 1.0))


def _fresh_receiver_pressure(scene, plan, restart_kind, episode_ticks, config):
    """[N] GEGEN 상대 수신 초반 압박. 팀 상실 이력을 복원하지 않는다."""
    owner_index = jnp.argmax(scene.possessor)
    opponent_owner = jnp.any(scene.possessor) & (scene.team_id != scene.team_id[owner_index])
    return (
        scene.ball_live & opponent_owner & (restart_kind == 0)
        & (plan == int(RuleTacticalPlan.GEGENPRESSING))
        & (episode_ticks > 0)
        & (episode_ticks <= config.fresh_receiver_pressure_control_ticks)
    )


def _effective_tactical_profile(
    player_tactical_plan: jax.Array, stamina_long: jax.Array
) -> jax.Array:
    """장기 stamina에 맞춰 공격적 전술 편차를 DFL_BASE 쪽으로 보간한다."""

    raw = _TACTICAL_PROFILE[player_tactical_plan]
    reserve = jnp.clip(
        jnp.asarray(stamina_long, jnp.float32) / _TACTIC_FATIGUE_KNEE,
        0.0,
        1.0,
    )
    reserve = reserve * reserve * (jnp.float32(3.0) - jnp.float32(2.0) * reserve)
    floor = _TACTIC_FATIGUE_FLOOR[player_tactical_plan]
    intensity = floor + (jnp.float32(1.0) - floor) * reserve
    neutral = _TACTICAL_PROFILE[jnp.int32(RuleTacticalPlan.DFL_BASE)]
    return neutral + intensity[:, None] * (raw - neutral)
