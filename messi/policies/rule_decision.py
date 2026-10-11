"""규칙 정책의 기존 receiver 보너스·점유 macro 가치 leaf."""

import jax
import jax.numpy as jnp

from messi.policies.rule_config import RulePolicyConfig
from messi.core.action import INTENT_SHOOT, INTENT_PASS, INTENT_CONTROL


def _inside_box_receiver_score_bonus(
    carrier_shot_quality: jax.Array,
    receiver_shot_quality: jax.Array,
    receiver_completion: jax.Array,
    receiver_in_service_zone: jax.Array,
    carrier_inside_box: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """더 좋은 박스 안 슛을 만드는 안전한 진입·연결 패스의 선택 보너스."""

    gain = jnp.maximum(
        jnp.asarray(receiver_shot_quality, jnp.float32)
        - jnp.asarray(carrier_shot_quality, jnp.float32),
        jnp.float32(0.0),
    )
    return jnp.where(
        jnp.asarray(receiver_in_service_zone, jnp.bool_)
        & (
            jnp.asarray(carrier_inside_box, jnp.bool_)
            | (
                (receiver_completion >= config.pass_completion_threshold)
                & (receiver_shot_quality >= config.shot_quality_threshold)
            )
        ),
        jnp.asarray(config.inside_box_receiver_shot_gain, jnp.float32)
        * jnp.asarray(receiver_completion, jnp.float32)
        * gain,
        jnp.float32(0.0),
    )


def _release_urgency_control_scale(
    possession_episode_ticks: jax.Array,
    viable_release: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """유효한 release가 있는 장기 점유에서만 CONTROL 가치를 완만히 낮춘다."""

    elapsed = jnp.asarray(possession_episode_ticks, jnp.float32)
    start = jnp.float32(config.release_urgency_start_ticks)
    duration = jnp.maximum(
        jnp.float32(
            config.release_urgency_full_ticks - config.release_urgency_start_ticks
        ),
        jnp.float32(1.0),
    )
    progress = jnp.clip((elapsed - start) / duration, 0.0, 1.0)
    scale = jnp.float32(1.0) - progress * (
        jnp.float32(1.0) - jnp.float32(config.release_urgency_control_floor)
    )
    return jnp.where(jnp.asarray(viable_release, jnp.bool_), scale, jnp.float32(1.0))


def _box_possession_macro_values(
    pass_value: jax.Array,
    control_value: jax.Array,
    carrier_shot_quality: jax.Array,
    receiver_shot_quality: jax.Array,
    target_completion: jax.Array,
    inside_attacking_box: jax.Array,
    target_in_service_zone: jax.Array,
    config: "RulePolicyConfig",
) -> tuple[jax.Array, jax.Array]:
    """박스 안에서 더 좋은 슛을 만들지 않는 recycle과 CONTROL만 할인한다."""

    receiver_gain = jnp.maximum(
        receiver_shot_quality - carrier_shot_quality, jnp.float32(0.0)
    )
    creates_better_shot = target_in_service_zone & (receiver_gain > 0.0)
    box_pass_value = jnp.where(
        creates_better_shot,
        jnp.clip(
            pass_value
            + jnp.asarray(config.inside_box_receiver_shot_gain, jnp.float32)
            * target_completion
            * receiver_gain,
            0.0,
            1.0,
        ),
        pass_value * jnp.asarray(config.inside_box_recycle_pass_scale, jnp.float32),
    )
    return (
        jnp.where(inside_attacking_box, box_pass_value, pass_value),
        jnp.where(
            inside_attacking_box,
            control_value
            * jnp.asarray(config.inside_box_control_macro_scale, jnp.float32),
            control_value,
        ),
    )


def _select_receiver(
    pass_score, safe_receiver, box_receiver_bonus, lane_completion,
    receiver_space_score, specialist_preference, pass_actor_valid,
    scene, restart_kind, pass_actor, config,
):
    """기존 안전 gate 뒤의 [N] 점수에서 score·int32[] target·bool[] 유효성을 반환한다.

    N>=1이고 scene 위치는 SI/team-frame이다. 안전 mask는 caller가 계산한다.
    동점은 첫 slot이며 모든 slot이 unsafe면 score=-inf, target=0, 유효성=false다.
    입력의 추가 검증이나 별도 물리 gate를 만들지 않는다.
    """
    has_box_improvement = jnp.any(
        safe_receiver & (box_receiver_bonus > 0.0)
        & (lane_completion >= config.pass_completion_threshold)
    )
    box_entry_ranking = (
        has_box_improvement & scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0)
        & (scene.position_team[pass_actor, 0] >= 0.0)
    )
    pass_score = pass_score - jnp.where(
        box_entry_ranking,
        0.25 * jnp.maximum(receiver_space_score - 2.0 * config.pressure_distance_m, 0.0),
        0.0,
    )
    pass_score = pass_score + jnp.where(has_box_improvement, 0.0, specialist_preference)
    pass_score = jnp.where(safe_receiver, pass_score, -jnp.inf)
    target_scalar = jnp.argmax(pass_score).astype(jnp.int32)
    has_target_scalar = pass_actor_valid & jnp.any(safe_receiver)
    return pass_score, target_scalar, has_target_scalar


def _select_possession_macro(
    shot_ready, shot_macro_value, has_target, availability, macro_pass_viable,
    pass_dwell_met, quick_relay, carry_release, pass_macro_value,
    pressure_escape_release, control_macro_value, forced_close_shot,
):
    """기존 [N] eligibility로 SHOOT/PASS/CONTROL 동점 순서와 강제 슛을 보존한다.

    availability는 [N,5] intent mask다. 반환은 bool[N] availability와 int32[N]
    intent다. 전부 비활성이면 availability=false이며 fallback slot0의 intent는
    SHOOT로 남지만 caller가 적용 여부를 판단한다. random key를 소비하지 않는다.
    """
    macro_value = jnp.stack(
        (
            jnp.where(shot_ready, shot_macro_value, -jnp.inf),
            jnp.where(
                has_target
                & availability[:, INTENT_PASS]
                & macro_pass_viable
                & (pass_dwell_met | quick_relay)
                & carry_release,
                pass_macro_value,
                -jnp.inf,
            ),
            jnp.where(
                availability[:, INTENT_CONTROL] & (~pressure_escape_release),
                control_macro_value,
                -jnp.inf,
            ),
        ),
        axis=-1,
    )
    macro_available = jnp.any(jnp.isfinite(macro_value), axis=-1)
    macro_choice = jnp.argmax(macro_value, axis=-1)
    macro_intent = jnp.take(
        jnp.asarray((INTENT_SHOOT, INTENT_PASS, INTENT_CONTROL), jnp.int32),
        macro_choice,
    )
    macro_intent = jnp.where(
        shot_ready & forced_close_shot,
        jnp.int32(INTENT_SHOOT),
        macro_intent,
    )
    return macro_available, macro_intent
