"""shared scene을 재사용하는 고정-shape rule-based player policy."""

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.action import (
    INTENT_CHALLENGE,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOOT,
    IntentAction,
)
from messi.core.action_mapping import linf_radial_encode
from messi.core.ball import BallPhysics
from messi.core.constants import (
    RK_CORNER,
    RK_FREEKICK,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    STANDARD_PLAYER_SLOTS,
)
from messi.core.contact import ContactPhysics
from messi.core.contact_response import ContactResponsePhysics
from messi.core.contest import (
    _challenge_success_favourability as _contest_challenge_success_favourability,
)
from messi.core.formation import (
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MIDFIELDER,
    ROLE_FULL_BACK,
    ROLE_WIDE_FORWARD,
    ROLE_WIDE_MIDFIELDER,
)
from messi.core.geometry import BallGeometry
from messi.core.goalkeeper import goalkeeper_hand_allowed_mask, held_ball_mask
from messi.core.numeric import (
    require_bounded_integer,
    require_float32_field_half_extent,
    require_float32_representable,
)
from messi.core.offside import (
    DEFAULT_OFFSIDE_GEOMETRY,
    OffsideGeometry,
    prospective_offside_mask,
)
from messi.core.passive_contact import DEFAULT_PASSIVE_CONTACT_PHYSICS
from messi.core.randomness import (
    RandomEvent,
    event_random_key,
    validate_prng_key,
)
from messi.core.timebase import DEFAULT_TIMEBASE, MAX_PHYSICS_DT_S, Timebase
from messi.environment.observation import (
    PolicyObservation,
    SharedScene,
    intent_availability,
    observe_all,
)
from messi.policies.aerial import aerial_route
from messi.policies.ground_lane import ground_arrival_times
from messi.policies.rule_actions import (
    _shot_compensation_eligible,
    _shot_request_direction,
    _shot_release_velocity,
    _nominal_shot_target_y,
    _shot_lane_open,
    _shot_aim_noise_scale,
    _smooth_release_space,
    _shot_opportunity_quality,
    _shot_quality_context,
    _regular_shot_quality_threshold,
    _shot_macro_ranking_value,
    _interior_limit_m,
    _unit,
    _ground_launch_angle,
)

# [restart kind, side (0 restart team / 1 defending team), pitch third,
#  (centroid x, positive-ball-side centroid y, x std, y std)]. FootballWorld의
# 비공개 K-League 위치장 fit을 팀 moment로 압축한 값이다. DFL 측정값이 아니다.
_RESTART_SHAPE_MOMENTS = jnp.asarray(
    (
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,
        (
            (
                (-26.136, 17.232, 9.932, 8.525),
                (2.184, 12.497, 9.048, 10.415),
                (31.300, 12.479, 8.774, 6.415),
            ),
            (
                (-35.615, 11.804, 5.288, 4.716),
                (-8.131, 13.985, 6.853, 8.132),
                (19.632, 18.576, 9.340, 6.619),
            ),
        ),
        (
            (
                (-9.900, 4.888, 11.715, 12.392),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (5.665, 5.080, 10.182, 9.794)),
        ),
        (
            ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (35.535, 1.582, 4.336, 4.349)),
            (
                (-41.315, 0.571, 1.257, 1.262),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
        ),
        (
            (
                (-3.699, 5.972, 9.285, 11.273),
                (10.844, 2.362, 8.553, 11.086),
                (34.235, -0.249, 4.607, 3.067),
            ),
            (
                (-40.510, 1.254, 0.354, 1.992),
                (-16.736, 2.334, 6.251, 8.731),
                (-1.467, 6.303, 7.904, 9.279),
            ),
        ),
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,
    ),
    dtype=jnp.float32,
)
_RESTART_SHAPE_VALID = jnp.any(jnp.abs(_RESTART_SHAPE_MOMENTS) > 0.0, axis=-1)
from messi.policies.rule_config import (
    _PRESSURE_COVER_SPEED_ENVELOPE_MPS,
    RuleTacticalPlan,
    RuleTacticalPlanSpec,
    RULE_TACTICAL_PLAN_SPECS,
    RULE_TACTICAL_PLAN_SCHEMA,
    RULE_POLICY_DFL_REFERENCE_SCHEMA,
    RULE_POLICY_DFL_REFERENCE_SHA256,
    _TACTICAL_PROFILE,
    _TACTIC_ATTACK_WIDTH,
    _TACTIC_ATTACK_DEPTH_SCALE,
    _TACTIC_DEFEND_WIDTH,
    _TACTIC_DEFEND_DEPTH_SCALE,
    _TACTIC_PASS_SCALE,
    _TACTIC_PROGRESS_BONUS,
    _TACTIC_COUNTERPRESS_TICKS,
    _TACTIC_COUNTERPRESS_COVER,
    _TACTIC_FULLBACK_ADVANCE,
    _TACTIC_CROSS_DEPTH_DELTA,
    _TACTIC_CROSS_WIDTH_DELTA,
    _TACTIC_FATIGUE_FLOOR,
    _TACTIC_FATIGUE_KNEE,
    _TACTIC_BOX_COMPRESSION,
    _TACTIC_MARKING_RADIUS_SCALE,
    _TACTIC_FLUID_ROTATION,
    _TACTIC_NARROW_BLOCK_SWITCH_VALUE,
    _TACTIC_SHORT_PASS_GAIN,
    _TACTIC_DIRECT_PASS_GAIN,
    _TACTIC_DEFENSIVE_LINE_SHIFT,
    _tactical_plan_code,
    make_rule_tactical_plan,
    DEFAULT_RULE_TACTICAL_PLAN,
    DEFAULT_RULE_TACTICAL_POOL,
    _normalize_rule_tactical_plan,
    RulePolicyConfig,
    DEFAULT_RULE_POLICY_CONFIG,
)




from messi.policies.rule_specialists import (
    _specialist_pass_bonus,
    _specialist_long_service,
    _specialist_macro_scale,
    _fresh_receiver_pressure,
    _effective_tactical_profile,
)










def _specialist_defensive_target(target, anchor, ball, role, plan, counterpress, eligible, *, position):
    """[N,2] team-frame 목표: Zona leash와 GEGEN 국소 압박 단계의 squeeze."""
    zona = (plan == int(RuleTacticalPlan.ZONA_MISTA)) & eligible
    zonal = anchor + jnp.clip(target - anchor, -8.0, 8.0)
    target = jnp.where(zona[:, None], zonal, target)
    mobile_role = (
        (role == ROLE_CENTRE_MIDFIELDER) | (role == ROLE_WIDE_MIDFIELDER)
        | (role == ROLE_CENTRE_FORWARD) | (role == ROLE_WIDE_FORWARD)
    )
    local = jnp.sum((position - ball) ** 2, axis=-1) <= 18.0**2
    squeeze = eligible & counterpress & mobile_role & local & (plan == int(RuleTacticalPlan.GEGENPRESSING))
    # 근처의 실제 선수만 goal-side outlet을 좁힌다. 중앙으로 모두 모이지 않게 좌우를 남긴다.
    side = jnp.where(position[:, 1] < ball[:, 1], -1.0, 1.0)
    press_target = ball + jnp.stack((-jnp.full_like(side, 2.0), 3.0 * side), axis=-1)
    delta = jnp.clip(0.45 * (press_target - target), jnp.asarray((-6.0, -4.0)), jnp.asarray((6.0, 4.0)))
    return jnp.where(squeeze[:, None], target + delta, target)


def _zona_assignment_from_score(score):
    """정방 [N,N], N>=1 비용에서 두 번의 상호 최솟값 담당을 선정한다.

    Non-finite 비용은 불가능한 edge다. 미배정 selected는 사용하지 않는다.
    1차 담당을 보존하고 중복 담당을 금지하지만 최대 cardinality matching은 아니다.
    """
    indices = jnp.arange(score.shape[0], dtype=jnp.int32)
    score = jnp.where(jnp.isfinite(score), score, jnp.inf)
    selected = jnp.argmin(score, axis=-1)
    primary = jnp.argmin(score, axis=0)
    assigned = jnp.any(jnp.isfinite(score), axis=-1) & (primary[selected] == indices)
    # Column의 1차 primary row를 gather하여 새 [N,N] 소유 행렬을 만들지 않는다.
    claimed = assigned[primary] & (selected[primary] == indices)
    remaining = jnp.where((~assigned[:, None]) & (~claimed[None, :]), score, jnp.inf)
    second_selected = jnp.argmin(remaining, axis=-1)
    second_primary = jnp.argmin(remaining, axis=0)
    second_assigned = jnp.any(jnp.isfinite(remaining), axis=-1) & (second_primary[second_selected] == indices)
    return jnp.where(assigned, selected, second_selected), assigned | second_assigned


def _zona_marking_target(scene, anchor, eligible, is_goalkeeper, config):
    """현재/예측 지역의 합집합 안에서 관측 속도로 담당 비용을 산정한다."""
    count = anchor.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    opponent_x = scene.position_team[:, 0:1] + scene.relative_position[..., 0]
    opponent_y = scene.position_team[:, 1:2] + scene.relative_position[..., 1]
    anchor_distance = (opponent_x - anchor[:, 0:1]) ** 2 + (opponent_y - anchor[:, 1:2]) ** 2
    predicted_x = opponent_x + config.marking_lead_s * (
        scene.relative_velocity[..., 0] + scene.velocity_team[:, 0:1]
    )
    predicted_y = opponent_y + config.marking_lead_s * (
        scene.relative_velocity[..., 1] + scene.velocity_team[:, 1:2]
    )
    predicted_anchor_distance = (
        (predicted_x - anchor[:, 0:1]) ** 2
        + (predicted_y - anchor[:, 1:2]) ** 2
    )
    radius = jnp.asarray(config.marking_activation_distance_m, jnp.float32)
    candidate = (
        eligible[:, None] & scene.opponent & scene.valid[None, :]
        & (~is_goalkeeper[None, :]) & (~scene.possessor[None, :])
        & ((anchor_distance <= radius**2) | (predicted_anchor_distance <= radius**2))
        & (scene.distance_squared <= (2.0 * radius) ** 2)
        & (opponent_x <= scene.ball_position_team[:, 0:1] + 0.5 * radius)
    )
    # Anchor 소속을 우선하되 실제로 먼 선수에게 맡기지 않는다. 행/열의 상호 최솟값만
    # 인정해 한 defender의 중복 담당과 한 runner로의 다중 추격을 동시에 막는다.
    score = jnp.where(candidate, predicted_anchor_distance + 0.15 * scene.distance_squared, jnp.inf)
    selected, assigned = _zona_assignment_from_score(score)
    selected_position = jnp.stack((opponent_x[indices, selected], opponent_y[indices, selected]), axis=-1)
    selected_velocity = scene.relative_velocity[indices, selected] + scene.velocity_team
    target = selected_position + config.marking_lead_s * selected_velocity
    target = target.at[:, 0].add(-config.marking_goal_side_m)
    return jnp.where(assigned[:, None], target, anchor), assigned




def _fluid_role_rotation_offset(
    role: jax.Array,
    lateral_anchor: jax.Array,
    ball_y: jax.Array,
    player_tactical_plan: jax.Array,
) -> jax.Array:
    """추가 pairwise 없이 공격 시 role cover/interchange offset을 반환한다."""

    # Fullback은 전진·안쪽, wide forward는 안쪽 half-space, CM은 비워진 바깥과
    # 후방 cover로 움직인다. 공 반대편은 강도를 줄여 rest-defence를 남긴다.
    same_ball_side = jnp.tanh(lateral_anchor * ball_y / jnp.float32(180.0))
    participation = jnp.float32(0.72) + jnp.float32(0.28) * same_ball_side
    x = jnp.select(
        (
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (6.0, -2.5, 2.0, -1.0, 2.5),
        default=0.0,
    ).astype(jnp.float32)
    y_scale = jnp.select(
        (
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (-0.28, 0.18, -0.18, 0.10, -0.34),
        default=0.0,
    ).astype(jnp.float32)
    strength = _TACTIC_FLUID_ROTATION[player_tactical_plan] * participation
    return strength[:, None] * jnp.stack((x, y_scale * lateral_anchor), axis=-1)


def _total_flank_exchange(
    target, scene, role, anchor, is_goalkeeper, attacking_shape,
    player_tactical_plan, same_team, config, pressure,
):
    """TOTAL의 같은 flank FB와 WF/WM이 관측 점유에 따라 공간 역할을 교환한다."""
    count = target.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    eligible = (
        scene.valid & scene.ball_live & (~is_goalkeeper) & attacking_shape
        & (player_tactical_plan == int(RuleTacticalPlan.TOTAL_FOOTBALL))
    )
    fullback = eligible & (role == ROLE_FULL_BACK)
    wide = eligible & ((role == ROLE_WIDE_FORWARD) | (role == ROLE_WIDE_MIDFIELDER))
    same_flank = anchor[:, None, 1] * anchor[None, :, 1] > 0.0
    candidate = (
        same_team & same_flank
        & ((fullback[:, None] & wide[None, :]) | (wide[:, None] & fullback[None, :]))
    )
    # 기존 거리 행렬의 mutual-nearest pair만 사용해 한 선수의 중복 역할을 막는다.
    partner = jnp.argmin(jnp.where(candidate, scene.distance_squared, jnp.inf), axis=-1)
    has_partner = jnp.any(candidate, axis=-1)
    reciprocal = has_partner & has_partner[partner] & (partner[partner] == index)
    wide_index = jnp.where(wide, index, partner)
    fullback_index = jnp.where(fullback, index, partner)
    side = jnp.sign(anchor[:, 1])
    ball_side = jnp.clip(side * scene.ball_position_team[:, 1] / 8.0, 0.0, 1.0)
    inward = jnp.clip(
        (jnp.abs(anchor[wide_index, 1]) - jnp.abs(scene.position_team[wide_index, 1]) - 3.0) / 6.0,
        0.0, 1.0,
    )
    already_ahead = jnp.clip(
        (scene.position_team[fullback_index, 0] - scene.position_team[wide_index, 0] + 1.0) / 6.0,
        0.0, 1.0,
    )
    wide_on_ball = scene.possessor[wide_index]
    unpressured_wide_on_ball = wide_on_ball & (~pressure[wide_index])
    phase = jnp.maximum(jnp.maximum(inward, already_ahead), unpressured_wide_on_ball.astype(jnp.float32))
    phase = phase * phase * (3.0 - 2.0 * phase)
    strength = jnp.where(reciprocal, ball_side * phase, 0.0)
    width = jnp.maximum(jnp.abs(target[:, 1]), jnp.abs(target[partner, 1]))
    desired = jnp.stack((
        jnp.where(fullback, jnp.maximum(target[:, 0], target[partner, 0]),
                  jnp.minimum(target[:, 0], target[partner, 0])),
        side * jnp.where(fullback, width, 0.65 * width),
    ), axis=-1)
    delta = jnp.clip(desired - target, jnp.asarray((-12.0, -8.0)), jnp.asarray((12.0, 8.0)))
    exchanged = target + strength[:, None] * delta
    # 외곽 carrier가 시작한 overlap은 고정 anchor 뒤에 갇히지 않게 현재 FB에서
    # 최대 12 m 앞까지만 목표 하한을 전진시킨다. 계획 수신자 override는 유지한다.
    overlap_x = jnp.minimum(
        scene.position_team[wide_index, 0] + config.planned_receiver_forward_lead_m,
        scene.position_team[:, 0] + 12.0,
    )
    exchanged = exchanged.at[:, 0].set(jnp.where(
        fullback & wide_on_ball,
        jnp.maximum(exchanged[:, 0], target[:, 0] + strength * (overlap_x - target[:, 0])),
        exchanged[:, 0],
    ))
    exchanged = jnp.clip(
        exchanged,
        jnp.asarray((-config.field_half_length_m + 1.0, -config.field_half_width_m + 1.0)),
        jnp.asarray((config.field_half_length_m - 1.0, config.field_half_width_m - 1.0)),
    )
    if config.offside_shoulder_enabled:
        shoulder = jnp.maximum(_onside_limit_x(scene) - config.offside_shoulder_margin_m, 0.0)
        exchanged = exchanged.at[:, 0].set(jnp.where(
            jnp.any(scene.possessor), jnp.minimum(exchanged[:, 0], shoulder), exchanged[:, 0],
        ))
    # 교환이 없는 모든 plan은 clamp나 float 재연산 결과도 적용하지 않는다.
    return jnp.where((strength > 0.0)[:, None], exchanged, target)


def _narrow_block_switch_bonus(
    pass_relative: jax.Array,
    lane_completion: jax.Array,
    opponent_mask: jax.Array,
    position_y: jax.Array,
    tactical_plan: jax.Array,
    field_half_width_m: float,
) -> jax.Array:
    """좁은 block 상대의 안전한 lateral switch 후보 가치를 O(N)으로 높인다."""

    half_width = jnp.asarray(field_half_width_m, jnp.float32)
    opponent_min = jnp.min(jnp.where(opponent_mask, position_y, jnp.inf))
    opponent_max = jnp.max(jnp.where(opponent_mask, position_y, -jnp.inf))
    has_opponent = jnp.any(opponent_mask)
    opponent_width = jnp.where(
        has_opponent, opponent_max - opponent_min, 2.0 * half_width
    )
    narrowness = jnp.clip(
        1.0 - opponent_width / jnp.maximum(2.0 * half_width, 1.0), 0.0, 1.0
    )
    lateral_fraction = jnp.clip(
        jnp.abs(pass_relative[:, 1]) / jnp.maximum(half_width, 1.0), 0.0, 1.0
    )
    return (
        _TACTIC_NARROW_BLOCK_SWITCH_VALUE[tactical_plan]
        * narrowness
        * lateral_fraction
        * lane_completion
    )


def _lead_protection_intensity(
    score_for: jax.Array, score_against: jax.Array, config: "RulePolicyConfig"
) -> jax.Array:
    """큰 리드에서만 0→1로 증가하는 연속적인 경기상태 강도."""

    span = jnp.float32(
        config.lead_protection_full_gap - config.lead_protection_start_gap + 1
    )
    value = (
        jnp.clip(
            (score_for - score_against).astype(jnp.float32)
            - jnp.float32(config.lead_protection_start_gap - 1),
            0.0,
            span,
        )
        / span
    )
    return value * value * (jnp.float32(3.0) - jnp.float32(2.0) * value)


def _stamina_offball_power(
    role: jax.Array,
    stamina_long: jax.Array,
    is_cover: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """비핵심 off-ball 이동의 장기 stamina 보존 계수를 계산한다."""

    reserve = jnp.clip(
        jnp.asarray(stamina_long, jnp.float32)
        / jnp.float32(config.stamina_conservation_knee),
        0.0,
        1.0,
    )
    reserve = reserve * reserve * (jnp.float32(3.0) - jnp.float32(2.0) * reserve)
    floor = jnp.asarray(config.role_fatigue_power_floor, jnp.float32)[
        jnp.clip(role.astype(jnp.int32), 0, 6)
    ]
    power = floor + (jnp.float32(1.0) - floor) * reserve
    cover_floor = jnp.float32(config.cover_fatigue_power_floor)
    return jnp.where(is_cover, jnp.maximum(power, cover_floor), power)


def _topology_wingback_advance(
    role: jax.Array,
    same_team: jax.Array,
    lateral_anchor: jax.Array,
    ball_y: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """Wide attacker가 없는 5인 수비선의 ball-side full-back 전진량."""

    wide_role = (role == ROLE_WIDE_MIDFIELDER) | (role == ROLE_WIDE_FORWARD)
    team_wide_role_count = jnp.sum(
        (same_team & wide_role[None, :]).astype(jnp.int32), axis=-1
    )
    defensive_role = (role == ROLE_CENTRE_BACK) | (role == ROLE_FULL_BACK)
    defensive_role_count = jnp.sum(
        (same_team & defensive_role[None, :]).astype(jnp.int32), axis=-1
    )
    ball_side_fullback = (
        (role == ROLE_FULL_BACK)
        & (team_wide_role_count == 0)
        & (defensive_role_count >= 5)
        & (lateral_anchor * ball_y >= 0.0)
    )
    return ball_side_fullback.astype(jnp.float32) * jnp.asarray(
        config.wingback_advance_m, jnp.float32
    )


def _box_runner_tracking_mask(
    opponent_position: jax.Array,
    defensive_box_phase: jax.Array,
    box_guard: jax.Array,
    is_chaser: jax.Array,
    is_cover: jax.Array,
    is_primary_marker: jax.Array,
    config: "RulePolicyConfig",
) -> jax.Array:
    """Own-box defensive phase의 primary marker를 12 m gate와 독립 유지한다."""

    in_threat_zone = (
        opponent_position[:, 0]
        <= -jnp.float32(config.field_half_length_m)
        + jnp.float32(config.box_threat_distance_m)
    ) & (
        jnp.abs(opponent_position[:, 1])
        <= jnp.float32(config.penalty_area_half_width_m + 4.0)
    )
    return (
        defensive_box_phase
        & box_guard
        & (~is_chaser)
        & (~is_cover)
        & is_primary_marker
        & in_threat_zone
    )














from messi.policies.rule_decision import (
    _inside_box_receiver_score_bonus,
    _release_urgency_control_scale,
    _box_possession_macro_values,
    _select_receiver,
    _select_possession_macro,
)




def _compress_longitudinal_team_shape(
    target: jax.Array,
    same_team: jax.Array,
    outfield: jax.Array,
    compression: jax.Array,
) -> jax.Array:
    """Outfield team centroid와 y를 보존하며 target의 x spread만 압축한다."""

    target = jnp.asarray(target, jnp.float32)
    team_outfield = (
        jnp.asarray(same_team, jnp.bool_) & jnp.asarray(outfield, jnp.bool_)[None, :]
    )
    count = jnp.maximum(jnp.sum(team_outfield, axis=-1), 1)
    centroid_x = (
        jnp.sum(jnp.where(team_outfield, target[None, :, 0], 0.0), axis=-1) / count
    )
    compressed_x = centroid_x + jnp.asarray(compression, jnp.float32) * (
        target[:, 0] - centroid_x
    )
    return target.at[:, 0].set(
        jnp.where(jnp.asarray(outfield, jnp.bool_), compressed_x, target[:, 0])
    )




def _pass_release_contest_clear(
    scene: SharedScene, actor: jax.Array, max_speed: jax.Array,
    goalkeeper_claim_available: jax.Array, config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
    *, is_goalkeeper: jax.Array,
) -> jax.Array:
    """고정 planning horizon의 관측 기반 위험 근사. 미래 intent를 예측하지 않는다."""
    ball_distance = jnp.linalg.norm(scene.ball_relative_position, axis=-1)
    ball_speed = jnp.linalg.norm(scene.ball_velocity_team[actor])
    ordinary_ready = scene.contact_ready
    challenge_ready = ordinary_ready & scene.challenge_ready
    radius = jnp.where(challenge_ready, contact_physics.challenge_radius_m,
                       contact_physics.carry_radius_m)
    # Predictive claim=false는 미래 release의 hand 불가를 뜻하지 않는다. 법적 hand
    # mask 자체는 관측에 없으므로 GK는 hand 폭을 보수적 상한으로 유지한다.
    radius = jnp.where(is_goalkeeper, jnp.maximum(radius, contact_physics.goalkeeper_radius_m), radius)
    reach = radius + config.ball_radius_m + (
        jnp.maximum(max_speed, 0.)+ball_speed
    )*jnp.float32(config.pass_release_lookahead_s)
    ready = ordinary_ready | goalkeeper_claim_available
    finite = (jnp.all(jnp.isfinite(ball_distance)) & jnp.all(jnp.isfinite(max_speed))
              & jnp.isfinite(ball_speed))
    threat = scene.opponent[actor] & ready & (ball_distance <= reach)
    return finite & (~jnp.any(threat))


def _pass_release_window(
    nearest_opponent_distance: jax.Array,
    config: "RulePolicyConfig",
    contact_physics: ContactPhysics,
    *,
    contest_clear=None,
) -> tuple[jax.Array, jax.Array]:
    """Pass의 압박 urgency와 실제 kick 공간을 분리한 bounded window.

    압박자가 멀 때는 기존 pass ranking을 거의 유지하고, pressure envelope에 들어오면
    CONTROL보다 조금 일찍 release하도록 value를 올린다. 반대로 challenge reach에 거의
    겹치면 completion이 높더라도 정상적인 kick 동작 공간이 아니므로 readiness를 닫는다.
    """

    close_distance = jnp.asarray(
        contact_physics.challenge_radius_m + config.ball_radius_m, jnp.float32
    )
    useful_distance = jnp.maximum(
        jnp.asarray(config.pressure_distance_m, jnp.float32),
        close_distance + jnp.float32(1.0e-3),
    )
    release_space = _smooth_release_space(
        nearest_opponent_distance, close_distance, useful_distance
    )
    minimum_release_space = jnp.float32(0.20)
    if contest_clear is not None:
        release_space = jnp.where(
            contest_clear, jnp.maximum(release_space, minimum_release_space), release_space
        )
    urgency = jnp.float32(1.0) - _smooth_release_space(
        nearest_opponent_distance,
        useful_distance,
        jnp.float32(2.0) * useful_distance,
    )
    # 먼 거리의 기존 ranking은 정확히 보존하고, 유효 압박 거리에 들어온 경우에만
    # 최대 50% urgency bonus를 준다. Close-space 감쇠는 이 bonus 뒤에도 지배적이다.
    value_scale = release_space * (jnp.float32(1.0) + jnp.float32(0.50) * urgency)
    return value_scale, release_space >= minimum_release_space






from messi.policies.rule_features import (
    _RulePolicyFeatures,
    _policy_features,
    _target_relative_velocity,
    _receiver_arrival_eta,
    _inside_attacking_penalty_area,
    _inside_own_penalty_area,
)









# Tracking-data fit으로 가장하지 않는 bounded tactical design prior다. Ball-follow와
# width 계수는 RulePolicyConfig의 static field로 노출한다.
_GOALKEEPER_MIN_DEPTH_M = 3.0
_GOALKEEPER_MAX_DEPTH_M = 8.0
_GOALKEEPER_POST_MARGIN_M = 0.35




class RulePolicyDiagnostics(NamedTuple):
    intent: jax.Array
    target_player: jax.Array
    pressure: jax.Array
    ball_distance: jax.Array
    availability: jax.Array
    prospective_offside: jax.Array
    goalkeeper_claim: jax.Array
    goalkeeper_holding: jax.Array
    pass_completion: jax.Array
    cross: jax.Array
    clearance: jax.Array
    counterpress: jax.Array
    carrier_valid: jax.Array
    carrier: jax.Array
    carrier_local_teammates: jax.Array
    carrier_local_opponents: jax.Array
    target_direction_family: jax.Array
    target_score_gap: jax.Array
    target_next_outlet_quality: jax.Array








def _height_compensated_launch(
    signed_launch: jax.Array, ball_height: jax.Array, contact_physics: ContactPhysics
) -> jax.Array:
    """공 높이에 따라 내려가는 actuator floor를 보정해 ground 각도를 유지한다.

    Foot actuator는 공이 높을수록 launch floor를 아래로 내리므로 같은 signed control이
    bouncing ball에서는 downward volley가 되어 turf에서 튀어 오른다. 정책은 launch를
    ground 기준 각도로 정의하고, 관측된 ball height에서 같은 각도를 실현하는 control로
    변환한다. 값은 [-1, 1]로 clip하며 pairwise 계산은 없다.
    """

    down = jnp.float32(contact_physics.ground_launch_down_max_radians)
    launch_max = jnp.float32(contact_physics.launch_max_radians)
    reference = jnp.float32(contact_physics.ground_launch_down_reference_height_m)
    radius = jnp.float32(BallGeometry().radius_m)
    target_angle = _ground_launch_angle(signed_launch, contact_physics)
    height_fraction = jnp.clip(
        (ball_height - radius) / jnp.maximum(reference - radius, 1.0e-3), 0.0, 1.0
    )
    floor = -(down + (launch_max - down) * height_fraction)
    unit = (target_angle - floor) / jnp.maximum(launch_max - floor, 1.0e-3)
    return jnp.clip(2.0 * unit - 1.0, -1.0, 1.0)


def _lofted_release_power(
    carry_m, signed_launch, config: RulePolicyConfig, contact_physics: ContactPhysics
):
    """목표 carry 거리를 ballistic range로 실현하는 kick power.

    Vacuum range v² sin(2θ)/g에 free-ball drag를 근사한 `aerial_carry_factor`를 곱한
    carry를 뒤집는다. 전체 power로 차는 clearance·cross·distribution이 pitch 밖으로
    나가는 대신 목표 지점 근처에 떨어지게 한다.
    """

    angle = _ground_launch_angle(signed_launch, contact_physics)
    range_term = jnp.maximum(
        jnp.sin(2.0 * angle) * jnp.float32(config.aerial_carry_factor), 0.05
    )
    speed = jnp.sqrt(
        jnp.maximum(jnp.asarray(carry_m, jnp.float32), 0.0)
        * jnp.float32(config.gravity_mps2)
        / range_term
    )
    return jnp.clip(speed / jnp.float32(config.kick_speed_max_mps), 0.05, 1.0)


def _aerial_receiver_quality(
    scene, distance, height, reach_height, max_speed, ball_control, is_goalkeeper, launch,
    minimum_power, ball_height, config, contact_physics,
):
    """[N] 수신 후보의 nominal 공중 경합 점수 [0,1]. SI/team-frame 입력이다.

    기존 [N,N] 거리·상대 mask로 비행시간 안에 접근할 수 있는 가장 높은 상대와
    비교한다. Reach는 점프·자세를 포함한 절대 높이이므로 키를 중복 가산하지 않는다.
    빈 공간은 작은 선수도 받을 수 있다. 정확한 flight/interception solver는 아니다.
    """
    power = jnp.maximum(
        _lofted_release_power(distance, launch, config, contact_physics), minimum_power
    )
    angle = _ground_launch_angle(launch, contact_physics)
    vertical_speed = power * jnp.float32(config.kick_speed_max_mps) * jnp.sin(angle)
    gravity = jnp.maximum(jnp.float32(config.gravity_mps2), 1.e-3)
    radius = jnp.float32(config.ball_radius_m)
    release_height = jnp.maximum(ball_height - radius, 0.0)
    flight_time = (vertical_speed + jnp.sqrt(vertical_speed**2 + 2.0 * gravity * release_height)) / gravity
    apex = release_height + vertical_speed**2 / (2.0 * gravity)
    # 골키퍼의 넓은 손 도달 반경은 landing 지점이 상대 penalty area일 때만 쓴다.
    keeper_area = (
        (scene.position_team[:, 0] >= config.field_half_length_m - config.penalty_area_length_m)
        & (jnp.abs(scene.position_team[:, 1]) <= config.penalty_area_half_width_m)
    )
    contact_radius = jnp.where(
        keeper_area[:, None] & is_goalkeeper[None, :],
        contact_physics.goalkeeper_radius_m, contact_physics.carry_radius_m,
    )
    opponent_eta = jnp.maximum(
        jnp.sqrt(scene.distance_squared + 1.e-6) - (contact_radius + radius), 0.0
    ) / jnp.maximum(max_speed[None, :], 1.e-3)
    contenders = scene.opponent & (opponent_eta <= flight_time[:, None])
    opponent_reach = jnp.minimum(reach_height[None, :], apex[:, None])
    opponent_strength = opponent_reach + 0.15 * ball_control[None, :]
    strongest = jnp.max(jnp.where(contenders, opponent_strength, -100.0), axis=-1)
    reachable_height = jnp.minimum(reach_height, apex)
    advantage = reachable_height + 0.15 * ball_control - strongest
    offered_height = jnp.minimum(apex, 2.2)
    reach_margin = jnp.maximum(reach_height - height, 1.e-3)
    jump_effort = jnp.clip((offered_height - height) / reach_margin, 0.0, 1.0)
    contested = jnp.any(contenders, axis=-1)
    quality = jax.nn.sigmoid(advantage / 0.35) - 0.08 * jump_effort
    quality = jnp.where(reach_height >= offered_height, quality, 0.0)
    return jnp.where(contested, jnp.clip(quality, 0.0, 1.0), 1.0)








def _ground_pass_controls(
    target_relative: jax.Array,
    target_velocity: jax.Array,
    config: RulePolicyConfig,
    *,
    ball_physics: BallPhysics = BallPhysics(),
    passer_position: jax.Array | None = None,
    release_speed_max_mps: jax.Array | None = None,
):
    """Team-frame receiver 상대속도로 bounded lead와 rolling-distance power를 계산한다.

    release_speed_max_mps는 actuator의 m/s 상한이며 생략하면 kick 상한을 쓴다.
    Throw의 release 높이와 비행·바운드는 여기서 모델링하지 않는다.
    """

    initial_distance = jnp.sqrt(
        jnp.sum(target_relative * target_relative, axis=-1) + 1.0e-6
    )
    arrival = jnp.asarray(config.pass_arrival_speed_mps, dtype=jnp.float32)
    knots = jnp.asarray(ball_physics.rolling_speed_knots_mps, dtype=jnp.float32)
    deceleration = jnp.asarray(
        ball_physics.rolling_deceleration_knots_mps2, dtype=jnp.float32
    )
    initial_estimate = jnp.sqrt(
        arrival * arrival + 2.0 * jnp.float32(2.0) * initial_distance
    )
    midpoint_speed = 0.5 * (arrival + initial_estimate)
    effective_deceleration = jnp.interp(midpoint_speed, knots, deceleration)
    release_speed = jnp.sqrt(
        arrival * arrival + 2.0 * effective_deceleration * initial_distance
    )
    mean_speed = jnp.maximum(0.5 * (release_speed + arrival), 1.0e-3)
    lead_time = jnp.minimum(
        initial_distance / mean_speed,
        jnp.asarray(config.pass_lead_time_cap_s, dtype=jnp.float32),
    )
    raw_lead = target_velocity * lead_time[:, None]
    lead_norm = jnp.sqrt(jnp.sum(raw_lead * raw_lead, axis=-1) + 1.0e-6)
    lead_scale = jnp.minimum(
        1.0,
        jnp.asarray(config.pass_lead_distance_cap_m, dtype=jnp.float32)
        / jnp.maximum(lead_norm, 1.0e-6),
    )
    lead = raw_lead * lead_scale[:, None]
    if passer_position is not None:
        # 경계 근처의 receiver를 pitch 밖으로 선행하지 않는다. 이미 1 m margin
        # 안쪽보다 경계에 가까운 receiver는 현재 위치를 보존해 짧은 pass를 역전하지 않는다.
        receiver_position = passer_position + target_relative
        limits = jnp.asarray(
            (
                _interior_limit_m(config.field_half_length_m, 1.0),
                _interior_limit_m(config.field_half_width_m, 1.0),
            ),
            dtype=jnp.float32,
        )
        lead = jnp.clip(
            lead,
            jnp.minimum(-limits - receiver_position, 0.0),
            jnp.maximum(limits - receiver_position, 0.0),
        )
    led_target = target_relative + lead
    distance = jnp.sqrt(jnp.sum(led_target * led_target, axis=-1) + 1.0e-6)
    # Lead가 거리를 바꾸면 속도 구간별 rolling 감속도 함께 달라진다. 초기 receiver
    # 거리에서 고른 감속을 그대로 재사용하면 움직이는 receiver pass의 power가 작게
    # 어긋나므로 led distance에서 한 번만 정제한다.
    led_initial_estimate = jnp.sqrt(
        arrival * arrival + 2.0 * effective_deceleration * distance
    )
    led_midpoint_speed = 0.5 * (arrival + led_initial_estimate)
    effective_deceleration = jnp.interp(led_midpoint_speed, knots, deceleration)
    release_speed = jnp.sqrt(
        arrival * arrival + 2.0 * effective_deceleration * distance
    )
    speed_max = jnp.asarray(
        config.kick_speed_max_mps
        if release_speed_max_mps is None
        else release_speed_max_mps,
        dtype=jnp.float32,
    )
    power = jnp.clip(
        release_speed / jnp.maximum(speed_max, jnp.float32(1.0e-3)),
        0.05,
        1.0,
    )
    return _unit(led_target), power


def _cross_spin_ground_eta_prior(
    release_speed, surface_cross, ball_height, vertical_velocity, vertical_spin,
    goalkeeper_holding, restart_kind, config,
):
    """Cross-spin만 미지원인 지면 행의 선택적 평균속도 prior; 비대상은 NaN.

    물리 ETA나 보수 상한이 아니다. 유한한 양의 release만 사용하며 기존 terminal
    speed보다 느린 prior는 만들지 않는다. NaN은 lane helper의 기존 fallback 표식이다.
    """
    speed = jnp.maximum(
        .5 * release_speed + .5 * config.pass_arrival_speed_mps,
        jnp.asarray(config.pass_arrival_speed_mps, jnp.float32),
    )
    eligible = (
        (ball_height <= config.ball_radius_m + .02)
        & (jnp.abs(vertical_velocity) <= .1)
        & (jnp.abs(vertical_spin) <= 1.)
        & (jnp.abs(surface_cross) > .5)
        & (~goalkeeper_holding)
        & (restart_kind == 0)
        & jnp.isfinite(ball_height) & jnp.isfinite(vertical_velocity)
        & jnp.isfinite(vertical_spin) & jnp.isfinite(surface_cross)
        & jnp.isfinite(release_speed) & (release_speed > 0.) & jnp.isfinite(speed)
    )
    return jnp.where(eligible, speed, jnp.float32(jnp.nan))


def _pass_lane_completion(
    receiver_relative: jax.Array,
    opponent_relative: jax.Array,
    opponent_mask: jax.Array,
    config: RulePolicyConfig,
    *,
    release_speed=None,
    surface_velocity=None,
    ground_valid=None,
    opponent_velocity=None,
    opponent_max_speed=None,
    release_origin_offset=None,
    receiver_distance=None,
    fallback_arrival_speed=None,
    ball_physics: BallPhysics = BallPhysics(),
) -> jax.Array:
    """한 passer 기준 모든 receiver lane의 저비용 interception completion."""

    physical_ground = release_speed is not None
    fallback_valid = None
    if fallback_arrival_speed is not None:
        fallback_valid = jnp.isfinite(fallback_arrival_speed) & (fallback_arrival_speed > 0.)
        if physical_ground:
            fallback_valid &= ~ground_valid
    lane_opponent_relative = opponent_relative[None, :, :]
    if release_origin_offset is not None:
        # ETA 미지원 행도 실제 release 원점과 같은 ray 기하를 사용한다.
        receiver_relative = receiver_relative/jnp.maximum(
            jnp.linalg.norm(receiver_relative, axis=-1, keepdims=True), 1.e-6,
        )*receiver_distance[:, None]
        lane_opponent_relative = opponent_relative[None, :, :]-release_origin_offset
    length_squared = jnp.sum(receiver_relative * receiver_relative, axis=-1)
    projection = jnp.sum(
        receiver_relative[:, None, :] * lane_opponent_relative, axis=-1
    ) / jnp.maximum(length_squared[:, None], 1.0e-6)
    projection = jnp.clip(projection, 0.0, 1.0)
    closest = projection[..., None] * receiver_relative[:, None, :]
    lateral = jnp.sqrt(
        jnp.sum((lane_opponent_relative - closest) ** 2, axis=-1) + 1.0e-6
    )
    distance = jnp.sqrt(length_squared + 1.0e-6)
    lane_arrival_speed = jnp.maximum(
        jnp.asarray(config.pass_arrival_speed_mps, dtype=jnp.float32), 1.0e-3
    )
    time_distance = distance
    if fallback_valid is not None:
        # 비대상 행의 scalar division 순서와 값을 보존하며 [N] 거리만 재척도화한다.
        selected_speed = jnp.where(
            fallback_valid, fallback_arrival_speed, lane_arrival_speed,
        )
        time_distance = distance * jnp.where(
            fallback_valid, lane_arrival_speed / selected_speed, 1.,
        )
    ball_time = projection * time_distance[:, None] / lane_arrival_speed
    if physical_ground:
        query = jnp.concatenate((projection*distance[:, None], distance[:, None]), axis=1)
        arrival = ground_arrival_times(
            query, release_speed, surface_velocity, physics=ball_physics,
            integration_dt_s=MAX_PHYSICS_DT_S,
        )
        ball_time = jnp.where(ground_valid[:, None], arrival[:, :-1], ball_time)
    # Receiver도 lane의 자기 쪽 끝을 지킨다. Receiver 뒤(goal side)에 선 marker는
    # receiver보다 먼저 공에 닿을 수 없으므로 lane 차단이 아니라 reception contest이며,
    # 그 경쟁은 별도 arrival advantage가 다룬다. Interceptor는 공과 receiver 중 먼저
    # 도착하는 쪽보다 빨라야 한다.
    receiver_time = (
        (1.0 - projection)
        * distance[:, None]
        / jnp.maximum(
            jnp.asarray(config.receiver_cover_speed_mps, dtype=jnp.float32), 1.0e-3
        )
    )
    response_time = jnp.maximum(
        jnp.minimum(ball_time, receiver_time)
        - jnp.asarray(config.opponent_reaction_s, dtype=jnp.float32),
        0.0,
    )
    reachable = (
        jnp.asarray(config.opponent_intercept_speed_mps, dtype=jnp.float32)
        * response_time
    )
    if physical_ground:
        gap = lane_opponent_relative-closest
        closing_speed = jnp.maximum(
            -jnp.sum(opponent_velocity[None, :, :]*gap/lateral[..., None], axis=-1), 0.,
        )
        # 관측 속도에서 가속하는 bounded prior다. 미래 intent를 아는 모델은 아니다.
        observed_reach = jnp.minimum(
            closing_speed*response_time + jnp.float32(2.)*response_time**2,
            opponent_max_speed[None, :]*response_time,
        )
        reachable = jnp.where(ground_valid[:, None], observed_reach, reachable)
    margin = (lateral - reachable) / jnp.maximum(
        jnp.asarray(config.pass_lane_softness_m, dtype=jnp.float32), 0.05
    )
    safe = jax.nn.sigmoid(margin)
    if physical_ground:
        passive = DEFAULT_PASSIVE_CONTACT_PHYSICS
        body_radius = max(passive.torso_radius_m, passive.head_radius_m,
                          .5*passive.shoulder_width_m+passive.leg_radius_m)+config.ball_radius_m
        direct_body = (projection > 0.) & (projection < 1.) & (lateral <= body_radius)
        safe = jnp.where(ground_valid[:, None] & direct_body, 0., safe)
    safe = jnp.where(opponent_mask[None, :], safe, 1.0)
    completion = jnp.min(safe, axis=-1)
    if physical_ground:
        completion = jnp.where(ground_valid & ~jnp.isfinite(arrival[:, -1]), 0., completion)
    return completion


def _contextual_bernoulli(reference_probability, context, logit_limit):
    """Reference 확률을 관측 context로 bounded하게 조정한 Bernoulli 확률.

    FootballWorld의 `_contextual_bernoulli_probability`와 같은 형태다. `context`가 `0.5`면
    reference 확률을 그대로 쓰고 `0`/`1`에서 `±logit_limit`만큼 logit을 옮긴다. 따라서
    확률은 항상 `(0, 1)`에 머문다.
    """

    reference = jnp.asarray(reference_probability, jnp.float32)
    logit = jnp.log(reference) - jnp.log1p(-reference)
    shift = jnp.asarray(logit_limit, jnp.float32) * (
        2.0 * jnp.clip(context, 0.0, 1.0) - 1.0
    )
    return jax.nn.sigmoid(logit + shift)


def _challenge_context(
    scene: SharedScene,
    features: _RulePolicyFeatures,
    config: RulePolicyConfig,
) -> jax.Array:
    """Challenge 제출 확률을 조정하는 관측 전용 context를 만든다.

    세 항을 bounded하게 합친다. 공이 가까울수록(proximity) 높고, 상대가 실제로 공을
    소유해 되찾을 기회가 있을 때(opportunity) 높고, 위치 자체가 아니라 빠르게 접근하는
    위험한 접촉일 때(foul risk) 낮다. 기존 `[N]` feature와 nearest-opponent gather만
    재사용하며 새 pairwise tensor를 만들지 않는다.
    """

    ball_distance = jnp.sqrt(features.ball_distance_squared + 1.0e-6)
    proximity = 1.0 - jnp.clip(
        ball_distance / jnp.maximum(jnp.float32(config.challenge_distance_m), 1.0e-3),
        0.0,
        1.0,
    )
    opponent_controls = jnp.any(scene.opponent & scene.possessor[None, :], axis=-1)
    opportunity = jnp.where(opponent_controls, jnp.float32(1.0), jnp.float32(0.35))
    # 실제 contest kernel은 actor와 verified carrier의 상대속도를 사용한다. Ball velocity는
    # dribble/contact 사이에 carrier와 다를 수 있으므로 foul-risk proxy에도 쓰지 않는다.
    # Observation의 possessor identity로 같은 carrier-relative closing 성분을 계산한다.
    carrier = jnp.argmax(scene.possessor.astype(jnp.int32)).astype(jnp.int32)
    carrier_relative_position = scene.relative_position[:, carrier]
    carrier_relative_velocity = scene.relative_velocity[:, carrier]
    approach = jnp.sum(
        -carrier_relative_velocity * _unit(carrier_relative_position), axis=-1
    )
    approach_risk = jnp.clip(approach / jnp.float32(6.0), 0.0, 1.0)
    foul_risk = approach_risk
    return jnp.clip(
        jnp.float32(config.challenge_proximity_gain) * proximity
        + jnp.float32(config.challenge_opportunity_gain) * opportunity
        - jnp.float32(config.challenge_foul_avoidance_gain) * foul_risk,
        0.0,
        1.0,
    )


def _challenge_success_favourability(
    scene: SharedScene,
    features: _RulePolicyFeatures,
    carrier: jax.Array,
    ball_control: jax.Array,
    max_speed: jax.Array,
    contact: ContactPhysics,
    config: RulePolicyConfig,
) -> jax.Array:
    """Contest와 같은 관측 geometry로 challenge 자세의 유리함을 계산한다."""

    count = scene.valid.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    safe_carrier = jnp.clip(carrier, 0, count - 1)
    actor_to_carrier = scene.relative_position[indices, safe_carrier]
    separation = jnp.sqrt(
        jnp.sum(actor_to_carrier * actor_to_carrier, axis=-1) + jnp.float32(1.0e-9)
    )
    approach = actor_to_carrier / separation[:, None]
    carrier_relative_velocity = scene.relative_velocity[indices, safe_carrier]
    closing = jnp.clip(
        jnp.maximum(jnp.sum(-carrier_relative_velocity * approach, axis=-1), 0.0)
        / jnp.maximum(max_speed + max_speed[safe_carrier], jnp.float32(1.0e-6)),
        0.0,
        1.0,
    )
    carrier_body = scene.body_forward_team[safe_carrier]
    carrier_body_observer = jnp.stack(
        (
            carrier_body[0]
            * scene.attack_direction[safe_carrier]
            * scene.attack_direction,
            jnp.full(count, carrier_body[1], dtype=jnp.float32),
        ),
        axis=-1,
    )
    rear_alignment = jnp.clip(
        jnp.sum(approach * _unit(carrier_body_observer), axis=-1), -1.0, 1.0
    )
    ball_distance = jnp.sqrt(features.ball_distance_squared + jnp.float32(1.0e-9))
    horizontal_extension = jnp.maximum(
        ball_distance - jnp.asarray(config.ball_radius_m, jnp.float32), 0.0
    )
    lunge = jnp.clip(
        (horizontal_extension - jnp.asarray(contact.carry_radius_m, jnp.float32))
        / jnp.maximum(
            jnp.asarray(
                contact.challenge_radius_m - contact.carry_radius_m, jnp.float32
            ),
            jnp.float32(1.0e-6),
        ),
        0.0,
        1.0,
    )
    control_advantage = jnp.clip(
        0.5 + 0.5 * (ball_control - ball_control[safe_carrier]), 0.0, 1.0
    )
    return _contest_challenge_success_favourability(
        closing,
        rear_alignment,
        lunge,
        control_advantage,
    )




def _challenge_attempt_probability(
    scene: SharedScene,
    features: _RulePolicyFeatures,
    nearest_opponent_relative: jax.Array,
    config: RulePolicyConfig,
) -> jax.Array:
    probability = _contextual_bernoulli(
        config.challenge_attempt_probability,
        _challenge_context(scene, features, config),
        config.challenge_context_logit_limit,
    )
    return jnp.where(
        _inside_own_penalty_area(scene, config),
        jnp.minimum(
            probability,
            jnp.asarray(config.challenge_penalty_area_probability_cap, jnp.float32),
        ),
        probability,
    )


def _contextual_quick_relay(
    control_ticks: jax.Array,
    completion: jax.Array,
    pressured: jax.Array,
    alignment: jax.Array,
    arrival_advantage: jax.Array,
    progress_m: jax.Array,
    config: RulePolicyConfig,
) -> jax.Array:
    """압박 또는 확실한 전진 outlet에서 안전한 첫 reception relay를 판정한다."""

    return (
        (control_ticks <= config.quick_relay_control_ticks)
        & (completion >= config.quick_relay_completion_floor)
        & (
            pressured
            | (
                (completion >= config.progressive_carry_release_completion)
                & (progress_m >= config.progressive_carry_release_m)
            )
        )
        & (alignment >= config.quick_relay_alignment_floor)
        & (arrival_advantage >= config.quick_relay_arrival_advantage_floor)
        & (progress_m >= -jnp.float32(2.0))
    )


def _ground_path_intercept(
    ball_position: jax.Array,
    ball_velocity: jax.Array,
    position: jax.Array,
    max_speed: jax.Array,
    config: RulePolicyConfig,
) -> tuple[jax.Array, jax.Array]:
    """각 선수가 굴러가는 공의 경로를 가장 빨리 만나는 시각과 지점을 계산한다.

    |d + v t| = s t의 가장 작은 양의 근을 닫힌 형태로 구한다. horizon 안에 만날 수 없으면
    horizon 시점의 경로 지점까지 남은 추격 거리를 더한 값으로 순위만 매긴다. Rolling 감속은
    velocity factor로 보수적으로 근사하며, 선수축 broadcasting만 사용하고 pairwise tensor는
    만들지 않는다.
    """

    velocity = ball_velocity * jnp.asarray(
        config.ground_intercept_velocity_factor, jnp.float32
    )
    delta = ball_position - position
    speed_squared = jnp.maximum(max_speed * max_speed, jnp.float32(1.0e-6))
    ball_speed_squared = jnp.sum(velocity * velocity, axis=-1)
    quadratic = ball_speed_squared - speed_squared
    linear = 2.0 * jnp.sum(delta * velocity, axis=-1)
    constant = jnp.sum(delta * delta, axis=-1)
    raw_discriminant = linear * linear - 4.0 * quadratic * constant
    root = jnp.sqrt(jnp.maximum(raw_discriminant, 0.0))
    player_faster = quadratic < 0.0
    faster_time = (linear + root) / jnp.maximum(-2.0 * quadratic, jnp.float32(1.0e-6))
    slower_time = (-linear - root) / jnp.maximum(2.0 * quadratic, jnp.float32(1.0e-6))
    # 공이 더 빠를 때는 실제 실근이 있고 그 근이 양수일 때만 만날 수 있다. Discriminant를
    # clip한 값으로 판정하면 접근 중이지만 닿지 않는 공을 만날 수 있다고 오판한다.
    slower_feasible = (
        (~player_faster) & (raw_discriminant >= 0.0) & (-linear - root > 0.0)
    )
    horizon = jnp.asarray(config.ground_intercept_horizon_s, jnp.float32)
    feasible = player_faster | slower_feasible
    feasible_time = jnp.clip(
        jnp.where(player_faster, faster_time, slower_time), 0.0, horizon
    )
    # 공이 더 빠르고 만날 수 없으면 horizon 뒤 경로 지점까지의 잔여 추격 거리로 순위를
    # 매긴다. 이 값은 horizon보다 크므로 실제 intercept 가능한 선수가 항상 앞선다.
    horizon_delta = delta + velocity * horizon
    infeasible_time = horizon + jnp.sqrt(
        jnp.sum(horizon_delta * horizon_delta, axis=-1) + 1.0e-6
    ) / jnp.sqrt(speed_squared)
    time = jnp.where(feasible, feasible_time, infeasible_time)
    point = ball_position + velocity * jnp.minimum(time, horizon)[:, None]
    return time, point


def _observed_aerial_reception(observation, config, contact_physics):
    """전 정책 공통의 [N] 도착 비용, [N,2] 경합 목표와 cover 기준점.

    9개 bounded 시간과 개인별 reach/standing 높이 교차 시각만 검사한다.
    미래 상태를 읽지 않는 nominal ballistic 근사이며 공중 pose/점프 action은 없다.
    """
    scene = observation.scene
    count = scene.valid.shape[0]
    radius = jnp.float32(config.ball_radius_m)
    height = jnp.maximum(observation.ball_height - radius, 0.0)
    vertical = observation.ball_vertical_velocity
    gravity = jnp.maximum(jnp.float32(config.gravity_mps2), 1.e-3)
    horizon = jnp.float32(config.aerial_intercept_horizon_s)
    landing = jnp.clip((vertical + jnp.sqrt(vertical**2 + 2.0 * gravity * height)) / gravity, 0.0, horizon)
    grid = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 9) * landing, (count, 9))
    levels = jnp.stack((observation.reach_height, observation.height), axis=-1)
    roots = (vertical + jnp.sqrt(jnp.maximum(vertical**2 + 2.0 * gravity * (height - levels), 0.0))) / gravity
    times = jnp.concatenate((grid, jnp.clip(roots, 0.0, landing)), axis=-1)
    points = scene.ball_position_team[:, None, :] + times[..., None] * scene.ball_velocity_team[:, None, :]
    ball_bottom = height + vertical * times - 0.5 * gravity * times**2
    distance = jnp.sqrt(jnp.sum((points - scene.position_team[:, None, :])**2, axis=-1) + 1.e-6)
    speed = jnp.maximum(observation.max_speed, 1.e-3)
    feasible = (
        (distance <= speed[:, None] * times + contact_physics.carry_radius_m + radius)
        & (ball_bottom <= observation.reach_height[:, None] + 1.e-5)
        & scene.valid[:, None]
        & scene.contact_ready[:, None]
    )
    effort = jnp.clip(
        (ball_bottom - observation.height[:, None])
        / jnp.maximum((observation.reach_height - observation.height)[:, None], 1.e-3), 0.0, 1.0,
    )
    costs = jnp.where(feasible, times + 0.03 * effort, 1.e6)
    best = jnp.argmin(costs, axis=-1)
    indices = jnp.arange(count)
    can_receive = jnp.any(feasible, axis=-1)
    fallback_point = scene.ball_position_team + landing * scene.ball_velocity_team
    fallback_cost = horizon + jnp.sqrt(jnp.sum((fallback_point - scene.position_team)**2, axis=-1) + 1.e-6) / speed
    own_cost = jnp.where(can_receive, costs[indices, best], fallback_cost)
    own_point = jnp.where(can_receive[:, None], points[indices, best], fallback_point)
    same_team = scene.team_id[:, None] == scene.team_id[None, :]
    own_pass = _own_pass_flight(
        scene, observation.possession_previous_team, observation.last_contact_actor,
        observation.last_contact_intent, observation.last_contact_kick_applied,
    )
    last_actor = jnp.clip(observation.last_contact_actor, 0, count - 1)
    is_last_actor = indices == last_actor
    alternate = jnp.any(same_team & (scene.valid & (~observation.is_goalkeeper) & (~is_last_actor))[None, :], axis=-1)
    candidate = scene.valid & can_receive & (~(own_pass & is_last_actor & alternate))
    team_costs = jnp.where(same_team & candidate[None, :], own_cost[None, :], 1.e6)
    opponent_costs = jnp.where(scene.opponent & candidate[None, :], own_cost[None, :], 1.e6)
    team_best = jnp.argmin(team_costs, axis=-1)
    opponent_best = jnp.argmin(opponent_costs, axis=-1)
    opponent_first = jnp.min(opponent_costs, axis=-1) + 0.05 < jnp.min(team_costs, axis=-1)
    winner = jnp.where(opponent_first, opponent_best, team_best)
    winner_point = own_point[winner].at[:, 0].multiply(scene.attack_direction[winner] * scene.attack_direction)
    has_winner = jnp.minimum(jnp.min(team_costs, axis=-1), jnp.min(opponent_costs, axis=-1)) < 1.e5
    contest_point = jnp.where(has_winner[:, None], winner_point, fallback_point)
    # 상대가 먼저 닿을 때는 이미 사라질 착지점 대신 그 접촉점의 goal-side에서 경합한다.
    contain_point = contest_point + jnp.asarray((-0.8, 0.0), jnp.float32)
    target = jnp.where(opponent_first[:, None], contain_point, own_point)
    contain_cost = jnp.sqrt(jnp.sum((contain_point - scene.position_team)**2, axis=-1) + 1.e-6) / speed
    chase_cost = jnp.where(opponent_first, contain_cost, own_cost)
    aerial = observation.ball_height >= config.aerial_intercept_height_m
    return (
        jnp.where(aerial, chase_cost, 0.0),
        jnp.where(aerial, target, scene.ball_position_team),
        jnp.where(aerial, contest_point, scene.ball_position_team),
    )


def _observed_aerial_ball_target(observation, config):
    return _observed_aerial_reception(observation, config, ContactPhysics())[1]


def _restart_shape_target(
    scene: SharedScene,
    formation_anchor: jax.Array,
    is_goalkeeper: jax.Array,
    restart_kind: jax.Array,
    own_restart: jax.Array,
    config: RulePolicyConfig,
) -> tuple[jax.Array, jax.Array]:
    """관측된 restart context를 압축 team moments에 투영한다."""

    if scene.valid.shape[0] != STANDARD_PLAYER_SLOTS:
        return jnp.zeros_like(scene.position_team), jnp.zeros_like(scene.valid)

    kind = jnp.where(restart_kind == RK_OFFSIDE, RK_FREEKICK, restart_kind)
    side = jnp.where(own_restart, 0, 1).astype(jnp.int32)
    ball = scene.ball_position_team
    zone = jnp.where(ball[:, 0] < -17.5, 0, jnp.where(ball[:, 0] > 17.5, 2, 1))
    moment = _RESTART_SHAPE_MOMENTS[kind, side, zone]
    valid = _RESTART_SHAPE_VALID[kind, side, zone]

    same_team = scene.team_id[:, None] == scene.team_id[None, :]
    outfield = same_team & scene.valid[None, :] & (~is_goalkeeper[None, :])
    count = jnp.maximum(jnp.sum(outfield, axis=-1), 1)
    centre = (
        jnp.sum(
            jnp.where(outfield[..., None], formation_anchor[None, :, :], 0.0), axis=1
        )
        / count[:, None]
    )
    centred = formation_anchor[None, :, :] - centre[:, None, :]
    variance = (
        jnp.sum(jnp.where(outfield[..., None], centred * centred, 0.0), axis=1)
        / count[:, None]
    )
    scale = jnp.sqrt(jnp.maximum(variance, jnp.float32(1.0)))
    normalized = (formation_anchor - centre) / scale
    spread = jnp.stack(
        (
            jnp.maximum(moment[:, 2], jnp.float32(3.0)),
            jnp.maximum(moment[:, 3], jnp.float32(5.0)),
        ),
        axis=-1,
    )
    ball_side = jnp.where(ball[:, 1] < 0.0, -1.0, 1.0)
    target = jnp.stack(
        (
            moment[:, 0] + normalized[:, 0] * spread[:, 0],
            ball_side * moment[:, 1] + normalized[:, 1] * spread[:, 1],
        ),
        axis=-1,
    )
    limits = jnp.asarray(
        (
            _interior_limit_m(config.field_half_length_m, 1.0),
            _interior_limit_m(config.field_half_width_m, 1.0),
        ),
        dtype=jnp.float32,
    )
    target = jnp.clip(target, -limits, limits)
    standard_pitch = jnp.isclose(config.field_half_length_m, 52.5) & jnp.isclose(
        config.field_half_width_m, 34.0
    )
    valid = (
        valid
        & scene.valid
        & (~is_goalkeeper)
        & (count == 10)
        & standard_pitch
        & jnp.any(scene.restart_taker)
    )
    return target, valid


def _onside_limit_x(scene: SharedScene) -> jax.Array:
    """각 observer의 team frame에서 Law 11 onside 한계 x를 O(N²) reduction 하나로 만든다.

    한계는 두 번째로 깊은 상대와 공 중 더 앞선 쪽이다. 기존 `relative_position`만
    재사용하며 새 pairwise tensor를 만들지 않는다. 동점 상대를 함께 제외하지 않도록
    가장 앞선 상대의 index만 masking한다.
    """

    count = scene.valid.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    opponent_x = jnp.where(
        scene.opponent & scene.valid[None, :],
        scene.position_team[:, 0:1] + scene.relative_position[..., 0],
        -jnp.inf,
    )
    deepest = jnp.argmax(opponent_x, axis=-1)
    second = jnp.max(
        jnp.where(indices[None, :] == deepest[:, None], -jnp.inf, opponent_x), axis=-1
    )
    second = jnp.where(jnp.isfinite(second), second, scene.ball_position_team[:, 0])
    return jnp.maximum(second, scene.ball_position_team[:, 0])


def _own_pass_flight(
    scene: SharedScene,
    previous_team: jax.Array,
    actor: jax.Array,
    intent: jax.Array,
    kick_applied: jax.Array,
) -> jax.Array:
    """유효한 마지막 접촉과 일치하는 자기 팀 pass 비행만 인정한다."""

    count = scene.valid.shape[0]
    valid_actor = (actor >= 0) & (actor < count)
    safe_actor = jnp.clip(actor, 0, count - 1)
    return (
        scene.ball_live
        & (~jnp.any(scene.possessor))
        & (scene.team_id == previous_team)
        & valid_actor
        & scene.valid[safe_actor]
        & (scene.team_id == scene.team_id[safe_actor])
        & (intent == INTENT_PASS)
        & kick_applied
    )


def _attack_shape_phase(
    scene: SharedScene,
    own_possession: jax.Array,
    own_pass_in_flight: jax.Array,
    restart_kind: jax.Array,
    config: RulePolicyConfig,
) -> jax.Array:
    """실제 소유와 움직이는 자기 팀 pass에서만 공격 지원 위치를 유지한다."""

    speed_squared = jnp.sum(scene.ball_velocity_team**2, axis=-1)
    moving_pass = (
        own_pass_in_flight
        & scene.ball_live
        & (restart_kind == 0)
        & (jnp.sqrt(speed_squared) >= jnp.float32(config.attack_pass_flight_min_speed_mps))
        & (speed_squared > 0.0)
        & jnp.bool_(config.attack_pass_flight_support_enabled)
    )
    return own_possession | moving_pass


def _final_third_shape_target(
    scene: SharedScene,
    role: jax.Array,
    support_anchor: jax.Array,
    is_goalkeeper: jax.Array,
    own_possession: jax.Array,
    config: RulePolicyConfig,
) -> tuple[jax.Array, jax.Array]:
    """Role별 staggered lane을 만드는 observation-only final-third target."""

    ball_x = scene.ball_position_team[:, 0]
    anchor_y = support_anchor[:, 1]
    x_offset = jnp.select(
        (
            role == ROLE_CENTRE_BACK,
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (-25.0, -12.0, -4.0, 2.0, 8.0, 6.0),
        default=0.0,
    ).astype(jnp.float32)
    lateral_scale = jnp.select(
        (
            role == ROLE_CENTRE_BACK,
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (0.75, 0.90, 0.55, 0.85, 0.20, 0.50),
        default=1.0,
    ).astype(jnp.float32)
    x_limit = _interior_limit_m(config.field_half_length_m, 8.0)
    y_limit = _interior_limit_m(config.field_half_width_m, 2.0)
    # Forward는 고정 ball offset 대신 관측 가능한 Law 11 한계 바로 뒤까지 전진한다.
    # 수비 line이 내려가면 그만큼 앞선 option이 생긴다. 기존 offset과의 최댓값은
    # line을 넘을 수 있으므로 실제 소유 중에는 아래에서 blend 결과도 제한한다.
    target_x = ball_x + x_offset
    if config.offside_shoulder_enabled:
        forward_role = (role == ROLE_CENTRE_FORWARD) | (role == ROLE_WIDE_FORWARD)
        shoulder_x = _onside_limit_x(scene) - jnp.float32(
            config.offside_shoulder_margin_m
        )
        target_x = jnp.where(forward_role, jnp.maximum(target_x, shoulder_x), target_x)
    role_target = jnp.stack(
        (
            jnp.clip(
                target_x,
                -x_limit,
                x_limit,
            ),
            jnp.clip(
                lateral_scale * anchor_y,
                -y_limit,
                y_limit,
            ),
        ),
        axis=-1,
    )
    phase_start = jnp.select(
        (
            role == ROLE_CENTRE_BACK,
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (8.0, 8.0, 10.0, 11.0, 12.0, 12.0),
        default=12.0,
    ).astype(jnp.float32)
    phase_end = jnp.select(
        (
            role == ROLE_CENTRE_BACK,
            role == ROLE_FULL_BACK,
            role == ROLE_CENTRE_MIDFIELDER,
            role == ROLE_WIDE_MIDFIELDER,
            role == ROLE_CENTRE_FORWARD,
            role == ROLE_WIDE_FORWARD,
        ),
        (22.0, 22.0, 24.0, 25.0, 26.0, 26.0),
        default=28.0,
    ).astype(jnp.float32)
    phase = jnp.clip(
        (ball_x - phase_start) / jnp.maximum(phase_end - phase_start, 1.0),
        0.0,
        1.0,
    )
    # Smoothstep은 threshold 부근의 속도 방향 급변을 줄이면서 [0, 1] 범위를 유지한다.
    phase = phase * phase * (jnp.float32(3.0) - jnp.float32(2.0) * phase)
    target = support_anchor + phase[:, None] * (role_target - support_anchor)
    if config.offside_shoulder_enabled:
        # 이미 방출된 pass의 runner는 현재 line을 넘어도 합법이므로 제한하지 않는다.
        # 원래 anchor가 line 밖인 경우도 있어 role_target 이전이 아닌 blend 뒤에 적용한다.
        cap_forward = forward_role & jnp.any(scene.possessor)
        target = target.at[:, 0].set(
            jnp.where(
                cap_forward,
                jnp.minimum(target[:, 0], jnp.maximum(shoulder_x, 0.0)),
                target[:, 0],
            )
        )
    active = (
        own_possession
        & scene.ball_live
        & (phase > 0.0)
        & scene.valid
        & (~is_goalkeeper)
    )
    return target, active


def _zona_cutback_support_target(
    scene: SharedScene,
    role: jax.Array,
    same_team: jax.Array,
    support_anchor: jax.Array,
    attacking_shape: jax.Array,
    player_tactical_plan: jax.Array,
    config: RulePolicyConfig,
) -> jax.Array:
    """좁은 수비 상대 wide attack에서 CM 한 명을 컷백 수신 위치로 보낸다.

    입력·출력은 [N, 2] team-frame m이며, 비대상 row는 원래 anchor와 같다.
    """

    if config.zona_cutback_support_strength == 0.0:
        return support_anchor
    count = role.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    opponent_y = scene.position_team[:, 1:2] + scene.relative_position[..., 1]
    valid_opponent = scene.opponent & scene.valid[None, :]
    minimum = jnp.min(jnp.where(valid_opponent, opponent_y, jnp.inf), axis=-1)
    maximum = jnp.max(jnp.where(valid_opponent, opponent_y, -jnp.inf), axis=-1)
    opponent_width = jnp.where(
        jnp.any(valid_opponent, axis=-1),
        maximum - minimum,
        jnp.float32(2.0 * config.field_half_width_m),
    )
    midfield = same_team & scene.valid[None, :] & (
        role[None, :] == ROLE_CENTRE_MIDFIELDER
    )
    midfield_distance = jnp.abs(
        scene.position_team[None, :, 1]
        - scene.ball_position_team[:, None, 1]
    )
    chosen = jnp.argmin(
        jnp.where(midfield, midfield_distance, jnp.inf), axis=-1
    )
    ball_x = scene.ball_position_team[:, 0]
    ball_y = scene.ball_position_team[:, 1]
    active = (
        (player_tactical_plan == int(RuleTacticalPlan.ZONA_MISTA))
        & attacking_shape & scene.ball_live & scene.valid
        & (indices == chosen) & jnp.any(midfield, axis=-1)
        & (ball_x >= jnp.float32(config.cross_min_depth_m))
        & (jnp.abs(ball_y) >= jnp.float32(config.cross_min_width_m))
        & (opponent_width <= jnp.float32(
            2.0 * config.penalty_area_half_width_m + 4.0
        ))
    )
    cutback = jnp.stack(
        (
            jnp.minimum(
                ball_x - jnp.float32(3.0),
                jnp.float32(config.field_half_length_m - 11.0),
            ),
            jnp.clip(jnp.float32(0.40) * ball_y, -8.0, 8.0),
        ),
        axis=-1,
    )
    strength = jnp.asarray(config.zona_cutback_support_strength, jnp.float32)
    return jnp.where(
        active[:, None],
        support_anchor + strength * (cutback - support_anchor),
        support_anchor,
    )


def _policy_from_scene(
    scene: SharedScene,
    prospective_offside: jax.Array,
    goalkeeper_claim: jax.Array,
    goalkeeper_holding: jax.Array,
    restart_retouch_blocked: jax.Array,
    config: RulePolicyConfig,
    *,
    ball_physics: BallPhysics,
    contact_physics: ContactPhysics,
    is_goalkeeper: jax.Array,
    max_speed: jax.Array,
    height: jax.Array,
    reach_height: jax.Array,
    ball_control: jax.Array,
    role: jax.Array,
    formation_anchor: jax.Array,
    restart_kind: jax.Array,
    restart_indirect: jax.Array,
    ball_chase_target_team: jax.Array,
    aerial_intercept_time: jax.Array,
    aerial_contest_target: jax.Array,
    counterpress_active: jax.Array,
    possession_previous_team: jax.Array,
    last_contact_actor: jax.Array,
    last_contact_intent: jax.Array,
    last_contact_kick_applied: jax.Array,
    possession_control_ticks: jax.Array,
    possession_episode_ticks: jax.Array,
    tactical_plan: jax.Array,
    ball_height: jax.Array,
    ball_vertical_velocity: jax.Array,
    policy_key,
    include_diagnostics: bool,
):
    count = scene.valid.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    player_tactical_plan = tactical_plan[scene.team_id]
    tactical_profile = _effective_tactical_profile(
        player_tactical_plan, scene.stamina_long
    )
    lead_protection = _lead_protection_intensity(
        scene.score_for, scene.score_against, config
    )
    tactical_profile = tactical_profile.at[:, _TACTIC_FULLBACK_ADVANCE].add(
        -lead_protection
        * jnp.asarray(config.lead_protection_fullback_retreat_m, jnp.float32)
    )
    same_team = scene.team_id[:, None] == scene.team_id[None, :]
    features = _policy_features(scene)
    availability = intent_availability(
        scene,
        goalkeeper_holding=goalkeeper_holding,
        restart_retouch_blocked=restart_retouch_blocked,
        contact=contact_physics,
        ball=BallGeometry(radius_m=config.ball_radius_m),
    )
    # Public availability is deliberately a conservative upper envelope based on
    # the largest configured contact radius.  The rule policy, however, knows the
    # intent it is about to submit and must not use goalkeeper/challenge reach as
    # ordinary foot/chest/head reach.  Otherwise a large goalkeeper radius opens
    # PASS/SHOOT for an outfield loose-ball actor even though the contact kernel
    # will reject that action at ``carry_radius_m``.
    ordinary_contact_reachable = (
        features.ball_distance_squared
        <= (
            jnp.asarray(contact_physics.carry_radius_m, jnp.float32)
            + jnp.asarray(config.ball_radius_m, jnp.float32)
        )
        ** 2
    )
    challenge_contact_reachable = (
        features.ball_distance_squared
        <= (
            jnp.asarray(contact_physics.challenge_radius_m, jnp.float32)
            + jnp.asarray(config.ball_radius_m, jnp.float32)
        )
        ** 2
    )
    open_play = scene.ball_live & (~jnp.any(scene.restart_taker))
    for ordinary_intent in (INTENT_PASS, INTENT_SHOOT):
        availability = availability.at[:, ordinary_intent].set(
            availability[:, ordinary_intent]
            & ((~open_play) | ordinary_contact_reachable)
        )
    availability = availability.at[:, INTENT_CHALLENGE].set(
        availability[:, INTENT_CHALLENGE] & ((~open_play) | challenge_contact_reachable)
    )
    control_receive_reachable = features.ball_distance_squared <= config.receive_distance_m**2
    if config.possessor_control_contact_extension_enabled:
        # Loose-ball 수신 gate는 유지하고 실제 소유자의 낮은 공만 접촉 범위까지 허용한다.
        control_receive_reachable = control_receive_reachable | (
            open_play
            & scene.valid
            & scene.possessor
            & (~is_goalkeeper)
            & (~goalkeeper_holding)
            & scene.contact_ready
            & jnp.isfinite(ball_height)
            & jnp.isfinite(height)
            & (ball_height >= 0.0)
            & (ball_height <= height * contact_physics.pelvis_height_factor + config.ball_radius_m)
        )
    availability = availability.at[:, INTENT_CONTROL].set(
        availability[:, INTENT_CONTROL]
        & (
            control_receive_reachable
            & ordinary_contact_reachable
            | goalkeeper_claim
        )
    )
    # Observation의 lawful predictive claim을 보존하되, direct policy 호출의 custom
    # ContactPhysics가 더 작은 goalkeeper envelope를 쓰면 같은 closest-point geometry로
    # fail-closed한다. Current-frame availability와 교집합하면 다음 control interval에
    # 들어오는 공을 미리 arm하는 기본 contract까지 잘못 닫힌다.
    claim_delta = scene.ball_relative_position[:, :2]
    claim_relative_velocity = scene.ball_velocity_team - scene.velocity_team
    claim_speed_squared = jnp.sum(claim_relative_velocity**2, axis=-1)
    claim_time = jnp.clip(
        -jnp.sum(claim_delta * claim_relative_velocity, axis=-1)
        / jnp.maximum(claim_speed_squared, jnp.float32(1.0e-6)),
        0.0,
        jnp.asarray(config.goalkeeper_claim_horizon_s, dtype=jnp.float32),
    )
    claim_closest_delta = claim_delta + claim_time[:, None] * claim_relative_velocity
    claim_contact_compatible = (
        jnp.sum(claim_closest_delta**2, axis=-1)
        <= (
            jnp.asarray(contact_physics.goalkeeper_radius_m, dtype=jnp.float32)
            + jnp.asarray(config.ball_radius_m, dtype=jnp.float32)
        )
        ** 2
    )
    # Predictive claim은 reach뿐 아니라 이 frame의 법적 availability도 만족해야 한다.
    # 특히 restart taker의 untouched second-touch lock을 마지막 GK override가 다시
    # CONTROL로 열어서는 안 된다.
    goalkeeper_claim_available = goalkeeper_claim
    goalkeeper_claim = (
        goalkeeper_claim & claim_contact_compatible & availability[:, INTENT_CONTROL]
    )
    pressure = (
        features.nearest_opponent_distance_squared <= config.pressure_distance_m**2
    )
    nearest_opponent = jnp.argmin(
        jnp.where(scene.opponent, scene.distance_squared, jnp.inf), axis=-1
    )
    observer_index = jnp.arange(count, dtype=jnp.int32)
    pass_actor_mask = scene.possessor | scene.restart_taker | goalkeeper_holding
    pass_actor = jnp.argmax(pass_actor_mask.astype(jnp.int32)).astype(jnp.int32)
    pass_actor_valid = jnp.any(pass_actor_mask)
    release_contest_clear = (
        pass_actor_valid & scene.possessor[pass_actor]
        & (restart_kind[pass_actor] == 0)
        & _pass_release_contest_clear(
            scene, pass_actor, max_speed, goalkeeper_claim_available, config, contact_physics,
            is_goalkeeper=is_goalkeeper,
        )
    )
    (
        goal_centre_vector,
        goal_distance_squared,
        goal_distance,
        centrality,
        opponent_goalkeeper,
        _goalkeeper_y,
        shot_quality,
        inside_attacking_box,
    ) = _shot_quality_context(
        scene, is_goalkeeper, config, contact_physics,
        carrier_release_clear=release_contest_clear,
    )
    nearest_opponent_relative = scene.relative_position[
        observer_index, nearest_opponent
    ]

    # 실제 contact actor 한 명의 receiver row만 평가한다.
    pass_relative = scene.relative_position[pass_actor]
    pass_distance = jnp.sqrt(scene.distance_squared[pass_actor] + jnp.float32(1.0e-6))
    pass_actor_nearest_opponent_distance = jnp.sqrt(
        features.nearest_opponent_distance_squared[pass_actor] + jnp.float32(1.0e-6)
    )
    pass_release_value_scale, pass_release_ready = _pass_release_window(
        pass_actor_nearest_opponent_distance, config, contact_physics,
        contest_clear=release_contest_clear,
    )
    # Restart는 상대가 법정 거리 밖에 있고 held goalkeeper release는 별도 안전 target
    # 검사를 쓴다. 두 경로를 open-play carrier의 kick-space gate로 닫지 않는다.
    protected_pass_release = (
        scene.restart_taker[pass_actor] | goalkeeper_holding[pass_actor]
    )
    pass_release_value_scale = jnp.where(
        protected_pass_release, jnp.float32(1.0), pass_release_value_scale
    )
    pass_release_ready = protected_pass_release | pass_release_ready
    legal_receiver = scene.teammate[pass_actor] & (~prospective_offside)
    # Receiver는 이미 현재 관측 위치에 있으므로 passer-receiver 전체 거리를
    # 이동시간으로 해석하면 먼 전방 receiver를 이중으로 벌점 준다. 현재 위치에서
    # 요구되는 bounded lead를 사용하며 추가 거리 prior는 설정으로만 남긴다.
    receiver_eta = _receiver_arrival_eta(pass_distance, max_speed, config)
    nearest_defender_speed = max_speed[nearest_opponent]
    defender_eta = features.receiver_space / jnp.maximum(
        nearest_defender_speed, jnp.float32(1.0e-3)
    )
    reception_advantage = jax.nn.sigmoid(
        jnp.float32(2.0) * (defender_eta - receiver_eta)
    )
    # FootballWorld Policy48의 DFL calibration 원칙을 따른다. 전진 거리를 meter
    # 그대로 더하면 completion 최대값보다 커져 먼 선수가 안전성을 무조건 덮어쓴다.
    # 기존 receiver row를 재사용해 progress를 pitch 길이로 bounded하고, 양의 전진
    # 가치만 completion에 조건부로 더한다. Backward recycling 비용은 별도 항이다.
    attacking_third_actor = scene.position_team[pass_actor, 0] >= jnp.asarray(
        config.field_half_length_m / 3.0, jnp.float32
    )
    progress_scale = jnp.where(
        attacking_third_actor,
        jnp.asarray(2.0 * config.field_half_length_m / 3.0, jnp.float32),
        jnp.asarray(config.field_half_length_m, jnp.float32),
    )
    signed_progress = jnp.clip(
        pass_relative[:, 0] / jnp.maximum(progress_scale, 1.0),
        -1.0,
        1.0,
    )
    backward_receiver = pass_relative[:, 0] < 0.0
    direct_outlet_ranking = (
        (player_tactical_plan[pass_actor] == int(RuleTacticalPlan.CATENACCIO))
        & scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0)
    )
    # Direct 전술에서는 이미 충분히 열린 outlet의 빈 공간을 끝없이 보상하지 않는다.
    # Lane/arrival 안전성은 유지하고 30~50 m의 후방 여유가 전진 가치를 덮는 것만 막는다.
    receiver_space_score = jnp.where(
        attacking_third_actor | backward_receiver | direct_outlet_ranking,
        jnp.minimum(
            features.receiver_space,
            jnp.asarray(2.0 * config.pressure_distance_m, jnp.float32),
        ),
        features.receiver_space,
    )
    # Own goalkeeper는 항상 열려 있어 completion·space가 최대이므로 마지막 outlet으로만
    # 사용한다. 또한 pass ray가 own goal mouth를 지나고 receiver 뒤의 rolling overrun이
    # goal line까지 닿으면 miscontrol 한 번이 자책골이 되므로 강하게 벌점한다. 두 항 모두
    # 이미 계산한 receiver row만 사용하며 새 pairwise tensor를 만들지 않는다.
    # Own-goal ray의 원점은 passer 중심이 아니라 실제 contact가 일어나는 공이다.
    # 두 위치는 contact radius 안에서도 약 1 m 어긋날 수 있고, goal-post 근처의
    # backward release에서는 그 차이만으로 mouth 교차 판정이 뒤집힌다.
    pass_release_position = scene.ball_position_team[pass_actor]
    pass_speed_max = jnp.where(
        scene.restart_taker[pass_actor] & (restart_kind[pass_actor] == RK_THROWIN),
        jnp.float32(contact_physics.throw_speed_max_mps),
        jnp.float32(config.kick_speed_max_mps),
    )
    candidate_direction, candidate_power = _ground_pass_controls(
        pass_relative,
        scene.velocity_team - scene.velocity_team[pass_actor],
        config,
        ball_physics=ball_physics,
        passer_position=scene.position_team[pass_actor],
        release_speed_max_mps=pass_speed_max,
    )
    # Foot actuator와 같이 incoming velocity에 requested impulse를 더한 실제 예상
    # release ray를 사용한다. Generated-speed cap은 전체 velocity를 양의 scalar로만
    # 줄이므로 planar 방향에는 영향을 주지 않는다. Height compensation 뒤 ground
    # launch angle은 원래 policy ground angle과 같으므로 horizontal impulse도 동일하다.
    ground_launch_angle = _ground_launch_angle(jnp.float32(-0.65), contact_physics)
    release_impulse_speed = (
        candidate_power
        * pass_speed_max
        * jnp.cos(ground_launch_angle)
    )
    expected_release_velocity = (
        scene.ball_velocity_team[pass_actor]
        + release_impulse_speed[:, None] * candidate_direction
    )
    expected_release_speed = jnp.sqrt(
        jnp.sum(expected_release_velocity * expected_release_velocity, axis=-1)
        + jnp.float32(1.0e-6)
    )
    ray_direction = expected_release_velocity / expected_release_speed[:, None]
    release_lane = ray_direction * pass_distance[:, None]
    ground_expected_z = (
        ball_vertical_velocity + candidate_power*pass_speed_max*jnp.sin(ground_launch_angle)
    )
    ground_release_norm = jnp.sqrt(expected_release_speed**2+ground_expected_z**2)
    incoming_norm = jnp.sqrt(jnp.sum(scene.ball_velocity_team[pass_actor]**2)+ball_vertical_velocity**2)
    ground_speed_cap = jnp.maximum(incoming_norm, jnp.float32(contact_physics.kick_speed_max_mps))
    ground_release_speed = expected_release_speed*jnp.minimum(1., ground_speed_cap/jnp.maximum(ground_release_norm, 1.e-6))
    ground_spin = scene.ball_spin_team[pass_actor]
    ground_surface = config.ball_radius_m*scene.attack_direction[pass_actor]*jnp.stack((-ground_spin[1], ground_spin[0]))
    ground_surface_along = jnp.sum(ray_direction*ground_surface, axis=-1)
    ground_surface_cross = ground_surface[0]*ray_direction[:, 1]-ground_surface[1]*ray_direction[:, 0]
    ground_model_valid = (
        (ball_height <= config.ball_radius_m+.02)
        & (jnp.abs(ball_vertical_velocity) <= .1)
        & (jnp.abs(ground_spin[2]) <= 1.)
        & (jnp.abs(ground_surface_cross) <= .5)
        & (~goalkeeper_holding[pass_actor])
        & (restart_kind[pass_actor] == 0)
    )
    fallback_arrival_speed = None
    if config.cross_spin_ground_eta_prior_enabled:
        fallback_arrival_speed = _cross_spin_ground_eta_prior(
            ground_release_speed, ground_surface_cross, ball_height,
            ball_vertical_velocity, ground_spin[2], goalkeeper_holding[pass_actor],
            restart_kind[pass_actor], config,
        )
    lane_completion = _pass_lane_completion(
        release_lane,
        pass_relative,
        scene.opponent[pass_actor],
        config,
        release_speed=ground_release_speed,
        surface_velocity=ground_surface_along,
        ground_valid=ground_model_valid,
        opponent_velocity=scene.relative_velocity[pass_actor]+scene.velocity_team[pass_actor],
        opponent_max_speed=max_speed,
        release_origin_offset=pass_release_position-scene.position_team[pass_actor],
        receiver_distance=jnp.linalg.norm(
            pass_relative-(pass_release_position-scene.position_team[pass_actor]), axis=-1,
        ),
        fallback_arrival_speed=fallback_arrival_speed,
        ball_physics=ball_physics,
    )
    # Soft interception probability는 passer 원점에 가까운 상대에게 reaction time 0을
    # 주므로, ray 위에 서 있어도 약 0.5의 completion을 남길 수 있다. 특히 held GK의
    # 짧은 배급에서는 첫 1--2 m 안의 몸에 바로 맞는 방향을 안전하다고 볼 수 없다.
    # Candidate마다 실제 공 위치에서 incoming velocity를 합친 release ray를 쓰고,
    # 공에서 receiver까지의 경로 중 pressure envelope 침범을 hard-block한다.
    release_relative = pass_relative - (
        pass_release_position - scene.position_team[pass_actor]
    )
    release_receiver_distance = jnp.linalg.norm(release_relative, axis=-1)
    release_forward = jnp.sum(
        ray_direction[:, None, :] * release_relative[None, :, :], axis=-1
    )
    release_lateral = jnp.sqrt(
        jnp.sum(
            (
                release_relative[None, :, :]
                - release_forward[..., None] * ray_direction[:, None, :]
            )
            ** 2,
            axis=-1,
        )
        + jnp.float32(1.0e-6)
    )
    release_corridor_end = jnp.minimum(
        release_receiver_distance,
        jnp.asarray(config.pressure_distance_m, jnp.float32),
    )
    release_corridor_blocked = jnp.any(
        scene.opponent[pass_actor][None, :]
        & (release_forward > jnp.float32(0.0))
        & (release_forward <= release_corridor_end[:, None])
        & (
            release_lateral
            <= jnp.asarray(
                contact_physics.challenge_radius_m + config.ball_radius_m,
                jnp.float32,
            )
        ),
        axis=-1,
    )
    # Candidate별 실제 expected release ray에서 목표보다 먼저 ordinary contact
    # envelope를 가로지르는 같은-team non-target이 있으면 그 후보를 제외한다.
    # Passer와 candidate 자신은 corridor blocker가 아니다.
    candidate_indices = jnp.arange(count, dtype=jnp.int32)
    non_target_teammate = (
        scene.teammate[pass_actor][None, :]
        & (candidate_indices[None, :] != candidate_indices[:, None])
        & (candidate_indices[None, :] != pass_actor)
    )
    teammate_release_corridor_blocked = jnp.any(
        non_target_teammate
        & (release_forward > jnp.float32(0.0))
        & (release_forward < release_receiver_distance[:, None])
        & (
            release_lateral
            <= jnp.asarray(
                contact_physics.carry_radius_m + config.ball_radius_m,
                jnp.float32,
            )
        ),
        axis=-1,
    )
    toward_own_goal = ray_direction[:, 0] < -1.0e-3
    own_goal_line_distance = (
        -jnp.asarray(config.field_half_length_m, jnp.float32) - pass_release_position[0]
    ) / jnp.where(toward_own_goal, ray_direction[:, 0], -1.0)
    own_goal_line_y = (
        pass_release_position[1] + own_goal_line_distance * ray_direction[:, 1]
    )
    # Ground contact 직후에는 기존 rolling spin과 planar velocity의 surface slip이
    # sticking transition을 만들 수 있다. 이 동안 trajectory는 initial release ray와
    # eventual rolling ray 사이로 휘므로 initial ray 하나만 검사하면 post 바깥에서
    # 시작해 mouth 안으로 들어오는 backward pass를 놓친다. Ball kernel의 closed-form
    # sticking delta를 candidate별 expected release에 적용해 두 ray의 envelope를 쓴다.
    pass_spin = scene.ball_spin_team[pass_actor]
    pass_attack_direction = scene.attack_direction[pass_actor]
    rolling_surface_velocity = (
        jnp.asarray(config.ball_radius_m, jnp.float32)
        * pass_attack_direction
        * jnp.stack((-pass_spin[1], pass_spin[0]))
    )
    sticking_fraction = jnp.asarray(
        ball_physics.inertia_ratio / (1.0 + ball_physics.inertia_ratio),
        jnp.float32,
    )
    expected_rolling_velocity = expected_release_velocity - sticking_fraction * (
        expected_release_velocity + rolling_surface_velocity[None, :]
    )
    expected_rolling_speed = jnp.sqrt(
        jnp.sum(expected_rolling_velocity * expected_rolling_velocity, axis=-1)
        + jnp.float32(1.0e-6)
    )
    rolling_ray_direction = expected_rolling_velocity / expected_rolling_speed[:, None]
    rolling_toward_own_goal = rolling_ray_direction[:, 0] < -1.0e-3
    rolling_goal_line_distance = (
        -jnp.asarray(config.field_half_length_m, jnp.float32) - pass_release_position[0]
    ) / jnp.where(rolling_toward_own_goal, rolling_ray_direction[:, 0], -1.0)
    rolling_goal_line_y = (
        pass_release_position[1]
        + rolling_goal_line_distance * rolling_ray_direction[:, 1]
    )
    own_goal_mouth_half_width = jnp.asarray(
        0.5 * config.goal_width_m + config.own_goal_hazard_margin_m,
        jnp.float32,
    )
    # A receiver can collect a backward pass before its eventual trajectory reaches
    # the goal line.  Only veto a goal-mouth crossing within the receiver's expected
    # travel distance (plus the configured overrun allowance); the infinite ray beyond
    # a safely reachable receiver is not part of this pass.
    own_goal_before_receiver = (
        jnp.minimum(own_goal_line_distance, rolling_goal_line_distance)
        - pass_distance
        <= jnp.asarray(config.own_goal_hazard_overrun_m, jnp.float32)
    )
    curve_enters_own_goal_mouth = (
        toward_own_goal
        & rolling_toward_own_goal
        & own_goal_before_receiver
        & (
            jnp.minimum(own_goal_line_y, rolling_goal_line_y)
            <= own_goal_mouth_half_width
        )
        & (
            jnp.maximum(own_goal_line_y, rolling_goal_line_y)
            >= -own_goal_mouth_half_width
        )
    )
    own_goal_line_hazard = toward_own_goal & (
        own_goal_line_distance - pass_distance
        <= jnp.asarray(config.own_goal_hazard_overrun_m, jnp.float32)
    )
    own_goal_hazard = (
        toward_own_goal & own_goal_before_receiver
        & (jnp.abs(own_goal_line_y) <= own_goal_mouth_half_width)
    ) | curve_enters_own_goal_mouth
    # PASS는 scoring action이 아니다. Goal line 바로 옆의 lateral target이나 blocked
    # shot fallback에서 incoming + requested velocity의 합이 상대 mouth를 먼저 가로지를
    # 수 있다. Initial/eventual rolling ray 중 하나 또는 그 사이 curve envelope가
    # 상대 goal aperture를 수신 접근보다 먼저 통과하는 후보만 hard veto한다.
    # Receiver를 이미 지난 무한 ray 연장으로 중앙 전진 pass를 금지하지 않는다.
    toward_opponent_goal = ray_direction[:, 0] > 1.0e-3
    opponent_goal_line_distance = (
        jnp.asarray(config.field_half_length_m, jnp.float32) - pass_release_position[0]
    ) / jnp.where(toward_opponent_goal, ray_direction[:, 0], 1.0)
    opponent_goal_line_y = (
        pass_release_position[1] + opponent_goal_line_distance * ray_direction[:, 1]
    )
    rolling_toward_opponent_goal = rolling_ray_direction[:, 0] > 1.0e-3
    rolling_opponent_goal_line_distance = (
        jnp.asarray(config.field_half_length_m, jnp.float32) - pass_release_position[0]
    ) / jnp.where(rolling_toward_opponent_goal, rolling_ray_direction[:, 0], 1.0)
    rolling_opponent_goal_line_y = (
        pass_release_position[1]
        + rolling_opponent_goal_line_distance * rolling_ray_direction[:, 1]
    )
    opponent_goal_mouth_half_width = jnp.asarray(0.5 * config.goal_width_m, jnp.float32)
    receiver_from_ball = pass_relative - (pass_release_position - scene.position_team[pass_actor])
    receiver_velocity = scene.relative_velocity[pass_actor] + scene.velocity_team[pass_actor]
    release_speed_scale = ground_release_speed / jnp.maximum(expected_release_speed, 1.0e-6)

    def receiver_before_goal(ball_velocity, goal_distance, ray_speed):
        # Initial/rolling 땅볼의 등속 closest approach일 뿐 실제 수신 보장은 아니다.
        # 현재 receiver 거리만 비교하면 incoming velocity가 receiver를 비껴가는
        # ray까지 허용하므로 접촉 envelope 안의 미래 접근을 함께 요구한다.
        relative_velocity = ball_velocity - receiver_velocity
        approach_time = jnp.maximum(
            jnp.sum(receiver_from_ball * relative_velocity, axis=-1)
            / jnp.maximum(jnp.sum(relative_velocity**2, axis=-1), 1.0e-6), 0.0,
        )
        miss_squared = jnp.sum((receiver_from_ball - approach_time[:, None] * relative_velocity)**2, axis=-1)
        goal_time = goal_distance / jnp.maximum(ray_speed, 1.0e-6)
        return (
            legal_receiver & (ray_speed > 1.0e-6) & (goal_distance > 0.0)
            & jnp.isfinite(approach_time) & jnp.isfinite(miss_squared) & jnp.isfinite(goal_time)
            & (approach_time < goal_time)
            & (miss_squared <= (contact_physics.carry_radius_m + config.ball_radius_m)**2)
        )

    initial_receiver_first = receiver_before_goal(
        expected_release_velocity * release_speed_scale[:, None],
        opponent_goal_line_distance, ground_release_speed,
    )
    rolling_receiver_first = receiver_before_goal(
        expected_rolling_velocity * release_speed_scale[:, None],
        rolling_opponent_goal_line_distance, expected_rolling_speed * release_speed_scale,
    )
    opponent_goal_hazard = (
        toward_opponent_goal
        & (opponent_goal_line_distance >= 0.0)
        & (~initial_receiver_first)
        & (jnp.abs(opponent_goal_line_y) <= opponent_goal_mouth_half_width)
    ) | (
        toward_opponent_goal
        & rolling_toward_opponent_goal
        & (opponent_goal_line_distance >= 0.0)
        & (rolling_opponent_goal_line_distance >= 0.0)
        & (~(initial_receiver_first & rolling_receiver_first))
        & (
            jnp.minimum(opponent_goal_line_y, rolling_opponent_goal_line_y)
            <= opponent_goal_mouth_half_width
        )
        & (
            jnp.maximum(opponent_goal_line_y, rolling_opponent_goal_line_y)
            >= -opponent_goal_mouth_half_width
        )
    )
    receiver_service_zone = (
        (scene.position_team[:, 0] >= config.field_half_length_m - config.penalty_area_length_m)
        & (jnp.abs(scene.position_team[:, 1]) <= config.penalty_area_half_width_m)
    )
    cross_origin = (
        scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0)
        & (scene.position_team[pass_actor, 0] >= config.cross_min_depth_m + tactical_profile[pass_actor, _TACTIC_CROSS_DEPTH_DELTA])
        & (jnp.abs(scene.position_team[pass_actor, 1]) >= config.cross_min_width_m + tactical_profile[pass_actor, _TACTIC_CROSS_WIDTH_DELTA])
    )
    corner_origin = scene.restart_taker[pass_actor] & (restart_kind[pass_actor] == RK_CORNER)
    aerial_cross_candidate = (cross_origin | corner_origin) & receiver_service_zone
    aerial_long_candidate = _specialist_long_service(
        player_tactical_plan[pass_actor], pass_distance, pass_relative[:, 0], jnp.ones_like(pass_distance),
        scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0),
    ) & (~is_goalkeeper[pass_actor])
    candidate_launch = jnp.where(aerial_cross_candidate, config.cross_launch, 0.10)
    candidate_loft_power = jnp.maximum(
        _lofted_release_power(pass_distance, candidate_launch, config, contact_physics),
        jnp.where(aerial_cross_candidate, config.cross_min_power, 0.0),
    )
    # 상대 선수도 passer의 team frame으로 변환한다. position_team 자체는 선수별
    # 자기 공격 좌표계이므로 상대 row를 그대로 route에 넘기면 좌우가 뒤집힌다.
    route = aerial_route(
        origin=jnp.concatenate((pass_release_position, jnp.reshape(ball_height, (1,)))),
        incoming_velocity=jnp.concatenate((scene.ball_velocity_team[pass_actor], jnp.reshape(ball_vertical_velocity, (1,)))),
        incoming_spin=scene.ball_spin_team[pass_actor],
        position=scene.position_team[pass_actor] + pass_relative,
        height=height, reach_height=reach_height, max_speed=max_speed,
        opponent=scene.opponent[pass_actor],
        eligible=legal_receiver & pass_actor_valid & (aerial_cross_candidate | aerial_long_candidate),
        power=candidate_loft_power,
        launch_angle=_ground_launch_angle(candidate_launch, contact_physics),
        intended_direction=candidate_direction,
        ball_physics=ball_physics, contact_physics=contact_physics,
        ball_geometry=BallGeometry(radius_m=config.ball_radius_m),
        field_half_extent=(config.field_half_length_m, config.field_half_width_m),
        goalkeeper=is_goalkeeper,
        opponent_penalty_x_min=config.field_half_length_m - config.penalty_area_length_m,
        penalty_half_width=config.penalty_area_half_width_m,
        nonreceiver_teammate=scene.teammate[pass_actor] & (indices != pass_actor),
    )
    use_aerial_route = route.viable & (route.score >= 0.65)
    # Corner는 전술에 관계없이 상대와의 헤딩 경합을 service 후보로 둔다.
    # 17-step 경로 근사의 도착시각 한 표본 이내(0.10 s) 경합만 허용하고,
    # 몸 충돌·경계·입력 오류는 delivery_feasible에서 계속 hard veto한다.
    # Open-play gate는 그대로 둔다.
    corner_contested_route = (
        corner_origin & route.delivery_feasible
        & (route.opponent_time_s + jnp.float32(0.10) >= route.receiver_time_s)
    )
    aerial_quality = jnp.where(use_aerial_route, route.score, 0.0)
    # Loft의 도착 여유 score는 확률이 아니다. 예측 차단이 없는 경로는 clear-lane
    # 등급 1로 두고, 도착 여유는 reception/ranking에 따로 반영한다.
    lane_completion = jnp.where(use_aerial_route, 1.0, lane_completion)
    reception_advantage = jnp.where(use_aerial_route, route.score, reception_advantage)
    pass_score = (
        +0.25 * receiver_space_score
        - config.pass_distance_penalty_per_m * pass_distance
        + config.pass_completion_gain * lane_completion
        + config.pass_reception_gain * reception_advantage
        + config.progressive_pass_value_gain
        * lane_completion
        * jnp.maximum(signed_progress, 0.0)
        - config.backward_pass_penalty
        * jnp.maximum(-pass_relative[:, 0], 0.0)
        / jnp.maximum(
            jnp.asarray(config.pressure_distance_m, jnp.float32),
            jnp.float32(1.0e-3),
        )
        - jnp.where(
            is_goalkeeper,
            jnp.asarray(config.goalkeeper_receiver_penalty, jnp.float32),
            jnp.float32(0.0),
        )
        - jnp.where(
            own_goal_hazard,
            jnp.asarray(config.own_goal_hazard_penalty, jnp.float32),
            jnp.float32(0.0),
        )
        - jnp.where(
            own_goal_line_hazard,
            jnp.asarray(config.own_goal_line_hazard_penalty, jnp.float32),
            jnp.float32(0.0),
        )
        + _narrow_block_switch_bonus(
            pass_relative,
            lane_completion,
            scene.opponent[pass_actor],
            scene.position_team[:, 1],
            player_tactical_plan[pass_actor],
            config.field_half_width_m,
        )
    )
    pass_score = pass_score + jnp.where(
        (aerial_cross_candidate | aerial_long_candidate) & (aerial_quality >= 0.40),
        4.0 * (aerial_quality - 0.5), 0.0,
    )
    specialist_preference = _specialist_pass_bonus(
        player_tactical_plan[pass_actor], pass_distance, pass_relative[:, 0],
        lane_completion, reception_advantage,
        scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0),
    )
    # 자기 goal mouth를 향하는 pass는 soft preference가 아니라 안전 제약이다.
    box_receiver_bonus = _inside_box_receiver_score_bonus(
        shot_quality[pass_actor],
        shot_quality,
        lane_completion,
        inside_attacking_box,
        inside_attacking_box[pass_actor],
        config,
    )
    # Macro의 [0,1] 가치와 receiver ranking의 completion 점수 단위를 맞춘다.
    pass_score = pass_score + config.pass_completion_gain * box_receiver_bonus
    # 벌점만 주면 모든 다른 outlet의 점수가 더 낮을 때 위험한 receiver가 다시
    # argmax로 선택되어, 장거리 backward pass가 그대로 own goal까지 굴러간다.
    goalkeeper_corridor_safe = (~goalkeeper_holding[pass_actor]) | (
        ~release_corridor_blocked
    )
    kickoff_receiver_far_enough = (restart_kind[pass_actor] != RK_KICKOFF) | (
        pass_distance
        >= jnp.asarray(config.kickoff_min_receiver_distance_m, jnp.float32)
    )
    safe_receiver = (
        legal_receiver
        & (~own_goal_hazard)
        & (~opponent_goal_hazard)
        & goalkeeper_corridor_safe
        & ((~teammate_release_corridor_blocked) | use_aerial_route)
        & kickoff_receiver_far_enough
    )
    # 최신 base가 찾은 안전한 박스 진입 기회를 짧은 순환/롱볼 취향으로 가리지 않는다.
    pass_score, target_scalar, has_target_scalar = _select_receiver(
        pass_score, safe_receiver, box_receiver_bonus, lane_completion,
        receiver_space_score, specialist_preference, pass_actor_valid,
        scene, restart_kind, pass_actor, config,
    )
    # JAX 0.10.2 GPU의 dynamic-slice/reduce fusion 오류를 피하기 위해 선택된 ETA를
    # 아주 작은 tie-break로 유지한다. Receiver score의 의미와 threshold는 보존한다.
    target_completion = lane_completion[target_scalar] * (
        jnp.float32(0.999) + jnp.float32(0.001) * reception_advantage[target_scalar]
    )
    target_progress = pass_relative[target_scalar, 0]
    target_position = scene.position_team[target_scalar]
    target_in_service_zone = (
        target_position[0]
        >= jnp.asarray(
            config.field_half_length_m - config.penalty_area_length_m, jnp.float32
        )
    ) & (
        jnp.abs(target_position[1])
        <= jnp.asarray(config.penalty_area_half_width_m, jnp.float32)
    )
    target_relative = pass_relative[target_scalar]
    pass_direction_scalar = candidate_direction[target_scalar][None, :]
    pass_power_scalar = candidate_power[target_scalar][None]
    pass_direction = jnp.broadcast_to(pass_direction_scalar, (count, 2))
    pass_power = jnp.broadcast_to(pass_power_scalar, (count,))
    target = jnp.full((count,), target_scalar, dtype=jnp.int32)
    has_target = pass_actor_mask & has_target_scalar
    goalkeeper_safe_distribution = (
        goalkeeper_holding
        & has_target
        & (target_completion >= config.pass_completion_threshold)
        # Held ball은 보통 own goal line 가까이서 release된다. Lateral/backward
        # target은 작은 residual own-goal velocity만 있어도 contact interval 안에
        # 선을 넘을 수 있으므로 team-frame 전방 성분을 안전 조건으로 요구한다.
        & (pass_direction_scalar[0, 0] >= jnp.float32(0.1))
    )
    goalkeeper_long_distribution = goalkeeper_holding & (~goalkeeper_safe_distribution)

    goalkeeper_index = jnp.argmax(opponent_goalkeeper, axis=-1)
    goalkeeper_position = (
        scene.position_team + scene.relative_position[observer_index, goalkeeper_index]
    )
    goalkeeper_velocity = (
        scene.velocity_team + scene.relative_velocity[observer_index, goalkeeper_index]
    )
    shot_compensation_eligible = _shot_compensation_eligible(
        scene.ball_position_team, ball_height, height, config, contact_physics
    )
    # Lane/value는 의도한 target으로 판단하고 아래 stochastic 오차는 실제 방출에만
    # 적용한다. 둘을 섞으면 오차가 큰 선수가 빗나가는 대신 슛 자체를 포기하며,
    # 실행 정확도를 교정하는 knob가 shot volume까지 비단조적으로 바꾼다.
    nominal_shot_target_y = _nominal_shot_target_y(
        scene.position_team,
        goalkeeper_position,
        goalkeeper_velocity,
        jnp.any(opponent_goalkeeper, axis=-1),
        config,
        contact_physics,
        ball_position=scene.ball_position_team,
        incoming_velocity=scene.ball_velocity_team,
        incoming_vertical_velocity=ball_vertical_velocity,
        compensation_eligible=shot_compensation_eligible,
    )
    shot_target_y = nominal_shot_target_y
    if config.shot_aim_noise_fraction > 0.0:
        aim_key = event_random_key(policy_key, RandomEvent.POLICY_SHOT_DIRECTION)
        aim_noise_scale = _shot_aim_noise_scale(
            goal_centre_vector,
            scene.body_forward_team,
            features.nearest_opponent_distance_squared,
            scene.stamina_short,
            config,
        )
        shot_target_y = shot_target_y + jax.random.normal(
            aim_key, (count,), dtype=jnp.float32
        ) * (jnp.float32(0.5 * config.goal_width_m) * aim_noise_scale)
    shot_origin = scene.position_team
    if config.shot_residual_velocity_compensation:
        shot_origin = jnp.where(shot_compensation_eligible[:, None], scene.ball_position_team, scene.position_team)
    goal_vector = jnp.stack(
        (
            config.field_half_length_m - shot_origin[:, 0],
            shot_target_y - shot_origin[:, 1],
        ),
        axis=-1,
    )
    # Lane은 실제 SHOOT 후보 각자의 observer row에서 계산한다. 무소유 first-time
    # shot에는 pass actor가 없어 argmax fallback slot 0을 쓰면 blocker 판정이 뒤집힌다.
    decision_goal_vector = jnp.stack(
        (
            config.field_half_length_m - shot_origin[:, 0],
            nominal_shot_target_y - shot_origin[:, 1],
        ),
        axis=-1,
    )
    shot_lane_open = _shot_lane_open(
        scene, decision_goal_vector, is_goalkeeper, height, reach_height,
        ball_height, ball_vertical_velocity, config, contact_physics,
    )
    pass_alignment = jnp.sum(
        _unit(target_relative[None, :])[0] * scene.body_forward_team[pass_actor]
    )
    quick_relay = _contextual_quick_relay(
        possession_episode_ticks,
        target_completion,
        pressure[pass_actor],
        pass_alignment,
        reception_advantage[target_scalar],
        target_progress,
        config,
    )
    pass_opportunity = (target_completion >= config.pass_completion_threshold) & (
        (target_progress >= config.progressive_pass_min_m) | pressure[pass_actor]
    )
    # 받은 직후 곧바로 다시 내주지 않도록 소유 에피소드 경과 하한을 둔다. 관측 가능한
    # quick-relay 조건을 만족하면 예외로 즉시 내줄 수 있다. 소유 에피소드는 같은 선수의
    # 반복 touch에서 되돌아가지 않으므로 이 하한이 실제로 구속한다.
    pass_dwell_met = possession_episode_ticks >= jnp.int32(
        config.pass_min_episode_ticks
    )
    if config.progressive_carry_min_ticks > 0:
        carry_commit_met = possession_episode_ticks >= jnp.int32(
            config.progressive_carry_min_ticks
        )
        decisive_progressive_release = (
            target_completion
            >= jnp.asarray(config.progressive_carry_release_completion, jnp.float32)
        ) & (
            target_progress
            >= jnp.asarray(config.progressive_carry_release_m, jnp.float32)
        )
        carry_release = (
            carry_commit_met
            | pressure[pass_actor]
            | quick_relay
            | decisive_progressive_release
        )
    else:
        carry_release = jnp.bool_(True)
    pass_ready = (
        has_target
        & availability[:, INTENT_PASS]
        & pass_opportunity
        & (pass_dwell_met | quick_relay)
        & carry_release
        & pass_release_ready
    )
    regular_shot_quality_threshold = _regular_shot_quality_threshold(
        inside_attacking_box, config
    )
    # 안정된 고품질 기회는 수신 직후에도 살린다. 반대로 빠른 공, 멀어진 공 또는
    # 몸을 돌려야 하는 수신은 기존 dwell을 유지한다. Lane/legality는 아래에서 적용한다.
    shot_alignment = jnp.sum(
        _unit(decision_goal_vector) * scene.body_forward_team, axis=-1
    )
    shot_relative_speed_squared = jnp.sum(
        (scene.ball_velocity_team - scene.velocity_team) ** 2, axis=-1
    )
    settled_clear_chance = (
        (shot_quality >= jnp.asarray(config.first_time_shot_quality_threshold, jnp.float32))
        & (features.ball_distance_squared <= jnp.float32(config.receive_distance_m**2))
        & (shot_relative_speed_squared <= jnp.float32(config.pass_arrival_speed_mps**2))
        & (shot_alignment >= jnp.float32(0.8))
    )
    shot_dwell_met = (
        (~scene.possessor)
        | (possession_episode_ticks >= jnp.int32(config.shot_min_episode_ticks))
        | settled_clear_chance
    )
    lane_clear = (
        shot_lane_open
        if config.shot_requires_open_lane
        else ((~pass_actor_valid) | shot_lane_open)
    )
    shot_ready = (
        shot_dwell_met
        & ((goal_distance_squared <= config.shoot_distance_m**2) | inside_attacking_box)
        & lane_clear
        & (shot_quality >= regular_shot_quality_threshold)
        & availability[:, INTENT_SHOOT]
    )
    # shoot_distance_m 밖에서도 lane이 비고 압박이 없고 중앙이면 원거리 슛을 받는다.
    long_shot_ready = (
        shot_dwell_met
        & (goal_distance_squared > config.shoot_distance_m**2)
        & (goal_distance_squared <= config.long_shot_distance_m**2)
        & (~inside_attacking_box)
        & shot_lane_open
        & (~pressure)
        & (centrality >= jnp.asarray(config.long_shot_centrality_min, jnp.float32))
        & availability[:, INTENT_SHOOT]
    )
    shot_ready = shot_ready | long_shot_ready
    forced_close_shot = goal_distance <= jnp.float32(
        config.shot_forced_distance_fraction * config.shoot_distance_m
    )
    valid_last_actor = (last_contact_actor >= 0) & (last_contact_actor < count)
    safe_last_actor = jnp.clip(last_contact_actor, 0, count - 1)
    # Goalkeeper hand parry와 passive save는 last-contact intent가 각각 CONTROL과
    # MOVE로 남는다. SHOOT만 rebound로 보던 기존 gate는 이 causal 경로를 일반
    # first-time shot으로 오분류했다. 상대 goalkeeper의 non-kick contact를 같은
    # rebound threshold로 묶되 deliberate distribution은 건드리지 않는다.
    opponent_goalkeeper_rebound = (
        valid_last_actor
        & scene.opponent[indices, safe_last_actor]
        & is_goalkeeper[safe_last_actor]
        & (~last_contact_kick_applied)
    )
    rebound_first_time = (~scene.possessor) & (
        (last_contact_intent == INTENT_SHOOT) | opponent_goalkeeper_rebound
    )
    first_time_threshold = jnp.where(
        rebound_first_time,
        jnp.asarray(config.rebound_shot_quality_threshold, jnp.float32),
        jnp.asarray(config.first_time_shot_quality_threshold, jnp.float32),
    )
    first_time_shot_allowed = (shot_quality >= first_time_threshold) | (
        forced_close_shot & (~rebound_first_time)
    )
    shot_ready = shot_ready & (scene.possessor | first_time_shot_allowed)
    previous_teammate = (
        valid_last_actor & same_team[:, safe_last_actor] & (indices != safe_last_actor)
    )
    fresh_unlinked_possession = (
        scene.possessor
        & (~previous_teammate)
        & (possession_episode_ticks < jnp.int32(config.turnover_shot_settle_ticks))
        & (
            shot_quality
            < jnp.asarray(config.first_time_shot_quality_threshold, jnp.float32)
        )
    )
    settled_fraction = jnp.clip(
        possession_episode_ticks.astype(jnp.float32)
        / jnp.maximum(
            jnp.asarray(config.turnover_shot_settle_ticks, jnp.float32),
            jnp.float32(1.0),
        ),
        0.0,
        1.0,
    )
    transition_shot_scale = (
        jnp.asarray(config.turnover_shot_value_scale, jnp.float32)
        + (
            jnp.float32(1.0)
            - jnp.asarray(config.turnover_shot_value_scale, jnp.float32)
        )
        * settled_fraction
    )
    pitch_progress = jnp.clip(
        (scene.position_team[:, 0] + config.field_half_length_m)
        / jnp.maximum(jnp.float32(2.0 * config.field_half_length_m), 1.0),
        0.0,
        1.0,
    )
    pass_progress = jnp.clip(
        target_progress / jnp.maximum(jnp.float32(config.field_half_length_m), 1.0),
        -1.0,
        1.0,
    )
    receiver_openness = jnp.clip(
        receiver_space_score[target_scalar]
        / jnp.maximum(jnp.float32(2.0 * config.pressure_distance_m), 1.0),
        0.0,
        1.0,
    )
    carrier_space = jnp.clip(
        features.nearest_opponent_distance_squared
        / jnp.maximum(jnp.float32(config.pressure_distance_m**2), 1.0),
        0.0,
        1.0,
    )
    pass_macro_value = jnp.clip(
        jnp.float32(0.68) * target_completion
        + jnp.float32(0.20) * jnp.maximum(pass_progress, 0.0)
        + jnp.float32(0.12) * receiver_openness,
        0.0,
        1.0,
    )
    pass_macro_value = jnp.clip(
        pass_macro_value + jnp.where(quick_relay, config.quick_relay_value_bonus, 0.0),
        0.0,
        1.0,
    )
    # FootballWorld의 pass-macro scale 분리를 따른다. 기본 scale은 pass cadence를
    # 낮추고, completion⁴와 최소 전진 거리로 정의한 progressive service quality가 높을
    # 때만 상한 scale로 부드럽게 복원한다. 기본값 1.0/1.0은 기존 동작과 같다.
    progressive_service_quality = (
        target_completion
        * target_completion
        * target_completion
        * target_completion
        * jnp.clip(
            target_progress
            / jnp.maximum(
                jnp.asarray(config.progressive_pass_min_m, jnp.float32), 1.0e-3
            ),
            0.0,
            1.0,
        )
    )
    pass_macro_value = pass_macro_value * (
        jnp.asarray(config.pass_macro_value_scale, jnp.float32)
        + jnp.asarray(
            config.progressive_pass_macro_scale - config.pass_macro_value_scale,
            jnp.float32,
        )
        * progressive_service_quality
    )
    pass_macro_value = pass_macro_value * (
        tactical_profile[:, _TACTIC_PASS_SCALE]
        + tactical_profile[:, _TACTIC_PROGRESS_BONUS] * jnp.maximum(pass_progress, 0.0)
    )
    pass_macro_value = pass_macro_value * pass_release_value_scale
    # Receiver 선정뿐 아니라 macro 경쟁에서도 안전한 박스 진입의 슛 개선을 반영한다.
    # 현재 슛이 더 좋거나 completion이 낮은 speculative service에는 보너스가 없다.
    box_entry_value = _inside_box_receiver_score_bonus(
        shot_quality[pass_actor], shot_quality[target_scalar], target_completion,
        target_in_service_zone, inside_attacking_box[pass_actor], config,
    )
    pass_macro_value = jnp.minimum(
        1.0,
        pass_macro_value + jnp.where(
            ~inside_attacking_box[pass_actor], box_entry_value * pass_release_value_scale, 0.0
        ),
    )
    direct_release = _specialist_long_service(
        player_tactical_plan[pass_actor], pass_distance[target_scalar], target_progress,
        target_completion, scene.ball_live[pass_actor] & (restart_kind[pass_actor] == 0),
    ) & (reception_advantage[target_scalar] >= 0.35) & (~is_goalkeeper[pass_actor])
    pass_macro_value = pass_macro_value * _specialist_macro_scale(
        direct_release, box_entry_value,
    )
    control_macro_value = jnp.clip(
        jnp.float32(0.45) * carrier_space
        + jnp.float32(0.35) * pitch_progress
        + jnp.float32(0.20) * scene.stamina_short,
        0.0,
        1.0,
    )
    # FootballWorld와 같이 lawful/physical shot eligibility와 전술적 pass 선호를
    # 분리한다. Shot이 실제 후보일 때만 기존 trade-off를 pass ranking에 적용한다.
    pass_macro_value = jnp.where(
        shot_ready,
        pass_macro_value * jnp.asarray(config.shot_pass_tradeoff, jnp.float32),
        pass_macro_value,
    )
    pass_macro_value, control_macro_value = _box_possession_macro_values(
        pass_macro_value,
        control_macro_value,
        shot_quality,
        shot_quality[target_scalar],
        target_completion,
        inside_attacking_box,
        target_in_service_zone,
        config,
    )
    # Macro ranking은 기존에는 completion threshold를 우회했다. 다만 전진 service와
    # 압박 탈출까지 같은 gate로 자르면 attacking-third progression이 크게 줄어든다.
    # 따라서 낮은 completion의 비압박·비전진 recycle만 CONTROL과 경쟁하지 못하게 한다.
    macro_pass_viable = (
        (target_completion >= config.pass_completion_threshold)
        | (target_progress >= config.progressive_pass_min_m)
        | pressure[pass_actor]
    ) & pass_release_ready
    macro_pass_candidate = (
        has_target
        & availability[:, INTENT_PASS]
        & macro_pass_viable
        & (pass_dwell_met | quick_relay)
        & carry_release
    )
    viable_release = shot_ready | macro_pass_candidate
    control_macro_value = control_macro_value * _release_urgency_control_scale(
        possession_episode_ticks, viable_release, config
    )
    pressure_escape_release = (
        pressure[pass_actor]
        & has_target
        & availability[:, INTENT_PASS]
        & macro_pass_viable
        & (possession_episode_ticks >= jnp.int32(config.pressure_escape_min_ticks))
        & (
            target_completion
            >= jnp.asarray(config.pressure_escape_completion_floor, jnp.float32)
        )
    )
    shot_macro_value = _shot_macro_ranking_value(
        shot_quality,
        inside_attacking_box,
        fresh_unlinked_possession,
        transition_shot_scale,
        config,
        goal_distance=goal_distance,
    )
    macro_available, macro_intent = _select_possession_macro(
        shot_ready, shot_macro_value, has_target, availability, macro_pass_viable,
        pass_dwell_met, quick_relay, carry_release, pass_macro_value,
        pressure_escape_release, control_macro_value, forced_close_shot,
    )
    cross_ready = (
        has_target
        & availability[:, INTENT_PASS]
        & (
            scene.position_team[:, 0]
            >= config.cross_min_depth_m + tactical_profile[:, _TACTIC_CROSS_DEPTH_DELTA]
        )
        & (
            jnp.abs(scene.position_team[:, 1])
            >= config.cross_min_width_m + tactical_profile[:, _TACTIC_CROSS_WIDTH_DELTA]
        )
        & target_in_service_zone
    )
    # Open-play cross는 PASS macro가 실제 경쟁에서 이긴 경우에만 launch
    # modifier로 사용한다. 그렇지 않으면 wide box 경계에서 선택된 SHOT과
    # forced-close SHOT까지 아래 최종 PASS override가 덮어쓴다.
    cross_ready = (
        cross_ready
        & (aerial_quality[target_scalar] >= 0.40)
        & scene.possessor
        & ((~shot_ready) | (macro_available & (macro_intent == INTENT_PASS)))
    )
    restart_corner = (
        scene.restart_taker & (restart_kind == RK_CORNER) & availability[:, INTENT_PASS]
    )
    cross_ready = cross_ready | (
        restart_corner & (
            (~has_target)
            | (target_in_service_zone & (
                (aerial_quality[target_scalar] >= 0.40)
                | corner_contested_route[target_scalar]
            ))
        )
    )
    clearance_ready = (
        pass_actor_mask
        & availability[:, INTENT_SHOOT]
        & pressure
        & (scene.position_team[:, 0] <= config.clearance_depth_m)
    )
    if config.clearance_requires_release_contest:
        controlled_open_play = (
            scene.possessor & (possession_episode_ticks > 0)
            & scene.ball_live & (~is_goalkeeper) & (~goalkeeper_holding)
            & (restart_kind == 0)
        )
        clearance_ready = clearance_ready & (~(controlled_open_play & release_contest_clear))
    if config.clearance_respects_safe_pass:
        safe_possession_pass = (
            scene.possessor & (possession_episode_ticks > 0)
            & (~goalkeeper_holding) & (restart_kind == 0)
            & has_target & pass_release_ready
            & (target_completion >= config.pass_completion_threshold)
            & macro_available & (macro_intent == INTENT_PASS)
        )
        clearance_ready = clearance_ready & (~safe_possession_pass)
    aerial_clearance_ready = (
        (~jnp.any(scene.possessor))
        & (ball_height >= config.aerial_intercept_height_m)
        & (features.ball_distance_squared <= jnp.float32(config.receive_distance_m**2))
        & availability[:, INTENT_SHOOT]
        & (
            pressure
            | (
                (scene.position_team[:, 0] <= -config.field_half_length_m + config.penalty_area_length_m)
                & (scene.ball_velocity_team[:, 0] < 0.0)
            )
        )
        & (scene.position_team[:, 0] <= config.clearance_depth_m)
    )
    clearance_ready = clearance_ready | aerial_clearance_ready
    control_ready = availability[:, INTENT_CONTROL]
    challenge_favourability = _challenge_success_favourability(
        scene,
        features,
        pass_actor,
        ball_control,
        max_speed,
        contact_physics,
        config,
    )
    carrier_position_for_challenge = (
        scene.position_team + scene.relative_position[observer_index, pass_actor]
    )
    own_goal_for_challenge = jnp.stack(
        (
            -jnp.full(count, config.field_half_length_m, dtype=jnp.float32),
            jnp.zeros(count, dtype=jnp.float32),
        ),
        axis=-1,
    )
    carrier_goal_distance = jnp.sqrt(
        jnp.sum((carrier_position_for_challenge - own_goal_for_challenge) ** 2, axis=-1)
        + jnp.float32(1.0e-6)
    )
    challenge_position_ready = (
        challenge_favourability
        >= jnp.asarray(config.challenge_success_favourability_floor, jnp.float32)
    ) | (
        carrier_goal_distance
        <= jnp.asarray(config.challenge_emergency_goal_distance_m, jnp.float32)
    )
    challenge_ready = (
        availability[:, INTENT_CHALLENGE]
        & (features.ball_distance_squared <= config.challenge_distance_m**2)
        & challenge_position_ready
    )
    if config.challenge_stochastic:
        # 거리 안이면 무조건 제출하던 결정론 판단은 파울을 맥락과 무관하게 만든다.
        # 관측 context로 조정한 Bernoulli로 cadence와 파울 회피를 함께 표현한다.
        challenge_key = event_random_key(policy_key, RandomEvent.POLICY_CHALLENGE)
        challenge_probability = _challenge_attempt_probability(
            scene,
            features,
            nearest_opponent_relative,
            config,
        )
        challenge_ready = challenge_ready & (
            jax.random.uniform(challenge_key, (count,)) < challenge_probability
        )

    # Restart taker와 공 소유자의 우선순위를 scalar select chain으로 고정한다.
    intent = jnp.full((count,), INTENT_MOVE, dtype=jnp.int32)
    intent = jnp.where(control_ready, INTENT_CONTROL, intent)
    intent = jnp.where(pass_ready, INTENT_PASS, intent)
    intent = jnp.where(shot_ready, INTENT_SHOOT, intent)
    intent = jnp.where(
        scene.possessor & macro_available,
        macro_intent,
        intent,
    )
    intent = jnp.where(cross_ready, INTENT_PASS, intent)
    intent = jnp.where(clearance_ready, INTENT_SHOOT, intent)
    intent = jnp.where(challenge_ready, INTENT_CHALLENGE, intent)
    restart_contact = scene.restart_taker & (
        availability[:, INTENT_PASS] | availability[:, INTENT_SHOOT]
    )
    restart_shoot = restart_contact & (
        (restart_kind == RK_PENALTY)
        | ((restart_kind == RK_KICKOFF) & (~has_target))
        | ((restart_kind == RK_FREEKICK) & (~restart_indirect) & shot_ready)
    )
    restart_intent = jnp.where(restart_shoot, INTENT_SHOOT, INTENT_PASS)
    intent = jnp.where(restart_contact, restart_intent, intent)
    intent = jnp.where(goalkeeper_claim, INTENT_CONTROL, intent)
    held_intent = jnp.where(goalkeeper_safe_distribution, INTENT_PASS, INTENT_SHOOT)
    intent = jnp.where(goalkeeper_holding, held_intent, intent)
    intent = jnp.where(scene.valid, intent, INTENT_MOVE)

    ball_direction = _unit(scene.ball_relative_position)
    ball_position = scene.ball_position_team
    ball_velocity = scene.ball_velocity_team
    near_x_boundary = (
        jnp.abs(ball_position[:, 0])
        >= config.field_half_length_m - config.boundary_control_margin_m
    )
    near_y_boundary = (
        jnp.abs(ball_position[:, 1])
        >= config.field_half_width_m - config.boundary_control_margin_m
    )
    outward_x = near_x_boundary & (ball_position[:, 0] * ball_velocity[:, 0] > 0.0)
    outward_y = near_y_boundary & (ball_position[:, 1] * ball_velocity[:, 1] > 0.0)
    boundary_control = outward_x | outward_y
    boundary_control_direction = _unit(
        jnp.stack(
            (
                jnp.where(outward_x, -jnp.sign(ball_position[:, 0]), 0.0),
                jnp.where(outward_y, -jnp.sign(ball_position[:, 1]), 0.0),
            ),
            axis=-1,
        )
    )
    field_scale = jnp.asarray(
        (config.field_half_length_m / 52.5, config.field_half_width_m / 34.0),
        dtype=jnp.float32,
    )
    one_m_x_limit = _interior_limit_m(config.field_half_length_m, 1.0)
    one_m_y_limit = _interior_limit_m(config.field_half_width_m, 1.0)
    support_x_limit = _interior_limit_m(config.field_half_length_m, 8.0)
    defensive_anchor = formation_anchor * field_scale
    lateral_anchor = defensive_anchor[:, 1]
    # 자기 anchor의 좌우 성향과 공의 좌우 성향이 같으면 +1, 반대면 -1이다. 같은
    # 라인의 두 선수도 anchor y가 다르므로 계수가 갈라진다.
    width_reference = jnp.maximum(jnp.float32(config.field_half_width_m), 1.0)
    anchor_side = jnp.tanh(lateral_anchor / width_reference)
    ball_side = jnp.tanh(scene.ball_position_team[:, 1] / width_reference)
    side_gain = (
        1.0
        + jnp.asarray(config.ball_side_follow_asymmetry, jnp.float32)
        * anchor_side
        * ball_side
    )
    defending_anchor = jnp.stack(
        (
            tactical_profile[:, _TACTIC_DEFEND_DEPTH_SCALE] * defensive_anchor[:, 0]
            + jnp.float32(config.defend_ball_follow_x) * scene.ball_position_team[:, 0],
            jnp.float32(config.defend_width_scale)
            * tactical_profile[:, _TACTIC_DEFEND_WIDTH]
            * lateral_anchor
            + jnp.float32(config.defend_ball_follow_y)
            * side_gain
            * scene.ball_position_team[:, 1],
        ),
        axis=-1,
    )
    # 공이 자기 골에 가까울수록 anchor를 공 주위로 수축시킨다. 상대 배치 순서를
    # 유지한 채 간격만 줄이므로 선수가 한 점에 겹치지 않는다. Lerp 하나라
    # pairwise 계산이 늘지 않는다.
    own_goal_offset = jnp.stack(
        (
            scene.ball_position_team[:, 0] + jnp.float32(config.field_half_length_m),
            scene.ball_position_team[:, 1],
        ),
        axis=-1,
    )
    ball_goal_distance = jnp.sqrt(
        jnp.sum(own_goal_offset * own_goal_offset, axis=-1) + 1.0e-6
    )
    threat = jnp.clip(
        1.0
        - ball_goal_distance
        / jnp.maximum(jnp.float32(config.box_threat_distance_m), 1.0e-3),
        0.0,
        1.0,
    )
    tactical_box_compression = (
        jnp.asarray(config.box_collapse_compression, jnp.float32)
        * _TACTIC_BOX_COMPRESSION[player_tactical_plan]
    )
    compression = 1.0 - threat * (1.0 - tactical_box_compression)
    defending_anchor = scene.ball_position_team + compression[:, None] * (
        defending_anchor - scene.ball_position_team
    )
    # Formation마다 최후방 숫자는 다르지만 박스 수비의 최소 책임까지 달라져서는 안 된다.
    # 각 팀 formation anchor의 x depth rank가 5보다 작은 outfield line을 고르고, own-box
    # threat에서는 이들이 carrier 한 점으로 모두 수축하지 않고 원래 lateral lane과
    # ball-side shift 사이를 지키게 한다. 같은 depth의 line은 함께 지켜 좌우 대칭을
    # 깨지 않으며, formation별 handicap 대신 같은 rank 규칙을 적용한다.
    box_outfield = scene.valid & (~is_goalkeeper)
    deeper_teammate = (
        same_team
        & box_outfield[None, :]
        & (formation_anchor[None, :, 0] < formation_anchor[:, None, 0])
    )
    defensive_depth_rank = jnp.sum(deeper_teammate, axis=-1)
    box_guard = box_outfield & (defensive_depth_rank < 5) & (threat > 0.0)
    box_lane_y = jnp.float32(0.72) * defending_anchor[:, 1] + jnp.float32(
        0.28
    ) * jnp.clip(
        scene.ball_position_team[:, 1],
        -jnp.float32(0.5 * config.penalty_area_half_width_m),
        jnp.float32(0.5 * config.penalty_area_half_width_m),
    )
    box_goal_side_x = jnp.minimum(
        defending_anchor[:, 0], scene.ball_position_team[:, 0] - jnp.float32(2.0)
    )
    box_guard_target = jnp.stack((box_goal_side_x, box_lane_y), axis=-1)
    defending_anchor = jnp.where(
        box_guard[:, None],
        defending_anchor
        + (threat * jnp.asarray(config.box_lane_guard_strength, jnp.float32))[:, None]
        * (box_guard_target - defending_anchor),
        defending_anchor,
    )
    defending_anchor = jnp.clip(
        defending_anchor,
        jnp.asarray((-one_m_x_limit, -one_m_y_limit), jnp.float32),
        jnp.asarray((one_m_x_limit, one_m_y_limit), jnp.float32),
    )
    # Team frame에서 own goal은 항상 (-half_length, 0)이다. 공–goal 선분 위의
    # bounded depth로 이동해 wide attack의 angle을 줄인다. Pairwise geometry나
    # sqrt 없이 이미 관측한 ball position만 재사용한다.
    goalkeeper_position_speed_squared = jnp.sum(
        scene.ball_velocity_team * scene.ball_velocity_team, axis=-1
    )
    goalkeeper_position_lead = (
        goalkeeper_position_speed_squared
        <= jnp.float32(config.goalkeeper_intercept_max_ball_speed_mps**2)
    ).astype(jnp.float32) * jnp.asarray(
        config.goalkeeper_position_ball_lead_s, jnp.float32
    )
    goalkeeper_reference_ball = (
        scene.ball_position_team
        + goalkeeper_position_lead[:, None] * scene.ball_velocity_team
    )
    ball_from_goal = jnp.maximum(
        goalkeeper_reference_ball[:, 0] + config.field_half_length_m,
        jnp.float32(1.0),
    )
    goalkeeper_depth_capacity = 2.0 * config.field_half_length_m
    goalkeeper_max_depth = min(_GOALKEEPER_MAX_DEPTH_M, goalkeeper_depth_capacity)
    goalkeeper_min_depth = min(_GOALKEEPER_MIN_DEPTH_M, goalkeeper_max_depth)
    goalkeeper_depth = jnp.clip(
        jnp.float32(0.12) * ball_from_goal,
        goalkeeper_min_depth,
        goalkeeper_max_depth,
    )
    goalkeeper_lateral_limit = jnp.maximum(
        jnp.float32(0.5 * config.goal_width_m - _GOALKEEPER_POST_MARGIN_M),
        jnp.float32(0.0),
    )
    goalkeeper_anchor = jnp.stack(
        (
            -config.field_half_length_m + goalkeeper_depth,
            jnp.clip(
                goalkeeper_reference_ball[:, 1] * goalkeeper_depth / ball_from_goal,
                -goalkeeper_lateral_limit,
                goalkeeper_lateral_limit,
            ),
        ),
        axis=-1,
    )
    outfield = scene.valid & (~is_goalkeeper)
    outfield_teammate = same_team & outfield[None, :]
    # Ground ball은 현재 위치가 아니라 굴러가는 경로 위의 intercept 지점을 향한다.
    # Receiver가 공보다 앞에 있으면 chaser가 되고, 공 뒤에서 뒤쫓는 선수는 밀려난다.
    aerial_ball = ball_height >= jnp.asarray(
        config.aerial_intercept_height_m, jnp.float32
    )
    ground_intercept_time, ground_intercept_point = _ground_path_intercept(
        scene.ball_position_team,
        scene.ball_velocity_team,
        scene.position_team,
        max_speed,
        config,
    )
    chase_time_squared = jnp.where(
        aerial_ball,
        aerial_intercept_time * aerial_intercept_time,
        ground_intercept_time * ground_intercept_time,
    )
    ball_chase_target_team = jnp.where(
        aerial_ball, ball_chase_target_team, ground_intercept_point
    )
    valid_last_actor = (last_contact_actor >= 0) & (last_contact_actor < count)
    safe_last_actor = jnp.clip(last_contact_actor, 0, count - 1)
    own_pass_in_flight = _own_pass_flight(
        scene,
        possession_previous_team,
        last_contact_actor,
        last_contact_intent,
        last_contact_kick_applied,
    )
    alternate_chaser = outfield_teammate & (indices[None, :] != safe_last_actor)
    has_alternate_chaser = jnp.any(alternate_chaser, axis=-1)
    chaser_candidate = jnp.where(
        (own_pass_in_flight & has_alternate_chaser)[:, None],
        alternate_chaser,
        outfield_teammate,
    )
    chaser = jnp.argmin(
        jnp.where(
            chaser_candidate,
            chase_time_squared[None, :],
            jnp.inf,
        ),
        axis=-1,
    )
    has_outfield = jnp.any(outfield_teammate, axis=-1)
    chaser = jnp.where(has_outfield, chaser, jnp.arange(count, dtype=jnp.int32))
    is_chaser = jnp.arange(count, dtype=jnp.int32) == chaser
    cover_candidates = outfield_teammate & (indices[None, :] != chaser[:, None])
    cover = jnp.argmin(
        jnp.where(
            cover_candidates,
            features.ball_distance_squared[None, :],
            jnp.inf,
        ),
        axis=-1,
    )
    has_cover = jnp.any(cover_candidates, axis=-1)
    is_cover = has_cover & (indices == cover)
    own_possession = jnp.any(same_team & scene.possessor[None, :], axis=-1)
    attacking_shape = _attack_shape_phase(
        scene, own_possession, own_pass_in_flight, restart_kind, config
    )
    opponent_possession = jnp.any(scene.opponent & scene.possessor[None, :], axis=-1)
    own_goalkeeper_recovering = jnp.any(
        same_team & is_goalkeeper[None, :] & (~scene.contact_ready[None, :]),
        axis=-1,
    )
    goalkeeper_recovery_loose_ball = (
        scene.ball_live
        & (~jnp.any(scene.possessor))
        & own_goalkeeper_recovering
        & (threat > 0.0)
    )
    defensive_box_phase = opponent_possession | goalkeeper_recovery_loose_ball
    wingback_advance = _topology_wingback_advance(
        role,
        same_team,
        lateral_anchor,
        scene.ball_position_team[:, 1],
        config,
    )
    support_anchor = jnp.stack(
        (
            jnp.clip(
                tactical_profile[:, _TACTIC_ATTACK_DEPTH_SCALE] * defensive_anchor[:, 0]
                + jnp.float32(config.attack_ball_follow_x)
                * scene.ball_position_team[:, 0]
                - lead_protection
                * jnp.asarray(
                    config.lead_protection_attack_depth_retreat_m, jnp.float32
                )
                + jnp.where(
                    role == ROLE_FULL_BACK,
                    tactical_profile[:, _TACTIC_FULLBACK_ADVANCE],
                    0.0,
                )
                + wingback_advance,
                -support_x_limit,
                support_x_limit,
            ),
            jnp.float32(config.attack_width_scale)
            * tactical_profile[:, _TACTIC_ATTACK_WIDTH]
            * defensive_anchor[:, 1]
            + jnp.float32(config.attack_ball_follow_y)
            * side_gain
            * scene.ball_position_team[:, 1],
        ),
        axis=-1,
    )
    support_anchor = support_anchor + _fluid_role_rotation_offset(
        role,
        lateral_anchor,
        scene.ball_position_team[:, 1],
        player_tactical_plan,
    )
    support_anchor = _compress_longitudinal_team_shape(
        support_anchor,
        same_team,
        outfield,
        jnp.asarray(config.attack_longitudinal_compression, jnp.float32),
    )
    defending_anchor = defending_anchor.at[:, 0].add(
        jnp.where(outfield, _TACTIC_DEFENSIVE_LINE_SHIFT[player_tactical_plan], 0.0)
    )
    defending_anchor = _compress_longitudinal_team_shape(
        defending_anchor,
        same_team,
        outfield,
        jnp.asarray(config.defend_longitudinal_compression, jnp.float32),
    )
    final_third_target, final_third_active = _final_third_shape_target(
        scene,
        role,
        support_anchor,
        is_goalkeeper,
        attacking_shape,
        config,
    )
    support_anchor = jnp.where(
        final_third_active[:, None], final_third_target, support_anchor
    )
    support_anchor = _total_flank_exchange(
        support_anchor, scene, role, defensive_anchor, is_goalkeeper, attacking_shape,
        player_tactical_plan, same_team, config, pressure,
    )
    support_anchor = _zona_cutback_support_target(
        scene, role, same_team, support_anchor, attacking_shape,
        player_tactical_plan, config,
    )
    live_target = jnp.where(attacking_shape[:, None], support_anchor, defending_anchor)
    planned_receiver = (
        own_possession
        & has_target_scalar
        & (indices == target_scalar)
        & (~scene.possessor)
        & (~prospective_offside)
    )
    receiver_run_target = (
        scene.position_team
        + jnp.asarray(config.planned_receiver_velocity_lead_s, jnp.float32)
        * scene.velocity_team
        + jnp.stack(
            (
                jnp.full(
                    count, config.planned_receiver_forward_lead_m, dtype=jnp.float32
                ),
                jnp.zeros(count, dtype=jnp.float32),
            ),
            axis=-1,
        )
    )
    receiver_run_target = jnp.stack(
        (
            jnp.clip(
                receiver_run_target[:, 0],
                -one_m_x_limit,
                one_m_x_limit,
            ),
            jnp.clip(
                receiver_run_target[:, 1],
                -one_m_y_limit,
                one_m_y_limit,
            ),
        ),
        axis=-1,
    )
    if config.offside_shoulder_enabled:
        receiver_onside_x = jnp.maximum(
            _onside_limit_x(scene) - jnp.float32(config.offside_shoulder_margin_m),
            0.0,
        )
        receiver_run_target = receiver_run_target.at[:, 0].set(
            jnp.where(
                (restart_kind == 0) & (~is_goalkeeper),
                jnp.minimum(receiver_run_target[:, 0], receiver_onside_x),
                receiver_run_target[:, 0],
            )
        )
    live_target = jnp.where(planned_receiver[:, None], receiver_run_target, live_target)
    cover_distance = jnp.where(
        counterpress_active,
        jnp.asarray(config.counterpress_cover_distance_m, dtype=jnp.float32)
        * tactical_profile[:, _TACTIC_COUNTERPRESS_COVER],
        jnp.asarray(config.pressure_cover_distance_m, dtype=jnp.float32),
    )
    cover_target = scene.ball_position_team + jnp.stack(
        (-cover_distance, jnp.zeros_like(cover_distance)), axis=-1
    )
    cover_target = (
        cover_target
        + jnp.asarray(config.pressure_cover_ball_lead_s, dtype=jnp.float32)
        * scene.ball_velocity_team
    )
    cover_target = jnp.where(
        (jnp.broadcast_to(aerial_ball, (count,)) & (~jnp.any(scene.possessor)))[:, None],
        aerial_contest_target + jnp.stack((-cover_distance, jnp.zeros_like(cover_distance)), axis=-1),
        cover_target,
    )
    live_target = jnp.where(
        ((~own_possession) & is_cover)[:, None], cover_target, live_target
    )
    nearest_opponent_position = scene.position_team + nearest_opponent_relative
    nearest_opponent_velocity = (
        scene.relative_velocity[observer_index, nearest_opponent] + scene.velocity_team
    )
    # 여러 defender가 같은 opponent를 향해 겹치지 않도록, 각 observer가 선택한
    # opponent별로 같은 team에서 가장 가까운 primary marker 한 명만 허용한다.
    # 기존 [N,N] distance matrix를 gather/reduction할 뿐 새 geometry는 만들지 않는다.
    distance_to_selected_opponent = scene.distance_squared[:, nearest_opponent].T
    primary_marker = jnp.argmin(
        jnp.where(
            same_team & scene.valid[None, :],
            distance_to_selected_opponent,
            jnp.inf,
        ),
        axis=-1,
    )
    is_primary_marker = indices == primary_marker
    predicted_marking_position = (
        nearest_opponent_position
        + jnp.asarray(config.marking_lead_s, jnp.float32) * nearest_opponent_velocity
    )
    marking_target = predicted_marking_position + jnp.stack(
        (
            -jnp.full(count, config.marking_goal_side_m, dtype=jnp.float32),
            jnp.zeros(count, dtype=jnp.float32),
        ),
        axis=-1,
    )
    marking = (
        opponent_possession
        & outfield
        & (~is_chaser)
        & (~is_cover)
        & is_primary_marker
        & (
            features.nearest_opponent_distance_squared
            <= (
                jnp.asarray(config.marking_activation_distance_m, jnp.float32)
                * _TACTIC_MARKING_RADIUS_SCALE[player_tactical_plan]
            )
            ** 2
        )
    )
    zona_eligible = (
        opponent_possession & outfield & (~is_chaser) & (~is_cover)
        & (player_tactical_plan == int(RuleTacticalPlan.ZONA_MISTA))
    )
    zona_target, zona_assigned = _zona_marking_target(
        scene, defending_anchor, zona_eligible, is_goalkeeper, config,
    )
    marking_target = jnp.where(zona_eligible[:, None], zona_target, marking_target)
    marking = jnp.where(zona_eligible, zona_assigned, marking)
    live_target = jnp.where(marking[:, None], marking_target, live_target)
    live_target = _specialist_defensive_target(
        live_target, defending_anchor, scene.ball_position_team, role, player_tactical_plan,
        (counterpress_active | _fresh_receiver_pressure(
            scene, player_tactical_plan, restart_kind, possession_episode_ticks, config,
        )) & ~marking,
        opponent_possession & outfield & (~is_chaser) & (~is_cover),
        position=scene.position_team,
    )
    # 일반 marking의 12 m activation gate는 own-box에서 뒤쪽 runner를 너무 일찍 놓는다.
    # 이미 계산한 primary marker를 재사용해 goal-mouth threat zone에서는 끝까지 추적한다.
    ranked_opponent_position = nearest_opponent_position
    box_runner_marking = _box_runner_tracking_mask(
        ranked_opponent_position,
        defensive_box_phase,
        box_guard,
        is_chaser,
        is_cover,
        is_primary_marker,
        config,
    )
    own_goal_centre = jnp.stack(
        (
            -jnp.full(count, config.field_half_length_m, dtype=jnp.float32),
            jnp.zeros(count, dtype=jnp.float32),
        ),
        axis=-1,
    )
    # Goal-side는 단순히 team-frame -x가 아니다. Wide runner에서 -x만 적용하면
    # marker가 runner와 같은 y에 남아 goal-centre로 향하는 대각선 shot lane을 비운다.
    box_runner_goalward = _unit(own_goal_centre - ranked_opponent_position)
    box_runner_target = ranked_opponent_position + (
        jnp.asarray(config.marking_goal_side_m, jnp.float32) * box_runner_goalward
    )
    runner_strength = jnp.asarray(config.box_runner_tracking_strength, jnp.float32)
    box_runner_target = live_target + runner_strength * (
        box_runner_target - live_target
    )
    live_target = jnp.where(box_runner_marking[:, None], box_runner_target, live_target)
    chase_due = is_chaser & ((~jnp.any(scene.possessor)) | opponent_possession)
    live_target = jnp.where(chase_due[:, None], ball_chase_target_team, live_target)
    # scene.challenge_ready는 환경의 challenge recovery lock을 그대로 노출한다. 회복
    # 중에 공으로 달려들면 CHALLENGE가 거부돼 압박이 사라지므로 containment로 바꾼다.
    retry_press = (
        opponent_possession
        & outfield
        & (~scene.challenge_ready)
        & (
            features.ball_distance_squared
            <= jnp.asarray(config.challenge_contain_distance_m, jnp.float32) ** 2
        )
    )
    contain = (
        opponent_possession
        & outfield
        & (
            (~scene.challenge_ready)
            | (availability[:, INTENT_CHALLENGE] & (~challenge_ready))
        )
        & (
            features.ball_distance_squared
            <= jnp.asarray(config.challenge_contain_distance_m, jnp.float32) ** 2
        )
    )
    carrier_position = (
        scene.position_team + scene.relative_position[observer_index, pass_actor]
    )
    carrier_velocity = (
        scene.velocity_team + scene.relative_velocity[observer_index, pass_actor]
    )
    contain_reference = jnp.where(
        retry_press[:, None],
        carrier_position
        + jnp.asarray(config.challenge_retry_lead_s, jnp.float32) * carrier_velocity,
        carrier_position,
    )
    # Declined/locked challenge도 carrier의 뒤쪽 x축이 아니라 carrier-own-goal
    # 선분 위를 지켜야 wide-angle shot lane을 실제로 닫는다.
    contain_goalward = _unit(own_goal_centre - contain_reference)
    contain_target = contain_reference + (
        jnp.asarray(config.challenge_contain_standoff_m, jnp.float32) * contain_goalward
    )
    live_target = jnp.where(contain[:, None], contain_target, live_target)
    live_target = jnp.where(
        scene.possessor[:, None],
        jnp.stack(
            (
                jnp.full((count,), config.field_half_length_m, dtype=jnp.float32),
                jnp.zeros((count,), dtype=jnp.float32),
            ),
            axis=-1,
        ),
        live_target,
    )
    if config.dribble_space_radius_m > 0.0:
        # SharedScene의 기존 [N, N] 거리·상대 위치를 그대로 줄인다. 후보 방향 K축이나
        # 새 pairwise tensor를 만들지 않으면서 한 명의 좌우 위치만 보던 결정을 근처
        # 수비수 전체가 덜 밀집한 방향으로 바꾼다.
        opponent_distance = jnp.sqrt(scene.distance_squared + jnp.float32(1.0e-6))
        space_weight = jnp.where(
            scene.opponent,
            jnp.square(
                jnp.clip(
                    1.0
                    - opponent_distance
                    / jnp.asarray(config.dribble_space_radius_m, jnp.float32),
                    0.0,
                    1.0,
                )
            ),
            0.0,
        )
        opponent_direction = scene.relative_position / jnp.maximum(
            opponent_distance[..., None], jnp.float32(1.0e-3)
        )
        dribble_evasion = -jnp.sum(
            space_weight[..., None] * opponent_direction, axis=1
        ) / jnp.maximum(jnp.sum(space_weight, axis=1, keepdims=True), 1.0)
        dribble_evasion = dribble_evasion * jnp.asarray(
            config.dribble_evasion_gain, dtype=jnp.float32
        )
    else:
        dribble_evasion = jnp.stack(
            (
                jnp.zeros(count, dtype=jnp.float32),
                -jnp.sign(nearest_opponent_relative[:, 1])
                * jnp.asarray(config.dribble_evasion_gain, dtype=jnp.float32),
            ),
            axis=-1,
        )
    dribble_vector = _unit(goal_centre_vector) + dribble_evasion
    # 공간 회피의 longitudinal 성분이 전진 벡터를 일부 상쇄해도 CONTROL power 자체가
    # 줄어들지는 않아야 한다. 기존 lateral-only 경로는 norm >= 1이라 결과가 동일하다.
    dribble_direction = dribble_vector / jnp.sqrt(
        jnp.sum(dribble_vector * dribble_vector, axis=-1, keepdims=True) + 1.0e-6
    )
    dribble_target = scene.position_team + dribble_direction * jnp.float32(6.0)
    live_target = jnp.where(
        (scene.possessor & pressure)[:, None], dribble_target, live_target
    )
    live_target = jnp.where(is_goalkeeper[:, None], goalkeeper_anchor, live_target)
    # Loose ball이 own goal mouth를 향해 굴러오면 goalkeeper는 공–goal centre 선분의
    # anchor 대신 관측된 ball ray 위의 closest-approach 지점으로 이동한다. Backpass는
    # hand claim이 금지되므로 foot control 위치를 미리 잡아야 하고, 상대 슛은 슛 line
    # 위에서 막아야 한다. Ball position·velocity만 재사용하는 O(1) 계산이다.
    ball_speed_squared = jnp.sum(ball_velocity * ball_velocity, axis=-1)
    ball_toward_own_goal = ball_velocity[:, 0] < -1.0e-3
    ball_goal_line_time = (
        -jnp.asarray(config.field_half_length_m, jnp.float32) - ball_position[:, 0]
    ) / jnp.where(ball_toward_own_goal, ball_velocity[:, 0], -1.0)
    ball_goal_line_y = ball_position[:, 1] + ball_goal_line_time * ball_velocity[:, 1]
    ball_into_own_mouth = ball_toward_own_goal & (
        jnp.abs(ball_goal_line_y)
        <= jnp.asarray(
            0.5 * config.goal_width_m + config.own_goal_hazard_margin_m, jnp.float32
        )
    )
    intercept_delta = ball_position - scene.position_team
    intercept_time = jnp.clip(
        -jnp.sum(intercept_delta * ball_velocity, axis=-1)
        / jnp.maximum(ball_speed_squared, jnp.float32(1.0e-6)),
        0.0,
        jnp.asarray(config.goalkeeper_intercept_horizon_s, jnp.float32),
    )
    intercept_point = ball_position + intercept_time[:, None] * ball_velocity
    # Goalkeeper는 goal line 뒤로 물러나 공을 받지 않는다. Ray 위 지점이 goal line을
    # 넘으면 line 앞 1 m 안쪽으로 clip한다.
    intercept_point = intercept_point.at[:, 0].set(
        jnp.maximum(
            intercept_point[:, 0],
            -jnp.float32(_interior_limit_m(config.field_half_length_m, 1.0)),
        )
    )
    intercept_offset = intercept_point - scene.position_team
    # 느리게 굴러오는 공에만 적용한다. 빠른 슛까지 ray 위로 따라가면 goalkeeper가 매번
    # shot line을 선점해 선방이 비현실적으로 많아진다. 실제 슛 저지는 hand claim과
    # contact geometry가 담당한다.
    goalkeeper_intercept = (
        is_goalkeeper
        & scene.ball_live
        & (~jnp.any(scene.possessor))
        & ball_into_own_mouth
        & (
            ball_speed_squared
            <= jnp.float32(config.goalkeeper_intercept_max_ball_speed_mps**2)
        )
        & (
            jnp.sum(intercept_offset * intercept_offset, axis=-1)
            <= jnp.float32(config.goalkeeper_intercept_radius_m**2)
        )
    )
    live_target = jnp.where(goalkeeper_intercept[:, None], intercept_point, live_target)
    live_target = jnp.where(
        (goalkeeper_claim | goalkeeper_holding)[:, None],
        scene.ball_position_team,
        live_target,
    )
    restart_present = jnp.any(scene.restart_taker)
    restart_owner_team = jnp.sum(
        jnp.where(scene.restart_taker, scene.team_id, jnp.int32(0)),
        dtype=jnp.int32,
    )
    own_restart = restart_present & (scene.team_id == restart_owner_team)
    dead_ball_target = jnp.where(own_restart[:, None], support_anchor, defending_anchor)
    restart_shape_target, restart_shape_valid = _restart_shape_target(
        scene,
        formation_anchor,
        is_goalkeeper,
        restart_kind,
        own_restart,
        config,
    )
    dead_ball_target = jnp.where(
        restart_shape_valid[:, None], restart_shape_target, dead_ball_target
    )
    # Restart 준비 중 taker까지 team shape target으로 보내면 공에서 멀어진 뒤 contact
    # availability를 영구히 잃을 수 있다. Taker는 layout이 완성될 때까지 공 위치를
    # 지켜야 하며, 나머지 선수만 restart shape를 만든다.
    dead_ball_target = jnp.where(
        scene.restart_taker[:, None], scene.ball_position_team, dead_ball_target
    )
    dead_ball_from_ball = dead_ball_target - scene.ball_position_team
    dead_ball_distance_squared = jnp.sum(
        dead_ball_from_ball * dead_ball_from_ball, axis=-1
    )
    restart_clearance = jnp.where(
        restart_kind == RK_THROWIN, jnp.float32(2.0), jnp.float32(9.15)
    )
    opponent_restart = (~own_restart) & restart_present
    dead_ball_inverse_distance = jax.lax.rsqrt(
        jnp.maximum(dead_ball_distance_squared, jnp.float32(1.0e-6))
    )
    dead_ball_target = jnp.where(
        (
            opponent_restart
            & (dead_ball_distance_squared < restart_clearance * restart_clearance)
        )[:, None],
        scene.ball_position_team
        + restart_clearance[:, None]
        * dead_ball_from_ball
        * dead_ball_inverse_distance[:, None],
        dead_ball_target,
    )
    anchor_direction = _unit(dead_ball_target - scene.position_team)
    live_direction = _unit(live_target - scene.position_team)
    move = jnp.where(scene.ball_live[:, None], live_direction, anchor_direction)
    # 공을 쫓지도 들고 있지도 않은 선수만 역할·stamina 파워로 줄인다. Chaser와
    # possessor는 도착 시점이 곧 경기 진행이므로 건드리지 않는다. Cover는 저 stamina
    # 상태에서도 별도 최소 파워를 보장한다.
    off_ball = ~(scene.possessor | chase_due | retry_press)
    role_power = jnp.asarray(config.role_offball_power, jnp.float32)[
        jnp.clip(role.astype(jnp.int32), 0, 6)
    ]
    fatigue_power = _stamina_offball_power(role, scene.stamina_long, is_cover, config)
    move_power = role_power * fatigue_power
    move = jnp.where(off_ball[:, None], move * move_power[:, None], move)
    force = jnp.where((intent == INTENT_PASS)[:, None], pass_direction, ball_direction)
    loose_control = (intent == INTENT_CONTROL) & (~scene.possessor)
    ground_loose_control = loose_control & (
        ball_height < jnp.asarray(config.aerial_intercept_height_m, jnp.float32)
    )
    force = jnp.where(
        ground_loose_control[:, None],
        -ball_direction * jnp.asarray(config.dribble_control_power, dtype=jnp.float32),
        force,
    )
    controlled_dribble = scene.possessor & (intent == INTENT_CONTROL)
    force = jnp.where(
        controlled_dribble[:, None],
        dribble_direction
        * jnp.asarray(config.dribble_control_power, dtype=jnp.float32),
        force,
    )
    reception_shield = (
        (player_tactical_plan == int(RuleTacticalPlan.CATENACCIO))
        & loose_control & scene.ball_live & (restart_kind == 0)
        & (~is_goalkeeper) & pressure & (~boundary_control) & (~cross_ready)
        & (ball_height >= jnp.asarray(config.aerial_intercept_height_m, jnp.float32))
        & (features.nearest_opponent_distance_squared > jnp.float32(1.e-6))
        & (features.ball_distance_squared <= jnp.float32(
            (contact_physics.carry_radius_m + config.ball_radius_m) ** 2
        ))
        & (ball_position[:, 0] > -config.field_half_length_m
           + config.boundary_control_margin_m + config.own_goal_hazard_overrun_m)
    )
    shield_direction = -_unit(nearest_opponent_relative)
    shield_force_power = jnp.sqrt(jnp.sum(force * force, axis=-1))
    shield_move_power = jnp.sqrt(jnp.sum(move * move, axis=-1))
    force = jnp.where(
        reception_shield[:, None], shield_direction * shield_force_power[:, None], force,
    )
    move = jnp.where(
        reception_shield[:, None], shield_direction * shield_move_power[:, None], move,
    )
    force = jnp.where(
        ((intent == INTENT_CONTROL) & boundary_control)[:, None],
        boundary_control_direction
        * jnp.asarray(config.boundary_control_power, dtype=jnp.float32),
        force,
    )
    # CONTROL은 incoming ball이 아직 goal을 향하지 않아도 requested relative
    # velocity가 성공하면서 own-goal ray로 바뀔 수 있다. 특히 높은 loose ball에서
    # 기본 force가 player->ball 방향인 경우 goal line 인근의 한 touch가 그대로
    # 자책골이 된다. Contact response와 같은 bounded-impulse 계산의 planar projection을
    # 사용해 실제 release 방향을 미리 검사한다.
    requested_control_velocity = scene.velocity_team + force * jnp.asarray(
        contact_physics.control_request_speed_max_mps, jnp.float32
    )
    control_required_impulse = requested_control_velocity - ball_velocity
    control_required_speed = jnp.sqrt(
        jnp.sum(control_required_impulse * control_required_impulse, axis=-1)
        + jnp.float32(1.0e-6)
    )
    control_completion = jnp.minimum(
        1.0,
        jnp.asarray(contact_physics.kick_speed_max_mps, jnp.float32)
        / control_required_speed,
    )
    expected_control_velocity = (
        ball_velocity + control_completion[:, None] * control_required_impulse
    )
    expected_control_speed = jnp.sqrt(
        jnp.sum(expected_control_velocity * expected_control_velocity, axis=-1)
        + jnp.float32(1.0e-6)
    )
    expected_control_direction = (
        expected_control_velocity / expected_control_speed[:, None]
    )
    control_toward_own_goal = expected_control_direction[:, 0] < -1.0e-3
    control_goal_line_distance = (
        -jnp.asarray(config.field_half_length_m, jnp.float32) - ball_position[:, 0]
    ) / jnp.where(control_toward_own_goal, expected_control_direction[:, 0], -1.0)
    control_goal_line_y = (
        ball_position[:, 1]
        + control_goal_line_distance * expected_control_direction[:, 1]
    )
    control_request_into_own_mouth = (
        control_toward_own_goal
        & (control_goal_line_distance >= 0.0)
        & (
            control_goal_line_distance
            <= jnp.asarray(config.own_goal_hazard_overrun_m, jnp.float32)
        )
        & (
            jnp.abs(control_goal_line_y)
            <= jnp.asarray(
                0.5 * config.goal_width_m + config.own_goal_hazard_margin_m,
                jnp.float32,
            )
        )
    )
    # Goal line에서 2 m 이상 떨어져 있어도 현재 ball ray가 own goal mouth를 향하면
    # 일반 settling touch(-ball_direction)는 공의 진행 방향을 그대로 goal 쪽으로
    # 밀 수 있다. FootballWorld의 goal-line control risk와 같은 우선순위로 team-frame
    # 전방(+x), 가까운 touchline 바깥쪽을 향하게 하고 boundary control power를 쓴다.
    # Header response의 방향 전환 능력과 무관하게 policy 요청 자체가 항상 안전해야 한다.
    # 이 예외는 full request power를 쓴다. Goal line을 향해 7 m/s로 움직이는 수비수에게
    # boundary_control_power=0.45만 주면 player velocity + 4.05 m/s가 여전히 own goal
    # 방향이라, outward direction을 요청해도 CONTROL response가 자책골을 만들 수 있다.
    own_goal_control_vector = jnp.stack(
        (
            jnp.ones(count, dtype=jnp.float32),
            jnp.float32(0.75) * jnp.sign(ball_position[:, 1]),
        ),
        axis=-1,
    )
    # 골문 밖으로 향하는 header도 own goal-line 바로 앞에서는 반복 corner를 만든다.
    # Header의 요청 방향과 ground response의 player momentum을 모두 검사한다.
    own_goal_line_touch = (
        ball_position[:, 0] <= -config.field_half_length_m + config.boundary_control_margin_m
    ) & ((force[:, 0] < 0.0) | (expected_control_velocity[:, 0] < 0.0))
    own_goal_threat_touch = (loose_control | (intent == INTENT_CHALLENGE)) & (
        ball_into_own_mouth | control_request_into_own_mouth | own_goal_line_touch
    )
    force = jnp.where(
        own_goal_threat_touch[:, None],
        _unit(own_goal_control_vector),
        force,
    )
    # Goalkeeper hand claim의 force는 catch 실패 시 parry 방향이 된다. Loose-ball
    # settling touch(자기 쪽, 낮은 power)를 그대로 쓰면 빠른 슛이 goalkeeper를 지나
    # own goal로 굴러가므로, own goal에서 멀어지는 전방·가까운 touchline 쪽으로 전체
    # power를 요청한다. Team frame의 +x는 항상 own goal 반대 방향이다.
    parry_vector = jnp.stack(
        (
            jnp.ones(count, dtype=jnp.float32),
            jnp.float32(0.75) * jnp.where(ball_position[:, 1] >= 0.0, 1.0, -1.0),
        ),
        axis=-1,
    )
    force = jnp.where(goalkeeper_claim[:, None], _unit(parry_vector), force)
    clearance_vector = jnp.stack(
        (
            jnp.ones(count, dtype=jnp.float32),
            jnp.where(scene.position_team[:, 1] >= 0.0, -0.35, 0.35),
        ),
        axis=-1,
    )
    # GK long fallback도 바로 앞 상대 쪽으로 고정 대각선 clearance를 보내지 않는다.
    # 상대의 lateral side가 명확하면 반대쪽을 고르고, 정면이면 기존 touchline fallback을
    # 사용한다. 일반 outfield clearance 방향은 그대로 둔다.
    nearest_opponent_lateral = nearest_opponent_relative[:, 1]
    goalkeeper_clearance_side = jnp.where(
        jnp.abs(nearest_opponent_lateral) > jnp.float32(0.15),
        -jnp.sign(nearest_opponent_lateral),
        jnp.where(scene.position_team[:, 1] >= 0.0, -1.0, 1.0),
    )
    goalkeeper_clearance_vector = jnp.stack(
        (
            jnp.ones(count, dtype=jnp.float32),
            jnp.float32(0.75) * goalkeeper_clearance_side,
        ),
        axis=-1,
    )
    shoot_direction = jnp.where(
        (clearance_ready | goalkeeper_long_distribution)[:, None],
        _unit(clearance_vector),
        _shot_request_direction(goal_vector, scene.ball_velocity_team, goal_distance, config, contact_physics,
                                compensation_eligible=shot_compensation_eligible),
    )
    force = jnp.where((intent == INTENT_SHOOT)[:, None], shoot_direction, force)
    ordinary_close_shot = (
        (intent == INTENT_SHOOT)
        & (goal_distance <= jnp.float32(config.close_shot_distance_m))
        & (~clearance_ready)
        & (~goalkeeper_long_distribution)
    )
    force = jnp.where(
        ordinary_close_shot[:, None],
        force * jnp.float32(config.close_shot_power),
        force,
    )
    held_direction = jnp.where(
        goalkeeper_safe_distribution[:, None],
        pass_direction,
        _unit(goalkeeper_clearance_vector),
    )
    force = jnp.where(goalkeeper_holding[:, None], held_direction, force)
    clearance_power = _lofted_release_power(
        config.clearance_carry_m, config.clearance_launch, config, contact_physics
    )
    distribution_power = _lofted_release_power(
        config.goalkeeper_distribution_carry_m,
        config.goalkeeper_distribution_launch,
        config,
        contact_physics,
    )
    force = jnp.where(clearance_ready[:, None], force * clearance_power, force)
    force = jnp.where(
        goalkeeper_long_distribution[:, None], force * distribution_power, force
    )
    cross_power = jnp.maximum(
        _lofted_release_power(
            pass_distance[target_scalar], config.cross_launch, config, contact_physics
        ),
        jnp.asarray(config.cross_min_power, jnp.float32),
    )
    long_service = _specialist_long_service(
        player_tactical_plan, pass_distance[target_scalar], target_progress,
        target_completion, scene.ball_live & (restart_kind == 0),
    ) & has_target & (~is_goalkeeper) & (~cross_ready) & (aerial_quality[target_scalar] >= 0.40)
    long_service_power = _lofted_release_power(
        pass_distance[target_scalar], 0.10, config, contact_physics
    )
    release_power = jnp.where(
        long_service, long_service_power, jnp.where(cross_ready, cross_power, pass_power)
    )
    force = jnp.where(
        (intent == INTENT_PASS)[:, None], force * release_power[:, None], force
    )
    restart_without_receiver = restart_contact & (intent == INTENT_PASS) & (~has_target)
    fallback_x = jnp.where(restart_kind == RK_CORNER, -1.0, 1.0)
    fallback_y = -jnp.sign(scene.position_team[:, 1]) * jnp.float32(0.5)
    fallback_vector = jnp.stack((fallback_x, fallback_y), axis=-1)
    fallback_direction = fallback_vector * jax.lax.rsqrt(
        jnp.sum(fallback_vector * fallback_vector, axis=-1, keepdims=True)
    )
    force = jnp.where(
        restart_without_receiver[:, None],
        fallback_direction
        * jnp.asarray(config.restart_fallback_power, dtype=jnp.float32),
        force,
    )
    launch = jnp.where(
        intent == INTENT_SHOOT,
        jnp.asarray(config.shot_launch, jnp.float32),
        jnp.float32(-0.65),
    )
    launch = jnp.where(
        clearance_ready, jnp.asarray(config.clearance_launch, jnp.float32), launch
    )
    launch = jnp.where(
        goalkeeper_long_distribution,
        jnp.asarray(config.goalkeeper_distribution_launch, jnp.float32),
        launch,
    )
    launch = jnp.where(
        cross_ready, jnp.asarray(config.cross_launch, dtype=jnp.float32), launch
    )
    launch = jnp.where(long_service & (intent == INTENT_PASS), jnp.float32(0.10), launch)
    launch = _height_compensated_launch(launch, ball_height, contact_physics)
    # 실험용 CAT 수신: 현재 수신자 row만 사용하며 HEAD는 trap이 아닌 아군 relay다.
    # 관측에 response 설정이 없으므로 기본 retention의 nominal 예측을 사용한다.
    relay_actor = jnp.argmin(jnp.where(reception_shield, features.ball_distance_squared, jnp.inf))
    relay_relative = scene.relative_position[relay_actor]
    relay_distance = jnp.sqrt(scene.distance_squared[relay_actor] + jnp.float32(1.e-6))
    relay_power = jnp.sqrt(jnp.sum(force[relay_actor] ** 2))
    relay_speed = jnp.sqrt(jnp.sum(ball_velocity[relay_actor] ** 2) + ball_vertical_velocity ** 2)
    relay_speed = relay_speed * (1.0 - relay_power * (1.0 - ContactResponsePhysics().header_speed_retention))
    relay_time = relay_distance / jnp.maximum(relay_speed, jnp.float32(1.e-3))
    relay_angle = jnp.clip(jnp.arctan2(
        height * contact_physics.pelvis_height_factor - ball_height
        + jnp.float32(0.5 * config.gravity_mps2) * relay_time ** 2,
        relay_distance,
    ), -contact_physics.launch_max_radians, contact_physics.launch_max_radians)
    relay_time = relay_distance / jnp.maximum(relay_speed * jnp.cos(relay_angle), jnp.float32(1.e-3))
    relay_contact_height = ball_height + relay_speed * jnp.sin(relay_angle) * relay_time - jnp.float32(0.5 * config.gravity_mps2) * relay_time ** 2
    relay_opponent_eta = features.receiver_space / jnp.maximum(max_speed[nearest_opponent], jnp.float32(1.e-3))
    relay_height_clear = (relay_opponent_eta > relay_time) | (reach_height >= reach_height[nearest_opponent] + jnp.float32(0.15))
    relay_legal = (
        scene.teammate[relay_actor] & scene.valid & (~is_goalkeeper) & (~prospective_offside)
        & (relay_distance <= jnp.float32(8.0))
        & (features.receiver_space >= config.pressure_distance_m)
        & (relay_contact_height >= config.ball_radius_m)
        & (relay_contact_height <= reach_height + config.ball_radius_m)
        & relay_height_clear
    )
    relay_target = jnp.argmin(jnp.where(relay_legal, relay_distance, jnp.inf))
    relay_contender_eta = jnp.sqrt(scene.distance_squared[relay_target] + jnp.float32(1.e-6)) / jnp.maximum(max_speed, jnp.float32(1.e-3))
    relay_contenders = scene.opponent[relay_target] & (relay_contender_eta <= relay_time[relay_target])
    relay_strongest_reach = jnp.max(jnp.where(relay_contenders, reach_height, jnp.float32(-100.0)))
    relay_all_opponents_clear = (~jnp.any(relay_contenders)) | (reach_height[relay_target] >= relay_strongest_reach + jnp.float32(0.15))
    relay_active = (
        jnp.any(reception_shield) & jnp.any(relay_legal)
        & relay_all_opponents_clear
        & (relay_power > 0.0) & jnp.isfinite(relay_speed) & (relay_speed > 0.0)
        & availability[relay_actor, INTENT_PASS]
        & (ball_height > height[relay_actor] * contact_physics.torso_top_height_factor + config.ball_radius_m)
        & (ball_height <= reach_height[relay_actor] + config.ball_radius_m)
        & (~own_goal_threat_touch[relay_actor])
        & (~((last_contact_actor == relay_actor) & (last_contact_intent == INTENT_PASS)
             & (~last_contact_kick_applied)
             & (jnp.sum(scene.ball_relative_position[relay_actor] * (ball_velocity[relay_actor] - scene.velocity_team[relay_actor])) > 0.0)))
    )
    relay_row = relay_active & (indices == relay_actor)
    relay_direction = _unit(relay_relative[relay_target])
    relay_height_fraction = jnp.clip((ball_height - config.ball_radius_m) / (contact_physics.ground_launch_down_reference_height_m - config.ball_radius_m), 0.0, 1.0)
    relay_floor = -(contact_physics.ground_launch_down_max_radians + (contact_physics.launch_max_radians - contact_physics.ground_launch_down_max_radians) * relay_height_fraction)
    relay_launch = 2.0 * (relay_angle[relay_target] - relay_floor) / (contact_physics.launch_max_radians - relay_floor) - 1.0
    intent = jnp.where(relay_row, INTENT_PASS, intent)
    force = jnp.where(relay_row[:, None], relay_direction * relay_power, force)
    launch = jnp.where(relay_row, relay_launch, launch)
    target = jnp.where(relay_row, relay_target, target)
    has_target = has_target | relay_row
    # 고공의 의도적 자기 header release가 멀어지는 동안 CONTROL로 재접촉하지 않는다.
    # 실제 body mechanism은 관측에 없으므로 높이 envelope를 쓰는 제한된 추정이다.
    relay_self_departing = (
        (player_tactical_plan == int(RuleTacticalPlan.CATENACCIO))
        & (intent == INTENT_CONTROL) & (~scene.possessor)
        & (last_contact_actor == indices) & (last_contact_intent == INTENT_PASS)
        & (~last_contact_kick_applied) & scene.ball_live & (restart_kind == 0)
        & (~is_goalkeeper) & (~boundary_control) & (~own_goal_threat_touch)
        & (ball_height > height * contact_physics.torso_top_height_factor + config.ball_radius_m)
        & (ball_height <= reach_height + config.ball_radius_m)
        & (jnp.sum(scene.ball_relative_position * (ball_velocity - scene.velocity_team), axis=-1) > 0.0)
    )
    intent = jnp.where(relay_self_departing, INTENT_MOVE, intent)
    move_power = jnp.sqrt(jnp.sum(move * move, axis=-1))
    force_power = jnp.sqrt(jnp.sum(force * force, axis=-1))
    action = IntentAction(
        intent=intent,
        move=linf_radial_encode(move, move_power),
        force_to_ball=linf_radial_encode(force, force_power),
        launch=launch,
        spin=jnp.zeros((count, 2), dtype=jnp.float32),
    )
    if not include_diagnostics:
        return action
    diagnostic_actor = jnp.where(relay_active, relay_actor, pass_actor)
    diagnostic_target = jnp.where(relay_active, relay_target, target_scalar)
    diagnostic_relative = jnp.where(relay_active, relay_relative[relay_target], target_relative)
    diagnostic_has_target = has_target_scalar | relay_active
    diagnostic_near = scene.distance_squared[diagnostic_actor] <= jnp.float32(8.0**2)
    diagnostic_teammates = jnp.float32(1.0) + jnp.sum(
        scene.teammate[diagnostic_actor] & diagnostic_near, dtype=jnp.float32
    )
    diagnostic_opponents = jnp.sum(
        scene.opponent[diagnostic_actor] & diagnostic_near, dtype=jnp.float32
    )
    target_cosine = diagnostic_relative[0] / jnp.maximum(
        jnp.sqrt(jnp.sum(diagnostic_relative * diagnostic_relative)), jnp.float32(1.0e-6)
    )
    target_direction_family = jnp.where(
        target_cosine <= -0.34,
        jnp.int32(-1),
        jnp.where(target_cosine < 0.34, jnp.int32(0), jnp.int32(1)),
    )
    second_pass_score = jnp.max(
        jnp.where(
            jnp.arange(count, dtype=jnp.int32) != target_scalar,
            pass_score,
            -jnp.inf,
        )
    )
    target_score_gap = jnp.where(
        has_target_scalar & jnp.isfinite(second_pass_score),
        pass_score[target_scalar] - second_pass_score,
        0.0,
    )
    next_outlet_relative = scene.relative_position[diagnostic_target]
    next_outlet_legal = scene.teammate[diagnostic_target] & (~prospective_offside)
    next_outlet_progress = jnp.clip(
        next_outlet_relative[:, 0]
        / jnp.maximum(jnp.asarray(config.field_half_length_m, jnp.float32), 1.0),
        0.0,
        1.0,
    )
    next_outlet_space = jnp.clip(
        features.receiver_space
        / jnp.maximum(jnp.asarray(2.0 * config.pressure_distance_m, jnp.float32), 1.0),
        0.0,
        1.0,
    )
    target_next_outlet_quality = jnp.max(
        jnp.where(
            next_outlet_legal,
            jnp.float32(0.7) * next_outlet_progress
            + jnp.float32(0.3) * next_outlet_space,
            0.0,
        )
    )
    diagnostics = RulePolicyDiagnostics(
        intent=intent,
        target_player=jnp.where(has_target, target, -1),
        pressure=pressure,
        ball_distance=jnp.sqrt(features.ball_distance_squared + 1.0e-6),
        availability=availability,
        prospective_offside=prospective_offside,
        goalkeeper_claim=goalkeeper_claim,
        goalkeeper_holding=goalkeeper_holding,
        pass_completion=jnp.where(relay_row, jnp.float32(0.0), jnp.where(has_target, target_completion, 0.0)),
        cross=cross_ready,
        clearance=clearance_ready,
        counterpress=counterpress_active,
        carrier_valid=pass_actor_valid | relay_active,
        carrier=jnp.where(relay_active, relay_actor, jnp.where(pass_actor_valid, pass_actor, -1)).astype(jnp.int32),
        carrier_local_teammates=jnp.where(pass_actor_valid | relay_active, diagnostic_teammates, 0.0),
        carrier_local_opponents=jnp.where(pass_actor_valid | relay_active, diagnostic_opponents, 0.0),
        target_direction_family=jnp.where(
            diagnostic_has_target, target_direction_family, jnp.int32(2)
        ),
        target_score_gap=jnp.where(relay_active, jnp.float32(0.0), target_score_gap),
        target_next_outlet_quality=jnp.where(
            diagnostic_has_target, target_next_outlet_quality, 0.0
        ),
    )
    return action, diagnostics


def rule_policy(
    observation: PolicyObservation,
    *,
    config: RulePolicyConfig = DEFAULT_RULE_POLICY_CONFIG,
    ball_physics: BallPhysics = BallPhysics(),
    contact_physics: ContactPhysics = ContactPhysics(),
    tactical_plan=None,
    key=None,
) -> IntentAction:
    """환경 private state를 읽지 않는 observation-only action policy.

    `key`는 stochastic challenge 또는 shot aim noise가 켜졌을 때 필요한 frame-local
    PRNG key다. 기본 config는 stochastic challenge를 사용하므로 key가 필요하다. 정책은
    key를 상태로 보존하지 않는다.
    """
    return _policy_from_observation(
        observation,
        config,
        ball_physics=ball_physics,
        contact_physics=contact_physics,
        tactical_plan=tactical_plan,
        key=key,
        include_diagnostics=False,
    )


def rule_policy_with_diagnostics(
    observation: PolicyObservation,
    *,
    config: RulePolicyConfig = DEFAULT_RULE_POLICY_CONFIG,
    ball_physics: BallPhysics = BallPhysics(),
    contact_physics: ContactPhysics = ContactPhysics(),
    tactical_plan=None,
    key=None,
) -> tuple[IntentAction, RulePolicyDiagnostics]:
    """동일 observation에서 평가·보고서용 policy 진단도 반환한다."""
    return _policy_from_observation(
        observation,
        config,
        ball_physics=ball_physics,
        contact_physics=contact_physics,
        tactical_plan=tactical_plan,
        key=key,
        include_diagnostics=True,
    )


def _goalkeeper_policy_masks(
    state,
    roster,
    config: RulePolicyConfig,
    timebase: Timebase,
):
    count = state.players.position.shape[0]
    indices = jnp.arange(count, dtype=jnp.int32)
    hand_allowed = goalkeeper_hand_allowed_mask(
        state,
        roster,
        jnp.asarray(
            (config.field_half_length_m, config.field_half_width_m),
            dtype=state.players.position.dtype,
        ),
        penalty_area_length_m=config.penalty_area_length_m,
        penalty_area_half_width_m=config.penalty_area_half_width_m,
    )
    delta = state.ball.position[:2] - state.players.position
    relative_velocity = state.ball.velocity[:2] - state.players.velocity
    relative_speed_squared = jnp.sum(relative_velocity * relative_velocity, axis=-1)
    closest_time = jnp.clip(
        -jnp.sum(delta * relative_velocity, axis=-1)
        / jnp.maximum(relative_speed_squared, jnp.float32(1.0e-6)),
        0.0,
        jnp.asarray(config.goalkeeper_claim_horizon_s, dtype=jnp.float32),
    )
    closest_delta = delta + closest_time[:, None] * relative_velocity
    distance_squared = jnp.sum(closest_delta * closest_delta, axis=-1)
    predicted_height = (
        state.ball.position[2]
        + state.ball.velocity[2] * closest_time
        - jnp.float32(0.5 * config.gravity_mps2) * closest_time * closest_time
    )
    requested_claim = (
        state.ball.live
        & state.players.active
        & roster.is_goalkeeper
        & hand_allowed
        & (
            distance_squared
            <= (config.goalkeeper_claim_radius_m + config.ball_radius_m) ** 2
        )
        & (predicted_height <= roster.reach_height + config.ball_radius_m)
        & (predicted_height >= -config.ball_radius_m)
        & (state.players.contact_lock_substeps <= timebase.decimation)
        & (state.players.aerial_recovery_substeps <= timebase.decimation)
    )
    held, holder = held_ball_mask(state, roster)
    holding = held & (indices == holder)
    return requested_claim, holding


def observe_policy(
    state,
    roster,
    *,
    config: RulePolicyConfig = DEFAULT_RULE_POLICY_CONFIG,
    offside_geometry: OffsideGeometry = DEFAULT_OFFSIDE_GEOMETRY,
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> PolicyObservation:
    """Authoritative state를 공용 full-observable policy 입력으로 투영한다."""

    scene = observe_all(state, roster, timebase=timebase)
    goalkeeper_claim, goalkeeper_holding = _goalkeeper_policy_masks(
        state, roster, config, timebase
    )
    prospective_offside = prospective_offside_mask(
        state,
        roster,
        ball_radius_m=config.ball_radius_m,
        geometry=offside_geometry,
    )
    direct_offside_exempt_restart = (
        (state.restart.kind == RK_THROWIN)
        | (state.restart.kind == RK_GOALKICK)
        | (state.restart.kind == RK_CORNER)
    )
    prospective_offside = jnp.where(
        direct_offside_exempt_restart,
        jnp.zeros_like(prospective_offside),
        prospective_offside,
    )
    return PolicyObservation(
        scene=scene,
        player_id=roster.player_id,
        is_goalkeeper=roster.is_goalkeeper,
        max_speed=roster.max_speed,
        reach_height=roster.reach_height,
        height=roster.height,
        ball_control=roster.ball_control,
        endurance_factor=roster.endurance_factor,
        role=roster.role,
        formation_anchor=roster.formation_anchor,
        formation_id=roster.formation_id,
        prospective_offside=prospective_offside,
        goalkeeper_claim_available=goalkeeper_claim,
        goalkeeper_holding=goalkeeper_holding,
        restart_retouch_blocked=(
            state.restart_release.active
            & state.restart_release.untouched
            & (
                jnp.arange(state.players.position.shape[0], dtype=jnp.int32)
                == state.restart_release.taker
            )
        ),
        restart_kind=jnp.broadcast_to(
            state.restart.kind, state.players.position.shape[:1]
        ),
        restart_indirect=jnp.broadcast_to(
            state.restart.indirect, state.players.position.shape[:1]
        ),
        ball_height=state.ball.position[2],
        ball_vertical_velocity=state.ball.velocity[2],
        possession_team=state.possession.team,
        possession_previous_team=state.possession.previous_team,
        possession_control_ticks=state.possession.control_ticks,
        # 소유 에피소드 경과 frame 수다. 첫 frame이 1이고 소유가 없으면 0이므로
        # `control_ticks`와 같은 단위로 비교할 수 있다.
        possession_episode_ticks=jnp.where(
            (state.possession.team >= 0) & (state.possession.episode_start_tick >= 0),
            state.control_tick - state.possession.episode_start_tick + jnp.int32(1),
            jnp.int32(0),
        ).astype(jnp.int32),
        last_contact_actor=state.possession.last_contact.actor,
        last_contact_intent=state.possession.last_contact.intent,
        last_contact_kick_applied=state.possession.last_contact.kick_applied,
    )


def _policy_from_observation(
    observation: PolicyObservation,
    config: RulePolicyConfig,
    *,
    ball_physics: BallPhysics,
    contact_physics: ContactPhysics,
    tactical_plan=None,
    key=None,
    include_diagnostics: bool,
):
    if not isinstance(observation, PolicyObservation):
        raise TypeError("observation은 PolicyObservation이어야 한다")
    # Static config 정합성을 먼저 확인한 뒤 runtime 인자를 본다.
    if np.float32(config.kick_speed_max_mps) != np.float32(
        contact_physics.kick_speed_max_mps
    ):
        raise ValueError(
            "RulePolicyConfig.kick_speed_max_mps와 "
            "ContactPhysics.kick_speed_max_mps는 float32에서 같아야 한다"
        )
    stochastic_policy = (
        config.challenge_stochastic or config.shot_aim_noise_fraction > 0.0
    )
    if stochastic_policy:
        if key is None:
            raise ValueError(
                "stochastic challenge 또는 shot aim noise가 켜진 rule_policy에는 "
                "key가 필요하다"
            )
        validate_prng_key(key, name="key")
    tactical_plan = _normalize_rule_tactical_plan(tactical_plan)
    player_tactical_plan = tactical_plan[observation.scene.team_id]
    counterpress_tick_scale = _effective_tactical_profile(
        player_tactical_plan, observation.scene.stamina_long
    )[:, _TACTIC_COUNTERPRESS_TICKS]
    counterpress_active = (
        observation.scene.ball_live
        & (observation.scene.team_id == observation.possession_previous_team)
        & (observation.scene.team_id != observation.possession_team)
        & (observation.possession_team >= 0)
        & (
            observation.possession_episode_ticks
            <= jnp.asarray(config.counterpress_control_ticks, jnp.float32)
            * counterpress_tick_scale
        )
    )
    aerial_time, aerial_target, aerial_contest = _observed_aerial_reception(
        observation, config, contact_physics
    )
    return _policy_from_scene(
        observation.scene,
        observation.prospective_offside,
        observation.goalkeeper_claim_available,
        observation.goalkeeper_holding,
        observation.restart_retouch_blocked,
        config,
        ball_physics=ball_physics,
        contact_physics=contact_physics,
        is_goalkeeper=observation.is_goalkeeper,
        max_speed=observation.max_speed,
        height=observation.height,
        reach_height=observation.reach_height,
        ball_control=observation.ball_control,
        role=observation.role,
        formation_anchor=observation.formation_anchor,
        restart_kind=observation.restart_kind,
        restart_indirect=observation.restart_indirect,
        ball_chase_target_team=aerial_target,
        aerial_intercept_time=aerial_time,
        aerial_contest_target=aerial_contest,
        counterpress_active=counterpress_active,
        possession_previous_team=observation.possession_previous_team,
        last_contact_actor=observation.last_contact_actor,
        last_contact_intent=observation.last_contact_intent,
        last_contact_kick_applied=observation.last_contact_kick_applied,
        possession_control_ticks=observation.possession_control_ticks,
        possession_episode_ticks=observation.possession_episode_ticks,
        tactical_plan=tactical_plan,
        policy_key=key,
        ball_height=observation.ball_height,
        ball_vertical_velocity=observation.ball_vertical_velocity,
        include_diagnostics=include_diagnostics,
    )
