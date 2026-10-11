"""규칙 정책의 static 설정·전술 코드·고정 계수 leaf. 행동 graph는 포함하지 않는다."""

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.geometry import BallGeometry
from messi.core.numeric import (
    require_bounded_integer,
    require_float32_field_half_extent,
    require_float32_representable,
)
from messi.core.timebase import DEFAULT_TIMEBASE

# Static config의 lead-distance 산술을 검증하기 위한 수치 envelope다. 실제 ball
# velocity를 clip하거나 physics 지원 범위를 새로 정의하지 않는다.
_PRESSURE_COVER_SPEED_ENVELOPE_MPS = 60.0


class RuleTacticalPlan(IntEnum):
    """경기 시작 때 팀별로 고정하는 base와 네 rule-policy 전술 코드.

    기존 artifact의 정수 의미를 보존하기 위해 ID 1은 재사용하지 않는다.
    """

    ZONA_MISTA = 0
    GEGENPRESSING = 2
    CATENACCIO = 3
    TOTAL_FOOTBALL = 4
    DFL_BASE = 5

    # 사용 중인 입력 alias. 새 artifact에는 canonical name을 기록한다.
    BALANCED = ZONA_MISTA
    GEGENPRESS = GEGENPRESSING
    LOW_BLOCK_COUNTER = CATENACCIO
    WIDE_OVERLOAD = TOTAL_FOOTBALL


DEFAULT_RULE_TACTICAL_POOL = (
    RuleTacticalPlan.DFL_BASE,
    RuleTacticalPlan.ZONA_MISTA,
    RuleTacticalPlan.TOTAL_FOOTBALL,
    RuleTacticalPlan.GEGENPRESSING,
    RuleTacticalPlan.CATENACCIO,
)


@dataclass(frozen=True, slots=True)
class RuleTacticalPlanSpec:
    """Host-side 설명용 전술 contract. JIT hot path에는 들어가지 않는다."""

    plan: RuleTacticalPlan
    principle: str
    in_possession: str
    out_of_possession: str
    tradeoff: str
    intended_advantage_over: RuleTacticalPlan | None


RULE_TACTICAL_PLAN_SPECS = (
    RuleTacticalPlanSpec(
        RuleTacticalPlan.DFL_BASE,
        "DFL pooled reference를 따라가는 공통 rule-policy kernel",
        "전술별 delta가 없는 중립 pass·shot·shape 선택",
        "전술별 delta가 없는 중립 pressure·cover·marking",
        "상성 cycle에 참여하지 않고 네 변형의 공통 현실성 기준으로만 사용",
        None,
    ),
    RuleTacticalPlanSpec(
        RuleTacticalPlan.ZONA_MISTA,
        "formation zone을 유지하면서 가까운 primary runner만 대인 인계",
        "균형 폭과 전진·재순환 선택, 좁은 block에는 제한적 lateral switch",
        "zonal anchor와 선택적 marking의 혼합",
        "direct counter의 lane은 균형 rest-defence로 닫지만 유동적 역할 교대에는 인계가 흔들릴 수 있음",
        RuleTacticalPlan.CATENACCIO,
    ),
    RuleTacticalPlanSpec(
        RuleTacticalPlan.GEGENPRESSING,
        "소유권 상실 직후 짧고 높은 집단 압박",
        "높은 line에서 짧은 재공격",
        "긴 counterpress window, 가까운 cover, 넓은 marking activation",
        "위치 점유의 짧은 outlet은 압박하지만 깊은 block의 direct counter에는 높은 line이 노출될 수 있음",
        None,
    ),
    RuleTacticalPlanSpec(
        RuleTacticalPlan.CATENACCIO,
        "깊고 좁은 block 뒤의 직접 counter",
        "탈압박 뒤 전진 pass 가치를 크게 부여",
        "낮은 line과 threat-zone box compression",
        "높은 line 뒤 공간은 counter로 공략하지만 균형 rest-defence와 switch에는 전진 lane이 닫힐 수 있음",
        RuleTacticalPlan.GEGENPRESSING,
    ),
    RuleTacticalPlanSpec(
        RuleTacticalPlan.TOTAL_FOOTBALL,
        "역할 교대와 넓은 전개로 고정 수비 책임을 흔듦",
        "fullback/CM/wide forward rotation과 좁은 block 상대 switch",
        "공통 cover·marking을 유지하되 높은 rotation 비용을 부담",
        "혼합 zone의 인계는 교대로 흔들지만 positional rest-defence가 빈 anchor를 공략할 수 있음",
        RuleTacticalPlan.ZONA_MISTA,
    ),
)


RULE_TACTICAL_PLAN_SCHEMA = "messi.rule-tactical-plan/29"
RULE_POLICY_DFL_REFERENCE_SCHEMA = "messi.dfl-rule-policy-reference/22"
RULE_POLICY_DFL_REFERENCE_SHA256 = (
    "080c17637d33b1a11574bc55c52f3245f6e6e637894719e10299f43e656aa18f"
)


# Columns: attack width, attack depth scale, defensive width, defensive depth
# scale, pass value scale, progressive pass bonus, counterpress tick scale,
# counterpress cover-distance scale, fullback advance, cross depth delta,
# cross width delta. DFL_BASE row는 DFL-audited common kernel의 neutral이고,
# 네 변형은 같은 feature/candidate 계산에 작은 tactic delta만 더한다. ID 1은
# 제거된 전술의 자리이며 neutral row로 남겨 과거 정수 ID를 재사용하지 않는다. 이 표는
# DFL pooled average의 직접 fit이 아니라 reference 분포를 벗어나지 않게 시작하는
# design prior이며, 실제 채택은 ordered matchup과 DFL distance로 검증한다.
_TACTICAL_PROFILE = jnp.asarray(
    (
        (1.00, 0.70, 1.00, 0.70, 1.00, 0.00, 1.00, 1.00, 0.0, 0.0, 0.0),
        (1.00, 0.70, 1.00, 0.70, 1.00, 0.00, 1.00, 1.00, 0.0, 0.0, 0.0),
        (0.96, 0.74, 0.94, 0.76, 0.95, 0.06, 1.35, 0.70, 2.0, 0.0, 0.0),
        (0.92, 0.76, 0.90, 0.62, 1.02, 0.28, 0.65, 1.15, 0.0, 0.0, 0.0),
        (1.22, 0.72, 1.02, 0.70, 0.98, 0.10, 1.00, 1.00, 7.0, -7.0, -5.0),
        (1.00, 0.70, 1.00, 0.70, 1.00, 0.00, 1.00, 1.00, 0.0, 0.0, 0.0),
    ),
    dtype=jnp.float32,
)
(
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
) = range(_TACTICAL_PROFILE.shape[1])

# GEGENPRESSING의 전진·압박 강도는 장기 stamina가 떨어질수록
# DFL_BASE profile 쪽으로 완만하게 물러난다. Floor는 전술 정체성을 남기고, 다른
# 세 plan의 1.0은 기존 profile을 정확히 유지한다. 값은 추적자료 fit이 아니라 90분
# matrix에서 후반 pass 붕괴를 줄이기 위한 policy design prior다.
_TACTIC_FATIGUE_FLOOR = jnp.asarray((1.0, 1.0, 0.55, 1.0, 1.0, 1.0), jnp.float32)
_TACTIC_FATIGUE_KNEE = jnp.float32(0.25)

# MESSI의 단순 1+cover block에서는 지나치게 깊은 narrow line만으로 goal 앞 lane을
# 닫지 못한다. CATENACCIO만 threat-zone anchor 간격을 추가로 줄인다.
_TACTIC_BOX_COMPRESSION = jnp.asarray((1.0, 1.0, 1.0, 0.80, 1.0, 1.0), jnp.float32)
# 기존 nearest-opponent/primary-marker 결과를 재사용하는 activation 반경이다.
# ZONA_MISTA는 zonal anchor를 기본으로 가까운 상대만 인계하고, GEGENPRESSING은
# 더 일찍 대인 압박을 붙는다.
_TACTIC_MARKING_RADIUS_SCALE = jnp.asarray(
    (1.00, 1.00, 1.15, 1.08, 1.00, 1.00), jnp.float32
)
# TOTAL_FOOTBALL의 role interchange를 위한 O(N) anchor offset 강도다. 다른
# 전술도 같은 산술을 통과하므로 executable variant가 늘지 않는다.
_TACTIC_FLUID_ROTATION = jnp.asarray((0.0, 0.0, 0.10, 0.0, 0.62, 0.0), jnp.float32)
_TACTIC_NARROW_BLOCK_SWITCH_VALUE = jnp.asarray(
    (0.45, 0.0, 0.05, 0.0, 2.00, 0.0), jnp.float32
)

# 실측 fit이 아닌 전술 분리를 위한 bounded prior. Base 행은 정확히 0이다.
_TACTIC_SHORT_PASS_GAIN = jnp.asarray((0.25, 0.0, 0.5, 0.0, 0.6, 0.0), jnp.float32)
_TACTIC_DIRECT_PASS_GAIN = jnp.asarray((0.25, 0.0, 0.6, 5.0, 0.4, 0.0), jnp.float32)
_TACTIC_DEFENSIVE_LINE_SHIFT = jnp.asarray((0.0, 0.0, 2.0, -4.0, 0.0, 0.0), jnp.float32)


def _tactical_plan_code(value, *, name: str) -> int:
    if isinstance(value, str):
        try:
            return int(RuleTacticalPlan[value.strip().upper()])
        except KeyError as exc:
            choices = ", ".join(plan.name for plan in RuleTacticalPlan)
            raise ValueError(f"{name}은 {choices} 중 하나여야 한다") from exc
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name}은 RuleTacticalPlan, 이름 또는 정수 코드여야 한다")
    try:
        return int(RuleTacticalPlan(int(value)))
    except ValueError as exc:
        raise ValueError(
            f"{name}은 {', '.join(plan.name for plan in RuleTacticalPlan)} 중 하나여야 한다"
        ) from exc


def make_rule_tactical_plan(
    team_0: RuleTacticalPlan | str | int = RuleTacticalPlan.DFL_BASE,
    team_1: RuleTacticalPlan | str | int = RuleTacticalPlan.DFL_BASE,
) -> jax.Array:
    """검증한 두 팀의 runtime 전술 코드를 `int32[2]`로 만든다."""

    return jnp.asarray(
        (
            _tactical_plan_code(team_0, name="team_0"),
            _tactical_plan_code(team_1, name="team_1"),
        ),
        dtype=jnp.int32,
    )


DEFAULT_RULE_TACTICAL_PLAN = make_rule_tactical_plan()


def _normalize_rule_tactical_plan(tactical_plan) -> jax.Array:
    plan = (
        DEFAULT_RULE_TACTICAL_PLAN
        if tactical_plan is None
        else jnp.asarray(tactical_plan)
    )
    if plan.shape != (2,):
        raise ValueError("tactical_plan shape은 (2,)여야 한다")
    if not jnp.issubdtype(plan.dtype, jnp.integer):
        raise TypeError("tactical_plan dtype은 integer여야 한다")
    plan = plan.astype(jnp.int32)
    valid = (
        (plan == int(RuleTacticalPlan.ZONA_MISTA))
        | (plan == int(RuleTacticalPlan.GEGENPRESSING))
        | (plan == int(RuleTacticalPlan.CATENACCIO))
        | (plan == int(RuleTacticalPlan.TOTAL_FOOTBALL))
        | (plan == int(RuleTacticalPlan.DFL_BASE))
    )
    # JIT runtime 값은 Python exception으로 검사할 수 없으므로 범위 밖 코드는
    # indexing wraparound 대신 neutral DFL_BASE로 fail-closed한다.
    return jnp.where(valid, plan, jnp.int32(RuleTacticalPlan.DFL_BASE))


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class RulePolicyConfig:
    shoot_distance_m: float = 24.0
    challenge_distance_m: float = 1.35
    challenge_stochastic: bool = True
    # 210경기에서 실제 contact당 foul은 1.87%였지만 5m 미만 release가 712회였다.
    # B128×H6000 paired screening에서 0.75가 진행성을 보존하며 골을 21% 줄였다.
    challenge_attempt_probability: float = 0.75
    # 박스 안에서 challenge 자체를 막으면 슛을 방치한다. 위치 상한은 중립값으로
    # 두고, 위험한 접근에서 거절된 수비수는 goal-side containment 후 다시 시도한다.
    # Penalty 남발 억제는 ContestPhysics의 접촉 맥락 확률에서 담당한다.
    challenge_penalty_area_probability_cap: float = 1.0
    challenge_context_logit_limit: float = 2.20
    challenge_proximity_gain: float = 0.55
    challenge_opportunity_gain: float = 0.65
    challenge_foul_avoidance_gain: float = 1.00
    # Contest와 같은 controlled-closing/signed-position/reach/control 문맥이
    # 이 값 이상일 때만 일반 challenge를 제출한다. 골문 인접 emergency는
    # 별도 거리에서 우회한다.
    challenge_success_favourability_floor: float = 0.50
    challenge_emergency_goal_distance_m: float = 8.0
    receive_distance_m: float = 1.15
    possessor_control_contact_extension_enabled: bool = False
    pressure_distance_m: float = 4.0
    pass_release_lookahead_s: float = 0.1
    pass_lead_m: float = 2.0
    pass_arrival_speed_mps: float = 7.0
    cross_spin_ground_eta_prior_enabled: bool = False
    pass_lead_time_cap_s: float = 1.15
    pass_lead_distance_cap_m: float = 8.0
    planned_receiver_forward_lead_m: float = 3.0
    planned_receiver_velocity_lead_s: float = 0.35
    kick_speed_max_mps: float = 34.76
    opponent_reaction_s: float = 0.18
    opponent_intercept_speed_mps: float = 8.0
    pass_lane_softness_m: float = 0.75
    receiver_cover_speed_mps: float = 4.0
    pass_completion_gain: float = 8.0
    pass_reception_gain: float = 4.0
    pass_distance_penalty_per_m: float = 0.02
    receiver_arrival_distance_fraction: float = 0.0
    progressive_pass_value_gain: float = 0.60
    backward_pass_penalty: float = 0.25
    pass_completion_threshold: float = 0.55
    progressive_pass_min_m: float = 4.0
    clearance_depth_m: float = -24.0
    clearance_respects_safe_pass: bool = False
    clearance_requires_release_contest: bool = True
    pressure_cover_distance_m: float = 4.8
    pressure_cover_ball_lead_s: float = 0.18
    marking_activation_distance_m: float = 12.0
    # 역할별 off-ball 도착 파워. 전원이 같은 정규화 이동을 쓰면 공이 움직일 때 20명이
    # 같은 벡터로 평행 이동해 off-ball 방향의 평균 코사인 유사도가 0.370이 된다.
    # ROLE_* 순서(GK, CB, FB, CM, WM, CF, WF)이고 전부 1.0이면 기존 동작이다.
    # Role에서 [N]으로 gather하는 것뿐이라 pairwise 계산이 늘지 않는다.
    role_offball_power: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # 장기 stamina가 knee 아래로 떨어지면 비핵심 off-ball 이동만 역할별 하한까지
    # smoothstep으로 줄인다. Chaser와 possessor는 적용 대상이 아니며, 수비 cover에는
    # 별도 최소 power를 보장한다. ROLE_* 순서(GK, CB, FB, CM, WM, CF, WF)다.
    stamina_conservation_knee: float = 0.60
    role_fatigue_power_floor: tuple[float, ...] = (
        0.80,
        0.85,
        0.70,
        0.70,
        0.38,
        0.45,
        0.38,
    )
    cover_fatigue_power_floor: float = 0.90
    # 같은 라인이라도 공이 있는 쪽 선수는 더 따라가고 반대쪽 선수는 폭을 유지해야
    # 공간이 생긴다. Ball-follow 계수를 자기 anchor의 좌우 성향과 공의 좌우 성향의
    # 곱으로 조절한다. 0이면 전원이 같은 계수를 쓰는 기존 동작이다. tanh 두 번과
    # 곱 두 번으로 끝나 pairwise 계산이 늘지 않는다.
    ball_side_follow_asymmetry: float = 0.0
    # 공이 자기 골에 이 거리 안으로 들어오면 형태 유지보다 수축이 우선이다. 현재는
    # 박스 안에서도 formation anchor를 그대로 지켜 간격이 유지되고 공 주변이 빈다.
    box_threat_distance_m: float = 24.0
    # 수축 정도. 1.0이면 간격을 그대로 지키는 기존 동작이고, 작을수록 공 주위로
    # 모인다. 상대 순서는 유지하므로 한 점에 겹치지 않는다.
    box_collapse_compression: float = 0.82
    # Formation별 승률 보너스 없이 최후방 5명이 own-box에서 lateral lane을 보존하는
    # 공통 강도다. 0은 기존 anchor/chaser/cover 동작을 복원한다.
    box_lane_guard_strength: float = 0.75
    # 최후방 수비수가 공 주위 anchor만 지키다 goal-mouth runner를 놓치지 않도록,
    # own-box threat에서는 상대의 골 위협 순서에 맞춰 서로 다른 선수를 맡는다.
    box_runner_tracking_strength: float = 1.0
    # Wide midfielder/forward가 없는 5인 수비선에서는 ball-side full-back 한 명만
    # wingback outlet으로 전진시킨다. Formation ID가 아니라 role topology로 판정한다.
    wingback_advance_m: float = 5.0
    # 210경기에서 5골 차 이후 다음 득점 hazard가 오히려 증가했다. 앞선 팀만
    # 3골 차부터 5골 차까지 연속적으로 전진 숫자를 줄여 runaway lead를 완충한다.
    lead_protection_start_gap: int = 3
    lead_protection_full_gap: int = 5
    lead_protection_attack_depth_retreat_m: float = 4.0
    lead_protection_fullback_retreat_m: float = 6.0
    # 태클 회복 중에는 CHALLENGE가 거부된다. 그대로 공에 달려들면 압박을 포기하는
    # 것과 같으므로, 이 거리 안에서는 상대와 자기 골 사이에 서서 lane을 막는다.
    challenge_contain_distance_m: float = 4.0
    # Containment는 물러나는 것이 아니라 붙어서 막는 것이다. marking_target의
    # velocity lead를 쓰면 빠른 carrier에서 과도하게 앞질러 압박이 풀린다. Carrier
    # 현재 위치에서 자기 골 쪽으로 이 거리만 띄운다.
    challenge_contain_standoff_m: float = 1.1
    # 실패한 challenger는 recovery 중에도 carrier의 짧은 미래 위치를 전력 추격한다.
    challenge_retry_lead_s: float = 0.20
    marking_goal_side_m: float = 2.0
    marking_lead_s: float = 0.25
    aerial_intercept_height_m: float = 0.8
    aerial_intercept_horizon_s: float = 2.0
    ground_intercept_horizon_s: float = 3.0
    ground_intercept_velocity_factor: float = 0.75
    gravity_mps2: float = 9.81
    goal_width_m: float = 7.32
    shot_target_half_width_fraction: float = 0.70
    # 실험 후보: 전 범위 foot 슛의 고정 power를 유지하며 incoming 수평 속도를 역산한다.
    # 착지/spin과 미래 GK 운동을 해결하지 않으며 경기 개선은 별도 검증한다.
    shot_residual_velocity_compensation: bool = False
    # Eligibility/quality/power와 독립인 SHOT macro 감쇠. 1.0은 기존 산술을 보존한다.
    shot_distance_macro_floor: float = 1.0
    shot_distance_macro_near_m: float = 8.0
    shot_distance_macro_far_m: float = 24.0
    # 조준이 완벽하면 유효 슛 비율이 참조보다 높고 빗나가는 슛이 부족하다. Goal mouth
    # 반폭 기준 기본 표준편차에 거리·압박·몸 방향·피로 gain을 더한다. Nominal lane과
    # 실행 오차를 분리한 뒤 B128×H6000 paired screening에서 0.8/1.6이 release와 진행성을
    # 보존하면서 골을 25% 줄였으므로 이를 기본 폭과 상한으로 사용한다.
    # 0이면 gain도 적용하지 않는 결정론 동작이다.
    shot_aim_noise_fraction: float = 0.8
    # 가까운 무압박·정렬된 기회에서만 base 오차를 줄인다. 장거리·압박·피로 gain은
    # 그대로 남으므로 어려운 슛의 분산을 일괄 축소하지 않는다.
    shot_aim_clean_base_scale: float = 0.65
    shot_aim_distance_gain: float = 0.22
    shot_aim_pressure_gain: float = 0.25
    shot_aim_alignment_gain: float = 0.18
    shot_aim_fatigue_gain: float = 0.18
    shot_aim_noise_max_fraction: float = 1.6
    # Active outfield reach의 상한(default carry+ball=1.21m)이다.
    # GK hand와 outfield passive body 반경은 이 cap과 독립적이다.
    shot_block_radius_m: float = 1.5
    shot_quality_threshold: float = 0.50
    shot_release_contest_relief_enabled: bool = False
    # 압박에 따른 quality 감점 하한이다. Lane/접촉 legality와 aim noise는 별개로
    # 유지하며, 기본 .35는 기존 고정식과 같다. 확률이나 xG를 뜻하지 않는다.
    shot_pressure_quality_floor: float = 0.75
    # 24 m까지 일반 shot 후보를 연 뒤 penalty area 밖에만 추가 opening을 요구할 수
    # 있다. 3-seed H6000에서 0.02/0.06은 in-box 비율을 과보정해 기본값은 0이다.
    # 이미 계산한 위치와 quality에 scalar margin만 더하므로 새 pairwise geometry나
    # executable branch를 만들지 않는다.
    outside_box_shot_quality_margin: float = 0.0
    # Shot quality의 세 base 항이다. Range는 shoot distance 경계의 sigmoid, opening은
    # 두 post가 만드는 실제 angle, centrality는 penalty-area 폭을 쓴다. 최종 quality에는
    # 별도로 goalkeeper/nearest-outfield release-space factor가 곱해진다.
    shot_range_weight: float = 0.52
    shot_opening_weight: float = 0.28
    shot_centrality_weight: float = 0.20
    shot_pass_tradeoff: float = 0.75
    # 단조 감소하는 shot_quality에 임계값 하나만 두면 "가까울 때만 쏜다"만 나온다.
    # 원거리 슛은 품질이 아니라 기회 조건(열린 lane·압박 없음·중앙)으로 따로 받는다.
    # shoot_distance_m 밖의 더 먼 슛은 이 전용 기회 gate를 통과해야 한다.
    long_shot_distance_m: float = 30.0
    # 0.85는 중앙 좌우 5.1 m다. 0.75(6.8 m)로 넓히면 분포는 더 밀리지만 총 편차가 나빠졌다.
    long_shot_centrality_min: float = 0.85
    # 기존 gate는 pass 대상이 없으면 blocker를 무시하고 쏜다. 항상 lane을 본다.
    shot_requires_open_lane: bool = True
    # 1.25 후보는 paired 완주 screening에서 빈도 증가가 확인되지 않아 유지하지 않는다.
    shot_value_gain: float = 1.15
    # 슛 eligibility나 실행은 유지하고, 확립된 possessor의 SHOT/PASS/CONTROL
    # macro ranking에서만 box 밖 SHOT 값을 낮춘다. 1.0은 exact neutral, 기본은 0.90이다.
    outside_box_shot_macro_scale: float = 0.90
    # 박스 안 회복 직후에도 유효한 슛은 macro ranking에 남긴다. Pass/CONTROL은
    # receiver가 더 좋은 슛을 만드는지에 따라 별도로 조정하며 eligibility는 바꾸지 않는다.
    inside_box_transition_shot_floor: float = 0.75
    inside_box_control_macro_scale: float = 0.65
    inside_box_recycle_pass_scale: float = 0.80
    inside_box_receiver_shot_gain: float = 0.50
    # Zona의 좁은 상대 block 앞 측면 공격에서 CM 컷백 지원 강도.
    # 0.65는 격리된 양 진영 100경기에서 관측 개선을 보인 설정이다.
    zona_cutback_support_strength: float = 0.65
    shot_forced_distance_fraction: float = 0.22
    # 무소유 ball은 macro PASS/CONTROL 경쟁을 거치지 않으므로 일반 gate만 쓰면 낮은
    # 품질 rebound와 first-time release가 연속된다. 확립된 possessor의 선택은 건드리지
    # 않고, first-time contact에만 더 높은 quality를 요구한다.
    first_time_shot_quality_threshold: float = 0.58
    # 직전 deliberate release가 SHOOT인 loose ball은 일반 first-time보다 더 명확한
    # 기회에서만 다시 찬다. 시간 state를 추가하지 않고 rebound 연쇄만 좁힌다.
    rebound_shot_quality_threshold: float = 0.68
    close_shot_distance_m: float = 8.0
    close_shot_power: float = 0.82
    # 탈취·rebound 직후의 낮은 품질 슛만 macro ranking에서 잠시 할인한다. 동료의
    # 패스를 받은 경우와 quality가 이 경계 이상인 명백한 찬스는 할인하지 않는다.
    turnover_shot_settle_ticks: int = 8
    turnover_shot_value_scale: float = 0.35
    # IntentAction.launch는 signed [-1, 1]이고 decode 시 [0, 1]로 바뀐다.
    # -0.55는 ground contact에서 약 0.06 rad의 낮은 goal-directed trajectory다.
    shot_launch: float = -0.55
    boundary_control_margin_m: float = 2.0
    boundary_control_power: float = 0.45
    dribble_control_power: float = 0.06
    dribble_evasion_gain: float = 0.55
    # 0이면 기존 nearest-opponent 좌우 회피를 유지한다. 양수면 이미 계산한 [N, N]
    # 상대 geometry에서 이 반경 안의 방향을 가중 평균해 열린 공간 쪽으로 운반한다.
    dribble_space_radius_m: float = 0.0
    cross_min_depth_m: float = 25.0
    cross_min_width_m: float = 15.0
    cross_min_power: float = 0.35
    cross_launch: float = 0.25
    clearance_launch: float = 0.28
    clearance_carry_m: float = 35.0
    goalkeeper_distribution_launch: float = 0.22
    goalkeeper_distribution_carry_m: float = 45.0
    aerial_carry_factor: float = 0.80
    restart_fallback_power: float = 0.55
    # Kickoff 직후 carry envelope 가장자리에 선 선수를 receiver로 고르면 사실상
    # self-tap처럼 시작한다. Law 8 minimum이 아니라 정책 품질 제약이며, 0이면 끈다.
    kickoff_min_receiver_distance_m: float = 4.0
    counterpress_control_ticks: int = 40
    fresh_receiver_pressure_control_ticks: int = DEFAULT_TIMEBASE.control_steps_for(1.0)
    counterpress_cover_distance_m: float = 2.5
    quick_relay_control_ticks: int = 20
    quick_relay_completion_floor: float = 0.60
    quick_relay_alignment_floor: float = 0.20
    quick_relay_arrival_advantage_floor: float = 0.25
    quick_relay_value_bonus: float = 0.10
    # 새 carrier가 받은 공을 매 frame 곧바로 다시 차는 oscillation을 막는다. 압박,
    # contextual quick relay, 높은 completion의 명백한 전진 service는 즉시 release한다.
    # 0이면 이 추가 commitment gate가 닫혀 기존 동작과 정확히 같다.
    progressive_carry_min_ticks: int = 0
    progressive_carry_release_completion: float = 0.78
    progressive_carry_release_m: float = 8.0
    pass_min_episode_ticks: int = 0
    # 압박을 받은 carrier가 안전한 outlet을 두고 CONTROL만 반복하지 않도록 한다.
    # 10 Hz decision cadence에서 20 tick은 2초이며, 그 전에는 carry/evasion 경쟁을
    # 그대로 보존한다. 경과 뒤에도 실제 lane completion이 충분할 때만 CONTROL을
    # macro 후보에서 닫아 shot 또는 안전 pass로 탈출한다.
    pressure_escape_min_ticks: int = 20
    pressure_escape_completion_floor: float = 0.80
    shot_min_episode_ticks: int = 5
    pass_macro_value_scale: float = 0.50
    progressive_pass_macro_scale: float = 0.80
    # 실제 PASS/SHOT 후보가 남은 장기 점유에서만 CONTROL ranking을 낮춘다.
    # 후보가 없으면 1.0을 유지해 경계 밖 kick이나 기계적 방출을 만들지 않는다.
    release_urgency_start_ticks: int = 10
    release_urgency_full_ticks: int = 40
    release_urgency_control_floor: float = 0.75
    defensive_depth_m: float = -20.0
    attack_ball_follow_x: float = 0.35
    attack_ball_follow_y: float = 0.18
    attack_pass_flight_support_enabled: bool = True
    attack_pass_flight_min_speed_mps: float = 1.0
    defend_ball_follow_x: float = 0.28
    defend_ball_follow_y: float = 0.22
    attack_width_scale: float = 1.05
    offside_shoulder_margin_m: float = 1.0
    offside_shoulder_enabled: bool = True
    defend_width_scale: float = 0.92
    goalkeeper_claim_radius_m: float = 2.0
    goalkeeper_claim_horizon_s: float = 0.1
    goalkeeper_receiver_penalty: float = 3.0
    own_goal_hazard_margin_m: float = 2.0
    own_goal_hazard_overrun_m: float = 30.0
    own_goal_hazard_penalty: float = 6.0
    # Outfield team centroid와 y를 보존하는 x축 전용 압축이다.
    attack_longitudinal_compression: float = 0.95
    defend_longitudinal_compression: float = 0.98
    own_goal_line_hazard_penalty: float = 2.0
    goalkeeper_intercept_horizon_s: float = 1.5
    goalkeeper_intercept_radius_m: float = 3.0
    goalkeeper_intercept_max_ball_speed_mps: float = 12.0
    # 느린 carry/cross에서 goalkeeper anchor가 현재 공 위치만 뒤쫓지 않도록 짧은
    # velocity lead를 쓴다. 빠른 shot은 intercept speed gate 밖이므로 예측하지 않는다.
    goalkeeper_position_ball_lead_s: float = 0.12
    penalty_area_length_m: float = 16.5
    penalty_area_half_width_m: float = 20.16
    field_half_length_m: float = 52.5
    field_half_width_m: float = 34.0
    ball_radius_m: float = 0.11

    def __post_init__(self) -> None:
        nonnegative = (
            "shoot_distance_m",
            "challenge_distance_m",
            "challenge_context_logit_limit",
            "challenge_penalty_area_probability_cap",
            "challenge_proximity_gain",
            "challenge_opportunity_gain",
            "challenge_foul_avoidance_gain",
            "challenge_success_favourability_floor",
            "challenge_emergency_goal_distance_m",
            "receive_distance_m",
            "attack_pass_flight_min_speed_mps",
            "pressure_distance_m",
            "pass_release_lookahead_s",
            "pass_lead_m",
            "pass_arrival_speed_mps",
            "pass_lead_time_cap_s",
            "pass_lead_distance_cap_m",
            "planned_receiver_forward_lead_m",
            "planned_receiver_velocity_lead_s",
            "kick_speed_max_mps",
            "opponent_reaction_s",
            "opponent_intercept_speed_mps",
            "pass_lane_softness_m",
            "receiver_cover_speed_mps",
            "pass_completion_gain",
            "pass_reception_gain",
            "pass_distance_penalty_per_m",
            "receiver_arrival_distance_fraction",
            "progressive_pass_value_gain",
            "backward_pass_penalty",
            "pass_completion_threshold",
            "progressive_pass_min_m",
            "pressure_cover_distance_m",
            "pressure_cover_ball_lead_s",
            "marking_activation_distance_m",
            "stamina_conservation_knee",
            "cover_fatigue_power_floor",
            "ball_side_follow_asymmetry",
            "box_threat_distance_m",
            "box_collapse_compression",
            "box_lane_guard_strength",
            "box_runner_tracking_strength",
            "wingback_advance_m",
            "lead_protection_attack_depth_retreat_m",
            "lead_protection_fullback_retreat_m",
            "challenge_contain_distance_m",
            "challenge_contain_standoff_m",
            "challenge_retry_lead_s",
            "marking_goal_side_m",
            "marking_lead_s",
            "aerial_intercept_height_m",
            "aerial_intercept_horizon_s",
            "ground_intercept_horizon_s",
            "ground_intercept_velocity_factor",
            "gravity_mps2",
            "goal_width_m",
            "shot_target_half_width_fraction",
            "shot_aim_noise_fraction",
            "shot_aim_clean_base_scale",
            "shot_aim_distance_gain",
            "shot_aim_pressure_gain",
            "shot_aim_alignment_gain",
            "shot_aim_fatigue_gain",
            "shot_aim_noise_max_fraction",
            "shot_block_radius_m",
            "shot_quality_threshold",
            "shot_pressure_quality_floor",
            "outside_box_shot_quality_margin",
            "shot_range_weight",
            "shot_opening_weight",
            "shot_centrality_weight",
            "shot_pass_tradeoff",
            "long_shot_distance_m",
            "long_shot_centrality_min",
            "shot_value_gain",
            "shot_forced_distance_fraction",
            "outside_box_shot_macro_scale",
            "inside_box_transition_shot_floor",
            "inside_box_control_macro_scale",
            "inside_box_recycle_pass_scale",
            "inside_box_receiver_shot_gain",
            "zona_cutback_support_strength",
            "first_time_shot_quality_threshold",
            "rebound_shot_quality_threshold",
            "close_shot_distance_m",
            "close_shot_power",
            "turnover_shot_value_scale",
            "boundary_control_margin_m",
            "boundary_control_power",
            "dribble_control_power",
            "dribble_evasion_gain",
            "dribble_space_radius_m",
            "cross_min_depth_m",
            "cross_min_width_m",
            "cross_min_power",
            "cross_launch",
            "clearance_carry_m",
            "goalkeeper_distribution_carry_m",
            "aerial_carry_factor",
            "restart_fallback_power",
            "kickoff_min_receiver_distance_m",
            "counterpress_cover_distance_m",
            "quick_relay_completion_floor",
            "quick_relay_alignment_floor",
            "quick_relay_arrival_advantage_floor",
            "quick_relay_value_bonus",
            "pressure_escape_completion_floor",
            "release_urgency_control_floor",
            "attack_longitudinal_compression",
            "defend_longitudinal_compression",
            "progressive_carry_release_completion",
            "progressive_carry_release_m",
            "pass_macro_value_scale",
            "progressive_pass_macro_scale",
            "attack_ball_follow_x",
            "attack_ball_follow_y",
            "defend_ball_follow_x",
            "defend_ball_follow_y",
            "attack_width_scale",
            "offside_shoulder_margin_m",
            "defend_width_scale",
            "goalkeeper_claim_radius_m",
            "goalkeeper_claim_horizon_s",
            "goalkeeper_receiver_penalty",
            "own_goal_hazard_margin_m",
            "own_goal_hazard_overrun_m",
            "own_goal_hazard_penalty",
            "own_goal_line_hazard_penalty",
            "goalkeeper_intercept_horizon_s",
            "goalkeeper_intercept_radius_m",
            "goalkeeper_intercept_max_ball_speed_mps",
            "goalkeeper_position_ball_lead_s",
            "penalty_area_length_m",
            "penalty_area_half_width_m",
        )
        for name in nonnegative:
            if require_float32_representable(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        field_half_length, field_half_width = require_float32_field_half_extent(
            (self.field_half_length_m, self.field_half_width_m),
            name="rule policy field_half_extent",
        )
        object.__setattr__(self, "field_half_length_m", field_half_length)
        object.__setattr__(self, "field_half_width_m", field_half_width)
        object.__setattr__(self, "goal_width_m", float(np.float32(self.goal_width_m)))
        object.__setattr__(
            self,
            "boundary_control_margin_m",
            float(np.float32(self.boundary_control_margin_m)),
        )
        BallGeometry(radius_m=self.ball_radius_m)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            maximum_position_delta = np.float32(2.0) * np.maximum(
                np.float32(self.field_half_length_m),
                np.float32(self.field_half_width_m),
            ) + np.maximum(
                np.float32(self.pressure_cover_distance_m),
                np.float32(self.counterpress_cover_distance_m),
            )
            maximum_cover_component = maximum_position_delta + (
                np.float32(self.pressure_cover_ball_lead_s)
                * np.float32(_PRESSURE_COVER_SPEED_ENVELOPE_MPS)
            )
            maximum_cover_distance_squared = (
                np.float32(2.0) * maximum_cover_component * maximum_cover_component
            )
        if not np.isfinite(maximum_cover_distance_squared):
            raise ValueError(
                "pressure_cover_ball_lead_s의 float32 cover target 거리 제곱은 "
                "유한해야 한다"
            )
        represented_kick_speed = np.float32(self.kick_speed_max_mps)
        represented_arrival_speed = np.float32(self.pass_arrival_speed_mps)
        if represented_kick_speed < np.finfo(np.float32).tiny:
            raise ValueError(
                "kick_speed_max_mps는 float32 최소 positive normal 이상이어야 한다"
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            kick_speed_reciprocal = np.reciprocal(represented_kick_speed)
        if not np.isfinite(kick_speed_reciprocal):
            raise ValueError("kick_speed_max_mps의 float32 reciprocal은 유한해야 한다")
        if represented_arrival_speed > represented_kick_speed:
            raise ValueError(
                "pass_arrival_speed_mps는 kick_speed_max_mps 이하여야 한다"
            )
        for name in (
            "pass_arrival_speed_mps",
            "shoot_distance_m",
            "challenge_distance_m",
            "receive_distance_m",
            "pressure_distance_m",
            "shot_block_radius_m",
            "marking_activation_distance_m",
            "dribble_space_radius_m",
            "kickoff_min_receiver_distance_m",
        ):
            represented = np.float32(getattr(self, name))
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                squared = represented * represented
            if not np.isfinite(squared):
                raise ValueError(f"{name}의 float32 제곱은 유한해야 한다")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            goalkeeper_claim_extent = np.float32(
                np.float32(self.goalkeeper_claim_radius_m)
                + np.float32(self.ball_radius_m)
            )
            goalkeeper_claim_squared = goalkeeper_claim_extent * goalkeeper_claim_extent
        if not np.isfinite(goalkeeper_claim_squared):
            raise ValueError(
                "goalkeeper_claim_radius_m + ball_radius_m의 float32 제곱은 "
                "유한해야 한다"
            )
        for name in (
            "shot_launch",
            "clearance_launch",
            "goalkeeper_distribution_launch",
        ):
            signed_launch = require_float32_representable(name, getattr(self, name))
            if not -1.0 <= signed_launch <= 1.0:
                raise ValueError(f"{name}는 [-1, 1] 범위여야 한다")
        if self.aerial_carry_factor <= 0.0 or self.aerial_carry_factor > 1.0:
            raise ValueError("aerial_carry_factor는 (0, 1] 범위여야 한다")
        if self.ground_intercept_velocity_factor > 1.0:
            raise ValueError("ground_intercept_velocity_factor는 1 이하여야 한다")
        defensive_depth = require_float32_representable(
            "defensive_depth_m", self.defensive_depth_m
        )
        clearance_depth = require_float32_representable(
            "clearance_depth_m", self.clearance_depth_m
        )
        if abs(defensive_depth) > self.field_half_length_m:
            raise ValueError("defensive_depth_m은 field 내부여야 한다")
        if self.penalty_area_length_m > 2.0 * self.field_half_length_m:
            raise ValueError("penalty_area_length_m은 field 길이를 넘을 수 없다")
        if self.penalty_area_half_width_m > self.field_half_width_m:
            raise ValueError("penalty_area_half_width_m은 field 폭을 넘을 수 없다")
        if not 0.0 < self.challenge_attempt_probability < 1.0:
            raise ValueError("challenge_attempt_probability는 (0, 1) 사이여야 한다")
        if self.challenge_penalty_area_probability_cap > 1.0:
            raise ValueError("challenge_penalty_area_probability_cap은 1 이하여야 한다")
        for name in (
            "challenge_proximity_gain",
            "challenge_opportunity_gain",
            "challenge_foul_avoidance_gain",
            "challenge_success_favourability_floor",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name}은 1 이하여야 한다")
        if self.pass_completion_threshold > 1.0:
            raise ValueError("pass_completion_threshold는 1 이하여야 한다")
        if self.progressive_carry_release_completion > 1.0:
            raise ValueError("progressive_carry_release_completion은 1 이하여야 한다")
        if self.pressure_escape_completion_floor > 1.0:
            raise ValueError("pressure_escape_completion_floor는 1 이하여야 한다")
        if self.receiver_arrival_distance_fraction > 1.0:
            raise ValueError("receiver_arrival_distance_fraction은 1 이하여야 한다")
        if clearance_depth > 0.0:
            raise ValueError("clearance_depth_m은 0 이하여야 한다")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2는 양수여야 한다")
        if self.goal_width_m <= 0.0:
            raise ValueError("goal_width_m은 양수여야 한다")
        if np.float32(self.goal_width_m) > np.float32(2.0 * self.field_half_width_m):
            raise ValueError("goal_width_m은 field 전체 폭 이하여야 한다")
        if np.float32(self.boundary_control_margin_m) > np.float32(
            min(self.field_half_length_m, self.field_half_width_m)
        ):
            raise ValueError(
                "boundary_control_margin_m은 field의 작은 half extent 이하여야 한다"
            )
        if self.shot_target_half_width_fraction > 1.0:
            raise ValueError("shot_target_half_width_fraction은 1 이하여야 한다")
        if not 0.0 < self.shot_aim_clean_base_scale <= 1.0:
            raise ValueError("shot_aim_clean_base_scale은 (0, 1] 범위여야 한다")
        if self.shot_aim_noise_fraction > self.shot_aim_noise_max_fraction:
            raise ValueError(
                "shot_aim_noise_max_fraction은 shot_aim_noise_fraction 이상이어야 한다"
            )
        if self.shot_forced_distance_fraction > 1.0:
            raise ValueError("shot_forced_distance_fraction은 1 이하여야 한다")
        if (
            self.outside_box_shot_macro_scale > 1.0
            or self.inside_box_transition_shot_floor > 1.0
            or self.inside_box_control_macro_scale > 1.0
            or self.inside_box_recycle_pass_scale > 1.0
            or self.inside_box_receiver_shot_gain > 1.0
            or self.zona_cutback_support_strength > 1.0
        ):
            raise ValueError("shot macro scale과 gain은 1 이하여야 한다")
        if self.attack_ball_follow_x > 1.0 or self.defend_ball_follow_x > 1.0:
            raise ValueError("ball follow 계수는 1 이하여야 한다")
        if not 0.0 < self.attack_longitudinal_compression <= 1.0:
            raise ValueError("attack_longitudinal_compression은 (0, 1] 범위여야 한다")
        if not 0.0 < self.defend_longitudinal_compression <= 1.0:
            raise ValueError("defend_longitudinal_compression은 (0, 1] 범위여야 한다")
        if self.attack_ball_follow_y > 1.0 or self.defend_ball_follow_y > 1.0:
            raise ValueError("ball follow 계수는 1 이하여야 한다")
        if (
            self.shot_quality_threshold > 1.0
            or self.shot_pressure_quality_floor > 1.0
            or self.shot_quality_threshold + self.outside_box_shot_quality_margin > 1.0
            or self.shot_pass_tradeoff > 1.0
            or self.first_time_shot_quality_threshold > 1.0
            or self.rebound_shot_quality_threshold > 1.0
            or self.close_shot_power > 1.0
            or self.turnover_shot_value_scale > 1.0
        ):
            raise ValueError(
                "shot quality 설정과 outside-box margin의 합은 1 이하여야 한다"
            )
        weight_total = (
            self.shot_range_weight
            + self.shot_opening_weight
            + self.shot_centrality_weight
        )
        if abs(weight_total - 1.0) > 1.0e-6:
            raise ValueError("shot quality 가중치의 합은 1이어야 한다")
        powers = tuple(self.role_offball_power)
        if len(powers) != 7:
            raise ValueError("role_offball_power는 ROLE_* 7개 값이어야 한다")
        canonical_powers = []
        for value in powers:
            scaled = require_float32_representable("role_offball_power", value)
            if not 0.0 < scaled <= 1.0:
                raise ValueError("role_offball_power는 (0, 1] 범위여야 한다")
            canonical_powers.append(scaled)
        object.__setattr__(self, "role_offball_power", tuple(canonical_powers))
        fatigue_floors = tuple(self.role_fatigue_power_floor)
        if len(fatigue_floors) != 7:
            raise ValueError("role_fatigue_power_floor는 ROLE_* 7개 값이어야 한다")
        canonical_floors = []
        for value in fatigue_floors:
            floor = require_float32_representable("role_fatigue_power_floor", value)
            if not 0.0 < floor <= 1.0:
                raise ValueError("role_fatigue_power_floor는 (0, 1] 범위여야 한다")
            canonical_floors.append(floor)
        object.__setattr__(self, "role_fatigue_power_floor", tuple(canonical_floors))
        if self.stamina_conservation_knee <= 0.0:
            raise ValueError("stamina_conservation_knee는 양수여야 한다")
        if self.cover_fatigue_power_floor > 1.0:
            raise ValueError("cover_fatigue_power_floor는 1 이하여야 한다")
        if self.ball_side_follow_asymmetry > 1.0:
            raise ValueError("ball_side_follow_asymmetry는 [0, 1] 범위여야 한다")
        if not 0.0 < self.box_collapse_compression <= 1.0:
            raise ValueError("box_collapse_compression은 (0, 1] 범위여야 한다")
        if self.box_lane_guard_strength > 1.0:
            raise ValueError("box_lane_guard_strength는 1 이하여야 한다")
        if self.box_runner_tracking_strength > 1.0:
            raise ValueError("box_runner_tracking_strength는 1 이하여야 한다")
        if self.long_shot_centrality_min > 1.0:
            raise ValueError("long_shot_centrality_min은 1 이하여야 한다")
        if self.long_shot_distance_m < self.shoot_distance_m:
            raise ValueError("long_shot_distance_m은 shoot_distance_m 이상이어야 한다")
        if not isinstance(self.shot_requires_open_lane, bool):
            raise TypeError("shot_requires_open_lane은 bool이어야 한다")
        if not isinstance(self.cross_spin_ground_eta_prior_enabled, bool):
            raise TypeError("cross_spin_ground_eta_prior_enabled는 bool이어야 한다")
        if not isinstance(self.clearance_respects_safe_pass, bool):
            raise TypeError("clearance_respects_safe_pass는 bool이어야 한다")
        if not isinstance(self.clearance_requires_release_contest, bool):
            raise TypeError("clearance_requires_release_contest는 bool이어야 한다")
        if not isinstance(self.shot_release_contest_relief_enabled, bool):
            raise TypeError("shot_release_contest_relief_enabled는 bool이어야 한다")
        if not isinstance(self.shot_residual_velocity_compensation, bool):
            raise TypeError("shot_residual_velocity_compensation는 bool이어야 한다")
        if not isinstance(self.possessor_control_contact_extension_enabled, bool):
            raise TypeError("possessor_control_contact_extension_enabled는 bool이어야 한다")
        floor = require_float32_representable("shot_distance_macro_floor", self.shot_distance_macro_floor)
        near = require_float32_representable("shot_distance_macro_near_m", self.shot_distance_macro_near_m)
        far = require_float32_representable("shot_distance_macro_far_m", self.shot_distance_macro_far_m)
        if not 0.0 <= floor <= 1.0:
            raise ValueError("shot_distance_macro_floor는 [0, 1] 범위여야 한다")
        if near < 0.0 or far <= near:
            raise ValueError("shot_distance_macro 거리는 0 <= near < far여야 한다")
        if not isinstance(self.attack_pass_flight_support_enabled, bool):
            raise TypeError("attack_pass_flight_support_enabled는 bool이어야 한다")
        if self.pass_macro_value_scale > 1.0 or self.progressive_pass_macro_scale > 1.0:
            raise ValueError("pass macro scale은 1 이하여야 한다")
        if self.progressive_pass_macro_scale < self.pass_macro_value_scale:
            raise ValueError(
                "progressive_pass_macro_scale은 pass_macro_value_scale 이상이어야 한다"
            )
        if self.boundary_control_power > 1.0 or self.dribble_control_power > 1.0:
            raise ValueError("control power는 1 이하여야 한다")
        if self.goalkeeper_claim_horizon_s > 1.0:
            raise ValueError("goalkeeper_claim_horizon_s는 1초 이하여야 한다")
        if self.pass_release_lookahead_s > 1.0:
            raise ValueError("pass_release_lookahead_s는 1초 이하여야 한다")
        if (
            self.cross_min_power > 1.0
            or self.cross_launch > 1.0
            or self.restart_fallback_power > 1.0
        ):
            raise ValueError("cross control은 1 이하여야 한다")
        for name in (
            "counterpress_control_ticks",
            "fresh_receiver_pressure_control_ticks",
            "quick_relay_control_ticks",
            "progressive_carry_min_ticks",
            "pass_min_episode_ticks",
            "pressure_escape_min_ticks",
            "release_urgency_start_ticks",
            "release_urgency_full_ticks",
            "shot_min_episode_ticks",
            "turnover_shot_settle_ticks",
            "lead_protection_start_gap",
            "lead_protection_full_gap",
        ):
            try:
                value = require_bounded_integer(name, getattr(self, name), minimum=0)
            except TypeError as error:
                raise ValueError(f"{name}는 음이 아닌 정수여야 한다") from error
            object.__setattr__(
                self,
                name,
                value,
            )
        if self.release_urgency_full_ticks <= self.release_urgency_start_ticks:
            raise ValueError(
                "release_urgency_full_ticks는 release_urgency_start_ticks보다 커야 한다"
            )
        if not 0.0 < self.release_urgency_control_floor <= 1.0:
            raise ValueError("release_urgency_control_floor는 (0, 1] 범위여야 한다")
        if self.lead_protection_start_gap < 1:
            raise ValueError("lead_protection_start_gap은 1 이상이어야 한다")
        if self.lead_protection_full_gap < self.lead_protection_start_gap:
            raise ValueError(
                "lead_protection_full_gap은 lead_protection_start_gap 이상이어야 한다"
            )
        if (
            self.quick_relay_completion_floor > 1.0
            or self.quick_relay_alignment_floor > 1.0
            or self.quick_relay_arrival_advantage_floor > 1.0
        ):
            raise ValueError("quick relay threshold는 1 이하여야 한다")


DEFAULT_RULE_POLICY_CONFIG = RulePolicyConfig()
