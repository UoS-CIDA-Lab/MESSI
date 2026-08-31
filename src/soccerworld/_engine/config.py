"""
환경에 필요한 객체 정의 및 특징값 정리
"""

import math
import numbers
from dataclasses import dataclass, field, fields

from .timebase import DEFAULT_MATCH_DURATION_SECONDS, DEFAULT_TIMEBASE, duration_to_ticks

_FLOAT32_MIN_SUBNORMAL = math.ldexp(1.0, -149)
# Rule-policy values are combined with pitch-scale coordinates, player sums,
# and (for tactical utilities) one another before the result is clipped.  Merely
# fitting one scalar into float32 is therefore insufficient: a value near
# FLOAT32_MAX can overflow a multiply or squared norm immediately.  2**20 is
# still many orders beyond every meaningful football tuning value while leaving
# deterministic headroom for those intermediates.
_RULE_POLICY_SCALAR_ABS_MAX = float(1 << 20)

# ``SoccerEnv`` models on-pitch slots, not a bench roster.  Association
# football permits at most eleven players per team, and every hot-path
# geometry/observation primitive scales at least quadratically in the combined
# slot count.  Keep the sporting rule and the practical JAX shape bound in one
# environment-level contract while still allowing asymmetric small-sided play.
MAX_ENV_TEAM_PLAYERS = 11

# One control action is held across this many physics ticks.  The defaults use
# six ticks (15 Hz control) and the native 30 Hz path uses three; 256 still
# permits 1 Hz control (90 ticks) with ample diagnostic headroom at the default
# 90 Hz physics rate.  Larger values create static scan lengths and optional
# `(decimation, State)` outputs per action, so they are not a useful SoccerEnv
# operating mode even though the general-purpose Timebase can represent them.
MAX_CONTROL_DECIMATION = 256

# Hard ceiling shared by every configurable position-relaxation loop.  The
# ordinary separation solver is Python-unrolled into a JIT graph, while
# restart/reconcile relaxation executes this many O(N^2) sweeps at runtime.
# Sixteen leaves substantial diagnostic headroom over the measured defaults
# (3 and 6) without exposing construction-time graph explosions or effectively
# unbounded compiled execution.  These are local approximate relaxations;
# adding hundreds of rounds cannot repair an inconsistent multi-body geometry.
MAX_POSITION_SOLVER_ROUNDS = 16

# A complete match in the observed corpus uses at most six replacements per
# team (12 rows total, including extra-time rules).  Thirty-two retains more
# than 2.5x headroom and supports 16 per team, while bounding constructor work
# and the per-step scheduled-transition loop against adversarial millions-row
# inputs.
MAX_SUBSTITUTION_SCHEDULE_ROWS = 32

# 벤치 크기는 '투입 가능 인원'이 아니라 '고를 수 있는 사람 수'다. 둘은 다르다 — IFAB가
# 허용하는 교체는 5명이지만, 9명 중 어느 5명을 고르는가가 그 자체로 결정이다. 점수차·
# 포지션·잔여 시간에 따라 다른 사람을 넣는 알고리즘을 표현하려면 벤치가 커야 한다.
# 30은 정규 선수단 규모(25~28)에 여유를 둔 값이고, State shape 상한 역할만 한다.
MAX_BENCH_SIZE = 30

# Rolling resistance is a piecewise-linear static lookup used in every ball
# physics tick and snapshotted into the rule-policy graph.  The calibrated
# table has ten knots; 256 preserves 25.6x tuning headroom while preventing a
# public config from embedding an effectively unbounded constant/search table
# in each compiled hot path.
MAX_ROLLING_TABLE_KNOTS = 256

# Both public RulePolicy sample counts become static JAX array dimensions.  At
# the 11v11 ceiling, one receiver float32 tensor at 256 samples is
# 22 * 22 * 256 * 4 bytes = 0.47 MiB.  That is already 25.6x the receiver
# default on every 15 Hz policy call; one million samples would make each such
# tensor 1.80 GiB.  The shared ceiling also gives style sampling 128x its
# default while limiting that independent `(samples, 5)` allocation to 5 KiB.
MAX_RULE_POLICY_SAMPLE_COUNT = 256

@dataclass(frozen=True)
class Ball:
    radius: float = 0.11
    """공 반지름(m)"""

@dataclass(frozen=True)
class Agent:
    id: int = 0
    """비음수 선수/person ID(int32 범위). ``NO_PLAYER`` 등 음수 규칙 sentinel과 겹치지 않는다."""
    speed: float = 7.96   # [DFL calib p99.9] 실측 선수 속력 p99.9
    """선수 최대속도(m/s) — vmax. **[p99.9 캘리브]** 실측 선수 속력 p99.9=7.96(데모는 override). @snapshot"""
    tall: float = 1.8
    """선수 키(m)"""
    reach_z_max: float = 2.7
    """선수 최대 reach 높이(m) — 공과 선수 간 수직거리 ≤ reach_z_max → reach=True"""
    ball_control: float = 0.5
    """선수 공제어력(0~1) — 트래핑 확률(trap_base×ctrl)·경합 로짓(w_ctrl)에 반영.
    이 필드의 초기 기준은 원본 DFL Ability_Base 캘리브레이션이다."""
    is_gk: bool = False
    """선수 골키퍼 여부 — 골킥 키커 우선·세트피스(페널티 수비 GK 면제) 등 규칙에 반영됨.
    ※능력치(reach_z_max/ball_control)는 이 플래그로 자동 조정되지 않는다 — GK를 다르게 주고
    싶으면 호출자가 Agent 필드에 직접 설정할 것(초기화는 받은 값을 그대로 사용)."""
    init_pos: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """선수 초기 위치(x, y)"""
    endurance_factor: float = 1.0
    """장·단기 stamina 소모율을 나누는 지구력 배율. 1.0은 기존 동역이다."""


@dataclass(frozen=True)
class Substitution:
    """관측된 교체 하나 — env가 정해진 tick에 적용하는 **외생** identity transition.

    교체는 정책의 행동이 아니다. 실경기 event/lineup에서 확정된 사실을 N-slot state에
    손실 없이 투영하기 위한 스케줄 항목이며, 행동공간을 넓히지 않는다. 코퍼스의
    ``substitutions.jsonl``(out/in/시각/pair_confidence)과 ``entities.jsonl``(roster 능력치)이
    이 dataclass의 원천이다.

    ``tick``이 T이면 **t=T로 표시된 state부터** 새 사람이 뛴다. 정책은 그 state를 관측한 뒤
    행동하므로 교체를 모르고 행동하는 프레임이 생기지 않는다.
    """

    tick: int
    """교체가 반영되는 control tick(에피소드 시작 기준)."""
    slot: int
    """사람이 바뀌는 N-slot 인덱스. 팀·공격방향은 슬롯 값을 그대로 유지한다."""
    player_id: int
    """투입 선수의 person id — 나간 선수 id를 재사용하지 않는다."""
    entry_pos: tuple[float, float]
    """투입 선수의 첫 관측 위치(world m). 나간 선수 위치에서 보간하지 않는다."""
    role_pos: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """투입 선수의 전술 role anchor(공격 접힘 프레임, m)."""
    speed: float = 7.96
    """투입 선수 vmax(m/s)."""
    tall: float = 1.8
    """투입 선수 키(m) — head_z 산출에 사용."""
    reach_z_max: float = 2.7
    """투입 선수 최대 reach 높이(m)."""
    ball_control: float = 0.5
    """투입 선수 공제어력(0~1)."""
    is_gk: bool = False
    """투입 선수 GK 여부."""
    stamina_long_entry: float = 1.0
    """입장 시점 장기 stamina prior. 실제 출전시간·워밍업 근거가 있으면 호출자가 조정한다."""
    stamina_short_entry: float = 1.0
    """입장 시점 단기 stamina prior. 교체 선수의 즉시 질주 여력을 별도로 표현한다."""
    yellow_cards: int = 0
    """투입 선수 person ledger의 누적 경고 — 나간 선수 값을 승계하지 않는다."""
    endurance_factor: float = 1.0
    """투입 선수의 stamina 소모율 배율. 1.0은 기존 동역이다."""

@dataclass(frozen=True)
class BenchPlayer:
    """교체 후보 한 명 — 벤치에 앉아 투입을 기다리는 사람.

    :class:`Substitution`이 '언제 누가 들어온다'는 **확정된 사실**인 것과 달리, 이쪽은
    '들어올 수 있는 사람'이다. 그래서 시각이 없다. 자동 교체 알고리즘은 이 명단에서 고르고,
    스케줄 재현은 이 명단을 쓰지 않아도 된다(둘은 독립적으로 켤 수 있다).
    """

    player_id: int
    """교체 후보의 person id — 출전 중인 선수 id와 겹치면 안 된다."""
    role_pos: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """투입 시 전술 role anchor(공격 접힘 프레임, m)."""
    speed: float = 7.96
    """vmax(m/s)."""
    tall: float = 1.8
    """키(m) — head_z 산출에 쓴다."""
    reach_z_max: float = 2.7
    """최대 reach 높이(m)."""
    ball_control: float = 0.5
    """공제어력(0~1)."""
    is_gk: bool = False
    """골키퍼인가. GK는 GK 자리로만 들어갈 수 있다."""
    endurance_factor: float = 1.0
    """stamina 소모율 배율. 1.0은 기존 동역이다."""


@dataclass(frozen=True)
class Stadium:
    width: float = 68.0
    """경기장 폭(m)"""
    length: float = 105.0
    """경기장 길이(m)"""
    goal_width: float = 7.32
    """골대 폭(m)"""
    goal_height: float = 2.44
    """골대 높이(m)"""
    penalty_area_length: float = 16.5
    """페널티 에어리어 길이(m)"""
    penalty_area_width: float = 40.32
    """페널티 에어리어 폭(m)"""
    goal_area_length: float = 5.5
    """골 에어리어 길이(m) — 골킥 스폿과 렌더 필드 마킹이 함께 사용."""
    goal_area_width: float = 18.32
    """골 에어리어 폭(m)"""
    center_circle_radius: float = 9.15
    """센터 서클 반지름(m)"""
    penalty_arc_radius: float = 9.15
    """페널티 아크 반지름(m)"""
    corner_arc_radius: float = 1.0
    """코너 플래그에서 코너 에어리어를 이루는 1/4원 반지름(m).
    상대 9.15m 이격은 공 중심이 아니라 이 아크에서 잰다."""

    @property # 함수를 attribute처럼 쓰기 위해 property 데코레이터 사용
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

@dataclass(frozen=True)
class RulePolicy:
    """관측 기반 룰 정책의 공용 튜너.

    역할/스타일 열 번호는 ``constants.py``의 불변 계약이고, 성능을 바꾸는 수치는 이 객체만
    소유한다. ``make_rule_based_policy(..., policy_config=...)``로 실험별 override할 수 있다.
    """

    use_shot_solver: bool = True
    style_sample_count: int = 2
    professional_control_floor: float = 0.30
    """정책 의사결정에서 하위 프로 수준으로 포화할 ``ball_control`` 값.

    이보다 낮은 입력도 플레이를 못 하는 아마추어처럼 계속 악화시키지 않는다. 환경의
    물리 경합은 원래 능력치를 그대로 쓰지만, 전술 선택·실행 오차의 정책 효과는 여기서
    포화해 모두가 기본적인 프로 플레이를 수행하게 한다.
    """
    professional_control_ceiling: float = 0.80
    """정책 의사결정에서 상위 프로 수준으로 포화할 ``ball_control`` 값."""
    professional_reach_floor_m: float = 2.35
    """공중 수신자 선호에서 하위 프로 수준으로 보는 최대 도달 높이(m)."""
    professional_reach_ceiling_m: float = 2.90
    """공중 수신자 선호에서 상위 프로 수준으로 보는 최대 도달 높이(m)."""
    receiver_control_value_gain: float = 0.08
    """동일한 패스 레인에서 제어력이 좋은 수신자에게 주는 최대 상·하 격차."""
    cross_receiver_ability_gain: float = 0.10
    """크로스 목표 가치에 제어력·도달 높이가 만드는 최대 상·하 격차."""
    dribble_control_retention_gain: float = 0.18
    """드리블 유지가 제어력 하·상위 프로 사이에서 달라지는 총 배율 폭."""
    execution_control_noise_gain: float = 0.32
    """킥 각오차가 제어력 하·상위 프로 사이에서 달라지는 총 배율 폭.

    기본값이면 하위 프로는 중간값보다 16% 큰 오차, 상위 프로는 16% 작은 오차를
    갖는다. 능력치 하나로 완벽/무능을 만들지 않으면서 영상에서 구분되는 정도다.
    """
    aerial_control_floor_relief: float = 0.05
    """상위 프로 공중 접촉자가 동료 연결을 택할 때 완화되는 완성률 하한의 총 폭."""
    shoot_gain: float = 2.6
    shot_commit_xg: float = 0.56
    """이 값을 넘는 오픈플레이 슛은 패스 가치가 조금 높아도 마무리한다.

    ``shot_min_xg``와 같은 비율(0.33→0.50)로 함께 올린다 — 하한만 올리고 확정 문턱을
    두면 "쏠 수 있는 슛은 거의 다 확정"이 되어 문턱이 하나로 붕괴한다."""
    first_time_shot_xg: float = 0.42
    """같은 팀의 빠른 패스를 원터치 슛/발리로 연결할 최소 xG. 위와 같은 비율로 맞춘다."""
    first_time_shot_distance: float = 30.0
    """빠른 패스 원터치 마무리를 허용할 최대 골 거리(m)."""
    shot_approach_margin: float = 3.0
    """슛 범위 바로 밖에서 직전 전진 터치를 이어 갈 최대 추가 거리(m)."""
    shot_approach_xg: float = 0.20
    """슛 범위 진입 드리블을 패스보다 우선할 최소 현재 위치 xG."""
    pass_lead_time_cap: float = 0.90
    """움직이는 수신자의 도착점을 예측하는 최대 시간(s)."""
    pass_lead_velocity_weight: float = 0.72
    """등속 수신자 예측을 실제 패스 목표에 반영하는 비율."""
    pass_lead_distance_cap: float = 6.5
    """현재 수신자 위치에서 패스 목표를 앞세울 수 있는 최대 거리(m)."""
    receive_prediction_horizon_s: float = 2.50
    """의도적 패스 뒤 수신 주자를 고를 미래 공 궤적의 최대 시간(s). 42m 백스핀 크로스의
    실측 체공시간 2.42s를 덮는다. 0.20s보다 짧은 창은 첫 표본도 창 안으로
    축소해 구성값을 실제 최대값으로 지킨다."""
    receive_prediction_samples: int = 10
    """수신 주자 도달시간 비교에 쓰는 미래 궤적 표본 수."""
    through_gap_weight: float = 0.55
    """온사이드 수신자와 최종 수비선 사이 여유를 킬패스 공간으로 쓰는 비율."""
    through_shoulder_cue_weight: float = 0.55
    """수비선 어깨의 정지한 온사이드 공격수가 패스 순간 출발할 의도 강도.

    현재 속도만 요구하면 라인에 도착해 감속한 공격수는 영원히 킬패스 후보가 되지 않는다.
    이 값은 공을 차기 전 오프사이드 판정은 그대로 둔 채, 패스와 동시에 시작하는 런을
    목표점 예측에 반영한다.
    """
    through_shoulder_gap_m: float = 2.5
    """최종 수비선에서 이 거리 안의 전방 수신자에게 타이밍 런 cue를 줄 범위(m)."""
    through_run_ahead: float = 3.5
    """전진 중인 온사이드 러너가 수비선 어깨에 붙었을 때도 목표로 둘 최소 라인 뒤 공간(m)."""
    through_min_progress: float = 6.0
    """킬패스 후보가 되기 위한 최소 수신자 전진 우위(m)."""
    through_bonus: float = 0.28
    """실제 전방 공간을 겨냥하는 패스의 가치 가산점."""
    forward_line_margin: float = 0.80
    """전방 주자가 최종 수비선 어깨에서 확보하는 온사이드 여유(m)."""
    forward_run_min_gap: float = 6.0
    """공보다 이만큼 앞선 수비선이 있을 때 포워드를 라인 어깨까지 올린다(m)."""
    combination_gain: float = 0.34
    """받은 뒤 실제로 안전한 두 번째 패스를 이어갈 수신자에 주는 최대 가치 가산점.

    단순한 주변 동료 수가 아니라 후보 수신점→제3선수의 실제 drive 속도,
    수비 도착시간과 순전진량을 함께 평가한다.
    """
    possession_combination_gain: float = 0.35
    """낮은 directness 팀이 짧은 2차 연결점을 더 선호하도록 combination에 주는 추가 배율."""
    continuation_support_lead_weight: float = 0.65
    """첫 패스 비행 중 지원자의 현재 속도를 2차 패스 위치 예측에 반영하는 비율."""
    continuation_support_lead_cap_s: float = 0.80
    """2-hop 평가에서 지원자 등속 외삽에 쓰는 최대 시간(s)."""
    continuation_min_distance: float = 5.0
    """2차 연결 패스로 인정할 최소 수신점 간 거리(m)."""
    continuation_max_distance: float = 24.0
    """2차 연결 패스로 인정할 최대 수신점 간 거리(m)."""
    continuation_backward_tolerance: float = 12.0
    """2-hop 출구가 첫 수신점보다 뒤여도 허용할 최대 거리(m).

    전방 수신자가 5~12m 뒤의 미드필더에게 되받아주는 벽패스도 첫 패스의 순전진을
    보존하는 현실적 출구다. 4m 하한은 이 패턴을 모두 폐기해 안전한 횡패스만 남겼다.
    """
    direct_distance_gain: float = 0.22
    """높은 directness 팀의 먼 전방 패스 후보에 주는 최대 가치 가산점."""
    possession_distance_penalty: float = 0.24
    """낮은 directness 팀이 불필요한 장거리 후보에 적용하는 최대 가치 감점."""
    pass_target_distance_penalty: float = 0.35
    """패스 행동의 가치와 분리해 **수신자 선택에만** 적용하는 장거리 후보 감점.

    여러 합법 수신자 중 실제보다 긴 목표를 반복해서 고르는 편향을 교정한다. directness가
    높은 팀에는 절반까지 완화되어 롱볼 스타일을 보존한다. 0.35는 약해진 전진 보상과
    함께 평균 패스 길이 17.88 m를 만들었다(실측 17.70 m, 240 s × 8 장기 acceptance).
    선택된 수신자의 원래 기대가치로 행동을 비교하므로 낮은 가치의 짧은 패스를 억지로
    실행하지 않는다.
    """
    pass_progress_base: float = 0.01
    """수신자 가치에서 전진량에 주는 기본 가중치.

    방향 분포를 직접 제어하는 항이다. 패스 거리 패널티와 달리 긴 횡패스까지 함께 벌하지
    않으며, ``pass_progress_direct_gain``과 합쳐 팀 directness별 전진 성향을 보존한다.
    """
    pass_progress_direct_gain: float = 0.05
    """directness=1일 때 전진량 가중치에 더하는 값."""
    pass_receiver_movement_gain: float = 0.12
    """패스 전 실제로 이동 중인 수신자의 도착점에 주는 최대 가치 가산점.

    DFL 7경기의 성공 패스 수신자는 킥 직전 1초에 중앙값 2.42m를 이동했다.
    이 항은 정지한 위치의 위협값만 비교하던 정책이 사선·체크런을 실제 패스로
    보상하게 한다. 물리 레인 완성률 하한은 그대로 유지된다.
    """
    pass_space_creation_gain: float = 0.16
    """수신자가 현재 위치보다 더 열린 도착점으로 이동할 때의 최대 가산점.

    움직임의 절대 크기보다 실제로 압박에서 벗어난 정도를 더 크게 보상한다. DFL에서
    성공 패스 수신자의 직전 1초 이동 중앙값은 2.42m이므로 전력 질주자를 고르는 정책이
    아니라 짧은 체크런으로 패스각을 만든 선수를 고르는 정책이어야 한다.
    """
    pass_shot_chain_gain: float = 0.40
    """첫 수신자에서 제3선수로 이어지는 물리 2-hop×xG 가치 가중치."""
    pass_forward_direction_cost: float = 0.04
    """일반 전방 후보의 방향 균형 비용. through 강도가 클수록 해제해 킬패스는 보존한다.

    물리 레인 도입 뒤 3분×3시드 영수증의 전방 비중이 22.6%(DFL 33.3%)로 내려갔다.
    합산 순전진을 소프트 보너스로 두는 계약에서 0.04로 이미 수비 도착 위험을 부담하는
    전방 후보의 중복 비용만 완화한다.
    """
    pass_short_forward_bonus: float = 0.06
    """점유형 팀의 짧은 전진 지상 연결에 주는 최대 가산점.

    거리 6~11m에서만 주로 작동하고 거리와 함께 0으로 줄어든다. 지원 위치와 같은
    directness start→full 구간에서 최대치→0으로 줄므로 tiki-taka의 짧은 전진 조합은
    횡패스 보너스와 경쟁하지만 balanced·long-ball의 먼 전방 패스를 더 밀지 않는다.
    """
    pass_sideways_direction_bonus: float = 0.03
    """DFL의 전/횡/후 약 1/3 분포를 위한 횡패스 수신자 선택 가산점."""
    pass_backward_direction_bonus: float = 0.12
    """압박 탈출·깊은 빌드업에서 후방패스에 주는 최대 수신자 선택 가산점."""
    pass_backward_recycle_floor: float = 0.05
    """무압박·비수신 국면에서 남기는 후방패스 보너스의 최소 비율.

    압박도와 자기 진영 깊이에 따라 ``pass_backward_direction_bonus``가 연속적으로
    복원된다. 막 받은 패스를 곧바로 되돌릴 때는 이 바닥도 적용하지 않아 A→B→A
    반복을 억제하되, 압박 탈출과 후방 빌드업의 안전판은 유지한다.
    """
    loft_distance_base: float = 29.0
    """로프트를 검토하기 시작하는 directness=0 기준 거리(m)."""
    loft_distance_direct_relief: float = 10.0
    """directness=1에서 로프트 거리 기준을 줄이는 양(m)."""
    loft_force_directness: float = 0.85
    """안전한 장거리 지상 레인도 의도적으로 로프트로 바꾸는 최소 directness.

    이 값보다 낮은 점유·균형 전술은 거리 기준을 넘더라도 지상 완성률 하한을
    통과한 레인을 낮게 연결한다. 지상 레인이 막혔으면 전술과 무관하게 로프트를
    검토하므로, 롱패스 자체를 없애는 스위치는 아니다.
    """
    loft_control_floor: float = 0.36
    """일반 로프트 패스에서 수신자 도착시간·상대 선점을 함께 본 최소 공중 수신 우위.

    크로스와 달리 일반 롱패스가 지상 레인 품질만으로 수신자를 고르던 경로를 막는다.
    """
    aerial_control_max_height: float = 1.45
    """압박이 없는 수신자가 헤더 대신 소프트 첫 터치를 시도할 최대 공 높이(m)."""
    pressured_aerial_control_max_height: float = 0.75
    """압박 중에도 헤더성 걷어내기 대신 통제할 낮은 바운드의 최대 높이(m).

    공 반경을 조금 넘었다는 이유만으로 발목~허리 높이의 공을 17m/s 전방 헤더처럼
    처리하지 않는다. 그보다 높은 실제 공중 경합은 기존 즉시 헤더 계약을 유지한다.
    """
    aerial_control_pressure_max: float = 0.55
    """공중볼을 바로 헤더/클리어하지 않고 냙하를 기다릴 최대 압박도."""
    aerial_duel_eta_window_s: float = 0.20
    """상대 서비스에 수비수가 직접 공중 경합을 제출할 최대 ETA 열세(s).

    이 범위를 벗어난 수비수는 공격 수신자와 같은 좌표로 달려들지 않고 골사이드
    세컨드볼을 준비한다. 공 단위 접촉 잠금 없이도 명백히 늦은 선수가 경합 자세를
    반복하는 현상을 막되, 실제 50:50 공은 양 팀 contest에 그대로 맡긴다.
    """
    aerial_defender_cover_distance: float = 3.0
    """직접 경합 ETA 창을 놓친 수비수가 예상 수신점 골사이드에 두는 간격(m)."""
    aerial_pass_min_distance: float = 3.0
    """공중 접촉으로 동료에게 연결할 최소 목표거리(m)."""
    aerial_pass_max_distance: float = 18.0
    """공중 접촉을 헤더성 동료 패스로 취급할 최대 목표거리(m)."""
    aerial_pass_completion_floor: float = 0.78
    """헤더성 동료 연결에 요구하는 기존 지상 레인/수신 ETA 완성률 하한.

    전용 공중 패스 실측 적합 전의 보수적 기본값이다. 이미 계산한 패스 품질을 재사용해
    정책 비용을 늘리지 않으면서, 열린 동료가 있을 때만 전방 고정 클리어를 대체한다.
    """
    aerial_pass_min_speed_mps: float = 5.0
    """최단 공중 동료 패스의 목표 출구속도(m/s)."""
    aerial_pass_max_speed_mps: float = 12.0
    """최장 공중 동료 패스의 목표 출구속도(m/s); 거리에 따라 최소값과 선형보간한다."""
    aerial_shot_speed_mps: float = 18.0
    """정책이 원터치 슛으로 선택한 공격 헤더의 목표 출구속도(m/s)."""
    aerial_clear_speed_mps: float = 16.0
    """안전한 동료 연결이 없는 공중볼 클리어의 목표 출구속도(m/s)."""
    aerial_pass_launch01: float = 0.10
    """헤더성 동료 패스의 지면 기준 발사각 fraction(접촉 높이 역매핑 전)."""
    aerial_clear_launch01: float = 0.18
    """공중볼 클리어의 지면 기준 발사각 fraction(접촉 높이 역매핑 전)."""
    pass_min_distance: float = 6.0
    """패스 후보로 인정할 최소 공↔수신자 거리(m).

    실측 패스 중 5 m 미만은 5.4%뿐인데, 3 m 하한에서는 시뮬이 22%를 그 구간에 쏟아냈다
    (인플레이 1분당 패스 21회 vs 실측 9.5회). 발밑 3~5 m 툭툭은 실제 축구의 패스가 아니라
    대부분 드리블 터치다.
    """
    pass_contact_clearance: float = 1.8
    """패스·크로스를 실제 릴리스하기 위한 캐리어-최근접 상대 최소거리(m).

    환경의 challenge reach(1.4m)+공 반경보다 작은 간격에서는 킥과 같은 substep의 몸 접촉이
    거의 확정이다. 압박은 릴리스를 앞당기지만 이미 접촉권 안에 들어온 뒤의 거짓 패스는
    허용하지 않고 먼저 통제/회피하도록 한다.
    """
    pass_lane_intercept_radius: float = 1.50
    """지상 패스 레인에서 수비수가 차단할 수 있는 공 중심 기준 반경(m).

    환경의 challenge reach와 공 반경을 합친 실제 접촉 기하에 맞춘다.
    """
    pass_lane_reaction_s: float = 0.18
    """정지·비접근 수비수가 새 패스 레인에 반응하기 전 지연시간(s)."""
    pass_lane_defender_acceleration_mps2: float = 5.5
    """패스 차단 ETA가 사용하는 보수적 수비 가속도(m/s²).

    이동 물리의 7.95/8.46m/s² 종·횡 상한보다 낮게 두어 즉시 최대가속을 가정하지 않되,
    현재 속도만 외삽하던 낙관 편향을 제거한다.
    """
    pass_receive_ahead_margin: float = 0.35
    """진행 중인 자기 팀 패스의 궤적 수신자가 공보다 앞서 있어야 하는 최소 종방향 여유(m).

    원 패서가 첫 0.067초 동안 공과 함께 달리며 같은 헤더를 연속 제출하는 현상을 막는다.
    """
    drive_arrive_speed_mps: float = 7.0
    """지상 drive 패스의 거리 역솔버가 목표점에서 노리는 수평 공속(m/s).

    발사속도를 임의 비율로 줄이지 않고 현재 공 물리에서 이 도착속도가 되는 발사속도를
    역산한다. 전체 오픈플레이 공속 분포와 패스 비행시간을 조절하는 정책 계수다.
    """
    ground_control_touch_speed_mps: float = 3.0
    """빠른 지상볼 트랩과 중립 루즈볼 확보에 쓰는 통제 터치 출구속도(m/s).

    두 상황은 모두 공을 멀리 차는 패스가 아니라 다음 행동을 위한 첫 터치다. 구 정책의
    루즈볼 상수 ``0.35 * f2b_speed_max``는 기본 환경에서 11.9 m/s였고, 같은 파일의
    지상 트랩 3.0 m/s와 모순되면서 전체 인플레이 공속을 부풀렸다. 하나의 물리 단위
    config를 공유해 ``f2b_speed_max`` 변경에도 의미가 보존되게 한다.
    """
    receive_control_lead_speed_mps: float = 0.75
    """달리는 패스 수신자의 첫 터치가 현재 선수 속도보다 앞서는 속도 여유(m/s).

    일반 ``ground_control_touch_speed_mps``는 정지·저속 수신자의 절대 출구속도 하한이다.
    달리는 선수에게도 그 값을 그대로 적용하면 선수가 공을 앞질러 가므로, 의도적 패스의
    소프트 수신에 한해 ``선수 현재 속도벡터 + 이 여유×다음 진행방향``을 첫 터치 목표로
    쓴다. 최종 속도는 ``carried_release_speed_mps``와 엔진의 DRIBBLE 분류 상한 안으로
    제한한다. 이 값을 별도 필드로 둬 DFL 수신 후 상대속도 분포에 맞춰 독립 보정할 수 있다.
    """
    settled_ball_speed_mps: float = 4.0
    """정지 트랩 없이 일반 패스·크로스·클리어를 허용할 공속 상한(m/s)."""
    carried_release_speed_mps: float = 8.0
    """직전 자기 팀 드리블 뒤 달리며 패스·슛을 이어갈 공속 상한(m/s).

    인입 패스에는 적용하지 않는다. 현재값 8m/s는 4m/s 일반 정착 문턱을 그대로 둔 채
    빠른 운반만 분리한다. 하나의 4m/s 문턱만 쓰면 빠른 운반자는 매번 공을
    완전히 세운 뒤에야 동료를 찾고, 최종 3분의 1에서 오프더볼 런이 사라진 뒤 패스한다.
    마지막 터치 종류가 관측되므로 tracking/event에서도 같은 조건을 복원할 수 있다.
    """
    gk_long_directness_threshold: float = 0.65
    """GK가 짧은 배급보다 롱볼 수신자를 우선하는 팀 directness 문턱."""
    gk_long_min_distance: float = 22.0
    """GK 롱배급 수신 후보의 최소 예상 수신거리(m)."""
    gk_long_min_progress: float = 6.0
    """GK 롱배급 수신 후보의 최소 전진 우위(m)."""
    gk_buildup_gain: float = 0.03
    """자기 진영에서 안전하게 열린 GK를 빌드업 출구로 쓰는 최대 가치 가산점.

    DFL에는 GK를 거치는 빌드업이 분명히 있지만, 0.40은 짧은 다중 시드에서 GK 패스
    비중을 38.5%(DFL 9.1%)까지 끌어올렸다. 3분×3시드의 0.12도 29.0%였고,
    소프트 2-hop 이전의 0.03 영수증은 12.9%였으므로 적극적 활용과 상시 후퇴를 분리한다.
    """
    gk_foot_pass_max_distance: float = 28.0
    """손 사용이 제한된 GK가 롱 클리어 대신 낮게 연결할 최대 거리(m)."""
    gk_foot_pass_pressure_max: float = 0.65
    """GK가 발밑 숏배급을 유지할 최대 합산 압박도."""
    shot_min_xg: float = 0.50
    """오픈플레이 슛을 허용할 최소 xG.

    0.03 하한에서는 시뮬 슛이 90분당 67회(실측 11회), 중앙값 거리 26.7 m(실측 16.7 m)였다.
    이중 stamina 환경에서 0.14도 팀당 인플레이분 슛이 실측의 4.87배였다.

    0.33은 DFL 목표 0.182회/팀·인플레이분을 겨냥한 값이었지만, K리그1 30경기 트래킹 기준
    0.230회와 비교하면 실측 슛이 **여전히 3.42배**였다(0.788회, 300 s×3 seed). 그 결과가
    가장 눈에 띄는 곳은 스코어다 — 90분 전체 롤아웃에서 12-17, 경기당 29골이 나왔다
    (K리그 평균 2.7골).

    0.45 / 0.50 / 0.58을 각각 60 시뮬분(슛 표본 26~48개)으로 재면 0.274 / 0.249 / 0.173이고,
    0.50이 실측 0.230의 1.08배로 가장 가깝다. 슛이 줄면 GK 캐치와 킥오프도 함께 내려간다
    — 90분 롤아웃에서 GK 홀드 105회·킥오프 30회였던 것이 그 연쇄다.
    """
    pass_release_cost: float = 0.24
    """패스 가치에서 빼는 '공을 놓는 비용'.

    캐리어가 매 프레임 '지금 패스 vs 계속 운반'을 비교하는데, 이 비용이 없으면 근소하게
    이득인 패스도 전부 실행돼 인플레이 1분당 패스가 18.4회가 된다(실측 9.5회 — 6.4초에 한 번).
    0.16에서도 16.06회로 과다해 0.24를 쓴다. 실제 선수는 받은 뒤 한두 터치를 하고
    내준다. 그 '한두 터치'가 이 상수다. 압박 중에는 ``pass_pressure_relief``만큼 완화된다.
    """
    pass_pressure_scale: float = 0.75
    """패스 릴리스 비용을 완화하기 시작하는 압박도 스케일.

    :func:`rule_policy.tactics.pressure`의 합산 압박도를 이 값으로 나눠 [0,1]로 접는다.
    가까운 상대 한 명이 있거나 협공이면 안전한 탈압박 패스를 불필요하게 지연하지 않는다.
    """
    pass_pressure_relief: float = 0.45
    """강압박에서 ``pass_release_cost``를 줄이는 최대 비율."""
    unpressured_backpass_cost: float = 0.02
    """무압박 후방 패스에 추가로 부과하는 최대 가치 비용.

    백패스 전체를 금지하지 않고 후퇴거리·압박·자기진영 깊이에 따라 연속적으로 적용한다.
    그래서 전진 이득이 없는 안전한 A→B→A 반환은 줄되, 압박 탈출과 깊은 빌드업은 남는다.
    """
    backpass_distance_scale: float = 15.0
    """``unpressured_backpass_cost``가 최대가 되는 후퇴거리(m)."""
    deep_backpass_relief: float = 0.65
    """자기 골라인에 가까워질수록 후방 패스 비용을 줄이는 최대 비율."""
    open_dribble_gain: float = 0.12
    """무압박이고 전방 5m가 열렸을 때 드리블 가치에 더하는 최대 배율.

    패스를 단순히 삭제하는 대신 캐리어가 실제로 공간을 운반하도록 만드는 대응 항이다.
    """
    pass_release_base_probability: float = 0.024
    """무압박·비위험지역에서 패스 최적안이 실제 릴리스되는 control-frame당 확률.

    15Hz에서 평시 hazard만의 평균 보류시간은 약 2.78초다. 패스를 보류한 프레임은 드리블로 폴백하며,
    seeded PRNG라 같은 경기 key의 결과는 완전히 재현된다. 킬패스는 별도 가산으로 더 빨리
    열리지만 무조건 우회하지 않아 전진 패스만 남는 선택 편향을 막는다.
    """
    pass_release_pressure_gain: float = 0.090
    """압박도가 최대일 때 frame당 패스 릴리스 확률에 더하는 값."""
    contact_release_probability: float = 0.30
    """밀착 상대 반대편에 안전한 낮은 출구가 있을 때 frame당 릴리스 확률.

    일반 패스 cadence와 탈취 직후 secure-window 확률에서 독립시켜, 실측의 밀착
    탈압박 시간 분포를 다른 패스 빈도를 움직이지 않고 보정할 수 있게 한다.
    """
    pass_release_final_third_gain: float = 0.044
    """수신 목표가 파이널서드 깊숙이 들어갈수록 릴리스 확률에 더하는 최대값."""
    pass_release_through_gain: float = 0.044
    """실제 라인 뒤 through 강도에 따라 frame당 릴리스 확률에 더하는 최대값."""
    pass_release_through_scale: float = 0.08
    """``pass_release_through_gain``이 최대가 되는 through-strength 스케일."""
    pass_release_closing_gain: float = 0.0
    """수비가 **좁혀 오는 속도**로 릴리스를 앞당기는 최대 가산. 기본값은 0(꺼짐)이다.

    압박도(``pass_release_pressure_gain``)는 수비가 도달한 뒤에야 오르므로 '붙은 다음에
    찬다'가 된다는 관찰에서 넣은 항이다. 그러나 8시드 A/B에서 성공률은 81.2%(게인 0)
    대 83.3%(게인 0.075)로 표준오차 안이었고, 4시드에서는 부호가 반대로 뒤집혔다.
    반면 **패스 시도는 26% 증가**했고 이 방향만 표본마다 일관됐다. 패스 빈도는 이미
    인플레이 분당 22회로 실측 9.5회의 2.3배라, 그쪽을 더 미는 항을 켤 근거가 없다.

    빈도를 먼저 실측에 맞춘 뒤 다시 평가할 값이다. 그때는 이 게인만 올리면 된다.
    """
    pass_release_closing_scale: float = 3.5
    """``pass_release_closing_gain``이 최대가 되는 접근률 스케일(m/s 합산)."""
    quick_relay_probability: float = 0.18
    """안전한 낮은 인입 패스를 첫 접촉에 다시 낮게 연결할 기준 seeded 확률.

    실제 확률은 제어력·몸 방향·압박·되돌림 여부로 조절된다. 1.0 override는 테스트와
    전술 실험에서 확정 릴레이를 뜻하며 문맥 배율 뒤에도 1.0을 보존한다.
    """
    quick_relay_completion_floor: float = 0.72
    """원터치 전환을 허용할 예측 지상 레인 완성확률 하한."""
    quick_relay_min_incoming_speed_mps: float = 6.0
    """원터치 전환 후보로 보는 자기 팀 인입 패스의 최소 수평 공속(m/s)."""
    quick_relay_max_incoming_speed_mps: float = 18.0
    """상위 프로도 먼저 통제해야 하는 원터치 인입 공속 상한(m/s)."""
    quick_relay_low_skill_speed_fraction: float = 0.75
    """하위 프로가 최소~최대 원터치 공속 구간에서 사용할 수 있는 비율.

    기본 0.75면 6~18m/s 구간 중 15m/s까지이며, 상위 프로로 갈수록 18m/s까지
    연속적으로 열린다. 절대 감산값이 아니므로 사용자가 속도 범위를 좁혀도 역전되지 않는다.
    """
    quick_relay_alignment_floor: float = -0.25
    """달리는 수신자의 진행방향과 다음 패스방향 내적 하한. 정지 선수는 면제한다."""
    quick_relay_return_cos: float = -0.65
    """인입 진행축을 즉시 거슬러 보내는 반환 패스로 보는 방향 내적 경계."""
    quick_relay_skill_gain: float = 0.70
    """하·상위 프로 사이의 원터치 시도 확률 배율 증가폭."""
    quick_relay_return_scale: float = 0.25
    """인입 방향을 거의 그대로 거슬러 보내는 즉시 반환 패스의 확률 배율."""
    quick_relay_floor_skill_relief: float = 0.04
    """상위 프로가 원터치 레인에 적용받는 완성률 하한 완화의 총 폭."""
    pass_completion_floor: float = 0.58
    """일반 지상패스로 허용할 예측 레인 완성 확률 하한.

    현재 DFL 공통 다음-touch 기준 목표 성공률은 79.9%다.
    ``env INTERCEPT 112 vs provider 9`` 비교는 제공자의
    태클 시도/성공과 env의 탈취 라벨을 서로 다른 taxonomy로 센 잘못된 근거라 폐기했다.
    양쪽에 같은 정의를 적용한 소유 지속시간과 ``rule_policy.profile``의 다음-touch 성공률로
    경합 부담을 재며, 이 하한은 애초에 접전 레인으로 찌르는 빈도를 정책 쪽에서 제한한다.
    """
    pass_forward_completion_floor: float = 0.72
    """±60도 전방 지상패스에 요구하는 예측 완성확률 하한.

    2차 표본에서 전방 패스의 실제 다음-touch 성공은 4/24였지만 횡·후방은 19/30이었다.
    전방을 금지하지 않고 같은 ETA 통화에서 더 큰 안전 여유를 요구해 성공한 전진만 남긴다.
    """
    through_completion_floor: float = 0.52
    """킬패스에만 허용하는 더 낮은 예측 완성 확률 하한.

    동일 4시드·4분 paired profile에서 0.42→0.50은 인플레이 팀당 패스량을
    9.44→9.53/분(실측 9.45)으로 유지하면서 기준 호환 성공률을
    77.6→80.3%, 전진 방향 비중을 56.9→52.5%로 개선했다. 0.56은 성공률이
    79.3%로 되돌아가므로 더 엄격한 값이 단조롭게 좋은 것은 아니다. 현행 일반/킬패스
    하한 0.60/0.52는 DFL의 동일 다음-touch 성공률 79.9%를 목표로 한 값이다.
    """
    cross_wide_fraction: float = 0.34
    """크로스 캐리어로 보는 최소 절대 y / pitch half-width."""
    cross_start_fraction: float = 0.10
    """크로스를 허용하는 최소 공격 전진도 x / pitch half-length."""
    cross_gain: float = 1.55
    """박스 침투자에게 공급하는 크로스의 전술 가치 배율."""
    cross_preference_ratio: float = 0.38
    """일반 패스 가치 대비 이 비율 이상인 크로스를 우선 서비스한다."""
    cross_dribble_preference_ratio: float = 0.48
    """드리블 가치 대비 이 비율 이상인 실수신자 크로스를 우선 서비스한다."""
    cross_direct_preference_gain: float = 0.30
    """directness가 높을수록 크로스 우선비율 문턱을 낮추는 상대 조정 폭."""
    cross_control_floor: float = 0.24
    """러너 도달시간과 상대 선점을 함께 본 최소 크로스 수신 우위."""
    wide_progression_gain: float = 0.30
    """공격진영의 더 넓은 동료에게 전개하는 패스의 최대 가치 가산점."""
    curl_spin_min: float = 0.28
    curl_spin_max: float = 0.82
    curl_distance_start: float = 11.0
    curl_goal_fraction: float = 0.75
    """감아차기가 노리는 골문 반폭의 비율. 포스트·공 반경·적분 오차를 남긴 먼 포스트 목표다."""
    shot_noise_base_rad: float = 0.070
    """규칙 정책 슛의 최소 각오차 표준편차(rad) — 수평·수직 공통.

    K리그 2,670개 슛의 결말은 골대밖 43.4%(빗나감·골대맞음 포함)·차단 20.1%·
    유효 24.1%·골 10.4%다(event_rates.json). 이 모델에는 슛이 골문을 벗어나는
    경로가 조준 오차뿐이므로, 그 43%를 조준 오차가 감당해야 한다. 중앙값 슛 거리
    16.7m에서 조준점(골문 반폭의 ``curl_goal_fraction``)부터 포스트까지 0.92m,
    크로스바까지 1.54m이고, 등방 정규오차 σ_m≈1.5m가 두 방향 합쳐 약 43%를 낸다 —
    각도로는 0.09rad다. 여기에 xG·압박 항이 더해져 그 근처가 되도록 기저를 잡았다.
    이전 값 0.018은 16m에서 횡오차 0.29m뿐이라 골대밖이 7%도 되지 않았고, 실측
    900초·3시드에서 골킥이 **0건**이었다."""
    shot_noise_xg_rad: float = 0.030
    """낮은 xG에서 추가되는 각오차 표준편차 계수(rad).

    ``tactics.shot_xg``는 이름과 달리 **확률이 아니라 슛 품질 점수**다. 페널티
    스폿에서만 실측 xG와 맞고(0.769 대 0.76), 16m 수비1+GK에서 0.364 대 0.07,
    25m에서 0.407 대 0.03으로 3~15배 크다. 이 항은 그 점수의 상대 순서만 쓴다."""
    shot_noise_pressure_rad: float = 0.010
    """상대 압박도에 따라 추가되는 각오차 표준편차 계수(rad)."""
    retouch_retreat_distance: float = 6.0
    """세트피스 키커가 타인 접촉 전 확보하려는 공 반대쪽 후퇴 거리(m)."""
    retouch_retreat_power: float = 0.60
    """재터치 금지 선수가 공에서 물러날 때의 최소 이동 파워.

    이 후퇴는 대형 표류가 아니라 **커밋된 행동**이다(특히 스로어는 라인 밖에 서 있다).
    오프더볼 순항 파워를 실측 속도에 맞춰 낮추면서 이 복귀까지 함께 느려지면, 던진 선수가
    5 m 경계 밖에서 어정거린다. 압박·추격과 같은 부류로 분리해 둔다.
    """
    retouch_pitch_inset: float = 1.0
    """재터치 금지 목표의 피치 안 여유(m).

    일반 세트피스 후퇴 목표의 경계이자, 스로어가 터치라인에 수직으로 재진입한 뒤 일반
    오프볼 움직임으로 복귀하는 깊이다.
    """
    support_count: int = 3
    """캐리어 주변 지정 서포트 슬롯 수.

    실측에서 공 10m 안에 있는 **자기 팀** 아웃필더는 중앙값 2명이다(캐리어 포함).
    5명을 6m 반경에 불러 모으면 화면이 공 주위로 뭉치고 대형이 사라진다.
    """
    support_scale: float = 1.0
    """서포트 슬롯 오프셋 배율. 1.0이면 슬롯 원좌표(≈8~12m)를 그대로 쓴다."""
    support_progression_blend: float = 1.0
    """근접 지원 슬롯의 횡삼각형(0)→전진 대각선(1) 최대 보간값.

    실제 보간은 팀 ``directness``에 따라 아래 start→full 구간에서 0→이 값으로
    연속 증가한다. 따라서 점유형 팀은 횡삼각형을 보존하고 균형·직선형 팀은 같은
    K리그 최근접 거리 10.2/14.2m 안에서 전방 대각선 체크런을 만든다.
    """
    support_progression_directness_start: float = 0.15
    """전진 대각선 지원을 섞기 시작하는 팀 directness."""
    support_progression_directness_full: float = 0.50
    """``support_progression_blend``를 전부 적용하는 팀 directness."""
    support_run_power: float = 0.34
    """캐리어 주변 3인 지원자의 능동적 이동 파워.

    DFL에서 캐리어 최근접 3인의 속도 중앙값은 2.52m/s, 나머지 오프더볼은
    1.84m/s였다. 전체 ``offball_cruise``를 올리지 않고 국소 지원 역할만 구분한다.
    """
    box_run_power: float = 0.47
    """파이널서드의 니어·파포스트 침투와 컷백 준비 주행 파워."""
    support_motion_alignment_gain: float = 0.20
    """후보 위치가 지원자의 현재 이동 방향을 자연스럽게 이을 때의 가산점."""
    press_contain_distance: float = 2.2
    """1차 압박수가 캐리어를 '가두는' 기본 거리(m). 실측 수비팀 최근접 거리 p25가 2.75m다."""
    press_tackle_distance: float = 0.9
    """실제로 볼을 뺏으러 들어가는 거리(m). 연속적인 도전 의도가 높을 때만 이 간격까지 좁힌다."""
    tackle_hazard_base_per_s: float = 0.18
    """통제된 캐리어와 접촉권에 들어온 1차 압박수의 기본 태클 hazard(회/s).

    프레임 확률이 아니라 연속시간 hazard로 저장해 control FPS가 바뀌어도
    ``1-exp(-hazard*dt)``의 시도 계약이 보존된다.
    """
    tackle_hazard_danger_gain_per_s: float = 0.33
    """공이 자기 골문에 가까워질수록 더하는 최대 태클 hazard(회/s)."""
    tackle_hazard_instability_gain_per_s: float = 0.36
    """캐리어의 긴 터치·공과의 상대속도에 더하는 최대 태클 hazard(회/s)."""
    tackle_hazard_approach_gain_per_s: float = 0.21
    """닫히는 속도와 골사이드 접근각이 좋을 때 더하는 최대 태클 hazard(회/s)."""
    tackle_hazard_counterpress_gain_per_s: float = 0.18
    """소유를 잃은 뒤 3초 역압박 창에서 더하는 태클 hazard(회/s)."""
    tackle_hazard_aggression_gain_per_s: float = 0.15
    """팀 aggression=1에서 더하는 최대 태클 hazard(회/s)."""
    tackle_hazard_contact_gain_per_s: float = 0.90
    """1차 압박수가 실제 challenge 접촉권 깊숙이 들어왔을 때 더하는 hazard(회/s).

    전역 시도율을 올리는 항이 아니다. 컨테인 거리에서는 0이고, 공과 약 0.9m까지
    붙은 경우에만 최대가 되어 수 초간 몸만 겹치고 발을 넣지 않는 꼬리를 줄인다.
    """
    tackle_hazard_cap_per_s: float = 1.2
    """통제된 캐리어 상대 태클 hazard 상한(회/s)."""
    tackle_entry_hazard_scale_per_s: float = 0.75
    """태클 hazard를 컨테인↔접촉 거리의 연속적인 진입 강도로 바꾸는 스케일(회/s)."""
    tackle_instability_speed_scale_mps: float = 4.0
    """공과 캐리어의 상대속도를 긴 터치 점수 1로 접는 속도(m/s)."""
    tackle_closing_speed_scale_mps: float = 3.5
    """압박수의 캐리어 접근속도를 접근 품질 1로 접는 속도(m/s)."""
    interception_hazard_base_per_s: float = 3.0
    """상대 소유 표기 아래 이동 중인 비통제 공에 발을 내는 기본 hazard(회/s)."""
    interception_hazard_speed_gain_per_s: float = 2.0
    """공속이 ``press_loose_ball_speed``에 도달할 때 더하는 인터셉트 hazard(회/s)."""
    interception_hazard_approach_gain_per_s: float = 1.0
    """공의 진행 경로를 골사이드에서 닫을 때 더하는 인터셉트 hazard(회/s)."""
    interception_hazard_cap_per_s: float = 6.0
    """비통제 공 인터셉트 hazard 상한(회/s)."""
    challenge_control_touch_speed_mps: float = 3.0
    """일반 태클·인터셉트가 소유 정착을 위해 만드는 낮은 첫 터치 공속(m/s).

    challenge 접촉 자체를 전방 롱킥으로 쓰지 않는다. 이 속도는 neutral loose-ball의
    ``ground_control_touch_speed_mps``와 별도로 보정할 수 있어 DFL 탈취 뒤 공속 분포를
    독립적으로 수용한다.
    """
    challenge_outlet_max_distance: float = 18.0
    """탈취 첫 접촉에서 낮은 안전 출구로 바로 연결할 최대 수신거리(m)."""
    challenge_outlet_completion_floor: float = 0.78
    """탈취 첫 접촉의 즉시 낮은 출구가 요구하는 예측 완성확률 하한."""
    challenge_outlet_max_incoming_speed_mps: float = 10.0
    """낮은 출구로 방향 전환할 수 있는 인입 공속 상한(m/s).

    더 빠른 공은 무리한 원터치 패스로 꺾지 않고 먼저 통제한다.
    """
    challenge_clearance_depth_fraction: float = 0.72
    """긴급 걷어내기를 검토하기 시작하는 자기 골 방향 깊이 ``-x/hx``.

    challenge뿐 아니라 통제 캐리어의 무옵션 긴급 걷어내기에도 같은 경계를 써,
    자기 진영이라는 이유만으로 중원 가까이서 롱볼을 택하지 않게 한다.
    """
    challenge_clearance_pressure_min: float = 0.78
    """안전 출구 없는 깊은 접촉이 실제 걷어내기가 되기 위한 최소 압박도."""
    press_retain_distance: float = 5.0
    """개시선 밖에서도 이미 붙은 1차 압박을 유지할 팀 최근접-공 거리(m).

    DFL 수비팀 최근접 거리 중앙값 4.8m를 포괄한다. 이 히스테리시스가 없으면 캐리어가
    개시선을 넘는 한 프레임에 압박수가 즉시 블록으로 복귀해 눈앞의 소유자를 놓친다.
    """
    press_cover_distance: float = 4.8
    """balanced 팀의 2차 커버가 공의 골사이드에 확보할 간격(m).

    2차 선수는 이 지점에서 다음 패스와 튀어나온 공을 준비하며 직접 킥/태클은 제출하지 않는다.
    """
    box_mark_max_runners: int = 3
    """자기 박스 부근에서 동시에 전담할 최대 상대 러너 수."""
    box_mark_runner_margin: float = 5.0
    """페널티구역 밖에서도 박스 침투 러너로 미리 보는 공간 여유(m)."""
    box_mark_ball_margin: float = 14.0
    """공이 페널티구역 앞 이 거리 안에 왔을 때 다중 박스 수비를 활성화한다(m)."""
    box_mark_goal_side_distance: float = 1.6
    """박스 마커가 담당 러너보다 자기 골 쪽에 확보하는 간격(m)."""
    press_engagement_x_base: float = 12.2
    """balanced 팀의 1차 압박 개시선(자기 공격 프레임 x, m).

    K리그 45경기 직접 압박 2,627건의 위치 p75=+12.18m를 쓴다. 이 선보다 상대 골 쪽의
    느린 통제 빌드업에는 달려들지 않고 데이터 위치장의 수비 블록을 유지한다.
    """
    press_engagement_aggression_gain: float = 45.0
    """aggression이 0.5에서 1만큼 변할 때 압박 개시선을 옮기는 거리(m)."""
    press_engagement_line_gain: float = 20.0
    """line height가 0.5에서 1만큼 변할 때 압박 개시선을 옮기는 거리(m).

    두 스타일 항을 합치면 gegenpress는 약 +37m(실측 직접 압박 p95=+34m),
    park-the-bus는 약 -1.6m(실측 중앙값=-2.9m)에서 압박을 시작한다.
    """
    press_loose_ball_speed: float = 3.5
    counterpress_window_s: float = 3.0
    """소유권을 잃은 뒤 역압박 역할을 유지하는 실시간(s).

    DFL 소유 전환과 같은 3초 회수 정의를 쓴다. 시간이 지나면 공을 무한정
    쫓지 않고 위치장 수비 블록으로 복귀한다.
    """
    counterpress_retain_distance: float = 8.0
    """전환 창 내에서만 적용하는 1차 역압박 최대 공 거리(m)."""
    counterpress_cover_distance: float = 6.5
    """전환 시 2차 선수가 즉시 태클 대신 탈출 패스를 막는 기준 거리(m)."""
    counterpress_rest_distance: float = 11.0
    """전환 시 3차 선수가 공 뒤에 남겨 두는 rest-defense 거리(m)."""
    secure_possession_window_s: float = 0.80
    """탈취 직후 트랩·회피·안전 출구를 우선하는 실시간(s)."""
    secure_pass_completion_floor: float = 0.76
    """탈취 직후 안전 출구로 인정하는 지상 패스 예측 완성률 하한."""
    secure_release_probability: float = 0.30
    """안정화 창에서 압박을 받을 때 안전 패스를 내는 frame당 하한 hazard."""
    dribble_evade_pressure: float = 0.52
    """전방이 막히지 않아도 다방향 회피 드리블을 검토하는 합산 압박도."""
    dribble_escape_space_weight: float = 0.52
    """다방향 드리블 후보의 빈 공간 기본 가중치."""
    dribble_escape_progress_weight: float = 0.34
    """다방향 드리블 후보의 전진 기본 가중치."""
    dribble_escape_momentum_weight: float = 0.14
    """현재 주행 방향을 부드럽게 이어가는 드리블 후보 가중치."""
    dribble_escape_skill_tradeoff: float = 0.14
    """하위 프로는 공간, 상위 프로는 전진을 더 택하게 하는 가중치 이동 폭."""
    unattended_ball_speed: float = 0.5
    """이 속력(m/s) 이하이고 아무도 곁에 없으면 소유 표기와 무관하게 루즈볼로 다룬다."""
    unattended_ball_radius: float = 5.0
    """방치 판정 거리(m). env의 ``possession_release_radius``와 같은 뜻이며, env가 소유권을
    중립으로 되돌리기 전이라도 정책이 스스로 공을 주우러 가게 하는 이중 안전장치다."""
    """개시선과 무관하게 이동 중인 비통제 공으로 간주해 압박할 최소 수평 공속(m/s)."""
    dribble_cone_length: float = 5.0
    width_base: float = 1.0
    """실측 위치장 y를 그대로 쓰는 기준 폭 배율(width 스타일 0.5에서 1.0)."""
    width_gain: float = 0.22
    """width 스타일이 ±0.5 벗어날 때 폭 배율을 바꾸는 폭. 0.22면 [0.78, 1.22]."""
    anchor_spread_y_attack: float = 1.30
    """공격 국면에서 위치장 y(폭)를 넓히는 배율.

    위치장은 **조건부 평균**이라 순간 산포보다 항상 좁다(젠슨 효과). 실측 팀 폭 중앙값은
    공격 35.4 m인데 표를 그대로 따르면 27.0 m가 나왔다 — 그 비(1.31)를 그대로 쓴다.
    """
    anchor_spread_y_defend: float = 1.18
    """수비 국면 폭 배율. 실측 28.6 m / 표 추종 24.1 m."""
    anchor_spread_x_attack: float = 1.35
    """공격 국면에서 팀 중심 기준 깊이 산포 배율. 실측 28.2 m / 표 추종 20.7 m."""
    anchor_spread_x_defend: float = 1.15
    """수비 국면 깊이 산포 배율. 실측 22.3 m / 표 추종 19.3 m."""
    style_line_span: float = 12.0
    """line 스타일이 블록 전체 x를 옮기는 폭(m). ±(span/2)까지 하이/로우 블록.

    위치장은 스타일 평균을 담고 있으므로 이 값은 그 위의 편차다. 게겐프레스(line=0.85)와
    파크더버스(0.15) 사이가 약 8.4 m 차이 나는데, 실측 팀 간 수비라인 높이 차이의 대략적인
    폭이다. 더 키우면 표가 담은 국면 구조를 스타일이 덮어써 대형이 축구가 아니게 된다.
    """
    arrival_radius: float = 6.5
    """오프더볼 도착 감속 반경(m). 목표가 이보다 멀면 최대 파워, 가까우면 선형 감속.

    실측 선수 속도는 중앙값 1.97 m/s이고 절반 이상이 2 m/s 미만이다(K리그 30경기).
    목표를 향해 항상 전력으로 달리는 정책은 이 분포를 절대 만들 수 없다 — 사람이 보기에
    '전원이 계속 뛰는' 비현실의 가장 큰 원인이다. 도착 감속이 걷기·조깅 구간을 만든다.
    """
    offball_cruise: float = 0.20
    """공에 커밋하지 않은 오프더볼 이동의 파워 상한. 1.0은 스프린트다.

    실측 대조로 정한 값이다. 0.72에서는 시뮬 속도 중앙값이 3.75 m/s(실측 2.05), 1인당
    이동거리가 21.9 km였다. 0.20은 새 역할별 추격 파워와 함께 K리그 30경기의 평균
    12.77 km/90분에 근접하도록 낮춘 값이다.
    """
    offball_surge_cap: float = 0.47
    """오프더볼 복귀·침투 주행이 올라갈 수 있는 파워 상한.

    1.0(전력)까지 열어 두면 스프린트가 90분당 101회 나온다(실측 20회). 실측 스프린트는
    드물고 짧다 — 대부분의 '빠른 이동'은 조깅~러닝 구간이다.

    **이 값으로 질주 빈도를 맞출 수는 없다.** 질주 판정이 7 m/s인데 기본 로스터는
    vmax가 전원 7.96으로 같아서, 필요한 파워가 7.0/7.96 = 0.879 **한 점**에 몰려 있다.
    즉 이 상한은 0.879 아래면 0명, 위면 22명인 이분 스위치다. 실측(60 시뮬분):

        0.47 → 90분당 11.3회 · 0.69 → 11.3회 · 0.92 → 43.5회 · 1.0 → 101회
        K리그1 30경기 실측 21.7회

    0.69를 시도했다가 되돌렸다. 질주는 전혀 늘지 않으면서(문턱 아래) 이동 부하만 늘어
    하프타임 stamina가 0.589에서 0.413으로 떨어졌고, 그 결과 교체가 더 이르게 몰렸다
    (실측 교체 중앙값 68.3분인데 50분 안에 다섯 장을 소진). 순손실이다.

    이 지표를 닫으려면 손잡이가 아니라 **로스터에 속도 이질성**이 있어야 한다 — 실제
    선수의 최고속도는 7.5~9.5 m/s로 흩어져 있고, 그래야 "빠른 윙어는 질주하고 느린
    센터백은 안 한다"가 표현된다. 로스터는 사용자가 정하는 것이므로 여기서 바꾸지 않고
    사실만 남긴다.
    """
    offball_walk_floor: float = 0.06
    """제자리에서도 미세 조정을 계속하는 최소 파워(완전 정지 프레임 방지)."""
    carrier_control_power: float = 0.45
    """공이 2.5 m 안에 있을 때 캐리어가 오버런하지 않는 통제 이동 파워."""
    carrier_chase_power: float = 0.84
    """공이 2.5 m보다 멀어진 캐리어의 회수 이동 파워."""
    press_run_power: float = 0.78
    """볼 압박수의 이동 파워. 장기 profile에서 전체 스프린트는 실측의 0.59배다."""
    press_cover_run_power: float = 0.62
    """2차 압박 커버의 이동 파워. 1차보다 낮춰 더블 태클·스웜이 되지 않게 한다."""
    mark_run_power: float = 0.70
    """전담 마커가 예측 위협 위치를 따라가는 이동 파워."""
    receive_run_power: float = 0.86
    """의도적 패스의 예측 수신 주자가 수신점으로 이동하는 파워."""
    loose_run_power: float = 0.78
    """팀별 ETA 1순위 루즈볼 추격수의 이동 파워."""
    loose_cover_distance: float = 5.0
    """ETA 2순위 루즈볼 선수가 예상 접촉점 골사이드에 확보할 세컨드볼 간격(m)."""
    loose_cover_run_power: float = 0.64
    """루즈볼 2순위 커버의 이동 파워. 직접 볼 접촉은 ETA 1순위만 시도한다."""
    aerial_run_power: float = 0.78
    """공중볼 낙하지점의 최근접 경합수가 이동하는 파워."""
    aerial_cover_run_power: float = 0.64
    """직접 경합 ETA 창을 놓친 수비수의 골사이드 세컨드볼 이동 파워."""
    overlap_start_x: float = 2.0
    """오버래핑 풀백이 발동하는 캐리어 최소 전진도(m)."""
    overlap_run_ahead: float = 11.0
    """오버래핑 풀백이 캐리어보다 앞서 달려 나가는 거리(m)."""
    wide_progress_gate: float = 0.10
    """측면 슬롯이 폭을 유지하기 시작하는 최소 공 전진도 x / pitch half-length."""
    gk_prediction_decel: float = 5.0
    anti_clump_radius: float = 9.0
    anti_clump_gain: float = 1.5
    restart_wait_margin: float = 0.75
    """재개 제한구역의 법정 경계보다 정책이 추가로 확보하는 대기 여유(m).

    환경의 ``legal_margin_floor``는 수치 재침범만 막는 최소 투영 여유이고, 이 값은
    다음 control frame의 정책 이동·관성으로 다시 제한구역에 들어가지 않게 하는 행동 여유다.
    """
    throwin_support_min_distance: float = 4.5
    """우리 스로인 비키커가 재개 지점에서 확보할 최소 전술 거리(m).

    K리그 4,265개 스로인에서 재개 지점과 가장 가까운 동료 거리의 p05가 4.49 m였다.
    이벤트 행위자(실제 스로어)를 위치장에 포함하면 그 역할의 조건부 중심이 공 위로 오염되고,
    런타임이 다른 역할을 스로어로 고를 때 두 선수가 라인에 겹친다. 재적합의 행위자 제외와
    함께 이 하한을 두어 희소-cell fallback에서도 같은 문제가 되살아나지 않게 한다.
    """
    restart_target_slowdown_radius: float = 4.0
    """재개 중 합법 대기점에 접근할 때 이동 파워를 선형 감쇠하는 거리(m).

    2 m는 관성 때문에 목표를 몇 미터 지나쳤다가 되돌아오는 진동을 만든다 — 벽·포스트처럼
    수십 cm 단위로 서야 하는 배치에서 특히 티가 난다. 4 m면 마지막 구간을 걸어 들어간다.
    """

    def __post_init__(self):
        if not isinstance(self.use_shot_solver, bool):
            raise TypeError("use_shot_solver must be bool")

        # Public policy configuration follows its annotations strictly.  In
        # particular bool is a Python Integral/Real subclass and previously
        # slipped into every float field (``shoot_gain=True``).  Accept NumPy
        # scalar kinds, but canonicalise them so equivalent configs have the
        # same stored Python types and downstream JAX weak-type behaviour.
        integer_fields = {
            "style_sample_count",
            "support_count",
            "receive_prediction_samples",
            "box_mark_max_runners",
        }
        for spec in fields(self):
            name = spec.name
            value = getattr(self, name)
            if name == "use_shot_solver":
                continue
            if name in integer_fields:
                if not isinstance(value, numbers.Integral) or isinstance(
                    value, (bool,)
                ):
                    raise TypeError(f"{name} must be an integer scalar")
                object.__setattr__(self, name, int(value))
            elif spec.type is float:
                if not isinstance(value, numbers.Real) or isinstance(
                    value, (bool,)
                ):
                    raise TypeError(f"{name} must be a real non-boolean scalar")
                canonical = float(value)
                if not math.isfinite(canonical):
                    raise ValueError(f"{name} must be finite")
                # RulePolicy executes in the environment's float32 JAX
                # contract.  A Python-finite value outside that range used to
                # pass validation and then become +/-inf (or zero) while the
                # action graph was traced.  In particular an overflowing
                # anti_clump_gain makes ``inf * 0`` and emits NaN movement.
                magnitude = abs(canonical)
                if magnitude > _RULE_POLICY_SCALAR_ABS_MAX or (
                    magnitude != 0.0 and magnitude < _FLOAT32_MIN_SUBNORMAL
                ):
                    raise ValueError(
                        f"{name} must remain in the numerically safe float32 "
                        f"policy range (absolute value <= "
                        f"{_RULE_POLICY_SCALAR_ABS_MAX:g})"
                    )
                canonical = 0.0 if canonical == 0.0 else canonical
                object.__setattr__(self, name, canonical)

        if not 1 <= self.style_sample_count <= MAX_RULE_POLICY_SAMPLE_COUNT:
            raise ValueError(
                "style_sample_count must be an integer in "
                f"[1, MAX_RULE_POLICY_SAMPLE_COUNT={MAX_RULE_POLICY_SAMPLE_COUNT}]"
            )
        if self.support_count <= 0 or self.support_count > 5:
            raise ValueError(
                "support_count must be an integer in [1, 5] (the policy has "
                "five distinct support slots)"
            )
        if not 2 <= self.receive_prediction_samples <= MAX_RULE_POLICY_SAMPLE_COUNT:
            raise ValueError(
                "receive_prediction_samples must be an integer in "
                f"[2, MAX_RULE_POLICY_SAMPLE_COUNT={MAX_RULE_POLICY_SAMPLE_COUNT}]"
            )
        if not 1 <= self.box_mark_max_runners <= 5:
            raise ValueError("box_mark_max_runners must be an integer in [1, 5]")
        numeric = tuple(
            value for name, value in vars(self).items()
            if name not in {"use_shot_solver", "style_sample_count", "support_count"}
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("RulePolicy values must be finite")
        if (
            self.shoot_gain <= 0.0
            or self.professional_control_floor <= 0.0
            or self.professional_control_ceiling <= 0.0
            or self.professional_reach_floor_m <= 0.0
            or self.professional_reach_ceiling_m <= 0.0
            or self.receiver_control_value_gain <= 0.0
            or self.cross_receiver_ability_gain <= 0.0
            or self.dribble_control_retention_gain <= 0.0
            or self.execution_control_noise_gain <= 0.0
            or self.aerial_control_floor_relief <= 0.0
            or self.support_scale <= 0.0
            or self.support_run_power <= 0.0
            or self.box_run_power <= 0.0
            or self.support_motion_alignment_gain <= 0.0
            or self.dribble_cone_length <= 0.0
            or self.gk_prediction_decel <= 0.0
            or self.anti_clump_radius <= 0.0
            or self.anti_clump_gain <= 0.0
            or self.restart_wait_margin <= 0.0
            or self.throwin_support_min_distance <= 0.0
            or self.restart_target_slowdown_radius <= 0.0
            or self.pass_lead_time_cap <= 0.0
            or self.pass_lead_distance_cap <= 0.0
            or self.receive_prediction_horizon_s <= 0.0
            or self.shot_approach_margin <= 0.0
            or self.through_min_progress <= 0.0
            or self.through_shoulder_gap_m <= 0.0
            or self.through_bonus <= 0.0
            or self.through_run_ahead <= 0.0
            or self.forward_line_margin <= 0.0
            or self.forward_run_min_gap <= 0.0
            or self.combination_gain <= 0.0
            or self.possession_combination_gain <= 0.0
            or self.pass_receiver_movement_gain <= 0.0
            or self.pass_space_creation_gain <= 0.0
            or self.pass_shot_chain_gain <= 0.0
            or self.continuation_support_lead_weight <= 0.0
            or self.continuation_support_lead_cap_s <= 0.0
            or self.continuation_min_distance <= 0.0
            or self.continuation_max_distance <= 0.0
            or self.continuation_backward_tolerance <= 0.0
            or self.direct_distance_gain <= 0.0
            or self.possession_distance_penalty <= 0.0
            or self.loft_distance_base <= 0.0
            or self.loft_distance_direct_relief <= 0.0
            or self.loft_control_floor <= 0.0
            or self.aerial_control_max_height <= 0.0
            or self.pressured_aerial_control_max_height <= 0.0
            or self.aerial_control_pressure_max <= 0.0
            or self.aerial_duel_eta_window_s <= 0.0
            or self.aerial_defender_cover_distance <= 0.0
            or self.aerial_pass_min_distance <= 0.0
            or self.aerial_pass_max_distance <= 0.0
            or self.aerial_pass_min_speed_mps <= 0.0
            or self.aerial_pass_max_speed_mps <= 0.0
            or self.aerial_shot_speed_mps <= 0.0
            or self.aerial_clear_speed_mps <= 0.0
            or self.gk_long_min_distance <= 0.0
            or self.gk_long_min_progress <= 0.0
            or self.gk_foot_pass_max_distance <= 0.0
            or self.quick_relay_min_incoming_speed_mps <= 0.0
            or self.quick_relay_max_incoming_speed_mps <= 0.0
            or self.quick_relay_low_skill_speed_fraction <= 0.0
            or self.quick_relay_skill_gain <= 0.0
            or self.quick_relay_return_scale <= 0.0
            or self.quick_relay_floor_skill_relief <= 0.0
            or self.cross_gain <= 0.0
            or self.wide_progression_gain <= 0.0
            or self.curl_distance_start <= 0.0
            or self.shot_noise_base_rad <= 0.0
            or self.shot_noise_xg_rad <= 0.0
            or self.shot_noise_pressure_rad <= 0.0
            or self.retouch_retreat_distance <= 0.0
            or self.retouch_pitch_inset <= 0.0
            or self.style_line_span <= 0.0
            or self.arrival_radius <= 0.0
            or self.overlap_run_ahead <= 0.0
            or self.press_contain_distance <= 0.0
            or self.press_tackle_distance <= 0.0
            or self.tackle_hazard_cap_per_s <= 0.0
            or self.tackle_entry_hazard_scale_per_s <= 0.0
            or self.tackle_instability_speed_scale_mps <= 0.0
            or self.tackle_closing_speed_scale_mps <= 0.0
            or self.interception_hazard_cap_per_s <= 0.0
            or self.challenge_control_touch_speed_mps <= 0.0
            or self.challenge_outlet_max_distance <= 0.0
            or self.challenge_outlet_max_incoming_speed_mps <= 0.0
            or self.press_retain_distance <= 0.0
            or self.press_cover_distance <= 0.0
            or self.box_mark_runner_margin <= 0.0
            or self.box_mark_ball_margin <= 0.0
            or self.box_mark_goal_side_distance <= 0.0
            or self.counterpress_window_s <= 0.0
            or self.counterpress_retain_distance <= 0.0
            or self.counterpress_cover_distance <= 0.0
            or self.counterpress_rest_distance <= 0.0
            or self.secure_possession_window_s <= 0.0
            or self.press_engagement_aggression_gain <= 0.0
            or self.press_engagement_line_gain <= 0.0
            or self.press_loose_ball_speed <= 0.0
            or self.unattended_ball_speed <= 0.0
            or self.unattended_ball_radius <= 0.0
            or self.pass_min_distance <= 0.0
            or self.pass_contact_clearance <= 0.0
            or self.pass_lane_intercept_radius <= 0.0
            or self.pass_lane_reaction_s <= 0.0
            or self.pass_lane_defender_acceleration_mps2 <= 0.0
            or self.pass_receive_ahead_margin <= 0.0
            or self.drive_arrive_speed_mps <= 0.0
            or self.ground_control_touch_speed_mps <= 0.0
            or self.receive_control_lead_speed_mps <= 0.0
            or self.settled_ball_speed_mps <= 0.0
            or self.carried_release_speed_mps <= 0.0
            or self.retouch_retreat_power <= 0.0
            or self.pass_release_cost <= 0.0
            or self.pass_pressure_scale <= 0.0
            or self.unpressured_backpass_cost <= 0.0
            or self.backpass_distance_scale <= 0.0
            or self.pass_release_through_scale <= 0.0
            or self.pass_release_closing_scale <= 0.0
            or self.pass_release_closing_gain < 0.0
            or self.dribble_evade_pressure <= 0.0
            or self.dribble_escape_space_weight <= 0.0
            or self.dribble_escape_progress_weight <= 0.0
            or self.dribble_escape_momentum_weight <= 0.0
            or self.dribble_escape_skill_tradeoff <= 0.0
            or self.offball_cruise <= 0.0
            or self.carrier_control_power <= 0.0
            or self.carrier_chase_power <= 0.0
            or self.press_run_power <= 0.0
            or self.press_cover_run_power <= 0.0
            or self.mark_run_power <= 0.0
            or self.receive_run_power <= 0.0
            or self.loose_run_power <= 0.0
            or self.loose_cover_distance <= 0.0
            or self.loose_cover_run_power <= 0.0
            or self.aerial_run_power <= 0.0
            or self.aerial_cover_run_power <= 0.0
        ):
            raise ValueError("RulePolicy gains, scales, and distances must be positive")
        fractions = (
            self.professional_control_floor,
            self.professional_control_ceiling,
            self.receiver_control_value_gain,
            self.cross_receiver_ability_gain,
            self.dribble_control_retention_gain,
            self.execution_control_noise_gain,
            self.aerial_control_floor_relief,
            self.wide_progress_gate,
            self.offball_cruise,
            self.offball_surge_cap,
            self.offball_walk_floor,
            self.support_progression_blend,
            self.support_progression_directness_start,
            self.support_progression_directness_full,
            self.support_run_power,
            self.box_run_power,
            self.carrier_control_power,
            self.carrier_chase_power,
            self.press_run_power,
            self.press_cover_run_power,
            self.mark_run_power,
            self.receive_run_power,
            self.loose_run_power,
            self.loose_cover_run_power,
            self.aerial_run_power,
            self.aerial_cover_run_power,
            self.pass_lead_velocity_weight,
            self.continuation_support_lead_weight,
            self.through_gap_weight,
            self.through_shoulder_cue_weight,
            self.loft_control_floor,
            self.aerial_pass_completion_floor,
            self.aerial_pass_launch01,
            self.aerial_clear_launch01,
            self.gk_long_directness_threshold,
            self.gk_buildup_gain,
            self.gk_foot_pass_pressure_max,
            self.cross_wide_fraction,
            self.cross_start_fraction,
            self.cross_preference_ratio,
            self.cross_dribble_preference_ratio,
            self.cross_direct_preference_gain,
            self.cross_control_floor,
            self.curl_spin_min,
            self.curl_spin_max,
            self.curl_goal_fraction,
            self.pass_completion_floor,
            self.pass_forward_completion_floor,
            self.through_completion_floor,
            self.first_time_shot_xg,
            self.shot_approach_xg,
            self.shot_min_xg,
            self.retouch_retreat_power,
            self.pass_pressure_relief,
            self.deep_backpass_relief,
            self.open_dribble_gain,
            self.pass_target_distance_penalty,
            self.pass_progress_base,
            self.pass_progress_direct_gain,
            self.pass_receiver_movement_gain,
            self.pass_space_creation_gain,
            self.pass_shot_chain_gain,
            self.support_motion_alignment_gain,
            self.pass_forward_direction_cost,
            self.pass_short_forward_bonus,
            self.pass_sideways_direction_bonus,
            self.pass_backward_direction_bonus,
            self.pass_backward_recycle_floor,
            self.loft_force_directness,
            self.pass_release_base_probability,
            self.pass_release_pressure_gain,
            self.contact_release_probability,
            self.pass_release_final_third_gain,
            self.pass_release_through_gain,
            self.pass_release_closing_gain,
            self.quick_relay_probability,
            self.quick_relay_completion_floor,
            self.quick_relay_low_skill_speed_fraction,
            self.quick_relay_skill_gain,
            self.quick_relay_return_scale,
            self.quick_relay_floor_skill_relief,
            self.secure_pass_completion_floor,
            self.secure_release_probability,
            self.dribble_evade_pressure,
            self.dribble_escape_space_weight,
            self.dribble_escape_progress_weight,
            self.dribble_escape_momentum_weight,
            self.dribble_escape_skill_tradeoff,
            self.challenge_outlet_completion_floor,
            self.challenge_clearance_depth_fraction,
            self.challenge_clearance_pressure_min,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("RulePolicy pitch fractions must lie in [0, 1]")
        if (
            self.support_progression_directness_start
            >= self.support_progression_directness_full
        ):
            raise ValueError(
                "support_progression_directness_start must be smaller than "
                "support_progression_directness_full"
            )
        if self.professional_control_floor >= self.professional_control_ceiling:
            raise ValueError(
                "professional_control_floor must be below "
                "professional_control_ceiling"
            )
        if self.professional_reach_floor_m >= self.professional_reach_ceiling_m:
            raise ValueError(
                "professional_reach_floor_m must be below "
                "professional_reach_ceiling_m"
            )
        if not -1.0 <= self.quick_relay_alignment_floor <= 1.0:
            raise ValueError("quick_relay_alignment_floor must lie in [-1, 1]")
        if not -1.0 <= self.quick_relay_return_cos <= 1.0:
            raise ValueError("quick_relay_return_cos must lie in [-1, 1]")
        if not (
            self.offball_walk_floor
            <= self.offball_cruise
            <= self.offball_surge_cap
        ):
            raise ValueError(
                "off-ball powers must satisfy "
                "offball_walk_floor <= offball_cruise <= offball_surge_cap"
            )
        if self.press_retain_distance < self.press_contain_distance:
            raise ValueError(
                "press_retain_distance must not be below press_contain_distance"
            )
        if self.counterpress_retain_distance < self.press_retain_distance:
            raise ValueError(
                "counterpress_retain_distance must not be below "
                "press_retain_distance"
            )
        if self.press_cover_distance <= self.press_tackle_distance:
            raise ValueError(
                "press_cover_distance must exceed press_tackle_distance"
            )
        if self.challenge_outlet_max_distance <= self.pass_min_distance:
            raise ValueError(
                "challenge_outlet_max_distance must exceed pass_min_distance"
            )
        tackle_hazard_terms = (
            self.tackle_hazard_base_per_s,
            self.tackle_hazard_danger_gain_per_s,
            self.tackle_hazard_instability_gain_per_s,
            self.tackle_hazard_approach_gain_per_s,
            self.tackle_hazard_counterpress_gain_per_s,
            self.tackle_hazard_aggression_gain_per_s,
            self.tackle_hazard_contact_gain_per_s,
            self.interception_hazard_base_per_s,
            self.interception_hazard_speed_gain_per_s,
            self.interception_hazard_approach_gain_per_s,
        )
        if any(value < 0.0 for value in tackle_hazard_terms):
            raise ValueError("tackle/interception hazard terms must be non-negative")
        if self.tackle_hazard_cap_per_s < self.tackle_hazard_base_per_s:
            raise ValueError(
                "tackle_hazard_cap_per_s must not be below the base hazard"
            )
        if (
            self.pressured_aerial_control_max_height
            > self.aerial_control_max_height
        ):
            raise ValueError(
                "pressured_aerial_control_max_height must not exceed "
                "aerial_control_max_height"
            )
        if self.aerial_pass_min_distance >= self.aerial_pass_max_distance:
            raise ValueError(
                "aerial_pass_min_distance must be smaller than "
                "aerial_pass_max_distance"
            )
        if self.aerial_pass_min_speed_mps > self.aerial_pass_max_speed_mps:
            raise ValueError(
                "aerial_pass_min_speed_mps must not exceed "
                "aerial_pass_max_speed_mps"
            )
        if (
            self.quick_relay_min_incoming_speed_mps
            >= self.quick_relay_max_incoming_speed_mps
        ):
            raise ValueError(
                "quick_relay_min_incoming_speed_mps must be below "
                "quick_relay_max_incoming_speed_mps"
            )
        aerial_floor_low = (
            self.aerial_pass_completion_floor
            - 0.5 * self.aerial_control_floor_relief
        )
        aerial_floor_high = (
            self.aerial_pass_completion_floor
            + 0.5 * self.aerial_control_floor_relief
        )
        if not (0.0 <= aerial_floor_low <= aerial_floor_high <= 1.0):
            raise ValueError(
                "ability-adjusted aerial pass completion floor must lie in [0, 1]"
            )
        if (
            self.interception_hazard_cap_per_s
            < self.interception_hazard_base_per_s
        ):
            raise ValueError(
                "interception_hazard_cap_per_s must not be below the base hazard"
            )
        if not 0.0 <= self.shot_commit_xg <= 1.0:
            raise ValueError("shot_commit_xg must lie in [0, 1]")
        if not 0.0 <= self.first_time_shot_xg <= 1.0:
            raise ValueError("first_time_shot_xg must lie in [0, 1]")
        if self.first_time_shot_distance <= 0.0:
            raise ValueError("first_time_shot_distance must be positive")
        if self.curl_spin_min > self.curl_spin_max:
            raise ValueError("curl_spin_min must not exceed curl_spin_max")
        if self.through_completion_floor > self.pass_completion_floor:
            raise ValueError(
                "through_completion_floor must not exceed pass_completion_floor"
            )
        if self.pass_forward_completion_floor < self.pass_completion_floor:
            raise ValueError(
                "pass_forward_completion_floor must not be below "
                "pass_completion_floor"
            )
        if self.quick_relay_completion_floor < self.pass_completion_floor:
            raise ValueError(
                "quick_relay_completion_floor must not be below "
                "pass_completion_floor"
            )
        quick_relay_floor_low = (
            self.quick_relay_completion_floor
            - 0.5 * self.quick_relay_floor_skill_relief
        )
        quick_relay_floor_high = (
            self.quick_relay_completion_floor
            + 0.5 * self.quick_relay_floor_skill_relief
        )
        if quick_relay_floor_low < self.pass_completion_floor:
            raise ValueError(
                "ability-adjusted quick relay floor must not fall below "
                "pass_completion_floor"
            )
        if quick_relay_floor_high > 1.0:
            raise ValueError(
                "ability-adjusted quick relay floor must not exceed 1"
            )
        if self.secure_pass_completion_floor < self.pass_completion_floor:
            raise ValueError(
                "secure_pass_completion_floor must not be below "
                "pass_completion_floor"
            )
        if self.gk_foot_pass_max_distance <= self.pass_min_distance:
            raise ValueError(
                "gk_foot_pass_max_distance must exceed pass_min_distance"
            )
        if self.carried_release_speed_mps < self.settled_ball_speed_mps:
            raise ValueError(
                "carried_release_speed_mps must not be below "
                "settled_ball_speed_mps"
            )
        if self.carried_release_speed_mps <= self.ground_control_touch_speed_mps:
            raise ValueError(
                "carried_release_speed_mps must exceed "
                "ground_control_touch_speed_mps"
            )
        if self.continuation_max_distance <= self.continuation_min_distance:
            raise ValueError(
                "continuation_max_distance must exceed "
                "continuation_min_distance"
            )
        if not -52.5 <= self.press_engagement_x_base <= 52.5:
            raise ValueError("press_engagement_x_base must lie within the pitch length")
        if self.loft_distance_direct_relief >= self.loft_distance_base:
            raise ValueError(
                "loft_distance_direct_relief must be smaller than loft_distance_base"
            )

@dataclass(frozen=True)
class Engine:
    """환경 엔진 상수. 물리 서브스텝은 ``dt_phys``이고 컨트롤 간격과 decimation은
    :class:`timebase.Timebase`가 ``control_fps``에서 파생한다.

    각 필드의 짧은 주석은 현재 의미와 단위를 기록한다. 시간 창은 초 단위 필드가 SSOT이고,
    파생 tick 필드는 ``init=False``다.
    """
    # 시간 / 프레임 관련 팩터
    dt_phys: float = DEFAULT_TIMEBASE.dt_phys
    """물리 계산 서브스텝 시간 간격(s). 기본 1/90 s — 기본 15 Hz 컨트롤은 정수
    서브스텝(6), K리그 트래킹 native 30 Hz는 정수 서브스텝(3)으로 떨어진다.
    25 Hz(DFL native)가 필요하면 dt_phys=0.01을 명시 주입할 것(1/90으로는 25 Hz가
    정수 서브스텝이 아니라 env 생성이 거부된다).
    dt 간 적분 편차는 실측 1 cm/2 s 수준(강한 로프트킥+스핀 자유전개, 1/90 vs 1/100)."""
    # 선수 관련 팩터
    r_player: float = 0.23
    """선수 반지름(m)"""
    player_boundary_margin: float = 5.0
    """데드볼 재개 배치에서 선수에게 허용하는 피치 바깥 작업 여유(m).

    x·y 양방향에 동일 적용한다. 오픈플레이 적분은 이 큰 작업영역을 쓰지 않고 선수 중심을
    라인 밖 ``r_player``까지만 허용한다. 스로인 키커와 기술구역 배치에는 이 값이 필요하다."""
    bench_first_x_offset: float = 4.0
    """퇴장 선수 벤치 배치의 첫 x 오프셋(m)."""
    bench_spacing: float = 4.0
    """퇴장 선수 벤치 슬롯 간격(m)."""
    bench_boundary_inset: float = 1.0
    """벤치 슬롯 x를 경기장 끝에서 안쪽으로 제한하는 여유(m)."""
    bench_touchline_inset: float = 0.4
    """벤치 줄이 터치라인에서 떨어진 거리(m)."""
    offpitch_zone_spacing: float = 2.5
    """오프피치 구역 사이 간격(m). 벤치·교체아웃·퇴장을 터치라인 거리로 구분한다."""
    a_max: float = 7.95  # [calib 2026-08-23] 진행방향 가속 상한 n=22,681,993 · 이전값 8.68
    """선수 최대 가속도(m/s^2) — [속도-명령 모델] 진행방향(종) **가속** 상한. **[p99.9 캘리브]** 실측 종가속 p99.9=8.68. @snapshot"""
    accel_norm_max: float = 8.46  # [DFL calib p99.9] 실측 횡가속 p99.9
    """[속도-명령 모델] **선회(횡·법선) 가속 상한**(m/s²). **[p99.9 캘리브]** 실측 횡가속 p99.9=8.46 @snapshot
    (선회는 종가속보다 약하다). 목표속도 클립의 수직 성분 캡. a_max(종)·brake_decel_max(종감속)와 3분할."""
    brake_decel_max: float = 10.51  # [DFL calib p99.9] 실측 종감속 p99.9
    """[속도-명령 모델] 진행방향(종) **감속** 상한(m/s²). **[p99.9 캘리브]** 실측 종감속 p99.9=10.51 @snapshot
    (감속이 가속보다 강함). 목표속도 클립의 종 음성분 캡."""
    sep_iters: int = 3
    """_separate 반복 횟수 — 선수 간 최소거리(``r_player + r_player``) 겹침 밀어내기.

    경계에 막힌 5인 충돌 fixture에서 2회는 7.78 mm, 3회는 2.40 mm의
    근사 침투를 남겼다. 3회의 전체 step 비용은 2회 대비 약 2.7% 증가에
    그쳐 기본값과 검증 하한으로 사용한다. 이 solver는 여전히 근사이며
    엄밀한 0-침투를 주장하지 않는다. 허용 범위는 3..16회다. 더 큰 값은
    Python-unrolled JIT 그래프만 비대하게 만들며 다체 제약의 해법이 아니다."""

    restart_slide_rounds: int = 6
    """재개 투영 후 겹침 해소 라운드 수. 투영은 침범자마다 독립이라 둘을 사실상 같은 지점으로
    보낼 수 있고(골킥은 x를 공유, 코너는 합법 원호가 얇아 같은 점으로 깔때기), 이 투영은
    서브스텝의 유일한 ``_separate`` 뒤에 호출되므로 그 겹침이 관측 상태에 그대로 남는다.
    각 라운드는 충돌쌍을 결정적 순서로 풀고 곧바로 규칙 제약에 재투영한다.

    검증은 정수 6 이상을 강제한다. 롤아웃에서 뽑은 실제 배치 350개 기준
    5회에서는 코너 침투가 1.4 cm 남았고, 6회에서 사라졌다. 다체 제약은 라운드를
    단순히 늘리는 것만으로 해결되지 않으며, ``restart_separation_relaxation``과 함께
    검증된 구간을 써야 한다. 성능을 위해 하한을 낮추면 세트피스 밀집 겹침이 돌아온다.
    허용 상한은 16회이며, 그 이상은 동일한 국소 완화를 반복할 뿐 구조적 불일치를
    해결하지 못하고 O(N²) 실행시간만 늘린다."""

    restart_separation_relaxation: float = 1.5
    """Over-relaxation coefficient for restart-position collision repair.

    Exact-contact correction (1.0) reconverges only asymptotically in crowded
    multi-body constraints. Environment validation therefore accepts only
    the measured-safe interval [1.5, 2.0]; values outside it can leave players
    overlapped even when the round count is increased.
    """

    sprint_speed: float = 5.5
    """감사 telemetry와 장기 추가 부하에서 사용하는 스프린트 기준 속도(m/s)."""
    long_stamina_sprint_mult: float = 3.0
    """장기 stamina의 스프린트 추가 부하 기울기. 기준속도는 ``sprint_speed``다."""
    long_stamina_idle_load: float = 0.12
    """on-pitch 선수가 정지해도 내는 장기 workload. 경기 국면과 무관하게 적용된다."""
    long_stamina_speed_ref: float = 2.4
    """장기 거리 workload의 기준 속도(m/s)."""
    long_stamina_speed_load: float = 0.90
    """거리(속도) workload 계수.

    실측 트래킹 127경기(DFL 7 + K리그 120)에 이 계수로 현행 workload 식을 적용하면 경기 중앙
    workload는 약 **0.82**다. 이 값을 ``long_stamina_reference_workload``로 분리해 계수와
    경기 종료 목표를 독립적으로 캘리브레이션할 수 있다.

    반면 현행 규칙 정책(balanced×balanced, 240 s × 8 seed)의 workload는 1.25이고,
    90분 환산 장기 drop은 1.45다. 원인은 계수가 아니라 **인플레이 비율**이다 — 인플레이 중
    평균 속력은 실측 2.367 대 env 2.386으로 거의 같지만, 인플레이 비율이 실측 52.2 %인
    반면 env는 85.8 %라 env 선수에게는
    실측 선수가 걷는 시간(데드볼)이 거의 없다. 따라서 이 계수를 낮추는 것은 잘못된 수리다."""
    long_stamina_accel_ref: float = 4.0
    """장기 가속 workload의 기준 가속도(m/s²). 실행 속도의 substep 차분으로 산출한다."""
    long_stamina_accel_load: float = 0.10
    """정규화된 가속도의 제곱 장기 workload 계수."""
    long_stamina_vmax_floor: float = 0.99
    """장기 stamina가 0일 때 지속 가능 최고속도의 비율. 실측 후반 p99 속도 저하가 작아
    기본값은 0.99이며, 장기 피로는 주로 단기 질주 반복 가능성을 통해 드러나게 할 수 있다."""
    long_stamina_end_frac: float = 0.05
    """기준 workload로 90분을 모두 뛴 선수의 목표 장기 stamina 잔여량."""
    long_stamina_tail_knee: float = 0.20
    """장기 stamina의 정보 보존형 꼬리가 시작되는 잔여량.

    이 값 위에서는 workload를 기존처럼 선형 적분한다. 아래에서는 같은 명목 피로를
    지수 좌표로 바꿔 90분을 넘기거나 workload가 큰 선수도 0에 clip되지 않게 한다.
    기준 선수의 90분 종료값은 ``long_stamina_end_frac`` 그대로다."""
    long_stamina_reference_duration_s: float = DEFAULT_MATCH_DURATION_SECONDS
    """장기 소모율의 고정 물리시간 기준(s). clip 길이와 독립인 90분이다."""
    long_stamina_reference_workload: float = 0.82
    """종료 목표를 보정하는 실측 평균 workload. 데이터셋 확장 시 독립적으로 재추정한다."""

    short_stamina_vmax_floor: float = 0.70
    """단기 stamina가 완전히 고갈됐을 때 장기 속도 상한 대비 순간 최고속도 비율."""
    short_stamina_headroom_knee: float = 0.25
    """이 잔여량 이상이면 단기 stamina가 최고속도를 제한하지 않는다."""
    short_stamina_depletion_s: float = 10.0
    """속도 강도 1에서 단기 stamina 1을 소모하는 기준 시간(s)."""
    short_stamina_depletion_speed_frac: float = 0.70
    """장기 속도 상한 대비 단기 소모가 시작되는 속도 비율."""
    short_stamina_speed_exponent: float = 2.0
    """단기 속도 부하 곡률. 1보다 크면 임계 근처 이동을 더 널널하게 포용한다."""
    short_stamina_accel_ref: float = 6.0
    """단기 부하에 쓰는 양의 속력 가속 기준(m/s²)."""
    short_stamina_accel_load: float = 0.20
    """양의 속력 가속이 단기 소모에 더하는 최대 가중치."""
    short_stamina_recovery_tau_s: float = 60.0
    """완전 휴식에서 단기 결손이 지수적으로 회복되는 시정수(s)."""
    short_stamina_recovery_speed_frac: float = 0.70
    """장기 속도 상한 대비 단기 회복이 0이 되는 속도 비율."""
    short_stamina_recovery_exponent: float = 1.0
    """저강도 속도에 따른 단기 회복 곡률."""
    short_stamina_long_recovery_penalty: float = 0.0
    """장기 피로가 단기 회복률을 낮추는 선택 계수(0~1). 기본값 0은 두 효과를 분리한다."""
    possession_release_speed: float = 0.5
    """이 속력(m/s) 이하이면 공이 '서 있다'고 본다 — 소유권 해제 판정의 한 축."""
    possession_release_radius: float = 5.0
    """활성 선수가 이 거리(m) 밖에만 있으면 공이 '방치됐다'고 본다.

    소유권은 접촉으로만 바뀌므로, 아무도 건드리지 않으면 마지막 터치 팀에 **영원히**
    래치된다. 규칙 정책은 소유가 중립일 때만 루즈볼 추격을 켜므로(policy의 ``loose_phase``),
    래치된 채 아무도 곁에 없으면 양 팀이 대형만 유지하고 공이 멈춰 선 경기가 만들어진다.
    실측에서 최근접 선수가 9.8 m 떨어진 채 90초가 흘렀다. 규칙 vs 규칙은 접촉이 계속돼
    소유가 뒤집히므로 이 상태에 빠지지 않는다 — 다르게 행동하는 정책만 노출시키는 잠복
    결함이라 env에서 닫는다."""
    # obs 정규화 관련 팩터
    norm_player_vel: float = 20.0           # Calib 요구
    norm_ball_vel: float = 50.0             # Calib 요구
    norm_ball_z: float = 20.0          # Calib 요구
    norm_spin: float | None = None
    """공 스핀 관측 정규화 분모. None이면 ``spin_max``에서 자동 파생."""
    norm_body_z: float = 3.0
    norm_score: float = 5.0

    # 행동 공간 관련 팩터
    launch_max: float = 1.0
    """force2ball 최대 발사각(rad, ≈57°). [DFL calib/fit_kick] 강한 발킥(v>12) 발사각
    p99.5=0.96 반올림 — 구 1.2(69°)는 실측 상회."""

    # 도달 볼륨 관련 팩터
    reach_xy_carry: float = 1.1
    """공 소유팀 선수의 터치 반지름(m). [K리그 실측 2026-08-13] 이벤트 액터-공 거리
    중앙 0.84·p90 1.57 — 구 1.6(역할 혼합 p95)에서 축소. 소유 판정은 poss_team."""
    reach_xy_challenge: float = 1.4
    """비소유(도전) 선수의 개입 반지름(m) — 태클·인터셉트·루즈볼. carry(1.1)~여기(1.4)
    사이는 **런지 존**: 파울 로짓 가산(Foul.k_lunge)·재시도 쿨다운 연장
    (challenge_cooldown_extra_s)이 거리 비례로 걸린다. 중립(poss=-1)일 땐 전원 challenge."""
    challenge_cooldown_extra_s: float = 0.84
    """런지 최대치에서의 추가 쿨다운(초) — 슬라이딩/런지 회복 시간. 총 쿨다운은
    cooldown_s(0.16)+λ·이 값 ≈ 최대 1.0 s. 관찰 캘리브 불가(프로는 안전할 때만
    긴 태클 시도 — 선택 편향)라 RL 착취 방지용 설계 파라미터."""
    reach_height_factor: float = 0.85
    """선수 머리 높이 factor — head_z = tall * reach_height_factor.
    계수가 1이면 머리 높이가 키와 같고, 더 작으면 그 비율만큼 낮아진다."""
    gk_reach_xy: float = 2.0  # Calib 요구
    """골키퍼가 '자기 페널티박스 안'에 있을 때의 수평 reach 반지름(m) — 손 사용·다이브 커버."""
    gk_backpass_target_radius: float = 5.0
    """의도적 발 플레이/스로인의 예측 종착점이 자기 GK를 겨냥했다고 보는 오차 반경(m).

    실제 규칙은 결과가 정확히 도착했는지가 아니라 동료가 골키퍼에게 보내려는 의도였는지를
    본다. 액션의 제출 방향·세기와 굴림 감속 SSOT로 무경합 종착점을 추정해 이 반경 안이면
    손 처리 금지 provenance를 설정한다. 5m는 부정확한 백패스도 포용하면서 같은 방향의 더
    짧은 중간 패스는 제외하는 설계 prior이며 데이터로 교체 가능한 명시적 파라미터다."""

    # 골키퍼 캐치/홀드
    gk_catch_speed_cap: float = 21.3  # [calib 2026-08-23] 호환 하드캡 / 확률곡선 50% 공속
    """GK 캐치 판정의 수평 공속 기준(m/s). ``gk_catch_speed_scale==0``인 레거시
    모드에서는 이하=캐치·초과=parry인 하드캡이고, scale>0이면 캐치 확률이 50 %인
    로지스틱 중앙점이다. K리그 save catch/punch 라벨을 GK reach 진입 공속에 연결한다."""
    gk_catch_speed_scale: float = 9.9  # [calib 2026-08-23] GK 캐치 확률 속도 전이폭 n=135 · 이전값 0.0
    """속도별 GK 캐치 확률 ``sigmoid((cap-speed)/scale)``의 전이폭(m/s).

    0은 기존 하드캡과 정확히 같은 호환 모드다. 양수 후보는 K리그 catch/punch 표본의
    로지스틱 기울기에서 추정하고, 같은 시드 rollout 검증 뒤에만 활성화한다."""
    gk_hold_s: float = 8.0
    """GK 캐치 홀드 창(초) — IFAB 홀드 규칙. 이 안에 배급 안 하면 만료 강제 배급.
    (obs rt_norm·restart 만료 강제킥이 이 값을 파생 — 바꾸면 자동 전파.)"""
    gk_hold_substeps: int = field(init=False)
    """gk_hold_s의 서브스텝 환산 — __post_init__ 파생(단일 진실원천은 초 단위 필드)."""

    # challenge/re-challenge recovery (compatibility name: cooldown)
    cooldown_s: float = 0.16
    """도전/재도전 기본 쿨다운(초). 기존 State/config 직렬화 호환을 위해 ``cooldown`` 이름은
    유지하지만, 현재 소유팀의 공 제어에는 적용하지 않는다. 도전자(상대 소유 또는 루즈볼)에만
    게이트되고, 도전으로 소유를 얻은 동안은 남은 시간이 무시되며 소유를 다시 잃으면 재활성화된다.
    같은 control frame 내 반복 접촉은 이 타이머가 아니라 player별 touch gate가 막는다.

    **데이터 하한이라 서브스텝 환산이 이 값을 넘으면 안 된다**(넘게 반올림되면 데이터가 허용하는
    0.16 s 재도전을 env가 금지) — __post_init__가 초과 시 한 스텝 내려 잡는다.
    0.16s는 DFL event XML의 같은선수 킥→재터치 클린 최소 간격(3,121쌍 중 미만 0건)에서 가져온
    **회복시간 proxy/design prior**다. 현재 계약은 challenge-only이므로 직접적인 재도전 캘리브레이션
    근거는 아니며, challenger-specific 추정 전까지 값을 유지한다. @snapshot 트래킹 검출의 3프레임
    간격 '터치'는 듀얼 분열·유령 검출이었음(replay 병목의 원인 재귀속)."""
    cooldown_substeps: int = field(init=False)
    """challenge cooldown_s의 서브스텝 환산(하한 비초과 정책) — __post_init__ 파생."""

    contact_interval_s: float = 1.0 / 15.0
    """동일 선수가 실제 force-to-ball 시도를 다시 낼 수 있는 최소 물리시간(s).

    control frame이 아니라 물리 시계에 묶이므로 control FPS를 15→30 Hz로 바꿔도 접촉 빈도는
    자동으로 15 Hz 상한에 정렬된다. 낮은 control FPS에서는 새 명령이 없으므로 control 주기가 상한이다.
    """
    contact_lock_substeps: int = field(init=False)
    """``contact_interval_s``의 물리 substep 환산값."""

    aerial_attempt_lock_s: float = 0.5
    """선수 키보다 높은 공에 대한 도달 가능한 능동 시도 후 회복시간(s).

    실제 접촉·경합 승패와 무관하게 시작하며 이동은 허용한다. 이 타이머는 능동적인 다음
    볼 개입만 막고, 몸에 맞는 공의 수동 굴절은 계속 처리한다. 따라서 능동 킥 뒤 자기 몸과의
    즉시 재충돌까지 막는 :attr:`contact_interval_s`와는 별도 상태다.
    """
    aerial_attempt_lock_substeps: int = field(init=False)
    """``aerial_attempt_lock_s``의 물리 substep 올림 환산값."""

    # 세트피스 재개
    restart_s: float = 180.0
    """일반 재개 타이머의 전체 창(초).

    키커 접근 실패를 안전하게 담는 상태 창이다. 실제 발사 시점은 아래
    ``*_restart_delay_s``가 정하고, 이 값 자체가 모든 재개에서 180초를 기다리라는 뜻은 아니다.
    """
    restart_substeps: int = field(init=False)
    setup_hold_s: float = 3.0
    """GK가 손으로 통제한 뒤 결정론적으로 배급하는 시간(초).

    IFAB이 정한 8초는 **상한**이지 목표가 아니다. 실제 GK는 한도를 채우지 않고
    2~4초에 배급한다. 이 값이 상한에 가까우면 GK가 매 홀드마다 정확히 그만큼 서 있고,
    Law 9상 그 시간이 인플레이·점유로 잡히므로 지표가 '정지한 공'에 지배된다.
    일반 재개와 페널티는 키커가 공에 도착한 뒤 아래의 3초 데드볼 유지 게이트를 쓴다.
    """
    setup_hold_substeps: int = field(init=False)
    throwin_restart_delay_s: float = 3.0
    """키커가 공에 도착한 뒤 스로인을 허용하기까지의 데드볼 유지시간(s)."""
    throwin_restart_delay_substeps: int = field(init=False)
    goalkick_restart_delay_s: float = 3.0
    """키커가 공에 도착한 뒤 골킥을 허용하기까지의 데드볼 유지시간(s)."""
    goalkick_restart_delay_substeps: int = field(init=False)
    corner_restart_delay_s: float = 3.0
    """키커가 공에 도착한 뒤 코너킥을 허용하기까지의 데드볼 유지시간(s)."""
    corner_restart_delay_substeps: int = field(init=False)
    freekick_restart_delay_s: float = 3.0
    """키커가 공에 도착한 뒤 프리킥을 허용하기까지의 데드볼 유지시간(s)."""
    freekick_restart_delay_substeps: int = field(init=False)
    offside_restart_delay_s: float = 3.0
    """키커 도착 뒤 오프사이드 간접 FK를 허용하기까지의 데드볼 유지시간(s)."""
    offside_restart_delay_substeps: int = field(init=False)
    post_goal_kickoff_delay_s: float = 0.0
    """득점 뒤 다음 킥오프의 추가 대기시간(s).

    0은 득점이 난 control frame에서는 재개 상태만 만들고, 다음 control
    step에서 바로 킥오프를 연다. 경기 시작·후반 시작 킥오프도
    경기 시계 밖 준비를 이미 마쳐 ``restart_t=1``로 즉시 가능하다.
    """
    post_goal_kickoff_delay_substeps: int = field(init=False)
    throwin_clear: float = 2.0
    """스로인 상대 이격거리(m). 수치 안정화용으로 안쪽에 둔 공 중심이 아니라
    Law 15가 정한 실제 터치라인 위 재개 지점에서 잰다."""
    clear_dist: float = 9.15
    """코너·프리킥·오프사이드 재개의 규정 이격거리(m). 프리킥·오프사이드는
    공에서, 코너는 ``Stadium.corner_arc_radius`` 아크에서 잰다. 골킥은 이 반경을
    쓰지 않고 페널티박스 이탈만 요구하며, 페널티 아크는 Stadium.penalty_arc_radius가 SSOT다."""
    kicker_speed: float = 6.0
    """키커 강제이동 속도(m/s) — 조깅 접근 수준. 도착 전 스태미나 무소모"""
    kicker_arrive_r: float = 1.6
    """키커 도착 판정 반경(m). 일반 터치 reach와 별개의 세트피스 접근 기준."""
    goalkick_depth: float = 5.10
    """골라인에서 골킥 공 중심까지의 인필드 깊이(m). K리그 setPiece goalKick
    1,712건 중앙값 5.097m를 반올림했다(골 에어리어 안에서만 허용)."""
    goalkick_lateral_offset: float = 6.25
    """골킥 공의 중앙선 기준 좌우 절대 오프셋(m). 같은 1,712건의 |y|
    중앙값 6.260m를 반올림했으며, 직전 아웃의 터치라인 쪽 부호를 쓴다."""
    # 재탈취 지연
    ctrl_lock_s: float = 0.16
    """재탈취 지연(초). cooldown_s와 같은 하한 비초과 정책."""
    ctrl_lock_substeps: int = field(init=False)

    # ── 오프사이드 ─────────────────────────────────────────────
    offside_margin: float = 0.5
    """2번째 최종수비 라인 여유(m) — 이보다 앞선 동료만 오프사이드 위치로 플래그."""
    # ── 페널티 / 차징 ───────────────────────────────────────────
    penalty_spot: float = 11.0
    """골라인에서 페널티 스폿까지(m)."""
    penalty_s: float = 140.0
    """페널티 키커 접근과 재개 상태를 담는 안전 창(초)."""
    penalty_substeps: int = field(init=False)
    """페널티 빌드업 윈도우(서브스텝) — 전원 박스 밖 정렬 시간."""
    penalty_restart_delay_s: float = 3.0
    """키커가 공에 도착한 뒤 페널티킥을 허용하기까지의 데드볼 유지시간(s)."""
    penalty_restart_delay_substeps: int = field(init=False)
    challenge_cooldown_extra: int = field(init=False)
    """challenge_cooldown_extra_s의 서브스텝 환산 — __post_init__ 파생."""
    charge_speed: float = 5.5
    """상대 보유자에게 이 접근속도(m/s) 이상으로 돌진하면 차징 후보."""

    # ── 경합 승자 선정 로짓 가중 ─────────────────────────────────
    # ★식별성 정직 표기(균일 능력치 기본 로스터에선): w_dist만 유효 지렛대. w_time은 time=dist/vmax라
    #  균일 vmax에서 **w_dist와 완전 공선**(따로 식별 안 됨), w_height·w_ctrl은 능력치가 균일이면
    #  전원 공통상수 → argmax에 **무효**. 이 셋은 **이질(heterogeneous) 로스터(선수별 speed/reach_z/
    #  ball_control 다름)에서만 식별·발현**한다. 균일 기본에서 실효 경합모델 = 거리 + 소유(w_poss).
    w_dist: float = 1.0
    """경합 점수 가중 — 공까지 거리(가까울수록↑). 유효 주 지렛대."""
    w_time: float = 1.0
    """경합 점수 가중 — 도달시간(거리/vmax). ★균일 vmax에선 w_dist와 공선(별도 식별 불가) — 이질 vmax에서만 발현."""
    w_height: float = 0.55
    """경합 점수 가중 — 높이 적합(reach_z 대비). ★균일 reach_z에선 공통상수=무효 — 이질 reach_z에서만 발현."""
    w_poss: float = 0.0
    """경합 점수 가중 — 현 소유팀 보너스. **중립(0)으로 설정**. 근거: (1) 부모 fit_contest_true의
    β_poss=0.82(→0.54)는 부재 스크립트라 감사불가이고, 현재 보존된 유일 감사가능 통계(이벤트 소유측
    승률 48.2%<50%)와 부호가 어긋난다. (2) **실측 롤아웃에서 w_poss∈{0,0.27,0.54}가 소유 전환율
    (10.2/분)·점유 분할에 무영향**(등거리 tie만 좌우, 대부분 경합은 거리지배 → 행동 비활성). 검증
    불가능한 양의 바이어스를 행동 이득도 없이 싣지 않는다(중립=감사가능 통계와 정합). 재적합 시 복원 가능."""
    w_ctrl: float = 0.3
    """경합 점수 가중 — 볼컨트롤 능력치. ★균일 ctrl에선 공통상수=무효 — 이질 ball_control에서만 발현."""
    contest_temp: float = 0.66
    """Gumbel 소프트맥스 온도(낮을수록 결정적). [상속] 부모 fit_contest_true — 진라벨 β_dist=1.21에
    프레임노이즈 역보정(시뮬 σ_δ=2f) β_true≈1.68 → temp=0.66. 시뮬 기반 +39% 보정은 감사불가(상속)."""

    # ── 킥 파워/높이 캡 ─────────────────────────────────────────
    f2b_speed_max: float = 34.76   # [DFL calib p99.9] 고신뢰(event-확인) 클린 킥속도 p99.9
    """풀파워 킥 상한 속도(m/s, ≈125km/h). **[p99.9 캘리브]** 고신뢰(event-확인) 클린 킥 직후속도
    p99.9=34.76(crude 44는 굴절오염이라 배제). @snapshot"""
    reach_block_limit: float = 31.0
    """reach 차단 가능 상한(m/s) — ``speed + reach_height_penalty × z``의 절편.

    이 선 위의 공은 reach 경합 후보에서 빠진다. **몸통 충돌은 그대로다** — ``ball.py``의
    스윕 판정은 이 게이트를 거치지 않으므로 직접 맞으면 여전히 굴절·트랩된다. 즉 "발만
    뻗어서 잡는 것"만 막고 "몸으로 막는 것"은 남긴다.

    [DFL calib] 공이 reach 반경(1.4 m) 안을 지나간 9,569건을 분모로, 실제 접촉이 일어난
    비율을 (높이 × 진입속도)에서 쟀다. 차단률은 절벽이 아니라 경사이고 높이·속도 **양쪽**이
    함께 떨어뜨린다 — 땅볼도 26 m/s를 넘으면 절반이 못 막힌다. 높이 구간별 0.5 교차 속도가
    26.5 / 23.0 / 26.8 / 25.2 / 20.5 / 19.8 m/s로 단조 감소해 선형 경계가 등고선을 잘 근사한다.

    값 선택은 **정상 플레이를 봉쇄하지 않는 쪽**을 최우선으로 했다. C=31에서 실제로 막아낸
    공을 차단해 버리는 경우가 94건(0.98%)뿐이고, 차단되는 815건 중 89.9%는 실측에서도 못
    막은 공이다. C=26이면 정확도는 0.914로 더 높지만 봉쇄가 231건으로 늘어난다.
    ``f2b_speed_max``(34.76)를 감안하면 풀파워 킥은 어느 높이에서도 차단되지 않는다."""
    reach_height_penalty: float = 3.5
    """높이 1 m마다 차단 가능 속도가 줄어드는 양(m/s). [DFL calib] 위 등고선의 기울기."""
    pelvis_frac: float = 0.55
    """골반 높이 = head_z×이값. 이하 접촉=발 슛(풀파워)."""
    chest_cap: float = 0.73
    """골반~머리 높이 접촉 파워 상한(×f2b_speed_max). [DFL fit_kick] 밴드별 출구속도 p99.5 비.

    development 7경기 재측정: 전체 접촉 0.716 / 이벤트 확인 0.730 / 확인·슛·패스·크로스 0.735.
    ``f2b_speed_max``와 ``header_cap``이 이벤트 확인 표본으로 잡혔으므로 같은 층인 0.73을 쓴다.

    이전 값 0.87은 "이 밴드엔 가슴 높이 발리킥이 섞여 순수 가슴트랩보다 높다"는 근거였지만
    재측정에서 지지되지 않았다 — 가슴 밴드 564건 중 25 m/s 초과가 3건뿐이고 그 평균 높이도
    0.95 m로 골반 경계(0.842 m) 바로 위였다. 게다가 출구속도 추정 잡음은 상위 분위수를
    **부풀리는** 방향이라 참값은 이보다 낮을 여지가 있다.

    라벨을 깨는 문제가 아니라 **RL 착취** 문제였다 — 상한이 관대하면 거부가 줄 뿐이지만,
    정책이 가슴 높이 공을 30 m/s로 때리는 것을 env가 허용하는데 실측에는 그런 접촉이 없다.
    발(1.00)과 헤더(0.71)는 재측정에서 그대로 지지됐다(0.955~0.962, 0.679~0.702)."""
    header_cap: float = 0.71
    """머리 위(header) 접촉 파워 상한(×f2b_speed_max). [DFL fit_kick] p99.5 24.0/33.7=0.71(≈86km/h 헤더). @snapshot"""
    launch_down_ground: float = 0.21
    """지면공 하향 발사각 하한(rad, ≈-12°) — 프레스/솔 트랩. [DFL fit_kick] 발킥 발사각 p0.5=-0.21."""
    launch_down_ref: float = 1.0
    """하향 발사각이 -launch_max까지 열리는 공 높이(m)."""

    # ── 스핀(킥 액션) ───────────────────────────────────────────
    spin_max: float = 78.4
    """액션 스핀[-1,1]→±이값(rad/s). [DFL calib/fit_magnus] 공중 커브 2,574세그 곡률비
    @snapshot |q0|=c_magnus·ω_z 피팅 p99.5와 문헌 앵커를 곱의 두 인자로 분해했다 —
    곱만 식별되므로 c_magnus·spin_max 곱이 실측 최대 휨을 재현하도록 spin_max 쪽을 조정.
    norm_spin과 동기 유지 필수."""
    spin_head_cap: float = 0.40
    """머리 높이 접촉 스핀 상한(×spin_max)."""
    spin_chest_cap: float = 0.55
    """가슴 높이 접촉 스핀 상한(×spin_max)."""

    # ── 태클 / 굴절 / 탈취 ──────────────────────────────────────
    tackle_prob: float = 0.28
    """태클 1회 성공(소유권 전환) 확률 — DFL 지상경합 전환율 27.9%."""
    tackle_out_cap: float = 0.79
    """탈취 터치 출구속도 상한(×f2b_speed_max). [DFL calib/fit_body] 소유전환(프록시) 터치
    1,822건 출구속도 p99.5=26.9/34.0."""
    intercept_speed: float = 6.0
    """레거시 직렬화/캘리브 보고서 호환용 공속 경계.

    TACKLE/INTERCEPT 의미는 공속이 아니라 실제 캐리어 제어 여부로 결정한다. 새
    동역학에서는 사용하지 않지만, 이전 config와 보고서를 읽을 수 있도록 필드는 보존한다."""
    deflect_prob: float = 0.0  # [calib 2026-08-23] 경합 실패 뒤 무작위 굴절 확률 n=89 · 이전값 0.5
    """경합 실패 뒤 진행방향 기준 무작위 굴절 채널의 확률. B-1에서는 0으로 꺼 "
    몸통 스윕 반사만 실제 굴절을 만들게 한다. 역산·구버전 config 호환을 위해 채널과 "
    하위 계수 코드는 보존한다."""
    deflect_out_frac: float = 0.61
    """굴절 출구속도 = 입사속도×이값 + deflect_out_base. [DFL calib/fit_reception] 무에너지유입
    수신 4,376건 Theil-Sen 회귀 기울기 0.61(95% CI 0.58~0.63). @snapshot"""
    deflect_out_base: float = 0.05
    """굴절 출구속도 기저(m/s). [DFL fit_reception] 회귀 절편 ≈0 — 구 2.0은 저속 굴절 과대."""
    deflect_angle_max: float = 1.2
    """굴절 시 입사 방향에서 허용하는 최대 좌우 회전각(rad). 정방향·역산이 함께 사용."""
    deflect_lift_frac: float = 0.10
    """굴절 출구속도 중 수직 성분 비율."""
    deflect_stationary_speed: float = 1.0
    """입사 방향 대신 도전자→공 방향을 쓰는 저속공 임계값(m/s)."""

    # ── 슛/드리블/스로인 분류·발사 ──────────────────────────────
    f2b_shoot_range: float = 30.0
    """규칙 정책이 슛을 시도하는 최대 거리(m). **슛 분류에는 쓰이지 않는다** — 분류는
    거리와 무관하게 조준선이 골문을 지나는가로 판정하므로 30m 밖 중거리포도 슛이다."""
    dribble_speed_max: float = 9.0
    """자기팀 자기터치가 이 속도 이하면 드리블 터치로 라벨.

    같은 값이 슛 분류의 **하한**이기도 하다 — 이 아래는 '놓은 공'(제어 터치)이고 위가
    '친 공'이다. 두 분기가 같은 경계를 공유해야 분류가 평가 순서와 무관하게 배타적이다."""
    shot_aim_mouth_scale: float = 3.0
    """슛으로 인정하는 조준 폭 — **골 반폭의 배수**.

    찬 공의 조준선이 골라인을 지나는 지점이 중앙에서 이 배수 × (골폭/2) 안이면 슛이다.
    기본 3.0은 골문(±1배) 양옆에 골폭 하나씩(±2배 추가)을 더한 폭이고, 기본 규격에서
    ±10.98 m다. 25 m 거리에서 23.8° 빗나간 강타까지 슛 시도로 본다.

    종전에는 골 **중앙** 방향의 고정 60° 원뿔(``shot_aim_cos``)을 썼는데, 각도는
    거리에 따라 뜻이 달라진다 — 5 m 앞에서는 페널티 지역을 통째로 덮고 30 m에서는
    골라인 폭 35 m(골문의 5배)를 덮었다. ``f2b_shoot_range``의 30 m 상한은 그 원뿔이
    원거리에서 터무니없어지는 것을 막는 땜질이었고, 그 대가로 30 m 밖 득점이 슛으로
    기록되지 않았다. 골라인 교차점으로 재면 두 문제가 함께 사라지고, 미터 단위라
    거리와 무관하게 같은 뜻을 갖는다. 경기장 규격이 바뀌어도 따라가도록 배수로 둔다."""
    throw_speed_max: float = 21.5
    """스로인(머리 위 손던지기) 최대 발사속도(m/s). [DFL calib/fit_throwin] 릴리즈 268건
    (터치라인·창 일관성·천장 필터) 강건속도 p99.5=21.5 — 전 캡 공통 p99.5 앵커 규약. @snapshot"""
    restart_min_ball_speed: float = 0.5
    """자동 재개 최소 공 속도(m/s). 세트업 완료 시 릴리스 타이밍은 환경이 소유하므로
    zero-power 정책 출력도 이 속도로 투영해 'kicked and clearly moves'를 만족하고 재개
    타임아웃/정지공 인플레이를 동시에 막는다. 오픈플레이 킥에는 적용하지 않는다."""
    throw_height: float = 0.47
    """스로인 시 머리 위 손 높이 가산(m). [DFL calib/fit_throwin] 릴리즈 z 중앙 2.00m에 발사고
    (head_z 1.53+가산)를 정렬 — z는 DFL 저신뢰 축이나 중앙값(n=268)은 노이즈에 강건, 편향 유의."""

    # ── 공 자유물리(비행/바운스/굴림) ──────────────────────────
    g: float = 9.81
    """중력가속도(m/s²)."""
    z_ground: float = 0.15
    """지면 접촉 밴드 상단(m) — 이하면 굴림 마찰 적용·공기저항 미적용."""
    ground_settle_vz: float = 0.5
    """이 하강속도(m/s) 미만 접지는 바운스 대신 정착(z→r_ball, vz→0)."""
    c_drag: float = 0.0180
    """공기저항 계수 — ``drag = -c_drag·|v|·v`` (공중 비행만).

    DFL7 14개 하프의 선수 비접촉 3D 비행 아크 442개에서 경기별 유효 중앙값은
    0.0162--0.0217, 전체 중앙값은 0.0181이었다. 5경기 train 중앙값 0.0180을 두
    held-out 경기의 연속 비행-바운스 30창으로 검증했으며, 단독 변경이 위치오차
    p50/p90을 0.690/1.548m에서 0.594/1.365m로 낮췄다. @snapshot"""
    c_magnus: float = 0.0058
    """마그누스 계수 — a = c_magnus·(spin×v). 휨(커브) 생성 (공중 비행만)."""
    c_ground_curl: float = 0.00506
    """지면에서 측스핀으로 생기는 수평 컬 계수:
    ``a_xy = c_ground_curl * (spin_z * z_hat × velocity)``."""
    e_rest: float = 0.773  # [calib 2026-08-23] 지면 반발계수 n=2,642 · 이전값 0.61
    """지면 반발계수(vz 부호반전 시 감쇠) — 실제 임팩트 전용. DFL 무접촉
    바운스의 충돌 전후 포물선을 충돌 순간으로 외삽해 캘리브레이션한다."""
    spin_decay: float = 0.31
    """비행 중 스핀 감쇠율(1/s)."""
    ball_inertia_ratio: float = 0.667
    """공 관성비 α = I/(m·r²). 축구공은 얇은 구껍질 → α≈2/3(속이 찬 구 2/5보다 큼) [문헌 앵커].
    바운스 시 스핀↔병진 접선 임펄스 결합의 물리 계수 — 접촉점 각운동량 보존을 강제한다."""
    bounce_tangential_e: float = 0.0
    """접선 반발계수 e_t ∈[0,1] — 바운스 시 회전 표면속도를 병진으로 전달하는
    탄성 계수. 0은 sticking 전달, 1은 완전탄성 상한이다. 전달 크기는
    ``(1+e_t)*alpha/(1+alpha)*(spin surface velocity)``이며 병진 변화와 회전 변화를
    하나의 접선 충격량으로 결합한다."""
    bounce_spin_vmin: float = 0.7
    """이 하강속도(m/s) 초과 임팩트만 스핀-속도 결합(굴림 미세진동 제외)."""
    bounce_h_keep: float = 0.732  # [DFL7 calib 2026-08-30] train n=468, held-out n=207
    """바운스 임팩트의 수평속도 보존율. DFL7 무접촉 바운스 675개의 진행방향
    투영 속도비 전체 중앙값 0.739, 5경기 train 중앙값 0.732를 사용한다. 현행
    스핀-접선 결합을 포함한 두 held-out 경기 연속 30창에서 ``c_drag=0.0180``과
    함께 p50/p90을 0.690/1.548m에서 0.501/1.014m로 낮췄다. 이벤트별 스핀·입사각
    이질성이 남으므로 하나의 바운스에서 재추정하지 않는다. @snapshot"""
    goal_frame_radius: float = 0.06
    """골 프레임(포스트·크로스바) 단면 반지름(m). IFAB Law 1이 폭·두께를 12 cm 이하로
    묶으므로 0.06 m가 규격 상한이다. 골문 안치수 ``goal_width``/``goal_height``는 프레임
    **안쪽** 면이라, 포스트 축은 y=±(goal_width/2 + r)·크로스바 축은 z=goal_height + r에
    놓인다. 0으로 두면 프레임 충돌이 완전히 꺼진다(구 동작 재현용)."""
    goal_frame_e_rest: float = 0.68
    """골 프레임 법선 반발계수. **미보정 공학 추정값이다** — 잔디 ``e_rest``=0.773은 공-잔디
    캘리브라 그대로 쓸 수 없고(프레임은 속 빈 알루미늄 관이라 링잉으로 에너지를 더 먹는다),
    K리그·DFL 2D 트래킹에는 프레임 충돌 라벨이 없어 fit할 앵커가 없다. 문헌의 공-강체벽
    COR(저속 0.75~0.8, 고속에서 감소)에 관 변형 손실을 얹은 값이다. 실측 앵커가 생기면 교체할 것."""
    goal_frame_mu: float = 0.35
    """골 프레임 접선 마찰계수(Coulomb). 접선 임펄스를 ``|J_t| ≤ mu·|J_n|``으로 제한한다.
    이 상한이 없으면 스치는 충돌까지 sticking(접촉점 슬립 완전 소거)으로 풀려 실제보다 크게
    꺾인다 — 방향 현실성은 이 한 계수가 좌우한다. **미보정 공학 추정값**(도색 알루미늄 대 가죽)."""
    roll_v_knots: tuple = (0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 40.0)
    """굴림 감속 테이블 — 속도 knot(m/s). roll_d_knots와 구간별 선형보간."""
    roll_d_knots: tuple = (0.70, 0.95, 1.09, 1.70, 6.04, 7.85, 9.72, 13.29, 17.39, 26.96)
    """굴림 감속 테이블 — 각 속도 knot에서의 감속도(m/s²). 굴림→슬라이딩 전이 포함."""

    # ── 공-몸통 충돌/트래핑 ─────────────────────────────────────
    leg_top: float = 0.5
    """지면공 밴드 상단(m) — 이 미만 공은 몸통 충돌 통과(다리 아래, 경합으로만 처리)."""
    body_top_frac: float = 1.0
    """몸통 충돌 상단 높이 = head_z×이값."""
    body_r: float = 0.15
    """수동 몸통 충돌 코어 반경(m). 게이트 = body_r + r_ball."""
    """코어 중앙 지름 관통 전체의 상호작용 확률(짧은 chord는 경로비례로 감소).
    DFL 몸통밴드 통과의 운동학 상호작용률과, 무작위 굴절을 끈 동일시드 rollout의
    도전자 굴절 빈도를 함께 맞춰 캘리브레이션한다."""
    e_body: float = 0.5  # [calib 2026-08-23] 몸통 반사 반발계수 n=89 · 이전값 0.45
    """몸통 스윕 반사의 법선 반발계수. 무작위 경합 굴절을 끈 동일시드 rollout에서 "
    도전자 운동학 굴절의 출구/입구 속도비를 DFL과 맞춰 캘리브레이션한다."""
    collide_speed_min: float = 2.5
    """공속 이 미만이면 몸통 충돌 무시(고밴드 저속 보호)."""
    trap_base: float = 0.48
    """기본 트래핑 확률 — 볼컨트롤·저속일수록 상승. [DFL calib/fit_reception] 몸통밴드 수신
    694건 env형 MLE: A(=trap_base·ctrl)=0.239 → ctrl 0.5 분해. 실측 트랩률은 완만한 플래토
    (~13%, 20m/s+에서야 급락) — 선형감쇠 형태의 근사임을 유의."""
    trap_speed_ref: float = 26.6
    """트래핑 확률 감쇠 기준 속도(m/s). [DFL fit_reception] 구 14.0은 14m/s+ 트랩 불가였으나
    실측은 14~20m/s에서도 12% 성공."""
    body_spin_keep: float = 0.35
    """몸통 접촉(바운스) 후 잔존 스핀 비율 — 몸에 맞은 공은 회전이 크게 소산된다(트랩은 0으로).
    DFL 미피팅 초기값(calibration 검증 예정)."""
    body_player_vel_transfer: float = 0.30
    """몸통 반사 공에 전달되는 선수 속도 비율."""
    trap_velocity_keep: float = 0.10
    """몸통 트랩 직후 남기는 수평 공속 비율."""
    trap_drop_distance: float = 0.35
    """트랩한 공을 선수 중심에서 접촉 법선 방향으로 내려놓는 거리(m)."""
    parry_lateral_keep: float = 0.40
    """GK parry에서 입사 y속도를 보존하는 비율."""
    parry_lift_frac: float = 0.35
    """GK parry 출구속도의 수직 성분 비율."""
    parry_punch_speed_cap: float = 27.0
    """전방 펀치가 가능한 입사 공속 상한(m/s). 이보다 빠르면 손으로 궤적을 되돌리지
    못하고 크로스바 위로 흘려보낸다(tip-over). 전방 펀치만 두면 골라인을 넘는 공이
    없어 코너가 0건이 된다 — 도입 전 실측에서 코너 0건이었다(K리그 8.63건).

    값은 코너율로 역산했다. 2,400초(8시드×300초) 표본에서 cap 24.0이 15.8회/90분,
    28.0이 6.8회, 32.0이 2.2회로 로그공간에서 거의 직선(기울기 −0.247/단위)이다.
    목표 8.63회를 두 앵커에서 각각 외삽하면 26.5와 27.0이 나와 27.0을 쓴다.
    **표본은 코너 1~7건이라 개별 측정의 오차가 ±40% 수준이다** — 같은 표본에서
    parry와 무관한 프리킥이 20.2와 31.5로 흔들린 것이 그 노이즈 바닥을 보여준다.
    방향(cap↑ → 팁오버↓ → 코너↓)은 물리적으로 확정이고 크기만 이 정도로 불확실하다."""
    parry_tip_keep: float = 0.55
    """tip-over에서 유지하는 입사 수평속도 비율. 손끝 접촉이라 감속이 펀치보다 작다."""
    parry_tip_clearance: float = 0.60
    """tip-over한 공이 골라인에서 크로스바 위로 지나야 할 여유 높이(m). 분기 판정은
    무항력 탄도식이라 실제보다 높게 예측한다 — 이 여유가 그 차이를 흡수한다."""
    parry_tip_max_range: float = 9.0
    """tip-over를 허용하는 공-골라인 거리 상한(m). 손끝 세이브는 골라인 가까이에서
    일어나고, 멀수록 무항력 근사가 무너져 넘긴다고 판정한 공이 골이 된다."""
    parry_tip_lift_max: float = 12.0
    """GK가 tip-over로 낼 수 있는 수직 초속도 상한(m/s). 이 값으로도 크로스바를
    못 넘기면 넘기지 않고 전방 펀치로 되돌아간다."""

    # ── 규칙/렌더가 공유하는 기하 허용치 ───────────────────────
    restart_field_inset: float = 0.25
    """코너·스로인 재개 스폿을 선 안쪽에 두는 공통 여유(m). 코너킥
    1,037건의 좌표 중앙(골라인 0.148m·터치라인 0.267m 안쪽)에 맞춘 값이다."""
    throwin_line_inset: float = 0.15
    """스로인 공 중심을 터치라인 안쪽에 두는 여유(m)."""
    free_kick_boundary_inset: float = 0.15
    """파울 재개 스폿을 필드 경계에서 안쪽으로 제한하는 수치 여유(m).
    K리그 프리킥 3,090건에서 구 1.0m 값은 42건(1.36%)을 최대 1m 옮겼지만,
    0.15m는 7건(0.23%)만 최대 0.15m 보정해 반칙 지점을 거의 보존한다."""
    goal_line_tolerance: float = 0.25
    """재개 이격 면제/페널티 GK 판정에서 허용하는 선수 중심의 최대 인필드
    깊이(m). 기본 0.25m는 선수 원반 반경 0.23m와 2cm 수치 여유를 합친 값이다.
    골라인 뒤쪽은 절댓값 밴드로 자르지 않고 Law 13/14에 따라 계속 합법이다."""
    goal_post_tolerance: float = 0.25
    """골라인 위 선수 중심의 포스트 바깥 y 여유(m). 선수 반경과 2cm 수치
    여유를 반영하며, 이보다 멀면 어떤 신체 부분도 두 포스트 사이에 있지 않다."""
    legal_margin_floor: float = 0.05
    """규칙상 면제된 선수에게 관측으로 내보내는 최소 양의 이격 여유(m)."""
    unrestricted_margin: float = 1.0
    """이격 의무가 없는 GK 홀드에서 관측으로 내보내는 최소 양의 여유(m)."""

    def __post_init__(self):
        if self.norm_spin is None:
            object.__setattr__(self, "norm_spin", self.spin_max)

        # 경계 검증이 빠져 있던 상수들. 각각 잘못된 부호가 **오류가 아니라 조용한 오작동**
        # 으로 나타나던 것들이라 여기서 막는다.
        #   offpitch_zone_spacing  < 0  → 벤치·교체아웃·퇴장의 터치라인 거리 순서가 뒤집힌다
        #   possession_release_*   < 0  → 방치 소유권 해제가 영영 안 걸리거나 항상 걸린다
        #   reach_block_limit      < 0  → 필드 선수의 의도적 공 접촉 후보가 전멸한다
        #   reach_height_penalty   < 0  → 공이 높을수록 차단이 쉬워져 실측 관계가 뒤집힌다
        positive = {
            "offpitch_zone_spacing": self.offpitch_zone_spacing,
            "reach_block_limit": self.reach_block_limit,
            "aerial_attempt_lock_s": self.aerial_attempt_lock_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        non_negative = {
            "possession_release_speed": self.possession_release_speed,
            "possession_release_radius": self.possession_release_radius,
            "reach_height_penalty": self.reach_height_penalty,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

        # 서브스텝 단위 파생 — 단일 진실원천은 초 단위 필드. dt_phys를 바꿔도 실시간 의미가
        # 유지되고, dataclasses.replace(cfg, dt_phys=...)가 __post_init__을 재실행해 자동 재파생.
        def nearest(seconds):
            """최근접 반올림 — 지속시간 계열(반 스텝 이내 오차가 최선)."""
            return duration_to_ticks(seconds, self.dt_phys, minimum=1)

        def capped(seconds):
            """하한 비초과 — cooldown류는 실측 최소 간격보다 길어지지 않게 내림한다."""
            return duration_to_ticks(
                seconds,
                self.dt_phys,
                rounding="floor",
                minimum=1,
            )

        def at_least(seconds):
            """실시간 최소간격 — 물리 tick이 규약보다 짧아지지 않게 올림."""
            return duration_to_ticks(
                seconds,
                self.dt_phys,
                rounding="ceil",
                minimum=1,
            )

        object.__setattr__(self, "challenge_cooldown_extra",
                           nearest(self.challenge_cooldown_extra_s))
        object.__setattr__(self, "gk_hold_substeps", capped(self.gk_hold_s))
        object.__setattr__(self, "cooldown_substeps", capped(self.cooldown_s))
        object.__setattr__(self, "contact_lock_substeps", at_least(self.contact_interval_s))
        object.__setattr__(
            self,
            "aerial_attempt_lock_substeps",
            at_least(self.aerial_attempt_lock_s),
        )
        object.__setattr__(self, "restart_substeps", nearest(self.restart_s))
        object.__setattr__(self, "setup_hold_substeps", nearest(self.setup_hold_s))
        object.__setattr__(
            self, "throwin_restart_delay_substeps",
            nearest(self.throwin_restart_delay_s),
        )
        object.__setattr__(
            self, "goalkick_restart_delay_substeps",
            nearest(self.goalkick_restart_delay_s),
        )
        object.__setattr__(
            self, "corner_restart_delay_substeps",
            nearest(self.corner_restart_delay_s),
        )
        object.__setattr__(
            self, "freekick_restart_delay_substeps",
            nearest(self.freekick_restart_delay_s),
        )
        object.__setattr__(
            self, "offside_restart_delay_substeps",
            nearest(self.offside_restart_delay_s),
        )
        object.__setattr__(
            self, "post_goal_kickoff_delay_substeps",
            duration_to_ticks(
                self.post_goal_kickoff_delay_s,
                self.dt_phys,
                rounding="nearest",
                minimum=0,
            ),
        )
        object.__setattr__(self, "ctrl_lock_substeps", capped(self.ctrl_lock_s))
        object.__setattr__(self, "penalty_substeps", nearest(self.penalty_s))
        object.__setattr__(
            self, "penalty_restart_delay_substeps",
            nearest(self.penalty_restart_delay_s),
        )


@dataclass(frozen=True)
class Foul:
    """반칙 선언·카드 기준(IFAB Law 12).

    k_lunge는 런지 존(reach_xy_carry~challenge) 개입의 파울 로짓 가산(λ 비례)이다.
    태클 파울과 차징 파울을 로지스틱으로 판정하고, 선언 시 확률적 카드→퇴장한다.
    보상엔 관여하지 않는 순수 규율(수적 열세는 물리로만 불리).

    태클 채널은 경합 승자의 탈취 파울뿐 아니라 근접 도전자가 공을 못 따고 끝난 loser-foul도
    ``contest.retained`` 분기로 표현한다. 카드/레드율은 파울 160·레드 2 소표본이라
    레드 비율의 불확실성은 여전히 크다(CI≈0.7–18%)."""

    k_lunge: float = 1.2
    """런지 분율 λ(0~1)당 태클 파울 로짓 가산 — 설계 파라미터(선택 편향으로 관찰 캘리브 불가)."""

    # 태클 파울 로짓
    tackle_bias: float = -3.22  # [calib 2026-08-23] 태클 파울 로짓 절편 n=15 · 이전값 -2.56
    """태클 파울 로짓 절편. 실측 인플레이 분당 파울 총량에 맞추되, 같은 시드
    rollout에서 소유 전환과 도전자 접촉이 유지되는지 함께 검증해 조정한다."""
    header_foul_bias: float = -0.87
    """공중(헤더) 경합 파울 억제 보정(지상 대비 반칙률 낮음)."""
    k_close: float = 0.0
    """접근속도 계수(적합 실질 0 — 특징 보존용)."""
    k_behind: float = -0.55
    """등 뒤 접근 계수(태클 채널). [주의·상속] 부호가 charge 채널 ``kc_behind``와 **상반**된다 — 같은
    '등 뒤' 특징이 채널마다 반대 부호라 정직하게 표기: 여기(태클)는 '공을 딴 성립 듀얼에선 뒤에서가 파울↓'
    라는 적합 부호를, charge는 '뒤에서 몸싸움=파울↑'를 반영(별개 물리 채널). 상속 적합값, 재검증 시 부호
    일관성 재검토 필요(REVIEW)."""
    k_balldist: float = 0.0
    """공-태클러 거리 계수(적합 실질 0)."""
    balldist_ref: float = 0.9
    """공거리 기준(m). ``k_balldist``가 비활성이면 보상식에 영향이 없다."""
    k_clean: float = 2.8
    """공을 깨끗이 딴 태클의 파울 억제(강)."""
    clean_ball_dist: float = 1.1
    """깨끗한 태클 인정 공거리(m)."""
    clean_ball_speed: float = 6.0
    """깨끗한 태클 인정 공속(m/s)."""
    tackle_p_min: float = 0.008
    """태클 파울 확률 하한."""
    tackle_p_max: float = 0.35
    """태클 파울 확률 상한(실제 수준)."""
    tackle_contact_range: float = 2.5
    """태클 파울 성립 최대 캐리어 거리(m) — 접촉 전제."""
    box_foul_bias: float = -0.67
    """박스 안 태클 파울 로짓 보정(페널티 직결이라 엄격)."""

    # 차징 파울 로짓
    charge_bias: float = -7.8
    """차징 파울 로짓 절편."""
    kc_speed: float = 0.18
    """차징 접근속도 계수."""
    kc_behind: float = 1.30
    """차징 등 뒤 접근 계수."""
    kc_shoulder: float = 1.20
    """어깨싸움(정당 몸싸움) 억제 계수."""
    kc_ballfar: float = 1.00
    """공이 먼 상태(볼 미플레이) 계수."""
    charge_play_dist: float = 1.5
    """공이 이 거리 밖이면 '볼 미플레이'로 간주(m)."""
    charge_contact_padding: float = 0.15
    """두 선수 반지름 합에 더하는 차징 접촉 허용 여유(m)."""
    charge_p_min: float = 0.003
    """차징 파울 확률 하한."""
    charge_p_max: float = 0.10
    """차징 파울 확률 상한. [B9] 0.18→0.10 — @snapshot 차징 로짓이 kc항으로 상한에 자주 포화해 상한 자체가
    실효 확률을 지배, 하향으로 차징 파울 과다 억제."""

    # 카드/퇴장 (K리그 일반 파울 동일행 카드 기준; 독립 징계 사건은 별도 구조 과제)
    card_per_foul: float = 0.1297
    """중앙선·45분 기준 카드 확률. K리그 120경기 일반 접촉 파울 2,676건의
    위치·시간 조건부 로지스틱 적합값이며, 전체 관측 카드율은 352/2,676=0.1315이다."""
    card_attack_progress_logit_weight: float = -1.2860
    """가해자 공격방향 진행도(0=자기 골, 1=상대 골)의 카드 로짓 계수.
    자기 진영의 득점기회 저지·위험 파울이 더 자주 경고되는 실측 경사를 보존한다."""
    card_elapsed_fraction_logit_weight: float = 0.8028
    """90분 경기 진행률의 카드 로짓 계수. 경기 후반 카드율 증가를 반영한다.
    짧은 실험 episode도 실제 시계의 첫 구간으로 취급하며 episode 길이에 재정규화하지 않는다."""
    red_given_card: float = 0.03125
    """카드 중 직접 레드 확률(=11/352). 두 번째 옐로 동시표기 8건은 직접 레드에서
    제외하며, 조건부 레드 모델을 적합하기에는 표본이 작아 이 집계 확률만 사용한다."""


@dataclass(frozen=True)
class Reward:
    """보상 설정. mode로 sparse/dense 전환(jit 정적). dense는 sparse(골)를 포함하고 전진
    셰이핑·소유획득 보너스를 더한다. 전진 셰이핑은 potential-based F=γ·Φ(s′)−Φ(s)(Ng 1999)
    로 shaping_gamma가 트레이너 할인율과 일치하고 Φ 연속일 때 무편향."""

    mode: str = "sparse"
    """"sparse" | "dense" — dense는 아래 가산항 포함."""
    goal: float = 1.0
    """골 보상 — 득점팀 +goal / 실점팀 -goal(양 모드 공통, per-player)."""
    advance: float = 0.10
    """[dense] 공 전진 셰이핑 가중: advance·(γΦ′−Φ), Φ=ball_x·attack_dir/hx."""
    shaping_gamma: float = 0.99
    """[dense] 셰이핑 γ — 트레이너 할인율과 일치시켜야 무편향."""
    poss_gain: float = 0.05
    """[dense] 소유권 획득 순간(팀 기준) 1회 보너스(상대는 -)."""
