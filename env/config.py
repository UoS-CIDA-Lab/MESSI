"""
환경에 필요한 객체 정의 및 특징값 정리
"""

import math
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Ball:
    radius: float = 0.11
    """공 반지름(m)"""
    mass: float = 0.43
    """공 질량(kg)"""
    area: float = 0.038
    """공 단면적(m^2)"""

@dataclass(frozen=True)
class Agent:
    id: int = 0
    """선수 ID(``range(N)``의 원소)."""
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
    싶으면 호출자가 Agent 필드에 직접 설정할 것(setup._meta는 받은 값을 그대로 사용)."""
    init_pos: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """선수 초기 위치(x, y)"""
    
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
     
    @property # 함수를 attribute처럼 쓰기 위해 property 데코레이터 사용
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

@dataclass(frozen=True)
class DeadBall:
    """데드볼 내장 배치 정책의 조정값.

    물리·규칙 계수인 :class:`Engine`과 분리하되 모든 배치 수치를 이 객체가 소유한다.
    ``movement._deadball_target`` 수식에 익명 거리·혼합비가 흩어지지 않게 하는 SSOT다.
    """

    attack_push_base: float = 16.0
    attack_push_gain: float = 12.0
    attack_own_inset: float = 3.0
    attack_opp_inset: float = 5.0
    attack_home_y_weight: float = 0.5
    defend_retreat: float = 10.0
    defend_field_inset: float = 2.0
    defend_home_y_weight: float = 0.6

    throw_attack_push_base: float = 20.0
    throw_attack_push_gain: float = 12.0
    throw_attack_own_inset: float = 6.0
    throw_attack_opp_inset: float = 4.0
    throw_attack_home_y_weight: float = 0.4
    throw_attack_y_inset: float = 3.0
    throw_defend_retreat: float = 4.0
    throw_defend_home_y_weight: float = 0.5
    throw_defend_field_inset: float = 2.0

    free_kick_keep_back_line: float = -0.5
    free_kick_push_base: float = 24.0
    free_kick_push_gain: float = 18.0
    free_kick_progress_floor: float = -0.2
    free_kick_own_inset: float = 8.0
    free_kick_opp_inset: float = 2.5
    box_depth_fraction: float = 0.5
    free_kick_commit_start: float = 0.1
    free_kick_commit_span: float = 0.5
    free_kick_home_y_weight: float = 0.3
    free_kick_box_y_padding: float = 3.0
    free_kick_back_push: float = 6.0
    free_kick_back_home_y_weight: float = 0.7
    corner_home_y_weight: float = 0.6

    penalty_line_padding: float = 1.0
    gk_line_offset: float = 0.5
    gk_ball_y_weight: float = 0.3
    field_inset: float = 1.0
    target_slowdown_radius: float = 2.0

    def __post_init__(self):
        values = vars(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("DeadBall values must be finite")
        weights = (
            self.attack_home_y_weight,
            self.defend_home_y_weight,
            self.throw_attack_home_y_weight,
            self.throw_defend_home_y_weight,
            self.box_depth_fraction,
            self.free_kick_home_y_weight,
            self.free_kick_back_home_y_weight,
            self.corner_home_y_weight,
            self.gk_ball_y_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("DeadBall blend weights must lie in [0, 1]")
        positive = (
            self.attack_push_base,
            self.attack_push_gain,
            self.attack_own_inset,
            self.attack_opp_inset,
            self.defend_retreat,
            self.defend_field_inset,
            self.throw_attack_push_base,
            self.throw_attack_push_gain,
            self.throw_attack_own_inset,
            self.throw_attack_opp_inset,
            self.throw_attack_y_inset,
            self.throw_defend_retreat,
            self.throw_defend_field_inset,
            self.free_kick_push_base,
            self.free_kick_push_gain,
            self.free_kick_own_inset,
            self.free_kick_opp_inset,
            self.free_kick_commit_span,
            self.free_kick_box_y_padding,
            self.free_kick_back_push,
            self.penalty_line_padding,
            self.gk_line_offset,
            self.field_inset,
            self.target_slowdown_radius,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("DeadBall distances and scales must be positive")
        if not -1.0 <= self.free_kick_progress_floor <= 1.0:
            raise ValueError("free_kick_progress_floor must lie in [-1, 1]")


@dataclass(frozen=True)
class RulePolicy:
    """관측 기반 룰 정책의 공용 튜너.

    역할/스타일 열 번호는 ``constants.py``의 불변 계약이고, 성능을 바꾸는 수치는 이 객체만
    소유한다. ``make_rule_based_policy(..., policy_config=...)``로 실험별 override할 수 있다.
    """

    use_shot_solver: bool = True
    style_sample_count: int = 2
    role_defender_max_x: float = -33.0
    role_midfielder_max_x: float = -18.0
    shoot_gain: float = 2.6
    support_count: int = 5
    support_scale: float = 0.55
    dribble_cone_length: float = 5.0
    width_base: float = 1.0
    width_gain: float = 0.22
    ballside_slide: float = 0.12
    wide_home_threshold: float = 22.0
    wide_progress_gate: float = 0.10
    wide_pitch_fraction: float = 0.78
    defense_x_home_weight: float = 0.35
    defense_x_line_weight: float = 0.78
    defense_y_home_weight: float = 0.66
    defense_y_ball_weight: float = 0.28
    gk_prediction_decel: float = 5.0
    anti_clump_radius: float = 9.0
    anti_clump_gain: float = 1.5

    def __post_init__(self):
        if not isinstance(self.use_shot_solver, bool):
            raise ValueError("use_shot_solver must be bool")
        if (
            not isinstance(self.style_sample_count, int)
            or isinstance(self.style_sample_count, bool)
            or self.style_sample_count <= 0
        ):
            raise ValueError("style_sample_count must be a positive integer")
        if (
            not isinstance(self.support_count, int)
            or isinstance(self.support_count, bool)
            or self.support_count <= 0
        ):
            raise ValueError("support_count must be a positive integer")
        numeric = tuple(
            value for name, value in vars(self).items()
            if name not in {"use_shot_solver", "style_sample_count", "support_count"}
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("RulePolicy values must be finite")
        if self.role_defender_max_x >= self.role_midfielder_max_x:
            raise ValueError("role_defender_max_x must be smaller than role_midfielder_max_x")
        if (
            self.shoot_gain <= 0.0
            or self.support_scale <= 0.0
            or self.dribble_cone_length <= 0.0
            or self.gk_prediction_decel <= 0.0
            or self.anti_clump_radius <= 0.0
            or self.anti_clump_gain <= 0.0
        ):
            raise ValueError("RulePolicy gains, scales, and distances must be positive")
        fractions = (self.wide_progress_gate, self.wide_pitch_fraction)
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("RulePolicy pitch fractions must lie in [0, 1]")

@dataclass(frozen=True)
class Engine:
    """환경 엔진 상수 (서브스텝은 ``dt_phys``, 컨트롤 간격은 ``decimation·dt_phys``).

    ── 캘리브 provenance·검증상태 (논문 정직성) ────────────────────────────────
    상수 주석의 `[DFL calib/fit_*]` 출처 표기는 **부모 프로젝트 AAMAS2027의 캘리브 파이프라인**
    (`../../AAMAS2027_1/calib/fit_*.py`)에서 적합된 값을 가리킨다 — 그 스크립트들은 구 env(가속·10D)
    데이터 파이프라인(`_env_ref`·`harvest.py`)에 결합돼 있어 SoccerBC로 그대로 이식되지 않았다(상속값).
    SoccerBC 자체 `calib/`는 **프레임 동기 불필요**한 부분집합만 독립 재검증한다:
      · 재검증 가능(SoccerBC calib): vmax·a_max·brake·accel_norm·turn_rate(트래킹 유한차분 p99.9),
        tackle_prob·파울/카드율(이벤트 개수), reach_xy(이벤트-확인 비순환), c_ground_curl(REVIEW).
      · **재검증 불가 → 상속·N/A**: 프레임정밀이 필요한 값(launch_max·spin_*·gk_catch_speed_cap·
        throw_speed_max·throw_height)은 이벤트 시각 지터(±0.85s)로 SoccerBC 트래킹 단독 재측정 불가 —
        report가 **N/A로 명시**하며, config 값은 부모 캘리브 상속(3자리 유효숫자 주장은 그 한계 내).
      · **미적합(추정/문헌 앵커)**: c_drag·c_magnus(문헌 앵커), ball_inertia_ratio(문헌 2/3),
        bounce_tangential_e(sticking 기본, fit_bounce 부재), body_spin_keep, 스태미나 계열(설계 추정).
    즉 "DFL-캘리브"는 **위 재검증 가능 부분집합**에 대한 주장이며, 상속·추정값은 그 한계를 명시한다.
    ────────────────────────────────────────────────────────────────────────────
    """
    # 시간 / 프레임 관련 팩터
    dt_phys: float = 0.01
    """물리 계산 서브스텝 시간 간격(s)."""
    decimation: int = 10
    """컨트롤 스텝당 서브스텝 수 — 1 컨트롤 스텝 = decimation * dt_phys 초""" 

    # 선수 관련 팩터
    r_player: float = 0.23
    """선수 반지름(m)""" 
    bench_first_x_offset: float = 4.0
    """퇴장 선수 벤치 배치의 첫 x 오프셋(m)."""
    bench_spacing: float = 4.0
    """퇴장 선수 벤치 슬롯 간격(m)."""
    bench_boundary_inset: float = 1.0
    """벤치 슬롯 x를 경기장 끝에서 안쪽으로 제한하는 여유(m)."""
    bench_touchline_inset: float = 0.4
    """벤치 슬롯 y를 터치라인 안쪽에 두는 거리(m)."""
    a_max: float = 8.68  # [DFL calib p99.9] 실측 종가속 p99.9
    """선수 최대 가속도(m/s^2) — [속도-명령 모델] 진행방향(종) **가속** 상한. **[p99.9 캘리브]** 실측 종가속 p99.9=8.68. @snapshot"""
    accel_norm_max: float = 8.46  # [DFL calib p99.9] 실측 횡가속 p99.9
    """[속도-명령 모델] **선회(횡·법선) 가속 상한**(m/s²). **[p99.9 캘리브]** 실측 횡가속 p99.9=8.46 @snapshot
    (선회는 종가속보다 약하다). 목표속도 클립의 수직 성분 캡. a_max(종)·brake_decel_max(종감속)와 3분할."""
    turn_rate: float = 4.76  # [DFL calib p99.9] 실측 속도-heading 각속도 p99.9
    """선수 최대 회전속도(rad/s) — facing 회전(파울/렌더용). **[p99.9 캘리브]** 실측 속도방향 각속도 p99.9=4.76. @snapshot
    이동 속도 선회는 accel_norm_max가 지배."""
    brake_decel_max: float = 10.51  # [DFL calib p99.9] 실측 종감속 p99.9
    """[속도-명령 모델] 진행방향(종) **감속** 상한(m/s²). **[p99.9 캘리브]** 실측 종감속 p99.9=10.51 @snapshot
    (감속이 가속보다 강함). 목표속도 클립의 종 음성분 캡."""
    sep_iters: int = 2
    """_separate 반복 횟수 — 선수 간 최소거리(``r_player + r_player``) 겹침 밀어내기."""

    sprint_speed: float = 5.5
    """스프린트 기준 속도(m/s) — 초과 시 추가 스태미나 소모"""
    stamina_sprint_mult: float = 3.0
    """스프린트 소모 배율의 외삽 기울기. 기준속도는 ``sprint_speed``, 끝점 배율은
    ``stamina_sprint_mult``이며 GK는 스태미나를 소모하지 않는다(movement 참조)."""
    vmax_floor: float = 0.91
    """유효 vmax = vmax×(floor + (1-floor)×stamina) — 스태미나 소진 시 속도 하한 비율"""
    stamina_end_frac: float = 0.35
    """경기 종료 시 기저(비질주) 선수 목표 잔여 스태미나 — drain_base 산출 기준"""

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
    reach_xy: float = 1.6    # 필드 선수 고정 reach 반지름
    """선수 reach 반지름(m) — 공과 선수 간 거리 ≤ reach_xy → reach=True. 속도-의존 확장(reach_xy_slide)
    폐지 → **필드 선수 고정 상수**. GK만 별도(gk_reach_xy).

    **[2026-07-29 재캘리브: 1.2 → 1.6, p90 컷 → p95 컷]** @snapshot
    구값 1.2는 [DFL calib/eval_touches]의 '접촉거리 p90≈1.25m'에서 왔으나, 그 집합은 **트래킹
    임펄스에 스냅되지 않은 이벤트**(액터가 공에서 7~23m 떨어진 행정 타임스탬프 프레임)를 포함해
    분포가 오염돼 있었다. J03WMX 전·후반 전체 3,322접촉을 라벨 출처로 분해해 재측정:
      · 임펄스 증거 있음(n=2,888): p50 0.42 · p90 1.05 · p95 1.71 · p99 2.51
      · 임펄스 증거 없음(n=434)  : p50 1.30 · p90 10.38 — 물리적 접촉이 아님(오염원)
    @snapshot 깨끗한 집합(필드+임펄스, n=2,703)의 **p95=1.73m**를 채택 근거로 삼았다.
    ※깨끗한 집합의 p90은 1.06m로 구 캘리브 근거(1.25)보다 **작다** — 즉 구값 1.2는 이미 p92~93
    수준이었고, 이번 변경은 '분위수 기준을 p90→p95로 한 단계 완화'로 읽어야 정확하다.

    효과: 실측 접촉 중 env가 표현 못 하는 비율이 2.7% → 1.6%(접촉 프레임을 정확히 잡았을 때 기준). @snapshot
    미표현이 남는 대표 부류는 인터셉트·태클 등 수비 개입 동작이다.

    ★파생 영향: ``kicker_arrive_r``가 None이면 이 값에서 파생되므로 세트피스 키커의 도착 판정
    반경도 함께 ``reach_xy``로 맞춰진다(의도적으로 결합 유지 — 강제 세트피스 킥은 in_reach를 우회하므로
    arrive_r이 reach보다 크면 도달 밖에서 발사된다). 분리하려면 kicker_arrive_r을 명시 지정할 것."""
    reach_height_factor: float = 0.85
    """선수 머리 높이 factor — head_z = tall * reach_height_factor.
    계수가 1이면 머리 높이가 키와 같고, 더 작으면 그 비율만큼 낮아진다."""
    # [폐지] reach_max_speed / reach_xy_slide — 속도-의존 도달 확장 메커니즘 제거(_in_reach 고정반경). 미사용.
    gk_reach_xy: float = 2.0  # Calib 요구
    """골키퍼가 '자기 페널티박스 안'에 있을 때의 수평 reach 반지름(m) — 손 사용·다이브 커버."""

    # 골키퍼 캐치/홀드
    gk_catch_speed_cap: float = 26.4
    """GK가 박스 안에서 공을 '잡을 수 있는' 최대 공속(m/s). 이하=캐치(홀드), 초과=parry(쳐냄).
    [DFL calib] GK 캐치(수신 후 정지) 230건 입사속도 p99.5=26.4 — 설계 초기값 25와 근접. @snapshot"""
    gk_hold_substeps: int = 800
    """GK 캐치 홀드 창(서브스텝). 실제 시간은 ``gk_hold_substeps * dt_phys``이며 IFAB 홀드 규칙을 따른다.
    이 안에 배급 안 하면 만료 강제 배급.
    (obs rt_norm·restart 만료 강제킥이 이 값을 파생 — 바꾸면 자동 전파.)"""

    # force2ball (킥)
    cooldown_substeps: int = 16
    """force2ball 쿨다운(서브스텝) — 실제 시간은 ``cooldown_substeps * dt_phys``.
    [DFL calib/fit_cooldown 검증] event XML 무검열 라벨의 같은선수 킥→재터치 클린 최소 간격이
    정확히 0.16s(3,121쌍 중 미만 0건) — 데이터 하한과 일치, 유지. @snapshot 트래킹 검출의 3프레임 간격
    '터치'는 듀얼 분열·유령 검출이었음(replay 병목의 원인 재귀속)."""

    # 세트피스 재개
    restart_substeps: int = 500
    """재개 윈도우(서브스텝) — 선수 배치 시간 확보"""
    setup_hold_substeps: int = 300
    """재개 전 선수 배치 유지 시간(서브스텝) — 진영이 잡히고 나서 재개 전까지 슛하지 않도록 강제.
    [B] 킥 타이밍 강제 하에선 이 값이 곧 '키커 도착→결정론적 발사'까지의 지연(=데드볼 엔진 배치 시간)이다.
    실제 축구 세트피스에 배치 시간이 필요하다는 점을 반영한다. 이 값은 모든 종류별 재개
    window보다 작아야 ``setup_done``이 성립한다. ``kickoff_instant``인
    킥오프는 이 홀드를 우회(즉시 발사). 종류별 세분(프리킥>스로인 등)은 향후 per-kind 값으로 확장 가능."""
    throwin_clear: float = 2.0
    """스로인 재개 시 선수 반경(clear) 거리(m) — 스로인 재개 시 공과 선수 간 거리 ≥ throwin_clear → 규정 준수."""
    clear_dist: float = 9.15 
    """그 외 재개 시 선수 반경(clear) 거리(m) — 재개 시 공과 선수 간 거리 ≥ clear_dist → 규정 준수."""
    kicker_speed: float = 6.0
    """키커 강제이동 속도(m/s) — 조깅 접근 수준. 도착 전 스태미나 무소모"""
    kicker_arrive_r: float | None = None
    """키커 도착 판정 반경(m). None이면 ``reach_xy``에서 자동 파생."""
    kickoff_instant: bool = True
    """킥오프 즉시 발동 — True면 킥오프(RK_KICKOFF)는 setup_hold·카운트다운 대기 없이 키커가 스폿에
    도착하는 즉시 강제 킥. 경기 시작/득점 후 전원이 배치 카운트다운을 서서 기다리는 것을 없앤다.
    다른 세트피스(스로인·프리킥·골킥·페널티)는 영향 없음. False면 종전대로 restart_substeps 대기."""
    deadball_engine: bool = False
    """★ 이 플래그를 끄면 데드볼 이동도 학습 대상이다(2026-08-03 변경).

    True면 재개(`restart_t > 0`) 동안 비-키커 전원의 이동을 `_deadball_move`가 강제한다.
    그런데 실측하면 그 휴리스틱이 체계적으로 틀렸다:

    | 지표 (재개 순간, 현실/엔진) | 킥오프 | 골킥 | 프리킥 |
    |---|---|---|---|
    | 수비 전진도 | −12.9/**−34.9** | **+2.5/−34.9** | −18.5/**−34.9** |
    | 수비↔공 거리 | 19.6/**38.2** | 42.6/**77.7** | 29.5/**43.8** |
    | 공 10 m 내 수비 | 2.9/**0.0** | 0.2/0.0 | 1.1/0.6 |

    골킥에서는 **부호가 반대**다 — 현실은 전진 압박인데 엔진은 자기 진영으로 물러난다(37 m 차).
    `_deadball_target` 자신도 TODO에 "수비벽 선발·오프사이드 라인·마킹 매칭"을 미구현으로
    남겨 뒀다. 코너킥만 전용 분기가 있어 잘 맞는다.

    이제 실데이터 데드볼 라벨이 있다(`recon/deadball.py` — 695구간·이동감독 157만).
    엔진 대신 그걸로 배운다. 룰 정책 60초 실측으로 **재개가 정상 종료되고**(구간 길이
    75f, 엔진 켠 76f와 동일) 정체 0건임을 확인했다. `move_forced`가 13.8 % → 5.0 %로
    줄고(남은 5 %는 키커 강제 + 퇴장) 그만큼이 학습 범주로 들어온다.

    **어블레이션용으로 코드는 남겨 둔다** — True로 되돌리면 예전 동작 그대로다.
    "휴리스틱 데드볼 vs 학습"은 그 자체로 논문 결과다.

    ── 원래 설명 ──
    데드볼 내장 엔진 — True면 재개(restart_t>0, 데드볼) 동안 비-키커 전원의 이동을 env 내장
    컨트롤러(_deadball_move)가 공수 역할별로 강제 제어한다. 정책은 ball_alive에서만 이동을 몰고,
    데드볼 이동은 엔진 소관(→ 그 프레임 전원 move_forced=True로 BC 마스킹, 골/하프타임 텔레포트도 흡수).
    킥 자체(release 순간의 파라미터)는 여전히 정책이 소유(타이밍만 env가 setup_done에 고정). 껐다 켜서
    순수 정책 데드볼과 비교 가능. 컨트롤러의 정교함(벽·마킹·런)은 _deadball_target 두뇌를 교체해 개선."""
    
    # 재탈취 지연
    ctrl_lock_substeps: int = 16
    """재탈취 지연(서브스텝) — 실제 시간은 ``ctrl_lock_substeps * dt_phys``."""

    # ── 오프사이드 ─────────────────────────────────────────────
    offside_margin: float = 0.5
    """2번째 최종수비 라인 여유(m) — 이보다 앞선 동료만 오프사이드 위치로 플래그."""
    pass_protect: int = 240
    """패스 후 오프사이드 판정 윈도우(서브스텝) — 롱볼 도달까지 창 유지."""

    # ── 페널티 / 차징 ───────────────────────────────────────────
    penalty_spot: float = 11.0
    """골라인에서 페널티 스폿까지(m)."""
    penalty_substeps: int = 540
    """페널티 빌드업 윈도우(서브스텝) — 전원 박스 밖 정렬 시간."""
    penalty_settle_speed: float = 0.5
    """인플레이 페널티 종결 판정 임계 속도(m/s) — 킥 후 골/아웃/세이브 없이 공이
    지면에서 이 속도 미만으로 정지하면 '무득점 종결'로 간주(플라이트 플래그 해소 페일세이프)."""
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
    β_poss=0.82(→0.54)는 부재 스크립트라 감사불가이고, SoccerBC의 유일 감사가능 통계(이벤트 소유측
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
    pelvis_frac: float = 0.55
    """골반 높이 = head_z×이값. 이하 접촉=발 슛(풀파워)."""
    chest_cap: float = 0.87
    """골반~머리 높이 접촉 파워 상한(×f2b_speed_max). [DFL fit_kick] 실측비 p99.5 29.4/33.7=0.87 — @snapshot
    이 밴드엔 '가슴 높이 발리킥'(발을 들어 풀스윙)이 포함되므로 순수 가슴트랩보다 높은 게 실측."""
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
    """탈취 시 공속 이 이상=인터셉트, 미만=태클(이벤트 분류)."""
    deflect_prob: float = 0.50
    """P(굴절 | 경합 & 파울아님 & 탈취실패) — 접촉했으나 소유 불변 루즈볼."""
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
    f2b_shoot_range: float = 26.0
    """이 거리 안 + 골 정면 조준이면 슛으로 분류."""
    dribble_speed_max: float = 9.0
    """자기팀 자기터치가 이 속도 이하면 드리블 터치로 라벨."""
    shot_aim_cos: float = 0.5
    """슛 분류의 골 방향 코사인 임계값."""
    throw_speed_max: float = 21.5
    """스로인(머리 위 손던지기) 최대 발사속도(m/s). [DFL calib/fit_throwin] 릴리즈 268건
    (터치라인·창 일관성·천장 필터) 강건속도 p99.5=21.5 — 전 캡 공통 p99.5 앵커 규약. @snapshot"""
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
    c_drag: float = 0.0154
    """공기저항 계수 — drag = -c_drag·|v|·v (공중 비행만)."""
    c_magnus: float = 0.0058
    """마그누스 계수 — a = c_magnus·(spin×v). 휨(커브) 생성 (공중 비행만)."""
    c_ground_curl: float = 0.0
    """지상 굴림 컬 계수 — a_xy = c_ground_curl·(spin_z ẑ×v) (지면만·수평만). 영이면 비활성이다.
    측스핀이 굴러가는 공을 횡으로 휘게 하는 효과이나, 스핀 비관측이라 트래킹만으론 **깨끗이 식별 불가**:
    fit_ground_curl의 지상/공중 p99 비는 부트스트랩 95%CI [0.73,1.02](공중 컬과 통계적 구분 불가)이고,
    지상 median |q|=0.05/s의 비영 baseline이 피치기울기·노이즈·감속 등 **비스핀 confound**를 시사한다.
    방어 불가능한 물리항을 기본 탑재하지 않는다(보수적) — 물리 코드 경로는 유지(×0)하여 스핀 관측/전용
    실험으로 재검증되면 값 복원(예: 0.00506)만으로 재활성 가능. 근거는 calib/report.py 참조."""
    e_rest: float = 0.61
    """지면 반발계수(vz 부호반전 시 감쇠) — 실제 임팩트 전용."""
    spin_decay: float = 0.31
    """비행 중 스핀 감쇠율(1/s)."""
    ball_inertia_ratio: float = 0.667
    """공 관성비 α = I/(m·r²). 축구공은 얇은 구껍질 → α≈2/3(속이 찬 구 2/5보다 큼) [문헌 앵커].
    바운스 시 스핀↔병진 접선 임펄스 결합의 물리 계수 — 접촉점 각운동량 보존을 강제한다."""
    bounce_tangential_e: float = 0.0
    """접선 반발계수 e_t ∈[0,1] — 바운스 시 스핀→병진 전달의 탄성. 0=sticking(잔디 그립,
    무자유파라미터·표준 거친바운스, 에너지 감소), 1=완전탄성(superball, 에너지 정확보존).
    전달 크기 = (1+e_t)·α/(1+α)·(스핀 표면속도)로 **각운동량 보존·에너지 비창출**(구식 독립
    전달+고정소산의 에너지 창출 폐기). 이상적으론 DFL 바운스로 fit(fit_bounce, 현재 부재) — 기본 sticking."""
    bounce_spin_vmin: float = 0.7
    """이 하강속도(m/s) 초과 임팩트만 스핀-속도 결합(굴림 미세진동 제외)."""
    bounce_h_keep: float = 0.89
    """바운스 임팩트 시 수평속도 보존율. [DFL calib/fit_bounce] 무접촉 바운스 1,698건 실측
    수평속도비 중앙 0.869에서 같은 서브스텝에 이미 적용되는 굴림 감속분(≈0.975)을 제한 값 —
    구현은 수평 불변(1.0)이어서 실측 대비 바운스 후 공이 ~13% 빨랐다. 스핀 결합 효과와 일부
    혼재(스핀 비관측) — 무스핀 바운스가 실측 중앙에 맞도록 설정."""
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
    body_hit_prob: float = 0.90
    """코어 정면 관통당 총 상호작용 확률(스치는 관통은 경로비례로 자동 감소).
    [DFL calib/fit_body] 몸통밴드 통과 에피소드 1,248건 상호작용률 89.6%(속도별 86~97%)."""
    e_body: float = 0.45
    """몸통 반발계수(바운스)."""
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

    # ── 규칙/렌더가 공유하는 기하 허용치 ───────────────────────
    restart_field_inset: float = 0.50
    """코너·스로인 재개 스폿을 선 안쪽에 두는 공통 여유(m)."""
    throwin_line_inset: float = 0.15
    """스로인 공 중심을 터치라인 안쪽에 두는 여유(m)."""
    free_kick_boundary_inset: float = 1.0
    """파울 재개 스폿을 필드 경계에서 안쪽으로 제한하는 여유(m)."""
    goal_line_tolerance: float = 1.2
    """재개 이격 면제/페널티 GK 판정에서 '골라인 위'로 인정하는 x 밴드(m)."""
    goal_post_tolerance: float = 0.30
    """골라인 위 선수 판정의 포스트 바깥 y 여유(m)."""
    legal_margin_floor: float = 0.05
    """규칙상 면제된 선수에게 관측으로 내보내는 최소 양의 이격 여유(m)."""
    unrestricted_margin: float = 1.0
    """이격 의무가 없는 GK 홀드에서 관측으로 내보내는 최소 양의 여유(m)."""

    def __post_init__(self):
        # dataclass 본문의 ``kicker_arrive_r = reach_xy`` 같은 복사는 클래스 정의 시 한 번만
        # 평가되어 Engine(reach_xy=...) override와 어긋난다. 파생값은 인스턴스 생성 뒤 동기화한다.
        if self.kicker_arrive_r is None:
            object.__setattr__(self, "kicker_arrive_r", self.reach_xy)
        if self.norm_spin is None:
            object.__setattr__(self, "norm_spin", self.spin_max)


@dataclass(frozen=True)
class Foul:
    """반칙 선언·카드 기준(IFAB Law 12). 태클 파울과 차징 파울을 로지스틱으로 판정하고,
    선언 시 확률적 카드→퇴장. 로짓 계수는 DFL analyze_fouls 적합값(부호·크기 실측 채택, 상속).
    보상엔 관여하지 않는 순수 규율(수적 열세는 물리로만 불리).

    태클 채널은 경합 승자의 탈취 파울뿐 아니라 근접 도전자가 공을 못 따고 끝난 loser-foul도
    ``contest.retained`` 분기로 표현한다. 카드/레드율은 파울 160·레드 2 소표본이라
    레드 비율의 불확실성은 여전히 크다(CI≈0.7–18%)."""

    # 태클 파울 로짓
    tackle_bias: float = -2.56
    """태클 파울 절편 — 듀얼당 파울률 앵커.

    **[2026-07-29 재캘리브: -1.66 → -2.56, Δ=-0.90]** 구 앵커 "DFL P(foul|duel)=160/1412=0.113"의
    **분모가 틀렸다**. 그 0.113의 분모는 DFL이 *주석을 붙인* `TacklingGame`(2.35/분)인데, env가
    실제로 파울을 추첨하는 상황은 **기하학적 근접 경합**(상대가 공 reach 안, 실측 5.64/분)이다.
    같은 기하 정의를 실측에 적용하면(calib `check_geometric_duels`):
        P(foul | 기하 듀얼) = 0.22 파울/분 ÷ 5.64 듀얼/분 = **0.0383**
    반면 env 창발값은 0.1585(룰 정책 롤아웃) → **4.1배 과대**. 로짓 하향 응답을 실측해
    (Δ-1.2에서 0.0206) 로그선형 보간한 Δ=-0.90을 채택.

    ※`charge_bias`도 동일 Δ로 함께 내린다(두 채널 합이 총 파울률을 만들므로).
    ※`tackle_p_max`/`charge_p_max` 포화는 아님이 확인됨 — p_max를 함께 낮춰도 결과 불변이었다.
    ※**듀얼 수 자체는 보정하지 않는다**: env의 듀얼 발생률(13.7/분)이 실측(5.64/분)보다 높지만
    그건 env 규칙이 아니라 **정책의 성질**(룰 정책이 공 주변에 몰림)이다. 절대 파울 수를 맞추려고
    확률을 더 내리면 정책이 바뀌는 순간 파울이 사라진다. env 규칙인 P(foul|duel)만 실측에 맞춘다.
    카드율은 이 파울율×card_per_foul로 자동 종속."""
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
    """차징 파울 절편(대폭 억제). [2026-07-29] `tackle_bias`와 동일 Δ-0.90 하향 — 차징은 태클과
    별개 파울 채널이라(DFL은 단일 Foul 범주) 두 채널을 함께 스케일 다운해야 총 듀얼당 파울률이
    실측 목표 0.0383에 수렴한다. 측정된 채널 비중은 태클 85% · 차징 15%로 태클이 지배적."""
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

    # 카드/퇴장 (DFL: 파울 160건 중 카드 37 = 옐로 35 + 레드 2)
    card_per_foul: float = 0.23125
    """파울당 카드 확률(=37/160)."""
    red_given_card: float = 0.054054
    """카드 중 레드 확률(=2/37). 레드 1장 or 옐로 2장 = 퇴장."""


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
