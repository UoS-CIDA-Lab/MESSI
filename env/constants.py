"""환경 전역 불변값의 단일 진실원천.

물리·규칙의 조정 가능한 값은 :mod:`config`의 dataclass에 두고, 여기에는 팀/상태 코드,
행동 레이아웃, sentinel, 수치 안정성 상수처럼 모든 환경에서 바뀌면 안 되는 값만 둔다.
핫패스 수식에 익명 숫자를 직접 쓰지 않도록 이 파일의 이름 있는 상수를 사용한다.
"""

import math
from types import MappingProxyType


# 공통 팀/선수 sentinel
TEAM_0 = 0
TEAM_1 = 1
TEAM_COUNT = 2
NO_TEAM = -1
NO_PLAYER = -1
NO_EVENT = -1
SAMPLED_WINNER = -2

# 수치 안정성. float32에서 실제로 구분 가능한 확률 하한과 기하/노름 하한을 분리한다.
PROB_EPS = 1.0e-6
GEOMETRY_EPS = 1.0e-6
DIV_EPS = 1.0e-9
SQUARED_EPS = 1.0e-12
SAFE_NORM_EPS = 1.0e-18
STATIONARY_SPEED_EPS = 1.0e-3
COINCIDENT_DISTANCE_EPS = 1.0e-4

# 공용 기하
HALF_TURN = math.pi
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# 경기&공 상태 
BALL_DEAD = 0
"""인플레이 상태 아님"""
BALL_ALIVE = 1
"""인플레이 상태"""

# 인플레이 재개 종류
RK_NONE = 0
"""재개 없음(인플레이)"""
RK_KICKOFF = 1
"""킥오프 — 경기 시작·득점 후·후반 시작"""
RK_THROWIN = 2
"""스로인 — 터치라인 아웃"""
RK_GOALKICK = 3
"""골킥 — 공격팀 최종터치 후 골라인 아웃"""
RK_CORNER = 4
"""코너킥 — 수비팀 최종터치 후 골라인 아웃"""
RK_FREEKICK = 5
"""프리킥 — 태클/차징 파울·스로인 재터치 등"""
RK_PENALTY = 6
"""페널티킥 — 자기 박스 안 파울. 키커가 실제 킥, GK는 필드선수로 경합(물리 플레이)"""
RK_OFFSIDE = 7
"""오프사이드 콜에 의한 수비팀 프리킥(FREEKICK과 통계 분리용 별도 코드)"""
RK_GK_HOLD = 8
"""GK 캐치 홀드 — GK가 자기 박스 안에서 공을 잡아 소유 정지. ``Engine.gk_hold_substeps`` 창 안에
GK가 스로/킥으로 배급, 만료 시 강제 배급. 상대는 도전 불가(clear). 배급 전까지 인플레이 아님."""
RESTART_COUNT = 9

# 터치 종류
TOUCH_NONE = 0
"""접촉 없음."""
TOUCH_PASS = 1
"""발 패스 킥."""
TOUCH_SHOOT = 2
"""발 슛 킥."""
TOUCH_PASS_HEAD = 3
"""헤딩 패스 — head_z 위 접촉."""
TOUCH_SHOOT_HEAD = 4
"""헤딩 슛 — head_z 위 접촉."""
TOUCH_DRIBBLE = 5
"""드리블 소프트 자기터치 — 패스/슛과 속도 임계로 구분."""
TOUCH_TACKLE = 6
"""태클로 공 탈취(파울 태클 포함 라벨)."""
TOUCH_GK_CATCH = 7
"""GK 캐치/세이브."""
TOUCH_INTERCEPT = 8
"""인터셉트 — 빠르게 움직이는 공(상대 패스 등)을 탈취."""
TOUCH_DEFLECT = 9
"""굴절 — 공에 닿았지만 탈취 아님. IFAB상 '의도적 플레이'가 아니라
오프사이드 플래그를 리셋하지 않음(INTERCEPT/TACKLE은 리셋됨)."""
TOUCH_PARRY = 10
"""GK가 잡지 못하고 쳐낸 세이브. 일정 속도 이상인 경우 공을 잡을 수 없음"""
TOUCH_COUNT = 11
"""터치 코드 종류 수(TOUCH_NONE..TOUCH_PARRY = 0..10) — obs one-hot 폭. 코드 추가 시 갱신."""

# 반칙 종류 
FOUL_NONE = 0
"""반칙 없음."""
FOUL_TACKLE = 1
"""태클 파울 — 공 경합 중 foul_prob 발동."""
FOUL_CHARGE = 2
"""차징 파울 — 몸싸움, 잡기."""
FOUL_THROW = 3
"""스로인 재터치 위반 — 스로어가 타인 접촉 전 재터치."""


ACTION_DIM = 8
"""행동 차원 — [kick_gate, move(2), f2b(2), f2b_launch, spin_side, spin_back].
이동·킥은 각각 **L∞ radial stretch 2D**(방향+크기를 한 2D 벡터로, box→원판 전단사·방향보존·등방).
kick_gate(dim[0])는 **[0,1] 범위 · threshold 0.5**(0.5 초과 시 킥). 그 외 차원은 [-1,1](tanh 정책 가정).
env가 클립·디코드 — _decode / spatial.stretch_decode 참조."""

ACTION_MIN = -1.0
ACTION_MAX = 1.0
KICK_GATE_THRESHOLD = 0.5
ACTION_KICK_GATE = 0
ACTION_MOVE = slice(1, 3)
ACTION_KICK_VECTOR = slice(3, 5)
ACTION_LAUNCH = 5
ACTION_SPIN_SIDE = 6
ACTION_SPIN_BACK = 7

# 저장 데이터/체크포인트가 레이아웃 변경을 조용히 통과하지 않게 노출하는 스키마 버전.
ACTION_SCHEMA_VERSION = 1
OBS_SCHEMA_VERSION = 6
"""2 — self 블록에 `abs_vel`(자기 절대속도, 관측자 attack_dir로 접힘) 2차원 추가(442 → 444).
3 — context 블록에 `pass_t`(오프사이드 창 잔여, `pass_protect` 정규화) 1차원 추가(444 → 445).
4 — 페널티 침범 래치 노출(445 → 468): others마다 `pen_encroach` 1비트(+21)와 setpiece 블록의
    팀 요약 `pen_encroach_ours_any`/`pen_encroach_theirs_any`(+2). 자기 래치(`self_pen_encroach`)만
    보이면 같은 관측에서 골 인정/재실행과 선수별 카드가 갈린다(events의 재실행 매트릭스가
    `penalty_encroach_mask`를 팀별로 가르기 때문).
5 — context에 `second_half_kickoff_ours` 1차원 추가(468 → 469). `kickoff_team`(전반 킥오프 팀)은
    후반 킥오프 팀을 결정하는데(`events._halftime_switch`) 관측에 없어, 두 상태의 obs가 완전히
    같은데 하프타임 전이 결과가 갈렸다. 경기 전체에 걸친 상수라 짧은 history로도 복원 불가.
6 — 라이브 간접 프리킥의 `restart_indirect`를 재개 타이머로 가리지 않고, 재터치 제한의 출처가
    스로인인지 일반 세트피스인지 나타내는 `retouch_is_throw` 1차원을 context에 추가(469 → 470).
    둘 다 직접 득점의 유효성을 바꾸므로 같은 관측에 상충하는 다음 상태가 생기지 않아야 한다.
구 442/444/445/468/469차원 체크포인트와 호환되지 않는다."""
STATE_SCHEMA_VERSION = 3
"""2 — 중앙 상태에 정확한 pending/setpiece/throw taker, pass_t, last-touch code, 페널티 래치와
후반 킥오프 팀을 포함한다.
3 — 라이브 간접 프리킥에서도 `restart_indirect`를 1로 유지한다. 차원은 538로 동일하지만 의미가
바뀌므로 기존 중앙 크리틱 체크포인트와 조용히 섞이지 않게 버전을 올린다."""

ROTATE_180 = -1
"""상대팀 포메이션 180° 회전(점대칭) 계수 — setup._meta에서 init_pos의 x·y 양쪽에 곱함."""

YELLOW_CARD_SEND_OFF_COUNT = 2

# SoccerEnv() 기본 생성자가 실제로 동작하도록 쓰는 표준 11인 포메이션.
# 좌표는 팀의 공격 프레임(+x 공격)이며 상대 팀은 setup._meta에서 점대칭한다.
DEFAULT_FORMATION = (
    (-45.0, 0.0),
    (-35.0, 25.0),
    (-35.0, -25.0),
    (-35.0, 15.0),
    (-35.0, -15.0),
    (-20.0, 20.0),
    (-20.0, -20.0),
    (-20.0, 0.0),
    (-10.0, 25.0),
    (-10.0, -25.0),
    (-12.0, 0.0),
)
DEFAULT_TEAM_SIZE = len(DEFAULT_FORMATION)
DEFAULT_GAME_DURATION = 135_000
DEFAULT_CONTROL_FPS = 25.0

# 공간 차원 상수
DIM_X = 0
DIM_Y = 1
DIM_Z = 2
DIM_ALL = 3

# 룰 정책의 불변 범주 레이아웃. 역할 경계·전술 강도처럼 조정 가능한 값은 config.RulePolicy가
# 소유하고, 여기에는 배열 열의 의미와 enum만 둔다.
ROLE_GK = 0
ROLE_DEFENDER = 1
ROLE_MIDFIELDER = 2
ROLE_FORWARD = 3
ROLE_COUNT = 4

STYLE_LINE = 0
STYLE_TEMPO = 1
STYLE_WIDTH = 2
STYLE_AGGRESSION = 3
STYLE_DIRECTNESS = 4
STYLE_DIM = 5

# 이름으로 공개되는 불변 프리셋. 런타임 정책 튜너는 config.RulePolicy에 있으며, 호출자가
# team_styles 배열을 직접 전달하면 이 프리셋을 우회할 수 있다.
STYLE_PRESETS = MappingProxyType({
    "balanced":     (0.5, 0.5, 0.5, 0.5, 0.5),
    "gegenpress":   (0.85, 0.8, 0.55, 0.9, 0.45),
    "park_the_bus": (0.15, 0.35, 0.35, 0.35, 0.6),
    "tiki_taka":    (0.7, 0.45, 0.35, 0.55, 0.15),
    "long_ball":    (0.4, 0.7, 0.75, 0.5, 0.95),
})


# 관측/central-state 스키마. 생성 코드와 학습 코드가 같은 이름·폭을 참조하도록 상수화한다.
OBS_SELF_FEATURES = (
    ("abs_pos", 2),
    ("abs_vel", 2),
    ("stamina", 1),
    ("vmax", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("is_gk", 1),
    ("yellow", 1),
    ("cooldown", 1),
    ("ball_ctrl", 1),
    ("in_reach", 1),
    ("f2b_avail", 1),
    ("own_offside", 1),
    ("pass_signal", 1),
    ("retouch", 1),
    ("ctrl_lock", 1),
)
OBS_OTHER_FEATURES = (
    ("rel_pos", 2),
    ("abs_pos", 2),
    ("abs_vel", 2),
    ("stamina", 1),
    ("vmax", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("is_gk", 1),
    ("yellow", 1),
    ("offside", 1),
    ("pen_encroach", 1),
    ("ball_ctrl", 1),
    ("ctrl_lock", 1),
    ("cooldown", 1),
    ("is_taker", 1),
    ("team_flag", 1),
)
OBS_SETPIECE_FEATURES = (
    ("is_sp_ours", 1),
    ("rk_onehot", RESTART_COUNT),
    ("rt_norm", 1),
    ("is_kicker_locked", 1),
    ("enc_margin", 1),
    ("any_encroacher", 1),
    ("is_fk_indirect", 1),
    ("self_is_taker", 1),
    ("self_kicker_ready", 1),
    ("pen_flight", 1),
    ("self_pen_encroach", 1),
    ("pen_encroach_ours_any", 1),
    ("pen_encroach_theirs_any", 1),
)
OBS_BALL_FEATURES = (
    ("rel_pos", 3),
    ("abs_pos", 3),
    ("abs_vel", 3),
    ("abs_spin", 3),
    ("ball_alive", 1),
    ("poss_ours", 1),
)
OBS_CONTEXT_FEATURES = (
    ("setpiece", sum(size for _, size in OBS_SETPIECE_FEATURES)),
    ("retouch_is_throw", 1),
    ("off_line", 1),
    ("pass_t", 1),
    ("time_left", 1),
    ("last_touch", 1),
    ("last_touch_code", TOUCH_COUNT),
    ("score_diff", 1),
    ("second_half_kickoff_ours", 1),
)

STATE_PLAYER_FEATURES = (
    ("pos", 2),
    ("vel", 2),
    ("face", 2),
    ("vmax", 1),
    ("ball_ctrl", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("stamina", 1),
    ("cooldown", 1),
    ("ctrl_lock", 1),
    ("on_pitch", 1),
    ("yellow", 1),
    ("pending_taker", 1),
    ("setpiece_taker", 1),
    ("throw_taker", 1),
    ("penalty_encroach", 1),
    ("offside", 1),
    ("team", 1),
    ("is_gk", 1),
)
STATE_BALL_FEATURES = (("pos", 3), ("vel", 3), ("spin", 3))
STATE_GAME_FEATURES = (
    ("poss_onehot", TEAM_COUNT + 1),
    ("last_touch_team_onehot", TEAM_COUNT + 1),
    ("attack_dir0", 1),
    ("ball_alive", 1),
    ("restart_t", 1),
    ("pass_t", 1),
    ("time_left", 1),
    ("is_fk_indirect", 1),
    ("restart_team_onehot", TEAM_COUNT + 1),
    ("pass_team_onehot", TEAM_COUNT + 1),
    ("score", TEAM_COUNT),
    ("rk_onehot", RESTART_COUNT),
    ("last_touch_code", TOUCH_COUNT),
    ("kickoff_team", TEAM_COUNT),
    ("penalty_flight_team", TEAM_COUNT + 1),
)

# star-import를 쓰는 기존 모듈과 호환하되 math 등 구현 모듈은 새어 나가지 않게 제한한다.
__all__ = tuple(name for name in globals() if name.isupper())
