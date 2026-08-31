"""환경 전역 불변값의 단일 진실원천.

물리·규칙의 조정 가능한 값은 :mod:`config`의 dataclass에 두고, 여기에는 팀/상태 코드,
행동 레이아웃, sentinel, 수치 안정성 상수처럼 모든 환경에서 바뀌면 안 되는 값만 둔다.
핫패스 수식에 익명 숫자를 직접 쓰지 않도록 이 파일의 이름 있는 상수를 사용한다.
"""

import math
from types import MappingProxyType

from .timebase import DEFAULT_TIMEBASE

# 공통 팀/선수 sentinel
TEAM_0 = 0
TEAM_1 = 1
TEAM_COUNT = 2
NO_TEAM = -1
NO_PLAYER = -1
NO_EVENT = -1
SAMPLED_WINNER = -2
# Goalkeeper handling restriction code.  The State field name ends in ``_team``
# but its value is deliberately causal, not merely a team id:
# 0/1 mean a team-mate's deliberate kick/throw to that team's goalkeeper;
# 2/3 mean that goalkeeper released hand possession and no other player has
# touched the ball yet.  The causes have the same current hand legality but a
# different transition when the keeper clearly kicks to release the ball.
GK_HANDLING_BACKPASS_OFFSET = 0
GK_HANDLING_RELEASE_OFFSET = TEAM_COUNT
GK_HANDLING_RESTRICTION_COUNT = 2 * TEAM_COUNT
# ``throw_taker``/``setpiece_taker`` normally store a live slot index.  When
# that identity leaves the pitch before anybody else touches the ball, the
# actor-specific double-touch restriction disappears but the ball's
# direct-goal/indirect-FK provenance does not.  This contextual sentinel keeps
# that phase alive without aliasing the replacement identity now occupying the
# old slot.  Its numeric value may equal a sentinel used by a different field;
# the State fields have disjoint contracts.
DEPARTED_TAKER = -2

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
GK가 스로/킥으로 배급, 만료 시 강제 배급. 상대는 도전 불가(clear).

**공은 인플레이다**(IFAB Law 9 — 골키퍼가 손으로 통제하는 동안 공은 죽지 않는다). 따라서 재개가
아니고 상대에게 이격 의무도 제외구역도 없어 박스 안 어디에나 있을 수 있다. ``restart_t``를
타이머로 재사용하므로 물리 적분은 그대로 멈추지만(``env`` 적분 게이트가
``BALL_ALIVE & ~restart_timer_active``다), 인플레이 시간·점유율·역할 누적은 정상 계상된다.

**의도된 규약 이탈**: IFAB 현행은 8초 초과 시 상대 코너킥이지만 이 엔진은 **강제 배급**을 유지한다.
코너 제재를 넣으면 정책이 "8초를 채워 코너를 내주는" 선택지를 배울 수 있고, 그건 학습에서 착취
표면이 된다. 강제 배급은 그 분기 자체를 만들지 않는다."""
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
TOUCH_BODY_TRAP = 11
"""가슴·몸통으로 공을 죽인 수동 트랩. 발의 의도적 플레이인
``TOUCH_DRIBBLE``과 분리해 GK 백패스·오프사이드·BC 인과성 판정에서 오인하지 않는다."""
TOUCH_COUNT = 12
"""터치 코드 종류 수(TOUCH_NONE..TOUCH_BODY_TRAP = 0..11). 코드 추가 시 갱신."""

# 한 물리 서브스텝의 접촉 파이프라인 순서. 능동 force-to-ball 판정이 먼저이고,
# 그 결과를 반영한 뒤 수동 몸통 충돌을 판정한다. State의 control-frame 접촉 버퍼
# 두 번째 축과 replay ``contact_phase``가 이 순서를 공유한다.
TOUCH_EVENT_FORCE = 0
TOUCH_EVENT_BODY = 1
TOUCH_EVENT_PHASE_COUNT = 2

# 반칙 종류
FOUL_NONE = 0
"""반칙 없음."""
FOUL_TACKLE = 1
"""태클 파울 — 공 경합 중 foul_prob 발동."""
FOUL_CHARGE = 2
"""차징 파울 — 몸싸움, 잡기."""
FOUL_THROW = 3
"""스로인 재터치 위반 — 스로어가 타인 접촉 전 재터치."""
FOUL_SETPIECE = 4
"""세트피스 재터치 위반 — 프리킥·코너·골킥·페널티 키커가 타인 접촉 전 재터치.

스로인과 제재는 같다(수비 간접 FK). 종류를 나누는 이유는 재개 출처가 다르고,
합쳐 놓으면 텔레메트리에서 스로인 위반과 구분할 수 없기 때문이다.
"""

INJECTABLE_FOUL_KINDS = frozenset({FOUL_TACKLE, FOUL_CHARGE})
"""``inject_charge``가 붙일 수 있는 반칙 코드.

주입은 인플레이 접촉 반칙을 재현하는 pin이라 그 두 종류만 의미가 있다. ``FOUL_NONE``은
반칙이 아니고(주입을 끄려면 ``actor < 0``을 쓴다), 재터치 위반 둘은 재개 절차가 만드는
반칙이라 이 경로로 붙이면 프리킥에 스로인 위반 라벨이 달린다. 호스트 검증과 런타임
게이트가 **같은 집합**을 본다 — 갈리면 한쪽이 통과시킨 값을 다른 쪽이 조용히 버린다.
"""

# Observed-foul reconstruction can pin the disciplinary outcome independently
# of the restart.  SAMPLE keeps the ordinary contextual draw; the other values
# are facts supplied by a replay/data adapter.
DISCIPLINE_SAMPLE = -1
DISCIPLINE_NONE = 0
DISCIPLINE_YELLOW = 1
DISCIPLINE_RED = 2
DISCIPLINE_OUTCOMES = frozenset(
    {DISCIPLINE_SAMPLE, DISCIPLINE_NONE, DISCIPLINE_YELLOW, DISCIPLINE_RED}
)


# 공이 판정선을 넘어 발생한 사건의 종류 — control frame 안의 substep 버퍼에 기록된다.
BALL_EVENT_NONE = 0
"""이 substep에는 라인 통과 사건이 없었다."""
BALL_EVENT_GOAL = 1
"""골 — 공 전체가 골문 안쪽으로 골라인을 통과."""
BALL_EVENT_CORNER = 2
"""골라인 아웃, 최종 터치가 수비팀 → 코너킥."""
BALL_EVENT_GOALKICK = 3
"""골라인 아웃, 최종 터치가 공격팀 → 골킥."""
BALL_EVENT_THROWIN = 4
"""터치라인 아웃 → 스로인."""
BALL_EVENT_COUNT = 5
"""``BALL_EVENT_*`` 도메인 크기."""


# 공이 골 프레임(포스트·크로스바)에 맞고 튄 사건 — ``BALL_EVENT_*``와 **별도 버퍼**다.
# 같은 substep에 프레임 충돌과 라인 통과가 함께 일어나기 때문이다: 크로스바를 맞고
# 그대로 골라인을 넘는 공은 한 물리 tick 안에서 두 사건을 다 만든다(반사 뒤 남은 이동은
# 최대 dt·|v| ≈ 0.44 m인데 골라인은 바로 거기 있다). 한 칸을 나눠 쓰면 하필 그
# 흥미로운 경우만 덮어써 사라진다.
WOODWORK_NONE = 0
"""이 substep에는 골 프레임 충돌이 없었다."""
WOODWORK_POST = 1
"""골포스트(수직 기둥) 충돌."""
WOODWORK_CROSSBAR = 2
"""크로스바(수평 기둥) 충돌."""
WOODWORK_COUNT = 3
"""``WOODWORK_*`` 도메인 크기."""


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
ACTION_SCHEMA_VERSION = 2
SELF_TOUCH_RELATION = 2.0
"""``last_touch_relation``에서 '내가 마지막으로 찼다'를 뜻하는 값.

팀 관계 {-1 상대, 0 없음, +1 우리}에 얹는 네 번째 값이다. 슬롯 번호가 아니라 관측자
기준 관계라 identity는 여전히 노출되지 않는다.
"""

OBS_SCHEMA_VERSION = 29
"""관측 레이아웃과 값 해석의 계약. 차원·열 순서는 :data:`OBS_PLAYER_FEATURES`,
:data:`OBS_BALL_FEATURES`, :data:`OBS_CONTEXT_FEATURES`가 단일 진실원천이고 이 상수는 그
레이아웃이 바뀌었음을 노출한다. 차원이 같아도 열의 **의미**가 바뀌면 함께 올린다 — 저장
데이터와 정책 체크포인트가 조용히 섞이는 것을 막는 것이 이 상수의 유일한 목적이다."""
STATE_SCHEMA_VERSION = 26
"""중앙 상태 벡터의 레이아웃 계약. 열 구성은 :data:`STATE_PLAYER_FEATURES`,
:data:`STATE_BALL_FEATURES`, :data:`STATE_GAME_FEATURES`가 단일 진실원천이다. 중앙 크리틱과
복원 파이프라인이 읽는 벡터이므로 관측과 별개로 버전을 관리한다."""

# 중앙 벡터와 별개인 시뮬레이터 State/dynamics 계약 버전. dataset manifest fingerprint가
# 코드 의미 변경을 조용히 섞지 않도록 명시적으로 올린다.
ENV_DYNAMICS_VERSION = 60
"""물리·규칙 전이의 계약. ``SoccerEnv.dynamics_fingerprint``에 들어가며, 같은
(상태, 행동, 키)가 다른 다음 상태를 내면 올린다. 전이 결과가 같아도 정책·보상이 읽는 State가
달라지면 함께 올린다 — 그때도 학습 데이터와 체크포인트는 서로 혼용할 수 없기 때문이다."""
STATE_CONTAINER_SCHEMA_VERSION = 22
"""``State`` 컨테이너(필드 집합과 순서)의 계약. 벡터 레이아웃을 다루는
:data:`STATE_SCHEMA_VERSION`과 달리 pytree 구조 자체가 바뀔 때 올린다."""
MOVEMENT_DYNAMICS_VERSION = 17
"""이동 커널(속도 명령 → 실현 속도)의 계약. 마찰 타원·실효 최고속·에너지
결합이 바뀌면 올린다. 이동 라벨 역산(``inverse.infer_move_action``)이 이 버전에 묶인다."""
ENERGY_DYNAMICS_VERSION = 5
"""stamina 전이(장기 누적 workload·단기 headroom)의 계약. 실효 최고속을 통해
이동과 경합에 함께 들어가므로 이동 동역학과 별도로 관리한다."""
AFFORDANCE_SCHEMA_VERSION = 3
"""``affordance_view``가 반환하는 파생 신호의 계약. 저장 관측에 넣지 않고 매
프레임 결정론적으로 재계산하는 값들이라, 키 집합이나 시간 범위(현 substep / 현 control frame)가
바뀌면 올린다."""

# 슬롯 참여 상태 — `on_pitch`/`sent_off` 두 비트를 하나의 categorical로 접는다. '교체로 빠진
# 슬롯'과 '퇴장으로 빠진 슬롯'은 남은 경기의 최소 인원 판정(IFAB_MIN_TEAM_PLAYERS)과 재투입
# 가능성이 다르므로 서로 다른 코드여야 한다. projection 한 비트로는 둘을 구분할 수 없다.
SLOT_INACTIVE = 0
"""온피치가 아님 — 교체로 빠졌거나 아직 투입되지 않은 슬롯."""
SLOT_ACTIVE = 1
"""물리·규칙에 참여 중인 슬롯."""
SLOT_SENT_OFF = 2
"""퇴장 — 되돌릴 수 없고 팀 인원을 영구히 줄인다."""
SLOT_STATUS_COUNT = 3

# taker_mask 비트 — pending/setpiece/throw 세 개의 N차원 마스크를 슬롯당 정수 하나로 접는다.
# 세 비트는 동시에 설 수 있으므로 categorical이 아니라 비트마스크다(예: 스로인을 던진 선수가
# 다음 프리킥의 지정 키커가 되는 경우 pending과 throw가 함께 선다).
TAKER_BIT_PENDING = 1
"""이번 재개의 지정 키커(``pending_taker``) — 킥 예정."""
TAKER_BIT_SETPIECE = 2
"""직전 세트피스 실행자(``setpiece_taker``) — 타인 접촉 전 재터치 금지."""
TAKER_BIT_THROW = 4
"""직전 스로인 실행자(``throw_taker``) — 재터치 금지 + 직접 득점 불가."""
TAKER_MASK_COUNT = 8

# Global provenance left when an original throw/set-piece taker has departed.
# Active takers remain represented by the per-slot ``taker_mask`` above; these
# bits deliberately contain no actor identity.
DEPARTED_TAKER_BIT_THROW = 1
DEPARTED_TAKER_BIT_SETPIECE = 2
DEPARTED_TAKER_MASK_COUNT = 4
"""``departed_taker_mask``가 취할 수 있는 값의 수(2비트) — embedding 폭."""

ROLE_GAIN_EXACT_MAX_SAMPLES = 1_048_576
"""``round(1/role_gain - 1)``이 표본 수를 정확히 되돌리는, 보수적으로 선언한 상한.

float32에 ``1/(1+n)``을 저장했다 되돌리면 상대오차가 ``n``에 비례해 커진다. 실측 상 반올림이
깨지는 지점은 약 1.26e7이며, 여기서는 그보다 한 자릿수 낮은 값을 계약으로 공개한다. 정규 경기
한 하프의 control step 수는 15 Hz 기준 4만 수준이라 실사용 범위와 12배 이상 여유가 있다."""

# 슬롯이 참여하지 않을 때 중앙 상태에서 살려 두는 열.
#
# ``status_code``는 전이에 직접 관여한다 — 참여 여부가 인원 하한 판정을 좌우하고, INACTIVE와
# SENT_OFF의 구분이 그 슬롯에 예정된 교체의 성립 여부를 가른다.
#
# ``team_id``는 env 계산에 쓰이지 않는다(인원 하한은 ``active_player & team_id``라 비참여 슬롯의
# 팀 값을 읽지 않는다). 그럼에도 남기는 이유는 **슬롯→팀 대응이 벡터 안에서 닫히게** 하기
# 위해서다. 어느 팀이 수적 열세인지는 값 추정에 직접 쓰이는 사실인데, 이 열이 없으면 슬롯
# 인덱스와 로스터 크기를 벡터 밖에서 알아야만 읽을 수 있다.
#
# ``is_gk``는 뺀다. GK 역할 보존은 ``project_substitution``의 ``values_valid``에서 쓰이지만 그
# 경로는 ``eligible``이 ``on_pitch & ~sent_off``를 요구하므로 **참여 중인 슬롯에서만** 도달한다.
# 비참여 슬롯의 GK 비트는 어떤 전이에도 다시 등장하지 않는 죽은 신호다.
SUBSTITUTION_ENTRY_DIRECTIONS = 8
"""교체 투입 좌표가 막혔을 때 탐색하는 방향 수 — 결정적 후보 집합의 각 분해능."""

SUBSTITUTION_ENTRY_RINGS = 3
"""교체 투입 대체 좌표 동심원 수. 반경은 ``min_d``의 1~3배 — 그보다 멀면 지정 좌표의 의미가
사라지고, 그만큼 밀집한 배치는 실경기에 존재하지 않는다."""

INACTIVE_SLOT_LIVE_COLUMNS = ("status_code", "team_id")

ROTATE_180 = -1
"""상대팀 포메이션 180° 회전(점대칭) 계수 — 초기화에서 init_pos의 x·y 양쪽에 곱함."""

YELLOW_CARD_SEND_OFF_COUNT = 2
IFAB_MIN_TEAM_PLAYERS = 7
"""정규 경기의 최소 팀 인원. 소규모 사용자 로스터에는 초기 인원 수를 상한으로 적용한다."""
DEFAULT_MAX_SUBSTITUTIONS = 5
"""팀당 교체 인원의 기본값. 대회 규정과 실험 설계에 맞게 ``SoccerEnv`` 생성 시
``max_substitutions``로 바꿀 수 있다. 벤치 명단 크기와는 독립적이다."""

# 이전 내부 이름을 유지한다. 5는 엔진의 하드 상한이 아니라 현행 일반 규칙의
# 기본값이며, 실제 상한은 환경별 ``max_substitutions``에 들어 있다.
IFAB_MAX_SUBSTITUTIONS_PER_TEAM = DEFAULT_MAX_SUBSTITUTIONS
IFAB_MAX_SUBSTITUTION_WINDOWS = 3
"""[IFAB Law 3] 하프타임을 제외한 교체 기회 횟수. 한 기회에 여러 명을 함께 바꿀 수 있다."""

# 결정 텔레메트리 코드 — "결정자는 제안하고 환경이 승인한다"의 **승인 쪽 결과**다.
#
# 제안이 조용히 사라지면 학습된 결정 정책은 무엇을 고쳐야 하는지 알 수 없다. 거절 자체는
# 규칙이라 없앨 수 없지만(공이 살아났으면 못 바꾼다), **왜** 거절됐는지는 알려 줄 수 있다.
# 문자열이 아니라 정수 코드인 이유는 이 값이 JIT 안에서 만들어져 info로 나가기 때문이다.
SUB_DECISION_APPLIED = 0
"""교체가 실제로 적용됐다."""
SUB_DECISION_NOT_REQUESTED = 1
"""그 자리에 제안이 없었다(``out_slot``이 ``NO_PLAYER``)."""
SUB_DECISION_BALL_LIVE = 2
"""[IFAB Law 3] 경계 시점에 공이 인플레이라 교체 창이 닫혀 있었다."""
SUB_DECISION_TERMINAL = 3
"""이미 끝난 경기다."""
SUB_DECISION_SLOT_RANGE = 4
"""슬롯 인덱스가 로스터 밖이다."""
SUB_DECISION_BENCH_RANGE = 5
"""벤치 인덱스가 벤치 밖이다."""
SUB_DECISION_BENCH_EMPTY = 6
"""그 벤치 자리에 사람이 없다(비었거나 이미 투입됐다)."""
SUB_DECISION_WRONG_TEAM = 7
"""그 슬롯은 제안한 팀의 것이 아니다."""
SUB_DECISION_SLOT_INACTIVE = 8
"""그 슬롯은 이미 경기장 밖이다(교체 아웃·퇴장)."""
SUB_DECISION_SLOT_ALREADY_CHANGED = 9
"""이 경계에서 그 슬롯이 이미 바뀌었다 — 스케줄이 먼저 썼거나 앞선 rank가 썼다."""
SUB_DECISION_GK_ROLE = 10
"""GK 자리는 GK로만 채운다 — 아니면 팀이 GK 없이 남는다."""
SUB_DECISION_NO_CARD = 11
"""팀의 교체 인원을 다 썼다."""
SUB_DECISION_WINDOW_BUDGET = 12
"""[IFAB Law 3] 교체 기회 3회를 다 썼고 지금은 새 기회다."""
SUB_DECISION_PLACEMENT = 13
"""사전 검사는 통과했지만 투영이 identity를 바꾸지 못했다(중복 신원·배치 실패)."""
SUB_DECISION_CODE_COUNT = 14

FORMATION_DECISION_APPLIED = 0
"""포메이션 명령이 적용됐다."""
FORMATION_DECISION_UNCHANGED = 1
"""지금 레이아웃을 그대로 요청했다 — 거절이 아니라 '바꿀 것 없음'이다."""
FORMATION_DECISION_OUT_OF_RANGE = 2
"""레이아웃 인덱스가 표 밖이다."""
FORMATION_DECISION_TERMINAL = 3
"""이미 끝난 경기다."""
FORMATION_DECISION_CODE_COUNT = 4

TAKER_DECISION_APPLIED = 0
"""결정자의 키커 제안이 승인됐다."""
TAKER_DECISION_NOT_CALLED = 1
"""이 프레임에는 키커 지정 요청 자체가 없었다."""
TAKER_DECISION_SLOT_RANGE = 2
"""슬롯 인덱스가 로스터 밖이라 최근접 fallback으로 떨어졌다."""
TAKER_DECISION_SLOT_INACTIVE = 3
"""그 슬롯은 경기장 밖이다."""
TAKER_DECISION_WRONG_TEAM = 4
"""재개권을 가진 팀의 선수가 아니다."""
TAKER_DECISION_NOT_GK = 5
"""GK 홀드 배급은 그 GK만 할 수 있다(규칙이 아니라 법이다)."""
TAKER_DECISION_CODE_COUNT = 6

# 경기장 밖 선수의 구역 — 셋은 **되돌릴 수 있는가**가 다르므로 위치로 구분한다.
OFFPITCH_BENCH = 0
"""아직 투입되지 않은 교체 후보. 들어올 수 있다."""
OFFPITCH_RETIRED = 1
"""교체로 나간 선수. 같은 경기에 다시 들어올 수 없다."""
OFFPITCH_SENT_OFF = 2
"""퇴장. 다시 들어올 수 없고 팀 인원도 줄어든다."""
OFFPITCH_ZONE_COUNT = 3

# SoccerEnv() 기본 생성자가 실제로 동작하도록 쓰는 표준 11인 포메이션.
# 좌표는 팀의 공격 프레임(+x 공격)이며 상대 팀은 초기화에서 점대칭한다.
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
DEFAULT_CONTROL_FPS = DEFAULT_TIMEBASE.control_fps
POSSESSION_CONTEXT_SECONDS = 5.0
"""Actor/critic vectors expose possession age over this causal window.

Five seconds covers the measured three-second counterpress window plus a
two-second secure-possession check.  The raw State counter is not clipped;
only its bounded vector encoding is, so no transition information is hidden.
"""

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


# 관측/central-state 스키마(Causal-Compact). 생성 코드와 학습 코드가 같은 이름·폭을 참조하도록
# 상수화한다. 세 층을 분리하는 것이 이 레이아웃의 요지다.
#   · 프레임 인과 상태  — 여기 실린 것. 다음 전이를 결정하는 값만.
#   · 선수 프로필       — identity가 바뀔 때만 변하는 값(PLAYER_PROFILE_FEATURES). 랜덤 액세스
#                        벡터에는 포함하되 시퀀스 저장에서는 sidecar로 뺄 수 있다.
#   · 파생 affordance   — AFFORDANCE_FEATURES. 상태+config에서 매 프레임 결정론적으로 재계산되며
#                        저장하지 않는다(원천 상태처럼 중복 저장하면 두 계약이 갈린다).
OBS_ANCHOR_FEATURES = (
    ("abs_pos", 2),
    ("abs_vel", 2),
)
"""관측자 앵커 — 이 4차원이 있어야 토큰의 상대량에서 절대 좌표를 복원할 수 있다.
필드 경계·골대는 절대 좌표에만 있으므로 앵커 자체는 제거 대상이 아니다."""
OBS_PLAYER_FEATURES = (
    ("rel_pos", 2),
    ("rel_vel", 2),
    ("stamina_short", 1),
    ("stamina_long", 1),
    ("endurance_factor", 1),
    ("cooldown", 1),
    ("contact_lock", 1),
    ("aerial_recovery", 1),
    ("ctrl_lock", 1),
    ("yellow", 1),
    ("offside_latch", 1),
    ("role_pos", 2),
    ("role_gain", 1),
    # 규범 앵커 — '어디에 서야 하나'. ``role_pos``(서술: 현재 전술 epoch에서 어디에
    # 있었나)와 다른 축이다.
    # 이것 없이는 규칙 정책이 포메이션을 생성 시점 상수로만 알 수 있어, 경기 중 지휘가
    # 관측에 도달하지 못한다.
    ("formation_home", 2),
    # 실측 위치장 표가 레이아웃마다 다르므로 연속 앵커와 함께 목표 index를 싣는다.
    ("layout_index", 1),
    ("status_code", 1),
    ("taker_mask", 1),
    ("team_relation", 1),
    ("vmax", 1),
    ("ball_ctrl", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("is_gk", 1),
)
"""전 슬롯 동형 토큰(자기 자신 포함, 슬롯 순서). 자기 토큰은 rel_pos=rel_vel=0이라
관측자와 타 선수를 하나의 entity encoder에 넣을 수 있다. 절대량은
``abs_j = anchor + rel_j``로 정확히 복원된다."""
OBS_BENCH_FEATURES = (
    ("available", 1),
    ("is_gk", 1),
    ("role_pos", 2),
    # 감독이 실제로 쓰는 값이다 — 역할 적합도만으로는 "누가 더 나은가"를 못 가른다.
    # 관측에 없으면 학습된 감독이 배포 시점에 없는 입력에 의존하고, 같은 obs에서 교체 후
    # 능력치와 다음 전이가 달라진다.
    ("vmax", 1),
    ("ball_ctrl", 1),
    ("endurance_factor", 1),
)
STATE_BENCH_FEATURES = OBS_BENCH_FEATURES
"""중앙 상태의 벤치 토큰 — 팀 절대 인덱스(0/1) 순서, ``role_pos``는 각 팀 공격 접힘 프레임.

obs만 고치면 중앙 크리틱이 여전히 벤치를 못 본다 — 프로필을 바꿔도 ``get_state()``가
완전히 같아서, 같은 학습 입력에서 교체 후 능력치와 다음 전이가 달라진다."""

_BENCH_TOKEN_DOC = """벤치 **자리별** 토큰. 우리 팀 자리 전부, 그다음 상대 팀 자리 전부 순서다.

인원수만 실으면 계약이 깨진다 — 제안이 ``bench_index``로 특정 자리를 지목하는데,
``[-1, 9102]``와 ``[9101, -1]``은 가용 인원이 똑같이 1이라 obs가 같지만 ``index=0``의
결과가 갈린다. 저장 주소가 계약에 새어 나온 셈이라, 그 주소가 가리키는 **내용**을 실어야
Markov가 성립한다.

``role_pos``는 like-for-like 대체에 필요하다 — 나간 자리의 앵커에 가장 가까운 벤치 선수를
고르려면 벤치에 **누가** 있는지 알아야 하고, 인원수로는 애초에 불가능하다.
신원(player_id)은 싣지 않는다. 관측이 사람을 특정하지 않는다는 원칙(identity 비노출)은
경기장 안과 밖에 똑같이 적용한다."""

OBS_BALL_FEATURES = (
    ("rel_pos", 2),
    ("abs_z", 1),
    ("rel_vel", 2),
    ("abs_vel_z", 1),
    ("abs_spin", 3),
    ("ball_alive", 1),
    ("possession_relation", 1),
)
"""공의 x·y는 관측자 기준 상대량, z는 지면 기준 절대량(관측자에게 높이 오프셋이 없다)."""
OBS_CONTEXT_FEATURES = (
    ("restart_owner_relation", 1),
    ("restart_kind_code", 1),
    ("restart_steps", 1),
    ("restart_indirect", 1),
    ("departed_taker_mask", 1),
    ("pass_owner_relation", 1),
    ("offside_active", 1),
    ("time_left", 1),
    ("possession_steps", 1),
    ("previous_possession_relation", 1),
    ("last_touch_relation", 1),
    ("last_touch_code", 1),
    ("gk_handling_restricted_relation", 1),
    ("score_diff", 1),
    ("second_half_kickoff_ours", 1),
    # ── 교체·포메이션 자원 ────────────────────────────────────────────
    # 이 값들은 **다음 전이를 바꾼다**. 싣지 않으면 완전히 같은 obs에서 다음 상태가 갈리고
    # (실측: subs_remaining 1과 0에서 같은 제안이 투입/유지로 나뉘고, layout_since_t만
    # 달라도 다음 레이아웃이 [30,19]와 [0,0]으로 갈렸다), 같은 입력에 서로 다른 BC 라벨이
    # 붙는다. ``role_gain``을 실은 것과 같은 이유다.
    #
    # 결정자 view가 "전부 관측에서 얻을 수 있다"는 계약도 이것들 없이는 성립하지 않는다 —
    # 학습된 교체 정책이 배포 시점에 없는 입력에 의존하게 된다.
    ("subs_remaining_ours", 1),
    ("subs_remaining_theirs", 1),
    ("sub_windows_used_ours", 1),
    ("sub_windows_used_theirs", 1),
    ("sub_window_open_ours", 1),
    ("sub_window_open_theirs", 1),
    ("bench_available_ours", 1),
    ("bench_available_theirs", 1),
    ("bench_gk_available_ours", 1),
    ("bench_gk_available_theirs", 1),
    # (t - layout_since_t) / game_duration. 승인 명령 직후 0이고 다음 명령까지 단조 증가한다.
    # 교체는 이 값을 reset하지 않는다. identity는 player_id/slot_generation이 따로 소유한다.
    ("layout_hold_elapsed_ours", 1),
    ("layout_hold_elapsed_theirs", 1),
)
"""전역 맥락. ``*_code``는 정수 categorical이라 연속값으로 해석하면 안 되고 embedding 대상이다.
``*_relation``은 관측자 팀 기준 ±1(해당 없음 0)이라 팀 인덱스 누출 없이 관점 불변이다.
``*_ours``/``*_theirs``도 같은 이유로 팀 인덱스가 아니라 관측자 기준으로 접힌다."""

STATE_PLAYER_FEATURES = (
    ("pos", 2),
    ("vel", 2),
    ("stamina_short", 1),
    ("stamina_long", 1),
    ("endurance_factor", 1),
    ("cooldown", 1),
    ("contact_lock", 1),
    ("aerial_recovery", 1),
    ("ctrl_lock", 1),
    ("yellow", 1),
    ("offside_latch", 1),
    ("role_pos", 2),
    ("role_gain", 1),
    # 규범 앵커 — '어디에 서야 하나'. ``role_pos``(서술: 현재 전술 epoch에서 어디에
    # 있었나)와 다른 축이다.
    # 이것 없이는 규칙 정책이 포메이션을 생성 시점 상수로만 알 수 있어, 경기 중 지휘가
    # 관측에 도달하지 못한다.
    ("formation_home", 2),
    # 실측 위치장 표가 레이아웃마다 다르므로 연속 앵커와 함께 목표 index를 싣는다.
    ("layout_index", 1),
    ("status_code", 1),
    ("taker_mask", 1),
    ("vmax", 1),
    ("ball_ctrl", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("team_id", 1),
    ("is_gk", 1),
)
"""중앙 크리틱용 절대 프레임 슬롯 토큰. ``face``는 ``vel``·``attack_dir``의 결정함수라 빠졌다
(movement.facing_from_velocity)."""
STATE_BALL_FEATURES = (("pos", 3), ("vel", 3), ("spin", 3))
STATE_GAME_FEATURES = (
    ("poss_team_code", 1),
    ("possession_steps", 1),
    ("previous_poss_team_code", 1),
    ("last_touch_team_code", 1),
    ("gk_handling_restricted_team_code", 1),
    ("attack_dir_team0", 1),
    ("ball_alive", 1),
    ("restart_steps", 1),
    ("offside_active", 1),
    ("time_left", 1),
    ("restart_indirect", 1),
    ("departed_taker_mask", 1),
    ("restart_team_code", 1),
    ("pass_team_code", 1),
    ("score", TEAM_COUNT),
    ("restart_kind_code", 1),
    ("last_touch_code", 1),
    ("kickoff_team_code", 1),
    # 중앙 크리틱도 같은 값을 봐야 한다 — obs만 고치면 state가 여전히 Markov가 아니다.
    # 관측자가 없으므로 팀 절대 인덱스(0/1) 순서다.
    ("subs_remaining", TEAM_COUNT),
    ("sub_windows_used", TEAM_COUNT),
    ("sub_window_open", TEAM_COUNT),
    ("bench_available", TEAM_COUNT),
    ("bench_gk_available", TEAM_COUNT),
    ("layout_hold_elapsed", TEAM_COUNT),
)
"""팀 코드는 ``NO_TEAM``(-1)/``TEAM_0``(0)/``TEAM_1``(1) 원값이다. one-hot을 쓰지 않는 이유는
역변환이 자명해 정보가 같고, 폭만 3배이기 때문이다."""

PLAYER_PROFILE_FEATURES = (
    ("vmax", 1),
    ("ball_ctrl", 1),
    ("endurance_factor", 1),
    ("reach_z", 1),
    ("head_z", 1),
    ("is_gk", 1),
    ("team_id", 1),
)
"""identity(``player_id``, ``slot_generation``)가 바뀔 때만 갱신되는 값. 단독 프레임 벡터에는
포함하지만, 연속 경기 시퀀스를 저장할 때는 identity 키의 sidecar로 빼면 프레임 payload가
슬롯당 7차원 줄어든다. ``role_pos``/``role_gain``은 매 라이브 프레임 변하므로 프로필이 아니다."""

# 지구력 계수는 원값을 관측에 공개하지만, 규칙 결정자가 그 값을 곧바로 배율로 쓰면
# 허용 범위의 극단값이 전술 점수를 지배한다. 1.0에서 정확히 0이고 양쪽 최대 25%까지만
# 반영하는 보수적 centered delta를 모든 내장 결정자가 공유한다. 실제 물리 소모율은 이
# 상수가 아니라 energy 모듈이 소유한다.
ENDURANCE_FACTOR_MIN = 0.01
ENDURANCE_FACTOR_MAX = 100.0
ENDURANCE_FACTOR_REFERENCE = 1.0
ENDURANCE_DECISION_DELTA_CAP = 1.0
ENDURANCE_DECISION_GAIN = 0.25

AFFORDANCE_FEATURES = (
    ("in_reach", 1),
    ("f2b_avail", 1),
    ("ball_rank", 1),
    ("off_line", 1),
    ("enc_margin", 1),
    ("any_encroacher", 1),
    ("kicker_locked", 1),
    ("kicker_ready", 1),
    ("is_taker", 1),
    ("pass_signal", 1),
    ("is_sp_ours", 1),
    ("move_forced", 1),
    ("kick_gated", 1),
    ("kick_forced", 1),
)
"""상태와 config에서 결정론적으로 재계산되는 파생량 — **저장하지 않는다**. 정책에는 compact
벡터와 함께 전달하되 데이터셋에는 원천 상태처럼 중복 기록하지 않는다. 중복 저장은 두 계약이
서로 갈릴 여지를 만들고, 재계산은 비용이 거의 없다."""

# star-import를 쓰는 기존 모듈과 호환하되 math 등 구현 모듈은 새어 나가지 않게 제한한다.
__all__ = tuple(name for name in globals() if name.isupper())
