"""세트피스 키커 선택 — 누가 찰 것인가.

종전에는 **공에 가장 가까운 선수**가 전부 찼다. 그래서 센터백이 스로인을 던지고, 중앙
공격수가 코너를 올리며, 마침 뒤로 처져 있던 수비수가 페널티를 찼다. 재개 종류가 선택
함수에 전달조차 되지 않았으므로 종류별로 다르게 고를 방법이 없었다.

이 모듈은 그 선택을 종류별 규칙으로 바꾼다. 역할은 **고정 슬롯 번호가 아니라 현재
``formation_home``의 깊이와 폭**으로 판정하므로, 포메이션이 바뀌거나 교체가 일어나도
규칙이 그대로 유지된다.

규약
----
    decide(view, key) -> slot_index      # NO_PLAYER면 후보 없음

선택은 **재개가 열리는 순간 한 번만** 한다. 재개 도중에는 바꾸지 않고, 키커가 교체·퇴장된
경우에만 같은 규칙으로 다시 고른다 — 매 서브스텝 다시 고르면 비용도 크고 키커가 흔들린다.

비용
----
이 규칙은 공짜가 아니다. 240 스텝 CPU 실측(22명, 기본 config)::

    세 축 전부 꺼짐            1.55 ms/step
    키커만 auto               2.59 ms/step   (+67%)
    벤치+감독 auto             1.73 ms/step   (+12%)
    전부 켬                   2.98 ms/step   (+92%)

재개를 만드는 지점이 열 곳이고 그 각각이 뷰(규범 앵커 + 역할 분류)를 만들기 때문이다.
그래서 환경 기본값은 여전히 ``restart_taker_mode="nearest"``다 — 대량 BC 수집처럼
스텝 비용이 지배적인 곳에서는 끄고, 경기다움이 필요한 곳(데모·렌더·평가)에서 켠다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constants import (
    ENDURANCE_DECISION_DELTA_CAP,
    ENDURANCE_DECISION_GAIN,
    ENDURANCE_FACTOR_REFERENCE,
    NO_PLAYER,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_PENALTY,
    RK_THROWIN,
)

# ── 역할 코드 — formation_home의 깊이·폭에서 유도한다 ─────────────────────
TAKER_RULES_VERSION = 4
"""키커 선택 규칙의 식별자. 역할 판정·점수식·계수가 바뀌면 올린다 — 같은 모드 이름으로
다른 궤적이 나오면 산출물이 조용히 섞인다."""

ROLE_GK = 0
ROLE_CENTRE_BACK = 1
ROLE_FULL_BACK = 2
ROLE_CENTRE_MID = 3
ROLE_WIDE_MID = 4
ROLE_CENTRE_FORWARD = 5
ROLE_WIDE_FORWARD = 6
ROLE_COUNT = 7

ROLE_NAMES = ("GK", "CB", "FB", "CM", "WM", "CF", "WF")
"""역할 코드 → 짧은 이름. 렌더 HUD와 진단 출력이 같은 이름을 쓰게 한다 —
역할이 화면에 안 보이면 키커 선택이 맞는지 눈으로 확인할 방법이 없다."""

_WIDE_PER_LINE = 2
"""한 라인에서 측면으로 세는 인원 — **가장 바깥 두 명**이다.

절대 문턱으로 자르면 포메이션마다 결과가 뒤집힌다(킥오프 대형에서 수비 4명이 전부 측면
으로 잡혔다). 그렇다고 "최대 폭의 60% 이상"으로 자르는 것도 안전하지 않았다: 저자 백4의
앵커가 ±25/±15라 문턱이 정확히 15.0이고 안쪽 두 명도 정확히 15.0이어서, float32에서
``0.6 * 25.0``이 15.000000953674316으로 올림된 **우연** 하나로만 2+2가 됐다. 앵커가
±24/±15였다면 백4 전원이 풀백이 된다.

순위로 자르면 그 우연이 사라지고 축구적으로도 옳다. 백4→풀백2+센터백2, 백5→윙백2+
센터백3, 미드3→측면2+중앙1, 스리톱→윙2+중앙1이 전부 이 한 규칙에서 나온다. 좌우 대칭
앵커는 |y|가 같으므로 둘 다 순위 0이 되어 함께 측면이 된다."""
_WIDE_MIN_LINE = 3
"""이보다 인원이 적은 라인에는 측면을 두지 않는다 — 두 명뿐인 라인에 윙은 없다."""
_WIDE_TIE_TOL = 1e-3
"""폭 순위를 셀 때 같은 폭으로 볼 차이(m). 좌우 대칭 앵커가 부동소수 오차로 서로 다른
순위를 받지 않게 한다 — 그러면 한쪽만 측면이 되어 대형이 좌우 비대칭이 된다."""
_DEPTH_TOL = 0.5
"""같은 라인으로 볼 앵커 깊이 차(m). 한 라인의 구성원은 같은 깊이를 공유한다."""


def classify_roles(formation_home, is_gk, team_id=None, active=None):
    """규범 앵커에서 역할 코드 ``(N,)``을 유도한다.

    라인은 **앵커 깊이가 같은 무리**다. 고정 비율(수비 40 % / 공격 72 %)로 자르면 어떤
    모양이든 같은 구성이 나온다 — 실측으로 5-4-1·3-5-2·4-3-3·4-2-4가 전부 수비4·미드4·
    공격2로 분류됐다. 그러면 "포메이션이 바뀌면 역할도 따라간다"는 계약이 성립하지 않는다.
    앵커 자체를 보면 레이아웃의 실제 라인 분할이 그대로 나온다.

    슬롯 번호를 쓰지 않는 이유는 포메이션이 바뀌면 같은 슬롯이 다른 역할이 되기 때문이다.

    경기장 치수는 받지 않는다. 라인은 앵커 **사이의** 깊이 차로, 측면은 라인 **안에서의**
    상대 폭으로 정하므로 절대 치수가 들어갈 자리가 없다 — 종전 시그니처는 half_length·
    half_width를 받아 곧바로 ``del``했고, 호출부는 쓰이지도 않는 값을 넘기고 있었다.
    """

    gk = jnp.asarray(is_gk, bool)
    n = formation_home.shape[0]
    teams = (jnp.zeros(n, jnp.int32) if team_id is None
             else jnp.asarray(team_id, jnp.int32))
    live = (jnp.ones(n, bool) if active is None else jnp.asarray(active, bool))
    live = live & (~gk)
    depth = formation_home[:, 0]
    lateral = jnp.abs(formation_home[:, 1])

    band = jnp.zeros(n, jnp.int32)          # 0 수비 · 1 미드 · 2 공격
    wide = jnp.zeros(n, bool)
    for side in (TEAM_0_LOCAL, TEAM_1_LOCAL):
        mine = live & (teams == side)
        # 이 선수의 라인 번호 = 자기보다 **뚜렷하게 깊은 서로 다른 깊이**의 개수.
        #
        # "서로 다른"을 세려면 각 깊이의 **대표 슬롯 하나만** 세면 된다. 대표는 그 깊이를
        # 가진 우리 팀 선수 중 가장 앞선 슬롯이다. 종전에는 그 대표 판정에 팀 마스크를
        # 걸지 않아, 앞 슬롯의 상대 팀 선수가 우리 라인을 지웠다 — 팀 1이 어느 포메이션에서든
        # 수비 10·미드 0·공격 0으로 뭉갰다.
        same_depth = (
            mine[None, :] & mine[:, None]
            & (jnp.abs(depth[None, :] - depth[:, None]) <= _DEPTH_TOL)
        )
        earlier = jnp.arange(n)[None, :] < jnp.arange(n)[:, None]
        representative = mine & (~jnp.any(same_depth & earlier, axis=1))
        deeper_reps = (
            representative[None, :] & mine[:, None]
            & (depth[None, :] < depth[:, None] - _DEPTH_TOL)
        )
        mine_band = jnp.clip(jnp.sum(deeper_reps, axis=1), 0, 2).astype(jnp.int32)
        band = jnp.where(mine, mine_band, band)
        for line in range(3):
            group = mine & (mine_band == line)
            # 이 선수보다 **뚜렷하게 더 바깥**인 같은 라인 인원 수 = 폭 순위.
            # 대칭 앵커(±y)는 lateral이 같아 서로를 세지 않으므로 둘 다 순위 0이다.
            wider = jnp.sum(
                group[None, :] & group[:, None]
                & (lateral[None, :] > lateral[:, None] + _WIDE_TIE_TOL),
                axis=1)
            big_enough = jnp.sum(group) >= _WIDE_MIN_LINE
            wide = wide | (group & big_enough & (wider < _WIDE_PER_LINE))

    role = jnp.where(
        band == 0, jnp.where(wide, ROLE_FULL_BACK, ROLE_CENTRE_BACK),
        jnp.where(band == 2,
                  jnp.where(wide, ROLE_WIDE_FORWARD, ROLE_CENTRE_FORWARD),
                  jnp.where(wide, ROLE_WIDE_MID, ROLE_CENTRE_MID)))
    return jnp.where(gk, ROLE_GK, role).astype(jnp.int32)


TEAM_0_LOCAL = 0
TEAM_1_LOCAL = 1


@dataclass(frozen=True)
class SetPiecePlan:
    """팀의 세트피스 지정 키커 — ``player_id`` 우선순위 목록.

    감독이 명시적으로 정한 순번이다. 목록에 **활성 선수**가 있으면 규칙 점수보다 우선한다 —
    실측 재현이나 사용자 전술이 규칙에 밀리면 안 되기 때문이다. 목록이 비어 있거나 아무도
    활성이 아니면 아래 규칙 점수로 넘어간다.
    """

    penalty: tuple[int, ...] = ()
    direct_free_kick: tuple[int, ...] = ()
    indirect_free_kick: tuple[int, ...] = ()
    corner_left: tuple[int, ...] = ()
    corner_right: tuple[int, ...] = ()
    throw_left: tuple[int, ...] = ()
    throw_right: tuple[int, ...] = ()
    kickoff: tuple[int, ...] = ()
    goal_kick: tuple[int, ...] = ()


PLAN_FIELDS = (
    "penalty", "direct_free_kick", "indirect_free_kick",
    "corner_left", "corner_right", "throw_left", "throw_right",
    "kickoff", "goal_kick",
)
"""``SetPiecePlan``의 슬롯 순서 — 런타임 표의 행 순서와 같아야 한다."""


class SetPieceTakerView(NamedTuple):
    """키커 결정자가 보는 입력.

    ``plan``과 ``player_id``를 빼면 전부 compact 관측에서 복원할 수 있는 값이다.
    그 둘은 예외이고, 그것이 계약의 **정확한 경계**다.

    감독이 :class:`SetPiecePlan`으로 키커를 지정하면 선택이 슬롯이 아니라 **사람 id**로
    걸린다. compact obs/state는 player_id를 싣지 않으므로, 능력치와 위치가 같은 두 슬롯의
    id만 맞바꾼 두 상태는 관측 벡터가 완전히 같은데도 지정 키커가 달라진다 — 즉 계획을
    쓰는 순간 관측만으로 다음 전이가 결정되지 않는다.

    그래서 계획은 **기본값이 아니다**(``setpiece_plans=None``이면 ``plan_slot``이 음수라
    이 경로가 통째로 꺼지고 규칙 점수만 남는다). 계획을 쓰는 실험은 그 사실을 알고
    써야 하고, dynamics 지문에 계획 내용이 통째로 실려 산출물이 섞이지 않게 한다.
    """

    restart_kind: jnp.ndarray      # ()
    restart_indirect: jnp.ndarray  # ()
    restart_team: jnp.ndarray      # ()
    restart_spot: jnp.ndarray      # (2,)
    attack_dir: jnp.ndarray        # () 재개 팀의 공격 방향 부호(+1/-1)
    half_length: jnp.ndarray       # () 경기장 반길이(m)
    goalkeeper_only: jnp.ndarray   # () 골킥에서 GK를 우선할 것인가
    player_id: jnp.ndarray         # (N,)
    active_player: jnp.ndarray     # (N,)
    team_id: jnp.ndarray           # (N,)
    is_gk: jnp.ndarray             # (N,)
    player_pos: jnp.ndarray        # (N,2)
    role: jnp.ndarray              # (N,) classify_roles 결과
    stamina_long: jnp.ndarray      # (N,)
    endurance_factor: jnp.ndarray  # (N,) 1.0 중심의 선수별 지구력 계수
    player_ctrl: jnp.ndarray       # (N,)
    reach_z: jnp.ndarray           # (N,)
    plan: jnp.ndarray              # (2, P, D) 팀×세트피스×우선순위의 player_id
    plan_slot: jnp.ndarray         # () 이번 재개가 쓸 plan 행. 음수면 지정 없음


# ── 종류별 역할 선호 — K리그1 2026 실측 ────────────────────────────────
#
# 종전 표는 저자 사전값(1순위 60점 / 2순위 30점 / 그 외 0)이었고, 실측과 어긋난 곳이
# 여럿이었다. 120경기 세트피스 7,500여 건(``bc/data/k_league``)의 키커를 라인업
# ``position_name``으로 역할에 접어 재추정한 결과가 아래다. 어긋났던 대표적인 두 곳:
#
#   * 코너의 1순위가 표에는 없던 **중앙 미드필더**였다(점유율 43%). 표는 측면 미드와
#     풀백을 1순위로 두고 있었다.
#   * 자기 진영 프리킥은 **골키퍼가 61%**로 압도적인데, 표는 GK에 일괄 -100을 걸어
#     구조적으로 배제하고 있었다.
#
# 점유율을 그대로 쓰면 인원이 많은 역할이 과대평가된다(중앙 미드는 팀당 2.4명, 측면
# 미드는 0.8명). 그래서 **선수 1인당 성향** = 점유율 / 선발 XI 평균 인원 으로 정규화한
# 뒤 로그를 취한다. 상수 항은 argmax가 같은 행 안에서만 비교하므로 무의미하다 —
# 의미가 있는 것은 같은 행 안의 **차이**뿐이다.
_AFFINITY_SCALE = 12.0
"""로그 성향 → 점수 환산 계수. 거리 항과 같은 단위(점)로 맞춘다.

성향비 10배가 약 27점이고, 스로인의 거리 가중치가 1.0점/m이므로 "성향이 10배인 역할을
제치려면 27 m 더 가까워야 한다"로 읽힌다."""
_AFFINITY_FLOOR = 1e-3
"""관측되지 않은 (종류, 역할) 조합의 성향 하한. 0을 그대로 로그에 넣을 수 없고, 하한을
두면 후보가 전멸했을 때도 순위가 정의된다(-82.9점 ≈ 종전의 GK 일괄 페널티)."""

# 행 = 재개 종류. 프리킥은 **재개 지점의 진영**에 따라 키커가 완전히 달라져서 3분할한다
# (자기 진영은 GK, 중원은 CB·CM, 상대 진영은 미드·풀백). 직접/간접은 나누지 않는다 —
# 실측 자료가 그 둘을 구분하지 않고, 구분해야 할 만큼 키커가 다르지도 않다.
# 값은 **선수 1인당 성향**(합 1로 정규화)이다.
_PROPENSITY = {
    #                GK      CB      FB      CM      WM      CF      WF
    "throwin":      (0.000,  0.078,  0.765,  0.022,  0.058,  0.018,  0.059),
    "corner":       (0.000,  0.016,  0.111,  0.266,  0.339,  0.065,  0.202),
    "freekick_def": (0.771,  0.157,  0.040,  0.025,  0.004,  0.002,  0.001),
    "freekick_mid": (0.099,  0.268,  0.200,  0.278,  0.077,  0.018,  0.060),
    "freekick_att": (0.000,  0.014,  0.185,  0.230,  0.269,  0.128,  0.174),
    "penalty":      (0.000,  0.000,  0.023,  0.076,  0.229,  0.505,  0.167),
    "goalkick":     (0.881,  0.118,  0.001,  0.000,  0.000,  0.000,  0.000),
    # 킥오프는 실측 자료에 세트피스로 태깅되지 않는다. 중앙 공격수가 굴리고 미드가 받는
    # 통상 관행을 그대로 둔다 — 유일하게 측정에 근거하지 않은 행이다.
    "kickoff":      (0.000,  0.010,  0.040,  0.160,  0.130,  0.500,  0.160),
}

_ROLE_ORDER = (ROLE_GK, ROLE_CENTRE_BACK, ROLE_FULL_BACK, ROLE_CENTRE_MID,
               ROLE_WIDE_MID, ROLE_CENTRE_FORWARD, ROLE_WIDE_FORWARD)
"""``_PROPENSITY`` 열 순서 → 역할 코드. 코드 값과 열 순서가 다르므로 명시한다."""

_DISTANCE_WEIGHT = {
    # 스로인은 그 사이드의 풀백이 그대로 던진다 — 역할과 근접이 함께 걸린다.
    "throwin": 1.0,
    # 코너·프리킥은 키커가 걸어온다. 거리는 좌/우 대칭을 깨는 정도로만 쓴다.
    "corner": 0.35,
    "freekick_def": 0.25, "freekick_mid": 0.5, "freekick_att": 0.4,
    # 페널티는 위치가 아니라 기술로 정한다.
    "penalty": 0.05,
    "goalkick": 0.3, "kickoff": 0.2,
}

_FREEKICK_ROWS = ("freekick_def", "freekick_mid", "freekick_att")
"""자기 골라인 기준 진영 3분할. ``_zone_row``가 이 순서로 고른다."""

_DEFAULT_TEMPERATURE = _AFFINITY_SCALE
"""키커 추첨의 온도. :data:`_AFFINITY_SCALE`과 같게 두는 것이 기본이다.

점수를 그대로 argmax하면 규칙이 **한 역할로 붕괴한다**. 실측 대조에서 코너 키커가
측면 미드 95.7%로 나왔지만 K리그1 실측은 중앙 미드 43.1% · 측면 미드 18.2% ·
풀백 14.9% · 측면 공격수 14.9%로 넓게 퍼진다. 실제 축구에서 코너를 늘 같은 사람이
차는 팀은 없고, 있다 해도 그것은 감독의 **지정**(:class:`SetPiecePlan`)이지 규칙의
성질이 아니다.

그래서 점수에 Gumbel 잡음을 더해 뽑는다(Gumbel-max = softmax 표집). 적합도가
``_AFFINITY_SCALE * log(성향)``이므로 온도를 같은 값으로 두면 다른 항이 없을 때
선택 확률이 **실측 성향과 정확히 같아진다**.

난수는 ``(episode_seed, tick, 재개 종류)``에서 파생된 키다 — 같은 상황은 같은 사람을
고르고, 다른 에피소드는 다르게 고른다. 재현성은 그대로다."""


def _role_affinity():
    """``(종류, 역할)`` 적합도 표. 호스트에서 한 번 만든다.

    성향의 로그에 :data:`_AFFINITY_SCALE`을 곱한 값이다. 관측되지 않은 조합은
    :data:`_AFFINITY_FLOOR`로 바닥을 깐다.
    """

    kinds = list(_PROPENSITY)
    rows = []
    for kind in kinds:
        row = [0.0] * ROLE_COUNT
        for column, role in enumerate(_ROLE_ORDER):
            share = max(_PROPENSITY[kind][column], _AFFINITY_FLOOR)
            row[role] = _AFFINITY_SCALE * math.log(share)
        rows.append(row)
    return kinds, jnp.asarray(rows, jnp.float32)


_KIND_NAMES, _AFFINITY = _role_affinity()
_DISTANCE = jnp.asarray([_DISTANCE_WEIGHT[k] for k in _KIND_NAMES], jnp.float32)


def _decision_stamina(stamina, endurance_factor):
    """Conservative readiness estimate; factor 1.0 is exactly the old value."""

    centered = jnp.asarray(endurance_factor, jnp.float32) - jnp.float32(
        ENDURANCE_FACTOR_REFERENCE
    )
    delta = (
        jnp.clip(
            centered,
            -jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
            jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
        )
        * jnp.float32(ENDURANCE_DECISION_GAIN)
    )
    return jnp.clip(jnp.asarray(stamina, jnp.float32) + delta, 0.0, 1.0)


def _zone_row(view):
    """프리킥 재개 지점의 진영 → ``_FREEKICK_ROWS`` 행 번호.

    ``progress``는 재개 팀 기준 0(자기 골라인)~1(상대 골라인)이다. 팀 축을 접지 않으면
    한쪽 팀의 자기 진영 프리킥이 상대 진영 행을 쓰게 되어 골키퍼 대신 윙어가 찬다.
    """

    index = {name: i for i, name in enumerate(_KIND_NAMES)}
    span = jnp.maximum(2.0 * view.half_length, 1e-6)
    progress = jnp.clip(
        (view.restart_spot[0] * view.attack_dir + view.half_length) / span,
        0.0, 1.0)
    row = jnp.int32(index[_FREEKICK_ROWS[0]])
    row = jnp.where(progress >= 1.0 / 3.0, index[_FREEKICK_ROWS[1]], row)
    row = jnp.where(progress >= 2.0 / 3.0, index[_FREEKICK_ROWS[2]], row)
    return row.astype(jnp.int32)


def _kind_row(view):
    """재개 종류를 적합도 표의 행 번호로 바꾼다.

    프리킥(직접·간접·오프사이드)만 진영에 따라 행이 갈린다 — 나머지 종류는 일어나는
    위치가 이미 정해져 있어서(코너는 코너에서, 골킥은 골에어리어에서) 진영이 정보를
    더 주지 않는다.
    """

    index = {name: i for i, name in enumerate(_KIND_NAMES)}
    kind = view.restart_kind
    row = _zone_row(view)
    row = jnp.where(kind == RK_THROWIN, index["throwin"], row)
    row = jnp.where(kind == RK_CORNER, index["corner"], row)
    row = jnp.where(kind == RK_GOALKICK, index["goalkick"], row)
    row = jnp.where(kind == RK_KICKOFF, index["kickoff"], row)
    row = jnp.where(kind == RK_PENALTY, index["penalty"], row)
    return row.astype(jnp.int32)


def nearest_taker(view, key):
    """종전 방식 — 공에 가장 가까운 선수. 호환·A/B 비교용으로 남긴다."""

    del key
    base = (view.team_id == view.restart_team) & view.active_player
    gk_mask = base & view.is_gk
    use_gk = view.goalkeeper_only & jnp.any(gk_mask)
    mask = jnp.where(use_gk, gk_mask, base)
    dist = jnp.linalg.norm(
        view.player_pos - view.restart_spot[None, :], axis=1)
    taker = jnp.argmin(jnp.where(mask, dist, jnp.inf)).astype(jnp.int32)
    return jnp.where(jnp.any(mask), taker, jnp.int32(NO_PLAYER))


class RuleTaker:
    """규칙 키커 — 역할 적합도 + 기술 + 거리 - 구조 보존 비용.

    점수식::

        역할 적합도 + control_weight * ball_control + stamina_weight * stamina
        - 종류별 거리 가중치 * 재개 지점까지 거리 - 구조 보존 비용

    거리 가중치가 종류마다 다른 것이 핵심이다. 스로인은 역할과 접근 거리를 함께 보지만
    (멀리 있는 풀백을 부르면 경기가 멈춘다), 페널티는 거리가 거의 영향을 주지 않는다
    (누가 찰지는 위치가 아니라 기술로 정한다).

    역할 적합도는 K리그1 120경기 실측이고(:data:`_PROPENSITY`), 아래 세 계수는 아직
    사전값이다.

    클로저가 아니라 클래스다 — 결정자는 env 속성이라 env와 함께 pickle되고, 지역 함수를
    돌려주면 ``restart_taker_mode="auto"`` env가 통째로 직렬화 불가가 된다.
    """

    def __init__(self, *, control_weight: float = 20.0,
                 stamina_weight: float = 10.0,
                 box_presence_cost: float = 15.0,
                 temperature: float = _DEFAULT_TEMPERATURE):
        self.control_weight = float(control_weight)
        self.stamina_weight = float(stamina_weight)
        self.box_presence_cost = float(box_presence_cost)
        if temperature < 0.0:
            raise ValueError(f"temperature must not be negative, got {temperature}")
        self.temperature = float(temperature)

    def __call__(self, view, key):
        control_weight = self.control_weight
        stamina_weight = self.stamina_weight
        box_presence_cost = self.box_presence_cost
        eligible = (view.team_id == view.restart_team) & view.active_player
        row = _kind_row(view)

        # ── 1) 감독이 지정한 순번이 있으면 그것이 이긴다 ──
        wanted = view.plan[view.restart_team, jnp.clip(view.plan_slot, 0, None)]
        has_plan = view.plan_slot >= 0
        # 우선순위 d의 선수가 활성이면 그 슬롯. 앞선 순번이 이긴다.
        match = (view.player_id[None, :] == wanted[:, None]) & eligible[None, :]
        found = jnp.any(match, axis=1)
        depth_pick = jnp.argmax(found)
        slot_pick = jnp.argmax(match[depth_pick]).astype(jnp.int32)
        planned = has_plan & jnp.any(found)

        # ── 2) 규칙 점수 ──
        affinity = _AFFINITY[row][view.role]
        distance = jnp.linalg.norm(
            view.player_pos - view.restart_spot[None, :], axis=1)
        # 코너에서는 제공권 좋은 중앙 선수를 박스에 남긴다 — 키커로 빼면 표적이 사라진다.
        keep_in_box = (
            (view.restart_kind == RK_CORNER)
            & jnp.isin(view.role,
                       jnp.asarray([ROLE_CENTRE_FORWARD, ROLE_CENTRE_BACK]))
        )
        structure = jnp.where(
            keep_in_box,
            box_presence_cost * jnp.clip(view.reach_z / 3.0, 0.0, 1.0),
            0.0)
        score = (
            affinity
            + control_weight * view.player_ctrl
            + stamina_weight * _decision_stamina(
                view.stamina_long, view.endurance_factor
            )
            - _DISTANCE[row] * distance
            - structure
        )
        # GK 강제는 **GK 홀드에만** 필요하다. [IFAB Law 16] 골킥은 GK 전용이 아니므로
        # 선호(적합도 +1000)만 주고 강제하지 않는다 — 강제하면 감독이 지정한 합법적인 필드
        # 선수를 무시하게 된다(실측: 골킥 계획에 슬롯3을 넣었는데 GK 0이 선택됐다).
        gk_only = view.restart_kind == RK_GK_HOLD
        usable = eligible & jnp.where(gk_only, view.is_gk, True)
        # Gumbel-max 표집 — 점수/온도를 로짓으로 하는 softmax와 같다. 온도 0이면
        # 잡음이 사라져 종전과 비트 단위로 같은 argmax가 된다(A/B·디버깅용).
        #
        # 잡음은 **슬롯 번호가 아니라 사람 신원(player_id)**을 따라야 한다. 슬롯으로
        # 색인하면 이 환경의 핵심 불변식인 R180 회전 + 팀 교환 공변성이 깨진다 — 회전한
        # 세계에서 슬롯 i는 원래 슬롯 i+N/2의 사람을 담고 있는데, 잡음은 여전히 i번째
        # 값을 받아 대응 슬롯이 뽑히지 않는다(실측: 스로인·코너·프리킥에서 어긋남).
        # player_id 순위로 색인하면 잡음이 사람을 따라가므로 회전과 교환에 공변한다.
        if self.temperature > 0.0:
            identity_rank = jnp.argsort(jnp.argsort(view.player_id))
            noise = jax.random.gumbel(key, score.shape, score.dtype)[identity_rank]
            score = score / jnp.float32(self.temperature) + noise
        # 동점이면 낮은 슬롯 — 결정론이어야 재현된다. argmax가 첫 최대를 준다.
        ruled = jnp.argmax(jnp.where(usable, score, -jnp.inf)).astype(jnp.int32)
        ruled = jnp.where(jnp.any(usable), ruled, jnp.int32(NO_PLAYER))

        # 지정 키커도 GK 전용 재개에서는 법을 못 넘는다.
        planned = planned & jnp.where(gk_only, view.is_gk[slot_pick], True)
        return jnp.where(planned, slot_pick, ruled).astype(jnp.int32)


def make_rule_taker(**overrides):
    """규칙 키커를 만든다."""

    return RuleTaker(**overrides)


TAKERS = {
    "nearest": nearest_taker,
    "auto": None,      # resolve에서 make_rule_taker()로 만든다
}
"""이름으로 고를 수 있는 기본 결정자. 사용자 함수는 이름 대신 직접 넘긴다."""


def resolve(mode):
    """이름 또는 사용자 함수를 키커 결정자로 바꾼다."""

    if callable(mode):
        return mode
    if mode == "auto":
        return make_rule_taker()
    if mode in TAKERS and TAKERS[mode] is not None:
        return TAKERS[mode]
    raise ValueError(
        f"restart_taker_mode must be callable or one of {sorted(TAKERS)}, "
        f"got {mode!r}")


def compile_plans(plans, team_ids, depth=None):
    """``{team: SetPiecePlan}`` → 고정 shape 표 ``(2, P, D)``.

    런타임에 파이썬 dict를 볼 수 없으므로 생성 시점에 편다. ``depth``는 우선순위 목록의
    최대 길이이고, 짧은 목록은 :data:`NO_PLAYER`로 채운다.
    """

    import numpy as np

    rows = {}
    for team, plan in (plans or {}).items():
        if team not in team_ids:
            raise ValueError(f"set-piece plan team must be one of {team_ids}, "
                             f"got {team!r}")
        if not isinstance(plan, SetPiecePlan):
            raise ValueError(
                f"set-piece plan must be a SetPiecePlan, got {plan!r}")
        rows[int(team)] = plan
    if depth is None:
        depth = max(
            (len(getattr(plan, name)) for plan in rows.values()
             for name in PLAN_FIELDS),
            default=0)
    depth = max(int(depth), 1)
    table = np.full((len(team_ids), len(PLAN_FIELDS), depth), NO_PLAYER,
                    np.int32)
    for team, plan in rows.items():
        for index, name in enumerate(PLAN_FIELDS):
            ids = tuple(getattr(plan, name))
            if len(ids) > depth:
                raise ValueError(
                    f"set-piece plan {name} has {len(ids)} entries but the "
                    f"table depth is {depth}")
            for position, pid in enumerate(ids):
                if not isinstance(pid, int) or isinstance(pid, bool):
                    raise ValueError(
                        f"set-piece plan {name} entries must be ints, "
                        f"got {pid!r}")
                table[team, index, position] = pid
    return jnp.asarray(table)
