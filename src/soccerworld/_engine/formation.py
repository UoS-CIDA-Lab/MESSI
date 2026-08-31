"""포메이션 지휘 — 경기 중에 팀의 **모양**을 바꾼다.

교체가 '누가 뛰는가'를 바꾼다면 이쪽은 '어디에 서는가'를 바꾼다. 두 결정은 같은 자리에서
같은 규약으로 내려진다 — 결정자는 **제안**하고 환경이 **승인**한다.

시뮬레이션에서 '실시간 명령'을 푸는 법
--------------------------------------
선수에게 메시지를 보내는 채널은 필요 없다. 선수는 이미 매 틱 관측을 읽고 있으므로,
**그 관측에 실린 조건 입력을 바꾸는 것**이 곧 명령이다. 교체가 정확히 그렇게 동작한다
(:class:`config.BenchPlayer`의 ``role_pos``가 투입 선수의 anchor prior가 된다).

여기서 쓰는 채널은 규범적 앵커 ``formation_home``이다. ``state.role_pos``와 혼동하면 안 된다:

  ``role_pos``        **서술적** — 이 선수가 현재 전술 epoch에서 실제로 어디 있었나
                                  (라이브볼 위치의 누적평균)
  ``formation_home``  **규범적** — 이 선수가 어디에 서야 하나(지휘관이 쓰는 값)

서술값을 전술 홈으로 쓰면 정책이 자기 과거 평균을 쫓게 되므로 둘을 나눈다.
승인된 포메이션 명령은 해당 팀의 전술 epoch도 새로 연다. 이때 ``role_pos``는 목표 앵커가
아니라 명령 경계의 실제 위치로 재기준화된다. ``role_pos_count`` 감소는 교체 증거가 아니며,
identity 변경은 오직 ``player_id``/``slot_generation``으로 판정한다.

적용 시점
---------
교체는 [IFAB Law 3] 때문에 데드볼 전용이지만 포메이션에는 그런 규칙이 없다. 승인된 명령은
라이브볼·데드볼 모두 명령 경계에서 규범 앵커를 즉시 바꾼다. 이것은 선수 좌표의 순간이동이
아니다. 선수는 기존 행동 정책과 물리를 통해 새 앵커를 향해 점진적으로 이동한다.

결정자 규약
-----------
    decide(view, key) -> layout_index        # (2,) int32, 팀별. 음수면 그대로 둔다

``view``는 :class:`FormationView`이고 전부 관측에서 얻을 수 있는 값이다 —
:mod:`substitution`과 같은 정보 경계다. 학습된 지휘관이 배포 시점에 없는 입력에 의존하면
안 되기 때문이다.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .constants import TEAM_0, TEAM_1

NO_CHANGE = -1
"""이 팀은 이번에 모양을 바꾸지 않는다."""


class FormationView(NamedTuple):
    """지휘관에게 주는 입력 — 전부 관측 가능한 값이다."""

    t: jnp.ndarray                 # () 현재 control tick
    game_duration: jnp.ndarray     # () 전체 길이(tick)
    ball_dead: jnp.ndarray         # () 교체 등 다른 감독 판단에 쓰는 현재 국면
    score: jnp.ndarray             # (2,)
    team_id: jnp.ndarray           # (N,)
    active_player: jnp.ndarray     # (N,)
    is_gk: jnp.ndarray             # (N,)
    formation_home: jnp.ndarray    # (N,2) 공격 접힘 프레임의 현재 규범 앵커
    stamina_long: jnp.ndarray      # (N,)
    endurance_factor: jnp.ndarray  # (N,) 1.0 중심의 선수별 지구력 계수
    territory: jnp.ndarray         # (2,) 팀별 영역 점유(공 x 평균을 자기 공격 방향으로 접은 값)
    layout_index: jnp.ndarray      # (2,) 현재 명령된 목표 레이아웃
    layout_since_t: jnp.ndarray    # (2,) 마지막으로 명령한 tick
    control_fps: jnp.ndarray       # () tick <-> 초 환산 — 규칙의 시간 상수는 초로 둔다


# ── 레이아웃 표 ───────────────────────────────────────────────────────────
# 포메이션은 (수비·미드·공격) 라인 분할이고, 블록은 그 전체를 앞뒤로 민다. 둘을 곱해
# 작은 categorical을 만든다 — 규칙이 다루기 쉽고, BC에서는 그대로 categorical 헤드이며,
# 트래킹 데이터에서 국면별 앵커를 클러스터링해 라벨을 만들 수 있다.
SHAPES = (
    ("4-4-2", (4, 4, 2)),
    ("4-3-3", (4, 3, 3)),
    ("3-5-2", (3, 5, 2)),
    ("5-4-1", (5, 4, 1)),
    ("4-2-4", (4, 2, 4)),
    # K리그1 2026 240 팀-경기의 선발 형태를 라인업 포지션에서 복원하면
    # 4-4-2 39.6% · 4-3-3 34.6% · 3-4-3 19.2% · 3-5-2 4.2% · 4-2-4 1.2% · 5-4-1 0.8%다.
    # 3-4-3(윙백을 수비로 세면 5-2-3)이 다섯 경기 중 하나인데 표에 없었다 — 지휘관이
    # 실제로 쓰이는 모양을 명령조차 할 수 없었다. 이 하나를 더하면 실측 형태의 98.8%를
    # 표현한다. **끝에** 붙여야 기존 layout_index의 의미가 그대로 유지된다.
    ("3-4-3", (3, 4, 3)),
)
"""필드 10명 기준 라인 분할. 로스터가 작으면 비율대로 줄인다."""

BLOCKS = (
    ("low", -0.14),
    ("mid", 0.0),
    ("high", 0.14),
)
"""블록 높이 — 하프길이에 대한 비율로 전 라인을 민다."""

WIDTHS = (
    ("narrow", 0.72),
    ("normal", 1.0),
)
"""폭 배율 — 라인 안의 y 간격."""

KICKOFF_LAYOUT = 0
"""표의 0번은 **이 경기의 킥오프 포메이션 그대로**다.

합성 레이아웃을 기본값으로 두면 명령이 없는데도 env가 저자가 지정한 포메이션 대신 내
4-4-2 표를 쓰게 된다(실측: 앵커가 최대 27.9 m 어긋났다). 지휘관이 아무 말도 하지 않으면
종전과 완전히 같은 모양이어야 한다."""

LAYOUTS = (("kickoff", None, 0.0, 1.0),) + tuple(
    (f"{shape} {block} {width}", lines, offset, scale)
    for shape, lines in SHAPES
    for block, offset in BLOCKS
    for width, scale in WIDTHS
)
"""``layout_index``가 가리키는 표. 이름은 사람이 읽고 로그에 남기기 위한 것이다."""

LAYOUT_NAMES = tuple(row[0] for row in LAYOUTS)

LAYOUT_INDEX_CAPACITY = 63
"""관측이 ``layout_index``를 정규화할 때 쓰는 **고정** 분모.

종전에는 ``len(LAYOUTS) - 1``로 나눴다. 그러면 표에 모양을 하나 더할 때마다 **기존 모든
레이아웃의 관측값이 바뀐다** — 4-4-2가 어제는 0.033이고 오늘은 0.028이 되므로, 표를 늘린
것만으로 학습된 정책과 수집된 데이터가 조용히 어긋난다. 고정 용량으로 나누면 표가 자라도
기존 값이 그대로 남고, 새 모양만 뒤에 새 값을 받는다.

63은 현재 37개(1 + 6모양 × 3블록 × 2폭)에 여유를 둔 값이다. 이 상한을 넘길 만큼 표가
커지면 그때는 categorical을 스칼라로 싣는 것 자체를 다시 봐야 한다."""

# 라인별 기본 깊이(하프길이 대비). 자기 골대가 -1, 상대 골대가 +1인 공격 접힘 프레임이다.
LINE_DEPTH = (-0.55, -0.08, 0.34)
"""수비·미드·공격 라인의 기본 x. 블록 오프셋이 여기에 더해진다."""

LINE_SPREAD = (
    {3: 0.52, 4: 0.62, 5: 0.70},          # 수비
    {2: 0.30, 3: 0.45, 4: 0.66, 5: 0.74},  # 미드 — 플랫4의 윙어가 가장 넓다
    {1: 0.00, 2: 0.24, 3: 0.62, 4: 0.70},  # 공격 — 투톱은 좁고 스리톱은 윙을 쓴다
)
"""라인별·인원별 폭(하프폭 대비). 인원이 같아도 라인마다 다르다 — 투톱을 백4와 같은
폭으로 벌리면 스트라이커가 터치라인에 선다.

이 값들은 **측정이 아니라 사전값**이다. K리그·PFF 트래킹에서 국면별 앵커를 뽑아
재캘리브레이션하는 것이 맞고, 그때까지는 눈으로 보고 납득되는 배치를 쓴다."""


def _spread(line_index, size, fallback=0.62):
    """표에 없는 인원수(소규모 로스터)는 인원에 비례해 벌린다."""

    table = LINE_SPREAD[line_index]
    if size in table:
        return table[size]
    return min(fallback, 0.16 + 0.13 * size)


def _line_sizes(lines, outfield):
    """라인 분할을 실제 필드 인원수에 비례 배분한다.

    소규모 로스터(4v3 같은 계약 테스트 config)에서도 정의돼야 한다. 비례 배분 후 남는
    인원은 미드필드에 준다 — 가장 덜 극단적인 선택이다.
    """

    total = sum(lines)
    sizes = [int(outfield * n / total) for n in lines]
    while sum(sizes) < outfield:
        sizes[1] += 1
    while sum(sizes) > outfield and sizes[1] > 0:
        sizes[1] -= 1
    return tuple(sizes)


def slot_ranks(kickoff_home, gk_mask, team_id):
    """슬롯마다 (깊이 순위, 좌우 키)를 킥오프 포메이션에서 **한 번** 정한다.

    모양을 바꿀 때 선수가 경기장을 가로지르면 안 된다. 그래서 배정을 임의로 하지 않고
    순위로 한다 — 킥오프에서 가장 깊었던 사람이 어느 모양에서도 가장 깊은 라인에 가고,
    가장 왼쪽이었던 사람이 라인 안에서 계속 왼쪽에 선다.

    순위를 **현재 앵커**가 아니라 킥오프에서 뽑는 이유는 경로 의존을 없애기 위해서다.
    현재 앵커로 매번 다시 매기면 같은 목표 layout도 지나온 경로에 따라 달라져, 관측에서
    재구성할 수 없는 숨은 전이 상태가 된다.

    반환은 ``(depth_rank, side_key)``. GK는 순위에서 빠진다(자기 골문을 지킨다).
    """

    home = np.asarray(kickoff_home, np.float64)
    gk = np.asarray(gk_mask, bool)
    teams = np.asarray(team_id, np.int64)
    depth_rank = np.full(len(home), -1, np.int32)
    for team in (TEAM_0, TEAM_1):
        mine = np.flatnonzero((teams == team) & (~gk))
        if not len(mine):
            continue
        # 깊은 쪽(자기 골대에 가까운 -x)이 rank 0. 동률은 슬롯 순서 — 결정론적이어야 한다.
        order = mine[np.argsort(home[mine, 0], kind="stable")]
        depth_rank[order] = np.arange(len(order), dtype=np.int32)
    return depth_rank, home[:, 1].astype(np.float32)


def kickoff_line_shape(kickoff_home, gk_mask, team_id, tol=0.5):
    """저자가 준 킥오프 배치에서 **암묵적 라인 구조**를 읽는다.

    킥오프 레이아웃은 합성 템플릿이 아니라 저자가 지정한 배치라 ``lines``가 없다. 그러면
    퇴장이 나도 재배치할 근거가 없어, **기본 설정에서만 재배치가 안 되는** 상태가 된다.
    저자 배치에도 깊이 무리라는 라인이 있으므로 그것을 읽어 다른 레이아웃과 같게 다룬다.

    반환은 ``(라인 인원 3개, 라인 깊이 3개, 라인 폭 3개)``다. 무리가 셋보다 적으면 뒤를
    비운다.
    """

    home = np.asarray(kickoff_home, np.float64)
    gk = np.asarray(gk_mask, bool)
    teams = np.asarray(team_id, np.int64)
    return tuple(_one_team_shape(home, gk, teams, side, tol)
                 for side in (TEAM_0, TEAM_1))


def _one_team_shape(home, gk, teams, side, tol):
    """한 팀의 킥오프 라인 구조.

    **팀마다 따로** 뽑아야 한다. 한쪽에서 뽑아 양 팀에 쓰면 비대칭 로스터(4v3)에서 전원인
    팀의 저자 앵커까지 최대 27 m 움직인다 — 저자가 지정한 배치를 이유 없이 갈아엎는 셈이다.
    """

    sizes, depths, spreads = [], [], []
    mine = np.flatnonzero((teams == side) & (~gk))
    if not len(mine):
        return (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    order = mine[np.argsort(home[mine, 0])]
    groups = [[order[0]]]
    for slot in order[1:]:
        if abs(home[slot, 0] - home[groups[-1][0], 0]) <= tol:
            groups[-1].append(slot)
        else:
            groups.append([slot])
    # 셋보다 많으면 가운데를 합친다 — 수비·미드·공격 세 밴드가 이 표의 규약이다.
    while len(groups) > 3:
        merge = min(range(1, len(groups) - 1),
                    key=lambda i: abs(home[groups[i][0], 0]
                                      - home[groups[i + 1][0], 0]))
        groups[merge] = groups[merge] + groups.pop(merge + 1)
    for group in groups:
        sizes.append(len(group))
        depths.append(float(np.mean(home[group, 0])))
        spreads.append(float(np.max(np.abs(home[group, 1]))))
    while len(sizes) < 3:
        sizes.append(0); depths.append(0.0); spreads.append(0.0)
    return tuple(sizes), tuple(depths), tuple(spreads)


def layout_line_plan(half_length, half_width, max_outfield,
                     kickoff_shape=None):
    """레이아웃 × **필드 인원수**별 라인 기하를 미리 편다.

    퇴장하면 팀 인원이 줄고, 남은 선수들이 그 인원에 맞는 형태로 다시 서야 한다. 종전에는
    킥오프 인원으로 표를 한 번만 만들어서, 퇴장한 슬롯만 비고 나머지는 11인용 자리에 그대로
    남았다 — 대형에 구멍이 뚫린 채로 경기가 이어졌다.

    인원수는 0..``max_outfield``의 작은 정수라 전 조합을 호스트에서 펼 수 있다. 런타임에는
    인원수로 인덱싱만 하므로, 앵커는 (목표 layout, **활성 마스크**)에서 결정된다 —
    ``active_player``도 관측에 있으므로 재구성 가능성을 잃지 않는다.

    반환은 ``(L, M+1, 3)`` 세 개와 ``(L, 3)`` 하나다: 라인 크기·시작 순위·좌우 폭, 그리고
    라인 깊이.
    """

    layouts = len(LAYOUTS)
    span = max_outfield + 1
    # 팀 축이 필요하다 — 킥오프 레이아웃의 라인 구조는 팀마다 다르다(합성 레이아웃은
    # 같지만 한 표로 두는 편이 인덱싱이 단순하다).
    sizes = np.zeros((layouts, 2, span, 3), np.int32)
    starts = np.zeros((layouts, 2, span, 3), np.int32)
    spread = np.zeros((layouts, 2, span, 3), np.float32)
    depth = np.zeros((layouts, 2, 3), np.float32)
    for index, (_, lines, offset, scale) in enumerate(LAYOUTS):
        if lines is None:
            # 킥오프 레이아웃 — 저자 배치에서 읽은 라인 구조를 쓴다. 이게 없으면 기본
            # 설정에서만 퇴장 후 재배치가 안 된다.
            if kickoff_shape is None:
                continue
            for side in range(2):
                shape, shape_depth, shape_spread = kickoff_shape[side]
                total = max(sum(shape), 1)
                for line in range(3):
                    depth[index, side, line] = shape_depth[line]
                for count in range(span):
                    row = (_line_sizes(shape, count)
                           if (count and sum(shape)) else (0, 0, 0))
                    sizes[index, side, count] = row
                    starts[index, side, count] = np.cumsum([0, row[0], row[1]])
                    for line, size in enumerate(row):
                        # 인원이 줄면 폭도 비례해 좁아진다.
                        spread[index, side, count, line] = (
                            shape_spread[line] * (count / total) if size else 0.0)
            continue
        for side in range(2):
            for line in range(3):
                depth[index, side, line] = (LINE_DEPTH[line] + offset) * half_length
            for count in range(span):
                row = _line_sizes(lines, count) if count else (0, 0, 0)
                sizes[index, side, count] = row
                starts[index, side, count] = np.cumsum([0, row[0], row[1]])
                for line, size in enumerate(row):
                    spread[index, side, count, line] = (
                        _spread(line, size) * scale * half_width if size else 0.0)
    return {"sizes": jnp.asarray(sizes), "starts": jnp.asarray(starts),
            "spread": jnp.asarray(spread), "depth": jnp.asarray(depth),
            "max_outfield": max_outfield,
            # 저자 배치가 전제한 인원. 이보다 줄면 킥오프 레이아웃도 재배치한다.
            # 저자 배치가 전제한 인원 — **팀마다** 다르다(4v3 같은 비대칭 로스터).
            "kickoff_outfield": jnp.asarray(
                [0, 0] if kickoff_shape is None
                else [int(sum(kickoff_shape[side][0])) for side in range(2)],
                jnp.int32)}


def active_anchors(plan, layout_index, active, gk_mask, team_id, ranks,
                   kickoff_home):
    """지금 뛰고 있는 인원에 맞춘 앵커 ``(N, 2)`` — 공격 접힘 프레임.

    순위는 킥오프에서 한 번 정한 것을 쓰되(경로 의존을 만들지 않는다) **살아 있는 선수만**
    남겨 다시 매긴다. 그래서 한 명이 퇴장하면 나머지가 10인용 형태로 자동 재배치된다.

    GK는 어느 모양에서도 킥오프 앵커를 지킨다 — 라인에 넣으면 팀이 골문을 비운다.
    킥오프 레이아웃(0번)도 저자가 준 배치를 그대로 쓴다.
    """

    depth_rank, side_key = ranks
    depth_rank = jnp.asarray(depth_rank, jnp.int32)
    side_key = jnp.asarray(side_key, jnp.float32)
    home = jnp.asarray(kickoff_home, jnp.float32)
    gk = jnp.asarray(gk_mask, bool)
    team = jnp.asarray(team_id, jnp.int32)
    live = jnp.asarray(active, bool) & (~gk)
    n = home.shape[0]
    far = jnp.int32(n + 1)

    out = home
    for side in (TEAM_0, TEAM_1):
        mine = live & (team == side)
        count = jnp.sum(mine).astype(jnp.int32)
        index = jnp.asarray(layout_index, jnp.int32)[side]
        capped = jnp.clip(count, 0, plan["max_outfield"])
        sizes = plan["sizes"][index, side, capped]        # (3,)
        starts = plan["starts"][index, side, capped]      # (3,)
        spread = plan["spread"][index, side, capped]      # (3,)
        depth = plan["depth"][index, side]                # (3,)

        # 깊이 순위: 살아 있는 선수만 재압축. 죽은 슬롯은 뒤로 밀어 순위에서 뺀다.
        depth_order = jnp.argsort(jnp.where(mine, depth_rank, far))
        compact = jnp.argsort(depth_order).astype(jnp.int32)   # 슬롯 -> 압축 순위
        # 이 선수가 속한 **실제 라인 번호**다. 종전에는 조건을 만족하는 라인의 **개수**를
        # 셌는데, 앞 라인이 비어 있으면(예: 3-4-3에 필드 선수가 1명이라 sizes=[0,1,0])
        # 그 라인이 개수에서 빠져 미드필더가 수비 라인 깊이에 선다(실측: 의도 x=-4.2 m
        # 대신 -28.875 m). 개수가 아니라 **가장 큰 참인 인덱스**를 골라야 한다.
        line_ids = jnp.arange(3, dtype=jnp.int32)
        covers = (compact[:, None] >= starts[None, :]) & (sizes[None, :] > 0)
        line = jnp.max(jnp.where(covers, line_ids[None, :], -1), axis=1)
        # 어느 라인에도 못 걸리면(빈 계획) 첫 비어 있지 않은 라인으로 보낸다.
        line = jnp.where(line < 0, jnp.argmax(sizes > 0).astype(jnp.int32), line)
        line = jnp.clip(line, 0, 2)

        # 라인 안에서는 좌우 키 순서. (라인, 좌우)로 사전식 정렬하면 라인 시작점 기준
        # 위치가 그대로 나온다.
        lexical = jnp.where(
            mine, line.astype(jnp.float32) * 4.0 + side_key * 1e-3, 1e6)
        side_order = jnp.argsort(lexical)
        side_rank = jnp.argsort(side_order).astype(jnp.int32)
        within = side_rank - starts[line]
        size = sizes[line]
        span = jnp.where(
            size > 1,
            within.astype(jnp.float32) / jnp.maximum(size - 1, 1).astype(jnp.float32)
            * 2.0 - 1.0,
            0.0)
        placed = jnp.stack([depth[line], span * spread[line]], axis=1)
        # 킥오프 레이아웃과 GK·비활성은 원래 앵커를 지킨다.
        # 킥오프 레이아웃도 인원이 줄면 재배치한다. 전원이 있을 때만 저자 배치를 그대로
        # 쓴다 — 저자가 지정한 배치는 그 인원을 전제로 한 것이다.
        #
        # 비교 대상은 **저자 배치가 전제한 인원**이다. 현재 활성 수와 비교하면 항진식이 되어
        # (count는 곧 활성 수다) 재배치가 영영 걸리지 않는다.
        full = count >= plan["kickoff_outfield"][side]
        use = mine & ((index != KICKOFF_LAYOUT) | (~full))
        out = jnp.where(use[:, None], placed, out)
    return out


def no_change(view, key):
    """아무것도 바꾸지 않는 지휘관 — 고정 포메이션(종전 동작)."""

    del view, key
    return jnp.full((2,), NO_CHANGE, jnp.int32)


def make_scoreline_rule(
    *,
    earliest_fraction: float = 0.60,
    hold_seconds: float = 60.0,
    chase_layout: str = "4-2-4 high normal",
    protect_layout: str = "5-4-1 low narrow",
    base_layout: str = "kickoff",
):
    """내장 규칙 지휘관 — 스코어와 남은 시간으로 모양을 고른다.

    실제 감독이 하는 가장 뚜렷하고 트래킹 데이터에서도 보이는 행동만 넣는다. 지고 있으면
    라인을 올리고 공격 인원을 늘리며, 이기고 있으면 내려앉는다. 경기 초반에는 바꾸지 않고,
    한 번 바꾸면 ``hold_seconds`` 동안 유지한다 — 매 틱 흔들리면 관측이 잡음이 된다.
    """

    return ScoreCommander(
        earliest_fraction=earliest_fraction, hold_seconds=hold_seconds,
        chase=LAYOUT_NAMES.index(chase_layout),
        protect=LAYOUT_NAMES.index(protect_layout),
        base=LAYOUT_NAMES.index(base_layout))


class ScoreCommander:
    """스코어 기반 규칙 지휘관.

    클로저가 아니라 클래스다 — 지휘관은 env 속성이라 env와 함께 pickle된다.
    """

    def __init__(self, *, earliest_fraction, hold_seconds, chase, protect, base):
        self.earliest_fraction = float(earliest_fraction)
        self.hold_seconds = float(hold_seconds)
        self.chase = int(chase)
        self.protect = int(protect)
        self.base = int(base)

    def __call__(self, view, key):
        del key
        earliest_fraction = self.earliest_fraction
        hold_seconds = self.hold_seconds
        chase, protect, base = self.chase, self.protect, self.base
        # 비율 비교는 XLA 역수 곱셈 반올림에 걸린다(t=300/600이 0.49999997).
        # tick 공간에서 비교한다 — :mod:`manager`·:mod:`substitution`과 같은 규약이다.
        late = view.t.astype(jnp.float32) >= (
            jnp.float32(earliest_fraction)
            * view.game_duration.astype(jnp.float32))
        # 유지시간은 **초**로 정하고 tick은 시계에서 파생한다 — control_fps가 바뀌어도
        # 같은 시간이어야 한다.
        hold_ticks = jnp.maximum(
            jnp.round(jnp.float32(hold_seconds) * view.control_fps), 1.0)
        hold_elapsed = (
            view.t - view.layout_since_t
        ).astype(jnp.float32) >= hold_ticks

        margin = jnp.stack([
            view.score[0] - view.score[1],
            view.score[1] - view.score[0],
        ]).astype(jnp.int32)
        want = jnp.where(margin < 0, chase,
                         jnp.where(margin > 0, protect, base)).astype(jnp.int32)
        change = late & hold_elapsed & (want != view.layout_index)
        return jnp.where(change, want, jnp.int32(NO_CHANGE))


DECIDERS = {
    "fixed": no_change,
    "auto": make_scoreline_rule(),
    "external": no_change,
}
"""이름으로 고를 수 있는 기본 지휘관.

``fixed``와 ``external``은 둘 다 스스로 바꾸지 않지만 뜻이 다르다 — ``fixed``는 '이 경기는
고정 포메이션이다'이고 ``external``은 '모양은 호출자(학습 정책)가 정한다'이다.
:mod:`substitution`과 같은 규약이다."""


def resolve(mode):
    """모드 이름 또는 사용자 함수를 지휘관으로 바꾼다."""

    if callable(mode):
        return mode
    if mode in DECIDERS:
        return DECIDERS[mode]
    raise ValueError(
        f"formation mode must be callable or one of {sorted(DECIDERS)}, got {mode!r}"
    )
