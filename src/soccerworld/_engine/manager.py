"""감독 — 한 팀의 벤치와 전술을 맡는다.

교체와 포메이션은 **같은 사람의 두 결정**이다. 따로 두면 세 가지가 어긋난다.

  * 같은 값을 두 번 만든다 — 스태미나·스코어·팀 구성을 두 뷰가 각각 계산했다.
  * 연동된 결정을 표현할 수 없다 — "수비수를 빼고 공격수를 넣으면서 4-2-4로 전환"은
    두 결정자가 서로를 모르면 나올 수 없다.
  * 꽂는 지점이 둘이라, 사용자가 감독 하나를 갈아 끼우려면 두 군데를 건드려야 한다.

규약
----
    decide(view, key) -> ManagerDecision(out_slot, bench_index, layout)

``view``는 :class:`ManagerView`이고 전부 관측에서 얻을 수 있는 값이다. 특권 정보를 넣지
않는 이유는 BC/RL 정책과 같은 정보 경계를 공유해야 학습된 감독이 배포 시에도 같은 입력을
받기 때문이다. 승인은 여전히 환경이 한다 — 감독은 **제안**할 뿐이다.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import formation as formation_module
from .constants import (
    ENDURANCE_DECISION_DELTA_CAP,
    ENDURANCE_DECISION_GAIN,
    ENDURANCE_FACTOR_REFERENCE,
    IFAB_MAX_SUBSTITUTION_WINDOWS,
    NO_PLAYER,
    TEAM_0,
    TEAM_1,
)
from .setpiece_taker import (
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MID,
    ROLE_COUNT,
    ROLE_FULL_BACK,
    ROLE_GK,
    ROLE_WIDE_FORWARD,
    ROLE_WIDE_MID,
    classify_roles,
)

MANAGER_RULES_VERSION = 3
"""Version of the named ``manager="auto"`` decision semantics."""


class ManagerView(NamedTuple):
    """감독이 보는 것 — 전부 관측 가능한 값이다."""

    t: jnp.ndarray                 # () 현재 control tick
    game_duration: jnp.ndarray     # () 전체 길이(tick)
    control_fps: jnp.ndarray       # () tick <-> 초
    ball_dead: jnp.ndarray         # () 지금 교체가 법적으로 가능한가
    score: jnp.ndarray             # (2,)
    # ── 경기장 안 ──
    team_id: jnp.ndarray           # (N,)
    active_player: jnp.ndarray     # (N,)
    is_gk: jnp.ndarray             # (N,)
    stamina_long: jnp.ndarray      # (N,)
    endurance_factor: jnp.ndarray  # (N,) 1.0 중심의 선수별 지구력 계수
    stamina_short: jnp.ndarray     # (N,)
    yellow_cards: jnp.ndarray      # (N,)
    formation_home: jnp.ndarray    # (N,2) 규범 앵커(공격 접힘)
    vmax: jnp.ndarray              # (N,)
    player_ctrl: jnp.ndarray       # (N,)
    # ── 벤치 ──
    # 벤치 자리의 **신원**이 단일 진실원천이다(NO_PLAYER = 비었거나 이미 투입됨).
    # 가용 여부만 담으면 감독 경로와 ``substitution_view``가 서로 다른 값을 보게 되고,
    # 두 경로를 오가는 결정자가 같은 상태에서 다른 자리를 고른다.
    bench_player_id: jnp.ndarray   # (2,B)
    bench_is_gk: jnp.ndarray       # (2,B)
    bench_role_pos: jnp.ndarray    # (2,B,2) 그 선수의 앵커 — like-for-like에 쓴다
    bench_vmax: jnp.ndarray        # (2,B)
    bench_ctrl: jnp.ndarray        # (2,B)
    bench_endurance_factor: jnp.ndarray  # (2,B)
    # ── 자원 ──
    subs_remaining: jnp.ndarray    # (2,)
    max_substitutions: jnp.ndarray  # () int32; 팀당 총 교체 상한
    sub_windows_used: jnp.ndarray  # (2,)
    sub_window_open: jnp.ndarray   # (2,) 이번 정지에서 이미 바꿨는가
    half_time: jnp.ndarray         # () 하프타임 정지인가 — 기회를 쓰지 않는다
    max_simultaneous: jnp.ndarray  # () int32
    # ── 전술 ──
    layout_index: jnp.ndarray      # (2,)
    layout_since_t: jnp.ndarray    # (2,)
    ball_progress: jnp.ndarray     # (2,) 공 x를 자기 공격 방향으로 접은 값 / 하프길이
    team_centroid: jnp.ndarray     # (2,) 팀 중심 x(접힘) / 하프길이
    team_width: jnp.ndarray        # (2,) 팀 좌우 산포 / 하프폭


class ManagerDecision(NamedTuple):
    """감독의 제안. 승인은 환경이 한다."""

    out_slot: jnp.ndarray          # (2,K)  NO_PLAYER면 그 자리는 비운다
    bench_index: jnp.ndarray       # (2,K)
    layout: jnp.ndarray            # (2,)   음수면 모양을 바꾸지 않는다


class DecisionParams(NamedTuple):
    """감독·키커 결정자의 **파라미터**를 담는 동적 PyTree.

    학습된 결정 정책은 함수와 파라미터가 수명이 다르다. 함수 구조는 경기 내내 고정이지만
    가중치는 갱신마다 바뀐다. 그런데 결정자를 env 생성 시 클로저로 묶으면 그 둘이 하나로
    붙어, 새 체크포인트를 꽂을 때마다 **다른 파이썬 객체**가 되어 전체 스텝이 재컴파일된다.

    그래서 함수는 정적으로 두고(``manager=``·``restart_taker_mode=``) 파라미터만 이 컨테이너로
    ``step_env_array``/``reset_state``에 넘긴다. 같은 shape·dtype·트리 구조의 새 가중치는
    JIT 캐시에 그대로 맞으므로 재컴파일이 없다.

    교체·포메이션 축에는 이미 다른 탈출구가 있었다 — 결정을 밖에서 만들어 ``substitution=``·
    ``formation=``으로 주입하는 것이다. 키커 축에는 그 탈출구가 없었다. 재개는 서브스텝 스캔
    한가운데에서 열리므로 호출자가 "언제 누구를 지정해야 하는지" 미리 알 수 없어, 결정을
    주입하는 방식 자체가 성립하지 않는다. 그 축은 **파라미터를 넣고 결정자를 안에서 부르는**
    이 경로가 유일한 방법이다.

    필드가 ``None``이면 그 축에는 파라미터가 없다는 뜻이고, 그 축의 결정자는 2인자 규약
    (``decide(view, key)``)이어야 한다.
    """

    manager: object = None
    taker: object = None


def _hold(view):
    """팀별 인원 (2,) — 활성 필드 선수 수."""

    live = view.active_player & (~view.is_gk)
    return jnp.stack([
        jnp.sum(live & (view.team_id == TEAM_0)),
        jnp.sum(live & (view.team_id == TEAM_1)),
    ]).astype(jnp.int32)


def _endurance_delta(endurance_factor):
    """Return the conservative, 1.0-centred policy-only endurance delta."""

    centered = jnp.asarray(endurance_factor, jnp.float32) - jnp.float32(
        ENDURANCE_FACTOR_REFERENCE
    )
    return (
        jnp.clip(
            centered,
            -jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
            jnp.float32(ENDURANCE_DECISION_DELTA_CAP),
        )
        * jnp.float32(ENDURANCE_DECISION_GAIN)
    )


def _decision_stamina(stamina, endurance_factor):
    """Combine current stamina and endurance without changing factor-1 results."""

    return jnp.clip(
        jnp.asarray(stamina, jnp.float32) + _endurance_delta(endurance_factor),
        0.0,
        1.0,
    )


def _decision_capacity(value, endurance_factor):
    """Conservative sustained-capacity estimate used only for bench ranking."""

    return jnp.asarray(value, jnp.float32) * (
        jnp.float32(1.0) + _endurance_delta(endurance_factor)
    )


# ── 교체: 누가 나가는가 ─────────────────────────────────────────────────
#
# K리그1 2026 120경기 교체 1,148건을 나간 선수의 라인업 포지션으로 집계하고, 선발 XI의
# 평균 역할 인원으로 나눠 **선수 1인당 교체될 확률**로 정규화한 값이다.
#
#     WF .259  CF .221  WM .220  CM .165  FB .100  CB .049  GK ≈ 0
#
# 즉 측면 공격수가 센터백보다 5배 자주 교체된다. 피로만 보면 이 차이가 나오지 않는다 —
# 실제 감독은 "누가 지쳤나"만이 아니라 "그 자리를 바꾸는 것이 싼가"를 함께 본다.
# 평균 0으로 옮겨 긴급도에 더한다(양수면 더 잘 나간다).
_SUB_ROLE_PROPENSITY = {
    ROLE_GK: 0.000, ROLE_CENTRE_BACK: 0.049, ROLE_FULL_BACK: 0.100,
    ROLE_CENTRE_MID: 0.165, ROLE_WIDE_MID: 0.220,
    ROLE_CENTRE_FORWARD: 0.221, ROLE_WIDE_FORWARD: 0.259,
}
_SUB_ROLE_BIAS = jnp.asarray(
    [_SUB_ROLE_PROPENSITY[r] for r in range(ROLE_COUNT)], jnp.float32)
_SUB_ROLE_BIAS = _SUB_ROLE_BIAS / jnp.mean(_SUB_ROLE_BIAS[1:]) - 1.0
"""역할별 교체 성향 편향. GK는 평균 계산에서 뺀다 — 실측 0.26%라 평균을 끌어내린다."""


# ── 교체: 역할 적합 ──────────────────────────────────────────────────────
def _substitution_proposal(view, cfg):
    """교체 쌍을 **점수**로 고른다.

    종전에는 가장 지친 선수를 뺀 뒤 벤치에서 앞자리부터 채웠다 — 포지션도 능력도 보지
    않아 센터백 자리에 스트라이커가 들어갔다. 이제 두 단계로 나눈다.

      나갈 사람  = 피로 + 경고 위험 (자기 라인의 인원이 얇으면 억제)
      들어올 사람 = 나간 자리의 앵커에 가까울수록(역할 적합) + 능력이 나을수록

    ``bench_role_pos``가 관측에 실려 있어야 이 매칭이 가능하다 — 인원수만으로는 애초에
    표현되지 않는다.
    """

    width = int(view.max_simultaneous)
    take = max(1, min(int(cfg["max_per_window"]), width))
    rank = jnp.arange(width, dtype=jnp.int32)
    elapsed = view.t.astype(jnp.float32) / jnp.maximum(
        view.game_duration.astype(jnp.float32), 1.0)
    # 실측: 교체 1,148건 중 45분 이전은 28건(2.4%)뿐이고 대부분 부상·경고다. 나머지는
    # 하프타임(15.9%)과 후반에 몰린다. 그래서 정기 교체는 ``earliest_fraction`` 이후에만
    # 열고, 경고를 안은 선수만 시각과 무관하게 뺄 수 있게 한다.
    # 경과 비율을 만들어 비교하면 안 된다. XLA가 나눗셈을 역수 곱셈으로 바꾸는 탓에
    # ``t / game_duration``이 t=300, duration=600에서 0.5가 아니라 0.49999997이 되고,
    # ``>= 0.50``이 1 ULP 차이로 거짓이 된다 — 하필 그 tick이 하프타임이라 실측 교체의
    # 최빈값(15.9%)이 구조적으로 제안되지 않았다. tick 공간에서 비교하면 그 반올림이
    # 개입할 자리가 없다(0.5 × 600 = 300은 정확하다).
    late = view.t.astype(jnp.float32) >= (
        jnp.float32(cfg["earliest_fraction"])
        * view.game_duration.astype(jnp.float32))

    n_slots = view.team_id.shape[0]
    n_bench = view.bench_player_id.shape[1]
    # 반환 폭 K가 로스터보다 클 수 있다(2v2에 K=5). 마지막 인덱스로 채우고 범위 밖은
    # 마스크로 무효화한다 — 잘라내면 ``rank``와 브로드캐스트가 깨진다.
    slot_take = jnp.minimum(rank, max(n_slots - 1, 0))

    booked = view.yellow_cards > 0
    decision_stamina = _decision_stamina(
        view.stamina_long, view.endurance_factor
    )
    # 역할은 앵커에서 유도한다 — 슬롯 번호로 굳히면 포메이션이 바뀔 때 뜻이 달라진다.
    # 키커 결정자와 **같은 함수**를 쓴다: 한 경기 안에서 "이 선수는 윙어다"가 두 정책에서
    # 서로 다른 뜻이면 안 된다.
    role = classify_roles(view.formation_home, view.is_gk,
                          view.team_id, view.active_player)
    # 뺄 가치: 지칠수록·경고를 안고 있을수록·그 역할이 원래 자주 교체될수록 크다.
    urgency = (
        (1.0 - decision_stamina) * cfg["fatigue_weight"]
        + booked.astype(jnp.float32) * cfg["booked_weight"]
        + _SUB_ROLE_BIAS[role] * cfg["role_bias_weight"]
    )
    # 피로만 보면 실측 교체 횟수가 나오지 않는다. K리그1 240 팀-경기 중 187(78%)이
    # 5명을 **다 썼고** 중앙값이 68분이다 — 그 중 상당수는 지쳐서가 아니라 경기를
    # 바꾸려고 하는 전술 교체다. 그래서 문턱을 시간과 남은 카드로 완화한다:
    # 늦을수록, 그리고 카드를 많이 쥐고 있을수록 바꿀 이유가 커진다.
    pressure = jnp.clip(
        (elapsed - jnp.float32(cfg["earliest_fraction"]))
        / jnp.maximum(1.0 - jnp.float32(cfg["earliest_fraction"]), 1e-6),
        0.0, 1.0)
    # 남은 카드의 **비율**이다. 상대 팀 잔량으로 나누면 두 팀이 똑같이 3장을 썼을 때
    # 둘 다 1.0이 되어 "카드를 아직 많이 쥐고 있다"로 읽힌다 — 척도는 팀당 상한이어야 한다.
    budget_share = (view.subs_remaining.astype(jnp.float32)
                    / jnp.float32(max(int(view.max_substitutions), 1)))
    relaxed_floor = (jnp.float32(cfg["stamina_floor"])
                     + jnp.float32(cfg["tactical_gain"]) * pressure * budget_share)
    eligible = view.active_player & (~view.is_gk) & (
        (decision_stamina < relaxed_floor[
            jnp.clip(view.team_id, 0, 1)]) | booked)

    if n_bench == 0:
        # 벤치가 없으면 교체 자체가 불가능하다. 빈 축에 max/argmax를 걸면 죽는다
        # (실측: zero-size array to reduction operation max).
        none = jnp.full((2, width), NO_PLAYER, jnp.int32)
        return none, none

    out_slots, bench_idx = [], []
    for team in (TEAM_0, TEAM_1):
        mine = view.team_id == team
        # 가치가 큰 순서로 뺀다(부호를 뒤집어 argsort).
        cost = jnp.where(mine & eligible, -urgency, jnp.inf)
        picks = jnp.argsort(cost)[slot_take].astype(jnp.int32)
        picked_ok = jnp.isfinite(cost[picks]) & (rank < n_slots)

        avail = (view.bench_player_id[team] >= 0) & (~view.bench_is_gk[team])
        # 역할 적합 — 나간 자리의 앵커와 벤치 선수 앵커의 거리. 가까울수록 좋다.
        gap = jnp.linalg.norm(
            view.bench_role_pos[team][None, :, :]
            - view.formation_home[picks][:, None, :], axis=-1)
        bench_capacity = _decision_capacity(
            view.bench_vmax[team], view.bench_endurance_factor[team]
        )
        field_capacity = _decision_capacity(
            view.vmax[picks], view.endurance_factor[picks]
        )
        upgrade = (
            (bench_capacity[None, :] - field_capacity[:, None])
            * cfg["speed_weight"]
            + (view.bench_ctrl[team][None, :] - view.player_ctrl[picks][:, None])
            * cfg["control_weight"])
        fit = jnp.where(avail[None, :], -gap * cfg["role_weight"] + upgrade,
                        -jnp.inf)
        # **쓸 후보만** 벤치를 가져간다. 종전에는 K개 전부가 먼저 점유한 뒤 나중에
        # ``rank < take``로 뒤쪽을 버려서, 버려질 4·5순위가 유일한 벤치를 선점했다
        # (실측: 교체 가능한 슬롯과 벤치가 있는데 제안이 전부 -1).
        budget = jnp.minimum(
            jnp.int32(take), view.subs_remaining[team]).astype(jnp.int32)
        # 한 정지에 몇 명을 바꾸는가. 실측 분포는 1명 65% · 2명 31% · 3명 3.6% ·
        # 4명 0.1%(평균 1.40명)이다. 상한만 두면 매 정지가 상한을 쓴다 — 자격이 있는
        # 사람을 전부 바꾸기 때문이다. 그래서 두 번째·세 번째 교체에는 **더 깊은 피로**를
        # 요구한다. rank 0의 문턱은 ``stamina_floor``와 같아서 첫 교체는 달라지지 않는다.
        # 첫 교체는 완화된 문턱을 쓰지만, **두 번째부터는 완화를 주지 않는다**. 전술
        # 교체는 "경기를 바꾸려고 한 명 바꾼다"이지 "늦었으니 세 명 바꾼다"가 아니다 —
        # 실측 동시 인원은 1명 65% · 2명 31% · 3명 3.6%다. 완화된 문턱을 랭크 전체에
        # 적용하면 후반에 자격자가 한꺼번에 생겨 매 정지가 상한을 쓴다.
        rank_floor = jnp.where(
            rank == 0,
            relaxed_floor[team],
            jnp.float32(cfg["stamina_floor"])
            - rank.astype(jnp.float32)
            * jnp.float32(cfg["extra_stamina_step"]))
        deep_enough = (decision_stamina[picks] < rank_floor) | booked[picks]
        usable = picked_ok & (rank < budget) & deep_enough
        fit = jnp.where(usable[:, None], fit, -jnp.inf)
        order = jnp.argsort(-jnp.max(fit, axis=1))
        chosen = jnp.full((width,), NO_PLAYER, jnp.int32)

        # ``fit``·``order``를 기본인자로 묶는다. 지금은 같은 반복 안에서 곧바로 쓰이므로
        # 동작이 같지만, 클로저가 팀 루프 변수를 늦게 참조하는 형태는 한 줄만 옮겨도
        # 조용히 다른 팀의 값을 읽는다(ruff B023).
        def claim(step, carry, fit=fit, order=order):
            picked, used = carry
            row = order[step]
            score = jnp.where(used, -jnp.inf, fit[row])
            best = jnp.argmax(score).astype(jnp.int32)
            ok = jnp.isfinite(score[best])
            picked = picked.at[row].set(jnp.where(ok, best, NO_PLAYER))
            used = used | (jnp.arange(n_bench) == best) & ok
            return picked, used

        chosen, _ = jax.lax.fori_loop(
            0, width, claim, (chosen, jnp.zeros(n_bench, bool)))

        # [IFAB Law 3] 하프타임 교체는 3회 기회에 포함되지 않는다. 이 면제가 승인부에만
        # 있었을 때는, 기회를 다 쓴 팀이 하프타임에 **제안조차 하지 않았다** — 환경은
        # 승인했을 교체를 아무도 요청하지 않는 어긋남이다. 제안자와 승인자가 같은 법을
        # 봐야 한다.
        has_window = (
            (view.sub_windows_used[team] < IFAB_MAX_SUBSTITUTION_WINDOWS)
            | view.half_time
        )
        allowed = (
            (late | booked[picks])
            & view.ball_dead
            & picked_ok
            & (chosen >= 0)
            & usable
            & has_window
            # 이번 정지에서 아직 아무도 안 바꿨을 때만 — 한 정지에 한 번 결정한다.
            & (~view.sub_window_open[team])
        )
        routine_out = jnp.where(allowed, picks, jnp.int32(NO_PLAYER))
        routine_bench = jnp.where(allowed, chosen, jnp.int32(NO_PLAYER))

        # 활동 가능한 GK가 없는 팀은 경기 시각과 무관한 비상상황이다. 일반 피로 교체와
        # 섞으면 GK 투입보다 앞선 rank가 벤치·동시교체 폭을 차지할 수 있으므로, 합법한
        # 정지에서는 정확히 한 쌍만 제안하고 나머지 rank를 닫는다. 나갈 필드 선수는 이미
        # 관측 가능한 urgency(피로·경고·역할 성향)가 가장 큰 선수, 벤치 GK는 가장 앞의
        # 가용 자리로 결정한다. 둘 다 동률이면 ``argmax``의 낮은 인덱스 규약이 고정한다.
        active_team = mine & view.active_player
        goalkeeper_missing = ~jnp.any(active_team & view.is_gk)
        field_candidates = active_team & (~view.is_gk)
        goalkeeper_bench = (
            (view.bench_player_id[team] >= 0) & view.bench_is_gk[team]
        )
        emergency_out = jnp.argmax(
            jnp.where(field_candidates, urgency, -jnp.inf)
        ).astype(jnp.int32)
        emergency_bench = jnp.argmax(
            goalkeeper_bench.astype(jnp.int32)
        ).astype(jnp.int32)
        emergency_allowed = (
            goalkeeper_missing
            & jnp.any(field_candidates)
            & jnp.any(goalkeeper_bench)
            & view.ball_dead
            & (view.subs_remaining[team] > 0)
            & has_window
            & (~view.sub_window_open[team])
        )
        emergency_out_row = jnp.full((width,), NO_PLAYER, jnp.int32).at[0].set(
            jnp.where(emergency_allowed, emergency_out, NO_PLAYER)
        )
        emergency_bench_row = jnp.full(
            (width,), NO_PLAYER, jnp.int32
        ).at[0].set(jnp.where(emergency_allowed, emergency_bench, NO_PLAYER))
        # GK가 없는데 비상 투입도 불가능하면 필드→필드 교체를 fallback처럼 내지 않는다.
        # 예산·창·벤치가 다시 합법해질 때까지 제안 없음이 정확한 신호다.
        out_slots.append(jnp.where(
            goalkeeper_missing, emergency_out_row, routine_out
        ))
        bench_idx.append(jnp.where(
            goalkeeper_missing, emergency_bench_row, routine_bench
        ))
    return jnp.stack(out_slots), jnp.stack(bench_idx)


# ── 포메이션: 31개 레이아웃 점수화 ───────────────────────────────────────
def _layout_features():
    """레이아웃별 정적 성질 — 공격성·폭·수비 인원. 점수 함수의 입력이다."""

    attack, width, defenders = [], [], []
    for name, lines, offset, scale in formation_module.LAYOUTS:
        if lines is None:                      # 킥오프 형태 — 중립으로 둔다
            attack.append(0.0); width.append(1.0); defenders.append(4.0)
            continue
        total = sum(lines)
        attack.append(offset * 3.0 + (lines[2] - lines[0]) / total)
        width.append(scale)
        defenders.append(float(lines[0]))
    return (jnp.asarray(attack, jnp.float32), jnp.asarray(width, jnp.float32),
            jnp.asarray(defenders, jnp.float32))


def _formation_proposal(view, cfg, out_slot):
    """전체 레이아웃 표를 점수화해 최고점을 고른다.

    종전에는 득실 **부호**만 봤다 — 1점 차와 4점 차, 61분과 89분, 11명과 10명을 똑같이
    다뤘다. 이제 긴급도(남은 시간 × 득실차), 인원 차, 체력, 영역 점유를 함께 본다.

    바꾸는 데는 비용이 있다. 현재 모양에 가산점을 줘서 근소한 차이로 흔들리지 않게 한다.
    """

    attack, width, defenders = _layout_features()
    hold_ticks = jnp.maximum(
        jnp.round(jnp.float32(cfg["hold_seconds"]) * view.control_fps), 1.0)
    elapsed = view.t.astype(jnp.float32) / jnp.maximum(
        view.game_duration.astype(jnp.float32), 1.0)
    hold_elapsed = (
        view.t - view.layout_since_t
    ).astype(jnp.float32) >= hold_ticks
    # 교체 제안과 **같은 이유**로 tick 공간에서 비교한다(역수 곱셈 ULP).
    late = view.t.astype(jnp.float32) >= (
        jnp.float32(cfg["earliest_fraction"])
        * view.game_duration.astype(jnp.float32))
    counts = _hold(view)
    decision_stamina = _decision_stamina(
        view.stamina_long, view.endurance_factor
    )

    want = []
    for team in (TEAM_0, TEAM_1):
        other = TEAM_1 - team
        margin = (view.score[team] - view.score[other]).astype(jnp.float32)
        # 긴급도: 남은 시간이 적을수록, 점수차가 클수록 크다. 지고 있으면 양수.
        remaining = jnp.clip(1.0 - elapsed, 0.0, 1.0)
        chase = -margin * (1.0 + cfg["urgency_gain"] * (1.0 - remaining))
        # 인원 차 — 열세면 물러선다. 우세면 밀어붙인다.
        edge = (counts[team] - counts[other]).astype(jnp.float32)
        # 체력 — 지친 팀은 넓게 벌리기 어렵다.
        mine = (view.team_id == team) & view.active_player
        fitness = jnp.sum(jnp.where(mine, decision_stamina, 0.0)) / jnp.maximum(
            jnp.sum(mine), 1).astype(jnp.float32)
        # 영역 — 상대 진영에서 놀고 있으면 그 상태를 유지할 모양을 선호한다.
        territory = view.team_centroid[team]

        score = (
            attack * (chase * cfg["chase_gain"]
                      + edge * cfg["edge_gain"]
                      + territory * cfg["territory_gain"])
            + width * (fitness - cfg["width_fitness_pivot"]) * cfg["width_gain"]
            - jnp.abs(defenders - (4.0 - jnp.clip(edge, -2.0, 2.0)))
            * cfg["shape_gain"]
        )
        # 현재 모양 가산점 — 근소한 차이로 흔들리지 않게 한다.
        score = score.at[view.layout_index[team]].add(cfg["incumbent_bonus"])
        best = jnp.argmax(score).astype(jnp.int32)

        # 교체와 연동: 이번에 사람을 바꾸는 팀은 같은 정지에 모양도 함께 정리한다.
        substituting = jnp.any(out_slot[team] >= 0)
        # 퇴장은 경기 시각과 무관한 즉시 전술 비상상황이다. 종전에는 ``late``가
        # 모든 포메이션 변경을 막아, 전반 조기 퇴장 뒤에도 earliest_fraction까지
        # 11인용 명령을 유지했다. 수적 우열은 활성 인원만으로 관측 가능하며, 양 팀이
        # 열세를 보호하거나 우세를 활용할 수 있게 routine hold와 분리한다.
        numerical_imbalance = edge != 0.0
        change = (
            ((hold_elapsed[team] & (late | substituting)) | numerical_imbalance)
            & (best != view.layout_index[team])
        )
        want.append(jnp.where(change, best, jnp.int32(-1)))
    return jnp.stack(want)


DEFAULT_RULES = {
    # 실측 교체 시각은 p25 57.7분 · p50 68.3분 · p75 80.1분이고, 최빈값은 **하프타임
    # 정각**(15.9%)이다. 종전 0.55는 90분 기준 49.5분이라 그 최빈값을 구조적으로
    # 배제했다. 0.50이면 하프타임이 첫 기회가 된다.
    "earliest_fraction": 0.50,
    # 90분 전체 롤아웃에서 stamina_long은 하프타임 0.589, 종료 시 평균 0.158(p25 0.001)
    # 까지 떨어진다. 0.45는 45분에 이미 12명이 걸리는 문턱이라 다섯 장을 50분 안에 다
    # 써 버렸다(실측 교체 중앙값 68.3분). 궤적상 0.25는 대략 70% 경과에 해당한다.
    "stamina_floor": 0.25,
    # 두 번째·세 번째 동시 교체가 요구하는 추가 피로. 실측 평균 동시 인원 1.40명.
    # 문턱이 0.25로 내려갔으므로 간격도 함께 좁힌다 — 0.08이면 3순위 문턱이 0.09라
    # 사실상 두 명이 상한이 된다.
    "extra_stamina_step": 0.05,
    "fatigue_weight": 1.0,
    "booked_weight": 0.35,
    # 역할 편향의 세기. 0이면 순수 피로 기준(종전 동작)이다.
    "role_bias_weight": 0.25,
    # 전술 교체의 세기. 경기 막바지에 카드를 다 쥐고 있으면 체력 문턱이 이만큼 올라간다.
    # 0이면 순수 피로 교체(종전 동작)이고, 그때는 5장을 다 쓰는 78%가 재현되지 않는다.
    "tactical_gain": 0.45,
    "role_weight": 0.05,
    "speed_weight": 0.4,
    "control_weight": 0.3,
    "max_per_window": 3,
    "hold_seconds": 60.0,
    "urgency_gain": 2.0,
    "chase_gain": 0.6,
    "edge_gain": 0.5,
    "territory_gain": 0.4,
    "width_gain": 0.3,
    "width_fitness_pivot": 0.6,
    "shape_gain": 0.25,
    "incumbent_bonus": 0.35,
}
"""규칙 감독의 계수.

교체 쪽(``earliest_fraction``·``extra_stamina_step``·``role_bias_weight``)은 K리그1
2026 120경기 교체 1,148건에서 **측정**했다. 포메이션 쪽 이득 계수는 아직 사전값이다 —
실측 자료에 경기 중 포메이션 전환이 라벨로 붙어 있지 않아 같은 방식으로 뽑을 수 없다."""


class RuleManager:
    """규칙 감독 — 교체와 포메이션을 **한 뷰**에서 함께 정한다.

    클로저가 아니라 클래스다. 감독은 env 속성이므로 env와 함께 pickle된다 —
    지역 함수를 돌려주면 ``manager="auto"`` env가 통째로 직렬화 불가가 된다
    (실측: ``Can't pickle local object make_rule_manager.<locals>.decide``).
    :class:`ComposedManager`가 같은 이유로 이미 클래스였고, 이쪽만 남아 있었다.
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg)

    def __call__(self, view, key):
        del key
        out_slot, bench_index = _substitution_proposal(view, self.cfg)
        layout = _formation_proposal(view, self.cfg, out_slot)
        return ManagerDecision(out_slot, bench_index, layout)


class IdleManager:
    """아무것도 하지 않는 감독 — 고정 명단·고정 포메이션."""

    def __call__(self, view, key):
        del key
        none = jnp.full((TEAM_COUNT_LOCAL, int(view.max_simultaneous)),
                        NO_PLAYER, jnp.int32)
        return ManagerDecision(none, none, jnp.full((2,), -1, jnp.int32))


def make_rule_manager(**overrides):
    """규칙 감독을 만든다. 알 수 없는 계수 이름은 거부한다."""

    cfg = dict(DEFAULT_RULES)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"unknown manager rule keys: {sorted(unknown)}")
    cfg.update(overrides)
    return RuleManager(cfg)


def make_idle_manager():
    """유휴 감독을 만든다."""

    return IdleManager()


TEAM_COUNT_LOCAL = 2

MANAGERS = {
    "idle": make_idle_manager,
    "auto": make_rule_manager,
}
"""이름으로 고를 수 있는 기본 감독. 사용자 함수는 이름 대신 직접 넘긴다."""


class ComposedManager:
    """두 결정자를 하나로 묶는 어댑터.

    클로저가 아니라 **클래스**여야 한다. 로컬 함수를 감독으로 두면 env가 pickle 불가가
    되어 직렬화 계약이 깨진다(실측: ``Can't pickle local object
    SoccerEnv._build_manager.<locals>.composed``). 감독은 env 속성이므로 env와 함께
    직렬화된다.
    """

    def __init__(self, substitution_decider, formation_decider,
                 to_substitution_view, to_formation_view):
        self.substitution_decider = substitution_decider
        self.formation_decider = formation_decider
        self.to_substitution_view = to_substitution_view
        self.to_formation_view = to_formation_view

    def __call__(self, view, key):
        k_sub, k_form = jax.random.split(key)
        out_slot, bench_index = self.substitution_decider(
            self.to_substitution_view(view), k_sub)
        layout = self.formation_decider(self.to_formation_view(view), k_form)
        return ManagerDecision(out_slot, bench_index, layout)


def resolve(mode):
    """감독 이름 또는 사용자 함수를 결정 함수로 바꾼다."""

    if callable(mode):
        return mode
    if mode in MANAGERS:
        return MANAGERS[mode]()
    raise ValueError(
        f"manager must be callable or one of {sorted(MANAGERS)}, got {mode!r}")
