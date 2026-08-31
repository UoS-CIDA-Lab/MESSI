"""교체 결정 — 언제 누구를 바꿀 것인가.

이 모듈은 **결정**만 한다. 적용(위치 투영·충돌 회피·identity 전이)은 ``env.project_substitution``
이 이미 하고 있고 그대로 재사용한다. 둘을 나누는 이유는 결정 방식이 여러 가지이기 때문이다.

  ``schedule``  실측 재현 — 관측된 교체를 그 시각 이후 첫 데드볼에 적용한다
  ``auto``      내장 규칙 — 피로·카드·시간대를 보고 스스로 고른다
  ``external``  호출자가 정한다 — env는 스스로 제안하지 않는다(학습되는 교체 정책)
  사용자 함수    아래 :func:`decide` 규약만 맞추면 그대로 꽂힌다

**교체 축과 행동 축은 독립이다.** 어느 조합이든 설정만으로 만들어진다:

  행동 = 신경망, 교체 = 규칙   ``substitution_mode="auto"``, 교체 제안 주입 안 함
  행동 = 규칙,   교체 = 학습   ``substitution_mode="external"`` + ``step(..., substitution=...)``
  둘 다 학습                  ``"external"`` + 주입(교체 헤드가 행동 헤드와 같은 obs를 본다)
  둘 다 규칙                  ``"auto"`` + 규칙 기반 행동 정책

주입이 있으면 주입이 모드를 이긴다 — 제안은 행동과 같은 **매 스텝 데이터**이고 모드는
주입이 없을 때의 기본값이다. 그래서 ``"auto"``로 두고 일부 스텝만 주입하면 나머지 스텝은
내장 규칙이 바꿔 버린다. 학습이 교체를 온전히 소유해야 하면 ``"external"``을 쓸 것.

[IFAB Law 3] 교체는 **경기 중단 시에만** 가능하다. 그래서 어떤 결정자가 무엇을 내든 적용
단계에서 데드볼 여부·잔여 인원·GK 대응을 다시 확인한다. 타인이 만든 알고리즘이 규칙을
어겨도 환경이 깨지지 않아야 하기 때문이다 — 결정자는 **제안**하고 환경이 **승인**한다.

결정자 규약
-----------
    decide(view, key) -> (out_slot, bench_index)      # 각 (2, K) 또는 (2,)

``view``는 :class:`SubstitutionView`이고 전부 관측에서 얻을 수 있는 값이다. 특권 정보를
넣지 않는 이유는 BC/RL 정책과 같은 정보 경계를 공유해야 학습된 교체 정책이 배포 시에도
같은 입력을 받기 때문이다.

반환은 팀별 배열 두 개다. [IFAB Law 3] 한 정지에서 여러 명을 함께 바꿀 수 있으므로 폭은
``(2, K)``이고 ``K = view.max_simultaneous``다. 한 명만 바꾸는 흔한 경우를 위해 ``(2,)``도
받는다 — 그때는 첫 자리만 쓴다. ``out_slot``이 :data:`NO_PLAYER`면 그 자리는 비운다.
JAX 순수 함수여야 하고 shape는 고정이다.

여러 명을 **한 호출에** 내는 것과 연속 tick으로 나눠 내는 것은 다르다. 트리플 교체의
의도는 원자적이라, 나눠 내면 도중에 공이 살아났을 때 남은 교체가 사라진다.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .constants import (
    IFAB_MAX_SUBSTITUTION_WINDOWS, NO_PLAYER, TEAM_0, TEAM_1,
)


class SubstitutionView(NamedTuple):
    """결정자에게 주는 입력 — 전부 관측 가능한 값이다.

    특권 정보(상대 벤치 상세, 내부 난수 상태 등)는 넣지 않는다. 결정자를 바꿔 끼울 때
    입력 계약이 흔들리지 않도록 필드를 늘릴 때도 이 원칙을 지킨다.

    ``half_time``이 여기 있는 이유는 **제안자와 승인자가 같은 법을 봐야** 하기 때문이다.
    [IFAB Law 3]에서 하프타임 교체는 3회 기회에 포함되지 않는데, 그 면제는 환경의 승인
    단계에만 있었다. 결정자는 ``sub_windows_used < 3``만 보고 있었으므로 기회를 다 쓴 팀은
    하프타임에도 제안조차 하지 않았다 — 환경은 승인했을 교체를 아무도 요청하지 않는다.
    """

    t: jnp.ndarray                 # () 현재 control tick
    game_duration: jnp.ndarray     # () 전체 길이(tick)
    ball_dead: jnp.ndarray         # () 지금 교체가 법적으로 가능한가
    team_id: jnp.ndarray           # (N,)
    active_player: jnp.ndarray     # (N,)
    is_gk: jnp.ndarray             # (N,)
    stamina_long: jnp.ndarray      # (N,)
    stamina_short: jnp.ndarray     # (N,)
    yellow_cards: jnp.ndarray      # (N,)
    score: jnp.ndarray             # (2,)
    bench_player_id: jnp.ndarray   # (2, B)  NO_PLAYER = 없음/투입됨
    bench_is_gk: jnp.ndarray       # (2, B)
    subs_remaining: jnp.ndarray    # (2,)
    sub_windows_used: jnp.ndarray  # (2,)
    sub_window_open_t: jnp.ndarray # (2,) 이번 정지에서 이미 교체했는가(-1이면 아직)
    half_time: jnp.ndarray         # () 지금이 하프타임 정지인가
    max_simultaneous: int          # 한 정지에서 함께 바꿀 수 있는 인원(반환 폭)


def no_substitution(view, key):
    """아무것도 바꾸지 않는 결정자 — ``schedule`` 모드의 기본값이다."""

    del key
    none = jnp.full((2, view.max_simultaneous), NO_PLAYER, jnp.int32)
    return none, none


class FatigueRule:
    """내장 규칙 결정자 — 지친 선수를 후반에 뺀다.

    세 가지를 본다. 모두 관측에 있는 값이다.

      * 장기 stamina가 ``long_stamina_floor`` 아래인 필드 선수
      * 경고를 받은 선수는 더 이른 문턱(``booked_stamina_floor``)에서 교체 — 2차 경고
        퇴장은 팀 인원을 줄이므로 피로보다 비용이 크다
      * ``earliest_fraction`` 이전에는 교체하지 않는다 — 실제 감독도 초반에는 부상이
        아니면 카드를 쓰지 않는다

    GK는 이 규칙으로 교체하지 않는다. GK 교체는 부상·전술 등 이 규칙이 보는 신호로
    설명되지 않고, 잘못 빼면 팀이 GK 없이 남는다.

    ``max_per_window``는 **규칙의 절제**이지 법이 아니다. [IFAB Law 3]은 한 정지에 몇 명을
    바꾸든 허용하므로 env는 막지 않는다 — 다만 첫 정지에 카드를 전부 털어 넣는 감독은 없다.
    법은 환경이, 취향은 결정자가 강제한다는 분리를 여기서도 지킨다.

    상한이 **정지당**으로 성립하려면 이미 이번 정지에서 바꿨는지를 봐야 한다. 그러지 않으면
    창이 열려 있는 동안 매 tick 다시 발동해 결국 벤치를 비운다(실측: 3명을 넣은 다음 tick에
    남은 2명까지 들어가 잔여가 0이 됐다). ``sub_window_open_t``가 그 사실을 준다.
    """

    def __init__(self, *, long_stamina_floor=0.45, earliest_fraction=0.55,
                 booked_stamina_floor=0.60, max_per_window=3):
        self.long_stamina_floor = float(long_stamina_floor)
        self.earliest_fraction = float(earliest_fraction)
        self.booked_stamina_floor = float(booked_stamina_floor)
        self.max_per_window = int(max_per_window)

    def __call__(self, view, key):
        del key
        long_stamina_floor = self.long_stamina_floor
        earliest_fraction = self.earliest_fraction
        booked_stamina_floor = self.booked_stamina_floor
        width = int(view.max_simultaneous)
        take = max(1, min(self.max_per_window, width))
        # 비율을 만들어 비교하면 XLA의 역수 곱셈 반올림 때문에 정확한 경계 tick에서
        # 1 ULP 차이로 거짓이 된다(t=300, duration=600에서 0.49999997). tick 공간에서
        # 비교한다 — :mod:`manager`의 같은 게이트와 같은 규약이다.
        window_open = view.t.astype(jnp.float32) >= (
            jnp.float32(earliest_fraction)
            * view.game_duration.astype(jnp.float32))

        floor = jnp.where(
            view.yellow_cards > 0, booked_stamina_floor, long_stamina_floor
        )
        tired = (
            view.active_player
            & (~view.is_gk)
            & (view.stamina_long < floor)
        )
        # 가장 지친 순서로 고른다. 동률이면 낮은 슬롯 — 결정론적이어야 재현된다.
        cost = jnp.where(tired, view.stamina_long, jnp.inf)
        rank = jnp.arange(width, dtype=jnp.int32)

        # 반환 폭 K가 로스터나 벤치보다 클 수 있다(2v2에 K=5). 잘라내면 길이가 K보다 짧아져
        # ``rank``와 브로드캐스트가 깨지므로, 마지막 인덱스로 채우고 **유효 범위 밖은
        # 마스크로 무효화**한다.
        n_slots = view.team_id.shape[0]
        n_bench = view.bench_player_id.shape[1]
        if n_bench == 0:
            # 벤치가 없으면 교체 자체가 불가능하다. 빈 축에 gather를 걸면 죽는다 —
            # ``bench=None`` + ``substitution_mode="auto"`` 조합에서 첫 호출에 터졌다.
            # :func:`manager._substitution_proposal`이 이미 같은 방어를 갖고 있고,
            # 호환 경로인 이쪽만 빠져 있었다.
            none = jnp.full((2, width), NO_PLAYER, jnp.int32)
            return none, none
        slot_take = jnp.minimum(rank, max(n_slots - 1, 0))
        bench_take = jnp.minimum(rank, max(n_bench - 1, 0))

        out_slots, bench_idx = [], []
        for team in (TEAM_0, TEAM_1):
            mine = view.team_id == team
            team_cost = jnp.where(mine, cost, jnp.inf)
            picks = jnp.argsort(team_cost)[slot_take].astype(jnp.int32)
            picked_ok = jnp.isfinite(team_cost[picks]) & (rank < n_slots)

            # 벤치에서 필드 선수를 고른다(GK는 GK 자리에만 들어간다). 사용 가능한 자리를
            # 앞으로 모아 순서대로 배정한다 — 같은 자리를 두 번 쓰지 않는다.
            avail = (view.bench_player_id[team] >= 0) & (~view.bench_is_gk[team])
            bench_order = jnp.argsort(~avail)[bench_take].astype(jnp.int32)
            bench_ok = avail[bench_order] & (rank < n_bench)

            allowed = (
                window_open
                & view.ball_dead
                & picked_ok
                & bench_ok
                & (rank < take)
                & (rank < view.subs_remaining[team])
                # 이번 정지에서 아직 아무도 안 바꿨을 때만 — 한 정지에 한 번 결정한다.
                & (view.sub_window_open_t[team] < 0)
                # [IFAB Law 3] 하프타임 교체는 3회 기회에 포함되지 않는다. 면제가
                # 승인부에만 있으면 기회를 다 쓴 팀이 하프타임에 제안조차 하지 않아,
                # 환경이 승인했을 교체를 아무도 요청하지 않게 된다.
                & ((view.sub_windows_used[team] < IFAB_MAX_SUBSTITUTION_WINDOWS)
                   | view.half_time)
            )
            out_slots.append(jnp.where(allowed, picks, jnp.int32(NO_PLAYER)))
            bench_idx.append(jnp.where(allowed, bench_order, jnp.int32(NO_PLAYER)))
        return jnp.stack(out_slots), jnp.stack(bench_idx)


def make_fatigue_rule(**overrides):
    """피로 규칙 결정자를 만든다.

    클래스 인스턴스를 돌려준다 — 결정자는 env 속성이라 env와 함께 pickle되고,
    지역 함수를 돌려주면 ``substitution_mode="auto"`` env가 직렬화 불가가 된다.
    """

    return FatigueRule(**overrides)


DECIDERS = {
    "schedule": no_substitution,
    "auto": make_fatigue_rule(),
    "external": no_substitution,
}
"""이름으로 고를 수 있는 기본 결정자. 사용자 함수는 이름 대신 직접 넘긴다.

``schedule``과 ``external``은 둘 다 스스로 제안하지 않지만 뜻이 다르다. ``schedule``은
"이 경기의 교체는 대본에 있다"이고 ``external``은 "교체는 호출자(학습 정책)가 정한다"이다.
이름이 갈려 있어야 주입을 빠뜨렸을 때 의도가 드러난다."""


def resolve(mode):
    """모드 이름 또는 사용자 함수를 결정자로 바꾼다."""

    if callable(mode):
        return mode
    if mode in DECIDERS:
        return DECIDERS[mode]
    raise ValueError(
        f"substitution mode must be callable or one of {sorted(DECIDERS)}, got {mode!r}"
    )
