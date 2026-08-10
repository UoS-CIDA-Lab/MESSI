"""
tactics.py — 규칙 정책이 '추상 의사결정'(패스/슛/드리블/전개)을 내리기 위한 **정량 전술 지표** 모음.

설계 철학
---------
정책의 매크로 결정(슛할까·누구에게 패스할까·드리블할까)을 애드혹 가중합이 아니라 **공통 통화**
(≈ 팀의 득점 기여 기대값)로 비교하려면, 각 선택지의 가치를 재는 재사용 가능한 지표가 필요하다.
이 모듈이 그 지표를 단일 진실원천으로 제공한다. policy.py는 여기서 값을 받아 argmax만 한다.

좌표 규약 (전 함수 공통)
------------------------
모든 입력은 **관측자 자기 공격 프레임(미터)** 이다 — get_obs 폴딩과 동일. 따라서:
  · 상대 골대 G = [+hx, 0], 우리 골대 = [-hx, 0], +x = 전방(전진).
  · 배치 축 0은 항상 관측자(에이전트) N개. 상대/동료 축은 O = N-1.
각 에이전트가 '자기 obs'만으로 자기 프레임에서 값을 재므로 분산적·일관적이다(월드 좌표 불필요).

핵심 신호 — **상대 속도(opp_vel)로 선점(anticipation)**
-----------------------------------------------------
obs가 타 선수 절대속도(abs_vel)를 주는 이유가 여기 있다: '지금 어디 있나'가 아니라 '패스가 도착할
때 어디 있을까'로 차단·압박을 예측한다. pressure·lane_completion·mark는 전부 opp_vel로 미래를 본다.

반환 형태
---------
질의점 q는 (N,2)(선수당 1점) 또는 (N,K,2)(선수당 K개 후보)를 받는다. 반환은 각각 (N,) 또는 (N,K).
"""
from __future__ import annotations
import jax.numpy as jnp

from constants import DIV_EPS
from spatial import _safe_norm, _unit


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _prep_q(q):
    """q (N,2)→(N,1,2)[single] 또는 (N,K,2)→그대로. ndim은 트레이스타임 정적이라 파이썬 분기 안전."""
    if q.ndim == 2:
        return q[:, None, :], True
    return q, False


def _future(pos, vel, t):
    """등속 선점 위치 pos + vel·t. t는 스칼라 또는 브로드캐스트 가능 배열."""
    return pos + vel * t


def _sigmoid(x):
    """수치안정 시그모이드(자기완결 — jax.nn 의존 회피)."""
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -30.0, 30.0)))


# ── 지표 1: 압박도(pressure) ─────────────────────────────────────────────────
def pressure(q, opp_pos, opp_vel, opp_mask, tau=5.0, lead=0.35):
    """질의점 q에 대한 **상대 압박도** — 가까운·다가오는 상대일수록 크다.

    상대를 lead초 선점(opp+vel·lead)한 뒤 q까지 거리로 exp(-d/τ) 가중 합. τ=5m는 '압박 반경' 스케일.
    여러 상대가 겹치면 합산되어 협공(더블팀)이 자연히 큰 값이 된다. 반환 (N,) 또는 (N,K).
    """
    qk, single = _prep_q(q)                                         # (N,K,2)
    of = _future(opp_pos, opp_vel, lead)                            # (N,O,2)
    d = _safe_norm(qk[:, :, None, :] - of[:, None, :, :], axis=-1)  # (N,K,O)
    w = jnp.exp(-d / tau) * opp_mask[:, None, :]
    p = jnp.sum(w, axis=-1)                                         # (N,K)
    return p[:, 0] if single else p


# ── 지표 2: 공간(openness) ───────────────────────────────────────────────────
def openness(q, opp_pos, opp_vel, opp_mask, lead=0.3, cap=15.0):
    """질의점 q의 **여유 공간** = 선점한 최근접 상대까지 거리(캡). 클수록 열려 있음. (N,) 또는 (N,K)."""
    qk, single = _prep_q(q)
    of = _future(opp_pos, opp_vel, lead)
    d = _safe_norm(qk[:, :, None, :] - of[:, None, :, :], axis=-1)  # (N,K,O)
    d = jnp.where(opp_mask[:, None, :], d, jnp.inf)
    nn = jnp.clip(jnp.min(d, axis=-1), 0.0, cap)                    # (N,K)
    return nn[:, 0] if single else nn


# ── 지표 3: 패스 성공확률(lane_completion) — 상대 속도 인터셉트 예측 ──────────
def lane_completion(src, dst, opp_pos, opp_vel, opp_mask,
                    v_ball=20.0, r_int=1.25, soft=1.1, react=0.35):
    """src(캐리어)→dst(수신 후보)로 지상 패스가 **상대에게 차단되지 않을 확률** [0,1].

    각 상대에 대해:
      · 레인축 u=unit(dst−src), 길이 L. 상대의 레인 위 최근접 파라미터 s=clip((opp−src)·u, 0, L).
      · 공이 그 지점에 도달하는 시간 t_ball = s / v_ball.
      · 상대의 레인 수직 현재 갭 perp, 그리고 **속도의 레인쪽 성분**(close)으로 t_ball 뒤 예상 갭
        perp_at = perp − max(0, close)·max(0, t_ball−react). (react=반응지연)
      · 예상 갭이 인터셉트 반경 r_int 안이면 차단 → block = sigmoid((r_int − perp_at)/soft).
    완성확률 = Π(1 − block). src·dst 사이(0<s<L)의 상대만 고려. dst는 (N,2) 또는 (N,M,2).
    반환 (N,) 또는 (N,M). **opp_vel로 미래 차단을 보는 게 이 지표의 핵심.**
    """
    d2, single = _prep_q(dst)                                       # (N,M,2)
    rel = d2 - src[:, None, :]                                      # (N,M,2)
    L = _safe_norm(rel, axis=-1)                                    # (N,M)
    u = rel / (L[:, :, None] + DIV_EPS)                            # (N,M,2) 레인 단위방향

    o_rel = opp_pos[:, None, :, :] - src[:, None, None, :]          # (N,1,O,2) src기준 상대
    # 각 (M,O): 레인축 투영 s, 수직 성분
    s = jnp.einsum('nmod,nmd->nmo', o_rel, u)                       # (N,M,O) 레인 위 위치
    s_cl = jnp.clip(s, 0.0, L[:, :, None])
    lane_pt = src[:, None, None, :] + u[:, :, None, :] * s_cl[:, :, :, None]   # (N,M,O,2)
    gap_vec = opp_pos[:, None, :, :] - lane_pt                      # (N,M,O,2) 상대→레인점
    perp = _safe_norm(gap_vec, axis=-1)                            # (N,M,O) 현재 수직 갭

    t_ball = s_cl / v_ball                                          # (N,M,O)
    perp_dir = gap_vec / (perp[:, :, :, None] + DIV_EPS)          # 레인에서 상대로
    close = -jnp.einsum('nmod,nmod->nmo', opp_vel[:, None, :, :], perp_dir)   # 레인쪽 접근속도(+)
    perp_at = perp - jnp.maximum(close, 0.0) * jnp.maximum(t_ball - react, 0.0)

    between = opp_mask[:, None, :] & (s > 0.5) & (s < L[:, :, None] - 0.3)     # 캐리어·수신자 사이만
    block = _sigmoid((r_int - perp_at) / soft)                    # (N,M,O) 상대별 차단 확률
    block = jnp.where(between, block, 0.0)
    comp = jnp.prod(1.0 - block, axis=-1)                          # (N,M)
    return comp[:, 0] if single else comp


# ── 지표 4: 슛 기대값(shot_xg) ───────────────────────────────────────────────
def shot_xg(src, opp_pos, opp_mask, o_gk_mask, hx, goal_w,
            b0=1.6, b_ang=1.7, b_dist=0.09, b_obs=1.3, b_gk=1.1):
    """위치 src에서의 **슛 득점 기대값** [0,1]. 거리·골문 각도·차폐·GK 위치를 로지스틱으로 결합.

    다른 지표와 달리 상대 속도를 쓰지 않는다 — 슛은 즉발이라 차폐 판정에 선점(anticipation)이
    들어갈 자리가 없다(패스 레인과의 차이).

      · 골문 각도 ang: 두 골포스트 [hx,±goal_w/2]를 잇는 시야각(넓을수록↑).
      · 거리 d: 멀수록↓.
      · 차폐 obs: src→골 원뿔(콘) 안 상대 수(가까울수록 무겁게). GK는 별도 항 b_gk.
    xg = sigmoid(b0 + b_ang·ang − b_dist·d − b_obs·obs − b_gk·gk_block).
    src는 (N,2)(선수당 1점) 또는 (N,K,2)(선수당 K 후보) → 반환 (N,) 또는 (N,K)."""
    q, single = _prep_q(src)                                       # (N,K,2)
    G = jnp.array([hx, 0.0])
    to_goal = G[None, None, :] - q                                # (N,K,2)
    d = _safe_norm(to_goal, axis=-1)                             # (N,K)
    postL = jnp.array([hx, goal_w * 0.5])
    postR = jnp.array([hx, -goal_w * 0.5])
    aL = _unit(postL[None, None, :] - q)
    aR = _unit(postR[None, None, :] - q)
    ang = jnp.arccos(jnp.clip(jnp.sum(aL * aR, axis=-1), -1.0, 1.0))  # (N,K) 골문 시야각

    gdir = to_goal / (d[:, :, None] + DIV_EPS)                    # (N,K,2) 슛 방향
    rel = opp_pos[:, None, :, :] - q[:, :, None, :]               # (N,K,O,2)
    along = jnp.einsum('nkod,nkd->nko', rel, gdir)               # (N,K,O) 슛방향 투영
    perp = _safe_norm(rel - along[:, :, :, None] * gdir[:, :, None, :], axis=-1)  # (N,K,O)
    in_cone = opp_mask[:, None, :] & (along > 0.3) & (along < d[:, :, None]) & (perp < 2.2)
    w_near = jnp.clip(
        1.0 - along / (d[:, :, None] + DIV_EPS), 0.0, 1.0
    )                                                             # 가까운 차폐 무겁게
    obs = jnp.sum(jnp.where(in_cone & (~o_gk_mask[:, None, :]), w_near, 0.0), axis=-1)   # (N,K)
    gk_block = jnp.sum(jnp.where(in_cone & o_gk_mask[:, None, :], w_near, 0.0), axis=-1)
    z = b0 + b_ang * ang - b_dist * d - b_obs * obs - b_gk * gk_block
    xg = _sigmoid(z)
    return xg[:, 0] if single else xg


# ── 지표 5: 피치 위협 표면(pitch_value) ──────────────────────────────────────
def pitch_value(q, hx, hy):
    """위치 q의 **정적 위협값** [0,1] — 공을 그 위치에 두는 것 자체의 가치(상대 무관, 전개 가치).

    전진도 adv=(x+hx)/(2hx)를 볼록(^1.35)하게 키우고 중앙·파이널서드 가중. 골 근처 램프 추가.
    xg가 근거리 슛 위협을 담고 pitch_value가 빌드업 가치를 담아, 상위 로직에서 max로 합친다.
    """
    qk, single = _prep_q(q)                                        # (N,K,2)
    x, y = qk[:, :, 0], qk[:, :, 1]
    adv = jnp.clip((x + hx) / (2.0 * hx), 0.0, 1.0)
    central = 1.0 - jnp.clip(jnp.abs(y) / hy, 0.0, 1.0)
    base = adv ** 1.35 * (0.72 + 0.28 * central)
    # 파이널서드(전진 0.72↑) 추가 위협 램프
    final_third = jnp.clip((adv - 0.72) / 0.28, 0.0, 1.0)
    v = jnp.clip(0.08 + 0.72 * base + 0.20 * final_third * (0.5 + 0.5 * central), 0.0, 1.0)
    return v[:, 0] if single else v


# 위협 통화(threat) = max(슛 가치, pitch_value) — 상위 결정의 공통 단위다. 별도 함수로 두지
# 않는 이유는 policy.py가 슛 항에 `RulePolicy.shoot_gain`을 곱한 값으로 max를 취하기 때문이다
# (드리블 항은 GAIN 없이 결합 — 비대칭이 의도된 설계라 여기서 일괄 정의할 수 없다).
