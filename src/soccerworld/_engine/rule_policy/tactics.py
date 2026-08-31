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
obs의 anchor 절대속도와 선수 토큰 상대속도를 합쳐 타 선수 절대속도(opp_vel)를 복원하는 이유가
여기 있다: '지금 어디 있나'가 아니라 '패스가 도착할 때 어디 있을까'로 차단·압박을 예측한다.
pressure·lane_completion·mark는 전부 opp_vel로 미래를 본다.

반환 형태
---------
질의점 q는 (N,2)(선수당 1점) 또는 (N,K,2)(선수당 K개 후보)를 받는다. 반환은 각각 (N,) 또는 (N,K).
"""
from __future__ import annotations
import jax.numpy as jnp

from ..constants import DIV_EPS
from ..spatial import _safe_norm, _unit


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


def moving_pass_target(
    src,
    receiver_pos,
    receiver_vel,
    offside_line_x,
    hx,
    hy,
    *,
    lead_time_cap=0.90,
    velocity_weight=0.72,
    lead_distance_cap=6.5,
    through_gap_weight=0.55,
    through_shoulder_cue_weight=0.55,
    through_shoulder_gap_m=2.5,
    through_run_ahead=3.5,
    through_min_progress=6.0,
    nominal_ball_speed=18.0,
):
    """움직이는 수신자의 *도착 위치*와 킬패스 강도를 계산한다.

    ``receiver_pos``/``receiver_vel``은 ``(N,M,2)``이고 모든 좌표는 공격 프레임이다.
    패스 순간의 오프사이드는 수신자의 현재 위치로 판정되므로, 이미 온사이드인 선수가
    최종 수비선 뒤 공간으로 달려갈 수 있게 목표점만 앞으로 둔다. 무조건 앞에 차지 않고
    실제 전방 속도와 수비선 앞 여유가 함께 있을 때 추가 lead를 준다. 이미 라인 어깨에
    도착해 감속한 공격수도 패스 순간 시작하는 타이밍 런 의도가 있으므로 좁은 shoulder
    구간에서는 제한된 cue를 더한다. 오프사이드 적법성은 현재 위치로 별도 판정된다.

    반환 ``(target, lead_distance, through_strength)``의 마지막 두 값은 ``(N,M)``이다.
    """

    rel = receiver_pos - src[:, None, :]
    distance = _safe_norm(rel, axis=-1)
    flight_t = jnp.clip(
        distance / max(float(nominal_ball_speed), DIV_EPS), 0.0, lead_time_cap
    )
    motion = receiver_vel * (flight_t * velocity_weight)[:, :, None]
    motion_norm = _safe_norm(motion, axis=-1)
    motion = motion * (
        jnp.minimum(1.0, lead_distance_cap / (motion_norm + DIV_EPS))[:, :, None]
    )

    progress = receiver_pos[:, :, 0] - src[:, None, 0]
    line_gap = jnp.clip(
        offside_line_x[:, None] - receiver_pos[:, :, 0], 0.0, lead_distance_cap
    )
    forward_speed = jnp.clip(receiver_vel[:, :, 0] / 7.0, 0.0, 1.0)
    shoulder_cue = (
        through_shoulder_cue_weight
        * jnp.clip(
            1.0 - line_gap / max(float(through_shoulder_gap_m), DIV_EPS),
            0.0,
            1.0,
        )
    )
    run_intent = jnp.maximum(forward_speed, shoulder_cue)
    progress_gate = _sigmoid((progress - through_min_progress) / 1.5)
    # 러너가 최종선 어깨에 가까워질수록 ``line_gap``이 0이 되어 가장 좋은 타이밍에 오히려
    # 라인 뒤 목표가 사라지던 역설을 막는다. 실제 전진속도와 전진 우위가 있을 때만 최소
    # run-ahead를 보장하므로 정지 선수·횡패스에는 적용되지 않는다.
    through_space = jnp.maximum(
        line_gap * through_gap_weight,
        through_run_ahead,
    )
    through_extra = through_space * progress_gate * run_intent
    target = receiver_pos + motion
    target = target.at[:, :, 0].add(through_extra)
    target = jnp.stack(
        [
            jnp.clip(target[:, :, 0], -hx + 2.0, hx - 1.5),
            jnp.clip(target[:, :, 1], -hy + 1.5, hy - 1.5),
        ],
        axis=-1,
    )
    lead_distance = _safe_norm(target - receiver_pos, axis=-1)
    through_strength = jnp.clip(
        (through_extra / (lead_distance_cap + DIV_EPS))
        * jnp.clip(progress / 20.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    return target, lead_distance, through_strength


def combination_value(receiver_target, support_pos, support_mask, hx, hy):
    """수신 후 2차 패스를 이어갈 구조의 가치 ``[0,1]``.

    첫 패스의 수신 목표 ``(N,M,2)``에서 같은 팀 지원점 ``(N,K,2)``까지 5–26m이고
    지나치게 후퇴하지 않는 연결을 찾는다. 이 값은 패스 자체의 성공률이 아니라 *다음 행동의
    존재 여부*라서 상위 정책에서 작은 보너스로만 사용한다. ``support_mask``는 모든 수신자가
    같은 후보를 쓰는 ``(N,K)`` 또는 자기 자신을 제외한 ``(N,M,K)``를 받을 수 있다.
    """

    rel = support_pos[:, None, :, :] - receiver_target[:, :, None, :]
    distance = _safe_norm(rel, axis=-1)
    forward = rel[:, :, :, 0]
    mask = support_mask[:, None, :] if support_mask.ndim == 2 else support_mask
    legal = (
        mask
        & (distance > 5.0)
        & (distance < 26.0)
        & (forward > -7.0)
    )
    spacing = jnp.exp(-jnp.abs(distance - 13.0) / 8.0)
    direction = jnp.clip((forward + 7.0) / 22.0, 0.0, 1.0)
    continuation = pitch_value(support_pos, hx, hy)[:, None, :]
    score = spacing * (0.45 + 0.55 * direction) * continuation
    return jnp.max(jnp.where(legal, score, 0.0), axis=-1)


def aerial_reception(
    target,
    receiver_pos,
    receiver_vmax,
    flight_t,
    player_pos,
    player_vel,
    player_vmax,
    opponent_mask,
    *,
    arrival_slack=0.35,
    arrival_soft=0.45,
    contest_soft=0.65,
):
    """크로스 목표별 러너 도착·공중 경합 우위 확률 ``[0,1]``.

    ``target``/``receiver_pos``는 ``(N,M,2)``, 전체 타 선수는 ``(N,O,2)``이다. 수신자가
    체공시간 안에 목표점에 도달하는 정도와, 가장 빠른 상대보다 먼저 도달할 여유를 곱한다.
    상대의 현재 속도는 체공시간의 일부(최대 0.6s)만 선점해 이미 닫히는 공간을 반영한다.
    """

    receiver_eta = _safe_norm(target - receiver_pos, axis=-1) / (
        receiver_vmax + DIV_EPS
    )
    lead_t = jnp.clip(0.35 * flight_t, 0.0, 0.6)
    opponent_future = (
        player_pos[:, None, :, :]
        + player_vel[:, None, :, :] * lead_t[:, :, None, None]
    )
    opponent_distance = _safe_norm(
        target[:, :, None, :] - opponent_future, axis=-1
    )
    opponent_eta = opponent_distance / (player_vmax[:, None, :] + DIV_EPS)
    nearest_opp_eta = jnp.min(
        jnp.where(opponent_mask[:, None, :], opponent_eta, jnp.inf), axis=-1
    )
    arrival = _sigmoid(
        (flight_t + arrival_slack - receiver_eta) / arrival_soft
    )
    contest = _sigmoid((nearest_opp_eta - receiver_eta) / contest_soft)
    return arrival * (0.35 + 0.65 * contest)


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


# ── 지표 1b: 접근률(closing) ─────────────────────────────────────────────────
def closing(q, q_vel, opp_pos, opp_vel, opp_mask, tau=12.0):
    """질의점 q로 **좁혀 오는 속도**(m/s) — 압박의 수준이 아니라 변화율이다.

    ``pressure``는 0.35초 선점만 보므로 12 m 밖에서 전력으로 달려오는 수비와
    9.5 m에 서 있는 수비를 거의 같은 값으로 낸다. 그 지표만으로 결정하면 캐리어는
    수비가 **이미 붙은 뒤에야** 반응하고, 그때 낸 패스는 끊긴다(실측: 성공 패스의
    보유 1.20 s 대 차단 패스 2.80 s).

    여기서는 각 상대의 q를 향한 **상대속도 성분**만 취해(멀어지는 상대는 0) 거리로
    감쇠 가중해 합산한다. τ는 압박 반경(5 m)보다 크게 잡는다 — 아직 멀지만 빠르게
    좁혀 오는 주자를 보는 것이 이 지표의 목적이기 때문이다. 반환 (N,) 또는 (N,K).
    """
    qk, single = _prep_q(q)                                          # (N,K,2)
    rel = opp_pos[:, None, :, :] - qk[:, :, None, :]                 # (N,K,O,2)
    d = _safe_norm(rel, axis=-1)                                     # (N,K,O)
    unit = rel / (d[..., None] + DIV_EPS)
    v_rel = opp_vel[:, None, :, :] - qk_vel_expand(q_vel)[:, :, None, :]
    approach = -jnp.sum(v_rel * unit, axis=-1)                       # (N,K,O) 양수=접근
    approach = jnp.maximum(approach, 0.0)
    w = jnp.exp(-d / tau) * opp_mask[:, None, :]
    out = jnp.sum(approach * w, axis=-1)                             # (N,K)
    return out[:, 0] if single else out


def qk_vel_expand(q_vel):
    """질의점 속도 (N,2)→(N,1,2). 질의점이 여러 개여도 속도는 관측자 하나의 것이다."""
    return q_vel[:, None, :] if q_vel.ndim == 2 else q_vel


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
def lane_completion(
    src,
    dst,
    opp_pos,
    opp_vel,
    opp_mask,
    v_ball=20.0,
    r_int=1.25,
    soft=1.1,
    react=0.35,
    opponent_vmax=None,
    defender_accel=0.0,
):
    """src(캐리어)→dst(수신 후보)로 지상 패스가 **상대에게 차단되지 않을 확률** [0,1].

    각 상대에 대해:
      · 레인축 u=unit(dst−src), 길이 L. 상대의 레인 위 최근접 파라미터 s=clip((opp−src)·u, 0, L).
      · 공이 그 지점에 도달하는 시간 t_ball = s / v_ball.
      · 상대의 레인 수직 현재 갭 perp, 그리고 **속도의 레인쪽 성분**(close)으로 t_ball 뒤 예상 갭
        perp_at = perp − max(0, close)·max(0, t_ball−react). (react=반응지연)
      · ``opponent_vmax``가 주어지면 반응 후 현재 접근속도에
        ``0.5·defender_accel·t²``를 더하고 vmax로 상한을 둔다. 정지한
        2차 압박자가 패스를 보고 가속하는 구간을 현재 속도 0으로 고정하지 않는다.
      · 예상 갭이 인터셉트 반경 r_int 안이면 차단 → block = sigmoid((r_int − perp_at)/soft).
    완성확률 = Π(1 − block). src·dst 사이(0<s<L)의 상대만 고려. dst는 (N,2) 또는 (N,M,2).
    ``v_ball``은 스칼라 또는 후보별 ``(N,M,1)`` 평균 공속을 받을 수 있다.
    ``opponent_vmax``는 ``(N,O)``이며 생략하면 이전 등속 계약과 같다. 반환
    (N,) 또는 (N,M). **실제 후보 공속과 수비 도착 물리로 미래 차단을
    보는 것이 이 지표의 핵심.**
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
    response_t = jnp.maximum(t_ball - react, 0.0)
    closing_speed = jnp.maximum(close, 0.0)
    reachable = closing_speed * response_t
    if opponent_vmax is not None:
        opponent_vmax = jnp.asarray(opponent_vmax)
        accelerated = (
            reachable + 0.5 * defender_accel * response_t * response_t
        )
        reachable = jnp.minimum(
            accelerated,
            opponent_vmax[:, None, :] * response_t,
        )
    perp_at = perp - reachable

    between = opp_mask[:, None, :] & (s > 0.5) & (s < L[:, :, None] - 0.3)     # 캐리어·수신자 사이만
    block = _sigmoid((r_int - perp_at) / soft)                    # (N,M,O) 상대별 차단 확률
    block = jnp.where(between, block, 0.0)
    comp = jnp.prod(1.0 - block, axis=-1)                          # (N,M)
    return comp[:, 0] if single else comp


def two_hop_continuation(
    receiver_target,
    support_target,
    support_mask,
    opponent_pos,
    opponent_vel,
    opponent_vmax,
    opponent_mask,
    next_ball_speed,
    hx,
    hy,
    *,
    min_distance=5.0,
    max_distance=24.0,
    backward_tolerance=4.0,
    intercept_radius=1.5,
    reaction_s=0.18,
    defender_accel=5.5,
    shot_value=None,
    shot_gain=0.0,
):
    """수신 후 제3선수로 연결할 수 있는 물리적 2-hop 가치 ``[0,1]``.

    ``receiver_target``은 첫 패스 후보 ``(N,M,2)``, ``support_target``은 각
    후보가 공을 받을 시점의 제3선수 예측점 ``(N,M,K,2)``이다.
    단순히 근처 동료가 있는지를 세지 않고, 두 점 사이의 실제 drive
    평균속도·수비 가속 ETA·출구 순전진·간격을 함께 평가한다.
    """

    n, m, k = support_target.shape[:3]
    opponents = opponent_pos.shape[1]
    src = receiver_target.reshape(n * m, 2)
    dst = support_target.reshape(n * m, k, 2)
    opp_pos = jnp.broadcast_to(
        opponent_pos[:, None, :, :], (n, m, opponents, 2)
    ).reshape(n * m, opponents, 2)
    opp_vel = jnp.broadcast_to(
        opponent_vel[:, None, :, :], (n, m, opponents, 2)
    ).reshape(n * m, opponents, 2)
    # ``support_mask``와 수비 마스크는 서로 다른 축이므로 독립 확장한다.
    defender_mask = jnp.broadcast_to(
        opponent_mask[:, None, :], (n, m, opponents)
    ).reshape(n * m, opponents)
    vmax = jnp.broadcast_to(
        opponent_vmax[:, None, :], (n, m, opponents)
    ).reshape(n * m, opponents)
    speed = next_ball_speed
    if speed.ndim == 3:
        speed = speed[:, :, :, None]
    speed = speed.reshape(n * m, k, 1)
    lane = lane_completion(
        src,
        dst,
        opp_pos,
        opp_vel,
        defender_mask,
        v_ball=speed,
        r_int=intercept_radius,
        react=reaction_s,
        opponent_vmax=vmax,
        defender_accel=defender_accel,
    ).reshape(n, m, k)

    rel = support_target - receiver_target[:, :, None, :]
    distance = _safe_norm(rel, axis=-1)
    forward = rel[:, :, :, 0]
    legal = (
        support_mask
        & (distance >= min_distance)
        & (distance <= max_distance)
        & (forward >= -backward_tolerance)
    )
    spacing_mid = 0.5 * (min_distance + max_distance)
    spacing_scale = 0.5 * (max_distance - min_distance) + DIV_EPS
    spacing = jnp.exp(-jnp.abs(distance - spacing_mid) / spacing_scale)
    progress = jnp.clip(
        (forward + backward_tolerance)
        / (12.0 + backward_tolerance),
        0.0,
        1.0,
    )
    destination = pitch_value(
        support_target.reshape(n * m, k, 2), hx, hy
    ).reshape(n, m, k)
    if shot_value is None:
        shot_value = jnp.zeros_like(destination)
    else:
        shot_value = jnp.asarray(shot_value)
        if shot_value.shape != destination.shape:
            raise ValueError(
                "shot_value must match support_target's (N,M,K) prefix, "
                f"got {shot_value.shape} for {destination.shape}"
            )
    score = lane * spacing * (
        0.20
        + 0.55 * progress
        + 0.25 * jnp.maximum(destination, shot_value)
        + shot_gain * shot_value
    )
    return jnp.max(jnp.where(legal, score, 0.0), axis=-1)


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
