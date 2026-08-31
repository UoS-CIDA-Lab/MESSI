"""장기 소크 — 40종 불변식을 jit/scan 안에서 검사해 수십만 프레임을 덮는다.

    python -m soccerworld._engine.soak                         # 11개 config · 180,000 프레임
    python -m soccerworld._engine.soak --scale 0.05            # 각 시나리오 5% 빠른 점검
    python -m soccerworld._engine.soak --scenario "rule policy C"
    python -m soccerworld._engine.soak --keep-backend          # 호출 환경의 JAX backend 사용

호스트로 상태를 꺼내 검사하면 프레임마다 device sync가 걸려 수천 프레임이 한계다. 검사 자체를
scan 안에 넣고 위반 카운터만 누적하면 90분 경기 전체를 여러 config로 돌릴 수 있다.

직접 실행의 기본 backend는 재현성과 GPU 메모리 격리를 위해 CPU다. GPU 등 호출 환경의 backend를
의도적으로 사용하려면 ``--keep-backend``를 명시한다. float32와 Threefry 설정은 backend와 별개인
동역학 계약이므로 어느 경우에도 고정한다.
"""

import argparse
import math
import os
import sys
import time


def _prepare_standalone_environment(keep_backend):
    """직접 실행에서 JAX를 import하기 전에 감사 실행 계약을 고정한다."""

    os.environ["JAX_ENABLE_X64"] = "0"
    os.environ["JAX_THREEFRY_PARTITIONABLE"] = "0"
    if not keep_backend:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"


# JAX는 import/첫 초기화 뒤에는 backend를 바꿀 수 없다. 정식 argparse 검증은 main이 맡되,
# 직접 실행에 필요한 이 한 플래그만 먼저 읽어 GPU가 초기화되기 전에 기본 CPU 계약을 적용한다.
if __name__ == "__main__":
    _prepare_standalone_environment("--keep-backend" in sys.argv[1:])

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .config import Agent, Foul, Reward, Substitution
from .constants import (
    BALL_ALIVE, BALL_DEAD, BALL_EVENT_COUNT, BALL_EVENT_NONE,
    DEPARTED_TAKER, DIM_X, FOUL_NONE, GEOMETRY_EPS,
    GK_HANDLING_RESTRICTION_COUNT, NO_PLAYER, NO_TEAM, RESTART_COUNT, RK_GK_HOLD,
    RK_NONE, RK_PENALTY, TEAM_0, TEAM_1, TOUCH_COUNT, WOODWORK_COUNT,
    WOODWORK_NONE, YELLOW_CARD_SEND_OFF_COUNT,
)
# ``python -m ...soak --help`` must stay a genuinely light bootstrap.  Importing
# SoccerEnv pulls in JaxMARL, whose module-level arrays initialize a backend
# before argparse can exit; ``--keep-backend`` then fails on CPU-only hosts with
# an installed CUDA plugin.  Library imports still receive the normal eager
# symbols, while the standalone path loads them in ``main`` only after parsing.
if __name__ != "__main__":
    from .env import SoccerEnv
    from .restart import restart_timer_active

CHECKS = (
    "finite_players", "finite_ball", "finite_energy",
    "team_domain", "kickoff_domain", "kind_domain", "touch_code_domain",
    "taker_domain", "dead_iff_timer", "timer_window", "kind_when_timer",
    "taker_team", "taker_active", "dead_ball_at_rest",
    "player_bounds", "speed_cap", "ball_z", "stamina_range",
    "cooldown_range", "contact_range", "ctrl_range", "yellow_range", "pass_range",
    "sent_off_off_pitch", "offside_team", "offside_window",
    "score_monotone", "sent_off_monotone", "time_step", "role_count_nonneg",
    "goal_recentres_ball", "ball_runaway", "role_count_monotone",
    "no_deep_overlap",          # 활성 선수 간 최소거리(2·r_player) 침투 1cm(분리기 사양) 초과 금지
    "no_encroacher_at_rest",    # 진입 때부터 살아 있던 재개는 프레임 끝에 침범자가 0이어야 한다
    "terminal_state_identity",  # terminal 진입 상태의 다음 State는 모든 leaf가 항등이어야 한다
    "role_pos_in_field",        # role anchor는 필드 경계 안의 좌표 평균이어야 한다
    "slot_generation_monotone", # 슬롯 세대는 되돌아가지 않는다
    "foul_latch_coherent",      # FOUL_NONE이면 actor/victim도 둘 다 NO_PLAYER
    "pass_latch_coherent",      # active phase iff valid team and at least one flag
    "ball_event_coherent",      # 라인 통과 칸은 kind/control_t/team이 함께 차거나 함께 빈다
    "woodwork_coherent",        # 프레임 충돌 칸은 kind/control_t가 함께 차고, 접촉점이 골라인 띠 안이다
    "gk_hold_is_live",          # [IFAB Law 9] GK 홀드는 타이머가 돌아도 공이 인플레이다
    "ball_state_domain",        # ball_state는 ALIVE/DEAD 둘 중 하나여야 한다
    "finite_scalars",           # cooldown·vmax·role_pos_count가 NaN이면 범위 검사가 통째로 샌다
)


def _unexplained_role_count_decrease(prev, state):
    """포메이션·identity·period 경계로 설명되지 않는 role epoch 감소 여부."""

    formation_epoch = (
        (state.layout_index[state.team_id]
         != prev.layout_index[state.team_id])
        & (state.layout_since_t[state.team_id] == state.t)
    )
    return jnp.any(
        (state.role_pos_count < prev.role_pos_count)
        & (state.attack_dir == prev.attack_dir)
        & (state.slot_generation == prev.slot_generation)
        & (~formation_epoch)
    )


def build(env, policy=None):
    e = env.e_cfg
    N = env.N
    bound_x = env.hx + e.player_boundary_margin
    bound_y = env.hy + e.player_boundary_margin
    max_cd = e.cooldown_substeps + e.challenge_cooldown_extra

    def violations(prev, s):
        active = s.active_player
        pos, vel = s.player_pos, s.player_vel
        rt = s.restart_t
        dead = s.ball_state == BALL_DEAD
        movement_margin = jnp.where(dead | (rt > 0),
                                    e.player_boundary_margin,
                                    e.r_player)
        movement_bound_x = env.hx + movement_margin
        movement_bound_y = env.hy + movement_margin
        is_hold = s.restart_kind == RK_GK_HOLD
        window = jnp.where(s.restart_kind == RK_PENALTY, e.penalty_substeps,
                  jnp.where(s.restart_kind == RK_GK_HOLD, e.gk_hold_substeps,
                            e.restart_substeps))
        taker = s.pending_taker
        safe_taker = jnp.clip(taker, 0, N - 1)
        has_taker = (taker >= 0) & (rt > 0)
        speed = jnp.linalg.norm(vel, axis=1)
        cap = env.effective_vmax(s.vmax, s.stamina_long, s.stamina_short)

        def team_ok(v):
            return (v == NO_TEAM) | (v == TEAM_0) | (v == TEAM_1)

        flagged = s.offside_flag
        terminal_changed = jnp.any(jnp.stack([
            jnp.any(before != after)
            for before, after in zip(
                jax.tree_util.tree_leaves(prev), jax.tree_util.tree_leaves(s)
            )
        ]))
        out = [
            ~(jnp.isfinite(pos).all() & jnp.isfinite(vel).all()
              & jnp.isfinite(s.player_facing).all() & jnp.isfinite(s.role_pos).all()),
            ~(jnp.isfinite(s.ball_pos).all() & jnp.isfinite(s.ball_vel).all()
              & jnp.isfinite(s.ball_spin).all()),
            ~(jnp.isfinite(s.stamina_long).all()
              & jnp.isfinite(s.endurance_factor).all()
              & jnp.isfinite(s.stamina_short).all()),
            ~(team_ok(s.poss_team) & team_ok(s.previous_poss_team)
              & team_ok(s.last_touch_team)
              & ((s.gk_handling_restricted_team == NO_TEAM)
                 | ((s.gk_handling_restricted_team >= 0)
                    & (s.gk_handling_restricted_team
                       < GK_HANDLING_RESTRICTION_COUNT)))
              & team_ok(s.restart_team) & team_ok(s.pass_team)),
            ~((s.kickoff_team == TEAM_0) | (s.kickoff_team == TEAM_1)),
            ~((s.restart_kind >= 0) & (s.restart_kind < RESTART_COUNT)),
            ~((s.last_touch_code >= 0) & (s.last_touch_code < TOUCH_COUNT)),
            ~(((taker == NO_PLAYER) | ((taker >= 0) & (taker < N)))
              & ((s.setpiece_taker == NO_PLAYER)
                 | (s.setpiece_taker == DEPARTED_TAKER)
                 | ((s.setpiece_taker >= 0) & (s.setpiece_taker < N)))
              & ((s.throw_taker == NO_PLAYER)
                 | (s.throw_taker == DEPARTED_TAKER)
                 | ((s.throw_taker >= 0) & (s.throw_taker < N)))),
            # GK 캐치 홀드는 ``restart_t``를 타이머로 쓰지만 [IFAB Law 9]상 공은 인플레이다.
            # 그 한 종류만 '타이머 활성 ⟺ 데드' 대응에서 뺀다 — 다만 **빼기만 하면**
            # "홀드는 반드시 인플레이"라는 그 계약 자체를 아무도 검사하지 않는다.
            # 아래 ``gk_hold_is_live``가 그 몫을 맡는다.
            ((rt > 0) != dead) & (~is_hold),
            (rt > 0) & (rt > window),
            (rt > 0) & (s.restart_kind == RK_NONE),
            has_taker & (s.team_id[safe_taker] != s.restart_team),
            has_taker & (~active[safe_taker]),
            dead & (jnp.linalg.norm(s.ball_vel) > 1e-3),
            jnp.any(active & (
                (jnp.abs(pos[:, 0]) > movement_bound_x + 1e-3)
                | (jnp.abs(pos[:, 1]) > movement_bound_y + 1e-3)
            )),
            jnp.any(active & (speed > cap + 1e-3)),
            s.ball_pos[2] < env.r_ball - 1e-3,
            jnp.any(
                (s.stamina_long < -1e-6) | (s.stamina_long > 1 + 1e-6)
                | (s.stamina_short < -1e-6) | (s.stamina_short > 1 + 1e-6)
            ),
            jnp.any((s.cooldown < -1e-6) | (s.cooldown > max_cd + 1e-6)),
            jnp.any((s.contact_lock_t < 0) | (s.contact_lock_t > e.contact_lock_substeps)),
            jnp.any((s.aerial_recovery_t < 0)
                    | (s.aerial_recovery_t > e.aerial_attempt_lock_substeps)),
            jnp.any((s.ctrl_lock_t < 0) | (s.ctrl_lock_t > e.ctrl_lock_substeps)),
            jnp.any((s.endurance_factor < 0.01) | (s.endurance_factor > 100.0)),
            jnp.any((s.yellow_cards < 0) | (s.yellow_cards > YELLOW_CARD_SEND_OFF_COUNT)),
            (s.pass_t < 0) | (s.pass_t > 1),
            jnp.any(s.sent_off & s.on_pitch),
            jnp.any(flagged & (s.team_id != s.pass_team)),
            jnp.any(flagged) & (s.pass_t <= 0),
            jnp.any(s.score < prev.score),
            jnp.any(prev.sent_off & (~s.sent_off)),
            # terminal freeze — 종료된 상태의 step은 항등이라 t가 늘지 않는다. 판정은 env의
            # SSOT를 그대로 쓴다(시간제한 ∨ 최소 인원 미달).
            jnp.where(env._is_terminal(prev), s.t != prev.t, s.t != prev.t + 1),
            jnp.any(s.role_pos_count < 0),
            # 득점하면 공이 센터스팟으로 회수돼야 한다. 킥오프는 같은 control frame 안에서
            # 소비될 수 있으므로(즉시 발동 규약) restart_kind == KICKOFF을 요구하지 않고
            # '공이 센터 근처로 돌아왔는가'로 검사한다.
            jnp.any(s.score > prev.score)
            & (jnp.linalg.norm(s.ball_pos[:2]) > 12.0),
            # 공이 경기장에서 폭주하면 물리가 깨진 것이다(아웃은 이벤트가 즉시 회수한다).
            (jnp.abs(s.ball_pos[0]) > env.hx + 40.0) | (jnp.abs(s.ball_pos[1]) > env.hy + 40.0),
            # role anchor epoch는 하프·교체뿐 아니라 승인된 포메이션 명령에서도 다시 열린다.
            # count 감소만으로 교체를 추론하지 않는다: identity SSOT는 slot_generation이고,
            # 전술 경계는 layout_index/layout_since_t의 원자적 변경이다. 팀별 명령이므로 감소한
            # 슬롯마다 자기 팀에 실제 레이아웃 경계가 있었는지 검사한다.
            _unexplained_role_count_decrease(prev, s),

            # --- 추가 7종 ---
            # 선수 비침투. 잠금은 분리기의 **문서화된 사양과 같은 값**이어야 한다.
            # ``movement.py``의 Jacobi 완화는 다자 경합에서 최악 1.0 cm의 근사 오차를
            # 남기고 다음 프레임에 스스로 해소하며, 그 1 cm를 없애는 대신 전면
            # Gauss-Seidel(4.6배 느림)도 조건부 실행(23% 느림)도 의도적으로 거절했다.
            #
            # 종전 잠금은 5 mm로 그 사양보다 좁았다. 짧은 표본에서는 드러나지 않다가
            # 384,000프레임 소크에서 3건이 걸렸고(재개 시나리오 2 · rule policy B 1),
            # 같은 시드로 다시 재면 최악 침투가 6.855 mm였다 — **사양 안, 잠금 밖**이다.
            # 물리가 규격을 어긴 것이 아니라 잠금이 규격을 잘못 옮긴 것이므로 사양에
            # 맞춘다. 이보다 깊은 침투는 그때야말로 solver 회귀다.
            jnp.min(jnp.where(
                active[:, None] & active[None, :] & (~jnp.eye(N, dtype=bool)),
                jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2), jnp.inf,
            )) < (2.0 * env.r_player - 0.010),
            # 재개 이격. 프레임 중간에 새로 생긴 재개는 다음 의사결정 step에서 투영되므로
            # **이어지는** 재개에만 요구한다. 새 인스턴스 판정은 runtime과 같은
            # restart_reopened SSOT를 쓰고, 지정 키커가 같은지도 확인한다.
            restart_timer_active(prev.restart_t) & restart_timer_active(rt)
            & (~env.restart_reopened(
                prev.restart_t, prev.restart_kind, prev.restart_team, s))
            & (taker == prev.pending_taker)
            & jnp.any(env._restart_encroacher_mask(s)),
            # 정상 퇴장으로 6명이 된 State는 **유효 terminal 결과**다. 검증할 계약은 최소
            # 인원 자체가 아니라 terminal이 흡수 상태라는 점이다.
            env._is_terminal(prev) & terminal_changed,
            # role anchor는 관측 좌표의 러닝 평균이므로 경계를 벗어날 수 없다.
            jnp.any((s.role_pos_count > 0)
                    & ((jnp.abs(s.role_pos[:, 0]) > bound_x + 1e-3)
                       | (jnp.abs(s.role_pos[:, 1]) > bound_y + 1e-3))),
            jnp.any(s.slot_generation < prev.slot_generation),
            (s.foul_kind == FOUL_NONE)
            & ((s.foul_actor != NO_PLAYER) | (s.foul_victim != NO_PLAYER)),
            # 활성 ⟺ (유효 팀 ∧ 플래그 하나 이상). 비활성이면 ``pass_team``도 함께
            # 비어 있어야 한다 — 그러지 않으면 pass_t=0인데 pass_team=TEAM_0인 반쪽
            # 래치가 정상으로 통과한다.
            ((s.pass_t > 0)
             != ((s.pass_team >= 0) & jnp.any(s.offside_flag)))
            | ((s.pass_t <= 0) & (s.pass_team != NO_TEAM)),
            # 라인 통과 버퍼는 칸 단위로 전부 차거나 전부 비어야 한다. 한 필드만 쓰이면
            # 소비자가 교차점 없는 사건이나 사건 없는 교차점을 보게 된다.
            # ``team_ok``는 NO_TEAM도 허용하므로 그것만으로는 "함께 찬다"를 검사하지
            # 못한다 — 실제 사건에 team=NO_TEAM을 넣거나 빈 칸에 team=TEAM_0을 넣어도
            # 통과했다. 채워진 칸은 **실제 팀**, 빈 칸은 **NO_TEAM**이어야 한다.
            jnp.any(
                ((s.ball_event_kind != BALL_EVENT_NONE)
                 != (s.ball_event_control_t >= 0))
                | ((s.ball_event_kind != BALL_EVENT_NONE)
                   & ((s.ball_event_team != TEAM_0)
                      & (s.ball_event_team != TEAM_1)))
                | ((s.ball_event_kind == BALL_EVENT_NONE)
                   & (s.ball_event_team != NO_TEAM))
                | (s.ball_event_kind < BALL_EVENT_NONE)
                | (s.ball_event_kind >= BALL_EVENT_COUNT)
            ),
            # 프레임 충돌 버퍼도 칸 단위로 함께 차거나 함께 빈다. 접촉점은 캡슐 표면이므로
            # 골라인에서 r_ball+rf보다 멀 수 없다 — 이 상한이 깨지면 스윕 판정이 엉뚱한
            # 캡슐을 집었다는 뜻이라, 좌표를 그대로 믿는 소비자(리플레이·슛 결말)가 오염된다.
            jnp.any(
                ((s.woodwork_kind != WOODWORK_NONE)
                 != (s.woodwork_control_t >= 0))
                | (s.woodwork_kind < WOODWORK_NONE)
                | (s.woodwork_kind >= WOODWORK_COUNT)
                | ((s.woodwork_kind != WOODWORK_NONE)
                   & (jnp.abs(jnp.abs(s.woodwork_pos[:, DIM_X]) - env.hx)
                      > env.r_ball + e.goal_frame_radius + GEOMETRY_EPS))
                | ((s.woodwork_kind == WOODWORK_NONE)
                   & jnp.any(s.woodwork_pos != 0.0, axis=-1))
            ),
            # ── 위 라벨 3종에 대응하는 새 검사 ──
            # [IFAB Law 9] GK가 손으로 잡고 있어도 공은 인플레이다. 홀드인데 데드면
            # 위반이고, 홀드인데 타이머가 없어도 위반이다(어느 쪽이든 그 종류가 아니다).
            is_hold & (dead | (rt <= 0)),
            ~((s.ball_state == BALL_ALIVE) | (s.ball_state == BALL_DEAD)),
            # NaN은 ``<``와 ``>``가 **둘 다 거짓**이라 범위 검사를 통째로 빠져나간다.
            # 범위를 재는 값은 먼저 유한해야 한다.
            ~(jnp.isfinite(s.cooldown).all() & jnp.isfinite(s.vmax).all()
              & jnp.isfinite(s.role_pos_count).all()),
        ]
        assert len(out) == len(CHECKS), (len(out), len(CHECKS))
        return jnp.asarray(out)

    def body(carry, key):
        state, counts = carry
        k_act, k_env = jax.random.split(key)
        if policy is None:
            action = jax.random.uniform(k_act, (N, 8), minval=-1.0, maxval=1.0)
        else:
            action = policy(env.get_obs_array(state), k_act, env.affordance_view(state))
        _, nxt, _, _, _ = env.step_env_array(
            k_env, state, action, include_bc_info=False, compute_observation=False)
        counts = counts + violations(state, nxt).astype(jnp.int32)
        return (nxt, counts), None

    @jax.jit
    def run(seed, steps_key):
        state = env.reset_state(seed)
        counts = jnp.zeros(len(CHECKS), jnp.int32)
        (final, counts), _ = lax.scan(body, (state, counts), steps_key)
        return final, counts

    return run


def soak(label, env, steps, seed, policy=None):
    run = build(env, policy)
    keys = jax.random.split(jax.random.PRNGKey(seed), steps)
    started = time.time()
    final, counts = run(jax.random.PRNGKey(seed), keys)
    counts = np.asarray(counts)
    took = time.time() - started
    bad = {CHECKS[i]: int(counts[i]) for i in range(len(CHECKS)) if counts[i]}
    status = "OK" if not bad else "위반"
    print(f"  {label:24s} {steps:7d} 프레임 {took:6.1f}s  {status}")
    if bad:
        for name, count in sorted(bad.items(), key=lambda kv: -kv[1]):
            print(f"      {name:24s} {count}")
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            f"{len(CHECKS)}개 환경 불변식을 JIT scan으로 검사합니다. "
            "옵션이 없으면 전체 장기 소크를 실행합니다."
        )
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="각 시나리오 프레임 수 배율(0보다 큰 값, 기본 1.0)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        metavar="LABEL",
        help="정확한 시나리오 라벨만 실행(여러 번 지정 가능)",
    )
    parser.add_argument(
        "--keep-backend",
        action="store_true",
        help=(
            "JAX backend를 CPU로 강제하지 않고 호출 환경 설정을 유지"
            "(float32/Threefry 계약은 계속 고정)"
        ),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.scale) or args.scale <= 0.0:
        parser.error("--scale must be a finite positive number")

    global SoccerEnv, restart_timer_active
    from .env import SoccerEnv
    from .restart import restart_timer_active

    total_frames = 0
    findings = {}
    scenarios = []

    scenarios.append(("default 90min", SoccerEnv(halftime=True, game_duration=81_000), 20_000, 1))
    scenarios.append(("no halftime", SoccerEnv(halftime=False, game_duration=81_000), 20_000, 2))
    scenarios.append((
        "foul storm",
        SoccerEnv(halftime=True, game_duration=81_000,
                  foul_config=Foul(tackle_p_min=0.4, tackle_p_max=0.95,
                                   charge_p_min=0.35, charge_p_max=0.95,
                                   card_per_foul=0.85, red_given_card=0.4)),
        20_000, 3))
    scenarios.append((
        "dense reward",
        SoccerEnv(halftime=True, game_duration=81_000, reward_config=Reward(mode="dense")),
        12_000, 5))

    team0 = [Agent(id=i, init_pos=(-22.0 - 3 * i, 6.0 * (i % 3) - 6.0), is_gk=(i == 0))
             for i in range(4)]
    team1 = [Agent(id=100 + i, init_pos=(-20.0 - 3 * i, 5.0 * (i % 3) - 5.0), is_gk=(i == 0))
             for i in range(3)]
    scenarios.append((
        "small roster 4v3",
        SoccerEnv(4, 3, team0, team1, halftime=True, game_duration=81_000), 12_000, 6))

    subs = [Substitution(tick=1500, slot=3, player_id=9001, entry_pos=(-20.0, 5.0),
                         role_pos=(-18.0, 4.0)),
            Substitution(tick=3000, slot=14, player_id=9002, entry_pos=(20.0, -5.0),
                         role_pos=(18.0, -4.0)),
            Substitution(tick=6000, slot=3, player_id=9003, entry_pos=(-10.0, 0.0),
                         role_pos=(-9.0, 0.0))]
    scenarios.append((
        "substitutions",
        SoccerEnv(halftime=True, game_duration=81_000, substitutions=subs), 12_000, 7))

    # 하프타임은 t=40,500이라 90분 config를 20,000 프레임 돌려도 도달하지 못한다.
    # 전환 자체를 밟는 config를 따로 둔다(진영·공격방향 반전, 전원 재배치).
    scenarios.append((
        "halftime crossing",
        SoccerEnv(halftime=True, game_duration=6_000), 12_000, 11))

    scenarios.append((
        "30 Hz control",
        SoccerEnv(halftime=True, game_duration=81_000, control_fps=30.0), 12_000, 8))

    from rule_policy import make_rule_based_policy
    for name, styles, seed in (("rule policy A", ("gegenpress", "park_the_bus"), 11),
                               ("rule policy B", ("tiki_taka", "long_ball"), 12),
                               ("rule policy C", ("balanced", "balanced"), 13)):
        penv = SoccerEnv(halftime=True, game_duration=81_000)
        pol = make_rule_based_policy(penv, match_key=jax.random.PRNGKey(seed), team_styles=styles)
        scenarios.append((name, penv, 20_000, seed, pol))

    labels = {entry[0] for entry in scenarios}
    if args.scenario:
        unknown = sorted(set(args.scenario) - labels)
        if unknown:
            parser.error(
                "unknown --scenario label(s): " + ", ".join(repr(x) for x in unknown)
                + "; available: " + ", ".join(sorted(labels))
            )
        selected = set(args.scenario)
        scenarios = [entry for entry in scenarios if entry[0] in selected]

    scaled = []
    for entry in scenarios:
        label, env, steps, seed, *rest = entry
        scaled_steps = max(1, int(math.floor(steps * args.scale + 0.5)))
        scaled.append((label, env, scaled_steps, seed, *rest))
    scenarios = scaled

    print(
        f"장기 소크 — {len(CHECKS)}종 불변식을 scan 안에서 매 프레임 검사 "
        f"(scale={args.scale:g}, backend={jax.default_backend()})"
    )
    for entry in scenarios:
        label, env, steps, seed = entry[:4]
        pol = entry[4] if len(entry) > 4 else None
        bad = soak(label, env, steps, seed, pol)
        total_frames += steps
        if bad:
            findings[label] = bad
    print()
    print(f"총 {total_frames:,} 프레임")
    if findings:
        print("=== 위반 요약 ===")
        for label, bad in findings.items():
            print(f"  {label}: {bad}")
        # 위반을 **출력만** 하고 0으로 끝나면 자동화(CI·스크립트)에서 통과한 소크와
        # 구분되지 않는다. 불변식이 깨진 실행은 실패로 끝나야 한다.
        return 1
    print("모든 불변식 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
