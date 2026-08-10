"""
policy.py — 관측(obs) 기반 룰 정책. env가 제대로 구현됐는지 눈으로(렌더) 검증하는 용도.

원본 SOCCER policy는 월드 절대상태(st)를 직접 읽었지만, 이 클론 정책은 **오직 obs 벡터만**
입력받아 action(8-D)을 낸다 — 그래서 env 폴더 밖 테스트/렌더 코드가 `obs → action`만으로
돌릴 수 있고, 동시에 "obs가 플레이에 충분한 마르코프 정보를 담는가"를 스스로 검증한다.

핵심 기하: 클론 obs는 **공격 프레임 폴딩**(에이전트별 자기중심, ×attack_dir)이고 action의
방향도 공격 프레임(env가 ×attack_dir로 월드 복원)이다. 따라서 정책은 월드 좌표를 몰라도 된다 —
모든 판단이 "내 공격 프레임"에서 이뤄지며 상대 골대는 항상 [+hx, 0], 우리 골대는 [-hx, 0]이다.
각 에이전트가 자기 obs만으로 전원·공의 공격 프레임 위치를 복원해 분산적으로 자기 행동을 정한다.

팀 성향(team_styles, (2,5) ∈[0,1]): 초기화 세팅으로 팀 색깔을 바꾼다.
  0 line_height 로우블록↔하이라인 · 1 tempo 점유↔직접 · 2 width 중앙↔측면
  3 aggression 지역↔맨압박 · 4 directness 숏빌드업↔롱볼
obs는 팀 정체를 감추므로(폴딩) 팀별 성향은 팩토리에서 team_id로 주입한다(정적 메타).

env 규칙 준수(진화한 env에 정합, obs로 판정):
  · 카드(self yellow): 경고 1장 선수는 태클 라인에 안 들어가고 컨테인/조키 — 2차 경고=퇴장 자충수 방지.
  · 간접 FK(is_fk_indirect, IFAB Law 13): 키커는 골 직격 금지, 반드시 동료로 연결(백패스·재터치 IDFK 포함).
  · GK 캐치/홀드(RK_GK_HOLD): 홀드 만료 전 GK가 롱 클리어(롱볼/무옵션) 또는 열린 동료 배급.

API:
  make_rule_based_policy(env, match_key=None, team_styles=None, policy_config=None) -> policy_fn
  policy_fn(obs, key) -> action        # obs (N, obs_dim) → action (N, ACTION_DIM=8)
  프리셋: STYLE_PRESETS["gegenpress"|"park_the_bus"|"tiki_taka"|"long_ball"|"balanced"]
"""
from __future__ import annotations
import types
import numpy as np
import jax
import jax.numpy as jnp

from constants import *
from config import RulePolicy
from spatial import _unit, stretch_encode
import tactics as T

def _sample_style(key, policy_config):
    """중앙편향(두 균등난수 평균 → 삼각분포) 성향 샘플 (5,)∈[0,1]."""
    return jnp.mean(
        jax.random.uniform(key, (policy_config.style_sample_count, STYLE_DIM)),
        axis=0,
    )


def _resolve_styles(match_key, team_styles, policy_config):
    """team_styles((2,5) 배열/이름/None)를 (2,5) jnp 배열로 정규화.
    None이면 match_key로 팀별 독립 샘플, 이름이면 프리셋."""
    def one(spec, key):
        if spec is None:
            return _sample_style(key, policy_config)
        if isinstance(spec, str):
            if spec not in STYLE_PRESETS:
                raise ValueError(
                    f"unknown team style {spec!r}; choose from {tuple(STYLE_PRESETS)}"
                )
            return jnp.asarray(STYLE_PRESETS[spec], jnp.float32)
        return jnp.asarray(spec, jnp.float32)

    keys = jax.random.split(
        match_key if match_key is not None else jax.random.PRNGKey(0),
        TEAM_COUNT,
    )
    if team_styles is None or isinstance(team_styles, str):
        s = team_styles
        styles = jnp.stack([one(s, key) for key in keys])
    else:
        if len(team_styles) != TEAM_COUNT:
            raise ValueError(f"team_styles must contain {TEAM_COUNT} teams")
        styles = jnp.stack([one(team_styles[i], keys[i]) for i in range(TEAM_COUNT)])
    if styles.shape != (TEAM_COUNT, STYLE_DIM):
        raise ValueError(
            f"team_styles must resolve to shape {(TEAM_COUNT, STYLE_DIM)}, got {styles.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(styles))) or not bool(
        jnp.all((styles >= 0.0) & (styles <= 1.0))
    ):
        raise ValueError("team style values must be finite and lie in [0, 1]")
    return styles


def _roles_from_home(home_att, gk, policy_config):
    """공격 프레임 홈 x(깊이)로 역할 유도. GK / DEF(≤-33) / MID(-33~-18) / FWD."""
    x = np.asarray(home_att[:, 0])
    r = np.where(
        x <= policy_config.role_defender_max_x,
        ROLE_DEFENDER,
        np.where(
            x <= policy_config.role_midfielder_max_x,
            ROLE_MIDFIELDER,
            ROLE_FORWARD,
        ),
    )
    r = np.where(np.asarray(gk) > 0.5, ROLE_GK, r)
    return jnp.asarray(r, jnp.int32)


def _calibrate_kick_solver(env, n_speed=56, nstep=450):
    """팩토리 1회: env의 **실제 공 물리**(ball_step_only)로 (발사속도→도달거리)를 롤아웃 캘리브하고
    (도달거리→필요 발사속도) 역테이블을 만든다. 런타임에 jnp.interp로 목표거리에 맞는 정밀 파워를 낸다.
    드래그·굴림·바운스가 전부 반영된 실물리 기반이라 R≈v²sin2θ/g 같은 근사보다 정확하다.

    두 궤적 모드:
      loft  — 높은 아크(수비수 넘김). '착지 거리'(apex 후 첫 지면 복귀 x) 기준. 크로스·롱패스·골킥·GK 롱배급.
      drive — 낮은 지상 드라이브. '도착 트래핑 속도까지 감속한 거리' 기준. 짧은 발밑 패스(리시버가 트랩 가능).
    반환: {loft_R, loft_v, drive_R, drive_v(모두 (n_speed,), interp용 단조증가), loft_launch01, drive_launch01}.
    """
    e = env.e_cfg
    rb = env.r_ball
    theta_loft, theta_drive = 0.55, 0.06        # rad — 높은 아크 / 거의 지면
    arrive_speed = 8.0                          # drive 패스 도착 목표 속력(트래핑 가능)
    ground = rb + 0.03
    speeds = jnp.linspace(4.0, e.f2b_speed_max, n_speed)

    def rollout(v0, theta):
        pos = jnp.array([0.0, 0.0, rb])
        vel = jnp.array([v0 * jnp.cos(theta), 0.0, v0 * jnp.sin(theta)])
        spin = jnp.zeros(3)
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[2], jnp.linalg.norm(v2[:2]))
        _, out = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return out                              # (xs, zs, spd) 각 (nstep,)

    xs_l, zs_l, _ = jax.vmap(lambda v: rollout(v, theta_loft))(speeds)     # (n_speed, nstep)
    xs_d, _, spd_d = jax.vmap(lambda v: rollout(v, theta_drive))(speeds)
    idx = jnp.arange(nstep)
    # loft 착지거리 = apex 이후 첫 지면 복귀 지점의 x
    apex = jnp.argmax(zs_l, axis=1)
    land = (zs_l <= ground) & (idx[None, :] > apex[:, None])
    land_idx = jnp.where(jnp.any(land, axis=1), jnp.argmax(land, axis=1), nstep - 1)
    loft_R = jnp.take_along_axis(xs_l, land_idx[:, None], axis=1)[:, 0]
    # drive 도착거리 = xy속력이 arrive_speed 이하로 처음 감속한 지점의 x
    dm = spd_d <= arrive_speed
    d_idx = jnp.where(jnp.any(dm, axis=1), jnp.argmax(dm, axis=1), nstep - 1)
    drive_R = jnp.take_along_axis(xs_d, d_idx[:, None], axis=1)[:, 0]
    # interp용 단조증가 보장(수치 흔들림 제거)
    loft_R = jax.lax.cummax(loft_R, axis=0)
    drive_R = jax.lax.cummax(drive_R, axis=0)
    lf = -e.launch_down_ground                  # 지면공 발사각 하한(launch_lo(r_ball))
    to01 = lambda th: float((th - lf) / (e.launch_max - lf))   # 목표각→action launch01 역매핑
    return dict(loft_R=loft_R, loft_v=speeds, drive_R=drive_R, drive_v=speeds,
                loft_launch01=to01(theta_loft), drive_launch01=to01(theta_drive))


def _calibrate_throw_solver(env, launch_h, n_speed=48, nstep=350):
    """스로인(손 던지기) 전용 캘리브 — 발사가 발킥이 아니라 **손 높이(launch_h)**에서 throw_speed_max
    스케일로 나간다(env _apply_force2ball의 throw 경로). 발킥 솔버로는 틀리므로 별도 캘리브.
    반환: {throw_R, throw_v(=속도, interp용 단조증가), throw_launch01}."""
    e = env.e_cfg
    rb = env.r_ball
    theta = 0.5
    ground = rb + 0.03
    speeds = jnp.linspace(3.0, e.throw_speed_max, n_speed)

    def rollout(v0):
        pos = jnp.array([0.0, 0.0, launch_h])
        vel = jnp.array([v0 * jnp.cos(theta), 0.0, v0 * jnp.sin(theta)])
        spin = jnp.zeros(3)
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[2])
        _, (xs, zs) = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return xs, zs

    xs, zs = jax.vmap(rollout)(speeds)
    idx = jnp.arange(nstep)
    apex = jnp.argmax(zs, axis=1)
    land = (zs <= ground) & (idx[None, :] > apex[:, None])
    land_idx = jnp.where(jnp.any(land, axis=1), jnp.argmax(land, axis=1), nstep - 1)
    throw_R = jax.lax.cummax(jnp.take_along_axis(xs, land_idx[:, None], axis=1)[:, 0], axis=0)
    lf = -e.launch_down_ground                    # 스로인 launch_ang도 스폿(지면) 기준 remap을 탄다
    throw_launch01 = float((theta - lf) / (e.launch_max - lf))
    return dict(throw_R=throw_R, throw_v=speeds, throw_launch01=throw_launch01)


def _calibrate_shot_solver(env, n_theta=56, nstep=220):
    """슛 전용 캘리브 — 목적이 '착지'가 아니라 '골 구석에 빠르게'. 고파워(≈0.9) 슛이 목표거리 d에서
    낮은 코너 높이(z_target)를 통과하도록 **발사각(launch01)을 거리별로 역산**한다. 파워는 높게 고정.
    반환: {shot_d, shot_launch01(거리별, interp용), shot_pow(스칼라)}."""
    e = env.e_cfg
    rb = env.r_ball
    v_shot = 0.97 * e.f2b_speed_max              # 거의 최대 파워 — GK 반응시간 최소화(피니싱)
    z_target = 0.9                                # 낮은 코너 목표 높이(m)
    thetas = jnp.linspace(-0.08, 0.5, n_theta)

    def rollout(theta):
        pos = jnp.array([0.0, 0.0, rb])
        vel = jnp.array([v_shot * jnp.cos(theta), 0.0, v_shot * jnp.sin(theta)])
        spin = jnp.zeros(3)
        def body(carry, _):
            p, v, s = carry
            p2, v2, s2 = env.ball_step_only(p, v, s)
            return (p2, v2, s2), (p2[0], p2[2])
        _, (xs, zs) = jax.lax.scan(body, (pos, vel, spin), None, length=nstep)
        return xs, zs

    XS, ZS = jax.vmap(rollout)(thetas)            # (n_theta, nstep)
    d_grid = jnp.linspace(4.0, e.f2b_shoot_range, 40)
    # 각 (d, theta): 궤적에서 x=d일 때의 z (x는 전진 중 단조증가)
    Z_at = jax.vmap(lambda d: jax.vmap(lambda i: jnp.interp(d, XS[i], ZS[i]))(jnp.arange(n_theta)))(d_grid)
    valid = (Z_at > 0.1) & (Z_at < env.goal_h - 0.2)                           # 지면~크로스바 하단 사이
    cost = jnp.where(valid, jnp.abs(Z_at - z_target), 1e6)
    best = jnp.argmin(cost, axis=1)               # (n_d,) 목표높이에 가장 근접한 발사각
    lf = -e.launch_down_ground
    shot_launch01 = jnp.clip((thetas[best] - lf) / (e.launch_max - lf), 0.0, 1.0)
    return dict(shot_d=d_grid, shot_launch01=shot_launch01, shot_pow=float(v_shot / e.f2b_speed_max))


def make_rule_based_policy(env, match_key=None, team_styles=None, policy_config=None):
    """팩토리: env에서 정적 기하(팀·역할·홈·정규화 상수)를 스냅샷하고 팀 성향을 고정해
    obs→action 클로저를 반환. env State는 일절 건드리지 않아 학습 에이전트와 100% 독립."""
    policy_config = RulePolicy() if policy_config is None else policy_config
    if not isinstance(policy_config, RulePolicy):
        raise TypeError(
            f"policy_config must be RulePolicy or None, got {type(policy_config).__name__}"
        )
    e, s = env.e_cfg, env.s_cfg
    st0 = env.reset_state(jax.random.PRNGKey(0))
    adir0 = st0.attack_dir
    home_att = env._kickoff_positions(st0) * adir0[:, None]      # 공격 프레임 홈(+x=전방)
    gk = st0.gk_indices.astype(jnp.float32)
    solver_cache = getattr(env, "_rule_policy_solver_cache", None)
    if solver_cache is None:
        ksolve = _calibrate_kick_solver(env)        # 발킥 솔버(거리→속도 역테이블)
        launch_h = float(st0.head_z[1]) + e.throw_height       # 스로인 손 높이(대표값)
        tsolve = _calibrate_throw_solver(env, launch_h)        # 스로인 전용 솔버
        ssolve = _calibrate_shot_solver(env)                   # 슛 발사각 솔버
        solver_cache = (ksolve, tsolve, ssolve)
        env._rule_policy_solver_cache = solver_cache
    else:
        ksolve, tsolve, ssolve = solver_cache

    spec = env.obs_spec()
    sf = spec["self"]["features"]
    of = spec["others"]["features"]
    bf = spec["ball"]["features"]
    ctx = spec["context"]["features"]
    spf = spec["context"]["setpiece_features"]
    sp0 = ctx["setpiece"][0]

    ctx_ns = types.SimpleNamespace(
        N=env.N, n=env.n_agents, hx=env.hx, hy=env.hy, length=s.length, width=s.width,
        goal_w=env.goal_w, r_ball=env.r_ball, f2b_max=e.f2b_speed_max, shoot_range=e.f2b_shoot_range,
        reach_xy=e.reach_xy, drib_max=e.dribble_speed_max, launch_max=e.launch_max,
        n_pvel=e.norm_player_vel, n_bvel=e.norm_ball_vel, n_bz=e.norm_ball_z,
        clear_dist=e.clear_dist, game_dur=env.game_duration, gravity=e.g,
        team_id=st0.team_id.astype(jnp.int32), gk=gk,
        roles=_roles_from_home(home_att, gk, policy_config), home_att=home_att,
        styles=_resolve_styles(match_key, team_styles, policy_config),
        policy=policy_config,
        others_start=spec["others"]["start"], others_size=spec["others"]["size"],
        i_self_pos=sf["abs_pos"], i_self_vel=sf["abs_vel"], i_in_reach=sf["in_reach"],
        i_f2b_avail=sf["f2b_avail"], i_is_gk=sf["is_gk"], i_retouch=sf["retouch"],
        i_offside=sf["own_offside"], i_cooldown=sf["cooldown"],
        o_relpos=of["rel_pos"], o_gk=of["is_gk"], o_taker=of["is_taker"], o_team=of["team_flag"],
        o_absvel=of["abs_vel"], o_offside=of["offside"],   # [지표] 상대 속도(선점)·오프사이드(패스 무효 게이트)
        i_pass_signal=sf["pass_signal"][0], i_own_offside=sf["own_offside"][0],
        i_ball_pos=bf["rel_pos"], i_ball_alive=bf["ball_alive"], i_poss=bf["poss_ours"],
        i_ball_vel=bf["abs_vel"],   # 공 절대속도
        i_rt=(sp0 + spf["rt_norm"][0]), i_sp_ours=(sp0 + spf["is_sp_ours"][0]),
        i_kick_lock=(sp0 + spf["is_kicker_locked"][0]), i_enc=(sp0 + spf["enc_margin"][0]),
        i_rk=(sp0 + spf["rk_onehot"][0]), i_fk_indirect=(sp0 + spf["is_fk_indirect"][0]),
        i_yellow=sf["yellow"][0], i_vmax=sf["vmax"][0], i_last_touch=ctx["last_touch"][0],
        i_off_line=ctx["off_line"][0],
        pen_len=env.pen_len, pen_hw=env.pen_hw, gk_catch_cap=e.gk_catch_speed_cap,
        loft_R=ksolve["loft_R"], loft_v=ksolve["loft_v"],
        drive_R=ksolve["drive_R"], drive_v=ksolve["drive_v"],
        loft_launch01=ksolve["loft_launch01"], drive_launch01=ksolve["drive_launch01"],
        throw_R=tsolve["throw_R"], throw_v=tsolve["throw_v"], throw_launch01=tsolve["throw_launch01"],
        throw_speed_max=e.throw_speed_max,
        shot_d=ssolve["shot_d"], shot_launch01=ssolve["shot_launch01"], shot_pow=ssolve["shot_pow"],
    )

    def policy_fn(obs, key):
        """obs (N, obs_dim) → action (N, ACTION_DIM=8). 공격 프레임 입출력. step_env_array에 직접 투입."""
        return _rule_based_actions(jnp.asarray(obs), key, ctx_ns)

    def act_dict(obs_dict, key):
        """JaxMARL dict 어댑터 — get_obs dict → action dict. env.step(dict API)와 함께 쓸 때."""
        stacked = jnp.stack([obs_dict[a] for a in env.agents])
        act = policy_fn(stacked, key)
        return {a: act[i] for i, a in enumerate(env.agents)}

    policy_fn.act_dict = act_dict
    policy_fn.team_styles = ctx_ns.styles
    policy_fn.roles = ctx_ns.roles
    policy_fn.policy_config = policy_config
    return policy_fn


def _rule_based_actions(obs, key, c):
    """관측만으로 국면(세트피스/공격/수비/루즈볼)을 분해해 각 에이전트 행동을 정한다.
    모든 좌표는 자기 공격 프레임(m): 상대 골 [+hx,0], 우리 골 [-hx,0], +x=전방."""
    N = c.N
    policy = c.policy
    k_pass, k_noise = jax.random.split(key, 2)   # 패스 gumbel tie-break · 킥 각오차

    # ── obs 복원(공격 프레임, 미터) ─────────────────────────────────────────
    def col(i0, i1):
        return obs[:, i0:i1]

    my_field = col(*c.i_self_pos) * jnp.array([c.hx, c.hy])            # 내 위치
    my_vel = col(*c.i_self_vel) * c.n_pvel                             # 내 절대속도(공격 프레임, m/s)
    my_speed = jnp.linalg.norm(my_vel, axis=1)
    in_reach = obs[:, c.i_in_reach[0]] > 0.5
    f2b_avail = obs[:, c.i_f2b_avail[0]] > 0.5
    is_gk = obs[:, c.i_is_gk[0]] > 0.5
    retouch = obs[:, c.i_retouch[0]] > 0.5
    cooldown = obs[:, c.i_cooldown[0]]
    booked = obs[:, c.i_yellow] > 0.5           # 경고 1장 — 2차 경고=퇴장이라 태클 자제(조키 전환)

    ball_rel = col(*c.i_ball_pos)
    ball_field = my_field + ball_rel[:, :2] * jnp.array([c.length, c.width])
    ball_z = ball_rel[:, 2] * c.n_bz
    ball_vel = col(*c.i_ball_vel) * c.n_bvel
    ball_speed = jnp.linalg.norm(ball_vel[:, :2], axis=1)
    ball_alive = obs[:, c.i_ball_alive[0]] > 0.5
    # 공중볼 낙하 지점(간이 탄도) — 리시버가 '떨어질 곳'으로 달려가 헤더/발리 경합하게.
    z_above = jnp.maximum(ball_z - c.r_ball, 0.0)
    gravity = c.gravity
    t_land = jnp.clip(
        (
            ball_vel[:, DIM_Z]
            + jnp.sqrt(ball_vel[:, DIM_Z] ** 2 + 2.0 * gravity * z_above)
        )
        / gravity,
        0.0,
        2.5,
    )
    ball_landing = ball_field + ball_vel[:, :2] * t_land[:, None]
    airborne = ball_z > 1.2
    poss = obs[:, c.i_poss[0]]                                        # +1 우리 / -1 상대 / 0 없음
    attacking = poss > 0.5
    defending = poss < -0.5

    others = obs[:, c.others_start:c.others_start + (N - 1) * c.others_size].reshape(N, N - 1, c.others_size)
    o_rel = others[:, :, c.o_relpos[0]:c.o_relpos[1]] * jnp.array([c.length, c.width])
    others_field = my_field[:, None, :] + o_rel                      # (N,N-1,2) 상대·동료 위치
    o_gk = others[:, :, c.o_gk[0]] > 0.5
    o_team = others[:, :, c.o_team[0]]
    mate = o_team > 0.5
    opp = o_team < -0.5
    active_o = jnp.abs(o_team) > 0.5
    # [지표] 타 선수 절대속도(공격프레임, m/s) — 압박·패스차단·마킹 선점의 핵심 신호.
    o_vel = others[:, :, c.o_absvel[0]:c.o_absvel[1]] * c.n_pvel      # (N,O,2)
    o_offside = others[:, :, c.o_offside[0]] > 0.5                    # 오프사이드 위치 동료(패스 무효 게이트)

    is_sp_ours = obs[:, c.i_sp_ours]                                 # +1 우리 재개 / -1 상대 / 0
    enc_margin = obs[:, c.i_enc]                                     # 침범 여유(음수=침범)
    rk = obs[:, c.i_rk:c.i_rk + RESTART_COUNT]                       # restart_kind one-hot
    restart_active = rk[:, RK_NONE] < 0.5

    sty = c.styles[c.team_id]
    line_h, tempo, width_s, aggr, direct = (
        sty[:, STYLE_LINE],
        sty[:, STYLE_TEMPO],
        sty[:, STYLE_WIDTH],
        sty[:, STYLE_AGGRESSION],
        sty[:, STYLE_DIRECTNESS],
    )

    # ── 캐리어/최근접 판정(자기 obs 내 거리 비교) ───────────────────────────
    ball_dist = jnp.linalg.norm(ball_field - my_field, axis=1)
    # 세트피스 taker(is_taker=재터치 금지/킥 예정)와 재터치 제한 중인 자신은 '공 회수자'에서 제외 —
    # 방금 찬 키커가 제 공 위에서 얼어붙지 않고, 다른 동료가 collector로 나서게 한다.
    o_is_taker = others[:, :, c.o_taker[0]] > 0.5
    d_ball_o = jnp.linalg.norm(ball_field[:, None, :] - others_field, axis=2)
    mate_ball = jnp.where(mate & (~o_is_taker), d_ball_o, jnp.inf)
    field_ball = jnp.where(active_o & (~o_gk) & (~o_is_taker), d_ball_o, jnp.inf)
    am_nearest_mate = (~retouch) & (
        ball_dist <= jnp.min(mate_ball, axis=1) + GEOMETRY_EPS
    )
    am_nearest_field = (~is_gk) & (~retouch) & (
        ball_dist <= jnp.min(field_ball, axis=1) + GEOMETRY_EPS
    )
    team_rank = jnp.sum((mate_ball < ball_dist[:, None]).astype(jnp.float32), axis=1)  # 0=우리팀 최근접
    carrier_j = jnp.argmin(mate_ball, axis=1)
    carrier_field = jnp.where(am_nearest_mate[:, None], my_field, jnp.take_along_axis(
        others_field, carrier_j[:, None, None], axis=1)[:, 0, :])

    opp_goal = jnp.array([c.hx, 0.0])
    own_goal = jnp.array([-c.hx, 0.0])
    ball_adv = ball_field[:, 0]                                      # 공 전진도(+ = 상대 진영)

    # ── 국면 B: 공격(우리 점유) — 지표 기반 기대가치 결정 ─────────────────────
    # 캐리어는 슛/패스/드리블/전개를 **공통 통화(≈득점 기여 기대값)**로 비교해 argmax. 모든
    # 에이전트가 '내가 캐리어라면'을 계산하고 am_nearest_mate 게이팅으로 실제 캐리어만 실행(분산·일관).
    to_goal_vec = opp_goal[None, :] - my_field
    d_goal = jnp.linalg.norm(to_goal_vec, axis=1) + DIV_EPS

    # 지표(tactics): 자기·동료 슛 xg, 압박(상대속도 선점), 위협통화 threat, 패스 성공확률(인터셉트 예측)
    xg_self = T.shot_xg(my_field, others_field, opp, o_gk, c.hx, c.goal_w)        # (N,)
    xg_mate = T.shot_xg(others_field, others_field, opp, o_gk, c.hx, c.goal_w)    # (N,M)
    press_self = T.pressure(my_field, others_field, o_vel, opp)                          # (N,)
    # ★슛 가치 통화 통일: '이 위치에서의 슛 가치' = xg·RulePolicy.shoot_gain.
    # 슛/패스/드리블이 모두 같은 단위로
    #  '지금 슛 vs 더 좋은 슛을 만든 뒤 슛'을 비교한다 → range에 들었다고 때리지 않고, 패스·드리블로
    #  xg가 더 오르면 그쪽을 택하다가 지금 슛이 최선일 때만 슛(argmax가 crossover를 자동 결정).
    sv_self = xg_self * policy.shoot_gain                                                       # (N,)
    threat_mate = jnp.maximum(
        xg_mate * policy.shoot_gain,
        T.pitch_value(others_field, c.hx, c.hy),
    )  # (N,M) 수신 후 슛/전개 가치
    comp = T.lane_completion(my_field, others_field, others_field, o_vel, opp)           # (N,M) 성공확률

    # 슛 조준(GK 반대 포스트)
    opp_gk_y = jnp.sum(jnp.where(opp & o_gk, others_field[:, :, 1], 0.0), axis=1)
    aim = jnp.stack([
        jnp.full(N, c.hx),
        -jnp.sign(opp_gk_y + DIV_EPS) * (c.goal_w / 2 * 0.90),
    ], axis=1)
    dir_shot = _unit(aim - my_field)

    # 패스 후보 평가: V_pass_j = 성공확률 × 수신 후 위협. 오프사이드 위치·근거리·초원거리 동료 제외.
    mate_d = jnp.linalg.norm(others_field - my_field[:, None, :], axis=2)
    mate_adv = others_field[:, :, 0] - my_field[:, 0:1]
    pass_ok = mate & (mate_d > 3.0) & (mate_d < 50.0) & (~o_offside)
    # 측면전환 소보너스(동률깨기·스타일) — 결정은 위협통화가 지배.
    switch_bonus = (jnp.clip((jnp.abs(others_field[:, :, 1]) - jnp.abs(my_field[:, 1:2])) / c.hy, 0.0, 1.0)
                    * 0.06 * (0.5 + width_s[:, None]))
    # 전진 보상: 안전한 옆·뒤 패스(comp↑·threat 비슷)보다 전방 찔러주기를 우선. directness에 비례.
    prog = jnp.clip(mate_adv, 0.0, 25.0) / 25.0                                          # (N,M) 전진량 정규화
    prog_w = 0.60 + 0.65 * direct[:, None]      # 전진패스 강조↑(파이널서드 침투 — 미드 순환 억제)
    V_pass_mate = jnp.where(pass_ok, comp * (threat_mate + prog_w * prog) + switch_bonus, -1.0)  # (N,M)
    pass_u = jnp.clip(
        jax.random.uniform(k_pass, (N, N - 1)), PROB_EPS, 1.0 - PROB_EPS
    )
    gumbel = -jnp.log(-jnp.log(pass_u)) * 0.03
    best_j = jnp.argmax(V_pass_mate + gumbel, axis=1)
    V_pass = jnp.take_along_axis(V_pass_mate, best_j[:, None], axis=1)[:, 0]             # (N,)
    best_mate_field = jnp.take_along_axis(others_field, best_j[:, None, None], axis=1)[:, 0, :]
    best_mate_adv = jnp.take_along_axis(mate_adv, best_j[:, None], axis=1)[:, 0]
    best_comp = jnp.take_along_axis(comp, best_j[:, None], axis=1)[:, 0]
    d_pass = jnp.linalg.norm(best_mate_field - my_field, axis=1) + DIV_EPS
    dir_pass = _unit(best_mate_field - my_field)
    has_pass = jnp.any(pass_ok, axis=1)
    lane_block = best_comp < 0.6      # 선택 패스의 지상 레인이 상당히 막힘(성공확률↓) → 로프트로 넘김

    # 드리블 가치: 전방 5m 위협 × 유지확률(압박 반비례). 전방이 상대 콘으로 막히면 감가.
    fwd_goal = _unit(opp_goal[None, :] - my_field)
    opp_fwd = jnp.where(opp, others_field[:, :, 0] - my_field[:, 0:1], -1e9)
    opp_side = jnp.where(opp, jnp.abs(others_field[:, :, 1] - my_field[:, 1:2]), 1e9)
    fwd_blocked = jnp.any(
        (opp_fwd > 0.5)
        & (opp_fwd < policy.dribble_cone_length)
        & (opp_side < 3.0),
        axis=1,
    )
    keep_prob = jnp.clip(1.0 - 0.45 * press_self, 0.25, 1.0)
    # 드리블 목표 = 골 방향 5m 전진(단, 골 2m 앞까지만 — 오버런 방지). 그 지점의 슛 가치(xg·GAIN)를
    # 전개 가치(pitch)와 max로 결합 → **드리블로 더 좋은 슛 위치를 만드는 것**이 슛 통화로 평가된다.
    step_len = jnp.clip(d_goal - 2.0, 0.0, 5.0)
    drib_step = my_field + fwd_goal * step_len[:, None]
    xg_step = T.shot_xg(drib_step, others_field, opp, o_gk, c.hx, c.goal_w)       # 드리블 후 슛 가치
    # 드리블 후 슛 가치 xg_step은 **GAIN 없이**(≤1) — 드리블은 '접근수단'이라 pitch와 같은 스케일로만
    # 반영(GAIN 스케일 시 무한 접근-드리블로 과드리블). 근접해 xg_self가 크면 V_shoot=xg·GAIN이 이겨 슛.
    # 0.85 배 — 드리블은 보조수단(맨 제치기·근접). 전진패스를 1차 진행수단으로.
    V_drib = jnp.maximum(T.pitch_value(drib_step, c.hx, c.hy), xg_step) * keep_prob * 0.85
    V_drib = jnp.where(fwd_blocked, V_drib * 0.45, V_drib)

    # 슛 가치: 사거리 안에서만(밖이면 0 → 반드시 접근).
    # 통화는 sv_self=xg·RulePolicy.shoot_gain.
    in_range = d_goal < c.shoot_range
    V_shoot = jnp.where(in_range, sv_self, 0.0)

    # 전개(클리어) 가치: 자기 진영 깊이 + 강압박 + 무옵션일 때만 안전판(전방 롱 클리어).
    deep = my_field[:, 0] < -c.hx / 3.0
    V_clear = jnp.where(deep & (press_self > 0.6) & (~has_pass), 0.14, 0.02)

    # 스타일 변조: tempo↑ 패스선호, 낮으면 드리블. directness↑ 슛 선호.
    V_pass = V_pass * (0.85 + 0.35 * tempo)
    V_drib = V_drib * (0.90 + 0.25 * (1.0 - tempo))
    V_shoot = V_shoot * (0.90 + 0.25 * direct)

    # ★추상 행동 선택 — 공통 통화 argmax(0 슛 / 1 패스 / 2 드리블 / 3 전개)
    opt = jnp.argmax(jnp.stack([V_shoot, V_pass, V_drib, V_clear], axis=1), axis=1)
    # 슛 = argmax가 '지금 슛'을 택했을 때(=패스·드리블로 더 좋은 슛을 만드는 것보다 지금 슛이 나을 때).
    # 하드 거리강제(d_goal<14) 제거 — range에 들었다고 무조건 때리지 않는다. 단 확실한 대찬스(xg>0.55)는
    # 개선 여지 무관하게 즉시 슛(탭인 흘리지 않음).
    shoot = ((opt == 0) | (xg_self > 0.55)) & in_range & (xg_self > 0.03)
    chose_pass = (opt == 1) & (~shoot)
    carrier_pass_all = chose_pass & has_pass
    # 크로스: 측면·전진 상황의 패스는 박스로 로프트(패스의 하위 유형).
    wide = jnp.abs(my_field[:, 1]) > c.hy * 0.35
    crossing = carrier_pass_all & wide & (ball_adv > c.hx / 3.0) & (d_goal > 14.0)
    cross_side = jnp.sign(my_field[:, 1] + DIV_EPS)
    box_far = jnp.stack([jnp.full(N, c.hx - 8.0), -cross_side * 5.0], axis=1)
    dir_cross = _unit(box_far - my_field)
    carrier_pass = carrier_pass_all & (~crossing)
    carrier_clear = (opt == 3) & (~shoot)                            # V_clear가 이미 무옵션·강압박·자기진영 게이트
    # 나머지 전부 드리블(항상 폴백) — '드리블을 골랐다'와 '패스를 골랐지만 무옵션'을 함께 흡수한다.
    carrier_drib = (~shoot) & (~carrier_pass_all) & (~carrier_clear)

    c_kick_dir = jnp.where(shoot[:, None], dir_shot,       # fwd_goal은 위 드리블 가치 계산에서 이미 정의됨
                 jnp.where(crossing[:, None], dir_cross,
                 jnp.where(carrier_pass[:, None], dir_pass, fwd_goal)))
    # 정밀 킥 솔버(팩토리 캘리브): 목표거리 R → 필요 파워(=발사속도/f2b_max). loft=아크(착지)·drive=지상(트래핑 도착).
    def loft_pow_of(R):  return jnp.clip(jnp.interp(R, c.loft_R, c.loft_v) / c.f2b_max, 0.15, 1.0)
    def drive_pow_of(R): return jnp.clip(jnp.interp(R, c.drive_R, c.drive_v) / c.f2b_max, 0.12, 1.0)
    def throw_pow_of(R): return jnp.clip(jnp.interp(R, c.throw_R, c.throw_v) / c.throw_speed_max, 0.12, 1.0)  # 손 던지기
    shot_launch = jnp.interp(d_goal, c.shot_d, c.shot_launch01)    # 슛 발사각(거리별 솔버) — 저각 강타로 골 도달

    lofted = carrier_pass & ((d_pass > 14.0) | (lane_block & (best_mate_adv > 3.0)))  # 긴 볼·막힌 전진패스만 띄움
    pass_pow = drive_pow_of(d_pass)                                # 지상 드라이브 — 리시버에 트래핑 속도로 도착
    loft_pow = loft_pow_of(d_pass)                                 # 아크 — 수비수 넘겨 목표 지점 착지
    drib_pow = jnp.clip((my_speed + 1.5) / c.f2b_max, 0.03, 0.35)   # 내 속도보다 살짝 앞서 밀어 놓기
    d_cross = jnp.linalg.norm(box_far - my_field, axis=1)          # 크로스 목표(박스 원거리) 거리
    cross_pow = loft_pow_of(d_cross)                               # 크로스도 솔버로 정밀 착지
    clear_pow = loft_pow_of(jnp.clip(d_goal, 20.0, 42.0))         # 캐리어 클리어 — 전방 롱 아크(솔버)
    c_kick_pow = jnp.where(shoot, (c.shot_pow if policy.use_shot_solver else 0.95), jnp.where(crossing, cross_pow,
                 jnp.where(carrier_pass, jnp.where(lofted, loft_pow, pass_pow),
                 jnp.where(carrier_drib, drib_pow, clear_pow))))
    # 발사각: 슛은 거리별 저각(shot_launch), 크로스·클리어는 loft, 드라이브 패스는 drive.
    c_launch = jnp.where(shoot, (shot_launch if policy.use_shot_solver else 0.05), jnp.where(crossing, c.loft_launch01,
               jnp.where(carrier_pass, jnp.where(lofted, c.loft_launch01, c.drive_launch01),
               jnp.where(carrier_drib, 0.0, c.loft_launch01))))
    c_spin_s = jnp.where(shoot, -jnp.sign(my_field[:, 1] + DIV_EPS) * 0.6,
               jnp.where(crossing, -cross_side * (0.6 + 0.35 * width_s), 0.0))
    c_spin_b = jnp.where(shoot, -0.4, jnp.where(crossing, 0.4, 0.0))

    # 통제된(느린) 공에서만 의도적 킥. 빠른 공은 아래 트래핑이 먼저 잡는다(2-터치).
    settled = ball_speed < 4.0
    drib_touch = carrier_drib & (ball_speed < 2.5)
    shoot_ok = shoot & (settled | (d_goal < 16.0))                 # 박스 근거리는 첫 터치 슛(트래핑 대기 없이 바로)
    carrier_kick = shoot_ok | ((crossing | carrier_pass | carrier_clear | drib_touch) & settled)

    # 오프더볼 배치: 홈 형태의 좌우 간격·서열을 보존한 채(과대 확장 금지) 공 전진도만큼 전방 이동하고
    # 블록 전체를 공 쪽으로 살짝 슬라이드(볼사이드 컴팩트). 예전엔 y를 ×1.15~1.65로 부풀리고 |y|>18
    # 전원(6/10)을 터치라인(±29m)에 붙박아 '양끝 바벨·중앙 공동'이 생겼다(측정: 측면 51%·중앙 18%).
    # 공격(우리 점유) 시 전방 침투 가속 — 포워드는 파이널서드로, 미드는 그 뒤로 올라가 '전진 패스 타깃'을
    # 만든다(전방 주자가 없으면 전진 패스를 넣을 곳이 없어 볼이 미드필드에서 맴돈다 = 전진성 패스의 전제).
    role_run = jnp.where(
        c.roles == ROLE_FORWARD,
        16.0,
        jnp.where(c.roles == ROLE_MIDFIELDER, 8.0, 0.0),
    )
    att_run = jnp.where(attacking, role_run + 0.22 * jnp.clip(ball_adv, 0.0, None), 0.0)
    fwd_push = jnp.clip(ball_adv * 0.30 + line_h * 10.0 + att_run, -6.0, 46.0)
    off_x = c.home_att[:, 0] + fwd_push
    # 온사이드 유지: 공격 시 전방 주자는 최종수비 라인(off_line)까지만 — 라인을 넘으면 오프사이드로
    # 전진 패스가 무효가 된다. 라인 어깨에서 대기하다 스루패스에 뛰어들게(전진성 패스 성립).
    off_line_x = obs[:, c.i_off_line] * c.hx
    off_x = jnp.where(attacking, jnp.minimum(off_x, off_line_x), off_x)
    off_y = (
        c.home_att[:, 1] * (policy.width_base + policy.width_gain * width_s)
        + ball_field[:, 1] * policy.ballside_slide
    )
    # 폭은 '진짜 윙어'(가장 넓은 역할)만, 그것도 파이널서드에서만 유지 — 전원 조기 광폭 방지.
    is_wide = jnp.abs(c.home_att[:, 1]) > policy.wide_home_threshold
    hold_wide = is_wide & (ball_adv > c.hx * policy.wide_progress_gate)
    off_y = jnp.where(
        hold_wide,
        jnp.sign(c.home_att[:, 1]) * (c.hy * policy.wide_pitch_fraction),
        off_y,
    )
    off_target = jnp.stack([jnp.clip(off_x, -c.hx + 4.0, c.hx - 6.0),
                            jnp.clip(off_y, -c.hy + 3.0, c.hy - 3.0)], axis=1)
    # 서포트: 대형 유지를 위해 '캐리어 최근접 동료 상위 3명만' 지정 슬롯(좌/우/후방)으로 —
    # 거리 환대(7~16m) 방식은 조건 맞는 전원이 ±11m 두 슬롯에 수렴해 겹침·대형 붕괴를 만들었다.
    # 순위는 각자 자기 obs의 거리 비교로 일관 계산(캐리어 자신은 d≈0이라 -1로 제외).
    d_carrier = jnp.linalg.norm(my_field - carrier_field, axis=1)
    d_carrier_o = jnp.linalg.norm(others_field - carrier_field[:, None, :], axis=2)
    srank = (
        jnp.sum(
            (mate & (d_carrier_o < d_carrier[:, None] - GEOMETRY_EPS)).astype(jnp.float32),
            axis=1,
        )
        - 1.0
    )
    is_support = ((~am_nearest_mate) & (~hold_wide) & (~is_gk)
                  & (d_carrier < 18.0) & (srank >= 0.0)
                  & (srank < policy.support_count))
    slot = jnp.where((srank < 0.5)[:, None], jnp.array([[4.0, 11.0]]),
           jnp.where((srank < 1.5)[:, None], jnp.array([[4.0, -11.0]]),
           jnp.where((srank < 2.5)[:, None], jnp.array([[-8.0, 0.0]]),
           jnp.where((srank < 3.5)[:, None], jnp.array([[10.0, 6.0]]), jnp.array([[10.0, -6.0]]))))) * policy.support_scale
    support_pos = carrier_field + slot
    # [지표] 서포트 미세조정 — 고정 슬롯이 대형 서열을 잡고, 지표가 '어디가 진짜 열렸나'를 국소 보정.
    # 슬롯 주변 후보 중 (공간 openness + 캐리어→나 레인 성공확률 + 전진위협)을 최대화하는 점으로 당긴다.
    # 상대속도 선점이 openness·lane_completion 양쪽에 들어가 '곧 닫힐 공간'을 피한다.
    cand_off = jnp.array([[0., 0.], [5., 0.], [-4., 0.], [0., 6.], [0., -6.], [5., 5.], [5., -5.]])
    cand = support_pos[:, None, :] + cand_off[None, :, :]                       # (N,K,2)
    op_c = T.openness(cand, others_field, o_vel, opp) / 12.0                    # (N,K) 공간
    lane_c = T.lane_completion(carrier_field, cand, others_field, o_vel, opp)   # (N,K) 캐리어→후보 레인
    pv_c = T.pitch_value(cand, c.hx, c.hy)                                      # (N,K) 전진위협
    # 위협(전진) 우선 가중 — openness만 좇으면 붐비는 파이널서드를 피해 미드로 후퇴한다(침투 저해).
    best_c = jnp.argmax(0.6 * op_c + 0.9 * lane_c + 1.2 * pv_c, axis=1)
    support_ref = jnp.take_along_axis(cand, best_c[:, None, None], axis=1)[:, 0, :]
    support_pos = 0.55 * support_pos + 0.45 * support_ref                       # 슬롯(전진배치) 비중↑
    off_target = jnp.where(is_support[:, None], support_pos, off_target)
    drib_target = ball_field                                        # 캐리어는 공에 직접 호밍(오버런 방지)
    attack_target = jnp.where(am_nearest_mate[:, None], drib_target, off_target)
    carrier_pow = jnp.where(ball_dist < 2.5, 0.45, 0.9)             # 근접 감속(오버런 억제)
    attack_pow = jnp.where(am_nearest_mate, carrier_pow,
                 jnp.where(is_support, 0.7, 0.72))

    # ── 국면 C: 수비(상대 점유) — 능동 압박 + 맨마킹 + 컴팩트 블록(골라인 붕괴 방지) ──
    # 최근접 수비수가 볼로 강압박(공격성/위험지역이면 2차 압박 가세), 비압박 수비수는 침투한
    # 위협 상대를 골사이드 마킹, 나머지는 공 높이의 컴팩트 라인 유지. '전원 골대 앞 뭉침' 제거.
    ball_deep = ball_adv < -c.hx / 3.0
    # 압박 완화: 상시 3인 스웜은 미드필드에서 공을 즉시 뺏어 공격이 파이널서드까지 전개되지 못하게
    # 한다(측정: 파이널서드 공격 터치 ~0). 1차만 상시 압박하고, 2차는 공격적 팀·위험지역, 3차는 드물게 —
    # 캐리어에 빌드업·전진패스 시간을 줘 공격이 전개되게(→슛·골 기회 생성).
    # 압박은 '조금만' 완화 — 높은 압박이 턴오버로 공격 기회를 만들어(게겐프레싱), 너무 빼면 양 팀이
    # 로우블록으로 앉아 침투가 죽는다(측정: 완전 완화 시 슛 6→0). 2차 압박 문턱만 살짝 올려(0.5→0.65)
    # 소극적 팀은 미드필드 더블프레스를 줄이되, 공격적 팀·위험지역은 유지.
    press1 = defending & (team_rank < 0.5)
    press2 = defending & (team_rank >= 0.5) & (team_rank < 1.5) & ((aggr > 0.5) | ball_deep)
    press3 = defending & (team_rank >= 1.5) & (team_rank < 2.5) & (aggr > 0.65) & ball_deep
    press = press1 | press2 | press3
    # 1차는 공에 바짝 붙어 탈취 시도, 2·3차는 공 진행방향을 살짝 리드해 패스레인 차단·커버.
    # 부킹된 1차 압박수는 태클 라인(0.8m 밀착)에 안 들어가고 컨테인(3m 조키)으로 — 2차 경고=퇴장 회피.
    press_lunge = press1 & (~booked)
    press_target = jnp.where(press_lunge[:, None],
                             ball_field + _unit(own_goal[None, :] - ball_field) * 0.8,
                             ball_field + ball_vel[:, :2] * 0.3 + _unit(own_goal[None, :] - ball_field) * 3.0)

    # [지표·선점] 위협 상대 = 우리 골쪽 침투 깊이 + 골로 달리는 속도 가점. 상대를 0.4s 선점(o_vel)해
    # '지금 위치'가 아니라 '곧 있을 위치'를 마킹 → 런에 뒷북치지 않는다(상대속도 활용의 핵심).
    opp_fut = others_field + o_vel * 0.25                                       # 선점 위치
    toward_own = jnp.clip(-o_vel[:, :, 0], 0.0, 8.0)                            # 우리 골(-x) 방향 이동속도
    opp_threat = jnp.where(opp & (~o_gk), -opp_fut[:, :, 0] + 0.4 * toward_own, -jnp.inf)
    has_threat = jnp.any(opp & (~o_gk), axis=1)
    mark_j = jnp.argmax(opp_threat, axis=1)
    mark_pos = jnp.take_along_axis(opp_fut, mark_j[:, None, None], axis=1)[:, 0, :]   # 선점 위치 마킹
    mark_target = mark_pos + _unit(own_goal[None, :] - mark_pos) * 2.0
    mark_goalside = (-mark_pos[:, 0]) > (-ball_adv - 3.0)
    # 마킹 클레임: 그 위협에 '내가 최근접 동료(GK 제외)'일 때만 마킹 — 전원이 동일 argmax
    # 타깃(최심 침투자)으로 수렴해 수비 라인이 통째로 무너지던 것을 1명 전담으로 제한.
    # 나머지 수비수는 컴팩트 라인(def_target)을 유지해 서로 간의 진영이 보존된다.
    d_mark_me = jnp.linalg.norm(mark_pos - my_field, axis=1)
    d_mark_mates = jnp.linalg.norm(others_field - mark_pos[:, None, :], axis=2)
    claim_mark = (
        d_mark_me
        <= jnp.min(jnp.where(mate & (~o_gk), d_mark_mates, jnp.inf), axis=1)
        + GEOMETRY_EPS
    )
    do_mark = defending & (~press) & (~is_gk) & has_threat & mark_goalside & claim_mark

    line_x = jnp.clip(ball_adv - 5.0 + line_h * 12.0, -c.hx + 12.0, 12.0)
    def_x = jnp.clip(
        c.home_att[:, 0] * policy.defense_x_home_weight
        + line_x * policy.defense_x_line_weight,
        -c.hx + 10.0,
        14.0,
    )
    # 수비 블록은 홈 y서열을 보존하되 좌우로 압축하고(과대폭 금지) 공 쪽으로 함께 슬라이드 —
    # 실제 수비는 컴팩트 유닛으로 볼사이드 이동. 홈 y비중을 낮추고 볼 추종을 올려 블록 폭을
    # ~46m→~36m로 조인다(예전 0.85/0.2는 사이드라인까지 벌어져 선수 간 간극이 컸다).
    def_y = (
        c.home_att[:, 1] * policy.defense_y_home_weight
        + ball_field[:, 1] * policy.defense_y_ball_weight
    )
    def_target = jnp.stack([def_x, def_y], axis=1)
    def_target = jnp.where(do_mark[:, None], mark_target, def_target)
    def_target = jnp.where(press[:, None], press_target, def_target)
    d_def = jnp.linalg.norm(def_target - my_field, axis=1)
    def_pow = jnp.where(press, 0.97, jnp.where(do_mark, 0.82,
              jnp.clip(0.5 + 0.5 * d_def / 12.0, 0.5, 0.92)))

    # ── 국면 루즈볼(무점유): 최근접 필드 선수가 공으로 직행 ─────────────────
    # 별도 불리언 없이 아래 target 결합의 else 가지(~attacking & ~defending)가 곧 루즈볼이다.
    chase_target = ball_field + ball_vel[:, :2] * 0.3
    loose_target = jnp.where(am_nearest_field[:, None], chase_target, off_target)
    loose_pow = jnp.where(am_nearest_field, 0.98, 0.5)

    # ── 국면 통합(오픈플레이) ───────────────────────────────────────────────
    target = jnp.where(attacking[:, None], attack_target,
             jnp.where(defending[:, None], def_target, loose_target))
    move_pow = jnp.where(attacking, attack_pow, jnp.where(defending, def_pow, loose_pow))

    # ── 공중볼 경합: 최근접 필드 선수는 '떨어질 곳'으로 전력 질주 → 헤더/발리 다툼(env가 접촉 높이로 헤더 판정) ──
    aerial_go = am_nearest_field & airborne & ball_alive & (~restart_active)
    target = jnp.where(aerial_go[:, None], ball_landing, target)
    move_pow = jnp.where(aerial_go, 0.99, move_pow)

    kick_dir = jnp.where(am_nearest_mate[:, None] & attacking[:, None], c_kick_dir,
                         _unit(opp_goal[None, :] - my_field))
    # 비캐리어 킥(압박 탈취 클리어 등)도 솔버로 전방 롱 아크 — 임의 0.6/저각 대신 정밀 착지.
    kick_pow = jnp.where(am_nearest_mate & attacking, c_kick_pow, loft_pow_of(jnp.clip(d_goal, 18.0, 40.0)))
    launch01 = jnp.where(am_nearest_mate & attacking, c_launch, c.loft_launch01)
    spin_s = jnp.where(am_nearest_mate & attacking, c_spin_s, 0.0)
    spin_b = jnp.where(am_nearest_mate & attacking, c_spin_b, 0.0)
    # 부킹된 선수는 압박 태클(공 다툼 킥)을 안 한다 — 파울→2차 경고→퇴장의 자충수 방지.
    do_kick = ((am_nearest_mate & attacking & carrier_kick) | (press & in_reach & (~booked))) & f2b_avail

    # ── 국면 A: 세트피스 오버라이드 ─────────────────────────────────────────
    # 키커(재개팀 & in_reach & 팀내 공 최근접)는 골/열린 동료로 킥(env가 setup 완료 시 실제 발사).
    # 나머지는 배치·후퇴.
    # 페널티는 IFAB Law 14대로 키커 빼고 '양 팀 전원'이 박스+아크 밖 — 이격 의무를 양 팀에 걸고,
    # 벗어난 뒤에도 여유 밴드(1~3m)에선 제자리 홀드해 home_att(박스 안)로 재진입하는 진동을 끊는다.
    # (진동 → 킥 순간 상시 침범 → 결과의존 재실행+카드 무한 루프의 원인이었음.)
    # 이격 여유 margin은 obs에서 `Engine.clear_dist`로 정규화돼 들어오므로 설정 변경을 자동 반영한다.
    is_corner = rk[:, RK_CORNER] > 0.5
    is_gkk = rk[:, RK_GOALKICK] > 0.5
    is_throw = rk[:, RK_THROWIN] > 0.5
    is_pen = rk[:, RK_PENALTY] > 0.5
    # 키커 판정은 '재개팀 & in_reach'만으로는 과대 — 스폿 근처 동료(파울 당한 선수 등)까지
    # 키커로 오인해 pen_park 면제 + 공 위 주차(침범 기록 → 재실행 루프 재개방)가 된다.
    # 지정 키커는 env가 공 뒤에 핀하므로 '팀 내 공 최근접'(taker 포함 전 동료 대비)으로 좁힌다.
    mate_ball_all = jnp.where(mate, d_ball_o, jnp.inf)
    am_ball_nearest_all = (
        ball_dist <= jnp.min(mate_ball_all, axis=1) + GEOMETRY_EPS
    )
    is_kicker = restart_active & (is_sp_ours > 0.5) & in_reach & am_ball_nearest_all
    am_taker = is_kicker | (obs[:, c.i_kick_lock] > 0.5)             # 지정 키커(접근 중 포함)
    encroaching = (restart_active & (~am_taker) & (~is_pen) & (is_sp_ours < 0.5)
                   & (enc_margin < 0.055))
    # 페널티 합법 대기점: 박스 전면 1.2m 밖(x) + 아크(9.15m) 회피 |y| 확보. '공 반대' 후퇴는
    # 스폿-골 사이 수비수를 박스 안쪽(자기 골라인)으로 밀어 넣으므로 직사각형+아크 구역엔
    # 대기점 직행이 맞다. GK는 전체 제외 — 수비GK는 env 면제(골라인 합법), 공격GK는 아래 GK
    # 오버라이드가 자기 골문(합법 위치)으로 보낸다.
    bs = jnp.where(ball_field[:, 0] >= 0, 1.0, -1.0)
    wait_x = bs * (c.hx - 17.7)
    y_min = jnp.sqrt(jnp.clip(9.65 ** 2 - (ball_field[:, 0] - wait_x) ** 2, 0.0, None))
    ys = jnp.where(my_field[:, 1] >= 0, 1.0, -1.0)
    wait_y = jnp.clip(jnp.where(jnp.abs(my_field[:, 1]) < y_min, ys * y_min, my_field[:, 1]),
                      -c.hy + 1.0, c.hy - 1.0)
    pen_wait = jnp.stack([wait_x, wait_y], axis=1)
    pen_park = restart_active & is_pen & (~am_taker) & (~is_gk) & (enc_margin < 0.33)
    # 일반 세트피스도 같은 진동이 생긴다(9.15m 원 밖 0.5m까지 후퇴 → home_att로 재진입 →
    # 킥 순간 반쯤 안쪽 → retake 체인). 이격 의무자(상대팀)는 경계 밖 밴드(margin<0.16 —
    # clear_r 정규화라 FK류 ≈1.5m, 스로인 ≈0.3m)에서 제자리 홀드.
    fk_hold = (restart_active & (~is_pen) & (~am_taker) & (is_sp_ours < 0.5) & (~is_gk)
               & (~encroaching) & (enc_margin < 0.16))
    retreat_dir = _unit(my_field - ball_field)

    long_opt = direct > 0.45
    # 골킥 정밀화: 열린 동료가 있으면 그 동료 거리로 솔버 정밀 착지, 롱성향·무옵션이면 다운필드 측면
    # 인-피치 지점으로 loft. 파워·발사각을 솔버 모드(loft/drive)에 정확히 페어링해 목표에 떨어뜨린다.
    gk_far_tgt = jnp.stack([jnp.clip(my_field[:, 0] + 42.0, -c.hx + 5.0, c.hx - 8.0),
                            jnp.sign(my_field[:, 1] + DIV_EPS) * (c.hy - 14.0)], axis=1)
    d_gk_far = jnp.linalg.norm(gk_far_tgt - my_field, axis=1)
    gkk_long = is_gkk & (long_opt | (~has_pass))                # 롱 골킥(무옵션 포함)
    gkk_short_loft = is_gkk & (~gkk_long) & (d_pass > 22.0)     # 숏 골킥이지만 먼 동료 → 아크
    gkk_dir = jnp.where(gkk_long[:, None], _unit(gk_far_tgt - my_field), dir_pass)
    gkk_pow = jnp.where(gkk_long, loft_pow_of(d_gk_far),
              jnp.where(gkk_short_loft, loft_pow_of(d_pass), drive_pow_of(d_pass)))
    gkk_launch = jnp.where(gkk_long | gkk_short_loft, c.loft_launch01, c.drive_launch01)
    # 방향: 페널티=슛, 골킥=gkk, 그 외 짧은 재개(코너·스로·킥오프·FK)=열린 동료(dir_pass) 우선.
    sp_kick_dir = jnp.where(is_pen[:, None], dir_shot,
                  jnp.where(is_gkk[:, None], gkk_dir,
                  jnp.where(has_pass[:, None], dir_pass, _unit(opp_goal[None, :] - my_field))))
    # 파워·발사각을 종류별 솔버에 페어링: 페널티=슛, 코너=loft(아크), 스로=손던지기, 킥오프/FK=drive(지상).
    d_sp = jnp.clip(d_pass, 4.0, 40.0)                             # 동료까지 거리(클램프)
    sp_kick_pow = jnp.where(is_pen, (c.shot_pow if policy.use_shot_solver else 0.9),
                  jnp.where(is_gkk, gkk_pow,
                  jnp.where(is_corner, loft_pow_of(d_sp),
                  jnp.where(is_throw, throw_pow_of(d_sp), drive_pow_of(d_sp)))))
    sp_launch = jnp.where(is_pen, (shot_launch if policy.use_shot_solver else 0.04),
                jnp.where(is_gkk, gkk_launch,
                jnp.where(is_corner, c.loft_launch01,
                jnp.where(is_throw, c.throw_launch01, c.drive_launch01))))

    # 간접 FK(IFAB Law 13, is_fk_indirect): 직접골 무효 → 키커는 골 조준 금지, 반드시 동료로 연결.
    # dir_pass는 항상 어떤 동료를 가리키므로 골대 직격을 구조적으로 막는다(백패스 IDFK·재터치 IDFK 포함).
    is_fk_indirect = obs[:, c.i_fk_indirect] > 0.5
    sp_kick_dir = jnp.where(is_fk_indirect[:, None], dir_pass, sp_kick_dir)
    sp_kick_pow = jnp.where(is_fk_indirect, pass_pow, sp_kick_pow)       # drive(솔버)
    sp_launch = jnp.where(is_fk_indirect, c.drive_launch01, sp_launch)   # drive와 페어링

    # GK 캐치/홀드 배급(RK_GK_HOLD): 홀드 만료 전 GK가 배급. 롱볼 성향·무옵션이면 전방 측면 인-피치로
    # 사거리 역산 롱 클리어, 아니면 열린 동료로 짧게(사이드암/발배급). 약한 골정면 poke를 대체.
    is_hold = rk[:, RK_GK_HOLD] > 0.5
    gk_dist_long = jnp.stack([jnp.clip(my_field[:, 0] + 45.0, -c.hx + 5.0, c.hx - 10.0),
                              jnp.sign(my_field[:, 1] + DIV_EPS) * (c.hy - 12.0)], axis=1)
    gk_dist_long_opt = long_opt | (~has_pass)
    d_gk_dist = jnp.linalg.norm(gk_dist_long - my_field, axis=1)
    sp_kick_dir = jnp.where(is_hold[:, None],
                            jnp.where(gk_dist_long_opt[:, None], _unit(gk_dist_long - my_field), dir_pass),
                            sp_kick_dir)
    sp_kick_pow = jnp.where(is_hold,
                            jnp.where(gk_dist_long_opt, loft_pow_of(d_gk_dist), pass_pow),  # 솔버 정밀
                            sp_kick_pow)
    sp_launch = jnp.where(is_hold, jnp.where(gk_dist_long_opt, c.loft_launch01, c.drive_launch01), sp_launch)

    target = jnp.where(is_kicker[:, None], ball_field, target)          # 키커는 공으로(env가 강제이동)
    target = jnp.where((restart_active & (~is_kicker) & (~encroaching) & (~pen_park))[:, None],
                       c.home_att, target)
    target = jnp.where(pen_park[:, None], pen_wait, target)             # 페널티: 합법 대기점 직행
    target = jnp.where(fk_hold[:, None], my_field, target)              # 경계 밖 밴드 제자리 홀드
    target = jnp.where(encroaching[:, None], my_field + retreat_dir * 6.0, target)
    move_pow = jnp.where(restart_active,
                         jnp.where(encroaching | (pen_park & (enc_margin < 0.11)), 0.95, 0.7),
                         move_pow)
    kick_dir = jnp.where(is_kicker[:, None], sp_kick_dir, kick_dir)
    kick_pow = jnp.where(is_kicker, sp_kick_pow, kick_pow)
    launch01 = jnp.where(is_kicker, sp_launch, launch01)
    # 세트피스 킥은 스핀 커맨드를 쓰지 않는다 — 사이드스핀만 0으로 두고 백스핀을 남기면
    # 슛/크로스 분기에서 계산된 c_spin_b(-0.4/+0.4)가 키커에게 그대로 새어 든다(비대칭 누락).
    spin_s = jnp.where(is_kicker, 0.0, spin_s)
    spin_b = jnp.where(is_kicker, 0.0, spin_b)
    do_kick = jnp.where(restart_active, is_kicker, do_kick)             # 세트피스 중엔 키커만 킥 시도

    # ── GK: 스위퍼-키퍼 — 각도수비 + 위협 예측 시 조기 진출해 인터셉트·소유(캐치), 필요시만 클리어 ──
    own_gx = own_goal[0]
    bvx = ball_vel[:, 0]
    ball_vxy = ball_vel[:, :2]
    d_ball_goal = jnp.linalg.norm(ball_field - own_goal[None, :], axis=1) + DIV_EPS
    heading_goal = (bvx * jnp.sign(own_gx) > 0.5) & (d_ball_goal < 32.0)
    # 골라인 교차 예측(슛 커버 각도수비의 기본 위치)
    t_cross = jnp.clip((own_gx - ball_field[:, 0]) / jnp.where(jnp.abs(bvx) < 0.3, jnp.sign(own_gx) * 0.3, bvx), 0.0, 3.0)
    cross_y = jnp.clip(ball_field[:, 1] + ball_vel[:, 1] * t_cross, -c.goal_w * 0.5, c.goal_w * 0.5)
    gk_out = jnp.clip(0.6 + d_ball_goal * 0.045, 0.6, 4.0)
    angle_pos = own_goal[None, :] + _unit(ball_field - own_goal[None, :]) * gk_out[:, None]
    gk_y = jnp.where(heading_goal, cross_y, jnp.clip(angle_pos[:, 1], -c.goal_w * 0.5, c.goal_w * 0.5))
    gk_pos = jnp.stack([jnp.clip(own_gx + gk_out, own_gx + 0.3, own_gx + 4.5), gk_y], axis=1)

    # 공 미래 경로 예측(굴림+드래그 평균 감속 근사) → GK가 시간 내 닿는 최이른 인터셉트점.
    my_vmax = obs[:, c.i_vmax] * c.n_pvel                            # 자기 최고속(≈GK 이동속도)
    ts = jnp.linspace(0.08, 2.2, 12)                                # 미래 시각 샘플(s)
    bsp = jnp.linalg.norm(ball_vxy, axis=1) + DIV_EPS
    bhat = ball_vxy / bsp[:, None]
    s_stop = 0.5 * bsp ** 2 / policy.gk_prediction_decel             # 정지까지 이동거리
    s_t = jnp.minimum(jnp.maximum(
        0.0,
        bsp[:, None] * ts[None, :]
        - 0.5 * policy.gk_prediction_decel * ts[None, :] ** 2,
    ),
                      s_stop[:, None])
    ball_fut = ball_field[:, None, :] + bhat[:, None, :] * s_t[:, :, None]       # (N,12,2)
    d_gk_fut = jnp.linalg.norm(ball_fut - my_field[:, None, :], axis=2)          # (N,12)
    reach_by = d_gk_fut <= my_vmax[:, None] * ts[None, :] + 0.5     # GK가 그 시각까지 닿나(+마진)
    any_reach = jnp.any(reach_by, axis=1)
    first_k = jnp.argmax(reach_by, axis=1)
    intercept = jnp.take_along_axis(ball_fut, first_k[:, None, None], axis=1)[:, 0, :]
    # 조기 진출 판정: 상대보다 먼저 붙고(gk_wins) + 잡을 수 있는 속도(claimable). '소유=캐치'는
    # env상 자기 박스 안에서만 가능하므로 진출 목표를 박스 안으로 클리핑(claim_pt) — 밖으로 나가면
    # 클리어가 돼 소유 실패 + 골문 노출. 빠른 슛엔 스위핑하지 않고 라인 각도수비(gk_pos·cross_y) 유지.
    d_opp_int = jnp.min(jnp.where(opp, jnp.linalg.norm(others_field - intercept[:, None, :], axis=2), jnp.inf), axis=1)
    # GK는 '명확히 이길 때만' 스위핑(마진 -1.5m) — 경합 스루볼을 다 수거하면 전진 패스 공격이 전부
    # 죽는다(스위퍼 vs 전진패스 자기충돌). 확실히 먼저 닿는 공만 나가고, 50/50 볼은 공격수에게 넘겨 슛 기회.
    gk_wins_int = jnp.linalg.norm(intercept - my_field, axis=1) <= d_opp_int - 1.5
    ball_low = ball_z < 1.7
    claimable = ball_speed < c.gk_catch_cap * 0.9                   # 잡을 수 있는 속도(슛엔 라인 유지)
    claim_pt = jnp.stack([jnp.clip(intercept[:, 0], own_gx + 0.3, own_gx + c.pen_len - 1.0),
                          jnp.clip(intercept[:, 1], -c.pen_hw + 1.0, c.pen_hw - 1.0)], axis=1)  # 박스 안으로 제한
    # ★위협은 '공이 자기 골로 향하는가'(소유 무관)로 판정 — 우리 팀 백패스·굴절이 자기 골로 굴러가는
    # 자살골 상황을 소유(attacking) 게이트로 놓치던 버그 수정. 상대 슛뿐 아니라 아군 공도 위협이면 진출.
    ball_toward_own = ball_vel[:, 0] * jnp.sign(own_gx) > 0.3       # 자기 골 방향 이동
    threat = ball_toward_own | (~attacking)                        # 골로 향함 or 상대/루즈볼
    sweep = (is_gk & (~restart_active) & any_reach & ball_low & claimable & gk_wins_int
             & (ball_field[:, 0] < -c.hx * 0.30) & threat & (d_ball_goal < 24.0))
    # 근접 스매더: 아주 가까우면(<7m) 소유 무관 돌진, 골로 향하는 근거리는 잡을 수 있을 때만 진출.
    gk_rush = is_gk & (~restart_active) & ((d_ball_goal < 7.0)
                                           | (heading_goal & (d_ball_goal < 12.0) & claimable))
    gk_come = sweep | gk_rush                                       # 진출(스위퍼 or 최종 돌진)
    gk_active = is_gk & (~is_kicker)
    gk_target = jnp.where(gk_rush[:, None], intercept, jnp.where(sweep[:, None], claim_pt, gk_pos))
    target = jnp.where(gk_active[:, None], gk_target, target)
    move_pow = jnp.where(gk_active, jnp.where(gk_come, 1.0, 0.92), move_pow)

    # 소유 vs 클리어: 자기 박스 안 + 잡을 수 있는 속도면 '캐치'(do_kick=False → env gk_claim이 홀드).
    # 박스 밖이거나 못 잡을 만큼 빠르면 솔버로 인-피치 정밀 클리어.
    gk_in_box = (my_field[:, 0] < own_gx + c.pen_len) & (jnp.abs(my_field[:, 1]) < c.pen_hw)
    gk_reach_ball = is_gk & in_reach & ball_alive & (cooldown <= 0) & (~restart_active)
    # ★우리 팀이 마지막으로 찬 공(백패스·굴절)은 env가 '손 캐치'를 금지한다 → 캐치 시도(do_kick=False)하면
    # GK가 아무것도 못 하고 공이 골로 굴러 들어간다(자살골 원인). 그런 공은 '발 클리어'(do_kick=True)로 처리.
    gk_ball_ours = obs[:, c.i_last_touch] > 0.5                     # 직전 터치가 우리 팀(백패스/굴절)
    gk_catch = (gk_reach_ball & gk_in_box & (ball_speed < c.gk_catch_cap * 0.95)
                & (~gk_ball_ours))                                 # 상대·중립 공만 캐치(env 규칙 정합)
    gk_clear = gk_reach_ball & (~gk_catch)                          # 박스밖·고속·아군 백패스 → 발 클리어
    clear_tgt = jnp.stack([jnp.clip(my_field[:, 0] + 45.0, -c.hx + 5.0, c.hx - 10.0),
                           jnp.sign(my_field[:, 1] + DIV_EPS) * (c.hy - 12.0)], axis=1)
    d_clear = jnp.linalg.norm(clear_tgt - my_field, axis=1)
    kick_dir = jnp.where(gk_clear[:, None], _unit(clear_tgt - my_field), kick_dir)
    kick_pow = jnp.where(gk_clear, loft_pow_of(d_clear), kick_pow)  # 솔버 정밀 클리어(θ=loft)
    launch01 = jnp.where(gk_clear, c.loft_launch01, launch01)
    spin_s = jnp.where(gk_clear, 0.0, spin_s)
    spin_b = jnp.where(gk_clear, 0.0, spin_b)
    # GK의 킥 의도: 클리어만 True(캐치는 False라 env가 잡는다). GK는 여기서 완전히 결정 — 이후 미덮음.
    do_kick = jnp.where(is_gk & (~restart_active), gk_clear, do_kick)

    # ── 재터치 가드: 방금 세트피스/스로인 찬 선수는 공 반대로 물러나고 킥 금지 ──
    move_dir = _unit(target - my_field)
    move_dir = jnp.where(retouch[:, None], retreat_dir, move_dir)
    do_kick = do_kick & (~retouch)

    # ── 캐리어 드리블: 킥 안 하는 캐리어는 공 지나 3m로 몰고 감(오버런 자동 회수) ──
    is_dribbling = am_nearest_mate & attacking & carrier_drib & (~do_kick) & (~restart_active) & (~is_gk)
    move_dir = jnp.where(is_dribbling[:, None], _unit(drib_target - my_field), move_dir)

    # ── 인입 볼 트래핑/헤더: 빠른 지상볼은 소프트 트래핑, 공중볼은 그 자리서 헤더/발리(env가 접촉높이로 판정) ──
    # 글루가 없어 굴러오는 패스/루즈볼은 능동 트래핑, 떨어지는 공은 최근접이 그대로 때려 경합.
    trap_now = (am_nearest_field & ball_alive & in_reach & (~restart_active) & (~retouch)
                & (cooldown <= 0) & ((~settled) | airborne)
                & (~(am_nearest_mate & attacking & shoot_ok)))     # 첫 터치 슛하는 캐리어는 트랩 대신 슛
    head_dir = jnp.where((ball_adv > c.hx / 3.0)[:, None], _unit(opp_goal[None, :] - my_field),
                         jnp.stack([jnp.ones(N), jnp.zeros(N)], axis=1))     # 공격진영 헤더는 골 겨냥
    kick_dir = jnp.where(trap_now[:, None], head_dir, kick_dir)
    kick_pow = jnp.where(trap_now, jnp.where(airborne, 0.5, 3.0 / c.f2b_max), kick_pow)  # 공중볼은 세게(헤더/발리)
    launch01 = jnp.where(trap_now, jnp.where(airborne, 0.15, 0.0), launch01)
    spin_s = jnp.where(trap_now, 0.0, spin_s)
    spin_b = jnp.where(trap_now, 0.0, spin_b)
    do_kick = do_kick | trap_now

    # ── anti-clump: 공 미커밋 오프더볼은 동료 반발을 이동방향에 강하게 블렌딩(넓게 벌리기) ──
    # 세트피스 홀드/후퇴 중인 선수도 committed — 반발 블렌딩이 홀드 타겟을 표류시키면
    # 제한구역 경계 재진입 진동(retake 루프)이 재발한다.
    committed = (am_nearest_field | press | is_kicker | is_gk | do_kick | retouch | is_dribbling
                 | trap_now | fk_hold | pen_park | encroaching)
    rel_tm = my_field[:, None, :] - others_field
    dist_tm = jnp.linalg.norm(rel_tm, axis=2) + DIV_EPS
    near_tm = mate & (dist_tm < policy.anti_clump_radius)
    push = jnp.sum(jnp.where(near_tm[:, :, None], rel_tm / dist_tm[:, :, None]
                             * (1.0 - dist_tm / policy.anti_clump_radius)[:, :, None], 0.0), axis=1)
    move_dir = jnp.where(
        (~committed)[:, None],
        _unit(move_dir + policy.anti_clump_gain * push),
        move_dir,
    )

    # ── 실행 노이즈(킥 각오차) + 액션 조립(공격 프레임) ───────────────────────
    # 슛은 노이즈를 대폭 줄여 겨냥한 구석에 정확히 꽂는다(피니싱 — GK 리치 밖 배치).
    noise_scale = jnp.where(shoot, 0.012, 0.04 + 0.0028 * ball_dist)
    ang = jax.random.normal(k_noise, (N,)) * noise_scale
    cs, sn = jnp.cos(ang), jnp.sin(ang)
    kick_dir = jnp.stack([cs * kick_dir[:, 0] - sn * kick_dir[:, 1],
                          sn * kick_dir[:, 0] + cs * kick_dir[:, 1]], axis=1)

    # 이동·킥 = L∞ radial stretch 2D(방향+크기를 한 벡터로). env _decode의 stretch_decode와 역쌍.
    mv_v = stretch_encode(move_dir, jnp.clip(move_pow, 0.0, 1.0))       # (N,2)
    kick_v = stretch_encode(kick_dir, jnp.clip(kick_pow, 0.0, 1.0))     # (N,2)
    return jnp.stack([
        jnp.where(do_kick, 1.0, 0.0),        # 킥 게이트 dim[0]∈[0,1](0.5 초과=킥): 킥 1.0 / 미킥 0.0
        mv_v[:, 0], mv_v[:, 1],
        kick_v[:, 0], kick_v[:, 1],
        2.0 * jnp.clip(launch01, 0.0, 1.0) - 1.0,
        jnp.clip(spin_s, -1.0, 1.0), jnp.clip(spin_b, -1.0, 1.0),
    ], axis=1)


if __name__ == "__main__":
    from env import SoccerEnv
    env = SoccerEnv(game_duration=3000, control_fps=25)

    policy = make_rule_based_policy(env, match_key=jax.random.PRNGKey(0),
                                    team_styles=("gegenpress", "park_the_bus"))
    print("team_styles=\n", np.asarray(policy.team_styles))

    key = jax.random.PRNGKey(1)
    obs, state = env.reset_array(key)

    @jax.jit
    def step(carry, k):
        obs, st = carry
        k_pol, k_env = jax.random.split(k)
        act = policy(obs, k_pol)
        obs2, st2, rew, done, info = env.step_env_array(k_env, st, act)
        return (obs2, st2), (st2.poss_team, st2.score, st2.ball_state)

    (obs, state), (poss, score, bstate) = jax.lax.scan(step, (obs, state), jax.random.split(key, 1500))
    alive_frac = float((bstate == BALL_ALIVE).mean())
    print(f"1500 steps: final score={state.score.tolist()} "
          f"ball-alive frac={alive_frac:.2f} NaN={bool(jnp.isnan(state.ball_pos).any())}")
    print("possession split:", [int((poss == t).sum()) for t in (-1, 0, 1)])
