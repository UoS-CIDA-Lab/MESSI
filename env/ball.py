"""공 물리 — 자유 비행/굴림/바운스(ball_step_only)와 공-몸통 수동 충돌·트래핑(_ball_body).

두 경로 모두 궤적에서 역산 가능하게 설계: 자유물리는 완전 결정적(입력 pos/vel/spin → 출력 유일),
몸통 충돌은 '결정적 propensity + 분리된 단일 난수 2개(관통 hazard·트래핑)'라 관측 결과
(트랩=발밑 감속 정착(잔여속도 10%) / 바운스=반사 / 통과=불변)만으로 어느 분기였는지 특정된다.
"""
import jax
import jax.numpy as jnp

from constants import *
from spatial import _safe_norm, _unit


class BallPhysics:
    def ball_step_only(self, pos, vel, spin=None):
        """공 단독 1서브스텝(선수 무관, 완전 결정적) → (pos, vel, spin).

        지면 접촉 밴드(z ≤ z_ground) 안에서는 '지면에 붙은' 것으로 보고 굴림 마찰만 적용하고
        공기저항은 끈다. 밴드 안 저속 하강(< ground_settle_vz)은 바운스 대신 정착(z→r_ball, vz→0)
        시켜 중력↔반발 미세진동을 제거한다 — 반발계수 e_rest는 실제 임팩트에만 쓴다.
        """
        e_cfg = self.e_cfg
        dt = e_cfg.dt_phys
        if spin is None:
            spin = jnp.zeros(DIM_ALL)

        airborne = pos[DIM_Z] > e_cfg.z_ground   # 지면 판정과 동일 임계(공기/지면 경계 seam 제거 — ~airborne=grounded)
        drag = jnp.where(airborne, -e_cfg.c_drag * _safe_norm(vel) * vel, jnp.zeros(DIM_ALL))
        magnus = jnp.where(airborne, e_cfg.c_magnus * jnp.cross(spin, vel), jnp.zeros(DIM_ALL))
        # 지상 굴림 컬: 측스핀(spin_z)이 접촉면 마찰·측면슬립으로 굴러가는 공을 횡으로 휘게 함
        # (바깥발 감아차기 그라운드 패스). 공중 Magnus와 별개 메커니즘 → **지면(~airborne)·수평만**,
        # spin_z만 사용(백/톱스핀은 바운스·굴림에 반영, 횡휨 아님). cross([0,0,spin_z],vel)=(-sz·vy, sz·vx, 0)
        # 이라 z성분 자동 0. 계수 c_ground_curl은 DFL 지상패스 곡률 calib(fit_ground_curl.py).
        spin_z_only = jnp.array([0.0, 0.0, spin[DIM_Z]])
        ground_curl = jnp.where(airborne, jnp.zeros(DIM_ALL),
                                e_cfg.c_ground_curl * jnp.cross(spin_z_only, vel))
        accel = jnp.array([0.0, 0.0, -e_cfg.g]) + drag + magnus + ground_curl
        vel = vel + dt * accel
        pos = pos + dt * vel

        grounded = pos[DIM_Z] <= e_cfg.z_ground
        falling = grounded & (vel[DIM_Z] < 0)
        bounce = falling & (vel[DIM_Z] < -e_cfg.ground_settle_vz)   # 실제 임팩트만 반발
        settle = falling & (~bounce)                               # 저속 하강 = 지면 정착

        # 임팩트 시 스핀↔병진 접선 결합 — **각운동량 보존 접선 반발 모델**(에너지 비창출).
        # 톱/백스핀의 접촉점(바닥 −r ẑ) 표면 접선속도 기여 u_spin = ω×(−r ẑ)|xy = r·[−spin_y, spin_x].
        # 하나의 마찰 임펄스가 병진 Δv와 스핀 Δω를 **커플링**해 바꾼다(구식: 병진에 rω 통째 전달 +
        # 스핀 고정 50% 소산 → 각운동량 비보존·에너지 창출(창출/소거 2.6배)이었음):
        #   Δv = −(1+e_t)·α/(1+α)·u_spin,  Δω = [Δv_y, −Δv_x]/(α·r)  (접촉점 각운동량 보존).
        # |Δv| ≤ (1+e_t)·α/(1+α)·|u_spin| ≤ superball 상한(e_t=1). e_t=0이면 공이 굴러 나간다(sticking).
        impact = grounded & (vel[DIM_Z] < -e_cfg.bounce_spin_vmin)
        alpha = e_cfg.ball_inertia_ratio
        u_spin = self.r_ball * jnp.array([-spin[DIM_Y], spin[DIM_X]])
        dv = -(1.0 + e_cfg.bounce_tangential_e) * (alpha / (1.0 + alpha)) * u_spin
        dspin = jnp.array([dv[DIM_Y], -dv[DIM_X]]) / (alpha * self.r_ball)
        vel = vel.at[:DIM_Z].add(jnp.where(impact, dv, jnp.zeros(DIM_Z)))
        spin = spin.at[:DIM_Z].add(jnp.where(impact, dspin, jnp.zeros(DIM_Z)))

        # 임팩트 수평 마찰: 실측 총 ~13% 손실 중 같은 서브스텝 굴림 감속분을 제외한 ~11%
        # (fit_bounce, config bounce_h_keep 참조) — settle 제외 실임팩트만.
        vel = vel.at[:DIM_Z].multiply(jnp.where(bounce, e_cfg.bounce_h_keep, 1.0))
        vel = vel.at[DIM_Z].set(jnp.where(bounce, -e_cfg.e_rest * vel[DIM_Z],
                                          jnp.where(settle, 0.0, vel[DIM_Z])))
        pos_z = jnp.where(settle, self.r_ball, jnp.maximum(pos[DIM_Z], self.r_ball))
        pos = pos.at[DIM_Z].set(pos_z)

        # 지면 굴림 감속 — 속도-감속 knot 테이블 구간별 선형보간(굴림→슬라이딩 전이 포함)
        ball_speed = _safe_norm(vel[:DIM_Z]) + DIV_EPS
        decel = jnp.interp(ball_speed, jnp.asarray(e_cfg.roll_v_knots), jnp.asarray(e_cfg.roll_d_knots))
        roll_scale = jnp.maximum(0.0, ball_speed - decel * dt) / ball_speed
        vel = vel.at[:DIM_Z].multiply(jnp.where(grounded, roll_scale, 1.0))

        spin = spin * (1.0 - e_cfg.spin_decay * dt)
        return pos, vel, spin

    def _ball_step(self, state):
        """State의 공을 ball_step_only로 1서브스텝 전진."""
        pos, vel, spin = self.ball_step_only(state.ball_pos, state.ball_vel, state.ball_spin)
        return state._replace(ball_pos=pos, ball_vel=vel, ball_spin=spin)

    def _ball_body(self, state, key, suppress=None, touch_before=None):
        """공-몸통 수동 충돌·트래핑(허벅지/몸통 높이 밴드).

        지면공(z < leg_top)은 다리 아래로 통과시켜 능동 경합(_apply_force2ball)에만 맡기고,
        leg_top~body_top 밴드에서 접근 중(closing>0)인 공만 가장 가까운 선수의 몸통 코어(body_r)와
        확률적으로 상호작용시킨다. 분기: 트래핑(발밑 감속 정착·소유 획득) / 바운스(반사) / 통과(불변).
        재터치 금지 키커는 트래핑(소유) 불가 — 단순 바운스(물리)는 허용.

        suppress: reconstruct용 추첨 봉쇄 pin(스칼라 bool). True면 hit 추첨을 '통과'로 고정
        (trap은 hit 종속이라 함께 봉쇄) — 관측에 없는 몸통 굴절이 공 궤적을 흔들지 않게.
        실측에서 실제로 몸에 맞은 굴절은 터치 검출→킥 역산으로 재생되므로 정보 손실 없음.
        추첨(uniform)은 그대로 소비해 RNG 열 불변. None/False면 포워드 불변.

        touch_before: 이 물리 서브스텝의 접촉 직전 ``state.touch`` 스냅샷. 몸통 충돌 배제를
        '이번 서브스텝에 새로 생긴 접촉'으로 한정해 배제 창이 ``decimation``에 종속되지 않게 한다.
        None이면 종전(컨트롤 스텝 누적) 동작.
        """
        e_cfg = self.e_cfg
        players = jnp.arange(self.N)
        ball_xy = state.ball_pos[:DIM_Z]
        ball_z = state.ball_pos[DIM_Z]
        rel = ball_xy[None, :] - state.player_pos
        dist_xy = jnp.linalg.norm(rel, axis=1)
        vel_xy_b = state.ball_vel[:DIM_Z]
        ball_speed3 = jnp.linalg.norm(state.ball_vel)
        to_player = _unit(-rel)
        closing = jnp.sum(vel_xy_b[None, :] * to_player, axis=1)

        body_top = e_cfg.body_top_frac * state.head_z
        in_band = (ball_z >= e_cfg.leg_top) & (ball_z < body_top)
        gate_r = e_cfg.body_r + self.r_ball
        # 스윕 판정(터널링 방지): 후보를 현재 위치가 아니라 이 서브스텝 공 경로(선분 pos→pos+v·dt)와
        # 선수의 최소거리로 잡는다. 풀파워 킥(34m/s)은 서브스텝당 0.34m > gate_r(0.26m)를 이동하므로
        # 이산 위치 샘플만 보면 강슛일수록 몸통 게이트를 통째로 건너뛴다.
        seg = vel_xy_b * e_cfg.dt_phys
        seg_len2 = jnp.dot(seg, seg) + SQUARED_EPS
        t_close = jnp.clip(jnp.sum((state.player_pos - ball_xy[None, :]) * seg[None, :], axis=1)
                           / seg_len2, 0.0, 1.0)
        closest = ball_xy[None, :] + t_close[:, None] * seg[None, :]
        sweep_dist = jnp.linalg.norm(state.player_pos - closest, axis=1)
        # 이번 **물리 서브스텝**에 이미 접촉한 선수 제외(같은 서브스텝 이중 상호작용 방지).
        # ``state.touch``는 컨트롤 스텝 단위로만 0 초기화되므로(env.step_env_array) 스냅샷 없이
        # ``touch > 0``만 보면 배제 창이 남은 서브스텝 전체로 늘어나 그 길이가 ``decimation``
        # (=control_fps)에 종속된다 — 컨트롤 레이트가 물리를 바꾸는 결함(AUDIT 미수정 관찰 2).
        # ``touch_before``(접촉 직전 스냅샷)와 비교해 이번 서브스텝에 새로 생긴 접촉만 배제한다.
        # ``restart._throwin_restriction``·``offside._offside_check``와 동일한 규약.
        # None이면 종전 동작(호환용).
        if touch_before is None:
            excluded = state.touch > TOUCH_NONE
        else:
            excluded = (state.touch > TOUCH_NONE) & (state.touch != touch_before)
        is_taker = (((players == state.setpiece_taker) & (state.setpiece_taker >= 0))
                    | ((players == state.throw_taker) & (state.throw_taker >= 0)))
        # 수직 낙하 케이스: 수평 접근이 없어도 밴드 안으로 떨어지는 공(가슴트랩 상황)은 후보에 포함.
        falling_hit = (state.ball_vel[DIM_Z] < -e_cfg.collide_speed_min) & (dist_xy < gate_r)
        candidate = ((sweep_dist < gate_r) & in_band & (~excluded) & state.active_player
                     & ((closing > 0) | falling_hit) & (ball_speed3 > e_cfg.collide_speed_min)
                     & (state.ball_state == BALL_ALIVE))
        idx = jnp.argmin(jnp.where(candidate, sweep_dist, jnp.inf))
        active = candidate[idx]
        team_i = state.team_id[idx].astype(jnp.int32)

        k_hit, k_trap = jax.random.split(key)
        # 서브스텝 hazard: 코어 정면 관통(경로≈gate_r)당 총 상호작용확률 = body_hit_prob.
        # 스치는 관통은 경로가 짧아 자동으로 더 낮다. 경로는 최대 풀 관통 현(2·gate_r)으로 캡.
        path = jnp.minimum(ball_speed3 * e_cfg.dt_phys, 2.0 * gate_r)
        p_hit = 1.0 - jnp.power(1.0 - e_cfg.body_hit_prob, path / gate_r)
        sup = jnp.bool_(False) if suppress is None else jnp.bool_(suppress)  # python True 방어(~True=-2 int화)
        hit = active & (~sup) & (jax.random.uniform(k_hit) < p_hit)
        p_trap = e_cfg.trap_base * state.player_ctrl[idx] * jnp.clip(1.0 - ball_speed3 / e_cfg.trap_speed_ref, 0.0, 1.0)
        trap = hit & (jax.random.uniform(k_trap) < p_trap) & (~is_taker[idx])
        bounce = hit & (~trap)

        # 반사 법선은 스윕 최근접점 기준(터널링 케이스에서도 올바른 접촉 방향).
        # 정면 관통(경로가 선수 중심을 지나 최근접점≈중심)은 법선이 퇴화하므로 입사 반대 방향으로 폴백.
        n_raw = closest[idx] - state.player_pos[idx]
        n_len = jnp.linalg.norm(n_raw)
        normal = jnp.where(
            n_len > GEOMETRY_EPS,
            n_raw / (n_len + DIV_EPS),
            -_unit(vel_xy_b[None, :])[0],
        )
        vel_xy = vel_xy_b
        # [버그수정] 입사 성분(v·n<0)만 반사. falling_hit 후보는 공이 몸에서 멀어지는 중(v·n>0)일 수 있어,
        # 그대로 반사하면 나가는 공을 안으로 당기는 비물리 발생 → jnp.minimum(v·n, 0)으로 나가는 성분 배제.
        vn = jnp.minimum(jnp.dot(vel_xy, normal), 0.0)
        vel_reflect = (
            vel_xy
            - (1.0 + e_cfg.e_body) * vn * normal
            + e_cfg.body_player_vel_transfer * state.player_vel[idx]
        )
        new_vel_xy = jnp.where(
            bounce,
            vel_reflect,
            jnp.where(trap, e_cfg.trap_velocity_keep * vel_xy, vel_xy),
        )
        new_vel_z = jnp.where(bounce, state.ball_vel[DIM_Z] * e_cfg.e_body,
                              jnp.where(trap, 0.0, state.ball_vel[DIM_Z]))
        new_vel = jnp.array([new_vel_xy[DIM_X], new_vel_xy[DIM_Y], new_vel_z])

        # 트랩은 선수 '앞'(접촉 방향 0.35m)에 내려놓는다 — 선수 중심에 겹쳐 놓으면 다음 서브스텝
        # 분리·경합이 비물리적으로 꼬인다. 몸통 접촉은 스핀을 크게 소산(트랩=0, 바운스=keep 비율).
        drop = state.player_pos[idx] + normal * e_cfg.trap_drop_distance
        new_pos = jnp.array([jnp.where(trap, drop[DIM_X], ball_xy[DIM_X]),
                             jnp.where(trap, drop[DIM_Y], ball_xy[DIM_Y]),
                             jnp.where(trap, self.r_ball, ball_z)])

        ball_pos = jnp.where(hit, new_pos, state.ball_pos)
        ball_vel = jnp.where(hit, new_vel, state.ball_vel)
        ball_spin = jnp.where(trap, jnp.zeros(DIM_ALL),
                    jnp.where(bounce, state.ball_spin * e_cfg.body_spin_keep, state.ball_spin))
        poss = jnp.where(trap, team_i, state.poss_team).astype(jnp.int32)
        last_touch = jnp.where(hit, team_i, state.last_touch_team).astype(jnp.int32)
        # 라벨 의미론: 상대 소유 공을 몸통 트랩 = 인터셉트(오프사이드 리셋 트리거와도 정합),
        # 자기팀 공 트랩 = 드리블(컨트롤). 바운스는 굴절(TOUCH_DEFLECT) — 스로인 재터치 해제 감지·
        # 오프사이드 간섭 판정이 touch 기록에 의존하므로 무기록이면 규칙 오검이 생긴다.
        trap_code = jnp.where((state.poss_team >= 0) & (team_i != state.poss_team),
                              jnp.int32(TOUCH_INTERCEPT), jnp.int32(TOUCH_DRIBBLE))
        new_code = jnp.where(trap, trap_code,
                   jnp.where(bounce, jnp.int32(TOUCH_DEFLECT), state.touch[idx]))
        touch = state.touch.at[idx].set(new_code)
        last_touch_code = jnp.where(hit, new_code, state.last_touch_code).astype(jnp.int32)
        return state._replace(ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                              poss_team=poss, last_touch_team=last_touch, touch=touch,
                              last_touch_code=last_touch_code)
