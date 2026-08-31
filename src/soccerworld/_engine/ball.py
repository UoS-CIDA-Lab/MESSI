"""공 물리 — 자유 비행/굴림/바운스(ball_step_only), 공-몸통 수동 충돌·트래핑(_ball_body),
골 프레임(포스트·크로스바) 반사(_goal_frame_bounce / 공개 goal_frame_impact).

프레임 반사는 ``ball_step_only`` **밖**에 둔다. 그 함수의 '완전 결정적 자유 비행' 계약이
역산과 rule_policy 사거리 캘리브의 근거라, 경기장 기하를 그 안에 섞으면 두 소비자가 함께
깨진다. 반사는 자유물리가 tick 끝점을 만든 **뒤** 적용되고, 발생 사실은 ``State.woodwork_*``에
남아 역산 하네스가 그 창을 알아볼 수 있다.

앞의 두 경로는 궤적에서 역산 가능하게 설계: 자유물리는 완전 결정적(입력 pos/vel/spin → 출력 유일),
몸통 충돌은 '결정적 propensity + 분리된 단일 난수 2개(관통 hazard·트래핑)'라 관측 결과
(트랩=발밑 감속 정착(잔여속도 10%) / 바운스=반사 / 통과=불변)만으로 어느 분기였는지 특정된다.
"""
import jax
import jax.numpy as jnp

from soccerworld.core.randomness import RandomEvent, select_random_key

from .constants import (
    BALL_ALIVE,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    GEOMETRY_EPS,
    GK_HANDLING_RELEASE_OFFSET,
    NO_TEAM,
    SQUARED_EPS,
    TEAM_0,
    TOUCH_BODY_TRAP,
    TOUCH_DEFLECT,
    TOUCH_NONE,
    WOODWORK_CROSSBAR,
    WOODWORK_NONE,
    WOODWORK_POST,
)
from .restart import _coerce_public_float32
from .spatial import _safe_norm, _unit


class BallPhysics:
    # Exact-response test subclasses disable only this static specialization.
    _specialize_ball_body_no_hit = True

    def ball_step_only(self, pos, vel, spin=None):
        """공 단독 1서브스텝(선수 무관, 완전 결정적) → (pos, vel, spin).

        지면 접촉 밴드(z ≤ z_ground) 안에서는 '지면에 붙은' 것으로 보고 굴림 마찰만 적용하고
        공기저항은 끈다. 밴드 안 저속 하강(< ground_settle_vz)은 바운스 대신 정착(z→r_ball, vz→0)
        시켜 중력↔반발 미세진동을 제거한다 — 반발계수 e_rest는 실제 임팩트에만 쓴다.
        """
        pos, _ = _coerce_public_float32("pos", pos, (DIM_ALL,))
        vel, _ = _coerce_public_float32("vel", vel, (DIM_ALL,))
        if spin is None:
            spin = jnp.zeros(DIM_ALL, dtype=jnp.float32)
        else:
            spin, _ = _coerce_public_float32("spin", spin, (DIM_ALL,))

        e_cfg = self.e_cfg
        dt = e_cfg.dt_phys

        airborne = pos[DIM_Z] > e_cfg.z_ground   # 지면 판정과 동일 임계(공기/지면 경계 seam 제거 — ~airborne=grounded)
        # A resting/rolling ball is supported by the ground.  Applying one
        # unconstrained gravity tick first and deciding whether it impacted
        # afterwards makes that support disappear when ``g * dt`` exceeds
        # ``ground_settle_vz``: a perfectly stationary ball then manufactures
        # a bounce solely because the configured physics tick is larger.  Use
        # the *pre-step* contact velocity to distinguish support from a real
        # incoming impact.  Upward motion is deliberately not supported -- a
        # ball that has just bounced must continue its ballistic flight.
        supported = (
            (~airborne)
            & (vel[DIM_Z] <= 0.0)
            & (vel[DIM_Z] >= -e_cfg.ground_settle_vz)
        )
        # Velocity-dependent forces are integrated below with bounded exact
        # updates; explicit Euler can reverse drag or make pure curl explode.
        # 지상 굴림 컬: 측스핀(spin_z)이 접촉면 마찰·측면슬립으로 굴러가는 공을 횡으로 휘게 함
        # (바깥발 감아차기 그라운드 패스). 공중 Magnus와 별개 메커니즘 → **지면(~airborne)·수평만**,
        # spin_z만 사용(백/톱스핀은 바운스·굴림에 반영, 횡휨 아님). cross([0,0,spin_z],vel)=(-sz·vy, sz·vx, 0)
        # 이라 z성분 자동 0. 계수 c_ground_curl은 DFL 지상패스 곡률 calib(fit_ground_curl.py).
        gravity_z = jnp.where(supported, 0.0, -e_cfg.g)
        vel_gravity = vel + dt * jnp.array([0.0, 0.0, gravity_z])

        # Quadratic drag has the closed-form damping v/(1+c|v|dt).
        # Magnus is the skew-symmetric flow dv/dt=(c*spin)×v, i.e. an
        # exact Rodrigues rotation.  Both remain bounded for every finite
        # non-negative public coefficient and retain the calibrated equation
        # to first order in dt.
        air_speed = _safe_norm(vel_gravity)
        air_damped = vel_gravity / (1.0 + e_cfg.c_drag * air_speed * dt)
        omega = e_cfg.c_magnus * spin
        omega_mag = _safe_norm(omega)
        axis = omega / omega_mag
        angle = omega_mag * dt
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        air_rotated = (
            air_damped * cos_a
            + jnp.cross(axis, air_damped) * sin_a
            + axis * jnp.dot(axis, air_damped) * (1.0 - cos_a)
        )

        # Ground curl is the same pure rotation restricted to xy and driven
        # only by spin_z: Jv=(-vy,vx).  Exact integration prevents the old
        # perpendicular Euler increment from manufacturing kinetic energy.
        ground_angle = e_cfg.c_ground_curl * spin[DIM_Z] * dt
        cos_g, sin_g = jnp.cos(ground_angle), jnp.sin(ground_angle)
        ground_xy = jnp.array([
            cos_g * vel_gravity[DIM_X] - sin_g * vel_gravity[DIM_Y],
            sin_g * vel_gravity[DIM_X] + cos_g * vel_gravity[DIM_Y],
        ])
        ground_rotated = vel_gravity.at[:DIM_Z].set(ground_xy)
        vel = jnp.where(airborne, air_rotated, ground_rotated)
        pos = pos + dt * vel

        grounded = pos[DIM_Z] <= e_cfg.z_ground
        falling = grounded & (vel[DIM_Z] < 0)
        bounce = falling & (vel[DIM_Z] < -e_cfg.ground_settle_vz)   # 실제 임팩트만 반발
        settle = falling & (~bounce)                               # 저속 하강 = 지면 정착

        # 임팩트 시 스핀↔병진 접선 결합. 관성비 α/(1+α)로 회전 표면속도를
        # 병진속도에 전달한 뒤 DFL7에서 보정한 수평 보존율을 적용한다.
        impact = grounded & (vel[DIM_Z] < -e_cfg.bounce_spin_vmin)
        alpha = e_cfg.ball_inertia_ratio
        u_spin = self.r_ball * jnp.array([-spin[DIM_Y], spin[DIM_X]])
        dv = (
            -(1.0 + e_cfg.bounce_tangential_e)
            * (alpha / (1.0 + alpha))
            * u_spin
        )
        dspin = jnp.array([dv[DIM_Y], -dv[DIM_X]]) / (
            alpha * self.r_ball
        )
        vel = vel.at[:DIM_Z].add(jnp.where(impact, dv, jnp.zeros(DIM_Z)))
        spin = spin.at[:DIM_Z].add(jnp.where(impact, dspin, jnp.zeros(DIM_Z)))

        vel = vel.at[:DIM_Z].multiply(
            jnp.where(bounce, e_cfg.bounce_h_keep, 1.0)
        )
        vel = vel.at[DIM_Z].set(jnp.where(bounce, -e_cfg.e_rest * vel[DIM_Z],
                                          jnp.where(settle, 0.0, vel[DIM_Z])))
        # Every supported sample in the numerical ground band represents the
        # same contact manifold.  Canonicalise it to r_ball; otherwise
        # z∈(r_ball,z_ground] with vz=0 hovers forever.
        pos_z = jnp.where(
            settle | supported,
            self.r_ball,
            jnp.maximum(pos[DIM_Z], self.r_ball),
        )
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

    def goal_frame_impact(self, pos_before, pos_after, vel, spin=None):
        """공개 골 프레임 판정 — ``(kind, contact_pos, pos, vel, spin)``.

        ``ball_step_only``은 **자유 비행 전용**이고 앞으로도 그렇다. 프레임 충돌은 그 뒤에
        오는 별도 분기이므로, 궤적 역산 하네스가 포워드를 그대로 미러링하려면 같은 분기를
        직접 부를 수 있어야 한다. 관측 구간 안에서 이 함수가 ``WOODWORK_NONE``이 아닌 값을
        내면 ``ball_step_only``만으로 한 역적분은 성립하지 않는다 — 그 창은 제외하거나
        접촉점에서 끊어야 한다. 런타임 기록은 ``State.woodwork_*``에 남는다.

        ``pos_before``→``pos_after``는 자유물리가 만든 한 물리 tick의 끝점이고, ``vel``·
        ``spin``은 그 tick 끝의 값이다(``_ball_step_after_body``가 넘기는 것과 같다).
        """

        pos_before, _ = _coerce_public_float32("pos_before", pos_before, (DIM_ALL,))
        pos_after, _ = _coerce_public_float32("pos_after", pos_after, (DIM_ALL,))
        vel, _ = _coerce_public_float32("vel", vel, (DIM_ALL,))
        if spin is None:
            spin = jnp.zeros(DIM_ALL, dtype=jnp.float32)
        else:
            spin, _ = _coerce_public_float32("spin", spin, (DIM_ALL,))
        if not self._goal_frame_active:
            return (
                jnp.int32(WOODWORK_NONE), pos_after, pos_after, vel, spin,
            )
        return self._goal_frame_bounce(pos_before, pos_after, vel, spin)

    def _goal_frame_capsule_entry(
        self, pos_before, seg, centers, perp_axes, along_axis, along_lo, along_hi,
    ):
        """축 정렬 원기둥 **몸통**에 대한 스윕 구-원기둥 진입 → ``(hit, s, offset)``.

        ``perp_axes``는 축에 수직인 두 좌표축 index, ``along_axis``는 축 방향이다.
        포스트는 z축(perp=xy), 크로스바는 y축(perp=xz) 원기둥이라 두 번 부른다.

        **축 범위는 점이 아니라 구간으로 교차시킨다.** 종전에는 무한 원기둥 진입근
        ``s_in`` **한 점**에서만 축좌표를 범위와 비교했는데, 이음매를 지나는 궤적은 진입
        순간 두 원기둥 모두 축좌표가 아직 범위 **밖**이다(포스트는 z가 ``along_hi`` 위,
        크로스바는 y가 span 밖). 공은 그 다음 세그먼트 중간에서 범위로 들어오지만 다시
        보는 곳이 없어 둘 다 탈락했다 — 상단 구석을 정확히 노린 슛이 통째로 관통했다.

        ``along(s)``는 s에 대해 선형이므로 ``[along_lo, along_hi]``를 s 구간 ``[sa, sb]``로
        풀어 ``[s_in, s_out]``·``[0, 1]``과 교차시키고, 교집합의 **시작**을 접촉 시점으로
        쓴다. 축방향 속도가 0이면 축좌표가 상수라 범위 안이면 전 구간 유효, 밖이면 무효다.

        끝면은 여기서 다루지 않는다 — 둥근 끝(모서리)은 ``_goal_frame_sphere_entry``가
        맡는다. 그래야 몸통 합집합에 남는 바깥 위 노치가 메워진다.
        """

        radius_sum = self.r_ball + self.e_cfg.goal_frame_radius
        i, j = perp_axes
        d = jnp.stack([seg[i], seg[j]])
        a0 = jnp.stack([pos_before[i], pos_before[j]])[None, :] - centers
        # 구성은 ``_ball_body``의 스윕 판정과 같다 — 두 조건을 각각 s 구간으로 만들고
        # 교차시킨다. 퇴화(변위 0)도 거기와 같이 **정적 포함 판정**으로 대체한다:
        # 수직 낙하하는 공은 수직 평면 변위가 0이라 '접촉 없음'으로 두면 포스트 판정에서
        # 통째로 빠진다(y가 post_y보다 바깥이면 크로스바도 못 잡아 둘 다 놓친다).
        quad_a = jnp.sum(d * d)
        perp_moving = quad_a > SQUARED_EPS
        quad_b = 2.0 * jnp.sum(a0 * d[None, :], axis=-1)
        perp_dist2 = jnp.sum(a0 * a0, axis=-1)
        quad_c = perp_dist2 - radius_sum * radius_sum
        disc = quad_b * quad_b - 4.0 * quad_a * quad_c
        root = jnp.sqrt(jnp.maximum(disc, 0.0))
        denom = jnp.where(perp_moving, 2.0 * quad_a, 1.0)
        perp_lo = jnp.where(perp_moving, (-quad_b - root) / denom, 0.0)
        perp_hi = jnp.where(perp_moving, (-quad_b + root) / denom, 1.0)
        perp_valid = jnp.where(
            perp_moving,
            (disc > 0.0) & (perp_hi >= 0.0) & (perp_lo <= 1.0),
            perp_dist2 < radius_sum * radius_sum,
        )

        # 축 범위도 **점이 아니라 구간**으로 푼다. 종전에는 진입근 한 점에서만 축좌표를
        # 비교했는데, 이음매를 지나는 궤적은 진입 순간 포스트는 z가 범위 위·크로스바는
        # y가 span 밖이라 둘 다 탈락했다. 공은 세그먼트 중간에서 범위로 들어오는데 다시
        # 보는 곳이 없어, 상단 구석을 정확히 노린 슛이 통째로 관통했다.
        along_0 = pos_before[along_axis]
        along_d = seg[along_axis]
        along_moving = jnp.abs(along_d) > GEOMETRY_EPS
        inv = 1.0 / jnp.where(along_moving, along_d, 1.0)
        edge_lo = (along_lo - along_0) * inv
        edge_hi = (along_hi - along_0) * inv
        span_lo = jnp.where(along_moving, jnp.minimum(edge_lo, edge_hi), 0.0)
        span_hi = jnp.where(along_moving, jnp.maximum(edge_lo, edge_hi), 1.0)
        along_valid = jnp.where(
            along_moving,
            (span_hi >= 0.0) & (span_lo <= 1.0),
            (along_0 >= along_lo) & (along_0 <= along_hi),
        )

        lo = jnp.maximum(jnp.maximum(perp_lo, span_lo), 0.0)
        hi = jnp.minimum(jnp.minimum(perp_hi, span_hi), 1.0)
        # 시작부터 겹쳐 있으면 ``perp_lo`` < 0이라 lo가 0으로 잘린다 — 파고든 상태를
        # 다음 tick으로 넘기지 않고 이 tick에서 빼낸다.
        hit = perp_valid & along_valid & (lo <= hi)
        s = jnp.clip(lo, 0.0, 1.0)
        offset = a0 + s[:, None] * d[None, :]
        return hit, s, offset

    def _goal_frame_sphere_entry(self, pos_before, seg, centers):
        """스윕 구-구 진입 → ``(hit, s, offset)``. 프레임 모서리(둥근 끝면)를 맡는다.

        몸통 원기둥 둘만 합치면 바깥 위 모서리에 노치가 남는다 — 두 축 모두에서 반지름
        안이면서 두 축 범위는 모두 밖인 영역이다. 실제 골대 모서리는 둥글고, 포스트와
        크로스바를 **캡슐**로 보면 두 캡슐의 끝 반구가 같은 점 ``(±hx, ±post_y, bar_z)``에
        중심을 둔다. 그 구를 그대로 판정에 넣으면 합집합에 구멍이 없다.

        크로스바 span만 늘리는 대안은 틀렸다 — 참거리가 반경합 밖인 점
        (예: ``(hx+0.15, post_y+0.16, bar_z)``, 참거리 0.219)을 명중으로 잡는다.
        """

        radius_sum = self.r_ball + self.e_cfg.goal_frame_radius
        a0 = pos_before[None, :] - centers
        quad_a = jnp.sum(seg * seg)
        moving = quad_a > SQUARED_EPS
        quad_b = 2.0 * jnp.sum(a0 * seg[None, :], axis=-1)
        dist2 = jnp.sum(a0 * a0, axis=-1)
        quad_c = dist2 - radius_sum * radius_sum
        disc = quad_b * quad_b - 4.0 * quad_a * quad_c
        root = jnp.sqrt(jnp.maximum(disc, 0.0))
        denom = jnp.where(moving, 2.0 * quad_a, 1.0)
        entry = jnp.where(moving, (-quad_b - root) / denom, 0.0)
        exit_ = jnp.where(moving, (-quad_b + root) / denom, 1.0)
        # 정지 표본은 ``_ball_body``와 같이 정적 포함으로 판정한다.
        valid = jnp.where(
            moving,
            (disc > 0.0) & (exit_ >= 0.0) & (entry <= 1.0),
            dist2 < radius_sum * radius_sum,
        )
        lo = jnp.maximum(entry, 0.0)
        hi = jnp.minimum(exit_, 1.0)
        hit = valid & (lo <= hi)
        s = jnp.clip(lo, 0.0, 1.0)
        offset = a0 + s[:, None] * seg[None, :]
        return hit, s, offset

    def _goal_frame_bounce(self, pos_before, pos_after, vel, spin):
        """골 프레임(포스트·크로스바) 충돌 → ``(kind, contact_pos, pos, vel, spin)``.

        ``kind``는 :data:`WOODWORK_NONE`/``_POST``/``_CROSSBAR``이고, ``NONE``이면 나머지
        셋은 입력 그대로다. 한 물리 tick의 직선 구간을 6개 캡슐(포스트 4 + 크로스바 2)로
        스윕해 **가장 이른** 진입을 고른다. 40 m/s 공은 tick당 0.44 m를 지나가는데 프레임
        단면은 지름 0.12 m라, 끝점만 보는 판정으로는 그냥 관통한다.

        고체는 **캡슐 6개의 합집합**이다 — 몸통 원기둥 6개(포스트 4 + 크로스바 2)와 그
        둥근 끝면인 모서리 구 4개. 몸통만 쓰면 상단 구석에 노치가 남고, 모서리를 정확히
        지나는 슛이 통째로 관통한다(실측: 참거리 0.0000인데 미검출).

        **기하** — ``goal_width``/``goal_height``는 프레임 **안쪽** 면 치수라(IFAB Law 1),
        반지름 ``rf`` 포스트의 축은 ``y = ±(goal_w/2 + rf)``, 크로스바 축은 ``z = goal_h + rf``다.
        골문 안은 비어 있으므로 득점 궤적은 어떤 캡슐과도 만나지 않는다 — 골/아웃 판정에
        손대지 않고 프레임만 얹힌다.

        **반사** — ``ball_step_only``의 지면 바운스와 **같은 임펄스 모델**이고 법선만
        일반화했다. 접촉점은 공 중심에서 ``-r_ball·n``, 접촉점 슬립은 ``u = v_t - r_ball(ω×n)``:

            법선  v_n' = -e·v_n
            접선  Δv_t = -J_t·û,   J_t = min( α/(1+α)·|u| ,  μ·(1+e)·|v_n| )
            스핀  Δω   = -(n × Δv_t) / (α·r_ball)

        ``Δω``는 접촉점 각운동량을 보존한다 — n=ẑ를 넣으면 지면식과 항별로 같아진다.
        Coulomb 상한 ``μ·J_n``이 **방향 현실성의 핵심**이다. 이것이 없으면 스치는 충돌까지
        슬립이 완전히 죽는 sticking으로 풀려 실제보다 크게 꺾이고 스핀도 과하게 뒤집힌다.
        상한이 있으면 정면 충돌은 되튀어 나오고, 얕은 각도는 프레임을 타고 흐르며,
        감아 찬 공은 정반사와 다른 각으로 튄다.

        **알려진 한계: 한 tick에 반사는 한 번이다.** ``pos_out``은 접촉점에서 반사 속도로
        남은 시간만큼 직진시켜 만들고, 그 경로를 다시 프레임에 대해 검사하지 않는다. 포스트를
        맞고 튄 공이 같은 tick 안에서 크로스바로 들어가면 두 번째 반사가 **한 tick 늦는다**
        (계량: 3만 표본 중 1차 명중 26,996건, 그 중 반사 후 고체 안 2 cm 초과 침투 319건
        =1.18%, 6 cm 초과 199건. 전부 이음매 부근).

        **관통도 끼임도 진동도 아니다 — 실측으로 닫았다.** 침투한 319건만 뽑아 자유비행 한
        tick 전진 + 프레임 판정을 세 번 반복했다. 다음 tick에서 **219건이 실제로 발화**해
        놓쳤던 두 번째 반사가 그대로 일어나고, 그 tick 이후 **고체 안에 남은 표본은 0건**이다.
        이후 두 tick은 추가 접촉 0건이며 고체까지 거리 중앙값이 0.2982 → 0.4191 → 0.5412로
        단조 증가한다. 흡수 경로는 다음 tick 진입에서 ``perp_lo``가 음수가 되어 정적 포함으로
        잡히고 아래 push-out이 표면 밖으로 빼내는 것이다.

        실효 오차는 시간 11 ms·위치 6 cm이고, 이음매를 연달아 맞고 이상한 각도로 나오는 장면
        자체는 재현된다. 한 tick 안에서 풀려면 반복 반사 루프가 필요한데 고정 형상 스캔에
        루프를 넣는 비용이 이 오차보다 크다.

        **순서** — 이 반사는 ``_events``의 라인 통과 판정 **앞**에 있고 판정 구간의 시작점을
        접촉점으로 바꾼다. 크로스바 맞고 골라인을 넘는 공이 골이 되는 것은 그 덕분이다.
        포스트가 골라인 위에 있으므로 '공 전체 통과'와 프레임 접촉이 함께 성립하는 구간은
        ``|x - hx| ≤ r_ball + rf`` (≤0.17 m) 띠뿐이고, 그 안에서 재개 스폿 오차는 같은 폭으로
        유계다. 프레임은 선수 접촉이 아니므로 ``last_touch``·오프사이드 국면은 건드리지
        않는다 — 포스트 리바운드를 오프사이드 위치의 공격수가 잡으면 반칙이라는 Law 11의
        결론이 그 불변으로 그대로 나온다.
        """

        e_cfg = self.e_cfg
        seg = pos_after - pos_before

        post_hit, post_s, post_off = self._goal_frame_capsule_entry(
            pos_before, seg, self._goal_post_axis_xy, (DIM_X, DIM_Y), DIM_Z,
            0.0, self.goal_h + e_cfg.goal_frame_radius,
        )
        bar_hit, bar_s, bar_off = self._goal_frame_capsule_entry(
            pos_before, seg, self._goal_bar_axis_xz, (DIM_X, DIM_Z), DIM_Y,
            -self._goal_bar_half_span, self._goal_bar_half_span,
        )
        # 모서리 구는 **위쪽 네 곳만** 둔다. 포스트 아래쪽 끝 반구는 z<=0.17까지 닿는데
        # 그 구간은 몸통 원기둥 z∈[0, bar_z]가 이미 덮고, 자유물리가 매 서브스텝
        # ``pos_z >= r_ball``로 자르므로 z<0에서 시작하는 궤적은 런타임에 존재하지 않는다.
        # (조밀 격자 대조: 도달 가능 집합 25,303 참명중에서 놓침 0·오검출 0.)
        corner_hit, corner_s, corner_off = self._goal_frame_sphere_entry(
            pos_before, seg, self._goal_corner_xyz,
        )
        zero = jnp.zeros_like(post_off[:, 0])
        post_normal = jnp.stack([post_off[:, 0], post_off[:, 1], zero], axis=-1)
        bar_zero = jnp.zeros_like(bar_off[:, 0])
        bar_normal = jnp.stack([bar_off[:, 0], bar_zero, bar_off[:, 1]], axis=-1)

        hit = jnp.concatenate([post_hit, bar_hit, corner_hit])
        s_all = jnp.concatenate([post_s, bar_s, corner_s])
        normal_all = jnp.concatenate(
            [post_normal, bar_normal, corner_off], axis=0
        )
        kind_all = jnp.concatenate([
            jnp.full(post_hit.shape, WOODWORK_POST, jnp.int32),
            jnp.full(bar_hit.shape, WOODWORK_CROSSBAR, jnp.int32),
            # 모서리는 포스트와 크로스바가 **공유하는** 끝면이라 어느 한쪽이 아니다.
            # 크로스바 높이에 있으므로 그쪽으로 라벨한다 — 코드를 새로 만들면 K리그
            # 라벨(shotHitGoalpost 하나)보다 세분화돼 대조할 기준이 없어진다.
            jnp.full(corner_hit.shape, WOODWORK_CROSSBAR, jnp.int32),
        ])
        # 한 tick에 여러 요소를 스치는 경우(이음매에서는 몸통 둘과 모서리 구가 함께
        # 걸린다)는 **먼저 닿은 쪽**이 물리적 접촉이다. 나중 것은 이미 방향이 바뀐 뒤라
        # 이 구간에서 성립하지 않는다.
        order = jnp.where(hit, s_all, jnp.inf)
        idx = jnp.argmin(order)
        any_hit = jnp.any(hit)

        s = s_all[idx]
        kind = jnp.where(any_hit, kind_all[idx], jnp.int32(WOODWORK_NONE))
        offset = normal_all[idx]
        offset_norm = _safe_norm(offset)
        normal = offset / jnp.maximum(offset_norm, DIV_EPS)

        # 접촉 순간의 공 중심. 파고든 상태로 들어왔다면 표면 밖으로 밀어내 다음 tick이
        # 같은 캡슐을 다시 집지 않게 한다(반사 뒤 v·n>0이라 정상 경로에서는 무해한 no-op).
        radius_sum = self.r_ball + e_cfg.goal_frame_radius
        contact_center = pos_before + s * seg
        push = jnp.maximum(radius_sum + GEOMETRY_EPS - offset_norm, 0.0)
        contact_center = contact_center + push * normal

        v_n = jnp.dot(vel, normal)
        approaching = v_n < 0.0
        v_tan = vel - v_n * normal
        # 접촉점 표면속도 = 병진 접선 + ω×(-r·n). 지면식(n=ẑ)에서 r·(-ω_y, ω_x, 0)와 같다.
        slip = v_tan - self.r_ball * jnp.cross(spin, normal)
        slip = slip - jnp.dot(slip, normal) * normal
        slip_mag = _safe_norm(slip)
        slip_dir = slip / jnp.maximum(slip_mag, DIV_EPS)

        alpha = e_cfg.ball_inertia_ratio
        j_normal = (1.0 + e_cfg.goal_frame_e_rest) * jnp.maximum(-v_n, 0.0)
        j_stick = (alpha / (1.0 + alpha)) * slip_mag
        j_tan = jnp.minimum(j_stick, e_cfg.goal_frame_mu * j_normal)
        dv_tan = -j_tan * slip_dir

        vel_out = (
            v_tan + dv_tan
            + (-e_cfg.goal_frame_e_rest * v_n) * normal
        )
        spin_out = spin - jnp.cross(normal, dv_tan) / (alpha * self.r_ball)
        # 이미 멀어지는 중이면(파고든 상태로 진입한 경우) 위치만 빼내고 속도는 보존한다.
        vel_out = jnp.where(approaching, vel_out, vel)
        spin_out = jnp.where(approaching, spin_out, spin)

        # 남은 구간은 반사된 속도로 간다 — ``_ball_step_after_body``의 몸통 충돌 보정과
        # 같은 조각별 선형 구성이다.
        pos_out = contact_center + (1.0 - s) * e_cfg.dt_phys * vel_out
        pos_out = pos_out.at[DIM_Z].set(
            jnp.maximum(pos_out[DIM_Z], self.r_ball)
        )

        hit_now = any_hit
        return (
            kind,
            jnp.where(hit_now, contact_center, pos_after),
            jnp.where(hit_now, pos_out, pos_after),
            jnp.where(hit_now, vel_out, vel),
            jnp.where(hit_now, spin_out, spin),
        )

    def _ball_step_after_body(
        self,
        state,
        ball_pos_before_body,
        ball_vel_before_body,
        body_touch_mask,
        body_toi,
        *,
        return_event_start=False,
        return_woodwork=False,
    ):
        """Advance one tick while preserving a swept body contact's time of impact.

        ``_ball_body`` tests the segment that the ball would traverse during
        this physics tick.  Applying its reflected velocity for the *entire*
        tick made a late collision happen at tick start: a 34 m/s ball hitting
        a torso at 90 % of the segment ended 0.51 m behind the continuous
        piecewise-linear endpoint.  The collision state itself stays at the
        physical pre-step/drop position for rule adjudication; only this
        integration copy receives the equivalent pre-position correction

            p_virtual + v_after*dt
              = p_before + v_before*toi*dt + v_after*(1-toi)*dt.

        A trap has an explicit player-relative drop point rather than the
        incoming contact point, hence its correction starts from that drop.
        Drag, Magnus and gravity still use the bounded full-tick integrator;
        at the 1/90 s default their operator-splitting remainder is second
        order, while the former position error was first order and up to twice
        the whole tick travel.
        """

        before_pos = jnp.asarray(ball_pos_before_body, dtype=jnp.float32)
        before_vel = jnp.asarray(ball_vel_before_body, dtype=jnp.float32)
        if before_pos.shape != (DIM_ALL,) or before_vel.shape != (DIM_ALL,):
            raise ValueError(
                "ball_pos_before_body and ball_vel_before_body must both "
                f"have shape ({DIM_ALL},)"
            )
        body_touch_mask = jnp.asarray(body_touch_mask, dtype=bool)
        if body_touch_mask.shape != (self.N,):
            raise ValueError(
                f"body_touch_mask must have shape ({self.N},), got "
                f"{body_touch_mask.shape}"
            )
        body_toi = jnp.asarray(body_toi, dtype=jnp.float32)
        if body_toi.shape != ():
            raise ValueError(f"body_toi must be scalar, got {body_toi.shape}")

        hit = jnp.any(body_touch_mask) & (state.ball_state == BALL_ALIVE)
        actor = jnp.argmax(body_touch_mask.astype(jnp.int32))
        trapped = hit & (state.touch[actor] == TOUCH_BODY_TRAP)
        pre_duration = jnp.clip(body_toi, 0.0, 1.0) * self.e_cfg.dt_phys
        bounce_virtual = (
            before_pos + pre_duration * (before_vel - state.ball_vel)
        )
        trap_virtual = state.ball_pos - pre_duration * state.ball_vel
        virtual_pos = jnp.where(trapped, trap_virtual, bounce_virtual)
        integration_state = state._replace(
            ball_pos=jnp.where(hit, virtual_pos, state.ball_pos)
        )
        stepped = self._ball_step(integration_state)
        # Referee line-crossing interpolation must follow the outgoing piece
        # of a swept collision, not the chord from the tick's original point
        # to its final point.  Incoming contacts are already gated to occur no
        # later than a whole-ball field exit.  A bounce therefore starts its
        # adjudication segment at the physical TOI; a trap starts at its
        # explicit player-relative drop point.  No-hit calls retain the
        # ordinary pre-integration state coordinate.
        contact_pos = before_pos + pre_duration * before_vel
        event_start = jnp.where(
            hit,
            jnp.where(trapped, state.ball_pos, contact_pos),
            state.ball_pos,
        )
        # 골 프레임은 이 tick의 **나가는** 조각에서만 만날 수 있다 — 몸통 충돌이 있었다면
        # 그 접촉점부터가 실제 경로다. 반사가 일어나면 라인 통과 판정의 시작점도 프레임
        # 접촉점으로 옮겨야 크로스바 맞고 들어간 공이 골로 판정된다.
        woodwork_kind = jnp.int32(WOODWORK_NONE)
        woodwork_pos = jnp.zeros(DIM_ALL, jnp.float32)
        woodwork_vel_in = jnp.zeros(DIM_ALL, jnp.float32)
        if self._goal_frame_active:
            vel_in = stepped.ball_vel
            (
                woodwork_kind, woodwork_pos, frame_pos, frame_vel, frame_spin,
            ) = self._goal_frame_bounce(
                event_start, stepped.ball_pos, vel_in, stepped.ball_spin,
            )
            struck = woodwork_kind != jnp.int32(WOODWORK_NONE)
            woodwork_vel_in = jnp.where(
                struck, vel_in, jnp.zeros_like(vel_in)
            )
            stepped = stepped._replace(
                ball_pos=frame_pos, ball_vel=frame_vel, ball_spin=frame_spin,
            )
            event_start = jnp.where(struck, woodwork_pos, event_start)
            woodwork_pos = jnp.where(
                struck, woodwork_pos, jnp.zeros_like(woodwork_pos)
            )

        result = (stepped,)
        if return_event_start:
            result = result + (event_start,)
        if return_woodwork:
            result = result + (
                (woodwork_kind, woodwork_pos, woodwork_vel_in),
            )
        return result[0] if len(result) == 1 else result

    def _ball_body(
        self, state, key, suppress=None, *, touch_before,
        force_touch_mask=None, randomness=None, substep_index=0,
        return_event=False
    ):
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
        호출자는 반드시 이 스냅샷을 제공한다.
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
        # The horizontal path was swept, but using only the *starting* z band
        # still lets a fast vertical ball cross the whole torso between two
        # accepted large-dt samples.  Compute the time interval in which the
        # xy segment lies inside the body cylinder and intersect it with the
        # per-player torso-height interval.  This is an exact segment/cylinder
        # gate (the subsequent stochastic hit model remains unchanged).
        rel_dot_seg = jnp.sum(rel * seg[None, :], axis=1)
        rel_len2 = jnp.sum(rel * rel, axis=1)
        seg_len2_raw = jnp.dot(seg, seg)
        disc = rel_dot_seg ** 2 - seg_len2_raw * (rel_len2 - gate_r ** 2)
        root = jnp.sqrt(jnp.maximum(disc, 0.0))
        xy_moving = seg_len2_raw > SQUARED_EPS
        xy_lo_raw = (-rel_dot_seg - root) / (seg_len2_raw + SQUARED_EPS)
        xy_hi_raw = (-rel_dot_seg + root) / (seg_len2_raw + SQUARED_EPS)
        xy_lo = jnp.where(xy_moving, xy_lo_raw, 0.0)
        xy_hi = jnp.where(xy_moving, xy_hi_raw, 1.0)
        xy_valid = jnp.where(
            xy_moving,
            (disc >= 0.0) & (xy_hi >= 0.0) & (xy_lo <= 1.0),
            dist_xy < gate_r,
        )

        dz = state.ball_vel[DIM_Z] * e_cfg.dt_phys
        z_moving = jnp.abs(dz) > GEOMETRY_EPS
        t_leg = (e_cfg.leg_top - ball_z) / (
            jnp.where(z_moving, dz, 1.0)
        )
        t_body = (body_top - ball_z) / (
            jnp.where(z_moving, dz, 1.0)
        )
        z_lo = jnp.where(z_moving, jnp.minimum(t_leg, t_body), 0.0)
        z_hi = jnp.where(z_moving, jnp.maximum(t_leg, t_body), 1.0)
        z_valid = jnp.where(
            z_moving,
            (z_hi >= 0.0) & (z_lo <= 1.0),
            in_band,
        )
        contact_lo = jnp.maximum(jnp.maximum(xy_lo, z_lo), 0.0)
        contact_hi = jnp.minimum(jnp.minimum(xy_hi, z_hi), 1.0)
        swept_body = xy_valid & z_valid & (contact_lo <= contact_hi)

        # A player may physically stand up to five metres outside the pitch.
        # The swept collision predictor must therefore stop at the instant the
        # whole ball first leaves play.  Otherwise a body hit *after* a
        # touchline/goal-line crossing is processed before ``events`` and can
        # rewrite the last-touch team of an already dead ball.
        def axis_exit_fraction(start, delta, boundary):
            end = start + delta
            positive_exit = (delta > 0.0) & (end > boundary)
            negative_exit = (delta < 0.0) & (end < -boundary)
            positive_t = (boundary - start) / (delta + DIV_EPS)
            negative_t = (-boundary - start) / (delta - DIV_EPS)
            crossing = jnp.where(
                positive_exit,
                positive_t,
                jnp.where(negative_exit, negative_t, 1.0),
            )
            return jnp.where(
                jnp.abs(start) > boundary,
                0.0,
                jnp.clip(crossing, 0.0, 1.0),
            )

        x_exit_t = axis_exit_fraction(
            ball_xy[DIM_X], seg[DIM_X], self.hx + self.r_ball
        )
        y_exit_t = axis_exit_fraction(
            ball_xy[DIM_Y], seg[DIM_Y], self.hy + self.r_ball
        )
        field_exit_t = jnp.minimum(x_exit_t, y_exit_t)
        # A valid forward transition adjudicates the exit at the end of the
        # preceding tick, but public replay/state-reconstruction callers may
        # inject a still-live sample that already lies wholly outside.  In that
        # case ``axis_exit_fraction`` deliberately returns zero; accepting an
        # overlap at the same zero would nevertheless let an outside torso
        # rewrite the last touch before ``events`` gets to repair the state.
        # Equality with the boundary remains in play (the whole ball has not
        # crossed), hence the strict comparison here.
        already_out = (
            (jnp.abs(ball_xy[DIM_X]) > self.hx + self.r_ball)
            | (jnp.abs(ball_xy[DIM_Y]) > self.hy + self.r_ball)
        )
        swept_body = swept_body & (~already_out) & (
            contact_lo <= field_exit_t + GEOMETRY_EPS
        )
        # 이번 **물리 서브스텝**에 이미 접촉한 선수 제외(같은 서브스텝 이중 상호작용 방지).
        # ``state.touch``는 컨트롤 스텝 단위로만 0 초기화되므로(env.step_env_array) 스냅샷 없이
        # ``touch > 0``만 보면 배제 창이 남은 서브스텝 전체로 늘어나 그 길이가 ``decimation``
        # (즉 제어 frame 길이)에 종속된다.
        # ``touch_before``(접촉 직전 스냅샷)와 비교해 이번 서브스텝에 새로 생긴 접촉만 배제한다.
        # ``restart._throwin_restriction``·``offside._offside_check``와 동일한 규약.
        excluded = (
            ((state.touch > TOUCH_NONE) & (state.touch != touch_before))
            if force_touch_mask is None
            else jnp.asarray(force_touch_mask, dtype=bool)
        )
        is_taker = (((players == state.setpiece_taker) & (state.setpiece_taker >= 0))
                    | ((players == state.throw_taker) & (state.throw_taker >= 0)))
        # 수직 낙하 케이스: 수평 접근이 없어도 밴드 안으로 떨어지는 공(가슴트랩 상황)은 후보에 포함.
        falling_hit = (
            state.ball_vel[DIM_Z] < -e_cfg.collide_speed_min
        ) & swept_body
        # ``contact_lock_t`` is the physical re-contact debounce established by
        # an active f2b contact.  Without it, a kick can launch the ball through
        # the kicker's own torso and the same player passively overwrites that
        # kick with DEFLECT a few substeps later.  The same lock already blocks
        # another active f2b attempt; it must also block this passive route.
        candidate = (swept_body & (~excluded) & (state.contact_lock_t <= 0)
                     & state.active_player
                     & ((closing > 0) | falling_hit) & (ball_speed3 > e_cfg.collide_speed_min)
                     & (state.ball_state == BALL_ALIVE))
        # Continuous collision ordering is owned by the earliest time of
        # impact.  Choosing the smallest perpendicular sweep distance first
        # allowed a centred *later* torso to win over an earlier grazing one,
        # so the ball could pass through the first player in a long tick.
        # After TOI, use perpendicular distance only as a geometric tie-break.
        # If both are exactly equal, global slot order is not a physical fact
        # and is not covariant when equal-sized roster halves are exchanged
        # (slots 3/4 become 7/0 in 4v4).  Resolve only that exact set with
        # compact-visible, rotation-invariant geometry: positive oriented side
        # of the ball path, then within-team rank.
        # A fully symmetric residual has no deterministic single-winner rule
        # that is also permutation-equivariant, so stable slot order is the
        # final fail-safe.  Never use player_id: identity is intentionally
        # absent from compact State/obs and would make equal vectors transition
        # differently.  Every non-tie keeps the earliest-contact winner bit for bit.
        earliest_entry = jnp.min(jnp.where(candidate, contact_lo, jnp.inf))
        entry_tie = candidate & (contact_lo == earliest_entry)
        minimum_distance = jnp.min(jnp.where(
            entry_tie, sweep_dist, jnp.inf
        ))
        contact_tie = entry_tie & (sweep_dist == minimum_distance)
        offset_at_closest = state.player_pos - closest
        oriented_side = (
            seg[DIM_X] * offset_at_closest[:, DIM_Y]
            - seg[DIM_Y] * offset_at_closest[:, DIM_X]
        )
        best_side = jnp.max(jnp.where(
            contact_tie, oriented_side, -jnp.inf
        ))
        side_tie = contact_tie & (oriented_side == best_side)
        within_team_rank = jnp.where(
            state.team_id == TEAM_0,
            players,
            players - self.n_agents,
        )
        best_rank = jnp.min(jnp.where(
            side_tie, within_team_rank, self.N
        ))
        final_tie = side_tie & (within_team_rank == best_rank)
        idx = jnp.argmax(final_tie.astype(jnp.int32))
        active = candidate[idx]
        team_i = state.team_id[idx].astype(jnp.int32)

        # 첫 키는 쓰지 않지만 split은 남긴다 — 없애면 ``k_trap``이 달라져 trap 추첨
        # 열이 통째로 밀린다. 전이를 바꾸지 않으려는 의도적 보존이다.
        _, k_trap = jax.random.split(key)
        k_trap = select_random_key(
            randomness, RandomEvent.BODY_TRAP, substep_index, k_trap
        )
        # 몸통 충돌은 **결정론**이다. ``swept_body``는 이미 선분–원기둥 정확 교차이고,
        # 그 위에 확률을 얹으면 "몸에 정통으로 맞았는데 그냥 통과"가 생긴다. 실제로 33 m/s
        # 공을 선수 정면에 놓고 60회 굴려도 감속이 0인 경우가 나왔다 — 물리로 성립하지 않는다.
        #
        # 이 결정론은 reach 속도 게이트(movement._in_reach)와 짝을 이룬다. 빠르고 높은 공은
        # "발 뻗어 잡기"에서 빠지되 "몸으로 막기"는 반드시 성립해야 하기 때문이다. 둘 중
        # 하나라도 확률이면 그 계약이 깨진다.
        #
        # 통과가 필요한 경우는 기하로 이미 갈린다 — 밴드 밖(다리 아래·머리 위), 코어 반경
        # 밖, 멀어지는 방향, 정지에 가까운 공은 애초에 ``candidate``가 아니다.
        #
        # "스쳐도 크게 굴절되지는 않는다"는 성질에 별도 확률 상수가 필요하지 않다. 반사식이
        # 이미 그것을 표현한다 — 접촉 법선의 입사 성분만 뒤집으므로
        # 접선 스침은 v·n ≈ 0이라 저절로 작아진다(정면 Δv 18.75 m/s 대 스침 2.65 m/s).
        sup = jnp.bool_(False) if suppress is None else jnp.bool_(suppress)  # python True 방어(~True=-2 int화)
        hit = active & (~sup)
        p_trap = e_cfg.trap_base * state.player_ctrl[idx] * jnp.clip(1.0 - ball_speed3 / e_cfg.trap_speed_ref, 0.0, 1.0)
        trap = hit & (jax.random.uniform(k_trap) < p_trap) & (~is_taker[idx])
        bounce = hit & (~trap)

        def hit_response(_):
            # 반사 법선은 스윕의 **첫 접촉점** 기준이다. 최근접점을 쓰면 안 된다 — 선분 내부의
            # 최근접 반경은 진행벡터와 직교해 v·n≈0이 되므로 비중심 충돌이 거의 반사되지 않고,
            # 중심을 정확히 지나는 퇴화 폴백만 180° 가까이 되튀는 구조적 각도 오류가 된다. 원통 진입시각과 torso-height 진입시각 중 늦은 contact_lo에서의
            # 선수→공 반경을 써 터널링 경로도 실제 첫 겹침 방향으로 정반사한다.
            contact_point = ball_xy + contact_lo[idx] * seg
            n_raw = contact_point - state.player_pos[idx]
            n_len = jnp.linalg.norm(n_raw)
            incoming_opposite = -_unit(vel_xy_b[None, :])[0]
            facing_fallback = jnp.asarray([
                jnp.cos(state.player_facing[idx]),
                jnp.sin(state.player_facing[idx]),
            ])
            degenerate_normal = jnp.where(
                jnp.linalg.norm(vel_xy_b) > GEOMETRY_EPS,
                incoming_opposite,
                facing_fallback,
            )
            normal = jnp.where(
                n_len > GEOMETRY_EPS,
                n_raw / (n_len + DIV_EPS),
                degenerate_normal,
            )
            vel_xy = vel_xy_b
            # 입사 성분(v·n<0)만 반사. falling_hit 후보는 공이 몸에서 멀어지는 중(v·n>0)일 수 있어,
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
            # A side impact damps the vertical component without reflecting it.
            # ``falling_hit`` also admits the distinct top-surface case where a
            # vertically descending ball enters the torso band with no horizontal
            # closing component.  Keeping the old sign there labelled a bounce
            # while letting the ball continue through the player's body.  Reflect
            # only that top entry; horizontal/oblique side contacts retain the
            # calibrated vertical damping semantics.
            top_impact = falling_hit[idx] & (closing[idx] <= 0.0)
            bounced_vz = jnp.where(
                top_impact,
                -state.ball_vel[DIM_Z] * e_cfg.e_body,
                state.ball_vel[DIM_Z] * e_cfg.e_body,
            )
            new_vel_z = jnp.where(
                bounce,
                bounced_vz,
                jnp.where(trap, 0.0, state.ball_vel[DIM_Z]),
            )
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
            # [2026-08-13] 몸통 바운스=비통제 굴절 → 중립(-1) — K리그 벤더 소유 기준 정합.
            poss = jnp.where(trap, team_i,
                    jnp.where(bounce, jnp.int32(-1), state.poss_team)).astype(jnp.int32)
            last_touch = jnp.where(hit, team_i, state.last_touch_team).astype(jnp.int32)
            # 몸통 트랩은 이전 소유팀과 무관하게 동일한 **물리 provenance**다. 상대 공을
            # 컨트롤했다는 전술 의미(interception)는 ``state.poss_team != team_i``에서 파생할
            # 수 있지만, TOUCH_INTERCEPT로 기록하면 inverse가 능동 태클 킥으로 오인해 제출하지
            # 않은 f2b action을 역산한다. BODY_TRAP은 deliberate control이되 발로 찬 행위가
            # 아니므로 GK 백패스와 kick_applied에서도 명시적으로 구별된다.
            new_code = jnp.where(trap, jnp.int32(TOUCH_BODY_TRAP),
                       jnp.where(bounce, jnp.int32(TOUCH_DEFLECT), state.touch[idx]))
            touch = state.touch.at[idx].set(new_code)
            last_touch_code = jnp.where(hit, new_code, state.last_touch_code).astype(jnp.int32)
            # Any intervening physical touch by another player ends the handling
            # restriction.  A restricted goalkeeper's own BODY_TRAP/DEFLECT does
            # not: neither a team-mate back-pass nor a post-release re-handling ban
            # is cured by the same keeper first letting the ball hit their body.
            handling_code = state.gk_handling_restricted_team
            handling_team = jnp.where(
                handling_code >= GK_HANDLING_RELEASE_OFFSET,
                handling_code - GK_HANDLING_RELEASE_OFFSET,
                handling_code,
            )
            restricted_gk_hit = (
                hit
                & (state.gk_indices[idx] == 1)
                & (handling_team == team_i)
            )
            gk_handling_restricted_team = jnp.where(
                hit & (~restricted_gk_hit),
                jnp.int32(NO_TEAM),
                state.gk_handling_restricted_team,
            ).astype(jnp.int32)
            # 행위자는 팀·코드와 **한 쌍**이다. 종전에는 몸통 접촉이 team과 code만 갱신하고
            # actor를 그대로 두어, 블록된 슛 뒤 State가 서로 다른 두 접촉을 가리켰다
            # (실측: actor=슈터(team 0) · last_touch_team=1 · last_touch_code=DEFLECT).
            # ``observation``의 last-touch 관계 열이 그 셋을 함께 읽으므로, 슈터에게는
            # 수비수의 굴절이 자기 접촉으로 보였다.
            last_touch_actor = jnp.where(
                hit, idx.astype(jnp.int32), state.last_touch_actor
            ).astype(jnp.int32)
            result = state._replace(
                ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                poss_team=poss, last_touch_team=last_touch, touch=touch,
                last_touch_code=last_touch_code,
                last_touch_actor=last_touch_actor,
                gk_handling_restricted_team=gk_handling_restricted_team,
            )
            # ``touch`` is a control-frame accumulator, so comparing code values
            # cannot detect a second BODY_TRAP/DEFLECT by the same actor.  Runtime
            # rule adjudication requests this explicit per-substep event mask;
            # public/direct callers get the State-only return.
            body_event = (players == idx) & hit
            # Time of impact is returned only with the internal event form.  It is
            # consumed by ``_ball_step_after_body`` after rule adjudication, so the
            # direct State-only primitive keeps its contact-state contract.
            body_toi = jnp.where(hit, jnp.clip(contact_lo[idx], 0.0, 1.0), 0.0)
            return result, body_event, body_toi

        if self._specialize_ball_body_no_hit:
            result, body_event, body_toi = jax.lax.cond(
                hit,
                hit_response,
                lambda _: (
                    state,
                    jnp.zeros(self.N, dtype=bool),
                    jnp.float32(0.0),
                ),
                operand=None,
            )
        else:
            result, body_event, body_toi = hit_response(None)
        return (result, body_event, body_toi) if return_event else result
