"""역동역학 헬퍼 — reconstruct/BC용. 관측된 접촉 결과에서 그 결과를 만든 '액션' 또는 '단일 draw'를
되찾는다. `_apply_force2ball`의 모든 접촉 분기가 결정적 propensity + 단일 draw 구조라 전부 역산된다.

분기별 자유도(reconstruct가 복원해야 하는 양):
 · **킥**(free_play·tackle_ok: Play/Shot/FK/Corner/GoalKick/KickOff/Penalty·TacklingGame승) →
   방향·파워·발사각·사이드/백스핀 액션(킥=L∞ stretch 2D). `infer_kick_action`(정확 역함수, 재추첨 불요).
 · **스로인 테이크**(ThrowIn) → 킥과 동일 구조지만 속도 스케일이 `throw_speed_max`(power_cap·스핀
   없음). `infer_throwin_action`.
 · **굴절**(deflect: 탈취 실패 루즈볼·BallDeflection) → 자유도는 **랜덤 각도 draw** 하나. 관측
   출구방향에서 `defl_ang`과 그것을 만든 uniform `u∈[0,1]`을 복원(출구속도는 입사속도의 결정함수).
   `infer_deflect`.
 · **GK parry**(BallClaiming 고속) → 자유도 0. 입사속도·GK attack_dir의 **결정함수**라 액션이 없다.
   `predict_parry`가 그 속도를 정방향 계산(관측치와 대조·pin).
 · **GK 캐치/홀드·백패스**(BallClaiming 저속·백패스) → 데드볼, 자유도 0(위치·재개는 결정적).

DFL 데이터 구조와의 접합(reconstruct 하네스):
 events XML 라벨(`calib/out/events_labels.npz`: `st_type`[Play/TacklingGame/ThrowIn/BallClaiming/
 ShotAtGoal/FreeKick/…], `st_player`, `st_frame`)이 '언제·누가·무슨 접촉'을 주고, 트래킹이 접촉
 직후 공 속도·스핀을 준다. 하네스는 ①`forced_winner=actor`로 경합 승자를 pin해 env가 그 분기를
 타게 하고(★GK 캐치/parry 분기는 want_kick 게이트 때문에 **GK 액션의 킥게이트도 꺼야** 성립)
 ②`touch_code`(env가 확정한 TOUCH_*)로 `infer_contact`를 디스패치해 액션/draw를 복원,
 ③복원 액션을 `_decode`에 넣거나 draw를 pin해 궤적을 결정적으로 재생한다. 포워드 물리는 불변(순수
 부가 모듈). 주의: 관측점은 `force2ball` 직후(같은 서브스텝의 `_ball_body`·`ball_step` 이전)라,
 트래킹의 컨트롤스텝 속도에서 접촉순간으로 `ball_step_only`를 역적분(드래그·마그누스 비해석 → 뉴턴)한
 값을 넣어야 정확하다.
"""
import jax.numpy as jnp

from constants import *
from spatial import _safe_norm, _unit, stretch_encode


class Inverse:
    def infer_kick_action(self, state, kicker, ball_vel_after, ball_spin_after, ball_z_at_contact):
        """관측된 킥 결과(공 속도·스핀) → 그 킥을 만든 8-D 액션(정책 프레임)으로 역산.

        _apply_force2ball 킥 분기의 정확한 역함수. 이동 차원(1:3)은 0으로 두고 킥 관련 차원만 채운다
        (킥게이트=+1, f2b_dir, f2b_pow, f2b_launch, spin_side, spin_back). 반환은 raw 액션(tanh/_u01
        적용 전 스케일)이라 그대로 _decode에 넣으면 원 결과가 재현된다.

        Args:
            state: 킥 시점 State (head_z·attack_dir 참조)
            kicker: 킥한 선수 인덱스(int)
            ball_vel_after: 킥 직후 공 속도 (3,) 월드 프레임
            ball_spin_after: 킥 직후 공 스핀 (3,) 월드 프레임
            ball_z_at_contact: 킥 접촉 시점 공 높이(발사각 하한·파워캡 결정)
        Return:
            action: (ACTION_DIM=8,) — kicker의 복원 액션(정책 프레임)
        """
        e_cfg = self.e_cfg
        head_z = state.head_z[kicker]
        pelvis_z = e_cfg.pelvis_frac * head_z
        header = ball_z_at_contact > head_z
        is_chest = (ball_z_at_contact > pelvis_z) & (~header)
        power_cap = jnp.where(header, e_cfg.header_cap, jnp.where(is_chest, e_cfg.chest_cap, 1.0))
        spin_cap = jnp.where(header, e_cfg.spin_head_cap, jnp.where(is_chest, e_cfg.spin_chest_cap, 1.0))

        # 속도 → 파워·발사각·방향(월드). |kick_vel|=speed(단위 방향벡터 노름 1이라).
        speed = _safe_norm(ball_vel_after)
        vel_xy = ball_vel_after[:DIM_Z]
        vel_xy_norm = _safe_norm(vel_xy)
        launch_ang = jnp.arctan2(ball_vel_after[DIM_Z], vel_xy_norm)
        dir_world = vel_xy / (vel_xy_norm + DIV_EPS)

        f2b_pow_val = jnp.clip(speed / (power_cap * e_cfg.f2b_speed_max), 0.0, 1.0)
        launch_floor = self.launch_lo(ball_z_at_contact)
        launch_frac = jnp.clip(
            (launch_ang - launch_floor) / (e_cfg.launch_max - launch_floor + DIV_EPS),
            0.0,
            1.0,
        )

        # 스핀 → 사이드(수직축, 프레임 불변)·백스핀(진행방향 lateral축)
        spin_scale = e_cfg.spin_max * spin_cap + DIV_EPS
        spin_side = ball_spin_after[DIM_Z] / spin_scale
        lateral_world = jnp.array([-dir_world[DIM_Y], dir_world[DIM_X]])
        spin_back = -(ball_spin_after[DIM_X] * lateral_world[DIM_X]
                      + ball_spin_after[DIM_Y] * lateral_world[DIM_Y]) / spin_scale

        # 방향은 정책 프레임으로 재폴딩(×attack_dir, involution). 스핀은 프레임 무관.
        attack_dir = state.attack_dir[kicker]
        f2b_dir_action = dir_world * attack_dir                          # 단위 방향(정책 프레임)

        # raw 액션 조립(8-D) — 킥은 L∞ stretch 2D(방향+파워→[3:5]), _u01 역(x=2u−1)
        f2b_v = stretch_encode(f2b_dir_action, f2b_pow_val)              # (2,) : unit=dir, ‖·‖∞=pow
        action = jnp.zeros(ACTION_DIM)
        action = action.at[ACTION_KICK_GATE].set(ACTION_MAX)
        action = action.at[ACTION_KICK_VECTOR].set(f2b_v)
        action = action.at[ACTION_LAUNCH].set(2.0 * launch_frac - 1.0)
        action = action.at[ACTION_SPIN_SIDE].set(
            jnp.clip(spin_side, ACTION_MIN, ACTION_MAX)
        )
        action = action.at[ACTION_SPIN_BACK].set(
            jnp.clip(spin_back, ACTION_MIN, ACTION_MAX)
        )
        return action

    def infer_throwin_action(self, state, taker, ball_vel_after, ball_z_at_contact):
        """관측된 스로인 릴리즈(공 속도) → 그 스로를 만든 8-D 액션(정책 프레임)으로 역산.

        스로인 테이크 분기(`_apply_force2ball` is_throw_take)의 정확한 역함수. 킥과 방향·발사각 매핑은
        같지만 속도 스케일이 `throw_speed_max`(손 스로라 body power_cap 없음)이고 스핀 커맨드가 없다
        (릴리즈 스핀=0). 따라서 스핀 차원(6,7)은 0으로 둔다.

        Args:
            state: 스로 시점 State(attack_dir 참조). ball_z_at_contact는 스로 스폿의 공 높이.
            taker: 스로인 던진 선수 인덱스(int).
            ball_vel_after: 릴리즈 직후 공 속도 (3,) 월드 프레임.
            ball_z_at_contact: 스로 스폿의 공 높이(릴리즈 손 높이 아님 — forward가 hands 재배치
                전의 z로 launch_lo를 계산하므로 동일 값을 넣어야 발사각 하한이 일치).
        Return:
            action: (ACTION_DIM=8,) — taker의 복원 액션(정책 프레임).
        """
        e_cfg = self.e_cfg
        speed = _safe_norm(ball_vel_after)
        vel_xy = ball_vel_after[:DIM_Z]
        vel_xy_norm = _safe_norm(vel_xy)
        launch_ang = jnp.arctan2(ball_vel_after[DIM_Z], vel_xy_norm)
        dir_world = vel_xy / (vel_xy_norm + DIV_EPS)

        f2b_pow_val = jnp.clip(speed / (e_cfg.throw_speed_max + DIV_EPS), 0.0, 1.0)
        launch_floor = self.launch_lo(ball_z_at_contact)
        launch_frac = jnp.clip(
            (launch_ang - launch_floor) / (e_cfg.launch_max - launch_floor + DIV_EPS),
            0.0,
            1.0,
        )

        f2b_dir_action = dir_world * state.attack_dir[taker]        # 정책 프레임 재폴딩(involution)
        f2b_v = stretch_encode(f2b_dir_action, f2b_pow_val)         # L∞ stretch 2D(방향+파워)
        action = jnp.zeros(ACTION_DIM)
        action = action.at[ACTION_KICK_GATE].set(ACTION_MAX)
        action = action.at[ACTION_KICK_VECTOR].set(f2b_v)
        action = action.at[ACTION_LAUNCH].set(2.0 * launch_frac - 1.0)
        return action                                              # 스핀 차원(6,7)=0

    def infer_deflect(self, state, winner, ball_vel_after):
        """관측된 굴절(루즈볼) 결과 → 그 굴절의 **랜덤 각도 draw**를 복원(단일 draw 역산).

        굴절 분기(`_apply_force2ball` deflect)의 자유도는 입사방향을
        ``±e_cfg.deflect_angle_max`` 회전시킨 각도 하나뿐이고
        출구속도는 입사속도의 결정함수다. 관측 출구방향에서 회전각 `defl_ang`과 그것을 만든
        uniform `u`를 복원한다. reconstruct는 `u`(또는 각도)를 pin해 굴절을 재생한다.

        Args:
            state: 접촉 시점 State(`state.ball_vel`=입사속도, winner 위치 참조 — forward와 동일).
            winner: 굴절시킨 선수 인덱스(int) — 저속 입사 시 기본 방향(도전자→공) 산출용.
            ball_vel_after: 굴절 직후 공 속도 (3,) 월드 프레임.
        Return:
            dict(defl_ang, u, defl_speed): 회전각(rad)·복원 uniform[0,1]·기대 출구속력(관측 대조용).
        """
        e_cfg = self.e_cfg
        ball_speed0 = _safe_norm(state.ball_vel[:DIM_Z])
        ball_dir = jnp.where(
            ball_speed0 > e_cfg.deflect_stationary_speed,
            state.ball_vel[:DIM_Z] / (ball_speed0 + DIV_EPS),
                             _unit((state.ball_pos[:DIM_Z] - state.player_pos[winner])[None, :])[0])
        out_xy = ball_vel_after[:DIM_Z]
        defl_dir = out_xy / (_safe_norm(out_xy) + DIV_EPS)
        cross = ball_dir[DIM_X] * defl_dir[DIM_Y] - ball_dir[DIM_Y] * defl_dir[DIM_X]
        dot = ball_dir[DIM_X] * defl_dir[DIM_X] + ball_dir[DIM_Y] * defl_dir[DIM_Y]
        defl_ang = jnp.arctan2(cross, dot)                         # 입사→출구 부호있는 회전각
        u = jnp.clip(
            defl_ang / (2.0 * e_cfg.deflect_angle_max) + 0.5, 0.0, 1.0
        )
        defl_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        return {"defl_ang": defl_ang, "u": u, "defl_speed": defl_speed}

    def predict_parry(self, state, gk):
        """GK parry 출구속도를 정방향 계산 — 자유도 0(입사속도·GK attack_dir의 결정함수).

        parry 분기(`_apply_force2ball` gk_parry)는 액션이 없다(GK 본능 반사). reconstruct는 이 값을
        관측 parry 속도와 대조해 접촉을 확인하고, `forced_winner=gk`로 승자를 pin하되 **GK 액션의
        킥게이트를 꺼야**(want_kick 게이트 — 켜져 있으면 gk_claim 자체가 불성립) 결정적으로 재생된다.

        Args:
            state: 접촉 시점 State(`state.ball_vel`=입사속도, attack_dir 참조).
            gk: parry한 골키퍼 인덱스(int).
        Return:
            parry_vel: (3,) 예측 parry 출구속도(월드). 보존·리프트 비율은 Engine 설정을 공유.
        """
        e_cfg = self.e_cfg
        ball_speed0 = _safe_norm(state.ball_vel[:DIM_Z])
        parry_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        return jnp.array([state.attack_dir[gk] * parry_speed,
                          state.ball_vel[DIM_Y] * e_cfg.parry_lateral_keep,
                          e_cfg.parry_lift_frac * parry_speed])

    def infer_contact(self, state, actor, touch_code, restart_kind,
                      ball_vel_after, ball_spin_after=None, ball_z_at_contact=None):
        """DFL 접촉 이벤트 → 분기별 역함수 디스패처(host-side, reconstruct 하네스 직결).

        env가 확정한 `touch_code`(TOUCH_*)와 `restart_kind`로 분기해 액션 또는 draw를 복원한다.
        스로인 테이크는 라벨이 TOUCH_PASS라 `restart_kind==RK_THROWIN`으로 킥과 구분한다. 반환 dict의
        "kind"로 하네스가 처리 방식을 안다(action=_decode 주입 / draw=uniform pin / deterministic=대조만).

        Args:
            state: 접촉 시점 State. actor: 접촉 선수 인덱스. touch_code: env TOUCH_* 결과.
            restart_kind: 접촉 시점 재개 종류(스로인 구분). ball_vel_after/ball_spin_after: 접촉 직후 공
            속도·스핀(트래킹서 접촉순간으로 역적분한 값). ball_z_at_contact: 접촉 시 공 높이.
        Return:
            dict(kind, ...): kind="action"(action 8-D) / "draw"(u·defl_ang) / "deterministic"(velocity) /
            "dead"(자유도 없는 데드볼 접촉).
        """
        spin = jnp.zeros(DIM_ALL) if ball_spin_after is None else ball_spin_after
        zc = state.ball_pos[DIM_Z] if ball_z_at_contact is None else ball_z_at_contact
        # INTERCEPT 포함: contest _apply_force2ball의 tackle_ok&fast 분기는 TACKLE과 동일한
        # kick_vel_tackle을 공에 적용한 params 킥이다(코드 라벨만 fast로 갈림). 복원은 suppress_body=True라
        # 몸통 트랩(ball.py) INTERCEPT가 억제되어 INTERCEPT는 오직 이 contest 킥 분기에서만 발생 →
        # infer_kick_action으로 정확히 역산된다(TACKLE과 동일 역함수). 누락 시 아래 deterministic로
        # 흘러 킥 속도를 롤포워드에 반영하지 못해 복원 트래젝토리가 발산한다.
        kick_codes = (TOUCH_PASS, TOUCH_SHOOT, TOUCH_PASS_HEAD, TOUCH_SHOOT_HEAD,
                      TOUCH_DRIBBLE, TOUCH_TACKLE, TOUCH_INTERCEPT)
        if int(touch_code) == TOUCH_PASS and int(restart_kind) == RK_THROWIN:
            return {"kind": "action", "action": self.infer_throwin_action(state, actor, ball_vel_after, zc)}
        if int(touch_code) in kick_codes:
            return {"kind": "action",
                    "action": self.infer_kick_action(state, actor, ball_vel_after, spin, zc)}
        if int(touch_code) == TOUCH_DEFLECT:
            return {"kind": "draw", **self.infer_deflect(state, actor, ball_vel_after)}
        if int(touch_code) == TOUCH_PARRY:
            return {"kind": "deterministic", "velocity": self.predict_parry(state, actor)}
        if int(touch_code) == TOUCH_GK_CATCH:
            return {"kind": "dead"}                                # 캐치/홀드=데드볼, 자유도 없음
        # 몸통 트랩·바운스(ball.py 결정적 물리)·기타 미분류 접촉 — 역산할 액션이 없다.
        # (INTERCEPT/DRIBBLE 몸통트랩은 복원 시 suppress_body로 억제되므로 여기 도달하지 않는다.)
        # 하네스는 관측 outcome을 pin하고 결정적 물리를 그대로 롤 포워드한다.
        return {"kind": "deterministic"}

    def infer_move_action(self, state, next_player_vel):
        """[속도-명령 모델 guide.md §1] 관측된 다음 속도 = **목표 속도 그 자체**. 이동 액션(mv_dir, mv_pow)
        역산 → (N, ACTION_DIM) 이동 차원.

        _move가 목표속도 v_cmd = mv_pow·mv_dir·vmax를 받으므로 역산은 자명:
        mv_dir = unit(next_vel), mv_pow = |next_vel|/vmax(유효최고속). 방향은 정책 프레임 재폴딩(×attack_dir).
        구모델(가속 역산 a≈(V′−V)/dt)의 plant&cut·drag 근사 오차·데드비트 과응답 이슈가 원천 소멸한다
        (속도 명령이라 1적분·직접 매핑). 실제 속도가 종/횡 캡을 넘으면 _vel_substep이 도달분만 실현.

        Args:
            state: 현재 State
            next_player_vel: (N,2) 다음 컨트롤스텝 선수 속도(트래킹 데이터의 유한차분 등)
        Return:
            action: (N, ACTION_DIM)
        """
        e_cfg = self.e_cfg
        vmax = state.vmax * (e_cfg.vmax_floor + (1.0 - e_cfg.vmax_floor) * state.stamina)  # _move와 동일
        speed = _safe_norm(next_player_vel, axis=1, keepdims=True)
        dir_world = next_player_vel / (speed + DIV_EPS)
        mv_pow = jnp.clip(speed[:, 0] / (vmax + DIV_EPS), 0.0, 1.0)     # (N,) 목표속도/vmax
        dir_action = dir_world * state.attack_dir[:, None]              # 정책 프레임 재폴딩(단위)

        # 이동은 L∞ stretch 2D — 방향+크기를 [1:3] 한 벡터로(별도 pow dim 없음)
        mv_v = stretch_encode(dir_action, mv_pow)                       # (N,2): unit=dir, ‖·‖∞=mv_pow
        action = jnp.zeros((self.N, ACTION_DIM))
        action = action.at[:, ACTION_MOVE].set(mv_v)
        return action
