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
 · **GK catch/parry**(BallClaiming) → 결과는 속도별 Bernoulli draw, 결과가 parry일 때 출구속도는
   입사속도·GK attack_dir의 **결정함수**다. `infer_contact`가 관측 결과를
   `forced_gk_touch` pin으로 내고 `predict_parry`가 parry 속도를 대조한다. 백패스는 결정적 IDFK다.

DFL 데이터 구조와의 접합(reconstruct 하네스):
 events XML 라벨(`calib/out/events_labels.npz`: `st_type`[Play/TacklingGame/ThrowIn/BallClaiming/
 ShotAtGoal/FreeKick/…], `st_player`, `st_frame`)이 '언제·누가·무슨 접촉'을 주고, 트래킹이 접촉
 직후 공 속도·스핀을 준다. 하네스는 ①`forced_winner=actor`로 경합 승자를 pin해 env가 그 분기를
 타게 하고(★GK 캐치/parry 분기는 want_kick 게이트 때문에 **GK 액션의 킥게이트도 꺼야** 성립)
 ②`touch_code`(env가 확정한 TOUCH_*)로 `infer_contact`를 디스패치해 액션/draw/GK 결과를 복원,
 ③복원 액션을 `_decode`에 넣거나 draw를 pin해 궤적을 결정적으로 재생한다. 포워드 물리는 불변(순수
 부가 모듈). 주의: 관측점은 `force2ball` 직후(같은 서브스텝의 `_ball_body`·`ball_step` 이전)라,
 트래킹의 컨트롤스텝 속도에서 접촉순간으로 `ball_step_only`를 역적분(드래그·마그누스 비해석 → 뉴턴)한
 값을 넣어야 정확하다.

 ★역적분 구간에 **골 프레임 충돌**이 있으면 그 역적분은 성립하지 않는다. `ball_step_only`은
 자유 비행 전용이고 포스트·크로스바 반사는 그 뒤의 별도 분기라, 프레임을 맞고 온 속도를
 자유물리로 되돌리면 접촉순간 속도가 틀린다. 포워드를 그대로 미러링하려면 공개
 `goal_frame_impact(pos_before, pos_after, vel, spin)`으로 같은 분기를 재현하거나, 런타임이
 남긴 `State.woodwork_*`(substep별 kind/접촉점/입사속도)로 그 창을 제외해야 한다.
 이 모듈의 함수들 자체는 **접촉 순간**(force2ball 직후, 적분 이전)에서 동작하므로 프레임
 충돌이 그 사이에 끼지 않는다 — 영향은 호출자의 역적분 구간에만 있다.
"""
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np

from .constants import (
    ACTION_DIM,
    ACTION_KICK_GATE,
    ACTION_KICK_VECTOR,
    ACTION_LAUNCH,
    ACTION_MAX,
    ACTION_MIN,
    ACTION_MOVE,
    ACTION_SPIN_BACK,
    ACTION_SPIN_SIDE,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    RESTART_COUNT,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
    TOUCH_COUNT,
    TOUCH_DEFLECT,
    TOUCH_DRIBBLE,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_PARRY,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
)
from .spatial import _safe_norm, _unit, stretch_encode


class Inverse:
    @staticmethod
    def _inverse_index(name, value, upper):
        if (not isinstance(value, numbers.Integral)
                or isinstance(value, (bool, np.bool_))):
            raise TypeError(f"{name} must be an integer scalar, got {value!r}")
        result = int(value)
        if not 0 <= result < upper:
            raise ValueError(f"{name} must lie in [0, {upper}), got {result}")
        return result

    @staticmethod
    def _inverse_real(name, value):
        array = np.asarray(value)
        if (array.shape != () or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
                or np.issubdtype(array.dtype, np.complexfloating)):
            raise TypeError(f"{name} must be a finite real scalar")
        result = float(array)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _inverse_vector(name, value, shape):
        array = np.asarray(value)
        if (array.shape != shape or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
                or np.issubdtype(array.dtype, np.complexfloating)):
            raise TypeError(f"{name} must be a real vector with shape {shape}")
        array = array.astype(np.float64, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _inverse_is_traced(*values):
        """Whether a low-level inverse is being staged by JAX.

        Host reconstruction observations can and should raise on an impossible
        target.  Python exceptions cannot depend on staged values, while the
        same low-level helpers are intentionally jittable for known forward
        outcomes.  Keep the algebra traceable and apply strict validation to
        every concrete/eager public call; ``infer_contact`` remains the strict
        host dispatcher for external observations.
        """

        return any(
            isinstance(leaf, jax.core.Tracer)
            for value in values
            for leaf in jax.tree_util.tree_leaves(value)
        )

    def _validate_kick_observation_host(
        self, state, actor, ball_vel_after, ball_spin_after,
        ball_z_at_contact, *, is_throw,
    ):
        """Canonicalise and reject targets outside the forward kick manifold."""

        actor = self._inverse_index("actor", actor, self.N)
        velocity = self._inverse_vector(
            "ball_vel_after", ball_vel_after, (DIM_ALL,)
        )
        spin = (
            np.zeros(DIM_ALL, np.float64)
            if ball_spin_after is None
            else self._inverse_vector(
                "ball_spin_after", ball_spin_after, (DIM_ALL,)
            )
        )
        contact_z = self._inverse_real(
            "ball_z_at_contact", ball_z_at_contact
        )

        e_cfg = self.e_cfg
        speed = float(np.linalg.norm(velocity))
        horizontal = float(np.linalg.norm(velocity[:DIM_Z]))
        head_z = float(np.asarray(state.head_z)[actor])
        pelvis_z = e_cfg.pelvis_frac * head_z
        header = contact_z > head_z
        chest = contact_z > pelvis_z and not header
        power_cap = (
            e_cfg.header_cap if header
            else e_cfg.chest_cap if chest else 1.0
        )
        max_speed = (
            e_cfg.throw_speed_max if is_throw
            else power_cap * e_cfg.f2b_speed_max
        )
        tol = 2.0e-4 * max(1.0, max_speed)
        if speed > max_speed + tol:
            raise ValueError(
                f"observed contact speed {speed:.6g} exceeds forward "
                f"maximum {max_speed:.6g}"
            )
        if is_throw and speed < e_cfg.restart_min_ball_speed - tol:
            raise ValueError(
                f"observed throw-in speed {speed:.6g} is below forward "
                f"minimum {e_cfg.restart_min_ball_speed:.6g}"
            )
        if speed > tol:
            if horizontal <= tol:
                raise ValueError(
                    "observed kick has no horizontal direction and cannot "
                    "be represented below launch_max"
                )
            launch = math.atan2(velocity[DIM_Z], horizontal)
            launch_lo = float(np.asarray(
                self.launch_lo(jnp.float32(contact_z))
            ))
            if not launch_lo - tol <= launch <= e_cfg.launch_max + tol:
                raise ValueError(
                    f"observed launch angle {launch:.6g} lies outside "
                    f"[{launch_lo:.6g}, {e_cfg.launch_max:.6g}]"
                )

        if not is_throw:
            spin_cap = (
                e_cfg.spin_head_cap if header
                else e_cfg.spin_chest_cap if chest else 1.0
            )
            max_spin = e_cfg.spin_max * spin_cap
            if (abs(spin[DIM_Z]) > max_spin + tol
                    or np.linalg.norm(spin[:DIM_Z]) > max_spin + tol):
                raise ValueError("observed spin exceeds the forward contact cap")
            if horizontal > tol:
                direction = velocity[:DIM_Z] / horizontal
                longitudinal = float(np.dot(spin[:DIM_Z], direction))
                if abs(longitudinal) > tol:
                    raise ValueError(
                        "observed horizontal spin has a longitudinal "
                        "component absent from forward dynamics"
                    )
            elif np.linalg.norm(spin[:DIM_Z]) > tol:
                raise ValueError(
                    "zero-speed kick cannot uniquely invert horizontal spin"
                )

        return (
            actor,
            jnp.asarray(velocity, jnp.float32),
            jnp.asarray(spin, jnp.float32),
            jnp.float32(contact_z),
        )

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

        Concrete/eager calls reject non-finite or forward-unrepresentable
        observations.  When staged under ``jax.jit`` this is the algebraic
        primitive for already validated/forward-produced outcomes; external
        observations must pass through host-side ``infer_contact`` first.
        """
        if not self._inverse_is_traced(
            kicker, ball_vel_after, ball_spin_after, ball_z_at_contact
        ):
            kicker, ball_vel_after, ball_spin_after, ball_z_at_contact = (
                self._validate_kick_observation_host(
                    state, kicker, ball_vel_after, ball_spin_after,
                    ball_z_at_contact, is_throw=False,
                )
            )
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

        Concrete/eager calls validate the forward manifold, including the
        mandatory restart minimum release speed.  Traced calls are the pure
        algebraic primitive for observations prevalidated by ``infer_contact``.
        """
        if not self._inverse_is_traced(
            taker, ball_vel_after, ball_z_at_contact
        ):
            taker, ball_vel_after, _, ball_z_at_contact = (
                self._validate_kick_observation_host(
                    state, taker, ball_vel_after, None,
                    ball_z_at_contact, is_throw=True,
                )
            )
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
        traced = self._inverse_is_traced(winner, ball_vel_after)
        velocity_host = None
        if not traced:
            winner = self._inverse_index("winner", winner, self.N)
            velocity_host = self._inverse_vector(
                "ball_vel_after", ball_vel_after, (DIM_ALL,)
            )
            ball_vel_after = jnp.asarray(velocity_host, jnp.float32)
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
        result = {"defl_ang": defl_ang, "u": u, "defl_speed": defl_speed}
        if not traced:
            expected_speed = float(np.asarray(defl_speed))
            observed_xy = float(np.linalg.norm(velocity_host[:DIM_Z]))
            expected_z = e_cfg.deflect_lift_frac * expected_speed
            angle = float(np.asarray(defl_ang))
            tol = 2.0e-4 * max(1.0, expected_speed)
            if (abs(angle) > e_cfg.deflect_angle_max + tol
                    or abs(observed_xy - expected_speed) > tol
                    or abs(velocity_host[DIM_Z] - expected_z) > tol):
                raise ValueError(
                    "observed deflection is outside the forward angle/speed manifold"
                )
        return result

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
        # Include State leaves in trace detection.  A common compiled use closes
        # over a Python goalkeeper index (``jit(lambda s: predict_parry(s, 0))``);
        # inspecting that traced State through NumPy would otherwise raise even
        # though the index itself is static and valid.
        traced = self._inverse_is_traced(state, gk)
        if not traced:
            gk = self._inverse_index("gk", gk, self.N)
            team = int(np.asarray(state.team_id)[gk])
            is_goalkeeper = int(np.asarray(state.gk_indices)[gk]) == 1
            is_active = bool(np.asarray(state.active_player)[gk])
            if team not in (TEAM_0, TEAM_1) or not is_goalkeeper or not is_active:
                raise ValueError(
                    "gk must identify an active goalkeeper on a valid team"
                )
            safe_gk = jnp.int32(gk)
            valid_gk = jnp.bool_(True)
        else:
            gk_input = jnp.asarray(gk)
            if gk_input.shape != ():
                raise ValueError(
                    f"gk must be an integer scalar, got shape {gk_input.shape}"
                )
            # Reject unsigned tracers outright.  With x64 disabled a wide host
            # uint can be narrowed before this compiled boundary (for example
            # 2**32 + slot -> slot), so no unsigned dtype can preserve the
            # public source-width contract here.
            if (not jnp.issubdtype(gk_input.dtype, jnp.signedinteger)
                    or jnp.issubdtype(gk_input.dtype, jnp.bool_)):
                raise TypeError(
                    "gk must have signed-integer dtype in JAX-compiled calls, "
                    f"got {gk_input.dtype}"
                )
            in_range = (gk_input >= 0) & (gk_input < self.N)
            # Sanitize in the original signed width before int32 conversion.
            # The safe index is used only to make the gather total; valid_gk
            # masks its value so a negative/out-of-range request cannot alias
            # slot zero (or the last slot) into a causal parry prediction.
            safe_source = jnp.where(
                in_range, gk_input, jnp.asarray(0, dtype=gk_input.dtype)
            )
            safe_gk = safe_source.astype(jnp.int32)
            team = state.team_id[safe_gk]
            valid_gk = (
                in_range
                & ((team == TEAM_0) | (team == TEAM_1))
                & (state.gk_indices[safe_gk] == 1)
                & state.active_player[safe_gk]
            )

        e_cfg = self.e_cfg
        ball_speed0 = _safe_norm(state.ball_vel[:DIM_Z])
        attack_dir = state.attack_dir[safe_gk]
        parry_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        punch = jnp.array([
            attack_dir * parry_speed,
            state.ball_vel[DIM_Y] * e_cfg.parry_lateral_keep,
            e_cfg.parry_lift_frac * parry_speed,
        ])
        # tip-over 분기는 ``_apply_force2ball``과 **같은 식**이어야 한다. 여기서 펀치만
        # 예측하면 손끝 세이브가 전부 "예측과 다른 접촉"으로 보여 재구성이 깨진다.
        own_goal_x = -attack_dir * self.hx
        tip_h = state.ball_vel[:DIM_Z] * e_cfg.parry_tip_keep
        to_line = jnp.abs(own_goal_x - state.ball_pos[DIM_X])
        flight_t = to_line / jnp.maximum(jnp.abs(tip_h[DIM_X]), DIV_EPS)
        need_vz = (
            (self.goal_h + e_cfg.parry_tip_clearance - state.ball_pos[DIM_Z])
            / jnp.maximum(flight_t, DIV_EPS)
            + 0.5 * e_cfg.g * flight_t
        )
        # 무항력 탄도식은 거리가 길수록 실제보다 높게 예측한다(실측: 12m에서
        # 0.57m·16m에서 0.76m 모자라 골이 됐다). 손끝 세이브는 원래 골라인 가까이에서
        # 일어나므로 사거리를 제한하고 여유 높이를 그 범위에 맞춰 잡는다.
        tip_over = (
            (ball_speed0 > e_cfg.parry_punch_speed_cap)
            & (state.ball_vel[DIM_X] * attack_dir < 0.0)
            & (need_vz > 0.0)
            & (need_vz <= e_cfg.parry_tip_lift_max)
            & (to_line <= e_cfg.parry_tip_max_range)
        )
        parry = jnp.where(
            tip_over,
            jnp.array([tip_h[DIM_X], tip_h[DIM_Y], need_vz]),
            punch,
        )
        return jnp.where(valid_gk, parry, jnp.zeros_like(parry))

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
        actor = self._inverse_index("actor", actor, self.N)
        touch_code = self._inverse_index("touch_code", touch_code, TOUCH_COUNT)
        restart_kind = self._inverse_index(
            "restart_kind", restart_kind, RESTART_COUNT
        )
        velocity_host = self._inverse_vector(
            "ball_vel_after", ball_vel_after, (DIM_ALL,)
        )
        spin_host = (
            np.zeros(DIM_ALL, np.float64)
            if ball_spin_after is None
            else self._inverse_vector(
                "ball_spin_after", ball_spin_after, (DIM_ALL,)
            )
        )
        zc_host = self._inverse_real(
            "ball_z_at_contact",
            np.asarray(state.ball_pos)[DIM_Z]
            if ball_z_at_contact is None else ball_z_at_contact,
        )
        spin = jnp.asarray(spin_host, jnp.float32)
        velocity = jnp.asarray(velocity_host, jnp.float32)
        zc = jnp.float32(zc_host)
        actor_is_gk = bool(np.asarray(state.gk_indices)[actor])
        if touch_code in (TOUCH_PARRY, TOUCH_GK_CATCH) and not actor_is_gk:
            raise ValueError(
                "TOUCH_PARRY/TOUCH_GK_CATCH require a goalkeeper actor"
            )
        # INTERCEPT 포함: contest _apply_force2ball의 tackle_ok&fast 분기는 TACKLE과 동일한
        # kick_vel_tackle을 공에 적용한 params 킥이다(코드 라벨만 fast로 갈림). 수동 몸통
        # 컨트롤은 소유 관계와 무관하게 TOUCH_BODY_TRAP이므로 INTERCEPT는 여기서 오직 능동
        # contest 킥을 뜻한다. 누락 시 아래 deterministic로 흘러 킥 속도를 롤포워드에
        # 반영하지 못해 복원 트래젝토리가 발산한다.
        kick_codes = (TOUCH_PASS, TOUCH_SHOOT, TOUCH_PASS_HEAD, TOUCH_SHOOT_HEAD,
                      TOUCH_DRIBBLE, TOUCH_TACKLE, TOUCH_INTERCEPT)
        action_contact = touch_code in kick_codes
        is_throw = touch_code == TOUCH_PASS and restart_kind == RK_THROWIN
        if action_contact:
            e_cfg = self.e_cfg
            speed = float(np.linalg.norm(velocity_host))
            horizontal = float(np.linalg.norm(velocity_host[:DIM_Z]))
            head_z = float(np.asarray(state.head_z)[actor])
            pelvis_z = e_cfg.pelvis_frac * head_z
            header = zc_host > head_z
            chest = zc_host > pelvis_z and not header
            head_label = touch_code in (TOUCH_PASS_HEAD, TOUCH_SHOOT_HEAD)
            non_head_label = touch_code in (
                TOUCH_PASS, TOUCH_SHOOT, TOUCH_DRIBBLE
            )
            if head_label and not header:
                raise ValueError(
                    "a *_HEAD touch code requires contact above head_z"
                )
            if non_head_label and header and not is_throw:
                raise ValueError(
                    "a non-head pass/shot/dribble code cannot represent "
                    "contact above head_z"
                )
            power_cap = (
                e_cfg.header_cap if header
                else e_cfg.chest_cap if chest else 1.0
            )
            max_speed = (
                e_cfg.throw_speed_max if is_throw
                else power_cap * e_cfg.f2b_speed_max
            )
            # 태클만 캡을 받는다. 정방향에서 ``interception``은 ``free_play``의
            # 부분집합이라(:mod:`contest`) 공에 ``kick_vel``이 캡 없이 실린다 — 반면
            # 역산은 INTERCEPT에도 ``tackle_out_cap``을 걸어, 정방향이 합법적으로 만든
            # 빠른 인터셉트(기본 설정에서 30 m/s)를 최대 27.46 m/s라며 거부했다.
            # 역산의 상한은 정방향이 낼 수 있는 값보다 **느슨해야** 안전하다.
            if touch_code == TOUCH_TACKLE:
                max_speed = min(
                    max_speed, e_cfg.tackle_out_cap * e_cfg.f2b_speed_max
                )
            tol = 2.0e-4 * max(1.0, max_speed)
            if speed > max_speed + tol:
                raise ValueError(
                    f"observed contact speed {speed:.6g} exceeds forward "
                    f"maximum {max_speed:.6g}"
                )
            if speed > tol:
                if horizontal <= tol:
                    raise ValueError(
                        "observed kick has no horizontal direction and cannot "
                        "be represented below launch_max"
                    )
                launch = math.atan2(velocity_host[DIM_Z], horizontal)
                launch_lo = float(np.asarray(self.launch_lo(jnp.float32(zc_host))))
                if not launch_lo - tol <= launch <= e_cfg.launch_max + tol:
                    raise ValueError(
                        f"observed launch angle {launch:.6g} lies outside "
                        f"[{launch_lo:.6g}, {e_cfg.launch_max:.6g}]"
                    )
            if is_throw:
                if np.linalg.norm(spin_host) > tol:
                    raise ValueError("throw-in forward dynamics always emits zero spin")
            else:
                spin_cap = (
                    e_cfg.spin_head_cap if header
                    else e_cfg.spin_chest_cap if chest else 1.0
                )
                max_spin = e_cfg.spin_max * spin_cap
                if (abs(spin_host[DIM_Z]) > max_spin + tol
                        or np.linalg.norm(spin_host[:DIM_Z]) > max_spin + tol):
                    raise ValueError("observed spin exceeds the forward contact cap")
                if horizontal > tol:
                    direction = velocity_host[:DIM_Z] / horizontal
                    longitudinal = float(np.dot(spin_host[:DIM_Z], direction))
                    if abs(longitudinal) > tol:
                        raise ValueError(
                            "observed horizontal spin has a longitudinal "
                            "component absent from forward dynamics"
                        )
                elif np.linalg.norm(spin_host[:DIM_Z]) > tol:
                    raise ValueError(
                        "zero-speed kick cannot uniquely invert horizontal spin"
                    )

        if is_throw:
            return {"kind": "action", "action": self.infer_throwin_action(state, actor, velocity, zc)}
        if touch_code in kick_codes:
            return {"kind": "action",
                    "action": self.infer_kick_action(state, actor, velocity, spin, zc)}
        if touch_code == TOUCH_DEFLECT:
            result = self.infer_deflect(state, actor, velocity)
            expected_speed = float(np.asarray(result["defl_speed"]))
            observed_xy = float(np.linalg.norm(velocity_host[:DIM_Z]))
            expected_z = self.e_cfg.deflect_lift_frac * expected_speed
            angle = float(np.asarray(result["defl_ang"]))
            tol = 2.0e-4 * max(1.0, expected_speed)
            if (abs(angle) > self.e_cfg.deflect_angle_max + tol
                    or abs(observed_xy - expected_speed) > tol
                    or abs(velocity_host[DIM_Z] - expected_z) > tol):
                raise ValueError(
                    "observed deflection is outside the forward angle/speed manifold"
                )
            return {"kind": "draw", **result}
        if touch_code == TOUCH_PARRY:
            predicted = self.predict_parry(state, actor)
            if not np.allclose(
                velocity_host, np.asarray(predicted), rtol=2.0e-4, atol=2.0e-4
            ):
                raise ValueError("observed parry velocity is not forward-representable")
            return {
                "kind": "deterministic",
                "velocity": predicted,
                "forced_gk_touch": TOUCH_PARRY,
            }
        if touch_code == TOUCH_GK_CATCH:
            if np.linalg.norm(velocity_host) > 2.0e-4:
                raise ValueError("GK catch/hold output velocity must be zero")
            return {
                "kind": "dead",
                "forced_gk_touch": TOUCH_GK_CATCH,
            }                                                        # 캐치/홀드=데드볼
        # 몸통 트랩·바운스(ball.py 결정적 물리)·기타 미분류 접촉 — 역산할 액션이 없다.
        # TOUCH_BODY_TRAP은 상대 소유 공을 컨트롤한 경우도 포함하며, 그 '인터셉트' 의미는
        # 접촉 전 소유팀과 actor 팀의 관계에서 파생한다. provenance 자체는 능동 킥이 아니다.
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
        velocity_host = self._inverse_vector(
            "next_player_vel", next_player_vel, (self.N, DIM_Z)
        )
        next_player_vel = jnp.asarray(velocity_host, jnp.float32)
        # 실효 vmax는 ``Movement.effective_vmax``가 단일 진실원천이다. 여기서 같은 식을 다시
        # 적으면 이동 클립과 역산이 조용히 갈라진다 — 역산은 정확히 forward가 자르는 그 캡을
        # 기준으로 파워를 정규화해야 한다.
        vmax = self.effective_vmax(
            state.vmax, state.stamina_long, state.stamina_short
        )
        if np.any(
            np.linalg.norm(velocity_host, axis=1)
            > np.asarray(vmax) + 2.0e-4
        ):
            raise ValueError("next_player_vel exceeds effective_vmax")
        speed = _safe_norm(next_player_vel, axis=1, keepdims=True)
        dir_world = next_player_vel / (speed + DIV_EPS)
        mv_pow = jnp.clip(speed[:, 0] / (vmax + DIV_EPS), 0.0, 1.0)     # (N,) 목표속도/vmax
        dir_action = dir_world * state.attack_dir[:, None]              # 정책 프레임 재폴딩(단위)

        # 이동은 L∞ stretch 2D — 방향+크기를 [1:3] 한 벡터로(별도 pow dim 없음)
        mv_v = stretch_encode(dir_action, mv_pow)                       # (N,2): unit=dir, ‖·‖∞=mv_pow
        action = jnp.zeros((self.N, ACTION_DIM))
        action = action.at[:, ACTION_MOVE].set(mv_v)
        return action
