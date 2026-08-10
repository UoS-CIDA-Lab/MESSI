"""오프사이드 — 패스 순간 플래그된 동료가 pass_t 창 안에서 공을 먼저 터치하면 콜 → 수비팀 프리킥.

콜 여부는 궤적에서 직접 관측된다(플래그 선수의 터치 → RK_OFFSIDE 재개). 확률 요소가 없는
결정적 규칙이라 복원 시 재추첨이 필요 없다.
"""
import jax.numpy as jnp

from constants import *


class Offside:
    def _offside_check(self, state, touch_before=None):
        """플래그된 동료가 창 안에서 공을 터치하면 오프사이드 콜(수비팀 FK). 그 외에는 창을 감쇠·해제.

        수비의 '의도적 플레이'(탈취 TACKLE·인터셉트)나 pass_t 소진 시 플래그·창을 리셋한다.
        굴절(DEFLECT)은 IFAB상 창을 닫지 않으므로 여기서 리셋 트리거에서 제외된다.

        ``touch_before``는 **이 물리 서브스텝의 접촉 직전** ``state.touch`` 스냅샷이다.
        ``state.touch``는 컨트롤 스텝 단위로만 0으로 초기화되므로(env.step_env_array), 스냅샷 없이
        ``touch > 0``만 보면 **플래그가 서기 전(같은 스텝의 앞 서브스텝)에 찍힌 터치**가 그대로 콜
        조건을 만족시켜 오프사이드를 오검한다(패스 순간 set_flags → 같은 서브스텝의 stale touch로
        즉시 콜). 이번 서브스텝에 새로 생긴 접촉만 콜/리셋 트리거로 쓴다 —
        ``restart._throwin_restriction``의 재터치 판정과 동일한 규약이다.
        None이면 종전 동작(호환용).
        """
        e_cfg = self.e_cfg
        if touch_before is None:
            new_touch = state.touch > TOUCH_NONE
        else:
            new_touch = (state.touch > TOUCH_NONE) & (state.touch != touch_before)
        active = state.pass_t > 0
        flagged_touch = state.offside_flag & new_touch & (state.team_id == state.pass_team)
        called = active & jnp.any(flagged_touch)
        idx = jnp.argmax(flagged_touch)
        defender = (TEAM_1 - state.pass_team).astype(jnp.int32)

        off_pos = state.player_pos[idx]
        inset = e_cfg.free_kick_boundary_inset
        spot = jnp.array([
            jnp.clip(off_pos[DIM_X], -self.hx + inset, self.hx - inset),
            jnp.clip(off_pos[DIM_Y], -self.hy + inset, self.hy - inset),
            self.r_ball,
        ])
        taker = self._designate_taker(state, spot[:DIM_Z], defender, jnp.bool_(False))

        restart_kind = jnp.where(called, RK_OFFSIDE, state.restart_kind).astype(jnp.int32)
        restart_team = jnp.where(called, defender, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(called, jnp.int32(e_cfg.restart_substeps), state.restart_t)
        ball_state = jnp.where(called, BALL_DEAD, state.ball_state).astype(jnp.int32)
        ball_pos = jnp.where(called, spot, state.ball_pos)
        ball_vel = jnp.where(called, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(called, jnp.zeros(DIM_ALL), state.ball_spin)
        poss = jnp.where(called, defender, state.poss_team).astype(jnp.int32)
        pending_taker = jnp.where(called, taker, state.pending_taker).astype(jnp.int32)
        throw_taker = jnp.where(called, jnp.int32(-1), state.throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(called, jnp.int32(-1), state.setpiece_taker).astype(jnp.int32)

        # 오프사이드 콜도 새 재개(수비 FK)를 세우는 전이 지점이라, 다른 재개 경로(events·fouls·
        # restart)와 동일하게 진행 중이던 페널티 플라이트·간접FK 플래그를 클리어한다. 페널티 킥은
        # 리바운드 오프사이드 창을 무장(contest offside_restart_kick)하므로, 이 클리어가 없으면
        # 스테일 penalty_flight_team이 오프사이드 재개를 넘어 살아남아 뒤늦은 페널티 '재실행'을
        # 오발동해(스코어·배치 되돌림) 정당한 후속 골을 취소할 수 있다.
        penalty_flight = jnp.where(called, jnp.int32(-1), state.penalty_flight_team).astype(jnp.int32)
        penalty_enc = jnp.where(called, jnp.zeros_like(state.penalty_encroach_mask), state.penalty_encroach_mask)
        restart_indirect = jnp.where(called, jnp.bool_(True), state.restart_indirect)   # IFAB Law 11: 오프사이드=간접 FK(직접골 무효)

        deliberate = (((state.touch == TOUCH_TACKLE) | (state.touch == TOUCH_INTERCEPT))
                      & new_touch)
        defender_touch = jnp.any(deliberate & (state.team_id != state.pass_team) & (state.pass_team >= 0))
        pass_t_dec = jnp.where(active, state.pass_t - 1, state.pass_t)
        clear = called | defender_touch | (pass_t_dec <= 0)
        off_flag = jnp.where(clear, jnp.zeros(self.N, bool), state.offside_flag)
        pass_team = jnp.where(clear, jnp.int32(-1), state.pass_team).astype(jnp.int32)
        pass_t_new = jnp.where(clear, jnp.int32(0), pass_t_dec).astype(jnp.int32)

        return state._replace(restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
                              ball_state=ball_state, ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                              poss_team=poss, pending_taker=pending_taker, pass_t=pass_t_new,
                              offside_flag=off_flag, pass_team=pass_team, throw_taker=throw_taker,
                              setpiece_taker=setpiece_taker, penalty_flight_team=penalty_flight,
                              penalty_encroach_mask=penalty_enc, restart_indirect=restart_indirect)
