"""오프사이드 — 패스 순간 플래그된 동료가 해당 플레이 국면에서 공을 먼저
터치하면 콜 → 수비팀 프리킥. 국면은 시간이 아니라 다음 동료 플레이·수비의 의도적
플레이·아웃/재개에서만 종료된다.

콜 여부는 궤적에서 직접 관측된다(플래그 선수의 터치 → RK_OFFSIDE 재개). 확률 요소가 없는
결정적 규칙이라 복원 시 재추첨이 필요 없다.
"""
import jax
import jax.numpy as jnp

from .constants import (
    BALL_ALIVE,
    BALL_DEAD,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    NO_TEAM,
    RK_OFFSIDE,
    TEAM_0,
    TEAM_1,
    TOUCH_BODY_TRAP,
    TOUCH_DEFLECT,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PARRY,
    TOUCH_TACKLE,
)
from .restart import restart_timer_active


class Offside:
    def _offside_positions_for_play(
        self, state, actor, team, ball_x_at_play=None
    ):
        """Return team-mates in an offside position at this deliberate play.

        Active f2b plays and controlled body traps must use exactly the same
        Law-11 line/ball/half predicates.  Keeping the geometry here avoids a
        silent split where contest kicks and passive-physics control arm
        different receiver sets.
        """

        attack_dir = state.attack_dir[actor]
        rule_pos = self._player_pos_for_field_rules(state.player_pos)
        x_att = rule_pos[:, DIM_X] * attack_dir
        ball_x_world = (
            state.ball_pos[DIM_X]
            if ball_x_at_play is None
            else jnp.asarray(ball_x_at_play, dtype=state.ball_pos.dtype)
        )
        ball_x_att = ball_x_world * attack_dir
        defenders = (state.team_id != team) & state.active_player
        x_def = jnp.where(defenders, x_att, -jnp.inf)
        line = jax.lax.top_k(x_def, 2)[0][1]
        n_def = jnp.sum(defenders.astype(jnp.int32))
        line = jnp.where(n_def >= 2, line, jnp.inf)
        return (
            (state.team_id == team)
            & state.active_player
            & (self.player_indices != actor)
            & (x_att > line + self.e_cfg.offside_margin)
            & (x_att > ball_x_att)
            & (x_att > 0)
        )

    def _normalize_pass_latch(self, state, clear=False):
        """Keep the offside/pass-window triplet structurally coherent.

        ``pass_t`` is a binary active latch kept for the public State layout;
        it owns the lifetime of ``pass_team`` and ``offside_flag``.  A new
        referee restart can force-clear the latch; invalid or empty flag sets
        clear it automatically.  There is deliberately no elapsed-time expiry:
        IFAB Law 11 ends the phase through a subsequent play, not a stopwatch.
        Centralizing this
        rule prevents goal/out and new-restart writers from exposing a zero
        timer with stale identities for even one public frame.
        """

        valid_team = (state.pass_team == TEAM_0) | (state.pass_team == TEAM_1)
        flags_match_team = jnp.all(
            (~state.offside_flag) | (state.team_id == state.pass_team)
        )
        keep = (
            (state.pass_t > 0)
            & valid_team
            & jnp.any(state.offside_flag)
            & flags_match_team
            & (~jnp.asarray(clear, dtype=bool))
        )
        return state._replace(
            pass_t=jnp.where(keep, jnp.int32(1), jnp.int32(0)).astype(jnp.int32),
            pass_team=jnp.where(
                keep, state.pass_team, jnp.int32(NO_TEAM)
            ).astype(jnp.int32),
            offside_flag=jnp.where(
                keep, state.offside_flag, jnp.zeros_like(state.offside_flag)
            ),
        )

    def _offside_check(
        self, state, touch_before, touch_after_force=None,
        body_ball_pos=None, force_touch_mask=None, body_touch_mask=None,
    ):
        """플래그된 동료가 해당 플레이 국면에서 공을 터치하면 오프사이드 콜(수비팀 FK).

        수비의 '의도적 플레이'나 새 국면에서 플래그·래치를 리셋한다. 의도적 플레이는
        제외목록으로 정의한다 — 굴절(DEFLECT)과 세이브(GK_CATCH·PARRY)만 국면을 닫지
        않고, 탈취·인터셉트·트랩은 물론 클리어링 패스·헤딩 클리어·드리블·슛도 닫는다.
        허용목록으로 두면 수비가 루즈볼을 걷어낸 뒤에도 국면이 살아 있어, 그 공을 만진
        이전 플래그 공격수에게 **거짓 오프사이드**가 선언된다.

        ``touch_before``는 **이 물리 서브스텝의 접촉 직전** ``state.touch`` 스냅샷이고,
        ``touch_after_force``는 능동 경합 직후(몸통 충돌 전) 스냅샷이다.
        ``body_ball_pos``는 몸통 충돌 직전 공 위치다. 트랩 저장점은 선수 앞 0.35 m로
        이동하므로 그 사후 좌표로 Law-11의 패스 순간을 잡으면 경계 선수가 잘못 분류된다.
        ``state.touch``는 컨트롤 스텝 단위로만 0으로 초기화되므로(env.step_env_array), 스냅샷 없이
        ``touch > 0``만 보면 **플래그가 서기 전(같은 스텝의 앞 서브스텝)에 찍힌 터치**가 그대로 콜
        조건을 만족시켜 오프사이드를 오검한다(패스 순간 set_flags → 같은 서브스텝의 stale touch로
        즉시 콜). 이번 서브스텝에 새로 생긴 접촉만 콜/리셋 트리거로 쓴다 —
        ``restart._throwin_restriction``의 재터치 판정과 동일한 규약이다.
        두 스냅샷으로 ``force2ball -> body``의 실제 접촉 순서를 보존한다. 수비수가 먼저
        의도적으로 플레이해 오프사이드 국면을 끝낸 뒤 플래그 공격수 몸에 맞은 공을 오프사이드로
        되돌리면 안 되고, 반대로 플래그 공격수가 먼저 건드린 뒤 수비 몸에 맞은 경우에는 이미
        성립한 오프사이드를 지우면 안 된다.

        직접 호출의 하위 호환을 위해 ``touch_after_force=None``이면 모든 새 접촉을 첫 phase로
        취급한다. 환경 전이 경로는 항상 명시적 스냅샷을 넘긴다.
        """
        e_cfg = self.e_cfg
        if touch_after_force is None:
            touch_after_force = state.touch
        if body_ball_pos is None:
            body_ball_pos = state.ball_pos

        force_touch = (
            ((touch_after_force > TOUCH_NONE)
             & (touch_after_force != touch_before))
            if force_touch_mask is None
            else jnp.asarray(force_touch_mask, dtype=bool)
        )
        body_touch = (
            ((state.touch > TOUCH_NONE)
             & (state.touch != touch_after_force))
            if body_touch_mask is None
            else jnp.asarray(body_touch_mask, dtype=bool)
        )
        active = state.pass_t > 0

        def phase(flags, phase_team, new_touch, phase_active, codes):
            flagged_touch = flags & new_touch & (state.team_id == phase_team)
            called = phase_active & jnp.any(flagged_touch)
            idx = jnp.argmax(flagged_touch)
            # [IFAB Law 11.2] 국면은 상대의 **의도적 플레이**로 끝난다. 예외는 '의도적
            # 세이브'다. 그래서 허용목록이 아니라 제외목록으로 정의한다 — 허용목록은
            # 새 터치 코드가 생길 때마다 조용히 국면을 안 닫는 쪽으로 틀린다.
            # DEFLECT는 이 환경의 정의 자체가 '탈취 아님, 의도적 플레이 아님'이고,
            # GK_CATCH/PARRY는 세이브라 둘 다 국면을 닫지 않는다.
            not_deliberate = (
                (codes == TOUCH_DEFLECT)
                | (codes == TOUCH_PARRY)
                | (codes == TOUCH_GK_CATCH)
            )
            deliberate = new_touch & (codes > TOUCH_NONE) & (~not_deliberate)
            defender_touch = phase_active & jnp.any(
                deliberate
                & (state.team_id != phase_team)
                & (phase_team >= 0)
            )
            cleared = called | defender_touch
            next_flags = jnp.where(cleared, jnp.zeros(self.N, bool), flags)
            return called, idx, defender_touch, next_flags

        called_force, idx_force, defender_force, old_flags_after_force = phase(
            state.offside_flag, state.pass_team, force_touch, active,
            touch_after_force,
        )
        force_closed = called_force | defender_force

        # A successful tackle/interception is itself a new deliberate play.
        # First adjudicate it against the old phase (a flagged attacker cannot
        # escape by labelling the same touch TACKLE), then arm the controller's
        # team before the later body-collision phase.  Without this ordered
        # re-arm, a defender could intentionally play the ball to an already
        # offside team-mate with no Law-11 consequence.
        force_control = force_touch & (
            (touch_after_force == TOUCH_TACKLE)
            | (touch_after_force == TOUCH_INTERCEPT)
        ) & (state.ball_state == BALL_ALIVE) & (
            ~restart_timer_active(state.restart_t)
        )
        force_actor = jnp.argmax(force_control).astype(jnp.int32)
        force_played = jnp.any(force_control) & (~called_force)
        force_team = state.team_id[force_actor].astype(jnp.int32)
        force_flags = self._offside_positions_for_play(
            state, force_actor, force_team
        )
        force_has_flags = jnp.any(force_flags)
        flags_for_body = jnp.where(
            force_played, force_flags, old_flags_after_force
        )
        team_after_force = jnp.where(
            force_played,
            jnp.where(force_has_flags, force_team, jnp.int32(NO_TEAM)),
            jnp.where(force_closed, jnp.int32(NO_TEAM), state.pass_team),
        ).astype(jnp.int32)
        active_after_force = (
            (~called_force) & jnp.any(flags_for_body)
            & (team_after_force >= 0)
        )

        called_body, idx_body, defender_body, flags_after_body = phase(
            flags_for_body, team_after_force, body_touch,
            active_after_force, state.touch,
        )
        called = called_force | called_body
        idx = jnp.where(called_force, idx_force, idx_body).astype(jnp.int32)
        called_team = jnp.where(
            called_force, state.pass_team,
            jnp.where(called_body, team_after_force, jnp.int32(TEAM_0)),
        ).astype(jnp.int32)
        defender = (TEAM_1 - called_team).astype(jnp.int32)

        off_pos = state.player_pos[idx]
        inset = e_cfg.free_kick_boundary_inset
        spot = jnp.array([
            jnp.clip(off_pos[DIM_X], -self.hx + inset, self.hx - inset),
            jnp.clip(off_pos[DIM_Y], -self.hy + inset, self.hy - inset),
            self.r_ball,
        ])
        # 오프사이드 재개는 간접 프리킥과 같은 규칙이다.
        taker = self._designate_taker_when(
            called,
            state, spot[:DIM_Z], defender, jnp.bool_(False),
            jnp.int32(RK_OFFSIDE), jnp.bool_(True))

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

        # 오프사이드 콜은 간접 프리킥으로 재개한다.
        restart_indirect = jnp.where(called, jnp.bool_(True), state.restart_indirect)   # IFAB Law 11: 오프사이드=간접 FK(직접골 무효)

        clear = called | defender_body
        off_flag = jnp.where(clear, jnp.zeros(self.N, bool), flags_after_body)
        pass_team = jnp.where(
            clear, jnp.int32(NO_TEAM), team_after_force
        ).astype(jnp.int32)
        pass_t_new = jnp.where(
            clear, jnp.int32(0), active_after_force.astype(jnp.int32)
        ).astype(jnp.int32)

        # A controlled chest/body trap is a new deliberate play for Law 11,
        # even though it is not a foot kick for the goalkeeper back-pass law
        # or BC kick causality.  First honour an offence/defender-clear from
        # the pre-existing window, then (unless an offence was called) arm a
        # fresh window from the controller's team and current positions.  This
        # handles both directions: an attacker who moved from onside to
        # offside is newly flagged, while stale flags disappear when the line
        # has moved the other way.  Passive DEFLECT never reaches this branch.
        body_control = body_touch & (state.touch == TOUCH_BODY_TRAP)
        body_actor = jnp.argmax(body_control).astype(jnp.int32)
        body_played = jnp.any(body_control) & (~called)
        body_team = state.team_id[body_actor].astype(jnp.int32)
        body_flags = self._offside_positions_for_play(
            state, body_actor, body_team,
            ball_x_at_play=body_ball_pos[DIM_X],
        )
        off_flag = jnp.where(body_played, body_flags, off_flag)
        body_has_flags = jnp.any(body_flags)
        pass_team = jnp.where(
            body_played,
            jnp.where(body_has_flags, body_team, jnp.int32(NO_TEAM)),
            pass_team,
        ).astype(jnp.int32)
        pass_t_new = jnp.where(
            body_played, body_has_flags.astype(jnp.int32), pass_t_new
        ).astype(jnp.int32)

        result = state._replace(
            restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
            ball_state=ball_state, ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
            poss_team=poss, pending_taker=pending_taker, pass_t=pass_t_new,
            offside_flag=off_flag, pass_team=pass_team, throw_taker=throw_taker,
            setpiece_taker=setpiece_taker, restart_indirect=restart_indirect,
            gk_handling_restricted_team=jnp.where(
                called, jnp.int32(NO_TEAM),
                state.gk_handling_restricted_team,
            ).astype(jnp.int32),
        )
        return self._normalize_pass_latch(result)
