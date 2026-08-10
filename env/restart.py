import jax.numpy as jnp


from constants import *

class Restart:
    def canonical_restart_spot(
        self,
        state,
        restart_kind,
        restart_team,
        observed_ball_pos=None,
    ):
        """Return SoccerEnv's legal spot for an externally observed restart.

        Reconstruction supplies only the observed kind/team boundary.  Spot
        geometry remains owned by the environment so DFL adapters cannot grow
        a second, subtly different implementation of touchline/goal-area
        insets.
        """

        e_cfg = self.e_cfg
        observed = (
            state.ball_pos
            if observed_ball_pos is None
            else jnp.asarray(observed_ball_pos)
        )
        inset = e_cfg.restart_field_inset
        center = jnp.asarray([0.0, 0.0, self.r_ball])
        throwin = jnp.asarray([
            jnp.clip(observed[DIM_X], -self.hx + inset, self.hx - inset),
            jnp.sign(observed[DIM_Y] + DIV_EPS)
            * (self.hy - e_cfg.throwin_line_inset),
            self.r_ball,
        ])
        corner = jnp.asarray([
            jnp.sign(observed[DIM_X] + DIV_EPS) * (self.hx - inset),
            jnp.sign(observed[DIM_Y] + DIV_EPS) * (self.hy - inset),
            self.r_ball,
        ])
        goalkick = jnp.asarray([
            jnp.sign(observed[DIM_X] + DIV_EPS)
            * (self.hx - self.s_cfg.goal_area_length),
            0.0,
            self.r_ball,
        ])
        safe_team = jnp.clip(restart_team, TEAM_0, TEAM_1)
        team_slot = jnp.where(safe_team == TEAM_0, 0, self.n_agents)
        attack = state.attack_dir[team_slot]
        penalty = jnp.asarray([
            attack * (self.hx - e_cfg.penalty_spot),
            0.0,
            self.r_ball,
        ])
        free = jnp.asarray([
            jnp.clip(
                observed[DIM_X],
                -self.hx + e_cfg.free_kick_boundary_inset,
                self.hx - e_cfg.free_kick_boundary_inset,
            ),
            jnp.clip(
                observed[DIM_Y],
                -self.hy + e_cfg.free_kick_boundary_inset,
                self.hy - e_cfg.free_kick_boundary_inset,
            ),
            self.r_ball,
        ])
        return jnp.where(
            restart_kind == RK_KICKOFF,
            center,
            jnp.where(
                restart_kind == RK_THROWIN,
                throwin,
                jnp.where(
                    restart_kind == RK_CORNER,
                    corner,
                    jnp.where(
                        restart_kind == RK_GOALKICK,
                        goalkick,
                        jnp.where(restart_kind == RK_PENALTY, penalty, free),
                    ),
                ),
            ),
        )

    def prepare_observed_restart(
        self,
        state,
        restart_kind,
        restart_team,
        pending_taker,
        restart_t,
        observed_ball_pos=None,
    ):
        """Initialize one observed referee boundary without advancing physics.

        This pure adapter is used by offline reconstruction exactly once at a
        DFL dead-ball onset.  Player positions are not teleported; the normal
        dead-ball engine subsequently walks the taker and reorganizes all
        players.  Random or simulated outs remain independently suppressible.
        """

        spot = self.canonical_restart_spot(
            state,
            restart_kind,
            restart_team,
            observed_ball_pos,
        )
        return state._replace(
            ball_pos=spot,
            ball_vel=jnp.zeros(DIM_ALL),
            ball_spin=jnp.zeros(DIM_ALL),
            ball_state=jnp.int32(BALL_DEAD),
            poss_team=restart_team.astype(jnp.int32),
            last_touch_team=restart_team.astype(jnp.int32),
            restart_team=restart_team.astype(jnp.int32),
            restart_t=restart_t.astype(jnp.int32),
            restart_kind=restart_kind.astype(jnp.int32),
            pending_taker=pending_taker.astype(jnp.int32),
            setpiece_taker=jnp.int32(NO_PLAYER),
            throw_taker=jnp.int32(NO_PLAYER),
            offside_flag=jnp.zeros_like(state.offside_flag),
            pass_team=jnp.int32(NO_TEAM),
            pass_t=jnp.int32(0),
            foul_kind=jnp.int32(FOUL_NONE),
            foul_actor=jnp.int32(NO_PLAYER),
            foul_victim=jnp.int32(NO_PLAYER),
            penalty_flight_team=jnp.int32(NO_TEAM),
            penalty_encroach_mask=jnp.zeros_like(
                state.penalty_encroach_mask
            ),
            restart_indirect=jnp.bool_(False),
        )

    def synchronize_observed_restart(
        self,
        state,
        restart_kind,
        restart_team,
        pending_taker,
        restart_t,
    ):
        """Pin only the observed referee phase/timer until physical release."""

        return state._replace(
            ball_state=jnp.int32(BALL_DEAD),
            restart_team=restart_team.astype(jnp.int32),
            restart_t=restart_t.astype(jnp.int32),
            restart_kind=restart_kind.astype(jnp.int32),
            pending_taker=pending_taker.astype(jnp.int32),
        )

    def _encroach_geometry(self, state):
        """세트피스 침범 판정 기하(단일 진실원천) — contest 재실행 판정과 obs 인코딩이 공유.
        render_rich의 제한구역 오버레이도 이 기하 규약을 따른다(numpy 측 복제 — light 렌더엔 오버레이 없음).
        재개 활성 게이팅은 호출자 책임(여기는 순수 기하만).
        반환:
          clear_r     : 규정 이격 반경(m) — 킥오프=센터서클 / 스로인=throwin_clear / 그 외 clear_dist
          encroachers : bool[N] 침범자.
                        · 골킥: 상대팀 & 온피치 & 재개팀 자기 박스 안 — 반경 조항 없음
                          (IFAB Law 16은 '박스 밖'만 요구. 반경 9.15 추가 요구는 과엄격이라
                          2026-07-16 제거 — 박스 밖 8m 합법 압박을 침범 오판·retake 루프 유발)
                        · 페널티: 온피치 & 공격 끝 박스+아크 안 & 키커·수비GK 제외 (★양 팀 걸침)
                        · GK_HOLD: 침범자 없음(상대는 '도전 불가'일 뿐 이격 의무 없음)
                        · 그 외: 상대팀 & 온피치 & 반경 안 & **자기 골라인 밴드 제외**(IFAB Law 13)
          margin      : float[N] 규정 준수까지의 서명 여유거리(m) — 음수=침범 깊이, 양수=여유.
                        골킥=자기 박스 이탈 sd / 페널티=공격 박스+아크 이탈 sd /
                        골라인 면제자와 GK_HOLD는 Engine의 양의 margin floor로 클램프.
        ★주의: 페널티 침범자는 수비팀만이 아니라 '키커·수비GK 제외 전원'이다. obs 인코딩에서
          기존 골킥용 is_def(수비팀만) 마스크를 페널티에 그대로 쓰면 공격수 쇄도를 놓친다."""
        e_cfg = self.e_cfg
        s_cfg = self.s_cfg
        pen_hw = s_cfg.penalty_area_width / 2.0
        pen_len = s_cfg.penalty_area_length
        hx = self.hx
        clear_r = jnp.where(state.restart_kind == RK_KICKOFF, s_cfg.center_circle_radius,
                            jnp.where(state.restart_kind == RK_THROWIN, e_cfg.throwin_clear, e_cfg.clear_dist))
        d_spot = jnp.linalg.norm(state.player_pos - state.ball_pos[:2][None, :], axis=1)
        px, py = state.player_pos[:, 0], state.player_pos[:, 1]
        # adir_rt = 재개팀의 공격 방향(골킥이면 수비팀, 페널티면 공격팀 — restart_kind에 따라 팀 역할이 다름).
        adir_rt = jnp.where(state.restart_team == 0, state.attack_dir[0], state.attack_dir[self.n_agents])

        # 박스 서명거리 헬퍼: x∈[x_lo,x_hi] & |py|≤pen_hw 직사각형. 내부 = -(최근접 경계 깊이), 외부 = 경계 유클리드.
        def _box_signed(goal_x):
            box_back = goal_x - jnp.sign(goal_x) * pen_len   # 골라인에서 필드 안쪽으로 pen_len
            x_lo = jnp.minimum(goal_x, box_back); x_hi = jnp.maximum(goal_x, box_back)
            in_box = (x_lo <= px) & (px <= x_hi) & (jnp.abs(py) <= pen_hw)
            depth_in = jnp.minimum(jnp.minimum(px - x_lo, x_hi - px), pen_hw - jnp.abs(py))
            dx_out = jnp.maximum(jnp.maximum(x_lo - px, px - x_hi), 0.0)
            dy_out = jnp.maximum(jnp.abs(py) - pen_hw, 0.0)
            sd = jnp.where(in_box, -depth_in, jnp.sqrt(dx_out ** 2 + dy_out ** 2))
            return in_box, sd

        # 골킥: 재개팀 '자기' 박스(수비 끝, -adir_rt*hx). 페널티: '공격 끝' 박스(상대 골대, +adir_rt*hx).
        gk_in_box, gk_sd = _box_signed(-adir_rt * hx)
        pen_in_box, pen_sd = _box_signed(adir_rt * hx)

        # 페널티 면제자: 키커(pending_taker) + 수비팀 GK(골라인 잔류 허용). 그 외 전원이 박스 밖이어야 함.
        ar = self.player_indices
        is_kicker = (ar == state.pending_taker) & (state.pending_taker >= 0)
        defending_team = 1 - state.restart_team
        is_def_gk = (state.gk_indices == 1) & (state.team_id == defending_team)
        # IFAB Law 14: 박스 밖 **그리고** 스폿 9.15m(페널티 아크) 밖이어야 규정 준수 —
        # 박스 밖이라도 아크 안(박스 모서리 바깥 초승달 지대) 대기는 침범.
        pen_arc = d_spot < s_cfg.penalty_arc_radius
        # 골라인 잔류(밴드 1.2m: 정책 GK가 0.5~1m 앞 각도커팅까지 '라인 위' 인정 — 과소밴드 시 루프 재발).
        # Law 13 예외(비페널티): 자기 골라인 위 수비수는 9.15m 이격 면제(골문 앞 간접FK서 영구침범→
        # 카드누적 퇴장 루프 방지). Law 14(페널티): 수비 GK는 골라인 위에서만 면제. 양쪽서 쓰므로 먼저 계산.
        own_gx = -state.attack_dir * hx
        on_goal_line = (
            (jnp.abs(px - own_gx) <= e_cfg.goal_line_tolerance)
            & (jnp.abs(py) <= s_cfg.goal_width / 2.0 + e_cfg.goal_post_tolerance)
        )
        # 페널티 침범: 비-GK는 박스·아크 밖 의무, 수비 GK는 '골라인 이탈' 시 침범(예전 전면 면제 →
        # GK가 스폿 옆 주차해도 합법이던 것, B2 — Law 14 GK 골라인 의무). 키커는 면제.
        pen_encroach = (((pen_in_box | pen_arc) & (~is_def_gk)) | (is_def_gk & (~on_goal_line))) \
                       & state.active_player & (~is_kicker)

        margin_r = d_spot - clear_r
        is_pen = state.restart_kind == RK_PENALTY
        is_gkk = state.restart_kind == RK_GOALKICK
        pen_margin = jnp.minimum(pen_sd, d_spot - s_cfg.penalty_arc_radius)
        margin = jnp.where(is_pen, pen_margin,
                           jnp.where(is_gkk, gk_sd, margin_r))   # 골킥=박스 서명거리만(Law 16)
        is_def = state.team_id != state.restart_team
        margin = jnp.where((~is_pen) & is_def & on_goal_line,
                           jnp.maximum(margin, e_cfg.legal_margin_floor), margin)
        # GK 홀드는 이격 의무가 없다(상대는 '도전 불가'일 뿐) — 기본 clear_dist가 흘러들면
        # obs가 존재하지 않는 9.15m 후퇴 의무를 신호(음수 margin·any_enc)하게 된다.
        is_hold = state.restart_kind == RK_GK_HOLD
        margin = jnp.where(
            is_hold, jnp.maximum(margin, e_cfg.unrestricted_margin), margin
        )
        encroachers = jnp.where(
            is_pen,
            pen_encroach,
            is_def & state.active_player & (~on_goal_line) & (~is_hold)
            & jnp.where(is_gkk, gk_in_box, d_spot < clear_r))
        return clear_r, encroachers, margin

    def _setpiece_kick_lock(self, state):
        """키커 강제이동/정렬 중 킥 불가 여부 판단.
        키커는 setup 완료 전까지 강제이동, 도착 후엔 제자리 고정. setup 완료 전까지 킥 불가.
        setup 완료: 도착 && 카운트다운 소진. (도착 전엔 카운트다운이 흐르지 않음)
        Args:
            state: State
        Return: (전부 스칼라 bool — pending_taker 단일 인덱스 기준)
            kicker_locked: 키커 강제이동/정렬 중 킥 불가 여부
            setup_done: setup 완료 여부 (도착 && 카운트다운 소진)
            arrived: 키커 도착 여부
            active: 키커 강제이동/정렬 활성화 여부
        """
        e_cfg = self.e_cfg
        restart_active = state.restart_t > 0
        is_pen = state.restart_kind == RK_PENALTY
        has_taker = state.pending_taker >= 0
        active = restart_active & has_taker            # 페널티도 포함(물리 플레이: 키커 스폿 접근·킥)
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        tgt = self._kicker_target(state)
        d_tgt = jnp.linalg.norm(state.player_pos[taker] - tgt)
        arrived = active & (d_tgt <= e_cfg.kicker_arrive_r)
        # 도착 후에만 restart_t가 흐르므로 setup 종료 = 도착 && 카운트다운 소진. 초기 창은 종류별 상이
        # (페널티=penalty_substeps, 그 외=restart_substeps).
        window = jnp.where(is_pen, e_cfg.penalty_substeps,
                 jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps, e_cfg.restart_substeps))
        # 킥오프 즉시 발동: setup_hold 없이 도착 즉시 setup 완료(그 뒤 _kick_gate가 바로 강제 킥).
        instant_ko = jnp.bool_(e_cfg.kickoff_instant) & (state.restart_kind == RK_KICKOFF)
        setup_done = arrived & (instant_ko | (state.restart_t <= (window - e_cfg.setup_hold_substeps)))
        kicker_locked = active & (~setup_done)        # 도착 전 or 정렬 중 = 킥 불가
        return kicker_locked, setup_done, arrived, active

    def _throwin_restriction(
        self, state, throw_taker_before, setpiece_taker_before, touch_before
    ):
        """세트피스 키커 재터치 금지(Law 15 스로인 + 코너/골킥/FK/킥오프 일반).

        키커는 공이 다른 선수에 닿기 전 재터치하면 위반 → 상대팀 프리킥. 다른 선수가 먼저 닿으면
        제한 해제(합법). ``*_before``와 ``touch_before``는 바로 이 물리 서브스텝의 접촉 직전
        값이다. 컨트롤 스텝 입구 값을 재사용하면 같은 스텝 후반에 제3자가 먼저 터치한 사실을
        놓쳐 다음 프레임에 원 키커를 오심 처리한다. touch 확정(force2ball·ball_body) 뒤 호출한다.
        """
        e_cfg = self.e_cfg
        N = self.N

        new_touch = (state.touch > TOUCH_NONE) & (state.touch != touch_before)

        def _detect(taker_before, taker_now):
            # 이번 서브스텝의 재개 킥으로 제한이 새로 생긴 경우에는 그 최초 킥 자체는 재터치가
            # 아니다. 다만 같은 서브스텝에 다른 선수 몸에 맞았다면 즉시 제한을 해제한다.
            active = taker_now >= 0
            existed = taker_before >= 0
            effective = jnp.where(existed, taker_before, taker_now)
            taker = jnp.clip(effective, 0, N - 1)
            others = self.player_indices != taker
            other_touched = active & jnp.any(new_touch & others)
            taker_touched = active & existed & new_touch[taker]
            foul = taker_touched & (~other_touched)
            return foul, other_touched, taker

        foul_throw, other_throw, taker_throw = _detect(
            throw_taker_before, state.throw_taker
        )
        foul_sp, other_sp, taker_sp = _detect(
            setpiece_taker_before, state.setpiece_taker
        )
        foul = foul_throw | foul_sp
        taker = jnp.where(foul_throw, taker_throw, taker_sp)

        offender_team = state.team_id[taker].astype(jnp.int32)
        defender = (TEAM_1 - offender_team).astype(jnp.int32)
        px, py = state.ball_pos[DIM_X], state.ball_pos[DIM_Y]
        inset = e_cfg.free_kick_boundary_inset
        fk_spot = jnp.array([
            jnp.clip(px, -self.hx + inset, self.hx - inset),
            jnp.clip(py, -self.hy + inset, self.hy - inset),
            self.r_ball,
        ])
        taker_fk = self._designate_taker(state, fk_spot[:DIM_Z], defender, jnp.bool_(False))

        restart_kind = jnp.where(foul, RK_FREEKICK, state.restart_kind).astype(jnp.int32)
        restart_team = jnp.where(foul, defender, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(foul, jnp.int32(e_cfg.restart_substeps), state.restart_t).astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state).astype(jnp.int32)
        ball_pos = jnp.where(foul, fk_spot, state.ball_pos)
        ball_vel = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_spin)
        poss = jnp.where(foul, defender, state.poss_team).astype(jnp.int32)
        pending_taker = jnp.where(foul, taker_fk, state.pending_taker).astype(jnp.int32)
        foul_kind = jnp.where(foul, jnp.int32(FOUL_THROW), state.foul_kind)
        # actor/victim 갱신 — 누락 시 렌더·통계가 stale 값 또는 -1(P[-1]=마지막 선수)을 지목한다.
        foul_actor = jnp.where(foul, taker.astype(jnp.int32), state.foul_actor)
        foul_victim = jnp.where(foul, jnp.int32(-1), state.foul_victim)   # 재터치는 피해자 없음
        # 각 제한 해제: 자기 케이스가 반칙이거나 다른 선수가 먼저 닿으면 해제
        throw_taker = jnp.where(foul | foul_throw | other_throw, jnp.int32(-1), state.throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(foul | foul_sp | other_sp, jnp.int32(-1), state.setpiece_taker).astype(jnp.int32)
        # 재터치 반칙으로 새 재개가 서면 진행 중이던 페널티 플라이트는 무효(스테일 재실행 방지)
        pen_flight = jnp.where(foul, jnp.int32(-1), state.penalty_flight_team).astype(jnp.int32)
        pen_enc = jnp.where(foul, jnp.zeros_like(state.penalty_encroach_mask), state.penalty_encroach_mask)
        # 간접FK 플래그: 재터치 반칙은 IFAB Law 13/15상 **간접 FK**(직접골 무효)로 선다 — 특히
        # 자기 박스 안 재터치(GK 배급 재캐치 등)가 직접 FK면 상대에게 공짜 골 각이 된다.
        # 두 번째 터치(other_*)는 기존 IDFK 보호 종료(False).
        restart_indirect = jnp.where(foul, jnp.bool_(True),
                                     jnp.where(other_sp | other_throw, jnp.bool_(False),
                                               state.restart_indirect))
        # 새 재개는 오프사이드 창을 리셋(다른 재개 전이와 동일 규약) — 스테일 pass_signal obs 방지.
        off_clear = jnp.where(foul, jnp.zeros(self.N, bool), state.offside_flag)
        pass_t_clear = jnp.where(foul, jnp.int32(0), state.pass_t).astype(jnp.int32)
        pass_team_clear = jnp.where(foul, jnp.int32(-1), state.pass_team).astype(jnp.int32)
        return state._replace(restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
                              ball_state=ball_state, ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                              poss_team=poss, pending_taker=pending_taker, foul_kind=foul_kind,
                              foul_actor=foul_actor, foul_victim=foul_victim,
                              throw_taker=throw_taker, setpiece_taker=setpiece_taker,
                              penalty_flight_team=pen_flight, penalty_encroach_mask=pen_enc,
                              restart_indirect=restart_indirect,
                              offside_flag=off_clear, pass_t=pass_t_clear, pass_team=pass_team_clear)
