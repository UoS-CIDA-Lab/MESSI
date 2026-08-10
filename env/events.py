"""이벤트 판정 — 득점/아웃(코너·골킥·스로인)/페널티 해소 라우팅(_events), 킥오프 배치, 하프타임 전환.

전부 결정적 기하 판정(페널티도 물리 플레이 — 킥 궤적이 골/세이브/아웃으로 자연 해소, xG 주사위
없음)이라 관측 궤적(공 위치·재개 종류·스코어)에서 이벤트를 그대로 역산할 수 있다.
"""
import jax
import jax.numpy as jnp

from constants import *


class Events:
    def _kickoff_taker(self, state, team, positions):
        """킥오프 키커 지정 — team 온피치 선수 중 센터스팟(0,0)에 유클리드 최근접(원본
        _designate_taker와 동일 기준). 포메이션 고정이라 매 킥오프 동일 슬롯이 결정적으로
        선택된다(동률은 최소 인덱스). |x|(라인 거리)만 보면 y가 큰 윙어가 중앙 스트라이커를
        제치고 뽑혀 20m+를 걸어오는 오선택이 난다."""
        base = (state.team_id == team) & state.active_player
        dist_to_spot = jnp.linalg.norm(positions, axis=1)
        taker = jnp.argmin(jnp.where(base, dist_to_spot, jnp.inf)).astype(jnp.int32)
        return jnp.where(jnp.any(base), taker, jnp.int32(NO_PLAYER))

    def _kickoff_positions(self, state):
        """킥오프 기준 포메이션(N,2) — base_formation을 현재 공격방향 부호로 **점대칭**(x·y 동시
        반전, 180° 회전). 골 후 킥오프·후반 전환의 선수 재배치에 공용.
        전 구간 회전 규약(setup의 ROTATE_180도 x·y 양쪽, obs 폴딩도 z회전)이므로 x만 뒤집으면
        진영 교대 시 좌/우 레인이 공격 프레임 기준으로 반전된다(전반 왼쪽 윙이 후반 오른쪽 윙이 됨)."""
        flip = jnp.where(state.attack_dir[0] > 0, 1.0, -1.0)
        return self.base_formation * flip

    def _events(self, state, key, arrived=None, sp_active=None, suppress_retake=None,
                suppress_restart=None):
        """득점·아웃·페널티 해소를 판정하고 재개(restart) 상태·배치를 라우팅. 반환 (state, scored).

        goal: 골문 통과 → 킥오프(득점팀 상대 공). out_end: 골라인 아웃 → 코너/골킥. out_side:
        터치라인 아웃 → 스로인. penalty: 물리 플레이 킥의 터미널(골/아웃/세이브/정지)을 관측해
        침범 여부와 결합, IFAB 재실행 매트릭스로 해소(xG 주사위 없음 — 아래 침범 재실행 블록).
        스로인 직접골은 양방향 무효(상대골→골킥/자책→코너, Law 15), 그 외 세트피스 직접 자책골은
        무효(코너 처리). 비이벤트 재개 프레임은 키커 도착 후에만 restart_t를 감소시킨다.
        """
        e_cfg = self.e_cfg
        hx, hy = self.hx, self.hy
        x = state.ball_pos[DIM_X]
        y = state.ball_pos[DIM_Y]
        z = state.ball_pos[DIM_Z]

        # 페널티는 물리 플레이(키커가 실제 킥, GK는 필드선수로 경합) — xG 주사위 없음.
        # 골/세이브/아웃은 아래 일반 로직(골 검출·contest·아웃)으로 자연 해소된다.
        adir0 = state.attack_dir[0]
        team_plus = jnp.where(adir0 > 0, TEAM_0, TEAM_1).astype(jnp.int32)  # +x 골대를 공격하는 팀
        team_minus = (TEAM_1 - team_plus).astype(jnp.int32)
        # IFAB: 공 '전체'가 라인을 넘어야 골/아웃 → 판정선 = 라인 + r_ball(공 중심 기준 환산).
        # 골문 통과(y·z)는 서브스텝 말 샘플이 아니라 판정선 '교차 시점'의 값으로 역내삽 —
        # 40m/s 공은 샘플까지 최대 0.4m를 지나쳐 크로스바·포스트 근처 오분류 밴드가 생긴다.
        line_x = hx + self.r_ball
        vx, vy, vz = state.ball_vel[DIM_X], state.ball_vel[DIM_Y], state.ball_vel[DIM_Z]
        overshoot = jnp.abs(x) - line_x
        frac_dt = jnp.clip(overshoot / (jnp.abs(vx) + DIV_EPS) / e_cfg.dt_phys, 0.0, 1.0)
        y_cross = y - vy * e_cfg.dt_phys * frac_dt
        z_cross = z - vz * e_cfg.dt_phys * frac_dt
        in_goal = (jnp.abs(y_cross) <= self.goal_w / 2) & (z_cross <= self.goal_h)
        goal_plus = in_goal & (x >= line_x)
        goal_minus = in_goal & (x <= -line_x)
        scored_field = jnp.where(goal_plus, team_plus, jnp.where(goal_minus, team_minus, -1))
        throw_active = state.throw_taker >= 0
        scored_field = jnp.where(throw_active, jnp.int32(-1), scored_field)  # 스로인 직접골 무효
        sp_take_active = state.setpiece_taker >= 0
        own_goal_direct = sp_take_active & (scored_field == (1 - state.restart_team).astype(jnp.int32))
        scored_field = jnp.where(own_goal_direct, jnp.int32(-1), scored_field)  # 세트피스 직접 자책 무효
        # 간접 프리킥(백패스 IDFK 등): 두 번째 터치(setpiece_taker 클리어) 전 직접 골은 무효(IFAB Law 13).
        # 무효 시 아래 out_end 라우팅이 골킥(상대 골)·코너(자기 골)로 자동 처리.
        indirect_direct = sp_take_active & state.restart_indirect & (scored_field >= 0)
        scored_field = jnp.where(indirect_direct, jnp.int32(-1), scored_field)
        is_goal_field = scored_field >= 0

        out_end = (jnp.abs(x) > line_x) & (~is_goal_field)
        out_side = (jnp.abs(y) > hy + self.r_ball) & (~is_goal_field) & (~out_end)
        out = out_end | out_side
        opp = (1 - state.last_touch_team).astype(jnp.int32)
        def_side = jnp.where(x > 0, team_minus, team_plus).astype(jnp.int32)
        corner = out_end & (state.last_touch_team == def_side)
        goalkick = out_end & (~corner)

        is_goal = is_goal_field
        scored = scored_field.astype(jnp.int32)
        # suppress_restart(reconstruct용 pin): sim 공이 드리프트로 라인을 넘어도 재개(골/아웃→세트피스·
        # 킥오프=전원 포메이션 스냅)를 유발하지 않게 이벤트 플래그를 봉쇄한다. 실경기 연속 창엔 재개가
        # 없으므로(관측 근거) sim 아웃은 순수 드리프트 아티팩트 — 위치 pin 아님(공은 그대로 굴러가고
        # 선수는 free-running 유지, 공 ADE엔 손실이 그대로 반영). suppress_charge/body/retake와 동류.
        if suppress_restart is not None:
            keep = ~suppress_restart
            is_goal = is_goal & keep; is_goal_field = is_goal_field & keep
            out = out & keep; out_side = out_side & keep; out_end = out_end & keep
            corner = corner & keep; goalkick = goalkick & keep
            scored_field = jnp.where(keep, scored_field, jnp.int32(-1))
            scored = jnp.where(keep, scored, jnp.int32(-1))
        event = is_goal_field | out

        center = jnp.array([0.0, 0.0, self.r_ball])
        spot_inset = e_cfg.restart_field_inset
        side_spot = jnp.array([
            jnp.clip(x, -hx + spot_inset, hx - spot_inset),
            jnp.sign(y) * (hy - e_cfg.throwin_line_inset),
            self.r_ball,
        ])
        corner_spot = jnp.array([
            jnp.sign(x) * (hx - spot_inset),
            jnp.sign(y + DIV_EPS) * (hy - spot_inset),
            self.r_ball,
        ])
        goalkick_spot = jnp.array([
            jnp.sign(x) * (hx - self.s_cfg.goal_area_length),
            0.0,
            self.r_ball,
        ])
        ball_pos = jnp.where(is_goal, center,
                    jnp.where(corner, corner_spot,
                     jnp.where(goalkick, goalkick_spot,
                      jnp.where(out_side, side_spot, state.ball_pos))))
        ball_vel = jnp.where(event, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(event, jnp.zeros(DIM_ALL), state.ball_spin)

        concede = jnp.where(is_goal, (1 - scored).astype(jnp.int32), jnp.int32(-1))
        poss = jnp.where(is_goal, concede, jnp.where(out, opp, state.poss_team)).astype(jnp.int32)
        last_touch = jnp.where(is_goal, concede, jnp.where(out, opp, state.last_touch_team)).astype(jnp.int32)
        score = state.score + jnp.where(
            is_goal,
            jnp.where(jnp.arange(TEAM_COUNT) == scored, 1, 0),
            jnp.zeros(TEAM_COUNT, jnp.int32),
        )
        restart_kind = jnp.where(is_goal, RK_KICKOFF,
                        jnp.where(corner, RK_CORNER,
                         jnp.where(goalkick, RK_GOALKICK,
                          jnp.where(out_side, RK_THROWIN, state.restart_kind)))).astype(jnp.int32)
        restart_team_out = jnp.where(out, opp, state.restart_team)
        restart_team = jnp.where(is_goal, concede, restart_team_out).astype(jnp.int32)

        # 비이벤트 재개 프레임: 키커 도착 후에만 restart_t 감소(도착 전 정지 — 페널티 포함, 비재개는 항상 감소).
        if arrived is None:
            countdown = jnp.maximum(0, state.restart_t - 1)
        else:
            can_count = (~sp_active) | arrived
            countdown = jnp.where(can_count, jnp.maximum(0, state.restart_t - 1), state.restart_t)
        restart_t = jnp.where(event, jnp.int32(e_cfg.restart_substeps), countdown).astype(jnp.int32)
        ball_state = jnp.where(event, BALL_DEAD, jnp.where(restart_t > 0, state.ball_state, BALL_ALIVE)).astype(jnp.int32)
        # 카운트다운 소진 페일세이프(킥 미실행): restart_kind·pending_taker도 리셋(obs is_taker 오염 방지).
        revived = (~event) & (state.restart_t > 0) & (restart_t == 0)
        restart_kind = jnp.where(revived, jnp.int32(RK_NONE), restart_kind).astype(jnp.int32)
        pending_taker_rv = jnp.where(revived, jnp.int32(-1), state.pending_taker).astype(jnp.int32)

        goalkick_any = goalkick                              # 골킥만 GK가 키커(페널티 세이브 분기 제거됨)
        ko_pos = self._kickoff_positions(state)
        taker = self._designate_taker(state, ball_pos[:DIM_Z], restart_team, goalkick_any)
        taker = jnp.where(is_goal, self._kickoff_taker(state, restart_team, ko_pos), taker)
        pending_taker = jnp.where(event, taker, pending_taker_rv).astype(jnp.int32)
        player_pos = jnp.where(is_goal, ko_pos, state.player_pos)
        player_vel = jnp.where(is_goal, jnp.zeros_like(state.player_vel), state.player_vel)
        off_flag = jnp.where(event, jnp.zeros_like(state.offside_flag), state.offside_flag)
        pass_t = jnp.where(event, jnp.int32(0), state.pass_t)
        throw_taker = jnp.where(event, jnp.int32(-1), state.throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(event, jnp.int32(-1), state.setpiece_taker).astype(jnp.int32)
        foul_kind = jnp.where(is_goal, jnp.int32(FOUL_NONE), state.foul_kind)

        # ── 페널티 침범 재실행(결과 의존, IFAB) ──────────────────────────────
        # in-flight 페널티가 터미널(골/아웃/세이브/정지)에 도달하면 침범 팀·결과로 재실행 판정.
        #   공격팀 침범 & 골 → 골 취소·재실행 · 수비팀 침범 & 무득점 → 재실행 · 그 외 → 결과 인정.
        # 침범자 마스크(양 팀)는 킥 순간 contest가 기록해 뒀고 여기서 team_id로 갈라 쓴다.
        in_flight = state.penalty_flight_team >= 0
        attacking = jnp.clip(state.penalty_flight_team, 0, 1).astype(jnp.int32)
        defending = (1 - attacking).astype(jnp.int32)
        enc_mask = state.penalty_encroach_mask
        att_enc = jnp.any(enc_mask & (state.team_id == attacking))
        def_enc = jnp.any(enc_mask & (state.team_id == defending))

        scored_pen = in_flight & is_goal & (scored == attacking)
        ball_speed_xy = jnp.linalg.norm(state.ball_vel[:DIM_Z])
        grounded = z <= e_cfg.z_ground
        adir_att = jnp.where(attacking == 0, state.attack_dir[0], state.attack_dir[self.n_agents])
        # '세이브' 확정은 수비 터치 + 공이 골에서 멀어지는 중(vx·adir_att<0)일 때만 — 불충분한 파리로
        # 공이 여전히 골로 향하면(다음 서브스텝 득점 가능) 조기 no-goal 종결하지 않는다(B4).
        saved = (state.last_touch_team == defending) & (state.ball_vel[DIM_X] * adir_att < 0)
        # 정지 페일세이프는 3D로 판정 — 지면 접촉 순간 튀는(vz 큰) 공을 수평속도만 보고 '정지'로
        # 오분류해 아직 살아있는 페널티를 조기 종료시키지 않도록 vz도 함께 게이트한다.
        settled = ((state.ball_state == BALL_ALIVE) & grounded
                   & (ball_speed_xy < e_cfg.penalty_settle_speed)
                   & (jnp.abs(state.ball_vel[DIM_Z]) < e_cfg.penalty_settle_speed))
        # GK가 페널티를 잡아 홀드로 죽인 종결(caught save) — contest가 이미 RK_GK_HOLD·BALL_DEAD·vel=0으로
        # 만들어 out/saved/settled 어디에도 안 걸린다. 이 케이스를 해소에 포함하지 않으면 penalty_flight_team이
        # 스테일로 남아 후속 오픈플레이 터미널에서 유령 페널티(재실행·득점무효·오심카드)를 오발동한다.
        gk_caught_pen = in_flight & (state.ball_state == BALL_DEAD) & (state.restart_kind == RK_GK_HOLD)
        missed_pen = in_flight & (~is_goal) & (out | saved | settled | gk_caught_pen)
        # 해소는 '어느 방향이든 골'을 포함 — scored==defending(공격측 역방향 자책) 분기를 빼면
        # 플라이트가 골·킥오프를 관통해 스테일 생존, 이후 오픈플레이 settled/out에서 유령 재실행됨.
        resolved_pen = (in_flight & is_goal) | missed_pen
        # suppress_retake pin: 침범-재실행 판정도 심판 재량(관측 근거 — 실경기에서 결과가
        # 인정됨)으로 억제 가능. 재개 소비 시 retake(contest)와 같은 채널의 events 쪽 적용점.
        sup_rt = jnp.bool_(False) if suppress_retake is None else jnp.bool_(suppress_retake)  # python True 방어
        retake = ((scored_pen & att_enc) | (missed_pen & def_enc)) & (~sup_rt)

        pen_spot = jnp.array([adir_att * (hx - e_cfg.penalty_spot), 0.0, self.r_ball])
        retake_taker = self._designate_taker(state, pen_spot[:DIM_Z], attacking, jnp.bool_(False))

        ball_pos = jnp.where(retake, pen_spot, ball_pos)
        ball_vel = jnp.where(retake, jnp.zeros(DIM_ALL), ball_vel)
        ball_spin = jnp.where(retake, jnp.zeros(DIM_ALL), ball_spin)
        ball_state = jnp.where(retake, jnp.int32(BALL_DEAD), ball_state).astype(jnp.int32)
        restart_kind = jnp.where(retake, jnp.int32(RK_PENALTY), restart_kind).astype(jnp.int32)
        restart_team = jnp.where(retake, attacking, restart_team).astype(jnp.int32)
        restart_t = jnp.where(retake, jnp.int32(e_cfg.penalty_substeps), restart_t).astype(jnp.int32)
        poss = jnp.where(retake, attacking, poss).astype(jnp.int32)
        last_touch = jnp.where(retake, attacking, last_touch).astype(jnp.int32)
        pending_taker = jnp.where(retake, retake_taker, pending_taker).astype(jnp.int32)
        score = jnp.where(retake, state.score, score)               # 재실행 시 득점 취소
        scored = jnp.where(retake, jnp.int32(-1), scored).astype(jnp.int32)
        player_pos = jnp.where(retake, state.player_pos, player_pos)  # 킥오프 재배치 되돌림
        player_vel = jnp.where(retake, state.player_vel, player_vel)
        # facing은 속도 파생값이므로 속도를 건드린 뒤 **무조건** 다시 만든다. 득점 시 위치·속도만
        # 리셋하고 facing을 두면 파생 계약이 깨진다 — 기본값 kickoff_instant=True에서는 아래
        # ko_snap의 pre-snap이 우연히 고쳐 주지만, False로 두면 그대로 남는다(실측 오차 1.571 rad).
        # 비이벤트 프레임에서는 속도가 그대로라 이 재계산이 항등이므로 포워드 불변이다.
        player_facing = self.facing_from_velocity(player_vel, state.attack_dir)
        throw_taker = jnp.where(retake, jnp.int32(-1), throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(retake, jnp.int32(-1), setpiece_taker).astype(jnp.int32)
        off_flag = jnp.where(retake, jnp.zeros_like(off_flag), off_flag)

        # 해소 시 플라이트 플래그 클리어(재실행이면 다시 킥될 때 contest가 재기록).
        penalty_flight_team = jnp.where(resolved_pen, jnp.int32(-1), state.penalty_flight_team).astype(jnp.int32)
        penalty_encroach_mask = jnp.where(resolved_pen, jnp.zeros_like(enc_mask), enc_mask)
        # 새 재개(골/아웃/페널티재실행/페일세이프 복귀)는 전부 직접 — 간접 플래그 클리어(스테일 방지).
        # 그 외(진행 중 IDFK)는 carry해 두 번째 터치 전까지 유지.
        restart_indirect = jnp.where(event | retake | revived, jnp.bool_(False), state.restart_indirect)

        new_state = state._replace(
            ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
            player_pos=player_pos, player_vel=player_vel, player_facing=player_facing,
            poss_team=poss, last_touch_team=last_touch,
            score=score, restart_team=restart_team, restart_t=restart_t, restart_kind=restart_kind,
            ball_state=ball_state, pending_taker=pending_taker, offside_flag=off_flag, pass_t=pass_t,
            foul_kind=foul_kind, throw_taker=throw_taker, setpiece_taker=setpiece_taker,
            penalty_flight_team=penalty_flight_team, penalty_encroach_mask=penalty_encroach_mask,
            restart_indirect=restart_indirect)

        # 재실행 유발 팀 침범자에게 확률 카드(공격팀 골취소 또는 수비팀 세이브·아웃 유발자).
        card_att = retake & scored_pen & att_enc
        card_def = retake & missed_pen & def_enc
        card_mask = jnp.where(card_att, enc_mask & (state.team_id == attacking),
                              jnp.where(card_def, enc_mask & (state.team_id == defending),
                                        jnp.zeros_like(enc_mask)))
        new_state = self._draw_cards(new_state, card_mask, key)
        # 방금 지정한 재개 키커가 그 카드로 퇴장하는 경우(재실행 키커=스폿 최근접 공격수가 침범
        # 카드 대상과 겹침) 벤치 고정 → 도착 불가 → can_count 동결 → 강제킥·revive 모두 도달
        # 불가한 영구 데드락 — 활성 최근접으로 재지정한다(모든 pending_taker 비활성 케이스 공용 가드).
        pt = new_state.pending_taker
        taker_dead = ((new_state.restart_t > 0) & (pt >= 0)
                      & (~new_state.active_player[jnp.clip(pt, 0, self.N - 1)]))
        re_taker = self._designate_taker(new_state, new_state.ball_pos[:DIM_Z],
                                         new_state.restart_team, jnp.bool_(False))
        new_state = new_state._replace(
            pending_taker=jnp.where(taker_dead, re_taker, pt).astype(jnp.int32))

        # ★즉시 킥오프 키커 pre-snap(BC 라벨 드롭 방지): 득점 후 킥오프는 키커가 포메이션에서 시작해
        # 다음 스텝 진입 시 arrived=False → action_agency가 kick_forced=False로 산출하는데, 그 스텝의
        # 서브스텝에서 _apply_kicker_move가 스냅→즉시 강제킥이 발사(kick_applied=True)되어 진입 마스크가
        # 킥 dim을 닫은 채 라벨이 드롭된다. reset_array의 오프닝 킥오프와 동일하게 여기서 미리 스냅해
        # 다음 스텝 진입 arrived=True로 만들어 kick_forced가 킥 dim을 연다. instant_ko(kickoff_instant)
        # ON일 때만·활성 킥오프에만 적용(코너/골킥/스로 taker·비-instant는 무변경).
        ko_snap = ((new_state.restart_kind == RK_KICKOFF) & (new_state.restart_t > 0)
                   & jnp.bool_(bool(self.e_cfg.kickoff_instant)))
        snapped = self._apply_kicker_move(new_state)
        new_state = new_state._replace(
            player_pos=jnp.where(ko_snap, snapped.player_pos, new_state.player_pos),
            player_vel=jnp.where(ko_snap, snapped.player_vel, new_state.player_vel),
            player_facing=jnp.where(ko_snap, snapped.player_facing, new_state.player_facing))
        return new_state, scored

    def _halftime_switch(self, state, do):
        """후반 전환(do=True일 때만 적용) — 진영·공격방향 반전, 중앙 킥오프 재배치, 후반 킥오프팀 설정.
        스태미나는 리셋하지 않는다(실측상 하프타임 회복 없음)."""
        N = self.N
        pos_2h = -self._kickoff_positions(state)          # 진영 교대 = 점대칭(x·y 반전, 회전 규약)
        adir_2h = -state.attack_dir
        # facing은 속도 파생값 — 정지 재배치라 새 공격 방향이 그대로 정지 규약 facing이 된다.
        facing_2h = self.facing_from_velocity(jnp.zeros((N, DIM_Z)), adir_2h)
        second_kick = (1 - state.kickoff_team).astype(jnp.int32)
        center = jnp.array([0.0, 0.0, self.r_ball])
        state_2h = state._replace(player_pos=pos_2h, attack_dir=adir_2h)
        taker = self._kickoff_taker(state_2h, second_kick, pos_2h)
        switched = state._replace(
            player_pos=pos_2h, player_vel=jnp.zeros((N, 2)), player_facing=facing_2h,
            attack_dir=adir_2h, ball_pos=center, ball_vel=jnp.zeros(DIM_ALL), ball_spin=jnp.zeros(DIM_ALL),
            poss_team=second_kick, ball_state=jnp.int32(BALL_DEAD),
            restart_team=second_kick, restart_t=jnp.int32(self.e_cfg.restart_substeps),
            restart_kind=jnp.int32(RK_KICKOFF), pending_taker=taker, last_touch_team=second_kick,
            ctrl_lock_t=jnp.zeros(N, dtype=jnp.int32), cooldown=jnp.zeros(N), offside_flag=jnp.zeros(N, bool),
            pass_team=jnp.int32(-1), pass_t=jnp.int32(-1),
            throw_taker=jnp.int32(-1), setpiece_taker=jnp.int32(-1),
            penalty_flight_team=jnp.int32(-1), penalty_encroach_mask=jnp.zeros(N, bool),
            restart_indirect=jnp.bool_(False), last_touch_code=jnp.int32(TOUCH_NONE),
            foul_kind=jnp.int32(FOUL_NONE), foul_actor=jnp.int32(-1), foul_victim=jnp.int32(-1),
        )
        # ★2H 킥오프 키커 pre-snap(BC 라벨 드롭 방지 — _events 골 분기와 동일 사유): instant_ko는
        # 스냅 즉시 발사라, 스냅이 스텝 진입 후(서브스텝)에 일어나면 그 스텝 라벨이 드롭된다. 전환 상태에서
        # 미리 스냅해 다음 스텝 진입 arrived=True로 만든다. instant_ko ON일 때만.
        if bool(self.e_cfg.kickoff_instant):
            switched = self._apply_kicker_move(switched)
        return jax.tree_util.tree_map(lambda a, b: jnp.where(do, a, b), switched, state)
