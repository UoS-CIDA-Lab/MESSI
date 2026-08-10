import jax.numpy as jnp

from constants import *
from spatial import _safe_norm, _unit

class Movement:
    def _in_own_box(self, pos, attack_dir, clamp_x=False):
        """자기 진영 페널티 박스 내부 판정(bool) — own_goal = -attack_dir*hx 기준.
        pos[...,DIM_X/DIM_Y] 사용(공(3,)·선수(N,2)·단일(2,) 모두 브로드캐스트). clamp_x=True면
        |x|<=hx(피치 안)도 요구 — GK reach·ball_in_box처럼 골라인 밖 배제가 필요할 때.
        movement/contest/fouls에 복붙돼 있던 박스 판정의 단일 진실원천(런타임은 XLA CSE로 동일)."""
        x = pos[..., DIM_X]
        y = pos[..., DIM_Y]
        own_goal_x = -attack_dir * self.hx
        inside = (jnp.abs(own_goal_x - x) <= self.pen_len) & (jnp.abs(y) <= self.pen_hw)
        if clamp_x:
            inside = inside & (jnp.abs(x) <= self.hx)
        return inside

    def facing_from_velocity(self, player_vel, attack_dir):
        """facing = **현재 속도 방향**(단일 진실원천). 독립 적분 상태가 아니라 파생값이다.

        `state.player_facing`은 이 함수의 캐시일 뿐이며, 속도가 바뀌는 지점(`_move`)에서 매번
        다시 계산된다. 따라서 facing은 **관측 가능한 양(속도·attack_dir)만으로 완전히 결정**되고,
        obs가 facing을 직접 담지 않아도 정책이 잃는 정보가 없다(self `abs_vel`·others `abs_vel`).

        정지(|v| ≤ eps) 시 방향은 정의되지 않으므로 **자기 공격 방향**으로 둔다. 은닉 래치(직전
        방향 유지)를 두면 한 프레임 관측만으로는 복원할 수 없는 상태가 되살아나므로 쓰지 않는다.
        arctan2(±0, ±0)의 IEEE −0 우연에 기대지 않도록 정지 분기를 명시적으로 가른다.

        회전 속도 상한은 별도 계수가 아니라 **마찰 타원의 횡가속 캡(`accel_norm_max`)**이 만든다 —
        속도 방향이 물리적으로 꺾일 수 있는 만큼만 facing도 꺾인다.
        """
        speed = _safe_norm(player_vel, axis=1)
        ang = jnp.arctan2(player_vel[:, DIM_Y], player_vel[:, DIM_X])
        rest = jnp.where(attack_dir >= 0.0, 0.0, HALF_TURN)
        return jnp.where(speed > STATIONARY_SPEED_EPS, ang, rest)

    def _in_reach(self, state):
        """
        선수 reach 여부 계산: 수평거리 ≤ Rxy & 공높이 ≤ reach_z + r_ball.
        **[변경] 속도-의존 도달 확장(reach_xy_slide 커밋 보간) 폐지** — 필드 선수는
        ``Engine.reach_xy``, GK는 자기 박스 안에서 ``Engine.gk_reach_xy``를 쓴다.
        접근속도로 reach가 늘어나던 메커니즘 제거.
        """
        e_cfg = self.e_cfg
        rel_xy = state.ball_pos[:DIM_Z][None, :] - state.player_pos
        d_xy = jnp.linalg.norm(rel_xy, axis=-1)

        # 골키퍼는 자기 페널티박스 안에서 확장 reach(gk_reach_xy). 그 외는 고정 필드 reach_xy.
        gk_in_box = (state.gk_indices == 1) & self._in_own_box(
            state.player_pos, state.attack_dir, clamp_x=True)
        Rxy = jnp.where(gk_in_box, e_cfg.gk_reach_xy, e_cfg.reach_xy) + self.r_ball
        reachable = (d_xy <= Rxy) & (state.ball_pos[DIM_Z] <= state.reach_z + self.r_ball)
        return reachable, d_xy

    def _gk_reactive_claim(self, state):
        """오픈플레이에서 자기 박스 안 GK가 도달 가능한 라이브 공에 대해 want_f2b 없이도 경합 후보가
        되게 하는 마스크(N,) — 키퍼 본능. contest 후보 마스크에만 더해지고 이동 facing엔 섞이지 않는다
        (do_kick과 분리). 실제 캐치/parry/백패스 분기는 contest._apply_force2ball이 승자 기준으로 판정.

        ★합법 캐치일 때만 발동: 잡는 행위 자체가 반칙인 공 — ①자신이 재터치 금지 추적자(자기
        세트피스/홀드 배급의 2차 터치 = IDFK) ②같은팀 발패스(백패스 = IDFK) — 은 본능 캐치에서
        제외한다. 실제 GK는 그런 공에 손을 대지 않는다(발 플레이는 want_f2b 경로로 여전히 가능).
        이 게이트가 없으면 '홀드 배급 → 공이 gk_reach_xy(2.0 m)를 못 벗어남 → cooldown 만료 즉시 자동
        재캐치 = 재터치 IDFK → 상대 슛 → 캐치 홀드 → 배급 → …' 무한 루프가 된다."""
        reachable, _ = self._in_reach(state)
        gk_in_box = (state.gk_indices == 1) & self._in_own_box(
            state.player_pos, state.attack_dir, clamp_x=True)
        alive = state.ball_state == BALL_ALIVE
        restart_active = state.restart_t > 0
        ar = self.player_indices
        retouch_locked = (state.setpiece_taker == ar) | (state.throw_taker == ar)
        # 백패스 = 동료가 발로 의도적으로 플레이한 공(IFAB) — 패스뿐 아니라 드리블 터치도 포함.
        # 헤더/가슴 백패스 캐치는 합법(코드 미포함)이 맞다.
        backpass = ((state.last_touch_team == state.team_id)
                    & ((state.last_touch_code == TOUCH_PASS)
                       | (state.last_touch_code == TOUCH_DRIBBLE)))
        # cooldown 게이트: parry 직후(cooldown 세팅) 같은 컨트롤 스텝에 리바운드를 즉시 재캐치하지 못하게.
        return (gk_in_box & reachable & alive & (~restart_active)
                & (~retouch_locked) & (~backpass)
                & (state.cooldown <= 0) & state.active_player)
    
    def _kicker_target(self, state):
        """키커 목표 위치(a 옵션): 공에서 자기 골 방향으로 (r_player+r_ball) 뒤.
        차러 들어가는 자세. 공이 스폿에 있을 때 그 뒤에 서도록. pending_taker<0(오픈플레이)이면
        clip으로 인덱스 안전화 — 실제 적용은 호출부의 active 게이트가 막는다."""
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        adt = state.attack_dir[taker]                 # 키커 공격 방향
        back = jnp.array([-adt, 0.0])                 # 자기 골 방향(= 공격 반대)
        return state.ball_pos[:DIM_Z] + back * (self.r_player + self.r_ball)

    def _apply_kicker_move(self, state):
        """세트피스 활성(active) 중에만 키커를 목표 위치로 강제 이동(최대속도, 스태미나 무소모).
        도착 전까지만 당기고 오픈플레이(pending_taker<0)에선 아무도 움직이지 않는다 — active 게이트가
        없으면 pending_taker=-1이 클립되어 player[0]을 매 서브스텝 공으로 끌어 궤적을 오염시킨다."""
        e_cfg = self.e_cfg
        _, _, _, active = self._setpiece_kick_lock(state)
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        tgt = self._kicker_target(state)
        cur = state.player_pos[taker]
        to_tgt = tgt - cur
        d = jnp.linalg.norm(to_tgt) + DIV_EPS
        # 킥오프 즉시 발동: 키커를 스폿으로 한 번에 스냅(걸어오는 대기 제거) → 첫 서브스텝에 도착·강제 킥.
        instant_ko = jnp.bool_(e_cfg.kickoff_instant) & (state.restart_kind == RK_KICKOFF)
        walk = jnp.minimum(d, e_cfg.kicker_speed * e_cfg.dt_phys)
        step_len = jnp.where(instant_ko, d, walk)
        new_tk_pos = cur + to_tgt / d * step_len
        P = jnp.where(active, state.player_pos.at[taker].set(new_tk_pos), state.player_pos)
        Vp = jnp.where(
            active,
            state.player_vel.at[taker].set(jnp.zeros(DIM_Z)),
            state.player_vel,
        )
        # facing은 속도 파생값이라 여기서 따로 지정하지 않는다(V=0인 pin된 키커 → 정지 규약).
        Fc = self.facing_from_velocity(Vp, state.attack_dir)
        return state._replace(player_pos=P, player_vel=Vp, player_facing=Fc)

    def _deadball_target(self, state):
        """데드볼 내장 엔진 v1 '두뇌' — 재개 종류·공수 역할별 각 선수 목표 위치(N,2, 월드).
        배선(라우팅·마스킹)과 분리된 교체 가능 함수. 정교화(벽·마킹·오프더볼 런)는 여기만 갈아끼운다.

        v2 규칙(공수 역할·능동 배치):
          - 백본 = 킥오프 포메이션 슬롯(_kickoff_positions, 공격방향 점대칭).
          - 기본(킥오프·프리킥·골킥·GK홀드·오프사이드): **재개(공격)팀은 재개 전진도(공 위치)에 비례해
            공격적으로 침투** — 재개가 상대 진영에 가까울수록 더 깊이 밀어붙이고 공 레인으로 강하게 수렴.
            수비팀은 자기 골 쪽으로 컴팩트 블록. GK는 자기 골라인 앞에서 공 y 약간 추종.
          - 코너: 공격팀은 상대 박스 침투, 수비팀은 자기 박스 마킹.
          - 클리어런스: 수비(비재개팀) 목표가 공에서 clear_r(킥오프=센터서클/스로인=throwin_clear/그외
            =clear_dist) 안이면 밖으로 밀어 반칙(침범) 방지.
        TODO(정교화): 프리킥 수비벽 선발·오프사이드 라인 존중·페널티 아크·마킹 매칭은 반복 개선 단계에서.
        """
        e_cfg, s_cfg, d_cfg = self.e_cfg, self.s_cfg, self.d_cfg
        hx, hy = self.hx, self.hy
        ax = state.attack_dir                              # (N,) 공격 x부호
        ball = state.ball_pos[:DIM_Z]                      # (2,)
        ball_y = ball[DIM_Y]
        home = self._kickoff_positions(state)              # (N,2) 포메이션 백본
        is_att = state.team_id == state.restart_team
        is_gk = state.gk_indices == 1
        kind = state.restart_kind

        # ── 능동 역할 바이어스 ──
        # 재개 전진도 adv = 공의 공격프레임 x위치(-1 자기골 ~ +1 상대골). 공격팀은 adv가 클수록(상대 진영)
        # 더 공격적으로 침투(base 16m → 최대 28m). 공 쪽으로도 강하게 수렴해 지원각·침투를 만든다.
        adv = jnp.clip(ax * ball[DIM_X] / hx, -1.0, 1.0)                 # (N,)
        att_push = (
            d_cfg.attack_push_base
            + d_cfg.attack_push_gain * jnp.clip(adv, 0.0, 1.0)
        )
        att_x = jnp.clip(
            home[:, DIM_X] + ax * att_push,
            -hx + d_cfg.attack_own_inset,
            hx - d_cfg.attack_opp_inset,
        )
        att_y = (
            d_cfg.attack_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.attack_home_y_weight) * ball_y
        )
        def_x = jnp.clip(
            home[:, DIM_X] - ax * d_cfg.defend_retreat,
            -hx + d_cfg.defend_field_inset,
            hx - d_cfg.defend_field_inset,
        )
        def_y = (
            d_cfg.defend_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.defend_home_y_weight) * ball_y
        )
        tgt = jnp.stack([jnp.where(is_att, att_x, def_x),
                         jnp.where(is_att, att_y, def_y)], axis=1)

        # ── 스로인: 능동 배치 — 공격(스로잉)팀은 상대 골 방향으로 전진 옵션 제공(자기 진영 후퇴 금지)·
        #    볼사이드로 이동, 수비팀은 골사이드에서 볼사이드 압박(깊이 후퇴 최소). 기본 케이스의 수비 10m
        #    후퇴 + 미약한 전진이 '자기 진영으로 되돌아가는' 원인이라 스로인 전용으로 분리. ──
        th_att_push = (
            d_cfg.throw_attack_push_base
            + d_cfg.throw_attack_push_gain * jnp.clip(adv, 0.0, 1.0)
        )
        th_att_x = jnp.clip(
            home[:, DIM_X] + ax * th_att_push,
            -hx + d_cfg.throw_attack_own_inset,
            hx - d_cfg.throw_attack_opp_inset,
        )
        th_att_y = jnp.clip(
            d_cfg.throw_attack_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.throw_attack_home_y_weight) * ball_y,
            -hy + d_cfg.throw_attack_y_inset,
            hy - d_cfg.throw_attack_y_inset,
        )
        th_def_x = jnp.clip(
            home[:, DIM_X] - ax * d_cfg.throw_defend_retreat,
            -hx + d_cfg.throw_defend_field_inset,
            hx - d_cfg.throw_defend_field_inset,
        )
        th_def_y = jnp.clip(
            d_cfg.throw_defend_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.throw_defend_home_y_weight) * ball_y,
            -hy + d_cfg.throw_defend_field_inset,
            hy - d_cfg.throw_defend_field_inset,
        )
        throw_tgt = jnp.stack([jnp.where(is_att, th_att_x, th_def_x),
                               jnp.where(is_att, th_att_y, th_def_y)], axis=1)
        tgt = jnp.where(kind == RK_THROWIN, throw_tgt, tgt)

        # ── 프리킥(RK_FREEKICK): 공격팀 능동 침투 — 미드·전방은 상대 박스로 강하게 올라가 상대와 섞이고,
        #    자기 수비라인(CB)만 역습 대비 커버로 잔류. 수비팀은 기본 블록(마킹)+아래 clear_dist 이격. ──
        home_adv = jnp.clip(ax * home[:, DIM_X] / hx, -1.0, 1.0)         # 홈 역할 깊이(-1 자기골~+1 상대골)
        keep_back = home_adv < d_cfg.free_kick_keep_back_line
        # 전진 프리킥(adv↑)일수록 '상대 박스 절대위치'로 수렴 → 미드까지 박스에 들어가 상대와 섞인다.
        # (홈+고정푸시만으론 깊은 홈의 미드가 박스에 못 미친다.) 수비 진영 프리킥(adv≤0)은 홈+전진(빌드업).
        fk_push = (
            d_cfg.free_kick_push_base
            + d_cfg.free_kick_push_gain
            * jnp.clip(adv, d_cfg.free_kick_progress_floor, 1.0)
        )
        push_x = jnp.clip(
            home[:, DIM_X] + ax * fk_push,
            -hx + d_cfg.free_kick_own_inset,
            hx - d_cfg.free_kick_opp_inset,
        )
        box_abs_x = ax * (hx - self.pen_len * d_cfg.box_depth_fraction)
        commit = jnp.clip(
            (adv - d_cfg.free_kick_commit_start) / d_cfg.free_kick_commit_span,
            0.0,
            1.0,
        )
        fk_fwd_x = (1.0 - commit) * push_x + commit * box_abs_x         # 전진 FK일수록 박스로 침투(상대와 섞임)
        fk_fwd_y = jnp.clip(
            d_cfg.free_kick_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.free_kick_home_y_weight) * ball_y,
            -self.pen_hw - d_cfg.free_kick_box_y_padding,
            self.pen_hw + d_cfg.free_kick_box_y_padding,
        )
        fk_back_x = jnp.clip(
            home[:, DIM_X] + ax * d_cfg.free_kick_back_push,
            -hx + d_cfg.defend_field_inset,
            hx - d_cfg.defend_field_inset,
        )
        fk_back_y = (
            d_cfg.free_kick_back_home_y_weight * home[:, DIM_Y]
            + (1.0 - d_cfg.free_kick_back_home_y_weight) * ball_y
        )
        fk_att_x = jnp.where(keep_back, fk_back_x, fk_fwd_x)
        fk_att_y = jnp.where(keep_back, fk_back_y, fk_fwd_y)
        fk_tgt = jnp.stack([jnp.where(is_att, fk_att_x, def_x),
                            jnp.where(is_att, fk_att_y, def_y)], axis=1)
        tgt = jnp.where(kind == RK_FREEKICK, fk_tgt, tgt)

        # ── 코너: 박스 침투/마킹 ──
        box_x = jnp.where(
            is_att,
            ax * (hx - self.pen_len * d_cfg.box_depth_fraction),
            -ax * (hx - self.pen_len * d_cfg.box_depth_fraction),
        )
        box_y = jnp.clip(
            home[:, DIM_Y] * d_cfg.corner_home_y_weight,
            -self.pen_hw,
            self.pen_hw,
        )
        corner_tgt = jnp.stack([box_x, box_y], axis=1)
        tgt = jnp.where(kind == RK_CORNER, corner_tgt, tgt)

        # ── 페널티: 키커·수비GK 외 전원을 스폿에서 clear_dist 밖(아크 뒤·박스 바깥) 라인에 정렬 →
        #    박스/아크 침범 0. 페널티 침범은 반경이 아닌 박스/아크 멤버십이라 아래 clear_r 푸시로는 못 막으므로
        #    전용 배치가 필요하다. GK는 바로 아래 골라인 타깃으로 덮어써 band 안에 세운다.
        taker_i = jnp.clip(state.pending_taker, 0, self.N - 1)
        adir_att = state.attack_dir[taker_i]                            # 재개(공격)팀 공격 방향
        pen_line_x = adir_att * (
            hx - e_cfg.penalty_spot - e_cfg.clear_dist - d_cfg.penalty_line_padding
        )
        pen_tgt = jnp.stack([jnp.broadcast_to(pen_line_x, ax.shape),
                             jnp.clip(
                                 home[:, DIM_Y],
                                 -hy + d_cfg.defend_field_inset,
                                 hy - d_cfg.defend_field_inset,
                             )], axis=1)
        tgt = jnp.where(kind == RK_PENALTY, pen_tgt, tgt)

        # ── GK: 자기 골라인 앞 0.5m(페널티 goal-line band 안 → 침범 미판정), 공 y 약간 추종 ──
        gk_y = jnp.broadcast_to(
            jnp.clip(d_cfg.gk_ball_y_weight * ball_y, -self.goal_w, self.goal_w),
            ax.shape,
        )
        gk_tgt = jnp.stack(
            [-ax * hx + ax * d_cfg.gk_line_offset, gk_y],
            axis=1,
        )
        tgt = jnp.where(is_gk[:, None], gk_tgt, tgt)

        # ── 클리어런스(수비만): 공에서 clear_r 밖으로 ──
        clear_r = jnp.where(kind == RK_KICKOFF, s_cfg.center_circle_radius,
                  jnp.where(kind == RK_THROWIN, e_cfg.throwin_clear, e_cfg.clear_dist))
        rel = tgt - ball[None, :]
        d = jnp.linalg.norm(rel, axis=1, keepdims=True)
        pushed = ball[None, :] + rel / (d + DIV_EPS) * clear_r
        need = ((~is_att) & (d[:, 0] < clear_r))[:, None]
        tgt = jnp.where(need, pushed, tgt)

        # 필드 경계 클립
        tgt = tgt.at[:, DIM_X].set(
            jnp.clip(tgt[:, DIM_X], -hx + d_cfg.field_inset, hx - d_cfg.field_inset)
        )
        tgt = tgt.at[:, DIM_Y].set(
            jnp.clip(tgt[:, DIM_Y], -hy + d_cfg.field_inset, hy - d_cfg.field_inset)
        )
        return tgt

    def _deadball_move(self, state):
        """데드볼 엔진 이동 명령(mv_dir[N,2] 월드, mv_pow[N]) — 각 선수를 _deadball_target로.
        목표 2m 내에서 파워를 선형 감쇠해 오버슈트·진동을 줄인다. _move가 그대로 물리에 태운다."""
        tgt = self._deadball_target(state)
        to_tgt = tgt - state.player_pos
        d = jnp.linalg.norm(to_tgt, axis=1)
        mv_dir = to_tgt / (d[:, None] + DIV_EPS)
        mv_pow = jnp.clip(d / self.d_cfg.target_slowdown_radius, 0.0, 1.0)
        return mv_dir, mv_pow

    def _vel_substep(self, player_vel, v_cmd, vmax, eff_move):
        """이동 속도 갱신 커널 1서브스텝 — **속도-명령 모델**(guide.md §1). 목표속도 v_cmd를 향한
        속도 변화를 현재 진행방향 기준 **종(가·감속)·횡(선회)**으로 분해해 각각 상한으로 클립한다.
        → '물리적으로 가능한 속도 변화'를 환경 레벨에서 보장(정책은 '가고 싶은 속도'만 내고 도달
        가능성·관성은 여기서 처리). _move(포워드)·inverse.infer_move_action(역산)이 공유하는 단일 진실원천.

        캡(서브스텝당 Δv = cap·dt): 종가속 a_max / 종감속 brake_decel_max / 횡가속 accel_norm_max.
        이 3분할 클립이 구모델의 plant&cut 제동·drag를 대체한다(급선회는 횡캡이, 반전은 종감속캡이 제한).

        Args:
            player_vel: (N,2) 현재 속도 / v_cmd: (N,2) 목표 속도(월드) /
            vmax: (N,) 유효 최고속(스태미나 반영) / eff_move: bool[N] 적용 마스크
        Return: (N,2) 갱신 속도
        """
        e_cfg = self.e_cfg
        dt = e_cfg.dt_phys
        # 목표속도를 vmax 크기로 우선 제한(방향 보존)
        cmd_sp = _safe_norm(v_cmd, axis=1, keepdims=True)
        v_cmd = v_cmd * jnp.minimum(vmax[:, None] / (cmd_sp + DIV_EPS), 1.0)

        # 속도 변화(Δv)를 현재 진행방향(정지 시 목표방향) 기준 종·횡으로 분해
        speed = _safe_norm(player_vel, axis=1, keepdims=True)             # 현재 속력 (N,1)
        vhat = jnp.where(
            speed > STATIONARY_SPEED_EPS,
            player_vel / (speed + DIV_EPS),
            _unit(v_cmd),
        )
        dv = v_cmd - player_vel
        along = jnp.sum(dv * vhat, axis=1, keepdims=True)                 # 종성분(부호: +가속 −감속)
        perp = dv - along * vhat                                          # 횡성분(선회)

        # 마찰-타원 클립(friction ellipse / g-g diagram) — 종·횡 캡을 반축으로 하는 타원 안으로 제한.
        # 구 박스(성분별 독립 클립)는 코너(동시 최대 종·횡)가 √(cap_종²+cap_횡²)로 단일축 캡을 40~55%
        # 초과했다(가속턴 12.1·제동턴 13.5 > 캡 8.5~10.5). 타원이면 어느 방향으로도 그 방향 캡을 못 넘고
        # 합성가속 ≤ max(종캡,횡캡)이라 물리적(마찰이 종·횡에 공유). 균일 스케일이라 명령 방향은 보존.
        cap_along = jnp.where(along >= 0.0, e_cfg.a_max * dt, e_cfg.brake_decel_max * dt)  # 종 비대칭(가속/제동)
        perp_mag = _safe_norm(perp, axis=1, keepdims=True)                                 # (N,1)
        n_along = along / (cap_along + DIV_EPS)                                            # 정규화 종
        n_perp = perp_mag / (e_cfg.accel_norm_max * dt + DIV_EPS)                          # 정규화 횡
        over = jnp.sqrt(
            n_along ** 2 + n_perp ** 2 + SAFE_NORM_EPS
        )                                                                                 # 타원 반경(>1=밖)
        scale = 1.0 / jnp.maximum(over, 1.0)                                               # 경계로 방사 투영(방향 보존)
        # ↑ jnp.maximum(over,1)≥1이라 1/x 항상 안전(미사용분기 inf·sqrt(0) grad-NaN 원천 차단). over≤1→scale 1.
        dv_clip = (along * scale) * vhat + perp * scale

        V = player_vel + jnp.where(eff_move[:, None], dv_clip, 0.0)
        # 안전 vmax 캡(클립 후 잔여 초과분 방지)
        sp = _safe_norm(V, axis=1, keepdims=True) + DIV_EPS
        return V * jnp.minimum(vmax[:, None] / sp, 1.0)

    def _move(self, state, eff_move, mv_dir, mv_pow):
        """선수 이동 1서브스텝 적분(**속도-명령 모델**): 목표속도 v_cmd=mv_pow·vmax·mv_dir 형성 →
        `_vel_substep`이 마찰-타원 클립(종 가·감속·횡 반축)으로 도달가능 Δv만 실현 → 위치·분리 → facing·스태미나.

        구 가속-명령(plant&cut 제동·드래그 서보)은 폐기됐다 — 이동 라벨의 노이즈(가속 역산 지터) 때문에
        속도-명령으로 재설계(guide.md §1). 가속 상한은 진행방향 정렬 프레임의 **타원**(a_max/brake_decel_max/
        accel_norm_max 반축)으로 강제되어 어느 방향으로도 캘리브 캡을 넘지 않는다(합성가속 ≤ max 축캡).

        Args:
            state:    State
            eff_move: bool[N] — 이번 스텝 이동 명령을 실제 적용받는가(키커 강제이동 대상 등은 False)
            mv_dir:   float[N,2] — 이동 방향 단위벡터(월드 프레임)
            mv_pow:   float[N] — 이동 강도 [0,1]
        Return:
            State — player_pos / player_vel / player_facing / stamina 갱신
            (facing은 갱신된 속도의 파생값 — `facing_from_velocity`)
        """
        e_cfg = self.e_cfg
        dt = e_cfg.dt_phys
        # [속도-명령 모델] 이동 액션 = 목표속도 = 강도(mv_pow)·방향(mv_dir)·유효최고속(vmax).
        # 구모델의 a_vec = a_max·pow·dir(가속)를 대체 — 정책 부담↓(1적분), 클립은 _vel_substep이 처리.
        vmax = state.vmax * (e_cfg.vmax_floor + (1.0 - e_cfg.vmax_floor) * state.stamina)
        v_cmd = (mv_pow[:, None] * mv_dir) * vmax[:, None]
        V = self._vel_substep(state.player_vel, v_cmd, vmax, eff_move)

        # 위치 적분 + 필드 경계 클립
        P_free = state.player_pos + dt * V
        P = P_free.at[:, 0].set(jnp.clip(P_free[:, 0], -self.hx, self.hx))
        P = P.at[:, 1].set(jnp.clip(P[:, 1], -self.hy, self.hy))
        # 경계에서 잘린 축은 **속도 성분도 0으로** 만든다. 위치만 클립하면 위치는 멈췄는데 상태
        # 속도는 바깥으로 전속인 '유령 속도'가 남아, ①obs의 self/others `abs_vel`이 거짓을 말하고
        # ②facing(속도 파생)이 바깥을 향하며 ③스태미나가 계속 소모되고 ④반대 명령을 줘도 마찰
        # 타원 캡 때문에 실제 복귀까지 ~0.8 s가 걸린다. 클립은 그 축으로 **바깥으로** 밀 때만
        # 발생하므로 축 성분을 0으로 두는 것이 곧 바깥 법선 성분 제거다(안쪽 이동은 클립되지 않음).
        V = jnp.where(P != P_free, 0.0, V)

        # 퇴장 선수(active_player=False)는 터치라인 안쪽 벤치 라인(y=-hy+0.4)에 고정·정지
        # → 물리/규칙에서 유령화 방지(P 클립이 [-hy,hy]라 피치 밖 배치는 불가)
        bench_x = jnp.clip(
            -self.hx + e_cfg.bench_first_x_offset
            + self.player_indices * e_cfg.bench_spacing,
            -self.hx + e_cfg.bench_boundary_inset,
            self.hx - e_cfg.bench_boundary_inset,
        )
        bench = jnp.stack(
            [bench_x, jnp.full(self.N, -(self.hy - e_cfg.bench_touchline_inset))],
            axis=1,
        )
        P = jnp.where(state.active_player[:, None], P, bench)
        V = jnp.where(state.active_player[:, None], V, jnp.zeros_like(V))
        # [B3] 강제이동 중인 세트피스 키커(~eff_move)는 분리 밀어내기에서 제외(pin)한다. 상대 다수가
        # 스폿을 점거해도 키커가 목표 뒤로 확실히 도달(arrived=True)해 카운트다운·강제킥이 정상 진행 —
        # pin이 없으면 _apply_kicker_move가 매 서브스텝 재배치해도 _separate가 도로 밀어내 d_tgt가
        # kicker_arrive_r 밖에 갇혀 영구 데드볼(도착 전엔 카운트다운이 흐르지 않음). 오픈플레이에선
        # ~eff_move가 전부 False라 정상 분리.
        P, V = self._separate(P, V, state.active_player, pinned=~eff_move)

        # 분리 보정도 위치를 경계까지 클립할 수 있다. 적분 직후의 유령 속도를 위에서 지웠더라도,
        # _separate가 선수를 경계로 밀어낸 뒤 바깥 법선 속도를 그대로 돌려주면 동일 문제가 다시
        # 생긴다. 경계의 **바깥쪽 성분만** 제거해 안쪽 복귀와 접선 이동은 보존한다.
        at_left = P[:, DIM_X] <= -self.hx + GEOMETRY_EPS
        at_right = P[:, DIM_X] >= self.hx - GEOMETRY_EPS
        at_bottom = P[:, DIM_Y] <= -self.hy + GEOMETRY_EPS
        at_top = P[:, DIM_Y] >= self.hy - GEOMETRY_EPS
        vx_out = (at_left & (V[:, DIM_X] < 0.0)) | (at_right & (V[:, DIM_X] > 0.0))
        vy_out = (at_bottom & (V[:, DIM_Y] < 0.0)) | (at_top & (V[:, DIM_Y] > 0.0))
        V = V.at[:, DIM_X].set(jnp.where(vx_out, 0.0, V[:, DIM_X]))
        V = V.at[:, DIM_Y].set(jnp.where(vy_out, 0.0, V[:, DIM_Y]))

        # facing = 갱신된 속도의 방향(파생값). 독립 적분·turn_rate 상한·킥 방향 지향이 모두 사라져
        # facing이 obs로 복원 불가능한 은닉 상태를 들고 있지 않다(facing_from_velocity 참조).
        # 회전율은 마찰 타원의 횡가속 캡(accel_norm_max)이 이미 물리적으로 제한한다.
        facing = self.facing_from_velocity(V, state.attack_dir)

        # 스태미나: 필드 선수만 소모. 라이브 = 기저(stamina_drain_base) + 스프린트 추가분.
        speed_now = _safe_norm(V, axis=1)
        field = state.gk_indices == 0
        restart_active = state.restart_t > 0
        over = jnp.clip((speed_now - e_cfg.sprint_speed) / e_cfg.sprint_speed, 0.0, 2.0)
        sprint_extra = jnp.where(speed_now > e_cfg.sprint_speed,
                                 (e_cfg.stamina_sprint_mult - 1.0) * over, 0.0)
        drain = self.stamina_drain_base * (1.0 + sprint_extra)
        # [클론 변경 v2] 재개(데드볼) 중에는 스태미나를 **고정**한다 — 소모도 회복도 없다.
        #
        #   v1은 속도 비례 소모였다(`drain * clip(speed/sprint_speed, 0, 1)`). 그런데
        #   **현실과 env의 시간 스케일이 다르다**: 실제 데드볼은 p50 19.8초라 선수가
        #   걸어서 제자리를 잡지만(실측 속력 p50 1.03 m/s), env 재개창은 5초 안팎이라
        #   같은 거리를 **뛰어야** 한다. 속도 비례로 두면 env 정책만 스태미나를 더 쓴다 —
        #   실측으로 전력 이동 시 인플레이와 소모가 같았다(0.000368 vs 0.000371).
        #
        #   고정은 어느 쪽으로도 편향되지 않는다. 무소모(원본)면 데드볼이 공짜 휴식이
        #   되어 일부러 공을 내보내는 이득이 생기고, 속도 비례(v1)면 시간 스케일 차이가
        #   그대로 벌점이 된다. **재개 구간을 스태미나 회계에서 통째로 빼는 것**이 맞다.
        drain = jnp.where(restart_active, 0.0, drain)
        stamina = jnp.clip(state.stamina - dt * jnp.where(field, drain, 0.0), 0.0, 1.0)

        return state._replace(player_pos=P, player_vel=V, player_facing=facing, stamina=stamina)

    def _separate(self, P, V, active=None, pinned=None):
        """선수 겹침 해소: 최소거리(2×r_player) 미만인 쌍을 절반씩 밀어냄. sep_iters회 반복.
        완전히 겹친(거리≈0) 쌍은 방향이 정의되지 않으므로 황금각 분산 방향으로 밀어 데드락 방지.
        퇴장 선수(active=False)는 밀지도 밀리지도 않음(벤치 유령화 방지).

        Args:
            P: float[N,2] 위치
            V: float[N,2] 속도 — 분리는 위치만 조정, V는 그대로 통과
            active: bool[N] 온피치 여부(None이면 전원 참여)
            pinned: bool[N] 위치 고정 선수(None이면 없음). 고정 선수는 움직이지 않고, 겹친 상대가
                    전체 보정량을 받는다. 둘 다 고정인 쌍만 분리하지 않는다.
        Return:
            (P, V)
        """
        N = P.shape[0]
        min_d = 2.0 * self.r_player
        eye = jnp.eye(N, dtype=bool)
        ang = jnp.arange(N) * GOLDEN_ANGLE
        tie = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
        active_mask = jnp.ones(N, dtype=bool) if active is None else active
        pinned_mask = jnp.zeros(N, dtype=bool) if pinned is None else pinned
        movable = active_mask & (~pinned_mask)
        pair_on = (
            active_mask[:, None]
            & active_mask[None, :]
            & (~(pinned_mask[:, None] & pinned_mask[None, :]))
        )
        # 수신자 i가 움직일 비율: 보통 양쪽 1/2, 상대 j가 pinned면 i가 전량.
        receiver_share = (
            jnp.where(pinned_mask[None, :], 1.0, 0.5) * movable[:, None]
        )
        for _ in range(self.e_cfg.sep_iters):
            diff = P[:, None, :] - P[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff * diff, axis=2) + SQUARED_EPS)
            degenerate = dist < COINCIDENT_DISTANCE_EPS
            fallback = tie[:, None, :] - tie[None, :, :]
            direction_raw = jnp.where(degenerate[:, :, None], fallback, diff)
            direction = direction_raw / (
                _safe_norm(direction_raw, axis=2, keepdims=True) + DIV_EPS
            )
            overlap = jnp.where(
                eye | (~pair_on), 0.0, jnp.clip(min_d - dist, 0.0, min_d)
            )
            P = P + jnp.sum(
                overlap[:, :, None] * direction * receiver_share[:, :, None],
                axis=1,
            )
            P = P.at[:, 0].set(jnp.clip(P[:, 0], -self.hx, self.hx))
            P = P.at[:, 1].set(jnp.clip(P[:, 1], -self.hy, self.hy))
        return P, V
