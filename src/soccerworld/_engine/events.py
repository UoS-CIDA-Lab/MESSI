"""이벤트 판정 — 득점/아웃(코너·골킥·스로인) 라우팅(_events), 킥오프 배치, 하프타임 전환.

전부 결정적 기하 판정이라 관측 궤적(공 위치·재개 종류·스코어)에서 이벤트를 그대로 역산할 수 있다.
"""
import jax
import jax.numpy as jnp

from .constants import (
    BALL_ALIVE,
    BALL_DEAD,
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    DEPARTED_TAKER,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    FOUL_NONE,
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
    TEAM_COUNT,
    TOUCH_NONE,
)
from .restart import restart_timer_active


class Events:
    def _broken_pending_taker(self, state):
        """Whether an active restart points at an ineligible designated slot.

        Playback/public callers may inject a negative/out-of-range pending
        value, or a slot that has left the pitch or belongs to the other team.
        All are one compact-visible invalid class.  Keep this predicate shared
        by event repair and the control-frame causality boundary: a successful
        reassignment must be observed before release, while an invalid value
        with no eligible replacement canonicalizes to ``NO_PLAYER`` and lets
        the existing timeout fail-safe drain.
        """

        safe = jnp.clip(state.pending_taker, 0, self.N - 1)
        in_range = (state.pending_taker >= 0) & (state.pending_taker < self.N)
        return (
            restart_timer_active(state.restart_t)
            & ((~in_range)
               | (~state.active_player[safe])
               | (state.team_id[safe] != state.restart_team)
               # GK 홀드에 필드 선수가 앉아 있는 것도 같은 무효 부류다. 복구 쪽에만 GK
               # 규칙을 두고 **탐지**에서 빠뜨리면, 불법 상태가 정상으로 판정돼 그대로
               # 남는다(실측: 필드 선수 5가 GK 홀드 키커로 유지됐다).
               | ((state.restart_kind == RK_GK_HOLD)
                  & (state.gk_indices[safe] != 1)))
        )

    def _repair_broken_pending_taker(self, state):
        """Return ``(repaired_state, repaired)`` using the taker-selection SSOT.

        키커가 교체·퇴장되면 **같은 규칙으로 다시 고른다**. 종전에는 이 경로가 재개 종류를
        넘기지 않아 복구만 최근접으로 돌아갔다 — 스로인 키커가 빠지면 그 자리를 센터백이
        이어받는 식이었다. 종류를 함께 넘겨 최초 선택과 복구가 한 규칙을 쓰게 한다.

        GK 홀드는 반드시 GK가 이어받는다. 이것은 규칙이 아니라 법이라 결정자 안에서도
        같은 게이트로 강제된다.
        """

        broken = self._broken_pending_taker(state)
        # 복구는 **진행 중인** 재개라 state의 종류가 곧 현재 종류다.
        # 키커가 실제로 깨졌을 때만 결정자를 돌린다 — 이 함수는 제어 프레임마다
        # 세 곳에서 불리는데 그 중 대부분은 멀쩡한 키커를 두고 지나간다.
        replacement = self._designate_taker_when(
            broken,
            state, state.ball_pos[:DIM_Z], state.restart_team,
            (state.restart_kind == RK_GOALKICK)
            | (state.restart_kind == RK_GK_HOLD),
            state.restart_kind, state.restart_indirect,
        ).astype(jnp.int32)
        # GK 홀드에서 GK가 없으면 이어받을 사람이 없다 — 필드 선수를 GK로 둔갑시키지 않고
        # 실패로 두어 기존 타임아웃 배수로가 흐르게 한다.
        hold_ok = (state.restart_kind != RK_GK_HOLD) | (
            (replacement >= 0)
            & state.gk_indices[jnp.clip(replacement, 0, self.N - 1)] == 1)
        replacement_valid = (
            (replacement >= 0) & (replacement < self.N) & hold_ok)
        repaired = broken & replacement_valid
        # Canonicalize an unrepairable malformed reference to NO_PLAYER.  The
        # existing timeout path can then drain instead of preserving a hidden
        # inactive/OOR identity that compact State intentionally masks.
        canonical_replacement = jnp.where(
            replacement_valid, replacement, jnp.int32(NO_PLAYER)
        )
        pending = jnp.where(
            broken, canonical_replacement, state.pending_taker
        ).astype(jnp.int32)
        return state._replace(pending_taker=pending), repaired

    def _kickoff_taker(self, state, team, positions):
        """킥오프 키커 지정 — **다른 재개와 같은 결정자**를 쓴다.

        종전에는 이 함수만 센터스팟 최근접으로 따로 골랐다. 그래서 감독이 지정한 킥오프
        순번이 무시됐고(실측: 계획 슬롯 11 대신 21이 선택), 킥오프만 규칙에서 떨어져 나갔다.
        재개는 전부 한 규칙을 공유해야 한다.

        위치는 킥오프 포메이션 좌표를 쓴다 — 이 시점의 ``player_pos``는 아직 재배치 전이라
        실제 서 있을 자리와 다르다.
        """

        return self._kickoff_taker_when(jnp.bool_(True), state, team, positions)

    def _kickoff_taker_when(self, required, state, team, positions):
        """킥오프 키커 — 실제로 킥오프가 열릴 때만 결정자를 돌린다.

        득점과 하프타임은 드문 사건인데 결정자는 매 서브스텝·매 전환 후보에서 돌고 있었다.
        조건을 넘기면 결과는 같고(호출부가 어차피 ``jnp.where``로 버리던 값이다) 비용만
        사라진다.
        """

        snapped = state._replace(player_pos=jnp.asarray(positions, jnp.float32))
        return self._designate_taker_when(
            required,
            snapped, jnp.zeros(2, jnp.float32), team, jnp.bool_(False),
            jnp.int32(RK_KICKOFF), jnp.bool_(False))

    def _kickoff_positions(self, state):
        """킥오프 기준 포메이션(N,2) — base_formation을 현재 공격방향 부호로 **점대칭**(x·y 동시
        반전, 180° 회전). 골 후 킥오프·후반 전환의 선수 재배치에 공용.
        전 구간 회전 규약(setup의 ROTATE_180도 x·y 양쪽, obs 폴딩도 z회전)이므로 x만 뒤집으면
        진영 교대 시 좌/우 레인이 공격 프레임 기준으로 반전된다(전반 왼쪽 윙이 후반 오른쪽 윙이 됨)."""
        flip = jnp.where(state.attack_dir[0] > 0, 1.0, -1.0)
        return self.base_formation * flip

    def _events(
        self, state, arrived=None, sp_active=None, suppress_restart=None,
        ball_pos_before=None, preserve_restart_timer=None,
    ):
        """규칙 primitive — ``(state, score)``만 반환하는 좁은 표면."""

        state, scored, _ = self._events_with_restart_mask(
            state,
            arrived=arrived,
            sp_active=sp_active,
            suppress_restart=suppress_restart,
            ball_pos_before=ball_pos_before,
            preserve_restart_timer=preserve_restart_timer,
        )
        return state, scored

    def _events_with_restart_mask(
        self, state, arrived=None, sp_active=None, suppress_restart=None,
        ball_pos_before=None, preserve_restart_timer=None, substep_index=None,
    ):
        """득점·아웃을 판정하고 재개(restart) 상태·배치를 라우팅.

        반환은 ``(state, scored, kickoff_restart_forced)``이며 마지막 값은 득점 뒤 킥오프
        사전 스냅·재개 거리 투영으로 위치가 강제된 슬롯의 ``bool[N]`` 마스크다. 역사적 public/rule
        primitive :meth:`_events`는 앞의 두 값만 반환한다.

        goal: 골문 통과 → 킥오프(득점팀 상대 공). out_end: 골라인 아웃 → 코너/골킥. out_side:
        터치라인 아웃 → 스로인. 스로인 직접골은 양방향 무효(상대골→골킥/자책→코너, Law 15), 그 외 세트피스 직접 자책골은
        무효(코너 처리). 비이벤트 재개 프레임의 restart_t는 합법 키커가 공에
        도착할 때까지 멈추고, 도착 뒤 재개별 데드볼 유지시간을 감소한다.
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
        line_y = hy + self.r_ball
        # Referee boundaries apply only while the ball is in play.  Observed
        # restart synchronization deliberately preserves the current measured
        # ball coordinate, which may already lie beyond a line; re-adjudicating
        # that stationary dead ball would overwrite the observed restart with
        # a second throw-in/goal-kick event.
        live_for_event = (
            (state.ball_state == BALL_ALIVE)
            & (~restart_timer_active(state.restart_t))
        )
        x_out_raw = live_for_event & (jnp.abs(x) > line_x)
        y_out_raw = live_for_event & (jnp.abs(y) > line_y)

        if ball_pos_before is None:
            # 직접 호출은 한 번의 물리 적분 결과가 아닐 수 있다. 적분 전 표본이 없으면
            # 터치라인 교차 시점을 복원할 수 없으므로 골라인 우선으로 판정한다.
            # 환경 전이는 언제나 아래의 정확한 적분 전 좌표를 넘긴다.
            vx, vy, vz = state.ball_vel[DIM_X], state.ball_vel[DIM_Y], state.ball_vel[DIM_Z]
            overshoot = jnp.abs(x) - line_x
            frac_dt = jnp.clip(
                overshoot / (jnp.abs(vx) + DIV_EPS) / e_cfg.dt_phys, 0.0, 1.0
            )
            y_cross = y - vy * e_cfg.dt_phys * frac_dt
            z_cross = z - vz * e_cfg.dt_phys * frac_dt
            # Compatibility calls without a pre-integration sample cannot
            # reconstruct the exact touchline-crossing abscissa.
            x_side_cross = x
            z_side_cross = z
            end_first = jnp.bool_(True)
        else:
            before = jnp.asarray(ball_pos_before)
            if before.shape != (DIM_ALL,):
                raise ValueError(
                    f"ball_pos_before must have shape ({DIM_ALL},), got {before.shape}"
                )

            def crossing_fraction(before_coord, after_coord, boundary):
                before_abs = jnp.abs(before_coord)
                after_abs = jnp.abs(after_coord)
                delta = after_abs - before_abs
                radial_raw = jnp.where(
                    jnp.abs(delta) > GEOMETRY_EPS,
                    (boundary - before_abs) / delta,
                    1.0,
                )
                # 보통의 한 tick 경로는 원점 한쪽에 머무르며, 그때는 절대좌표 식이
                # 정확하다.  Custom but valid high-speed
                # configurations can cross x=0/y=0 and the opposite boundary
                # in one tick, however.  ``abs(after)-abs(before)`` is not the
                # slope of that V-shaped path and can move the crossing far
                # earlier than it occurred.  Solve the signed segment against
                # the boundary on the *after* side for that case.
                same_side = ((before_coord >= 0.0) == (after_coord >= 0.0))
                target = jnp.where(after_coord >= 0.0, boundary, -boundary)
                signed_delta = after_coord - before_coord
                signed_raw = jnp.where(
                    jnp.abs(signed_delta) > GEOMETRY_EPS,
                    (target - before_coord) / signed_delta,
                    1.0,
                )
                raw = jnp.where(same_side, radial_raw, signed_raw)
                return jnp.clip(raw, 0.0, 1.0)

            x_cross = crossing_fraction(before[DIM_X], x, line_x)
            y_cross_frac = crossing_fraction(before[DIM_Y], y, line_y)
            # At a corner a fast diagonal ball can clear both complete-ball
            # boundaries in one 1/90-s integration.  The first crossing owns
            # the restart; unconditional end-line priority misroutes an earlier
            # touchline exit as a goal kick/corner.
            end_first = (~y_out_raw) | (x_cross <= y_cross_frac)
            y_cross = before[DIM_Y] + x_cross * (y - before[DIM_Y])
            z_cross = before[DIM_Z] + x_cross * (z - before[DIM_Z])
            # A throw-in is taken where the ball crossed the touchline, not at
            # its end-of-substep sample.  The latter can be almost one physics
            # tick farther downfield and silently shifts every diagonal exit.
            x_side_cross = (
                before[DIM_X] + y_cross_frac * (x - before[DIM_X])
            )
            z_side_cross = (
                before[DIM_Z] + y_cross_frac * (z - before[DIM_Z])
            )
        # 공 전체가 포스트·크로스바 안쪽으로 통과해야 득점이다. 중심 좌표를 명목상 골문
        # 경계와 비교하면 반지름만큼 포스트/크로스바를 관통한 공도 골로 판정된다.
        goal_half_clear = self.goal_w / 2.0 - self.r_ball
        crossbar_clear = self.goal_h - self.r_ball
        in_goal = (jnp.abs(y_cross) <= goal_half_clear) & (z_cross <= crossbar_clear)
        # Equality means the ball's trailing surface is still tangent to the
        # goal line.  IFAB requires the *whole* ball to cross; use the same
        # strict boundary as x_out_raw rather than scoring one sample early.
        goal_plus = in_goal & live_for_event & (x > line_x) & end_first
        goal_minus = in_goal & live_for_event & (x < -line_x) & end_first
        scored_field = jnp.where(goal_plus, team_plus, jnp.where(goal_minus, team_minus, -1))
        # Public/playback states may carry a malformed positive sentinel.
        # It is not a real restart provenance: treating every ``>= 0`` value
        # as active suppresses otherwise valid goals forever, even though no
        # player row can own the out-of-range latch.
        throw_active = (
            ((state.throw_taker >= 0) & (state.throw_taker < self.N))
            | (state.throw_taker == DEPARTED_TAKER)
        )
        scored_field = jnp.where(throw_active, jnp.int32(-1), scored_field)  # 스로인 직접골 무효
        sp_take_active = (
            ((state.setpiece_taker >= 0) & (state.setpiece_taker < self.N))
            | (state.setpiece_taker == DEPARTED_TAKER)
        )
        own_goal_direct = sp_take_active & (scored_field == (1 - state.restart_team).astype(jnp.int32))
        scored_field = jnp.where(own_goal_direct, jnp.int32(-1), scored_field)  # 세트피스 직접 자책 무효
        # 간접 프리킥(백패스 IDFK 등): 두 번째 터치(setpiece_taker 클리어) 전 직접 골은 무효(IFAB Law 13).
        # 무효 시 아래 out_end 라우팅이 골킥(상대 골)·코너(자기 골)로 자동 처리.
        # ``restart_indirect`` is ball-phase provenance, not player identity.
        # If the original taker has been substituted, DEPARTED_TAKER keeps the
        # phase alive and the bool remains authoritative until a real touch.
        # ``restart_indirect`` only constrains a live ball while a real
        # set-piece provenance latch owns that phase.  Public/playback states
        # can contain an out-of-range positive taker; it is normalized to
        # NO_PLAYER below and must be NONE-equivalent for this same event as
        # well, rather than suppressing one goal before self-healing.
        indirect_direct = (
            state.restart_indirect & sp_take_active & (scored_field >= 0)
        )
        scored_field = jnp.where(indirect_direct, jnp.int32(-1), scored_field)
        is_goal_field = scored_field >= 0

        out_end = x_out_raw & end_first & (~is_goal_field)
        out_side = y_out_raw & ((~x_out_raw) | (~end_first)) & (~is_goal_field)
        out = out_end | out_side
        # ``1 - last_touch_team``은 ``last_touch_team``이 ``NO_TEAM``(-1)이면 존재하지 않는 팀 2를
        # 낸다. 그 값이 restart_team·poss_team에 그대로 실리면 지정 키커 없는 재개와
        # categorical 도메인 붕괴가 된다. 더 중요하게는, 팀 폴백만 poss를 쓰고
        # corner 분기는 raw last_touch를 쓰면 ``GOALKICK + 공격팀 재개``같은 불가능한
        # 조합이 나온다. 유효 최종터치 → 유효 소유팀 → 해당 골라인 공격팀(기본
        # 골킥 해석) 순의 하나의 effective provenance를 종류·팀 라우팅이 공유한다.
        valid_last_touch = (
            (state.last_touch_team == TEAM_0)
            | (state.last_touch_team == TEAM_1)
        )
        valid_poss = (state.poss_team == TEAM_0) | (state.poss_team == TEAM_1)
        valid_kickoff_team = (
            (state.kickoff_team == TEAM_0)
            | (state.kickoff_team == TEAM_1)
        )
        # A touchline exit exactly at x=+0/-0 has no closer attacking end.
        # Reusing a fixed world-side team there breaks team-swap symmetry in
        # the deliberate unknown-provenance recovery path.  ``kickoff_team``
        # is a valid match-long team latch and therefore a covariant tie-break.
        # If a public/playback State corrupts even that latch, the remaining
        # perfectly centred state has no information from which a covariant
        # single team can be chosen.  Fail closed to a compact-visible fixed
        # team.  player_id must not participate: identity is deliberately not
        # encoded in compact State/obs, so using it makes equal vectors acquire
        # different restart teams.
        centre_fallback_team = jnp.where(
            valid_kickoff_team,
            state.kickoff_team,
            jnp.int32(TEAM_0),
        ).astype(jnp.int32)
        end_attacker = jnp.where(
            x > 0.0,
            team_plus,
            jnp.where(x < 0.0, team_minus, centre_fallback_team),
        ).astype(jnp.int32)
        effective_last_touch = jnp.where(
            valid_last_touch,
            state.last_touch_team,
            jnp.where(valid_poss, state.poss_team, end_attacker),
        ).astype(jnp.int32)
        opp = (TEAM_1 - effective_last_touch).astype(jnp.int32)
        def_side = jnp.where(x > 0, team_minus, team_plus).astype(jnp.int32)
        corner = out_end & (effective_last_touch == def_side)
        goalkick = out_end & (~corner)

        is_goal = is_goal_field
        scored = scored_field.astype(jnp.int32)
        # suppress_restart(reconstruct용 pin): sim 공이 드리프트로 라인을 넘어도 재개(골/아웃→세트피스·
        # 킥오프=전원 포메이션 스냅)를 유발하지 않게 이벤트 플래그를 봉쇄한다. 실경기 연속 창엔 재개가
        # 없으므로(관측 근거) sim 아웃은 순수 드리프트 아티팩트 — 위치 pin 아님(공은 그대로 굴러가고
        # 선수는 free-running 유지, 공 ADE엔 손실이 그대로 반영). suppress_charge/body와 동류.
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
            jnp.clip(x_side_cross, -hx + spot_inset, hx - spot_inset),
            jnp.sign(y) * (hy - e_cfg.throwin_line_inset),
            self.r_ball,
        ])
        # A corner belongs to the side on which the ball crossed the goal
        # line, not the side of its tick-end overshoot.  At an exactly central
        # over-bar crossing both corners are equidistant; a fixed +y fallback
        # breaks 180-degree/team-swap symmetry and sends both ends to the same
        # touchline.  Goal-line sign is a deterministic covariant tie-break
        # (and equals the corner-taking side's attack direction).
        corner_side = jnp.where(
            jnp.abs(y_cross) > GEOMETRY_EPS,
            jnp.sign(y_cross),
            jnp.sign(x),
        )
        corner_spot = jnp.array([
            jnp.sign(x) * (hx - spot_inset),
            corner_side * (hy - spot_inset),
            self.r_ball,
        ])
        goalkick_spot = jnp.array([
            jnp.sign(x) * (hx - e_cfg.goalkick_depth),
            jnp.where(
                jnp.abs(y_cross) > GEOMETRY_EPS,
                jnp.sign(y_cross),
                jnp.sign(x),
            ) * e_cfg.goalkick_lateral_offset,
            self.r_ball,
        ])
        ball_pos = jnp.where(is_goal, center,
                    jnp.where(corner, corner_spot,
                     jnp.where(goalkick, goalkick_spot,
                      jnp.where(out_side, side_spot, state.ball_pos))))
        ball_vel = jnp.where(event, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(event, jnp.zeros(DIM_ALL), state.ball_spin)

        # 라인 통과 telemetry — 위에서 재개 스폿을 정하는 데 이미 쓴 교차점이다.
        # 재개 스폿은 코너·골킥처럼 고정 좌표로 뭉개지고 골은 공을 센터서클로 보내므로,
        # 여기서 남기지 않으면 '어디로 어떤 속도로 나갔는가'가 프레임 경계에서 사라진다.
        # 좌표는 **공 중심**이 판정선 위에 있던 순간의 값이다(IFAB 전체통과 기준선).
        ball_event_kind = jnp.where(
            is_goal, jnp.int32(BALL_EVENT_GOAL),
            jnp.where(
                corner, jnp.int32(BALL_EVENT_CORNER),
                jnp.where(
                    goalkick, jnp.int32(BALL_EVENT_GOALKICK),
                    jnp.where(
                        out_side, jnp.int32(BALL_EVENT_THROWIN),
                        jnp.int32(BALL_EVENT_NONE),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        ball_event_team = jnp.where(
            is_goal, scored, jnp.where(out, opp, jnp.int32(NO_TEAM))
        ).astype(jnp.int32)
        goal_line_cross = jnp.stack([
            jnp.sign(x) * line_x, y_cross, z_cross
        ]).astype(state.ball_pos.dtype)
        touch_line_cross = jnp.stack([
            x_side_cross, jnp.sign(y) * line_y, z_side_cross
        ]).astype(state.ball_pos.dtype)
        ball_event_pos = jnp.where(out_side, touch_line_cross, goal_line_cross)

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

        # 비이벤트 재개 프레임: 지정 키커가 실제 공 접촉 거리에 들어올
        # 때까지 시계를 얼리고, 도착 뒤에만 종류별 유지시간을 춘소한다.
        # 진입 상태의 지정 키커가 비활성/타 팀 슬롯이면 이 호출의 뒤쪽 복구 경로가
        # 새 키커를 지정할 때까지 타이머를 보존한다. 특히 restart_t==1에서 먼저 0으로
        # 소진하면 restart_kind/pending_taker가 revived 경로에서 지워져 복구 자체가
        # 불가능해지고, 무킥 상태로 공만 살아나는 조용한 규칙 우회가 된다.
        _, entry_taker_repaired = self._repair_broken_pending_taker(state)
        preserve_timer = (
            jnp.bool_(False)
            if preserve_restart_timer is None
            else jnp.asarray(preserve_restart_timer, dtype=bool)
        )
        if preserve_timer.shape != ():
            raise ValueError(
                "preserve_restart_timer must be scalar, got shape "
                f"{preserve_timer.shape}"
            )
        if arrived is None:
            countdown = jnp.where(
                entry_taker_repaired | preserve_timer,
                state.restart_t,
                jnp.maximum(0, state.restart_t - 1),
            )
        else:
            safe_pending = jnp.clip(state.pending_taker, 0, self.N - 1)
            pending_in_range = (
                (state.pending_taker >= 0)
                & (state.pending_taker < self.N)
            )
            # Contact locks decay at the *end* of a physics tick.  If an
            # already-arrived taker reaches restart_t==1 with one short lock
            # remaining, decrementing the restart first revives a live loose
            # ball without the mandatory kick.  Preserve only this final timer
            # tick; the lock still decays below and the next substep releases.
            # The arrival-relative hold cannot manufacture a live loose ball.
            # Hold timer=1 while a valid designated taker is still outside
            # physical reach, or has arrived with a transient contact lock.
            # An invalid/unrepairable taker has ``sp_active=False`` and keeps
            # the existing timeout path instead of deadlocking forever.
            active_for_release = (
                jnp.bool_(False)
                if sp_active is None
                else jnp.asarray(sp_active, dtype=bool)
            )
            if active_for_release.shape != ():
                raise ValueError(
                    "sp_active must be scalar, got shape "
                    f"{active_for_release.shape}"
                )
            release_blocked_at_expiry = (
                active_for_release
                & (state.restart_t <= 1)
                & (
                    (~arrived)
                    | (
                        pending_in_range
                        & (
                            (state.contact_lock_t[safe_pending] > 0)
                            | (state.aerial_recovery_t[safe_pending] > 0)
                        )
                    )
                )
            )
            can_count = (
                (~entry_taker_repaired)
                & (~preserve_timer)
                & (~release_blocked_at_expiry)
                # A valid set-piece taker's approach is part of the dead-ball
                # duration but not part of the three-second post-arrival hold.
                # Invalid/no-taker states retain the timeout escape path.
                & ((~active_for_release) | arrived)
            )
            countdown = jnp.where(can_count, jnp.maximum(0, state.restart_t - 1), state.restart_t)
        restart_t = jnp.where(event, jnp.int32(e_cfg.restart_substeps), countdown).astype(jnp.int32)
        ball_state = jnp.where(
            event,
            BALL_DEAD,
            jnp.where(restart_timer_active(restart_t), state.ball_state, BALL_ALIVE),
        ).astype(jnp.int32)
        # 카운트다운 소진 페일세이프(킥 미실행): restart_kind·pending_taker도 리셋(obs is_taker 오염 방지).
        revived = (
            (~event)
            & restart_timer_active(state.restart_t)
            & (~restart_timer_active(restart_t))
        )
        # Both a referee event and a timeout revival terminate the preceding
        # dead/live phase.  Treat this as one SSOT below so no Law-11 or foul
        # provenance can leak through one branch but not another.
        phase_ended = event | revived
        restart_kind = jnp.where(revived, jnp.int32(RK_NONE), restart_kind).astype(jnp.int32)
        pending_taker_rv = jnp.where(revived, jnp.int32(-1), state.pending_taker).astype(jnp.int32)

        goalkick_any = goalkick                              # 골킥만 GK가 키커(페널티 세이브 분기 제거됨)
        ko_pos = self._kickoff_positions(state)
        # 아웃오브플레이 재개 — 스로인·코너·골킥이 여기서 갈린다. 종류를 넘기지 않으면
        # 이전 state의 RK_NONE을 읽어 전부 같은 규칙으로 골랐다.
        taker = self._designate_taker_when(
            event & (~is_goal),
            state, ball_pos[:DIM_Z], restart_team, goalkick_any,
            restart_kind, jnp.bool_(False))
        taker = jnp.where(
            is_goal,
            self._kickoff_taker_when(is_goal, state, restart_team, ko_pos),
            taker)
        pending_taker = jnp.where(event, taker, pending_taker_rv).astype(jnp.int32)
        # 진행 중인 재개의 지정 키커가 더는 합법이 아니면(퇴장·교체 아웃으로 비활성이 되거나
        # 재개팀 소속이 아니게 되면) 활성 명단에서 다시 뽑는다. 이 복구가 없으면 그 재개는
        # **영구 데드볼**이 된다 — 강제 킥은 ``st.active_player``를 요구해 발사되지 않고,
        # 카운트다운은 벤치로 치워진 키커가 스폿에 '도착'하지 못해 얼어붙어 실패세이프(revived)
        # 조차 돌지 않는다. 전방 전이로는 도달하지 않지만(데드볼 중에는 파울이 나지 않아 키커가
        # 퇴장당할 수 없다) traced playback·데이터셋 재구성이 상태를 주입할 수 있다.
        # taker가 -1인 경우는 건드리지 않는다 — 그때는 sp_active가 꺼져 카운트다운이 흘러
        # revived로 스스로 풀린다.
        repair_probe = state._replace(
            ball_pos=ball_pos,
            restart_t=restart_t,
            restart_kind=restart_kind,
            restart_team=restart_team,
            pending_taker=pending_taker,
        )
        taker_invalid = self._broken_pending_taker(repair_probe)
        repaired_probe, _ = self._repair_broken_pending_taker(
            repair_probe
        )   # 무효일 때만 결정자가 돈다 — 게이트는 그 함수 안에 있다.
        pending_taker = jnp.where(
            (~event) & taker_invalid,
            repaired_probe.pending_taker,
            pending_taker,
        ).astype(jnp.int32)
        player_pos = jnp.where(is_goal, ko_pos, state.player_pos)
        player_vel = jnp.where(is_goal, jnp.zeros_like(state.player_vel), state.player_vel)
        off_flag = jnp.where(phase_ended, jnp.zeros_like(state.offside_flag), state.offside_flag)
        pass_t = jnp.where(phase_ended, jnp.int32(0), state.pass_t)
        # Heal malformed public/playback latches even on a non-event tick.
        # Downstream retouch logic is slot-indexed, so an out-of-range value
        # cannot carry any lawful provenance and must be equivalent to NONE.
        valid_throw_taker = (
            ((state.throw_taker >= 0) & (state.throw_taker < self.N))
            | (state.throw_taker == DEPARTED_TAKER)
        )
        valid_setpiece_taker = (
            ((state.setpiece_taker >= 0) & (state.setpiece_taker < self.N))
            | (state.setpiece_taker == DEPARTED_TAKER)
        )
        throw_taker_in = jnp.where(
            valid_throw_taker, state.throw_taker, jnp.int32(NO_PLAYER)
        )
        setpiece_taker_in = jnp.where(
            valid_setpiece_taker, state.setpiece_taker, jnp.int32(NO_PLAYER)
        )
        throw_taker = jnp.where(event, jnp.int32(NO_PLAYER), throw_taker_in).astype(jnp.int32)
        setpiece_taker = jnp.where(
            event, jnp.int32(NO_PLAYER), setpiece_taker_in
        ).astype(jnp.int32)
        # A timeout fail-safe ends the dead-ball phase just as surely as a
        # taken restart.  Carrying its foul latch into live play exposes stale
        # actor/victim telemetry and can make a later legitimate tackle by the
        # old actor look like the stopped foul to ``_kick_applied``.
        foul_kind = jnp.where(
            phase_ended, jnp.int32(FOUL_NONE), state.foul_kind
        )
        # Referee placement/ownership is not a deliberate player touch.  Once
        # the ball leaves play (or a timed-out restart is made live), an older
        # PASS/DRIBBLE code must not survive and later trigger the goalkeeper
        # back-pass offence against the administratively assigned team.
        last_touch_code = jnp.where(
            phase_ended, jnp.int32(TOUCH_NONE), state.last_touch_code
        )
        gk_handling_restricted_team = jnp.where(
            phase_ended,
            jnp.int32(NO_TEAM),
            state.gk_handling_restricted_team,
        ).astype(jnp.int32)
        # facing은 속도 파생값이므로 득점 재배치 후 다시 계산한다.
        player_facing = self.facing_from_velocity(player_vel, state.attack_dir)

        # 새 재개(골/아웃/페일세이프 복귀)는 전부 직접 — 간접 플래그 클리어(스테일 방지).
        # 그 외(진행 중 IDFK)는 carry해 두 번째 터치 전까지 유지.
        malformed_setpiece_taker = (
            (state.setpiece_taker != NO_PLAYER) & (~valid_setpiece_taker)
        )
        restart_indirect = jnp.where(
            phase_ended | malformed_setpiece_taker,
            jnp.bool_(False),
            state.restart_indirect,
        )

        new_state = state._replace(
            ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
            player_pos=player_pos, player_vel=player_vel, player_facing=player_facing,
            poss_team=poss, last_touch_team=last_touch,
            # 골·아웃은 팀 귀속을 바꾸므로 개별 행위자는 더 이상 유효하지 않다.
            last_touch_actor=jnp.where(
                event, jnp.int32(NO_PLAYER), state.last_touch_actor
            ).astype(jnp.int32),
            score=score, restart_team=restart_team, restart_t=restart_t, restart_kind=restart_kind,
            ball_state=ball_state, pending_taker=pending_taker, offside_flag=off_flag, pass_t=pass_t,
            foul_kind=foul_kind, throw_taker=throw_taker, setpiece_taker=setpiece_taker,
            last_touch_code=last_touch_code,
            gk_handling_restricted_team=gk_handling_restricted_team,
            restart_indirect=restart_indirect)
        if substep_index is not None:
            # 이 substep 칸에만 쓴다. 사건이 없으면 빈 칸 값을 그대로 넣어 프레임 안에서
            # 멱등이다. ``substep_index``는 물리 스캔만 알고 있으므로, 규칙 primitive를
            # 직접 부르는 호출자는 이 telemetry를 남기지 않는다.
            occurred = ball_event_kind != jnp.int32(BALL_EVENT_NONE)
            new_state = new_state._replace(
                ball_event_kind=new_state.ball_event_kind.at[substep_index].set(
                    ball_event_kind
                ),
                ball_event_team=new_state.ball_event_team.at[substep_index].set(
                    jnp.where(occurred, ball_event_team, jnp.int32(NO_TEAM))
                ),
                ball_event_pos=new_state.ball_event_pos.at[substep_index].set(
                    jnp.where(occurred, ball_event_pos,
                              jnp.zeros_like(ball_event_pos))
                ),
                ball_event_vel=new_state.ball_event_vel.at[substep_index].set(
                    jnp.where(occurred, state.ball_vel,
                              jnp.zeros_like(state.ball_vel))
                ),
                ball_event_control_t=(
                    new_state.ball_event_control_t.at[substep_index].set(
                        jnp.where(occurred, state.t, jnp.int32(-1))
                    )
                ),
            )
        new_state = self._normalize_pass_latch(new_state, clear=phase_ended)
        # ``_events`` is also a directly exercised rule primitive.  Do not rely
        # on the outer environment scan to repair actor/victim identities after
        # clearing ``foul_kind`` here.
        new_state = self._normalize_foul_latch(new_state)
        new_state = jax.lax.cond(
            is_goal,
            self._project_inactive_players,
            lambda current: current,
            new_state,
        )
        # 킥오프 키커 pre-snap(BC 라벨 드롭 방지): 득점 후 킥오프는 키커가 포메이션에서 시작해
        # 다음 스텝 진입 시 arrived=False → action_agency가 kick_forced=False로 산출하는데, 그 스텝의
        # 서브스텝에서 _apply_kicker_move가 스냅→즉시 강제킥이 발사(kick_applied=True)되어 진입 마스크가
        # 킥 dim을 닫은 채 라벨이 드롭된다. reset_array의 오프닝 킥오프와 동일하게 여기서 미리 스냅해
        # 다음 스텝 진입 arrived=True로 만들어 kick_forced가 킥 dim을 연다.
        ko_snap = (
            (new_state.restart_kind == RK_KICKOFF)
            & restart_timer_active(new_state.restart_t)
        )
        new_state, kickoff_restart_forced = jax.lax.cond(
            ko_snap,
            self._snap_kickoff_taker_with_mask,
            lambda current: (
                current, jnp.zeros(self.N, dtype=bool)
            ),
            new_state,
        )
        return new_state, scored, kickoff_restart_forced

    def _halftime_switch(self, state, do):
        """후반 전환(do=True일 때만 적용) — 진영·공격방향 반전, 중앙 킥오프 재배치, 후반 킥오프팀 설정.
        스태미나는 리셋하지 않는다(실측상 하프타임 회복 없음)."""

        return self._halftime_switch_with_mask(state, do)[0]

    def _halftime_switch_with_mask(self, state, do):
        """Apply halftime and retain any internal kickoff-rule projection mask."""

        N = self.N
        pos_2h = -self._kickoff_positions(state)          # 진영 교대 = 점대칭(x·y 반전, 회전 규약)
        adir_2h = -state.attack_dir
        # facing은 속도 파생값 — 정지 재배치라 새 공격 방향이 그대로 정지 규약 facing이 된다.
        facing_2h = self.facing_from_velocity(jnp.zeros((N, DIM_Z)), adir_2h)
        second_kick = (1 - state.kickoff_team).astype(jnp.int32)
        center = jnp.array([0.0, 0.0, self.r_ball])
        state_2h = state._replace(player_pos=pos_2h, attack_dir=adir_2h)
        taker = self._kickoff_taker_when(do, state_2h, second_kick, pos_2h)
        switched = state._replace(
            player_pos=pos_2h, player_vel=jnp.zeros((N, 2)), player_facing=facing_2h,
            attack_dir=adir_2h, ball_pos=center, ball_vel=jnp.zeros(DIM_ALL), ball_spin=jnp.zeros(DIM_ALL),
            poss_team=second_kick, ball_state=jnp.int32(BALL_DEAD),
            # 하프타임 휴식은 경기 시계 밖에서 이미 끝났다. 다음 기록 프레임은
            # 경기 시작과 같이 즉시 가능한 킥오프로 연다; 득점 뒤 킥오프만 full window를 쓴다.
            restart_team=second_kick, restart_t=jnp.int32(1),
            restart_kind=jnp.int32(RK_KICKOFF), pending_taker=taker, last_touch_team=second_kick,
            ctrl_lock_t=jnp.zeros(N, dtype=jnp.int32),
            contact_lock_t=jnp.zeros(N, dtype=jnp.int32),
            aerial_recovery_t=jnp.zeros(N, dtype=jnp.int32),
            # 진영이 바뀌므로 role anchor 누적을 리셋한다(데이터셋이 period별로 쌓는 것과 동일).
            # 접힘 프레임에서는 prior가 그대로라 role_pos 자체는 유지한다.
            role_pos_count=jnp.zeros(N, dtype=state.role_pos_count.dtype),
            cooldown=jnp.zeros(N), offside_flag=jnp.zeros(N, bool),
            pass_team=jnp.int32(NO_TEAM), pass_t=jnp.int32(0),
            throw_taker=jnp.int32(-1), setpiece_taker=jnp.int32(-1),
            # 코드와 행위자는 한 쌍이다. 코드만 지우면 ``last_touch_actor >= 0``이
            # 남아 관측의 last-touch 관계 열이 전반에 공을 만진 선수를 계속 가리킨다 —
            # 다음 전이와 무관한 정보가 obs에 남는 Markov 계약 위반이다.
            restart_indirect=jnp.bool_(False), last_touch_code=jnp.int32(TOUCH_NONE),
            last_touch_actor=jnp.int32(NO_PLAYER),
            gk_handling_restricted_team=jnp.int32(NO_TEAM),
            foul_kind=jnp.int32(FOUL_NONE), foul_actor=jnp.int32(-1), foul_victim=jnp.int32(-1),
        )
        # 2H 킥오프도 전환 state에서 미리 스냅해 다음 step 진입 라벨을 연다.
        switched, restart_forced = self._snap_kickoff_taker_with_mask(switched)
        switched = self._project_inactive_players(switched)
        result = jax.tree_util.tree_map(
            lambda a, b: jnp.where(do, a, b), switched, state
        )
        return result, restart_forced & do
