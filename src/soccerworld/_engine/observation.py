import jax
import jax.numpy as jnp
import numpy as np

from . import formation as formation_module
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
    AFFORDANCE_FEATURES,
    AFFORDANCE_SCHEMA_VERSION,
    BALL_ALIVE,
    DEPARTED_TAKER,
    DEPARTED_TAKER_BIT_SETPIECE,
    DEPARTED_TAKER_BIT_THROW,
    DEPARTED_TAKER_MASK_COUNT,
    DIM_ALL,
    DIM_X,
    DIM_Z,
    DIV_EPS,
    ENDURANCE_FACTOR_MAX,
    ENDURANCE_FACTOR_MIN,
    ENDURANCE_FACTOR_REFERENCE,
    GK_HANDLING_RELEASE_OFFSET,
    GK_HANDLING_RESTRICTION_COUNT,
    IFAB_MAX_SUBSTITUTION_WINDOWS,
    INACTIVE_SLOT_LIVE_COLUMNS,
    KICK_GATE_THRESHOLD,
    NO_TEAM,
    OBS_ANCHOR_FEATURES,
    OBS_BALL_FEATURES,
    OBS_BENCH_FEATURES,
    OBS_CONTEXT_FEATURES,
    OBS_PLAYER_FEATURES,
    PLAYER_PROFILE_FEATURES,
    POSSESSION_CONTEXT_SECONDS,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_PENALTY,
    ROLE_GAIN_EXACT_MAX_SAMPLES,
    SELF_TOUCH_RELATION,
    SLOT_ACTIVE,
    SLOT_INACTIVE,
    SLOT_SENT_OFF,
    SLOT_STATUS_COUNT,
    STATE_BALL_FEATURES,
    STATE_BENCH_FEATURES,
    STATE_GAME_FEATURES,
    STATE_PLAYER_FEATURES,
    TAKER_BIT_PENDING,
    TAKER_BIT_SETPIECE,
    TAKER_BIT_THROW,
    TAKER_MASK_COUNT,
    TEAM_0,
    TEAM_1,
    TEAM_COUNT,
    TOUCH_COUNT,
    YELLOW_CARD_SEND_OFF_COUNT,
)
from .restart import restart_timer_active
from .spatial import _u01, stretch_decode


def _team_ball_rank(player_pos, ball_pos, active, team_id):
    """Return each active player's stable within-team ball-distance rank."""

    slots = jnp.arange(player_pos.shape[0], dtype=jnp.int32)
    distance = jnp.linalg.norm(player_pos - ball_pos[None, :2], axis=-1)
    distance = jnp.where(active, distance, 1e6)
    member = (team_id[:, None] == team_id[None, :]) & active[None, :]
    ahead = (
        (distance[None, :] < distance[:, None])
        | (
            (distance[None, :] == distance[:, None])
            & (slots[None, :] < slots[:, None])
        )
    )
    rank = jnp.sum(member & ahead, axis=1).astype(jnp.float32)
    team_size = jnp.sum(member, axis=1)
    return jnp.where(
        active,
        rank / jnp.maximum(team_size - 1, 1).astype(jnp.float32),
        0.0,
    )


class Observation:
    def _normalize_gk_handling_latch(self, state):
        """Keep goalkeeper-handling provenance valid only in live open play."""

        valid_code = (
            (state.gk_handling_restricted_team >= 0)
            & (state.gk_handling_restricted_team
               < GK_HANDLING_RESTRICTION_COUNT)
        )
        live_open_play = (
            (state.ball_state == BALL_ALIVE)
            & (~restart_timer_active(state.restart_t))
        )
        keep = valid_code & live_open_play
        return state._replace(
            gk_handling_restricted_team=jnp.where(
                keep,
                state.gk_handling_restricted_team,
                jnp.int32(NO_TEAM),
            ).astype(jnp.int32)
        )

    def _contact_lock_allows_f2b(self, state):
        """물리시간 기반 접촉 간격이 끝났는지 여부(관측·agency·runtime 공용)."""

        return state.contact_lock_t <= 0

    def _aerial_recovery_allows_f2b(self, state):
        """Whether a previous aerial attempt permits another voluntary F2B."""

        return state.aerial_recovery_t <= 0

    def _ctrl_lock_allows_f2b(self, state):
        """Whether the deterministic possession-loss lock permits challenge."""

        locked_opponent = ((state.poss_team >= 0)
                           & (state.team_id != state.poss_team)
                           & (state.ctrl_lock_t > 0))
        return ~locked_opponent

    def _retouch_allows_f2b(self, state):
        """Voluntary f2b legality after a restart, shared by obs and runtime.

        The previous throw/set-piece taker cannot touch the ball again until
        another player does.  Forced restart release is the sole exception and
        is applied separately by ``env._kick_gate``.
        """

        ar = self.player_indices
        return ~((ar == state.throw_taker) | (ar == state.setpiece_taker))

    def _cooldown_allows_f2b(self, state):
        """Whether cooldown permits this player's ball action.

        ``cooldown`` is a remaining re-challenge lock, not a universal touch
        lock.  The team that currently possesses the ball may keep playing it;
        an opponent, or every player while the ball is loose, must wait for the
        lock to expire.  Keeping this predicate here lets observation,
        action-agency and the transition gate share the same contract.
        """

        possession_team_action = (
            (state.poss_team >= 0) & (state.team_id == state.poss_team)
        )
        return (state.cooldown <= 0) | possession_team_action

    def _cooldown_norm_scale(self):
        """Maximum re-challenge lock, in physics substeps, for observations."""

        return float(self.e_cfg.cooldown_substeps + self.e_cfg.challenge_cooldown_extra)

    # ── compact 인코딩 헬퍼 — 관측·중앙 상태가 같은 구현을 공유한다 ──────────────
    def _status_code(self, state):
        """슬롯 참여 상태를 categorical 하나로 접는다.

        projection인 ``active_player`` 한 비트로는 '교체로 빠진 슬롯'과 '퇴장으로 빠진
        슬롯'이 같은 값이 된다. 둘은 팀 인원 하한(IFAB) 판정과 이후 투입 가능성이 달라
        전이가 갈리므로 서로 다른 코드여야 한다. 퇴장이 온피치 비트보다 우선한다.
        """

        return jnp.where(
            state.sent_off, SLOT_SENT_OFF,
            jnp.where(state.on_pitch, SLOT_ACTIVE, SLOT_INACTIVE),
        ).astype(jnp.float32)

    def _taker_mask(self, state):
        """pending/setpiece/throw 세 개의 N차원 마스크를 슬롯당 비트마스크 하나로 접는다.

        세 비트는 동시에 설 수 있으므로 categorical이 아니라 비트마스크다.
        """

        ar = self.player_indices
        pending = (ar == state.pending_taker) & (state.pending_taker >= 0)
        setpiece = (ar == state.setpiece_taker) & (state.setpiece_taker >= 0)
        throw = (ar == state.throw_taker) & (state.throw_taker >= 0)
        return (
            pending.astype(jnp.float32) * TAKER_BIT_PENDING
            + setpiece.astype(jnp.float32) * TAKER_BIT_SETPIECE
            + throw.astype(jnp.float32) * TAKER_BIT_THROW
        )

    @staticmethod
    def _departed_taker_mask(state):
        """Actor-free restart provenance retained across an identity boundary.

        A substitution can remove the original throw/set-piece taker before
        anybody else touches the ball.  The replacement player must not inherit
        that person's retouch lock, but direct-goal adjudication remains active.
        The per-slot ``taker_mask`` therefore cannot carry this state; expose the
        two independent global latches as one compact bitmask instead.
        """

        return (
            (state.throw_taker == DEPARTED_TAKER).astype(jnp.float32)
            * DEPARTED_TAKER_BIT_THROW
            + (state.setpiece_taker == DEPARTED_TAKER).astype(jnp.float32)
            * DEPARTED_TAKER_BIT_SETPIECE
        )

    @staticmethod
    def taker_bits(taker_mask):
        """``taker_mask`` → (pending, setpiece_retouch, throw_retouch) bool. 소비자 공용 디코더."""

        code = jnp.round(taker_mask).astype(jnp.int32)
        return (
            (code & TAKER_BIT_PENDING) > 0,
            (code & TAKER_BIT_SETPIECE) > 0,
            (code & TAKER_BIT_THROW) > 0,
        )

    @staticmethod
    def _role_gain(state):
        """누적평균 갱신 이득 ``1/(1+n)``.

        ``role_pos``는 현재 전술 epoch 라이브볼 위치의 인과적 누적평균이라 다음 갱신이
        ``role_pos += (x - role_pos) * gain``이다. 즉 표본 수 ``n``이 전이에 기여하는 전부가 이
        이득 하나이고, 이 값 없이 ``role_pos``만 실으면 같은 벡터에서 다음 상태가 갈린다
        (엄밀히 Markov 정보를 잃는 압축). ``n = 1/gain - 1``로 역산되므로 표본 수 자체도 보존된다.
        """

        return 1.0 / (1.0 + state.role_pos_count)

    def _decode(self, act_arr, attack_dir):
        """행동 배열 (N,ACTION_DIM=8) → 행동 그룹 튜플.
        방향 액션(mv_dir/f2b_dir)은 자기중심(공격) 프레임으로 입력받아 ×attack_dir로 월드 프레임 변환.
        사이드스핀(z축 유사벡터)은 180° 회전에서 불변이라 언폴딩 금지 — 원본(거울 규약)의
        contest × attack_dir * spin을 옮겨오면 안 됨. 백스핀은 진행방향 기준 축이라 프레임 무관."""
        act_arr = jnp.asarray(act_arr)
        expected_shape = (self.N, ACTION_DIM)
        if act_arr.shape != expected_shape:
            raise ValueError(f"act_arr must have shape {expected_shape}, got {act_arr.shape}")
        # NaN 무해화: 폭주 정책의 NaN 액션 하나가 clip을 통과(clip(NaN)=NaN)해 공·전 선수
        # 상태를 회복 불능으로 오염시키는 것을 차단(적대 퍼징 발견). NaN→0(중립 커맨드).
        act_arr_clip = jnp.clip(
            jnp.nan_to_num(act_arr, nan=0.0), ACTION_MIN, ACTION_MAX
        )
        want_f2b = act_arr_clip[:, ACTION_KICK_GATE] > KICK_GATE_THRESHOLD
        # 이동·킥은 L∞ radial stretch 2D — 방향(unit)·크기(‖·‖∞)를 한 2D 벡터에서 디코드(등방·전단사·방향보존).
        mv_dir_ego, mv_pow = stretch_decode(act_arr_clip[:, ACTION_MOVE])
        mv_dir = mv_dir_ego * attack_dir[:, None]                       # 정책 프레임 → 월드
        f2b_dir_ego, f2b_pow = stretch_decode(act_arr_clip[:, ACTION_KICK_VECTOR])
        f2b_dir = f2b_dir_ego * attack_dir[:, None]
        f2b_launch = _u01(act_arr_clip[:, ACTION_LAUNCH]) * self.e_cfg.launch_max
        spin_side = act_arr_clip[:, ACTION_SPIN_SIDE]
        spin_back = act_arr_clip[:, ACTION_SPIN_BACK]
        return (
            want_f2b, mv_dir, mv_pow,
            f2b_dir, f2b_pow, f2b_launch,
            spin_side, spin_back
        )

    def get_obs_array(self, state):
        """전 에이전트 관측 (N, obs_dim) — Causal-Compact 레이아웃.

        구성은 anchor, N개의 동일한 player token, ball, context, 양 팀 bench block이다.
        관측자 self와 타 선수를 나누지 않고 **전 슬롯을 같은 폭의 토큰**으로 낸다(자기
        토큰은 rel_pos=rel_vel=0). 덕분에 하나의 entity encoder가 N개 토큰을 그대로 먹을
        수 있다(기본 11대11에서는 22개). 각 block의 실제 폭과 열 위치는
        :meth:`obs_spec` 및 :data:`constants.OBS_PLAYER_FEATURES`가 유일한 진실원천이다.

        절대량은 앵커에서 정확히 복원되므로 싣지 않는다::

            p_j = anchor_pos + rel_pos_j            (둘 다 field_half 정규화)
            v_j = anchor_vel + rel_vel_j            (둘 다 norm_player_vel 정규화)
            p_ball = anchor_pos + ball_rel_pos
            v_ball = anchor_vel * norm_player_vel + ball_rel_vel * norm_ball_vel

        선수 위치·속도의 상대량과 앵커가 **같은 정규화 상수**를 쓰는 것이 핵심이다. rel_pos만
        ``field_size``로 나누면 복원이 단순 덧셈이 아니게 되고 절대 좌표를 따로 실어야 한다.

        파생 affordance(``in_reach``/``f2b_avail``/``ball_rank``/``off_line``/침범 기하/키커 잠금/
        ``pass_signal``/``is_sp_ours``)는 여기 없다 — :meth:`affordance_view`가 같은 상태에서
        결정론적으로 재계산한다. 관측과 affordance를 한 벡터에 섞으면 저장 데이터가 파생량을
        원천처럼 굳혀 두 계약이 갈린다.
        """

        N = self.N
        e_cfg = self.e_cfg
        att = state.attack_dir[:, None]                     # (N,1) 관측자별 접힘 부호
        pos = state.player_pos
        vel = state.player_vel

        # ── 관측자 앵커 — 필드 경계·골대는 절대 좌표에만 있으므로 제거 불가 ──
        anchor_pos = pos / self.field_half * att
        anchor_vel = vel / e_cfg.norm_player_vel * att
        anchor_block = jnp.concatenate([anchor_pos, anchor_vel], axis=1)     # (N,4)

        # ── 슬롯 토큰 (관측자 i, 대상 j) ──
        def tok(values):
            """슬롯별 스칼라 (N,) → 전 관측자 공통 토큰 열 (N,N,1)."""

            return jnp.broadcast_to(
                jnp.asarray(values, jnp.float32)[None, :], (N, N))[:, :, None]

        rel_pos = (pos[None, :, :] - pos[:, None, :]) * att[:, :, None] / self.field_half
        # 상대 속도 — '지금 어디'가 아니라 '곧 어디'를 보게 해 패스 차단·마킹의 선점을 가능케 한다
        # (tactics.lane_completion·pressure). 절대 속도는 앵커와 더하면 나온다.
        rel_vel = (vel[None, :, :] - vel[:, None, :]) * att[:, :, None] / e_cfg.norm_player_vel
        # role_pos는 각 선수 자기 공격 프레임에 저장돼 있다. 월드로 편 뒤 관측자 프레임으로 다시
        # 접어 **모든 토큰을 한 프레임**에 둔다. 각자 프레임 그대로 두면 팀 부호를 보고
        # 해석해야 해서 같은 좌표가 토큰마다 다른 뜻이 된다.
        role_world = state.role_pos * state.attack_dir[:, None]
        role_tok = role_world[None, :, :] * att[:, :, None] / self.field_half
        # 규범 앵커도 각자 공격 프레임에 있다. role_pos와 **같은 규약**으로 월드에 편 뒤
        # 관측자 프레임으로 접는다 — 두 앵커가 다른 프레임에 있으면 비교가 불가능하다.
        formation_world = self.formation_home(state) * state.attack_dir[:, None]
        formation_tok = (
            formation_world[None, :, :] * att[:, :, None] / self.field_half)
        layout_norm = jnp.float32(self._layout_scale())
        layout_index_tok = (
            state.layout_index[state.team_id].astype(jnp.float32) / layout_norm)
        team_relation = jnp.where(
            state.team_id[None, :] == state.team_id[:, None], 1.0, -1.0)

        status_code = self._status_code(state)
        taker_mask = self._taker_mask(state)
        cooldown_n = state.cooldown / self._cooldown_norm_scale()
        contact_n = state.contact_lock_t / jnp.float32(e_cfg.contact_lock_substeps)
        aerial_recovery_n = (
            state.aerial_recovery_t
            / jnp.float32(e_cfg.aerial_attempt_lock_substeps)
        )
        ctrl_lock_n = state.ctrl_lock_t / jnp.float32(e_cfg.ctrl_lock_substeps)

        players_block = jnp.concatenate(
            [
                rel_pos,
                rel_vel,
                tok(state.stamina_short),
                tok(state.stamina_long),
                tok(state.endurance_factor),
                tok(cooldown_n),
                tok(contact_n),
                tok(aerial_recovery_n),
                tok(ctrl_lock_n),
                tok(state.yellow_cards),
                tok(state.offside_flag),
                role_tok,
                tok(self._role_gain(state)),
                formation_tok,
                tok(layout_index_tok),
                tok(status_code),
                tok(taker_mask),
                team_relation[:, :, None],
                tok(state.vmax / e_cfg.norm_player_vel),
                tok(state.player_ctrl),
                tok(state.reach_z / e_cfg.norm_body_z),
                tok(state.head_z / e_cfg.norm_body_z),
                tok(state.gk_indices),
            ], axis=-1
        )
        # 비활성 슬롯의 토큰은 0으로 지운다. 다만
        # status_code만은 마스크 뒤에 되살린다 — 교체로 빠진 슬롯과 퇴장한 슬롯이 같은
        # 0 토큰이 되면 팀 인원 판정이 관측에서 사라진다.
        token_layout = self._layout(OBS_PLAYER_FEATURES, 0)[0]
        status_col = token_layout["status_code"][0]
        player_size = sum(size for _, size in OBS_PLAYER_FEATURES)
        players_start = sum(size for _, size in OBS_ANCHOR_FEATURES)
        target_active = state.active_player[None, :].astype(jnp.float32)
        players_block = players_block * target_active[:, :, None]
        players_block = players_block.at[:, :, status_col].set(
            jnp.broadcast_to(status_code[None, :], (N, N)))
        players_block = players_block.reshape(N, -1)

        # ── 공 — x·y는 관측자 상대량, z는 지면 기준 절대량 ──
        ball_rel_pos = (state.ball_pos[None, :DIM_Z] - pos) * att / self.field_half
        ball_rel_vel = (state.ball_vel[None, :DIM_Z] - vel) * att / e_cfg.norm_ball_vel
        ball_z = jnp.full((N, 1), state.ball_pos[DIM_Z] / e_cfg.norm_ball_z)
        ball_vel_z = jnp.full((N, 1), state.ball_vel[DIM_Z] / e_cfg.norm_ball_vel)
        # 스핀도 속도와 같은 180° 규약(x·y만 반전, z 유지). 회전에 유사벡터 예외는 없다 —
        # 원본 거울 규약의 y·z 반전을 옮겨오면 반대편 팀의 휘는 방향이 뒤집힌다.
        ball_spin = jnp.broadcast_to(state.ball_spin, (N, DIM_ALL))
        ball_spin = ball_spin.at[:, :DIM_Z].multiply(att) / e_cfg.norm_spin
        ball_alive = jnp.full((N, 1), (state.ball_state == BALL_ALIVE), jnp.float32)
        poss_relation = self._team_relation(state, state.poss_team)[:, None]
        ball_block = jnp.concatenate(
            [ball_rel_pos, ball_z, ball_rel_vel, ball_vel_z,
             ball_spin, ball_alive, poss_relation], axis=-1
        )

        # ── 전역 맥락 ──
        rt_window = jnp.where(
            state.restart_kind == RK_PENALTY, e_cfg.penalty_substeps,
            jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps,
                      e_cfg.restart_substeps))
        # 재개 소유·종류는 게이트를 걸지 않고 원값을 낸다. ``restart_kind_code == RK_NONE``이
        # 활성 여부를 이미 말해 주므로, 소유를 0으로 가리면 정보만 사라진다.
        restart_owner = self._team_relation(state, state.restart_team)[:, None]
        restart_kind_code = jnp.full((N, 1), state.restart_kind, jnp.float32)
        restart_steps = jnp.full((N, 1), state.restart_t / rt_window, jnp.float32)
        # 간접 프리킥 래치는 공이 라이브가 된 뒤에도 두 번째 터치 전 직접골 판정을 바꾸므로
        # 재개 타이머로 가리지 않는다.
        restart_indirect = jnp.full((N, 1), state.restart_indirect, jnp.float32)
        departed_taker_mask = jnp.full(
            (N, 1), self._departed_taker_mask(state), jnp.float32)
        pass_owner = self._team_relation(state, state.pass_team)[:, None]
        offside_active = jnp.full(
            (N, 1), state.pass_t > 0, jnp.float32)
        time_left = jnp.full(
            (N, 1), jnp.clip(1.0 - state.t / jnp.float32(self.game_duration), 0.0, 1.0),
            jnp.float32)
        possession_window = jnp.float32(
            self.control_fps * POSSESSION_CONTEXT_SECONDS
        )
        possession_steps = jnp.full(
            (N, 1),
            jnp.clip(state.possession_t / possession_window, 0.0, 1.0),
            jnp.float32,
        )
        previous_possession_relation = self._team_relation(
            state, state.previous_poss_team
        )[:, None]
        # 마지막 터치의 **관계**. 팀 관계 {-1, 0, +1}에 '나 자신'을 +2로 얹는다.
        # 차원은 그대로이고 기존 세 값의 의미도 그대로다. 슬롯 번호를 노출하지 않으므로
        # identity 비노출 원칙과 관점 불변성이 유지된다 — 각 관측자가 "내가 방금 찼는가"만
        # 알 수 있다. 이게 없으면 정책이 공 진행축 투영으로 원 패서를 추정해야 하는데,
        # 그 추정은 원 패서를 막지도 못하고 도착한 수신자를 배제한다(실측 양방향 오류).
        last_touch_relation = self._team_relation(state, state.last_touch_team)
        was_self = (
            (state.last_touch_actor >= 0)
            & (self.player_indices == state.last_touch_actor)
        )
        last_touch_relation = jnp.where(
            was_self, jnp.float32(SELF_TOUCH_RELATION), last_touch_relation
        )[:, None]
        # 마지막 터치 '종류'는 팀 무관 범주값이라 관점 폴딩 없이 그대로 낸다. 골키퍼 손
        # 제한은 이 코드에서 재추론하지 않는다 — 목표 의도를 보존한 별도 causal relation이
        # 아래에 있어 INTERCEPT/TACKLE 백패스와 우연한 아군 굴절을 구분한다.
        last_touch_code = jnp.full((N, 1), state.last_touch_code, jnp.float32)
        handling_code = state.gk_handling_restricted_team
        handling_team = jnp.where(
            handling_code >= GK_HANDLING_RELEASE_OFFSET,
            handling_code - GK_HANDLING_RELEASE_OFFSET,
            handling_code,
        )
        handling_cause_scale = jnp.where(
            handling_code >= GK_HANDLING_RELEASE_OFFSET, 2.0, 1.0
        )
        # Signed categorical relation: ±1=team-mate back-pass/throw-in,
        # ±2=the goalkeeper's own release before another player touch, 0=none.
        # Encoding the cause in the existing scalar keeps the transition fully
        # Markov without widening every actor row again.
        gk_handling_restricted = jnp.where(
            handling_code >= 0,
            self._team_relation(state, handling_team) * handling_cause_scale,
            0.0,
        )[:, None]
        tid = state.team_id
        score_diff = ((state.score[tid] - state.score[1 - tid]) / e_cfg.norm_score)[:, None]
        # 후반 킥오프가 우리 것인가(±1). ``kickoff_team``은 **전반** 킥오프 팀이고 후반은 그
        # 반대다(events._halftime_switch). 이 비트가 없으면 두 상태의 관측이 완전히 같은데
        # 하프타임 전이 결과가 갈린다. 경기 내내 상수라 짧은 history로도 복원할 수 없다.
        sh_kickoff_ours = jnp.where(tid == (TEAM_1 - state.kickoff_team), 1.0, -1.0)[:, None]
        # 교체·포메이션 자원. 팀 인덱스가 아니라 **관측자 기준**으로 접어 관점 불변을 지킨다.
        squad = self.squad_resources(state)
        squad_cols = []
        for name in ("subs_remaining", "sub_windows_used", "sub_window_open",
                     "bench_available", "bench_gk_available", "layout_hold_elapsed"):
            value = squad[name]
            squad_cols.append(value[tid][:, None])
            squad_cols.append(value[1 - tid][:, None])
        context_block = jnp.concatenate(
            [restart_owner, restart_kind_code, restart_steps, restart_indirect,
             departed_taker_mask,
             pass_owner, offside_active, time_left,
             possession_steps, previous_possession_relation,
             last_touch_relation, last_touch_code,
             gk_handling_restricted, score_diff, sh_kickoff_ours,
             *squad_cols], axis=-1
        )

        # 벤치 자리별 토큰 — 우리 팀 자리 전부, 그다음 상대 팀. 팀 인덱스가 아니라 관측자
        # 기준으로 접어 관점 불변을 지킨다(자리 순서는 벤치 배열 순서 그대로다).
        if self.bench_size:
            seated_all = (state.bench_player_id >= 0).astype(jnp.float32)
            gk_all = state.bench_is_gk.astype(jnp.float32)
            # 벤치 role_pos도 **그 선수 팀의** 공격 접힘 프레임에 저장돼 있다. 경기장 안
            # ``role_pos``와 똑같이 (1) 대상 팀 부호로 월드에 펴고 (2) 관측자 부호로 접어야
            # 한다. 펴는 단계를 빠뜨리면 관점 대칭이 깨진다 — 실측으로 같은 앵커가 팀0
            # 관측자에게는 우리·상대 모두 (10,5), 팀1 관측자에게는 모두 (-10,-5)로 나왔다.
            team_dir = jnp.stack([
                jnp.sum(jnp.where(state.team_id == TEAM_0, state.attack_dir, 0.0))
                / jnp.maximum(jnp.sum(state.team_id == TEAM_0), 1),
                jnp.sum(jnp.where(state.team_id == TEAM_1, state.attack_dir, 0.0))
                / jnp.maximum(jnp.sum(state.team_id == TEAM_1), 1),
            ])
            anchor_all = (state.bench_role_pos * team_dir[:, None, None]
                          / self.field_half)

            speed_all = state.bench_vmax / e_cfg.norm_player_vel
            ctrl_all = state.bench_player_ctrl
            endurance_all = state.bench_endurance_factor

            def bench_side(team_index):
                seated = seated_all[team_index][:, :, None]
                gk = gk_all[team_index][:, :, None]
                anchor = anchor_all[team_index] * att[:, None, :]
                # ``team_index``는 관측자별 팀 배열 (N,)이라 인덱싱 결과가 이미 (N, 자리수)다.
                # 다른 열들과 같은 규약으로 특징 축만 붙인다.
                speed = speed_all[team_index][:, :, None]
                ctrl = ctrl_all[team_index][:, :, None]
                endurance = endurance_all[team_index][:, :, None]
                # 비어 있는 자리는 전부 0 — 없는 사람의 속성이 읽히면 안 된다.
                return jnp.concatenate(
                    [seated, gk * seated, anchor * seated,
                     speed * seated, ctrl * seated,
                     endurance * seated], axis=-1).reshape(N, -1)

            bench_block = jnp.concatenate(
                [bench_side(tid), bench_side(1 - tid)], axis=-1)
        else:
            bench_block = jnp.zeros((N, 0), jnp.float32)

        full = jnp.concatenate(
            [anchor_block, players_block, ball_block, context_block,
             bench_block], axis=-1)
        full = full * state.active_player[:, None].astype(jnp.float32)
        # 행 마스크에서도 자기 토큰의 status_code만은 되살린다. 이것이 없으면 퇴장한 관측자의
        # 자기 status가 0(INACTIVE)으로 읽혀 다른 선수 행에 실린 2(SENT_OFF)와 어긋나고,
        # 무엇보다 '이 행이 마스크됐는지'를 판정할 열이 사라져 row_validity가 서술로만 남는다.
        # 규약을 하나로 통일한다 — status_code는 토큰 마스크든 행 마스크든 항상 살아남는다.
        rows = jnp.arange(N)
        self_status = players_start + rows * player_size + status_col
        return full.at[rows, self_status].set(status_code)

    def _team_relation(self, state, team):
        """관측자 팀 기준 팀 부호 (N,) — 우리 +1 / 상대 -1 / 해당 없음 0.

        팀 인덱스를 그대로 노출하지 않으므로 관점 불변이고, ``NO_TEAM``을 0으로 접어 '소유 없음'과
        '어느 팀 소유'가 같은 값이 되지 않는다.
        """

        return jnp.where(
            team >= 0, jnp.where(state.team_id == team, 1.0, -1.0), 0.0
        ).astype(jnp.float32)

    def get_state(self, state):
        """중앙집중 크리틱용 전역 상태 벡터 — Causal-Compact 레이아웃.

        구성은 N개의 player token, ball, game, 양 팀 bench block이고 전부 절대 프레임
        (폴딩 없음)이다. 실제 폭과 열 위치는 :meth:`state_spec` 및
        :data:`constants.STATE_PLAYER_FEATURES`가 유일한 진실원천이다.
        다음 전이를 바꾸는 값만 싣는다는 것이 이 벡터의 계약이다. 그래서

        * ``face``는 빠졌다 — ``vel``과 ``attack_dir``의 결정함수다(movement.facing_from_velocity).
        * pending/setpiece/throw 세 마스크는 ``taker_mask`` 비트마스크 하나로 접었다.
        * ``on_pitch`` 한 비트는 ``status_code``로 바뀌어 교체 아웃과 퇴장을 구분한다.
        * 팀·재개·터치 one-hot은 categorical 코드다(역변환이 자명해 정보가 같고 폭만 3~11배였다).
        * ``role_gain``이 새로 들어간다 — 이것 없이 ``role_pos``만 실으면 같은 벡터에서
          다음 ``role_pos``가 갈려 Markov 상태가 아니다.
        * 비참여 슬롯은 :data:`constants.INACTIVE_SLOT_LIVE_COLUMNS`만 남기고 마스킹한다.
          그 슬롯의 stamina·카드·role은 이후 어떤 전이도 바꾸지 않는다 — 교체 투입은
          ``Substitution`` 행의 값을 쓰고 퇴장은 되돌릴 수 없다. 남겨 두면 보상·전이가
          똑같은 두 상태가 서로 다른 벡터가 되어 "전이를 바꾸는 값만"이라는 계약이 깨진다.

        파생 affordance는 여기 없다. :meth:`affordance_view`가 재계산한다.
        정확한 레이아웃은 :meth:`state_spec`이 단일 진실원천이다.
        """

        e_cfg = self.e_cfg

        # ── players, 절대 프레임 ──
        # role_pos는 각 선수의 공격 프레임에 저장되므로 현재 attack_dir로 월드로 되돌린다.
        player_columns = jnp.concatenate(
            [
                state.player_pos / self.field_half,
                state.player_vel / e_cfg.norm_player_vel,
                state.stamina_short[:, None],
                state.stamina_long[:, None],
                state.endurance_factor[:, None],
                (state.cooldown / self._cooldown_norm_scale())[:, None],
                (state.contact_lock_t / jnp.float32(e_cfg.contact_lock_substeps))[:, None],
                (state.aerial_recovery_t
                 / jnp.float32(e_cfg.aerial_attempt_lock_substeps))[:, None],
                (state.ctrl_lock_t / jnp.float32(e_cfg.ctrl_lock_substeps))[:, None],
                state.yellow_cards.astype(jnp.float32)[:, None],
                state.offside_flag.astype(jnp.float32)[:, None],
                state.role_pos * state.attack_dir[:, None] / self.field_half,
                self._role_gain(state)[:, None],
                (self.formation_home(state) * state.attack_dir[:, None]
                 / self.field_half),
                (state.layout_index[state.team_id].astype(jnp.float32)
                 / jnp.float32(self._layout_scale()))[:, None],
                self._status_code(state)[:, None],
                self._taker_mask(state)[:, None],
                (state.vmax / e_cfg.norm_player_vel)[:, None],
                state.player_ctrl[:, None],
                (state.reach_z / e_cfg.norm_body_z)[:, None],
                (state.head_z / e_cfg.norm_body_z)[:, None],
                state.team_id.astype(jnp.float32)[:, None],
                state.gk_indices.astype(jnp.float32)[:, None],
            ], axis=1
        )
        # 참여 슬롯은 그대로, 비참여 슬롯은 live 열만 남긴다. status_code/team_id는 팀 인원
        # 하한과 슬롯→팀 대응을 보존한다. is_gk는 남기지 않는다 — 교체는 on_pitch 슬롯에만
        # 들어오고 퇴장 슬롯은 영구 비참여라, 비참여 슬롯의 GK 비트는 이후 전이에 쓰이지 않는다.
        layout = self._layout(STATE_PLAYER_FEATURES, 0)[0]
        keep = np.zeros(player_columns.shape[1], bool)
        for name in INACTIVE_SLOT_LIVE_COLUMNS:
            lo, hi = layout[name]
            keep[lo:hi] = True
        alive = state.active_player[:, None] | jnp.asarray(keep)[None, :]
        players_block = jnp.where(alive, player_columns, 0.0).reshape(-1)

        # ── ball (9), 절대 프레임 ──
        ball_block = jnp.concatenate(
            [
                state.ball_pos / jnp.array([self.hx, self.hy, e_cfg.norm_ball_z]),
                state.ball_vel / e_cfg.norm_ball_vel,
                state.ball_spin / e_cfg.norm_spin,
            ]
        )

        # ── game (19), 전역. ``*_code``는 원값 categorical이다 ──
        rt_window = jnp.where(
            state.restart_kind == RK_PENALTY, e_cfg.penalty_substeps,
            jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps,
                      e_cfg.restart_substeps))
        game_scalars = jnp.stack(
            [
                state.poss_team.astype(jnp.float32),
                jnp.clip(
                    state.possession_t
                    / jnp.float32(
                        self.control_fps * POSSESSION_CONTEXT_SECONDS
                    ),
                    0.0,
                    1.0,
                ),
                state.previous_poss_team.astype(jnp.float32),
                state.last_touch_team.astype(jnp.float32),
                state.gk_handling_restricted_team.astype(jnp.float32),
                state.attack_dir[0],                 # 팀0 공격방향 = 절대 프레임 해석 기준
                (state.ball_state == BALL_ALIVE).astype(jnp.float32),
                (state.restart_t / rt_window).astype(jnp.float32),
                (state.pass_t > 0).astype(jnp.float32),
                jnp.clip(1.0 - state.t / jnp.float32(self.game_duration), 0.0, 1.0),
                state.restart_indirect.astype(jnp.float32),
                self._departed_taker_mask(state),
                state.restart_team.astype(jnp.float32),
                state.pass_team.astype(jnp.float32),
            ]
        )
        game_tail = jnp.stack(
            [
                state.restart_kind.astype(jnp.float32),
                state.last_touch_code.astype(jnp.float32),
                state.kickoff_team.astype(jnp.float32),
            ]
        )
        score_n = state.score.astype(jnp.float32) / e_cfg.norm_score
        # obs와 **같은 함수**에서 만든다. 두 경로가 각자 계산하면 갈릴 수 있다.
        squad = self.squad_resources(state)
        e_cfg = self.e_cfg
        game_squad = jnp.concatenate([
            squad[name] for name in (
                "subs_remaining", "sub_windows_used", "sub_window_open",
                "bench_available", "bench_gk_available", "layout_hold_elapsed")
        ])
        # 자리별 벤치 프로필. obs만 고치면 중앙 크리틱이 벤치를 못 본다.
        if self.bench_size:
            bench_state = jnp.concatenate([
                (state.bench_player_id >= 0).astype(jnp.float32)[:, :, None],
                state.bench_is_gk.astype(jnp.float32)[:, :, None],
                state.bench_role_pos / self.field_half,
                (state.bench_vmax / e_cfg.norm_player_vel)[:, :, None],
                state.bench_player_ctrl[:, :, None],
                state.bench_endurance_factor[:, :, None],
            ], axis=-1).reshape(-1)
        else:
            bench_state = jnp.zeros((0,), jnp.float32)
        game_block = jnp.concatenate(
            [game_scalars, score_n, game_tail, game_squad, bench_state])

        return jnp.concatenate([players_block, ball_block, game_block])

    def affordance_view(self, state):
        """저장하지 않는 파생 affordance — 상태와 config만으로 결정론적으로 재계산한다.

        여기 있는 값은 전부 compact 벡터의 원천 상태에서 다시 만들 수 있다. 그래서 관측에
        굳혀 두지 않는다: 데이터셋이 파생량을 원천처럼 저장하면 나중에 규칙이 바뀔 때
        저장된 값과 재계산 값이 조용히 갈리고, 어느 쪽이 진실인지 알 방법이 없어진다.

        정책에는 compact 벡터와 함께 이 view를 넘긴다. 관측만 받는 학습 정책도 배포 시
        같은 함수를 호출하면 되므로 teacher/student가 보는 신호는 동일하다.

        반환 dict(모두 float32 (N,)): :data:`constants.AFFORDANCE_FEATURES`의 이름과 순서.
        """

        N = self.N
        e_cfg = self.e_cfg
        ar = self.player_indices

        in_reach, _ = self._in_reach(state)
        # ``f2b_avail`` is exact *current-substep* readiness and preserves the
        # useful distinction from control-frame action potential.  The latter
        # is ``~action_agency()["kick_gated"]`` / public get_avail: movement
        # can enter reach, timers can expire, and another player can clear a
        # retouch latch before a later opportunity in this same frame.
        agency = self.action_agency(state)
        f2b_avail = (in_reach & state.active_player
                     & (state.ball_state == BALL_ALIVE)
                     & self._cooldown_allows_f2b(state)
                     & self._contact_lock_allows_f2b(state)
                     & self._aerial_recovery_allows_f2b(state)
                     & self._ctrl_lock_allows_f2b(state)
                     & self._retouch_allows_f2b(state))

        # 팀내 공거리 랭크(0=최근접, 1=최원거리) — 동률 모호로 인한 방관자 교착을 깨는
        # **서술적** 좌표다. 로스터가 비대칭일 수 있으므로 11인 경계로 자르지 않고 실제 팀으로
        # 나눈다. 순위 집합과 분모 **둘 다** 활성 인원 기준이어야 한다. 분모만 전체 슬롯 수로
        # 두면 퇴장이 나온 팀에서 최원거리 선수의 랭크가 1.0에 못 미쳐(10인이면 0.9) 관측에서
        # 재계산한 값과 어긋난다. 비참여 슬롯의 랭크는 정의되지 않으므로 0으로 지운다.
        active = state.active_player
        ball_rank = _team_ball_rank(
            state.player_pos,
            state.ball_pos,
            active,
            state.team_id,
        )

        # 팀별 2번째 최종수비 라인(오프사이드 기준선)을 각 선수에게 배분한다.
        adir_team = state.attack_dir[self.team_indices]
        rule_pos = self._player_pos_for_field_rules(state.player_pos)
        x_att = rule_pos[None, :, DIM_X] * adir_team[:, None]                # (2,N)
        opp = ((state.team_id[None, :] != jnp.arange(TEAM_COUNT)[:, None])
               & state.active_player[None, :])
        x_opp = jnp.where(opp, x_att, -jnp.inf)
        line2 = jax.lax.top_k(x_opp, 2)[0][:, 1]
        line2 = jnp.where(opp.sum(axis=1) >= 2, line2, self.hx)
        off_line = jnp.clip(line2 / self.hx, -1.0, 1.0)[state.team_id.astype(jnp.int32)]

        restart_active = restart_timer_active(state.restart_t)
        kicker_locked_s, _, _, sp_active = self._setpiece_kick_lock(state)
        is_pending = ar == state.pending_taker
        kicker_locked = jnp.where(sp_active & kicker_locked_s & is_pending, 1.0, 0.0)
        # This is frame-level release readiness, not entry-substep geometry.
        # It is deliberately identical to agency.kick_forced so policy kick
        # parameters are present whenever runtime will consume the restart.
        kicker_ready = agency["kick_forced"].astype(jnp.float32)

        clear_r, encroachers, enc_m = self._encroach_geometry(state)
        is_pen = state.restart_kind == RK_PENALTY
        is_ko = state.restart_kind == RK_KICKOFF
        is_def = state.team_id != state.restart_team
        # 페널티 subject는 키커를 뺀 전원이다. 수비 GK도 골라인 밴드를 지켜야 하므로 포함한다.
        # 현재 위반자(encroachers)를 subject로 쓰면 경계 밖으로 나가는 순간 margin이 0이 되어
        # '밖에서 대기하라'는 신호가 사라지고 경계 재진입 진동이 생긴다.
        safe_pending = jnp.clip(state.pending_taker, 0, N - 1)
        valid_pending = (
            (state.pending_taker >= 0)
            & (state.pending_taker < N)
            & state.active_player[safe_pending]
            & (state.team_id[safe_pending] == state.restart_team)
        )
        is_kicker_m = is_pending & valid_pending
        is_def_gk_m = (state.gk_indices == 1) & (state.team_id != state.restart_team)
        # 비참여 슬롯은 어떤 재개의 거리 의무 주체도 아니다. penalty 분기만 active를 걸고
        # 일반 재개는 is_def만 쓰면 벤치 선수가 공 근처에 남아 있을 때 enc_margin<0인데
        # any_encroacher=0인 모순 affordance가 생긴다.
        subject = state.active_player & jnp.where(
            is_pen | is_ko, ~is_kicker_m, is_def
        )
        enc_scale = jnp.where(
            is_pen & is_def_gk_m,
            jnp.maximum(e_cfg.goal_line_tolerance, DIV_EPS),
            clear_r,
        )
        enc_margin = jnp.where(
            restart_active & subject, jnp.clip(enc_m / enc_scale, -1, 1), 0.0)
        any_encroacher = jnp.broadcast_to(
            jnp.where(restart_active & jnp.any(encroachers), 1.0, 0.0), (N,))

        pending, setpiece_bit, throw_bit = self.taker_bits(self._taker_mask(state))
        is_taker = (pending | setpiece_bit | throw_bit).astype(jnp.float32)
        pass_signal = jnp.where(
            (state.pass_t > 0) & (state.pass_team >= 0),
            self._team_relation(state, state.pass_team), 0.0)
        is_sp_ours = jnp.where(
            restart_active, self._team_relation(state, state.restart_team), 0.0)

        return {
            "in_reach": in_reach.astype(jnp.float32),
            "f2b_avail": f2b_avail.astype(jnp.float32),
            "ball_rank": ball_rank,
            "off_line": off_line,
            "enc_margin": enc_margin,
            "any_encroacher": any_encroacher,
            "kicker_locked": kicker_locked,
            "kicker_ready": kicker_ready,
            "is_taker": is_taker,
            "pass_signal": pass_signal,
            "is_sp_ours": is_sp_ours,
            "move_forced": agency["move_forced"].astype(jnp.float32),
            "kick_gated": agency["kick_gated"].astype(jnp.float32),
            "kick_forced": agency["kick_forced"].astype(jnp.float32),
        }

    def affordance_array(self, state):
        """:meth:`affordance_view`를 (N, len(AFFORDANCE_FEATURES)) 배열로 편 것."""

        view = self.affordance_view(state)
        return jnp.stack([view[name] for name, _ in AFFORDANCE_FEATURES], axis=1)

    def get_obs(self, state):
        """JaxMARL 규약 dict 어댑터 — 계산은 get_obs_array(단일 진실원천)."""
        full = self.get_obs_array(state)
        return {a: full[i] for i, a in enumerate(self._agent_keys)}

    def get_avail_actions_array(self, state):
        """행동 마스크 (N, 2) = [MOVE, F2B].

        실제 전이에서 쓰는 :meth:`action_agency`를 그대로 투영한다. 따라서 데드볼 내장 엔진이
        전원 이동을 덮어쓰는 프레임도 MOVE=0이고, 강제 세트피스 킥의 파라미터는 F2B=1이다.
        F2B는 국면 단위 제어 채널 권한이다. 현재 서브스텝의 도달·타이머·재터치
        합법성은 :meth:`affordance_view`의 ``f2b_avail``이 더 엄격하게 나타낸다.
        """
        agency = self.action_agency(state)
        return jnp.stack(
            [~agency["move_forced"], ~agency["kick_gated"]], axis=1
        ).astype(jnp.float32)

    def get_avail_actions(self, state):
        """JaxMARL 규약 dict 어댑터 — 계산은 get_avail_actions_array(단일 진실원천)."""
        mask = self.get_avail_actions_array(state)
        return {a: mask[i] for i, a in enumerate(self._agent_keys)}

    # ── 열 인코딩 메타데이터 ────────────────────────────────────────────────
    # slice만으로는 소비자가 열을 잘못 쓴다. 특히 팀 코드는 ``NO_TEAM = -1``이라 그대로
    # embedding index에 넣으면 예외가 나거나 마지막 category를 가리키고, ``*_steps``는 이름과
    # 달리 raw step이 아니라 [0,1] 비율이다. 그래서 각 열이 무엇인지 기계가 읽을 수 있게 낸다.
    @staticmethod
    def _categorical(cardinality, offset=0, values=None):
        """정수 범주 — ``embedding_index = round(value) + offset``."""

        spec = {"kind": "categorical", "cardinality": int(cardinality), "offset": int(offset)}
        if values is not None:
            spec["values"] = values
        return spec

    @staticmethod
    def _departed_taker_encoding():
        """Encoding for actor-free throw/set-piece provenance."""

        return {
            "kind": "bitmask",
            "cardinality": DEPARTED_TAKER_MASK_COUNT,
            "bits": ("throw", "setpiece"),
            "weights": (
                DEPARTED_TAKER_BIT_THROW,
                DEPARTED_TAKER_BIT_SETPIECE,
            ),
        }

    @staticmethod
    def _ratio(normalizer, raw_unit, note=None):
        """``value = raw / normalizer``인 [0,1] 비율. 이름이 raw 단위처럼 보여도 raw가 아니다."""

        spec = {"kind": "ratio", "normalizer": normalizer, "raw_unit": raw_unit,
                "range": (0.0, 1.0)}
        if note is not None:
            spec["note"] = note
        return spec

    @staticmethod
    def _scaled(normalizer, raw_unit, note=None):
        """``value = raw / normalizer``인 무계 연속량 — 엄격 클립이 아니다(README §행동과 관측)."""

        spec = {"kind": "scaled", "normalizer": normalizer, "raw_unit": raw_unit}
        if note is not None:
            spec["note"] = note
        return spec

    # 아래 넷은 상수 dict가 아니라 **팩토리**다. 모듈 수준 dict를 공유하면 한 열의 메타데이터를
    # 고친 것이 같은 kind를 쓰는 다른 모든 열과 이후 spec 호출까지 오염시킨다(예: ball_alive를
    # 건드리면 is_gk와 다음 obs_spec()이 함께 바뀐다). spec은 호출자에게 넘기는 값이므로
    # 매번 새 객체여야 한다.
    def _squad_spec(self, observation):
        """교체·포메이션 자원 열의 스펙. obs는 관측자 기준, state는 팀 절대 인덱스다."""

        unit = lambda note: {"kind": "unit", "range": (0.0, 1.0), "note": note}
        base = {
            "subs_remaining": unit("잔여 교체 인원 / 상한"),
            "sub_windows_used": unit("쓴 교체 기회 / 3"),
            "sub_window_open": self._binary(),
            "bench_available": unit("투입 가능한 필드 선수 / 벤치 크기"),
            "bench_gk_available": self._binary(),
            "layout_hold_elapsed": unit(
                "마지막 승인 포메이션 명령 이후 경과시간 / 경기 길이"
            ),
        }
        if not observation:
            return base
        out = {}
        for name, spec in base.items():
            out[f"{name}_ours"] = dict(spec)
            out[f"{name}_theirs"] = dict(spec)
        return out

    def squad_resources(self, state):
        """교체·포메이션 **자원**을 팀별로 (2,) 배열 여섯 개로 편다.

        obs와 중앙 state가 같은 값을 봐야 한다 — 한쪽만 고치면 다른 쪽이 여전히 Markov가
        아니다. 그래서 두 경로가 이 함수 하나를 공유한다.

        전부 다음 전이를 바꾸는 값이다. 승인 층이 잔여 인원·기회를 보고, 결정자가 창이
        열려 있는지와 벤치에 누가 남았는지를 본다. 관측에 없으면 같은 obs에서 다음 상태가
        갈린다.
        """

        subs_cap = jnp.float32(max(int(self.max_substitutions), 1))
        window_cap = jnp.float32(IFAB_MAX_SUBSTITUTION_WINDOWS)
        bench_cap = jnp.float32(max(int(self.bench_size), 1))
        seated = state.bench_player_id >= 0
        duration = jnp.float32(max(int(self.game_duration), 1))
        return {
            "subs_remaining": state.subs_remaining.astype(jnp.float32) / subs_cap,
            "sub_windows_used": (
                state.sub_windows_used.astype(jnp.float32) / window_cap),
            # 열림 여부만 나간다. 열린 **시각**을 그대로 실으면 경기 시계와 중복이고,
            # 결정자가 보는 것도 "이번 정지에서 이미 바꿨는가" 하나다.
            "sub_window_open": (state.sub_window_open_t >= 0).astype(jnp.float32),
            "bench_available": (
                jnp.sum(seated & (~state.bench_is_gk), axis=1).astype(jnp.float32)
                / bench_cap),
            "bench_gk_available": (
                jnp.any(seated & state.bench_is_gk, axis=1).astype(jnp.float32)),
            # 마지막 승인 명령 이후 경과. 정확한 정의는
            # ``clip((t - layout_since_t) / game_duration, 0, 1)``이다. 규칙이 보는 것은
            # "충분히 지났는가"이므로 절대 tick 대신 경기 길이 비율을 싣는다. 교체는
            # layout_since_t를 쓰지 않으므로 이 신호를 reset하지 않는다.
            "layout_hold_elapsed": jnp.clip(
                (state.t - state.layout_since_t).astype(jnp.float32) / duration,
                0.0, 1.0),
        }

    def _layout_count(self):
        """레이아웃 표의 크기 — 스펙 메타데이터가 보고하는 값."""

        return len(formation_module.LAYOUTS)

    @staticmethod
    def _layout_scale():
        """``layout_index``를 관측에 실을 때의 **고정** 분모.

        표 크기로 나누면 모양을 하나 더할 때마다 기존 레이아웃의 관측값이 전부 바뀐다.
        고정 용량으로 나누면 표가 자라도 기존 값이 불변이다
        (:data:`formation.LAYOUT_INDEX_CAPACITY` 참고).
        """

        return float(formation_module.LAYOUT_INDEX_CAPACITY)

    @staticmethod
    def _binary():
        return {"kind": "binary", "values": (0.0, 1.0)}

    @staticmethod
    def _sign():
        return {"kind": "sign", "values": (-1.0, 1.0)}

    @staticmethod
    def _relation():
        return {"kind": "relation", "values": (-1.0, 0.0, 1.0),
                "note": "관측자 팀 기준 우리 +1 / 상대 -1 / 해당 없음 0"}

    @staticmethod
    def _unit():
        return {"kind": "unit", "range": (0.0, 1.0)}

    @staticmethod
    def _endurance_factor_encoding():
        """Raw player multiplier; decision-time clipping is a policy concern."""

        return {
            "kind": "bounded_multiplier",
            "range": (ENDURANCE_FACTOR_MIN, ENDURANCE_FACTOR_MAX),
            "reference": ENDURANCE_FACTOR_REFERENCE,
            "note": "원시 지구력 계수. 1.0이 기준이며 관측 단계에서 정규화하지 않는다",
        }

    def _team_code(self):
        return self._categorical(
            TEAM_COUNT + 1, offset=1, values="NO_TEAM=-1, TEAM_0=0, TEAM_1=1")

    def _gk_handling_code(self):
        return self._categorical(
            GK_HANDLING_RESTRICTION_COUNT + 1,
            offset=1,
            values=(
                "NONE=-1, BACKPASS_TEAM_0=0, BACKPASS_TEAM_1=1, "
                "OWN_RELEASE_TEAM_0=2, OWN_RELEASE_TEAM_1=3"
            ),
        )

    @staticmethod
    def _gk_handling_relation_encoding():
        return {
            "kind": "signed_categorical_relation",
            "values": (-2.0, -1.0, 0.0, 1.0, 2.0),
            "note": (
                "관측자 기준 ±1=동료의 의도적 백패스/스로인, "
                "±2=GK 손 배급 뒤 타인 미접촉, 부호는 우리 + / 상대 -"
            ),
        }

    def _restart_steps_encoding(self):
        """재개 진행도 — 분모가 재개 종류에 따라 갈리므로 그 분기를 **구조화해서** 낸다.

        ``normalizer``를 "kind-dependent" 같은 문자열로 두면 사람만 읽을 수 있고 소비자는
        원래 substep 수를 복원할 수 없다. selector/cases/default 형태면 기계가
        ``raw = value * resolve(restart_kind_code)``로 정확히 되돌린다.

        분기는 **dict key가 아니라 리스트**로 낸다. JSON은 객체 key를 문자열로 강제하므로
        ``{6: 486.0}``이 왕복 뒤 ``{"6": 486.0}``이 되고, 정수로 조회하던 소비자는 조용히
        default로 떨어진다(penalty 486 → 450, GK hold 720 → 450). 리스트 안의 정수는 왕복해도
        정수 그대로다.
        """

        e_cfg = self.e_cfg
        return self._ratio(
            {
                "selector": "restart_kind_code",
                "cases": [
                    {"when": int(RK_PENALTY), "name": "RK_PENALTY",
                     "value": float(e_cfg.penalty_substeps)},
                    {"when": int(RK_GK_HOLD), "name": "RK_GK_HOLD",
                     "value": float(e_cfg.gk_hold_substeps)},
                ],
                "default": float(e_cfg.restart_substeps),
                "resolve": "cases에서 when == selector값인 항목의 value, 없으면 default",
            },
            "physics substep",
            "분모가 재개 종류별로 다르다 — 같은 0.5가 종류마다 다른 절대 시간이다",
        )

    def _player_encoding(self, *, observation):
        """선수 토큰 열 인코딩. 관측 토큰과 중앙 슬롯 토큰이 공유하는 부분이 대부분이다."""

        e_cfg = self.e_cfg
        # 관측 토큰은 비참여 슬롯에서 0으로 지워지므로 role_gain=0이 나온다. 역변환
        # ``1/value - 1``은 거기서 0으로 나누기가 된다 — 마스크 센티널임을 명시해야 한다.
        # 역변환은 반드시 반올림을 포함한다. float32에서 1/(1+n)을 저장했다 되돌리면
        # n=6이 5.999999687로 나오므로, round 없이 쓰면 정수 카운트가 하나씩 어긋난다.
        role_gain = {
            "kind": "reciprocal_count", "range": (0.0, 1.0),
            "inverse": "sample_count = round(1 / value - 1)",
            "inverse_domain": "value > 0",
            "inverse_is_integer": True,
            "inverse_exact_max": ROLE_GAIN_EXACT_MAX_SAMPLES,
            "note": "누적평균 갱신 이득. 0에 가까울수록 표본이 많다",
        }
        if observation:
            role_gain["masked_value"] = 0.0
            role_gain["note"] += "; 0은 값이 아니라 비참여 슬롯 마스크 센티널이다"
        shared = {
            "stamina_short": self._unit(),
            "stamina_long": self._unit(),
            "endurance_factor": self._endurance_factor_encoding(),
            "cooldown": self._ratio(self._cooldown_norm_scale(), "physics substep"),
            "contact_lock": self._ratio(float(e_cfg.contact_lock_substeps), "physics substep"),
            "aerial_recovery": self._ratio(
                float(e_cfg.aerial_attempt_lock_substeps), "physics substep"
            ),
            "ctrl_lock": self._ratio(float(e_cfg.ctrl_lock_substeps), "physics substep"),
            "yellow": {"kind": "count", "range": (0, YELLOW_CARD_SEND_OFF_COUNT)},
            "offside_latch": self._binary(),
            "role_gain": role_gain,
            "status_code": self._categorical(
                SLOT_STATUS_COUNT, values="INACTIVE=0, ACTIVE=1, SENT_OFF=2"),
            "taker_mask": {"kind": "bitmask", "cardinality": TAKER_MASK_COUNT,
                           "bits": ("pending", "setpiece_retouch", "throw_retouch"),
                           "weights": (TAKER_BIT_PENDING, TAKER_BIT_SETPIECE, TAKER_BIT_THROW)},
            "vmax": self._scaled(float(e_cfg.norm_player_vel), "m/s"),
            "ball_ctrl": self._unit(),
            "reach_z": self._scaled(float(e_cfg.norm_body_z), "m"),
            "head_z": self._scaled(float(e_cfg.norm_body_z), "m"),
            "is_gk": self._binary(),
        }
        half = [float(self.hx), float(self.hy)]
        layout_max = self._layout_scale()
        shared.update({
            "formation_home": self._scaled(half, "m", "규범 앵커 — 어디에 서야 하나"),
            "layout_index": self._scaled(layout_max, "layout", "현재 명령된 목표 모양"),
        })
        if observation:
            shared.update({
                "rel_pos": self._scaled(half, "m", "앵커와 같은 정규화 — abs = anchor + rel"),
                "rel_vel": self._scaled(float(e_cfg.norm_player_vel), "m/s",
                                        "앵커와 같은 정규화 — abs = anchor + rel"),
                "role_pos": self._scaled(
                    half, "m", "현재 전술 epoch 누적평균; 관측자 공격 프레임"),
                "team_relation": self._relation(),
            })
        else:
            shared.update({
                "pos": self._scaled(half, "m"),
                "vel": self._scaled(float(e_cfg.norm_player_vel), "m/s"),
                "role_pos": self._scaled(
                    half, "m", "현재 전술 epoch 누적평균; 절대 프레임(attack_dir로 펼침)"),
                "team_id": self._categorical(TEAM_COUNT, values="TEAM_0=0, TEAM_1=1"),
            })
        return shared

    @staticmethod
    def _layout(features, start=0):
        """(이름, 크기) 리스트 → {이름: (start, end)} 슬라이스 맵과 다음 오프셋. 오프셋 수동계산 방지용."""
        spec, offset = {}, start
        for name, size in features:
            spec[name] = (offset, offset + size)
            offset += size
        return spec, offset

    def obs_spec(self):
        """get_obs_array (N, obs_dim) 레이아웃의 단일 진실원천. 트레이너가 인덱스 하드코딩 없이
        블록·특징 슬라이스를 읽도록 {블록: {start,end,features,...}} 반환. get_obs_array와 순서 일치.

        ``players``는 관측자 자신을 포함한 **전 슬롯**이며 슬롯 인덱스 순서다. 즉 관측자 i의
        토큰은 ``players.start + i * players.size``에서 시작하고 rel_pos/rel_vel이 0이다.
        """

        e_cfg = self.e_cfg
        anchor_spec, a1 = self._layout(OBS_ANCHOR_FEATURES, 0)
        player_size = sum(s for _, s in OBS_PLAYER_FEATURES)
        players_end = a1 + self.N * player_size
        status_col = self._layout(OBS_PLAYER_FEATURES, 0)[0]["status_code"][0]
        ball_spec, b1 = self._layout(OBS_BALL_FEATURES, players_end)
        ctx_spec, c1 = self._layout(OBS_CONTEXT_FEATURES, b1)
        bench_size = sum(s for _, s in OBS_BENCH_FEATURES)
        bench_end = c1 + TEAM_COUNT * self.bench_size * bench_size
        half = [float(self.hx), float(self.hy)]
        return {
            "dim": bench_end,
            # 관측자 자신이 비참여면 그 행 전체가 0이다. 판정은 서술이 아니라 **읽을 수 있는
            # 열**로 준다 — 자기 토큰의 status_code는 행 마스크를 뚫고 살아남으므로, 그 값
            # 하나로 이 행이 마스크됐는지와 결장 종류(교체 아웃/퇴장)를 함께 알 수 있다.
            "row_validity": {"column": "players[self].status_code",
                             "offset": a1 + status_col,
                             "stride": player_size,
                             "valid_when": SLOT_ACTIVE,
                             "masked_value": 0.0,
                             "note": "행 i의 판정 열 = offset + i * stride. 이 값이 "
                                     "SLOT_ACTIVE가 아니면 그 행의 나머지는 전부 0이다"},
            "anchor": {"start": 0, "end": a1, "features": anchor_spec,
                       "encoding": {
                           "abs_pos": self._scaled(half, "m"),
                           "abs_vel": self._scaled(float(e_cfg.norm_player_vel), "m/s")}},
            "players": {"start": a1, "end": players_end, "count": self.N,
                        "size": player_size,
                        "features": self._layout(OBS_PLAYER_FEATURES, 0)[0],
                        "encoding": self._player_encoding(observation=True),
                        # 토큰 단위 유효성 — status_code만 마스크를 뚫고 살아남는다.
                        "validity": {
                            "column": "status_code",
                            "valid_when": SLOT_ACTIVE,
                            "masked_value": 0.0,
                            "always_valid": ("status_code",),
                            "note": "status_code != ACTIVE인 슬롯은 나머지 열이 전부 0으로 "
                                    "지워진다. 그 0은 관측값이 아니라 마스크이므로 역변환·"
                                    "통계에 넣으면 안 된다(특히 role_gain=0)",
                        }},
            "ball": {"start": players_end, "end": b1, "features": ball_spec,
                     "encoding": {
                         "rel_pos": self._scaled(half, "m", "abs = anchor_pos + rel_pos"),
                         "abs_z": self._scaled(float(e_cfg.norm_ball_z), "m"),
                         "rel_vel": self._scaled(
                             float(e_cfg.norm_ball_vel), "m/s",
                             "정규화 상수가 앵커와 달라 abs = anchor_vel*norm_player_vel"
                             " + rel_vel*norm_ball_vel"),
                         "abs_vel_z": self._scaled(float(e_cfg.norm_ball_vel), "m/s"),
                         "abs_spin": self._scaled(float(e_cfg.norm_spin), "rad/s"),
                         "ball_alive": self._binary(),
                         "possession_relation": self._relation()}},
            "context": {"start": b1, "end": c1, "features": ctx_spec,
                        "encoding": {
                            "restart_owner_relation": self._relation(),
                            "restart_kind_code": self._categorical(
                                RESTART_COUNT, values="RK_NONE=0 … RK_GK_HOLD=8"),
                            "restart_steps": self._restart_steps_encoding(),
                            "restart_indirect": self._binary(),
                            "departed_taker_mask": self._departed_taker_encoding(),
                            "pass_owner_relation": self._relation(),
                            "offside_active": self._binary(),
                            "time_left": self._ratio(
                                float(self.game_duration), "control step"),
                            "possession_steps": self._ratio(
                                float(
                                    self.control_fps
                                    * POSSESSION_CONTEXT_SECONDS
                                ),
                                "control step",
                                "현재 소유 라벨이 유지된 시간; 5초에서 clip",
                            ),
                            "previous_possession_relation": self._relation(),
                            "last_touch_relation": {
                                "kind": "relation",
                                "values": (-1.0, 0.0, 1.0, SELF_TOUCH_RELATION),
                                "note": ("관측자 기준 우리 +1 / 상대 -1 / 없음 0 / "
                                         "내가 마지막으로 찼으면 +2"),
                            },
                            "last_touch_code": self._categorical(
                                TOUCH_COUNT, values="TOUCH_NONE=0 … TOUCH_BODY_TRAP=11"),
                            "gk_handling_restricted_relation":
                                self._gk_handling_relation_encoding(),
                            "score_diff": self._scaled(float(e_cfg.norm_score), "goal"),
                            "second_half_kickoff_ours": self._sign(),
                            **self._squad_spec(True)}},
            "bench": {
                "start": c1, "end": bench_end,
                "slots": int(self.bench_size),
                "size": bench_size,
                "order": "우리 팀 자리 전부, 그다음 상대 팀 자리 전부",
                "features": self._layout(OBS_BENCH_FEATURES, 0)[0],
                "encoding": {
                    "available": self._binary(),
                    "is_gk": self._binary(),
                    "role_pos": self._scaled(half, "m", "관측자 공격 프레임"),
                    "vmax": self._scaled(float(e_cfg.norm_player_vel), "m/s"),
                    "ball_ctrl": self._unit(),
                    "endurance_factor": self._endurance_factor_encoding(),
                },
                "note": ("인원수만으로는 ``bench_index`` 제안의 결과가 갈린다 — "
                         "자리별 내용이 있어야 Markov가 성립하고 like-for-like "
                         "대체가 표현된다. 신원(player_id)은 싣지 않는다."),
            },
        }

    def state_spec(self):
        """get_state (state_dim,) 레이아웃의 단일 진실원천. get_state와 순서 일치."""
        e_cfg = self.e_cfg
        player_size = sum(s for _, s in STATE_PLAYER_FEATURES)
        players_end = self.N * player_size
        ball_spec, b1 = self._layout(STATE_BALL_FEATURES, players_end)
        game_spec, g1 = self._layout(STATE_GAME_FEATURES, b1)
        # 자리별 벤치 프로필이 게임 블록 뒤에 붙는다. 이 길이를 빼먹으면 공개 계약이
        # 실제 ``get_state()`` 길이와 어긋난다 — 소비자가 잘못된 오프셋으로 읽는다.
        bench_size = sum(v for _, v in STATE_BENCH_FEATURES)
        bench_start, bench_end = g1, g1 + TEAM_COUNT * self.bench_size * bench_size
        half = [float(self.hx), float(self.hy)]
        return {
            "dim": bench_end,
            "players": {"start": 0, "end": players_end, "count": self.N,
                        "size": player_size,
                        "features": self._layout(STATE_PLAYER_FEATURES, 0)[0],
                        "encoding": self._player_encoding(observation=False),
                        # 비참여 슬롯은 live 열만 남기고 지운다. 관측 토큰이 status_code
                        # 하나만 남기는 것보다 넓은데, 중앙 상태에서는 team_id가 슬롯→팀 대응과
                        # 인원 하한 판정을 위해 여전히 필요하기 때문이다. is_gk는 비참여 슬롯에서
                        # 다시 읽히지 않으므로 마스크한다.
                        "validity": {
                            "column": "status_code",
                            "valid_when": SLOT_ACTIVE,
                            "masked_value": 0.0,
                            "always_valid": tuple(INACTIVE_SLOT_LIVE_COLUMNS),
                            "note": "비참여 슬롯의 동적 필드는 이후 어떤 전이도 바꾸지 않으므로"
                                    " 0으로 지운다 — 교체 투입은 Substitution 행의 값을 쓰고"
                                    " 퇴장은 되돌릴 수 없다",
                        }},
            "ball": {"start": players_end, "end": b1, "features": ball_spec,
                     "encoding": {
                         "pos": self._scaled(
                             [float(self.hx), float(self.hy), float(e_cfg.norm_ball_z)], "m"),
                         "vel": self._scaled(float(e_cfg.norm_ball_vel), "m/s"),
                         "spin": self._scaled(float(e_cfg.norm_spin), "rad/s")}},
            "game": {"start": b1, "end": g1, "features": game_spec,
                     "encoding": {
                         # 팀 코드는 같은 인코딩이지만 **각각 새 객체**여야 한다.
                         # 한 객체를 공유하면 한 열을 고친 것이 나머지까지 바꾼다.
                         "poss_team_code": self._team_code(),
                         "possession_steps": self._ratio(
                             float(
                                 self.control_fps * POSSESSION_CONTEXT_SECONDS
                             ),
                             "control step",
                             "현재 소유 라벨이 유지된 시간; 5초에서 clip",
                         ),
                         "previous_poss_team_code": self._team_code(),
                         "last_touch_team_code": self._team_code(),
                         "gk_handling_restricted_team_code": self._gk_handling_code(),
                         "restart_team_code": self._team_code(),
                         "pass_team_code": self._team_code(),
                         "kickoff_team_code": self._categorical(
                             TEAM_COUNT, values="TEAM_0=0, TEAM_1=1 — NO_TEAM이 없다"),
                         "attack_dir_team0": self._sign(),
                         "ball_alive": self._binary(),
                         "restart_steps": self._restart_steps_encoding(),
                         "offside_active": self._binary(),
                         "time_left": self._ratio(float(self.game_duration), "control step"),
                         "restart_indirect": self._binary(),
                         "departed_taker_mask": self._departed_taker_encoding(),
                         "score": self._scaled(float(e_cfg.norm_score), "goal"),
                         "restart_kind_code": self._categorical(
                             RESTART_COUNT, values="RK_NONE=0 … RK_GK_HOLD=8"),
                         "last_touch_code": self._categorical(
                             TOUCH_COUNT, values="TOUCH_NONE=0 … TOUCH_BODY_TRAP=11"),
                         **self._squad_spec(False)}},
            "bench": {
                "start": bench_start, "end": bench_end,
                "slots": int(self.bench_size),
                "size": bench_size,
                "order": "팀0 자리 전부, 그다음 팀1 자리 전부",
                "features": self._layout(STATE_BENCH_FEATURES, 0)[0],
                "encoding": {
                    "available": self._binary(),
                    "is_gk": self._binary(),
                    "role_pos": self._scaled(half, "m", "각 팀 공격 접힘 프레임"),
                    "vmax": self._scaled(float(e_cfg.norm_player_vel), "m/s"),
                    "ball_ctrl": self._unit(),
                    "endurance_factor": self._endurance_factor_encoding(),
                },
                "note": "obs만 고치면 중앙 크리틱이 벤치를 못 본다",
            },
        }

    def affordance_spec(self):
        """:meth:`affordance_array` (N, aff_dim) 레이아웃. 저장 대상이 아니라 런타임 계약이다.

        저장하지 않더라도 **정책이 실제로 먹는 입력**이므로 obs/state와 같은 수준의 계약이
        필요하다 — 열별 인코딩, 참여 게이트 여부, 그리고 독립 스키마 버전.
        버전이 없으면 파생량의 의미만 바뀐 변경이 obs/action 스키마도 fingerprint도 건드리지
        않아 의미가 다른 체크포인트가 조용히 섞인다.
        """

        features, dim = self._layout(AFFORDANCE_FEATURES, 0)
        f2b_avail = self._binary()
        f2b_avail.update(
            active_gated=True,
            masked_value=0.0,
            temporal_scope="current physics substep",
            note=("instantaneous runtime readiness: entry reach + exact timers + "
                  "retouch legality; public F2B action availability is the wider "
                  "control-frame potential ~kick_gated"),
        )
        kicker_ready = self._binary()
        kicker_ready.update(
            temporal_scope="current control frame",
            note="강제 키커 이동·setup 카운트다운·contact lock을 거쳐 이 frame에 release",
        )
        kick_gated = self._binary()
        kick_gated.update(
            temporal_scope="current control frame",
            note="국면/활성 기반 F2B 채널 hard gate; ready-now는 f2b_avail",
        )
        kick_forced = self._binary()
        kick_forced.update(
            temporal_scope="current control frame",
            note="이 frame 안에 강제 재개 release가 예측된 지정 키커",
        )
        encoding = {
            # 참여 조건이 걸린 열 — 비참여 슬롯에서 항상 0이다.
            "f2b_avail": f2b_avail,
            # 순위 정규화는 가역 ratio가 아니다. 분모가 활성 인원에 따라 달라지고 1v1에서는
            # ``max(active-1, 1)``의 하한이 걸린다. normalizer 문자열로 위장하지 않고
            # 비가역임을 명시한다 — 소비자가 원래 순위를 복원하려면 활성 인원을 알아야 한다.
            "ball_rank": {"kind": "ordinal_ratio", "range": (0.0, 1.0),
                          "denominator": "max(active_squad_size - 1, 1)",
                          "invertible": False, "active_gated": True, "masked_value": 0.0,
                          "note": "0=최근접, 1=최원거리. 순위 집합과 분모 모두 활성 인원 기준"},
            # 참여와 무관하게 정의되는 열 — 비참여 슬롯도 의미 있는 값을 갖는다
            # (예: 결장자는 move_forced=True로 env가 기술구역으로 옮긴다).
            "in_reach": self._binary(),
            "off_line": self._scaled(float(self.hx), "m", "관측자 팀의 2번째 최종수비 라인"),
            # 재개 종류별 이격 반경으로 나눈 뒤 [-1,1]로 잘라 낸 값이라 가역이 아니다.
            "enc_margin": {"kind": "bounded_signed", "range": (-1.0, 1.0),
                           "denominator": "재개 종류별 이격 반경(페널티 수비 GK는 골라인 허용오차)",
                           "clipped": True, "invertible": False,
                           "active_gated": True, "masked_value": 0.0,
                           "note": "음수면 제한구역 침범. 재개 비활성이면 0"},
            "any_encroacher": self._binary(),
            "kicker_locked": self._binary(),
            "kicker_ready": kicker_ready,
            "is_taker": self._binary(),
            "pass_signal": self._relation(),
            "is_sp_ours": self._relation(),
            "move_forced": self._binary(),
            "kick_gated": kick_gated,
            "kick_forced": kick_forced,
        }
        for name, spec in encoding.items():
            spec.setdefault("active_gated", False)
        return {
            "dim": dim,
            "count": self.N,
            "schema_version": AFFORDANCE_SCHEMA_VERSION,
            "features": features,
            "encoding": encoding,
            "stored": False,
            "validity": {
                "source": "state.active_player",
                "column": "status_code",
                "valid_when": SLOT_ACTIVE,
                "note": "active_gated=True인 열만 참여 조건이 걸린다. 나머지는 비참여 슬롯에도"
                        " 정의된다 — env가 결장자를 강제 이동시키므로 move_forced 등은 참이다",
            },
        }

    def profile_spec(self):
        """identity가 바뀔 때만 갱신되는 슬롯 프로필의 레이아웃.

        시퀀스 저장에서 프레임 payload를 줄이려는 소비자를 위한 계약이다. 여기 이름은
        :data:`constants.STATE_PLAYER_FEATURES`의 부분집합이라 프레임 벡터에서 그대로 잘라내
        ``(player_id, slot_generation)`` 키의 sidecar로 옮길 수 있다.
        """

        features, dim = self._layout(PLAYER_PROFILE_FEATURES, 0)
        state_features = self._layout(STATE_PLAYER_FEATURES, 0)[0]
        obs_features = self._layout(OBS_PLAYER_FEATURES, 0)[0]
        return {
            "dim": dim,
            "features": features,
            "state_columns": {name: state_features[name] for name, _ in PLAYER_PROFILE_FEATURES},
            # 관측 토큰에는 team_id 대신 관점 상대값 team_relation이 있으므로 그 이름만 대응된다.
            "obs_columns": {name: obs_features[name] for name, _ in PLAYER_PROFILE_FEATURES
                            if name in obs_features},
            "sidecar_dim": dim * self.N,
            "state_frame_payload_dim": self.state_spec()["dim"] - dim * self.N,
            "obs_frame_payload_dim": self.obs_spec()["dim"] - sum(
                size for name, size in PLAYER_PROFILE_FEATURES if name in obs_features) * self.N,
        }
