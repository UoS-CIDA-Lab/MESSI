import jax
import jax.numpy as jnp

from spatial import _u01, stretch_decode
from constants import *

class Observation:
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
        """ 전 에이전트 관측 (N, obs_dim)
        """
        N = self.N 
        e_cfg = self.e_cfg
        s_cfg = self.s_cfg
        
        att_dir = state.attack_dir[:, None]
        
        player_pos = state.player_pos
        player_vel = state.player_vel
        others_idx = self.others_idx
        active_others_idx = state.active_player[others_idx].astype(jnp.float32)
        
        self_abs_pos = jnp.stack(
            [
                player_pos[:, DIM_X] / self.hx, 
                player_pos[:, DIM_Y] / self.hy
            ], axis=1
        ) * att_dir #위치 좌표계를 -1~1로 정규화하는 것

        # 자기 절대속도(관측자 attack_dir로 접힘). facing이 이 값의 파생량이므로(movement.
        # facing_from_velocity) 이걸 주면 obs에 facing 성분이 따로 없어도 정보 손실이 없다.
        self_abs_vel = player_vel * att_dir / e_cfg.norm_player_vel

        rel_pos = (
            (player_pos[others_idx] - player_pos[:, None, :])
            * att_dir[:, :, None]
            / self.field_size
        )

        # 타 선수 **절대 속도**(관측자 attack_dir로 접힘) — '지금 어디'가 아니라 '곧 어디'를 보게 해
        # 패스 차단·마킹의 anticipation(선점)을 가능케 한다(tactics.py의 lane_completion·pressure).
        oth_abs_vel = player_vel[others_idx] * att_dir[:, :, None] / e_cfg.norm_player_vel
        # 타 선수 **절대 위치**(위치는 절대·상대 둘 다 부여) — 국소 상호작용은 rel_pos, 필드 맥락은 abs_pos.
        oth_abs_pos = player_pos[others_idx] / self.field_half * att_dir[:, :, None]

        team_flag = jnp.where(
            state.team_id[others_idx] == state.team_id[:, None], 
            1.0, 
            -1.0
        )
        
        
        rb_pos = (state.ball_pos[None, :DIM_Z] - player_pos) * att_dir
        rb_pos_obs = jnp.stack(
            [
                rb_pos[:, DIM_X] / s_cfg.length, 
                rb_pos[:, DIM_Y] / s_cfg.width,
                jnp.full((N,), state.ball_pos[DIM_Z] / e_cfg.norm_ball_z)
            ], axis=1
        )
        
        # [guide.md §3] 공 **절대 위치**(x·y는 attack_dir 접힘, z 유지) — 상대위치 rb_pos_obs와 병행.
        ball_abs_pos = jnp.stack(
            [
                state.ball_pos[DIM_X] * att_dir[:, 0] / self.hx,
                state.ball_pos[DIM_Y] * att_dir[:, 0] / self.hy,
                jnp.full((N,), state.ball_pos[DIM_Z] / e_cfg.norm_ball_z)
            ], axis=1
        )
        # 공 절대 속도(rb_vel_obs)는 기존이 이미 절대(state.ball_vel, 선수속도 안 뺌) → 그대로 유지.
        rb_vel_obs = jnp.broadcast_to(state.ball_vel, (N, DIM_ALL))
        rb_vel_obs = rb_vel_obs.at[:, :DIM_Z].multiply(att_dir) / e_cfg.norm_ball_vel

        rb_spin_obs = jnp.broadcast_to(state.ball_spin, (N, DIM_ALL))
        # 회전(180°) 규약: 스핀도 속도와 동일 변환(x·y만 반전, z 유지). 회전엔 유사벡터 예외가
        # 없음 — 원본 거울 규약의 y·z 반전을 옮겨오면 반대편 팀의 휘는 방향이 뒤집힘.
        rb_spin_obs = rb_spin_obs.at[:, :DIM_Z].multiply(att_dir) / e_cfg.norm_spin
        
        in_reach, _ = self._in_reach(state)
        f2b_avail = in_reach & (state.ball_state == BALL_ALIVE) & (state.cooldown <= 0)
        
        norm_cooldown = (state.cooldown / e_cfg.cooldown_substeps)[:, None]
        is_gk_self = state.gk_indices[:, None].astype(jnp.float32)

        # 공격팀 기준 관측: poss_team_flag=1.0이면 소유, -1.0이면 비소유, 0.0이면 소유팀 없음
        poss_team_flag = jnp.where(state.team_id == state.poss_team, 1.0, -1.0)
        poss_ours = jnp.where(state.poss_team >= 0, poss_team_flag, 0.0)[:, None]
        
        # own_offside: 자기 자신의 오프사이드 플래그
        own_offside = state.offside_flag.astype(jnp.float32)[:, None]
        
        adir_team = state.attack_dir[self.team_indices]  # 팀0/팀1 공격방향
        x_att = player_pos[None, :, DIM_X] * adir_team[:, None]             # (2, N)
        opp = (
            state.team_id[None, :] != jnp.arange(TEAM_COUNT)[:, None]
        ) & state.active_player[None, :]
        x_opp = jnp.where(opp, x_att, -jnp.inf)
        line2 = jax.lax.top_k(x_opp, 2)[0][:, 1]                            # (2,) 팀별 2번째 최종수비
        n_opp = opp.sum(axis=1)
        line2 = jnp.where(n_opp >= 2, line2, self.hx)
        off_line = (jnp.clip(line2 / self.hx, -1.0, 1.0)
                    [state.team_id.astype(jnp.int32)])[:, None]             # (N,1) 팀별 라인을 선수에 배분
        # 오프사이드 창 잔여 시간(전역 스칼라, pass_protect 정규화). pass_signal이 '창이 열렸는가·
        # 누구 것인가'만 주므로 이것 없이는 창이 곧 닫히는지 알 수 없다 — 수신자가 언제까지
        # 플래그에 걸리는지, 수비가 언제까지 기다리면 되는지가 관측에서 빠진다(get_state는 이미 보유).
        pass_t_norm = jnp.clip(state.pass_t / jnp.float32(e_cfg.pass_protect), 0.0, 1.0)
        pass_t_norm = jnp.broadcast_to(pass_t_norm, (N,))[:, None]
        pass_active = state.pass_t > 0
        pass_signal = jnp.where(
            pass_active & (state.pass_team >= 0),
            jnp.where(state.team_id == state.pass_team, 1.0, -1.0),
            0.0,
        )[:, None]   # (N,1) ∈ {-1, 0, +1}
                
        ball_alive = jnp.broadcast_to((state.ball_state == BALL_ALIVE), (N,))[:, None].astype(jnp.float32)

        restart_active = state.restart_t > 0
        ours = (state.team_id == state.restart_team).astype(jnp.float32) * 2.0 - 1.0
        is_sp_ours = jnp.where(restart_active, ours, 0.0)[:, None]
        # restart_kind one-hot 인코딩 — 모든 선수에게 동일하므로 브로드캐스트.
        rk_onehot = jax.nn.one_hot(state.restart_kind, RESTART_COUNT)
        rk_onehot = jnp.broadcast_to(rk_onehot[None, :], (N, RESTART_COUNT))
        # restart_t 정규화 — 분모는 config의 재개 종류별 창을 그대로 사용한다.
        rt_window = jnp.where(state.restart_kind == RK_PENALTY, e_cfg.penalty_substeps,
                    jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps,
                              e_cfg.restart_substeps))
        rt_norm = jnp.where(restart_active, state.restart_t / rt_window, 0.0)
        rt_norm = jnp.broadcast_to(rt_norm, (N,))[:, None]
        
        # is_kicker_locked: 내가 키커이고 아직 킥 불가(도착 전 or 정렬 카운트다운 중)
        kicker_locked, setup_done_sp, _, sp_active = self._setpiece_kick_lock(state)
        ar_sp = self.player_indices
        is_kicker_locked = jnp.where(
            sp_active & kicker_locked & (ar_sp == state.pending_taker), 1.0, 0.0)[:, None]
        # [H5-B] 자기 지정키커·발사허용 신호 — 릴리즈 킥(=BC 킥 라벨)이 obs만으로 예측 가능해야 한다.
        #  self_is_taker: 내가 지정 세트피스 참가자(pending/setpiece/throw). others_is_taker의 self판.
        #  self_kicker_ready: 내가 pending 키커 & setup 완료(=이 순간 강제 발사). is_kicker_locked의 보수.
        self_is_taker = (((ar_sp == state.pending_taker) & (state.pending_taker >= 0))
                         | ((ar_sp == state.setpiece_taker) & (state.setpiece_taker >= 0))
                         | ((ar_sp == state.throw_taker) & (state.throw_taker >= 0))).astype(jnp.float32)[:, None]
        self_kicker_ready = jnp.where(
            sp_active & setup_done_sp & (ar_sp == state.pending_taker), 1.0, 0.0)[:, None]
        # [H5-A] 페널티 은닉 상태 노출(재실행 결정 상태 → Markov 관측화).
        #  pen_flight: 페널티 비행 중 여부(자기 기준: 없음0 / 우리팀킥+1 / 상대-1). 비페널티엔 flight_team=-1→0.
        #  self_pen_encroach: 내가 침범자로 래치됐나(킥 순간 기록). 비페널티엔 all-False→0.
        pf = state.penalty_flight_team
        pen_flight = jnp.where(pf >= 0, jnp.where(state.team_id == pf, 1.0, -1.0), 0.0)[:, None]
        self_pen_encroach = state.penalty_encroach_mask.astype(jnp.float32)[:, None]
        # 팀 요약 래치 — 자기 침범만 보이면 같은 관측에서 골 인정/재실행이 갈린다(events의
        # 재실행 매트릭스가 penalty_encroach_mask를 공격/수비로 갈라 쓰기 때문). 관측자 팀 기준
        # ours/theirs이며 재개 활성 여부로 게이팅하지 않는다 — 래치는 페널티 해소 시 env가 지운다.
        same_team = state.team_id[:, None] == state.team_id[None, :]
        pen_any = state.penalty_encroach_mask[None, :]
        pen_ours_any = jnp.any(pen_any & same_team, axis=1).astype(jnp.float32)[:, None]
        pen_theirs_any = jnp.any(pen_any & (~same_team), axis=1).astype(jnp.float32)[:, None]
        
        clear_r, encroachers, enc_m = self._encroach_geometry(state)
        is_pen = state.restart_kind == RK_PENALTY
        is_def = state.team_id != state.restart_team
        # 페널티 subject = 이격 '의무자' 전원(키커·수비GK만 면제) — 과거에 encroachers(현재
        # 위반자)를 썼더니 경계 밖으로 나가는 순간 margin이 0으로 사라져 '밖에서 대기하라'는
        # 신호가 정책에 전달될 수 없었다(경계 재진입 진동 → 무한 재실행의 관측 측 원인).
        is_kicker_m = (self.player_indices == state.pending_taker) & (state.pending_taker >= 0)
        is_def_gk_m = (state.gk_indices == 1) & (state.team_id != state.restart_team)
        pen_subject = state.active_player & (~is_kicker_m) & (~is_def_gk_m)
        subject = jnp.where(is_pen, pen_subject, is_def)      # 페널티=의무자 전원 / 그외=수비팀만
        enc_margin = jnp.where(restart_active & subject, jnp.clip(enc_m/clear_r,-1,1), 0.0)[:, None]
        
        any_enc = jnp.where(restart_active & jnp.any(encroachers), 1.0, 0.0)
        any_enc = jnp.broadcast_to(any_enc, (N,))[:, None]
        # 간접 프리킥 플래그 — 재개 타이머가 끝나 공이 라이브가 된 뒤에도 두 번째 터치 전까지
        # 직접 골이 무효다(events._events). restart_active로 가리면 물리가 다른 두 라이브 상태가
        # 같은 관측이 되므로, 래치 자체를 두 번째 터치/이벤트가 해제할 때까지 그대로 노출한다.
        is_fk_indirect = jnp.where(state.restart_indirect, 1.0, 0.0)
        is_fk_indirect = jnp.broadcast_to(is_fk_indirect, (N,))[:, None]
        setpiece = jnp.concatenate(
            [
                is_sp_ours,
                rk_onehot,
                rt_norm,
                is_kicker_locked,
                enc_margin,
                any_enc,
                is_fk_indirect,
                self_is_taker,
                self_kicker_ready,
                pen_flight,
                self_pen_encroach,
                pen_ours_any,
                pen_theirs_any
            ], axis=1
        )

        # facing은 obs에 별도 성분으로 넣지 않는다 — 속도 방향의 결정함수라(movement.
        # facing_from_velocity) self/others의 abs_vel에 이미 담겨 있고, 넣으면 중복이다.
        # state.player_facing은 파울 계산(fouls.py·contest.py)·렌더가 쓰는 파생 캐시다.
        vmax_n = (state.vmax / e_cfg.norm_player_vel)[:, None]
        ctrl_n = state.player_ctrl[:, None]
        yellow_n = state.yellow_cards[:, None].astype(jnp.float32)

        ar = self.player_indices
        retouch = (
            ((ar == state.setpiece_taker) & (state.setpiece_taker >= 0)) |
            ((ar == state.throw_taker) & (state.throw_taker >= 0))
        ).astype(jnp.float32)[:, None]

        # 동일한 retouch/is_taker 비트라도 스로인과 일반 세트피스는 직접골 규칙이 다르다.
        # 스로인은 어느 골문으로든 직접 득점할 수 없고, 일반 세트피스는 상대 골 직접 득점이
        # 가능하므로(events._events), 현재 재터치 제한의 출처를 전 선수에게 따로 알린다.
        retouch_is_throw = jnp.broadcast_to(
            state.throw_taker >= 0, (N,)
        ).astype(jnp.float32)[:, None]

        # is_taker: 지정 세트피스 참가자(킥 예정 pending_taker + 재터치금지 setpiece/throw_taker).
        # 세트피스 수비 시 '누가 킥하는가'를 타 선수 관측으로 알 수 있게 others에 노출(Markov 빈틈 보강).
        is_taker_all = (
            ((ar == state.pending_taker) & (state.pending_taker >= 0)) |
            ((ar == state.setpiece_taker) & (state.setpiece_taker >= 0)) |
            ((ar == state.throw_taker) & (state.throw_taker >= 0))
        ).astype(jnp.float32)
        others_is_taker = is_taker_all[others_idx]                          # (N, N-1)

        time_left = jnp.clip(1.0 - state.t / jnp.float32(self.game_duration), 0.0, 1.0)
        time_left = jnp.broadcast_to(time_left, (N,))[:, None]
        self_ctrl_lock = (state.ctrl_lock_t / jnp.float32(e_cfg.ctrl_lock_substeps))[:, None]

        reach_n = (state.reach_z / e_cfg.norm_body_z)[:, None]
        head_n = (state.head_z / e_cfg.norm_body_z)[:, None]

        last_touch = jnp.where(state.last_touch_team >= 0,
                               jnp.where(state.team_id == state.last_touch_team, 1.0, -1.0), 0.0)[:, None]
        # 마지막 터치 '종류' one-hot — 팀무관 범주값이라 관점 폴딩 없이 전 선수
        # 동일 broadcast. last_touch(팀부호)와 결합하면 GK가 '우리팀 발 플레이(PASS/DRIBBLE)=백패스'를
        # 판별 가능(catch-vs-IDFK 게이트, movement._gk_reactive_claim / contest.gk_backpass의 은닉 상태 노출).
        last_touch_code_oh = jnp.broadcast_to(
            jax.nn.one_hot(state.last_touch_code, TOUCH_COUNT), (N, TOUCH_COUNT))

        tid = state.team_id
        score_diff = ((state.score[tid] - state.score[1 - tid]) / e_cfg.norm_score)[:, None]

        # 후반 킥오프가 우리 것인가(±1). `kickoff_team`은 **전반** 킥오프 팀이고 후반은 그 반대다
        # (`events._halftime_switch`의 second_kick). 이 비트가 없으면 두 상태의 obs가 완전히 같은데
        # 하프타임 전이 결과가 갈린다. 경기 내내 상수라 짧은 history로도 복원할 수 없어 관측이 답이다.
        second_half_kick = TEAM_1 - state.kickoff_team
        sh_kickoff_ours = jnp.where(tid == second_half_kick, 1.0, -1.0)[:, None]
        
        others_cd_idx = state.cooldown[others_idx] / e_cfg.cooldown_substeps

        # observation 블록 결합
        self_block = jnp.concatenate(
            [
                self_abs_pos,
                self_abs_vel,
                state.stamina[:, None],
                vmax_n,
                reach_n,
                head_n,
                is_gk_self,
                yellow_n,
                norm_cooldown,
                ctrl_n,
                in_reach[:, None].astype(jnp.float32),
                f2b_avail[:, None].astype(jnp.float32),
                own_offside,
                pass_signal,
                retouch,
                self_ctrl_lock
            ], axis=1
        )

        # 타 선수 특징은 언마스크로 쌓고 마지막에 active 마스크를 블록 전체에 한 번만 곱한다.
        # rel_pos/others_cd/is_taker/team_flag는 산출부에서 이미 마스킹됐지만 마스크가
        # {0,1}이라 재적용이 멱등(bit 동일) — 특징마다 흩어졌던 * active_others_idx를 여기로 통일.
        others_block = jnp.concatenate(
            [
                rel_pos,
                oth_abs_pos,
                oth_abs_vel,
                state.stamina[others_idx][:, :, None],
                state.vmax[others_idx][:, :, None] / e_cfg.norm_player_vel,
                state.reach_z[others_idx][:, :, None] / e_cfg.norm_body_z,
                state.head_z[others_idx][:, :, None] / e_cfg.norm_body_z,
                state.gk_indices[others_idx][:, :, None].astype(jnp.float32),
                state.yellow_cards[others_idx].astype(jnp.float32)[:, :, None],
                state.offside_flag[others_idx].astype(jnp.float32)[:, :, None],
                # 타 선수 페널티 침범 래치 — 팀 요약만으로는 선수별 카드·퇴장 결과를 예측할 수 없다.
                # 퇴장자 열은 블록 끝 active 마스크가 함께 0으로 만든다.
                state.penalty_encroach_mask[others_idx].astype(jnp.float32)[:, :, None],
                state.player_ctrl[others_idx][:, :, None],
                self_ctrl_lock[others_idx],
                others_cd_idx[:, :, None],
                others_is_taker[:, :, None],
                team_flag[:, :, None],
            ], axis=-1
        )
        others_block = (others_block * active_others_idx[:, :, None]).reshape(N, -1)

        ball_block = jnp.concatenate(
            [
                rb_pos_obs,
                ball_abs_pos,
                rb_vel_obs,
                rb_spin_obs,
                ball_alive,
                poss_ours,
            ], axis=-1
        )

        context_block = jnp.concatenate(
            [
                setpiece,
                retouch_is_throw,
                off_line,
                pass_t_norm,
                time_left,
                last_touch,
                last_touch_code_oh,
                score_diff,
                sh_kickoff_ours
            ], axis=-1
        )
        full = jnp.concatenate(
            [
                self_block,
                others_block,
                ball_block,
                context_block
            ], axis=-1
        ) * state.active_player[:, None].astype(jnp.float32)
        return full

    def get_state(self, state):
        """중앙집중 크리틱용 전역 상태 벡터.

        절대 좌표의 선수·공뿐 아니라 다음 전이를 바꾸는 재터치 제한, 오프사이드 남은 창,
        마지막 터치 종류, 페널티 비행/침범 래치, 후반 킥오프 팀을 명시한다. 예전의 합쳐진
        ``is_taker``와 단순 ``pass_signal``은 서로 다른 물리 상태를 같은 벡터로 만들었으므로
        엄밀한 Markov 상태가 아니었다. 정확한 레이아웃은 :meth:`state_spec`이 단일 진실원천이다.
        """
        e_cfg = self.e_cfg
        ar = self.player_indices

        # ── players, 절대 프레임(폴딩 없음) ──
        players_pos_n = state.player_pos / self.field_half
        players_vel_n = state.player_vel / e_cfg.norm_player_vel
        players_face = jnp.stack([jnp.cos(state.player_facing), jnp.sin(state.player_facing)], axis=1)
        players_vmax_n = (state.vmax / e_cfg.norm_player_vel)[:, None]
        players_ctrl = state.player_ctrl[:, None]
        players_reach_n = (state.reach_z / e_cfg.norm_body_z)[:, None]
        players_head_n = (state.head_z / e_cfg.norm_body_z)[:, None]
        players_stamina = state.stamina[:, None]
        players_cooldown_n = (state.cooldown / e_cfg.cooldown_substeps)[:, None]
        players_ctrl_lock_n = (state.ctrl_lock_t / e_cfg.ctrl_lock_substeps)[:, None]
        players_on_pitch = state.active_player.astype(jnp.float32)[:, None]
        players_yellow = state.yellow_cards.astype(jnp.float32)[:, None]
        players_pending = ((ar == state.pending_taker) & (state.pending_taker >= 0)).astype(jnp.float32)[:, None]
        players_setpiece = ((ar == state.setpiece_taker) & (state.setpiece_taker >= 0)).astype(jnp.float32)[:, None]
        players_throw = ((ar == state.throw_taker) & (state.throw_taker >= 0)).astype(jnp.float32)[:, None]
        players_pen_encroach = state.penalty_encroach_mask.astype(jnp.float32)[:, None]
        players_offside = state.offside_flag.astype(jnp.float32)[:, None]
        players_team = (state.team_id.astype(jnp.float32) * 2.0 - 1.0)[:, None]
        players_gk = state.gk_indices.astype(jnp.float32)[:, None]
        players_block = jnp.concatenate(
            [
                players_pos_n,
                players_vel_n,
                players_face,
                players_vmax_n,
                players_ctrl,
                players_reach_n,
                players_head_n,
                players_stamina,
                players_cooldown_n,
                players_ctrl_lock_n,
                players_on_pitch,
                players_yellow,
                players_pending,
                players_setpiece,
                players_throw,
                players_pen_encroach,
                players_offside,
                players_team,
                players_gk,
            ], axis=1
        ).reshape(-1)

        # ── ball (9), 절대 프레임 ──
        ball_pos_n = state.ball_pos / jnp.array([self.hx, self.hy, e_cfg.norm_ball_z])
        ball_vel_n = state.ball_vel / e_cfg.norm_ball_vel
        ball_spin_n = state.ball_spin / e_cfg.norm_spin
        ball_block = jnp.concatenate([ball_pos_n, ball_vel_n, ball_spin_n])   # (9,)

        # ── game, 전역 ──
        def team_onehot(team):
            return jax.nn.one_hot(jnp.clip(team, NO_TEAM, TEAM_1) + 1, TEAM_COUNT + 1)

        poss_onehot = team_onehot(state.poss_team)                 # [무, 팀0, 팀1]
        last_touch_team = team_onehot(state.last_touch_team)
        attack_dir0 = state.attack_dir[0]                        # 팀0 공격방향(절대 프레임 해석 기준)
        ball_alive = (state.ball_state == BALL_ALIVE).astype(jnp.float32)
        rt_window_g = jnp.where(state.restart_kind == RK_PENALTY, e_cfg.penalty_substeps,
                      jnp.where(state.restart_kind == RK_GK_HOLD, e_cfg.gk_hold_substeps,
                                e_cfg.restart_substeps))
        restart_t_norm = jnp.where(state.restart_t > 0, state.restart_t / rt_window_g, 0.0)
        restart_team = team_onehot(state.restart_team)
        pass_team = team_onehot(jnp.where(state.pass_t > 0, state.pass_team, NO_TEAM))
        pass_t_norm = jnp.clip(state.pass_t / jnp.float32(e_cfg.pass_protect), 0.0, 1.0)
        time_left = jnp.clip(1.0 - state.t / jnp.float32(self.game_duration), 0.0, 1.0)
        score_n = state.score.astype(jnp.float32) / e_cfg.norm_score
        rk_onehot = jax.nn.one_hot(state.restart_kind, RESTART_COUNT)
        # 간접 프리킥 래치는 공이 라이브가 된 뒤에도 두 번째 터치 전 직접골 판정을 바꾼다.
        is_fk_indirect = state.restart_indirect.astype(jnp.float32)
        last_touch_code = jax.nn.one_hot(state.last_touch_code, TOUCH_COUNT)
        kickoff_team = jax.nn.one_hot(state.kickoff_team, TEAM_COUNT)
        penalty_flight = team_onehot(state.penalty_flight_team)
        game_block = jnp.concatenate(
            [
                poss_onehot,
                last_touch_team,
                jnp.stack([attack_dir0, ball_alive, restart_t_norm, pass_t_norm,
                           time_left, is_fk_indirect]),
                restart_team,
                pass_team,
                score_n,
                rk_onehot,
                last_touch_code,
                kickoff_team,
                penalty_flight,
            ]
        )

        return jnp.concatenate([players_block, ball_block, game_block])

    def get_obs(self, state):
        """JaxMARL 규약 dict 어댑터 — 계산은 get_obs_array(단일 진실원천)."""
        full = self.get_obs_array(state)
        return {a: full[i] for i, a in enumerate(self.agents)}

    def get_avail_actions_array(self, state):
        """행동 마스크 (N, 2) = [MOVE, F2B].

        실제 전이에서 쓰는 :meth:`action_agency`를 그대로 투영한다. 따라서 데드볼 내장 엔진이
        전원 이동을 덮어쓰는 프레임도 MOVE=0이고, 강제 세트피스 킥의 파라미터는 F2B=1이다.
        """
        agency = self.action_agency(state)
        return jnp.stack(
            [~agency["move_forced"], ~agency["kick_gated"]], axis=1
        ).astype(jnp.float32)

    def get_avail_actions(self, state):
        """JaxMARL 규약 dict 어댑터 — 계산은 get_avail_actions_array(단일 진실원천)."""
        mask = self.get_avail_actions_array(state)
        return {a: mask[i] for i, a in enumerate(self.agents)}

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
        블록·특징 슬라이스를 읽도록 {블록: {start,end,features,...}} 반환. get_obs_array와 순서 일치."""
        self_spec, o1 = self._layout(OBS_SELF_FEATURES, 0)
        other_size = sum(s for _, s in OBS_OTHER_FEATURES)
        n_others = self.N - 1
        others_end = o1 + n_others * other_size
        ball_spec, b1 = self._layout(OBS_BALL_FEATURES, others_end)
        ctx_spec, c1 = self._layout(OBS_CONTEXT_FEATURES, b1)
        return {
            "dim": c1,
            "self": {"start": 0, "end": o1, "features": self_spec},
            "others": {"start": o1, "end": others_end, "count": n_others,
                       "size": other_size, "features": self._layout(OBS_OTHER_FEATURES, 0)[0]},
            "ball": {"start": others_end, "end": b1, "features": ball_spec},
            "context": {"start": b1, "end": c1, "features": ctx_spec,
                        "setpiece_features": self._layout(OBS_SETPIECE_FEATURES, 0)[0]},
        }

    def state_spec(self):
        """get_state (state_dim,) 레이아웃의 단일 진실원천. get_state와 순서 일치."""
        player_size = sum(s for _, s in STATE_PLAYER_FEATURES)
        players_end = self.N * player_size
        ball_spec, b1 = self._layout(STATE_BALL_FEATURES, players_end)
        game_spec, g1 = self._layout(STATE_GAME_FEATURES, b1)
        return {
            "dim": g1,
            "players": {"start": 0, "end": players_end, "count": self.N,
                        "size": player_size, "features": self._layout(STATE_PLAYER_FEATURES, 0)[0]},
            "ball": {"start": players_end, "end": b1, "features": ball_spec},
            "game": {"start": b1, "end": g1, "features": game_spec},
        }
