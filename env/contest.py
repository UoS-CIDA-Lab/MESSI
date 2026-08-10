"""볼 경합 해결 — 승자 선정(_contest_winner)과 킥/탈취/파울/굴절/세트피스 힘 적용(_apply_force2ball).
세트피스 taker 지정(_designate_taker)·발사각 하한(launch_lo)도 포함. 태클 파울·오프사이드 플래그 겸함.

복원 관점: 승자는 궤적에서 '누가 공을 건드렸나'로 직접 관측되고, 파울/탈취/굴절 각 분기는 단일
uniform draw로 갈리며, 굴절각도 단일 draw의 가역 함수다. 따라서 관측 결과(공 속도·소유변화·재개
종류)로부터 어느 분기·어떤 draw였는지 결정적으로 역산할 수 있다(재추첨 매칭 불필요).
"""
import jax
import jax.numpy as jnp

from constants import *
from spatial import _unit


class Contest:
    def _contest_winner(self, state, candidate, dist_xy, key, forced_winner=None):
        """경합 후보 중 승자 1명 선정 — 거리·도달시간·높이적합·소유·볼컨트롤 점수의 Gumbel argmax.

        contest_temp가 낮을수록 점수 최대인 선수가 거의 확실히 승자(결정적). 승자는 궤적에서 직접
        관측되므로 복원 시 Gumbel 재추첨 없이 주입 가능 — forced_winner로 관측 승자를 pin한다:
        ≥0이면 그 선수를 승자로, ==-1이면 무승자, ==-2(또는 None)이면 정상 샘플.
        단 **pin도 합법 후보(candidate: 도달·쿨다운·게이트·온피치)여야 성립** — 도달은 추첨이
        아니라 확정 규칙이므로 우회 대상이 아니다. 후보 밖 pin은 불발(-1, any_cand=False)로
        처리해 복원 sim의 발산이 지표에 정직하게 드러나게 한다.
        반환 (winner_idx 또는 -1, any_cand).
        """
        e_cfg = self.e_cfg
        time_to_reach = dist_xy / (state.vmax + GEOMETRY_EPS)
        height_fit = jnp.clip(
            1.0 - state.ball_pos[DIM_Z] / (state.reach_z + GEOMETRY_EPS), 0.0, 1.0
        )
        poss_bias = (state.team_id == state.poss_team).astype(jnp.float32)
        score = (-e_cfg.w_dist * dist_xy - e_cfg.w_time * time_to_reach + e_cfg.w_height * height_fit
                 + e_cfg.w_poss * poss_bias + e_cfg.w_ctrl * state.player_ctrl)
        u = jnp.clip(jax.random.uniform(key, score.shape), PROB_EPS, 1.0 - PROB_EPS)
        gumbel = -jnp.log(-jnp.log(u))
        candidate = candidate & state.active_player                # 퇴장 선수는 경합 불가
        noisy = jnp.where(candidate, score / e_cfg.contest_temp + gumbel, -jnp.inf)
        winner = jnp.argmax(noisy)
        sampled_winner = jnp.where(jnp.any(candidate), winner, NO_PLAYER)
        sampled_any = jnp.any(candidate)
        if forced_winner is None:
            return sampled_winner, sampled_any
        use = forced_winner >= NO_PLAYER                           # SAMPLED_WINNER sentinel → 샘플
        forced_ok = ((forced_winner >= 0) & (forced_winner < candidate.shape[0])
                     & candidate[jnp.clip(forced_winner, 0, candidate.shape[0] - 1)])
        eff_forced = jnp.where(
            forced_ok, forced_winner, jnp.int32(NO_PLAYER)
        )   # 후보 밖 pin → 불발
        return (jnp.where(use, eff_forced, sampled_winner),
                jnp.where(use, forced_ok, sampled_any))

    def _designate_taker(self, state, ball_xy, restart_team, goalkeeper_only):
        """세트피스 키커 지정 — restart_team 온피치 선수 중 공에 최근접(골킥이면 GK 우선).
        GK 퇴장 등으로 GK 마스크가 비면 팀 최근접으로 폴백 — 마스크 전무 시 argmin(∞)=0이
        상대팀 선수를 키커로 지정·견인하는 폭주를 막는다(IFAB상 골킥은 아무나 차도 됨)."""
        dist = jnp.linalg.norm(state.player_pos - ball_xy[None, :], axis=1)
        base = (state.team_id == restart_team) & state.active_player
        gk_mask = base & (state.gk_indices == 1)
        use_gk = goalkeeper_only & jnp.any(gk_mask)
        mask = jnp.where(use_gk, gk_mask, base)
        taker = jnp.argmin(jnp.where(mask, dist, jnp.inf)).astype(jnp.int32)
        return jnp.where(jnp.any(mask), taker, jnp.int32(NO_PLAYER))

    def launch_lo(self, ball_z):
        """공 높이 ball_z에서 허용되는 최소(하향) 발사각(rad). 지면공은 -launch_down_ground,
        ball_z≥launch_down_ref에서 -launch_max까지 열림. 액션 발사각[0,1]은 [launch_lo, launch_max]로 재매핑."""
        e_cfg = self.e_cfg
        t = jnp.clip((ball_z - self.r_ball) / (e_cfg.launch_down_ref - self.r_ball), 0.0, 1.0)
        return -(e_cfg.launch_down_ground + (e_cfg.launch_max - e_cfg.launch_down_ground) * t)

    def _apply_force2ball(self, state, winner, any_cand, f2b_dir, f2b_pow, f2b_launch, key,
                          spin_side=None, spin_back=None, want_kick=None,
                          forced_freeplay=None, suppress_retake=None):
        """경합 승자의 공 접촉 결과 적용 — 킥/탈취/파울/굴절/세트피스 소비/오프사이드 플래그/카드.

        분기(전부 관측 결과에서 역산 가능):
          free_play : 오픈플레이·세트피스 킥 — 승자 커맨드(방향·파워·발사각·스핀)로 발사.
          tackle_ok : 상대 소유 탈취 성공(uniform<tackle_prob) — 커맨드 적용, 출구속도 상한 캡.
          foul      : 태클 파울(uniform<p_foul) — 공 데드 + FK/페널티 + 확률 카드.
          deflect   : 접촉했으나 탈취 실패(uniform<deflect_prob) — 소유 불변 루즈볼, 각도 랜덤.
          gk_claim  : 자기 박스 GK의 무의도 승리(want_kick 승자는 하이재킹 안 함) —
                      저속=캐치홀드(RK_GK_HOLD) / 고속=parry / 백패스=IDFK(포워드에선 movement
                      합법성 게이트로 사실상 미도달 — forced_winner 복원·방어용 분기).

        forced_freeplay: reconstruct용 분기 pin(스칼라 bool). True면 opp_poss를 젖혀 관측 터치를
        free_play(결정론 킥)로 강제 — tackle/foul/deflect 추첨을 구조적으로 무력화한다. 상태
        (poss_team)를 덮어쓰지 않는 순수 pin이며, 소유는 터치 성사 시 new_poss가 스스로 갱신.
        None/False면 포워드 불변.
        suppress_retake: reconstruct 세트피스 에피소드용 pin(스칼라 bool). True면 재개 소비 시
        침범 retake 판정만 봉쇄 — 근거는 관측("심판이 실제로 진행시켰다": 골킥 박스 잔류 18%에서
        실경기는 대부분 속행, IFAB 심판 재량). 확정 규칙 우회가 아니라 관측된 재량 결과의 pin.
        None/False면 포워드 불변."""
        e_cfg = self.e_cfg
        f_cfg = self.f_cfg
        k_foul, k_tackle, k_card, k_deflect, k_defl_dir = jax.random.split(key, 5)

        ball_z = state.ball_pos[DIM_Z]
        head_z = state.head_z[winner]
        pelvis_z = e_cfg.pelvis_frac * head_z
        header = ball_z > head_z
        is_chest = (ball_z > pelvis_z) & (~header)
        power_cap = jnp.where(header, e_cfg.header_cap, jnp.where(is_chest, e_cfg.chest_cap, 1.0))
        speed = f2b_pow[winner] * power_cap * e_cfg.f2b_speed_max

        # 발사각 재매핑 — 액션 [0,launch_max]를 [launch_lo(ball_z), launch_max]로(하향 타격 커맨드화)
        launch_floor = self.launch_lo(ball_z)
        launch_ang = launch_floor + (f2b_launch[winner] / e_cfg.launch_max) * (e_cfg.launch_max - launch_floor)
        kick_dir = f2b_dir[winner]
        kick_vel = speed * jnp.array([jnp.cos(launch_ang) * kick_dir[DIM_X],
                                      jnp.cos(launch_ang) * kick_dir[DIM_Y],
                                      jnp.sin(launch_ang)])

        ss = 0.0 if spin_side is None else spin_side[winner]
        sb = 0.0 if spin_back is None else spin_back[winner]
        spin_cap = jnp.where(header, e_cfg.spin_head_cap, jnp.where(is_chest, e_cfg.spin_chest_cap, 1.0))
        lateral = jnp.array([-kick_dir[DIM_Y], kick_dir[DIM_X], 0.0])
        # 사이드스핀(수직축)은 180° 회전 불변 → attack_dir 언폴딩 없이 직접 적용(B1, _decode 규약과 짝).
        # 백스핀은 진행방향 lateral축이라 프레임 무관.
        kick_spin = e_cfg.spin_max * spin_cap * ((-sb) * lateral + ss * jnp.array([0.0, 0.0, 1.0]))

        win_team = state.team_id[winner].astype(jnp.int32)
        attack_dir = state.attack_dir[winner]
        # ★속력 규약 = xy(수평)만 — GK 캐치/parry/굴절 속도(gk_catch_speed_cap·deflect_out_*)가 이
        # xy 규약으로 세팅·캘리브됐다. 3D(vz 포함)로 바꾸면 급강하 공을 더 어렵게 보지만 캡이 미보정
        # 상태가 되어 catch↓·parry↑로 조용히 치우친다 → 재캘리브 없인 xy 유지가 충실(감사 B9 결정).
        ball_speed0 = jnp.linalg.norm(state.ball_vel[:DIM_Z])
        poss = state.poss_team
        ff = jnp.bool_(False) if forced_freeplay is None else forced_freeplay
        has_possessor = (poss >= 0) & jnp.any(
            (state.team_id == poss) & state.active_player
        )
        # poss_team만 남고 해당 팀 온피치 선수가 전무한 극단 상태는 루즈볼로 취급한다.
        # 그렇지 않으면 argmin(all inf)=0인 유령 캐리어를 상대로 태클/파울이 발생한다.
        opp_poss = has_possessor & (win_team != poss) & (~ff)     # 복원 pin: 분기 추첨 무력화
        locked = opp_poss & (state.ctrl_lock_t[winner] > 0)        # 승자(도전자)가 재탈취 지연 중
        restart_active = state.restart_t > 0

        # ── GK 캐치/홀드/parry (자기 박스 안 GK가 라이브 공을 잡음) ──────────────
        # env가 '자기 박스 안 GK'를 want_f2b 없이도 경합 후보로 넣으므로(리액티브 클레임) GK가 승자일 수 있다.
        # 자기 박스 GK가 라이브 공을 이기면: ①직전 터치가 같은팀 발패스(백패스)→반칙·간접FK
        # ②잡을 수 있으면(저속)→캐치 홀드(RK_GK_HOLD) ③너무 빠르면→parry(쳐냄). 아래 free_play·contest에서 배제.
        winner_is_gk = state.gk_indices[winner] == 1
        w_in_own_box = self._in_own_box(state.player_pos[winner], attack_dir, clamp_x=True)
        # want_kick(승자의 킥 의도)가 있으면 캐치로 하이재킹하지 않는다 — GK가 발로 클리어하려는
        # 공(백패스 포함 — 발 플레이는 IFAB상 합법)을 손 캐치/백패스 IDFK로 바꿔치기하면 안 됨.
        # 리액티브 클레임(무의도)은 movement가 합법 캐치일 때만 후보로 올린다(재터치/백패스 제외).
        winner_wants_kick = jnp.bool_(False) if want_kick is None else want_kick[winner]
        gk_claim = (any_cand & winner_is_gk & w_in_own_box & (~winner_wants_kick)
                    & (state.ball_state == BALL_ALIVE) & (~restart_active))
        gk_backpass = (gk_claim & (state.last_touch_team == win_team)
                       & ((state.last_touch_code == TOUCH_PASS)
                          | (state.last_touch_code == TOUCH_DRIBBLE)))   # movement 게이트와 동일 정의
        gk_hold_catch = gk_claim & (~gk_backpass) & (ball_speed0 <= e_cfg.gk_catch_speed_cap)
        gk_parry = gk_claim & (~gk_backpass) & (ball_speed0 > e_cfg.gk_catch_speed_cap)

        contest = any_cand & opp_poss & (~locked) & (~gk_claim)

        # 태클 파울 — 도전 수비수(fouler)가 캐리어를 반칙. [#10c] 도전자가 경합을 이기든(opp_poss)
        # 지든(possessor가 공 유지) 반칙은 그 도전 수비수가 저지른다 → '파울하고 공 뺏김'을 표현.
        dist_ball_all = jnp.linalg.norm(state.player_pos - state.ball_pos[:DIM_Z][None, :], axis=1)
        carrier = jnp.argmin(jnp.where((state.team_id == poss) & state.active_player, dist_ball_all, jnp.inf))
        carrier_pos = state.player_pos[carrier]
        # 도전 수비수: opp_poss면 argmax 승자(winner), 아니면 캐리어 최근접 상대(chal — 지고도 파울).
        opp_of_poss = (state.team_id != poss) & state.active_player & (poss >= 0)
        d_carrier_all = jnp.linalg.norm(state.player_pos - carrier_pos[None, :], axis=1)
        chal = jnp.argmin(jnp.where(opp_of_poss, d_carrier_all, jnp.inf))
        chal_exists = opp_of_poss[chal]
        fouler = jnp.where(opp_poss, winner, chal).astype(jnp.int32)
        fpx, fpy = state.player_pos[fouler, DIM_X], state.player_pos[fouler, DIM_Y]
        # [버그수정 #10c] 박스·페널티 기하는 **파울러 자기 공격방향** 기준. attack_dir(=winner)은 킥용이라,
        # loser-foul(fouler=chal=상대팀)에선 부호가 반대 → 박스판정·페널티스폿이 반대 골대에 찍히는 버그.
        foul_dir = state.attack_dir[fouler]
        carrier_face_dir = jnp.array([jnp.cos(state.player_facing[carrier]), jnp.sin(state.player_facing[carrier])])
        fouler_from_carrier = _unit((state.player_pos[fouler] - carrier_pos)[None, :])[0]
        behind = jnp.clip(-jnp.dot(fouler_from_carrier, carrier_face_dir), 0.0, 1.0)
        to_carrier = _unit((carrier_pos - state.player_pos[fouler])[None, :])[0]
        v_close = jnp.clip(
            jnp.dot(state.player_vel[fouler], to_carrier),
            0.0,
            e_cfg.norm_player_vel,
        )
        ball_dist = jnp.linalg.norm(state.ball_pos[:DIM_Z] - state.player_pos[fouler])
        clean_win = (ball_dist < f_cfg.clean_ball_dist) & (ball_speed0 < f_cfg.clean_ball_speed)

        fouler_in_box = self._in_own_box(state.player_pos[fouler], foul_dir, clamp_x=False)
        box_penalty_bias = jnp.where(fouler_in_box, f_cfg.box_foul_bias, 0.0)
        air_bias = jnp.where(header, f_cfg.header_foul_bias, 0.0)
        logit = (f_cfg.tackle_bias + f_cfg.k_close * v_close + f_cfg.k_behind * behind
                 + f_cfg.k_balldist * jnp.maximum(0.0, ball_dist - f_cfg.balldist_ref)
                 - f_cfg.k_clean * clean_win.astype(jnp.float32) + box_penalty_bias + air_bias)
        p_foul = jnp.clip(jax.nn.sigmoid(logit), f_cfg.tackle_p_min, f_cfg.tackle_p_max)
        contact_range = jnp.linalg.norm(state.player_pos[fouler] - carrier_pos) < f_cfg.tackle_contact_range
        # loser-foul: possessor가 공 유지(~opp_poss)해도 근접 도전자가 **돌진(lunge)**하면 파울 가능.
        # 상시 근접이 아닌 실제 태클 시도만 파울화(폭증 방지). recon-안전: any_cand & ~ff 게이트로 재구성 결정성 유지.
        lunge = v_close > e_cfg.charge_speed
        retained = (has_possessor & any_cand & (~opp_poss) & (~ff) & (~gk_claim) & (~restart_active)
                    & (state.ball_state == BALL_ALIVE) & chal_exists & lunge)
        foul = (contest | retained) & contact_range & (jax.random.uniform(k_foul) < p_foul)
        # loser-foul: 파울러가 경합 **패자**인 분기(retained). 승자는 파울을 '당한' 쪽이라
        # 아래 터치 라벨·last_touch 갱신에서 별도 취급해야 한다(contest-foul은 승자=파울러).
        loser_foul = foul & (~opp_poss)
        tackle_ok = contest & (~foul) & (jax.random.uniform(k_tackle) < e_cfg.tackle_prob)
        deflect = contest & (~foul) & (~tackle_ok) & (jax.random.uniform(k_deflect) < e_cfg.deflect_prob)
        free_play = any_cand & (~opp_poss) & (~gk_claim)
        consume = free_play & restart_active & (win_team == state.restart_team)

        # 세트피스 침범 시 킥 무효화 + 재실행(retake). 판정 기하는 restart._encroach_geometry가 단일 진실원천.
        # GK 홀드 배급은 이격 규정이 없어(상대는 도전만 불가) retake 대상에서 제외한다.
        # 퀵 프리킥(2026-07-16, IFAB Law 13 정합): FK(오프사이드 FK 포함)는 '차는 행위 = 이격
        # 요구 포기' — 근접 상대가 있어도 속행(가로채이면 찬 팀의 리스크, 실측 릴리즈 순간 침범
        # 49%가 전부 속행). retake 면제해도 데드볼 강탈은 재개 킥 게이트(지정 키커만)가 막고,
        # 이격 후퇴 압력은 obs margin에 그대로 남는다. 킥오프(Law 8)·코너·스로인은 현행 유지.
        _, encroachers, _ = self._encroach_geometry(state)
        sup_rt = jnp.bool_(False) if suppress_retake is None else jnp.bool_(suppress_retake)  # python True 방어
        quick_ok = ((state.restart_kind == RK_FREEKICK) | (state.restart_kind == RK_OFFSIDE))
        retake = (consume & (state.restart_kind != RK_PENALTY)
                  & (state.restart_kind != RK_GK_HOLD) & (~quick_ok)
                  & jnp.any(encroachers) & (~sup_rt))
        eff_consume = consume & (~retake)

        stop_ball = foul
        kick_vel_tackle = kick_vel * jnp.minimum(
            1.0,
            e_cfg.tackle_out_cap * e_cfg.f2b_speed_max
            / (jnp.linalg.norm(kick_vel) + DIV_EPS),
        )
        new_vel = jnp.where(free_play, kick_vel, jnp.where(tackle_ok, kick_vel_tackle, state.ball_vel))
        new_vel = jnp.where(stop_ball, jnp.zeros(DIM_ALL), new_vel)

        # 굴절 출구: 진행방향(정지면 도전자→공 방향)에서 설정된 각도 범위로 랜덤 회전.
        ball_dir = jnp.where(
            ball_speed0 > e_cfg.deflect_stationary_speed,
            state.ball_vel[:DIM_Z] / (ball_speed0 + DIV_EPS),
                             _unit((state.ball_pos[:DIM_Z] - state.player_pos[winner])[None, :])[0])
        defl_ang = (2.0 * jax.random.uniform(k_defl_dir) - 1.0) * e_cfg.deflect_angle_max
        cos_a, sin_a = jnp.cos(defl_ang), jnp.sin(defl_ang)
        defl_dir = jnp.array([cos_a * ball_dir[DIM_X] - sin_a * ball_dir[DIM_Y],
                              sin_a * ball_dir[DIM_X] + cos_a * ball_dir[DIM_Y]])
        defl_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        vel_deflect = jnp.array([
            defl_dir[DIM_X] * defl_speed,
            defl_dir[DIM_Y] * defl_speed,
            e_cfg.deflect_lift_frac * defl_speed,
        ])
        new_vel = jnp.where(deflect, vel_deflect, new_vel)

        new_spin = jnp.where(free_play | tackle_ok, kick_spin, state.ball_spin)
        new_spin = jnp.where(eff_consume & (state.restart_kind == RK_THROWIN), jnp.zeros(DIM_ALL), new_spin)
        new_spin = jnp.where(stop_ball | retake | deflect, jnp.zeros(DIM_ALL), new_spin)
        new_poss = jnp.where(free_play | tackle_ok, win_team, poss)

        opp_goal_x = attack_dir * self.hx
        px, py = state.player_pos[winner, DIM_X], state.player_pos[winner, DIM_Y]
        to_goal = jnp.array([opp_goal_x - px, -py])
        dist_goal = jnp.linalg.norm(to_goal)
        aim_goal = jnp.dot(kick_dir, to_goal / (dist_goal + DIV_EPS)) > e_cfg.shot_aim_cos
        is_shot = free_play & (dist_goal < e_cfg.f2b_shoot_range) & aim_goal
        fast = ball_speed0 > e_cfg.intercept_speed
        is_dribble = (free_play & (win_team == poss) & (poss >= 0)
                      & (~header) & (~is_shot) & (speed < e_cfg.dribble_speed_max))
        code = jnp.where(tackle_ok & fast, TOUCH_INTERCEPT,
                jnp.where(tackle_ok, TOUCH_TACKLE,
                 jnp.where(is_shot, jnp.where(header, TOUCH_SHOOT_HEAD, TOUCH_SHOOT),
                  jnp.where(free_play,
                   jnp.where(is_dribble, TOUCH_DRIBBLE,
                    jnp.where(header, TOUCH_PASS_HEAD, TOUCH_PASS)), TOUCH_NONE))))
        code = jnp.where(deflect, TOUCH_DEFLECT, code)
        # 태클 파울 라벨은 **경합 승자가 파울러인** contest 분기에만 붙인다.
        # loser-foul(retained)에서 승자는 파울을 당한 쪽이므로 그에게 TOUCH_TACKLE을 찍으면
        # ①'피파울자가 태클했다'는 거짓 라벨이 되고 ②stop_ball로 공이 정지(new_vel=0)했는데도
        # env._kick_applied의 인과킥 집합에 TACKLE이 들어 있어 **거짓 인과킥**이 된다
        # (기존 foul_actor 제외 가드는 파울러만 막으므로 이 경로를 못 잡는다).
        # 휘슬로 플레이가 멎었으므로 승자의 터치는 성립하지 않는다 → TOUCH_NONE.
        code = jnp.where(foul & (~loser_foul), TOUCH_TACKLE, code)
        code = jnp.where(loser_foul, TOUCH_NONE, code)
        # 스로인 테이크는 발킥 분류기(speed=발킥 속도)와 무관 — 실발사는 throw_vel이므로 패스로 고정 라벨
        code = jnp.where(eff_consume & (state.restart_kind == RK_THROWIN), jnp.int32(TOUCH_PASS), code)
        code = jnp.where(retake, TOUCH_NONE, code).astype(jnp.int32)
        touch = state.touch.at[winner].set(jnp.where(any_cand & (code > 0), code, state.touch[winner]))

        # [IFAB Law 14] 페널티는 **반칙 위치**(파울러)로 판정 — 공 위치 무관. 과거 &ball_in_box는
        # 공이 순간 박스 밖이면 박스 안 반칙을 FK로 잘못 강등했다(charge 채널은 이미 제거, 정합).
        in_box = fouler_in_box     # #10c: 파울러(도전 수비수) 위치로 판정
        penalty = foul & in_box
        freekick = foul & (~in_box)
        foul_team = poss
        pen_spot = jnp.array([-foul_dir * (self.hx - e_cfg.penalty_spot), 0.0, self.r_ball])   # [버그수정] 파울러 골대 기준
        boundary_inset = e_cfg.free_kick_boundary_inset
        fk_spot = jnp.array([
            jnp.clip(fpx, -self.hx + boundary_inset, self.hx - boundary_inset),
            jnp.clip(fpy, -self.hy + boundary_inset, self.hy - boundary_inset),
            self.r_ball,
        ])
        restart_kind = jnp.where(penalty, RK_PENALTY, jnp.where(freekick, RK_FREEKICK, state.restart_kind))
        restart_kind = jnp.where(eff_consume, RK_NONE, restart_kind).astype(jnp.int32)
        restart_team = jnp.where(foul, foul_team, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(foul, jnp.where(penalty, e_cfg.penalty_substeps, e_cfg.restart_substeps), state.restart_t)
        restart_t = jnp.where(eff_consume, 0, restart_t)
        restart_t = jnp.where(retake, jnp.int32(e_cfg.restart_substeps), restart_t).astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state)
        ball_state = jnp.where(eff_consume, BALL_ALIVE, ball_state).astype(jnp.int32)
        ball_pos = jnp.where(penalty, pen_spot, jnp.where(freekick, fk_spot, state.ball_pos))
        new_vel = jnp.where(foul | retake, jnp.zeros(DIM_ALL), new_vel)

        is_throw_take = eff_consume & (state.restart_kind == RK_THROWIN)
        throw_speed = f2b_pow[winner] * e_cfg.throw_speed_max
        throw_vel = throw_speed * jnp.array([jnp.cos(launch_ang) * kick_dir[DIM_X],
                                             jnp.cos(launch_ang) * kick_dir[DIM_Y], jnp.sin(launch_ang)])
        hands = jnp.array([px, py, state.head_z[winner] + e_cfg.throw_height])
        new_vel = jnp.where(is_throw_take, throw_vel, new_vel)
        ball_pos = jnp.where(is_throw_take, hands, ball_pos)

        taker_foul = self._designate_taker(state, ball_pos[:DIM_Z], foul_team, jnp.bool_(False))
        pending_taker = jnp.where(foul, taker_foul, state.pending_taker)
        pending_taker = jnp.where(eff_consume, jnp.int32(-1), pending_taker).astype(jnp.int32)

        throw_taker = jnp.where(eff_consume, jnp.int32(-1), state.throw_taker)
        throw_taker = jnp.where(is_throw_take, winner.astype(jnp.int32), throw_taker)
        throw_taker = jnp.where(foul, jnp.int32(-1), throw_taker).astype(jnp.int32)

        # 비스로인 세트피스 키커는 타인 접촉 전 재터치 금지(직접 득점은 허용 → throw_taker와 분리)
        is_sp_take = eff_consume & (state.restart_kind != RK_THROWIN)
        setpiece_taker = jnp.where(eff_consume, jnp.int32(-1), state.setpiece_taker)
        setpiece_taker = jnp.where(is_sp_take, winner.astype(jnp.int32), setpiece_taker)
        setpiece_taker = jnp.where(foul, jnp.int32(-1), setpiece_taker).astype(jnp.int32)

        # 오프사이드 플래그: 소유팀이 공을 '플레이'(패스/슛/드리블)한 순간 전방 동료 재계산.
        is_played = ((code == TOUCH_PASS) | (code == TOUCH_PASS_HEAD)
                     | (code == TOUCH_SHOOT) | (code == TOUCH_SHOOT_HEAD)
                     | (code == TOUCH_DRIBBLE))
        offside_receive = (any_cand & (state.pass_t > 0) & state.offside_flag[winner]
                           & (win_team == state.pass_team))
        # Law 11 오프사이드 예외는 골킥/스로인/코너뿐 — GK 홀드 배급·킥오프도 플래그를 무장해야
        # 수비라인 뒤 상주(체리피킹) 익스플로잇이 안 생긴다(페널티 리바운드 포함).
        offside_restart_kick = eff_consume & ((state.restart_kind == RK_FREEKICK)
                                              | (state.restart_kind == RK_OFFSIDE)
                                              | (state.restart_kind == RK_PENALTY)
                                              | (state.restart_kind == RK_GK_HOLD)
                                              | (state.restart_kind == RK_KICKOFF))
        set_flags = (any_cand & free_play & (win_team == poss) & is_played
                     & ((~restart_active) | offside_restart_kick) & (~offside_receive))
        x_att = state.player_pos[:, DIM_X] * attack_dir
        ball_x_att = state.ball_pos[DIM_X] * attack_dir
        x_def = jnp.where((state.team_id != win_team) & state.active_player, x_att, -jnp.inf)
        line = jax.lax.top_k(x_def, 2)[0][1]                       # 2번째 최종수비
        n_def = jnp.sum(((state.team_id != win_team) & state.active_player).astype(jnp.int32))
        line = jnp.where(n_def >= 2, line, jnp.inf)                # 수비 2명 미만이면 판정 기준 부재
        mate_ahead = ((state.team_id == win_team) & state.active_player & (jnp.arange(self.N) != winner)
                      & (x_att > line + e_cfg.offside_margin) & (x_att > ball_x_att) & (x_att > 0))
        off_flag = jnp.where(set_flags, mate_ahead, state.offside_flag)
        deliberate_def = tackle_ok | foul                          # 의도적 수비 플레이는 라인 리셋
        off_flag = jnp.where(deliberate_def, jnp.zeros(self.N, bool), off_flag)
        pass_team = jnp.where(set_flags, win_team, state.pass_team).astype(jnp.int32)
        pass_t = jnp.where(set_flags, jnp.int32(e_cfg.pass_protect), state.pass_t)

        cooldown = state.cooldown.at[winner].set(
            jnp.where(any_cand, jnp.float32(e_cfg.cooldown_substeps), state.cooldown[winner]))
        # retake(침범 무효화)는 '접촉'으로 세지 않는다 — 킥이 무효화되어 공은 스폿에 정지하고
        # per-player touch도 code=TOUCH_NONE으로 기록되지 않는데(위 code/touch 참조), touched_real에
        # 남겨 두면 last_touch_team만 키커 팀으로 넘어가고 last_touch_code는 TOUCH_NONE으로 갱신되어
        # obs가 '방금 누가 찼는데 터치 종류는 없음'이라는 모순된 상태를 내보낸다(AUDIT 미수정 관찰 1).
        # loser_foul도 제외 — 승자 코드가 TOUCH_NONE이라 남겨 두면 retake와 똑같이
        # 'last_touch_team만 갱신 + last_touch_code=NONE'인 모순 상태가 obs로 나간다.
        touched_real = (any_cand & (free_play | tackle_ok | foul | deflect)
                        & (~retake) & (~loser_foul))
        last_touch = jnp.where(touched_real, win_team, state.last_touch_team).astype(jnp.int32)
        last_touch_code = jnp.where(touched_real, code, state.last_touch_code).astype(jnp.int32)
        foul_kind = jnp.where(foul, jnp.int32(FOUL_TACKLE),
                              jnp.where(eff_consume, jnp.int32(FOUL_NONE), state.foul_kind))
        # #10c: 파울러(도전 수비수)에 최근접한 소유팀 선수 = 파울 당한 피해자(fouler 위치 기준).
        dist_to_fouler = jnp.linalg.norm(state.player_pos - state.player_pos[fouler][None, :], axis=1)
        victim = jnp.argmin(jnp.where((state.team_id == poss) & (jnp.arange(self.N) != fouler)
                                      & state.active_player, dist_to_fouler, jnp.inf))
        foul_actor = jnp.where(foul, fouler.astype(jnp.int32), state.foul_actor)   # #10c: 파울러=도전 수비수
        foul_victim = jnp.where(foul, victim.astype(jnp.int32), state.foul_victim)

        # 페널티 킥 순간: 침범 기록(팀별 분리는 events 해소부가 team_id로 수행). 킥은 그대로 진행되고
        # 재실행 여부는 이후 궤적의 터미널(골/아웃/세이브)이 관측될 때 IFAB 규칙으로 결정된다.
        pen_kick = eff_consume & (state.restart_kind == RK_PENALTY)
        # 파울로 새 재개가 서면 진행 중이던 페널티 플라이트는 무효 — 클리어하지 않으면 무관한
        # 후속 국면에서 스테일 플래그가 뒤늦은 '재실행'을 오발동한다(스코어·배치 되돌림 오염).
        penalty_flight_team = jnp.where(pen_kick, state.restart_team,
                              jnp.where(foul, jnp.int32(-1), state.penalty_flight_team)).astype(jnp.int32)
        penalty_encroach_mask = jnp.where(pen_kick, encroachers,
                                jnp.where(foul, jnp.zeros_like(encroachers), state.penalty_encroach_mask))

        # ── GK 캐치/홀드/parry/백패스 결과 오버라이드 (gk_claim은 위 free_play·contest에서 배제됨) ──
        gk_x, gk_y = state.player_pos[winner, DIM_X], state.player_pos[winner, DIM_Y]
        opp_team = (1 - win_team).astype(jnp.int32)
        hold_ball_pos = jnp.array([gk_x, gk_y, self.r_ball])                 # 잡은 공은 GK 발밑 정지
        bp_spot = jnp.array([
            jnp.clip(gk_x, -self.hx + boundary_inset, self.hx - boundary_inset),
            jnp.clip(gk_y, -self.hy + boundary_inset, self.hy - boundary_inset),
            self.r_ball,
        ])
        bp_taker = self._designate_taker(state, bp_spot[:DIM_Z], opp_team, jnp.bool_(False))
        # parry: 자기 골 반대(자기 attack_dir 방향)로 쳐냄 + 살짝 위 — 입사속도·attack_dir의 결정 함수(역산 가능).
        parry_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        parry_vec = jnp.array([
            attack_dir * parry_speed,
            state.ball_vel[DIM_Y] * e_cfg.parry_lateral_keep,
            e_cfg.parry_lift_frac * parry_speed,
        ])
        gk_code = jnp.where(gk_parry, jnp.int32(TOUCH_PARRY), jnp.int32(TOUCH_GK_CATCH))
        gk_dead = gk_hold_catch | gk_backpass

        ball_pos = jnp.where(gk_hold_catch, hold_ball_pos, jnp.where(gk_backpass, bp_spot, ball_pos))
        new_vel = jnp.where(gk_dead, jnp.zeros(DIM_ALL), jnp.where(gk_parry, parry_vec, new_vel))
        new_spin = jnp.where(gk_claim, jnp.zeros(DIM_ALL), new_spin)
        ball_state = jnp.where(gk_dead, jnp.int32(BALL_DEAD), ball_state).astype(jnp.int32)
        restart_kind = jnp.where(gk_hold_catch, jnp.int32(RK_GK_HOLD),
                        jnp.where(gk_backpass, jnp.int32(RK_FREEKICK), restart_kind)).astype(jnp.int32)
        restart_team = jnp.where(gk_hold_catch, win_team,
                        jnp.where(gk_backpass, opp_team, restart_team)).astype(jnp.int32)
        restart_t = jnp.where(gk_hold_catch, jnp.int32(e_cfg.gk_hold_substeps),
                     jnp.where(gk_backpass, jnp.int32(e_cfg.restart_substeps), restart_t)).astype(jnp.int32)
        new_poss = jnp.where(gk_hold_catch, win_team,
                    jnp.where(gk_backpass, opp_team, new_poss)).astype(jnp.int32)
        pending_taker = jnp.where(gk_hold_catch, winner.astype(jnp.int32),
                         jnp.where(gk_backpass, bp_taker.astype(jnp.int32), pending_taker)).astype(jnp.int32)
        touch = jnp.where(gk_claim, touch.at[winner].set(gk_code), touch)
        last_touch = jnp.where(gk_claim, win_team, last_touch).astype(jnp.int32)
        last_touch_code = jnp.where(gk_claim, gk_code, last_touch_code).astype(jnp.int32)
        setpiece_taker = jnp.where(gk_dead, jnp.int32(-1), setpiece_taker).astype(jnp.int32)
        throw_taker = jnp.where(gk_dead, jnp.int32(-1), throw_taker).astype(jnp.int32)
        foul_kind = jnp.where(gk_backpass, jnp.int32(FOUL_NONE), foul_kind)  # 백패스=카드 없는 기술 반칙
        # 간접FK 플래그: 백패스=True / 일반 파울·GK 홀드캐치=False(직접) / 그 외 carry(진행 중 IDFK 유지).
        # GK가 진행 중 간접FK(백패스·오프사이드 IDFK)를 잡으면 홀드는 새 직접 재개다 — 클리어하지 않으면
        # 스테일 IDFK가 GK 배급까지 살아 정당한 직접골을 무효화(events indirect_direct)한다.
        restart_indirect = jnp.where(gk_backpass, jnp.bool_(True),
                            jnp.where(foul | gk_hold_catch, jnp.bool_(False), state.restart_indirect))

        state = state._replace(ball_pos=ball_pos, ball_vel=new_vel, ball_spin=new_spin,
                               poss_team=new_poss.astype(jnp.int32), cooldown=cooldown,
                               last_touch_team=last_touch, touch=touch, foul_kind=foul_kind,
                               foul_actor=foul_actor, foul_victim=foul_victim,
                               restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
                               ball_state=ball_state, pending_taker=pending_taker, offside_flag=off_flag,
                               pass_team=pass_team, pass_t=pass_t, throw_taker=throw_taker,
                               setpiece_taker=setpiece_taker,
                               penalty_flight_team=penalty_flight_team,
                               penalty_encroach_mask=penalty_encroach_mask,
                               last_touch_code=last_touch_code, restart_indirect=restart_indirect)
        card_mask = ((jnp.arange(self.N) == fouler) & foul) | (encroachers & retake)   # #10c: 카드=파울러
        state = self._draw_cards(state, card_mask, k_card)
        return state
