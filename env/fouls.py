"""반칙 규율 — 확률적 카드/퇴장(_draw_cards)과 차징 파울(_charge_foul).

차징: 공 보유자에게 규정 이상 속도로 돌진하면 로지스틱 확률로 파울 → FK/페널티(데드볼).
설계상 파울 판정은 '결정적 로짓 → 단일 uniform draw' 한 번이라, 관측된 재개(restart_kind·
ball_state·foul_kind)로부터 파울 발생 여부와 그 draw의 반공간을 그대로 역산할 수 있다.
카드도 카드추첨·색추첨 두 draw로 분리돼 퇴장 결과에서 각 분기를 특정 가능.
"""
import jax
import jax.numpy as jnp

from constants import *
from spatial import _unit


class Fouls:
    def _draw_cards(self, state, foul_mask, key):
        """foul_mask(bool[N]) 선수에게 확률적 카드 → 퇴장 갱신. 태클·차징·세트피스 침범 공용.

        파울당 card_per_foul 확률로 카드, 카드면 red_given_card 확률로 레드(아니면 옐로).
        레드 1장 또는 옐로 누적 2장이면 퇴장(active_player=False). 보상 무관(순수 규율).
        """
        f_cfg = self.f_cfg
        k_card, k_color = jax.random.split(key)
        eligible = foul_mask & state.active_player                 # 이미 퇴장한 선수는 제외
        carded = eligible & (jax.random.uniform(k_card, (self.N,)) < f_cfg.card_per_foul)
        red = carded & (jax.random.uniform(k_color, (self.N,)) < f_cfg.red_given_card)
        yellows = state.yellow_cards + (carded & (~red)).astype(jnp.int32)
        sent_off = (
            (~state.active_player)
            | red
            | (yellows >= YELLOW_CARD_SEND_OFF_COUNT)
        )
        return state._replace(yellow_cards=yellows, active_player=~sent_off)

    def _charge_foul(self, state, key, suppress=None):
        """공 보유자에게 무리한 돌진(차징) → 확률적 파울. 파울이면 공 데드 + FK/페널티 배치.

        가해자 = 보유팀 캐리어에게 규정속도(charge_speed) 이상 접근 중인 최근접 상대.
        로짓은 접근속도·등뒤·어깨싸움·볼 미플레이로 구성, sigmoid→[charge_p_min, charge_p_max] 클립.
        가해자 자기 박스 + 공도 박스 안이면 페널티, 아니면 프리킥.

        suppress: reconstruct용 추첨 봉쇄 pin(스칼라 bool). True면 파울 추첨을 '불발'로 고정 —
        관측에 없는 파울이 창을 탈선시키지 않게. 추첨(uniform)은 그대로 소비해 RNG 열 불변.
        None/False면 포워드 불변.
        """
        e_cfg = self.e_cfg
        f_cfg = self.f_cfg
        key, k_card = jax.random.split(key)
        poss = state.poss_team
        has_poss = (poss >= 0) & jnp.any(
            (state.team_id == poss) & state.active_player
        )
        ball_xy = state.ball_pos[:DIM_Z]

        dist_ball = jnp.linalg.norm(state.player_pos - ball_xy[None, :], axis=1)
        carrier = jnp.argmin(jnp.where((state.team_id == poss) & state.active_player, dist_ball, jnp.inf))
        carrier_pos = state.player_pos[carrier]
        dist_carrier = jnp.linalg.norm(state.player_pos - carrier_pos[None, :], axis=1)
        to_carrier = _unit(carrier_pos[None, :] - state.player_pos)
        closing = jnp.sum(state.player_vel * to_carrier, axis=1)
        is_charger = ((state.team_id != poss) & state.active_player
                      & (
                          dist_carrier
                          < (2.0 * self.r_player + f_cfg.charge_contact_padding)
                      )
                      & (closing > e_cfg.charge_speed) & has_poss
                      & (state.ball_state == BALL_ALIVE) & (state.restart_t == 0))
        idx = jnp.argmax(jnp.where(is_charger, closing, -jnp.inf))
        active = is_charger[idx]

        carrier_face = state.player_facing[carrier]
        carrier_face_dir = jnp.array([jnp.cos(carrier_face), jnp.sin(carrier_face)])
        to_charger = _unit((state.player_pos[idx] - carrier_pos)[None, :])[0]
        behind = jnp.clip(-jnp.dot(to_charger, carrier_face_dir), 0.0, 1.0)
        vel_align = jnp.clip(jnp.dot(_unit(state.player_vel[idx][None, :])[0],
                                     _unit(state.player_vel[carrier][None, :])[0]), 0.0, 1.0)
        side = 1.0 - jnp.abs(jnp.dot(to_charger, carrier_face_dir))
        shoulder = vel_align * side                                # 나란히 달리는 정당 몸싸움
        ball_far = (jnp.linalg.norm(ball_xy - carrier_pos) > f_cfg.charge_play_dist).astype(jnp.float32)
        logit = (f_cfg.charge_bias + f_cfg.kc_speed * jnp.maximum(0.0, closing[idx] - e_cfg.charge_speed)
                 + f_cfg.kc_behind * behind - f_cfg.kc_shoulder * shoulder + f_cfg.kc_ballfar * ball_far)
        p_charge = jnp.clip(jax.nn.sigmoid(logit), f_cfg.charge_p_min, f_cfg.charge_p_max)
        sup = jnp.bool_(False) if suppress is None else jnp.bool_(suppress)  # python True 방어(~True=-2 int화)
        foul = active & (~sup) & (jax.random.uniform(key) < p_charge)

        attack_dir = state.attack_dir[idx]
        px, py = carrier_pos[DIM_X], carrier_pos[DIM_Y]
        # 박스 판정은 가해자 위치 기준(contest 태클 파울과 동일 규약) — 접촉거리 <0.75m라 캐리어와
        # 실질 차이는 작지만 판정 주체를 통일. FK 스폿은 반칙 지점(캐리어 위치) 유지가 IFAB 정합.
        # IFAB: 페널티는 '반칙이 일어난 지점'(가해자/접촉 위치)이 박스 안이면 성립 — 공 위치 무관.
        # 예전 & ball_in_box는 경계 근처 차징에서 공이 박스 밖이면 페널티를 FK로 잘못 강등했다(B5).
        in_box = self._in_own_box(state.player_pos[idx], attack_dir, clamp_x=False)
        penalty = foul & in_box
        freekick = foul & (~in_box)

        pen_spot = jnp.array([-attack_dir * (self.hx - e_cfg.penalty_spot), 0.0, self.r_ball])
        inset = e_cfg.free_kick_boundary_inset
        fk_spot = jnp.array([
            jnp.clip(px, -self.hx + inset, self.hx - inset),
            jnp.clip(py, -self.hy + inset, self.hy - inset),
            self.r_ball,
        ])
        restart_kind = jnp.where(penalty, RK_PENALTY, jnp.where(freekick, RK_FREEKICK, state.restart_kind)).astype(jnp.int32)
        restart_team = jnp.where(foul, poss, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(foul, jnp.where(penalty, e_cfg.penalty_substeps, e_cfg.restart_substeps), state.restart_t).astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state).astype(jnp.int32)
        ball_pos = jnp.where(penalty, pen_spot, jnp.where(freekick, fk_spot, state.ball_pos))
        ball_vel = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_spin)

        taker = self._designate_taker(state, ball_pos[:DIM_Z], poss, jnp.bool_(False))
        pending_taker = jnp.where(foul, taker, state.pending_taker).astype(jnp.int32)
        foul_kind = jnp.where(foul, jnp.int32(FOUL_CHARGE), state.foul_kind)
        foul_actor = jnp.where(foul, idx.astype(jnp.int32), state.foul_actor)
        foul_victim = jnp.where(foul, carrier.astype(jnp.int32), state.foul_victim)
        throw_taker = jnp.where(foul, jnp.int32(-1), state.throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(foul, jnp.int32(-1), state.setpiece_taker).astype(jnp.int32)

        # 차징 파울로 새 재개가 서면 진행 중이던 페널티 플라이트는 무효(스테일 재실행 방지)
        pen_flight = jnp.where(foul, jnp.int32(-1), state.penalty_flight_team).astype(jnp.int32)
        pen_enc = jnp.where(foul, jnp.zeros_like(state.penalty_encroach_mask), state.penalty_encroach_mask)
        # 차징 파울은 직접 FK/페널티 → 진행 중이던 간접FK 플래그도 클리어(스테일 direct-골 무효 방지)
        restart_indirect = jnp.where(foul, jnp.bool_(False), state.restart_indirect)
        # 새 재개는 오프사이드 창을 리셋한다(offside.py·events.py·contest.py 전이와 동일 규약).
        # 안 지우면 스테일 offside_flag/pass_t/pass_team가 이 FK/페널티로 넘어가 obs(pass_signal 등)를
        # 오염시킨다(라벨은 pass_t 자체 소진으로 무해하나 잠재 위험 제거).
        off_clear = jnp.where(foul, jnp.zeros(self.N, bool), state.offside_flag)
        pass_t_clear = jnp.where(foul, jnp.int32(0), state.pass_t).astype(jnp.int32)
        pass_team_clear = jnp.where(foul, jnp.int32(-1), state.pass_team).astype(jnp.int32)
        state = state._replace(restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
                               ball_state=ball_state, ball_pos=ball_pos, ball_vel=ball_vel, ball_spin=ball_spin,
                               pending_taker=pending_taker, foul_kind=foul_kind, foul_actor=foul_actor,
                               foul_victim=foul_victim, throw_taker=throw_taker, setpiece_taker=setpiece_taker,
                               penalty_flight_team=pen_flight, penalty_encroach_mask=pen_enc,
                               restart_indirect=restart_indirect,
                               offside_flag=off_clear, pass_t=pass_t_clear, pass_team=pass_team_clear)
        state = self._draw_cards(state, (jnp.arange(self.N) == idx) & foul, k_card)
        return state
