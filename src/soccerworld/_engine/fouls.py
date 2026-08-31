"""반칙 규율 — 확률적 카드/퇴장(_draw_cards)과 차징 파울(_charge_foul).

차징: 공 보유자에게 규정 이상 속도로 돌진하면 로지스틱 확률로 파울 → FK/페널티(데드볼).
설계상 파울 판정은 '결정적 로짓 → 단일 uniform draw' 한 번이라, 관측된 재개(restart_kind·
ball_state·foul_kind)로부터 파울 발생 여부와 그 draw의 반공간을 그대로 역산할 수 있다.
카드도 카드추첨·색추첨 두 draw로 분리돼 퇴장 결과에서 각 분기를 특정 가능.
"""
import jax
import jax.numpy as jnp

from soccerworld.core.randomness import RandomEvent, select_random_key

from .constants import (
    BALL_ALIVE,
    BALL_DEAD,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DISCIPLINE_OUTCOMES,
    DISCIPLINE_RED,
    DISCIPLINE_SAMPLE,
    DISCIPLINE_YELLOW,
    FOUL_CHARGE,
    FOUL_NONE,
    INJECTABLE_FOUL_KINDS,
    NO_PLAYER,
    NO_TEAM,
    RK_FREEKICK,
    RK_PENALTY,
    TEAM_1,
    YELLOW_CARD_SEND_OFF_COUNT,
)
from .restart import restart_timer_active
from .spatial import _unit
from .timebase import DEFAULT_MATCH_DURATION_SECONDS


def _relative_closing_speed(player_pos, player_vel, carrier):
    """Project velocity relative to the carrier onto each approach direction."""

    carrier_pos = player_pos[carrier]
    to_carrier = _unit(carrier_pos[None, :] - player_pos)
    relative_velocity = player_vel - player_vel[carrier][None, :]
    return jnp.sum(relative_velocity * to_carrier, axis=1)


class Fouls:
    def _normalize_foul_latch(self, state):
        """Keep ``foul_kind`` and its actor/victim identities coherent.

        The identities describe the currently latched foul, not a record of
        past ones.  Centralizing the clear rule prevents a future restart
        writer from clearing only the kind and silently exposing stale IDs.
        """

        clear = state.foul_kind == FOUL_NONE
        return state._replace(
            foul_actor=jnp.where(clear, jnp.int32(NO_PLAYER), state.foul_actor),
            foul_victim=jnp.where(clear, jnp.int32(NO_PLAYER), state.foul_victim),
        )

    def _card_probability(self, state, foul_mask, foul_pos):
        """Return the fitted card probability for one declared contact foul.

        The K-League fit uses only features that the simulator and event feed
        define identically: the offence location folded into the fouler's
        attack direction and elapsed physical match time. Contact severity is
        intentionally not assigned an unmeasured coefficient.
        """

        f_cfg = self.f_cfg
        fouler = jnp.argmax(foul_mask)
        progress = jnp.clip(
            0.5 * (foul_pos[DIM_X] * state.attack_dir[fouler] / self.hx + 1.0),
            0.0,
            1.0,
        )
        elapsed_fraction = jnp.clip(
            state.t
            / jnp.asarray(
                self.control_fps * DEFAULT_MATCH_DURATION_SECONDS,
                dtype=jnp.float32,
            ),
            0.0,
            1.0,
        )
        reference = jnp.asarray(f_cfg.card_per_foul, dtype=jnp.float32)
        safe_reference = jnp.clip(reference, 1.0e-6, 1.0 - 1.0e-6)
        reference_logit = jnp.log(safe_reference) - jnp.log1p(-safe_reference)
        logit = (
            reference_logit
            + f_cfg.card_attack_progress_logit_weight * (progress - 0.5)
            + f_cfg.card_elapsed_fraction_logit_weight
            * (elapsed_fraction - 0.5)
        )
        contextual = jax.nn.sigmoid(logit)
        return jnp.where(
            reference <= 0.0,
            jnp.float32(0.0),
            jnp.where(reference >= 1.0, jnp.float32(1.0), contextual),
        )

    def _draw_cards(
        self,
        state,
        foul_mask,
        key,
        *,
        foul_pos=None,
        discipline=DISCIPLINE_SAMPLE,
        randomness=None,
        substep_index=0,
        card_event=RandomEvent.TACKLE_CARD,
        color_event=RandomEvent.TACKLE_CARD_COLOR,
    ):
        """Apply a sampled or observed disciplinary outcome to the fouler.

        Ordinary play samples a contextual card and then the aggregate direct
        red rate. Reconstruction may pin none/yellow/red while consuming the
        same RNG draws, so downstream streams remain stable.
        """
        f_cfg = self.f_cfg
        if foul_pos is None:
            # Keep the private helper source-compatible for diagnostics and
            # downstream subclasses.  Engine call sites always pass the
            # physical contact position explicitly.
            foul_pos = state.ball_pos
        k_card, k_color = jax.random.split(key)
        k_card = select_random_key(
            randomness, card_event, substep_index, k_card
        )
        k_color = select_random_key(
            randomness, color_event, substep_index, k_color
        )
        eligible = foul_mask & state.active_player                 # 이미 퇴장한 선수는 제외
        card_probability = self._card_probability(state, eligible, foul_pos)
        sampled_carded = eligible & (
            jax.random.uniform(k_card, (self.N,)) < card_probability
        )
        sampled_red = sampled_carded & (
            jax.random.uniform(k_color, (self.N,)) < f_cfg.red_given_card
        )
        discipline = jnp.asarray(discipline, jnp.int32)
        sample = discipline == DISCIPLINE_SAMPLE
        forced_card = eligible & (
            (discipline == DISCIPLINE_YELLOW) | (discipline == DISCIPLINE_RED)
        )
        carded = jnp.where(sample, sampled_carded, forced_card)
        red = jnp.where(
            sample,
            sampled_red,
            eligible & (discipline == DISCIPLINE_RED),
        )
        yellows = state.yellow_cards + (carded & (~red)).astype(jnp.int32)
        sent_off = state.sent_off | red | (yellows >= YELLOW_CARD_SEND_OFF_COUNT)
        on_pitch = state.on_pitch & (~sent_off)
        updated = state._replace(
            yellow_cards=yellows,
            sent_off=sent_off,
            on_pitch=on_pitch,
        )
        newly_inactive = jnp.any(state.active_player & (~updated.active_player))
        return jax.lax.cond(
            newly_inactive,
            self._project_inactive_players,
            lambda current: current,
            updated,
        )

    def _charge_foul(
        self,
        state,
        key,
        suppress=None,
        inject=None,
        *,
        randomness=None,
        substep_index=0,
    ):
        """공 보유자에게 무리한 돌진(차징) → 확률적 파울. 파울이면 공 데드 + FK/페널티 배치.

        가해자 = 보유팀 캐리어에게 규정속도(charge_speed) 이상 접근 중인 최근접 상대.
        로짓은 접근속도·등뒤·어깨싸움·볼 미플레이로 구성, sigmoid→[charge_p_min, charge_p_max] 클립.
        피해 캐리어의 접촉 위치가 가해자 자기 박스면 페널티, 아니면 프리킥.

        suppress: reconstruct용 추첨 봉쇄 pin(스칼라 bool). True면 파울 추첨을 '불발'로 고정 —
        관측에 없는 파울이 창을 탈선시키지 않게. 추첨(uniform)은 그대로 소비해 RNG 열 불변.
        None/False면 포워드 불변.

        inject: ``suppress``의 짝인 주입 pin(dict 또는 None). suppress가 관측에 **없는** 파울을
        지우듯, inject는 관측에 **있는데** 이 추상 모델이 만들 수 없는 파울을 넣는다. 핸드볼·
        오프볼 홀딩·공격 측 파울은 확률이 낮아서가 아니라 ``active``(근접·접근속도·비소유팀)가
        False라 **어떤 키로도 나오지 않는다** — 추첨 pin만으로는 복원이 한 방향으로만 닫힌다.

            {"actor": int32, "victim": int32, "pos": float32[2],
             "kind": int32(선택), "discipline": int32(선택)}

        ``actor``가 음수면 주입하지 않는다. ``victim``은 핸드볼처럼 피해자가 없으면 ``NO_PLAYER``.
        ``pos``는 접촉 위치이며 FK 스폿과 PK 판정에 함께 쓰인다(자연 경로의 ``carrier_pos``와 같은
        규약). 주입은 **가해자가 수비 측이라고 가정하지 않는다** — 재개는 가해자의 반대 팀에게 간다.
        추첨은 그대로 소비하므로 RNG 열은 불변이고, ``inject=None``이면 포워드가 비트 단위로 불변이다.
        """
        e_cfg = self.e_cfg
        f_cfg = self.f_cfg
        key, k_card = jax.random.split(key)
        key = select_random_key(
            randomness, RandomEvent.CHARGE_FOUL, substep_index, key
        )
        poss = state.poss_team
        has_poss = (poss >= 0) & jnp.any(
            (state.team_id == poss) & state.active_player
        )
        ball_xy = state.ball_pos[:DIM_Z]

        dist_ball = jnp.linalg.norm(state.player_pos - ball_xy[None, :], axis=1)
        carrier = jnp.argmin(jnp.where((state.team_id == poss) & state.active_player, dist_ball, jnp.inf))
        carrier_pos = state.player_pos[carrier]
        dist_carrier = jnp.linalg.norm(state.player_pos - carrier_pos[None, :], axis=1)
        # Closing speed is relative to the carrier.  Absolute challenger speed marks two
        # players running shoulder-to-shoulder at the same velocity as a high-speed charge.
        closing = _relative_closing_speed(
            state.player_pos,
            state.player_vel,
            carrier,
        )
        is_charger = ((state.team_id != poss) & state.active_player
                      & (
                          dist_carrier
                          < (2.0 * self.r_player + f_cfg.charge_contact_padding)
                      )
                      & (closing > e_cfg.charge_speed) & has_poss
                      & (state.ball_state == BALL_ALIVE)
                      & (~restart_timer_active(state.restart_t)))
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

        # ── 관측 파울 주입(재구성 전용) ────────────────────────────────────────
        # 이 함수는 substep scan 안에서 호출되므로 주입도 라이브·비재개일 때만 연다.
        # 첫 발화가 공을 데드로 만들면 같은 control frame의 남은 substep은 자동으로 닫힌다.
        if inject is None:
            inj_on = jnp.bool_(False)
            inj_actor = jnp.int32(NO_PLAYER)
            inj_victim = jnp.int32(NO_PLAYER)
            inj_pos = jnp.zeros(DIM_Z, jnp.float32)
            inj_kind = jnp.int32(FOUL_CHARGE)
            inj_discipline = jnp.int32(DISCIPLINE_SAMPLE)
        else:
            inj_actor = jnp.asarray(inject["actor"], jnp.int32)
            inj_victim = jnp.asarray(inject["victim"], jnp.int32)
            inj_pos = jnp.asarray(inject["pos"], jnp.float32)
            inj_kind = jnp.asarray(inject.get("kind", FOUL_CHARGE), jnp.int32)
            discipline_input = jnp.asarray(
                inject.get("discipline", DISCIPLINE_SAMPLE)
            )
            discipline_type_ok = (
                discipline_input.shape == ()
                and jnp.issubdtype(discipline_input.dtype, jnp.signedinteger)
            )
            inj_discipline = (
                discipline_input.astype(jnp.int32)
                if discipline_type_ok
                else jnp.int32(2**30)
            )
            safe_inj = jnp.clip(inj_actor, 0, self.N - 1)
            # 값 검증은 호스트(``step_env_array``)와 여기 두 층에 있다. 호스트는 구체
            # 값만 볼 수 있으므로 트레이서로 들어온 주입은 여기서 fail-closed해야 한다 —
            # 검사 없이 통과시키면 victim=999가 그대로 State에, kind=999가 BC 라벨에
            # 남고 pos=NaN은 중앙 프리킥으로 둔갑한다. 두 층은 같은 집합/범위를 본다.
            inj_kind_ok = jnp.any(jnp.stack([
                inj_kind == jnp.int32(code)
                for code in sorted(INJECTABLE_FOUL_KINDS)
            ]))
            inj_discipline_ok = jnp.any(jnp.stack([
                inj_discipline == jnp.int32(code)
                for code in sorted(DISCIPLINE_OUTCOMES)
            ]))
            inj_on = (
                (inj_actor >= 0) & (inj_actor < self.N)
                & (inj_victim >= NO_PLAYER) & (inj_victim < self.N)
                & inj_kind_ok
                & inj_discipline_ok
                & jnp.all(jnp.isfinite(inj_pos))
                & state.active_player[safe_inj]
                & (state.ball_state == BALL_ALIVE)
                & (~restart_timer_active(state.restart_t))
            )

        foul = jnp.where(inj_on, jnp.bool_(True), foul)
        idx = jnp.where(inj_on, inj_actor, idx)
        # 피해자·접촉위치·재개팀은 주입 시 관측값을 그대로 쓴다.
        victim_eff = jnp.where(inj_on, inj_victim, carrier.astype(jnp.int32))
        foul_pos = jnp.where(inj_on, inj_pos, carrier_pos)
        kind_eff = jnp.where(inj_on, inj_kind, jnp.int32(FOUL_CHARGE))
        # 자연 경로는 가해자가 항상 비소유팀이라 재개가 poss로 가지만, 주입은 공격 측
        # 파울도 표현할 수 있어야 하므로 **가해자의 반대 팀**을 SSOT로 쓴다.
        fouled_team = jnp.where(
            inj_on, jnp.int32(TEAM_1) - state.team_id[idx], poss
        ).astype(jnp.int32)

        attack_dir = state.attack_dir[idx]
        px, py = foul_pos[DIM_X], foul_pos[DIM_Y]
        # 캐리어 위치는 이 추상 차징 모델의 접촉점 proxy다. FK 스폿과 페널티 여부가
        # 같은 위치를 사용해야 PA 경계에서 서로 모순되지 않는다. 공 위치는 무관하다.
        # 선수 이동은 골라인 뒤 5m까지 가능하다. Law 12의 경기장 밖 반칙 규정에
        # 따라 가장 가까운 경계점이 자기 페널티구역 골라인이면 페널티로 분류한다.
        in_box = self._foul_in_own_box(foul_pos, attack_dir)
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
        restart_team = jnp.where(foul, fouled_team, state.restart_team).astype(jnp.int32)
        restart_t = jnp.where(foul, jnp.where(penalty, e_cfg.penalty_substeps, e_cfg.restart_substeps), state.restart_t).astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state).astype(jnp.int32)
        ball_pos = jnp.where(penalty, pen_spot, jnp.where(freekick, fk_spot, state.ball_pos))
        ball_vel = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_vel)
        ball_spin = jnp.where(foul, jnp.zeros(DIM_ALL), state.ball_spin)

        # 파울 재개는 페널티 또는 직접 프리킥이다 — 이 구분이 키커를 가른다.
        taker = self._designate_taker_when(
            foul,
            state, ball_pos[:DIM_Z], fouled_team, jnp.bool_(False),
            jnp.where(penalty, jnp.int32(RK_PENALTY), jnp.int32(RK_FREEKICK)),
            jnp.bool_(False))
        pending_taker = jnp.where(foul, taker, state.pending_taker).astype(jnp.int32)
        foul_kind = jnp.where(foul, kind_eff, state.foul_kind)
        foul_actor = jnp.where(foul, idx.astype(jnp.int32), state.foul_actor)
        foul_victim = jnp.where(foul, victim_eff, state.foul_victim)
        throw_taker = jnp.where(foul, jnp.int32(-1), state.throw_taker).astype(jnp.int32)
        setpiece_taker = jnp.where(foul, jnp.int32(-1), state.setpiece_taker).astype(jnp.int32)

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
                               restart_indirect=restart_indirect,
                               gk_handling_restricted_team=jnp.where(
                                   foul, jnp.int32(NO_TEAM),
                                   state.gk_handling_restricted_team,
                               ).astype(jnp.int32),
                               offside_flag=off_clear, pass_t=pass_t_clear, pass_team=pass_team_clear)
        injected_discipline = jnp.where(
            inj_on,
            inj_discipline,
            jnp.int32(DISCIPLINE_SAMPLE),
        )
        state = self._draw_cards(
            state,
            (jnp.arange(self.N) == idx) & foul,
            k_card,
            foul_pos=foul_pos,
            discipline=injected_discipline,
            randomness=randomness,
            substep_index=substep_index,
            card_event=RandomEvent.CHARGE_CARD,
            color_event=RandomEvent.CHARGE_CARD_COLOR,
        )
        return self._normalize_pass_latch(state, clear=foul)
