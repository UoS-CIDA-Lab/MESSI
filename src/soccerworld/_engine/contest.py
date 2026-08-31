"""볼 경합 해결 — 승자 선정(_contest_winner)과 킥/탈취/파울/굴절/세트피스 힘 적용(_apply_force2ball).
세트피스 taker 지정(_designate_taker)·발사각 하한(launch_lo)도 포함. 태클 파울·오프사이드 플래그 겸함.

복원 관점: 승자는 궤적에서 '누가 공을 건드렸나'로 직접 관측되고, 파울/탈취/굴절 각 분기는 단일
uniform draw로 갈리며, 굴절각도 단일 draw의 가역 함수다. 따라서 관측 결과(공 속도·소유변화·재개
종류)로부터 어느 분기·어떤 draw였는지 결정적으로 역산할 수 있다(재추첨 매칭 불필요).
"""
import jax
import jax.numpy as jnp
import numpy as np

from soccerworld.core.randomness import RandomEvent, select_random_key

from . import setpiece_taker as setpiece_taker_module
from .constants import (
    BALL_ALIVE,
    BALL_DEAD,
    DIM_ALL,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    FOUL_NONE,
    FOUL_TACKLE,
    GEOMETRY_EPS,
    GK_HANDLING_RELEASE_OFFSET,
    NO_PLAYER,
    NO_TEAM,
    PROB_EPS,
    RESTART_COUNT,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    SAMPLED_WINNER,
    TEAM_0,
    TEAM_1,
    TOUCH_DEFLECT,
    TOUCH_DRIBBLE,
    TOUCH_GK_CATCH,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PARRY,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
)
from .restart import _coerce_public_float32, restart_timer_active
from .spatial import _unit


class Contest:
    def _gk_catch_probability(self, horizontal_speed):
        """GK가 reach에서 공을 잡을 조건부 확률.

        scale=0은 속도 임계 이하만 잡는 하드캡이고, 양수는 K리그 catch/punch 로지스틱
        적합과 같은 매개화다.
        """

        e_cfg = self.e_cfg
        speed = jnp.asarray(horizontal_speed, jnp.float32)
        # Config is immutable Python state, so this branch is static at trace
        # time.  Keeping exp/div/random out of the graph makes the ``scale=0``
        # cap as cheap to compile as a plain comparison.
        if e_cfg.gk_catch_speed_scale <= 0.0:
            return (speed <= e_cfg.gk_catch_speed_cap).astype(jnp.float32)
        return jax.nn.sigmoid(
            (jnp.float32(e_cfg.gk_catch_speed_cap) - speed)
            / jnp.float32(e_cfg.gk_catch_speed_scale)
        )

    def _play_targets_own_goalkeeper(
        self, state, actor, team, direction, launch_speed
    ):
        """Whether a submitted horizontal play is aimed at the actor's own GK.

        Law 12 depends on the team-mate's intent, not on whether an inaccurate
        ball eventually arrives exactly at the goalkeeper.  Direction alone is
        insufficient: a short pass to an intermediate defender may share the
        same ray as the goalkeeper.  Infer the unopposed ground endpoint from
        submitted speed using the rolling-deceleration SSOT, then test that
        endpoint against a configurable disc around the active own GK.  A
        rolling path which physically enters the keeper's claim corridor also
        counts: otherwise a firm pass through the keeper to a farther endpoint
        evades the restriction even though the keeper is its first recipient.

        This remains a broad, calibratable intent proxy.  It does not require
        the realised ball to reach the keeper, so an inaccurate attempted
        back-pass is still restricted while a clearly shorter same-ray pass is
        not.
        """

        own_gk_mask = (
            (state.team_id == team)
            & (state.gk_indices == 1)
            & state.active_player
        )
        own_gk = jnp.argmax(own_gk_mask.astype(jnp.int32))
        to_gk = state.player_pos[own_gk] - state.ball_pos[:DIM_Z]
        direction = _unit(jnp.asarray(direction)[None, :])[0]
        along = jnp.dot(to_gk, direction)
        cross_track = jnp.abs(
            direction[DIM_X] * to_gk[DIM_Y]
            - direction[DIM_Y] * to_gk[DIM_X]
        )
        launch_speed = jnp.maximum(
            jnp.asarray(launch_speed, jnp.float32), 0.0
        )
        roll_decel = jnp.interp(
            launch_speed,
            jnp.asarray(self.e_cfg.roll_v_knots, jnp.float32),
            jnp.asarray(self.e_cfg.roll_d_knots, jnp.float32),
        )
        intended_distance = (
            launch_speed * launch_speed / (2.0 * roll_decel + DIV_EPS)
        )
        endpoint_error = jnp.sqrt(
            cross_track * cross_track
            + (along - intended_distance) * (along - intended_distance)
        )
        # Endpoint-only intent missed the observed 143.3 s back-pass: its
        # unopposed endpoint lay beyond the GK, while the submitted ground path
        # itself crossed the GK's handling reach.  Keep the finite-segment
        # condition so a short pass on the same ray which stops before a more
        # distant goalkeeper remains legal.
        enters_claim_corridor = (
            (cross_track <= self.e_cfg.gk_reach_xy + self.r_ball)
            & (along <= intended_distance + GEOMETRY_EPS)
        )
        return (
            jnp.any(own_gk_mask)
            & (actor != own_gk)
            & (along > GEOMETRY_EPS)
            & (launch_speed > GEOMETRY_EPS)
            & (
                (endpoint_error <= self.e_cfg.gk_backpass_target_radius)
                | enters_claim_corridor
            )
        )

    def _lunge_fraction(self, ball_center_distance):
        """공 표면 기준 carry→challenge 런지 분율(0~1)."""
        e_cfg = self.e_cfg
        surface_distance = jnp.maximum(
            jnp.asarray(ball_center_distance) - self.r_ball,
            0.0,
        )
        return jnp.clip(
            (surface_distance - e_cfg.reach_xy_carry)
            / (e_cfg.reach_xy_challenge - e_cfg.reach_xy_carry + DIV_EPS),
            0.0,
            1.0,
        )

    def _contest_winner(self, state, candidate, dist_xy, key, forced_winner=None):
        """Select a winner, avoiding score/Gumbel work when no slot is eligible.

        Candidate availability is a scalar physical gate.  The no-candidate result is always
        ``(NO_PLAYER, False)``, including for a forced request, because pins cannot bypass reach,
        cooldown, roster, or action gates.  Keep the complete calculation in
        :meth:`_contest_winner_full` as the single active-path implementation and exact oracle.
        """

        candidate = candidate & state.active_player
        any_candidate = jnp.any(candidate)
        return jax.lax.cond(
            any_candidate,
            lambda _: self._contest_winner_full(
                state, candidate, dist_xy, key, forced_winner
            ),
            lambda _: (jnp.int32(NO_PLAYER), jnp.bool_(False)),
            operand=None,
        )

    def _contest_winner_full(
        self, state, candidate, dist_xy, key, forced_winner=None
    ):
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
        # Exactly one sentinel requests sampling.  Every other value is a
        # forced request; an out-of-range request resolves to NO_PLAYER.  Using
        # ``>= NO_PLAYER`` here made invalid negative values (-3, -5, ...)
        # silently fall through to stochastic sampling under JIT, even though
        # step_env_array reported ``forced_winner_invalid=True`` and documented
        # a safe no-winner result.
        use = forced_winner != SAMPLED_WINNER
        forced_ok = ((forced_winner >= 0) & (forced_winner < candidate.shape[0])
                     & candidate[jnp.clip(forced_winner, 0, candidate.shape[0] - 1)])
        eff_forced = jnp.where(
            forced_ok, forced_winner, jnp.int32(NO_PLAYER)
        )   # 후보 밖 pin → 불발
        return (jnp.where(use, eff_forced, sampled_winner),
                jnp.where(use, forced_ok, sampled_any))

    def setpiece_taker_view(self, state, ball_xy, restart_team, goalkeeper_only,
                            restart_kind, restart_indirect=False):
        """키커 결정자가 보는 입력 — 학습 헤드도 **같은 것**을 본다.

        :meth:`substitution_view`·:meth:`formation_view`·:meth:`manager_view`와 같은
        정보 경계다. 종전에는 이 뷰만 :meth:`_designate_taker` 안에 갇혀 있어서, 학습
        키커 헤드를 만들려는 쪽이 입력을 재현할 방법이 없었다 — 결정자를 바꿔 끼울 수
        있다는 계약이 키커 축에서만 반쪽이었다.
        """

        return setpiece_taker_module.SetPieceTakerView(
            restart_kind=jnp.asarray(restart_kind, jnp.int32),
            restart_indirect=jnp.asarray(restart_indirect, bool),
            restart_team=jnp.asarray(restart_team, jnp.int32),
            restart_spot=ball_xy,
            # 프리킥 키커는 재개 지점이 **자기 진영인지 상대 진영인지**로 갈린다(자기
            # 진영은 골키퍼가 61%). 팀 축을 접어야 하므로 재개 팀의 공격 방향을 함께
            # 싣는다. 한 팀의 attack_dir는 전원 같으므로 팀 마스크 평균의 부호면 된다.
            attack_dir=jnp.sign(jnp.sum(jnp.where(
                state.team_id == restart_team, state.attack_dir, 0.0))),
            half_length=jnp.asarray(self.hx, jnp.float32),
            goalkeeper_only=jnp.asarray(goalkeeper_only, bool),
            player_id=state.player_id,
            active_player=state.active_player,
            team_id=state.team_id,
            is_gk=state.gk_indices == 1,
            player_pos=state.player_pos,
            role=setpiece_taker_module.classify_roles(
                self.formation_home(state), state.gk_indices == 1,
                state.team_id, state.active_player),
            stamina_long=state.stamina_long,
            endurance_factor=state.endurance_factor,
            player_ctrl=state.player_ctrl,
            reach_z=state.reach_z,
            plan=self._setpiece_plans,
            plan_slot=self._setpiece_plan_slot(restart_kind, restart_indirect, ball_xy),
        )

    def _designate_taker_when(self, required, state, ball_xy, restart_team,
                              goalkeeper_only, restart_kind,
                              restart_indirect=False, key=None):
        """지정이 **실제로 필요한 사건에서만** 키커 결정자를 돌린다.

        키커 지정은 희소한 사건이다 — 재개가 새로 열리거나, 키커가 교체·퇴장으로 무효가
        되었을 때뿐이다. 그런데 호출부는 재개를 만들 수 있는 **후보 지점** 열 곳에 흩어져
        있고, 전부 결과만 ``jnp.where``로 골랐다. 그러면 사건이 거짓이어도 뷰 조립(규범
        앵커 + 역할 분류 + 슬롯당 15개 배열)과 결정자 추론이 매 서브스텝 그대로 돈다 —
        제어 프레임당 아홉 곳 × decimation회다. 학습 키커 헤드를 꽂으면 그 비용이 전부
        신경망 추론이 된다.

        ``lax.cond``는 양쪽 가지를 **추적**하지만 런타임에는 고른 가지만 **실행**한다.
        그래서 조건을 여기로 옮기면 결과는 한 비트도 달라지지 않고 비용만 사라진다
        (거짓 가지의 값은 호출부의 ``jnp.where``가 어차피 버리던 값이다). 결정자는
        state에서 키를 유도하므로 호출을 건너뛰어도 다른 난수열이 밀리지 않는다.

        vmap 아래에서는 ``lax.cond``가 ``select``로 낮아져 양쪽이 다 실행된다 — 배치
        롤아웃에서는 이 절약이 사라진다는 뜻이지, 결과가 달라진다는 뜻은 아니다.
        """

        return jax.lax.cond(
            jnp.asarray(required, bool),
            lambda _: self._designate_taker(
                state, ball_xy, restart_team, goalkeeper_only, restart_kind,
                restart_indirect, key),
            lambda _: jnp.int32(NO_PLAYER),
            None,
        )

    def _taker_proposal_legal(
        self, state, proposed, restart_team, restart_kind
    ):
        """Shared active/team/role gate for every taker proposal."""

        proposed = jnp.asarray(proposed, jnp.int32)
        safe = jnp.clip(proposed, 0, self.N - 1)
        return (
            (proposed >= 0)
            & (proposed < self.N)
            & state.active_player[safe]
            & (
                state.team_id[safe]
                == jnp.asarray(restart_team, jnp.int32)
            )
            & (
                (jnp.asarray(restart_kind, jnp.int32) != RK_GK_HOLD)
                | (state.gk_indices[safe] == 1)
            )
        )

    def _designate_taker(self, state, ball_xy, restart_team, goalkeeper_only,
                         restart_kind, restart_indirect=False, key=None):
        """세트피스 키커 지정 — **모든 재개가 이 한 지점을 지난다**.

        종전에는 공에 최근접인 선수가 전부 찼다. 재개 종류가 이 함수에 전달조차 되지 않아
        종류별로 다르게 고를 방법이 없었고, 그래서 센터백이 스로인을 던지고 마침 뒤에 있던
        수비수가 페널티를 찼다. 이제 종류를 함께 받아 :mod:`setpiece_taker`의 결정자에게
        넘긴다 — 스로인·코너·파울·페널티·오프사이드·킥오프·잘못된 키커 복구가 같은 규칙을
        공유해야 규칙이 하나로 유지된다.

        ``restart_kind``는 **필수**다. 기본값을 ``state.restart_kind``로 두었더니 호출부가
        아무도 넘기지 않았고, 새 재개를 만드는 시점의 state는 아직 **이전 값**(대개 RK_NONE)
        이라 스로인을 만들면서 직접 프리킥 규칙으로 키커를 골랐다(실측: 스로인에서 FB 대신
        CF). 인자를 필수로 두면 같은 실수가 다시 나지 않는다.

        GK 마스크가 비면(퇴장 등) 팀 최근접으로 폴백한다 — 마스크 전무 시 argmin(∞)=0이
        상대팀 선수를 키커로 지정·견인하는 폭주를 막는다(IFAB상 골킥은 아무나 차도 됨).
        """

        kind = restart_kind
        indirect = restart_indirect
        # 고정 키를 쓰면 확률적 학습 정책이 매 사건 같은 난수를 받는다. 재개 시각과
        # 종류로 갈라 두면 같은 상황에서는 재현되고 다른 상황에서는 달라진다.
        if key is None:
            # 에피소드 시드 · 시각 · 종류를 함께 접는다. 시드를 빼면 서로 다른 시드의 두
            # 경기가 같은 시각·같은 재개에서 같은 난수를 받아, 확률적 키커 정책이
            # 에피소드 간에 전혀 달라지지 않는다.
            key = jax.random.fold_in(
                jax.random.fold_in(
                    jax.random.fold_in(jax.random.PRNGKey(0x5E7),
                                       state.episode_seed),
                    state.t),
                jnp.asarray(kind, jnp.int32))
        key = select_random_key(
            getattr(self, "_active_randomness_control", None),
            RandomEvent.SET_PIECE_TAKER,
            0,
            key,
        )
        view = self.setpiece_taker_view(
            state, ball_xy, restart_team, goalkeeper_only, kind, indirect)
        native = self._native_taker_command

        def configured_proposal(_):
            # 파라미터를 받는 결정자면 3인자 규약으로 부른다. 어느 쪽인지는 생성 시
            # 확정된 정적 플래그라 이 분기가 프로그램을 갈라 놓지 않는다.
            proposed = (
                self._restart_taker_decider(self._taker_params, view, key)
                if self._taker_takes_params
                else self._restart_taker_decider(view, key))
            # dtype은 **변환 전에** 본다. 먼저 int32로 캐스팅하면 3.9가 조용히
            # 슬롯 3으로 승인된다.
            raw = proposed
            proposed = jnp.asarray(proposed)
            if not jnp.issubdtype(proposed.dtype, jnp.integer):
                raise ValueError(
                    "restart taker decider must return an integer slot, got dtype "
                    f"{proposed.dtype}")
            if proposed.ndim:
                raise ValueError(
                    f"restart taker decider must return a scalar, got shape "
                    f"{proposed.shape}")
            # 폭도 dtype과 같은 이유로 좁히기 전에 본다. 트레이서는 이미 런타임
            # dtype이므로 NumPy 변환이 실패하고 아래 JAX 승인층으로 넘어간다.
            limits = np.iinfo(np.int32)
            try:
                host = np.asarray(raw)
            except (TypeError, ValueError):
                host = None
            if host is not None and host.size and np.issubdtype(
                host.dtype, np.integer
            ):
                if int(host.min()) < limits.min or int(host.max()) > limits.max:
                    raise ValueError(
                        "restart taker decider slot must fit in int32 "
                        f"([{limits.min}, {limits.max}]), got {int(host.min())}")
            return proposed.astype(jnp.int32)

        if native is None:
            # The frozen public path stays structurally unchanged: no table
            # lookup or conditional is added to ``step_env_array``/reset.
            proposed = configured_proposal(None)
        else:
            # ``StepCommand`` carries the whole team x restart-kind table, so a
            # restart born inside ``lax.scan`` can select its override without
            # a host callback or knowing the event one frame in advance.
            team = jnp.asarray(restart_team, jnp.int32)
            restart = jnp.asarray(kind, jnp.int32)
            addressable = (
                (team >= TEAM_0) & (team <= TEAM_1)
                & (restart >= RK_KICKOFF) & (restart < RESTART_COUNT)
            )
            safe_team = jnp.clip(team, TEAM_0, TEAM_1)
            safe_restart = jnp.clip(restart, RK_NONE, RESTART_COUNT - 1)
            requested = addressable & native.requested[safe_team, safe_restart]
            external = native.player_slot[safe_team, safe_restart].astype(jnp.int32)
            # On CPU/small batches this avoids the configured selector entirely
            # for an externally owned event.  Under vmap XLA may lower it to a
            # select, but the fixed PyTree remains valid and semantics match.
            proposed = jax.lax.cond(
                requested,
                lambda _: external,
                configured_proposal,
                None,
            )

        # 결정자는 **제안**하고 환경이 승인한다 — 키커도 교체·포메이션과 같은 규약이다.
        # 외부 슬롯도 이 동일한 active/team/GK gate를 통과하므로 공개 명령이 규칙을
        # 우회하지 않는다. 거부되면 기존의 법적 최근접 fallback을 사용한다.
        legal = self._taker_proposal_legal(
            state, proposed, restart_team, kind
        )
        # 거부되면 규칙이 아니라 **법이 허용하는 최소 선택**으로 떨어진다 — 팀의 활성 선수
        # 중 재개 지점 최근접. 여기서 다시 결정자를 부르면 같은 값이 돌아온다.
        fallback = setpiece_taker_module.nearest_taker(view, key)
        return jnp.where(legal, proposed, fallback).astype(jnp.int32)

    def _setpiece_plan_slot(self, kind, indirect, spot):
        """이번 재개가 참조할 :class:`SetPiecePlan` 행. 지정이 없으면 -1."""

        names = setpiece_taker_module.PLAN_FIELDS
        index = {name: i for i, name in enumerate(names)}
        kind = jnp.asarray(kind, jnp.int32)
        left = spot[DIM_Y] < 0.0
        slot = jnp.int32(-1)
        slot = jnp.where(kind == RK_PENALTY, index["penalty"], slot)
        slot = jnp.where(
            kind == RK_FREEKICK,
            jnp.where(jnp.asarray(indirect, bool),
                      index["indirect_free_kick"], index["direct_free_kick"]),
            slot)
        slot = jnp.where(kind == RK_OFFSIDE, index["indirect_free_kick"], slot)
        slot = jnp.where(
            kind == RK_CORNER,
            jnp.where(left, index["corner_left"], index["corner_right"]), slot)
        slot = jnp.where(
            kind == RK_THROWIN,
            jnp.where(left, index["throw_left"], index["throw_right"]), slot)
        slot = jnp.where(kind == RK_KICKOFF, index["kickoff"], slot)
        slot = jnp.where(kind == RK_GOALKICK, index["goal_kick"], slot)
        return slot.astype(jnp.int32)

    def launch_lo(self, ball_z):
        """공 높이 ball_z에서 허용되는 최소(하향) 발사각(rad). 지면공은 -launch_down_ground,
        ball_z≥launch_down_ref에서 -launch_max까지 열림. 액션 발사각[0,1]은 [launch_lo, launch_max]로 재매핑."""
        ball_z, _ = _coerce_public_float32("ball_z", ball_z, ())
        e_cfg = self.e_cfg
        t = jnp.clip((ball_z - self.r_ball) / (e_cfg.launch_down_ref - self.r_ball), 0.0, 1.0)
        return -(e_cfg.launch_down_ground + (e_cfg.launch_max - e_cfg.launch_down_ground) * t)

    def _apply_force2ball(
        self,
        state,
        winner,
        any_cand,
        f2b_dir,
        f2b_pow,
        f2b_launch,
        key,
        spin_side=None,
        spin_back=None,
        want_kick=None,
        forced_freeplay=None,
        contest_candidate=None,
        forced_gk_touch=None,
        *,
        randomness=None,
        substep_index=0,
        return_event=False,
    ):
        """Apply one eligible contact or the exact no-candidate normalization path.

        The contact implementation is intentionally not duplicated: the active branch delegates
        to :meth:`_apply_force2ball_full`.  A no-candidate call historically still repaired card,
        pass-latch, and foul-latch invariants for directly constructed States.  Preserve those
        writes (and the same card RNG split) while skipping the large contact-only graph.
        """

        winner_in_range = (winner >= 0) & (winner < self.N)
        candidate_active = any_cand & winner_in_range

        def active(_):
            return self._apply_force2ball_full(
                state,
                winner,
                any_cand,
                f2b_dir,
                f2b_pow,
                f2b_launch,
                key,
                spin_side=spin_side,
                spin_back=spin_back,
                want_kick=want_kick,
                forced_freeplay=forced_freeplay,
                contest_candidate=contest_candidate,
                forced_gk_touch=forced_gk_touch,
                randomness=randomness,
                substep_index=substep_index,
                return_event=return_event,
            )

        def inactive(_):
            return self._apply_force2ball_no_candidate(
                state,
                key,
                randomness=randomness,
                substep_index=substep_index,
                return_event=return_event,
            )

        return jax.lax.cond(candidate_active, active, inactive, operand=None)

    def _apply_force2ball_no_candidate(
        self,
        state,
        key,
        *,
        randomness=None,
        substep_index=0,
        return_event=False,
    ):
        """Preserve the legacy no-contact normalization without contact scoring."""

        _, _, k_card, _, _ = jax.random.split(key, 5)
        normalized = self._draw_cards(
            state,
            jnp.zeros(self.N, dtype=bool),
            k_card,
            foul_pos=state.ball_pos,
            randomness=randomness,
            substep_index=substep_index,
        )
        normalized = self._normalize_pass_latch(
            normalized, clear=jnp.bool_(False)
        )
        normalized = self._normalize_foul_latch(normalized)
        if return_event:
            return normalized, jnp.zeros(self.N, dtype=bool)
        return normalized

    def _apply_force2ball_full(
        self,
        state,
        winner,
        any_cand,
        f2b_dir,
        f2b_pow,
        f2b_launch,
        key,
        spin_side=None,
        spin_back=None,
        want_kick=None,
        forced_freeplay=None,
        contest_candidate=None,
        forced_gk_touch=None,
        *,
        randomness=None,
        substep_index=0,
        return_event=False,
    ):
        """경합 승자의 공 접촉 결과 적용 — 킥/탈취/파울/굴절/세트피스 소비/오프사이드 플래그/카드.

        분기(전부 관측 결과에서 역산 가능):
          free_play : 오픈플레이·세트피스 킥 — 승자 커맨드(방향·파워·발사각·스핀)로 발사.
          tackle_ok : 상대 소유 탈취 성공(uniform<tackle_prob) — 커맨드 적용, 출구속도 상한 캡.
          foul      : 태클 파울(uniform<p_foul) — 공 데드 + FK/페널티 + 확률 카드.
          deflect   : 접촉했으나 탈취 실패(uniform<deflect_prob) — 소유 불변 루즈볼, 각도 랜덤.
          gk_claim  : 자기 박스 GK의 무의도 승리(want_kick 승자는 하이재킹 안 함) —
                      속도별 확률=캐치홀드(RK_GK_HOLD) / 나머지=parry / 백패스=IDFK(포워드에선
                      movement 합법성 게이트로 사실상 미도달 — forced_winner 복원·방어용 분기).

        forced_freeplay: reconstruct용 분기 pin(스칼라 bool). True면 opp_poss를 젖혀 관측 터치를
        free_play(결정론 킥)로 강제 — tackle/foul/deflect 추첨을 구조적으로 무력화한다. 상태
        (poss_team)를 덮어쓰지 않는 순수 pin이며, 소유는 터치 성사 시 new_poss가 스스로 갱신.
        None/False면 포워드 불변.
        contest_candidate: 이 substep의 reach/cooldown/retouch/ctrl-lock을 모두 통과한 실제 후보
        mask. 소유자가 argmax를 이긴 retained 분기에서 실제 패배 도전자를 찾는 데 사용한다.
        단순 최근접 상대를 쓰면 더 가까운 수동 선수가 파울·cooldown을 대신 받는다.
        forced_gk_touch: reconstruct용 GK 결과 pin. ``TOUCH_GK_CATCH`` 또는
        ``TOUCH_PARRY``면 확률 draw만 덮어쓰고, None/``TOUCH_NONE``이면 정상 샘플한다.
        백패스·박스·도달 같은 확정 규칙은 우회하지 않는다.
        """
        e_cfg = self.e_cfg
        f_cfg = self.f_cfg
        k_foul, k_tackle, k_card, k_deflect, k_defl_dir = jax.random.split(key, 5)
        k_foul = select_random_key(
            randomness, RandomEvent.TACKLE_FOUL, substep_index, k_foul
        )
        k_tackle = select_random_key(
            randomness, RandomEvent.TACKLE_SUCCESS, substep_index, k_tackle
        )
        k_deflect = select_random_key(
            randomness, RandomEvent.DEFLECTION, substep_index, k_deflect
        )
        k_defl_dir = select_random_key(
            randomness,
            RandomEvent.DEFLECTION_DIRECTION,
            substep_index,
            k_defl_dir,
        )

        # NO_PLAYER(-1)를 그대로 배열 인덱스로 쓰면 JAX가 마지막 slot로 해석한다. 현재 분기들이
        # any_cand로 값 변경을 막더라도 새 필드 하나의 게이트 누락이 slot[-1]을 오염시킬 수 있으므로,
        # 읽기·scatter에 들어가기 전에 센티널을 안전 인덱스로 정규화하고 후보 유효성을 닫는다.
        winner_in_range = (winner >= 0) & (winner < self.N)
        any_cand = any_cand & winner_in_range
        winner = jnp.clip(winner, 0, self.N - 1).astype(jnp.int32)

        ball_z = state.ball_pos[DIM_Z]
        head_z = state.head_z[winner]
        pelvis_z = e_cfg.pelvis_frac * head_z
        header = ball_z > head_z
        is_chest = (ball_z > pelvis_z) & (~header)
        power_cap = jnp.where(header, e_cfg.header_cap, jnp.where(is_chest, e_cfg.chest_cap, 1.0))
        requested_speed = f2b_pow[winner] * power_cap * e_cfg.f2b_speed_max

        # 발사각 재매핑 — 액션 [0,launch_max]를 [launch_lo(ball_z), launch_max]로(하향 타격 커맨드화)
        launch_floor = self.launch_lo(ball_z)
        launch_ang = launch_floor + (f2b_launch[winner] / e_cfg.launch_max) * (e_cfg.launch_max - launch_floor)
        win_team = state.team_id[winner].astype(jnp.int32)
        attack_dir = state.attack_dir[winner]
        kick_dir_raw = f2b_dir[winner]
        restart_take = (
            any_cand
            & restart_timer_active(state.restart_t)
            & (win_team == state.restart_team)
        )
        speed = jnp.where(
            restart_take,
            jnp.maximum(requested_speed, e_cfg.restart_min_ball_speed),
            requested_speed,
        )
        # A zero radial action has neither power nor direction.  The minimum
        # restart speed above therefore needs a deterministic legal bearing:
        # near a pitch boundary point toward the centre (corner/throw-in and
        # boundary FK), elsewhere use the taker's attacking direction.
        attack_fallback = jnp.asarray([attack_dir, 0.0])
        boundary_band = max(
            e_cfg.restart_field_inset,
            e_cfg.throwin_line_inset,
            e_cfg.free_kick_boundary_inset,
        )
        near_boundary = (
            (jnp.abs(state.ball_pos[DIM_X])
             >= self.hx - boundary_band - GEOMETRY_EPS)
            | (jnp.abs(state.ball_pos[DIM_Y])
               >= self.hy - boundary_band - GEOMETRY_EPS)
        )
        centre_raw = -state.ball_pos[:DIM_Z]
        centre_fallback = jnp.where(
            jnp.linalg.norm(centre_raw) > GEOMETRY_EPS,
            _unit(centre_raw[None, :])[0],
            attack_fallback,
        )
        zero_fallback = jnp.where(
            near_boundary, centre_fallback, attack_fallback
        )
        kick_dir_raw = jnp.where(
            restart_take & (jnp.linalg.norm(kick_dir_raw) <= GEOMETRY_EPS),
            zero_fallback,
            kick_dir_raw,
        )
        # IFAB Law 14 requires the penalty kick to move forward.  Release is
        # automatic once setup is complete, so rejecting a non-forward command
        # would only time out the restart.  Project only this restart action to
        # the nearest strict-forward half-plane and keep its requested speed.
        penalty_take = (
            restart_take
            & (state.restart_kind == RK_PENALTY)
        )
        forward_axis = jnp.asarray([attack_dir, 0.0])
        raw_unit = _unit(kick_dir_raw[None, :])[0]
        forward_component = jnp.dot(raw_unit, forward_axis)
        lateral_raw = raw_unit - forward_component * forward_axis
        lateral_norm = jnp.linalg.norm(lateral_raw)
        lateral_fallback = jnp.asarray([0.0, 1.0])
        lateral_unit = jnp.where(
            lateral_norm > GEOMETRY_EPS,
            lateral_raw / (lateral_norm + DIV_EPS),
            lateral_fallback,
        )

        # A merely mathematical epsilon is not enough in a float32 world.
        # At the penalty mark, ``GEOMETRY_EPS`` of a 0.5 m/s release advances
        # by only ~5e-9 m per tick, below one float32 ULP; repeated additions
        # then leave x *exactly unchanged* while the ball travels sideways and
        # the penalty is nevertheless consumed.  Require four representable
        # x steps of headroom in the first physics tick.  This is still the
        # nearest legal half-plane projection (typically cos ~= 0.001), not a
        # tactical forward rewrite.
        forward_target = jnp.where(
            attack_dir > 0.0, jnp.float32(jnp.inf), jnp.float32(-jnp.inf)
        )
        forward_ulp = jnp.abs(
            jnp.nextafter(state.ball_pos[DIM_X], forward_target)
            - state.ball_pos[DIM_X]
        )
        required_forward_speed = 4.0 * forward_ulp / e_cfg.dt_phys
        # A custom launch_max can approach pi/2 closely enough that even a
        # fully forward direction has no representable x displacement.  Lower
        # only an offending penalty elevation to the nearest angle that can
        # realize the numerical Law-14 half-plane; all ordinary launches and
        # all non-penalty kicks remain untouched.
        max_penalty_launch = jnp.arccos(jnp.clip(
            required_forward_speed / (speed + DIV_EPS), 0.0, 1.0
        ))
        effective_launch = jnp.where(
            penalty_take & (launch_ang > max_penalty_launch),
            max_penalty_launch,
            launch_ang,
        )
        horizontal_speed = speed * jnp.maximum(jnp.cos(effective_launch), 0.0)
        min_forward_cos = jnp.clip(
            (4.0 * forward_ulp)
            / (horizontal_speed * e_cfg.dt_phys + DIV_EPS),
            GEOMETRY_EPS,
            1.0,
        )
        projected_lateral = jnp.sqrt(
            jnp.maximum(0.0, 1.0 - min_forward_cos * min_forward_cos)
        )
        constrained_dir = (
            min_forward_cos * forward_axis
            + projected_lateral * lateral_unit
        )
        # A purely backward command has no meaningful lateral side to
        # preserve; the closest deterministic legal direction is straight
        # forward, matching the previous behaviour for that case.
        constrained_dir = jnp.where(
            lateral_norm > GEOMETRY_EPS, constrained_dir, forward_axis
        )
        penalty_dir = jnp.where(
            forward_component >= min_forward_cos, raw_unit, constrained_dir
        )
        kick_dir = jnp.where(penalty_take, penalty_dir, kick_dir_raw)

        # Law 15: the ball is not in play until it enters the field.  The env
        # owns restart timing and therefore cannot simply reject an outward
        # action and wait for a new policy decision; doing so used to consume
        # the throw, adjudicate the never-entered ball as a fresh out, and
        # award the opponents a throw-in.  Project only an outward/parallel
        # throw to the nearest representably inward half-plane, preserving as
        # much along-touchline direction as possible.
        throw_take = restart_take & (state.restart_kind == RK_THROWIN)
        touchline_side = jnp.where(
            state.ball_pos[DIM_Y] != 0.0,
            jnp.sign(state.ball_pos[DIM_Y]),
            jnp.where(state.player_pos[winner, DIM_Y] >= 0.0, 1.0, -1.0),
        )
        inward_axis = jnp.asarray([0.0, -touchline_side])
        throw_raw_unit = _unit(kick_dir[None, :])[0]
        inward_component = jnp.dot(throw_raw_unit, inward_axis)
        tangent_raw = throw_raw_unit - inward_component * inward_axis
        tangent_norm = jnp.linalg.norm(tangent_raw)
        tangent_fallback = jnp.asarray([attack_dir, 0.0])
        tangent_unit = jnp.where(
            tangent_norm > GEOMETRY_EPS,
            tangent_raw / (tangent_norm + DIV_EPS),
            tangent_fallback,
        )
        inward_target = jnp.where(
            touchline_side > 0.0, jnp.float32(-jnp.inf), jnp.float32(jnp.inf)
        )
        # 릴리스 좌표와 릴리스 속도를 **실제 값으로** 잡는다. 종전에는 아래 두 가지를
        # 썼는데 둘 다 이 순간의 값이 아니다.
        #   * ``state.ball_pos[DIM_Y]`` — 릴리스 전 스폿. 실제 공은 손 위치(아래
        #     ``throw_release_limit``)에서 떠난다.
        #   * ``speed`` — 일반 발차기 속도(기본 34.76 m/s). 스로인은
        #     ``throw_speed_max``(기본 21.5)로 나간다.
        # 그래서 "최소 restart_min_ball_speed 만큼은 확실히 안쪽으로 움직인다"는 아래
        # 보장이 실제로는 21.5/34.76 배로 깎여 0.5 m/s 대신 0.309 m/s였다. 속도비가
        # 더 큰 설정에서는 안쪽 변위가 float32 ULP 아래로 내려가 재개만 소비하고 공은
        # 라인 밖에 머무를 수도 있다.
        throw_release_inset = jnp.minimum(e_cfg.legal_margin_floor,
                                          0.5 * self.r_ball)
        throw_release_limit = self.hy + self.r_ball - throw_release_inset
        release_y = touchline_side * throw_release_limit
        requested_throw_speed = f2b_pow[winner] * e_cfg.throw_speed_max
        # 재개로 소비되는 스로인은 항상 하한까지 끌어올려진다(아래 ``throw_speed``와 동일).
        throw_speed_taken = jnp.maximum(requested_throw_speed,
                                        e_cfg.restart_min_ball_speed)
        inward_ulp = jnp.abs(
            jnp.nextafter(release_y, inward_target) - release_y
        )
        throw_horizontal_speed = (
            throw_speed_taken * jnp.maximum(jnp.cos(launch_ang), 0.0))
        representable_inward_cos = (
            4.0 * inward_ulp
            / (throw_horizontal_speed * e_cfg.dt_phys + DIV_EPS)
        )
        # A merely representable component can leave an oblique throw
        # skimming outside the line for tens of seconds.  Reuse the restart
        # "clearly moves" speed as the minimum entry component; very slow or
        # highly lofted throws simply point fully inward.
        clear_entry_cos = (
            jnp.minimum(e_cfg.restart_min_ball_speed, throw_horizontal_speed)
            / (throw_horizontal_speed + DIV_EPS)
        )
        min_inward_cos = jnp.clip(
            jnp.maximum(representable_inward_cos, clear_entry_cos),
            GEOMETRY_EPS,
            1.0,
        )
        projected_tangent = jnp.sqrt(
            jnp.maximum(0.0, 1.0 - min_inward_cos * min_inward_cos)
        )
        inward_dir = (
            min_inward_cos * inward_axis
            + projected_tangent * tangent_unit
        )
        inward_dir = jnp.where(
            tangent_norm > GEOMETRY_EPS, inward_dir, inward_axis
        )
        throw_dir = jnp.where(
            inward_component >= min_inward_cos, throw_raw_unit, inward_dir
        )
        kick_dir = jnp.where(throw_take, throw_dir, kick_dir)
        kick_vel = speed * jnp.array([
            jnp.cos(effective_launch) * kick_dir[DIM_X],
            jnp.cos(effective_launch) * kick_dir[DIM_Y],
            jnp.sin(effective_launch),
        ])

        ss = 0.0 if spin_side is None else spin_side[winner]
        sb = 0.0 if spin_back is None else spin_back[winner]
        spin_cap = jnp.where(header, e_cfg.spin_head_cap, jnp.where(is_chest, e_cfg.spin_chest_cap, 1.0))
        lateral = jnp.array([-kick_dir[DIM_Y], kick_dir[DIM_X], 0.0])
        # 사이드스핀(수직축)은 180° 회전 불변 → attack_dir 언폴딩 없이 직접 적용(B1, _decode 규약과 짝).
        # 백스핀은 진행방향 lateral축이라 프레임 무관.
        kick_spin = e_cfg.spin_max * spin_cap * ((-sb) * lateral + ss * jnp.array([0.0, 0.0, 1.0]))

        # ★속력 규약 = xy(수평)만 — GK 캐치/parry/굴절 속도(gk_catch_speed_cap·deflect_out_*)가 이
        # xy 규약으로 세팅·캘리브됐다. 3D(vz 포함)로 바꾸면 급강하 공을 더 어렵게 보지만 캡이 미보정
        # 상태가 되어 catch↓·parry↑로 조용히 치우친다 → 재캘리브 없이는 xy 규약을 유지한다.
        ball_speed0 = jnp.linalg.norm(state.ball_vel[:DIM_Z])
        poss = state.poss_team
        ff = jnp.bool_(False) if forced_freeplay is None else forced_freeplay
        poss_mask = (state.team_id == poss) & state.active_player
        has_possessor = (poss >= 0) & jnp.any(poss_mask)
        dist_ball_all = jnp.linalg.norm(
            state.player_pos - state.ball_pos[:DIM_Z][None, :], axis=1
        )
        carrier = jnp.argmin(jnp.where(poss_mask, dist_ball_all, jnp.inf))
        carrier_is_handling_gk = (
            (state.gk_indices[carrier] == 1)
            & self._ball_in_each_own_box(state)[carrier]
        )
        carrier_radius = jnp.where(
            carrier_is_handling_gk,
            e_cfg.gk_reach_xy,
            e_cfg.reach_xy_carry,
        )
        carrier_controls = (
            has_possessor
            & (dist_ball_all[carrier] <= carrier_radius + self.r_ball)
            & (state.ball_pos[DIM_Z] <= state.reach_z[carrier] + self.r_ball)
        )
        # ``poss_team`` is a team-level phase/provenance latch: while a pass is
        # travelling it deliberately remains on the passing side.  TACKLE,
        # however, means a duel with an actual ball carrier.  Requiring a
        # reachable carrier prevents an opponent who cuts out a loose/travelling
        # pass from entering the tackle/foul lottery merely because that latch
        # has not changed yet.
        #
        # poss_team만 남고 해당 팀 온피치 선수가 전무한 극단 상태도 같은 이유로
        # 루즈볼로 취급한다. 그렇지 않으면 argmin(all inf)=0인 유령 캐리어를
        # 상대로 태클/파울이 발생한다.
        opp_poss = carrier_controls & (win_team != poss) & (~ff)  # 복원 pin: 분기 추첨 무력화
        locked = opp_poss & (state.ctrl_lock_t[winner] > 0)        # 승자(도전자)가 재탈취 지연 중
        restart_active = restart_timer_active(state.restart_t)

        # ── GK 캐치/홀드/parry (자기 박스 안 GK가 라이브 공을 잡음) ──────────────
        # env가 '자기 박스 안 GK'를 want_f2b 없이도 경합 후보로 넣으므로(리액티브 클레임) GK가 승자일 수 있다.
        # 자기 박스 GK가 라이브 공을 이기면: ①동료의 의도적 GK 대상 발킥/스로인 provenance가
        # 살아 있으면(백패스)→반칙·간접FK
        # ②잡을 수 있으면(저속)→캐치 홀드(RK_GK_HOLD) ③너무 빠르면→parry(쳐냄). 아래 free_play·contest에서 배제.
        winner_is_gk = state.gk_indices[winner] == 1
        # IFAB handling area is determined by the ball, not by the goalkeeper's
        # feet.  A keeper just inside the front line cannot reach two metres out
        # of the box and catch; conversely a keeper whose centre is just outside
        # may legally reach back to a ball that remains on/inside the line.
        ball_in_winner_own_box = self._ball_in_each_own_box(state)[winner]
        # want_kick(승자의 킥 의도)가 있으면 캐치로 하이재킹하지 않는다 — GK가 발로 클리어하려는
        # 공(백패스 포함 — 발 플레이는 IFAB상 합법)을 손 캐치/백패스 IDFK로 바꿔치기하면 안 됨.
        # 리액티브 클레임(무의도)은 movement가 합법 캐치일 때만 후보로 올린다(재터치/백패스 제외).
        winner_wants_kick = jnp.bool_(False) if want_kick is None else want_kick[winner]
        gk_claim = (any_cand & winner_is_gk & ball_in_winner_own_box
                    & (~winner_wants_kick)
                    & (state.ball_state == BALL_ALIVE) & (~restart_active))
        handling_code0 = state.gk_handling_restricted_team
        handling_team0 = jnp.where(
            handling_code0 >= GK_HANDLING_RELEASE_OFFSET,
            handling_code0 - GK_HANDLING_RELEASE_OFFSET,
            handling_code0,
        )
        handling_from_release0 = (
            handling_code0 >= GK_HANDLING_RELEASE_OFFSET
        )
        gk_handling_offence = (
            gk_claim & (handling_team0 == win_team)
        )  # movement 게이트와 동일 SSOT
        # 실측 catch/punch는 같은 속도에서도 겹친다. scale>0이면 관측 로지스틱 곡선을
        # 그대로 쓰고, 0이면 기존 하드캡과 bit-for-bit 같은 호환 분기를 유지한다.
        # 별도 draw는 기존 다섯 draw의 열을 밀지 않도록 원 key에서 fold_in해 파생한다.
        if e_cfg.gk_catch_speed_scale <= 0.0:
            sampled_gk_catch = ball_speed0 <= e_cfg.gk_catch_speed_cap
        else:
            catch_prob = self._gk_catch_probability(ball_speed0)
            gk_catch_key = select_random_key(
                randomness,
                RandomEvent.GOALKEEPER_CATCH,
                substep_index,
                jax.random.fold_in(key, jnp.uint32(0x474B4348)),
            )
            sampled_gk_catch = (
                jax.random.uniform(gk_catch_key)
                < catch_prob
            )
        if forced_gk_touch is not None:
            force_catch = forced_gk_touch == TOUCH_GK_CATCH
            force_parry = forced_gk_touch == TOUCH_PARRY
            sampled_gk_catch = jnp.where(
                force_catch | force_parry, force_catch, sampled_gk_catch
            )
        gk_hold_catch = gk_claim & (~gk_handling_offence) & sampled_gk_catch
        gk_parry = gk_claim & (~gk_handling_offence) & (~sampled_gk_catch)

        contest = any_cand & opp_poss & (~locked) & (~gk_claim)

        # 태클 파울 — 도전 수비수(fouler)가 캐리어를 반칙. [#10c] 도전자가 경합을 이기든(opp_poss)
        # 지든(possessor가 공 유지) 반칙은 그 도전 수비수가 저지른다 → '파울하고 공 뺏김'을 표현.
        carrier_pos = state.player_pos[carrier]
        # 도전 수비수: opp_poss면 argmax 승자(winner), 아니면 캐리어 최근접 상대(chal — 지고도 파울).
        eligible = (state.active_player if contest_candidate is None
                    else (jnp.asarray(contest_candidate, bool) & state.active_player))
        opp_of_poss = (state.team_id != poss) & eligible & (poss >= 0)
        d_carrier_all = jnp.linalg.norm(state.player_pos - carrier_pos[None, :], axis=1)
        chal = jnp.argmin(jnp.where(opp_of_poss, d_carrier_all, jnp.inf))
        chal_exists = opp_of_poss[chal]
        fouler = jnp.where(opp_poss, winner, chal).astype(jnp.int32)
        # The abstract lunge radius can put the challenger's centre as much as
        # 2.5 m from the victim.  Law 12 locates a contact offence where the
        # opponent is struck, not at the tackler's centre; the carrier is the
        # observable contact-location proxy used for both the box decision and
        # the free-kick spot (the charge channel follows the same convention).
        foul_pos = carrier_pos
        fpx, fpy = foul_pos[DIM_X], foul_pos[DIM_Y]
        # 박스·페널티 기하는 **파울러 자기 공격방향** 기준. attack_dir(=winner)은 킥용이라,
        # loser-foul(fouler=chal=상대팀)에선 부호가 반대 → 박스판정·페널티스폿이 반대 골대에 찍히는 버그.
        foul_dir = state.attack_dir[fouler]
        carrier_face_dir = jnp.array([jnp.cos(state.player_facing[carrier]), jnp.sin(state.player_facing[carrier])])
        fouler_from_carrier = _unit((state.player_pos[fouler] - carrier_pos)[None, :])[0]
        behind = jnp.clip(-jnp.dot(fouler_from_carrier, carrier_face_dir), 0.0, 1.0)
        to_carrier = _unit((carrier_pos - state.player_pos[fouler])[None, :])[0]
        v_close = jnp.clip(
            jnp.dot(state.player_vel[fouler] - state.player_vel[carrier], to_carrier),
            0.0,
            e_cfg.norm_player_vel,
        )
        ball_dist = jnp.linalg.norm(state.ball_pos[:DIM_Z] - state.player_pos[fouler])
        clean_win = (ball_dist < f_cfg.clean_ball_dist) & (ball_speed0 < f_cfg.clean_ball_speed)

        # 선수 물리는 골라인 뒤 5m까지 허용한다. Law 12에 따라 그곳의 반칙은 가장
        # 가까운 경계점이 자기 페널티구역 골라인이면 페널티이므로 공통 어댑터가
        # 피치 내부 박스와 자기 골라인 뒤의 해당 구간을 함께 분류한다.
        fouler_in_box = self._foul_in_own_box(foul_pos, foul_dir)
        box_penalty_bias = jnp.where(fouler_in_box, f_cfg.box_foul_bias, 0.0)
        air_bias = jnp.where(header, f_cfg.header_foul_bias, 0.0)
        # 런지 분율 λ: reach_xy_*는 선수 중심에서 **공 표면**까지의
        # 거리지만 ball_dist는 공 중심까지의 거리다. _in_reach가 자격 반경에
        # r_ball을 더하는 규약과 같게 표면 거리로 환산해야 런지 존이
        # carry+r_ball ~ challenge+r_ball에 정확히 맞는다.
        lunge_fraction = self._lunge_fraction(ball_dist)
        logit = (f_cfg.tackle_bias + f_cfg.k_lunge * lunge_fraction
                 + f_cfg.k_close * v_close + f_cfg.k_behind * behind
                 + f_cfg.k_balldist * jnp.maximum(0.0, ball_dist - f_cfg.balldist_ref)
                 - f_cfg.k_clean * clean_win.astype(jnp.float32) + box_penalty_bias + air_bias)
        p_foul = jnp.clip(jax.nn.sigmoid(logit), f_cfg.tackle_p_min, f_cfg.tackle_p_max)
        contact_range = jnp.linalg.norm(state.player_pos[fouler] - carrier_pos) < f_cfg.tackle_contact_range
        # loser-foul: possessor가 공 유지(~opp_poss)해도 근접 도전자가 **돌진(lunge)**하면 파울 가능.
        # 상시 근접이 아닌 실제 태클 시도만 파울화(폭증 방지). recon-안전: any_cand & ~ff 게이트로 재구성 결정성 유지.
        loser_is_lunging = v_close > e_cfg.charge_speed
        retained = (carrier_controls & any_cand & (~opp_poss) & (~ff) & (~gk_claim) & (~restart_active)
                    & (state.ball_state == BALL_ALIVE) & chal_exists & loser_is_lunging)
        foul = (contest | retained) & contact_range & (jax.random.uniform(k_foul) < p_foul)
        # loser-foul: 파울러가 경합 **패자**인 분기(retained). 승자는 파울을 '당한' 쪽이라
        # 아래 터치 라벨·last_touch 갱신에서 별도 취급해야 한다(contest-foul은 승자=파울러).
        loser_foul = foul & (~opp_poss)
        tackle_ok = contest & (~foul) & (jax.random.uniform(k_tackle) < e_cfg.tackle_prob)
        deflect = contest & (~foul) & (~tackle_ok) & (jax.random.uniform(k_deflect) < e_cfg.deflect_prob)
        free_play = any_cand & (~opp_poss) & (~gk_claim)
        # A deliberate touch that cuts out the other side's travelling/loose
        # ball is an interception regardless of incoming speed.  Speed cannot
        # distinguish a slow pass interception from a tackle; presence of a
        # controlled carrier can.  ``forced_freeplay`` remains a reconstruction
        # pin and therefore suppresses this semantic override.
        interception = (
            free_play & has_possessor & (~carrier_controls)
            & (win_team != poss) & (~ff) & (~restart_active)
            & (state.ball_state == BALL_ALIVE)
        )
        consume = free_play & restart_active & (win_team == state.restart_team)
        # Restart exclusion zones are enforced before contact by
        # Restart._project_restart_positions.  A legal kick is therefore always
        # consumed once; there is no encroachment retake/card branch.
        eff_consume = consume

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
        new_spin = jnp.where(stop_ball | deflect, jnp.zeros(DIM_ALL), new_spin)
        # [2026-08-13] K리그 벤더 정합: 비통제 접촉(deflect)은 중립(-1) — 실측상 벤더
        # neutral은 "터치했으나 통제 실패" 순간에 발생(전이 시 직전 소유팀 최근접 0.92 m).
        new_poss = jnp.where(free_play | tackle_ok, win_team,
                    jnp.where(deflect, jnp.int32(-1), poss))

        opp_goal_x = attack_dir * self.hx
        px, py = state.player_pos[winner, DIM_X], state.player_pos[winner, DIM_Y]
        # 슛인가는 **조준선이 골문을 지나는가**로 판정한다. 찬 공의 수평 방향을 골라인까지
        # 연장해 교차점을 구하고, 그 지점이 골문 중앙에서 허용 폭 안이면 슛이다.
        #
        # 종전에는 골 중앙 방향의 고정 60° 원뿔이었다. 각도는 거리에 따라 뜻이 달라져서
        # 5 m 앞에서는 페널티 지역을 통째로 덮고 30 m에서는 골라인 폭 35 m를 덮었고,
        # 그래서 ``f2b_shoot_range``의 30 m 상한으로 막아야 했다. 그 상한의 대가는
        # **30 m 밖 득점이 슛으로 기록되지 않는 것**이었다. 골라인 교차점은 미터 단위라
        # 거리와 무관하게 같은 뜻이고, 두 문제가 함께 사라진다.
        dir_x, dir_y = kick_dir[DIM_X], kick_dir[DIM_Y]
        # 골라인 쪽으로 유의미하게 향할 때만 교차점이 정의된다. 골라인 뒤에서 차면
        # ``travel``이 음수라 같은 게이트가 걸러 낸다.
        toward_goal_line = dir_x * attack_dir > GEOMETRY_EPS
        travel = (opp_goal_x - px) / jnp.where(toward_goal_line, dir_x, jnp.float32(1.0))
        # 감아 찬 공은 직선으로 가지 않는다. 정책은 사이드 스핀으로 골문 밖을 겨눠 안으로
        # 휘게 차므로(실측: 28.5 m 원터치 마무리가 직선 기준으로는 골문 밖이었다), 직선
        # 조준선만 보면 **커브 슛이 전부 패스가 된다**.
        #
        # 휘어짐은 env 자신의 마그누스 항에서 유도한다. ``ball.py``가 속도 벡터를
        # ``omega = c_magnus * spin`` 의 각속도로 회전시키므로, 수평면에서는 스핀의 z성분이
        # 진행방향을 회전시킨다. 작은 각에서 경로는 반지름 ``v/omega``의 원호이고, 거리 L을
        # 간 뒤의 횡변위는 ``L^2 * omega / (2 v)``다. 그 변위의 y성분이 좌법선의 y성분
        # (``dir_x``)이다. 1차 근사이며 지면 커브(``c_ground_curl``)는 포함하지 않는다.
        curl_rate = e_cfg.c_magnus * new_spin[DIM_Z]
        curl_drift = 0.5 * curl_rate * travel * travel / jnp.maximum(speed, DIV_EPS)
        y_cross = py + travel * dir_y + curl_drift * dir_x
        aim_half_width = e_cfg.shot_aim_mouth_scale * (0.5 * self.goal_w)
        aim_goal = (
            toward_goal_line
            & (travel > 0.0)
            & (jnp.abs(y_cross) <= aim_half_width)
        )
        # 슛은 **친 공**이어야 한다. 종전에는 기하(거리·조준)만 보고 속도를 전혀 보지
        # 않아, 골대 쪽으로 향한 3 m/s짜리 제어 터치도 SHOOT이 됐다 — 실측 240초에서
        # SHOOT 라벨 26건 중 16건(61.5%)이 드리블 상한 미만이었고 중앙값이 7.2 m/s였다.
        #
        # 경계를 새로 정할 필요는 없다. ``dribble_speed_max``가 이미 '놓은 공과 친 공'을
        # 가르는 값이고 아래 ``is_dribble``이 그 아래를 제어 터치로 규정하는데, ``is_shot``이
        # 먼저 평가되면서 **골대 근처에서만** 그 규정을 무효화하고 있었다. 즉 의도 축(기하)이
        # 물리 축(속도)을 덮어써, 같은 접촉이 일어난 위치에 따라 제어 터치 자격을 잃었다.
        # 같은 경계를 두 분기가 공유하면 분류가 평가 순서와 무관하게 배타적이 된다.
        # **수신자 테스트는 두지 않는다.** 조준선 위의 동료를 패스 대상으로 세면 골문을
        # 겨눈 강타의 43 %가 걸러지지만, 그 안에 규칙 정책의 진짜 마무리가 섞인다 —
        # 실측으로 29.3 m 감아차기 마무리가 14.8 m 앞·조준선 1.5 m 옆의 동료 때문에
        # 패스로 분류됐다. 같은 (상태, 행동)으로 '골문을 향한 슛'과 '중앙으로 뛰어드는
        # 동료를 향한 킬패스'를 모두 만들 수 있으므로, 액션에 킥 종류가 없는 한 기하로
        # 의도를 가르는 것은 원리적으로 불가능하다.
        #
        # 두 오류 중 무엇을 감수할지의 문제이고 답은 명확하다. 이 라벨의 주 소비자는
        # 규칙 정책을 모방하는 BC이므로 **교사의 슛을 패스로 적는 쪽**이 훨씬 나쁘다.
        # 골문을 향한 강한 전진 패스가 슛으로 남는 것은 감수한다.

        # 도달성 — 골라인까지 굴러갈 수 없는 공은 슛이 아니다. 사거리는 env 자신의
        # 감속 테이블에서 미리 적분해 둔 값을 보간해 쓴다(``_roll_range_knots``).
        # 공중 구간이 빠져 있어 보수적 하한이므로, 띄워 찬 장거리 슛을 잘라내지 않도록
        # 판정은 ``travel``이 그 하한을 **넘을 때만** 닫는다.
        roll_reach = jnp.interp(
            speed,
            jnp.asarray(e_cfg.roll_v_knots, jnp.float32),
            self._roll_range_knots,
        )
        reachable = travel <= roll_reach

        struck = speed >= e_cfg.dribble_speed_max
        is_shot = free_play & struck & aim_goal & reachable
        # A restart release is a pass/shot provenance event even when its
        # projected minimum speed lies below the open-play dribble threshold.
        # Calling a kickoff/FK/goal-kick/corner/GK distribution DRIBBLE makes
        # downstream back-pass/offside/inverse logic believe the taker carried
        # a live ball rather than putting it into play.
        # 속도가 dribble/control 범위인 자유 플레이 접촉은 직전 소유 표기가 중립이어도
        # DRIBBLE이다. ``win_team == poss >= 0``을 요구하면 중립 루즈볼을 3m/s로 내려놓는
        # 제어 터치가 PASS가 되고, 그 거짓 패스가 다음 프레임 즉시 패스와 offside/pass
        # latch를 다시 무장해 짧은 접촉 연쇄와 낮은 패스 성공률을 만든다.
        # 상대의 통제 공을 끊은 접촉은 위 ``interception`` 분기가 우선하므로 의미가 보존된다.
        is_dribble = (
            free_play
            & (~restart_active)
            & (~header)
            & (~struck)          # ``is_shot``과 같은 경계의 반대편 — 배타성이 구조로 보장된다.
        )
        code = jnp.where(interception, TOUCH_INTERCEPT,
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
        code = code.astype(jnp.int32)
        touch = state.touch.at[winner].set(jnp.where(any_cand & (code > 0), code, state.touch[winner]))

        # [IFAB Law 12/14] 페널티는 **접촉 위치**(피해 캐리어 proxy)로 판정한다 — 공 위치는 무관하다.
        # 공 위치를 함께 요구하면 공이 순간 박스 밖일 때 박스 안 반칙이 FK로 강등된다.
        in_box = fouler_in_box
        penalty = foul & in_box
        freekick = foul & (~in_box)
        foul_team = poss
        pen_spot = jnp.array([-foul_dir * (self.hx - e_cfg.penalty_spot), 0.0, self.r_ball])   # 파울러 골대 기준
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
        restart_t = restart_t.astype(jnp.int32)
        ball_state = jnp.where(foul, BALL_DEAD, state.ball_state)
        ball_state = jnp.where(eff_consume, BALL_ALIVE, ball_state).astype(jnp.int32)
        ball_pos = jnp.where(penalty, pen_spot, jnp.where(freekick, fk_spot, state.ball_pos))
        new_vel = jnp.where(foul, jnp.zeros(DIM_ALL), new_vel)

        is_throw_take = eff_consume & (state.restart_kind == RK_THROWIN)
        # 방향 계산에서 쓴 것과 **같은 값**이다 — 두 곳이 갈리면 방향이 보장하는 안쪽
        # 성분과 실제 속도가 어긋난다.
        throw_speed = jnp.where(
            is_throw_take, throw_speed_taken, requested_throw_speed)
        throw_vel = throw_speed * jnp.array([jnp.cos(launch_ang) * kick_dir[DIM_X],
                                             jnp.cos(launch_ang) * kick_dir[DIM_Y], jnp.sin(launch_ang)])
        # 스로어는 Law 15 자세를 위해 터치라인 밖에 서지만, 공 중심까지 손 좌표(py)에 놓으면
        # 릴리스 순간 이미 hy+r_ball 아웃 판정선을 0.12m 넘는다. 공만 판정선 바로 안쪽에
        # 투영해 '손은 밖, 공은 아직 전체가 라인을 넘지 않음'이라는 연속시간 릴리스를 표현한다.
        # 이후 바깥 방향 커맨드는 정상적으로 다시 아웃 판정을 받는다.
        throw_release_y = jnp.clip(py, -throw_release_limit, throw_release_limit)
        hands = jnp.array([
            px,
            throw_release_y,
            state.head_z[winner] + e_cfg.throw_height,
        ])
        new_vel = jnp.where(is_throw_take, throw_vel, new_vel)
        ball_pos = jnp.where(is_throw_take, hands, ball_pos)

        # 파울 재개는 직접 프리킥(또는 페널티). 종류를 넘기지 않으면 이전 state 값을 읽는다.
        taker_foul = self._designate_taker_when(
            foul,
            state, ball_pos[:DIM_Z], foul_team, jnp.bool_(False),
            jnp.int32(RK_FREEKICK), jnp.bool_(False))
        pending_taker = jnp.where(foul, taker_foul, state.pending_taker)
        pending_taker = jnp.where(eff_consume, jnp.int32(-1), pending_taker).astype(jnp.int32)

        throw_taker = jnp.where(eff_consume, jnp.int32(-1), state.throw_taker)
        throw_taker = jnp.where(is_throw_take, winner.astype(jnp.int32), throw_taker)
        throw_taker = jnp.where(foul, jnp.int32(-1), throw_taker).astype(jnp.int32)

        # 비스로인 세트피스 키커는 타인 접촉 전 재터치 금지(직접 득점은 허용 → throw_taker와 분리).
        # GK_HOLD는 세트피스가 아니라 플레이 중 키퍼 배급이다. 배급 후 같은 GK가
        # 발로 다시 플레이하는 것은 합법하고, 손으로 재취급하는 경우는 아래
        # ``gk_handling_restricted_team`` provenance가 별도로 IDFK를 판정한다.
        is_sp_take = (eff_consume
                      & (state.restart_kind != RK_THROWIN)
                      & (state.restart_kind != RK_GK_HOLD))
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
        # Offside is armed by a team-mate's deliberate play, not by the
        # possession label that happened to precede it.  In particular a
        # controlled pass/shot from a neutral loose ball changes ``new_poss``
        # to ``win_team`` above and must establish the same receiving window as
        # an ordinary same-possession pass.  Requiring ``win_team == poss``
        # left every such loose-ball pass exempt from offside.
        set_flags = (any_cand & free_play & is_played
                     & ((~restart_active) | offside_restart_kick) & (~offside_receive))
        mate_ahead = self._offside_positions_for_play(
            state, winner, win_team
        )
        off_flag = jnp.where(set_flags, mate_ahead, state.offside_flag)
        # A successful tackle/interception must retain the *previous* Law-11
        # phase until ``Offside._offside_check`` adjudicates this exact touch.
        # Clearing it here makes ``_normalize_pass_latch`` discard pass_team
        # and the flags before the ordered force-contact phase can see them;
        # a flagged attacker could then win a tackle without being called
        # offside.  The offside phase handler clears the old defender window
        # and arms the tackler's new team window in physical-touch order.
        deliberate_def = foul
        off_flag = jnp.where(deliberate_def, jnp.zeros(self.N, bool), off_flag)
        has_new_flags = jnp.any(mate_ahead)
        pass_team = jnp.where(
            set_flags,
            jnp.where(has_new_flags, win_team, jnp.int32(NO_TEAM)),
            state.pass_team,
        ).astype(jnp.int32)
        pass_t = jnp.where(
            set_flags, has_new_flags.astype(jnp.int32), state.pass_t
        ).astype(jnp.int32)

        # Challenge/re-challenge cooldown.  It belongs to the actor that
        # attempted a challenge, never to a same-possession-team controller.
        # On a normal contest that actor is ``winner``; in the retained
        # (loser-foul) branch the possessor won the argmax and ``chal`` is the
        # actual opposing lunger, so cooldown must target ``chal`` even when
        # the foul draw misses.  Otherwise that losing challenger can retry on
        # every physics substep without a possession transition/ctrl_lock.
        win_dist = dist_ball_all[winner]
        win_lunge = self._lunge_fraction(win_dist)
        winner_challenged = any_cand & ((win_team != poss) | (poss < 0))
        winner_cd = (jnp.float32(e_cfg.cooldown_substeps)
                     + win_lunge * e_cfg.challenge_cooldown_extra)
        cooldown = state.cooldown.at[winner].set(jnp.where(
            winner_challenged, winner_cd, state.cooldown[winner]))
        chal_dist = dist_ball_all[chal]
        chal_lunge = self._lunge_fraction(chal_dist)
        chal_cd = (jnp.float32(e_cfg.cooldown_substeps)
                   + chal_lunge * e_cfg.challenge_cooldown_extra)
        cooldown = cooldown.at[chal].set(jnp.where(
            retained, jnp.maximum(cooldown[chal], chal_cd), cooldown[chal]))
        # loser_foul도 제외 — 승자 코드가 TOUCH_NONE이라 남겨 두면
        # 'last_touch_team만 갱신 + last_touch_code=NONE'인 모순 상태가 obs로 나간다.
        touched_real = (any_cand & (free_play | tackle_ok | foul | deflect)
                        & (~loser_foul))
        last_touch = jnp.where(touched_real, win_team, state.last_touch_team).astype(jnp.int32)
        last_touch_code = jnp.where(touched_real, code, state.last_touch_code).astype(jnp.int32)
        last_touch_actor = jnp.where(
            touched_real, winner.astype(jnp.int32), state.last_touch_actor
        ).astype(jnp.int32)
        foot_targets_own_gk = self._play_targets_own_goalkeeper(
            state, winner, win_team, kick_dir, speed
        )
        throw_targets_own_gk = self._play_targets_own_goalkeeper(
            state, winner, win_team, kick_dir, throw_speed
        )
        deliberate_foot_to_gk = (
            any_cand
            & (free_play | tackle_ok)
            & (~foul)
            & (~header)
            & (~is_chest)
            & (~is_throw_take)
            & (speed > GEOMETRY_EPS)
            & foot_targets_own_gk
        )
        deliberate_throw_to_gk = (
            is_throw_take
            & (throw_speed > GEOMETRY_EPS)
            & throw_targets_own_gk
        )
        gk_distribution_release = (
            eff_consume
            & (state.restart_kind == RK_GK_HOLD)
            & winner_is_gk
        )
        restricted_gk_retouch = (
            touched_real
            & winner_is_gk
            & (handling_team0 == win_team)
        )
        # Law 12's narrow exception applies only to a team-mate back-pass or
        # throw-in: after the goalkeeper clearly kicks/attempts to kick it to
        # release the ball into play, handling is allowed.  It does *not* cure
        # the separate ban created when the goalkeeper first released hand
        # possession.  A successful submitted foot contact is this engine's
        # observable clearance-attempt event.
        keeper_clearly_kicks_to_release = (
            restricted_gk_retouch
            & (~handling_from_release0)
            & winner_wants_kick
            & (free_play | tackle_ok)
            & (~foul)
            & (~header)
            & (~is_chest)
            & (~is_throw_take)
            & (speed > GEOMETRY_EPS)
        )
        preserve_restricted_gk_retouch = (
            restricted_gk_retouch & (~keeper_clearly_kicks_to_release)
        )
        # A real contact by *another* player ends both handling restrictions:
        # direct receipt from a team-mate's deliberate kick/throw, and a
        # goalkeeper re-handling the ball after releasing possession.  The
        # restricted goalkeeper's own body play is not an intervening-player
        # touch.  Their foot play clears only the Law-12 back-pass cause via
        # the explicit clearance exception above, never their own-release ban.
        # Consuming GK_HOLD is the instant the keeper releases the ball and
        # therefore arms the second restriction.  A newly targeted team-mate
        # foot play or throw then arms the same action-legality latch.
        gk_handling_restricted_team = jnp.where(
            (touched_real & (~preserve_restricted_gk_retouch)) | foul | gk_claim,
            jnp.int32(NO_TEAM),
            state.gk_handling_restricted_team,
        )
        gk_handling_restricted_team = jnp.where(
            deliberate_foot_to_gk | deliberate_throw_to_gk,
            win_team,
            gk_handling_restricted_team,
        )
        gk_handling_restricted_team = jnp.where(
            gk_distribution_release,
            win_team + jnp.int32(GK_HANDLING_RELEASE_OFFSET),
            gk_handling_restricted_team,
        ).astype(jnp.int32)
        foul_kind = jnp.where(foul, jnp.int32(FOUL_TACKLE),
                              jnp.where(eff_consume, jnp.int32(FOUL_NONE), state.foul_kind))
        # This foul branch is explicitly a challenge on ``carrier``.  Looking
        # up the nearest possession-team player to the abstract lunger could
        # instead label a nearby supporting team-mate as the victim, while the
        # restart spot/box decision above correctly used the carrier contact
        # proxy.  Keep event identity and geometry on the same causal actor.
        victim = carrier
        foul_actor = jnp.where(foul, fouler.astype(jnp.int32), state.foul_actor)   # #10c: 파울러=도전 수비수
        foul_victim = jnp.where(foul, victim.astype(jnp.int32), state.foul_victim)

        # ── GK 캐치/홀드/parry/백패스 결과 오버라이드 (gk_claim은 위 free_play·contest에서 배제됨) ──
        opp_team = (1 - win_team).astype(jnp.int32)
        # Handling legality is decided at the ball's contact point.  A keeper
        # whose centre is just outside the area may legally reach back across
        # the line, but snapping the catch to the keeper's feet would then
        # manufacture an outside-box GK_HOLD immediately after a legal catch.
        # Preserve the contact x/y and only settle z/velocity for the hold.
        hold_ball_pos = jnp.array([
            state.ball_pos[DIM_X], state.ball_pos[DIM_Y], self.r_ball,
        ])
        # The technical offence occurs where the goalkeeper handles the ball,
        # not at the keeper's feet (reach can separate them by up to ~2 m).
        # Law 13 additionally moves an attacking IDFK from inside the
        # defenders' goal area to the parallel goal-area line.
        bp_spot = self._attacking_indirect_fk_spot(
            state.ball_pos[:DIM_Z], attack_dir
        )
        # 백패스는 **간접** 프리킥이다.
        bp_taker = self._designate_taker_when(
            gk_handling_offence,
            state, bp_spot[:DIM_Z], opp_team, jnp.bool_(False),
            jnp.int32(RK_FREEKICK), jnp.bool_(True))
        # parry: 자기 골 반대(자기 attack_dir 방향)로 쳐냄 + 살짝 위 — 입사속도·attack_dir의 결정 함수(역산 가능).
        parry_speed = e_cfg.deflect_out_frac * ball_speed0 + e_cfg.deflect_out_base
        parry_punch = jnp.array([
            attack_dir * parry_speed,
            state.ball_vel[DIM_Y] * e_cfg.parry_lateral_keep,
            e_cfg.parry_lift_frac * parry_speed,
        ])
        # 손끝 세이브(tip-over) — 되돌릴 수 없을 만큼 빠른 슛은 진행방향을 유지한 채
        # 크로스바 위로 흘려보낸다. 전방 펀치만 두면 골라인을 넘는 공이 아예 없어
        # 코너가 0건이 된다(실측 900초·3시드에서 코너 0 · K리그 8.6건). 실제 GK도
        # 세이브의 절반을 펀치로 처리하고(K리그 636건 중 50.8%) 그 상당수가 코너다.
        # 분기 기준은 접촉 전 상태의 결정 함수라 inverse.predict_parry가 그대로 재현한다.
        own_goal_x = -attack_dir * self.hx
        tip_h = state.ball_vel[:DIM_Z] * e_cfg.parry_tip_keep
        tip_speed_x = jnp.abs(tip_h[DIM_X])
        to_line = jnp.abs(own_goal_x - state.ball_pos[DIM_X])
        flight_t = to_line / jnp.maximum(tip_speed_x, DIV_EPS)
        # 골라인에서 크로스바보다 clearance만큼 위에 있으려면 필요한 수직 초속도.
        need_vz = (
            (self.goal_h + e_cfg.parry_tip_clearance - state.ball_pos[DIM_Z])
            / jnp.maximum(flight_t, DIV_EPS)
            + 0.5 * e_cfg.g * flight_t
        )
        # 골 쪽으로 오는 공만, 그리고 GK가 실제로 낼 수 있는 힘 안에서만 넘긴다.
        toward_goal = state.ball_vel[DIM_X] * attack_dir < 0.0
        # 무항력 탄도식은 거리가 길수록 실제보다 높게 예측한다(실측: 12m에서
        # 0.57m·16m에서 0.76m 모자라 골이 됐다). 손끝 세이브는 원래 골라인 가까이에서
        # 일어나므로 사거리를 제한하고 여유 높이를 그 범위에 맞춰 잡는다.
        tip_over = (
            (ball_speed0 > e_cfg.parry_punch_speed_cap)
            & toward_goal
            & (need_vz > 0.0)
            & (need_vz <= e_cfg.parry_tip_lift_max)
            & (to_line <= e_cfg.parry_tip_max_range)
        )
        parry_tip = jnp.array([tip_h[DIM_X], tip_h[DIM_Y], need_vz])
        parry_vec = jnp.where(tip_over, parry_tip, parry_punch)
        gk_code = jnp.where(gk_parry, jnp.int32(TOUCH_PARRY), jnp.int32(TOUCH_GK_CATCH))
        # 공 속도 0·키커 초기화는 두 경우 모두 필요하지만, **인플레이 여부는 다르다**.
        # IFAB Law 9에서 골키퍼가 손으로 통제하는 동안 공은 인플레이다 — 재개가 아니므로
        # 상대에게 이격 의무도 제외구역도 없다(``_restart_projection_active``가 이미 GK_HOLD를
        # 뺀다). 실제로 죽는 것은 백패스 손취급 반칙으로 간접 프리킥이 열릴 때뿐이다.
        gk_dead = gk_hold_catch | gk_handling_offence
        gk_ball_dead = gk_handling_offence

        ball_pos = jnp.where(gk_hold_catch, hold_ball_pos,
                            jnp.where(gk_handling_offence, bp_spot, ball_pos))
        new_vel = jnp.where(gk_dead, jnp.zeros(DIM_ALL), jnp.where(gk_parry, parry_vec, new_vel))
        new_spin = jnp.where(gk_claim, jnp.zeros(DIM_ALL), new_spin)
        ball_state = jnp.where(gk_ball_dead, jnp.int32(BALL_DEAD), ball_state).astype(jnp.int32)
        restart_kind = jnp.where(gk_hold_catch, jnp.int32(RK_GK_HOLD),
                        jnp.where(gk_handling_offence, jnp.int32(RK_FREEKICK), restart_kind)).astype(jnp.int32)
        restart_team = jnp.where(gk_hold_catch, win_team,
                        jnp.where(gk_handling_offence, opp_team, restart_team)).astype(jnp.int32)
        restart_t = jnp.where(gk_hold_catch, jnp.int32(e_cfg.gk_hold_substeps),
                     jnp.where(gk_handling_offence, jnp.int32(e_cfg.restart_substeps), restart_t)).astype(jnp.int32)
        new_poss = jnp.where(gk_hold_catch, win_team,
                    jnp.where(gk_handling_offence, opp_team,
                     jnp.where(gk_parry, jnp.int32(-1), new_poss))).astype(jnp.int32)  # parry=비통제→중립
        pending_taker = jnp.where(gk_hold_catch, winner.astype(jnp.int32),
                         jnp.where(gk_handling_offence, bp_taker.astype(jnp.int32), pending_taker)).astype(jnp.int32)
        touch = jnp.where(gk_claim, touch.at[winner].set(gk_code), touch)
        last_touch = jnp.where(gk_claim, win_team, last_touch).astype(jnp.int32)
        last_touch_code = jnp.where(gk_claim, gk_code, last_touch_code).astype(jnp.int32)
        last_touch_actor = jnp.where(
            gk_claim, winner.astype(jnp.int32), last_touch_actor
        ).astype(jnp.int32)
        setpiece_taker = jnp.where(gk_dead, jnp.int32(-1), setpiece_taker).astype(jnp.int32)
        throw_taker = jnp.where(gk_dead, jnp.int32(-1), throw_taker).astype(jnp.int32)
        foul_kind = jnp.where(gk_handling_offence, jnp.int32(FOUL_NONE), foul_kind)  # 손 취급=카드 없는 기술 반칙
        # 간접FK 플래그: 백패스=True / 일반 파울·GK 홀드캐치=False(직접) / 그 외 carry(진행 중 IDFK 유지).
        # GK가 진행 중 간접FK(백패스·오프사이드 IDFK)를 잡으면 홀드는 새 직접 재개다 — 클리어하지 않으면
        # 스테일 IDFK가 GK 배급까지 살아 정당한 직접골을 무효화(events indirect_direct)한다.
        restart_indirect = jnp.where(gk_handling_offence, jnp.bool_(True),
                            jnp.where(foul | gk_hold_catch, jnp.bool_(False), state.restart_indirect))

        state = state._replace(ball_pos=ball_pos, ball_vel=new_vel, ball_spin=new_spin,
                               poss_team=new_poss.astype(jnp.int32), cooldown=cooldown,
                               last_touch_team=last_touch, touch=touch, foul_kind=foul_kind,
                               gk_handling_restricted_team=gk_handling_restricted_team,
                               foul_actor=foul_actor, foul_victim=foul_victim,
                               restart_kind=restart_kind, restart_team=restart_team, restart_t=restart_t,
                               ball_state=ball_state, pending_taker=pending_taker, offside_flag=off_flag,
                               pass_team=pass_team, pass_t=pass_t, throw_taker=throw_taker,
                               setpiece_taker=setpiece_taker,
                               last_touch_code=last_touch_code, last_touch_actor=last_touch_actor,
                               restart_indirect=restart_indirect)
        card_mask = (jnp.arange(self.N) == fouler) & foul
        state = self._draw_cards(
            state,
            card_mask,
            k_card,
            foul_pos=foul_pos,
            randomness=randomness,
            substep_index=substep_index,
        )
        state = self._normalize_pass_latch(state, clear=foul | gk_dead)
        # Restart consumption clears ``foul_kind`` here, before the outer
        # environment scan reaches events.  Keep this directly callable rule
        # primitive coherent on its own instead of exposing FOUL_NONE with
        # the old actor/victim identities until that later normalization.
        state = self._normalize_foul_latch(state)
        # ``touch`` is a control-frame accumulator, not an event stream.  A
        # second PASS/TACKLE/etc. by the same actor in a later physics tick
        # leaves the stored integer unchanged, so ``touch_after !=
        # touch_before`` cannot observe that contact.  Runtime Law-11 and
        # retouch adjudication therefore consume this explicit per-substep
        # event mask, just as ``Ball._ball_body`` already does for repeated
        # BODY_TRAP/DEFLECT contacts.  Direct/public callers retain the
        # State-only return unless they opt in.
        force_event = (
            (self.player_indices == winner)
            & any_cand
            & ((code > TOUCH_NONE) | gk_claim)
        )
        return (state, force_event) if return_event else state
