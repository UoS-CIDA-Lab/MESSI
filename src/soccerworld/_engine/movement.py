import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MAX_POSITION_SOLVER_ROUNDS
from .constants import (
    BALL_ALIVE,
    COINCIDENT_DISTANCE_EPS,
    DIM_X,
    DIM_Y,
    DIM_Z,
    DIV_EPS,
    GEOMETRY_EPS,
    GK_HANDLING_RELEASE_OFFSET,
    GOLDEN_ANGLE,
    HALF_TURN,
    OFFPITCH_BENCH,
    OFFPITCH_RETIRED,
    OFFPITCH_SENT_OFF,
    RK_KICKOFF,
    RK_THROWIN,
    SAFE_NORM_EPS,
    SQUARED_EPS,
    STATIONARY_SPEED_EPS,
    SUBSTITUTION_ENTRY_DIRECTIONS,
    SUBSTITUTION_ENTRY_RINGS,
    TEAM_0,
    TOUCH_DRIBBLE,
    TOUCH_INTERCEPT,
    TOUCH_NONE,
    TOUCH_PASS,
    TOUCH_PASS_HEAD,
    TOUCH_SHOOT,
    TOUCH_SHOOT_HEAD,
    TOUCH_TACKLE,
)
from .energy import effective_speed_cap, stamina_step
from .restart import (
    _coerce_public_bool,
    _coerce_public_float32,
    _coerce_public_signed_int_scalar,
    restart_timer_active,
)
from .spatial import _safe_norm, _unit, reach_blockable


class _ReachContext(NamedTuple):
    in_reach: jax.Array
    distance_xy: jax.Array
    goalkeeper_can_handle: jax.Array


class Movement:
    def _within_team_rank(self, state):
        """Stable rank within each roster half, independent of team labels."""

        return jnp.where(
            state.team_id == TEAM_0,
            self.player_indices,
            self.player_indices - self.n_agents,
        )

    def _covariant_slot_side(self, state):
        """Deterministic per-slot ±1 side that rotates with team identity.

        Exact centreline geometry needs a tie-break, but global slot parity is
        not covariant: swapping two even-sized teams preserves parity while a
        180-degree rotation requires the chosen side to flip.  Fold parity to
        a within-team rank, then unfold it with the player's attack direction.
        Corresponding players therefore retain their rank and reverse the
        resulting world-side sign under a team swap/rotation.
        """

        within_team_rank = self._within_team_rank(state)
        rank_side = jnp.where(
            (within_team_rank & 1) == 0, 1.0, -1.0
        )
        return state.attack_dir * rank_side

    def _covariant_slot_directions(self, state):
        """Golden-angle unit vectors in a team-covariant 180-degree frame."""

        angle = self._within_team_rank(state).astype(jnp.float32) * GOLDEN_ANGLE
        folded = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1)
        return folded * state.attack_dir[:, None]

    def _player_pos_for_field_rules(self, player_pos):
        """라인 밖 물리 좌표를 IFAB 필드 규칙 계산용 실제 경계에 투영한다.

        선수는 물리적으로 margin까지 나갈 수 있지만, 오프사이드 같은 경기장 규칙에서 골라인
        밖 수비수는 골라인 위에 있는 것으로 취급해야 한다. 물리 위치 자체는 변경하지 않는다.
        """

        return jnp.stack(
            [jnp.clip(player_pos[..., DIM_X], -self.hx, self.hx),
             jnp.clip(player_pos[..., DIM_Y], -self.hy, self.hy)],
            axis=-1,
        )

    def _inactive_bench_positions(self, state=None):
        """비활성 slot의 결정적 기술구역 좌표(N,2).

        경기장 밖 선수는 셋으로 나뉘고 **되돌릴 수 있는가**가 다르다. 한 줄에 섞어 놓으면
        리플레이를 보는 사람이 '아직 들어올 수 있는 사람'과 '이미 끝난 사람'을 구분할 수
        없다. 그래서 터치라인에서의 거리로 구역을 나눈다.

          퇴장(:data:`OFFPITCH_SENT_OFF`)    터치라인에서 가장 멀리 — 돌아올 수 없다
          교체 아웃(:data:`OFFPITCH_RETIRED`) 중간
          그 외(:data:`OFFPITCH_BENCH`)      터치라인에 가장 가까이 — 들어올 수 있다

        ``state``가 없으면 구역을 알 수 없어 한 줄에 둔다 — 적분기와 투영이 서로 다른 줄을
        쓰면 내부 state와 밖으로 나가는 state가 갈리므로, 호출부는 가능하면 state를 준다.
        """

        e_cfg = self.e_cfg
        bench_x = jnp.clip(
            -self.hx + e_cfg.bench_first_x_offset
            + self.player_indices * e_cfg.bench_spacing,
            -self.hx + e_cfg.bench_boundary_inset,
            self.hx - e_cfg.bench_boundary_inset,
        )
        base_y = -(self.hy - e_cfg.bench_touchline_inset)
        if state is None:
            return jnp.stack([bench_x, jnp.full(self.N, base_y)], axis=1)
        zone = self._offpitch_zone(state)
        offset = zone.astype(jnp.float32) * e_cfg.offpitch_zone_spacing
        return jnp.stack([bench_x, base_y - offset], axis=1)

    def _offpitch_zone(self, state):
        """비활성 slot의 구역 코드 (N,) — 퇴장/교체아웃/벤치.

        퇴장은 ``sent_off``로 직접 안다. 교체 아웃은 슬롯 identity가 덮여 사라지므로
        슬롯만 봐서는 알 수 없다 — ``slot_generation``이 0보다 크면 그 슬롯에서 사람이
        한 번 이상 바뀌었다는 뜻이고, 지금 비활성이면 마지막 사람이 나간 것이다.
        """

        sent_off = state.sent_off
        replaced = state.slot_generation > 0
        return jnp.where(
            sent_off,
            jnp.int32(OFFPITCH_SENT_OFF),
            jnp.where(replaced, jnp.int32(OFFPITCH_RETIRED), jnp.int32(OFFPITCH_BENCH)),
        ).astype(jnp.int32)

    def _project_inactive_players(self, state):
        """외부로 반환되는 state에서 비활성 slot의 위치·속도·facing을 즉시 정규화한다."""

        active = state.active_player[:, None]
        player_pos = jnp.where(
            active, state.player_pos, self._inactive_bench_positions(state)
        )
        player_vel = jnp.where(active, state.player_vel, jnp.zeros_like(state.player_vel))
        player_facing = self.facing_from_velocity(player_vel, state.attack_dir)
        return state._replace(
            player_pos=player_pos,
            player_vel=player_vel,
            player_facing=player_facing,
        )

    def effective_vmax(self, vmax, stamina_long, stamina_short):
        """장기·단기 stamina가 반영된 실효 최고속의 공개 단일 진실원천.

        ``_move_with_energy``가 매 서브스텝 이 값으로 속력을 자른다. 외부에서 상태를 주입하는
        경로(교체 투입 등)도 같은 식으로 검증해야 ``|v| <= effective_vmax``라는 상태 불변식이
        유지된다. 불변식이 깨지면 다음 서브스텝의 vmax 캡이 마찰 타원을 훨씬 넘는 Δv를 만들고,
        그 프레임은 ``inverse.infer_move_action``으로 역산할 수 없게 된다.
        """

        vmax, _ = _coerce_public_float32(
            "vmax", vmax, minimum=0.0, fallback=0.0
        )
        stamina_long, _ = _coerce_public_float32(
            "stamina_long", stamina_long,
            minimum=0.0, maximum=1.0, fallback=0.0
        )
        stamina_short, _ = _coerce_public_float32(
            "stamina_short", stamina_short,
            minimum=0.0, maximum=1.0, fallback=0.0
        )
        shapes = [x.shape for x in (vmax, stamina_long, stamina_short) if x.shape != ()]
        if shapes and any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(
                "vmax, stamina_long and stamina_short must have the same shape "
                f"or be scalar, got {vmax.shape}, {stamina_long.shape}, "
                f"{stamina_short.shape}"
            )
        e_cfg = self.e_cfg
        return effective_speed_cap(
            vmax,
            stamina_long,
            stamina_short,
            long_floor=e_cfg.long_stamina_vmax_floor,
            short_floor=e_cfg.short_stamina_vmax_floor,
            short_knee=e_cfg.short_stamina_headroom_knee,
        )

    def stamina_transition(
        self,
        state,
        player_vel=None,
        previous_player_vel=None,
        locomotion_mask=None,
    ):
        """공개 energy adapter — env/dataset/inverse가 같은 물리 substep 식을 호출한다."""

        if player_vel is None:
            velocity = state.player_vel
        else:
            velocity, _ = _coerce_public_float32(
                "player_vel", player_vel, (self.N, DIM_Z)
            )
        if previous_player_vel is None:
            previous_velocity = state.player_vel
        else:
            previous_velocity, _ = _coerce_public_float32(
                "previous_player_vel", previous_player_vel, (self.N, DIM_Z)
            )
        if locomotion_mask is None:
            locomotion = state.active_player
        else:
            locomotion = _coerce_public_bool(
                "locomotion_mask", locomotion_mask, (self.N,)
            )
        return stamina_step(
            state.stamina_long,
            state.stamina_short,
            velocity,
            previous_velocity,
            state.active_player,
            locomotion,
            dt=self.e_cfg.dt_phys,
            long_drain_base=self.long_stamina_drain_base,
            long_tail_knee=self.e_cfg.long_stamina_tail_knee,
            long_tail_decay=self.long_stamina_tail_decay,
            sprint_speed=self.e_cfg.sprint_speed,
            long_sprint_mult=self.e_cfg.long_stamina_sprint_mult,
            long_idle_load=self.e_cfg.long_stamina_idle_load,
            long_speed_ref=self.e_cfg.long_stamina_speed_ref,
            long_speed_load_weight=self.e_cfg.long_stamina_speed_load,
            long_accel_ref=self.e_cfg.long_stamina_accel_ref,
            long_accel_load_weight=self.e_cfg.long_stamina_accel_load,
            vmax=state.vmax,
            endurance_factor=state.endurance_factor,
            long_vmax_floor=self.e_cfg.long_stamina_vmax_floor,
            short_depletion_s=self.e_cfg.short_stamina_depletion_s,
            short_depletion_speed_frac=self.e_cfg.short_stamina_depletion_speed_frac,
            short_speed_exponent=self.e_cfg.short_stamina_speed_exponent,
            short_accel_ref=self.e_cfg.short_stamina_accel_ref,
            short_accel_load_weight=self.e_cfg.short_stamina_accel_load,
            short_recovery_tau_s=self.e_cfg.short_stamina_recovery_tau_s,
            short_recovery_speed_frac=self.e_cfg.short_stamina_recovery_speed_frac,
            short_recovery_exponent=self.e_cfg.short_stamina_recovery_exponent,
            short_long_recovery_penalty=self.e_cfg.short_stamina_long_recovery_penalty,
        )

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

    def _foul_in_own_box(self, pos, attack_dir):
        """직접 프리킥 반칙이 자기 팀 페널티킥으로 분류되는 위치인지 판정한다.

        피치 안에서는 통상적인 자기 페널티구역을 사용한다. 경기장 밖에서 경기의
        일부로 일어난 반칙은 IFAB Law 12에 따라 가장 가까운 경계선 지점에서 재개하며,
        그 지점이 가해 팀 페널티구역의 골라인 구간이면 페널티킥이다. 따라서 자기
        골라인 *뒤*이면서 ``|y| <= pen_hw``인 좌표도 포함한다. 터치라인 밖, 상대
        골라인 뒤, 자기 골라인 뒤라도 페널티구역 폭 밖인 좌표는 포함하지 않는다.
        """

        x = pos[..., DIM_X]
        y = pos[..., DIM_Y]
        own_goal_x = -attack_dir * self.hx
        inside_pitch_box = self._in_own_box(pos, attack_dir, clamp_x=True)
        behind_own_goal = (
            ((x - own_goal_x) * attack_dir < 0.0)
            & (jnp.abs(y) <= self.pen_hw)
        )
        return inside_pitch_box | behind_own_goal

    def _ball_in_each_own_box(self, state):
        """공 중심이 각 선수의 자기 페널티박스 안인지 반환한다(bool[N]).

        골키퍼의 손 사용 가능 여부는 골키퍼의 발 위치가 아니라 **공의 위치**로
        결정된다. ``attack_dir``가 슬롯별이므로 하나의 공 좌표를 넣어도 선수별
        자기 박스 판정으로 브로드캐스트된다. 경계선은 페널티박스의 일부이며,
        골라인 뒤의 공은 허용하지 않도록 실제 피치 x 범위도 함께 요구한다.
        """

        return self._in_own_box(
            state.ball_pos[:DIM_Z], state.attack_dir, clamp_x=True
        )

    def facing_from_velocity(self, player_vel, attack_dir):
        """facing = **현재 속도 방향**(단일 진실원천). 독립 적분 상태가 아니라 파생값이다.

        `state.player_facing`은 이 함수의 캐시일 뿐이며, 속도가 바뀌는 지점(`_move`)에서 매번
        다시 계산된다. 따라서 facing은 **관측 가능한 양(속도·attack_dir)만으로 완전히 결정**되고,
        obs가 facing을 직접 담지 않아도 정책이 잃는 정보가 없다(anchor `abs_vel`과 각 토큰의
        `rel_vel`에서 전 선수 절대속도를 복원한다).

        정지(|v| ≤ eps) 시 방향은 정의되지 않으므로 **자기 공격 방향**으로 둔다. 은닉 래치(직전
        방향 유지)를 두면 한 프레임 관측만으로는 복원할 수 없는 상태가 되살아나므로 쓰지 않는다.
        arctan2(±0, ±0)의 IEEE −0 우연에 기대지 않도록 정지 분기를 명시적으로 가른다.

        회전 속도 상한은 별도 계수가 아니라 **마찰 타원의 횡가속 캡(`accel_norm_max`)**이 만든다 —
        속도 방향이 물리적으로 꺾일 수 있는 만큼만 facing도 꺾인다.
        """
        player_vel, _ = _coerce_public_float32(
            "player_vel", player_vel, (self.N, DIM_Z)
        )
        attack_dir, _ = _coerce_public_float32(
            "attack_dir", attack_dir, (self.N,),
            fallback=1.0, allowed_values=(-1.0, 1.0),
        )
        speed = _safe_norm(player_vel, axis=1)
        ang = jnp.arctan2(player_vel[:, DIM_Y], player_vel[:, DIM_X])
        rest = jnp.where(attack_dir >= 0.0, 0.0, HALF_TURN)
        return jnp.where(speed > STATIONARY_SPEED_EPS, ang, rest)

    def _reach_context(self, state):
        """Compute the shared player-ball reach geometry for one physical state."""

        e_cfg = self.e_cfg
        rel_xy = state.ball_pos[:DIM_Z][None, :] - state.player_pos
        d_xy = jnp.linalg.norm(rel_xy, axis=-1)

        # [2026-08-13] 역할별 reach: 소유팀 선수 = carry(1.1), 그 외(상대·중립 루즈볼) =
        # challenge(1.4, 런지 존 페널티는 contest에서). GK는 자기 박스 안 확장 유지.
        # 소유 판정은 poss_team — K리그 벤더 기준(터치 기반, 중립 있음)과 정합된 전이 규칙.
        carry_role = (state.team_id == state.poss_team) & (state.poss_team >= 0)
        base_r = jnp.where(carry_role, e_cfg.reach_xy_carry, e_cfg.reach_xy_challenge)
        gk_can_handle = (
            (state.gk_indices == 1) & self._ball_in_each_own_box(state)
        )
        Rxy = jnp.where(gk_can_handle, e_cfg.gk_reach_xy, base_r) + self.r_ball
        reachable = (d_xy <= Rxy) & (state.ball_pos[DIM_Z] <= state.reach_z + self.r_ball)

        # 빠르고 높은 공은 reach 안에 있어도 **의도적 제어 후보에서 뺀다**. 기하만 보면
        # 가슴 높이로 30 m/s에 날아가는 공도 발만 뻗으면 잡히는데 실측에는 그런 접촉이
        # 없다. 몸통 충돌 경로(ball.py의 스윕 판정)는 이 게이트를 거치지 않으므로 직접
        # 맞으면 여전히 굴절·트랩된다 — "발 뻗어 잡기"만 막고 "몸으로 막기"는 남는다.
        #
        # 속도 규약은 접촉 밴드와 같은 xy다. 수직 성분을 넣으면 낙하 중인
        # 공이 더 어렵게 보이지만 캘리브가 xy로 잡혀 있어 미보정 상태가 된다.
        ball_speed_xy = _safe_norm(state.ball_vel[jnp.newaxis, :DIM_Z], axis=1)[0]
        blockable = reach_blockable(
            ball_speed_xy, state.ball_pos[DIM_Z],
            e_cfg.reach_block_limit, e_cfg.reach_height_penalty,
        )
        # GK는 예외다 — 실제 키퍼는 이 선을 넘는 슛을 선방한다. 캘리브 표본에도 그 선방이
        # 섞여 있어 임계값 자체가 이미 GK 쪽으로 관대하다.
        return _ReachContext(
            in_reach=reachable & (blockable | gk_can_handle),
            distance_xy=d_xy,
            goalkeeper_can_handle=gk_can_handle,
        )

    def _in_reach(self, state):
        """
        선수 reach 여부 계산: 수평거리 ≤ Rxy & 공높이 ≤ reach_z + r_ball.
        필드 선수는 소유 역할에 따라 ``reach_xy_carry``/``reach_xy_challenge``를 쓰고,
        GK는 자기 박스 안에서 ``gk_reach_xy``를 쓴다.
        """

        context = self._reach_context(state)
        return context.in_reach, context.distance_xy

    def _gk_reactive_claim(
        self, state, reach_context: _ReachContext | None = None
    ):
        """오픈플레이에서 자기 박스 안 GK가 도달 가능한 라이브 공에 대해 want_f2b 없이도 경합 후보가
        되게 하는 마스크(N,) — 키퍼 본능. contest 후보 마스크에만 더해지고 이동 facing엔 섞이지 않는다
        (do_kick과 분리). 실제 캐치/parry/백패스 분기는 contest._apply_force2ball이 승자 기준으로 판정.

        ★합법 캐치일 때만 발동: 잡는 행위 자체가 반칙인 공 — ①자신이 일반
        세트피스 재터치 금지 추적자 ②같은팀 발패스/자신의 GK 배급(손 재취급 = IDFK)
        — 은 본능 캐치에서
        제외한다. 실제 GK는 그런 공에 손을 대지 않는다(발 플레이는 want_f2b 경로로 여전히 가능).
        이 게이트가 없으면 '홀드 배급 → 공이 gk_reach_xy(2.0 m)를 못 벗어남 → cooldown 만료 즉시 자동
        재캐치 = 재터치 IDFK → 상대 슛 → 캐치 홀드 → 배급 → …' 무한 루프가 된다."""
        if reach_context is None:
            reach_context = self._reach_context(state)
        reachable = reach_context.in_reach
        gk_can_handle = reach_context.goalkeeper_can_handle
        alive = state.ball_state == BALL_ALIVE
        restart_active = restart_timer_active(state.restart_t)
        ar = self.player_indices
        retouch_locked = (state.setpiece_taker == ar) | (state.throw_taker == ar)
        # 손처리 제한은 마지막 터치 종류가 아니라 접촉 순간 설정된 인과 provenance가
        # SSOT다. 0/1은 동료 백패스·직접 스로인, 2/3은 GK가 손에서 놓은 뒤 타인 미접촉을
        # 뜻한다. 어느 원인이든 현재 제한 팀을 복원해 리액티브 손 캐치를 막는다.
        handling_code = state.gk_handling_restricted_team
        handling_team = jnp.where(
            handling_code >= GK_HANDLING_RELEASE_OFFSET,
            handling_code - GK_HANDLING_RELEASE_OFFSET,
            handling_code,
        )
        handling_restricted = handling_team == state.team_id
        # A keeper who has just kicked a same-team play away from their own
        # goal must not immediately turn the still-reachable first 0.8 m of
        # that release into an automatic hand claim.  The causal handling
        # latch may legally clear after a team-mate back-pass foot play, but
        # "hands are legal" is not the same as "catch every departing ball".
        # Waiting until another player touches it, it slows, or it returns
        # toward goal removes the observed kick→0.067 s recatch loop without
        # blocking saves and genuine balls travelling toward the keeper.
        last_touch_was_team_play = (
            (state.last_touch_team == state.team_id)
            & (
                (state.last_touch_code == TOUCH_PASS)
                | (state.last_touch_code == TOUCH_PASS_HEAD)
                | (state.last_touch_code == TOUCH_SHOOT)
                | (state.last_touch_code == TOUCH_SHOOT_HEAD)
                | (state.last_touch_code == TOUCH_DRIBBLE)
                | (state.last_touch_code == TOUCH_TACKLE)
                | (state.last_touch_code == TOUCH_INTERCEPT)
            )
        )
        moving_outward = (
            state.ball_vel[DIM_X] * state.attack_dir
            > self.e_cfg.possession_release_speed
        )
        own_play_leaving = last_touch_was_team_play & moving_outward
        # challenge cooldown은 비소유/루즈볼에서만 효력. 소유를 얻은 뒤에는 남은 타이머가 정상
        # 제어를 막지 않지만, 다시 잃으면 재도전을 막는다. 같은 control step의 parry→재캐치는
        # cooldown 시간에 의존하지 않고 touch debounce가 구조적으로 차단한다.
        return (gk_can_handle & reachable & alive & (~restart_active)
                & (~retouch_locked) & (~handling_restricted)
                & (~own_play_leaving)
                & self._cooldown_allows_f2b(state)
                & self._contact_lock_allows_f2b(state)
                & self._aerial_recovery_allows_f2b(state)
                & (state.touch == TOUCH_NONE) & state.active_player)

    def _kicker_target(self, state):
        """키커 목표 위치(a 옵션): 공에서 자기 골 방향으로 (r_player+r_ball) 뒤.
        차러 들어가는 자세. 공이 스폿에 있을 때 그 뒤에 서도록. pending_taker<0(오픈플레이)이면
        clip으로 인덱스 안전화 — 실제 적용은 호출부의 active 게이트가 막는다."""
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        adt = state.attack_dir[taker]                 # 키커 공격 방향
        back = jnp.array([-adt, 0.0])                 # 자기 골 방향(= 공격 반대)
        kick_target = state.ball_pos[:DIM_Z] + back * (self.r_player + self.r_ball)
        # 스로인은 터치라인 밖에서 수행한다. 공과 같은 x에서 선수 반지름만큼
        # 라인 밖으로 배치해 상시 5m 외곽 물리 margin을 실제로 활용한다.
        touchline_side = jnp.where(state.ball_pos[DIM_Y] >= 0.0, 1.0, -1.0)
        throw_target = jnp.array([
            state.ball_pos[DIM_X],
            touchline_side * (self.hy + self.r_player),
        ])
        return jnp.where(state.restart_kind == RK_THROWIN, throw_target, kick_target)

    def _apply_kicker_move(self, state):
        """세트피스 활성(active) 중에만 키커를 목표 위치로 강제 이동(최대속도, 스태미나 무소모).
        도착 전까지만 당기고 오픈플레이(pending_taker<0)에선 아무도 움직이지 않는다 — active 게이트가
        없으면 pending_taker=-1이 클립되어 player[0]을 매 서브스텝 공으로 끌어 궤적을 오염시킨다."""
        _, _, _, active = self._setpiece_kick_lock(state)
        taker = jnp.clip(state.pending_taker, 0, self.N - 1)
        tgt = self._kicker_target(state)
        cur = state.player_pos[taker]
        taker_vmax = self.effective_vmax(
            state.vmax[taker], state.stamina_long[taker], state.stamina_short[taker]
        )
        new_tk_pos = self._advance_kicker_position(
            cur, tgt, state.restart_kind, taker_vmax
        )
        P = jnp.where(active, state.player_pos.at[taker].set(new_tk_pos), state.player_pos)
        # 강제이동 중 속도는 **실제 변위율**이어야 한다. 0으로 두면 위치는 kicker_speed로
        # 움직이는데 상태는 정지라고 말하게 되어, (a) 관측이 눈에 보이는 움직임과 어긋나고,
        # (b) facing이 진행방향 대신 기본값으로 잡히며, (c) inverse.infer_move_action이 그
        # 프레임을 재현하지 못한다(movement.effective_vmax 주석이 요구하는 상태 불변식).
        # 도착하면 변위가 0이 되어 자연스럽게 정지 속도가 된다.
        displacement = new_tk_pos - cur
        forced_velocity = displacement / self.e_cfg.dt_phys
        # Kick-off is an administrative placement rather than a traversed
        # path; exposing its potentially tens-of-metres-per-tick snap as a
        # physical velocity would violate the State speed contract.  Every
        # walking restart, by contrast, must report its actual displacement
        # rate instead of reporting zero-velocity ghost motion.
        forced_velocity = jnp.where(
            state.restart_kind == RK_KICKOFF,
            jnp.zeros(DIM_Z, dtype=forced_velocity.dtype),
            forced_velocity,
        )
        Vp = jnp.where(
            active,
            state.player_vel.at[taker].set(forced_velocity),
            state.player_vel,
        )
        Fc = self.facing_from_velocity(Vp, state.attack_dir)
        return state._replace(player_pos=P, player_vel=Vp, player_facing=Fc)

    def _snap_kickoff_taker(self, state):
        """Place a kickoff taker and reconcile the resulting position atomically.

        Kickoffs are the only restarts whose taker is snapped to the target in one
        physics tick.  A legal custom formation can therefore become illegal *because
        of* the snap: two players 0.68 m apart at ``(+0.34, 0)`` and ``(-0.34, 0)``
        collapse to the same point when the lower-slot player wins the taker tie.  The
        normal movement separator has not run yet on reset, goal, or halftime paths.

        Keep the designated taker pinned at the exact release target and move only the
        connected overlap component.  Re-run the restart projector afterwards so a
        displaced team-mate cannot be separated across the halfway line and an opponent
        cannot be separated into the centre circle.  All three kickoff creation paths
        call this primitive, while the per-substep walking hot path remains in
        :meth:`_apply_kicker_move`.
        """

        return self._snap_kickoff_taker_with_mask(state)[0]

    def _snap_kickoff_taker_with_mask(self, state):
        """Snap a kickoff taker and return the exact restart-projection mask.

        Reset only needs the legal State, while goal and halftime transitions
        also expose ``info["restart_position_forced"]``.  The latter must not
        lose a half/circle correction performed inside this atomic snap: once
        corrected, the next substep's idempotent projector cannot recover which
        slots the referee moved.  Collision reconciliation remains deliberately
        outside the returned mask; it is tracked by the broader non-policy
        position writer instead.
        """

        snapped = self._apply_kicker_move(state)
        taker = jnp.clip(snapped.pending_taker, 0, self.N - 1)
        valid_taker = (
            (snapped.pending_taker >= 0)
            & (snapped.pending_taker < self.N)
            & snapped.active_player[taker]
            & (snapped.team_id[taker] == snapped.restart_team)
            & (snapped.restart_kind == RK_KICKOFF)
            & restart_timer_active(snapped.restart_t)
        )
        taker_mask = (
            (self.player_indices == taker)
            & valid_taker
        )
        reconciled = self.reconcile_positions(
            snapped.player_pos,
            snapped.active_player,
            pinned=taker_mask,
            constrained=taker_mask,
            tie_direction=self._covariant_slot_directions(snapped),
            component_only=True,
        )
        moved = jnp.any(reconciled != snapped.player_pos, axis=1)
        velocity = jnp.where(
            moved[:, None], jnp.zeros_like(snapped.player_vel), snapped.player_vel
        )
        candidate = snapped._replace(
            player_pos=reconciled,
            player_vel=velocity,
            player_facing=self.facing_from_velocity(velocity, snapped.attack_dir),
        )
        candidate, restart_forced = self._project_restart_positions(candidate)
        clean = jax.tree_util.tree_map(
            lambda clean, original: jnp.where(valid_taker, clean, original),
            candidate,
            state,
        )
        return clean, restart_forced & valid_taker

    def _vel_substep(self, player_vel, v_cmd, vmax, eff_move):
        """이동 속도 갱신 커널 1서브스텝 — **속도-명령 모델**(guide.md §1). 목표속도 v_cmd를 향한
        속도 변화를 현재 진행방향 기준 **종(가·감속)·횡(선회)**으로 분해해 각각 상한으로 클립한다.
        → '물리적으로 가능한 속도 변화'를 환경 레벨에서 보장(정책은 '가고 싶은 속도'만 내고 도달
        가능성·관성은 여기서 처리). 포워드와 inverse.infer_move_action(역산)이 공유하는 단일 진실원천.

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
        # 성분별 독립 클립(박스)은 코너(동시 최대 종·횡)에서 √(cap_종²+cap_횡²)로 단일축 캡을 40~55%
        # 초과한다(가속턴 12.1·제동턴 13.5 > 캡 8.5~10.5). 타원이면 어느 방향으로도 그 방향 캡을 못 넘고
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

    def _move_with_energy(
        self,
        state,
        eff_move,
        mv_dir,
        mv_pow,
        separation_pinned=None,
        position_update_mask=None,
        locomotion_mask=None,
        return_domain_projection=False,
    ):
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
            separation_pinned: bool[N] | None — 충돌 분리에서도 고정할 선수.
                None이면 기존 계약대로 ``~eff_move``를 사용한다. 재개 위치 투영자는
                액션만 마스킹하고 충돌 해소는 허용해야 하므로 runtime이 키커만 전달한다.
            position_update_mask: bool[N] | None — 외부 규칙 이동/동결 뒤 이
                적분기가 다시 같은 선수를 이동시키지 않을 마스크. None이면 역사적
                동작대로 전 슬롯이 현재 속도로 적분된다.
            locomotion_mask: bool[N] | None — 정책이 만든 자기 추진 이동만 workload에
                포함할 마스크. None이면 ``position_update_mask``와 동일하다. 심판·세트피스
                엔진의 강제 위치 이동은 기본 부하만 내고 거리·가속 부하를 만들지 않는다.
            return_domain_projection: True면 데드볼 작업 영역에 있던 선수가
                프레임 중 오픈플레이 경계로 돌아와 축 클립된 마스크도 반환한다.
                정상 경기장 안에서 바깥 행동을 낸 경우는 포함하지 않는다.
        Return:
            기본은 ``(State, StaminaTransition)``. 요청 시 위치 영역 축소가
            쓴 ``bool[N]`` 마스크를 세 번째로 반환한다.
        """
        e_cfg = self.e_cfg
        dt = e_cfg.dt_phys
        # [속도-명령 모델] 이동 액션 = 목표속도 = 강도(mv_pow)·방향(mv_dir)·유효최고속(vmax).
        # 구모델의 a_vec = a_max·pow·dir(가속)를 대체 — 정책 부담↓(1적분), 클립은 _vel_substep이 처리.
        vmax = self.effective_vmax(
            state.vmax, state.stamina_long, state.stamina_short
        )
        v_cmd = (mv_pow[:, None] * mv_dir) * vmax[:, None]
        V = self._vel_substep(state.player_vel, v_cmd, vmax, eff_move)
        if position_update_mask is None:
            update_position = jnp.ones(self.N, dtype=bool)
        else:
            update_position = jnp.asarray(position_update_mask, dtype=bool)
            if update_position.shape != (self.N,):
                raise ValueError(
                    f"position_update_mask must have shape ({self.N},), got "
                    f"{update_position.shape}"
                )
        if locomotion_mask is None:
            locomotion = update_position
        else:
            locomotion = jnp.asarray(locomotion_mask, dtype=bool)
            if locomotion.shape != (self.N,):
                raise ValueError(
                    f"locomotion_mask must have shape ({self.N},), got "
                    f"{locomotion.shape}"
                )
        # A referee-projected/frozen player has no physical displacement in
        # this integrator.  Keeping an old nonzero velocity would create a
        # second ghost movement, false facing and stamina consumption.
        V = jnp.where(update_position[:, None], V, jnp.zeros_like(V))

        # 위치 적분 + 선수 허용 경계 클립. 오픈플레이에서 5 m짜리 재개 작업영역을 그대로
        # 허용하면 공을 쫓던 선수가 골라인 밖 수 m까지 달려 나간다. 실제 플레이 중에는 몸
        # 중심이 선수 반지름만큼 라인을 넘는 자연스러운 관성만 허용하고, 스로인 키커 배치 등
        # 심판/재개 엔진이 쓰는 데드볼 구간에는 기존 ``player_boundary_margin``을 유지한다.
        open_play = (
            (state.ball_state == BALL_ALIVE)
            & (~restart_timer_active(state.restart_t))
        )
        boundary_margin = jnp.where(
            open_play, jnp.float32(self.r_player),
            jnp.float32(e_cfg.player_boundary_margin),
        )
        bound_x = self.hx + boundary_margin
        bound_y = self.hy + boundary_margin
        P_free = jnp.where(
            update_position[:, None],
            state.player_pos + dt * V,
            state.player_pos,
        )
        P = P_free.at[:, 0].set(jnp.clip(P_free[:, 0], -bound_x, bound_x))
        P = P.at[:, 1].set(jnp.clip(P[:, 1], -bound_y, bound_y))
        if return_domain_projection:
            # A restart may release midway through a control frame. Players
            # can then start this physical tick inside the 5 m dead-ball
            # workspace but outside the tighter live-play domain. The axis
            # clamp below is an environment rewrite, not a policy displacement.
            # Match axes independently so an unrelated policy clamp on one
            # axis cannot mask a pre-existing violation on the other.
            outside_before = jnp.stack(
                [
                    jnp.abs(state.player_pos[:, DIM_X]) > bound_x,
                    jnp.abs(state.player_pos[:, DIM_Y]) > bound_y,
                ],
                axis=1,
            )
            axis_projected = P != P_free
            domain_projection = (
                state.active_player
                & jnp.any(outside_before & axis_projected, axis=1)
            )
        # 경계에서 잘린 축은 **속도 성분도 0으로** 만든다. 위치만 클립하면 위치는 멈췄는데 상태
        # 속도는 바깥으로 전속인 '유령 속도'가 남아, ①obs의 anchor/토큰 상대속도가 거짓을 말하고
        # ②facing(속도 파생)이 바깥을 향하며 ③스태미나가 계속 소모되고 ④반대 명령을 줘도 마찰
        # 타원 캡 때문에 실제 복귀까지 ~0.8 s가 걸린다. 클립은 그 축으로 **바깥으로** 밀 때만
        # 발생하므로 축 성분을 0으로 두는 것이 곧 바깥 법선 성분 제거다(안쪽 이동은 클립되지 않음).
        V = jnp.where(P != P_free, 0.0, V)

        # 비활성 선수(active_player=on_pitch & ~sent_off=False)는 결정적 기술구역 좌표에 고정·정지
        # → 확장된 선수 경계 안에서도 물리/규칙 유령화를 막는다.
        bench = self._inactive_bench_positions(state)
        P = jnp.where(state.active_player[:, None], P, bench)
        V = jnp.where(state.active_player[:, None], V, jnp.zeros_like(V))
        # [B3] 강제이동 중인 세트피스 키커(~eff_move)는 분리 밀어내기에서 제외(pin)한다. 상대 다수가
        # 스폿을 점거해도 키커가 목표 뒤로 확실히 도달(arrived=True)해 카운트다운·강제킥이 정상 진행 —
        # pin이 없으면 _apply_kicker_move가 매 서브스텝 재배치해도 _separate가 도로 밀어내 d_tgt가
        # kicker_arrive_r 밖에 갇혀 영구 데드볼(도착 전엔 카운트다운이 흐르지 않음). 오픈플레이에선
        # ~eff_move가 전부 False라 정상 분리.
        pinned = ~eff_move if separation_pinned is None else separation_pinned
        # 이동 핫패스는 값싼 Jacobi 완화를 쓴다. ``reconcile_positions``의 Gauss-Seidel이
        # 품질은 낫지만(다자 경합 근사 오차 최악 1.0 cm가 사라진다) N^2 순차 sweep이라
        # 전 서브스텝에 돌리면 **4.6배** 느려지고(실측 225 -> 49 step/s), 겹쳤을 때만
        # 조건부로 돌려도 23% 느리다(-> 173 step/s). 남는 1 cm는 선수 반지름의 4%이고 다음
        # 프레임에 스스로 해소되므로 그 값을 치를 이유가 없다. 규칙 개입이 만든 겹침처럼
        # 반드시 풀어야 하는 것은 ``reconcile_positions``가 그 경로에서 담당한다.
        P, V = self._separate(
            P,
            V,
            state.active_player,
            pinned=pinned,
            tie_direction=self._covariant_slot_directions(state),
            boundary_margin=boundary_margin,
        )
        # ``_separate``는 전달받은 활성 경기영역으로 모든 좌표를 수치 클립한다. 비활성
        # 슬롯은 그 영역 밖의 벤치/교체아웃/퇴장 구역이 곧 올바른 표시 위치이므로 복원한다.
        P = jnp.where(state.active_player[:, None], P, bench)

        # 분리 보정도 위치를 경계까지 클립할 수 있다. 적분 직후의 유령 속도를 위에서 지웠더라도,
        # _separate가 선수를 경계로 밀어낸 뒤 바깥 법선 속도를 그대로 돌려주면 동일 문제가 다시
        # 생긴다. 경계의 **바깥쪽 성분만** 제거해 안쪽 복귀와 접선 이동은 보존한다.
        at_left = P[:, DIM_X] <= -bound_x + GEOMETRY_EPS
        at_right = P[:, DIM_X] >= bound_x - GEOMETRY_EPS
        at_bottom = P[:, DIM_Y] <= -bound_y + GEOMETRY_EPS
        at_top = P[:, DIM_Y] >= bound_y - GEOMETRY_EPS
        vx_out = (at_left & (V[:, DIM_X] < 0.0)) | (at_right & (V[:, DIM_X] > 0.0))
        vy_out = (at_bottom & (V[:, DIM_Y] < 0.0)) | (at_top & (V[:, DIM_Y] > 0.0))
        V = V.at[:, DIM_X].set(jnp.where(vx_out, 0.0, V[:, DIM_X]))
        V = V.at[:, DIM_Y].set(jnp.where(vy_out, 0.0, V[:, DIM_Y]))

        # facing = 갱신된 속도의 방향(파생값). 독립 회전 적분·킥 방향 지향이 모두 사라져
        # facing이 obs로 복원 불가능한 은닉 상태를 들고 있지 않다(facing_from_velocity 참조).
        # 회전율은 마찰 타원의 횡가속 캡(accel_norm_max)이 이미 물리적으로 제한한다.
        facing = self.facing_from_velocity(V, state.attack_dir)

        # 에너지 전이는 별도 순수 함수가 유일한 구현이다. active_player는 on_pitch &
        # ~sent_off projection이므로 퇴장/교체-out 슬롯은 어떤 부하도 내지 않는다. 반대로
        # active 선수는 restart·동결 중에도 작은 생리적 기본 부하를 유지한다. 외부 위치 쓰기와
        # 세트피스 엔진 이동만 locomotion에서 빼 실제 자기 추진 거리/가속을 허위로 청구하지 않는다.
        energy = stamina_step(
            state.stamina_long,
            state.stamina_short,
            V,
            state.player_vel,
            state.active_player,
            locomotion & update_position,
            dt=e_cfg.dt_phys,
            long_drain_base=self.long_stamina_drain_base,
            long_tail_knee=e_cfg.long_stamina_tail_knee,
            long_tail_decay=self.long_stamina_tail_decay,
            sprint_speed=e_cfg.sprint_speed,
            long_sprint_mult=e_cfg.long_stamina_sprint_mult,
            long_idle_load=e_cfg.long_stamina_idle_load,
            long_speed_ref=e_cfg.long_stamina_speed_ref,
            long_speed_load_weight=e_cfg.long_stamina_speed_load,
            long_accel_ref=e_cfg.long_stamina_accel_ref,
            long_accel_load_weight=e_cfg.long_stamina_accel_load,
            vmax=state.vmax,
            endurance_factor=state.endurance_factor,
            long_vmax_floor=e_cfg.long_stamina_vmax_floor,
            short_depletion_s=e_cfg.short_stamina_depletion_s,
            short_depletion_speed_frac=e_cfg.short_stamina_depletion_speed_frac,
            short_speed_exponent=e_cfg.short_stamina_speed_exponent,
            short_accel_ref=e_cfg.short_stamina_accel_ref,
            short_accel_load_weight=e_cfg.short_stamina_accel_load,
            short_recovery_tau_s=e_cfg.short_stamina_recovery_tau_s,
            short_recovery_speed_frac=e_cfg.short_stamina_recovery_speed_frac,
            short_recovery_exponent=e_cfg.short_stamina_recovery_exponent,
            short_long_recovery_penalty=e_cfg.short_stamina_long_recovery_penalty,
        )

        # 이 서브스텝을 지배한 상한은 **진입 stamina**의 것이고, 방금 그 stamina가 내려갔다.
        # 그대로 두면 반환되는 state가 자기 자신의 ``effective_vmax``를 넘는다 —
        # 실측으로 short stamina 0.1 부근(실효최고속 곡선의 무릎)에서 1.9e-3 m/s까지
        # 벌어져 soak의 1e-3 잠금과 이동 계약 테스트의 1e-4 여유를 모두 넘는다.
        # 그런 state는 어떤 이동 액션으로도 표현되지 않으므로(주입 경로도 같은 이유로
        # 막는다) 갱신된 상한으로 다시 자른다. 방향은 보존하고 크기만 줄이며, 줄어드는
        # 폭은 최고속의 0.03 % 미만이라 facing(속도 방향 파생)도 바뀌지 않는다.
        cap_after = self.effective_vmax(
            state.vmax, energy.stamina_long, energy.stamina_short
        )
        speed_after = _safe_norm(V, axis=1, keepdims=True)
        V = V * jnp.minimum(cap_after[:, None] / (speed_after + DIV_EPS), 1.0)

        result = (
            state._replace(
                player_pos=P,
                player_vel=V,
                player_facing=facing,
                stamina_long=energy.stamina_long,
                stamina_short=energy.stamina_short,
            ),
            energy,
        )
        if return_domain_projection:
            return (*result, domain_projection)
        return result

    def nearest_free_position(
        self, desired, occupied, blockers, orientation=1.0
    ):
        """``desired``에서 가장 가까운, 아무와도 겹치지 않는 합법 좌표.

        위치 정합에는 두 종류가 있고 서로 대체할 수 없다. ``reconcile_positions``는 이미
        제자리 근처에 있는 선수들을 **국소적으로** 떼어놓는 완화이고, 이 함수는 임의의 지점으로
        **순간이동**해 들어오는 선수의 자리를 새로 찾는 배치다. 국소 완화로는 후자를 풀 수
        없다 — 두 선수 사이로 들어가면 양쪽에서 대칭으로 밀려 합이 상쇄되고 제자리에 남는다
        (실측: 0.30 m 겹침 유지). 그래서 결정적 후보 집합에서 고른다.

        우선 ``desired`` 자신과 그 둘레의 동심원(반경 ``min_d``의 1~3배) × 8방향에서
        가장 가까운 자리를 찾는다. 이 국소 후보를 전부 막는 합법 밀집 배치도 존재한다
        (11명을 0.461 m 간격의 비대칭 육각 격자로 놓으면 25개 후보가 모두 막힌다).
        그때 ``desired``를 그대로 반환하면 교체 선수가 기존 선수와 겹치므로, 뒤에
        ``N``개의 전역 보증 후보를 붙인다. 보증 후보끼리의 거리는 차단 반경의 두 배보다
        크다. 활성 blocker는 최대 ``N-1``명이므로 한 blocker가 보증 후보 하나씩을 막아도
        적어도 한 자리는 남는다는 비둘기집 원리로 유효 후보가 항상 존재한다. 극단적으로
        작은 사용자 도메인에 이 보증 grid 자체가 다 들어가지 않으면 가능한 만큼만 만들고,
        호출자가 반환점의 합법성을 검사해 fail-closed한다. 이미 비어 있는 local desired까지
        grid 용량 때문에 Python 예외로 거부해서는 안 된다.

        Args:
            desired: float[2] 원하는 좌표
            occupied: float[N,2] 현재 전체 위치
            blockers: bool[N] 비켜 줄 수 없는 슬롯(보통 자신을 뺀 활성 선수)
        Returns:
            float[2] — 허용 선수 경계 안에서 모든 blocker와 최소거리를 지키는 가장 가까운 후보.
        """

        desired, _ = _coerce_public_float32(
            "desired", desired, (DIM_Z,)
        )
        occupied, _ = _coerce_public_float32(
            "occupied", occupied, (self.N, DIM_Z)
        )
        blockers = _coerce_public_bool(
            "blockers", blockers, (self.N,)
        )
        orientation, _ = _coerce_public_float32(
            "orientation", orientation, (), fallback=1.0
        )

        min_d = 2.0 * self.r_player
        # Build every ordered candidate in a caller-supplied canonical frame,
        # then rotate it back by either 0 or 180 degrees.  This is stronger
        # than rotating only the local ring: the centered global grid also has
        # equal-cost ties whose row-major order otherwise selects the same
        # world corner after a team swap.  Internal callers pass the affected
        # slot's attack direction; the default preserves the standalone API.
        orientation = jnp.where(orientation < 0.0, -1.0, 1.0)
        angles = jnp.arange(SUBSTITUTION_ENTRY_DIRECTIONS, dtype=jnp.float32) * (
            2.0 * jnp.pi / SUBSTITUTION_ENTRY_DIRECTIONS
        )
        radii = (jnp.arange(SUBSTITUTION_ENTRY_RINGS, dtype=jnp.float32) + 1.0) * min_d
        ring = jnp.stack([
            (radii[:, None] * jnp.cos(angles)[None, :]).reshape(-1),
            (radii[:, None] * jnp.sin(angles)[None, :]).reshape(-1),
        ], axis=1)
        offsets = jnp.concatenate([jnp.zeros((1, DIM_Z), ring.dtype), ring], axis=0)
        desired_typed = desired.astype(ring.dtype)
        desired_canonical = desired_typed * orientation
        local_candidates = (
            desired_canonical[None, :] + offsets
        ) * orientation

        # 전역 fallback은 단순 탐색 확대가 아니라 **존재 보증**이다. 후보 간격이
        # 2 * clearance보다 크므로 한 blocker가 clearance 안에 둘을 동시에 넣을 수 없다.
        bound_x = self.hx + self.e_cfg.player_boundary_margin
        bound_y = self.hy + self.e_cfg.player_boundary_margin
        clearance = min_d + GEOMETRY_EPS
        guarantee_sep = 2.0 * clearance
        grid_cols_available = max(
            1, int(math.floor((2.0 * bound_x) / guarantee_sep)) + 1
        )
        grid_rows_available = max(
            1, int(math.floor((2.0 * bound_y) / guarantee_sep)) + 1
        )
        guarantee_count = min(
            self.N, grid_cols_available * grid_rows_available
        )
        # N개를 한 줄로 펴면 중앙의 막힌 투입이 수십 m 떨어진 하단 경계로 튈 수 있다.
        # 가능한 한 정사각형에 가까운 grid를 desired 주위에 중심 정렬해, 존재 보증과 최소
        # 개입을 함께 지킨다. 경계 근처에서는 grid 전체를 안쪽으로 평행이동한다.
        grid_cols = min(
            grid_cols_available, int(math.ceil(math.sqrt(guarantee_count)))
        )
        grid_rows_needed = int(math.ceil(guarantee_count / grid_cols))
        grid_index = jnp.arange(guarantee_count, dtype=jnp.int32)
        span_x = (grid_cols - 1) * guarantee_sep
        span_y = (grid_rows_needed - 1) * guarantee_sep
        origin_x = jnp.clip(
            desired_canonical[DIM_X] - 0.5 * span_x,
            -bound_x,
            bound_x - span_x,
        )
        origin_y = jnp.clip(
            desired_canonical[DIM_Y] - 0.5 * span_y,
            -bound_y,
            bound_y - span_y,
        )
        global_candidates = jnp.stack([
            origin_x + (grid_index % grid_cols).astype(ring.dtype) * guarantee_sep,
            origin_y + (grid_index // grid_cols).astype(ring.dtype) * guarantee_sep,
        ], axis=1) * orientation
        candidates = jnp.concatenate([local_candidates, global_candidates], axis=0)
        gaps = jnp.linalg.norm(candidates[:, None, :] - occupied[None, :, :], axis=2)
        clear = jnp.min(
            jnp.where(blockers[None, :], gaps, jnp.inf), axis=1
        ) >= clearance
        inside = (
            (jnp.abs(candidates[:, DIM_X]) <= bound_x)
            & (jnp.abs(candidates[:, DIM_Y]) <= bound_y)
        )
        cost = jnp.where(
            clear & inside,
            jnp.linalg.norm(candidates - desired_typed[None, :], axis=1),
            jnp.inf,
        )
        best = jnp.argmin(cost)
        # 정상 도메인에는 N-candidate 보증으로 finite best가 구조적으로 존재한다.
        # 축소 도메인/외부에서 깨진 blocker는 desired fallback 뒤 호출자의 합법성
        # 검사가 전이를 fail-closed한다.
        return jnp.where(
            jnp.isfinite(cost[best]), candidates[best], desired_typed
        )

    def reconcile_positions(self, pos, active, *, pinned=None, constrained=None,
                            normal=None, resolve=None, rounds=None,
                            tie_direction=None,
                            component_only=True):
        """겹침을 해소해 모든 활성 선수가 최소거리를 지키게 한다 — 위치 정합의 단일 지점.

        위치를 강제로 쓰는 경로는 여럿이다(이동 적분, 재개 이격 투영, 교체 투입, 킥오프
        재배치). 각 경로가 자기 불변식만 세우면 **마지막에 쓴 사람이 이긴다** — 실제로 재개
        투영이 이동의 분리 뒤에 호출돼 0.193 m 겹침을, 교체 투입이 스캔 뒤에 적용돼 좌표
        완전 일치를 관측 상태에 남겼다. 위치를 쓴 뒤 이 함수를 통과시키면 그 부류가 구조적으로
        막힌다.

        일반 Jacobi 분리는 한 쌍을 풀며 다른 쌍을 다시 만들고, 양쪽에서 대칭으로 밀리면 합이
        상쇄돼 제자리에 남는다. 여기서는 각 unordered pair를 슬롯 순서로 즉시 반영하는
        Gauss-Seidel sweep을 쓰고, 두 선수의 **실현 가능한** 방향 기여로 보정량을 재분배한다.

        Args:
            pos: float[N,2] 각자의 제약을 이미 만족하는 목표 위치
            active: bool[N] 온피치 여부 — 비활성은 밀지도 밀리지도 않는다
            pinned: bool[N] 움직이면 안 되는 슬롯(지정 키커, 기존 온피치 선수 등)
            constrained: bool[N] 제약면 **안쪽**으로 밀면 안 되는 슬롯
            normal: float[N,2] 제약면 바깥 법선(단위 또는 0). 0이면 제약 없음
            resolve: ``positions -> (positions, 새 제약자, 법선)``. 매 라운드 뒤 규칙 제약을
                복구한다. None이면 순수 분리만 한다.
            rounds: 반복 라운드 수(기본 ``engine.restart_slide_rounds``)
            tie_direction: float[N,2] 완전중첩 pair의 결정적 방향 basis. 실제 State
                전이는 within-team rank를 공격방향으로 unfold한 공변 basis를 전달한다.
            component_only: True면 ``constrained``와 겹침으로 연결된 성분만 푼다 — 규칙 개입이
                만든 충돌만 심판 개입으로 귀속하고 무관한 라이브 경합 쌍은 건드리지 않는다.
                False면 활성 전원을 대상으로 하는 일반 분리다.
        Returns:
            float[N,2] 정합된 위치
        """

        if type(component_only) is not bool:
            raise TypeError(
                "component_only must be a Python bool, got "
                f"{type(component_only).__name__}"
            )
        if resolve is not None and not callable(resolve):
            raise TypeError("resolve must be callable or None")

        pos, _ = _coerce_public_float32("pos", pos, (self.N, DIM_Z))
        active = _coerce_public_bool("active", active, (self.N,))
        zeros = jnp.zeros(self.N, dtype=bool)
        pinned = (
            zeros if pinned is None
            else _coerce_public_bool("pinned", pinned, (self.N,))
        )
        constrained = (
            zeros if constrained is None
            else _coerce_public_bool("constrained", constrained, (self.N,))
        )
        if normal is None:
            normal = jnp.zeros_like(pos)
        else:
            normal, _ = _coerce_public_float32(
                "normal", normal, (self.N, DIM_Z)
            )
        if tie_direction is not None:
            tie_direction, _ = _coerce_public_float32(
                "tie_direction", tie_direction, (self.N, DIM_Z)
            )
        if rounds is None:
            rounds = self.e_cfg.restart_slide_rounds
        else:
            rounds, _ = _coerce_public_signed_int_scalar(
                "rounds", rounds,
                minimum=0, maximum=MAX_POSITION_SOLVER_ROUNDS,
                fallback=MAX_POSITION_SOLVER_ROUNDS,
            )
        taker_pinned = pinned
        bound_x = self.hx + self.e_cfg.player_boundary_margin
        bound_y = self.hy + self.e_cfg.player_boundary_margin
        adjusted = pos
        affected = constrained if component_only else (active & (~pinned))
        # These values are invariant across relaxation rounds.  Keeping the
        # outer loop as a Python ``range`` unrolled the full restart geometry
        # callback once per round in every compiled step.  With six rounds and
        # the residual-repair branch that produced an 8-copy jaxpr and made a
        # focused restart file take 445 s.  A fori_loop executes the identical
        # ordered operations while tracing the round body exactly once.
        if tie_direction is None:
            angles = self.player_indices.astype(jnp.float32) * GOLDEN_ANGLE
            tie = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
        else:
            tie = tie_direction.astype(pos.dtype)
        rule_component = (resolve is not None) and component_only

        def feasible_one(vec, point, normal_i, constrained_i, pinned_i):
            inward_i = jnp.minimum(jnp.dot(vec, normal_i), 0.0)
            out = vec - jnp.where(
                constrained_i, inward_i * normal_i, 0.0
            )
            x_blocked = (
                ((point[DIM_X] <= -bound_x + GEOMETRY_EPS)
                 & (out[DIM_X] < 0.0))
                | ((point[DIM_X] >= bound_x - GEOMETRY_EPS)
                   & (out[DIM_X] > 0.0))
            )
            y_blocked = (
                ((point[DIM_Y] <= -bound_y + GEOMETRY_EPS)
                 & (out[DIM_Y] < 0.0))
                | ((point[DIM_Y] >= bound_y - GEOMETRY_EPS)
                   & (out[DIM_Y] > 0.0))
            )
            out = out.at[DIM_X].set(jnp.where(x_blocked, 0.0, out[DIM_X]))
            out = out.at[DIM_Y].set(jnp.where(y_blocked, 0.0, out[DIM_Y]))
            return jnp.where(pinned_i, jnp.zeros_like(out), out)

        def relaxation_round(_, carry):
            adjusted, affected, constrained_now, normal_now = carry
            # 규칙 투영이 만든 충돌의 연결 성분만 푼다. 전 roster를 매 호출마다 분리하면
            # 이미 합법인 상태도 다시 움직여 projector가 비멱등이 되고, 라이브 경합의
            # 근사 겹침까지 재개 심판 개입으로 잘못 귀속된다. 현재 affected와 겹친 이웃만
            # 한 홉씩 편입하면 필요한 비침범자는 움직이되 무관한 쌍은 그대로 남는다.
            pair_delta = adjusted[:, None, :] - adjusted[None, :, :]
            pair_dist2 = jnp.sum(pair_delta * pair_delta, axis=2)
            overlap = (
                (pair_dist2 < (2.0 * self.r_player) ** 2)
                & active[:, None]
                & active[None, :]
                & (~jnp.eye(self.N, dtype=bool))
            )
            affected = affected | jnp.any(overlap & affected[None, :], axis=1)
            # 일반 _separate의 Jacobi 합산은 한 쌍을 풀면서 다른 쌍을 다시 만들 수 있다.
            # 각 unordered pair를 슬롯 순서로 즉시 반영하는 Gauss-Seidel sweep을 쓴다.
            # 두 선수의 **실현 가능한** 방향 기여를 먼저 구하고 실제 상대 거리가 이번
            # sweep에서 정확히 min_d가 되도록 보정량을 재분배한다. 한쪽이 제약면/5m
            # 경계/키커 고정으로 못 움직이면 다른 쪽이 전량을 받는다.
            # A rule projection can push one legal blocker into the next one.
            # The connected component is therefore not static: computing it
            # once before the sweep grows it by only one hop per outer round
            # and a long, initially legal chain can outlive the configured
            # number of rounds.  Carry membership through the Gauss-Seidel
            # sweep so a newly contacted neighbour joins the causal component
            # immediately.  When this is a rule-aware reconciliation, every
            # joined blocker also keeps the supplied outward normal; otherwise
            # solving one pair may push it back through the exclusion surface.
            def separate_pair(pair_index, carry):
                points, connected = carry
                i = pair_index // self.N
                j = pair_index % self.N
                pi, pj = points[i], points[j]
                diff_ij = pi - pj
                dist_ij = jnp.sqrt(jnp.dot(diff_ij, diff_ij) + SQUARED_EPS)
                tie_ij = tie[i] - tie[j]
                raw = jnp.where(
                    dist_ij < COINCIDENT_DISTANCE_EPS, tie_ij, diff_ij
                )
                direction = raw / (jnp.linalg.norm(raw) + DIV_EPS)
                penetration = jnp.maximum(2.0 * self.r_player - dist_ij, 0.0)
                pair_on = (
                    (i < j)
                    & active[i] & active[j]
                    & (connected[i] | connected[j])
                    & (penetration > 0.0)
                )
                fi = feasible_one(
                    direction, pi, normal_now[i],
                    constrained_now[i] | (rule_component & connected[i]),
                    taker_pinned[i]
                )
                fj = feasible_one(
                    -direction, pj, normal_now[j],
                    constrained_now[j] | (rule_component & connected[j]),
                    taker_pinned[j]
                )
                gain_i = jnp.maximum(jnp.dot(fi, direction), 0.0)
                gain_j = jnp.maximum(jnp.dot(fj, -direction), 0.0)
                gain = gain_i + gain_j
                scale = jnp.where(
                    pair_on & (gain > DIV_EPS),
                    penetration * self.e_cfg.restart_separation_relaxation / gain,
                    0.0,
                )
                next_i = pi + scale * fi
                next_j = pj + scale * fj
                next_i = jnp.clip(next_i, jnp.array([-bound_x, -bound_y]),
                                  jnp.array([bound_x, bound_y]))
                next_j = jnp.clip(next_j, jnp.array([-bound_x, -bound_y]),
                                  jnp.array([bound_x, bound_y]))
                points = points.at[i].set(next_i).at[j].set(next_j)
                connected = connected.at[i].set(connected[i] | pair_on)
                connected = connected.at[j].set(connected[j] | pair_on)
                return points, connected

            adjusted, affected = jax.lax.fori_loop(
                0, self.N * self.N, separate_pair, (adjusted, affected)
            )
            adjusted = jnp.stack([
                jnp.clip(adjusted[:, DIM_X], -bound_x, bound_x),
                jnp.clip(adjusted[:, DIM_Y], -bound_y, bound_y),
            ], axis=1)
            # 매 라운드 뒤 합법성을 복구한다. 분리 때문에 새로 침범한 충돌 상대도 이후
            # 라운드에서는 같은 제약을 받으며, 그 슬롯의 강제 이동도 최종 mask에 포함된다.
            if resolve is not None:
                adjusted, newly_encroaching, normal_now = resolve(adjusted)
                constrained_now = constrained_now | newly_encroaching
                affected = affected | newly_encroaching
            return adjusted, affected, constrained_now, normal_now

        adjusted, affected, constrained, normal = jax.lax.fori_loop(
            0,
            rounds,
            relaxation_round,
            (adjusted, affected, constrained, normal),
        )
        return adjusted

    def repair_causal_overlaps(
        self, pos, active, causal, pinned, rule_constrained, resolve,
        orientation=None,
    ):
        """Fail-safe for a projection-induced collision wave.

        The relaxed local solver is intentionally bounded for runtime cost.
        A legal chain can nevertheless be arranged so that a projected player
        contacts more neighbours than those fixed sweeps can traverse (slot
        order must not decide the outcome).  This fallback runs only when such
        a residual pair actually exists.  It relocates one movable member of
        the causal component to the nearest globally collision-free candidate,
        then reapplies the restart geometry.  At most ``N`` placements are
        needed because every accepted candidate is clear of all current active
        players; the loop remains compact under JIT.
        """

        if not callable(resolve):
            raise TypeError("resolve must be callable")
        pos, _ = _coerce_public_float32("pos", pos, (self.N, DIM_Z))
        active = _coerce_public_bool("active", active, (self.N,))
        causal = _coerce_public_bool("causal", causal, (self.N,))
        pinned = _coerce_public_bool("pinned", pinned, (self.N,))
        rule_constrained = _coerce_public_bool(
            "rule_constrained", rule_constrained, (self.N,)
        )
        if orientation is None:
            orientation = jnp.ones(self.N, dtype=pos.dtype)
        else:
            orientation, _ = _coerce_public_float32(
                "orientation", orientation, (self.N,), fallback=1.0
            )

        min_d2 = (2.0 * self.r_player) ** 2
        eye = jnp.eye(self.N, dtype=bool)

        def one(_, carry):
            points, connected = carry
            delta = points[:, None, :] - points[None, :, :]
            overlap = (
                (jnp.sum(delta * delta, axis=2) < min_d2)
                & active[:, None] & active[None, :] & (~eye)
            )
            connected = connected | jnp.any(
                overlap & connected[None, :], axis=1
            )
            involved = (
                connected & active & (~pinned) & jnp.any(overlap, axis=1)
            )
            # Keep the already legal blocker when possible: moving it to a
            # nearby free point is less likely to fight the rule projector
            # than relocating the original encroacher off its legal surface.
            legal_blocker = involved & (~rule_constrained)
            candidates = jnp.where(jnp.any(legal_blocker), legal_blocker, involved)
            has_candidate = jnp.any(candidates)
            index = jnp.argmax(candidates.astype(jnp.int32))
            blockers = active.at[index].set(False)
            free = self.nearest_free_position(
                points[index], points, blockers,
                orientation=orientation[index],
            )
            placed = points.at[index].set(
                jnp.where(has_candidate, free, points[index])
            )
            connected = connected | (
                (self.player_indices == index) & has_candidate
            )
            return placed, connected

        def placement_pass(points, connected):
            return jax.lax.fori_loop(
                0, self.N, one, (points, connected)
            )

        # Do not nest the full restart projector inside the N-step placement
        # loop.  Although correct, that traces an enormous jaxpr at every env
        # reset (measured restart test time 5 s -> 447 s).  Hoist one final
        # projection out of the compact placement pass.  The placement itself
        # clears every current overlap against all active blockers; restoring
        # the same rule surface is then sufficient and avoids duplicating an
        # N-placement loop plus a second full geometry callback in every JIT.
        repaired, connected = placement_pass(pos, causal)
        repaired, _, _ = resolve(repaired)
        return repaired

    def _separate(
        self,
        P,
        V,
        active=None,
        pinned=None,
        tie_direction=None,
        boundary_margin=None,
    ):
        """선수 겹침 해소: 최소거리(2×r_player) 미만인 쌍을 절반씩 밀어냄. sep_iters회 반복.
        완전히 겹친(거리≈0) 쌍은 방향이 정의되지 않으므로 황금각 분산 방향으로 밀어 데드락 방지.
        퇴장 선수(active=False)는 밀지도 밀리지도 않음(벤치 유령화 방지).

        Args:
            P: float[N,2] 위치
            V: float[N,2] 속도 — 분리는 위치만 조정, V는 그대로 통과
            active: bool[N] 온피치 여부(None이면 전원 참여)
            pinned: bool[N] 위치 고정 선수(None이면 없음). 고정 선수는 움직이지 않고, 겹친 상대가
                    전체 보정량을 받는다. 둘 다 고정인 쌍만 분리하지 않는다.
            tie_direction: float[N,2] 완전중첩 pair의 결정적 방향 basis.
            boundary_margin: 수치 클립에 쓸 피치 바깥 여유. None이면 재개/공개 API의
                    기존 ``player_boundary_margin``을 쓴다.
        Return:
            (P, V)
        """
        N = P.shape[0]
        min_d = 2.0 * self.r_player
        eye = jnp.eye(N, dtype=bool)
        if tie_direction is None:
            ang = jnp.arange(N) * GOLDEN_ANGLE
            tie = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=1)
        else:
            tie = jnp.asarray(tie_direction, dtype=P.dtype)
            if tie.shape != P.shape:
                raise ValueError(
                    "tie_direction must match P shape, got "
                    f"{tie.shape} and {P.shape}"
                )
        active_mask = jnp.ones(N, dtype=bool) if active is None else active
        pinned_mask = jnp.zeros(N, dtype=bool) if pinned is None else pinned
        movable = active_mask & (~pinned_mask)
        pair_on = (
            active_mask[:, None]
            & active_mask[None, :]
            & (~(pinned_mask[:, None] & pinned_mask[None, :]))
        )
        # 일반 경로의 값싼 고정 share. 경계에 닿은 선수가 있을 때만 아래
        # ``boundary_aware_correction``이 실현 가능량을 다시 배분한다.
        receiver_share = (
            jnp.where(pinned_mask[None, :], 1.0, 0.5) * movable[:, None]
        )
        if boundary_margin is None:
            boundary_margin = self.e_cfg.player_boundary_margin
        bound_x = self.hx + boundary_margin
        bound_y = self.hy + boundary_margin
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
            # Boundary-aware pair shares.  A fixed 1/2+1/2 split silently loses the
            # outward half when one player is already on the 5 m boundary and the
            # final position is clipped.  At a corner that left a physically reachable
            # pair 2.6 cm inside each other after the configured two iterations (and an
            # exactly coincident pair 11.5 cm deep).  Remove each receiver's blocked
            # normal component first, measure how much separation that feasible vector
            # actually provides, then give the missing share to its partner.  This is
            # still the same vectorised Jacobi hot path; no sequential N^2 sweep is
            # introduced.
            def regular_correction(_):
                return jnp.sum(
                    overlap[:, :, None] * direction * receiver_share[:, :, None],
                    axis=1,
                )

            def boundary_aware_correction(_):
                x_blocked = (
                    ((P[:, None, DIM_X] <= -bound_x + GEOMETRY_EPS)
                     & (direction[:, :, DIM_X] < 0.0))
                    | ((P[:, None, DIM_X] >= bound_x - GEOMETRY_EPS)
                       & (direction[:, :, DIM_X] > 0.0))
                )
                y_blocked = (
                    ((P[:, None, DIM_Y] <= -bound_y + GEOMETRY_EPS)
                     & (direction[:, :, DIM_Y] < 0.0))
                    | ((P[:, None, DIM_Y] >= bound_y - GEOMETRY_EPS)
                       & (direction[:, :, DIM_Y] > 0.0))
                )
                feasible = direction.at[:, :, DIM_X].set(
                    jnp.where(x_blocked, 0.0, direction[:, :, DIM_X])
                )
                feasible = feasible.at[:, :, DIM_Y].set(
                    jnp.where(y_blocked, 0.0, feasible[:, :, DIM_Y])
                )
                feasible = feasible * movable[:, None, None]
                gain = jnp.maximum(jnp.sum(feasible * direction, axis=2), 0.0)
                pair_gain = gain + gain.T
                scale = jnp.where(
                    pair_gain > DIV_EPS,
                    overlap / jnp.maximum(pair_gain, DIV_EPS),
                    0.0,
                )
                return jnp.sum(scale[:, :, None] * feasible, axis=1)

            on_boundary = movable & (
                (P[:, DIM_X] <= -bound_x + GEOMETRY_EPS)
                | (P[:, DIM_X] >= bound_x - GEOMETRY_EPS)
                | (P[:, DIM_Y] <= -bound_y + GEOMETRY_EPS)
                | (P[:, DIM_Y] >= bound_y - GEOMETRY_EPS)
            )
            correction = jax.lax.cond(
                jnp.any(on_boundary),
                boundary_aware_correction,
                regular_correction,
                operand=None,
            )
            P = P + correction
            P = P.at[:, 0].set(jnp.clip(P[:, 0], -bound_x, bound_x))
            P = P.at[:, 1].set(jnp.clip(P[:, 1], -bound_y, bound_y))
        return P, V
