"""match clock, halftime, reward와 absorbing termination wrapper."""

import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_NONE,
    RK_PENALTY,
)
from messi.core.contact import MECHANISM_NONE, empty_contact_result
from messi.core.geometry import BallGeometry
from messi.core.restart import restart_active
from messi.core.roster import validate_player_capacity
from messi.core.stamina import ShortStamina, recover_short_stamina_at_rest
from messi.core.state import initial_player_body_forward
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase
from messi.core.transition import (
    ControlFrameResult,
    _empty_control_frame_receipt,
    step_control,
    step_control_with_events,
)
from messi.environment.clock import regulation_elapsed_ticks
from messi.environment.observation import SharedScene, observe_all


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class MatchConfig:
    halftime_seconds: float = 45.0 * 60.0
    fulltime_seconds: float = 90.0 * 60.0
    halftime_enabled: bool = True
    minimum_players_per_team: int = 7
    maximum_added_time_seconds_per_period: float = 10.0 * 60.0
    halftime_interval_s: float = 15.0 * 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.halftime_enabled, bool):
            raise TypeError("halftime_enabled는 bool이어야 한다")
        for name in (
            "fulltime_seconds",
            "maximum_added_time_seconds_per_period",
            "halftime_interval_s",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise TypeError(f"{name}은 유한한 실수여야 한다")
        if self.fulltime_seconds <= 0.0:
            raise ValueError("fulltime_seconds는 양수여야 한다")
        if self.maximum_added_time_seconds_per_period < 0.0:
            raise ValueError("maximum_added_time_seconds_per_period는 0 이상이어야 한다")
        if not 0.0 <= self.halftime_interval_s <= 15.0 * 60.0:
            raise ValueError("halftime_interval_s는 [0, 900] 범위여야 한다")
        if self.halftime_enabled:
            value = self.halftime_seconds
            if (
                not isinstance(value, numbers.Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise TypeError("halftime_seconds는 유한한 실수여야 한다")
            if value <= 0.0 or value >= self.fulltime_seconds:
                raise ValueError("halftime_seconds는 0보다 크고 fulltime보다 작아야 한다")
        else:
            object.__setattr__(self, "halftime_seconds", float(self.fulltime_seconds))
        if (
            not isinstance(self.minimum_players_per_team, numbers.Integral)
            or isinstance(self.minimum_players_per_team, bool)
        ):
            raise TypeError("minimum_players_per_team은 정수여야 한다")
        if not 0 <= self.minimum_players_per_team <= 11:
            raise ValueError("minimum_players_per_team은 0..11이어야 한다")

    def ticks(self, timebase: Timebase = DEFAULT_TIMEBASE) -> tuple[int, int]:
        fulltime = timebase.control_steps_for(self.fulltime_seconds, minimum=1)
        halftime = (
            timebase.control_steps_for(self.halftime_seconds, minimum=1)
            if self.halftime_enabled
            else fulltime
        )
        if fulltime > MAX_REGULATION_CONTROL_TICKS:
            raise ValueError("fulltime이 지원하는 model-clock 상한을 넘는다")
        if self.halftime_enabled and halftime >= fulltime:
            raise ValueError("halftime은 fulltime보다 앞서야 한다")
        return halftime, fulltime

    def maximum_added_time_ticks(
        self, timebase: Timebase = DEFAULT_TIMEBASE
    ) -> int:
        ticks = timebase.control_steps_for(
            self.maximum_added_time_seconds_per_period
        )
        if ticks > MAX_REGULATION_CONTROL_TICKS:
            raise ValueError("added time이 지원하는 model-clock 상한을 넘는다")
        return ticks


DEFAULT_MATCH_CONFIG = MatchConfig()
_DEFAULT_BALL_GEOMETRY = BallGeometry()

# float32 model-clock round trip과 int32 wall-clock headroom을 보장한다.
MAX_REGULATION_CONTROL_TICKS = 1 << 20
MAX_WALL_CONTROL_TICKS = (1 << 22) - 1

MATCH_END_NONE = 0
MATCH_END_FULL_TIME = 1
MATCH_END_MINIMUM_PLAYERS = 2
MATCH_END_RESTART_TAKER_UNAVAILABLE = 3
MATCH_END_WALL_CLOCK_LIMIT = 4


class MatchSetup(NamedTuple):
    second_half_positions: jax.Array
    second_half_distance_squared: jax.Array
    second_half_kickoff_team: jax.Array


class MatchTransition(NamedTuple):
    state: object
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    halftime_reset: jax.Array
    termination_reason: jax.Array


class MatchStatus(NamedTuple):
    """현재 authoritative state의 종료 상태."""

    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    reason: jax.Array


class EnvironmentStep(NamedTuple):
    transition: MatchTransition
    observation: SharedScene


def make_match_setup(initial_state) -> MatchSetup:
    second_half_positions = initial_state.players.position.at[:, 0].multiply(-1.0)
    return MatchSetup(
        second_half_positions=second_half_positions,
        second_half_distance_squared=jnp.sum(
            second_half_positions * second_half_positions, axis=-1
        ),
        second_half_kickoff_team=(jnp.int32(1) - initial_state.kickoff_team).astype(
            jnp.int32
        ),
    )


def _kickoff_taker_candidates(active, roster, kickoff_team) -> jax.Array:
    """Active outfield teammate를 우선하고 없으면 goalkeeper까지 허용한다."""

    teammate = active & (roster.team_id == kickoff_team)
    outfield = teammate & (~roster.is_goalkeeper)
    return jnp.where(jnp.any(outfield), outfield, teammate)


def _validate_match_setup(state, setup: MatchSetup) -> None:
    """MatchSetup의 static player축 contract를 physics 호출 전에 검증한다."""

    if not isinstance(setup, MatchSetup):
        raise TypeError("setup은 MatchSetup이어야 한다")
    player_count = state.players.position.shape[0]
    if setup.second_half_positions.shape != (player_count, 2):
        raise ValueError("setup.second_half_positions shape는 [player, 2]여야 한다")
    if setup.second_half_distance_squared.shape != (player_count,):
        raise ValueError(
            "setup.second_half_distance_squared shape는 player축과 같아야 한다"
        )
    if setup.second_half_kickoff_team.shape != ():
        raise ValueError("setup.second_half_kickoff_team은 scalar여야 한다")


def _validate_match_capacity(state, config: MatchConfig) -> None:
    """정적 roster capacity와 경기 최소 선수 설정의 호환성을 검증한다."""

    players_per_team = validate_player_capacity(state.players.position.shape[0])
    if config.minimum_players_per_team > players_per_team:
        raise ValueError(
            "minimum_players_per_team은 static players_per_team 이하여야 한다"
        )


def match_status(
    state,
    roster,
    *,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
) -> MatchStatus:
    """새 physics step 없이 현재 state의 종료 조건을 계산한다."""

    _validate_match_capacity(state, config)
    halftime_tick, fulltime_tick = config.ticks(timebase)
    maximum_added_ticks = config.maximum_added_time_ticks(timebase)
    counts = jnp.stack(
        [
            jnp.sum(state.players.active & (roster.team_id == team))
            for team in range(2)
        ]
    )
    below_minimum = jnp.any(counts < config.minimum_players_per_team)
    valid_restart_team = (state.restart.team == 0) | (state.restart.team == 1)
    safe_restart_team = jnp.clip(state.restart.team, 0, 1)
    ordinary_restart = restart_active(state.restart.kind) & (
        state.restart.kind != RK_GK_HOLD
    )
    restart_without_candidate = ordinary_restart & (
        (~valid_restart_team) | (counts[safe_restart_team] == 0)
    )
    terminated = below_minimum | restart_without_candidate
    penalty_incomplete = (state.restart.kind == RK_PENALTY) | (
        state.penalty_completion_active
    )
    second_half_wall = state.control_tick - jnp.maximum(
        state.first_half_wall_end_tick, jnp.int32(0)
    )
    second_half_wall_limit = jnp.int32(
        fulltime_tick - halftime_tick + maximum_added_ticks
    )
    if config.halftime_enabled:
        added_time_limit_reached = (
            (state.first_half_wall_end_tick >= 0)
            & (second_half_wall >= second_half_wall_limit)
        )
    else:
        added_time_limit_reached = state.control_tick >= jnp.int32(
            fulltime_tick + maximum_added_ticks
        )
    wall_clock_exhausted = state.control_tick >= jnp.int32(MAX_WALL_CONTROL_TICKS)
    # Regulation 경계에서는 진행 중인 penalty의 causal completion을 기다리지만,
    # configured wall-time 상한은 penalty 상태와 무관한 절대 종료 경계다.
    truncated = (
        (regulation_elapsed_ticks(state) >= fulltime_tick)
        & (~penalty_incomplete)
    ) | added_time_limit_reached
    truncated = truncated | wall_clock_exhausted
    reason = jnp.where(
        below_minimum,
        jnp.int32(MATCH_END_MINIMUM_PLAYERS),
        jnp.where(
            restart_without_candidate,
            jnp.int32(MATCH_END_RESTART_TAKER_UNAVAILABLE),
            jnp.where(
                wall_clock_exhausted,
                jnp.int32(MATCH_END_WALL_CLOCK_LIMIT),
                jnp.where(
                    truncated,
                    jnp.int32(MATCH_END_FULL_TIME),
                    jnp.int32(MATCH_END_NONE),
                ),
            ),
        ),
    )
    return MatchStatus(terminated, truncated, terminated | truncated, reason)


def _halftime_state(
    state,
    roster,
    setup: MatchSetup,
    halftime: jax.Array,
    *,
    halftime_tick: jax.Array,
    halftime_interval_s: float,
    short_stamina: ShortStamina,
    ball_geometry: BallGeometry,
):
    select = lambda new, old: jnp.where(halftime, new, old)
    active = state.players.active
    reset_player = halftime & active
    second_half_attack_direction = -state.attack_direction
    reset_body_forward = initial_player_body_forward(
        roster.team_id, second_half_attack_direction
    )
    players = state.players._replace(
        position=jnp.where(
            reset_player[:, None],
            setup.second_half_positions,
            state.players.position,
        ),
        velocity=jnp.where(
            reset_player[:, None],
            jnp.zeros_like(state.players.velocity),
            state.players.velocity,
        ),
        body_forward=jnp.where(
            reset_player[:, None], reset_body_forward, state.players.body_forward
        ),
        stamina_short=jnp.where(
            reset_player,
            recover_short_stamina_at_rest(
                state.players.stamina_short,
                halftime_interval_s,
                short=short_stamina,
            ),
            state.players.stamina_short,
        ),
        challenge_recovery_substeps=jnp.where(
            halftime,
            jnp.zeros_like(state.players.challenge_recovery_substeps),
            state.players.challenge_recovery_substeps,
        ),
        contact_lock_substeps=jnp.where(
            halftime,
            jnp.zeros_like(state.players.contact_lock_substeps),
            state.players.contact_lock_substeps,
        ),
        aerial_recovery_substeps=jnp.where(
            halftime,
            jnp.zeros_like(state.players.aerial_recovery_substeps),
            state.players.aerial_recovery_substeps,
        ),
        possession_loss_lock_substeps=jnp.where(
            halftime,
            jnp.zeros_like(state.players.possession_loss_lock_substeps),
            state.players.possession_loss_lock_substeps,
        ),
    )
    ball = state.ball._replace(
        position=jnp.where(
            halftime,
            jnp.asarray(
                (0.0, 0.0, ball_geometry.radius_m), state.ball.position.dtype
            ),
            state.ball.position,
        ),
        velocity=jnp.where(
            halftime, jnp.zeros_like(state.ball.velocity), state.ball.velocity
        ),
        spin=jnp.where(halftime, jnp.zeros_like(state.ball.spin), state.ball.spin),
        live=select(jnp.bool_(False), state.ball.live),
    )
    kickoff_team = setup.second_half_kickoff_team.astype(jnp.int32)
    eligible_taker = _kickoff_taker_candidates(players.active, roster, kickoff_team)
    selected_taker = jnp.where(
        jnp.any(eligible_taker),
        jnp.argmin(
            jnp.where(
                eligible_taker,
                setup.second_half_distance_squared,
                jnp.inf,
            )
        ),
        jnp.int32(NO_PLAYER),
    ).astype(jnp.int32)
    restart = state.restart._replace(
        kind=select(jnp.int32(RK_KICKOFF), state.restart.kind),
        team=select(kickoff_team, state.restart.team),
        substeps_remaining=select(jnp.int32(0), state.restart.substeps_remaining),
        taker=select(selected_taker, state.restart.taker),
        indirect=select(jnp.bool_(False), state.restart.indirect),
        opened_control_tick=select(
            state.control_tick, state.restart.opened_control_tick
        ),
    )
    possession = state.possession._replace(
        team=select(jnp.int32(NO_TEAM), state.possession.team),
        player=select(jnp.int32(NO_PLAYER), state.possession.player),
        control_ticks=select(jnp.int32(0), state.possession.control_ticks),
        episode_start_tick=select(
            jnp.int32(-1), state.possession.episode_start_tick
        ),
        previous_team=select(jnp.int32(NO_TEAM), state.possession.previous_team),
        last_contact=jax.tree.map(
            lambda empty, old: select(empty, old),
            empty_contact_result(),
            state.possession.last_contact,
        ),
    )
    cleared_release = state.restart_release._replace(
        active=select(jnp.bool_(False), state.restart_release.active),
        untouched=select(jnp.bool_(False), state.restart_release.untouched),
        kind=select(jnp.int32(RK_NONE), state.restart_release.kind),
        team=select(jnp.int32(NO_TEAM), state.restart_release.team),
        taker=select(jnp.int32(NO_PLAYER), state.restart_release.taker),
        indirect=select(jnp.bool_(False), state.restart_release.indirect),
        law11_direct_exempt=select(
            jnp.bool_(False), state.restart_release.law11_direct_exempt
        ),
        release_mechanism=select(
            jnp.int32(MECHANISM_NONE), state.restart_release.release_mechanism
        ),
    )
    return state._replace(
        players=players,
        ball=ball,
        restart=restart,
        possession=possession,
        attack_direction=jnp.where(
            halftime, second_half_attack_direction, state.attack_direction
        ),
        kickoff_team=select(kickoff_team, state.kickoff_team),
        restart_release=cleared_release,
        gk_backpass_team=select(jnp.int32(NO_TEAM), state.gk_backpass_team),
        first_half_wall_end_tick=select(
            state.control_tick, state.first_half_wall_end_tick
        ),
        first_half_live_extension_ticks=select(
            regulation_elapsed_ticks(state) - halftime_tick,
            state.first_half_live_extension_ticks,
        ),
        offside_flagged=jnp.where(
            halftime, jnp.zeros_like(state.offside_flagged), state.offside_flagged
        ),
        offside_direct_exempt_team=select(
            jnp.int32(NO_TEAM), state.offside_direct_exempt_team
        ),
        restart_layout_ready=select(jnp.bool_(False), state.restart_layout_ready),
        restart_taker_approach=select(
            jnp.bool_(False), state.restart_taker_approach
        ),
        penalty_completion_active=select(
            jnp.bool_(False), state.penalty_completion_active
        ),
        penalty_completion_team=select(
            jnp.int32(NO_TEAM), state.penalty_completion_team
        ),
    )


def _finish_match_transition(
    state,
    roster,
    stepped,
    setup: MatchSetup,
    *,
    already_done=False,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    short_stamina: ShortStamina = ShortStamina(),
    ball_geometry: BallGeometry = _DEFAULT_BALL_GEOMETRY,
) -> MatchTransition:
    """Control physics 결과에 match clock과 종료 규칙을 적용한다."""
    halftime_tick, _ = config.ticks(timebase)
    maximum_added_ticks = config.maximum_added_time_ticks(timebase)
    dead_ball_increment = restart_active(state.restart.kind) & (
        state.restart.kind != RK_GK_HOLD
    )
    stepped = stepped._replace(
        dead_ball_control_ticks=(
            state.dead_ball_control_ticks + dead_ball_increment.astype(jnp.int32)
        ).astype(jnp.int32)
    )
    penalty_incomplete = (stepped.restart.kind == RK_PENALTY) | (
        stepped.penalty_completion_active
    )
    first_half_wall_limit = jnp.int32(halftime_tick + maximum_added_ticks)
    crossed_half = (
        config.halftime_enabled
        & (state.first_half_wall_end_tick < 0)
        & (
            (
                (regulation_elapsed_ticks(stepped) >= halftime_tick)
                & (~penalty_incomplete)
            )
            | (stepped.control_tick >= first_half_wall_limit)
        )
    )
    stepped = _halftime_state(
        stepped,
        roster,
        setup,
        crossed_half,
        halftime_tick=jnp.int32(halftime_tick),
        halftime_interval_s=config.halftime_interval_s,
        short_stamina=short_stamina,
        ball_geometry=ball_geometry,
    )
    status = match_status(stepped, roster, config=config, timebase=timebase)
    already_done = jnp.asarray(already_done)
    terminated = status.terminated & (~already_done)
    truncated = status.truncated & (~already_done)
    done = already_done | terminated | truncated
    frozen = jax.tree.map(
        lambda new, old: jnp.where(already_done, old, new), stepped, state
    )
    score_delta = stepped.score - state.score
    reward = jnp.asarray(
        (2 * score_delta[0] - score_delta[1],
         2 * score_delta[1] - score_delta[0]),
        dtype=jnp.float32,
    )
    reward = jnp.where(already_done, jnp.zeros_like(reward), reward)
    return MatchTransition(
        frozen,
        reward,
        terminated,
        truncated,
        done,
        crossed_half & (~already_done),
        jnp.where(already_done, jnp.int32(MATCH_END_NONE), status.reason),
    )


def _regulation_elapsed_fraction(state, *, config: MatchConfig, timebase: Timebase):
    """Dead ball을 뺀 90분 진행도. Card 확률의 시간 조건으로 쓴다.

    이 값을 넘기지 않으면 기본값 0.0이 되어 card 확률이 경기 내내 상수가 된다.
    """

    reference = max(1, round(float(config.fulltime_seconds) * timebase.control_fps))
    return jnp.clip(
        regulation_elapsed_ticks(state).astype(jnp.float32) / jnp.float32(reference),
        0.0,
        1.0,
    )


def transition_match(
    state,
    roster,
    action,
    key,
    setup: MatchSetup,
    *,
    already_done=False,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    **kwargs,
) -> MatchTransition:
    """observation을 만들지 않는 lean match-aware transition."""
    _validate_match_setup(state, setup)
    _validate_match_capacity(state, config)
    short_stamina = kwargs.get("short_stamina", ShortStamina())
    ball_geometry = kwargs.get("ball_geometry", _DEFAULT_BALL_GEOMETRY)
    stepped = step_control(
        state,
        roster,
        action,
        key,
        timebase=timebase,
        regulation_elapsed_fraction=_regulation_elapsed_fraction(
            state, config=config, timebase=timebase
        ),
        **kwargs,
    )
    return _finish_match_transition(
        state,
        roster,
        stepped,
        setup,
        already_done=already_done,
        config=config,
        timebase=timebase,
        short_stamina=short_stamina,
        ball_geometry=ball_geometry,
    )


def transition_match_with_events(
    state,
    roster,
    action,
    key,
    setup: MatchSetup,
    *,
    already_done=False,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    **kwargs,
) -> tuple[MatchTransition, ControlFrameResult]:
    """Event Capture용 match transition과 compact receipt를 함께 반환한다."""
    _validate_match_setup(state, setup)
    _validate_match_capacity(state, config)
    short_stamina = kwargs.get("short_stamina", ShortStamina())
    ball_geometry = kwargs.get("ball_geometry", _DEFAULT_BALL_GEOMETRY)
    control = step_control_with_events(
        state,
        roster,
        action,
        key,
        timebase=timebase,
        regulation_elapsed_fraction=_regulation_elapsed_fraction(
            state, config=config, timebase=timebase
        ),
        **kwargs
    )
    transition = _finish_match_transition(
        state,
        roster,
        control.state,
        setup,
        already_done=already_done,
        config=config,
        timebase=timebase,
        short_stamina=short_stamina,
        ball_geometry=ball_geometry,
    )
    empty_receipt = _empty_control_frame_receipt(
        state.ball.position.dtype, state.players.position.shape[0]
    )
    receipt = jax.tree.map(
        lambda current, empty: jnp.where(already_done, empty, current),
        control.receipt,
        empty_receipt,
    )
    return transition, ControlFrameResult(transition.state, receipt)


def step_environment(
    state,
    roster,
    action,
    key,
    setup: MatchSetup,
    *,
    already_done=False,
    config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    **kwargs,
) -> EnvironmentStep:
    """Match transition과 다음 state의 shared observation을 반환한다."""

    transition = transition_match(
        state,
        roster,
        action,
        key,
        setup,
        already_done=already_done,
        config=config,
        timebase=timebase,
        **kwargs,
    )
    return EnvironmentStep(
        transition,
        observe_all(transition.state, roster, timebase=timebase),
    )
