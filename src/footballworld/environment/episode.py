"""Minimal match clock and absorbing rollout boundary for FootballWorld."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.body_foul import BodyFoul
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.config.perception import Perception
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.reach import Reach
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.action import IntentAction
from footballworld.core.constants import (
    GEOMETRY_EPS,
    IFAB_MAX_TEAM_PLAYERS,
    IFAB_MIN_TEAM_PLAYERS,
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_NONE,
    RK_PENALTY,
    STATIONARY_SPEED_EPS,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_NONE,
    LAW11_NONE,
    MECHANISM_NONE,
    OUTCOME_NONE,
    ContactResult,
)
from footballworld.core.state import (
    BallState,
    PossessionState,
    RestartReleaseProvenance,
    RestartState,
    State,
    body_forward_from_angle,
    initial_player_body_forward,
)
from footballworld.core.timebase import (
    DEFAULT_TIMEBASE,
    Timebase,
)
from footballworld.dynamics.action import (
    ACTION_FLAG_ENVIRONMENT_OVERWRITE,
    ACTION_FLAG_REFEREE_PROJECTION,
    ACTION_REASON_ENVIRONMENT_OVERWRITE,
    ACTION_REASON_REFEREE_PROJECTION,
    DISPLACEMENT_HALFTIME_RESET,
    DISPLACEMENT_REFEREE_PROJECTION,
    trace_action,
    trace_action_receipt,
)
from footballworld.dynamics.contest import ContestOverride
from footballworld.dynamics.stamina import recover_short_stamina_at_rest
from footballworld.environment.clock import regulation_elapsed_ticks
from footballworld.environment.transition import (
    ControlFrame,
    ControlFrameWithEvents,
    _ControlFrameWithEventsAndRenderSamples,
    _empty_frame_events,
    step_control_frame,
)
from footballworld.rules.offside import OffsideState, clear_offside_state
from footballworld.rules.restart import (
    repair_broken_restart_taker,
    select_restart_taker,
)
from footballworld.rules.restart_positioning import prepare_restart_positioning
from footballworld.rules.restart_timing import continuous_restart_approach_enabled

# Exact float32 counter round-trips are guaranteed below this wall-clock bound.
# Regulation is capped lower to reserve deterministic added-time headroom.
MAX_REGULATION_CONTROL_TICKS = 1 << 20
MAX_WALL_CONTROL_TICKS = (1 << 22) - 1


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Static regulation and abandonment settings for one rollout."""

    halftime_seconds: float = 45.0 * 60.0
    fulltime_seconds: float = 90.0 * 60.0
    halftime_enabled: bool = True
    halftime_interval_s: float = 15.0 * 60.0
    minimum_team_players: tuple[int, int] = (
        IFAB_MIN_TEAM_PLAYERS,
        IFAB_MIN_TEAM_PLAYERS,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.halftime_enabled, bool):
            raise TypeError("halftime_enabled must be a bool")
        if self.halftime_enabled and (
            not isinstance(self.halftime_seconds, numbers.Real)
            or isinstance(self.halftime_seconds, bool)
            or not math.isfinite(self.halftime_seconds)
            or self.halftime_seconds <= 0.0
        ):
            raise ValueError(
                "halftime_seconds must be finite and positive when halftime is enabled"
            )
        if (
            not isinstance(self.fulltime_seconds, numbers.Real)
            or isinstance(self.fulltime_seconds, bool)
            or not math.isfinite(self.fulltime_seconds)
            or self.fulltime_seconds <= 0.0
            or (
                self.halftime_enabled and self.fulltime_seconds <= self.halftime_seconds
            )
        ):
            relationship = (
                " and greater than halftime_seconds" if self.halftime_enabled else ""
            )
            raise ValueError(
                f"fulltime_seconds must be finite and positive{relationship}"
            )
        if not self.halftime_enabled:
            # An inactive boundary must not create distinct fingerprints or a
            # renderer-only validation failure. Canonicalize it to the sole
            # active regulation boundary on the host.
            object.__setattr__(self, "halftime_seconds", float(self.fulltime_seconds))
        if (
            not isinstance(self.halftime_interval_s, numbers.Real)
            or isinstance(self.halftime_interval_s, bool)
            or not math.isfinite(self.halftime_interval_s)
            or self.halftime_interval_s < 0.0
            or self.halftime_interval_s > 15.0 * 60.0
        ):
            raise ValueError("halftime_interval_s must be finite and between 0 and 900")
        if (
            not isinstance(self.minimum_team_players, tuple)
            or len(self.minimum_team_players) != 2
            or any(
                not isinstance(value, numbers.Integral)
                or isinstance(value, bool)
                or value < 0
                or value > IFAB_MAX_TEAM_PLAYERS
                for value in self.minimum_team_players
            )
        ):
            raise ValueError(
                "minimum_team_players must contain two integers from 0 to 11"
            )

    def clock_ticks(self, timebase: Timebase) -> tuple[int, int]:
        """Return static full-time and half-time control-tick boundaries."""

        fulltime = timebase.control_steps_for(self.fulltime_seconds, minimum=1)
        halftime = (
            timebase.control_steps_for(self.halftime_seconds, minimum=1)
            if self.halftime_enabled
            else fulltime
        )
        if fulltime > MAX_REGULATION_CONTROL_TICKS:
            raise ValueError("duration exceeds the reversible model-clock horizon")
        if self.halftime_enabled and halftime > np.iinfo(np.int32).max:
            raise ValueError("halftime exceeds the int32 control clock")
        if self.halftime_enabled and halftime >= fulltime:
            raise ValueError("halftime must precede fulltime after tick conversion")
        return fulltime, halftime


class MatchSetup(NamedTuple):
    """Per-match kickoff setup kept outside the recurrent rollout state."""

    second_half_positions: jax.Array
    second_half_kickoff_team: jax.Array


class RuleOutcome(NamedTuple):
    """Small rule-consequence sidecar returned with an episode transition."""

    score_delta: jax.Array
    restart_opened: jax.Array
    restart_kind: jax.Array
    restart_team: jax.Array


class EpisodeStep(NamedTuple):
    """One match-aware control transition."""

    frame: ControlFrame
    outcome: RuleOutcome
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    terminal_frozen: jax.Array
    halftime_reset: jax.Array


def make_match_setup(
    initial_state: State,
    *,
    second_half_positions: np.ndarray | jax.Array | None = None,
) -> MatchSetup:
    """Build the canonical second-half setup from the initial match state.

    The default second-half formation is a 180-degree rotation of the initial
    formation, never of the live state at the half-time boundary.  A caller
    may provide an independently designed second-half world formation.
    """

    first = np.asarray(initial_state.players.position, dtype=np.float32)
    expected = (first.shape[0], 2)
    if first.shape != expected or not np.all(np.isfinite(first)):
        raise ValueError("initial player positions must be finite [N, 2]")
    if second_half_positions is None:
        second = -first
    else:
        second = np.asarray(second_half_positions, dtype=np.float32)
        if second.shape != expected or not np.all(np.isfinite(second)):
            raise ValueError(
                f"second_half_positions must be finite with shape {expected}"
            )
    opening_team = int(np.asarray(initial_state.kickoff_team))
    if opening_team not in (TEAM_0, TEAM_1):
        raise ValueError("initial kickoff_team must identify one team")
    return MatchSetup(
        second_half_positions=jnp.asarray(second, dtype=jnp.float32),
        second_half_kickoff_team=jnp.int32(TEAM_1 - opening_team),
    )


def _active_per_team(state: State) -> jax.Array:
    active = state.players.active
    return jnp.stack(
        [jnp.sum(active & (state.players.team_id == team)) for team in range(2)]
    ).astype(jnp.int32)


def _status(
    state: State,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    active_per_team = _active_per_team(state)
    below_minimum = jnp.any(
        active_per_team < jnp.asarray(minimum_team_players, dtype=jnp.int32)
    )
    ordinary_restart = (
        (state.restart.kind > jnp.int32(RK_NONE))
        & (state.restart.kind < jnp.int32(RESTART_COUNT))
        & (state.restart.kind != jnp.int32(RK_GK_HOLD))
    )
    valid_restart_team = (state.restart.team == jnp.int32(TEAM_0)) | (
        state.restart.team == jnp.int32(TEAM_1)
    )
    safe_restart_team = jnp.clip(state.restart.team, TEAM_0, TEAM_1)
    has_restart_candidate = valid_restart_team & (
        active_per_team[safe_restart_team] > 0
    )
    terminated = below_minimum | (ordinary_restart & (~has_restart_candidate))
    live_play_ticks = regulation_elapsed_ticks(state)
    penalty_incomplete = (state.restart.kind == jnp.int32(RK_PENALTY)) | (
        state.penalty_completion_active
    )
    regulation_complete = (live_play_ticks >= jnp.int32(fulltime_tick)) & (
        ~penalty_incomplete
    )
    wall_clock_exhausted = state.control_tick.astype(jnp.int32) >= jnp.int32(
        MAX_WALL_CONTROL_TICKS
    )
    truncated = regulation_complete | wall_clock_exhausted
    return terminated, truncated


def _advance_penalty_completion(
    entry_state: State,
    frame: ControlFrame,
) -> ControlFrame:
    """Keep a taken period-ending penalty live until its outcome is settled.

    A clock boundary must not discard a pending penalty. A moving defending-
    goalkeeper parry remains live, but any new contact by the taker, another
    attacker, or an outfield
    defender completes the extended kick. No timeout or trajectory coefficient
    is introduced.
    """

    released = (entry_state.restart.kind == jnp.int32(RK_PENALTY)) & jnp.any(
        frame.kick_applied
    )
    active = entry_state.penalty_completion_active | released
    completion_team = jnp.where(
        released,
        entry_state.restart.team,
        entry_state.penalty_completion_team,
    ).astype(jnp.int32)

    speed_squared = jnp.sum(frame.state.ball.velocity * frame.state.ball.velocity)
    stopped = speed_squared <= jnp.asarray(
        STATIONARY_SPEED_EPS * STATIONARY_SPEED_EPS,
        dtype=frame.state.ball.velocity.dtype,
    )

    settling_followup = frame.penalty_settling_contact
    settled = active & (
        (~frame.state.ball.live)
        | (frame.state.restart.kind != jnp.int32(RK_NONE))
        | settling_followup
        | stopped
    )
    next_active = active & (~settled)
    next_team = jnp.where(next_active, completion_team, jnp.int32(NO_TEAM))
    return frame._replace(
        state=frame.state._replace(
            penalty_completion_active=next_active,
            penalty_completion_team=next_team,
        )
    )


def _dead_ball_for_added_time(state: State) -> jax.Array:
    """Count only out-of-play restarts, never goalkeeper possession."""

    return (state.restart.kind != jnp.int32(RK_NONE)) & (
        state.restart.kind != jnp.int32(RK_GK_HOLD)
    )


def _empty_contact(dtype) -> ContactResult:
    del dtype
    return ContactResult(
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        intent=jnp.int32(INTENT_MOVE),
        outcome=jnp.int32(OUTCOME_NONE),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=jnp.int32(LAW11_NONE),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_NONE),
    )


def _clear_restart_release() -> RestartReleaseProvenance:
    return RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )


def _halftime_state(
    state: State,
    setup: MatchSetup,
    *,
    halftime_interval_s: float,
    halftime_tick: int,
    short_stamina: ShortStamina,
    ball_geometry: Ball,
    stadium: Stadium,
    body: BodyContact,
) -> State:
    active = state.players.active
    attack_direction = -state.attack_direction
    canonical_position = setup.second_half_positions
    position = jnp.where(active[:, None], canonical_position, state.players.position)
    velocity = jnp.where(
        active[:, None], jnp.zeros_like(state.players.velocity), state.players.velocity
    )
    reset_body_forward = initial_player_body_forward(
        state.players.team_id, attack_direction
    )
    body_forward = jnp.where(
        active[:, None], reset_body_forward, state.players.body_forward
    )
    recovered_short = recover_short_stamina_at_rest(
        state.players.stamina_short,
        halftime_interval_s,
        short=short_stamina,
    )
    stamina_short = jnp.where(
        active, jnp.clip(recovered_short, 0.0, 1.0), state.players.stamina_short
    )
    players = state.players._replace(
        position=position,
        velocity=velocity,
        body_forward=body_forward,
        gaze_yaw=jnp.where(active, 0.0, state.players.gaze_yaw),
        stamina_short=stamina_short,
        challenge_recovery_substeps=jnp.zeros_like(
            state.players.challenge_recovery_substeps
        ),
        contact_lock_substeps=jnp.zeros_like(state.players.contact_lock_substeps),
        aerial_recovery_substeps=jnp.zeros_like(state.players.aerial_recovery_substeps),
        possession_loss_lock_substeps=jnp.zeros_like(
            state.players.possession_loss_lock_substeps
        ),
    )
    kickoff_team = setup.second_half_kickoff_team.astype(jnp.int32)
    center = jnp.asarray(
        [0.0, 0.0, ball_geometry.radius], dtype=state.ball.position.dtype
    )
    ball = BallState(
        position=center,
        velocity=jnp.zeros_like(state.ball.velocity),
        spin=jnp.zeros_like(state.ball.spin),
        live=jnp.bool_(False),
    )
    possession = PossessionState(
        team=jnp.int32(NO_TEAM),
        player=jnp.int32(NO_PLAYER),
        previous_team=jnp.int32(NO_TEAM),
        control_ticks=jnp.int32(0),
        last_contact=_empty_contact(state.ball.position.dtype),
    )
    provisional = state._replace(
        ball=ball,
        players=players,
        attack_direction=attack_direction,
        kickoff_team=kickoff_team,
        possession=possession,
        restart=RestartState(
            kind=jnp.int32(RK_KICKOFF),
            team=kickoff_team,
            substeps_remaining=jnp.int32(0),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            opened_control_tick=state.control_tick,
        ),
        restart_release=_clear_restart_release(),
        gk_backpass_team=jnp.int32(NO_TEAM),
        first_half_wall_end_tick=state.control_tick,
        first_half_live_extension_ticks=jnp.maximum(
            regulation_elapsed_ticks(state) - jnp.int32(halftime_tick),
            jnp.int32(0),
        ),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
    )
    taker = select_restart_taker(
        provisional, RK_KICKOFF, kickoff_team, center, stadium=stadium
    )
    designated = provisional._replace(restart=provisional.restart._replace(taker=taker))
    return designated


def _prepare_public_restart(
    frame: ControlFrame,
    *,
    ball_geometry: Ball,
    stadium: Stadium,
    body: BodyContact,
    restart_timing: RestartTiming,
) -> ControlFrame:
    """Return an observation-ready restart without charging teleports."""

    state = frame.state
    continuous_approach = continuous_restart_approach_enabled(
        state.restart.kind,
        config=restart_timing,
    )
    positioning = prepare_restart_positioning(
        state,
        stadium=stadium,
        ball=ball_geometry,
        body=body,
        preserve_taker=continuous_approach,
    )
    moved = jnp.any(
        jnp.abs(positioning.position - state.players.position) > GEOMETRY_EPS,
        axis=-1,
    )
    players = state.players._replace(
        position=positioning.position,
        body_forward=body_forward_from_angle(positioning.facing),
        velocity=jnp.where(
            moved[:, None],
            jnp.zeros_like(state.players.velocity),
            state.players.velocity,
        ),
    )
    return frame._replace(
        state=state._replace(
            players=players,
            restart_layout_ready=positioning.taker_ready,
        )
    )


def _zero_control_frame(state: State, offside_state: OffsideState) -> ControlFrame:
    player_count = state.players.position.shape[0]
    return ControlFrame(
        state=state,
        offside_state=offside_state,
        contact_attempted=jnp.zeros(player_count, dtype=jnp.bool_),
        kick_applied=jnp.zeros(player_count, dtype=jnp.bool_),
        restart_opened=jnp.bool_(False),
        event_budget_exhausted=jnp.bool_(False),
        # No override was consumed; conjunction over an empty execution is
        # valid rather than a replay error.
        contest_override_valid=jnp.bool_(True),
        penalty_settling_contact=jnp.bool_(False),
    )


def step_episode(
    state: State,
    offside_state: OffsideState,
    action: IntentAction,
    key: jax.Array,
    contest_overrides: ContestOverride,
    setup: MatchSetup,
    *,
    match: MatchConfig = MatchConfig(),
    timebase: Timebase = DEFAULT_TIMEBASE,
    boundary_margin_m: float = 0.0,
    locomotion_enabled: jax.Array | None = None,
    position_update_enabled: jax.Array | None = None,
    separation_pinned: jax.Array | None = None,
    goalkeeper_holding: GoalkeeperHolding = GoalkeeperHolding(),
    restart_timing: RestartTiming = RestartTiming(),
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    action_scale: ActionScale = ActionScale(),
    contact_timing: ContactTiming = ContactTiming(),
    contest_config: Contest = Contest(),
    body_foul_config: BodyFoul = BodyFoul(),
    player_physics: PlayerPhysics = PlayerPhysics(),
    perception: Perception = Perception(),
    body: BodyContact = BodyContact(),
    long_stamina: LongStamina = LongStamina(),
    short_stamina: ShortStamina = ShortStamina(),
    ball_physics: BallPhysics = BallPhysics(),
    _collect_events: bool = False,
    _render_fps: float | None = None,
) -> EpisodeStep:
    """Advance one policy frame or return an exact absorbing terminal state."""

    fulltime_tick, halftime_tick = match.clock_ticks(timebase)
    expected_positions = (state.players.position.shape[0], 2)
    if setup.second_half_positions.shape != expected_positions:
        raise ValueError(
            f"setup second_half_positions must have shape {expected_positions}"
        )
    if setup.second_half_kickoff_team.shape != ():
        raise ValueError("setup second_half_kickoff_team must be scalar")
    entry_terminated, entry_truncated = _status(
        state,
        fulltime_tick=fulltime_tick,
        minimum_team_players=match.minimum_team_players,
    )
    entry_done = entry_terminated | entry_truncated

    def advance(_):
        entry_state, _ = repair_broken_restart_taker(
            state,
            body=body,
            stadium=stadium,
        )
        first_half_open = jnp.bool_(match.halftime_enabled) & (
            entry_state.first_half_wall_end_tick < 0
        )
        period_end_tick = jnp.where(
            first_half_open,
            jnp.int32(halftime_tick),
            jnp.int32(fulltime_tick),
        )
        period_boundary_penalty_enforced = (
            regulation_elapsed_ticks(entry_state) >= period_end_tick
        )
        frame = step_control_frame(
            entry_state,
            offside_state,
            action,
            key,
            contest_overrides,
            timebase=timebase,
            boundary_margin_m=boundary_margin_m,
            locomotion_enabled=locomotion_enabled,
            position_update_enabled=position_update_enabled,
            separation_pinned=separation_pinned,
            minimum_team_players=match.minimum_team_players,
            goalkeeper_holding=goalkeeper_holding,
            restart_timing=restart_timing,
            ball_geometry=ball_geometry,
            stadium=stadium,
            reach=reach,
            action_scale=action_scale,
            contact_timing=contact_timing,
            contest_config=contest_config,
            body_foul_config=body_foul_config,
            player_physics=player_physics,
            perception=perception,
            body=body,
            long_stamina=long_stamina,
            short_stamina=short_stamina,
            ball_physics=ball_physics,
            period_boundary_penalty_enforced=period_boundary_penalty_enforced,
            _collect_events=_collect_events,
            _render_fps=_render_fps,
        )
        dead_ball_increment = _dead_ball_for_added_time(entry_state).astype(jnp.int32)
        frame = frame._replace(
            state=frame.state._replace(
                dead_ball_control_ticks=(
                    entry_state.dead_ball_control_ticks + dead_ball_increment
                ).astype(jnp.int32)
            )
        )
        frame = _advance_penalty_completion(entry_state, frame)
        terminated, truncated = _status(
            frame.state,
            fulltime_tick=fulltime_tick,
            minimum_team_players=match.minimum_team_players,
        )
        halftime = (
            jnp.bool_(match.halftime_enabled)
            & (frame.state.first_half_wall_end_tick < 0)
            & (regulation_elapsed_ticks(frame.state) >= jnp.int32(halftime_tick))
            & (~terminated)
            & (~truncated)
            & (~frame.state.penalty_completion_active)
            & (frame.state.restart.kind != jnp.int32(RK_PENALTY))
        )

        def reset_for_second_half(current):
            return current._replace(
                state=_halftime_state(
                    current.state,
                    setup,
                    halftime_interval_s=match.halftime_interval_s,
                    halftime_tick=halftime_tick,
                    short_stamina=short_stamina,
                    ball_geometry=ball_geometry,
                    stadium=stadium,
                    body=body,
                ),
                offside_state=clear_offside_state(current.offside_state),
                restart_opened=jnp.bool_(True),
            )

        # Halftime occurs once per match.  Keep its full restart-positioning
        # solve out of every ordinary control-frame execution.
        pre_halftime_position = frame.state.players.position
        frame = jax.lax.cond(
            halftime,
            reset_for_second_half,
            lambda current: current,
            frame,
        )
        if _collect_events:
            halftime_moved = halftime & jnp.any(
                jnp.abs(frame.state.players.position - pre_halftime_position)
                > GEOMETRY_EPS,
                axis=-1,
            )
            overwrite_rows = jnp.broadcast_to(
                halftime, frame.action_receipt.flags.shape
            )
            frame = frame._replace(
                action_receipt=frame.action_receipt._replace(
                    flags=(
                        frame.action_receipt.flags
                        | overwrite_rows.astype(jnp.uint32)
                        * jnp.uint32(ACTION_FLAG_ENVIRONMENT_OVERWRITE)
                    ),
                    displacement_source=(
                        frame.action_receipt.displacement_source
                        | halftime_moved.astype(jnp.uint16)
                        * jnp.uint16(DISPLACEMENT_HALFTIME_RESET)
                    ),
                    primary_reason=jnp.where(
                        overwrite_rows,
                        jnp.int16(ACTION_REASON_ENVIRONMENT_OVERWRITE),
                        frame.action_receipt.primary_reason,
                    ),
                )
            )
        return frame, halftime

    def freeze(_):
        frame = _zero_control_frame(state, offside_state)
        if _collect_events:
            event_fields = {
                "action_trace": trace_action(action, executed=jnp.bool_(False)),
                "action_receipt": trace_action_receipt(
                    action, executed=jnp.bool_(False)
                ),
                "events": _empty_frame_events(
                    timebase.decimation, state.ball.position.dtype
                ),
            }
            if _render_fps is None:
                frame = ControlFrameWithEvents(*frame, **event_fields)
            else:
                render_sample_count = round(float(_render_fps) / timebase.control_fps)
                frame = _ControlFrameWithEventsAndRenderSamples(
                    *frame,
                    **event_fields,
                    render_state=jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value, (render_sample_count,) + value.shape
                        ),
                        state,
                    ),
                    render_offside_state=jax.tree.map(
                        lambda value: jnp.broadcast_to(
                            value, (render_sample_count,) + value.shape
                        ),
                        offside_state,
                    ),
                )
        return frame, jnp.bool_(False)

    frame, halftime_reset = jax.lax.cond(
        entry_done,
        freeze,
        advance,
        operand=None,
    )
    output_terminated, output_truncated = _status(
        frame.state,
        fulltime_tick=fulltime_tick,
        minimum_team_players=match.minimum_team_players,
    )
    prepare_public = (
        (~entry_done)
        & (~output_terminated)
        & (~output_truncated)
        & (frame.state.restart.kind != jnp.int32(RK_NONE))
        & (~frame.state.restart_layout_ready)
    )
    pre_public_position = frame.state.players.position
    frame = jax.lax.cond(
        prepare_public,
        lambda current: _prepare_public_restart(
            current,
            ball_geometry=ball_geometry,
            stadium=stadium,
            body=body,
            restart_timing=restart_timing,
        ),
        lambda current: current,
        frame,
    )
    if _collect_events:
        referee_moved = prepare_public & jnp.any(
            jnp.abs(frame.state.players.position - pre_public_position) > GEOMETRY_EPS,
            axis=-1,
        )
        frame = frame._replace(
            action_receipt=frame.action_receipt._replace(
                flags=(
                    frame.action_receipt.flags
                    | referee_moved.astype(jnp.uint32)
                    * jnp.uint32(ACTION_FLAG_REFEREE_PROJECTION)
                ),
                displacement_source=(
                    frame.action_receipt.displacement_source
                    | referee_moved.astype(jnp.uint16)
                    * jnp.uint16(DISPLACEMENT_REFEREE_PROJECTION)
                ),
                primary_reason=jnp.where(
                    referee_moved & (~halftime_reset),
                    jnp.int16(ACTION_REASON_REFEREE_PROJECTION),
                    frame.action_receipt.primary_reason,
                ),
            )
        )
    if _render_fps is not None:
        # Episode-only boundaries (halftime and public restart projection)
        # occur after the physics scan. Preserve their authoritative endpoint
        # in the last renderer sample without enlarging the ordinary step.
        frame = frame._replace(
            render_state=jax.tree.map(
                lambda samples, endpoint: samples.at[-1].set(endpoint),
                frame.render_state,
                frame.state,
            ),
            render_offside_state=jax.tree.map(
                lambda samples, endpoint: samples.at[-1].set(endpoint),
                frame.render_offside_state,
                frame.offside_state,
            ),
        )
    score_delta = frame.state.score - state.score
    outcome = RuleOutcome(
        score_delta=score_delta,
        restart_opened=frame.restart_opened,
        restart_kind=jnp.where(
            frame.restart_opened, frame.state.restart.kind, RK_NONE
        ).astype(jnp.int32),
        restart_team=jnp.where(
            frame.restart_opened, frame.state.restart.team, NO_TEAM
        ).astype(jnp.int32),
    )
    return EpisodeStep(
        frame=frame,
        outcome=outcome,
        terminated=output_terminated,
        truncated=output_truncated,
        done=output_terminated | output_truncated,
        terminal_frozen=entry_done,
        halftime_reset=halftime_reset,
    )


__all__ = [
    "EpisodeStep",
    "MatchConfig",
    "MatchSetup",
    "RuleOutcome",
    "make_match_setup",
    "step_episode",
]
