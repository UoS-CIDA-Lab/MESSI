"""Fixed, invertible model views over the SI-unit rollout core.

Observer-relative entities are retained while absolute, observer-relative,
and ball-boundary scales remain semantically distinct. Runtime clipping,
per-match roster maxima, and flattened float categoricals
are rejected: FootballWorld uses immutable environment configuration only,
keeps the PyTree structure and dtypes, and never clips a normalized value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.config.perception import Perception
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.roster_sampling import RosterSampling
from footballworld.core.constants import RK_GK_HOLD, RK_KICKOFF, RK_NONE
from footballworld.core.contact import ContactResult
from footballworld.core.state import (
    BallState,
    PlayerState,
    PossessionState,
    RestartReleaseProvenance,
    RestartState,
    State,
)
from footballworld.core.timebase import Timebase
from footballworld.environment.clock import (
    MatchClockTicks,
    NormalizedMatchClock,
    match_clock_ticks,
    normalize_match_clock,
)
from footballworld.environment.episode import (
    MAX_WALL_CONTROL_TICKS,
    MatchConfig,
)
from footballworld.environment.management import (
    ManagerBenchObservation,
    ManagerObservation,
    ManagerOnFieldObservation,
    PlayerTacticalObservation,
    SquadSetup,
)
from footballworld.environment.observation import (
    BallObservation,
    MatchObservation,
    Observation,
    PlayerObservations,
    PossessionObservation,
    RestartObservation,
    RestartReleaseObservation,
    RosterMetadata,
    SelfObservation,
)
from footballworld.rules.gk_holding import holding_limit_substeps
from footballworld.rules.offside import OffsideState
from footballworld.rules.restart_timing import forced_release_delay_substeps

MODEL_OBSERVATION_SCHEMA_VERSION = 7
MODEL_STATE_SCHEMA_VERSION = 6
MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION = 7
MODEL_ROSTER_SCHEMA_VERSION = 1
MODEL_TACTICAL_OBSERVATION_SCHEMA_VERSION = 1


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class NormalizationContext:
    """Immutable semantic scales derived once from an environment definition."""

    position_scale_x_m: float
    position_scale_y_m: float
    relative_position_scale_x_m: float
    relative_position_scale_y_m: float
    ball_position_scale_x_m: float
    ball_position_scale_y_m: float
    ball_relative_position_scale_x_m: float
    ball_relative_position_scale_y_m: float
    player_speed_scale_mps: float
    player_relative_speed_scale_mps: float
    ball_speed_scale_mps: float
    ball_relative_speed_scale_mps: float
    ball_height_scale_m: float
    ball_spin_scale_radps: float
    min_player_speed_mps: float
    max_player_speed_mps: float
    min_height_m: float
    max_height_m: float
    min_reach_height_m: float
    max_reach_height_m: float
    min_ball_control: float
    max_ball_control: float
    min_endurance_factor: float
    max_endurance_factor: float
    gaze_yaw_limit_radians: float
    contact_lock_substeps: int
    challenge_lock_substeps: int
    aerial_lock_substeps: int
    possession_loss_lock_substeps: int
    restart_delay_substeps: int
    goalkeeper_hold_substeps: int
    halftime_tick: int
    fulltime_tick: int
    counter_scale_ticks: int
    halftime_enabled: bool


class NormalizedMatchObservation(NamedTuple):
    attack_direction: jax.Array
    kickoff_team: jax.Array
    score: jax.Array
    control_tick: jax.Array
    offside_direct_exempt_team: jax.Array
    gk_handling_restricted_team: jax.Array
    gk_handling_restriction_known: jax.Array
    clock: NormalizedMatchClock


class NormalizedObservation(NamedTuple):
    """One model-ready player view with the raw observation hierarchy intact."""

    valid: jax.Array
    self_state: SelfObservation
    players: PlayerObservations
    ball: BallObservation
    possession: PossessionObservation
    restart: RestartObservation
    restart_release: RestartReleaseObservation
    match: NormalizedMatchObservation


PlayerObservation = NormalizedObservation


class NormalizedRosterMetadata(NamedTuple):
    team_id: jax.Array
    player_id: jax.Array
    slot_generation: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


class NormalizedBallState(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    spin: jax.Array
    live: jax.Array


class NormalizedPlayerState(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    body_forward: jax.Array
    gaze_yaw: jax.Array
    team_id: jax.Array
    player_id: jax.Array
    on_pitch: jax.Array
    sent_off: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    challenge_recovery_substeps: jax.Array
    contact_lock_substeps: jax.Array
    aerial_recovery_substeps: jax.Array
    possession_loss_lock_substeps: jax.Array
    yellow_cards: jax.Array


class NormalizedPossessionState(NamedTuple):
    team: jax.Array
    player: jax.Array
    previous_team: jax.Array
    control_ticks: jax.Array
    last_contact: ContactResult


class NormalizedRestartState(NamedTuple):
    kind: jax.Array
    team: jax.Array
    substeps_remaining: jax.Array
    taker: jax.Array
    indirect: jax.Array
    opened_control_tick: jax.Array


class NormalizedGlobalState(NamedTuple):
    """Lossless-in-meaning model view of one authoritative :class:`State`."""

    control_tick: jax.Array
    clock: NormalizedMatchClock
    ball: NormalizedBallState
    players: NormalizedPlayerState
    attack_direction: jax.Array
    kickoff_team: jax.Array
    possession: NormalizedPossessionState
    restart: NormalizedRestartState
    score: jax.Array
    restart_release: RestartReleaseProvenance
    gk_backpass_team: jax.Array
    dead_ball_control_ticks: jax.Array
    first_half_wall_end_tick: jax.Array
    first_half_wall_end_known: jax.Array
    first_half_live_extension_ticks: jax.Array
    penalty_completion_active: jax.Array
    penalty_completion_team: jax.Array
    restart_layout_ready: jax.Array


class NormalizedGlobalRollout(NamedTuple):
    state: NormalizedGlobalState
    offside: OffsideState


class NormalizedPlayerTacticalObservation(NamedTuple):
    valid: jax.Array
    team: jax.Array
    formation_index: jax.Array
    formation_anchor: jax.Array
    formation_role: jax.Array


class NormalizedManagerObservation(NamedTuple):
    valid: jax.Array
    team: jax.Array
    on_field: ManagerOnFieldObservation
    bench: ManagerBenchObservation
    substitutions_remaining: jax.Array
    substitutions_max: jax.Array
    windows_remaining: jax.Array
    windows_max: jax.Array
    score: jax.Array
    control_tick: jax.Array
    restart_kind: jax.Array
    restart_team: jax.Array
    restart_position: jax.Array
    attack_direction: jax.Array
    restart_opened_control_tick: jax.Array
    formation_index: jax.Array
    formation_anchor: jax.Array
    formation_role: jax.Array
    formation_candidate_valid: jax.Array
    formation_candidate_signature: jax.Array
    formation_candidate_probability: jax.Array
    formation_candidate_attack_depth: jax.Array
    formation_candidate_width: jax.Array
    formation_candidate_defender_fraction: jax.Array
    clock: NormalizedMatchClock


def make_normalization_context(
    *,
    timebase: Timebase,
    match: MatchConfig,
    stadium: Stadium,
    ball: Ball,
    action_scale: ActionScale,
    contact_timing: ContactTiming,
    goalkeeper_holding: GoalkeeperHolding,
    restart_timing: RestartTiming,
    ball_physics: BallPhysics,
    roster_sampling: RosterSampling,
    perception: Perception,
) -> NormalizationContext:
    """Build all denominators from immutable, serializable environment input."""

    fulltime_tick, halftime_tick = match.clock_ticks(timebase)
    maximum_release_speed = max(
        action_scale.kick_speed_max_mps,
        action_scale.throw_speed_max_mps,
        action_scale.control_request_speed_max_mps,
    )
    maximum_reach = roster_sampling.max_height_m + roster_sampling.max_reach_margin_m
    maximum_release_height = max(
        maximum_reach,
        roster_sampling.max_height_m + action_scale.throw_release_height_addition_m,
    )
    # Translation, rotation and release height form one conservative mechanical
    # energy envelope. Drag, passive contacts and valid impacts do not add energy.
    specific_energy_twice = (
        maximum_release_speed * maximum_release_speed
        + ball_physics.ball_inertia_ratio
        * (ball.radius * action_scale.spin_max_radps) ** 2
        + 2.0 * ball_physics.g * maximum_release_height
    )
    ball_speed_scale = math.sqrt(specific_energy_twice)
    ball_height_scale = specific_energy_twice / (2.0 * ball_physics.g)
    ball_spin_scale = max(
        action_scale.spin_max_radps,
        ball_speed_scale / ball.radius,
    )

    def ticks(seconds: float) -> int:
        return max(1, round(seconds / timebase.dt_phys))

    return NormalizationContext(
        position_scale_x_m=stadium.half_length,
        position_scale_y_m=stadium.half_width,
        relative_position_scale_x_m=2.0 * stadium.half_length,
        relative_position_scale_y_m=2.0 * stadium.half_width,
        ball_position_scale_x_m=stadium.half_length + ball.radius,
        ball_position_scale_y_m=stadium.half_width + ball.radius,
        ball_relative_position_scale_x_m=stadium.length + ball.radius,
        ball_relative_position_scale_y_m=stadium.width + ball.radius,
        player_speed_scale_mps=roster_sampling.max_max_speed_mps,
        player_relative_speed_scale_mps=(2.0 * roster_sampling.max_max_speed_mps),
        ball_speed_scale_mps=ball_speed_scale,
        ball_relative_speed_scale_mps=(
            ball_speed_scale + roster_sampling.max_max_speed_mps
        ),
        ball_height_scale_m=ball_height_scale,
        ball_spin_scale_radps=ball_spin_scale,
        min_player_speed_mps=roster_sampling.min_max_speed_mps,
        max_player_speed_mps=roster_sampling.max_max_speed_mps,
        min_height_m=roster_sampling.min_height_m,
        max_height_m=roster_sampling.max_height_m,
        min_reach_height_m=(
            roster_sampling.min_height_m + roster_sampling.min_reach_margin_m
        ),
        max_reach_height_m=maximum_reach,
        min_ball_control=roster_sampling.min_ball_control,
        max_ball_control=roster_sampling.max_ball_control,
        min_endurance_factor=roster_sampling.min_endurance_factor,
        max_endurance_factor=roster_sampling.max_endurance_factor,
        gaze_yaw_limit_radians=math.radians(perception.gaze_yaw_limit_degrees),
        contact_lock_substeps=ticks(contact_timing.active_contact_interval_s),
        challenge_lock_substeps=ticks(
            contact_timing.challenge_recovery_s
            + contact_timing.max_lunge_extra_recovery_s
        ),
        aerial_lock_substeps=ticks(
            max(
                contact_timing.aerial_attempt_recovery_s,
                contact_timing.goalkeeper_dive_recovery_s,
            )
        ),
        possession_loss_lock_substeps=ticks(contact_timing.possession_loss_lock_s),
        restart_delay_substeps=forced_release_delay_substeps(
            timebase=timebase, config=restart_timing
        ),
        goalkeeper_hold_substeps=holding_limit_substeps(
            timebase=timebase, config=goalkeeper_holding
        ),
        halftime_tick=halftime_tick,
        fulltime_tick=fulltime_tick,
        counter_scale_ticks=MAX_WALL_CONTROL_TICKS + 1,
        halftime_enabled=match.halftime_enabled,
    )


def _xy_scale(context: NormalizationContext, dtype) -> jax.Array:
    return jnp.asarray(
        (context.position_scale_x_m, context.position_scale_y_m), dtype=dtype
    )


def _relative_xy_scale(context: NormalizationContext, dtype) -> jax.Array:
    return jnp.asarray(
        (
            context.relative_position_scale_x_m,
            context.relative_position_scale_y_m,
        ),
        dtype=dtype,
    )


def _unit_interval(value, lower: float, upper: float):
    return (value - lower) / (upper - lower)


def _from_unit_interval(value, lower: float, upper: float):
    return lower + value * (upper - lower)


def _counter(value, scale: int):
    return value.astype(jnp.float32) / jnp.float32(scale)


def _restore_counter(value, scale: int):
    return jnp.rint(value * jnp.float32(scale)).astype(jnp.int32)


def _restart_scale(kind, context: NormalizationContext):
    return jnp.where(
        kind == RK_GK_HOLD,
        jnp.float32(context.goalkeeper_hold_substeps),
        jnp.where(
            (kind == RK_NONE) | (kind == RK_KICKOFF),
            jnp.float32(1.0),
            jnp.float32(context.restart_delay_substeps),
        ),
    )


def _normalized_clock(
    state: State, context: NormalizationContext
) -> NormalizedMatchClock:
    return normalize_match_clock(
        match_clock_ticks(
            state,
            halftime_tick=context.halftime_tick,
            fulltime_tick=context.fulltime_tick,
            halftime_enabled=context.halftime_enabled,
        ),
        fulltime_tick=context.fulltime_tick,
        counter_scale_ticks=context.counter_scale_ticks,
    )


def denormalize_match_clock(
    clock: NormalizedMatchClock,
    context: NormalizationContext,
    *,
    valid: jax.Array = jnp.bool_(True),
) -> MatchClockTicks:
    """Recover exact tick facts, including the power-of-two sidecar counters."""

    second_half = clock.period == 2
    configured_period_duration = jnp.where(
        second_half,
        jnp.int32(context.fulltime_tick - context.halftime_tick),
        jnp.where(
            jnp.bool_(context.halftime_enabled),
            jnp.int32(context.halftime_tick),
            jnp.int32(context.fulltime_tick),
        ),
    )
    clock_valid = valid & ((clock.period == 1) | (clock.period == 2))
    period_duration = jnp.where(clock_valid, configured_period_duration, jnp.int32(0))
    return MatchClockTicks(
        period=jnp.where(clock_valid, clock.period, jnp.int32(0)),
        period_duration_ticks=period_duration,
        period_regulation_elapsed_ticks=jnp.where(
            clock_valid,
            jnp.rint(clock.period_regulation_progress * period_duration).astype(
                jnp.int32
            ),
            jnp.int32(0),
        ),
        match_regulation_elapsed_ticks=jnp.where(
            clock_valid,
            jnp.rint(clock.match_regulation_progress * context.fulltime_tick).astype(
                jnp.int32
            ),
            jnp.int32(0),
        ),
        period_dead_ball_ticks=jnp.where(
            clock_valid,
            _restore_counter(
                clock.period_dead_ball_counter, context.counter_scale_ticks
            ),
            jnp.int32(0),
        ),
        added_time_active=clock.added_time_active & clock_valid,
        added_time_elapsed_ticks=jnp.where(
            clock_valid,
            _restore_counter(
                clock.added_time_elapsed_counter, context.counter_scale_ticks
            ),
            jnp.int32(0),
        ),
        added_time_remaining_ticks=jnp.where(
            clock_valid,
            _restore_counter(
                clock.added_time_remaining_counter, context.counter_scale_ticks
            ),
            jnp.int32(0),
        ),
    )


def normalize_observation(
    observation: Observation,
    context: NormalizationContext,
) -> NormalizedObservation:
    """Normalize one or many raw observations without clipping any leaf."""

    xy = _xy_scale(context, observation.self_state.position.dtype)
    relative_xy = _relative_xy_scale(context, observation.self_state.position.dtype)
    ball = observation.ball.relative_state
    ball_position = ball[..., :3] / jnp.asarray(
        (
            context.ball_relative_position_scale_x_m,
            context.ball_relative_position_scale_y_m,
            context.ball_height_scale_m,
        ),
        dtype=ball.dtype,
    )
    ball_velocity = ball[..., 3:6] / jnp.float32(context.ball_relative_speed_scale_mps)
    ball_spin = ball[..., 6:9] / jnp.float32(context.ball_spin_scale_radps)
    players = observation.players
    restart_scale = _restart_scale(observation.restart.kind, context)
    return NormalizedObservation(
        valid=observation.valid,
        self_state=SelfObservation(
            player_index=observation.self_state.player_index,
            position=observation.self_state.position / xy,
            velocity=(
                observation.self_state.velocity
                / jnp.float32(context.player_speed_scale_mps)
            ),
            gaze_yaw=(
                observation.self_state.gaze_yaw
                / jnp.float32(context.gaze_yaw_limit_radians)
            ),
        ),
        players=players._replace(
            relative_position=players.relative_position / relative_xy,
            relative_velocity=(
                players.relative_velocity
                / jnp.float32(context.player_relative_speed_scale_mps)
            ),
            gaze_yaw=(players.gaze_yaw / jnp.float32(context.gaze_yaw_limit_radians)),
            challenge_recovery_substeps=_counter(
                players.challenge_recovery_substeps,
                context.challenge_lock_substeps,
            ),
            contact_lock_substeps=_counter(
                players.contact_lock_substeps, context.contact_lock_substeps
            ),
            aerial_recovery_substeps=_counter(
                players.aerial_recovery_substeps, context.aerial_lock_substeps
            ),
            possession_loss_lock_substeps=_counter(
                players.possession_loss_lock_substeps,
                context.possession_loss_lock_substeps,
            ),
        ),
        ball=observation.ball._replace(
            relative_state=jnp.concatenate(
                (ball_position, ball_velocity, ball_spin), axis=-1
            )
        ),
        possession=observation.possession._replace(
            control_ticks=_counter(
                observation.possession.control_ticks, context.counter_scale_ticks
            )
        ),
        restart=observation.restart._replace(
            substeps_remaining=(
                observation.restart.substeps_remaining.astype(jnp.float32)
                / restart_scale
            )
        ),
        restart_release=observation.restart_release,
        match=NormalizedMatchObservation(
            attack_direction=observation.match.attack_direction,
            kickoff_team=observation.match.kickoff_team,
            # Score is a small exact count, not a continuous physical value.
            score=observation.match.score,
            control_tick=_counter(
                observation.match.control_tick, context.counter_scale_ticks
            ),
            offside_direct_exempt_team=(observation.match.offside_direct_exempt_team),
            gk_handling_restricted_team=(observation.match.gk_handling_restricted_team),
            gk_handling_restriction_known=(
                observation.match.gk_handling_restriction_known
            ),
            clock=normalize_match_clock(
                observation.match.clock,
                fulltime_tick=context.fulltime_tick,
                counter_scale_ticks=context.counter_scale_ticks,
            ),
        ),
    )


def denormalize_observation(
    observation: NormalizedObservation,
    context: NormalizationContext,
) -> Observation:
    """Recover the raw observation values represented by a model view."""

    xy = _xy_scale(context, observation.self_state.position.dtype)
    relative_xy = _relative_xy_scale(context, observation.self_state.position.dtype)
    ball = observation.ball.relative_state
    ball_position = ball[..., :3] * jnp.asarray(
        (
            context.ball_relative_position_scale_x_m,
            context.ball_relative_position_scale_y_m,
            context.ball_height_scale_m,
        ),
        dtype=ball.dtype,
    )
    ball_velocity = ball[..., 3:6] * jnp.float32(context.ball_relative_speed_scale_mps)
    ball_spin = ball[..., 6:9] * jnp.float32(context.ball_spin_scale_radps)
    players = observation.players
    restart_scale = _restart_scale(observation.restart.kind, context)
    raw_clock = denormalize_match_clock(
        observation.match.clock,
        context,
        valid=observation.valid,
    )
    return Observation(
        valid=observation.valid,
        self_state=SelfObservation(
            player_index=observation.self_state.player_index,
            position=observation.self_state.position * xy,
            velocity=(
                observation.self_state.velocity
                * jnp.float32(context.player_speed_scale_mps)
            ),
            gaze_yaw=(
                observation.self_state.gaze_yaw
                * jnp.float32(context.gaze_yaw_limit_radians)
            ),
        ),
        players=players._replace(
            relative_position=players.relative_position * relative_xy,
            relative_velocity=(
                players.relative_velocity
                * jnp.float32(context.player_relative_speed_scale_mps)
            ),
            gaze_yaw=(players.gaze_yaw * jnp.float32(context.gaze_yaw_limit_radians)),
            challenge_recovery_substeps=_restore_counter(
                players.challenge_recovery_substeps,
                context.challenge_lock_substeps,
            ),
            contact_lock_substeps=_restore_counter(
                players.contact_lock_substeps, context.contact_lock_substeps
            ),
            aerial_recovery_substeps=_restore_counter(
                players.aerial_recovery_substeps, context.aerial_lock_substeps
            ),
            possession_loss_lock_substeps=_restore_counter(
                players.possession_loss_lock_substeps,
                context.possession_loss_lock_substeps,
            ),
        ),
        ball=observation.ball._replace(
            relative_state=jnp.concatenate(
                (ball_position, ball_velocity, ball_spin), axis=-1
            )
        ),
        possession=observation.possession._replace(
            control_ticks=_restore_counter(
                observation.possession.control_ticks, context.counter_scale_ticks
            )
        ),
        restart=observation.restart._replace(
            substeps_remaining=jnp.rint(
                observation.restart.substeps_remaining * restart_scale
            ).astype(jnp.int32)
        ),
        restart_release=observation.restart_release,
        match=MatchObservation(
            attack_direction=observation.match.attack_direction,
            kickoff_team=observation.match.kickoff_team,
            score=observation.match.score,
            control_tick=_restore_counter(
                observation.match.control_tick, context.counter_scale_ticks
            ),
            offside_direct_exempt_team=(observation.match.offside_direct_exempt_team),
            gk_handling_restricted_team=(observation.match.gk_handling_restricted_team),
            gk_handling_restriction_known=(
                observation.match.gk_handling_restriction_known
            ),
            clock=raw_clock,
        ),
    )


def normalize_roster_metadata(
    roster: RosterMetadata,
    context: NormalizationContext,
) -> NormalizedRosterMetadata:
    return NormalizedRosterMetadata(
        team_id=roster.team_id,
        player_id=roster.player_id,
        slot_generation=roster.slot_generation,
        is_goalkeeper=roster.is_goalkeeper,
        max_speed=_unit_interval(
            roster.max_speed,
            context.min_player_speed_mps,
            context.max_player_speed_mps,
        ),
        reach_height=_unit_interval(
            roster.reach_height,
            context.min_reach_height_m,
            context.max_reach_height_m,
        ),
        height=_unit_interval(
            roster.height, context.min_height_m, context.max_height_m
        ),
        ball_control=_unit_interval(
            roster.ball_control,
            context.min_ball_control,
            context.max_ball_control,
        ),
        endurance_factor=_unit_interval(
            roster.endurance_factor,
            context.min_endurance_factor,
            context.max_endurance_factor,
        ),
    )


def denormalize_roster_metadata(
    roster: NormalizedRosterMetadata,
    context: NormalizationContext,
) -> RosterMetadata:
    """Recover SI roster abilities from one normalized metadata tree."""

    return RosterMetadata(
        team_id=roster.team_id,
        player_id=roster.player_id,
        slot_generation=roster.slot_generation,
        is_goalkeeper=roster.is_goalkeeper,
        max_speed=_from_unit_interval(
            roster.max_speed,
            context.min_player_speed_mps,
            context.max_player_speed_mps,
        ),
        reach_height=_from_unit_interval(
            roster.reach_height,
            context.min_reach_height_m,
            context.max_reach_height_m,
        ),
        height=_from_unit_interval(
            roster.height, context.min_height_m, context.max_height_m
        ),
        ball_control=_from_unit_interval(
            roster.ball_control,
            context.min_ball_control,
            context.max_ball_control,
        ),
        endurance_factor=_from_unit_interval(
            roster.endurance_factor,
            context.min_endurance_factor,
            context.max_endurance_factor,
        ),
    )


def normalize_global_state(
    state: State, context: NormalizationContext
) -> NormalizedGlobalState:
    """Build a reversible structured model state without changing the carry."""

    xy = _xy_scale(context, state.players.position.dtype)
    p = state.players
    first_half_known = state.first_half_wall_end_tick >= 0
    return NormalizedGlobalState(
        control_tick=_counter(state.control_tick, context.counter_scale_ticks),
        clock=_normalized_clock(state, context),
        ball=NormalizedBallState(
            position=state.ball.position
            / jnp.asarray(
                (
                    context.ball_position_scale_x_m,
                    context.ball_position_scale_y_m,
                    context.ball_height_scale_m,
                ),
                dtype=state.ball.position.dtype,
            ),
            velocity=state.ball.velocity / context.ball_speed_scale_mps,
            spin=state.ball.spin / context.ball_spin_scale_radps,
            live=state.ball.live,
        ),
        players=NormalizedPlayerState(
            position=p.position / xy,
            velocity=p.velocity / context.player_speed_scale_mps,
            body_forward=p.body_forward,
            gaze_yaw=p.gaze_yaw / jnp.float32(context.gaze_yaw_limit_radians),
            team_id=p.team_id,
            player_id=p.player_id,
            on_pitch=p.on_pitch,
            sent_off=p.sent_off,
            is_goalkeeper=p.is_goalkeeper,
            max_speed=_unit_interval(
                p.max_speed,
                context.min_player_speed_mps,
                context.max_player_speed_mps,
            ),
            reach_height=_unit_interval(
                p.reach_height,
                context.min_reach_height_m,
                context.max_reach_height_m,
            ),
            height=_unit_interval(p.height, context.min_height_m, context.max_height_m),
            ball_control=_unit_interval(
                p.ball_control,
                context.min_ball_control,
                context.max_ball_control,
            ),
            endurance_factor=_unit_interval(
                p.endurance_factor,
                context.min_endurance_factor,
                context.max_endurance_factor,
            ),
            stamina_long=p.stamina_long,
            stamina_short=p.stamina_short,
            challenge_recovery_substeps=_counter(
                p.challenge_recovery_substeps, context.challenge_lock_substeps
            ),
            contact_lock_substeps=_counter(
                p.contact_lock_substeps, context.contact_lock_substeps
            ),
            aerial_recovery_substeps=_counter(
                p.aerial_recovery_substeps, context.aerial_lock_substeps
            ),
            possession_loss_lock_substeps=_counter(
                p.possession_loss_lock_substeps,
                context.possession_loss_lock_substeps,
            ),
            yellow_cards=p.yellow_cards,
        ),
        attack_direction=state.attack_direction,
        kickoff_team=state.kickoff_team,
        possession=NormalizedPossessionState(
            team=state.possession.team,
            player=state.possession.player,
            previous_team=state.possession.previous_team,
            control_ticks=_counter(
                state.possession.control_ticks, context.counter_scale_ticks
            ),
            last_contact=state.possession.last_contact,
        ),
        restart=NormalizedRestartState(
            kind=state.restart.kind,
            team=state.restart.team,
            substeps_remaining=(
                state.restart.substeps_remaining.astype(jnp.float32)
                / _restart_scale(state.restart.kind, context)
            ),
            taker=state.restart.taker,
            indirect=state.restart.indirect,
            opened_control_tick=_counter(
                state.restart.opened_control_tick, context.counter_scale_ticks
            ),
        ),
        score=state.score,
        restart_release=state.restart_release,
        gk_backpass_team=state.gk_backpass_team,
        dead_ball_control_ticks=_counter(
            state.dead_ball_control_ticks, context.counter_scale_ticks
        ),
        first_half_wall_end_tick=jnp.where(
            first_half_known,
            _counter(state.first_half_wall_end_tick, context.counter_scale_ticks),
            jnp.float32(0.0),
        ),
        first_half_wall_end_known=first_half_known,
        first_half_live_extension_ticks=_counter(
            state.first_half_live_extension_ticks, context.counter_scale_ticks
        ),
        penalty_completion_active=state.penalty_completion_active,
        penalty_completion_team=state.penalty_completion_team,
        restart_layout_ready=state.restart_layout_ready,
    )


def denormalize_global_state(
    state: NormalizedGlobalState,
    context: NormalizationContext,
) -> State:
    """Recover an SI-unit state; callers keep this out of recurrent stepping."""

    xy = _xy_scale(context, state.players.position.dtype)
    p = state.players
    return State(
        control_tick=_restore_counter(state.control_tick, context.counter_scale_ticks),
        ball=BallState(
            position=state.ball.position
            * jnp.asarray(
                (
                    context.ball_position_scale_x_m,
                    context.ball_position_scale_y_m,
                    context.ball_height_scale_m,
                ),
                dtype=state.ball.position.dtype,
            ),
            velocity=state.ball.velocity * context.ball_speed_scale_mps,
            spin=state.ball.spin * context.ball_spin_scale_radps,
            live=state.ball.live,
        ),
        players=PlayerState(
            position=p.position * xy,
            velocity=p.velocity * context.player_speed_scale_mps,
            body_forward=p.body_forward,
            gaze_yaw=p.gaze_yaw * jnp.float32(context.gaze_yaw_limit_radians),
            team_id=p.team_id,
            player_id=p.player_id,
            on_pitch=p.on_pitch,
            sent_off=p.sent_off,
            is_goalkeeper=p.is_goalkeeper,
            max_speed=_from_unit_interval(
                p.max_speed,
                context.min_player_speed_mps,
                context.max_player_speed_mps,
            ),
            reach_height=_from_unit_interval(
                p.reach_height,
                context.min_reach_height_m,
                context.max_reach_height_m,
            ),
            height=_from_unit_interval(
                p.height, context.min_height_m, context.max_height_m
            ),
            ball_control=_from_unit_interval(
                p.ball_control,
                context.min_ball_control,
                context.max_ball_control,
            ),
            endurance_factor=_from_unit_interval(
                p.endurance_factor,
                context.min_endurance_factor,
                context.max_endurance_factor,
            ),
            stamina_long=p.stamina_long,
            stamina_short=p.stamina_short,
            challenge_recovery_substeps=_restore_counter(
                p.challenge_recovery_substeps, context.challenge_lock_substeps
            ),
            contact_lock_substeps=_restore_counter(
                p.contact_lock_substeps, context.contact_lock_substeps
            ),
            aerial_recovery_substeps=_restore_counter(
                p.aerial_recovery_substeps, context.aerial_lock_substeps
            ),
            possession_loss_lock_substeps=_restore_counter(
                p.possession_loss_lock_substeps,
                context.possession_loss_lock_substeps,
            ),
            yellow_cards=p.yellow_cards,
        ),
        attack_direction=state.attack_direction,
        kickoff_team=state.kickoff_team,
        possession=PossessionState(
            team=state.possession.team,
            player=state.possession.player,
            previous_team=state.possession.previous_team,
            control_ticks=_restore_counter(
                state.possession.control_ticks, context.counter_scale_ticks
            ),
            last_contact=state.possession.last_contact,
        ),
        restart=RestartState(
            kind=state.restart.kind,
            team=state.restart.team,
            substeps_remaining=jnp.rint(
                state.restart.substeps_remaining
                * _restart_scale(state.restart.kind, context)
            ).astype(jnp.int32),
            taker=state.restart.taker,
            indirect=state.restart.indirect,
            opened_control_tick=_restore_counter(
                state.restart.opened_control_tick, context.counter_scale_ticks
            ),
        ),
        score=state.score,
        restart_release=state.restart_release,
        gk_backpass_team=state.gk_backpass_team,
        dead_ball_control_ticks=_restore_counter(
            state.dead_ball_control_ticks, context.counter_scale_ticks
        ),
        first_half_wall_end_tick=jnp.where(
            state.first_half_wall_end_known,
            jnp.rint(
                state.first_half_wall_end_tick
                * jnp.float32(context.counter_scale_ticks)
            ).astype(jnp.int32),
            jnp.int32(-1),
        ),
        first_half_live_extension_ticks=_restore_counter(
            state.first_half_live_extension_ticks, context.counter_scale_ticks
        ),
        penalty_completion_active=state.penalty_completion_active,
        penalty_completion_team=state.penalty_completion_team,
        restart_layout_ready=state.restart_layout_ready,
    )


def normalize_manager_observation(
    observation: ManagerObservation,
    state: State,
    squad: SquadSetup,
    context: NormalizationContext,
) -> NormalizedManagerObservation:
    """Normalize one manager view while preserving its own-bench boundary."""

    xy = _xy_scale(context, observation.on_field.position.dtype)
    on_field = observation.on_field
    bench = observation.bench
    valid = observation.valid
    normalized_on_field = on_field._replace(
        position=on_field.position / xy,
        velocity=on_field.velocity / context.player_speed_scale_mps,
        max_speed=_unit_interval(
            on_field.max_speed,
            context.min_player_speed_mps,
            context.max_player_speed_mps,
        ),
        height=_unit_interval(
            on_field.height, context.min_height_m, context.max_height_m
        ),
        reach_height=_unit_interval(
            on_field.reach_height,
            context.min_reach_height_m,
            context.max_reach_height_m,
        ),
        ball_control=_unit_interval(
            on_field.ball_control,
            context.min_ball_control,
            context.max_ball_control,
        ),
        endurance_factor=_unit_interval(
            on_field.endurance_factor,
            context.min_endurance_factor,
            context.max_endurance_factor,
        ),
    )
    normalized_on_field = jax.tree.map(
        lambda value, original: (
            jnp.where(
                on_field.team_mask[..., None]
                if value.ndim > on_field.team_mask.ndim
                else on_field.team_mask,
                value,
                jnp.zeros_like(value),
            )
            if jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        normalized_on_field,
        on_field,
    )
    normalized_bench = bench._replace(
        max_speed=_unit_interval(
            bench.max_speed,
            context.min_player_speed_mps,
            context.max_player_speed_mps,
        ),
        height=_unit_interval(bench.height, context.min_height_m, context.max_height_m),
        reach_height=_unit_interval(
            bench.reach_height,
            context.min_reach_height_m,
            context.max_reach_height_m,
        ),
        ball_control=_unit_interval(
            bench.ball_control,
            context.min_ball_control,
            context.max_ball_control,
        ),
        endurance_factor=_unit_interval(
            bench.endurance_factor,
            context.min_endurance_factor,
            context.max_endurance_factor,
        ),
    )
    normalized_bench = normalized_bench._replace(
        max_speed=jnp.where(bench.valid, normalized_bench.max_speed, 0.0),
        height=jnp.where(bench.valid, normalized_bench.height, 0.0),
        reach_height=jnp.where(bench.valid, normalized_bench.reach_height, 0.0),
        ball_control=jnp.where(bench.valid, normalized_bench.ball_control, 0.0),
        endurance_factor=jnp.where(bench.valid, normalized_bench.endurance_factor, 0.0),
    )
    clock = _normalized_clock(state, context)
    return NormalizedManagerObservation(
        valid=valid,
        team=observation.team,
        on_field=normalized_on_field,
        bench=normalized_bench,
        substitutions_remaining=observation.substitutions_remaining,
        substitutions_max=observation.substitutions_max,
        windows_remaining=observation.windows_remaining,
        windows_max=observation.windows_max,
        score=observation.score,
        control_tick=_counter(observation.control_tick, context.counter_scale_ticks),
        restart_kind=observation.restart_kind,
        restart_team=observation.restart_team,
        restart_position=observation.restart_position / xy,
        attack_direction=observation.attack_direction,
        restart_opened_control_tick=_counter(
            observation.restart_opened_control_tick, context.counter_scale_ticks
        ),
        formation_index=observation.formation_index,
        formation_anchor=observation.formation_anchor / xy,
        formation_role=observation.formation_role,
        formation_candidate_valid=observation.formation_candidate_valid,
        formation_candidate_signature=observation.formation_candidate_signature,
        formation_candidate_probability=observation.formation_candidate_probability,
        formation_candidate_attack_depth=(
            observation.formation_candidate_attack_depth
            / jnp.float32(context.position_scale_x_m)
        ),
        formation_candidate_width=(
            observation.formation_candidate_width
            / jnp.float32(context.position_scale_y_m)
        ),
        formation_candidate_defender_fraction=(
            observation.formation_candidate_defender_fraction
        ),
        clock=jax.tree.map(
            lambda value: jnp.where(valid, value, jnp.zeros_like(value)),
            clock,
        ),
    )


def normalize_player_tactics(
    observation: PlayerTacticalObservation,
    context: NormalizationContext,
) -> NormalizedPlayerTacticalObservation:
    xy = _xy_scale(context, observation.formation_anchor.dtype)
    return NormalizedPlayerTacticalObservation(
        valid=observation.valid,
        team=observation.team,
        formation_index=observation.formation_index,
        formation_anchor=observation.formation_anchor / xy,
        formation_role=observation.formation_role,
    )


def denormalize_manager_observation(
    observation: NormalizedManagerObservation,
    squad: SquadSetup,
    context: NormalizationContext,
) -> ManagerObservation:
    """Recover the SI manager observation represented by a model view."""

    xy = _xy_scale(context, observation.on_field.position.dtype)
    on_field = observation.on_field
    bench = observation.bench

    def on_field_ability(value, lower: float, upper: float):
        restored = _from_unit_interval(value, lower, upper)
        return jnp.where(on_field.team_mask, restored, jnp.zeros_like(restored))

    def bench_ability(value, lower: float, upper: float):
        restored = _from_unit_interval(value, lower, upper)
        return jnp.where(bench.valid, restored, jnp.zeros_like(restored))

    return ManagerObservation(
        valid=observation.valid,
        team=observation.team,
        on_field=on_field._replace(
            position=on_field.position * xy,
            velocity=on_field.velocity * context.player_speed_scale_mps,
            max_speed=on_field_ability(
                on_field.max_speed,
                context.min_player_speed_mps,
                context.max_player_speed_mps,
            ),
            height=on_field_ability(
                on_field.height, context.min_height_m, context.max_height_m
            ),
            reach_height=on_field_ability(
                on_field.reach_height,
                context.min_reach_height_m,
                context.max_reach_height_m,
            ),
            ball_control=on_field_ability(
                on_field.ball_control,
                context.min_ball_control,
                context.max_ball_control,
            ),
            endurance_factor=on_field_ability(
                on_field.endurance_factor,
                context.min_endurance_factor,
                context.max_endurance_factor,
            ),
        ),
        bench=bench._replace(
            max_speed=bench_ability(
                bench.max_speed,
                context.min_player_speed_mps,
                context.max_player_speed_mps,
            ),
            height=bench_ability(
                bench.height, context.min_height_m, context.max_height_m
            ),
            reach_height=bench_ability(
                bench.reach_height,
                context.min_reach_height_m,
                context.max_reach_height_m,
            ),
            ball_control=bench_ability(
                bench.ball_control,
                context.min_ball_control,
                context.max_ball_control,
            ),
            endurance_factor=bench_ability(
                bench.endurance_factor,
                context.min_endurance_factor,
                context.max_endurance_factor,
            ),
        ),
        substitutions_remaining=observation.substitutions_remaining,
        substitutions_max=observation.substitutions_max,
        windows_remaining=observation.windows_remaining,
        windows_max=observation.windows_max,
        score=observation.score,
        control_tick=_restore_counter(
            observation.control_tick, context.counter_scale_ticks
        ),
        restart_kind=observation.restart_kind,
        restart_team=observation.restart_team,
        restart_position=observation.restart_position * xy,
        attack_direction=observation.attack_direction,
        restart_opened_control_tick=_restore_counter(
            observation.restart_opened_control_tick, context.counter_scale_ticks
        ),
        formation_index=observation.formation_index,
        formation_anchor=observation.formation_anchor * xy,
        formation_role=observation.formation_role,
        formation_candidate_valid=observation.formation_candidate_valid,
        formation_candidate_signature=observation.formation_candidate_signature,
        formation_candidate_probability=observation.formation_candidate_probability,
        formation_candidate_attack_depth=(
            observation.formation_candidate_attack_depth
            * jnp.float32(context.position_scale_x_m)
        ),
        formation_candidate_width=(
            observation.formation_candidate_width
            * jnp.float32(context.position_scale_y_m)
        ),
        formation_candidate_defender_fraction=(
            observation.formation_candidate_defender_fraction
        ),
    )


def denormalize_player_tactics(
    observation: NormalizedPlayerTacticalObservation,
    context: NormalizationContext,
) -> PlayerTacticalObservation:
    """Recover SI tactical anchors from a normalized player view."""

    xy = _xy_scale(context, observation.formation_anchor.dtype)
    return PlayerTacticalObservation(
        valid=observation.valid,
        team=observation.team,
        formation_index=observation.formation_index,
        formation_anchor=observation.formation_anchor * xy,
        formation_role=observation.formation_role,
    )


__all__ = [
    "MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION",
    "MODEL_OBSERVATION_SCHEMA_VERSION",
    "MODEL_ROSTER_SCHEMA_VERSION",
    "MODEL_STATE_SCHEMA_VERSION",
    "MODEL_TACTICAL_OBSERVATION_SCHEMA_VERSION",
    "NormalizationContext",
    "NormalizedBallState",
    "NormalizedGlobalRollout",
    "NormalizedGlobalState",
    "NormalizedManagerObservation",
    "NormalizedObservation",
    "NormalizedPlayerTacticalObservation",
    "NormalizedRosterMetadata",
    "PlayerObservation",
    "denormalize_global_state",
    "denormalize_manager_observation",
    "denormalize_match_clock",
    "denormalize_observation",
    "denormalize_player_tactics",
    "denormalize_roster_metadata",
    "make_normalization_context",
    "normalize_global_state",
    "normalize_manager_observation",
    "normalize_observation",
    "normalize_player_tactics",
    "normalize_roster_metadata",
]
