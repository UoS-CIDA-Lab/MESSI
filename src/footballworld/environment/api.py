"""Minimal public rollout interface over the pure FootballWorld kernels."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

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
from footballworld.config.policies import PolicySelection
from footballworld.config.reach import Reach
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.roster import Player, PlayerProfile
from footballworld.config.roster_sampling import RosterSampling
from footballworld.config.stamina import LongStamina, ShortStamina
from footballworld.core.action import IntentAction
from footballworld.core.constants import RESTART_COUNT, RK_NONE, TEAM_0, TEAM_1
from footballworld.core.randomness import RandomEvent
from footballworld.core.state import State, body_forward_from_angle
from footballworld.core.timebase import DEFAULT_TIMEBASE, Timebase
from footballworld.environment.clock import MatchClockTicks, NormalizedMatchClock
from footballworld.environment.episode import (
    MatchConfig,
    MatchSetup,
    RuleOutcome,
    make_match_setup,
    step_episode,
)
from footballworld.environment.initialization import initialize_state
from footballworld.environment.management import (
    ActingGoalkeeperEvent,
    ManagementInitialization,
    ManagementRules,
    ManagerActingGoalkeeperCommand,
    ManagerAction,
    ManagerCommand,
    ManagerCommandReason,
    ManagerCommandResult,
    ManagerFormationCommand,
    ManagerSetPieceTakerCommand,
    ManagerState,
    ManagerSubstitutionCommand,
    OpeningFormationResult,
    SquadSetup,
    SubstitutionEvent,
    apply_manager_command,
    apply_opening_formation,
    emergency_goalkeeper_command,
    initialize_management,
    observe_player_tactics,
    reconcile_restart_after_roster_change,
)
from footballworld.environment.management import (
    ManagerObservation as SIManagerObservation,
)
from footballworld.environment.management import (
    PlayerTacticalObservation as SIPlayerTacticalObservation,
)
from footballworld.environment.management import (
    observe_manager as observe_manager_si,
)
from footballworld.environment.management import (
    opening_formation_available as _opening_formation_available,
)
from footballworld.environment.normalization import (
    NormalizationContext,
    NormalizedBallState,
    NormalizedGlobalRollout,
    NormalizedGlobalState,
    NormalizedManagerObservation,
    NormalizedObservation,
    NormalizedPlayerTacticalObservation,
    NormalizedRosterMetadata,
    denormalize_global_state,
    denormalize_manager_observation,
    denormalize_match_clock,
    denormalize_observation,
    denormalize_player_tactics,
    denormalize_roster_metadata,
    make_normalization_context,
    normalize_global_state,
    normalize_manager_observation,
    normalize_observation,
    normalize_player_tactics,
    normalize_roster_metadata,
)
from footballworld.environment.observation import (
    Observation as SIObservation,
)
from footballworld.environment.observation import (
    RosterMetadata as SIRosterMetadata,
)
from footballworld.environment.observation import (
    observe,
    observe_all,
    roster_metadata_from_state,
)
from footballworld.environment.substitution import (
    SubstitutionRequest,
    _apply_substitution,
    request_within_roster_domain,
)
from footballworld.environment.transition import (
    ActionReceipt,
    ActionTrace,
    FrameEvents,
    sample_control_contest_overrides,
)
from footballworld.environment.validation import (
    validate_environment_configuration,
    validate_formation_inside_pitch,
    validate_kickoff_team,
    validate_no_torso_overlap,
    validate_public_rosters,
)
from footballworld.rules.gk_holding import holding_limit_substeps
from footballworld.rules.offside import OffsideState
from footballworld.rules.restart_positioning import prepare_restart_positioning
from footballworld.rules.restart_timing import forced_release_delay_substeps
from footballworld.specs import (
    FlatTree,
    TreeSpec,
)
from footballworld.specs import (
    action_receipt_spec as _action_receipt_spec,
)
from footballworld.specs import (
    action_spec as _action_spec,
)
from footballworld.specs import (
    action_trace_spec as _action_trace_spec,
)
from footballworld.specs import (
    event_spec as _event_spec,
)
from footballworld.specs import (
    flatten_global_state as _flatten_global_state,
)
from footballworld.specs import (
    flatten_manager_observation as _flatten_manager_observation,
)
from footballworld.specs import (
    flatten_observation as _flatten_observation,
)
from footballworld.specs import (
    global_state_spec as _global_state_spec,
)
from footballworld.specs import (
    manager_observation_spec as _manager_observation_spec,
)
from footballworld.specs import (
    player_observation_spec as _player_observation_spec,
)
from footballworld.specs import (
    schema_versions as _schema_versions,
)

_ACTIVE_ROSTER_SAMPLING_STREAM = int(RandomEvent.ACTIVE_ROSTER_SAMPLING)
_BENCH_ROSTER_SAMPLING_STREAM = int(RandomEvent.BENCH_ROSTER_SAMPLING)

Observation = NormalizedObservation
PlayerObservation = NormalizedObservation
ManagerObservation = NormalizedManagerObservation
RosterMetadata = NormalizedRosterMetadata
PlayerTacticalObservation = NormalizedPlayerTacticalObservation


class SIRollout(NamedTuple):
    """Authoritative SI-unit recurrent carry; never a model observation."""

    state: State
    offside: OffsideState


# Backward-compatible constructor name; model state is global_state_view().
Rollout = SIRollout


class ResetResult(NamedTuple):
    """Prepared recurrent state and immutable per-match halftime setup."""

    rollout: Rollout
    setup: MatchSetup


class StepResult(NamedTuple):
    """Post-frame rollout and compact action-credit/safety telemetry."""

    rollout: Rollout
    outcome: RuleOutcome
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    terminal_frozen: jax.Array
    halftime_reset: jax.Array


class StepWithEventsResult(NamedTuple):
    """A normal step result plus exact physics-substep event telemetry."""

    rollout: Rollout
    outcome: RuleOutcome
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    terminal_frozen: jax.Array
    halftime_reset: jax.Array
    action_trace: ActionTrace
    action_receipt: ActionReceipt
    events: FrameEvents


class _StepWithEventsAndRenderSamplesResult(NamedTuple):
    """Eventful transition with bounded renderer-only physics samples."""

    rollout: Rollout
    outcome: RuleOutcome
    contact_attempted: jax.Array
    kick_applied: jax.Array
    restart_opened: jax.Array
    event_budget_exhausted: jax.Array
    contest_override_valid: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    terminal_frozen: jax.Array
    halftime_reset: jax.Array
    action_trace: ActionTrace
    action_receipt: ActionReceipt
    events: FrameEvents
    render_samples: Rollout


class SubstitutionStepResult(NamedTuple):
    """Post-event rollout and roster-metadata cache invalidation signal."""

    rollout: Rollout
    applied: jax.Array


class ManagerStepResult(NamedTuple):
    """Atomic field-state and manager-state result of a registered substitution."""

    rollout: Rollout
    management: ManagerState
    applied: jax.Array
    reason: jax.Array
    substitution_event: SubstitutionEvent
    acting_goalkeepers_applied: jax.Array
    acting_goalkeeper_events: ActingGoalkeeperEvent
    roster_metadata_changed: jax.Array


class ManagerCommandStepResult(NamedTuple):
    """One combined low-frequency management transition."""

    rollout: Rollout
    management: ManagerState
    substitutions_applied: jax.Array
    substitution_reasons: jax.Array
    substitution_events: SubstitutionEvent
    formations_applied: jax.Array
    formation_reasons: jax.Array
    acting_goalkeepers_applied: jax.Array
    acting_goalkeeper_reasons: jax.Array
    acting_goalkeeper_events: ActingGoalkeeperEvent
    set_piece_takers_applied: jax.Array
    set_piece_taker_reasons: jax.Array
    roster_metadata_changed: jax.Array


class OpeningFormationStepResult(NamedTuple):
    """Start-only physical formation placement and updated half-time setup."""

    rollout: Rollout
    setup: MatchSetup
    management: ManagerState
    applied: jax.Array


@jax.tree_util.register_static
@dataclass(frozen=True, slots=True)
class FootballWorld:
    """Static environment definition shared by eager, JIT, and VMAP rollouts."""

    timebase: Timebase = DEFAULT_TIMEBASE
    match: MatchConfig = field(default_factory=MatchConfig)
    boundary_margin_m: float = 0.0
    ball: Ball = field(default_factory=Ball)
    stadium: Stadium = field(default_factory=Stadium)
    reach: Reach = field(default_factory=Reach)
    action_scale: ActionScale = field(default_factory=ActionScale)
    contact_timing: ContactTiming = field(default_factory=ContactTiming)
    contest: Contest = field(default_factory=Contest)
    goalkeeper_holding: GoalkeeperHolding = field(default_factory=GoalkeeperHolding)
    restart_timing: RestartTiming = field(default_factory=RestartTiming)
    player_physics: PlayerPhysics = field(default_factory=PlayerPhysics)
    body: BodyContact = field(default_factory=BodyContact)
    long_stamina: LongStamina = field(default_factory=LongStamina)
    short_stamina: ShortStamina = field(default_factory=ShortStamina)
    ball_physics: BallPhysics = field(default_factory=BallPhysics)
    perception: Perception = field(default_factory=Perception)
    roster_sampling: RosterSampling = field(default_factory=RosterSampling)
    policies: PolicySelection = field(default_factory=PolicySelection)
    # Append new public configuration fields to preserve positional callers.
    body_foul: BodyFoul = field(default_factory=BodyFoul)

    def __post_init__(self) -> None:
        if type(self.match) is not MatchConfig:
            raise TypeError("match must be exactly MatchConfig")
        if type(self.roster_sampling) is not RosterSampling:
            raise TypeError("roster_sampling must be exactly RosterSampling")
        if type(self.policies) is not PolicySelection:
            raise TypeError("policies must be exactly PolicySelection")
        validate_environment_configuration(
            timebase=self.timebase,
            boundary_margin_m=self.boundary_margin_m,
            ball=self.ball,
            stadium=self.stadium,
            reach=self.reach,
            action_scale=self.action_scale,
            contact_timing=self.contact_timing,
            contest=self.contest,
            body_foul=self.body_foul,
            goalkeeper_holding=self.goalkeeper_holding,
            restart_timing=self.restart_timing,
            player_physics=self.player_physics,
            body=self.body,
            long_stamina=self.long_stamina,
            short_stamina=self.short_stamina,
            ball_physics=self.ball_physics,
            perception=self.perception,
        )
        # Derive the match-clock bounds at construction so an unsupported
        # duration fails on the host, not on the first traced episode step.
        self.match.clock_ticks(self.timebase)
        if not self.perception.limit_by_view_angle:
            # Only FOV width is inactive in full-view mode. Gaze still drives
            # body-relative pose and rendering, so preserve its two settings.
            object.__setattr__(
                self,
                "perception",
                Perception(
                    gaze_yaw_limit_degrees=self.perception.gaze_yaw_limit_degrees,
                    gaze_slew_rate_degrees_s=(self.perception.gaze_slew_rate_degrees_s),
                ),
            )

    def reset(
        self,
        team_0: Sequence[Player],
        team_1: Sequence[Player],
        *,
        kickoff_team: int = TEAM_0,
        second_half_positions: np.ndarray | jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> ResetResult:
        """Build and fully prepare one pre-kickoff rollout.

        Supplying ``key`` samples one fixed set of episode abilities; omitting it
        preserves each supplied profile exactly. Roster validation and packing
        are deliberately host-side. The returned
        state has already passed through restart positioning, so its first
        observation and first action are evaluated from the same player poses.
        """

        team_0 = tuple(team_0)
        team_1 = tuple(team_1)
        validate_kickoff_team(kickoff_team)
        validate_public_rosters(
            team_0,
            team_1,
            minimum_team_players=self.match.minimum_team_players,
            body=self.body,
            timebase=self.timebase,
            stadium=self.stadium,
            boundary_margin_m=self.boundary_margin_m,
            roster_sampling=self.roster_sampling,
        )
        initial = initialize_state(
            team_0,
            team_1,
            kickoff_team=kickoff_team,
            ball_geometry=self.ball,
            body=self.body,
            roster_sampling=self.roster_sampling,
            sampling_key=(
                None
                if key is None or not self.roster_sampling.enabled
                else jax.random.fold_in(key, _ACTIVE_ROSTER_SAMPLING_STREAM)
            ),
        )
        active = np.asarray(initial.state.players.active, dtype=bool)
        validate_formation_inside_pitch(
            np.asarray(initial.state.players.position)[active],
            stadium=self.stadium,
            name="initial formation",
        )
        validate_no_torso_overlap(
            np.asarray(initial.state.players.position)[active],
            np.arctan2(
                np.asarray(initial.state.players.body_forward)[active, 1],
                np.asarray(initial.state.players.body_forward)[active, 0],
            ),
            body=self.body,
            name="initial formation",
        )
        setup = make_match_setup(
            initial.state,
            second_half_positions=second_half_positions,
        )
        validate_formation_inside_pitch(
            np.asarray(setup.second_half_positions)[active],
            stadium=self.stadium,
            name="second-half formation",
        )
        if second_half_positions is not None:
            future_direction = -np.asarray(initial.state.attack_direction)
            team_id = np.asarray(initial.state.players.team_id)[active]
            second_half_facing = np.where(
                future_direction[team_id] >= 0.0,
                0.0,
                np.pi,
            )
            validate_no_torso_overlap(
                np.asarray(setup.second_half_positions)[active],
                second_half_facing,
                body=self.body,
                name="second-half formation",
            )
        positioning = prepare_restart_positioning(
            initial.state,
            stadium=self.stadium,
            ball=self.ball,
            body=self.body,
        )
        if not bool(np.asarray(positioning.taker_ready)):
            raise ValueError("initial kickoff taker could not be prepared")
        prepared = initial.state._replace(
            players=initial.state.players._replace(
                position=positioning.position,
                body_forward=body_forward_from_angle(positioning.facing),
            ),
            restart_layout_ready=positioning.taker_ready,
        )
        return ResetResult(
            rollout=Rollout(state=prepared, offside=initial.offside),
            setup=setup,
        )

    def step(
        self,
        rollout: Rollout,
        setup: MatchSetup,
        action: IntentAction,
        key: jax.Array,
    ) -> StepResult:
        """Advance one fixed-duration control frame without materializing views."""

        episode = step_episode(
            rollout.state,
            rollout.offside,
            action,
            key,
            sample_control_contest_overrides(self.timebase.decimation),
            setup,
            match=self.match,
            timebase=self.timebase,
            boundary_margin_m=self.boundary_margin_m,
            goalkeeper_holding=self.goalkeeper_holding,
            restart_timing=self.restart_timing,
            ball_geometry=self.ball,
            stadium=self.stadium,
            reach=self.reach,
            action_scale=self.action_scale,
            contact_timing=self.contact_timing,
            contest_config=self.contest,
            body_foul_config=self.body_foul,
            player_physics=self.player_physics,
            perception=self.perception,
            body=self.body,
            long_stamina=self.long_stamina,
            short_stamina=self.short_stamina,
            ball_physics=self.ball_physics,
        )
        frame = episode.frame
        return StepResult(
            rollout=Rollout(state=frame.state, offside=frame.offside_state),
            outcome=episode.outcome,
            contact_attempted=frame.contact_attempted,
            kick_applied=frame.kick_applied,
            restart_opened=frame.restart_opened,
            event_budget_exhausted=frame.event_budget_exhausted,
            contest_override_valid=frame.contest_override_valid,
            terminated=episode.terminated,
            truncated=episode.truncated,
            done=episode.done,
            terminal_frozen=episode.terminal_frozen,
            halftime_reset=episode.halftime_reset,
        )

    def step_with_events(
        self,
        rollout: Rollout,
        setup: MatchSetup,
        action: IntentAction,
        key: jax.Array,
    ) -> StepWithEventsResult:
        """Advance one frame or batch and retain exact substep events.

        This opt-in path does not infer or recompute events. It exposes values
        already produced by the physics and rules transition. For training,
        prefer :meth:`step` unless these details are being sampled, aggregated,
        or streamed to the host. Retained bytes scale with control frames and
        physics substeps; the renderer-only capture path emits only its bounded
        target-rate physics samples rather than a full 80 Hz state trajectory.
        The receipt is privileged causal telemetry, never a player observation.

        An action keeps integer categories at ``[N]`` and continuous controls
        at ``[N, 8]``. Its batched form is categorical ``[B, N]`` plus
        continuous ``[B, N, 8]``. The returned event leaves begin with
        ``[decimation, ...]`` for one match and
        ``[B, decimation, ...]`` for a batch. Batched rollouts may use either
        one shared setup or a setup with the same leading batch axis.
        Batched execution always uses ``lax.map`` to keep compilation and CPU
        temporary memory bounded for the branch-heavy transition.
        """

        if not isinstance(action, IntentAction):
            raise TypeError("action must be IntentAction")

        action_rank = action.move.ndim
        expected_intent_rank = action_rank - 1
        if action.intent.ndim != expected_intent_rank:
            raise ValueError(
                "intent category rank must be one less than the move "
                f"rank, got {action.intent.shape} and {action.move.shape}"
            )
        if action_rank == 2:
            return self._step_with_events_single(rollout, setup, action, key)
        if action_rank != 3:
            raise ValueError(
                "action must be single-player-batched [N, ...] or "
                "match-batched [B, N, ...], got "
                f"move field shape {action.move.shape}"
            )
        if rollout.state.players.position.ndim != 3:
            raise ValueError("batched actions require a batched rollout")
        batch_size = action.move.shape[0]
        if rollout.state.players.position.shape[0] != batch_size:
            raise ValueError("action and rollout batch sizes must match")
        setup_position_rank = setup.second_half_positions.ndim
        setup_team_rank = setup.second_half_kickoff_team.ndim
        if setup_position_rank == 2 and setup_team_rank == 0:
            setup_axis = None
        elif setup_position_rank == 3 and setup_team_rank == 1:
            if setup.second_half_positions.shape[0] != batch_size:
                raise ValueError("action and setup batch sizes must match")
            setup_axis = 0
        else:
            raise ValueError("setup must be shared [N, 2] or batched [B, N, 2]")
        if setup_axis is None:
            return jax.lax.map(
                lambda item: self._step_with_events_single(
                    item[0], setup, item[1], item[2]
                ),
                (rollout, action, key),
            )
        return jax.lax.map(
            lambda item: self._step_with_events_single(*item),
            (rollout, setup, action, key),
        )

    def _step_with_events_single(
        self,
        rollout: Rollout,
        setup: MatchSetup,
        action: IntentAction,
        key: jax.Array,
        *,
        _render_fps: float | None = None,
    ) -> StepWithEventsResult | _StepWithEventsAndRenderSamplesResult:
        """Implement one eventful transition for direct or mapped use."""

        episode = step_episode(
            rollout.state,
            rollout.offside,
            action,
            key,
            sample_control_contest_overrides(self.timebase.decimation),
            setup,
            match=self.match,
            timebase=self.timebase,
            boundary_margin_m=self.boundary_margin_m,
            goalkeeper_holding=self.goalkeeper_holding,
            restart_timing=self.restart_timing,
            ball_geometry=self.ball,
            stadium=self.stadium,
            reach=self.reach,
            action_scale=self.action_scale,
            contact_timing=self.contact_timing,
            contest_config=self.contest,
            body_foul_config=self.body_foul,
            player_physics=self.player_physics,
            perception=self.perception,
            body=self.body,
            long_stamina=self.long_stamina,
            short_stamina=self.short_stamina,
            ball_physics=self.ball_physics,
            _collect_events=True,
            _render_fps=_render_fps,
        )
        frame = episode.frame
        fields = {
            "rollout": Rollout(state=frame.state, offside=frame.offside_state),
            "outcome": episode.outcome,
            "contact_attempted": frame.contact_attempted,
            "kick_applied": frame.kick_applied,
            "restart_opened": frame.restart_opened,
            "event_budget_exhausted": frame.event_budget_exhausted,
            "contest_override_valid": frame.contest_override_valid,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "done": episode.done,
            "terminal_frozen": episode.terminal_frozen,
            "halftime_reset": episode.halftime_reset,
            "action_trace": frame.action_trace,
            "action_receipt": frame.action_receipt,
            "events": frame.events,
        }
        if _render_fps is None:
            return StepWithEventsResult(**fields)
        return _StepWithEventsAndRenderSamplesResult(
            **fields,
            render_samples=Rollout(
                state=frame.render_state,
                offside=frame.render_offside_state,
            ),
        )

    def normalization_context(self) -> NormalizationContext:
        """Return immutable model scales derived from this environment."""

        return make_normalization_context(
            timebase=self.timebase,
            match=self.match,
            stadium=self.stadium,
            ball=self.ball,
            action_scale=self.action_scale,
            contact_timing=self.contact_timing,
            goalkeeper_holding=self.goalkeeper_holding,
            restart_timing=self.restart_timing,
            ball_physics=self.ball_physics,
            roster_sampling=self.roster_sampling,
            perception=self.perception,
        )

    @staticmethod
    def action_spec(action: IntentAction | None = None) -> TreeSpec:
        """Describe the hierarchical action without entering a transition."""

        if action is None:
            action = IntentAction.neutral(1)
        return _action_spec(action)

    @staticmethod
    def action_trace_spec(trace: ActionTrace) -> TreeSpec:
        """Describe the submitted-action part of eventful telemetry."""

        return _action_trace_spec(trace)

    @staticmethod
    def action_receipt_spec(receipt: ActionReceipt) -> TreeSpec:
        """Describe the causal action-receipt part of eventful telemetry."""

        return _action_receipt_spec(receipt)

    def player_observation_spec(self, observation: NormalizedObservation) -> TreeSpec:
        """Describe one normalized player view and its configured scales."""

        return _player_observation_spec(observation, self.normalization_context())

    def manager_observation_spec(
        self, observation: NormalizedManagerObservation
    ) -> TreeSpec:
        """Describe one normalized manager view and its configured scales."""

        return _manager_observation_spec(observation, self.normalization_context())

    def global_state_spec(self, view: NormalizedGlobalRollout) -> TreeSpec:
        """Describe one normalized global state and its configured scales."""

        return _global_state_spec(view, self.normalization_context())

    def flatten_observation(
        self, observation: NormalizedObservation
    ) -> tuple[FlatTree, TreeSpec]:
        """Flatten a normalized player view with this environment's scales."""

        return _flatten_observation(
            observation,
            context=self.normalization_context(),
        )

    def flatten_manager_observation(
        self, observation: NormalizedManagerObservation
    ) -> tuple[FlatTree, TreeSpec]:
        """Flatten a normalized manager view with this environment's scales."""

        return _flatten_manager_observation(
            observation,
            context=self.normalization_context(),
        )

    def flatten_global_state(
        self, view: NormalizedGlobalRollout
    ) -> tuple[FlatTree, TreeSpec]:
        """Flatten a normalized global view with this environment's scales."""

        return _flatten_global_state(
            view,
            context=self.normalization_context(),
        )

    @staticmethod
    def event_spec(events: FrameEvents) -> TreeSpec:
        """Describe exact event telemetry without changing rollout work."""

        return _event_spec(events)

    @staticmethod
    def schema_versions():
        """Return immutable public host schema names."""

        return _schema_versions()

    def observe_si(self, rollout: Rollout, observer_index: jax.Array) -> SIObservation:
        """Return the SI-unit view used by the built-in rule policy."""

        fulltime_tick, halftime_tick = self.match.clock_ticks(self.timebase)
        return observe(
            rollout.state,
            rollout.offside,
            observer_index,
            decimation=self.timebase.decimation,
            restart_delay_substeps=forced_release_delay_substeps(
                timebase=self.timebase, config=self.restart_timing
            ),
            goalkeeper_hold_limit_substeps=holding_limit_substeps(
                timebase=self.timebase, config=self.goalkeeper_holding
            ),
            halftime_tick=halftime_tick,
            fulltime_tick=fulltime_tick,
            halftime_enabled=self.match.halftime_enabled,
            perception=self.perception,
        )

    def observe(self, rollout: Rollout, observer_index: jax.Array) -> Observation:
        """Return one normalized, structured observer-relative model view."""

        return normalize_observation(
            self.observe_si(rollout, observer_index), self.normalization_context()
        )

    def restore_observation(self, observation: NormalizedObservation) -> SIObservation:
        """Recover the SI values encoded by one normalized observation."""

        if not isinstance(observation, NormalizedObservation):
            raise TypeError("observation must be NormalizedObservation")
        return denormalize_observation(observation, self.normalization_context())

    def restore_match_clock(
        self, clock: NormalizedMatchClock, *, valid: jax.Array = jnp.bool_(True)
    ) -> MatchClockTicks:
        """Recover exact SI tick facts from any normalized model clock."""

        if not isinstance(clock, NormalizedMatchClock):
            raise TypeError("clock must be NormalizedMatchClock")
        return denormalize_match_clock(clock, self.normalization_context(), valid=valid)

    def observe_all_si(self, rollout: Rollout) -> SIObservation:
        """Return all raw SI views for the observation-only rule policy."""

        fulltime_tick, halftime_tick = self.match.clock_ticks(self.timebase)
        return observe_all(
            rollout.state,
            rollout.offside,
            decimation=self.timebase.decimation,
            restart_delay_substeps=forced_release_delay_substeps(
                timebase=self.timebase, config=self.restart_timing
            ),
            goalkeeper_hold_limit_substeps=holding_limit_substeps(
                timebase=self.timebase, config=self.goalkeeper_holding
            ),
            halftime_tick=halftime_tick,
            fulltime_tick=fulltime_tick,
            halftime_enabled=self.match.halftime_enabled,
            perception=self.perception,
        )

    def observe_all(self, rollout: Rollout) -> Observation:
        """Return normalized views with one leading observer-slot axis."""

        return normalize_observation(
            self.observe_all_si(rollout), self.normalization_context()
        )

    def global_state_view(self, rollout: Rollout) -> NormalizedGlobalRollout:
        """Return the normalized global state for logging, centralized use, and restore."""

        return NormalizedGlobalRollout(
            state=normalize_global_state(rollout.state, self.normalization_context()),
            offside=rollout.offside,
        )

    def restore_global_state_view(self, view: NormalizedGlobalRollout) -> Rollout:
        """Reconstruct one SI rollout from a normalized global-state view."""

        if not isinstance(view, NormalizedGlobalRollout):
            raise TypeError("view must be NormalizedGlobalRollout")
        return Rollout(
            state=denormalize_global_state(view.state, self.normalization_context()),
            offside=view.offside,
        )

    def substitute(
        self,
        rollout: Rollout,
        request: SubstitutionRequest,
    ) -> SubstitutionStepResult:
        """Apply an instantaneous, already-completed Law 3 substitution.

        Named-bench, re-entry, opportunity/window, referee-permission, and
        substitute-dismissal policy remains the competition layer's duty.
        The incoming identity inherits the outgoing tactical slot; walk-off
        and halfway-line entry procedure is intentionally abstracted away.
        Refresh cached roster metadata exactly when ``applied`` is true.
        """

        fulltime_tick, _ = self.match.clock_ticks(self.timebase)
        bounded_request = request._replace(
            enabled=(
                request.enabled
                & request_within_roster_domain(request, self.roster_sampling)
            )
        )
        event = _apply_substitution(
            rollout.state,
            rollout.offside,
            bounded_request,
            fulltime_tick=fulltime_tick,
            minimum_team_players=self.match.minimum_team_players,
            body=self.body,
            stadium=self.stadium,
        )
        active_restart = (event.state.restart.kind > RK_NONE) & (
            event.state.restart.kind < RESTART_COUNT
        )
        reconciled_state, taker_repaired = reconcile_restart_after_roster_change(
            event.state,
            event.applied,
            body=self.body,
            stadium=self.stadium,
        )

        def project_restart(candidate: State) -> State:
            positioning = prepare_restart_positioning(
                candidate,
                stadium=self.stadium,
                ball=self.ball,
                body=self.body,
            )
            return candidate._replace(
                players=candidate.players._replace(
                    position=positioning.position,
                    body_forward=body_forward_from_angle(positioning.facing),
                    velocity=jnp.where(
                        positioning.forced[:, None],
                        jnp.zeros_like(candidate.players.velocity),
                        candidate.players.velocity,
                    ),
                ),
                restart_layout_ready=positioning.taker_ready,
            )

        next_state = jax.lax.cond(
            (event.applied | taker_repaired) & active_restart,
            project_restart,
            lambda candidate: candidate,
            reconciled_state,
        )
        return SubstitutionStepResult(
            rollout=Rollout(next_state, event.offside_state),
            applied=event.applied,
        )

    def initialize_management(
        self,
        rollout: Rollout,
        team_0_bench: Sequence[PlayerProfile],
        team_1_bench: Sequence[PlayerProfile],
        *,
        rules: ManagementRules | None = None,
        formation_layouts: np.ndarray | jax.Array | None = None,
        formation_probabilities: np.ndarray | jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> ManagementInitialization:
        """Register manager-only benches outside the recurrent rollout.

        Pass the same episode key used by ``reset`` so active and bench
        profiles are sampled once from separate deterministic streams."""

        return initialize_management(
            rollout.state,
            team_0_bench,
            team_1_bench,
            rules=ManagementRules() if rules is None else rules,
            body=self.body,
            stadium=self.stadium,
            formation_layouts=formation_layouts,
            formation_probabilities=formation_probabilities,
            roster_sampling=self.roster_sampling,
            sampling_key=(
                None
                if key is None or not self.roster_sampling.enabled
                else jax.random.fold_in(key, _BENCH_ROSTER_SAMPLING_STREAM)
            ),
        )

    @staticmethod
    def observe_manager_si(
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        team: jax.Array,
    ) -> SIManagerObservation:
        """Return an SI-unit manager view with only that team bench."""

        return observe_manager_si(rollout.state, squad, management, team)

    @staticmethod
    def observe_managers_si(
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
    ) -> SIManagerObservation:
        """Stack both private manager views on a leading team axis."""

        views = (
            observe_manager_si(rollout.state, squad, management, TEAM_0),
            observe_manager_si(rollout.state, squad, management, TEAM_1),
        )
        return jax.tree.map(lambda *values: jnp.stack(values), *views)

    def observe_manager(
        self,
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        team: jax.Array,
    ) -> ManagerObservation:
        """Return a normalized manager view with causal match-clock context."""

        raw = self.observe_manager_si(rollout, squad, management, team)
        return normalize_manager_observation(
            raw, rollout.state, squad, self.normalization_context()
        )

    def observe_managers(
        self,
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
    ) -> ManagerObservation:
        """Stack normalized private manager views for a joint controller."""

        views = (
            self.observe_manager(rollout, squad, management, TEAM_0),
            self.observe_manager(rollout, squad, management, TEAM_1),
        )
        return jax.tree.map(lambda *values: jnp.stack(values), *views)

    def restore_manager_observation(
        self,
        observation: NormalizedManagerObservation,
        squad: SquadSetup,
    ) -> SIManagerObservation:
        """Recover the SI manager view represented by normalized output."""

        if not isinstance(observation, NormalizedManagerObservation):
            raise TypeError("observation must be NormalizedManagerObservation")
        return denormalize_manager_observation(
            observation, squad, self.normalization_context()
        )

    def manager_substitute(
        self,
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        action: ManagerAction,
    ) -> ManagerStepResult:
        """Apply a registered-bench action under authoritative match limits."""

        fulltime_tick, _ = self.match.clock_ticks(self.timebase)
        team = jnp.asarray(action.team, dtype=jnp.int32)
        valid_team = (team == 0) | (team == 1)
        safe_team = jnp.clip(team, 0, 1)
        substitutions = ManagerSubstitutionCommand.empty(1)
        requested = jnp.asarray(action.enabled, dtype=jnp.bool_) & valid_team
        substitutions = substitutions._replace(
            requested=substitutions.requested.at[safe_team, 0].set(requested),
            outgoing_index=substitutions.outgoing_index.at[safe_team, 0].set(
                jnp.where(
                    requested,
                    jnp.asarray(action.outgoing_index, dtype=jnp.int32),
                    -1,
                )
            ),
            incoming_bench_index=(
                substitutions.incoming_bench_index.at[safe_team, 0].set(
                    jnp.where(
                        requested,
                        jnp.asarray(action.incoming_bench_index, dtype=jnp.int32),
                        -1,
                    )
                )
            ),
        )
        command = ManagerCommand.empty(1)._replace(substitutions=substitutions)
        result = apply_manager_command(
            rollout.state,
            rollout.offside,
            squad,
            management,
            command,
            fulltime_tick=fulltime_tick,
            minimum_team_players=self.match.minimum_team_players,
            body=self.body,
            stadium=self.stadium,
            ball=self.ball,
        )
        return ManagerStepResult(
            rollout=Rollout(result.state, result.offside),
            management=result.management,
            applied=jnp.where(
                valid_team, result.substitutions_applied[safe_team, 0], False
            ),
            reason=jnp.where(
                valid_team,
                result.substitution_reasons[safe_team, 0],
                jnp.int32(ManagerCommandReason.INVALID_SLOT),
            ),
            substitution_event=jax.tree.map(
                lambda value: value[safe_team, 0], result.substitution_events
            ),
            acting_goalkeepers_applied=result.acting_goalkeepers_applied,
            acting_goalkeeper_events=result.acting_goalkeeper_events,
            roster_metadata_changed=result.roster_metadata_changed,
        )

    def observe_player_tactics_si(
        self,
        rollout: Rollout,
        management: ManagerState,
        observer_index: jax.Array,
    ) -> SIPlayerTacticalObservation:
        """Return SI tactical anchors for the built-in rule policy."""

        return observe_player_tactics(rollout.state, management, observer_index)

    def observe_player_tactics(
        self,
        rollout: Rollout,
        management: ManagerState,
        observer_index: jax.Array,
    ) -> PlayerTacticalObservation:
        """Return normalized own-team tactical targets and roles."""

        return normalize_player_tactics(
            self.observe_player_tactics_si(rollout, management, observer_index),
            self.normalization_context(),
        )

    def restore_player_tactics(
        self, observation: NormalizedPlayerTacticalObservation
    ) -> SIPlayerTacticalObservation:
        """Recover SI tactical anchors from normalized player output."""

        if not isinstance(observation, NormalizedPlayerTacticalObservation):
            raise TypeError("observation must be NormalizedPlayerTacticalObservation")
        return denormalize_player_tactics(observation, self.normalization_context())

    @staticmethod
    def opening_formation_available(
        rollout: Rollout,
        management: ManagerState,
    ) -> jax.Array:
        """Return the teams still eligible for one physical opening layout."""

        return _opening_formation_available(rollout.state, management)

    def opening_formation_command(
        self,
        rollout: Rollout,
        setup: MatchSetup,
        squad: SquadSetup,
        management: ManagerState,
        command: ManagerFormationCommand,
        *,
        mirror_second_half: bool = True,
    ) -> OpeningFormationStepResult:
        """Apply registered player positions only before the opening kickoff."""

        result: OpeningFormationResult = apply_opening_formation(
            rollout.state,
            rollout.offside,
            setup,
            squad,
            management,
            command,
            stadium=self.stadium,
            ball=self.ball,
            body=self.body,
            mirror_second_half=mirror_second_half,
        )
        return OpeningFormationStepResult(
            rollout=Rollout(result.state, result.offside),
            setup=result.setup,
            management=result.management,
            applied=result.applied,
        )

    def manager_command(
        self,
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        command: ManagerCommand,
    ) -> ManagerCommandStepResult:
        """Apply substitutions, shape targets, and a visible restart taker."""

        fulltime_tick, _ = self.match.clock_ticks(self.timebase)
        result: ManagerCommandResult = apply_manager_command(
            rollout.state,
            rollout.offside,
            squad,
            management,
            command,
            fulltime_tick=fulltime_tick,
            minimum_team_players=self.match.minimum_team_players,
            body=self.body,
            stadium=self.stadium,
            ball=self.ball,
        )
        return ManagerCommandStepResult(
            rollout=Rollout(result.state, result.offside),
            management=result.management,
            substitutions_applied=result.substitutions_applied,
            substitution_reasons=result.substitution_reasons,
            substitution_events=result.substitution_events,
            formations_applied=result.formations_applied,
            formation_reasons=result.formation_reasons,
            acting_goalkeepers_applied=result.acting_goalkeepers_applied,
            acting_goalkeeper_reasons=result.acting_goalkeeper_reasons,
            acting_goalkeeper_events=result.acting_goalkeeper_events,
            set_piece_takers_applied=result.set_piece_takers_applied,
            set_piece_taker_reasons=result.set_piece_taker_reasons,
            roster_metadata_changed=result.roster_metadata_changed,
        )

    def emergency_goalkeeper_command(
        self,
        rollout: Rollout,
        squad: SquadSetup,
        management: ManagerState,
        *,
        max_simultaneous: int = 1,
    ) -> ManagerCommand:
        """Propose bench-GK recovery plus a field-player fallback."""

        return emergency_goalkeeper_command(
            rollout.state,
            squad,
            management,
            stadium=self.stadium,
            max_simultaneous=max_simultaneous,
        )

    @staticmethod
    def roster_metadata_si(
        rollout: Rollout,
        management: ManagerState | None = None,
    ) -> SIRosterMetadata:
        """Return SI profiles for the built-in rule policy."""

        generation = None if management is None else management.slot_generation
        return roster_metadata_from_state(rollout.state, slot_generation=generation)

    def roster_metadata(
        self,
        rollout: Rollout,
        management: ManagerState | None = None,
    ) -> NormalizedRosterMetadata:
        """Return cacheable model-ready profiles on the physical slot axis."""

        return normalize_roster_metadata(
            self.roster_metadata_si(rollout, management), self.normalization_context()
        )

    def restore_roster_metadata(
        self, roster: NormalizedRosterMetadata
    ) -> SIRosterMetadata:
        """Recover the SI roster metadata encoded by a normalized tree."""

        if not isinstance(roster, NormalizedRosterMetadata):
            raise TypeError("roster must be NormalizedRosterMetadata")
        return denormalize_roster_metadata(roster, self.normalization_context())

    def render_mp4(
        self,
        states: Any,
        output_dir: str | os.PathLike[str],
        *,
        video_name: str = "match.mp4",
        frame_events: Any = None,
        observations: Any = None,
        slot_generations: Any = None,
        substitution_events: Any = None,
        acting_goalkeeper_events: Any = None,
        metadata: Any = None,
        match_index: int = 0,
        fps: float = 20.0,
        every: int = 1,
        workers: int = 1,
        chunk_frames: int | None = None,
        style: Any = None,
        exact_actions: bool = False,
    ) -> Any:
        """Render one selected match with the optional replay renderer.

        This is an eager host-side convenience method. The local import keeps
        Matplotlib, imageio, and ffmpeg out of the environment import and JAX
        transition graphs. Use a separate ``output_dir`` for each render;
        replay sidecars have stable names within that directory.

        ``slot_generations`` supplies exact reusable-slot identity. When it is
        omitted, sidecars use ``-1`` for untracked generations and never
        infer substitutions from player-id changes. ``substitution_events``
        and ``acting_goalkeeper_events`` accept stacked event trees or
        per-frame sequences with ``None`` on frames without a manager command.
        Committed events are recorded only in ``event.json`` and are not drawn
        on the video. These host-side inputs do not alter ``step`` /
        ``step_with_events`` compilation or rollout speed.

        ``fps`` is the video sampling rate before ``every`` decimation.
        The encoder rate is derived from the retained and sampled frame counts,
        preserving playback duration even when the final decimation group is
        partial. Tracking and event sidecars retain every supplied
        source frame and are timestamped from ``control_tick`` instead.
        """

        from footballworld.rendering import RenderStyle, render_mp4

        if style is None:
            style = RenderStyle()
        return render_mp4(
            states,
            output_dir,
            video_name=video_name,
            env=self,
            frame_events=frame_events,
            observations=observations,
            slot_generations=slot_generations,
            substitution_events=substitution_events,
            acting_goalkeeper_events=acting_goalkeeper_events,
            metadata=metadata,
            match_index=match_index,
            fps=fps,
            every=every,
            workers=workers,
            chunk_frames=chunk_frames,
            style=style,
            exact_actions=exact_actions,
        )


__all__ = [
    "ActingGoalkeeperEvent",
    "FootballWorld",
    "ManagementInitialization",
    "ManagementRules",
    "ManagerActingGoalkeeperCommand",
    "ManagerAction",
    "ManagerCommand",
    "ManagerCommandReason",
    "ManagerCommandStepResult",
    "ManagerFormationCommand",
    "ManagerObservation",
    "ManagerSetPieceTakerCommand",
    "ManagerState",
    "ManagerStepResult",
    "ManagerSubstitutionCommand",
    "MatchClockTicks",
    "NormalizationContext",
    "NormalizedBallState",
    "NormalizedGlobalRollout",
    "NormalizedGlobalState",
    "NormalizedManagerObservation",
    "NormalizedMatchClock",
    "NormalizedObservation",
    "NormalizedPlayerTacticalObservation",
    "NormalizedRosterMetadata",
    "OpeningFormationStepResult",
    "PlayerObservation",
    "PlayerTacticalObservation",
    "ResetResult",
    "Rollout",
    "SIManagerObservation",
    "SIObservation",
    "SIPlayerTacticalObservation",
    "SIRollout",
    "SIRosterMetadata",
    "SquadSetup",
    "StepResult",
    "StepWithEventsResult",
    "SubstitutionEvent",
    "SubstitutionRequest",
    "SubstitutionStepResult",
    "emergency_goalkeeper_command",
]
