"""Manager-only squad information kept outside the player rollout state."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.management import ManagementRules
from footballworld.config.roster import PlayerProfile, player_profile_values_valid
from footballworld.config.roster_sampling import RosterSampling
from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_NONE,
    RK_PENALTY,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.state import (
    State,
    body_forward_from_angle,
    initial_player_body_forward,
)
from footballworld.environment.clock import regulation_elapsed_ticks
from footballworld.environment.episode import MatchSetup
from footballworld.environment.observation import _safe_subject_index
from footballworld.environment.roster_sampling import (
    ProfileValues,
    sample_profile_values,
)
from footballworld.environment.substitution import (
    SubstitutionRequest,
    _apply_substitution,
    _empty_contact,
    _management_stoppage_open,
)
from footballworld.environment.tactics import (
    ROLE_CENTRE_BACK,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    classify_formation_roles,
)
from footballworld.rules.offside import OffsideState
from footballworld.rules.restart import (
    repair_broken_restart_taker,
    select_restart_taker,
)
from footballworld.rules.restart_positioning import prepare_restart_positioning

MANAGER_OBSERVATION_SCHEMA_VERSION = 9
PLAYER_TACTICAL_OBSERVATION_SCHEMA_VERSION = 1


class ManagerCommandReason(IntEnum):
    """Stable per-axis adjudication codes for well-shaped manager commands.

    Values through ``UNCHANGED`` define the stable public ``CommandReason``
    wire meanings. Additional reasons cover the atomic multi-substitution
    transaction. Shape and dtype failures remain host API
    errors rather than device-side decision codes.
    """

    NOT_REQUESTED = 0
    APPLIED = 1
    ACCEPTED_PENDING = 2
    INTERNAL_FALLBACK = 3
    ACTION_MASKED = 4
    INVALID_SLOT = 5
    INACTIVE_PLAYER = 6
    BENCH_UNAVAILABLE = 7
    NOT_DEAD_BALL = 8
    SUBSTITUTION_WINDOW_EXHAUSTED = 9
    SUBSTITUTION_LIMIT_EXHAUSTED = 10
    GOALKEEPER_CONSTRAINT = 11
    INVALID_FORMATION = 12
    NO_MATCHING_RESTART = 13
    WRONG_TEAM = 14
    INELIGIBLE_TAKER = 15
    TERMINAL = 16
    ALREADY_CHANGED = 17
    PLACEMENT_FAILED = 18
    FORMATION_IN_TRANSITION = 19
    UNCHANGED = 20
    DUPLICATE_REQUEST = 21
    INVALID_PROFILE = 22
    IDENTITY_CONFLICT = 23
    ATOMIC_TEAM_REJECTED = 24


def _safe_axis_index(name: str, value, size: int):
    """Raise for eager bad addresses and make traced bad addresses no-ops."""

    traced = any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )
    if not traced:
        source = np.asarray(value)
        if source.shape != ():
            raise ValueError(f"{name} must be scalar")
        if not np.issubdtype(source.dtype, np.integer) or np.issubdtype(
            source.dtype, np.bool_
        ):
            raise TypeError(f"{name} must be a non-boolean integer")
        integer = int(source)
        if not 0 <= integer < size:
            raise ValueError(f"{name} must lie in [0, {size})")
        return jnp.int32(integer), jnp.bool_(True)
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    if not jnp.issubdtype(array.dtype, jnp.integer) or jnp.issubdtype(
        array.dtype, jnp.bool_
    ):
        raise TypeError(f"{name} must be a non-boolean integer")
    valid = (array >= 0) & (array < size)
    return jnp.clip(array, 0, size - 1).astype(jnp.int32), valid


def _set_if_valid(array, index, value, valid):
    current = array[index]
    return array.at[index].set(
        jnp.where(valid, jnp.asarray(value, dtype=array.dtype), current)
    )


class SquadSetup(NamedTuple):
    """Static registered bench metadata, padded only on the bench axis."""

    valid: jax.Array
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    max_substitutions: jax.Array
    max_windows: jax.Array
    formation_probabilities: jax.Array
    formation_layouts: jax.Array
    formation_roles: jax.Array


class ManagerState(NamedTuple):
    """Small dynamic management carry, separate from physics and rules."""

    available: jax.Array
    slot_generation: jax.Array
    substitutions_used: jax.Array
    windows_used: jax.Array
    last_window_restart_tick: jax.Array
    formation_index: jax.Array
    formation_anchor: jax.Array
    formation_role: jax.Array
    opening_formation_committed: jax.Array


class ManagementInitialization(NamedTuple):
    squad: SquadSetup
    state: ManagerState


class ManagerAction(NamedTuple):
    """Select an on-pitch slot and an authoritative registered substitute."""

    enabled: jax.Array
    team: jax.Array
    outgoing_index: jax.Array
    incoming_bench_index: jax.Array


class ManagerSubstitutionCommand(NamedTuple):
    """Fixed-width, per-team substitution proposals for one stoppage."""

    requested: jax.Array
    outgoing_index: jax.Array
    incoming_bench_index: jax.Array

    @classmethod
    def empty(cls, max_simultaneous: int) -> ManagerSubstitutionCommand:
        if (
            not isinstance(max_simultaneous, int)
            or isinstance(max_simultaneous, bool)
            or max_simultaneous < 0
        ):
            raise ValueError("max_simultaneous must be a non-negative integer")
        shape = (2, max_simultaneous)
        return cls(
            requested=jnp.zeros(shape, dtype=jnp.bool_),
            outgoing_index=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            incoming_bench_index=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
        )

    def with_request(
        self,
        team: int | jax.Array,
        command_index: int | jax.Array,
        *,
        outgoing_index: int | jax.Array,
        incoming_bench_index: int | jax.Array,
    ) -> ManagerSubstitutionCommand:
        if self.requested.shape[1] == 0:
            traced = any(
                isinstance(leaf, jax.core.Tracer)
                for leaf in jax.tree_util.tree_leaves(command_index)
            )
            if traced:
                return self
            raise ValueError("cannot address an empty substitution command")
        team, team_valid = _safe_axis_index("team", team, 2)
        cell, cell_valid = _safe_axis_index(
            "command_index", command_index, self.requested.shape[1]
        )
        valid = team_valid & cell_valid
        index = (team, cell)
        return self._replace(
            requested=_set_if_valid(self.requested, index, True, valid),
            outgoing_index=_set_if_valid(
                self.outgoing_index, index, outgoing_index, valid
            ),
            incoming_bench_index=_set_if_valid(
                self.incoming_bench_index,
                index,
                incoming_bench_index,
                valid,
            ),
        )


class ManagerFormationCommand(NamedTuple):
    """Per-team request selecting a registered tactical-anchor layout."""

    requested: jax.Array
    layout_index: jax.Array

    @classmethod
    def empty(cls) -> ManagerFormationCommand:
        return cls(
            requested=jnp.zeros(2, dtype=jnp.bool_),
            layout_index=jnp.full(2, -1, dtype=jnp.int32),
        )

    def with_request(
        self, team: int | jax.Array, *, layout_index: int | jax.Array
    ) -> ManagerFormationCommand:
        team, valid = _safe_axis_index("team", team, 2)
        return self._replace(
            requested=_set_if_valid(self.requested, team, True, valid),
            layout_index=_set_if_valid(self.layout_index, team, layout_index, valid),
        )


class ManagerActingGoalkeeperCommand(NamedTuple):
    """Per-team fallback goalkeeper role selected during one stoppage."""

    requested: jax.Array
    player_slot: jax.Array

    @classmethod
    def empty(cls) -> ManagerActingGoalkeeperCommand:
        return cls(
            requested=jnp.zeros(2, dtype=jnp.bool_),
            player_slot=jnp.full(2, NO_PLAYER, dtype=jnp.int32),
        )

    def with_request(
        self, team: int | jax.Array, *, player_slot: int | jax.Array
    ) -> ManagerActingGoalkeeperCommand:
        team, valid = _safe_axis_index("team", team, 2)
        return self._replace(
            requested=_set_if_valid(self.requested, team, True, valid),
            player_slot=_set_if_valid(self.player_slot, team, player_slot, valid),
        )


class ManagerSetPieceTakerCommand(NamedTuple):
    """Per-team, per-restart taker requests for the currently open restart."""

    requested: jax.Array
    player_slot: jax.Array

    @classmethod
    def empty(cls) -> ManagerSetPieceTakerCommand:
        shape = (2, RESTART_COUNT)
        return cls(
            requested=jnp.zeros(shape, dtype=jnp.bool_),
            player_slot=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
        )

    def with_request(
        self,
        team: int | jax.Array,
        restart_kind: int | jax.Array,
        *,
        player_slot: int | jax.Array,
    ) -> ManagerSetPieceTakerCommand:
        team, team_valid = _safe_axis_index("team", team, 2)
        kind, kind_valid = _safe_axis_index("restart_kind", restart_kind, RESTART_COUNT)
        valid = team_valid & kind_valid
        index = (team, kind)
        return self._replace(
            requested=_set_if_valid(self.requested, index, True, valid),
            player_slot=_set_if_valid(self.player_slot, index, player_slot, valid),
        )


class ManagerCommand(NamedTuple):
    """One low-frequency, fixed-shape management transaction."""

    substitutions: ManagerSubstitutionCommand
    formations: ManagerFormationCommand
    acting_goalkeepers: ManagerActingGoalkeeperCommand
    set_piece_takers: ManagerSetPieceTakerCommand

    @classmethod
    def empty(cls, max_simultaneous: int) -> ManagerCommand:
        return cls(
            substitutions=ManagerSubstitutionCommand.empty(max_simultaneous),
            formations=ManagerFormationCommand.empty(),
            acting_goalkeepers=ManagerActingGoalkeeperCommand.empty(),
            set_piece_takers=ManagerSetPieceTakerCommand.empty(),
        )


class ManagerOnFieldObservation(NamedTuple):
    team_mask: jax.Array
    active: jax.Array
    player_id: jax.Array
    slot_generation: jax.Array
    position: jax.Array
    velocity: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    yellow_cards: jax.Array
    sent_off: jax.Array


class ManagerBenchObservation(NamedTuple):
    valid: jax.Array
    available: jax.Array
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


class ManagerObservation(NamedTuple):
    """Low-frequency manager view; never a player-policy input."""

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


class PlayerTacticalObservation(NamedTuple):
    """A player's own-team tactical targets, separate from physical state."""

    valid: jax.Array
    team: jax.Array
    formation_index: jax.Array
    formation_anchor: jax.Array
    formation_role: jax.Array


class SubstitutionEvent(NamedTuple):
    """One committed identity boundary in a reusable on-pitch slot."""

    occurred: jax.Array
    team: jax.Array
    player_slot: jax.Array
    outgoing_player_id: jax.Array
    incoming_player_id: jax.Array
    slot_generation: jax.Array
    control_tick: jax.Array

    @classmethod
    def empty(cls, shape: tuple[int, ...] = ()) -> SubstitutionEvent:
        """Return fixed-shape sentinels for an absent or rolled-back event."""

        return cls(
            occurred=jnp.zeros(shape, dtype=jnp.bool_),
            team=jnp.full(shape, NO_TEAM, dtype=jnp.int32),
            player_slot=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            outgoing_player_id=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            incoming_player_id=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            slot_generation=jnp.full(shape, -1, dtype=jnp.int32),
            control_tick=jnp.full(shape, -1, dtype=jnp.int32),
        )


class ActingGoalkeeperEvent(NamedTuple):
    """One role reassignment that does not change player identity."""

    occurred: jax.Array
    team: jax.Array
    player_slot: jax.Array
    player_id: jax.Array
    slot_generation: jax.Array
    environment_forced: jax.Array
    control_tick: jax.Array

    @classmethod
    def empty(cls, shape: tuple[int, ...] = ()) -> ActingGoalkeeperEvent:
        return cls(
            occurred=jnp.zeros(shape, dtype=jnp.bool_),
            team=jnp.full(shape, NO_TEAM, dtype=jnp.int32),
            player_slot=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            player_id=jnp.full(shape, NO_PLAYER, dtype=jnp.int32),
            slot_generation=jnp.full(shape, -1, dtype=jnp.int32),
            environment_forced=jnp.zeros(shape, dtype=jnp.bool_),
            control_tick=jnp.full(shape, -1, dtype=jnp.int32),
        )


class ManagerSubstitutionResult(NamedTuple):
    state: State
    offside: OffsideState
    management: ManagerState
    applied: jax.Array
    substitution_event: SubstitutionEvent


class ManagerCommandResult(NamedTuple):
    """Joint result with team-atomic masks and per-axis decision reasons."""

    state: State
    offside: OffsideState
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


class OpeningFormationResult(NamedTuple):
    """One start-only formation transaction and its updated half-time setup."""

    state: State
    offside: OffsideState
    setup: MatchSetup
    management: ManagerState
    applied: jax.Array


def reconcile_restart_after_roster_change(
    state: State,
    changed: jax.Array,
    *,
    body: BodyContact = BodyContact(),
    stadium: Stadium = Stadium(),
) -> tuple[State, jax.Array]:
    """Repair restart ownership only after a committed identity/role change.

    A pending taker is repaired inside the same public substitution boundary.
    The transaction also synchronizes explicit goalkeeper-hold ball and
    possession ownership. The
    outer gate preserves fail-closed no-op semantics for rejected commands.
    Restart positioning remains the caller's low-frequency postcondition so
    this helper never enters the player-control physics graph.
    """

    changed = jnp.asarray(changed, dtype=jnp.bool_)
    return jax.lax.cond(
        changed,
        lambda current: repair_broken_restart_taker(
            current,
            body=body,
            stadium=stadium,
        ),
        lambda current: (current, jnp.bool_(False)),
        state,
    )


def emergency_goalkeeper_command(
    state: State,
    squad: SquadSetup,
    management: ManagerState,
    *,
    stadium: Stadium,
    max_simultaneous: int = 1,
) -> ManagerCommand:
    """Propose bench-GK recovery with an acting-player fallback.

    The proposal removes the most advanced active outfielder when an available
    registered goalkeeper and substitution resource exist. Independently, it
    nominates the active outfielder nearest the own goal, so the authoritative
    transaction still has a legal fallback if the substitution is rejected.
    """

    command = ManagerCommand.empty(max_simultaneous)
    players = state.players
    substitution = command.substitutions
    acting = command.acting_goalkeepers
    for team in (TEAM_0, TEAM_1):
        active_team = players.active & (players.team_id == team)
        missing = ~jnp.any(active_team & players.is_goalkeeper)
        field = active_team & (~players.is_goalkeeper)
        has_field = jnp.any(field)
        attacking_x = players.position[:, 0] * state.attack_direction[jnp.int32(team)]
        outgoing = jnp.argmax(jnp.where(field, attacking_x, -jnp.inf)).astype(jnp.int32)

        own_goal = jnp.asarray(
            [
                -state.attack_direction[jnp.int32(team)] * stadium.half_length,
                0.0,
            ],
            dtype=players.position.dtype,
        )
        distance_squared = jnp.sum((players.position - own_goal) ** 2, axis=-1)
        acting_slot = jnp.argmin(jnp.where(field, distance_squared, jnp.inf)).astype(
            jnp.int32
        )
        acting = acting._replace(
            requested=acting.requested.at[team].set(missing & has_field),
            player_slot=acting.player_slot.at[team].set(
                jnp.where(missing & has_field, acting_slot, jnp.int32(NO_PLAYER))
            ),
        )

        if max_simultaneous > 0 and squad.valid.shape[1] > 0:
            bench_goalkeeper = (
                squad.valid[team]
                & management.available[team]
                & squad.is_goalkeeper[team]
            )
            has_bench_goalkeeper = jnp.any(bench_goalkeeper)
            incoming = jnp.argmax(bench_goalkeeper.astype(jnp.int32)).astype(jnp.int32)
            has_resource = management.substitutions_used[team] < squad.max_substitutions
            request = missing & has_field & has_bench_goalkeeper & has_resource
            substitution = substitution._replace(
                requested=substitution.requested.at[team, 0].set(request),
                outgoing_index=substitution.outgoing_index.at[team, 0].set(
                    jnp.where(request, outgoing, jnp.int32(NO_PLAYER))
                ),
                incoming_bench_index=(
                    substitution.incoming_bench_index.at[team, 0].set(
                        jnp.where(request, incoming, jnp.int32(NO_PLAYER))
                    )
                ),
            )

    return command._replace(
        substitutions=substitution,
        acting_goalkeepers=acting,
    )


def _validate_profile(
    profile: PlayerProfile, body: BodyContact, roster_sampling: RosterSampling
) -> None:
    if not isinstance(profile.player_id, numbers.Integral) or isinstance(
        profile.player_id, (bool, np.bool_)
    ):
        raise TypeError("bench player_id must be a non-boolean integer")
    if not 0 <= int(profile.player_id) <= np.iinfo(np.int32).max:
        raise ValueError("bench player_id must lie in the int32 identity domain")
    values = (
        profile.max_speed_mps,
        profile.height_m,
        profile.max_reach_height_m,
        profile.ball_control,
        profile.endurance_factor,
    )
    if not all(
        isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(
            "bench physical values must be finite non-boolean real numbers"
        )
    valid = player_profile_values_valid(
        profile.player_id,
        *values,
        head_radius=body.head_radius_m,
    )
    if not bool(valid):
        raise ValueError(f"invalid bench profile for player_id={profile.player_id}")
    reach_margin = profile.max_reach_height_m - profile.height_m
    normalized_domain = (
        roster_sampling.min_max_speed_mps
        <= profile.max_speed_mps
        <= roster_sampling.max_max_speed_mps
        and roster_sampling.min_height_m
        <= profile.height_m
        <= roster_sampling.max_height_m
        and roster_sampling.min_reach_margin_m
        <= reach_margin
        <= roster_sampling.max_reach_margin_m
        and roster_sampling.min_ball_control
        <= profile.ball_control
        <= roster_sampling.max_ball_control
        and roster_sampling.min_endurance_factor
        <= profile.endurance_factor
        <= roster_sampling.max_endurance_factor
    )
    if not normalized_domain:
        raise ValueError(
            f"bench profile exceeds the configured model-normalization domain: "
            f"player_id={profile.player_id}"
        )
    if not isinstance(profile.is_goalkeeper, bool):
        raise TypeError("bench is_goalkeeper must be bool")


def initialize_management(
    state: State,
    team_0_bench: Sequence[PlayerProfile],
    team_1_bench: Sequence[PlayerProfile],
    *,
    rules: ManagementRules | None = None,
    body: BodyContact | None = None,
    stadium: Stadium | None = None,
    formation_layouts: np.ndarray | jax.Array | None = None,
    formation_probabilities: np.ndarray | jax.Array | None = None,
    roster_sampling: RosterSampling = RosterSampling(),
    sampling_key: jax.Array | None = None,
) -> ManagementInitialization:
    """Register benches and fixed attacking-frame tactical layouts.

    Layout zero is always the kickoff shape reconstructed from the physical
    state. ``formation_layouts`` contains only additional ``[L, N, 2]``
    alternatives. ``formation_probabilities`` is an optional ``[L+1]`` or
    ``[2, L+1]`` rule-policy prior over the complete catalog. Omitting it
    assigns all mass to layout zero, preserving the caller's authored shape.
    Learned policies may still choose any registered layout.
    """

    rules = ManagementRules() if rules is None else rules
    body = BodyContact() if body is None else body
    stadium = Stadium() if stadium is None else stadium
    benches = (tuple(team_0_bench), tuple(team_1_bench))
    for bench in benches:
        for profile in bench:
            if type(profile) is not PlayerProfile:
                raise TypeError("bench entries must be exactly PlayerProfile")
            _validate_profile(profile, body, roster_sampling)
    on_field_ids = set(np.asarray(state.players.player_id, dtype=np.int64).tolist())
    bench_ids = [profile.player_id for bench in benches for profile in bench]
    if len(set(bench_ids)) != len(bench_ids):
        raise ValueError("bench player_id values must be unique across both teams")
    if on_field_ids.intersection(bench_ids):
        raise ValueError("bench player_id values must not duplicate on-pitch players")

    width = max(len(benches[0]), len(benches[1]))
    valid = np.zeros((2, width), dtype=np.bool_)
    player_id = np.full((2, width), NO_PLAYER, dtype=np.int32)
    is_goalkeeper = np.zeros((2, width), dtype=np.bool_)
    physical = np.zeros((2, width, 5), dtype=np.float32)
    for team, bench in enumerate(benches):
        for index, profile in enumerate(bench):
            valid[team, index] = True
            player_id[team, index] = profile.player_id
            is_goalkeeper[team, index] = profile.is_goalkeeper
            physical[team, index] = (
                profile.max_speed_mps,
                profile.height_m,
                profile.max_reach_height_m,
                profile.ball_control,
                profile.endurance_factor,
            )

    if sampling_key is not None and roster_sampling.enabled:
        sampled = sample_profile_values(
            ProfileValues(
                max_speed=physical[..., 0],
                height=physical[..., 1],
                reach_height=physical[..., 2],
                ball_control=physical[..., 3],
                endurance_factor=physical[..., 4],
            ),
            sampling_key,
            roster_sampling,
            valid=valid,
            minimum_height_m=float(
                np.nextafter(
                    np.float32(2.0 * body.head_radius_m),
                    np.float32(np.inf),
                )
            ),
        )
        physical = jnp.stack(sampled, axis=-1)

    player_count = state.players.position.shape[0]
    state_position = np.asarray(state.players.position, dtype=np.float32)
    team_id = np.asarray(state.players.team_id, dtype=np.int32)
    attack_direction = np.asarray(state.attack_direction, dtype=np.float32)
    kickoff_anchor = state_position * attack_direction[team_id, None]
    if formation_layouts is None:
        alternatives = np.empty((0, player_count, 2), dtype=np.float32)
    else:
        alternatives = np.asarray(formation_layouts, dtype=np.float32)
        if alternatives.ndim == 2:
            alternatives = alternatives[None, ...]
        if alternatives.ndim != 3 or alternatives.shape[1:] != (player_count, 2):
            raise ValueError("formation_layouts must have shape [L, player_count, 2]")
        if not np.all(np.isfinite(alternatives)):
            raise ValueError("formation_layouts must be finite")
        if np.any(np.abs(alternatives[..., 0]) > stadium.half_length):
            raise ValueError("formation layout x coordinates must lie on the pitch")
        if np.any(np.abs(alternatives[..., 1]) > stadium.half_width):
            raise ValueError("formation layout y coordinates must lie on the pitch")
    layouts = np.concatenate((kickoff_anchor[None, ...], alternatives), axis=0)
    layout_count = layouts.shape[0]
    if formation_probabilities is None:
        probabilities = np.zeros((2, layout_count), dtype=np.float32)
        probabilities[:, 0] = 1.0
    else:
        probabilities = np.asarray(formation_probabilities, dtype=np.float32)
        if probabilities.ndim == 1:
            probabilities = np.broadcast_to(
                probabilities[None, :], (2, probabilities.shape[0])
            ).copy()
        if probabilities.shape != (2, layout_count):
            raise ValueError(
                "formation_probabilities must have shape [L+1] or [2, L+1]"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("formation_probabilities must be finite and non-negative")
        total = probabilities.sum(axis=1, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError(
                "formation_probabilities must have positive mass for each team"
            )
        probabilities = probabilities / total
    layout_roles = jax.vmap(
        lambda anchor: classify_formation_roles(
            anchor, state.players.team_id, state.players.is_goalkeeper
        )
    )(jnp.asarray(layouts, dtype=jnp.float32))

    squad = SquadSetup(
        valid=jnp.asarray(valid),
        player_id=jnp.asarray(player_id),
        is_goalkeeper=jnp.asarray(is_goalkeeper),
        max_speed=jnp.asarray(physical[..., 0]),
        height=jnp.asarray(physical[..., 1]),
        reach_height=jnp.asarray(physical[..., 2]),
        ball_control=jnp.asarray(physical[..., 3]),
        endurance_factor=jnp.asarray(physical[..., 4]),
        max_substitutions=jnp.int32(rules.max_substitutions_per_team),
        max_windows=jnp.int32(rules.max_windows_per_team),
        formation_layouts=jnp.asarray(layouts, dtype=jnp.float32),
        formation_roles=layout_roles,
        formation_probabilities=jnp.asarray(probabilities, dtype=jnp.float32),
    )
    return ManagementInitialization(
        squad=squad,
        state=ManagerState(
            available=squad.valid,
            slot_generation=jnp.zeros(player_count, dtype=jnp.int32),
            substitutions_used=jnp.zeros(2, dtype=jnp.int32),
            windows_used=jnp.zeros(2, dtype=jnp.int32),
            last_window_restart_tick=jnp.full(2, -1, dtype=jnp.int32),
            formation_index=jnp.zeros(2, dtype=jnp.int32),
            formation_anchor=jnp.asarray(kickoff_anchor, dtype=jnp.float32),
            formation_role=layout_roles[0],
            opening_formation_committed=jnp.zeros(2, dtype=jnp.bool_),
        ),
    )


def _formation_candidate_signature(
    layout: jax.Array,
    role: jax.Array,
) -> jax.Array:
    """Identity-like catalog key used only for permutation-stable randomness."""

    quantized = jnp.rint(layout * jnp.float32(10_000.0)).astype(jnp.int32)
    values = jnp.concatenate((quantized.reshape(-1), role.astype(jnp.int32)))

    def combine(index, value):
        return (value ^ values[index].astype(jnp.uint32)) * jnp.uint32(16_777_619)

    return jax.lax.fori_loop(
        0, values.shape[0], combine, jnp.uint32(2_166_136_261)
    ).astype(jnp.int32)


def observe_manager(
    state: State,
    squad: SquadSetup,
    management: ManagerState,
    team: jax.Array,
) -> ManagerObservation:
    """Build one manager view without exposing the opposing registered bench."""

    safe_team, valid_team = _safe_subject_index(team, 2)
    team_mask = (state.players.team_id == safe_team) & valid_team
    float_mask = team_mask[:, None]
    bench_valid = squad.valid[safe_team] & valid_team
    layout_count = squad.formation_layouts.shape[0]
    layout_team = state.players.team_id[None, :] == safe_team
    layout_team = jnp.broadcast_to(layout_team, squad.formation_roles.shape)
    layout_outfield = layout_team & (squad.formation_roles != ROLE_GOALKEEPER)
    layout_count_outfield = jnp.maximum(jnp.sum(layout_outfield, axis=1), 1)
    candidate_attack_depth = (
        jnp.sum(
            jnp.where(layout_outfield, squad.formation_layouts[..., 0], 0.0), axis=1
        )
        / layout_count_outfield
    )
    candidate_width = jnp.max(
        jnp.where(
            layout_team,
            jnp.abs(squad.formation_layouts[..., 1]),
            0.0,
        ),
        axis=1,
    )
    candidate_defender_fraction = jnp.sum(
        layout_team
        & (
            (squad.formation_roles == ROLE_CENTRE_BACK)
            | (squad.formation_roles == ROLE_FULL_BACK)
        ),
        axis=1,
    ).astype(jnp.float32) / layout_count_outfield.astype(jnp.float32)
    candidate_valid = jnp.full((layout_count,), valid_team, dtype=jnp.bool_)
    return ManagerObservation(
        valid=valid_team,
        team=jnp.where(valid_team, safe_team, NO_TEAM),
        on_field=ManagerOnFieldObservation(
            team_mask=team_mask,
            active=state.players.active & team_mask,
            player_id=jnp.where(team_mask, state.players.player_id, NO_PLAYER),
            slot_generation=jnp.where(
                team_mask, management.slot_generation, jnp.int32(-1)
            ),
            position=jnp.where(float_mask, state.players.position, 0.0),
            velocity=jnp.where(float_mask, state.players.velocity, 0.0),
            is_goalkeeper=state.players.is_goalkeeper & team_mask,
            max_speed=jnp.where(team_mask, state.players.max_speed, 0.0),
            height=jnp.where(team_mask, state.players.height, 0.0),
            reach_height=jnp.where(team_mask, state.players.reach_height, 0.0),
            ball_control=jnp.where(team_mask, state.players.ball_control, 0.0),
            endurance_factor=jnp.where(team_mask, state.players.endurance_factor, 0.0),
            stamina_long=jnp.where(team_mask, state.players.stamina_long, 0.0),
            stamina_short=jnp.where(team_mask, state.players.stamina_short, 0.0),
            yellow_cards=jnp.where(team_mask, state.players.yellow_cards, 0),
            sent_off=state.players.sent_off & team_mask,
        ),
        bench=ManagerBenchObservation(
            valid=bench_valid,
            available=management.available[safe_team] & bench_valid,
            player_id=jnp.where(bench_valid, squad.player_id[safe_team], NO_PLAYER),
            is_goalkeeper=squad.is_goalkeeper[safe_team] & bench_valid,
            max_speed=jnp.where(bench_valid, squad.max_speed[safe_team], 0.0),
            height=jnp.where(bench_valid, squad.height[safe_team], 0.0),
            reach_height=jnp.where(bench_valid, squad.reach_height[safe_team], 0.0),
            ball_control=jnp.where(bench_valid, squad.ball_control[safe_team], 0.0),
            endurance_factor=jnp.where(
                bench_valid, squad.endurance_factor[safe_team], 0.0
            ),
        ),
        substitutions_remaining=jnp.where(
            valid_team,
            jnp.maximum(
                squad.max_substitutions - management.substitutions_used[safe_team], 0
            ),
            0,
        ),
        substitutions_max=jnp.where(valid_team, squad.max_substitutions, 0),
        windows_remaining=jnp.where(
            valid_team,
            jnp.maximum(squad.max_windows - management.windows_used[safe_team], 0),
            0,
        ),
        windows_max=jnp.where(valid_team, squad.max_windows, 0),
        score=jnp.where(valid_team, state.score, 0),
        control_tick=jnp.where(valid_team, state.control_tick, 0),
        restart_kind=jnp.where(valid_team, state.restart.kind, 0),
        restart_team=jnp.where(valid_team, state.restart.team, NO_TEAM),
        restart_position=jnp.where(
            valid_team & (state.restart.kind != RK_NONE),
            state.ball.position[:2],
            jnp.zeros(2, dtype=state.ball.position.dtype),
        ),
        attack_direction=jnp.where(
            valid_team,
            state.attack_direction[safe_team],
            jnp.asarray(0.0, dtype=state.attack_direction.dtype),
        ),
        restart_opened_control_tick=jnp.where(
            valid_team,
            state.restart.opened_control_tick,
            jnp.int32(-1),
        ),
        formation_index=jnp.where(valid_team, management.formation_index[safe_team], 0),
        formation_anchor=jnp.where(
            team_mask[:, None], management.formation_anchor, 0.0
        ),
        formation_role=jnp.where(team_mask, management.formation_role, -1),
        formation_candidate_valid=candidate_valid,
        formation_candidate_signature=jnp.where(
            candidate_valid,
            jax.vmap(_formation_candidate_signature)(
                jnp.where(layout_team[..., None], squad.formation_layouts, 0.0),
                jnp.where(layout_team, squad.formation_roles, jnp.int32(-1)),
            ),
            jnp.int32(0),
        ),
        formation_candidate_probability=jnp.where(
            candidate_valid,
            squad.formation_probabilities[safe_team],
            0.0,
        ),
        formation_candidate_attack_depth=jnp.where(
            candidate_valid, candidate_attack_depth, 0.0
        ),
        formation_candidate_width=jnp.where(candidate_valid, candidate_width, 0.0),
        formation_candidate_defender_fraction=jnp.where(
            candidate_valid, candidate_defender_fraction, 0.0
        ),
    )


def observe_player_tactics(
    state: State,
    management: ManagerState,
    observer_index: jax.Array,
) -> PlayerTacticalObservation:
    """Expose only the observer's team targets; opponent tactics stay masked."""

    player_count = state.players.position.shape[0]
    safe_observer, valid = _safe_subject_index(observer_index, player_count)
    team = state.players.team_id[safe_observer]
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    own = (state.players.team_id == team) & valid
    return PlayerTacticalObservation(
        valid=valid,
        team=jnp.where(valid, team, NO_TEAM),
        formation_index=jnp.where(valid, management.formation_index[safe_team], 0),
        formation_anchor=jnp.where(own[:, None], management.formation_anchor, 0.0),
        formation_role=jnp.where(own, management.formation_role, -1),
    )


def apply_manager_substitution(
    state: State,
    offside: OffsideState,
    squad: SquadSetup,
    management: ManagerState,
    action: ManagerAction,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    body: BodyContact,
    stadium: Stadium,
    ball: Ball = Ball(),
) -> ManagerSubstitutionResult:
    """Apply one registered substitution and account for its opportunity."""

    if squad.valid.shape[1] == 0:
        return ManagerSubstitutionResult(
            state=state,
            offside=offside,
            management=management,
            applied=jnp.bool_(False),
            substitution_event=SubstitutionEvent.empty(),
        )

    team = jnp.asarray(action.team, dtype=jnp.int32)
    outgoing = jnp.asarray(action.outgoing_index, dtype=jnp.int32)
    incoming = jnp.asarray(action.incoming_bench_index, dtype=jnp.int32)
    enabled = jnp.asarray(action.enabled, dtype=jnp.bool_)
    player_count = state.players.player_id.shape[0]
    safe_outgoing = jnp.clip(outgoing, 0, player_count - 1)
    team_valid = (team == TEAM_0) | (team == TEAM_1)
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    bench_width = squad.valid.shape[1]
    incoming_valid = (incoming >= 0) & (incoming < bench_width)
    safe_incoming = jnp.clip(incoming, 0, bench_width - 1)
    registered = (
        team_valid
        & incoming_valid
        & squad.valid[safe_team, safe_incoming]
        & management.available[safe_team, safe_incoming]
    )
    same_window = (state.restart.opened_control_tick >= 0) & (
        management.last_window_restart_tick[safe_team]
        == state.restart.opened_control_tick
    )
    halftime_window = (
        (state.first_half_wall_end_tick >= 0)
        & (state.control_tick == state.first_half_wall_end_tick)
        & (state.restart.kind == RK_KICKOFF)
        & (~state.ball.live)
    )
    within_limits = (
        management.substitutions_used[safe_team] < squad.max_substitutions
    ) & (
        halftime_window
        | same_window
        | (management.windows_used[safe_team] < squad.max_windows)
    )
    authorized = enabled & registered & within_limits
    request = SubstitutionRequest(
        enabled=authorized,
        team=team,
        outgoing_index=outgoing,
        incoming_player_id=squad.player_id[safe_team, safe_incoming],
        incoming_is_goalkeeper=squad.is_goalkeeper[safe_team, safe_incoming],
        incoming_max_speed=squad.max_speed[safe_team, safe_incoming],
        incoming_height=squad.height[safe_team, safe_incoming],
        incoming_reach_height=squad.reach_height[safe_team, safe_incoming],
        incoming_ball_control=squad.ball_control[safe_team, safe_incoming],
        incoming_endurance_factor=squad.endurance_factor[safe_team, safe_incoming],
        incoming_yellow_cards=jnp.int32(0),
    )
    event = _apply_substitution(
        state,
        offside,
        request,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
        body=body,
        stadium=stadium,
    )
    applied = event.applied
    reconciled_state, taker_repaired = reconcile_restart_after_roster_change(
        event.state,
        applied,
        body=body,
        stadium=stadium,
    )
    active_restart = (reconciled_state.restart.kind > RK_NONE) & (
        reconciled_state.restart.kind < RESTART_COUNT
    )

    def project_restart(candidate: State) -> State:
        positioning = prepare_restart_positioning(
            candidate,
            stadium=stadium,
            ball=ball,
            body=body,
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
        (applied | taker_repaired) & active_restart,
        project_restart,
        lambda candidate: candidate,
        reconciled_state,
    )
    next_generation = management.slot_generation.at[safe_outgoing].set(
        management.slot_generation[safe_outgoing] + applied.astype(jnp.int32)
    )
    substitution_event = SubstitutionEvent(
        occurred=applied,
        team=jnp.where(applied, team, jnp.int32(NO_TEAM)),
        player_slot=jnp.where(applied, outgoing, jnp.int32(NO_PLAYER)),
        outgoing_player_id=jnp.where(
            applied, state.players.player_id[safe_outgoing], jnp.int32(NO_PLAYER)
        ),
        incoming_player_id=jnp.where(
            applied,
            squad.player_id[safe_team, safe_incoming],
            jnp.int32(NO_PLAYER),
        ),
        slot_generation=jnp.where(
            applied, next_generation[safe_outgoing], jnp.int32(-1)
        ),
        control_tick=jnp.where(applied, state.control_tick, jnp.int32(-1)),
    )
    next_available = management.available.at[safe_team, safe_incoming].set(
        management.available[safe_team, safe_incoming] & (~applied)
    )
    next_substitutions = management.substitutions_used.at[safe_team].set(
        management.substitutions_used[safe_team] + applied.astype(jnp.int32)
    )
    opens_window = applied & (~same_window) & (~halftime_window)
    next_windows = management.windows_used.at[safe_team].set(
        management.windows_used[safe_team] + opens_window.astype(jnp.int32)
    )
    next_window_tick = management.last_window_restart_tick.at[safe_team].set(
        jnp.where(
            opens_window,
            state.restart.opened_control_tick,
            management.last_window_restart_tick[safe_team],
        )
    )
    return ManagerSubstitutionResult(
        state=next_state,
        offside=event.offside_state,
        management=management._replace(
            available=next_available,
            slot_generation=next_generation,
            substitutions_used=next_substitutions,
            windows_used=next_windows,
            last_window_restart_tick=next_window_tick,
        ),
        applied=applied,
        substitution_event=substitution_event,
    )


def _validate_manager_command(
    state: State,
    squad: SquadSetup,
    command: ManagerCommand,
) -> int:
    """Validate the fixed tree layout, leaving value legality to the kernel."""

    requested_shape = command.substitutions.requested.shape
    if len(requested_shape) != 2 or requested_shape[0] != 2:
        raise ValueError("command substitutions.requested must have shape (2, K)")
    width = requested_shape[1]
    substitution_shape = (2, width)
    expected = (
        (
            "substitutions.requested",
            command.substitutions.requested,
            substitution_shape,
            jnp.bool_,
        ),
        (
            "substitutions.outgoing_index",
            command.substitutions.outgoing_index,
            substitution_shape,
            jnp.int32,
        ),
        (
            "substitutions.incoming_bench_index",
            command.substitutions.incoming_bench_index,
            substitution_shape,
            jnp.int32,
        ),
        ("formations.requested", command.formations.requested, (2,), jnp.bool_),
        ("formations.layout_index", command.formations.layout_index, (2,), jnp.int32),
        (
            "acting_goalkeepers.requested",
            command.acting_goalkeepers.requested,
            (2,),
            jnp.bool_,
        ),
        (
            "acting_goalkeepers.player_slot",
            command.acting_goalkeepers.player_slot,
            (2,),
            jnp.int32,
        ),
        (
            "set_piece_takers.requested",
            command.set_piece_takers.requested,
            (2, RESTART_COUNT),
            jnp.bool_,
        ),
        (
            "set_piece_takers.player_slot",
            command.set_piece_takers.player_slot,
            (2, RESTART_COUNT),
            jnp.int32,
        ),
    )
    for name, value, shape, dtype in expected:
        array = jnp.asarray(value)
        if array.shape != shape:
            raise ValueError(f"command {name} must have shape {shape}")
        if array.dtype != jnp.dtype(dtype):
            raise TypeError(f"command {name} must have dtype {jnp.dtype(dtype)}")
    if squad.formation_layouts.ndim != 3:
        raise ValueError("registered formation layouts must have rank three")
    if squad.formation_layouts.shape[1:] != (state.players.position.shape[0], 2):
        raise ValueError("registered formation layouts must match roster slots")
    return width


def _apply_team_substitution_batch(
    state: State,
    offside: OffsideState,
    squad: SquadSetup,
    management: ManagerState,
    command: ManagerSubstitutionCommand,
    team: int,
    width: int,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    body: BodyContact,
    stadium: Stadium,
) -> tuple[
    State,
    OffsideState,
    ManagerState,
    jax.Array,
    jax.Array,
    SubstitutionEvent,
]:
    """Validate one unordered request set and commit it atomically."""

    if width == 0 or squad.valid.shape[1] == 0:
        reasons = jnp.where(
            command.requested[team],
            jnp.int32(ManagerCommandReason.BENCH_UNAVAILABLE),
            jnp.int32(ManagerCommandReason.NOT_REQUESTED),
        )
        return (
            state,
            offside,
            management,
            jnp.zeros(width, dtype=jnp.bool_),
            reasons,
            SubstitutionEvent.empty((width,)),
        )

    players = state.players
    player_count = players.position.shape[0]
    bench_width = squad.valid.shape[1]
    requested = command.requested[team]
    outgoing = command.outgoing_index[team]
    incoming = command.incoming_bench_index[team]
    outgoing_in_range = (outgoing >= 0) & (outgoing < player_count)
    incoming_in_range = (incoming >= 0) & (incoming < bench_width)
    safe_outgoing = jnp.clip(outgoing, 0, player_count - 1)
    safe_incoming = jnp.clip(incoming, 0, bench_width - 1)
    incoming_id = squad.player_id[team, safe_incoming]

    registered = (
        incoming_in_range
        & squad.valid[team, safe_incoming]
        & management.available[team, safe_incoming]
    )
    outgoing_valid = (
        outgoing_in_range
        & players.active[safe_outgoing]
        & (players.team_id[safe_outgoing] == team)
    )
    identity_unique = ~jnp.any(
        players.player_id[:, None] == incoming_id[None, :], axis=0
    )
    dtype = players.max_speed.dtype
    profile_valid = player_profile_values_valid(
        incoming_id,
        squad.max_speed[team, safe_incoming],
        squad.height[team, safe_incoming],
        squad.reach_height[team, safe_incoming],
        squad.ball_control[team, safe_incoming],
        squad.endurance_factor[team, safe_incoming],
        head_radius=jnp.asarray(body.head_radius_m, dtype=dtype),
    )
    cell_valid = outgoing_valid & registered & identity_unique & profile_valid

    requested_i32 = requested.astype(jnp.int32)
    outgoing_counts = (
        jnp.zeros(player_count, dtype=jnp.int32)
        .at[safe_outgoing]
        .add(requested_i32 * outgoing_in_range.astype(jnp.int32))
    )
    incoming_counts = (
        jnp.zeros(bench_width, dtype=jnp.int32)
        .at[safe_incoming]
        .add(requested_i32 * incoming_in_range.astype(jnp.int32))
    )
    duplicate_outgoing = jnp.any(outgoing_counts > 1)
    duplicate_incoming = jnp.any(incoming_counts > 1)

    request_count = jnp.sum(requested_i32, dtype=jnp.int32)
    any_requested = request_count > 0
    resources_valid = (
        management.substitutions_used[team] + request_count <= squad.max_substitutions
    )
    same_window = (state.restart.opened_control_tick >= 0) & (
        management.last_window_restart_tick[team] == state.restart.opened_control_tick
    )
    halftime_window = (
        (state.first_half_wall_end_tick >= 0)
        & (state.control_tick == state.first_half_wall_end_tick)
        & (state.restart.kind == RK_KICKOFF)
        & (~state.ball.live)
    )
    window_valid = (
        halftime_window
        | same_window
        | (management.windows_used[team] < squad.max_windows)
    )
    active_team = players.active & (players.team_id == team)
    initial_goalkeepers = jnp.sum(active_team & players.is_goalkeeper)
    outgoing_goalkeepers = jnp.sum(requested & players.is_goalkeeper[safe_outgoing])
    incoming_goalkeepers = jnp.sum(requested & squad.is_goalkeeper[team, safe_incoming])
    final_goalkeepers = (
        initial_goalkeepers - outgoing_goalkeepers + incoming_goalkeepers
    )
    management_open = _management_stoppage_open(
        state,
        offside,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
    )
    commit = (~any_requested) | (
        management_open
        & resources_valid
        & window_valid
        & (~duplicate_outgoing)
        & (~duplicate_incoming)
        & jnp.all((~requested) | cell_valid)
        & (final_goalkeepers <= 1)
    )
    applied = requested & commit
    reasons = jnp.full(
        (width,),
        jnp.int32(ManagerCommandReason.NOT_REQUESTED),
        dtype=jnp.int32,
    )
    reasons = jnp.where(
        requested,
        jnp.int32(ManagerCommandReason.ATOMIC_TEAM_REJECTED),
        reasons,
    )
    reasons = jnp.where(
        requested & (~management_open),
        jnp.int32(ManagerCommandReason.NOT_DEAD_BALL),
        reasons,
    )
    reasons = jnp.where(
        requested & management_open & (~resources_valid),
        jnp.int32(ManagerCommandReason.SUBSTITUTION_LIMIT_EXHAUSTED),
        reasons,
    )
    reasons = jnp.where(
        requested & management_open & resources_valid & (~window_valid),
        jnp.int32(ManagerCommandReason.SUBSTITUTION_WINDOW_EXHAUSTED),
        reasons,
    )
    reasons = jnp.where(
        requested & (duplicate_outgoing | duplicate_incoming),
        jnp.int32(ManagerCommandReason.DUPLICATE_REQUEST),
        reasons,
    )
    reasons = jnp.where(
        requested & outgoing_in_range & (~players.active[safe_outgoing]),
        jnp.int32(ManagerCommandReason.INACTIVE_PLAYER),
        reasons,
    )
    reasons = jnp.where(
        requested
        & outgoing_in_range
        & players.active[safe_outgoing]
        & (players.team_id[safe_outgoing] != team),
        jnp.int32(ManagerCommandReason.WRONG_TEAM),
        reasons,
    )
    reasons = jnp.where(
        requested & incoming_in_range & (~registered),
        jnp.int32(ManagerCommandReason.BENCH_UNAVAILABLE),
        reasons,
    )
    reasons = jnp.where(
        requested & incoming_in_range & registered & (~identity_unique),
        jnp.int32(ManagerCommandReason.IDENTITY_CONFLICT),
        reasons,
    )
    reasons = jnp.where(
        requested & incoming_in_range & registered & identity_unique & (~profile_valid),
        jnp.int32(ManagerCommandReason.INVALID_PROFILE),
        reasons,
    )
    reasons = jnp.where(
        requested & ((~outgoing_in_range) | (~incoming_in_range)),
        jnp.int32(ManagerCommandReason.INVALID_SLOT),
        reasons,
    )
    reasons = jnp.where(
        requested & cell_valid & (final_goalkeepers > 1),
        jnp.int32(ManagerCommandReason.GOALKEEPER_CONSTRAINT),
        reasons,
    )
    reasons = jnp.where(
        applied,
        jnp.int32(ManagerCommandReason.APPLIED),
        reasons,
    )

    outgoing_selector = applied[:, None] & (
        safe_outgoing[:, None] == jnp.arange(player_count, dtype=jnp.int32)[None, :]
    )
    outgoing_owner = jnp.argmax(outgoing_selector.astype(jnp.int32), axis=0)
    outgoing_changed = jnp.any(outgoing_selector, axis=0)

    def set_slots(base, replacement):
        """Gather one committed replacement per slot without padded scatters."""

        selected = replacement[outgoing_owner]
        mask = outgoing_changed
        while mask.ndim < base.ndim:
            mask = mask[..., None]
        return jnp.where(mask, selected, base)

    replacement_body_forward = initial_player_body_forward(
        players.team_id, state.attack_direction
    )[safe_outgoing]
    next_players = players._replace(
        velocity=set_slots(
            players.velocity, jnp.zeros((width, 2), dtype=players.velocity.dtype)
        ),
        body_forward=set_slots(players.body_forward, replacement_body_forward),
        gaze_yaw=set_slots(
            players.gaze_yaw, jnp.zeros(width, dtype=players.gaze_yaw.dtype)
        ),
        player_id=set_slots(players.player_id, incoming_id),
        on_pitch=set_slots(players.on_pitch, jnp.ones(width, dtype=jnp.bool_)),
        sent_off=set_slots(players.sent_off, jnp.zeros(width, dtype=jnp.bool_)),
        is_goalkeeper=set_slots(
            players.is_goalkeeper, squad.is_goalkeeper[team, safe_incoming]
        ),
        max_speed=set_slots(players.max_speed, squad.max_speed[team, safe_incoming]),
        reach_height=set_slots(
            players.reach_height, squad.reach_height[team, safe_incoming]
        ),
        height=set_slots(players.height, squad.height[team, safe_incoming]),
        ball_control=set_slots(
            players.ball_control, squad.ball_control[team, safe_incoming]
        ),
        endurance_factor=set_slots(
            players.endurance_factor, squad.endurance_factor[team, safe_incoming]
        ),
        stamina_long=set_slots(
            players.stamina_long, jnp.ones(width, dtype=players.stamina_long.dtype)
        ),
        stamina_short=set_slots(
            players.stamina_short, jnp.ones(width, dtype=players.stamina_short.dtype)
        ),
        challenge_recovery_substeps=set_slots(
            players.challenge_recovery_substeps,
            jnp.zeros(width, dtype=jnp.int32),
        ),
        contact_lock_substeps=set_slots(
            players.contact_lock_substeps, jnp.zeros(width, dtype=jnp.int32)
        ),
        aerial_recovery_substeps=set_slots(
            players.aerial_recovery_substeps, jnp.zeros(width, dtype=jnp.int32)
        ),
        possession_loss_lock_substeps=set_slots(
            players.possession_loss_lock_substeps,
            jnp.zeros(width, dtype=jnp.int32),
        ),
        yellow_cards=set_slots(players.yellow_cards, jnp.zeros(width, dtype=jnp.int32)),
    )
    replaced_last_actor = jnp.any(
        applied & (safe_outgoing == state.possession.last_contact.actor)
    )
    last_contact = jax.tree.map(
        lambda empty, old: jnp.where(replaced_last_actor, empty, old),
        _empty_contact(),
        state.possession.last_contact,
    )
    candidate_state = state._replace(
        players=next_players,
        possession=state.possession._replace(last_contact=last_contact),
    )
    taker_replaced = jnp.any(applied & (safe_outgoing == state.restart.taker))
    replacement_taker = select_restart_taker(
        candidate_state,
        candidate_state.restart.kind,
        candidate_state.restart.team,
        candidate_state.ball.position,
        stadium=stadium,
    )
    candidate_state = candidate_state._replace(
        restart=candidate_state.restart._replace(
            taker=jnp.where(
                taker_replaced, replacement_taker, candidate_state.restart.taker
            ).astype(jnp.int32)
        )
    )

    incoming_selector = applied[:, None] & (
        safe_incoming[:, None] == jnp.arange(bench_width, dtype=jnp.int32)[None, :]
    )
    consumed = jnp.any(incoming_selector, axis=0)
    next_available = management.available.at[team].set(
        management.available[team] & (~consumed)
    )
    generation_increment = outgoing_changed.astype(jnp.int32)
    next_generation = management.slot_generation + generation_increment
    opens_window = any_requested & commit & (~same_window) & (~halftime_window)
    next_management = management._replace(
        available=next_available,
        slot_generation=next_generation,
        substitutions_used=management.substitutions_used.at[team].set(
            management.substitutions_used[team] + jnp.sum(applied).astype(jnp.int32)
        ),
        windows_used=management.windows_used.at[team].set(
            management.windows_used[team] + opens_window.astype(jnp.int32)
        ),
        last_window_restart_tick=management.last_window_restart_tick.at[team].set(
            jnp.where(
                opens_window,
                state.restart.opened_control_tick,
                management.last_window_restart_tick[team],
            )
        ),
    )
    events = SubstitutionEvent(
        occurred=applied,
        team=jnp.where(applied, jnp.int32(team), jnp.int32(NO_TEAM)),
        player_slot=jnp.where(applied, safe_outgoing, jnp.int32(NO_PLAYER)),
        outgoing_player_id=jnp.where(
            applied, players.player_id[safe_outgoing], jnp.int32(NO_PLAYER)
        ),
        incoming_player_id=jnp.where(applied, incoming_id, jnp.int32(NO_PLAYER)),
        slot_generation=jnp.where(
            applied, next_generation[safe_outgoing], jnp.int32(-1)
        ),
        control_tick=jnp.where(applied, state.control_tick, jnp.int32(-1)),
    )
    return candidate_state, offside, next_management, applied, reasons, events


def _apply_acting_goalkeepers(
    state: State,
    offside: OffsideState,
    management: ManagerState,
    command: ManagerActingGoalkeeperCommand,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    stadium: Stadium,
) -> tuple[State, jax.Array, jax.Array, ActingGoalkeeperEvent]:
    """Repair a missing active goalkeeper without spending a substitution.

    A valid manager choice wins. If it is absent or invalid, the active
    outfielder nearest the own goal is selected deterministically. This is a
    low-frequency stoppage fail-safe, not part of ``env.step``.
    """

    players = state.players
    player_count = players.position.shape[0]
    index = jnp.arange(player_count, dtype=jnp.int32)
    next_is_goalkeeper = players.is_goalkeeper
    applied_rows = []
    reason_rows = []
    event_rows = []
    management_open = _management_stoppage_open(
        state,
        offside,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
    )
    for team in (TEAM_0, TEAM_1):
        active_team = players.active & (players.team_id == team)
        missing = ~jnp.any(active_team & next_is_goalkeeper)
        proposed = command.player_slot[team]
        in_range = (proposed >= 0) & (proposed < player_count)
        safe_proposed = jnp.clip(proposed, 0, player_count - 1)
        explicit_valid = (
            command.requested[team]
            & in_range
            & active_team[safe_proposed]
            & (~next_is_goalkeeper[safe_proposed])
        )

        candidates = active_team & (~next_is_goalkeeper)
        own_goal = jnp.asarray(
            [
                -state.attack_direction[jnp.int32(team)] * stadium.half_length,
                0.0,
            ],
            dtype=players.position.dtype,
        )
        goal_distance_squared = jnp.sum((players.position - own_goal) ** 2, axis=-1)
        fallback = jnp.argmin(
            jnp.where(candidates, goal_distance_squared, jnp.inf)
        ).astype(jnp.int32)
        has_candidate = jnp.any(candidates)
        chosen = jnp.where(explicit_valid, safe_proposed, fallback).astype(jnp.int32)
        applied = management_open & missing & has_candidate
        forced = applied & (~explicit_valid)
        requested = command.requested[team]
        reason = jnp.where(
            ~requested,
            jnp.where(
                applied,
                jnp.int32(ManagerCommandReason.INTERNAL_FALLBACK),
                jnp.int32(ManagerCommandReason.NOT_REQUESTED),
            ),
            jnp.where(
                ~missing,
                jnp.int32(ManagerCommandReason.ALREADY_CHANGED),
                jnp.where(
                    ~management_open,
                    jnp.int32(ManagerCommandReason.NOT_DEAD_BALL),
                    jnp.where(
                        ~in_range,
                        jnp.int32(ManagerCommandReason.INVALID_SLOT),
                        jnp.where(
                            ~players.active[safe_proposed],
                            jnp.int32(ManagerCommandReason.INACTIVE_PLAYER),
                            jnp.where(
                                players.team_id[safe_proposed] != team,
                                jnp.int32(ManagerCommandReason.WRONG_TEAM),
                                jnp.where(
                                    next_is_goalkeeper[safe_proposed]
                                    | (~has_candidate),
                                    jnp.int32(
                                        ManagerCommandReason.GOALKEEPER_CONSTRAINT
                                    ),
                                    jnp.int32(ManagerCommandReason.APPLIED),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        next_is_goalkeeper = jnp.where(
            applied & (index == chosen), jnp.bool_(True), next_is_goalkeeper
        )
        applied_rows.append(applied)
        reason_rows.append(reason)
        event_rows.append(
            ActingGoalkeeperEvent(
                occurred=applied,
                team=jnp.where(applied, jnp.int32(team), jnp.int32(NO_TEAM)),
                player_slot=jnp.where(applied, chosen, jnp.int32(NO_PLAYER)),
                player_id=jnp.where(
                    applied, players.player_id[chosen], jnp.int32(NO_PLAYER)
                ),
                slot_generation=jnp.where(
                    applied, management.slot_generation[chosen], jnp.int32(-1)
                ),
                environment_forced=forced,
                control_tick=jnp.where(applied, state.control_tick, jnp.int32(-1)),
            )
        )

    return (
        state._replace(players=players._replace(is_goalkeeper=next_is_goalkeeper)),
        jnp.stack(applied_rows),
        jnp.stack(reason_rows),
        jax.tree.map(lambda *values: jnp.stack(values), *event_rows),
    )


def _align_goalkeeper_roles(
    state: State,
    management: ManagerState,
) -> tuple[State, ManagerState, jax.Array]:
    """Move a newly assigned goalkeeper into the registered GK slot.

    The old formation-slot owner may be sent off. Swapping its anchor and role
    with the new active goalkeeper preserves one tactical GK role without
    changing the fixed roster shape. The move is administrative and therefore
    carries no velocity or stamina cost.
    """

    players = state.players
    count = players.position.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    position = players.position
    velocity = players.velocity
    body_forward = players.body_forward
    gaze_yaw = players.gaze_yaw
    anchor = management.formation_anchor
    role = management.formation_role
    reassigned_rows = []

    for team in (TEAM_0, TEAM_1):
        team_slot = players.team_id == team
        active_goalkeeper = team_slot & players.active & players.is_goalkeeper
        has_one = jnp.sum(active_goalkeeper.astype(jnp.int32)) == 1
        target = jnp.argmax(active_goalkeeper.astype(jnp.int32)).astype(jnp.int32)
        role_source_mask = team_slot & (role == ROLE_GOALKEEPER)
        has_source = jnp.any(role_source_mask)
        source = jnp.argmax(role_source_mask.astype(jnp.int32)).astype(jnp.int32)
        needs_reassignment = has_one & (role[target] != ROLE_GOALKEEPER)
        swap = needs_reassignment & has_source & (source != target)

        source_anchor = anchor[source]
        target_anchor = anchor[target]
        target_role = role[target]
        anchor = jnp.where(
            (swap & (index == target))[:, None],
            source_anchor,
            jnp.where((swap & (index == source))[:, None], target_anchor, anchor),
        )
        role = jnp.where(
            needs_reassignment & (index == target),
            jnp.int32(ROLE_GOALKEEPER),
            jnp.where(swap & (index == source), target_role, role),
        )

        source_active = has_source & players.active[source]
        source_world_anchor = source_anchor * state.attack_direction[jnp.int32(team)]
        target_world = jnp.where(
            has_source,
            jnp.where(source_active, position[source], source_world_anchor),
            position[target],
        )
        source_world = position[target]
        target_mask = needs_reassignment & has_source & (index == target)
        source_mask = swap & source_active & (index == source)
        position = jnp.where(
            target_mask[:, None],
            target_world,
            jnp.where(source_mask[:, None], source_world, position),
        )
        moved_mask = target_mask | source_mask
        velocity = jnp.where(moved_mask[:, None], jnp.zeros_like(velocity), velocity)
        target_body_forward = initial_player_body_forward(
            players.team_id, state.attack_direction
        )
        body_forward = jnp.where(moved_mask[:, None], target_body_forward, body_forward)
        gaze_yaw = jnp.where(moved_mask, 0.0, gaze_yaw)
        reassigned_rows.append(needs_reassignment)

    return (
        state._replace(
            players=players._replace(
                position=position,
                velocity=velocity,
                body_forward=body_forward,
                gaze_yaw=gaze_yaw,
            )
        ),
        management._replace(formation_anchor=anchor, formation_role=role),
        jnp.stack(reassigned_rows),
    )


def opening_formation_available(
    state: State,
    management: ManagerState,
) -> jax.Array:
    """Return teams whose authored kickoff pose may still be selected once."""

    coherent_opening = (
        (state.control_tick == 0)
        & (state.first_half_wall_end_tick < 0)
        & (~state.ball.live)
        & (state.restart.kind == RK_KICKOFF)
        & ((state.restart.team == TEAM_0) | (state.restart.team == TEAM_1))
        & (state.restart.opened_control_tick == 0)
    )
    return coherent_opening & (~management.opening_formation_committed)


def apply_opening_formation(
    state: State,
    offside: OffsideState,
    setup: MatchSetup,
    squad: SquadSetup,
    management: ManagerState,
    command: ManagerFormationCommand,
    *,
    stadium: Stadium,
    ball: Ball,
    body: BodyContact,
    mirror_second_half: bool = True,
) -> OpeningFormationResult:
    """Apply registered player positions exactly once before the opening kick.

    The selected attacking-frame layout becomes the long-lived tactical
    anchor. Its world-frame pose is then passed through the ordinary kickoff
    projector, so half, centre-circle, spacing, goalkeeper, and taker
    constraints remain authoritative. Later formation commands never enter
    this function and therefore remain non-teleporting tactical updates.
    """

    if not isinstance(mirror_second_half, bool):
        raise TypeError("mirror_second_half must be bool")
    if command.requested.shape != (2,) or command.layout_index.shape != (2,):
        raise ValueError("opening formation command fields must have shape [2]")
    if command.requested.dtype != jnp.bool_:
        raise TypeError("opening formation requested must have bool dtype")
    if command.layout_index.dtype != jnp.int32:
        raise TypeError("opening formation layout_index must have int32 dtype")
    player_count = state.players.position.shape[0]
    if setup.second_half_positions.shape != (player_count, 2):
        raise ValueError("setup second-half positions must have shape [N, 2]")
    if squad.formation_layouts.ndim != 3:
        raise ValueError("registered formation layouts must have rank three")
    if squad.formation_layouts.shape[1:] != (player_count, 2):
        raise ValueError("registered formation layouts must match player slots")
    if squad.formation_roles.shape != (
        squad.formation_layouts.shape[0],
        player_count,
    ):
        raise ValueError("registered formation roles must match layouts")

    available = opening_formation_available(state, management)
    layout_count = squad.formation_layouts.shape[0]
    position = state.players.position
    velocity = state.players.velocity
    body_forward = state.players.body_forward
    gaze_yaw = state.players.gaze_yaw
    anchor = management.formation_anchor
    role = management.formation_role
    formation_index = management.formation_index
    canonical_body_forward = initial_player_body_forward(
        state.players.team_id, state.attack_direction
    )
    applied_rows = []

    for team in (TEAM_0, TEAM_1):
        proposed = command.layout_index[team]
        in_range = (proposed >= 0) & (proposed < layout_count)
        safe_layout = jnp.clip(proposed, 0, layout_count - 1)
        apply = available[team] & command.requested[team] & in_range
        team_slot = state.players.team_id == team
        selected_anchor = squad.formation_layouts[safe_layout]
        selected_world = selected_anchor * state.attack_direction[team].astype(
            position.dtype
        )
        selector = apply & team_slot
        position = jnp.where(selector[:, None], selected_world, position)
        velocity = jnp.where(selector[:, None], jnp.zeros_like(velocity), velocity)
        body_forward = jnp.where(
            selector[:, None], canonical_body_forward, body_forward
        )
        gaze_yaw = jnp.where(selector, 0.0, gaze_yaw)
        anchor = jnp.where(selector[:, None], selected_anchor, anchor)
        role = jnp.where(
            selector,
            squad.formation_roles[safe_layout],
            role,
        )
        formation_index = formation_index.at[team].set(
            jnp.where(apply, safe_layout, formation_index[team])
        )
        applied_rows.append(apply)

    applied = jnp.stack(applied_rows)
    next_management = management._replace(
        formation_index=formation_index,
        formation_anchor=anchor,
        formation_role=role,
        opening_formation_committed=(management.opening_formation_committed | applied),
    )
    candidate = state._replace(
        players=state.players._replace(
            position=position,
            velocity=velocity,
            body_forward=body_forward,
            gaze_yaw=gaze_yaw,
        ),
        restart_layout_ready=state.restart_layout_ready & (~jnp.any(applied)),
    )

    def project_opening(value: State) -> State:
        positioning = prepare_restart_positioning(
            value,
            stadium=stadium,
            ball=ball,
            body=body,
        )
        return value._replace(
            players=value.players._replace(
                position=positioning.position,
                body_forward=body_forward_from_angle(positioning.facing),
                velocity=jnp.where(
                    positioning.forced[:, None],
                    jnp.zeros_like(value.players.velocity),
                    value.players.velocity,
                ),
            ),
            restart_layout_ready=positioning.taker_ready,
        )

    next_state = jax.lax.cond(
        jnp.any(applied),
        project_opening,
        lambda value: value,
        candidate,
    )
    second_half_positions = jnp.where(
        jnp.bool_(mirror_second_half) & jnp.any(applied),
        -next_state.players.position,
        setup.second_half_positions,
    )
    return OpeningFormationResult(
        state=next_state,
        offside=offside,
        setup=setup._replace(second_half_positions=second_half_positions),
        management=next_management,
        applied=applied,
    )


def apply_manager_command(
    state: State,
    offside: OffsideState,
    squad: SquadSetup,
    management: ManagerState,
    command: ManagerCommand,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    body: BodyContact,
    stadium: Stadium,
    ball: Ball,
) -> ManagerCommandResult:
    """Apply one low-frequency manager transaction outside the physics step.

    Substitutions are atomic per team: if any requested cell is illegal, none
    of that team's cells are committed. The opposing team's transaction is
    independent, matching the referee's real-world approval boundary.
    Formation changes only update policy targets. An emergency goalkeeper is
    the sole exception: that player moves administratively to the registered
    goalkeeper anchor before restart legality is re-projected.
    Taker requests apply only to a matching, visible non-hold restart. A
    different taker receives the old taker's legal pose atomically; final restart
    projection either commits identity and poses together or rolls that axis back.
    """

    width = _validate_manager_command(state, squad, command)
    next_state = state
    next_offside = offside
    next_management = management
    substitution_rows = []
    substitution_reason_rows = []
    substitution_event_rows = []
    for team in (TEAM_0, TEAM_1):
        (
            next_state,
            next_offside,
            next_management,
            applied,
            reasons,
            substitution_events,
        ) = _apply_team_substitution_batch(
            next_state,
            next_offside,
            squad,
            next_management,
            command.substitutions,
            team,
            width,
            fulltime_tick=fulltime_tick,
            minimum_team_players=minimum_team_players,
            body=body,
            stadium=stadium,
        )
        substitution_rows.append(applied)
        substitution_reason_rows.append(reasons)
        substitution_event_rows.append(substitution_events)
    substitutions_applied = jnp.stack(substitution_rows)
    substitution_reasons = jnp.stack(substitution_reason_rows)
    substitution_events = jax.tree.map(
        lambda *values: jnp.stack(values), *substitution_event_rows
    )

    formation_applied = []
    formation_reasons = []
    formation_index = next_management.formation_index
    formation_anchor = next_management.formation_anchor
    formation_role = next_management.formation_role
    regulation_tick = regulation_elapsed_ticks(next_state)
    active_count = jnp.stack(
        [
            jnp.sum(
                next_state.players.active & (next_state.players.team_id == active_team)
            )
            for active_team in (TEAM_0, TEAM_1)
        ]
    )
    match_open = (
        (regulation_tick < jnp.int32(fulltime_tick))
        | (next_state.restart.kind == RK_PENALTY)
    ) & jnp.all(active_count >= jnp.asarray(minimum_team_players, dtype=jnp.int32))
    layout_count = squad.formation_layouts.shape[0]
    for team in (TEAM_0, TEAM_1):
        proposed = command.formations.layout_index[team]
        in_range = (proposed >= 0) & (proposed < layout_count)
        safe_layout = jnp.clip(proposed, 0, layout_count - 1)
        apply = command.formations.requested[team] & in_range & match_open
        apply = apply & (safe_layout != formation_index[team])
        team_slot = next_state.players.team_id == team
        formation_anchor = jnp.where(
            (apply & team_slot)[:, None],
            squad.formation_layouts[safe_layout],
            formation_anchor,
        )
        formation_index = formation_index.at[team].set(
            jnp.where(apply, safe_layout, formation_index[team])
        )
        formation_role = jnp.where(
            apply & team_slot,
            squad.formation_roles[safe_layout],
            formation_role,
        )
        formation_applied.append(apply)
        reason = jnp.where(
            ~command.formations.requested[team],
            jnp.int32(ManagerCommandReason.NOT_REQUESTED),
            jnp.where(
                ~match_open,
                jnp.int32(ManagerCommandReason.TERMINAL),
                jnp.where(
                    ~in_range,
                    jnp.int32(ManagerCommandReason.INVALID_FORMATION),
                    jnp.where(
                        safe_layout == next_management.formation_index[team],
                        jnp.int32(ManagerCommandReason.UNCHANGED),
                        jnp.int32(ManagerCommandReason.APPLIED),
                    ),
                ),
            ),
        )
        formation_reasons.append(reason)
    next_management = next_management._replace(
        formation_index=formation_index,
        formation_anchor=formation_anchor,
        formation_role=formation_role,
    )

    (
        next_state,
        acting_goalkeepers_applied,
        acting_goalkeeper_reasons,
        acting_goalkeeper_events,
    ) = _apply_acting_goalkeepers(
        next_state,
        next_offside,
        next_management,
        command.acting_goalkeepers,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
        stadium=stadium,
    )
    next_state, next_management, goalkeeper_roles_reassigned = _align_goalkeeper_roles(
        next_state, next_management
    )
    roster_identity_changed = jnp.any(substitutions_applied) | jnp.any(
        acting_goalkeepers_applied
    )
    next_state, restart_taker_repaired = reconcile_restart_after_roster_change(
        next_state,
        roster_identity_changed,
        body=body,
        stadium=stadium,
    )

    players = next_state.players
    player_count = players.position.shape[0]
    proposed_taker = command.set_piece_takers.player_slot
    slot_in_range = (proposed_taker >= 0) & (proposed_taker < player_count)
    safe_slot = jnp.clip(proposed_taker, 0, player_count - 1)
    team_grid = jnp.arange(2, dtype=jnp.int32)[:, None]
    kind_grid = jnp.arange(RESTART_COUNT, dtype=jnp.int32)[None, :]
    active = players.active[safe_slot]
    correct_team = players.team_id[safe_slot] == team_grid
    role_eligible = kind_grid != RK_GK_HOLD
    matching_restart = (
        (~next_state.ball.live)
        & (next_state.restart.kind > RK_NONE)
        & (next_state.restart.team == team_grid)
        & (next_state.restart.kind == kind_grid)
    )
    takers_requested = (
        command.set_piece_takers.requested
        & slot_in_range
        & active
        & correct_team
        & role_eligible
        & matching_restart
        & match_open
    )
    selected = jnp.sum(
        jnp.where(takers_requested, safe_slot, jnp.int32(0)), dtype=jnp.int32
    )
    previous_taker = next_state.restart.taker
    safe_previous_taker = jnp.clip(previous_taker, 0, player_count - 1)
    previous_taker_valid = (
        (previous_taker >= 0)
        & (previous_taker < player_count)
        & players.active[safe_previous_taker]
        & (players.team_id[safe_previous_taker] == next_state.restart.team)
    )
    taker_change_requested = (
        jnp.any(takers_requested) & previous_taker_valid & (selected != previous_taker)
    )
    reposition = (
        jnp.any(substitutions_applied)
        | jnp.any(goalkeeper_roles_reassigned | acting_goalkeepers_applied)
        | jnp.any(jnp.stack(formation_applied))
        | restart_taker_repaired
    )
    active_restart = (next_state.restart.kind > RK_NONE) & (
        next_state.restart.kind < RESTART_COUNT
    )

    def project_restart(candidate: State) -> State:
        positioning = prepare_restart_positioning(
            candidate, stadium=stadium, ball=ball, body=body
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
        reposition & active_restart,
        project_restart,
        lambda candidate: candidate,
        next_state,
    )

    # A taker identity cannot be committed on its own. The previous taker is
    # already occupying the physical release pose, so swapping the old and new
    # actors' poses vacates it before the ordinary restart projector runs. Both
    # identities are administratively repositioned and therefore start at zero
    # velocity. If the fully projected layout is still invalid, every leaf is
    # restored from the legal pre-taker transaction above.
    player_index = jnp.arange(player_count, dtype=jnp.int32)
    previous_position = next_state.players.position
    previous_body_forward = next_state.players.body_forward
    previous_gaze_yaw = next_state.players.gaze_yaw
    old_slot = player_index == safe_previous_taker
    new_slot = player_index == selected
    swapped_position = jnp.where(
        old_slot[:, None],
        previous_position[selected],
        jnp.where(
            new_slot[:, None],
            previous_position[safe_previous_taker],
            previous_position,
        ),
    )
    swapped_body_forward = jnp.where(
        old_slot[:, None],
        previous_body_forward[selected],
        jnp.where(
            new_slot[:, None],
            previous_body_forward[safe_previous_taker],
            previous_body_forward,
        ),
    )
    swapped_gaze_yaw = jnp.where(
        old_slot,
        previous_gaze_yaw[selected],
        jnp.where(new_slot, previous_gaze_yaw[safe_previous_taker], previous_gaze_yaw),
    )
    swapped_slots = taker_change_requested & (old_slot | new_slot)
    taker_candidate = next_state._replace(
        players=next_state.players._replace(
            position=jnp.where(
                taker_change_requested, swapped_position, previous_position
            ),
            body_forward=jnp.where(
                taker_change_requested, swapped_body_forward, previous_body_forward
            ),
            gaze_yaw=jnp.where(
                taker_change_requested, swapped_gaze_yaw, previous_gaze_yaw
            ),
            velocity=jnp.where(
                swapped_slots[:, None],
                jnp.zeros_like(next_state.players.velocity),
                next_state.players.velocity,
            ),
        ),
        restart=next_state.restart._replace(
            taker=jnp.where(taker_change_requested, selected, previous_taker).astype(
                jnp.int32
            )
        ),
        restart_layout_ready=(
            next_state.restart_layout_ready & (~taker_change_requested)
        ),
    )
    projected_taker = jax.lax.cond(
        taker_change_requested & active_restart,
        project_restart,
        lambda candidate: candidate,
        taker_candidate,
    )
    taker_committed = taker_change_requested & projected_taker.restart_layout_ready
    next_state = jax.tree.map(
        lambda changed, original: jnp.where(taker_committed, changed, original),
        projected_taker,
        next_state,
    )
    takers_applied = takers_requested & taker_committed
    taker_reasons = jnp.full(
        command.set_piece_takers.requested.shape,
        jnp.int32(ManagerCommandReason.NOT_REQUESTED),
        dtype=jnp.int32,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested,
        jnp.int32(ManagerCommandReason.NO_MATCHING_RESTART),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested & (~slot_in_range),
        jnp.int32(ManagerCommandReason.INVALID_SLOT),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested & slot_in_range & (~active),
        jnp.int32(ManagerCommandReason.INACTIVE_PLAYER),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested & slot_in_range & active & (~correct_team),
        jnp.int32(ManagerCommandReason.WRONG_TEAM),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested
        & slot_in_range
        & active
        & correct_team
        & (~role_eligible),
        jnp.int32(ManagerCommandReason.INELIGIBLE_TAKER),
        taker_reasons,
    )
    same_taker = matching_restart & (safe_slot == previous_taker)
    taker_reasons = jnp.where(
        command.set_piece_takers.requested
        & matching_restart
        & slot_in_range
        & active
        & correct_team
        & role_eligible
        & (~previous_taker_valid),
        jnp.int32(ManagerCommandReason.INELIGIBLE_TAKER),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        command.set_piece_takers.requested
        & matching_restart
        & slot_in_range
        & active
        & correct_team
        & role_eligible
        & previous_taker_valid
        & same_taker,
        jnp.int32(ManagerCommandReason.UNCHANGED),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        takers_requested & previous_taker_valid & (~same_taker) & (~taker_committed),
        jnp.int32(ManagerCommandReason.PLACEMENT_FAILED),
        taker_reasons,
    )
    taker_reasons = jnp.where(
        takers_applied,
        jnp.int32(ManagerCommandReason.APPLIED),
        taker_reasons,
    )
    roster_metadata_changed = jnp.any(substitutions_applied) | jnp.any(
        acting_goalkeepers_applied
    )
    return ManagerCommandResult(
        state=next_state,
        offside=next_offside,
        management=next_management,
        substitutions_applied=substitutions_applied,
        substitution_reasons=substitution_reasons,
        substitution_events=substitution_events,
        formations_applied=jnp.stack(formation_applied),
        formation_reasons=jnp.stack(formation_reasons),
        acting_goalkeepers_applied=acting_goalkeepers_applied,
        acting_goalkeeper_reasons=acting_goalkeeper_reasons,
        acting_goalkeeper_events=acting_goalkeeper_events,
        set_piece_takers_applied=takers_applied,
        set_piece_taker_reasons=taker_reasons,
        roster_metadata_changed=roster_metadata_changed,
    )


__all__ = [
    "MANAGER_OBSERVATION_SCHEMA_VERSION",
    "PLAYER_TACTICAL_OBSERVATION_SCHEMA_VERSION",
    "ActingGoalkeeperEvent",
    "ManagementInitialization",
    "ManagementRules",
    "ManagerActingGoalkeeperCommand",
    "ManagerAction",
    "ManagerBenchObservation",
    "ManagerCommand",
    "ManagerCommandReason",
    "ManagerCommandResult",
    "ManagerFormationCommand",
    "ManagerObservation",
    "ManagerOnFieldObservation",
    "ManagerSetPieceTakerCommand",
    "ManagerState",
    "ManagerSubstitutionCommand",
    "ManagerSubstitutionResult",
    "OpeningFormationResult",
    "PlayerTacticalObservation",
    "SquadSetup",
    "SubstitutionEvent",
    "apply_manager_command",
    "apply_manager_substitution",
    "apply_opening_formation",
    "emergency_goalkeeper_command",
    "initialize_management",
    "observe_manager",
    "observe_player_tactics",
    "opening_formation_available",
    "reconcile_restart_after_roster_change",
]
