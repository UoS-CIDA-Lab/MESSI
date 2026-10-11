"""Physics rollout chunk 경계에서만 적용하는 fixed-slot substitution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_GK_HOLD,
    RK_KICKOFF,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import empty_contact_result
from messi.core.formation import ROLE_GOALKEEPER
from messi.core.numeric import require_bounded_integer
from messi.core.restart import restart_active
from messi.core.roster import BenchRoster, RosterSampling
from messi.environment.episode import MatchSetup, _validate_match_setup


@dataclass(frozen=True, slots=True)
class ManagementLimits:
    max_substitutions_per_team: int = 5
    max_windows_per_team: int = 3

    def __post_init__(self) -> None:
        for name in ("max_substitutions_per_team", "max_windows_per_team"):
            try:
                value = require_bounded_integer(name, getattr(self, name), minimum=0)
            except TypeError as error:
                raise ValueError(f"{name}은 0 이상의 정수여야 한다") from error
            object.__setattr__(
                self,
                name,
                value,
            )


DEFAULT_MANAGEMENT_LIMITS = ManagementLimits()
DEFAULT_ROSTER_SAMPLING = RosterSampling()


class ManagerState(NamedTuple):
    substitutions_used: jax.Array
    windows_used: jax.Array
    last_window_control_tick: jax.Array


class SubstitutionRequest(NamedTuple):
    enabled: jax.Array
    team: jax.Array
    outgoing_index: jax.Array
    incoming_player_id: jax.Array
    incoming_is_goalkeeper: jax.Array
    incoming_max_speed: jax.Array
    incoming_height: jax.Array
    incoming_reach_height: jax.Array
    incoming_ball_control: jax.Array
    incoming_endurance_factor: jax.Array
    incoming_yellow_cards: jax.Array


class SubstitutionResult(NamedTuple):
    state: object
    roster: object
    manager: ManagerState
    applied: jax.Array


class BenchSubstitutionRequest(NamedTuple):
    enabled: jax.Array
    team: jax.Array
    outgoing_index: jax.Array
    incoming_bench_index: jax.Array


class BenchSubstitutionResult(NamedTuple):
    state: object
    roster: object
    manager: ManagerState
    bench: BenchRoster
    applied: jax.Array


class BenchSubstitutionBatchRequest(NamedTuple):
    """한 team의 한 교체 window에 제출하는 fixed-width 요청."""

    enabled: jax.Array
    team: jax.Array
    outgoing_index: jax.Array
    incoming_bench_index: jax.Array


class BenchSubstitutionBatchResult(NamedTuple):
    state: object
    roster: object
    manager: ManagerState
    bench: BenchRoster
    applied: jax.Array
    applied_count: jax.Array
    setup: MatchSetup


class SetPieceTakerRequest(NamedTuple):
    enabled: jax.Array
    team: jax.Array
    restart_kind: jax.Array
    player_index: jax.Array


class SetPieceTakerResult(NamedTuple):
    state: object
    applied: jax.Array


class ActingGoalkeeperRequest(NamedTuple):
    enabled: jax.Array
    team: jax.Array
    player_index: jax.Array


class ActingGoalkeeperResult(NamedTuple):
    roster: object
    applied: jax.Array
    setup: MatchSetup


def empty_manager_state() -> ManagerState:
    return ManagerState(
        substitutions_used=jnp.zeros(2, dtype=jnp.int32),
        windows_used=jnp.zeros(2, dtype=jnp.int32),
        last_window_control_tick=jnp.full(2, -1, dtype=jnp.int32),
    )


def empty_substitution_request() -> SubstitutionRequest:
    return SubstitutionRequest(
        enabled=jnp.bool_(False),
        team=jnp.int32(NO_TEAM),
        outgoing_index=jnp.int32(NO_PLAYER),
        incoming_player_id=jnp.int32(-1),
        incoming_is_goalkeeper=jnp.bool_(False),
        incoming_max_speed=jnp.float32(0.0),
        incoming_height=jnp.float32(0.0),
        incoming_reach_height=jnp.float32(0.0),
        incoming_ball_control=jnp.float32(0.0),
        incoming_endurance_factor=jnp.float32(0.0),
        incoming_yellow_cards=jnp.int32(0),
    )


def empty_bench_substitution_request() -> BenchSubstitutionRequest:
    return BenchSubstitutionRequest(
        enabled=jnp.bool_(False),
        team=jnp.int32(NO_TEAM),
        outgoing_index=jnp.int32(NO_PLAYER),
        incoming_bench_index=jnp.int32(NO_PLAYER),
    )


def empty_set_piece_taker_request() -> SetPieceTakerRequest:
    return SetPieceTakerRequest(
        enabled=jnp.bool_(False),
        team=jnp.int32(NO_TEAM),
        restart_kind=jnp.int32(0),
        player_index=jnp.int32(NO_PLAYER),
    )


def empty_acting_goalkeeper_request() -> ActingGoalkeeperRequest:
    return ActingGoalkeeperRequest(
        enabled=jnp.bool_(False),
        team=jnp.int32(NO_TEAM),
        player_index=jnp.int32(NO_PLAYER),
    )


def _management_open(state) -> jax.Array:
    return (
        (~state.ball.live)
        & restart_active(state.restart.kind)
        & (state.restart.kind != RK_GK_HOLD)
        & (~jnp.any(state.offside_flagged))
        & (state.offside_direct_exempt_team == NO_TEAM)
    )


def _halftime_substitution_interval(state) -> jax.Array:
    """Second-half kickoff 전 IFAB substitution-opportunity 예외인지 판정한다."""

    return (
        (state.first_half_wall_end_tick >= 0)
        & (state.restart.kind == RK_KICKOFF)
        & (state.restart.opened_control_tick == state.first_half_wall_end_tick)
    )


def _align_goalkeeper_role(roster, active: jax.Array, team: jax.Array):
    """한 team의 유일한 active goalkeeper에게 GK role과 anchor를 옮긴다."""

    count = roster.team_id.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    team = jnp.asarray(team, dtype=jnp.int32)
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    valid_team = (team == TEAM_0) | (team == TEAM_1)
    team_mask = roster.team_id == safe_team
    goalkeeper_mask = active & team_mask & roster.is_goalkeeper
    role_source_mask = team_mask & (roster.role == ROLE_GOALKEEPER)
    has_one_goalkeeper = jnp.sum(goalkeeper_mask.astype(jnp.int32)) == 1
    has_one_role_source = jnp.sum(role_source_mask.astype(jnp.int32)) == 1
    target = jnp.argmax(goalkeeper_mask.astype(jnp.int32)).astype(jnp.int32)
    source = jnp.argmax(role_source_mask.astype(jnp.int32)).astype(jnp.int32)
    swap = valid_team & has_one_goalkeeper & has_one_role_source & (source != target)
    source_anchor = roster.formation_anchor[source]
    target_anchor = roster.formation_anchor[target]
    target_role = roster.role[target]
    role = jnp.where(
        swap & (index == target),
        jnp.int32(ROLE_GOALKEEPER),
        jnp.where(swap & (index == source), target_role, roster.role),
    )
    formation_anchor = jnp.where(
        (swap & (index == target))[:, None],
        source_anchor,
        jnp.where(
            (swap & (index == source))[:, None],
            target_anchor,
            roster.formation_anchor,
        ),
    )
    alignment_valid = valid_team & has_one_goalkeeper & has_one_role_source
    return (
        roster._replace(role=role, formation_anchor=formation_anchor),
        alignment_valid,
        source,
        target,
        swap,
    )


def _swap_match_setup_slots(
    setup: MatchSetup,
    source: jax.Array,
    target: jax.Array,
    enabled: jax.Array,
) -> MatchSetup:
    """GK role/anchor permutation을 immutable halftime slot snapshot에도 반영한다."""

    positions = setup.second_half_positions
    distance_squared = setup.second_half_distance_squared
    swapped_positions = positions.at[source].set(positions[target])
    swapped_positions = swapped_positions.at[target].set(positions[source])
    swapped_distance = distance_squared.at[source].set(distance_squared[target])
    swapped_distance = swapped_distance.at[target].set(distance_squared[source])
    return setup._replace(
        second_half_positions=jnp.where(enabled, swapped_positions, positions),
        second_half_distance_squared=jnp.where(
            enabled, swapped_distance, distance_squared
        ),
    )


def apply_set_piece_taker(
    state,
    roster,
    request: SetPieceTakerRequest,
) -> SetPieceTakerResult:
    """열린 restart의 taker를 active teammate로 원자적으로 바꾼다."""

    count = state.players.position.shape[0]
    player = jnp.asarray(request.player_index, dtype=jnp.int32)
    safe_player = jnp.clip(player, 0, count - 1)
    team = jnp.asarray(request.team, dtype=jnp.int32)
    valid_team = (team == TEAM_0) | (team == TEAM_1)
    valid_player = (player >= 0) & (player < count)
    restart_matches = (
        restart_active(state.restart.kind)
        & (state.restart.kind != RK_GK_HOLD)
        & (~state.ball.live)
        & (state.restart.team == team)
        & (state.restart.kind == request.restart_kind)
    )
    eligible = (
        valid_player
        & state.players.active[safe_player]
        & (roster.team_id[safe_player] == team)
    )
    applied = request.enabled & valid_team & restart_matches & eligible
    candidate = state._replace(
        restart=state.restart._replace(taker=player),
        restart_taker_approach=state.restart_layout_ready,
    )
    return SetPieceTakerResult(
        state=jax.tree.map(
            lambda new, old: jnp.where(applied, new, old), candidate, state
        ),
        applied=applied,
    )


def _apply_substitution(
    state,
    roster,
    manager: ManagerState,
    request: SubstitutionRequest,
    *,
    enforce_goalkeeper_count: bool,
    limits: ManagementLimits = DEFAULT_MANAGEMENT_LIMITS,
    sampling: RosterSampling = DEFAULT_ROSTER_SAMPLING,
) -> SubstitutionResult:
    """Dead-ball chunk 경계에서 roster slot identity와 ability를 원자적으로 교체한다."""

    count = state.players.position.shape[0]
    outgoing = jnp.asarray(request.outgoing_index, dtype=jnp.int32)
    safe_outgoing = jnp.clip(outgoing, 0, count - 1)
    team = jnp.asarray(request.team, dtype=jnp.int32)
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    valid_team = (team == TEAM_0) | (team == TEAM_1)
    management_open = _management_open(state)
    outgoing_valid = (
        (outgoing >= 0)
        & (outgoing < count)
        & state.players.active[safe_outgoing]
        & (roster.team_id[safe_outgoing] == team)
    )
    identity_unique = ~jnp.any(roster.player_id == request.incoming_player_id)
    reach_margin = request.incoming_reach_height - request.incoming_height
    profile_valid = (
        (request.incoming_player_id >= 0)
        & (request.incoming_max_speed >= sampling.min_max_speed_mps)
        & (request.incoming_max_speed <= sampling.max_max_speed_mps)
        & (request.incoming_height >= sampling.min_height_m)
        & (request.incoming_height <= sampling.max_height_m)
        & (reach_margin >= sampling.min_reach_margin_m)
        & (reach_margin <= sampling.max_reach_margin_m)
        & (request.incoming_ball_control >= sampling.min_ball_control)
        & (request.incoming_ball_control <= sampling.max_ball_control)
        & (request.incoming_endurance_factor >= sampling.min_endurance_factor)
        & (request.incoming_endurance_factor <= sampling.max_endurance_factor)
        & (request.incoming_yellow_cards >= 0)
        & (request.incoming_yellow_cards < 2)
    )
    candidate_goalkeeper = roster.is_goalkeeper.at[safe_outgoing].set(
        request.incoming_is_goalkeeper
    )
    goalkeeper_count = jnp.sum(
        state.players.active & (roster.team_id == safe_team) & candidate_goalkeeper
    )
    goalkeeper_valid = (
        goalkeeper_count == 1 if enforce_goalkeeper_count else jnp.bool_(True)
    )
    new_window = manager.last_window_control_tick[safe_team] != state.control_tick
    window_charge = new_window & (~_halftime_substitution_interval(state))
    substitution_available = (
        manager.substitutions_used[safe_team] < limits.max_substitutions_per_team
    )
    window_available = (~window_charge) | (
        manager.windows_used[safe_team] < limits.max_windows_per_team
    )
    applied = (
        request.enabled
        & valid_team
        & management_open
        & outgoing_valid
        & identity_unique
        & profile_valid
        & goalkeeper_valid
        & substitution_available
        & window_available
    )

    forward = jnp.stack(
        (
            state.attack_direction[safe_team],
            jnp.asarray(0.0, dtype=state.players.position.dtype),
        )
    )
    player_candidate = state.players._replace(
        velocity=state.players.velocity.at[safe_outgoing].set(0.0),
        body_forward=state.players.body_forward.at[safe_outgoing].set(forward),
        on_pitch=state.players.on_pitch.at[safe_outgoing].set(True),
        sent_off=state.players.sent_off.at[safe_outgoing].set(False),
        stamina_long=state.players.stamina_long.at[safe_outgoing].set(1.0),
        stamina_short=state.players.stamina_short.at[safe_outgoing].set(1.0),
        challenge_recovery_substeps=state.players.challenge_recovery_substeps.at[
            safe_outgoing
        ].set(0),
        contact_lock_substeps=state.players.contact_lock_substeps.at[safe_outgoing].set(
            0
        ),
        aerial_recovery_substeps=state.players.aerial_recovery_substeps.at[
            safe_outgoing
        ].set(0),
        possession_loss_lock_substeps=state.players.possession_loss_lock_substeps.at[
            safe_outgoing
        ].set(0),
        yellow_cards=state.players.yellow_cards.at[safe_outgoing].set(
            request.incoming_yellow_cards
        ),
    )
    roster_candidate = roster._replace(
        player_id=roster.player_id.at[safe_outgoing].set(request.incoming_player_id),
        is_goalkeeper=roster.is_goalkeeper.at[safe_outgoing].set(
            request.incoming_is_goalkeeper
        ),
        max_speed=roster.max_speed.at[safe_outgoing].set(request.incoming_max_speed),
        height=roster.height.at[safe_outgoing].set(request.incoming_height),
        reach_height=roster.reach_height.at[safe_outgoing].set(
            request.incoming_reach_height
        ),
        ball_control=roster.ball_control.at[safe_outgoing].set(
            request.incoming_ball_control
        ),
        endurance_factor=roster.endurance_factor.at[safe_outgoing].set(
            request.incoming_endurance_factor
        ),
    )
    last_contact_actor = state.possession.last_contact.actor == safe_outgoing
    next_contact = jax.tree.map(
        lambda empty, old: jnp.where(last_contact_actor, empty, old),
        empty_contact_result(),
        state.possession.last_contact,
    )
    state_candidate = state._replace(
        players=player_candidate,
        possession=state.possession._replace(last_contact=next_contact),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
    )
    manager_candidate = manager._replace(
        substitutions_used=manager.substitutions_used.at[safe_team].add(1),
        windows_used=manager.windows_used.at[safe_team].add(
            window_charge.astype(jnp.int32)
        ),
        last_window_control_tick=manager.last_window_control_tick.at[safe_team].set(
            state.control_tick
        ),
    )

    def select(new, old):
        return jnp.where(applied, new, old)

    return SubstitutionResult(
        state=jax.tree.map(select, state_candidate, state),
        roster=jax.tree.map(select, roster_candidate, roster),
        manager=jax.tree.map(select, manager_candidate, manager),
        applied=applied,
    )


def apply_substitution(
    state,
    roster,
    manager: ManagerState,
    request: SubstitutionRequest,
    *,
    limits: ManagementLimits = DEFAULT_MANAGEMENT_LIMITS,
    sampling: RosterSampling = DEFAULT_ROSTER_SAMPLING,
) -> SubstitutionResult:
    """Dead-ball chunk 경계에서 roster slot identity와 ability를 원자적으로 교체한다."""

    return _apply_substitution(
        state,
        roster,
        manager,
        request,
        enforce_goalkeeper_count=True,
        limits=limits,
        sampling=sampling,
    )


def _apply_bench_substitution(
    state,
    roster,
    manager: ManagerState,
    bench: BenchRoster,
    request: BenchSubstitutionRequest,
    *,
    enforce_goalkeeper_count: bool,
    limits: ManagementLimits = DEFAULT_MANAGEMENT_LIMITS,
    sampling: RosterSampling = DEFAULT_ROSTER_SAMPLING,
) -> BenchSubstitutionResult:
    """등록 bench index를 authoritative profile로 변환해 한 slot을 교체한다."""

    if bench.valid.ndim != 2 or bench.valid.shape[0] != 2:
        raise ValueError("bench field는 [2, bench_capacity] shape여야 한다")
    for leaf in jax.tree.leaves(bench):
        if leaf.shape != bench.valid.shape:
            raise ValueError("모든 bench field는 같은 shape여야 한다")
    width = bench.valid.shape[1]
    if width == 0:
        return BenchSubstitutionResult(state, roster, manager, bench, jnp.bool_(False))
    team = jnp.asarray(request.team, dtype=jnp.int32)
    safe_team = jnp.clip(team, TEAM_0, TEAM_1)
    incoming = jnp.asarray(request.incoming_bench_index, dtype=jnp.int32)
    safe_incoming = jnp.clip(incoming, 0, width - 1)
    incoming_valid = (
        ((team == TEAM_0) | (team == TEAM_1))
        & (incoming >= 0)
        & (incoming < width)
        & bench.valid[safe_team, safe_incoming]
        & bench.available[safe_team, safe_incoming]
    )
    explicit = SubstitutionRequest(
        enabled=request.enabled & incoming_valid,
        team=team,
        outgoing_index=request.outgoing_index,
        incoming_player_id=bench.player_id[safe_team, safe_incoming],
        incoming_is_goalkeeper=bench.is_goalkeeper[safe_team, safe_incoming],
        incoming_max_speed=bench.max_speed[safe_team, safe_incoming],
        incoming_height=bench.height[safe_team, safe_incoming],
        incoming_reach_height=bench.reach_height[safe_team, safe_incoming],
        incoming_ball_control=bench.ball_control[safe_team, safe_incoming],
        incoming_endurance_factor=bench.endurance_factor[safe_team, safe_incoming],
        incoming_yellow_cards=jnp.int32(0),
    )
    result = _apply_substitution(
        state,
        roster,
        manager,
        explicit,
        enforce_goalkeeper_count=enforce_goalkeeper_count,
        limits=limits,
        sampling=sampling,
    )
    available = bench.available.at[safe_team, safe_incoming].set(
        jnp.where(
            result.applied, jnp.bool_(False), bench.available[safe_team, safe_incoming]
        )
    )
    return BenchSubstitutionResult(
        result.state,
        result.roster,
        result.manager,
        bench._replace(available=available),
        result.applied,
    )


def apply_bench_substitution(
    state,
    roster,
    manager: ManagerState,
    bench: BenchRoster,
    request: BenchSubstitutionRequest,
    *,
    limits: ManagementLimits = DEFAULT_MANAGEMENT_LIMITS,
    sampling: RosterSampling = DEFAULT_ROSTER_SAMPLING,
) -> BenchSubstitutionResult:
    """등록 bench index를 authoritative profile로 변환해 한 slot을 교체한다."""

    return _apply_bench_substitution(
        state,
        roster,
        manager,
        bench,
        request,
        enforce_goalkeeper_count=True,
        limits=limits,
        sampling=sampling,
    )


def apply_bench_substitutions(
    state,
    roster,
    manager: ManagerState,
    bench: BenchRoster,
    request: BenchSubstitutionBatchRequest,
    *,
    setup: MatchSetup,
    limits: ManagementLimits = DEFAULT_MANAGEMENT_LIMITS,
    sampling: RosterSampling = DEFAULT_ROSTER_SAMPLING,
) -> BenchSubstitutionBatchResult:
    """한 team의 여러 bench 교체를 한 window에서 전부 적용하거나 전부 거부한다.

    요청 폭은 static compile shape다. ``enabled=False``인 slot은 padding이며, 활성
    요청의 team은 모두 같아야 한다. 이 함수는 management 경계 전용이므로 일반
    physics rollout graph에는 포함되지 않는다.
    """

    _validate_match_setup(state, setup)
    if bench.valid.ndim != 2 or bench.valid.shape[0] != 2:
        raise ValueError("bench field는 [2, bench_capacity] shape여야 한다")
    for leaf in jax.tree.leaves(bench):
        if leaf.shape != bench.valid.shape:
            raise ValueError("모든 bench field는 같은 shape여야 한다")
    shape = request.enabled.shape
    if request.enabled.ndim != 1:
        raise ValueError("batch substitution request는 rank-1이어야 한다")
    for leaf in jax.tree.leaves(request):
        if leaf.shape != shape:
            raise ValueError("batch substitution request field는 같은 shape여야 한다")
    if shape[0] == 0:
        return BenchSubstitutionBatchResult(
            state, roster, manager, bench, jnp.bool_(False), jnp.int32(0), setup
        )

    enabled = jnp.asarray(request.enabled, dtype=jnp.bool_)
    team = jnp.asarray(request.team, dtype=jnp.int32)
    outgoing = jnp.asarray(request.outgoing_index, dtype=jnp.int32)
    incoming = jnp.asarray(request.incoming_bench_index, dtype=jnp.int32)
    enabled_count = jnp.sum(enabled.astype(jnp.int32))
    first_slot = jnp.argmax(enabled.astype(jnp.int32))
    transaction_team = team[first_slot]
    coherent_team = jnp.all((~enabled) | (team == transaction_team))

    # 같은 pitch slot이나 bench slot을 한 transaction에서 두 번 바꾸는 요청은
    # 순서 의존성이 생기므로 사전에 원자적으로 거부한다.
    distinct_pair = ~jnp.eye(shape[0], dtype=jnp.bool_)
    enabled_pair = enabled[:, None] & enabled[None, :] & distinct_pair
    duplicate_outgoing = jnp.any(
        enabled_pair & (outgoing[:, None] == outgoing[None, :])
    )
    duplicate_incoming = jnp.any(
        enabled_pair & (incoming[:, None] == incoming[None, :])
    )
    bench_width = bench.valid.shape[1]
    if bench_width == 0:
        initial_identity_unique = jnp.bool_(False)
    else:
        safe_request_team = jnp.clip(team, TEAM_0, TEAM_1)
        safe_request_incoming = jnp.clip(incoming, 0, bench_width - 1)
        incoming_identity = bench.player_id[safe_request_team, safe_request_incoming]
        initial_identity_unique = ~jnp.any(
            enabled[:, None] & (incoming_identity[:, None] == roster.player_id[None, :])
        )
    preflight = (
        (enabled_count > 0)
        & coherent_team
        & (~duplicate_outgoing)
        & (~duplicate_incoming)
        & initial_identity_unique
    )

    def apply_one(carry, values):
        current_state, current_roster, current_manager, current_bench, all_valid = carry
        slot_enabled, slot_team, slot_outgoing, slot_incoming = values
        result = _apply_bench_substitution(
            current_state,
            current_roster,
            current_manager,
            current_bench,
            BenchSubstitutionRequest(
                enabled=slot_enabled & preflight,
                team=slot_team,
                outgoing_index=slot_outgoing,
                incoming_bench_index=slot_incoming,
            ),
            enforce_goalkeeper_count=False,
            limits=limits,
            sampling=sampling,
        )
        slot_valid = (~slot_enabled) | result.applied
        return (
            result.state,
            result.roster,
            result.manager,
            result.bench,
            all_valid & slot_valid,
        ), None

    initial = (state, roster, manager, bench, preflight)
    (
        (candidate_state, candidate_roster, candidate_manager, candidate_bench, valid),
        _,
    ) = jax.lax.scan(apply_one, initial, (enabled, team, outgoing, incoming))
    safe_transaction_team = jnp.clip(transaction_team, TEAM_0, TEAM_1)
    final_goalkeeper_count = jnp.sum(
        candidate_state.players.active
        & (candidate_roster.team_id == safe_transaction_team)
        & candidate_roster.is_goalkeeper
    )
    valid = valid & (final_goalkeeper_count == 1)
    (
        candidate_roster,
        goalkeeper_role_valid,
        role_source,
        role_target,
        role_swap,
    ) = _align_goalkeeper_role(
        candidate_roster, candidate_state.players.active, transaction_team
    )
    valid = valid & goalkeeper_role_valid
    candidate_setup = _swap_match_setup_slots(
        setup, role_source, role_target, role_swap
    )

    def select(candidate, original):
        return jnp.where(valid, candidate, original)

    return BenchSubstitutionBatchResult(
        state=jax.tree.map(select, candidate_state, state),
        roster=jax.tree.map(select, candidate_roster, roster),
        manager=jax.tree.map(select, candidate_manager, manager),
        bench=jax.tree.map(select, candidate_bench, bench),
        setup=jax.tree.map(select, candidate_setup, setup),
        applied=valid,
        applied_count=jnp.where(valid, enabled_count, jnp.int32(0)),
    )


def apply_acting_goalkeeper(
    state,
    roster,
    request: ActingGoalkeeperRequest,
    *,
    setup: MatchSetup,
) -> ActingGoalkeeperResult:
    """Management stoppage에서 active teammate 한 명을 goalkeeper로 지정한다."""

    _validate_match_setup(state, setup)
    count = state.players.position.shape[0]
    player = jnp.asarray(request.player_index, dtype=jnp.int32)
    safe_player = jnp.clip(player, 0, count - 1)
    team = jnp.asarray(request.team, dtype=jnp.int32)
    valid_team = (team == TEAM_0) | (team == TEAM_1)
    valid_player = (player >= 0) & (player < count)
    eligible = (
        valid_player
        & state.players.active[safe_player]
        & (roster.team_id[safe_player] == team)
    )
    applied = (
        request.enabled
        & valid_team
        & _management_open(state)
        & (~state.restart_layout_ready)
        & eligible
    )
    team_mask = roster.team_id == team
    candidate = roster._replace(
        is_goalkeeper=jnp.where(
            team_mask,
            jnp.arange(count, dtype=jnp.int32) == player,
            roster.is_goalkeeper,
        )
    )
    (
        candidate,
        goalkeeper_role_valid,
        role_source,
        role_target,
        role_swap,
    ) = _align_goalkeeper_role(
        candidate, state.players.active, team
    )
    applied = applied & goalkeeper_role_valid
    candidate_setup = _swap_match_setup_slots(
        setup, role_source, role_target, role_swap
    )
    return ActingGoalkeeperResult(
        roster=jax.tree.map(
            lambda new, old: jnp.where(applied, new, old), candidate, roster
        ),
        setup=jax.tree.map(
            lambda new, old: jnp.where(applied, new, old), candidate_setup, setup
        ),
        applied=applied,
    )


__all__ = [
    "ActingGoalkeeperRequest",
    "ActingGoalkeeperResult",
    "BenchSubstitutionBatchRequest",
    "BenchSubstitutionBatchResult",
    "BenchSubstitutionRequest",
    "BenchSubstitutionResult",
    "ManagementLimits",
    "ManagerState",
    "SetPieceTakerRequest",
    "SetPieceTakerResult",
    "SubstitutionRequest",
    "SubstitutionResult",
    "apply_acting_goalkeeper",
    "apply_bench_substitution",
    "apply_bench_substitutions",
    "apply_set_piece_taker",
    "apply_substitution",
    "empty_acting_goalkeeper_request",
    "empty_bench_substitution_request",
    "empty_manager_state",
    "empty_set_piece_taker_request",
    "empty_substitution_request",
]
