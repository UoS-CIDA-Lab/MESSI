"""Production match reset 경계의 roster, bench와 dynamic state 초기화."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.constants import NO_PLAYER, NO_TEAM, RK_KICKOFF, RK_NONE
from messi.core.contact import MECHANISM_NONE, empty_contact_result
from messi.core.formation import formation_positions
from messi.core.geometry import BallGeometry
from messi.core.numeric import require_float32_field_half_extent
from messi.core.randomness import RandomEvent, event_random_key, validate_prng_key
from messi.core.roster import (
    BenchRoster,
    PlayerProfile,
    RealizedRoster,
    RosterSampling,
    TeamRoster,
    _realize_bench,
    _realize_roster,
    _unselected_candidate_profiles,
)
from messi.core.state import (
    BallState,
    PlayerState,
    PossessionState,
    RestartReleaseProvenance,
    RestartState,
    State,
    initial_player_body_forward,
)
from messi.environment.episode import (
    MatchSetup,
    _kickoff_taker_candidates,
    make_match_setup,
)
from messi.environment.management import ManagerState, empty_manager_state


class ResetResult(NamedTuple):
    state: State
    roster: object
    bench: BenchRoster
    manager: ManagerState
    setup: MatchSetup
    candidate_index: jax.Array


def reset_environment(
    key: jax.Array,
    team_0: TeamRoster | None = None,
    team_1: TeamRoster | None = None,
    *,
    team_0_bench: Sequence[PlayerProfile] | None = None,
    team_1_bench: Sequence[PlayerProfile] | None = None,
    players_per_team: int = 11,
    kickoff_team: int | None = None,
    sampling: RosterSampling | None = None,
    formation_ids: tuple[int, int] | None = None,
    ball: BallGeometry = BallGeometry(),
    field_half_extent: tuple[float, float] = (52.5, 34.0),
) -> ResetResult:
    """한 경기의 고정 roster와 pre-kickoff dynamic state를 생성한다."""

    validate_prng_key(key)
    field_half_extent = require_float32_field_half_extent(field_half_extent)
    extent_float32 = np.asarray(field_half_extent, dtype=np.float32)
    sampling = RosterSampling() if sampling is None else sampling
    declared_bench_ids = tuple(
        profile.player_id
        for profiles in (team_0_bench, team_1_bench)
        if profiles is not None
        for profile in profiles
        if isinstance(profile, PlayerProfile) and profile.player_id is not None
    )
    realized: RealizedRoster = _realize_roster(
        team_0,
        team_1,
        key=key,
        players_per_team=players_per_team,
        sampling=sampling,
        formation_ids=formation_ids,
        reserved_player_ids=declared_bench_ids,
    )
    auto_team_0_bench = (
        team_0_bench is None
        and team_0 is not None
        and team_0.players is not None
        and len(team_0.players) > players_per_team
    )
    auto_team_1_bench = (
        team_1_bench is None
        and team_1 is not None
        and team_1.players is not None
        and len(team_1.players) > players_per_team
    )
    if auto_team_0_bench or auto_team_1_bench:
        remaining_0, remaining_1 = _unselected_candidate_profiles(
            TeamRoster() if team_0 is None else team_0,
            TeamRoster() if team_1 is None else team_1,
            realized.candidate_index,
            players_per_team=players_per_team,
            sampling=sampling,
            reserved_player_ids=declared_bench_ids,
        )
        team_0_bench = remaining_0 if auto_team_0_bench else team_0_bench
        team_1_bench = remaining_1 if auto_team_1_bench else team_1_bench
    active_player_ids = tuple(
        int(identity) for identity in np.asarray(realized.roster.player_id)
    )
    bench = _realize_bench(
        team_0_bench,
        team_1_bench,
        key=key,
        sampling=sampling,
        reserved_player_ids=active_player_ids,
    )
    if kickoff_team is None:
        kickoff_key = event_random_key(key, RandomEvent.OPENING_CANDIDATE, 0)
        kickoff_team = int(jax.random.randint(kickoff_key, (), 0, 2))
    elif (
        isinstance(kickoff_team, bool)
        or not isinstance(kickoff_team, Integral)
        or int(kickoff_team) not in (0, 1)
    ):
        raise ValueError("kickoff_team은 0, 1 또는 None이어야 한다")
    kickoff_team = int(kickoff_team)

    roster = realized.roster
    count = roster.team_id.shape[0]
    attack_direction = jnp.asarray((1.0, -1.0), dtype=jnp.float32)
    position = formation_positions(
        roster,
        attack_direction,
        field_half_extent=jnp.asarray(extent_float32, dtype=jnp.float32),
    )
    active = jnp.ones(count, dtype=jnp.bool_)
    players = PlayerState(
        position=position,
        velocity=jnp.zeros((count, 2), dtype=jnp.float32),
        body_forward=initial_player_body_forward(roster.team_id, attack_direction),
        on_pitch=active,
        sent_off=jnp.zeros(count, dtype=jnp.bool_),
        stamina_long=jnp.ones(count, dtype=jnp.float32),
        stamina_short=jnp.ones(count, dtype=jnp.float32),
        challenge_recovery_substeps=jnp.zeros(count, dtype=jnp.int32),
        contact_lock_substeps=jnp.zeros(count, dtype=jnp.int32),
        aerial_recovery_substeps=jnp.zeros(count, dtype=jnp.int32),
        possession_loss_lock_substeps=jnp.zeros(count, dtype=jnp.int32),
        yellow_cards=jnp.zeros(count, dtype=jnp.int32),
    )
    eligible = _kickoff_taker_candidates(active, roster, kickoff_team)
    distance_squared = jnp.sum(position * position, axis=-1)
    taker = jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf)).astype(jnp.int32)
    state = State(
        control_tick=jnp.int32(0),
        ball=BallState(
            position=jnp.asarray((0.0, 0.0, ball.radius_m), dtype=jnp.float32),
            velocity=jnp.zeros(3, dtype=jnp.float32),
            spin=jnp.zeros(3, dtype=jnp.float32),
            live=jnp.bool_(False),
        ),
        players=players,
        attack_direction=attack_direction,
        kickoff_team=jnp.int32(kickoff_team),
        possession=PossessionState(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=jnp.int32(NO_TEAM),
            control_ticks=jnp.int32(0),
            episode_start_tick=jnp.int32(-1),
            last_contact=empty_contact_result(),
        ),
        restart=RestartState(
            kind=jnp.int32(RK_KICKOFF),
            team=jnp.int32(kickoff_team),
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=jnp.bool_(False),
            opened_control_tick=jnp.int32(0),
        ),
        score=jnp.zeros(2, dtype=jnp.int32),
        restart_release=RestartReleaseProvenance(
            active=jnp.bool_(False),
            untouched=jnp.bool_(False),
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            law11_direct_exempt=jnp.bool_(False),
            release_mechanism=jnp.int32(MECHANISM_NONE),
        ),
        offside_flagged=jnp.zeros(count, dtype=jnp.bool_),
        offside_direct_exempt_team=jnp.int32(NO_TEAM),
        gk_backpass_team=jnp.int32(NO_TEAM),
        dead_ball_control_ticks=jnp.int32(0),
        first_half_wall_end_tick=jnp.int32(-1),
        first_half_live_extension_ticks=jnp.int32(0),
        penalty_completion_active=jnp.bool_(False),
        penalty_completion_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
        restart_taker_approach=jnp.bool_(False),
    )
    return ResetResult(
        state=state,
        roster=roster,
        bench=bench,
        manager=empty_manager_state(),
        setup=make_match_setup(state),
        candidate_index=realized.candidate_index,
    )


__all__ = ["ResetResult", "reset_environment"]
