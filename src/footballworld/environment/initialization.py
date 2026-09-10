"""Host-side construction of the minimal immutable rollout state."""

from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball
from footballworld.config.roster import Player, PlayerProfile
from footballworld.config.roster_sampling import RosterSampling
from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_KICKOFF,
    RK_NONE,
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
    PlayerState,
    PossessionState,
    RestartReleaseProvenance,
    RestartState,
    State,
    initial_player_body_forward,
)
from footballworld.environment.roster_sampling import (
    ProfileValues,
    sample_profile_values,
)
from footballworld.rules.offside import OffsideState, empty_offside_state
from footballworld.rules.restart import select_restart_taker


class Initialization(NamedTuple):
    """A rollout state and its separate clear Law 11 state."""

    state: State
    offside: OffsideState


def _validate_rosters(
    team_0: tuple[Player, ...],
    team_1: tuple[Player, ...],
) -> None:
    if not team_0 or not team_1:
        raise ValueError("both teams must contain at least one player")
    players = team_0 + team_1
    if any(type(player) is not Player for player in players):
        raise TypeError("team rosters must contain exactly Player entries")
    if any(type(player.profile) is not PlayerProfile for player in players):
        raise TypeError("starter profiles must be exactly PlayerProfile")

    player_ids = [player.profile.player_id for player in players]
    if any(
        not isinstance(player_id, Integral) or isinstance(player_id, (bool, np.bool_))
        for player_id in player_ids
    ):
        raise TypeError("starter player_id values must be non-boolean integers")
    if any(
        not 0 <= int(player_id) <= np.iinfo(np.int32).max for player_id in player_ids
    ):
        raise ValueError(
            "starter player_id values must lie in the int32 identity domain"
        )
    if len(set(map(int, player_ids))) != len(player_ids):
        raise ValueError("player_id values must be unique across both teams")

    for name, roster in (("team_0", team_0), ("team_1", team_1)):
        if any(type(player.profile.is_goalkeeper) is not bool for player in roster):
            raise TypeError(f"{name} is_goalkeeper values must be bool")
        if sum(player.profile.is_goalkeeper for player in roster) > 1:
            raise ValueError(f"{name} may contain at most one goalkeeper")


def _pack_rosters(
    team_0: tuple[Player, ...],
    team_1: tuple[Player, ...],
    body: BodyContact,
) -> tuple[np.ndarray, ...]:
    players = team_0 + team_1
    local_positions = np.asarray(
        [player.initial_position for player in players], dtype=np.float32
    )
    if local_positions.shape != (len(players), 2):
        raise ValueError("every initial_position must contain exactly two values")

    physical = np.asarray(
        [
            (
                player.profile.max_speed_mps,
                player.profile.height_m,
                player.profile.max_reach_height_m,
                player.profile.ball_control,
                player.profile.endurance_factor,
            )
            for player in players
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(local_positions)) or not np.all(np.isfinite(physical)):
        raise ValueError("roster positions and physical values must be finite")
    max_speed, stature, reach_height, ball_control, endurance = physical.T
    physically_valid = (
        (max_speed > 0.0)
        & (stature > 2.0 * body.head_radius_m)
        & (reach_height >= stature)
        & (ball_control >= 0.0)
        & (ball_control <= 1.0)
        & (endurance > 0.0)
    )
    if not np.all(physically_valid):
        raise ValueError("roster physical values are outside their valid domains")

    team_0_count = len(team_0)
    team_id = np.concatenate(
        (
            np.full(team_0_count, TEAM_0, dtype=np.int32),
            np.full(len(team_1), TEAM_1, dtype=np.int32),
        )
    )
    world_positions = local_positions.copy()
    world_positions[team_0_count:] *= -1.0
    player_id = np.asarray(
        [player.profile.player_id for player in players], dtype=np.int32
    )
    is_goalkeeper = np.asarray(
        [player.profile.is_goalkeeper for player in players], dtype=np.bool_
    )
    return world_positions, physical, team_id, player_id, is_goalkeeper


def initialize_state(
    team_0: Sequence[Player],
    team_1: Sequence[Player],
    *,
    kickoff_team: int = TEAM_0,
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
    roster_sampling: RosterSampling = RosterSampling(),
    sampling_key: jax.Array | None = None,
) -> Initialization:
    """Build one pre-kickoff phase state from folded rosters.

    Both rosters express ``initial_position`` in an attacking local frame where
    positive x points forward. Team 0 keeps that frame in world coordinates;
    team 1 is rotated by 180 degrees. This constructor does not project player
    positions for Law 8 or snap the selected taker to the centre mark.
    """

    if not isinstance(kickoff_team, Integral) or isinstance(
        kickoff_team, (bool, np.bool_)
    ):
        raise TypeError("kickoff_team must be a non-boolean integer")
    kickoff_team = int(kickoff_team)
    if kickoff_team not in (TEAM_0, TEAM_1):
        raise ValueError("kickoff_team must be TEAM_0 or TEAM_1")
    team_0 = tuple(team_0)
    team_1 = tuple(team_1)
    _validate_rosters(team_0, team_1)
    position, physical, team_id, player_id, is_goalkeeper = _pack_rosters(
        team_0, team_1, body
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
            minimum_height_m=float(
                np.nextafter(
                    np.float32(2.0 * body.head_radius_m),
                    np.float32(np.inf),
                )
            ),
        )
        physical = jnp.stack(sampled, axis=-1)

    player_count = position.shape[0]
    position = jnp.asarray(position, dtype=jnp.float32)
    team_id = jnp.asarray(team_id, dtype=jnp.int32)
    attack_direction = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    velocity = jnp.zeros((player_count, 2), dtype=jnp.float32)
    players = PlayerState(
        position=position,
        velocity=velocity,
        body_forward=initial_player_body_forward(team_id, attack_direction),
        gaze_yaw=jnp.zeros(player_count, dtype=jnp.float32),
        team_id=team_id,
        player_id=jnp.asarray(player_id, dtype=jnp.int32),
        on_pitch=jnp.ones(player_count, dtype=jnp.bool_),
        sent_off=jnp.zeros(player_count, dtype=jnp.bool_),
        is_goalkeeper=jnp.asarray(is_goalkeeper, dtype=jnp.bool_),
        max_speed=jnp.asarray(physical[..., 0], dtype=jnp.float32),
        reach_height=jnp.asarray(physical[..., 2], dtype=jnp.float32),
        height=jnp.asarray(physical[..., 1], dtype=jnp.float32),
        ball_control=jnp.asarray(physical[..., 3], dtype=jnp.float32),
        endurance_factor=jnp.asarray(physical[..., 4], dtype=jnp.float32),
        stamina_long=jnp.ones(player_count, dtype=jnp.float32),
        stamina_short=jnp.ones(player_count, dtype=jnp.float32),
        challenge_recovery_substeps=jnp.zeros(player_count, dtype=jnp.int32),
        contact_lock_substeps=jnp.zeros(player_count, dtype=jnp.int32),
        aerial_recovery_substeps=jnp.zeros(player_count, dtype=jnp.int32),
        possession_loss_lock_substeps=jnp.zeros(player_count, dtype=jnp.int32),
        yellow_cards=jnp.zeros(player_count, dtype=jnp.int32),
    )
    empty_contact = ContactResult(
        actor=jnp.int32(NO_PLAYER),
        mechanism=jnp.int32(MECHANISM_NONE),
        intent=jnp.int32(INTENT_MOVE),
        outcome=jnp.int32(OUTCOME_NONE),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=jnp.int32(LAW11_NONE),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_NONE),
    )
    position_ball = jnp.asarray([0.0, 0.0, ball_geometry.radius], dtype=jnp.float32)
    state = State(
        control_tick=jnp.int32(0),
        ball=BallState(
            position=position_ball,
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
            last_contact=empty_contact,
        ),
        restart=RestartState(
            kind=jnp.int32(RK_KICKOFF),
            team=jnp.int32(kickoff_team),
            substeps_remaining=jnp.int32(0),
            taker=jnp.int32(NO_PLAYER),
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
        gk_backpass_team=jnp.int32(NO_TEAM),
        restart_layout_ready=jnp.bool_(False),
    )
    taker = select_restart_taker(state, RK_KICKOFF, state.restart.team, position_ball)
    state = state._replace(restart=state.restart._replace(taker=taker))
    return Initialization(state=state, offside=empty_offside_state(player_count))
