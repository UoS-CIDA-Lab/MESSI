"""Internal fixed-shape substitution kernel for the public environment API."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Stadium
from footballworld.config.roster import player_profile_values_valid
from footballworld.config.roster_sampling import RosterSampling
from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RESTART_COUNT,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    TEAM_0,
    TEAM_1,
    YELLOW_CARD_SEND_OFF_COUNT,
)
from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_NONE,
    LAW11_NONE,
    MECHANISM_NONE,
    OUTCOME_NONE,
    ContactResult,
)
from footballworld.core.state import State, initial_player_body_forward
from footballworld.environment.clock import regulation_elapsed_ticks
from footballworld.rules.offside import OffsideState
from footballworld.rules.restart import select_restart_taker


class SubstitutionRequest(NamedTuple):
    """One fixed-shape roster-slot replacement selected by a caller policy."""

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


class _SubstitutionKernelResult(NamedTuple):
    state: State
    offside_state: OffsideState
    applied: jax.Array


def _empty_contact() -> ContactResult:
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


def request_within_roster_domain(
    request: SubstitutionRequest, config: RosterSampling
) -> jax.Array:
    """Return whether an incoming profile fits the immutable model domain."""

    reach_margin = request.incoming_reach_height - request.incoming_height
    return (
        (request.incoming_max_speed >= config.min_max_speed_mps)
        & (request.incoming_max_speed <= config.max_max_speed_mps)
        & (request.incoming_height >= config.min_height_m)
        & (request.incoming_height <= config.max_height_m)
        & (reach_margin >= config.min_reach_margin_m)
        & (reach_margin <= config.max_reach_margin_m)
        & (request.incoming_ball_control >= config.min_ball_control)
        & (request.incoming_ball_control <= config.max_ball_control)
        & (request.incoming_endurance_factor >= config.min_endurance_factor)
        & (request.incoming_endurance_factor <= config.max_endurance_factor)
    )


def _validate_layout(
    state: State,
    offside_state: OffsideState,
    request: SubstitutionRequest,
) -> None:
    player_count = state.players.position.shape[0]
    if player_count <= 0:
        raise ValueError("substitution requires at least one roster slot")
    if offside_state.flagged.shape != (player_count,):
        raise ValueError("offside flagged shape must match roster slots")
    fields = request._asdict()
    for name, value in fields.items():
        if jnp.shape(value) != ():
            raise ValueError(f"request {name} must be scalar")
    for name in ("enabled", "incoming_is_goalkeeper"):
        if jnp.asarray(fields[name]).dtype != jnp.bool_:
            raise TypeError(f"request {name} must have bool dtype")
    for name in (
        "team",
        "outgoing_index",
        "incoming_player_id",
        "incoming_yellow_cards",
    ):
        if jnp.asarray(fields[name]).dtype != jnp.int32:
            raise TypeError(f"request {name} must have int32 dtype")
    for name in (
        "incoming_max_speed",
        "incoming_height",
        "incoming_reach_height",
        "incoming_ball_control",
        "incoming_endurance_factor",
    ):
        if jnp.asarray(fields[name]).dtype != jnp.float32:
            raise TypeError(f"request {name} must have float32 dtype")


def _management_stoppage_open(
    state: State,
    offside_state: OffsideState,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
) -> jax.Array:
    """Return the shared referee-approved management boundary predicate.

    A goalkeeper hold is normally live play.  The sole exception is a
    malformed hold whose team has no active goalkeeper: exposing that rare
    boundary lets a manager substitute or designate an acting goalkeeper
    before the environment's corner-expiry fail-safe advances the match.
    """

    players = state.players
    player_count = players.position.shape[0]
    active_count = jnp.stack(
        [
            jnp.sum(players.active & (players.team_id == team))
            for team in (TEAM_0, TEAM_1)
        ]
    )
    regulation_tick = regulation_elapsed_ticks(state)
    match_open = (
        (regulation_tick < jnp.int32(fulltime_tick))
        | (state.restart.kind == RK_PENALTY)
    ) & jnp.all(active_count >= jnp.asarray(minimum_team_players, dtype=jnp.int32))

    restart_team_valid = (state.restart.team == TEAM_0) | (state.restart.team == TEAM_1)
    taker = state.restart.taker
    taker_in_range = (taker >= 0) & (taker < player_count)
    safe_taker = jnp.clip(taker, 0, player_count - 1)
    coherent_restart = (
        (~state.ball.live)
        & (state.restart.kind > RK_NONE)
        & (state.restart.kind < RESTART_COUNT)
        & (state.restart.kind != RK_GK_HOLD)
        & restart_team_valid
        & (state.restart.substeps_remaining >= 0)
        & taker_in_range
        & players.active[safe_taker]
        & (players.team_id[safe_taker] == state.restart.team)
        & (
            (state.restart.kind == RK_FREEKICK)
            | ((state.restart.kind == RK_OFFSIDE) & state.restart.indirect)
            | (
                (state.restart.kind != RK_FREEKICK)
                & (state.restart.kind != RK_OFFSIDE)
                & (~state.restart.indirect)
            )
        )
    )
    missing_hold_goalkeeper = (
        (state.restart.kind == RK_GK_HOLD)
        & restart_team_valid
        & (~jnp.any(
            players.active
            & players.is_goalkeeper
            & (players.team_id == state.restart.team)
        ))
    )
    release = state.restart_release
    causal_stoppage_clean = (
        (state.possession.team == NO_TEAM)
        & (state.possession.player == NO_PLAYER)
        & (state.possession.control_ticks == 0)
        & (~release.active)
        & (~release.untouched)
        & (release.kind == RK_NONE)
        & (release.team == NO_TEAM)
        & (release.taker == NO_PLAYER)
        & (~release.indirect)
        & (~release.law11_direct_exempt)
        & (release.release_mechanism == MECHANISM_NONE)
        & (state.gk_backpass_team == NO_TEAM)
        & (~jnp.any(offside_state.flagged))
        & (offside_state.direct_exempt_team == NO_TEAM)
    )
    return match_open & (
        (coherent_restart & causal_stoppage_clean) | missing_hold_goalkeeper
    )


def _apply_substitution(
    state: State,
    offside_state: OffsideState,
    request: SubstitutionRequest,
    *,
    fulltime_tick: int,
    minimum_team_players: tuple[int, int],
    body: BodyContact,
    stadium: Stadium,
) -> _SubstitutionKernelResult:
    """Apply one facade-validated rare event or return an exact no-op."""

    _validate_layout(state, offside_state, request)
    players = state.players
    player_count = players.position.shape[0]
    outgoing = request.outgoing_index
    outgoing_in_range = (outgoing >= 0) & (outgoing < player_count)
    safe_outgoing = jnp.clip(outgoing, 0, player_count - 1)
    request_team_valid = (request.team == TEAM_0) | (request.team == TEAM_1)

    management_open = _management_stoppage_open(
        state,
        offside_state,
        fulltime_tick=fulltime_tick,
        minimum_team_players=minimum_team_players,
    )
    outgoing_valid = (
        outgoing_in_range
        & players.on_pitch[safe_outgoing]
        & (~players.sent_off[safe_outgoing])
        & (players.team_id[safe_outgoing] == request.team)
    )
    identity_unique = ~jnp.any(players.player_id == request.incoming_player_id)
    dtype = players.max_speed.dtype
    profile_valid = player_profile_values_valid(
        request.incoming_player_id,
        request.incoming_max_speed,
        request.incoming_height,
        request.incoming_reach_height,
        request.incoming_ball_control,
        request.incoming_endurance_factor,
        head_radius=jnp.asarray(body.head_radius_m, dtype=dtype),
    )
    discipline_valid = (request.incoming_yellow_cards >= 0) & (
        request.incoming_yellow_cards < YELLOW_CARD_SEND_OFF_COUNT
    )

    candidate_goalkeepers = players.is_goalkeeper.at[safe_outgoing].set(
        request.incoming_is_goalkeeper
    )
    candidate_active = players.active.at[safe_outgoing].set(True)
    goalkeeper_count = jnp.stack(
        [
            jnp.sum(
                candidate_active & (players.team_id == team) & candidate_goalkeepers
            )
            for team in (TEAM_0, TEAM_1)
        ]
    )
    safe_request_team = jnp.clip(request.team, TEAM_0, TEAM_1)
    goalkeeper_valid = goalkeeper_count[safe_request_team] == 1
    applied = (
        request.enabled
        & management_open
        & request_team_valid
        & outgoing_valid
        & identity_unique
        & profile_valid
        & discipline_valid
        & goalkeeper_valid
    )

    def replace_slot(operand):
        current_state, current_offside = operand
        current = current_state.players
        body_forward = initial_player_body_forward(
            current.team_id, current_state.attack_direction
        )[safe_outgoing]
        replacement = current._replace(
            velocity=current.velocity.at[safe_outgoing].set(
                jnp.zeros(2, dtype=current.velocity.dtype)
            ),
            body_forward=current.body_forward.at[safe_outgoing].set(body_forward),
            gaze_yaw=current.gaze_yaw.at[safe_outgoing].set(0.0),
            player_id=current.player_id.at[safe_outgoing].set(
                request.incoming_player_id
            ),
            on_pitch=current.on_pitch.at[safe_outgoing].set(True),
            sent_off=current.sent_off.at[safe_outgoing].set(False),
            is_goalkeeper=current.is_goalkeeper.at[safe_outgoing].set(
                request.incoming_is_goalkeeper
            ),
            max_speed=current.max_speed.at[safe_outgoing].set(
                request.incoming_max_speed
            ),
            reach_height=current.reach_height.at[safe_outgoing].set(
                request.incoming_reach_height
            ),
            height=current.height.at[safe_outgoing].set(request.incoming_height),
            ball_control=current.ball_control.at[safe_outgoing].set(
                request.incoming_ball_control
            ),
            endurance_factor=current.endurance_factor.at[safe_outgoing].set(
                request.incoming_endurance_factor
            ),
            stamina_long=current.stamina_long.at[safe_outgoing].set(1.0),
            stamina_short=current.stamina_short.at[safe_outgoing].set(1.0),
            challenge_recovery_substeps=(
                current.challenge_recovery_substeps.at[safe_outgoing].set(0)
            ),
            contact_lock_substeps=(
                current.contact_lock_substeps.at[safe_outgoing].set(0)
            ),
            aerial_recovery_substeps=(
                current.aerial_recovery_substeps.at[safe_outgoing].set(0)
            ),
            possession_loss_lock_substeps=(
                current.possession_loss_lock_substeps.at[safe_outgoing].set(0)
            ),
            yellow_cards=current.yellow_cards.at[safe_outgoing].set(
                request.incoming_yellow_cards
            ),
        )
        last_actor_is_outgoing = (
            current_state.possession.last_contact.actor == safe_outgoing
        )
        last_contact = jax.tree.map(
            lambda empty, old: jnp.where(last_actor_is_outgoing, empty, old),
            _empty_contact(),
            current_state.possession.last_contact,
        )
        provisional = current_state._replace(
            players=replacement,
            possession=current_state.possession._replace(last_contact=last_contact),
        )
        replacement_taker = select_restart_taker(
            provisional,
            provisional.restart.kind,
            provisional.restart.team,
            provisional.ball.position,
            stadium=stadium,
        )
        next_taker = jnp.where(
            current_state.restart.taker == safe_outgoing,
            replacement_taker,
            current_state.restart.taker,
        ).astype(jnp.int32)
        return (
            provisional._replace(
                restart=provisional.restart._replace(taker=next_taker),
                restart_layout_ready=jnp.bool_(False),
            ),
            current_offside,
        )

    next_state, next_offside = jax.lax.cond(
        applied,
        replace_slot,
        lambda operand: operand,
        (state, offside_state),
    )
    return _SubstitutionKernelResult(next_state, next_offside, applied)


__all__ = ["SubstitutionRequest"]
