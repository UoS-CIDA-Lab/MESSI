"""Pure JAX transitions for touch and selected-challenge Law 11 offences."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball
from footballworld.core.constants import NO_PLAYER, NO_TEAM, TEAM_COUNT
from footballworld.core.contact import (
    LAW11_DEFLECTION_NO_RESET,
    LAW11_DELIBERATE_PLAY_RESET,
    LAW11_DELIBERATE_SAVE_NO_RESET,
    LAW11_DIRECT_RESTART_EXEMPTION,
    ContactOccurrence,
)
from footballworld.core.state import State


class OffsideState(NamedTuple):
    """Players flagged by each team's latest relevant touch.

    ``direct_exempt_team`` retains a goal-kick, throw-in, or corner exemption
    until that team next touches the ball or an opponent deliberately plays it.
    """

    flagged: jax.Array
    direct_exempt_team: jax.Array


class OffsideEvent(NamedTuple):
    """First touch-based offside offence in a contact batch."""

    occurred: jax.Array
    actor: jax.Array
    team: jax.Array
    position: jax.Array
    time_fraction: jax.Array


class OffsideStep(NamedTuple):
    """Law 11 state and the first offence after ordered contacts."""

    state: OffsideState
    event: OffsideEvent


def empty_offside_state(player_count: int) -> OffsideState:
    """Create an inactive fixed-shape Law 11 state."""

    return OffsideState(
        flagged=jnp.zeros(player_count, dtype=bool),
        direct_exempt_team=jnp.asarray(NO_TEAM, dtype=jnp.int32),
    )


def clear_offside_state(state: OffsideState) -> OffsideState:
    """Clear Law 11 state after any stoppage or boundary event."""

    return OffsideState(
        flagged=jnp.zeros_like(state.flagged, dtype=bool),
        direct_exempt_team=jnp.asarray(NO_TEAM, dtype=jnp.int32),
    )


def player_front_support(
    state: State,
    *,
    player_position: jax.Array | None = None,
    body: BodyContact = BodyContact(),
) -> jax.Array:
    """Return playable-body front coordinates for both attack directions.

    The returned shape is ``[TEAM_COUNT, player_count]``. Arms are excluded by
    Law 11. The support is the union of the oriented torso capsule, head sphere,
    and the foremost foot along the player's facing direction.
    """

    players = state.players
    player_position = players.position if player_position is None else player_position
    body_forward = players.body_forward
    shoulder_x = -body_forward[:, 1]
    half_core = 0.5 * (body.shoulder_width_m - body.torso_depth_m)
    torso_radius = 0.5 * body.torso_depth_m
    direction = state.attack_direction[:TEAM_COUNT, None]
    centre = direction * player_position[None, :, 0]
    torso = centre + half_core * jnp.abs(shoulder_x)[None, :] + torso_radius
    head = centre + body.head_radius_m
    # A fixed-shape gait envelope is symmetric along the facing axis: at least
    # one foot may be ahead on either end of a stride. This also preserves the
    # environment's 180-degree rotation/team-swap covariance.
    foot = centre + (jnp.abs(body_forward[:, 0])[None, :] * body.foot_forward_extent_m)
    return jnp.maximum(jnp.maximum(torso, head), foot)


def _second_last_opponent_lines(
    state: State,
    support: jax.Array,
) -> jax.Array:
    player_count = state.players.position.shape[0]
    player_index = jnp.arange(player_count, dtype=jnp.int32)
    team_index = jnp.arange(TEAM_COUNT, dtype=jnp.int32)[:, None]
    opponent = state.players.active[None, :] & (
        state.players.team_id[None, :] != team_index
    )
    candidates = jnp.where(opponent, support, -jnp.inf)
    first_index = jnp.argmax(candidates, axis=1)
    not_first = player_index[None, :] != first_index[:, None]
    return jnp.max(jnp.where(opponent & not_first, support, -jnp.inf), axis=1)


def _snapshot_for_touch(
    state: State,
    support: jax.Array,
    second_last_line: jax.Array,
    team: jax.Array,
    actor: jax.Array,
    ball_position: jax.Array,
    *,
    ball: Ball,
) -> jax.Array:
    player_count = state.players.position.shape[0]
    player_index = jnp.arange(player_count, dtype=jnp.int32)
    safe_team = jnp.clip(team, 0, TEAM_COUNT - 1)
    valid_team = (team >= 0) & (team < TEAM_COUNT)
    ball_front = state.attack_direction[safe_team] * ball_position[0] + ball.radius
    front = support[safe_team]
    return (
        valid_team
        & state.players.active
        & (state.players.team_id == team)
        & (player_index != actor)
        & (front > 0.0)
        & (front > ball_front)
        & (front > second_last_line[safe_team])
    )


def offside_position_snapshot(
    state: State,
    team: jax.Array,
    actor: jax.Array,
    ball_position: jax.Array,
    *,
    ball: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> jax.Array:
    """Flag offside positions at one team-mate's ball contact."""

    support = player_front_support(state, body=body)
    second_last_line = _second_last_opponent_lines(state, support)
    return _snapshot_for_touch(
        state,
        support,
        second_last_line,
        team,
        actor,
        ball_position,
        ball=ball,
    )


def resolve_offside_challenge(
    offside: OffsideState,
    state: State,
    contest,
    *,
    time_fraction: jax.Array = 0.0,
    ball: Ball = Ball(),
) -> OffsideEvent:
    """Return an offence for the selected contest challenger.

    ``ContestResult.challenger`` is populated only when the selected actor is
    actually challenging an opposing, physically verified ball carrier. A
    merely eligible or unselected candidate is therefore insufficient. The
    caller owns priority against a simultaneous contest foul and supplies the
    swept contact time.

    This implements only the observable ``challenging an opponent for the
    ball`` branch of Law 11. It deliberately does not infer obstruction of
    vision, close-ball attempts, or other opponent impact from geometry.
    """

    player_count = state.players.position.shape[0]
    actor = jnp.asarray(contest.challenger, dtype=jnp.int32)
    actor_in_range = (actor >= 0) & (actor < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    actor_team = state.players.team_id[safe_actor].astype(jnp.int32)
    valid_team = (actor_team >= 0) & (actor_team < TEAM_COUNT)
    offence = (
        contest.selected
        & state.ball.live
        & actor_in_range
        & (contest.actor == actor)
        & state.players.active[safe_actor]
        & valid_team
        & offside.flagged[safe_actor]
        & (offside.direct_exempt_team != actor_team)
    )
    dtype = state.ball.position.dtype
    position = jnp.asarray(
        [
            state.players.position[safe_actor, 0],
            state.players.position[safe_actor, 1],
            jnp.asarray(ball.radius, dtype=dtype),
        ],
        dtype=dtype,
    )
    return OffsideEvent(
        occurred=offence,
        actor=jnp.where(offence, actor, NO_PLAYER).astype(jnp.int32),
        team=jnp.where(offence, actor_team, NO_TEAM).astype(jnp.int32),
        position=jnp.where(offence, position, jnp.zeros_like(position)),
        time_fraction=jnp.where(
            offence,
            jnp.asarray(time_fraction, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        ),
    )


def resolve_offside_contacts(
    offside: OffsideState,
    state: State,
    occurrences: ContactOccurrence,
    *,
    player_path_start: jax.Array | None = None,
    player_path_delta: jax.Array | None = None,
    ball: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> OffsideStep:
    """Reduce chronological player contacts under the touch-based Law 11 subset.

    This deliberately excludes involvement without a ball touch, including
    obstructing an opponent's view, challenging, and impactful attempts.
    Contact arrays are expected in chronological physics-event order. False
    slots left by frame or ground events are harmless. Position is interpolated
    on the approved p0/p1 path; facing remains the substep-final value.
    """

    players = state.players
    player_count = players.position.shape[0]
    player_path_start = (
        players.position if player_path_start is None else player_path_start
    )
    player_path_delta = (
        jnp.zeros_like(players.position)
        if player_path_delta is None
        else player_path_delta
    )
    dtype = occurrences.position.dtype
    empty_event = OffsideEvent(
        occurred=jnp.asarray(False),
        actor=jnp.asarray(NO_PLAYER, dtype=jnp.int32),
        team=jnp.asarray(NO_TEAM, dtype=jnp.int32),
        position=jnp.zeros(3, dtype=dtype),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
    )

    def reduce_contact(carry, occurrence):
        current, event = carry
        contact_positions = (
            player_path_start + occurrence.time_fraction * player_path_delta
        )
        support = player_front_support(
            state, player_position=contact_positions, body=body
        )
        second_last_line = _second_last_opponent_lines(state, support)
        actor = occurrence.actor.astype(jnp.int32)
        valid_actor = occurrence.occurred & (actor >= 0) & (actor < player_count)
        safe_actor = jnp.clip(actor, 0, player_count - 1)
        actor_position = jnp.concatenate(
            (
                contact_positions[safe_actor],
                jnp.asarray([ball.radius], dtype=dtype),
            )
        )
        actor_team = players.team_id[safe_actor].astype(jnp.int32)
        valid_team = (actor_team >= 0) & (actor_team < TEAM_COUNT)
        process = valid_actor & valid_team & (~event.occurred)
        direct_release = occurrence.law11_effect == LAW11_DIRECT_RESTART_EXEMPTION
        exempt_receipt = actor_team == current.direct_exempt_team
        offence = (
            process
            & current.flagged[safe_actor]
            & (~exempt_receipt)
            & (~direct_release)
        )
        candidate_event = OffsideEvent(
            occurred=offence,
            actor=jnp.where(offence, actor, NO_PLAYER).astype(jnp.int32),
            team=jnp.where(offence, actor_team, NO_TEAM).astype(jnp.int32),
            position=jnp.where(offence, actor_position, jnp.zeros_like(actor_position)),
            time_fraction=jnp.where(
                offence, occurrence.time_fraction, jnp.asarray(0.0, dtype=dtype)
            ),
        )
        event = jax.tree_util.tree_map(
            lambda new, old: jnp.where(offence, new, old),
            candidate_event,
            event,
        )

        snapshot = _snapshot_for_touch(
            state,
            support,
            second_last_line,
            actor_team,
            actor,
            occurrence.position,
            ball=ball,
        )
        actor_team_mask = players.team_id == actor_team
        preserve_other_team = jnp.where(actor_team_mask, snapshot, current.flagged)
        deliberate_play = occurrence.law11_effect == LAW11_DELIBERATE_PLAY_RESET
        no_reset_touch = (occurrence.law11_effect == LAW11_DEFLECTION_NO_RESET) | (
            occurrence.law11_effect == LAW11_DELIBERATE_SAVE_NO_RESET
        )
        next_flagged = jnp.where(
            direct_release,
            jnp.zeros_like(current.flagged),
            jnp.where(
                deliberate_play,
                snapshot,
                jnp.where(no_reset_touch, preserve_other_team, current.flagged),
            ),
        )
        next_exempt_team = jnp.where(
            direct_release,
            actor_team,
            jnp.where(
                deliberate_play | (no_reset_touch & exempt_receipt),
                NO_TEAM,
                current.direct_exempt_team,
            ),
        ).astype(jnp.int32)
        candidate_state = OffsideState(
            flagged=next_flagged,
            direct_exempt_team=next_exempt_team,
        )
        update = process & (~offence)
        current = jax.tree_util.tree_map(
            lambda new, old: jnp.where(update, new, old),
            candidate_state,
            current,
        )
        return (current, event), None

    (next_state, event), _ = jax.lax.scan(
        reduce_contact,
        (offside, empty_event),
        occurrences,
    )
    return OffsideStep(state=next_state, event=event)
