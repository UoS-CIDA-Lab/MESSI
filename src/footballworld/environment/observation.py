"""Full-information agent observations derived without mutating physics.

Roster metadata is deliberately built separately so callers can publish or
cache it once rather than replicating it inside every per-player observation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_NONE,
)
from footballworld.core.contact import INTENT_SOURCE_NONE
from footballworld.core.state import State
from footballworld.environment.clock import MatchClockTicks, match_clock_ticks
from footballworld.rules.gk_handling_restriction import (
    goalkeeper_hand_restricted_team,
)
from footballworld.rules.offside import OffsideState
from footballworld.rules.restart_legality import (
    goalkeeper_restart_boundary_ready,
    restart_actor_mask,
)
from footballworld.rules.restart_timing import restart_may_release_within_frame

ROSTER_METADATA_SCHEMA_VERSION = 2
PLAYER_OBSERVATION_SCHEMA_VERSION = 13


class RosterMetadata(NamedTuple):
    """Public profiles; generation is -1 when no manager ledger is supplied."""

    team_id: jax.Array
    player_id: jax.Array
    slot_generation: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


class SelfObservation(NamedTuple):
    """Observer anchor in its team's current attacking coordinate frame."""

    player_index: jax.Array
    position: jax.Array
    velocity: jax.Array


class PlayerObservations(NamedTuple):
    """Fixed roster-slot observations relative to the observer."""

    relative_position: jax.Array
    relative_velocity: jax.Array
    facing_sin: jax.Array
    facing_cos: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    challenge_recovery_substeps: jax.Array
    contact_lock_substeps: jax.Array
    aerial_recovery_substeps: jax.Array
    possession_loss_lock_substeps: jax.Array
    yellow_cards: jax.Array
    on_pitch: jax.Array
    sent_off: jax.Array
    offside: jax.Array
    possessor: jax.Array
    restart_taker: jax.Array
    release_taker: jax.Array
    last_actor: jax.Array
    contact_may_occur_this_frame: jax.Array


class BallObservation(NamedTuple):
    """Ball state relative to the observer: position, velocity, then spin."""

    relative_state: jax.Array
    live: jax.Array


class LastContactObservation(NamedTuple):
    """The non-actor fields of the latest possession-producing contact."""

    mechanism: jax.Array
    intent: jax.Array
    intent_source: jax.Array
    outcome: jax.Array
    restart_kind: jax.Array
    law11_effect: jax.Array
    kick_applied: jax.Array


class PossessionObservation(NamedTuple):
    """Possession state; player identity is the per-slot ``possessor`` flag."""

    previous_team: jax.Array
    control_ticks: jax.Array
    last_contact: LastContactObservation


class RestartObservation(NamedTuple):
    """Public restart phase; taker identity is a visible per-slot flag."""

    kind: jax.Array
    team: jax.Array
    substeps_remaining: jax.Array
    indirect: jax.Array


class RestartReleaseObservation(NamedTuple):
    """Executed-restart provenance; taker identity is a per-slot flag."""

    active: jax.Array
    untouched: jax.Array
    kind: jax.Array
    indirect: jax.Array
    law11_direct_exempt: jax.Array
    release_mechanism: jax.Array


class MatchObservation(NamedTuple):
    """Globally public phase and score state."""

    score: jax.Array
    control_tick: jax.Array
    offside_direct_exempt_team: jax.Array
    gk_handling_restricted_team: jax.Array
    clock: MatchClockTicks


class Observation(NamedTuple):
    """One player observer's fixed-shape JAX tree; it never contains a bench."""

    valid: jax.Array
    self_state: SelfObservation
    players: PlayerObservations
    ball: BallObservation
    possession: PossessionObservation
    restart: RestartObservation
    restart_release: RestartReleaseObservation
    match: MatchObservation


# Explicit semantic name for new code while preserving the original API and
# existing checkpoint tree definition.
PlayerObservation = Observation


def roster_metadata_from_state(
    state: State, slot_generation: jax.Array | None = None
) -> RosterMetadata:
    """Extract profiles and an exact managed generation or -1 sentinel."""

    players = state.players
    if slot_generation is None:
        generation = jnp.full_like(players.player_id, -1, dtype=jnp.int32)
    else:
        generation = jnp.asarray(slot_generation)
        if generation.shape != players.player_id.shape:
            raise ValueError("slot_generation must match the roster-slot axis")
        if generation.dtype != jnp.int32:
            raise TypeError("slot_generation must have int32 dtype")
    return RosterMetadata(
        team_id=players.team_id,
        player_id=players.player_id,
        slot_generation=generation,
        is_goalkeeper=players.is_goalkeeper,
        max_speed=players.max_speed,
        reach_height=players.reach_height,
        height=players.height,
        ball_control=players.ball_control,
        endurance_factor=players.endurance_factor,
    )


def _mask_float(value: jax.Array, visible: jax.Array) -> jax.Array:
    mask = visible
    while mask.ndim < value.ndim:
        mask = mask[..., None]
    return jnp.where(mask, value, jnp.zeros_like(value))


def _actor_flag(
    actor: jax.Array,
    player_count: int,
    visible: jax.Array,
) -> jax.Array:
    indices = jnp.arange(player_count, dtype=jnp.int32)
    valid = (actor >= 0) & (actor < player_count)
    return valid & (indices == actor) & visible


def _safe_subject_index(value, size: int) -> tuple[jax.Array, jax.Array]:
    """Return a safe scalar address and fail-closed validity mask.

    Host values are inspected before conversion so booleans, fractional
    values, and out-of-range host integers cannot alias a valid subject. Compiled
    callers must supply scalar int32 identifiers because JAX may canonicalize
    wider integers before this function sees them.
    """

    traced = any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value)
    )
    if not traced:
        source = np.asarray(value)
        if source.shape != ():
            raise ValueError("subject index must be scalar")
        integer_dtype = np.issubdtype(source.dtype, np.integer) and not np.issubdtype(
            source.dtype, np.bool_
        )
        if not integer_dtype:
            return jnp.int32(0), jnp.bool_(False)
        integer = int(source)
        valid = 0 <= integer < size
        return jnp.int32(integer if valid else 0), jnp.bool_(valid)

    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError("subject index must be scalar")
    integer_dtype = jnp.issubdtype(array.dtype, jnp.integer) and not jnp.issubdtype(
        array.dtype, jnp.bool_
    )
    if not integer_dtype:
        return jnp.int32(0), jnp.bool_(False)
    valid = (array >= 0) & (array < size)
    return jnp.clip(array, 0, size - 1).astype(jnp.int32), valid


def observe(
    state: State,
    offside_state: OffsideState,
    observer_index: jax.Array,
    *,
    decimation: int,
    restart_delay_substeps: int,
    goalkeeper_hold_limit_substeps: int,
    halftime_tick: int,
    fulltime_tick: int,
    halftime_enabled: bool,
) -> Observation:
    """Build one full-information observation without touching physics.

    ``decimation`` is the fixed number of physics substeps in the upcoming
    control frame.  The contact affordance uses only hard state that can make
    contact impossible for that entire frame; it deliberately ignores ball
    position, speed, and height.  An invisible slot retains conservative
    eligibility from public phase, team, activity, and role facts rather than
    leaking a hidden actor identity or private recovery timer.
    """

    if not isinstance(decimation, int) or isinstance(decimation, bool):
        raise TypeError("decimation must be an integer")
    if decimation < 1:
        raise ValueError("decimation must be positive")

    players = state.players
    player_count = players.position.shape[0]
    safe_observer, observer_valid = _safe_subject_index(observer_index, player_count)
    observer_team = players.team_id[safe_observer]
    direction = state.attack_direction[observer_team]
    observer_position = players.position[safe_observer]
    observer_velocity = players.velocity[safe_observer]

    player_visible = jnp.ones(player_count, dtype=jnp.bool_) & observer_valid
    ball_visible = jnp.asarray(True, dtype=jnp.bool_) & observer_valid
    local_relative_position = (players.position - observer_position) * direction
    local_relative_velocity = (players.velocity - observer_velocity) * direction
    local_body_forward = players.body_forward * direction
    local_facing_cos = local_body_forward[:, 0]
    local_facing_sin = local_body_forward[:, 1]

    possession_actor = state.possession.player
    restart_actor = state.restart.taker
    release_actor = state.restart_release.taker
    last_actor = state.possession.last_contact.actor
    possessor = _actor_flag(possession_actor, player_count, player_visible)
    restart_taker = _actor_flag(restart_actor, player_count, player_visible)
    release_taker = _actor_flag(release_actor, player_count, player_visible)
    last_actor_flag = _actor_flag(last_actor, player_count, player_visible)
    offside = offside_state.flagged & player_visible

    full_frame_locked = (
        (players.contact_lock_substeps >= decimation)
        | (players.aerial_recovery_substeps >= decimation)
        | (players.possession_loss_lock_substeps >= decimation)
    )
    restart_active = state.restart.kind != RK_NONE
    restart_fires_this_frame = restart_may_release_within_frame(
        state,
        decimation,
        delay_substeps=restart_delay_substeps,
        hold_limit_substeps=goalkeeper_hold_limit_substeps,
    ) & goalkeeper_restart_boundary_ready(state)
    # A legal actor is still unable to touch the ball until the referee's
    # restart placement transaction has committed.
    restart_fires_this_frame = restart_fires_this_frame & state.restart_layout_ready
    structural_phase = jnp.where(
        restart_active,
        restart_actor_mask(state) & restart_fires_this_frame,
        state.ball.live & players.active,
    )
    contact_may_occur = structural_phase & (~full_frame_locked) & observer_valid

    ball_relative_position = jnp.concatenate(
        (
            (state.ball.position[:2] - observer_position) * direction,
            state.ball.position[2:3],
        )
    )
    ball_relative_velocity = jnp.concatenate(
        (
            (state.ball.velocity[:2] - observer_velocity) * direction,
            state.ball.velocity[2:3],
        )
    )
    ball_local_spin = jnp.concatenate(
        (state.ball.spin[:2] * direction, state.ball.spin[2:3])
    )
    ball_relative_state = jnp.concatenate(
        (ball_relative_position, ball_relative_velocity, ball_local_spin)
    )
    ball_relative_state = jnp.where(
        ball_visible, ball_relative_state, jnp.zeros_like(ball_relative_state)
    )

    handling_team = goalkeeper_hand_restricted_team(state)
    observed_handling_team = jnp.where(observer_valid, handling_team, NO_TEAM).astype(
        jnp.int32
    )
    last_contact = state.possession.last_contact

    return Observation(
        valid=observer_valid,
        self_state=SelfObservation(
            player_index=jnp.where(observer_valid, safe_observer, NO_PLAYER),
            position=jnp.where(observer_valid, observer_position * direction, 0.0),
            velocity=jnp.where(observer_valid, observer_velocity * direction, 0.0),
        ),
        players=PlayerObservations(
            relative_position=_mask_float(local_relative_position, player_visible),
            relative_velocity=_mask_float(local_relative_velocity, player_visible),
            facing_sin=_mask_float(local_facing_sin, player_visible),
            facing_cos=_mask_float(local_facing_cos, player_visible),
            stamina_long=_mask_float(players.stamina_long, player_visible),
            stamina_short=_mask_float(players.stamina_short, player_visible),
            challenge_recovery_substeps=jnp.where(
                player_visible, players.challenge_recovery_substeps, 0
            ),
            contact_lock_substeps=jnp.where(
                player_visible, players.contact_lock_substeps, 0
            ),
            aerial_recovery_substeps=jnp.where(
                player_visible, players.aerial_recovery_substeps, 0
            ),
            possession_loss_lock_substeps=jnp.where(
                player_visible, players.possession_loss_lock_substeps, 0
            ),
            yellow_cards=jnp.where(player_visible, players.yellow_cards, 0),
            on_pitch=players.on_pitch & observer_valid,
            sent_off=players.sent_off & observer_valid,
            offside=offside,
            possessor=possessor,
            restart_taker=restart_taker,
            release_taker=release_taker,
            last_actor=last_actor_flag,
            contact_may_occur_this_frame=contact_may_occur,
        ),
        ball=BallObservation(
            relative_state=ball_relative_state,
            live=state.ball.live & observer_valid,
        ),
        possession=PossessionObservation(
            previous_team=jnp.where(
                observer_valid,
                state.possession.previous_team,
                NO_TEAM,
            ),
            control_ticks=jnp.where(observer_valid, state.possession.control_ticks, 0),
            last_contact=LastContactObservation(
                mechanism=jnp.where(observer_valid, last_contact.mechanism, 0),
                intent=jnp.where(observer_valid, last_contact.intent, 0),
                intent_source=jnp.where(
                    observer_valid,
                    last_contact.intent_source,
                    INTENT_SOURCE_NONE,
                ),
                outcome=jnp.where(observer_valid, last_contact.outcome, 0),
                restart_kind=jnp.where(observer_valid, last_contact.restart_kind, 0),
                law11_effect=jnp.where(observer_valid, last_contact.law11_effect, 0),
                kick_applied=last_contact.kick_applied & observer_valid,
            ),
        ),
        restart=RestartObservation(
            kind=jnp.where(observer_valid, state.restart.kind, RK_NONE),
            team=jnp.where(observer_valid, state.restart.team, NO_TEAM),
            substeps_remaining=jnp.where(
                observer_valid, state.restart.substeps_remaining, 0
            ),
            indirect=state.restart.indirect & observer_valid,
        ),
        restart_release=RestartReleaseObservation(
            active=state.restart_release.active & observer_valid,
            untouched=state.restart_release.untouched & observer_valid,
            kind=jnp.where(observer_valid, state.restart_release.kind, RK_NONE),
            indirect=state.restart_release.indirect & observer_valid,
            law11_direct_exempt=(
                state.restart_release.law11_direct_exempt & observer_valid
            ),
            release_mechanism=jnp.where(
                observer_valid, state.restart_release.release_mechanism, 0
            ),
        ),
        match=MatchObservation(
            score=jnp.where(observer_valid, state.score, 0),
            control_tick=jnp.where(observer_valid, state.control_tick, 0),
            offside_direct_exempt_team=jnp.where(
                observer_valid, offside_state.direct_exempt_team, NO_TEAM
            ),
            gk_handling_restricted_team=observed_handling_team,
            clock=jax.tree.map(
                lambda value: jnp.where(observer_valid, value, jnp.zeros_like(value)),
                match_clock_ticks(
                    state,
                    halftime_tick=halftime_tick,
                    fulltime_tick=fulltime_tick,
                    halftime_enabled=halftime_enabled,
                ),
            ),
        ),
    )


def observe_all(
    state: State,
    offside_state: OffsideState,
    *,
    decimation: int,
    restart_delay_substeps: int,
    goalkeeper_hold_limit_substeps: int,
    halftime_tick: int,
    fulltime_tick: int,
    halftime_enabled: bool,
) -> Observation:
    """Vectorize ``observe`` over every fixed roster slot."""

    player_count = state.players.position.shape[0]
    return jax.vmap(
        lambda observer: observe(
            state,
            offside_state,
            observer,
            decimation=decimation,
            restart_delay_substeps=restart_delay_substeps,
            goalkeeper_hold_limit_substeps=(goalkeeper_hold_limit_substeps),
            halftime_tick=halftime_tick,
            fulltime_tick=fulltime_tick,
            halftime_enabled=halftime_enabled,
        )
    )(jnp.arange(player_count, dtype=jnp.int32))


__all__ = [
    "PLAYER_OBSERVATION_SCHEMA_VERSION",
    "ROSTER_METADATA_SCHEMA_VERSION",
    "BallObservation",
    "LastContactObservation",
    "MatchObservation",
    "Observation",
    "PlayerObservation",
    "PlayerObservations",
    "PossessionObservation",
    "RestartObservation",
    "RestartReleaseObservation",
    "RosterMetadata",
    "SelfObservation",
    "observe",
    "observe_all",
    "roster_metadata_from_state",
]
