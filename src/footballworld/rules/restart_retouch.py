"""Pure-JAX adjudication of a restart taker's prohibited second touch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.contact import (
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    ContactOccurrence,
)
from footballworld.core.state import RestartReleaseProvenance, State
from footballworld.rules.restart import select_restart_taker
from footballworld.rules.restart_spot import canonical_restart_spot


class RestartRetouchEvent(NamedTuple):
    """First prohibited contact found in the fixed contact sequence."""

    occurred: jax.Array
    offender: jax.Array
    source_restart_kind: jax.Array
    team: jax.Array
    restart_kind: jax.Array
    indirect: jax.Array
    goalkeeper_handling: jax.Array
    time_fraction: jax.Array
    contact_position: jax.Array
    restart_position: jax.Array


class RestartRetouchResolution(NamedTuple):
    """Post-physics state overridden by the first restart retouch offence."""

    state: State
    event: RestartRetouchEvent


class _RetouchScan(NamedTuple):
    active: jax.Array
    kind: jax.Array
    team: jax.Array
    taker: jax.Array
    found: jax.Array
    offender: jax.Array
    source_kind: jax.Array
    restart_team: jax.Array
    handling: jax.Array
    time_fraction: jax.Array
    position: jax.Array


def _is_set_piece(kind: jax.Array) -> jax.Array:
    return (
        (kind == RK_KICKOFF)
        | (kind == RK_THROWIN)
        | (kind == RK_GOALKICK)
        | (kind == RK_CORNER)
        | (kind == RK_FREEKICK)
        | (kind == RK_PENALTY)
        | (kind == RK_OFFSIDE)
    )


def detect_restart_retouch(
    pre_physics_state: State,
    contact_occurrences: ContactOccurrence,
) -> RestartRetouchEvent:
    """Find the first prohibited taker contact in four ordered contact slots.

    Deliberateness is carried by the mechanism rather than by a slot index
    when identifying the initial restart release. Every later ball contact by
    the same taker is a second-touch offence, including after a penalty kick.
    """

    release = pre_physics_state.restart_release
    dtype = contact_occurrences.position.dtype
    initial = _RetouchScan(
        active=release.active,
        kind=release.kind,
        team=release.team,
        taker=release.taker,
        found=jnp.bool_(False),
        offender=jnp.int32(NO_PLAYER),
        source_kind=jnp.int32(RK_NONE),
        restart_team=jnp.int32(NO_TEAM),
        handling=jnp.bool_(False),
        time_fraction=jnp.asarray(0.0, dtype=dtype),
        position=jnp.zeros(3, dtype=dtype),
    )

    def scan_contact(carry, inputs):
        occurred, actor, mechanism, position, time_fraction = inputs
        valid_team = (carry.team == TEAM_0) | (carry.team == TEAM_1)
        same_taker = occurred & carry.active & (actor == carry.taker)
        goalkeeper_handling = mechanism == MECHANISM_GOALKEEPER_HAND
        deliberate = occurred & (mechanism != MECHANISM_PASSIVE_BODY)
        set_piece_retouch = _is_set_piece(carry.kind)
        hold_handling = (carry.kind == RK_GK_HOLD) & goalkeeper_handling
        violation = (
            (~carry.found)
            & valid_team
            & same_taker
            & (set_piece_retouch | hold_handling)
        )
        restart_team = (TEAM_1 - carry.team).astype(jnp.int32)
        other_player_touch = occurred & carry.active & (actor != carry.taker)
        pending_release = (
            deliberate
            & (pre_physics_state.restart.kind != RK_NONE)
            & (actor == pre_physics_state.restart.taker)
        )
        next_active = carry.active & (~other_player_touch)
        next_active = jnp.where(pending_release, True, next_active)
        next_kind = jnp.where(
            pending_release, pre_physics_state.restart.kind, carry.kind
        ).astype(jnp.int32)
        next_team = jnp.where(
            pending_release, pre_physics_state.restart.team, carry.team
        ).astype(jnp.int32)
        next_taker = jnp.where(
            pending_release, pre_physics_state.restart.taker, carry.taker
        ).astype(jnp.int32)
        return _RetouchScan(
            active=next_active,
            kind=next_kind,
            team=next_team,
            taker=next_taker,
            found=carry.found | violation,
            offender=jnp.where(violation, actor, carry.offender).astype(jnp.int32),
            source_kind=jnp.where(violation, carry.kind, carry.source_kind).astype(
                jnp.int32
            ),
            restart_team=jnp.where(violation, restart_team, carry.restart_team).astype(
                jnp.int32
            ),
            handling=jnp.where(violation, goalkeeper_handling, carry.handling),
            time_fraction=jnp.where(violation, time_fraction, carry.time_fraction),
            position=jnp.where(violation, position, carry.position),
        ), None

    final, _ = jax.lax.scan(
        scan_contact,
        initial,
        (
            contact_occurrences.occurred,
            contact_occurrences.actor,
            contact_occurrences.mechanism,
            contact_occurrences.position,
            contact_occurrences.time_fraction,
        ),
    )
    return RestartRetouchEvent(
        occurred=final.found,
        offender=final.offender,
        source_restart_kind=final.source_kind,
        team=final.restart_team,
        restart_kind=jnp.where(final.found, RK_FREEKICK, RK_NONE).astype(jnp.int32),
        indirect=final.found,
        goalkeeper_handling=final.handling,
        time_fraction=final.time_fraction,
        contact_position=final.position,
        restart_position=final.position,
    )


def resolve_restart_retouch(
    pre_physics_state: State,
    post_physics_state: State,
    contact_occurrences: ContactOccurrence,
    *,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> RestartRetouchResolution:
    """Resolve retouch before any later boundary transition is applied."""

    detected = detect_restart_retouch(pre_physics_state, contact_occurrences)
    restart_position = canonical_restart_spot(
        RK_FREEKICK,
        detected.team,
        detected.contact_position,
        post_physics_state.attack_direction,
        indirect=jnp.bool_(True),
        stadium=stadium,
        ball=ball_geometry,
    )
    taker = select_restart_taker(
        post_physics_state,
        RK_FREEKICK,
        detected.team,
        restart_position,
        stadium=stadium,
    )
    ball = post_physics_state.ball._replace(
        position=restart_position,
        velocity=jnp.zeros_like(post_physics_state.ball.velocity),
        spin=jnp.zeros_like(post_physics_state.ball.spin),
        live=jnp.bool_(False),
    )
    possession = post_physics_state.possession._replace(
        team=jnp.int32(NO_TEAM),
        player=jnp.int32(NO_PLAYER),
        previous_team=post_physics_state.possession.team.astype(jnp.int32),
        control_ticks=jnp.int32(0),
    )
    restart = post_physics_state.restart._replace(
        kind=jnp.int32(RK_FREEKICK),
        team=detected.team,
        substeps_remaining=jnp.int32(0),
        taker=taker,
        indirect=jnp.bool_(True),
        opened_control_tick=post_physics_state.control_tick,
    )
    cleared_release = RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    candidate = post_physics_state._replace(
        ball=ball,
        possession=possession,
        restart=restart,
        restart_release=cleared_release,
        gk_backpass_team=jnp.int32(NO_TEAM),
    )
    state = jax.tree_util.tree_map(
        lambda changed, current: jnp.where(detected.occurred, changed, current),
        candidate,
        post_physics_state,
    )
    event = detected._replace(
        restart_position=jnp.where(
            detected.occurred,
            restart_position,
            jnp.zeros_like(restart_position),
        )
    )
    return RestartRetouchResolution(state=state, event=event)


def apply_restart_retouch(
    pre_physics_state: State,
    post_physics_state: State,
    contact_occurrences: ContactOccurrence,
    *,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> State:
    """Return only the state produced by restart-retouch adjudication."""

    return resolve_restart_retouch(
        pre_physics_state,
        post_physics_state,
        contact_occurrences,
        stadium=stadium,
        ball_geometry=ball_geometry,
    ).state
