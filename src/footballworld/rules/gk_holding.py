"""Pure-JAX goalkeeper eight-second hand-control transition.

IFAB Law 12.3 awards an opponent corner when a goalkeeper controls the ball
with the hands/arms for more than eight seconds.  The official FAQ places the
corner in the corner area on the side closest to the goalkeeper when
penalised. An exact
centre-line tie is not prescribed, so this module deterministically uses the
positive-y corner.

The caller must hold one policy action fixed for a complete control frame and
must not reinterpret the catch action as a release after ``opened`` becomes
true mid-frame.  This preserves the next-decision-boundary contract without
adding another dynamic state field. Offside state is intentionally outside this
module and must be cleared by the higher-level rules transition on expiry.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.core.constants import (
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_NONE,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.contact import MECHANISM_NONE
from footballworld.core.state import RestartReleaseProvenance, State
from footballworld.core.timebase import DEFAULT_TIMEBASE, Timebase
from footballworld.rules.restart import select_restart_taker


class GoalkeeperHoldingEvent(NamedTuple):
    """Opening or expiry of goalkeeper hand control in one physics tick."""

    opened: jax.Array
    expired: jax.Array
    goalkeeper: jax.Array
    goalkeeper_team: jax.Array
    restart_team: jax.Array
    corner_position: jax.Array


class GoalkeeperHoldingResolution(NamedTuple):
    """Post-physics state plus the eight-second event."""

    state: State
    event: GoalkeeperHoldingEvent


def holding_limit_substeps(
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    config: GoalkeeperHolding = GoalkeeperHolding(),
) -> int:
    """Convert the SI Law 12.3 limit once on the host.

    Floor is intentional: it is the number of complete physics intervals that
    can elapse without exceeding the legal duration.  The following interval
    is the first one that can produce the offence.
    """

    return timebase.physics_steps_for(
        config.hand_control_limit_s,
        rounding="floor",
    )


def _corner_position(
    state: State,
    goalkeeper: jax.Array,
    goalkeeper_team: jax.Array,
    goalkeeper_valid: jax.Array,
    *,
    stadium: Stadium,
    ball: Ball,
) -> jax.Array:
    """Return the prescribed corner, using held-ball side without a valid GK."""

    dtype = state.ball.position.dtype
    safe_goalkeeper = jnp.clip(goalkeeper, 0, state.players.position.shape[0] - 1)
    safe_team = jnp.clip(goalkeeper_team, TEAM_0, TEAM_1)
    defending_end = -state.attack_direction[safe_team]
    side_basis = jnp.where(
        goalkeeper_valid,
        state.players.position[safe_goalkeeper, 1],
        state.ball.position[1],
    )
    side = jnp.where(side_basis >= 0.0, 1.0, -1.0)
    return jnp.asarray(
        [
            defending_end * stadium.half_length,
            side * stadium.half_width,
            ball.radius,
        ],
        dtype=dtype,
    )


def step_goalkeeper_holding(
    pre_physics_state: State,
    post_physics_state: State,
    *,
    limit_substeps: int,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> GoalkeeperHoldingResolution:
    """Advance an existing GK hold or turn a hold over into a corner.

    Call exactly once after each physics substep.  ``limit_substeps`` is a
    host-derived static integer from :func:`holding_limit_substeps`.
    A newly opened hold counts its opening substep.  Releasing at exactly the
    limit remains legal; continuing through the following substep expires it.
    A malformed hold with no valid goalkeeper also expires from its valid
    restart team, using the held ball rather than an invalid actor for side.
    """

    was_held = pre_physics_state.restart.kind == RK_GK_HOLD
    is_held = post_physics_state.restart.kind == RK_GK_HOLD
    same_holder = pre_physics_state.restart.taker == post_physics_state.restart.taker
    opened = (~was_held) & is_held
    continued = was_held & is_held & same_holder
    expired = continued & (pre_physics_state.restart.substeps_remaining <= 0)

    limit = jnp.asarray(limit_substeps, dtype=jnp.int32)
    opened_remaining = jnp.maximum(limit - 1, 0)
    continued_remaining = jnp.maximum(
        pre_physics_state.restart.substeps_remaining - 1, 0
    )
    remaining = jnp.where(
        opened,
        opened_remaining,
        jnp.where(
            continued & (~expired),
            continued_remaining,
            post_physics_state.restart.substeps_remaining,
        ),
    ).astype(jnp.int32)
    counted_state = post_physics_state._replace(
        restart=post_physics_state.restart._replace(substeps_remaining=remaining)
    )

    goalkeeper = post_physics_state.restart.taker.astype(jnp.int32)
    player_count = post_physics_state.players.position.shape[0]
    goalkeeper_in_range = (goalkeeper >= 0) & (goalkeeper < player_count)
    safe_goalkeeper = jnp.clip(goalkeeper, 0, player_count - 1)
    player_team = post_physics_state.players.team_id[safe_goalkeeper]
    restart_team_value = post_physics_state.restart.team
    valid_restart_team = (restart_team_value == TEAM_0) | (restart_team_value == TEAM_1)
    goalkeeper_team = jnp.where(
        valid_restart_team, restart_team_value, player_team
    ).astype(jnp.int32)
    valid_team = (goalkeeper_team == TEAM_0) | (goalkeeper_team == TEAM_1)
    valid_goalkeeper = (
        goalkeeper_in_range
        & post_physics_state.players.active[safe_goalkeeper]
        & post_physics_state.players.is_goalkeeper[safe_goalkeeper]
        & (player_team == goalkeeper_team)
    )
    expired = expired & valid_team
    restart_team = (TEAM_1 - goalkeeper_team).astype(jnp.int32)
    corner_position = _corner_position(
        post_physics_state,
        safe_goalkeeper,
        goalkeeper_team,
        valid_goalkeeper,
        stadium=stadium,
        ball=ball_geometry,
    )
    taker = select_restart_taker(
        post_physics_state,
        RK_CORNER,
        restart_team,
        corner_position,
        stadium=stadium,
    )

    ball = post_physics_state.ball._replace(
        position=corner_position,
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
        kind=jnp.int32(RK_CORNER),
        team=restart_team,
        substeps_remaining=jnp.int32(0),
        taker=taker,
        indirect=jnp.bool_(False),
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
    corner_state = post_physics_state._replace(
        ball=ball,
        possession=possession,
        restart=restart,
        restart_release=cleared_release,
        gk_backpass_team=jnp.int32(NO_TEAM),
    )
    state = jax.tree_util.tree_map(
        lambda changed, current: jnp.where(expired, changed, current),
        corner_state,
        counted_state,
    )
    event_corner = jnp.where(expired, corner_position, jnp.zeros_like(corner_position))
    reported_goalkeeper = (opened | expired) & valid_goalkeeper & valid_team
    reported_team = (opened | expired) & valid_team
    event = GoalkeeperHoldingEvent(
        opened=opened,
        expired=expired,
        goalkeeper=jnp.where(reported_goalkeeper, goalkeeper, NO_PLAYER).astype(
            jnp.int32
        ),
        goalkeeper_team=jnp.where(reported_team, goalkeeper_team, NO_TEAM).astype(
            jnp.int32
        ),
        restart_team=jnp.where(expired, restart_team, NO_TEAM).astype(jnp.int32),
        corner_position=event_corner,
    )
    return GoalkeeperHoldingResolution(state=state, event=event)
