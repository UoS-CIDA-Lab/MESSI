"""Pure JAX state transition for a classified ball-boundary crossing."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    BALL_EVENT_CORNER,
    BALL_EVENT_GOAL,
    BALL_EVENT_GOALKICK,
    BALL_EVENT_NONE,
    BALL_EVENT_THROWIN,
    NO_PLAYER,
    NO_TEAM,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.contact import MECHANISM_NONE, MECHANISM_THROW
from footballworld.core.state import RestartReleaseProvenance, State
from footballworld.rules.ball_boundary import (
    BoundaryCrossing,
    BoundaryEvent,
    classify_boundary_event,
)
from footballworld.rules.restart import select_restart_taker
from footballworld.rules.restart_spot import canonical_restart_spot


class BoundaryResolution(NamedTuple):
    """State and final rule event produced by one boundary crossing."""

    state: State
    event: BoundaryEvent
    goal_awarded: jax.Array


def _last_touch_team(state: State) -> jax.Array:
    actor = state.possession.last_contact.actor
    player_count = state.players.team_id.shape[0]
    valid = (actor >= 0) & (actor < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    return jnp.where(valid, state.players.team_id[safe_actor], NO_TEAM).astype(
        jnp.int32
    )


def resolve_boundary_crossing(
    state: State,
    crossing: BoundaryCrossing,
    *,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> BoundaryResolution:
    """Apply one goal or out-of-play crossing to the authoritative state."""

    raw_event = classify_boundary_event(
        crossing,
        _last_touch_team(state),
        state.possession.team,
        state.attack_direction,
        state.kickoff_team,
    )
    raw_goal = raw_event.occurred & (raw_event.kind == BALL_EVENT_GOAL)
    release = state.restart_release
    valid_release_team = (release.team == TEAM_0) | (release.team == TEAM_1)
    from_restart = release.untouched & valid_release_team
    direct_own_goal = (
        raw_goal
        & from_restart
        & (release.kind != RK_GK_HOLD)
        & (release.team != raw_event.scoring_team)
    )
    direct_opponent_goal = (
        raw_goal & from_restart & (release.team == raw_event.scoring_team)
    )
    gk_hand_distribution = (release.kind == RK_GK_HOLD) & (
        release.release_mechanism == MECHANISM_THROW
    )
    opponent_goal_forbidden = direct_opponent_goal & (
        (release.kind == RK_THROWIN) | gk_hand_distribution | release.indirect
    )
    goal_awarded = raw_goal & (~direct_own_goal) & (~opponent_goal_forbidden)

    conceding_team = (TEAM_1 - raw_event.scoring_team).astype(jnp.int32)
    restart_team = jnp.where(
        goal_awarded,
        conceding_team,
        jnp.where(
            direct_own_goal,
            raw_event.scoring_team,
            jnp.where(opponent_goal_forbidden, conceding_team, raw_event.team),
        ),
    ).astype(jnp.int32)
    final_event_kind = jnp.where(
        direct_own_goal,
        BALL_EVENT_CORNER,
        jnp.where(opponent_goal_forbidden, BALL_EVENT_GOALKICK, raw_event.kind),
    ).astype(jnp.int32)
    restart_kind = jnp.where(
        goal_awarded,
        RK_KICKOFF,
        jnp.where(
            final_event_kind == BALL_EVENT_THROWIN,
            RK_THROWIN,
            jnp.where(
                final_event_kind == BALL_EVENT_CORNER,
                RK_CORNER,
                jnp.where(
                    final_event_kind == BALL_EVENT_GOALKICK,
                    RK_GOALKICK,
                    RK_NONE,
                ),
            ),
        ),
    ).astype(jnp.int32)
    restart_position = canonical_restart_spot(
        restart_kind,
        restart_team,
        crossing.position,
        state.attack_direction,
        stadium=stadium,
        ball=ball_geometry,
    )
    taker = select_restart_taker(
        state,
        restart_kind,
        restart_team,
        restart_position,
        stadium=stadium,
    )

    safe_scoring_team = jnp.clip(raw_event.scoring_team, TEAM_0, TEAM_1)
    score = state.score.at[safe_scoring_team].add(
        goal_awarded.astype(state.score.dtype)
    )
    ball = state.ball._replace(
        position=restart_position,
        velocity=jnp.zeros_like(state.ball.velocity),
        spin=jnp.zeros_like(state.ball.spin),
        live=jnp.asarray(False),
    )
    possession = state.possession._replace(
        team=jnp.asarray(NO_TEAM, dtype=jnp.int32),
        player=jnp.asarray(NO_PLAYER, dtype=jnp.int32),
        previous_team=state.possession.team.astype(jnp.int32),
        control_ticks=jnp.asarray(0, dtype=jnp.int32),
    )
    restart = state.restart._replace(
        kind=restart_kind,
        team=restart_team,
        substeps_remaining=jnp.asarray(0, dtype=jnp.int32),
        taker=taker,
        indirect=jnp.asarray(False),
        opened_control_tick=state.control_tick,
    )
    cleared_release = RestartReleaseProvenance(
        active=jnp.asarray(False),
        untouched=jnp.asarray(False),
        kind=jnp.asarray(RK_NONE, dtype=jnp.int32),
        team=jnp.asarray(NO_TEAM, dtype=jnp.int32),
        taker=jnp.asarray(NO_PLAYER, dtype=jnp.int32),
        indirect=jnp.asarray(False),
        law11_direct_exempt=jnp.asarray(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    candidate_state = state._replace(
        ball=ball,
        kickoff_team=jnp.where(goal_awarded, conceding_team, state.kickoff_team).astype(
            jnp.int32
        ),
        possession=possession,
        restart=restart,
        score=score,
        restart_release=cleared_release,
        gk_backpass_team=jnp.int32(NO_TEAM),
    )
    next_state = jax.tree_util.tree_map(
        lambda candidate, current: jnp.where(crossing.occurred, candidate, current),
        candidate_state,
        state,
    )
    final_event = BoundaryEvent(
        occurred=raw_event.occurred,
        kind=jnp.where(raw_event.occurred, final_event_kind, BALL_EVENT_NONE).astype(
            jnp.int32
        ),
        team=jnp.where(
            raw_event.occurred,
            jnp.where(goal_awarded, raw_event.scoring_team, restart_team),
            NO_TEAM,
        ).astype(jnp.int32),
        scoring_team=jnp.where(goal_awarded, raw_event.scoring_team, NO_TEAM).astype(
            jnp.int32
        ),
        time_fraction=raw_event.time_fraction,
        position=raw_event.position,
    )
    return BoundaryResolution(
        state=next_state,
        event=final_event,
        goal_awarded=goal_awarded,
    )


def apply_boundary_crossing(
    state: State,
    crossing: BoundaryCrossing,
    *,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> State:
    """Apply a crossing and return only the next state."""

    return resolve_boundary_crossing(
        state,
        crossing,
        stadium=stadium,
        ball_geometry=ball_geometry,
    ).state
