"""Goal-to-kickoff ownership regression contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, Player, PlayerProfile
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import BALL_EVENT_GOAL, RK_KICKOFF, TEAM_0, TEAM_1
from footballworld.rules.ball_boundary import CROSSING_GOAL_LINE, BoundaryCrossing
from footballworld.rules.boundary_resolution import resolve_boundary_crossing

_FORMATION = (
    (-50.0, 0.0),
    (-35.0, -24.0),
    (-35.0, -8.0),
    (-35.0, 8.0),
    (-35.0, 24.0),
    (-20.0, -18.0),
    (-20.0, 0.0),
    (-20.0, 18.0),
    (-8.0, -24.0),
    (-8.0, 0.0),
    (-8.0, 24.0),
)


def _team(identity_base: int) -> tuple[Player, ...]:
    return tuple(
        Player(
            PlayerProfile(
                player_id=identity_base + index,
                is_goalkeeper=index == 0,
            ),
            position,
        )
        for index, position in enumerate(_FORMATION)
    )


def test_goal_always_gives_kickoff_to_the_conceding_team_eager_and_jit() -> None:
    env = FootballWorld()
    state = env.reset(_team(1_000), _team(2_000)).rollout.state
    scorer = int(np.flatnonzero(np.asarray(state.players.team_id) == TEAM_0)[1])
    state = state._replace(
        ball=state.ball._replace(live=jnp.bool_(True)),
        possession=state.possession._replace(
            team=jnp.int32(TEAM_0),
            player=jnp.int32(scorer),
            last_contact=state.possession.last_contact._replace(
                actor=jnp.int32(scorer)
            ),
        ),
    )
    goal_x = float(np.asarray(state.attack_direction[TEAM_0])) * (
        Stadium().half_length + Ball().radius
    )
    crossing = BoundaryCrossing(
        occurred=jnp.bool_(True),
        axis=jnp.int32(CROSSING_GOAL_LINE),
        time_fraction=jnp.float32(0.5),
        position=jnp.asarray((goal_x, 0.0, 1.0), dtype=jnp.float32),
        through_goal=jnp.bool_(True),
    )

    eager = resolve_boundary_crossing(state, crossing)
    compiled = jax.jit(resolve_boundary_crossing)(state, crossing)
    for result in (eager, compiled):
        assert int(result.event.kind) == BALL_EVENT_GOAL
        assert int(result.event.scoring_team) == TEAM_0
        assert int(result.state.restart.kind) == RK_KICKOFF
        assert int(result.state.restart.team) == TEAM_1
        assert int(result.state.kickoff_team) == TEAM_1
        taker = int(result.state.restart.taker)
        assert int(result.state.players.team_id[taker]) == TEAM_1
