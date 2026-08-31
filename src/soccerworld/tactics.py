"""Stable formation and player-role vocabulary used by commands and diagnostics."""

from enum import IntEnum

from soccerworld._engine.constants import (
    BALL_ALIVE,
    BALL_DEAD,
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
from soccerworld._engine.formation import LAYOUT_NAMES as FORMATION_LAYOUT_NAMES
from soccerworld._engine.setpiece_taker import ROLE_NAMES as PLAYER_ROLE_NAMES
from soccerworld._engine.setpiece_taker import classify_roles as classify_player_roles


class Team(IntEnum):
    HOME = TEAM_0
    AWAY = TEAM_1


class BallState(IntEnum):
    """Public ball-in-play state stored by :class:`soccerworld.State`."""

    DEAD = BALL_DEAD
    ALIVE = BALL_ALIVE


class RestartKind(IntEnum):
    NONE = RK_NONE
    KICKOFF = RK_KICKOFF
    THROW_IN = RK_THROWIN
    GOAL_KICK = RK_GOALKICK
    CORNER = RK_CORNER
    FREE_KICK = RK_FREEKICK
    PENALTY = RK_PENALTY
    OFFSIDE = RK_OFFSIDE
    GOALKEEPER_HOLD = RK_GK_HOLD


__all__ = [
    "BallState",
    "FORMATION_LAYOUT_NAMES",
    "PLAYER_ROLE_NAMES",
    "RestartKind",
    "Team",
    "classify_player_roles",
]
