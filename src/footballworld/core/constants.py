"""Immutable scalar codes and layouts used by the rollout core."""

import math

# Teams and sentinels
TEAM_0 = 0
TEAM_1 = 1
TEAM_COUNT = 2

NO_TEAM = -1
NO_PLAYER = -1

# Float32 numerical guards
GEOMETRY_EPS = 1.0e-6
DIV_EPS = 1.0e-9
SQUARED_EPS = 1.0e-12
SAFE_NORM_EPS = 1.0e-18
STATIONARY_SPEED_EPS = 1.0e-3
COINCIDENT_DISTANCE_EPS = 1.0e-4

# Operational guard for the dynamic player-collision loop. This is a runtime
# resource bound, not a football or biomechanical coefficient. Valid public
# configurations are checked on the host so ordinary rollouts remain far below
# it; the traced guard only fails closed for reconstructed invalid states.
MAX_PLAYER_COLLISION_MICROSTEPS = 32


# Geometry
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


# Restart kind
RK_NONE = 0
RK_KICKOFF = 1
RK_THROWIN = 2
RK_GOALKICK = 3
RK_CORNER = 4
RK_FREEKICK = 5
RK_PENALTY = 6
RK_OFFSIDE = 7
RK_GK_HOLD = 8
RESTART_COUNT = 9


# Discipline outcomes
DISCIPLINE_NONE = 0
DISCIPLINE_YELLOW = 1
DISCIPLINE_RED = 2


# Boundary events
BALL_EVENT_NONE = 0
BALL_EVENT_GOAL = 1
BALL_EVENT_CORNER = 2
BALL_EVENT_GOALKICK = 3
BALL_EVENT_THROWIN = 4

WOODWORK_NONE = 0
WOODWORK_POST = 1
WOODWORK_CROSSBAR = 2


# Normalized continuous action bounds
ACTION_MIN = -1.0
ACTION_MAX = 1.0

# Explicit-intent action layout. The categorical intent is stored separately;
# these indices address only the eight continuous controls.
INTENT_ACTION_CONTINUOUS_DIM = 8
INTENT_ACTION_MOVE = slice(0, 2)
INTENT_ACTION_FORCE_TO_BALL = slice(2, 4)
INTENT_ACTION_LAUNCH = 4
INTENT_ACTION_SPIN_SIDE = 5
INTENT_ACTION_SPIN_BACK = 6
INTENT_ACTION_GAZE_CENTER = 7


# Law-level roster limits
YELLOW_CARD_SEND_OFF_COUNT = 2
IFAB_MIN_TEAM_PLAYERS = 7
IFAB_MAX_TEAM_PLAYERS = 11
