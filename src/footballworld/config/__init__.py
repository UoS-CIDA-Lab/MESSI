"""Immutable environment configuration types."""

from footballworld.config.action import ActionScale
from footballworld.config.ball_physics import BallPhysics
from footballworld.config.body_contact import BodyContact
from footballworld.config.body_foul import BodyFoul
from footballworld.config.contact_timing import ContactTiming
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.gk_holding import GoalkeeperHolding
from footballworld.config.management import ManagementRules
from footballworld.config.perception import Perception
from footballworld.config.player_physics import PlayerPhysics
from footballworld.config.policies import PolicySelection
from footballworld.config.reach import Reach
from footballworld.config.restart_timing import RestartTiming
from footballworld.config.roster import Player, PlayerProfile
from footballworld.config.roster_sampling import RosterSampling
from footballworld.config.stamina import LongStamina, ShortStamina

__all__ = [
    "ActionScale",
    "Ball",
    "BallPhysics",
    "BodyContact",
    "BodyFoul",
    "ContactTiming",
    "Contest",
    "GoalkeeperHolding",
    "LongStamina",
    "ManagementRules",
    "Perception",
    "Player",
    "PlayerPhysics",
    "PlayerProfile",
    "PolicySelection",
    "Reach",
    "RestartTiming",
    "RosterSampling",
    "ShortStamina",
    "Stadium",
]
