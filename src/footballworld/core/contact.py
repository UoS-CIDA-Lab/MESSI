"""Compact contact semantics for the rollout core."""

from typing import NamedTuple

import jax

# Physical mechanism
MECHANISM_NONE = 0
MECHANISM_FOOT = 1
MECHANISM_HEAD = 2
MECHANISM_CHEST = 3
MECHANISM_PASSIVE_BODY = 4
MECHANISM_GOALKEEPER_HAND = 5
MECHANISM_THROW = 6
MECHANISM_COUNT = 7


# Inferred contact semantics. MOVE means no deliberate ball contact.
INTENT_MOVE = 0
INTENT_CONTROL = 1
INTENT_PASS = 2
INTENT_SHOT = 3
INTENT_CLEAR = 4
# The public action calls this broad contest request CHALLENGE; interception is
# a realized outcome within that intent, not a seventh requested action.
# TACKLE remains an event-schema alias so existing logs retain code 5.
INTENT_CHALLENGE = 5
INTENT_TACKLE = INTENT_CHALLENGE
ACTION_INTENT_COUNT = 6
# Catch is a goalkeeper event inferred by physics, never a requested action.
INTENT_CATCH = 6
INTENT_COUNT = 7


# Intent provenance metadata. These codes are not selectable actions and do
# not extend the six-way public action vocabulary.
INTENT_SOURCE_SCHEMA = "footballworld.intent-source/2"
INTENT_SOURCE_NONE = 0
INTENT_SOURCE_POLICY = 1
INTENT_SOURCE_ENVIRONMENT_FORCED = 2
INTENT_SOURCE_NAMES = ("NONE", "POLICY", "ENVIRONMENT_FORCED")
INTENT_SOURCE_COUNT = len(INTENT_SOURCE_NAMES)


# Realized physical outcome
OUTCOME_NONE = 0
OUTCOME_RELEASE = 1
OUTCOME_TRAP = 2
OUTCOME_INTERCEPTION = 3
OUTCOME_TACKLE_WON = 4
OUTCOME_DEFLECTION = 5
OUTCOME_CATCH = 6
OUTCOME_PARRY = 7
OUTCOME_FOUL = 8
OUTCOME_MISCONTROL = 9
OUTCOME_COUNT = 10


# Direct causal effect on Law 11 state
LAW11_NONE = 0
LAW11_DELIBERATE_PLAY_RESET = 1
LAW11_DEFLECTION_NO_RESET = 2
LAW11_DELIBERATE_SAVE_NO_RESET = 3
LAW11_DIRECT_RESTART_EXEMPTION = 4
LAW11_EFFECT_COUNT = 5


class ContactResult(NamedTuple):
    """Realized physical response after contest context and control resolution."""

    actor: int
    mechanism: int
    intent: int
    outcome: int
    restart_kind: int
    law11_effect: int
    kick_applied: bool
    intent_source: int = INTENT_SOURCE_NONE


class ContactOccurrence(NamedTuple):
    """Fixed-shape location and timing of player contact with the ball."""

    occurred: jax.Array
    actor: jax.Array
    mechanism: jax.Array
    law11_effect: jax.Array
    position: jax.Array
    time_fraction: jax.Array
