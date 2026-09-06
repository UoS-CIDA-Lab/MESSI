"""Seeded possession-episode patterns for the rule player policy.

``TacticalPlan`` describes the team's persistent structural identity.  An
``AttackPattern`` is deliberately narrower: one reproducible attacking idea
selected for the currently observed team-possession episode.  Pattern codes
never change action legality.  They only apply bounded score and movement
biases in the caller, and every unavailable pattern falls back to the ordinary
policy ranking.

The prior table is a FootballWorld design prior, not a fit to tracking or event
data.  Its rows follow the stable :class:`TacticalPlan` integer order and its
columns follow :class:`AttackPattern`.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp

from footballworld.policies.rule_based.tactical_plan import TACTICAL_PLAN_COUNT


class AttackPattern(IntEnum):
    """Short-horizon ideas retained across one observed possession episode."""

    PROGRESSIVE_CARRY = 0
    THIRD_MAN = 1
    WIDE_OVERLOAD = 2
    SWITCH_PLAY = 3
    RUN_BEHIND_DIRECT = 4


ATTACK_PATTERN_COUNT = len(AttackPattern)

# salida lavolpiana, juego de posicion, gegenpress, catenaccio, zona mista.
# The values are normalized explicitly so future edits fail visibly instead of
# relying on categorical logits to hide a malformed probability row.
_TACTICAL_PATTERN_PRIOR = (
    (0.18, 0.28, 0.18, 0.20, 0.16),
    (0.16, 0.28, 0.22, 0.22, 0.12),
    (0.24, 0.20, 0.16, 0.12, 0.28),
    (0.12, 0.10, 0.12, 0.18, 0.48),
    (0.16, 0.18, 0.30, 0.22, 0.14),
)
_PATTERN_DRAW_STREAM = 0x4154504E
_CARRY_COMMIT_STREAM = 0x434D4954


def _prior_table() -> jax.Array:
    table = jnp.asarray(_TACTICAL_PATTERN_PRIOR, dtype=jnp.float32)
    if table.shape != (TACTICAL_PLAN_COUNT, ATTACK_PATTERN_COUNT):
        raise ValueError("attack-pattern prior must match tactical and pattern counts")
    return table / jnp.sum(table, axis=-1, keepdims=True)


def select_attack_pattern(
    episode_key: jax.Array,
    tactical_plan_code: jax.Array,
) -> jax.Array:
    """Draw one stable pattern from a team-possession episode key."""

    row = jnp.clip(
        jnp.asarray(tactical_plan_code, dtype=jnp.int32),
        0,
        TACTICAL_PLAN_COUNT - 1,
    )
    probability = _prior_table()[row]
    key = jax.random.fold_in(episode_key, _PATTERN_DRAW_STREAM)
    # One uniform variate plus cumulative thresholds preserves the categorical
    # prior without materializing one Gumbel variate per pattern. This keeps
    # observer-local episode draws small when the helper is vmapped.
    draw = jax.random.uniform(key, dtype=jnp.float32)
    return jnp.sum(draw >= jnp.cumsum(probability)[:-1]).astype(jnp.int32)


def progressive_carry_commit_seconds(
    episode_key: jax.Array,
    *,
    minimum_s: float,
    maximum_s: float,
) -> jax.Array:
    """Return one seeded carry commitment duration for the episode."""

    minimum = jnp.maximum(jnp.asarray(minimum_s, dtype=jnp.float32), 0.0)
    maximum = jnp.maximum(jnp.asarray(maximum_s, dtype=jnp.float32), minimum)
    draw = jax.random.uniform(
        jax.random.fold_in(episode_key, _CARRY_COMMIT_STREAM),
        dtype=jnp.float32,
    )
    return minimum + (maximum - minimum) * draw


__all__ = [
    "ATTACK_PATTERN_COUNT",
    "AttackPattern",
    "progressive_carry_commit_seconds",
    "select_attack_pattern",
]
