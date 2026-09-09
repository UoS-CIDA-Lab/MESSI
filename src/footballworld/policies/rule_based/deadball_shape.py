"""Compact K-League transfer prior for ordinary restart team shape.

The private player-level positioning field and its fitting pipeline are not
distributed with FootballWorld.  This module retains only team-level moments
needed by the built-in policy: attacking-frame centroid and longitudinal /
lateral spread for each supported restart phase and pitch third.  Individual
roles remain FootballWorld formation anchors, so the policy stays compatible
with its fixed-shape roster and formation contracts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    RK_FREEKICK,
    RK_OFFSIDE,
)
from footballworld.policies.rule_based.context import RulePolicyContext
from footballworld.policies.rule_based.state import ROLE_GOALKEEPER, RulePolicyState

# [restart kind, side (0 restart team / 1 defending team), pitch third,
#  (centroid x, positive-ball-side centroid y, x std, y std)].  These are
# compressed team moments from the SoccerWorld K-League transfer fit, not DFL
# measurements and not a player-level data product. Unsupported cells remain
# invalid and fall back to FootballWorld's ordinary formation field.
_MOMENTS = jnp.asarray(
    (
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,  # none
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,  # kickoff: procedural formation
        (  # throw-in
            (
                (-26.136, 17.232, 9.932, 8.525),
                (2.184, 12.497, 9.048, 10.415),
                (31.300, 12.479, 8.774, 6.415),
            ),
            (
                (-35.615, 11.804, 5.288, 4.716),
                (-8.131, 13.985, 6.853, 8.132),
                (19.632, 18.576, 9.340, 6.619),
            ),
        ),
        (  # goal kick: only physically reachable own/defending thirds
            (
                (-9.900, 4.888, 11.715, 12.392),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (5.665, 5.080, 10.182, 9.794)),
        ),
        (  # corner: only physically reachable own/defending thirds
            ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (35.535, 1.582, 4.336, 4.349)),
            (
                (-41.315, 0.571, 1.257, 1.262),
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
        ),
        (  # free kick
            (
                (-3.699, 5.972, 9.285, 11.273),
                (10.844, 2.362, 8.553, 11.086),
                (34.235, -0.249, 4.607, 3.067),
            ),
            (
                (-40.510, 1.254, 0.354, 1.992),
                (-16.736, 2.334, 6.251, 8.731),
                (-1.467, 6.303, 7.904, 9.279),
            ),
        ),
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,  # penalty: procedural formation
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,  # offside maps to free kick
        (((0.0, 0.0, 0.0, 0.0),) * 3,) * 2,  # goalkeeper hold: live shape
    ),
    dtype=jnp.float32,
)

_VALID = jnp.any(jnp.abs(_MOMENTS) > 0.0, axis=-1)


def restart_shape_target(
    context: RulePolicyContext,
    policy_state: RulePolicyState,
    restart_kind: jax.Array,
    own_restart: jax.Array,
    *,
    half_length: float,
    half_width: float,
) -> tuple[jax.Array, jax.Array]:
    """Return a compact measured-phase target and availability per observer."""

    rows = context.self_index.shape[0]
    row = jnp.arange(rows, dtype=jnp.int32)
    kind = jnp.asarray(restart_kind, dtype=jnp.int32)
    kind = jnp.where(kind == RK_OFFSIDE, jnp.int32(RK_FREEKICK), kind)
    side_index = jnp.where(own_restart, 0, 1).astype(jnp.int32)
    ball = context.ball_position[:, :2]
    zone = jnp.where(ball[:, 0] < -17.5, 0, jnp.where(ball[:, 0] > 17.5, 2, 1))
    moment = _MOMENTS[kind, side_index, zone]
    valid = _VALID[kind, side_index, zone] & context.ball_visible

    roles = policy_state.role[None, :]
    outfield = (
        context.same_team
        & context.participating
        & (roles != jnp.int32(ROLE_GOALKEEPER))
    )
    count = jnp.maximum(jnp.sum(outfield, axis=1), 1)
    anchors = policy_state.formation_anchor[None, :, :]
    center = jnp.sum(jnp.where(outfield[..., None], anchors, 0.0), axis=1)
    center = center / count[:, None]
    variance = (
        jnp.sum(
            jnp.where(outfield[..., None], (anchors - center[:, None, :]) ** 2, 0.0),
            axis=1,
        )
        / count[:, None]
    )
    scale = jnp.sqrt(jnp.maximum(variance, jnp.float32(1.0)))
    self_anchor = policy_state.formation_anchor[context.self_index]
    normalized = (self_anchor - center) / scale

    # Conditional means can collapse near-goal restart cells.  Five metres of
    # lateral and three metres of longitudinal standard deviation are bounded
    # numerical/tactical floors, not measured constants; they preserve player
    # separation while retaining the measured centroid and compactness.
    target_spread = jnp.stack(
        (
            jnp.maximum(moment[:, 2], jnp.float32(3.0)),
            jnp.maximum(moment[:, 3], jnp.float32(5.0)),
        ),
        axis=-1,
    )
    ball_side = jnp.where(ball[:, 1] < 0.0, -1.0, 1.0)
    target = jnp.stack(
        (
            moment[:, 0] + normalized[:, 0] * target_spread[:, 0],
            ball_side * moment[:, 1] + normalized[:, 1] * target_spread[:, 1],
        ),
        axis=-1,
    )
    limits = jnp.asarray((half_length - 1.0, half_width - 1.0), dtype=jnp.float32)
    target = jnp.clip(target, -limits, limits)
    self_is_gk = roles[row, context.self_index] == jnp.int32(ROLE_GOALKEEPER)
    # The transfer corpus contains full-strength shapes on a 105 x 68 m
    # pitch. Reduced teams and custom stadiums retain the ordinary procedural
    # formation rather than extrapolating this absolute-metre prior.
    standard_pitch = jnp.isclose(half_length, 52.5) & jnp.isclose(half_width, 34.0)
    valid = valid & context.self_active & (~self_is_gk) & (count == 10) & standard_pitch
    target = jnp.where(valid[:, None], target, context.self_position)
    return target.astype(jnp.float32), valid
