"""Observation-only timing gates for deliberate rule-policy contacts.

The helpers in this module are pure fixed-shape array calculations. They do
not keep a second contact timer, inspect rollout state, or introduce a random
stream. This keeps contact timing causal and reproducible while allowing the
top-level policy to retain ownership of the public ``IntentAction`` contract.
"""

from __future__ import annotations

import jax

from footballworld.core.constants import STATIONARY_SPEED_EPS
from footballworld.core.contact import INTENT_CONTROL, OUTCOME_TRAP


def dribble_recontact_ready(
    ball_offset: jax.Array,
    relative_ball_velocity: jax.Array,
    ball_speed: jax.Array,
    self_touched_last: jax.Array,
    last_contact_known: jax.Array,
    last_contact_intent: jax.Array,
    last_contact_outcome: jax.Array,
) -> jax.Array:
    """Re-arm a carrier's touch only after it catches its own pushed ball.

    A successful ``CONTROL`` contact used for a dribble leaves an observable
    ``OUTCOME_TRAP`` in FootballWorld. Immediately after that touch the ball is
    moving away from the carrier, so the derivative of squared carrier-ball
    distance is positive. Another touch is permitted only after that derivative
    turns negative (the carrier has caught the ball) or the ball is physically
    stationary. Unknown or other-player contacts never create a hidden block.

    The gate uses causal intent and outcome observations rather than a hidden
    touch-code API.
    """

    expected_shape = ball_offset.shape[:-1]
    if ball_offset.shape[-1:] != (2,):
        raise ValueError("ball_offset must have a final axis of length two")
    if relative_ball_velocity.shape != ball_offset.shape:
        raise ValueError("relative_ball_velocity must match ball_offset")
    for name, value in (
        ("ball_speed", ball_speed),
        ("self_touched_last", self_touched_last),
        ("last_contact_known", last_contact_known),
        ("last_contact_intent", last_contact_intent),
        ("last_contact_outcome", last_contact_outcome),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must match the leading ball_offset axes")

    # Writing the two-dimensional dot explicitly avoids adding a reduction to
    # this scalar gate in every batched policy graph.
    distance_derivative_sign = (
        relative_ball_velocity[..., 0] * ball_offset[..., 0]
        + relative_ball_velocity[..., 1] * ball_offset[..., 1]
    )
    self_control_followup = (
        last_contact_known
        & self_touched_last
        & (last_contact_intent == INTENT_CONTROL)
        & (last_contact_outcome == OUTCOME_TRAP)
    )
    carrier_caught_ball = distance_derivative_sign < 0.0
    stalled_ball = ball_speed <= STATIONARY_SPEED_EPS
    return (~self_control_followup) | carrier_caught_ball | stalled_ball


__all__ = ["dribble_recontact_ready"]
