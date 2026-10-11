"""Connect the public rule policy to bounded match recording."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.analysis.recording import make_match_recording_chunk
from messi.core.ball import BallPhysics
from messi.core.contact import ContactPhysics
from messi.core.offside import DEFAULT_OFFSIDE_GEOMETRY
from messi.core.timebase import DEFAULT_TIMEBASE, Timebase
from messi.environment.episode import DEFAULT_MATCH_CONFIG, MatchConfig
from messi.policies.rule_based import (
    DEFAULT_RULE_POLICY_CONFIG,
    DEFAULT_RULE_TACTICAL_PLAN,
    observe_policy,
    rule_policy_with_diagnostics,
)


class PolicyDecisionTelemetry(NamedTuple):
    carrier_valid: jax.Array
    carrier: jax.Array
    local_teammates: jax.Array
    local_opponents: jax.Array
    target: jax.Array
    target_direction_family: jax.Array
    target_completion: jax.Array
    target_score_gap: jax.Array
    target_next_outlet_quality: jax.Array
    selected_intent: jax.Array


def make_rule_policy_recording_chunk(
    max_steps: int,
    *,
    config=DEFAULT_RULE_POLICY_CONFIG,
    match_config: MatchConfig = DEFAULT_MATCH_CONFIG,
    timebase: Timebase = DEFAULT_TIMEBASE,
    **transition_kwargs,
):
    """Create a bounded match recorder using observation-only rule actions."""

    contact_physics = transition_kwargs.get("contact_physics", ContactPhysics())
    ball_physics = transition_kwargs.get("ball_physics", BallPhysics())
    offside_geometry = transition_kwargs.get(
        "offside_geometry", DEFAULT_OFFSIDE_GEOMETRY
    )

    def policy(state, roster, tactical_plan, policy_key):
        observation = observe_policy(
            state,
            roster,
            config=config,
            offside_geometry=offside_geometry,
            timebase=timebase,
        )
        action, diagnostics = rule_policy_with_diagnostics(
            observation,
            config=config,
            ball_physics=ball_physics,
            contact_physics=contact_physics,
            tactical_plan=tactical_plan,
            key=policy_key,
        )
        safe_carrier = jnp.clip(diagnostics.carrier, 0, action.intent.shape[0] - 1)
        return action, PolicyDecisionTelemetry(
            carrier_valid=diagnostics.carrier_valid,
            carrier=diagnostics.carrier,
            local_teammates=diagnostics.carrier_local_teammates,
            local_opponents=diagnostics.carrier_local_opponents,
            target=diagnostics.target_player[safe_carrier],
            target_direction_family=diagnostics.target_direction_family,
            target_completion=diagnostics.pass_completion[safe_carrier],
            target_score_gap=diagnostics.target_score_gap,
            target_next_outlet_quality=diagnostics.target_next_outlet_quality,
            selected_intent=action.intent[safe_carrier],
        )

    return make_match_recording_chunk(
        max_steps,
        policy,
        match_config=match_config,
        timebase=timebase,
        policy_returns_telemetry=True,
        policy_takes_key=True,
        policy_takes_context=True,
        policy_context=DEFAULT_RULE_TACTICAL_PLAN,
        **transition_kwargs,
    )
