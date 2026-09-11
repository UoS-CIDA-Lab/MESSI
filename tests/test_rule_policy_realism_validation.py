"""Focused adversarial checks for the rule-policy realism controls.

These tests exercise only public-observation geometry and causal policy memory.
The new coefficients are policy design priors, not measured football constants.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE
from footballworld.policies.rule_based.config import RulePolicyConfig
from footballworld.policies.rule_based.context import RulePolicyContext
from footballworld.policies.rule_based.possession import (
    POSSESSION_PASS,
    POSSESSION_SHOT,
    ShotPlan,
    _backward_pass_cost,
    _pass_width_context,
    decide_possession,
)
from footballworld.policies.rule_based.shape import shape_movement
from footballworld.policies.rule_based.state import (
    INACTIVE_AGE,
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MIDFIELDER,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    ROLE_WIDE_FORWARD,
    RulePolicyState,
    _next_carrier_age,
)
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    gather_tactical_profile,
    tactical_plan_code,
)
from footballworld.policies.rule_based.tactics import moving_receiver_target


def _carrier_context(*, teammate_available: bool = True) -> RulePolicyContext:
    """Build one carrier's fixed-shape, observation-local geometry."""

    position = jnp.asarray(
        ((0.0, 0.0), (12.0, 1.0), (42.0, 14.0), (42.0, -14.0)),
        dtype=jnp.float32,
    )
    teammate = jnp.asarray((False, teammate_available, False, False), dtype=jnp.bool_)
    opponent = jnp.asarray((False, False, True, True), dtype=jnp.bool_)
    return RulePolicyContext(
        self_index=jnp.int32(0),
        self_team=jnp.int32(0),
        self_active=jnp.bool_(True),
        self_goalkeeper=jnp.bool_(False),
        self_position=position[0],
        self_velocity=jnp.zeros((2,), dtype=jnp.float32),
        player_position=position,
        player_velocity=jnp.zeros_like(position),
        player_visible=jnp.ones((4,), dtype=jnp.bool_),
        participating=jnp.ones((4,), dtype=jnp.bool_),
        same_team=jnp.asarray((True, True, False, False), dtype=jnp.bool_),
        teammate=teammate,
        opponent=opponent,
        ball_position=jnp.asarray((0.0, 0.0, 0.11), dtype=jnp.float32),
        ball_velocity=jnp.zeros((3,), dtype=jnp.float32),
        ball_spin=jnp.zeros((3,), dtype=jnp.float32),
        ball_visible=jnp.bool_(True),
    )


def _possession_kwargs(*, teammate_available: bool = True):
    candidate = jnp.asarray((False, teammate_available, False, False), dtype=jnp.bool_)
    return {
        "half_length": 52.5,
        "half_width": 34.0,
        "goal_width": 7.32,
        "pass_target_xy": jnp.asarray(
            ((0.0, 0.0), (12.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
            dtype=jnp.float32,
        ),
        "pass_completion": jnp.asarray((0.0, 0.70, 0.0, 0.0), dtype=jnp.float32),
        "pass_candidate": candidate,
        "decision_key": None,
    }


def _fixed_shot(*, value: float, quality: float):
    def plan(*_args, **_kwargs):
        return ShotPlan(
            direction=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
            power=jnp.float32(1.0),
            launch=jnp.float32(-0.5),
            spin=jnp.zeros((2,), dtype=jnp.float32),
            value=jnp.float32(value),
            quality=jnp.float32(quality),
        )

    return plan


def _fixed_ranking_metrics(monkeypatch) -> None:
    """Remove unrelated geometric ranking variation from focused gate tests."""

    import footballworld.policies.rule_based.possession as possession_module

    def zeros(position, *_args, **_kwargs):
        return jnp.zeros(jnp.asarray(position).shape[:-1], dtype=jnp.float32)

    def pitch(position, *_args, **_kwargs):
        return jnp.full(
            jnp.asarray(position).shape[:-1], jnp.float32(0.20), dtype=jnp.float32
        )

    monkeypatch.setattr(possession_module, "pressure", zeros)
    monkeypatch.setattr(possession_module, "openness", zeros)
    monkeypatch.setattr(possession_module, "shot_quality", zeros)
    monkeypatch.setattr(possession_module, "pitch_value", pitch)


def test_productive_width_and_switch_reduce_only_safe_lateral_cost():
    central_wide, central_switch, central_scale = _pass_width_context(
        jnp.float32(0.0),
        jnp.asarray((0.0, 30.0), dtype=jnp.float32),
        jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        jnp.float32(34.0),
    )
    assert not bool(central_wide)
    np.testing.assert_array_equal(np.asarray(central_switch), (False, False))
    assert float(central_scale[0]) == pytest.approx(1.0)
    assert float(central_scale[1]) < 0.12

    wide, switch, safe_scale = _pass_width_context(
        jnp.float32(20.0),
        jnp.asarray((-20.0, 20.0), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        jnp.float32(34.0),
    )
    assert bool(wide)
    np.testing.assert_array_equal(np.asarray(switch), (True, False))
    assert float(safe_scale[0]) == pytest.approx(0.0)
    assert float(safe_scale[1]) == pytest.approx(1.0)

    _, _, unsafe_scale = _pass_width_context(
        jnp.float32(0.0),
        jnp.asarray((30.0,), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.float32(34.0),
    )
    assert float(unsafe_scale[0]) == pytest.approx(1.0)


def test_moving_receiver_lead_and_advanced_recycling_cost_are_bounded():
    config = RulePolicyConfig()
    receiver = jnp.asarray(((18.0, 2.0),), dtype=jnp.float32)
    target = moving_receiver_target(
        jnp.asarray((0.0, 0.0), dtype=jnp.float32),
        receiver,
        jnp.asarray(((12.0, 0.0),), dtype=jnp.float32),
        jnp.asarray((True,), dtype=jnp.bool_),
        half_length=52.5,
        half_width=34.0,
        velocity_weight=config.pass_receiver_velocity_weight,
        lead_time_cap_s=config.pass_receiver_lead_time_cap_s,
        lead_distance_cap_m=config.pass_receiver_lead_distance_cap_m,
    )
    lead = float(jnp.linalg.norm(target[0] - receiver[0]))
    assert 0.0 < lead <= config.pass_receiver_lead_distance_cap_m + 1e-6

    own_half = _backward_pass_cost(-0.5, 0.0, 0.25, 0.18, 0.12)
    advanced = _backward_pass_cost(-0.5, 0.0, 0.90, 0.18, 0.12)
    pressured = _backward_pass_cost(-0.5, 1.0, 0.90, 0.18, 0.12)
    assert float(advanced) > float(own_half)
    assert float(pressured) == pytest.approx(0.0)


def test_turnover_settle_suppresses_only_fresh_low_quality_shots(monkeypatch):
    """The settle window must not become a blanket shot prohibition."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.70, quality=0.30)
    )
    context = _carrier_context(teammate_available=True)
    config = replace(
        RulePolicyConfig(),
        solo_carry_value_decay=0.0,
        progressive_pass_value_gain=0.0,
        attack_pattern_receiver_gain=0.0,
        forward_pocket_receiver_gain=0.0,
        continuation_value_gain=0.0,
    )
    kwargs = _possession_kwargs(teammate_available=True)

    fresh = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(0.0),
        **kwargs,
    )
    settled = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(config.turnover_shot_settle_s),
        **kwargs,
    )
    linked = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(0.0),
        previous_actor=jnp.asarray((False, True, False, False)),
        **kwargs,
    )

    assert int(fresh.kind) != POSSESSION_SHOT
    assert int(settled.kind) == POSSESSION_SHOT
    assert int(linked.kind) == POSSESSION_SHOT

    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.70, quality=0.80)
    )
    clear_chance = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(0.0),
        **kwargs,
    )
    assert int(clear_chance.kind) == POSSESSION_SHOT


def test_pass_macro_scale_changes_only_macro_action_competition(monkeypatch):
    """A safe receiver remains available when another macro action wins."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.35, quality=0.35)
    )
    context = _carrier_context(teammate_available=True)
    kwargs = _possession_kwargs(teammate_available=True)
    common = {
        "solo_carry_value_decay": 0.0,
        "progressive_pass_value_gain": 0.0,
        "attack_pattern_receiver_gain": 0.0,
        "forward_pocket_receiver_gain": 0.0,
        "continuation_value_gain": 0.0,
    }

    pass_favoured = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        replace(RulePolicyConfig(), pass_macro_value_scale=1.0, **common),
        **kwargs,
    )
    shot_favoured = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        replace(RulePolicyConfig(), pass_macro_value_scale=0.10, **common),
        **kwargs,
    )

    assert int(pass_favoured.kind) == POSSESSION_PASS
    assert int(shot_favoured.kind) != POSSESSION_PASS
    assert int(shot_favoured.target) == 1


def test_pass_macro_uses_the_service_that_would_actually_be_executed(monkeypatch):
    """A weak sampled outlet must not borrow a stronger receiver's utility."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.30, quality=0.30)
    )
    context = _carrier_context(teammate_available=True)
    # Add a second teammate whose safe outlet is much stronger than the one
    # deliberately selected below.
    context = context._replace(
        teammate=jnp.asarray((False, True, True, False), dtype=jnp.bool_),
        opponent=jnp.asarray((False, False, False, True), dtype=jnp.bool_),
        same_team=jnp.asarray((True, True, True, False), dtype=jnp.bool_),
    )
    pass_candidate = jnp.asarray((False, True, True, False), dtype=jnp.bool_)
    kwargs = _possession_kwargs(teammate_available=True) | {
        "pass_candidate": pass_candidate,
        "pass_completion": jnp.asarray((0.0, 0.05, 0.95, 0.0), dtype=jnp.float32),
        "pass_target_xy": jnp.asarray(
            ((0.0, 0.0), (12.0, 1.0), (24.0, 5.0), (0.0, 0.0)),
            dtype=jnp.float32,
        ),
    }

    original_masked_categorical = possession_module._masked_categorical
    calls = 0

    def select_weak_service(values, eligible, key, temperature):
        nonlocal calls
        calls += 1
        if calls == 1:
            return jnp.int32(1), jnp.bool_(True)
        return original_masked_categorical(values, eligible, key, temperature)

    monkeypatch.setattr(
        possession_module, "_masked_categorical", select_weak_service
    )
    decision = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        replace(
            RulePolicyConfig(),
            solo_carry_value_decay=0.0,
            progressive_pass_value_gain=0.0,
            attack_pattern_receiver_gain=0.0,
            forward_pocket_receiver_gain=0.0,
            continuation_value_gain=0.0,
        ),
        **kwargs,
    )

    assert int(decision.target) == 1
    assert int(decision.kind) != POSSESSION_PASS

def test_receiver_does_not_inherit_team_episode_carry_urgency(monkeypatch):
    """An observed previous teammate makes old team-episode age inert."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.01, quality=0.01)
    )
    context = _carrier_context(teammate_available=True)
    config = RulePolicyConfig()
    kwargs = _possession_kwargs(teammate_available=True)
    previous_actor = jnp.asarray((False, True, False, False), dtype=jnp.bool_)

    new_episode = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.2),
        possession_episode_seconds=jnp.float32(0.2),
        previous_actor=previous_actor,
        **kwargs,
    )
    old_team_episode = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.2),
        possession_episode_seconds=jnp.float32(20.0),
        previous_actor=previous_actor,
        **kwargs,
    )

    for left, right in zip(new_episode, old_team_episode, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_carrier_age_bridges_self_control_lineage_and_is_fail_closed():
    """Self recontacts continue age; handoffs, losses and hidden rows are safe."""

    args = (
        jnp.asarray((13, 13, 13, 13, 13), dtype=jnp.int32),
        jnp.asarray((5, 5, 5, 5, 5), dtype=jnp.int32),
        jnp.asarray((5, 7, 0, 0, 0), dtype=jnp.int32),
        jnp.asarray((1, 1, 0, 0, 0), dtype=jnp.int32),
        jnp.asarray((True, True, False, False, False), dtype=jnp.bool_),
        jnp.asarray((True, True, False, False, False), dtype=jnp.bool_),
        jnp.asarray((True, True, True, True, False), dtype=jnp.bool_),
        jnp.asarray((False, False, True, False, False), dtype=jnp.bool_),
        jnp.ones((5,), dtype=jnp.int32),
    )
    expected = np.asarray((14, 0, 14, INACTIVE_AGE, 13), dtype=np.int32)

    eager = _next_carrier_age(*args)
    compiled = jax.jit(_next_carrier_age)(*args)

    np.testing.assert_array_equal(np.asarray(eager), expected)
    np.testing.assert_array_equal(np.asarray(compiled), expected)


def test_linked_receiver_must_release_at_personal_soft_limit(monkeypatch):
    """A prior teammate cannot exempt the current carrier from safe release."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.01, quality=0.01)
    )
    config = RulePolicyConfig()
    result = decide_possession(
        _carrier_context(teammate_available=True),
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(config.solo_carry_soft_limit_s),
        possession_episode_seconds=jnp.float32(20.0),
        previous_actor=jnp.asarray((False, True, False, False), dtype=jnp.bool_),
        **_possession_kwargs(teammate_available=True),
    )

    assert int(result.kind) == POSSESSION_PASS


@pytest.mark.parametrize(
    ("hide_previous", "offside_previous"),
    ((True, False), (False, True)),
    ids=("invisible-previous", "offside-previous"),
)
def test_linked_reception_uses_causal_identity_not_current_legality(
    monkeypatch, hide_previous, offside_previous
):
    """A linked receiver ignores current visibility/offside for episode memory."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    context = _carrier_context(teammate_available=True)
    if hide_previous:
        context = context._replace(
            player_visible=context.player_visible.at[1].set(False),
            teammate=context.teammate.at[1].set(False),
        )
    offside = jnp.zeros((4,), dtype=jnp.bool_).at[1].set(offside_previous)
    previous_actor = jnp.asarray((False, True, False, False), dtype=jnp.bool_)
    config = replace(
        RulePolicyConfig(),
        solo_carry_value_decay=0.0,
        progressive_pass_value_gain=0.0,
        attack_pattern_receiver_gain=0.0,
        forward_pocket_receiver_gain=0.0,
        continuation_value_gain=0.0,
        turnover_shot_value_scale=0.05,
    )
    kwargs = _possession_kwargs(teammate_available=True)

    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.01, quality=0.01)
    )
    young = decide_possession(
        context,
        offside,
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.2),
        possession_episode_seconds=jnp.float32(0.2),
        previous_actor=previous_actor,
        **kwargs,
    )
    old_team_episode = decide_possession(
        context,
        offside,
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.2),
        possession_episode_seconds=jnp.float32(20.0),
        previous_actor=previous_actor,
        **kwargs,
    )
    for left, right in zip(young, old_team_episode, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.70, quality=0.30)
    )
    linked = decide_possession(
        context,
        offside,
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(0.0),
        previous_actor=previous_actor,
        **kwargs,
    )
    unlinked = decide_possession(
        context,
        offside,
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_episode_seconds=jnp.float32(0.0),
        **kwargs,
    )
    assert int(linked.kind) == POSSESSION_SHOT
    assert int(unlinked.kind) != POSSESSION_SHOT


def test_long_unlinked_episode_changes_safe_release_but_not_unsafe_lane(monkeypatch):
    """Episode-age urgency is completion-gated and remains a soft preference."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.01, quality=0.01)
    )
    context = _carrier_context(teammate_available=True)
    config = RulePolicyConfig()
    safe = _possession_kwargs(teammate_available=True)
    safe["pass_target_xy"] = (
        safe["pass_target_xy"].at[1].set(jnp.asarray((1.0, 12.0), dtype=jnp.float32))
    )

    young = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.0),
        possession_episode_seconds=jnp.float32(0.0),
        **safe,
    )
    old = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.0),
        possession_episode_seconds=jnp.float32(2.0 * config.solo_carry_soft_limit_s),
        **safe,
    )
    assert int(old.kind) == POSSESSION_PASS

    unsafe = dict(safe)
    unsafe["pass_completion"] = safe["pass_completion"].at[1].set(0.40)
    unsafe_young = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.0),
        possession_episode_seconds=jnp.float32(0.0),
        **unsafe,
    )
    unsafe_old = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        possession_seconds=jnp.float32(0.0),
        possession_episode_seconds=jnp.float32(2.0 * config.solo_carry_soft_limit_s),
        **unsafe,
    )
    assert int(unsafe_old.kind) == int(unsafe_young.kind)
    # The fixture deliberately has a different young decision so this proves
    # episode age, rather than a pre-existing dominant pass, exercises the gate.
    assert int(young.kind) != int(old.kind)


def test_formation_lane_drift_is_soft_and_can_be_disabled(monkeypatch):
    """Aged unpressured carries prefer their own side without banning dribbles."""

    import footballworld.policies.rule_based.possession as possession_module

    _fixed_ranking_metrics(monkeypatch)
    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.01, quality=0.01)
    )
    context = _carrier_context(teammate_available=False)
    kwargs = _possession_kwargs(teammate_available=False)
    config = RulePolicyConfig()
    common = {
        "possession_seconds": jnp.float32(config.solo_carry_soft_limit_s),
        "possession_episode_seconds": jnp.float32(config.solo_carry_soft_limit_s),
    }

    positive = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        formation_anchor_y=jnp.float32(20.0),
        **common,
        **kwargs,
    )
    negative = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        config,
        formation_anchor_y=jnp.float32(-20.0),
        **common,
        **kwargs,
    )
    assert float(positive.direction[1]) > 0.0
    assert float(negative.direction[1]) < 0.0

    disabled = replace(config, dribble_shape_drift_penalty=0.0)
    disabled_positive = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        disabled,
        formation_anchor_y=jnp.float32(20.0),
        **common,
        **kwargs,
    )
    disabled_negative = decide_possession(
        context,
        jnp.zeros((4,), dtype=jnp.bool_),
        jnp.zeros((4,), dtype=jnp.bool_),
        disabled,
        formation_anchor_y=jnp.float32(-20.0),
        **common,
        **kwargs,
    )
    for left, right in zip(disabled_positive, disabled_negative, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _shape_fixture() -> tuple[RulePolicyContext, RulePolicyState]:
    anchor = jnp.asarray(
        (
            (-48.0, 0.0),
            (-34.0, 0.0),
            (-30.0, 22.0),
            (-16.0, -8.0),
            (-8.0, 0.0),
            (-5.0, -24.0),
        ),
        dtype=jnp.float32,
    )
    player_count = anchor.shape[0]
    self_index = jnp.arange(player_count, dtype=jnp.int32)
    team_id = jnp.asarray((0, 0, 0, 1, 1, 1), dtype=jnp.int32)
    same_team = team_id[None, :] == team_id[:, None]
    visible = jnp.ones((player_count, player_count), dtype=jnp.bool_)
    eye = jnp.eye(player_count, dtype=jnp.bool_)
    context = RulePolicyContext(
        self_index=self_index,
        self_team=team_id,
        self_active=jnp.ones((player_count,), dtype=jnp.bool_),
        self_goalkeeper=jnp.asarray((True, False, False, False, False, False)),
        self_position=anchor,
        self_velocity=jnp.zeros_like(anchor),
        player_position=jnp.broadcast_to(anchor, (player_count, player_count, 2)),
        player_velocity=jnp.zeros((player_count, player_count, 2), dtype=jnp.float32),
        player_visible=visible,
        participating=visible,
        same_team=same_team,
        teammate=same_team & (~eye),
        opponent=(~same_team) & visible,
        ball_position=jnp.zeros((player_count, 3), dtype=jnp.float32),
        ball_velocity=jnp.zeros((player_count, 3), dtype=jnp.float32),
        ball_spin=jnp.zeros((player_count, 3), dtype=jnp.float32),
        ball_visible=jnp.zeros((player_count,), dtype=jnp.bool_),
    )
    inactive = jnp.full((player_count,), INACTIVE_AGE, dtype=jnp.int32)
    no_player = jnp.full((player_count,), NO_PLAYER, dtype=jnp.int32)
    state = RulePolicyState(
        formation_anchor=anchor,
        role=jnp.asarray(
            (
                ROLE_GOALKEEPER,
                ROLE_CENTRE_BACK,
                ROLE_FULL_BACK,
                ROLE_CENTRE_MIDFIELDER,
                ROLE_CENTRE_FORWARD,
                ROLE_WIDE_FORWARD,
            ),
            dtype=jnp.int32,
        ),
        team_tactical_plan=jnp.zeros((2,), dtype=jnp.int32),
        team_slot_index=jnp.zeros((2, 11), dtype=jnp.int32),
        team_slot_valid=jnp.zeros((2, 11), dtype=jnp.bool_),
        restart_kind=jnp.full((player_count,), RK_NONE, dtype=jnp.int32),
        restart_age=inactive,
        possession_team=jnp.full((player_count,), NO_TEAM, dtype=jnp.int32),
        possession_age=inactive,
        carrier_age=inactive,
        attack_phase=inactive,
        current_possessor=no_player,
        previous_possessor=no_player,
        counterpress_age=inactive,
        loose_chaser=no_player,
        planned_receiver=no_player,
        planned_receiver_id=no_player,
        planned_arrival=jnp.zeros((player_count, 2), dtype=jnp.float32),
        planned_eta_ticks=jnp.zeros((player_count,), dtype=jnp.int32),
        service_opportunity=jnp.zeros((player_count,), dtype=jnp.bool_),
        secure_control_age=inactive,
        last_control_tick=jnp.zeros((player_count,), dtype=jnp.int32),
    )
    return context, state


def _shape_call(
    context,
    state,
    *,
    phase,
    lateral_shift=2.4,
    own_possession=None,
    opponent_possession=None,
    nearest_pressure=None,
    counterpress_active=None,
    tactical_plan=TacticalPlan.SALIDA_LAVOLPIANA,
):
    player_count = context.self_index.shape[0]
    if own_possession is None:
        own_possession = jnp.zeros((player_count,), dtype=jnp.bool_)
    if opponent_possession is None:
        opponent_possession = jnp.zeros((player_count,), dtype=jnp.bool_)
    if nearest_pressure is None:
        nearest_pressure = jnp.zeros((player_count,), dtype=jnp.bool_)
    if counterpress_active is None:
        counterpress_active = jnp.zeros((player_count,), dtype=jnp.bool_)
    return shape_movement(
        context,
        state,
        half_length=52.5,
        half_width=34.0,
        penalty_area_length=16.5,
        penalty_area_width=40.32,
        goal_width=7.32,
        cross_start_fraction=0.68,
        cross_wide_fraction=0.52,
        cross_target_central_fraction=0.45,
        box_mark_lead_s=0.25,
        box_mark_runner_margin_m=0.8,
        box_mark_ball_margin_m=1.2,
        box_mark_goal_side_distance_m=1.4,
        own_possession=jnp.asarray(own_possession, dtype=jnp.bool_),
        opponent_possession=jnp.asarray(opponent_possession, dtype=jnp.bool_),
        nearest_pressure=jnp.asarray(nearest_pressure, dtype=jnp.bool_),
        counterpress_active=jnp.asarray(counterpress_active, dtype=jnp.bool_),
        tactical=gather_tactical_profile(
            jnp.full(
                (player_count,),
                tactical_plan_code(tactical_plan),
                dtype=jnp.int32,
            )
        ),
        attack_pattern=jnp.full((player_count,), -1, dtype=jnp.int32),
        attack_phase=jnp.full((player_count,), -1, dtype=jnp.int32),
        attack_pattern_shape_shift_m=3.0,
        forward_pocket_shift_m=3.0,
        run_behind_receiver=jnp.int32(NO_PLAYER),
        run_behind_release=jnp.bool_(False),
        run_behind_timing_error=jnp.bool_(False),
        offside_line_error_m=jnp.float32(1.0),
        forward_run_min_gap_m=3.5,
        kickoff_path_phase=phase,
        kickoff_path_lateral_shift_m=lateral_shift,
    )


def test_central_final_third_attack_keeps_two_forwards_in_box_lanes():
    """Central progression must not make both box runners retreat to anchors."""

    context, state = _shape_fixture()
    player_count = context.self_index.shape[0]
    ball = jnp.broadcast_to(
        jnp.asarray((32.0, 0.0, 0.11), dtype=jnp.float32),
        (player_count, 3),
    )
    context = context._replace(
        ball_position=ball,
        ball_visible=jnp.ones((player_count,), dtype=jnp.bool_),
    )
    own_possession = np.asarray(context.self_team) == 1

    result = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        own_possession=own_possession,
    )

    # Slots 4 and 5 are the centre and wide forward for team 1. The visible
    # ball defines the Law-11 line here, so both remain just behind it rather
    # than falling back to their negative-x formation anchors.
    assert float(result.target[4, 0]) > 30.0
    assert float(result.target[5, 0]) > 30.0
    assert bool(result.urgent[4])
    assert bool(result.urgent[5])


def test_kickoff_waypoint_is_bounded_role_diverse_and_endpoint_inert():
    context, state = _shape_fixture()
    start = _shape_call(context, state, phase=jnp.float32(0.0))
    end = _shape_call(context, state, phase=jnp.float32(1.0))
    disabled = _shape_call(context, state, phase=jnp.float32(0.5), lateral_shift=0.0)
    before = _shape_call(context, state, phase=jnp.float32(-2.0))
    after = _shape_call(context, state, phase=jnp.float32(3.0))
    first = _shape_call(context, state, phase=jnp.float32(1.0 / 30.0))
    middle = _shape_call(context, state, phase=jnp.float32(0.5))

    for result in (end, disabled, before, after):
        np.testing.assert_allclose(result.target, start.target, rtol=0.0, atol=1e-6)
    first_delta = np.asarray(first.target - start.target)
    assert np.count_nonzero(np.linalg.norm(first_delta, axis=1) > 1e-3) >= 4
    delta_xy = np.asarray(middle.target - start.target)
    delta = delta_xy[:, 1]
    assert delta[0] == pytest.approx(0.0, abs=1e-6)
    assert np.count_nonzero(np.abs(delta) > 0.1) >= 4
    assert len(np.unique(np.round(np.abs(delta[np.abs(delta) > 0.1]), 3))) >= 3
    assert float(np.max(np.abs(delta_xy[:, 0]))) <= 1.5 * 2.4 + 1e-6
    assert float(np.max(np.abs(delta_xy[:, 1]))) <= 2.0 * 2.4 + 1e-6


def test_attacking_shape_advances_by_phase_and_establishes_width_before_service():
    context, state = _shape_fixture()
    player_count = context.self_index.shape[0]
    # Include selected wide roles on both sides of Team 0. A central ball must
    # preserve both formation-relative outlets without choosing an arbitrary run side.
    team_id = jnp.asarray((0, 0, 0, 1, 1, 0), dtype=jnp.int32)
    same_team = team_id[None, :] == team_id[:, None]
    eye = jnp.eye(player_count, dtype=jnp.bool_)
    context = context._replace(
        self_team=team_id,
        same_team=same_team,
        teammate=same_team & (~eye),
        opponent=(~same_team),
    )
    player_position = np.broadcast_to(
        np.asarray(context.self_position), (player_count, player_count, 2)
    ).copy()
    # Give Team 0 a visible, legal forward line so the test isolates shape
    # construction rather than the final Law-11 cap.
    player_position[np.ix_((0, 1, 2, 5), (3, 4), (0,))] = np.asarray(
        (42.0, 40.0), dtype=np.float32
    )[None, :, None]
    own = jnp.asarray((True, True, True, False, False, True), dtype=jnp.bool_)
    state = state._replace(
        role=state.role.at[5].set(ROLE_FULL_BACK),
        current_possessor=jnp.asarray((1, 1, 1, 4, 4, 1), dtype=jnp.int32),
    )

    def at_ball(x):
        ball = jnp.broadcast_to(
            jnp.asarray((x, 0.0, 0.11), dtype=jnp.float32),
            (player_count, 3),
        )
        observed = context._replace(
            player_position=jnp.asarray(player_position),
            ball_position=ball,
            ball_visible=jnp.ones((player_count,), dtype=jnp.bool_),
        )
        return _shape_call(observed, state, phase=jnp.float32(0.0), own_possession=own)

    build = at_ball(-35.0)
    progression = at_ball(0.0)
    final_third = at_ball(35.0)

    # The centre-back reference moves with build-up/progression/final-third
    # ball depth instead of saturating at the old global translation cap.
    assert float(build.target[1, 0]) < float(progression.target[1, 0])
    assert float(progression.target[1, 0]) < float(final_third.target[1, 0])
    # Both stable outer roles retain more than their anchor width without being
    # pinned to the 33 m policy boundary. A central ball creates no arbitrary
    # positive-side run or urgent role.
    assert 22.0 < float(progression.target[2, 1]) < 33.0
    assert -33.0 < float(progression.target[5, 1]) < -24.0
    assert not bool(progression.urgent[2])
    assert not bool(progression.urgent[5])


def test_tactical_profiles_allocate_bounded_direct_pressers_and_keep_cover():
    context, state = _shape_fixture()
    player_count = context.self_index.shape[0]
    # Team 0 has four visible outfielders ranked by distance to a Team 1 carrier.
    team_id = jnp.asarray((0, 0, 0, 1, 0, 0), dtype=jnp.int32)
    same_team = team_id[None, :] == team_id[:, None]
    eye = jnp.eye(player_count, dtype=jnp.bool_)
    positions = jnp.asarray(
        ((-45.0, 0.0), (-1.0, 0.0), (-2.0, 0.0), (0.0, 0.0), (-3.0, 0.0), (-4.0, 0.0)),
        dtype=jnp.float32,
    )
    player_position = jnp.broadcast_to(positions, (player_count, player_count, 2))
    ball = jnp.broadcast_to(
        jnp.asarray((-0.1, 0.0, 0.11), dtype=jnp.float32),
        (player_count, 3),
    )
    context = context._replace(
        self_team=team_id,
        same_team=same_team,
        teammate=same_team & (~eye),
        opponent=(~same_team),
        self_position=positions,
        player_position=player_position,
        ball_position=ball,
        ball_visible=jnp.ones((player_count,), dtype=jnp.bool_),
    )
    state = state._replace(
        role=(
            state.role.at[4]
            .set(ROLE_CENTRE_MIDFIELDER)
            .at[5]
            .set(ROLE_CENTRE_MIDFIELDER)
        )
    )
    defending = team_id == 0
    primary = jnp.arange(player_count) == 1

    catenaccio = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        opponent_possession=defending,
        nearest_pressure=primary,
        tactical_plan=TacticalPlan.CATENACCIO,
    )
    positional = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        opponent_possession=defending,
        nearest_pressure=primary,
        tactical_plan=TacticalPlan.JUEGO_DE_POSICION,
    )
    gegenpress = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        opponent_possession=defending,
        nearest_pressure=primary,
        tactical_plan=TacticalPlan.GEGENPRESS,
    )

    np.testing.assert_array_equal(
        np.asarray(catenaccio.direct_pressure),
        (False, True, False, False, False, False),
    )
    np.testing.assert_array_equal(
        np.asarray(positional.direct_pressure),
        (False, True, True, False, False, False),
    )
    np.testing.assert_array_equal(
        np.asarray(gegenpress.direct_pressure),
        (False, True, True, False, True, False),
    )
    # Rank three remains behind the three-player press as cover, not a fourth
    # claimant at the carrier.
    assert not bool(gegenpress.direct_pressure[5])
    assert float(gegenpress.target[5, 0]) < float(ball[5, 0])

    transition = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        opponent_possession=defending,
        nearest_pressure=primary,
        counterpress_active=defending,
        tactical_plan=TacticalPlan.GEGENPRESS,
    )
    np.testing.assert_array_equal(
        np.asarray(transition.direct_pressure),
        (False, True, False, False, False, False),
    )


def test_gegenpress_uses_two_not_three_direct_pressers_high_upfield():
    context, state = _shape_fixture()
    player_count = context.self_index.shape[0]
    team_id = jnp.asarray((0, 0, 0, 1, 0, 0), dtype=jnp.int32)
    same_team = team_id[None, :] == team_id[:, None]
    eye = jnp.eye(player_count, dtype=jnp.bool_)
    positions = jnp.asarray(
        ((-45.0, 0.0), (11.0, 0.0), (9.0, 0.0), (12.0, 0.0), (8.0, 0.0), (7.0, 0.0)),
        dtype=jnp.float32,
    )
    context = context._replace(
        self_team=team_id,
        same_team=same_team,
        teammate=same_team & (~eye),
        opponent=(~same_team),
        self_position=positions,
        player_position=jnp.broadcast_to(positions, (player_count, player_count, 2)),
        ball_position=jnp.broadcast_to(
            jnp.asarray((12.0, 0.0, 0.11), dtype=jnp.float32),
            (player_count, 3),
        ),
        ball_visible=jnp.ones((player_count,), dtype=jnp.bool_),
    )
    state = state._replace(
        role=(
            state.role.at[4]
            .set(ROLE_CENTRE_MIDFIELDER)
            .at[5]
            .set(ROLE_CENTRE_MIDFIELDER)
        )
    )
    result = _shape_call(
        context,
        state,
        phase=jnp.float32(0.0),
        opponent_possession=team_id == 0,
        nearest_pressure=jnp.arange(player_count) == 1,
        tactical_plan=TacticalPlan.GEGENPRESS,
    )
    assert int(np.count_nonzero(np.asarray(result.direct_pressure))) == 2


def test_new_scalar_inputs_fail_closed_before_tracing():
    context = _carrier_context()
    config = RulePolicyConfig()
    kwargs = _possession_kwargs()
    with pytest.raises(ValueError, match="possession_episode_seconds must be scalar"):
        decide_possession(
            context,
            jnp.zeros((4,), dtype=jnp.bool_),
            jnp.zeros((4,), dtype=jnp.bool_),
            config,
            possession_episode_seconds=jnp.zeros((1,), dtype=jnp.float32),
            **kwargs,
        )
    with pytest.raises(ValueError, match="formation_anchor_y must be scalar"):
        decide_possession(
            context,
            jnp.zeros((4,), dtype=jnp.bool_),
            jnp.zeros((4,), dtype=jnp.bool_),
            config,
            formation_anchor_y=jnp.zeros((1,), dtype=jnp.float32),
            **kwargs,
        )

    shape_context, state = _shape_fixture()
    player_count = shape_context.self_index.shape[0]
    with pytest.raises(ValueError, match="kickoff_path_phase"):
        _shape_call(
            shape_context,
            state,
            phase=jnp.zeros((player_count, 1), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="kickoff_path_lateral_shift_m"):
        _shape_call(
            shape_context,
            state,
            phase=jnp.float32(0.5),
            lateral_shift=jnp.ones((player_count,), dtype=jnp.float32),
        )


@pytest.mark.parametrize("field", ("turnover_shot_settle_s", "kickoff_path_window_s"))
def test_positive_policy_controls_fail_closed_at_zero(field):
    with pytest.raises(ValueError, match=rf"{field} must be greater than zero"):
        replace(RulePolicyConfig(), **{field: 0.0})


def test_possession_changes_have_fixed_eager_and_jit_output_shape(monkeypatch):
    import footballworld.policies.rule_based.possession as possession_module

    monkeypatch.setattr(
        possession_module, "plan_shot", _fixed_shot(value=0.20, quality=0.20)
    )
    context = _carrier_context()
    config = RulePolicyConfig()
    kwargs = _possession_kwargs()

    def choose(episode_age, anchor_y, previous_actor):
        return decide_possession(
            context,
            jnp.zeros((4,), dtype=jnp.bool_),
            jnp.zeros((4,), dtype=jnp.bool_),
            config,
            possession_episode_seconds=episode_age,
            formation_anchor_y=anchor_y,
            previous_actor=previous_actor,
            **kwargs,
        )

    args = (
        jnp.float32(4.4),
        jnp.float32(14.0),
        jnp.asarray((False, False, False, False), dtype=jnp.bool_),
    )
    eager = choose(*args)
    compiled = jax.jit(choose)(*args)
    assert jax.tree.structure(eager) == jax.tree.structure(compiled)
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True
    ):
        np.testing.assert_allclose(eager_leaf, compiled_leaf, rtol=1e-6, atol=1e-6)
