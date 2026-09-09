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
)
from footballworld.policies.rule_based.tactical_plan import gather_tactical_profile


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


def _shape_call(context, state, *, phase, lateral_shift=2.4):
    player_count = context.self_index.shape[0]
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
        own_possession=jnp.zeros((player_count,), dtype=jnp.bool_),
        opponent_possession=jnp.zeros((player_count,), dtype=jnp.bool_),
        nearest_pressure=jnp.zeros((player_count,), dtype=jnp.bool_),
        counterpress_active=jnp.zeros((player_count,), dtype=jnp.bool_),
        tactical=gather_tactical_profile(jnp.zeros((player_count,), dtype=jnp.int32)),
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
def test_new_time_windows_must_be_positive(field):
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
