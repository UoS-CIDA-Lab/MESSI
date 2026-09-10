import inspect
import json
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld import FootballWorld, managed, managed_batch
from footballworld.analysis import metrics as analysis_metrics
from footballworld.analysis.metrics import (
    _action_direction_unit,
    _attach_shot_history_and_context,
    _attack_normalized_position,
    _classify_shot_context,
    _shot_map_rows,
    _tracking_lookback_sample,
)
from footballworld.analysis.report import (
    _ball_density_chart,
    _comparison_rows,
    _pass_map_chart,
    _passing_network_charts,
    _reference_chart,
    _series_chart,
    _shot_context_chart,
    _shot_map_chart,
    _space_occupancy_chart,
    _space_occupancy_snapshot,
    _spatial_balance_bucket,
    _workload_charts,
    render_html,
)
from footballworld.policies.manager import (
    FunctionalManagerPolicy,
    FunctionalOpeningManagerPolicy,
    ManagerBoundaryState,
    manager_restart_unseen,
)
from footballworld.policies.player import FunctionalPlayerPolicy
from footballworld.rendering import (
    capture,
    events,
    integrity,
    publication,
    replay,
    tracking,
)
from footballworld.rendering.defaults import RenderStyle
from footballworld.rendering.publication import (
    _expected_video_sample_count,
    authoritative_completion_error,
    open_published_replay,
)
from footballworld.rendering.renderer import (
    _PerspectiveCamera,
    _resample_indices,
    _update_line_vertices,
    _update_polygon_vertices,
)
from footballworld.rendering.replay import METADATA_SCHEMA
from footballworld.rollout import _runtime_step_budget


class _FakeSquad:
    pass


def test_scalar_zero_horizon_skips_opening_transaction(monkeypatch):
    class FakeManagedState(NamedTuple):
        rollout: object
        setup: object
        management: object
        roster: object
        player_policy_state: object
        manager: object
        opening_formation_checked: bool = False

    class FakeSquad:
        pass

    monkeypatch.setattr(managed, "ManagedMatchState", FakeManagedState)
    monkeypatch.setattr(managed, "SquadSetup", FakeSquad)
    calls = []

    def opening(*args):
        calls.append(args)
        raise AssertionError("zero-horizon scalar run reached opening transaction")

    runner = managed.ManagedRunner(
        env=None,
        player_policy=None,
        chunk_steps=1,
        manager_policy=None,
        opening_policy=object(),
        _advance=None,
        _manager_initialize=None,
        _manager_decide=None,
        _refresh_roster=None,
        _apply_tactics=None,
        _opening_decide=opening,
        _can_handle_goalkeeper=False,
    )
    original = FakeManagedState(
        rollout=object(),
        setup=object(),
        management=object(),
        roster=object(),
        player_policy_state=object(),
        manager=SimpleNamespace(boundary=object(), policy=object()),
    )

    result = runner.run(
        original,
        FakeSquad(),
        jnp.asarray([0, 1], dtype=jnp.uint32),
        0,
    )

    assert calls == []
    assert result.state is original
    assert result.steps_executed == 0
    assert result.manager_decisions == 0
    np.testing.assert_array_equal(
        result.opening_formation_applied,
        np.zeros(2, dtype=np.bool_),
    )


def test_manager_boundary_state_rejects_non_integer_memory():
    float_tick = ManagerBoundaryState(
        processed_restart_tick=jnp.zeros(2, dtype=jnp.float32),
        processed_restart_kind=jnp.zeros(2, dtype=jnp.int32),
    )
    with np.testing.assert_raises_regex(
        TypeError, "processed_restart_tick must have an integer dtype"
    ):
        manager_restart_unseen(float_tick, jnp.int32(1), jnp.int32(1))

    bool_kind = ManagerBoundaryState(
        processed_restart_tick=jnp.zeros(2, dtype=jnp.int32),
        processed_restart_kind=jnp.zeros(2, dtype=jnp.bool_),
    )
    with np.testing.assert_raises_regex(
        TypeError, "processed_restart_kind must have an integer dtype"
    ):
        manager_restart_unseen(bool_kind, jnp.int32(1), jnp.int32(1))


def test_runtime_step_budget_fails_closed_under_jit():
    sanitize = jax.jit(lambda value: _runtime_step_budget(value, 4))

    assert int(sanitize(jnp.int32(4))) == 4
    assert int(sanitize(jnp.int32(-1))) == 0
    assert int(sanitize(jnp.int32(5))) == 0


def test_managed_runners_accept_an_external_player_policy():
    policy = FunctionalPlayerPolicy(
        initialize_fn=lambda *_args: jnp.int32(0),
        step_fn=lambda *_args: None,
    )

    scalar = managed.make_managed_runner(FootballWorld(), policy, chunk_steps=1)
    batched = managed_batch.make_managed_batch_runner(
        FootballWorld(), policy, chunk_steps=1
    )
    assert scalar.player_policy is policy
    assert batched.player_policy is policy


def test_functional_manager_adapters_reject_noncallable_functions():
    with np.testing.assert_raises_regex(TypeError, "initialize_fn must be callable"):
        FunctionalManagerPolicy(
            initialize_fn=None,
            step_fn=lambda *_args: None,
        )
    with np.testing.assert_raises_regex(TypeError, "step_fn must be callable"):
        FunctionalOpeningManagerPolicy(
            initialize_fn=lambda *_args: None,
            step_fn=None,
        )


def test_managed_runner_rejects_a_malformed_external_manager():
    malformed = SimpleNamespace(
        initialize=0,
        step=lambda *_args: None,
    )
    with np.testing.assert_raises_regex(
        TypeError,
        "manager policy initialize and step must be callable",
    ):
        managed.make_managed_runner(
            FootballWorld(),
            manager_policy=malformed,
        )


def test_managed_capture_transfers_chunk_controls_once(monkeypatch):
    calls = 0
    device_get = capture.jax.device_get

    def tracked_device_get(value):
        nonlocal calls
        calls += 1
        return device_get(value)

    monkeypatch.setattr(capture.jax, "device_get", tracked_device_get)
    result = SimpleNamespace(
        steps_executed=jnp.int32(1),
        valid=jnp.asarray([True, False]),
        steps=SimpleNamespace(event_budget_exhausted=jnp.asarray([False, True])),
        done=jnp.bool_(False),
        manager_required=jnp.bool_(True),
    )

    controls = capture._host_managed_chunk_control(result)

    assert calls == 1
    assert controls[0] == 1
    np.testing.assert_array_equal(controls[1], [True, False])
    np.testing.assert_array_equal(controls[2], [False, True])
    assert controls[3:] == (False, True)


def test_unmanaged_capture_transfers_chunk_controls_once(monkeypatch):
    calls = 0
    device_get = capture.jax.device_get

    def tracked_device_get(value):
        nonlocal calls
        calls += 1
        return device_get(value)

    monkeypatch.setattr(capture.jax, "device_get", tracked_device_get)
    result = SimpleNamespace(
        steps=SimpleNamespace(
            done=jnp.asarray([False, True]),
            event_budget_exhausted=jnp.asarray([True, False]),
        ),
        final_rollout=SimpleNamespace(
            state=SimpleNamespace(control_tick=jnp.int32(12))
        ),
    )

    controls = capture._host_event_chunk_control(result, jnp.int32(10))

    assert calls == 1
    np.testing.assert_array_equal(controls[0], [False, True])
    np.testing.assert_array_equal(controls[1], [True, False])
    assert controls[2] == 2

    result.final_rollout.state.control_tick = jnp.int32(10)
    assert capture._host_event_chunk_control(result, jnp.int32(10))[2] == 0


def test_capture_prefix_transfers_side_data_in_same_barrier(monkeypatch):
    calls = 0
    device_get = capture.jax.device_get

    def tracked_device_get(value):
        nonlocal calls
        calls += 1
        return device_get(value)

    monkeypatch.setattr(capture.jax, "device_get", tracked_device_get)
    values, side = capture._device_get_prefix(
        jnp.arange(16).reshape(4, 4),
        2,
        side=jnp.asarray([7, 8]),
    )

    assert calls == 1
    np.testing.assert_array_equal(values, np.arange(8).reshape(2, 4))
    np.testing.assert_array_equal(side, [7, 8])


def test_scalar_managed_runner_stops_when_match_is_terminal(monkeypatch):
    monkeypatch.setattr(managed, "SquadSetup", _FakeSquad)
    monkeypatch.setattr(managed, "_one_key", lambda _key: None)
    calls = 0
    device_get_calls = 0
    device_get = managed.jax.device_get

    def tracked_device_get(value):
        nonlocal device_get_calls
        device_get_calls += 1
        return device_get(value)

    monkeypatch.setattr(managed.jax, "device_get", tracked_device_get)

    def advance(
        rollout,
        _setup,
        _roster,
        policy_state,
        _boundary,
        _budget,
        _match_key,
    ):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            final_rollout=rollout,
            final_policy_state=policy_state,
            steps_executed=jnp.int32(0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
            manager_required=jnp.bool_(False),
        )

    state = managed.ManagedMatchState(
        rollout=object(),
        setup=object(),
        management=object(),
        roster=object(),
        player_policy_state=object(),
        manager=SimpleNamespace(boundary=object(), policy=object()),
        opening_formation_checked=True,
    )
    runner = managed.ManagedRunner(
        env=None,
        player_policy=None,
        chunk_steps=4,
        manager_policy=None,
        opening_policy=None,
        _advance=advance,
        _manager_initialize=None,
        _manager_decide=None,
        _refresh_roster=None,
        _apply_tactics=None,
        _opening_decide=None,
        _can_handle_goalkeeper=False,
    )

    result = runner.run(state, _FakeSquad(), object(), 10)

    assert calls == 1
    assert device_get_calls == 1
    assert result.steps_executed == 0
    assert result.state == state


def test_batch_managed_runner_retires_terminal_rows(monkeypatch):
    monkeypatch.setattr(managed_batch, "_batch_size", lambda _rollout: 2)
    monkeypatch.setattr(managed_batch, "_validate_batch_tree", lambda *_args: None)
    monkeypatch.setattr(managed_batch, "_validate_keys", lambda *_args: None)

    def advance(
        rollout,
        _setup,
        _roster,
        policy_state,
        _boundary,
        _budget,
        _match_keys,
    ):
        return SimpleNamespace(
            final_rollout=rollout,
            final_policy_state=policy_state,
            steps_executed=jnp.asarray([0, 1], dtype=jnp.int32),
            terminated=jnp.asarray([True, False]),
            truncated=jnp.asarray([False, False]),
            manager_required=jnp.asarray([False, False]),
        )

    opening_calls = 0

    def opening(rollout, setup, _squad, management, _keys, _active):
        nonlocal opening_calls
        opening_calls += 1
        return SimpleNamespace(
            rollout=rollout,
            setup=setup,
            management=management,
            applied=jnp.zeros((2, 2), dtype=jnp.bool_),
        )

    state = managed_batch.ManagedBatchState(
        rollout=object(),
        setup=object(),
        management=object(),
        roster=object(),
        player_policy_state=object(),
        manager=SimpleNamespace(boundary=object(), policy=object()),
    )
    runner = managed_batch.ManagedBatchRunner(
        env=None,
        player_policy=None,
        chunk_steps=4,
        manager_policy=None,
        opening_policy=None,
        _advance=advance,
        _manager_initialize=None,
        _manager_decide=None,
        _acknowledge=None,
        _refresh_roster=None,
        _apply_tactics=None,
        _opening_decide=opening,
        _can_handle_goalkeeper=False,
    )

    result = runner.run(state, object(), jnp.zeros((2, 2), dtype=jnp.uint32), [1, 1])

    np.testing.assert_array_equal(result.steps_executed, [0, 1])

    runner.run(
        result.state,
        object(),
        jnp.zeros((2, 2), dtype=jnp.uint32),
        [1, 1],
    )
    assert opening_calls == 1


def test_zero_distance_pass_map_is_renderable():
    row = {
        "team": 0,
        "start_m": [0.0, 0.0],
        "intended_target_m": None,
        "actual_end_m": [0.0, 0.0],
        "outcome": "same_team_next_contact",
        "passer_player_id": 1,
        "intended_receiver_player_id": 2,
        "actual_next_player_id": 2,
        "intended_receiver_match": True,
        "distance_m": None,
        "direction_family": "unknown",
    }
    report = {
        "visualizations": {
            "ball_density_2d": {
                "x_edges_m": [-52.5, 52.5],
                "y_edges_m": [-34.0, 34.0],
            },
            "pass_map": {"rows": [row]},
            "intent_actions": {"rows": []},
        }
    }

    rendered = _pass_map_chart(report, 0)

    assert "Unavailable m" in rendered


def test_pass_map_identifies_exact_cross_signature_and_line_break_proxy():
    source_contact = {
        "time_key": (10, 0, 0.0),
        "order": 0,
        "slot": 3,
        "actor": 3,
        "team": 0,
        "player_id": 13,
        "slot_generation": 0,
        "position": (0.0, 20.0, 0.11),
    }
    receiver_contact = {
        "time_key": (12, 0, 0.0),
        "order": 0,
        "slot": 8,
        "actor": 8,
        "team": 0,
        "player_id": 18,
        "slot_generation": 0,
        "position": (36.0, 4.0, 0.11),
    }
    rows = analysis_metrics._pass_map_rows(
        {
            "contacts": [source_contact, receiver_contact],
            "boundaries": [],
            "pass_sources": [
                {
                    "tick": 10,
                    "actor": 3,
                    "team": 0,
                    "player_id": 13,
                    "slot_generation": 0,
                    "intended_receiver_player_id": 18,
                    "applied_direction_unit": [1.0, 0.0],
                    "submitted_spin": [-0.18, 0.35],
                    "contact": source_contact,
                }
            ],
        },
        {10: (1.0, -1.0)},
        {10: {18: (4.0, 2.0)}},
        {
            10: [
                {"team": 1, "player_id": 21, "position": (40.0, 0.0)},
                {"team": 1, "player_id": 22, "position": (30.0, 5.0)},
                {"team": 1, "player_id": 23, "position": (10.0, -5.0)},
            ]
        },
        52.5,
    )

    assert len(rows) == 1
    assert rows[0]["rule_policy_cross_control_signature"] is True
    assert rows[0]["defensive_line_x_m_at_source"] == 30.0
    assert rows[0]["defensive_line_breaking_pass_proxy"] is True
    assert rows[0]["outcome"] == "same_team_next_contact"


def test_line_break_proxy_fails_closed_without_two_source_time_opponents():
    source_contact = {
        "time_key": (10, 0, 0.0),
        "order": 0,
        "slot": 3,
        "actor": 3,
        "team": 0,
        "player_id": 13,
        "slot_generation": 0,
        "position": (0.0, 0.0, 0.11),
    }
    receiver_contact = {
        "time_key": (12, 0, 0.0),
        "order": 0,
        "slot": 8,
        "actor": 8,
        "team": 0,
        "player_id": 18,
        "slot_generation": 0,
        "position": (40.0, 0.0, 0.11),
    }
    rows = analysis_metrics._pass_map_rows(
        {
            "contacts": [source_contact, receiver_contact],
            "boundaries": [],
            "pass_sources": [
                {
                    "tick": 10,
                    "actor": 3,
                    "team": 0,
                    "player_id": 13,
                    "slot_generation": 0,
                    "intended_receiver_player_id": 18,
                    "applied_direction_unit": [1.0, 0.0],
                    "submitted_spin": [0.0, 0.0],
                    "contact": source_contact,
                }
            ],
        },
        {10: (1.0, -1.0)},
        {10: {18: (4.0, 0.0)}},
        {10: [{"team": 1, "player_id": 21, "position": (30.0, 0.0)}]},
        52.5,
    )

    assert rows[0]["rule_policy_cross_control_signature"] is False
    assert rows[0]["defensive_line_x_m_at_source"] is None
    assert rows[0]["defensive_line_breaking_pass_proxy"] is False


def test_pass_map_shot_marker_exposes_attack_normalized_direction():
    np.testing.assert_allclose(
        _action_direction_unit({"force_to_ball": [0.3, -0.4]}),
        [0.6, -0.8],
    )
    assert _action_direction_unit({"force_to_ball": [0.0, 0.0]}) is None
    report = {
        "visualizations": {
            "ball_density_2d": {
                "x_edges_m": [-52.5, 52.5],
                "y_edges_m": [-34.0, 34.0],
            },
            "pass_map": {"rows": []},
            "intent_actions": {
                "rows": [
                    {
                        "control_tick": 10,
                        "team": 0,
                        "player_id": 9,
                        "slot_generation": 0,
                        "intent": 3,
                        "position_m": [20.0, 5.0],
                        "direction_unit": [0.6, -0.8],
                    }
                ]
            },
        }
    }

    rendered = _pass_map_chart(report, 0)

    assert "id='shot-direction-0'" in rendered
    assert "marker-end='url(#shot-direction-0)'" in rendered
    assert "#9 shot intent; direction (0.600, -0.800)" in rendered
    assert "x1='485.71'" in rendered


def test_dfl_reference_chart_shows_only_current_match_and_dfl():
    report = {
        "external_reference": {
            "title": "DFL guide",
            "source": {"provider": "DFL", "matches": 7, "tracking_hz": 15},
            "comparisons": [
                {
                    "label": "Forward pass share",
                    "comparability": "closest_comparable",
                    "current": 0.32,
                    "target": 0.38,
                    "baseline": 0.12,
                    "unit": "ratio",
                    "relative_delta": -0.16,
                    "definition": "Attack-normalized share.",
                    "caution": "",
                }
            ],
            "cautions": [
                "Baseline values pool four diagnostic seeds.",
                "Use held-out matches before calibration.",
            ],
            "interpretation": "Descriptive comparison.",
        }
    }

    rendered = _reference_chart(report)

    assert "Current match" in rendered
    assert "DFL guide" in rendered
    assert "Baseline values pool four diagnostic seeds." in rendered
    assert "12.0%" not in rendered
    assert "Use held-out matches before calibration." in rendered


def test_attack_normalization_is_a_180_degree_rotation():
    assert _attack_normalized_position((12.0, 7.5), 1.0) == (12.0, 7.5)
    assert _attack_normalized_position((12.0, 7.5), -1.0) == (-12.0, -7.5)


def _shot_source(contact, *, team=0, actor=2):
    return {
        "tick": int(contact["tick"]),
        "actor": actor,
        "team": team,
        "player_id": 102,
        "slot_generation": 0,
        "restart_kind": 0,
        "contact": contact,
    }


def _shot_rows(contacts, source, *, boundaries=(), woodwork=()):
    dataset = SimpleNamespace(
        metadata={"control_fps": 10.0},
        video_time_s=lambda tick: tick / 10.0,
    )
    return _shot_map_rows(
        {
            "contacts": contacts,
            "shot_sources": [source],
            "boundary_events": list(boundaries),
            "woodwork_events": list(woodwork),
        },
        {int(source["tick"]): (1.0, -1.0)},
        {},
        dataset,
    )


def test_shot_outcomes_follow_deflections_and_keep_goal_priority():
    launch = {
        "time_key": (10, 0, 0.1),
        "tick": 10,
        "actor": 2,
        "team": 0,
        "law11_effect": 1,
        "position": (31.0, -4.0),
    }
    deflection = {
        "time_key": (10, 1, 0.4),
        "tick": 10,
        "actor": 12,
        "team": 1,
        "law11_effect": 2,
        "position": (39.0, -2.0),
    }
    restart_contact = {
        "time_key": (11, 0, 0.2),
        "tick": 11,
        "actor": 12,
        "team": 1,
        "law11_effect": 1,
        "position": (0.0, 0.0),
    }
    rows = _shot_rows(
        [launch, deflection, restart_contact],
        _shot_source(launch),
        boundaries=(
            {
                "time_key": (10, 2, 0.8),
                "tick": 10,
                "kind": 1,
                "scoring_team": 0,
            },
        ),
    )

    assert [(row["category"], row["resolution"]) for row in rows] == [("goal", "goal")]
    assert rows[0]["position_m"] == [31.0, -4.0]


def test_shot_outcomes_distinguish_save_block_and_censoring():
    launch = {
        "time_key": (20, 0, 0.1),
        "tick": 20,
        "actor": 2,
        "team": 0,
        "law11_effect": 1,
        "position": (28.0, 3.0),
    }
    save = {
        "time_key": (20, 2, 0.7),
        "tick": 20,
        "actor": 11,
        "team": 1,
        "law11_effect": 3,
        "position": (50.0, 1.0),
    }
    saved = _shot_rows([launch, save], _shot_source(launch))[0]
    assert (saved["category"], saved["resolution"]) == ("on_target", "saved")

    block = dict(save, law11_effect=2)
    blocked = _shot_rows([launch, block], _shot_source(launch))[0]
    assert (blocked["category"], blocked["resolution"]) == (
        "off_target",
        "opponent_block",
    )

    unresolved = _shot_rows([launch], _shot_source(launch))[0]
    assert (unresolved["category"], unresolved["resolution"]) == (
        "unresolved",
        "capture_end",
    )


def test_spatial_balance_bucket_matches_javascript_half_up_boundaries():
    # Positive half-bucket positions must use JavaScript Math.round semantics.
    # Python's built-in round would instead choose an even bucket at .5.
    assert _spatial_balance_bucket(31.0, 1.0) == 1
    assert _spatial_balance_bucket(23.0, 9.0) == 5
    assert _spatial_balance_bucket(9.0, 23.0) == 12
    assert _spatial_balance_bucket(1.0, 31.0) == 16
    assert _spatial_balance_bucket(1.0, 0.0) == 0
    assert _spatial_balance_bucket(0.0, 1.0) == 16


def test_report_renders_time_occupancy_and_all_shot_symbols():
    report = {
        "visualizations": {
            "team_space_occupancy": {
                "x_edges_m": [-52.5, 0.0, 52.5],
                "y_edges_m": [-34.0, 0.0, 34.0],
                "windows": [
                    {
                        "index": 0,
                        "start_clock_s": 0.0,
                        "end_clock_s": 30.0,
                        "end_control_tick": 300,
                    },
                    {
                        "index": 1,
                        "start_clock_s": 30.0,
                        "end_clock_s": 60.0,
                        "end_control_tick": 600,
                    },
                ],
                "rows": [
                    [0, 0, 1, 1, 11.0],
                    [0, 1, 0, 0, 10.0],
                    [1, 0, 0, 1, 7.0],
                    [1, 1, 0, 1, 7.0],
                ],
            },
            "ball_density_2d": {
                "x_edges_m": [-52.5, 52.5],
                "y_edges_m": [-34.0, 34.0],
                "observed_live_seconds": 21.0,
                "excluded_out_of_pitch_seconds": 1.0,
                "windows": [
                    {
                        "index": 0,
                        "start_clock_s": 0.0,
                        "end_clock_s": 30.0,
                        "end_control_tick": 300,
                        "observed_live_seconds": 12.0,
                        "excluded_out_of_pitch_seconds": 1.0,
                    },
                    {
                        "index": 1,
                        "start_clock_s": 30.0,
                        "end_clock_s": 60.0,
                        "end_control_tick": 600,
                        "observed_live_seconds": 9.0,
                        "excluded_out_of_pitch_seconds": 0.0,
                    },
                ],
                "rows": [[0, 0, 0, 12.0], [1, 0, 0, 9.0]],
            },
            "shot_context": {
                "rows": [
                    {
                        "category": "counterattack",
                        "label": "Counterattack",
                        "shots": 4,
                        "goals": 1,
                        "shot_share": 1.0,
                        "goal_share": 1.0,
                        "goal_conversion": 0.25,
                    }
                ]
            },
            "shot_map": {
                "rows": [
                    {
                        "control_tick": 600,
                        "team": 0,
                        "clock_label": "01:00",
                        "player_id": 1,
                        "position_m": [30.0, 0.0],
                        "category": category,
                        "resolution": resolution,
                        "pre_shot_context": {
                            "category": "counterattack",
                            "label": "Counterattack",
                        },
                        "tracking_path": [
                            [570, -24.0, 3.0],
                            [580, -10.0, -5.0],
                            [590, 12.0, 4.0],
                        ],
                        "preceding_events": [
                            {
                                "type": "sequence_start",
                                "control_tick": 570,
                                "clock_label": "00:57",
                                "team": 0,
                                "player_id": None,
                                "label": "Regain",
                                "position_m": [-24.0, 3.0],
                            },
                            {
                                "type": "pass",
                                "clock_label": "00:58",
                                "team": 0,
                                "player_id": 2,
                                "label": "Pass",
                                "position_m": [-10.0, -5.0],
                            },
                            {
                                "type": "challenge",
                                "clock_label": "00:59",
                                "team": 0,
                                "player_id": 1,
                                "label": "Challenge",
                                "position_m": [12.0, 4.0],
                            },
                        ],
                    }
                    for category, resolution in (
                        ("goal", "goal"),
                        ("on_target", "saved"),
                        ("off_target", "opponent_block"),
                        ("unresolved", "capture_end"),
                    )
                ]
            },
        }
    }

    occupancy_data = report["visualizations"]["team_space_occupancy"]
    initial_occupancy, _ = _space_occupancy_snapshot(occupancy_data, 1)
    occupancy = _space_occupancy_chart(report)
    density = _ball_density_chart(report)
    shots = _shot_map_chart(report, 0)
    shot_context = _shot_context_chart(report)

    assert "Cumulative live-ball player density from kickoff" in occupancy
    assert "prefix.push(running.slice())" in occupancy
    assert 'T.textContent="Kickoff–"' in occupancy
    assert "cumulative attacking-half occupancy" in occupancy
    assert 'type="range"' in occupancy
    assert "not modeled territory control" in occupancy
    assert 'radialGradient id="spatial-balance-0"' in occupancy
    assert 'radialGradient id="spatial-balance-8"' in occupancy
    assert 'radialGradient id="spatial-balance-16"' in occupancy
    assert 'fill="#0b1628"' in occupancy
    assert "#ff365f" in occupancy and "#2787ff" in occupancy
    assert "red through balanced purple to blue" in occupancy
    assert "opacity shows their combined density" in occupancy
    assert 'value="1"' in occupancy
    assert initial_occupancy.count('class="spatial-density-cell"') == 3
    assert 'fill="url(#spatial-balance-8)"' in initial_occupancy
    assert "spatial-team-0" not in occupancy
    assert "spatial-team-1" not in occupancy
    assert "mix-blend-mode" not in occupancy

    assert "Ball-position density over time" in density
    assert 'class="network-time density-time" type="range"' in density
    assert "prefix.push(running.slice())" in density
    assert "cumulative live-ball density" in density
    assert 'T.textContent="Kickoff–"' in density

    assert "1 goals | 1 saved on target | 1 off target | 1 unresolved" in shots
    assert "opponent block" in shots
    assert "shot-route-track-shadow" in shots
    assert "shot-route-track" in shots
    assert "data-route-age='0.000' opacity='0.200'" in shots
    assert "data-route-age='1.000' opacity='1.000'" in shots
    assert shots.index("data-route-age='0.000'") < shots.index("data-route-age='1.000'")
    assert "shot-route-node shot-event-sequence_start" in shots
    assert "shot-route-node shot-event-pass" in shots
    assert "shot-route-node shot-event-challenge" in shots
    assert ">R</text>" in shots
    assert "marker-end='url(#shot-route-arrow-0)'" in shots
    assert "onpointerup='this.focus()'" in shots
    assert ">00:58</text>" in shots
    assert "attack start" in shots
    assert "control" in shots
    assert "challenge" in shots
    assert ">X</text>" in shots
    assert "00:58 · Pass #2" not in shots
    assert "Previous causal events" not in shots
    assert "shot-history-bg" not in shots
    assert "<table" not in shots
    assert "Counterattack" in shots
    assert shots.index("shot-symbol-layer") < shots.index("shot-overlay-layer")
    assert shots.index("shot-overlay-layer") < shots.index("shot-route-track")
    assert "shot-hit-target" in shots
    assert "Pre-shot context mix" in shot_context

    for row in report["visualizations"]["shot_map"]["rows"]:
        row["team"] = 1
    team_1_shots = _shot_map_chart(report, 1)
    assert team_1_shots.index("shot-symbol-layer") < team_1_shots.index(
        "shot-overlay-layer"
    )
    assert team_1_shots.index("shot-overlay-layer") < team_1_shots.index(
        "shot-route-track"
    )


def test_report_visualizations_fail_closed_without_verified_spatial_data():
    empty = {"visualizations": {}}
    assert "Time-resolved ball density is unavailable" in _ball_density_chart(empty)
    assert "Spatial occupancy is unavailable" in _space_occupancy_chart(empty)
    assert "Pitch geometry is unavailable" in _shot_map_chart(empty, 0)


def test_match_control_and_space_is_the_last_content_section():
    source = inspect.getsource(render_html)
    control = source.index('<section class="control-space-final">')
    assert source.index("<h2>Realized shot outcomes</h2>") < control
    assert source.index("<h2>Player workload</h2>") < control
    assert source.index("</details>") < control
    assert control < source.index('<div class="foot">')


def test_shot_progression_uses_only_positioned_same_attack_actions():
    shots = [
        {
            "control_tick": 100,
            "team": 0,
            "position_m": [30.0, 0.0],
            "category": "off_target",
            "restart_kind": 0,
        }
    ]

    def source(tick, team, position, intent=2):
        return {
            "tick": tick,
            "substep": 0,
            "order": 0,
            "intent": intent,
            "restart_kind": 0,
            "team": team,
            "player_id": 100 + tick,
            "contact": {"position": position},
        }

    summary = _attach_shot_history_and_context(
        shots,
        {
            "applied_sources": [
                source(40, 0, [-20.0, 1.0]),
                source(60, 0, [-5.0, 2.0]),
                source(70, 1, [0.0, 3.0]),
                source(85, 0, [8.0, -3.0], intent=1),
                source(90, 0, [10.0, -4.0]),
                source(100, 0, [20.0, 0.0]),
            ]
        },
        [],
        {
            (100, 0): {
                "origin": "opponent_regain",
                "start_clock_s": 5.0,
                "start_control_tick": 50,
                "elapsed_s": 5.0,
                "start_x_m": 0.0,
                "start_position_m": [0.0, 6.0],
                "tracking_path": [[50, 0.0, 6.0], [55, 2.0, 5.0], [80, 7.0, 3.0]],
            }
        },
        {},
        {100: (-1.0, 1.0)},
        SimpleNamespace(metadata={"control_fps": 10.0}),
    )

    assert [row["control_tick"] for row in shots[0]["preceding_events"]] == [
        50,
        60,
        85,
        90,
    ]
    assert [row["position_m"] for row in shots[0]["preceding_events"]] == [
        [0.0, 6.0],
        [5.0, -2.0],
        [-8.0, 3.0],
        [-10.0, 4.0],
    ]
    assert shots[0]["preceding_events"][0]["type"] == "sequence_start"
    assert shots[0]["preceding_events"][0]["label"] == "Regain"
    assert [row["type"] for row in shots[0]["preceding_events"]][1:3] == [
        "pass",
        "control_touch",
    ]
    assert shots[0]["tracking_path"] == [
        [50.0, 0.0, 6.0],
        [55.0, 2.0, 5.0],
        [80.0, 7.0, 3.0],
    ]
    assert all("absolute_position_m" not in row for row in shots[0]["preceding_events"])
    assert (
        "actual attack-normalized ball positions from tracking"
        in summary["causal_history_semantics"]
    )
    assert "no intermediate position is inferred" in summary["causal_history_semantics"]


def test_shot_progression_omits_same_tick_and_zero_length_sequence_starts():
    shots = [
        {
            "control_tick": 100,
            "team": 0,
            "position_m": [30.0, 0.0],
            "category": "off_target",
            "restart_kind": 0,
        },
        {
            "control_tick": 101,
            "team": 1,
            "position_m": [15.0, 2.0],
            "category": "off_target",
            "restart_kind": 0,
        },
    ]
    _attach_shot_history_and_context(
        shots,
        {"applied_sources": []},
        [],
        {
            (100, 0): {
                "origin": "restart",
                "start_clock_s": 10.0,
                "start_control_tick": 100,
                "elapsed_s": 0.0,
                "start_x_m": 30.0,
                "start_position_m": [30.0, 0.0],
            },
            (101, 1): {
                "origin": "loose_recovery",
                "start_clock_s": 9.9,
                "start_control_tick": 99,
                "elapsed_s": 0.2,
                "start_x_m": 15.0,
                "start_position_m": [15.0, 2.0],
            },
        },
        {},
        {100: (-1.0, 1.0), 101: (-1.0, 1.0)},
        SimpleNamespace(metadata={"control_fps": 10.0}),
    )

    assert all(row["preceding_events"] == [] for row in shots)


def test_tracking_lookback_route_does_not_require_an_event_node():
    sample = _tracking_lookback_sample(
        [
            (2, 47200, 4720.0, -12.0, 3.0),
            (2, 47330, 4733.0, -5.0, -4.0),
            (1, 47335, 4733.5, 99.0, 99.0),
        ],
        period=2,
        shot_clock_s=4734.1,
        direction=-1.0,
    )

    assert sample is not None
    assert sample["origin"] == "tracking_lookback"
    assert sample["route_basis"] == "bounded_tracking_lookback"
    assert sample["tracking_path"] == [
        [47200.0, 12.0, -3.0],
        [47330.0, 5.0, 4.0],
    ]

    shots = [
        {
            "control_tick": 47341,
            "team": 1,
            "position_m": [22.0, -10.0],
            "category": "on_target",
            "restart_kind": 5,
        }
    ]
    _attach_shot_history_and_context(
        shots,
        {
            "applied_sources": [
                {
                    "tick": 47330,
                    "substep": 0,
                    "order": 0,
                    "intent": 2,
                    "team": 1,
                    "player_id": 2001,
                    "contact": {"position": [-5.0, -4.0]},
                }
            ]
        },
        [],
        {(47341, 1): sample},
        {},
        {47341: (-1.0, 1.0)},
        SimpleNamespace(metadata={"control_fps": 10.0}),
    )
    assert shots[0]["tracking_path"]
    assert shots[0]["tracking_path_basis"] == "bounded_tracking_lookback"
    assert shots[0]["preceding_events"] == []


def test_pre_shot_context_heuristic_has_declared_exclusive_priority():
    shot = {"restart_kind": 0, "position_m": [35.0, 0.0]}

    counter = _classify_shot_context(
        shot,
        {"origin": "opponent_regain", "elapsed_s": 4.0, "start_x_m": 10.0},
    )
    quick = _classify_shot_context(
        shot,
        {"origin": "opponent_regain", "elapsed_s": 4.0, "start_x_m": 25.0},
    )
    sustained = _classify_shot_context(
        shot,
        {"origin": "loose_recovery", "elapsed_s": 14.0, "start_x_m": 20.0},
    )
    restart = _classify_shot_context(
        {"restart_kind": 4, "position_m": [35.0, 0.0]}, None
    )
    free_kick = _classify_shot_context(
        {"restart_kind": 5, "position_m": [35.0, 0.0]}, None
    )
    penalty = _classify_shot_context(
        {"restart_kind": 6, "position_m": [35.0, 0.0]}, None
    )

    assert counter["category"] == "counterattack"
    assert quick["category"] == "quick_after_regain"
    assert sustained["category"] == "sustained_buildup"
    assert restart["category"] == "restart_attack"
    assert free_kick == {
        "category": "free_kick",
        "label": "Free kick",
        "sequence_origin": "unavailable",
        "seconds_since_sequence_start": None,
        "forward_progress_m": None,
    }
    assert penalty["category"] == "penalty_kick"
    assert penalty["label"] == "Penalty kick"


def test_renderer_reuses_projection_depth_and_fixed_collection_paths():
    camera = _PerspectiveCamera(RenderStyle(width_px=320, height_px=240))
    points = np.asarray(((1.0, 2.0, 0.0), (-3.0, 4.0, 1.5)))
    projected, scale = camera.project_with_relative_scale(points)
    np.testing.assert_allclose(projected, camera.project(points))
    np.testing.assert_allclose(scale, camera.relative_scale(points))

    class Collection:
        def __init__(self, paths):
            self.paths = paths
            self.stale = False
            self.set_calls = 0

        def get_paths(self):
            return self.paths

        def set_segments(self, values):
            self.set_calls += 1
            self.paths = [
                SimpleNamespace(vertices=np.asarray(row).copy()) for row in values
            ]

        def set_verts(self, values):
            self.set_calls += 1
            self.paths = [
                SimpleNamespace(vertices=np.concatenate((row, row[:1])))
                for row in values
            ]

    lines = Collection([SimpleNamespace(vertices=np.zeros((2, 2))) for _ in range(2)])
    line_values = np.arange(8, dtype=float).reshape(2, 2, 2)
    _update_line_vertices(lines, line_values)
    line_path_ids = [id(path) for path in lines.paths]
    _update_line_vertices(lines, line_values + 10.0)
    assert lines.set_calls == 1
    assert [id(path) for path in lines.paths] == line_path_ids
    np.testing.assert_array_equal(lines.paths[1].vertices, line_values[1] + 10.0)
    assert lines.stale

    polygons = Collection(
        [SimpleNamespace(vertices=np.zeros((4, 2))) for _ in range(2)]
    )
    polygon_values = np.arange(12, dtype=float).reshape(2, 3, 2)
    _update_polygon_vertices(polygons, polygon_values)
    polygon_path_ids = [id(path) for path in polygons.paths]
    _update_polygon_vertices(polygons, polygon_values + 20.0)
    assert polygons.set_calls == 1
    assert [id(path) for path in polygons.paths] == polygon_path_ids
    np.testing.assert_array_equal(
        polygons.paths[0].vertices[:-1], polygon_values[0] + 20.0
    )
    np.testing.assert_array_equal(
        polygons.paths[0].vertices[-1], polygon_values[0, 0] + 20.0
    )
    assert polygons.stale


def test_passing_network_preserves_continuing_nodes_and_exposes_hover_heatmap():
    report = {
        "visualizations": {
            "pass_map": {
                "rows": [
                    {
                        "control_tick": 10,
                        "team": 0,
                        "passer_player_id": 1,
                        "passer_slot_generation": 0,
                        "actual_next_player_id": 2,
                        "actual_next_slot_generation": 0,
                        "outcome": "same_team_next_contact",
                        "start_m": [-10.0, 8.0],
                        "actual_end_m": [0.0, 4.0],
                    },
                    {
                        "control_tick": 30,
                        "team": 0,
                        "passer_player_id": 3,
                        "passer_slot_generation": 1,
                        "actual_next_player_id": 2,
                        "actual_next_slot_generation": 0,
                        "outcome": "same_team_next_contact",
                        "start_m": [-4.0, -6.0],
                        "actual_end_m": [2.0, 3.0],
                    },
                ]
            },
            "intent_actions": {"rows": []},
            "player_position_occupancy": {
                "x_edges_m": [-52.5, 0.0, 52.5],
                "y_edges_m": [-34.0, 0.0, 34.0],
                "rows": [
                    [10, 0, 1, 0, 0, 1, 1.0],
                    [10, 0, 2, 0, 1, 1, 1.0],
                    [30, 0, 3, 1, 0, 0, 1.0],
                    [30, 0, 2, 0, 1, 1, 1.0],
                ],
                "activity": [
                    {
                        "team": 0,
                        "player_id": 1,
                        "slot_generation": 0,
                        "first_tick": 0,
                        "last_tick": 20,
                        "initial_position_m": [-10.0, 8.0],
                    },
                    {
                        "team": 0,
                        "player_id": 2,
                        "slot_generation": 0,
                        "first_tick": 0,
                        "last_tick": 40,
                        "initial_position_m": [0.0, 4.0],
                    },
                    {
                        "team": 0,
                        "player_id": 3,
                        "slot_generation": 1,
                        "first_tick": 21,
                        "last_tick": 40,
                        "initial_position_m": [-4.0, -6.0],
                    },
                ],
            },
        },
        "timeline": [
            {
                "control_tick": 21,
                "clock_s": 2.1,
                "team": 0,
                "type": "substitution",
                "label": "#1 out, #3 in",
            }
        ],
    }

    rendered = _passing_network_charts(report)

    payload_text = rendered.split(
        '<script type="application/json" id="network-data">', 1
    )[1].split("</script>", 1)[0]
    payload = json.loads(payload_text)[0]
    assert payload["players"] == [
        [1, 0, 0, 20, -10.0, 8.0],
        [2, 0, 0, 40, 0.0, 4.0],
        [3, 1, 21, 40, -4.0, -6.0],
    ]
    assert "network-heatmap" in rendered
    assert 'addEventListener("pointerenter"' in rendered
    assert "d.passes.filter(x=>x.tick<=cut)" in rendered
    assert "x.tick>=start" not in rendered

    assert "pairs[key]||(pairs[key]={a:first,b:second,ab:0,ba:0})" in rendered
    assert "if(a===first)e.ab++;else e.ba++" in rendered
    assert "headSize=n=>n?Math.min(10,4+1.8*Math.sqrt(n)):0" in rendered
    assert "width=.9+Math.min(4.1,.6*Math.sqrt(total))" in rendered
    assert "2+Math.min(7,2.2*Math.sqrt(n))" in rendered
    assert 'markerUnits="userSpaceOnUse"' in rendered
    assert 'querySelectorAll(".network-edge")' in rendered


def test_foreign_host_publication_lock_is_not_reclaimed(monkeypatch, tmp_path):
    lock = tmp_path / ".match.render.lock"
    lock.write_text("host=other-render-node\npid=999999999\n", encoding="ascii")
    monkeypatch.setattr(integrity.socket, "gethostname", lambda: "this-render-node")
    monkeypatch.setattr(
        integrity.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(AssertionError("foreign PID probed")),
    )

    with np.testing.assert_raises(FileExistsError):
        integrity._reserve_output_lock(lock)
    assert lock.exists()


def test_publication_does_not_remove_replacement_lock(tmp_path):
    target = tmp_path / "match"
    lock = tmp_path / ".match.render.lock"

    with integrity.staged_output_directory(target):
        lock.unlink()
        lock.write_text("host=another-render-node\npid=123\n", encoding="ascii")

    assert target.is_dir()
    assert lock.read_text(encoding="ascii") == ("host=another-render-node\npid=123\n")


def test_tracking_reader_closes_archive_when_index_is_malformed(monkeypatch):
    class FakeArchive:
        def __init__(self):
            self.files = ["__index__"]
            self.closed = False

        def __getitem__(self, key):
            assert key == "__index__"
            return np.frombuffer(b"{", dtype=np.uint8)

        def close(self):
            self.closed = True

    archive = FakeArchive()
    monkeypatch.setattr(tracking.np, "load", lambda *_args, **_kwargs: archive)

    with np.testing.assert_raises(ValueError):
        tracking.NpzTrackingReader("malformed.npz")

    assert archive.closed


def test_publication_sample_count_uses_control_tick_span():
    metadata = {"time_axis": {"tracking_span_s": 1.0}}

    count = _expected_video_sample_count(
        metadata,
        source_frames=2,
        control_fps=10.0,
        sample_fps=20.0,
    )

    assert count == 22


def test_renderer_rejects_duplicate_control_ticks():
    frames = [
        SimpleNamespace(control_tick=10),
        SimpleNamespace(control_tick=10),
    ]

    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        _resample_indices(frames, 10.0, 20.0)


def test_publication_sample_count_never_divides_by_zero():
    metadata = {"time_axis": {"tracking_span_s": 0.0}}

    count = _expected_video_sample_count(
        metadata, source_frames=1, control_fps=10.0, sample_fps=0.01
    )

    assert count == 1


def test_scalar_managed_runner_rejects_invalid_progress(monkeypatch):
    monkeypatch.setattr(managed, "SquadSetup", _FakeSquad)
    monkeypatch.setattr(managed, "_one_key", lambda _key: None)
    state = managed.ManagedMatchState(
        rollout=object(),
        setup=object(),
        management=object(),
        roster=object(),
        player_policy_state=object(),
        manager=SimpleNamespace(boundary=object(), policy=object()),
        opening_formation_checked=True,
    )

    for progressed, message in (
        (0, "no progress without a boundary"),
        (-1, "invalid step count"),
        (5, "invalid step count"),
    ):

        def advance(
            rollout, _setup, _roster, policy_state, *_args, _progressed=progressed
        ):
            return SimpleNamespace(
                final_rollout=rollout,
                final_policy_state=policy_state,
                steps_executed=jnp.int32(_progressed),
                terminated=jnp.bool_(False),
                truncated=jnp.bool_(False),
                manager_required=jnp.bool_(False),
            )

        runner = managed.ManagedRunner(
            env=None,
            player_policy=None,
            chunk_steps=4,
            manager_policy=None,
            opening_policy=None,
            _advance=advance,
            _manager_initialize=None,
            _manager_decide=None,
            _refresh_roster=None,
            _apply_tactics=None,
            _opening_decide=None,
            _can_handle_goalkeeper=False,
        )

        with np.testing.assert_raises_regex(RuntimeError, message):
            runner.run(state, _FakeSquad(), object(), 4)


def test_legacy_publication_sample_count_matches_renderer_rounding():
    single = _expected_video_sample_count(
        {}, source_frames=1, control_fps=10.0, sample_fps=20.0
    )
    half_tie = _expected_video_sample_count(
        {}, source_frames=5, control_fps=10.0, sample_fps=5.0
    )
    low_rate = _expected_video_sample_count(
        {}, source_frames=2, control_fps=10.0, sample_fps=0.01
    )

    assert single == 1
    assert half_tie == 3
    assert low_rate == 1


def test_authoritative_publication_requires_every_video_decode_check():
    completion = {
        "done": True,
        "complete": True,
        "full_duration_complete": True,
        "terminal_basis": "regulation_complete",
        "maximum_steps": None,
        "event_budget_exhausted_count": 0,
        "publication_guard": {
            "authoritative": True,
            "status": "valid",
            "source_authority": "clean",
            "stable_during_capture": True,
        },
    }
    required = {
        "frame_count": True,
        "fps": True,
        "width_px": True,
        "height_px": True,
        "codec": True,
        "pixel_format": True,
        "duration_s": True,
    }
    output = {"video_decode_verification": {"enabled": True, "checks": required}}

    assert authoritative_completion_error(completion, output) is None
    metadata = {"completion": dict(completion)}
    assert (
        authoritative_completion_error(
            completion,
            output,
            metadata=metadata,
        )
        is None
    )
    metadata["completion"]["done"] = False
    assert (
        authoritative_completion_error(completion, output, metadata=metadata)
        is not None
    )
    output["video_decode_verification"]["checks"] = {"fps": True}
    assert authoritative_completion_error(completion, output) is not None


def test_current_publication_verifies_tracking_receipt_against_archive(
    monkeypatch, tmp_path
):
    children = {
        "video": tmp_path / "match.mp4",
        "event": tmp_path / "event.json",
        "tracking": tmp_path / "tracking.npz",
        "metadata": tmp_path / "metadata.json",
    }
    table = np.zeros(
        1,
        dtype=np.dtype([("frame", "<i8"), ("control_tick", "<i8")]),
    )
    table["frame"] = [0]
    table["control_tick"] = [1]
    tracking_index = tracking._write_archive(children["tracking"], (table,), {})
    tracking_receipt = tracking.tracking_storage_receipt(
        children["tracking"], tracking_index
    )
    children["video"].write_bytes(b"video")
    children["event"].write_text("{}\n", encoding="utf-8")
    metadata = {
        "schema": METADATA_SCHEMA,
        "source_frame_count": 1,
        "tracking_frame_count": 1,
        "video_sample_frame_count": 2,
        "video_frame_count": 2,
        "control_fps": 10.0,
        "video_sample_fps": 20.0,
        "video_fps": 20.0,
        "sample_every": 1,
        "tracking_storage": tracking_receipt,
        "time_axis": {
            "tracking_span_s": 0.0,
            "video_origin_time_s": 0.05,
            "video_time_step_s": 0.05,
        },
    }
    children["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output = {
        **{name: path.name for name, path in children.items()},
        "source_frame_count": 1,
        "tracking_frame_count": 1,
        "event_frame_count": 1,
        "video_frame_count": 2,
        "tracking_storage": tracking_receipt,
        "artifacts": {
            name: integrity.artifact_receipt(path) for name, path in children.items()
        },
    }
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(
        json.dumps(
            {
                "schema": integrity.RENDER_COMPLETION_SCHEMA,
                "outputs": [output],
            }
        ),
        encoding="utf-8",
    )
    artifact_calls = []
    artifact_receipt = publication._artifact_receipt

    def tracked_artifact_receipt(path):
        artifact_calls.append(path)
        return artifact_receipt(path)

    monkeypatch.setattr(publication, "_artifact_receipt", tracked_artifact_receipt)
    open_published_replay(tmp_path, require_authoritative=False)
    assert children["tracking"] not in artifact_calls

    metadata["tracking_storage"]["frame_rows"] = 99
    children["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output["tracking_storage"] = metadata["tracking_storage"]
    output["artifacts"]["metadata"] = integrity.artifact_receipt(children["metadata"])
    completion_path.write_text(
        json.dumps({"schema": integrity.RENDER_COMPLETION_SCHEMA, "outputs": [output]}),
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "archive disagrees"):
        open_published_replay(tmp_path, require_authoritative=False)

    duplicate_metadata = (
        children["metadata"]
        .read_text(encoding="utf-8")
        .replace(
            '{"schema":',
            '{"schema":"duplicate","schema":',
            1,
        )
    )
    children["metadata"].write_text(duplicate_metadata, encoding="utf-8")
    output["artifacts"]["metadata"] = integrity.artifact_receipt(children["metadata"])
    completion_path.write_text(
        json.dumps({"schema": integrity.RENDER_COMPLETION_SCHEMA, "outputs": [output]}),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "duplicate JSON key"):
        open_published_replay(tmp_path, require_authoritative=False)

    metadata["nonfinite_probe"] = float("nan")
    children["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output["artifacts"]["metadata"] = integrity.artifact_receipt(children["metadata"])
    completion_path.write_text(
        json.dumps({"schema": integrity.RENDER_COMPLETION_SCHEMA, "outputs": [output]}),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "non-finite JSON constant"):
        open_published_replay(tmp_path, require_authoritative=False)


def test_publication_rejects_duplicate_manifest_keys(tmp_path):
    manifest = tmp_path / "completion.json"
    manifest.write_text(
        (
            '{"schema":"footballworld.render-completion/2",'
            '"schema":"footballworld.render-completion/2","outputs":[]}'
        ),
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "duplicate JSON key"):
        open_published_replay(manifest, require_authoritative=False)

    manifest.write_text(
        (
            '{"schema":"footballworld.render-completion/2",'
            '"nonfinite_probe":NaN,"outputs":[]}'
        ),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "non-finite JSON constant"):
        open_published_replay(manifest, require_authoritative=False)

    writer_dir = tmp_path / "writer"
    writer_dir.mkdir()
    with np.testing.assert_raises_regex(ValueError, "JSON compliant"):
        integrity.write_completion_manifest(
            writer_dir,
            completion={"nonfinite_probe": float("nan")},
            outputs=[],
        )
    with np.testing.assert_raises_regex(ValueError, "JSON compliant"):
        replay._write_json(
            writer_dir / "metadata.json",
            {"nonfinite_probe": float("inf")},
        )


def test_non_pass_intents_retain_tracking_ticks_for_report_maps():
    intents = (
        analysis_metrics.INTENT_SHOT,
        analysis_metrics.INTENT_CLEAR,
        analysis_metrics.INTENT_CHALLENGE,
    )
    frames = [
        {
            "control_tick": tick,
            "frame_events": {
                "events": [
                    {
                        "type": "deliberate_contact",
                        "substep": 0,
                        "fields": {
                            "kick_applied": True,
                            "restart_kind": analysis_metrics.RESTART_NONE,
                            "intent": intent,
                            "actor": 0,
                            "mechanism": "foot",
                            "law11_effect": "deliberate",
                        },
                    }
                ]
            },
        }
        for tick, intent in enumerate(intents, start=10)
    ]
    frames.append(
        {
            "control_tick": 13,
            "frame_events": {
                "events": [
                    {
                        "type": "deliberate_contact",
                        "substep": 0,
                        "fields": {
                            "kick_applied": False,
                            "restart_kind": analysis_metrics.RESTART_NONE,
                            "intent": analysis_metrics.INTENT_CONTROL,
                            "actor": 0,
                            "mechanism": "foot",
                            "law11_effect": "deliberate",
                        },
                    },
                    {
                        "type": "contact",
                        "substep": 0,
                        "slot": 0,
                        "fields": {
                            "occurred": True,
                            "actor": 0,
                            "mechanism": "foot",
                            "law11_effect": "deliberate",
                            "position": [4.0, -2.0, 0.11],
                            "time_fraction": 0.5,
                        },
                    },
                ]
            },
        }
    )

    facts = analysis_metrics._collect_policy_contact_facts(
        {},
        {0: {"team": 0, "player_id": 1001, "slot_generation": 0}},
        frames,
    )

    assert facts["attack_direction_ticks"] == {10, 11, 12, 13}
    assert len(facts["control_sources"]) == 1
    assert facts["control_sources"][0]["kick_applied"] is False
    assert facts["control_sources"][0]["contact"]["position"] == (4.0, -2.0, 0.11)


def test_event_stream_rejects_ambiguous_or_nonfinite_json(tmp_path):
    event_path = tmp_path / "event.json"
    event_path.write_text(
        ('{"schema":"footballworld.events/15","schema":"duplicate","frames":[]}'),
        encoding="utf-8",
    )
    with np.testing.assert_raises_regex(ValueError, "duplicate JSON key"):
        events.open_events(event_path)

    event_path.write_text(
        (
            '{"schema":"footballworld.events/15","frames":['
            '{"frame":0,"frame":0,"control_tick":1}]}'
        ),
        encoding="utf-8",
    )
    with (
        events.open_events(event_path) as stream,
        np.testing.assert_raises_regex(ValueError, "duplicate JSON key"),
    ):
        list(stream)

    event_path.write_text(
        (
            '{"schema":"footballworld.events/15","frames":['
            '{"frame":0,"control_tick":1,"clock_s":NaN}]}'
        ),
        encoding="utf-8",
    )
    with (
        events.open_events(event_path) as stream,
        np.testing.assert_raises_regex(ValueError, "non-finite JSON constant"),
    ):
        list(stream)


def test_team_comparison_uses_clear_full_match_event_metrics():
    teams = [
        {
            "controlled_possession_share": 0.49,
            "goals": 1,
            "submitted_shots": 8,
            "open_play_completed_passes": 310,
            "open_play_pass_completion": 0.84,
            "penalty_area_entries": 12,
            "corners": 4,
            "fouls_committed": 7,
            "offsides": 0,
        },
        {
            "controlled_possession_share": 0.51,
            "goals": 2,
            "submitted_shots": 10,
            "open_play_completed_passes": 325,
            "open_play_pass_completion": 0.86,
            "penalty_area_entries": 15,
            "corners": 5,
            "fouls_committed": 9,
            "offsides": 1,
        },
    ]

    rendered = _comparison_rows({"teams": teams})

    assert "Full-match controlled possession" in rendered
    assert "Completed open-play passes" in rendered
    assert "Corners" in rendered
    assert "Fouls committed" in rendered
    assert "Offsides" in rendered
    assert "Average width" not in rendered
    assert "Average length" not in rendered
    assert "Compactness" not in rendered
    assert "Final-third entries" not in rendered


def test_cumulative_possession_chart_and_bidirectional_workload_badges():
    report = {
        "summary": {"captured_duration_s": 60.0},
        "visualizations": {
            "window_s": 30.0,
            "windows": [
                {"controlled_possession_share": [0.6, 0.4]},
                {"controlled_possession_share": [0.55, 0.45]},
            ],
        },
    }
    chart = _series_chart(
        report,
        title="Cumulative controlled possession",
        metric="controlled_possession_share",
        unit="%",
        percent=True,
    )
    workload = _workload_charts(
        {
            "players": [
                {
                    "team": 0,
                    "player_id": 10,
                    "distance_m": 5000.0,
                    "substitution": {
                        "exited": {"clock_label": "67:34"},
                    },
                },
                {
                    "team": 0,
                    "player_id": 17,
                    "distance_m": 2500.0,
                    "substitution": {
                        "entered": {"clock_label": "67:34"},
                    },
                },
            ]
        }
    )

    assert "2 cumulative checkpoints" in chart
    assert "Team 0, 55.0%" in chart
    assert "sub-badge exited" in workload
    assert "&darr; 67:34" in workload
    assert "sub-badge entered" in workload
    assert "&uarr; 67:34" in workload
