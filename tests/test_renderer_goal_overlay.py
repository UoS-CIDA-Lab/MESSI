"""Resolution and lifetime contracts for the goal celebration overlay."""

from __future__ import annotations

import numpy as np

from footballworld.rendering.defaults import RenderStyle
from footballworld.rendering.renderer import (
    _Adjudication,
    _carry_adjudication,
    _goal_overlay_font_sizes,
    _insert_goal_presentation_holds,
    _player_marker_geometry,
    _player_walk_pose_indices,
    _VisualFrame,
)


def _frame(control_tick: int, adjudication: _Adjudication | None) -> _VisualFrame:
    return _VisualFrame(
        control_tick=control_tick,
        video_time_s=None,
        first_half_wall_end_tick=27_000,
        ball_position=np.zeros(3, dtype=np.float32),
        ball_live=True,
        player_position=np.zeros((2, 2), dtype=np.float32),
        player_velocity=np.zeros((2, 2), dtype=np.float32),
        player_body_forward=np.zeros((2, 2), dtype=np.float32),
        player_gaze_yaw=np.zeros(2, dtype=np.float32),
        aerial_progress=np.zeros(2, dtype=np.float32),
        high_head_contact=np.zeros(2, dtype=bool),
        team_id=np.asarray((0, 1), dtype=np.int32),
        is_goalkeeper=np.asarray((True, True), dtype=bool),
        active=np.asarray((True, True), dtype=bool),
        sent_off=np.zeros(2, dtype=bool),
        stamina_long=np.ones(2, dtype=np.float32),
        stamina_short=np.ones(2, dtype=np.float32),
        yellow_cards=np.zeros(2, dtype=np.int32),
        possession_player=-1,
        score=np.asarray((1, 0), dtype=np.int32),
        requested_intent=np.zeros(2, dtype=np.int32),
        action_executed=True,
        adjudication=adjudication,
    )


def test_goal_overlay_typography_preserves_relative_size_at_1080p() -> None:
    assert _goal_overlay_font_sizes(RenderStyle(width_px=960, height_px=540)) == (
        20.0,
        8.5,
    )
    assert _goal_overlay_font_sizes(RenderStyle(width_px=1920, height_px=1080)) == (
        40.0,
        17.0,
    )


def test_player_marker_is_one_compound_head_and_body_path() -> None:
    vertices, codes = _player_marker_geometry()
    assert vertices.shape == (34, 2)
    assert codes.shape == (34,)
    assert np.count_nonzero(codes == 1) == 4
    assert np.count_nonzero(codes == 79) == 4
    assert np.max(np.abs(vertices[:, 0])) <= 0.5
    assert np.min(vertices[:, 1]) <= -1.0
    head = vertices[-13:]
    assert np.allclose(np.linalg.norm(head, axis=1), 0.34)
    assert np.max(head[:, 1]) > 0.3


def test_player_legs_animate_only_while_moving() -> None:
    velocity = np.asarray(((0.0, 0.0), (4.0, 0.0)), dtype=np.float32)
    active = np.asarray((True, True))
    still = _player_walk_pose_indices(velocity, active, video_seconds=0.0)
    later = _player_walk_pose_indices(velocity, active, video_seconds=0.1)
    assert still[0] == 3
    assert later[0] == 3
    assert still[1] != later[1]

    hidden = _player_walk_pose_indices(
        velocity, np.asarray((True, False)), video_seconds=0.1
    )
    assert hidden[1] == 3


def test_goal_presentation_survives_adjudication_hold() -> None:
    goal = _Adjudication(
        origin_control_tick=100,
        title="GOAL — HOME",
        detail="SCORE 1 : 0",
        accent="#ef476f",
        goal=True,
    )
    held, previous = _carry_adjudication(
        _frame(124, None),
        goal,
        control_fps=10.0,
        duration_seconds=2.5,
    )
    assert held.adjudication is goal
    assert previous is goal
    assert held.adjudication.goal

    expired, previous = _carry_adjudication(
        _frame(125, None),
        goal,
        control_fps=10.0,
        duration_seconds=2.5,
    )
    assert expired.adjudication is None
    assert previous is None


def test_goal_presentation_freezes_video_for_exactly_two_seconds() -> None:
    goal = _Adjudication(
        origin_control_tick=100,
        title="GOAL — HOME",
        detail="SCORE 1 : 0",
        accent="#ef476f",
        goal=True,
    )
    origin = _frame(100, goal)
    duplicate = _frame(100, goal)
    carried = _frame(101, goal)
    resumed = _frame(102, None)
    presented = _insert_goal_presentation_holds(
        [origin, duplicate, carried, resumed],
        video_fps=20.0,
        duration_seconds=2.0,
    )
    assert len(presented) == 42
    assert all(frame is origin for frame in presented[:40])
    assert presented[40].control_tick == 101
    assert presented[40].adjudication is None
    assert presented[41] is resumed
