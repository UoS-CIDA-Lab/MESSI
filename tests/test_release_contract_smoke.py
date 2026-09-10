"""Small public gates for release-facing timing contracts.

Deep semantic, adversarial, and calibration validation remains in the local
``.validation_tests`` suite and is intentionally excluded from distributions.
"""

from types import SimpleNamespace

import pytest

from footballworld.core.timebase import DEFAULT_TIMEBASE, _exact_render_grid
from footballworld.environment.episode import MatchConfig
from footballworld.rendering.capture import _event_render_grid
from footballworld.rendering.integrity import publication_authority
from footballworld.rendering.renderer import (
    SCOREBOARD_CLOCK_Y,
    SCOREBOARD_SCORE_Y,
    ReplayRenderer,
)


def test_bounded_capture_is_never_authoritative() -> None:
    assert publication_authority(dirty=False, maximum_steps=None) == {
        "status": "valid",
        "authoritative": True,
        "source_authority": "clean",
    }
    assert publication_authority(dirty=False, maximum_steps=3000) == {
        "status": "diagnostic-step-budget",
        "authoritative": False,
        "source_authority": "clean",
    }
    assert publication_authority(dirty=True, maximum_steps=None) == {
        "status": "diagnostic-dirty-source",
        "authoritative": False,
        "source_authority": "diagnostic-dirty",
    }


def test_default_timebase_contract() -> None:
    assert DEFAULT_TIMEBASE.physics_fps == pytest.approx(80.0)
    assert DEFAULT_TIMEBASE.control_fps == pytest.approx(10.0)
    assert DEFAULT_TIMEBASE.decimation == 8


def test_exact_event_render_grid_divides_physics_decimation() -> None:
    env = SimpleNamespace(timebase=DEFAULT_TIMEBASE)
    for fps, samples in ((10.0, 1), (20.0, 2), (40.0, 4), (80.0, 8)):
        assert _event_render_grid(env, fps) == (fps, samples)

    for fps in (30.0, 60.0):
        with pytest.raises(ValueError, match="divide the physics decimation"):
            _event_render_grid(env, fps)


def test_exact_render_indices_are_uniform_physics_endpoints() -> None:
    expected = {
        10.0: (7,),
        20.0: (3, 7),
        40.0: (1, 3, 5, 7),
        80.0: tuple(range(8)),
    }
    for fps, indices in expected.items():
        _, _, actual = _exact_render_grid(
            DEFAULT_TIMEBASE,
            fps,
            context="event render",
        )
        assert actual == indices


def test_halftime_validation_matches_renderer_contract() -> None:
    match = MatchConfig(
        halftime_enabled=False,
        halftime_seconds=6000.0,
        fulltime_seconds=5400.0,
    )
    assert match.halftime_seconds == match.fulltime_seconds == 5400.0
    assert match.clock_ticks(DEFAULT_TIMEBASE) == (54000, 54000)
    renderer = ReplayRenderer(
        halftime_enabled=False,
        halftime_seconds=6000.0,
        fulltime_seconds=5400.0,
    )
    assert renderer.halftime_seconds == renderer.fulltime_seconds == 5400.0


def test_540p_scoreboard_reserves_distinct_score_and_clock_rows() -> None:
    assert (SCOREBOARD_SCORE_Y - SCOREBOARD_CLOCK_Y) * 540 >= 24.0

    with pytest.raises(ValueError, match="greater than halftime_seconds"):
        MatchConfig(
            halftime_enabled=True,
            halftime_seconds=6000.0,
            fulltime_seconds=5400.0,
        )
    with pytest.raises(ValueError, match="less than fulltime_seconds"):
        ReplayRenderer(
            halftime_enabled=True,
            halftime_seconds=6000.0,
            fulltime_seconds=5400.0,
        )
