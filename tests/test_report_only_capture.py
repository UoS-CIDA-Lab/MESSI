import json
from pathlib import Path

from footballworld.rendering.capture import (
    _managed_capture_chunk_kernel,
    _replace_last_render_sample,
    _WindowSink,
)
from footballworld.rendering.window import ReplayWindow


class _FakeSpool:
    frame_count = 3

    def __init__(self, root: Path) -> None:
        self.root = root
        self.receipt = None

    def finalize(self, **receipt):
        self.receipt = receipt
        paths = (
            self.root / "event.json",
            self.root / "tracking.npz",
            self.root / "metadata.json",
        )
        for path in paths:
            path.write_bytes(b"x")
        return paths


def test_report_only_managed_kernel_omits_visual_samples(monkeypatch):
    selected = []
    marker = object()

    def compiled(runner, event_chunk_steps, render_fps):
        selected.append((runner, event_chunk_steps, render_fps))
        return marker

    monkeypatch.setattr(
        "footballworld.rendering.capture._compiled_managed_event_chunk",
        compiled,
    )
    runner = object()

    assert (
        _managed_capture_chunk_kernel(
            runner,
            256,
            20.0,
            render_video=False,
        )
        is marker
    )
    assert selected.pop() == (runner, 256, None)

    assert (
        _managed_capture_chunk_kernel(
            runner,
            256,
            20.0,
            render_video=True,
        )
        is marker
    )
    assert selected.pop() == (runner, 256, 20.0)


def test_manager_boundary_endpoint_replacement_skips_report_only_groups():
    replacement = object()
    report_only_groups = [[], []]

    _replace_last_render_sample(report_only_groups, replacement)

    assert report_only_groups == [[], []]

    previous = object()
    video_groups = [[previous], [previous]]
    _replace_last_render_sample(video_groups, replacement)

    assert video_groups[0][-1] is previous
    assert video_groups[-1][-1] is replacement


def test_report_only_sink_skips_mp4_and_preserves_source_grid(tmp_path):
    sink = object.__new__(_WindowSink)
    sink.window = ReplayWindow("match")
    sink.render_video = False
    sink.video = tmp_path / "report-only.json"
    sink.control_fps = 10.0
    sink.spool = _FakeSpool(tmp_path)

    result = sink.finish(
        elapsed=2.0,
        workers=4,
        completion={"done": False},
    )

    assert list(tmp_path.glob("*.mp4")) == []
    assert json.loads(result.video.read_text()) == {
        "schema": "footballworld.report-only-marker/1",
        "source_frame_count": 3,
        "video_generated": False,
    }
    assert result.video_generated is False
    assert result.frames == 3
    assert result.workers == 0
    assert sink.spool.receipt["video_sample_frame_count"] == 3
    assert sink.spool.receipt["video_sample_fps"] == 10.0
    assert sink.spool.receipt["video_frame_count"] == 3
    assert sink.spool.receipt["video_fps"] == 10.0
    assert sink.spool.receipt["sample_every"] == 1
    assert sink.spool.receipt["video_verification"] == "not_requested"
