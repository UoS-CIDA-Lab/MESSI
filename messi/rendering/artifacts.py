"""Export one recorded match as linked report and optional video artifacts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from messi._json import strict_json_dumps
from messi.analysis.recording import InMemoryMatchStore, record_match
from messi.analysis.report import write_match_report
from messi.environment.episode import DEFAULT_MATCH_CONFIG
from messi.rendering.report import render_match_report_html
from messi.rendering.video import RenderStyle, render_match_video

MATCH_ARTIFACT_SCHEMA = "messi.match-artifacts/1"


@dataclass(frozen=True, slots=True)
class MatchArtifactPaths:
    directory: Path
    manifest: Path
    report_json: Path
    report_markdown: Path
    report_html: Path
    dataset: None
    video: Path | None


def match_title_slug(title: str) -> str:
    """Turn a display title into a safe directory name."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    normalised = unicodedata.normalize("NFKC", title.strip())
    slug = re.sub(r"[^\w.-]+", "-", normalised, flags=re.UNICODE).strip("._-")
    slug = slug[:120].rstrip("._-")
    if not slug or slug in {".", ".."}:
        raise ValueError("title does not yield a safe directory name")
    return slug


def _video_settings(report, fps: int, control_fps, field_half_extent):
    spatial = report.spatial
    x_edges = np.asarray(spatial.x_edges_m, dtype=np.float64)
    y_edges = np.asarray(spatial.y_edges_m, dtype=np.float64)
    if (
        x_edges.ndim != 1
        or y_edges.ndim != 1
        or x_edges.size < 2
        or y_edges.size < 2
        or not np.all(np.isfinite(x_edges))
        or not np.all(np.isfinite(y_edges))
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
    ):
        raise ValueError("report has invalid pitch edges")
    extent = ((x_edges[-1] - x_edges[0]) / 2, (y_edges[-1] - y_edges[0]) / 2)
    centre = ((x_edges[-1] + x_edges[0]) / 2, (y_edges[-1] + y_edges[0]) / 2)
    if any(value <= 0 for value in extent) or not np.allclose(centre, 0, rtol=0, atol=1e-9):
        raise ValueError("report pitch must be centred on the origin")
    if not np.isfinite(report.control_dt_s) or report.control_dt_s <= 0:
        raise ValueError("report has an invalid control interval")
    expected_fps = 1.0 / report.control_dt_s
    if control_fps is not None and not np.isclose(
        control_fps, expected_fps, rtol=1e-9, atol=0
    ):
        raise ValueError("control_fps disagrees with the report")
    if field_half_extent is not None and not np.allclose(
        field_half_extent, extent, rtol=0, atol=1e-9
    ):
        raise ValueError("field_half_extent disagrees with the report")
    from messi.rendering.video import _validated_render_frame_rates

    _validated_render_frame_rates(fps, expected_fps)
    return float(expected_fps), tuple(float(value) for value in extent)


def _stored_trajectory(store: InMemoryMatchStore):
    """Join contiguous host chunks only when a video was explicitly requested."""

    chunks = store.chunks
    if not chunks:
        raise ValueError("no recorded frames are available for video rendering")
    expected_offset = 0
    for item in chunks:
        if item.match_index != 0 or item.frame_offset != expected_offset:
            raise ValueError("recorded video chunks must be contiguous for one match")
        expected_offset += int(np.asarray(item.chunk.steps_executed))
    if expected_offset == 0:
        raise ValueError("no recorded frames are available for video rendering")
    return SimpleNamespace(
        states=jax.tree.map(
            lambda *leaves: np.concatenate(leaves, axis=0),
            *(item.chunk.states for item in chunks),
        ),
        actions=jax.tree.map(
            lambda *leaves: np.concatenate(leaves, axis=0),
            *(item.chunk.actions for item in chunks),
        ),
        receipts=jax.tree.map(
            lambda *leaves: np.concatenate(leaves, axis=0),
            *(item.chunk.receipts for item in chunks),
        ),
    )


def export_match_artifacts(
    recording,
    root_directory: str | Path,
    title: str,
    *,
    roster=None,
    trajectory=None,
    fps: int = 20,
    control_fps: float | None = None,
    halftime_seconds: float = DEFAULT_MATCH_CONFIG.halftime_seconds,
    style: RenderStyle | None = None,
    field_half_extent=None,
    team_names: tuple[str, str] | None = None,
) -> MatchArtifactPaths:
    """Write JSON, Markdown, HTML, and optionally MP4 for one recorded match."""

    slug = match_title_slug(title)
    if trajectory is not None and roster is None:
        raise ValueError("roster is required when rendering a video")
    video_settings = None
    if trajectory is not None:
        video_settings = _video_settings(
            recording.report, fps, control_fps, field_half_extent
        )
    output = Path(root_directory) / slug
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifact"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_json, report_markdown = write_match_report(
        recording.report, artifacts, title=title.strip()
    )
    report_html = render_match_report_html(
        recording.report, output / "match-report.html", title=title.strip()
    )
    video = None
    if trajectory is not None:
        video = render_match_video(
            trajectory,
            roster,
            artifacts / "match.mp4",
            fps=fps,
            control_fps=video_settings[0],
            halftime_seconds=halftime_seconds,
            style=style,
            field_half_extent=video_settings[1],
            match_report=recording.report,
            team_names=team_names,
        )
    manifest = artifacts / "artifacts.json"
    manifest.write_text(
        strict_json_dumps(
            {
                "schema": MATCH_ARTIFACT_SCHEMA,
                "title": title.strip(),
                "files": {
                    "report_json": report_json.name,
                    "report_markdown": report_markdown.name,
                    "report_html": "../" + report_html.name,
                    "dataset": None,
                    "video": None if video is None else video.name,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return MatchArtifactPaths(
        output, manifest, report_json, report_markdown, report_html, None, video
    )


def record_and_export_match(
    chunk_kernel,
    initial_state,
    roster,
    match_key,
    setup,
    maximum_steps: int,
    root_directory: str | Path,
    title: str,
    *,
    include_video: bool = False,
    policy_context=None,
    fps: int = 20,
    style: RenderStyle | None = None,
    team_names: tuple[str, str] | None = None,
):
    """Run a bounded match capture and export its reports in one call.

    Video capture retains the full host trajectory. Leave it disabled for
    report-only long matches to keep host memory bounded by the chunk size.
    """

    match_title_slug(title)
    if not isinstance(include_video, bool):
        raise TypeError("include_video must be bool")
    store = InMemoryMatchStore() if include_video else None
    recording = record_match(
        chunk_kernel,
        initial_state,
        roster,
        match_key,
        setup,
        maximum_steps,
        chunk_sink=store,
        policy_context=policy_context,
    )
    trajectory = None if store is None else _stored_trajectory(store)
    paths = export_match_artifacts(
        recording,
        root_directory,
        title,
        roster=roster if include_video else None,
        trajectory=trajectory,
        fps=fps,
        halftime_seconds=getattr(
            chunk_kernel, "match_config", DEFAULT_MATCH_CONFIG
        ).halftime_seconds,
        style=style,
        team_names=team_names,
    )
    return recording, paths


__all__ = [
    "MATCH_ARTIFACT_SCHEMA",
    "MatchArtifactPaths",
    "export_match_artifacts",
    "match_title_slug",
    "record_and_export_match",
]
