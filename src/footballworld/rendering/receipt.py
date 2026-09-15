"""Host-only provenance for the one FootballWorld render path."""

from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from footballworld.rendering.defaults import RenderStyle

RENDER_SETTINGS_SCHEMA = "footballworld.render-settings/4"


def _dependency_versions() -> dict[str, str | None]:
    """Read installed distribution metadata without invoking an encoder."""

    result: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "matplotlib",
        "imageio",
        "imageio-ffmpeg",
        "Pillow",
    ):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def render_settings_receipt(
    *,
    style: RenderStyle,
    video_fps: float,
    workers_requested: int,
    workers_effective: int,
    encoder_threads_per_segment: int,
    render_chunk_frame_cap: int,
    segment_count: int,
    process_start_method: str | None,
    intent_ring_ordinary_radius_m: float,
    intent_ring_challenge_radius_m: float,
    intent_ring_goalkeeper_control_radius_m: float,
    event_chunk_steps: int | None,
    async_rgba_buffers_per_worker: int = 2,
) -> dict[str, Any]:
    """Return actual host render settings, never caller-authored metadata."""

    configured_threads = int(encoder_threads_per_segment)
    actual_threads = max(1, min(configured_threads, os.cpu_count() or 1))
    return {
        "schema": RENDER_SETTINGS_SCHEMA,
        "camera": "fixed_oblique_perspective_tight_pitch",
        "presentation": (
            "broadcast_3d_single_player_marker_with_intent_ring_"
            "minimap_and_transient_adjudication"
        ),
        "camera_azimuth_degrees": float(style.camera_azimuth_degrees),
        "camera_elevation_degrees": float(style.camera_elevation_degrees),
        "camera_distance_m": float(style.camera_distance_m),
        "camera_focal_length": float(style.camera_focal_length),
        "player_marker_area_points2": float(style.player_size),
        "player_marker": "single_team_coloured_circle",
        "video_fps": float(video_fps),
        "width_px": int(style.width_px),
        "height_px": int(style.height_px),
        "dpi": int(style.dpi),
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "crf": int(style.crf),
        "encoder_preset": style.encoder_preset,
        "encoder_threads_requested": int(style.encoder_threads),
        "encoder_threads_configured_per_segment": configured_threads,
        "encoder_threads_per_segment": actual_threads,
        "workers_requested": int(workers_requested),
        "workers_effective": int(workers_effective),
        "process_start_method": process_start_method,
        "render_worker_compute_backend": (
            "cpu_host_only" if workers_effective > 1 else "in_process"
        ),
        "async_rgba_buffers_per_worker": int(async_rgba_buffers_per_worker),
        "render_chunk_frame_cap": int(render_chunk_frame_cap),
        "segment_count": int(segment_count),
        "event_chunk_steps": (
            None if event_chunk_steps is None else int(event_chunk_steps)
        ),
        "intent_ring_overlay": {
            "visible_for_executed_non_move_intent": True,
            "ordinary_radius_m": float(intent_ring_ordinary_radius_m),
            "challenge_radius_m": float(intent_ring_challenge_radius_m),
            "goalkeeper_control_radius_m": float(
                intent_ring_goalkeeper_control_radius_m
            ),
            "radius_basis": "configured_horizontal_reach_plus_ball_radius",
            "geometry": "turf_projected_maximum_interference_envelope",
        },
        "adjudication_banner": (
            "single_transient_exact_goal_foul_or_offside_no_history"
        ),
        "adjudication_seconds": float(style.adjudication_seconds),
        "goal_hold_seconds": float(style.goal_hold_seconds),
        "goal_hold_semantics": "frozen_video_frame_environment_clock_unchanged",
        "adjudication_chunk_continuity": (
            "host_control_tick_carry_before_segment_submission"
        ),
        "python": platform.python_version(),
        "dependencies": _dependency_versions(),
        "ffmpeg_version_probe": "not_run",
    }


__all__ = ["RENDER_SETTINGS_SCHEMA", "render_settings_receipt"]
