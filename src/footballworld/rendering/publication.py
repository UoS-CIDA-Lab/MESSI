"""Fail-closed access to atomically published replay artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from footballworld.rendering.integrity import RENDER_COMPLETION_SCHEMA
from footballworld.rendering.replay import METADATA_SCHEMA


@dataclass(frozen=True, slots=True)
class PublishedReplay:
    """Resolved, verified paths for one completion-manifest output."""

    root: Path
    video: Path
    event: Path
    tracking: Path
    metadata: Path
    completion: dict[str, Any]
    output: dict[str, Any]


def _artifact_receipt(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _resolve_child(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("published replay child path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("published replay child path must remain inside its root")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("published replay child path escapes its root")
    if not resolved.is_file():
        raise FileNotFoundError(f"published replay child is missing: {resolved}")
    return resolved


def open_published_replay(
    path: str | Path,
    *,
    output_name: str | None = None,
    require_authoritative: bool = True,
    verify_hashes: bool = True,
) -> PublishedReplay:
    """Open one publication only after schema, status, paths, counts, and hashes.

    This cold host API trusts a recorded full-decode receipt rather than
    decoding the video on every open. Official publication creates that
    receipt before the staging directory is atomically renamed.
    """

    if type(require_authoritative) is not bool or type(verify_hashes) is not bool:
        raise TypeError("publication verification switches must be bool")
    source = Path(path)
    manifest_path = source / "completion.json" if source.is_dir() else source
    with manifest_path.open(encoding="utf-8") as stream:
        completion = json.load(stream)
    if not isinstance(completion, dict):
        raise TypeError("completion manifest root must be an object")
    if completion.get("schema") != RENDER_COMPLETION_SCHEMA:
        raise ValueError(f"published replay requires {RENDER_COMPLETION_SCHEMA!r}")
    root = manifest_path.parent.resolve()
    outputs = completion.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("completion manifest has no outputs")
    matches = []
    for candidate in outputs:
        if not isinstance(candidate, dict):
            raise TypeError("completion output records must be objects")
        name = Path(str(candidate.get("video", ""))).parent.as_posix()
        if output_name is None or name == output_name:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError("published replay output selection must be unique")
    output = matches[0]
    paths = {
        kind: _resolve_child(root, output.get(kind))
        for kind in ("video", "event", "tracking", "metadata")
    }
    for name in (
        "source_frame_count",
        "tracking_frame_count",
        "event_frame_count",
        "video_frame_count",
    ):
        value = output.get(name)
        if type(value) is not int or value < 1:
            raise ValueError(f"completion output has invalid {name}")
    if not (
        output["source_frame_count"]
        == output["tracking_frame_count"]
        == output["event_frame_count"]
    ):
        raise ValueError("published source sidecar frame counts disagree")
    artifacts = output.get("artifacts")
    with paths["metadata"].open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError("published replay metadata root must be an object")
    if metadata.get("schema") not in {
        "footballworld.replay-metadata/7",
        METADATA_SCHEMA,
    }:
        raise ValueError("published replay metadata schema is unsupported")
    source_frames = output["source_frame_count"]
    for name, expected in (
        ("source_frame_count", source_frames),
        ("tracking_frame_count", source_frames),
        ("video_frame_count", output["video_frame_count"]),
    ):
        if metadata.get(name) != expected:
            raise ValueError(f"published replay metadata disagrees on {name}")
    control_fps = metadata.get("control_fps")
    sample_fps = metadata.get("video_sample_fps")
    video_fps = metadata.get("video_fps")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in (control_fps, sample_fps, video_fps)
    ):
        raise ValueError("published replay metadata has invalid timebase rates")
    sample_every = metadata.get("sample_every")
    if type(sample_every) is not int or sample_every < 1:
        raise ValueError("published replay metadata has invalid sample_every")
    expected_samples = round(source_frames * sample_fps / control_fps)
    expected_video = (expected_samples + sample_every - 1) // sample_every
    if metadata.get("video_sample_frame_count") != expected_samples:
        raise ValueError("published video sample count disagrees with source duration")
    if output["video_frame_count"] != expected_video:
        raise ValueError("published video frame count disagrees with its sample grid")
    expected_video_fps = expected_video * sample_fps / expected_samples
    if not math.isclose(
        video_fps,
        expected_video_fps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("published video fps does not preserve sample duration")
    if metadata.get("schema") == METADATA_SCHEMA:
        time_axis = metadata.get("time_axis")
        if not isinstance(time_axis, dict):
            raise TypeError("published replay metadata has no time axis")
        origin_s = time_axis.get("video_origin_time_s")
        step_s = time_axis.get("video_time_step_s")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (origin_s, step_s)
            )
            or step_s <= 0.0
        ):
            raise ValueError("published replay has invalid absolute video time axis")
        if not math.isclose(step_s, 1.0 / video_fps, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("published video time step disagrees with video fps")
    if not isinstance(artifacts, dict):
        raise TypeError("published replay has no artifact receipts")
    if verify_hashes:
        for kind, child in paths.items():
            declared = artifacts.get(kind)
            if not isinstance(declared, dict):
                raise TypeError(f"published replay lacks {kind} receipt")
            if _artifact_receipt(child) != declared:
                raise ValueError(f"published replay {kind} hash or size changed")
    if require_authoritative:
        guard = completion.get("publication_guard")
        if not isinstance(guard, dict):
            raise ValueError("published replay has no publication guard")
        required = {
            "done": True,
            "complete": True,
            "full_duration_complete": True,
            "terminal_basis": "regulation_complete",
            "maximum_steps": None,
            "event_budget_exhausted_count": 0,
        }
        if any(completion.get(name) != value for name, value in required.items()):
            raise ValueError("published replay is not a complete regulation match")
        if (
            guard.get("authoritative") is not True
            or guard.get("status") != "valid"
            or guard.get("source_authority") not in {"clean", "frozen"}
            or guard.get("stable_during_capture") is not True
        ):
            raise ValueError("published replay source authority is not valid")
        decode = output.get("video_decode_verification")
        if not isinstance(decode, dict) or decode.get("enabled") is not True:
            raise ValueError("published replay lacks pre-publication full decode")
        checks = decode.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise ValueError("published replay video decode checks did not all pass")
    return PublishedReplay(
        root=root,
        video=paths["video"],
        event=paths["event"],
        tracking=paths["tracking"],
        metadata=paths["metadata"],
        completion=completion,
        output=output,
    )


__all__ = ["PublishedReplay", "open_published_replay"]
