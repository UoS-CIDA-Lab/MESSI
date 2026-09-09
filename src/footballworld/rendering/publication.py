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
from footballworld.rendering.tracking import (
    open_tracking,
    tracking_storage_receipt,
)


@dataclass(frozen=True, slots=True)
class PublishedReplay:
    """Resolved, verified paths for one completion-manifest output."""

    root: Path
    video: Path
    event: Path
    tracking: Path
    metadata: Path
    metadata_record: dict[str, Any]
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in published replay: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant in published replay: {value}")


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


def authoritative_completion_error(
    completion: dict[str, Any],
    output: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Return why a completion is non-authoritative, or ``None`` when valid."""

    if metadata is not None:
        manifest_payload = {
            name: value
            for name, value in completion.items()
            if name not in {"schema", "outputs"}
        }
        if metadata.get("completion") != manifest_payload:
            return "published replay metadata and manifest completion disagree"
    guard = completion.get("publication_guard")
    if not isinstance(guard, dict):
        return "published replay has no publication guard"
    required = {
        "done": True,
        "complete": True,
        "full_duration_complete": True,
        "terminal_basis": "regulation_complete",
        "maximum_steps": None,
        "event_budget_exhausted_count": 0,
    }
    if any(completion.get(name) != value for name, value in required.items()):
        return "published replay is not a complete regulation match"
    if (
        guard.get("authoritative") is not True
        or guard.get("status") != "valid"
        or guard.get("source_authority") not in {"clean", "frozen"}
        or guard.get("stable_during_capture") is not True
    ):
        return "published replay source authority is not valid"
    decode = output.get("video_decode_verification")
    if not isinstance(decode, dict) or decode.get("enabled") is not True:
        return "published replay lacks pre-publication full decode"
    checks = decode.get("checks")
    required_decode_checks = {
        "frame_count",
        "fps",
        "width_px",
        "height_px",
        "codec",
        "pixel_format",
        "duration_s",
    }
    if (
        not isinstance(checks, dict)
        or not required_decode_checks.issubset(checks)
        or not all(value is True for value in checks.values())
    ):
        return "published replay video decode checks did not all pass"
    return None


def _expected_video_sample_count(
    metadata: dict[str, Any],
    *,
    source_frames: int,
    control_fps: float,
    sample_fps: float,
) -> int:
    """Rebuild the renderer's causal-hold sample count from its time span."""

    time_axis = metadata.get("time_axis")
    tracking_span_s = (
        time_axis.get("tracking_span_s") if isinstance(time_axis, dict) else None
    )
    if tracking_span_s is None:
        if source_frames == 1:
            return 1
        legacy_covered_samples = source_frames * sample_fps / control_fps
        return max(1, math.floor(legacy_covered_samples + 0.5))
    if (
        isinstance(tracking_span_s, bool)
        or not isinstance(tracking_span_s, (int, float))
        or not math.isfinite(tracking_span_s)
        or tracking_span_s < 0.0
    ):
        raise ValueError("published replay has invalid tracking span")
    covered_seconds = tracking_span_s + 1.0 / control_fps
    return max(1, math.floor(covered_seconds * sample_fps + 0.5))


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
        completion = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
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
        metadata = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    if not isinstance(metadata, dict):
        raise TypeError("published replay metadata root must be an object")
    if metadata.get("schema") not in {
        "footballworld.replay-metadata/7",
        "footballworld.replay-metadata/8",
        METADATA_SCHEMA,
    }:
        raise ValueError("published replay metadata schema is unsupported")
    observed_tracking: dict[str, Any] | None = None
    if metadata.get("schema") == METADATA_SCHEMA:
        declared_tracking = metadata.get("tracking_storage")
        if not isinstance(declared_tracking, dict):
            raise TypeError("published replay metadata has no tracking receipt")
        if output.get("tracking_storage") != declared_tracking:
            raise ValueError(
                "published replay manifest and metadata tracking receipts disagree"
            )
        with open_tracking(paths["tracking"]) as tracking_reader:
            tracking_index = getattr(tracking_reader, "index", None)
            if not isinstance(tracking_index, dict):
                raise TypeError(
                    "current published replay tracking must use canonical NPZ"
                )
            observed_tracking = tracking_storage_receipt(
                paths["tracking"], tracking_index
            )
        if observed_tracking != declared_tracking:
            raise ValueError(
                "published replay tracking archive disagrees with its receipt"
            )
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
    expected_samples = _expected_video_sample_count(
        metadata,
        source_frames=source_frames,
        control_fps=control_fps,
        sample_fps=sample_fps,
    )
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
            if kind == "tracking" and observed_tracking is not None:
                observed_artifact = {
                    "bytes": observed_tracking["compressed_bytes"],
                    "sha256": observed_tracking["archive_sha256"],
                }
            else:
                observed_artifact = _artifact_receipt(child)
            if observed_artifact != declared:
                raise ValueError(f"published replay {kind} hash or size changed")
    if require_authoritative:
        authority_error = authoritative_completion_error(
            completion, output, metadata=metadata
        )
        if authority_error is not None:
            raise ValueError(authority_error)
    return PublishedReplay(
        root=root,
        video=paths["video"],
        event=paths["event"],
        tracking=paths["tracking"],
        metadata=paths["metadata"],
        metadata_record=metadata,
        completion=completion,
        output=output,
    )


__all__ = [
    "PublishedReplay",
    "authoritative_completion_error",
    "open_published_replay",
]
