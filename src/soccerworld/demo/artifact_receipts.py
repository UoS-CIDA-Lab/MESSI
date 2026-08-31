"""Fail-closed provenance and output receipts for repository render tools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_tree_sha256(states, *, jax, np) -> str:
    """Fingerprint every named state leaf in stable field order."""

    digest = hashlib.sha256()
    names = getattr(states, "_fields", ())
    leaves = tuple(states) if names else tuple(jax.tree_util.tree_leaves(states))
    if not names:
        names = tuple(f"leaf-{index}" for index in range(len(leaves)))
    for name, value in zip(names, leaves):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode() + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(repository), *arguments),
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"cannot establish render source provenance: {detail.strip()}") from exc
    return completed.stdout.strip()


def git_source_receipt(repository: Path) -> dict[str, object]:
    """Bind an artifact to one clean Git commit/tree, or refuse to certify it."""

    repository = repository.resolve()
    before = (
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "rev-parse", "HEAD^{tree}"),
    )
    dirty = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    after = (
        _git(repository, "rev-parse", "HEAD"),
        _git(repository, "rev-parse", "HEAD^{tree}"),
    )
    if dirty:
        raise RuntimeError("cannot certify render artifacts from a dirty source tree")
    if before != after:
        raise RuntimeError("source commit changed while provenance was being inspected")
    return {
        "git_commit": before[0],
        "git_tree": before[1],
        "dirty": False,
    }


def assert_git_source_unchanged(
    repository: Path,
    expected: dict[str, object],
) -> None:
    """Fail if source provenance changed during rollout or rendering."""

    actual = git_source_receipt(repository)
    if actual != expected:
        raise RuntimeError(
            "source commit/tree changed during render: "
            f"expected {expected}, got {actual}"
        )


def rule_policy_receipt(
    policy,
    *,
    expected_version: int,
    fingerprint: Callable[[object], str],
) -> dict[str, object]:
    """Return the version/config pair that defines rule-policy semantics."""

    if not hasattr(policy, "rule_policy_version"):
        raise RuntimeError("rule policy is missing rule_policy_version provenance")
    if not hasattr(policy, "policy_config"):
        raise RuntimeError("rule policy is missing policy_config provenance")
    actual_version = int(policy.rule_policy_version)
    if actual_version != int(expected_version):
        raise RuntimeError(
            "rule-policy version mismatch: "
            f"exported={expected_version}, policy={actual_version}"
        )
    config_sha256 = fingerprint(policy.policy_config)
    if len(config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config_sha256
    ):
        raise RuntimeError("policy_config_fingerprint returned a non-SHA-256 value")
    return {
        "rule_policy_version": actual_version,
        "policy_config_sha256": config_sha256,
        "fingerprint_method": "policy_config_fingerprint",
    }


def _ffprobe_receipt(path: Path) -> dict[str, object]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {"available": False}
    completed = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            os.fspath(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "available": True,
        "returncode": completed.returncode,
        "result": json.loads(completed.stdout) if completed.returncode == 0 else None,
        "stderr": completed.stderr.strip() if completed.returncode else "",
    }


def media_receipt(
    path: Path,
    *,
    expected_frames: int,
    expected_seconds: float,
    expected_fps: float,
) -> dict[str, object]:
    """Probe one encoded video and fail if it differs from its requested shape."""

    import imageio_ffmpeg

    frame_count, duration = imageio_ffmpeg.count_frames_and_secs(os.fspath(path))
    reader = imageio_ffmpeg.read_frames(os.fspath(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    duration_tolerance = max(0.05, 1.0 / expected_fps)
    if frame_count != expected_frames:
        raise RuntimeError(
            f"encoded frame mismatch for {path}: expected {expected_frames}, got {frame_count}"
        )
    if abs(duration - expected_seconds) > duration_tolerance:
        raise RuntimeError(
            f"encoded duration mismatch for {path}: expected {expected_seconds}, got {duration}"
        )
    actual_fps = float(metadata["fps"])
    if not math.isclose(actual_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(
            f"encoded FPS mismatch for {path}: expected {expected_fps}, got {actual_fps}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "encoded_frames": int(frame_count),
        "encoded_duration_seconds": float(duration),
        "encoded_fps": actual_fps,
        "resolution": [int(value) for value in metadata["size"]],
        "codec": metadata["codec"],
        "pixel_format": metadata["pix_fmt"],
        "probe_method": "imageio_ffmpeg.count_frames_and_secs + read_frames metadata",
        "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(),
        "ffprobe": _ffprobe_receipt(path),
    }
