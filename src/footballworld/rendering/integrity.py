"""Host-only provenance and atomic publication for replay artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import numpy as np

from footballworld.runtime import environment_fingerprint

RENDER_COMPLETION_SCHEMA = "footballworld.render-completion/2"


def _source_sha256() -> str:
    """Fingerprint the imported FootballWorld Python source, including dirty edits."""

    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible host configuration without changing its meaning."""

    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def publication_authority(
    *,
    dirty: bool,
    maximum_steps: int | None,
) -> dict[str, object]:
    """Classify source authority separately from match completion authority."""

    if dirty:
        status = "diagnostic-dirty-source"
    elif maximum_steps is not None:
        status = "diagnostic-step-budget"
    else:
        status = "valid"
    return {
        "status": status,
        "authoritative": (not dirty) and maximum_steps is None,
        "source_authority": "diagnostic-dirty" if dirty else "clean",
    }


def artifact_receipt(path: str | Path) -> dict[str, Any]:
    """Hash one finalized child artifact for the enclosing manifest."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"replay artifact is not a file: {source}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def validate_authoritative_completion(
    completion: Mapping[str, Any], guard: Mapping[str, Any] | None
) -> None:
    """Fail closed before publishing a caller-declared authoritative replay."""

    if guard is None or guard.get("authoritative") is not True:
        return
    source_authority = guard.get("source_authority")
    if source_authority not in {"clean", "frozen"}:
        raise RuntimeError(
            "authoritative publication requires source_authority='clean' or 'frozen'"
        )
    if guard.get("stable_during_capture") is not True:
        raise RuntimeError(
            "authoritative publication requires stable_during_capture=true"
        )
    if guard.get("status") != "valid":
        raise RuntimeError("authoritative publication guard status must be 'valid'")
    required = {
        "done": True,
        "complete": True,
        "full_duration_complete": True,
        "terminal_basis": "regulation_complete",
        "maximum_steps": None,
        "event_budget_exhausted_count": 0,
    }
    mismatches = {
        name: (expected, completion.get(name))
        for name, expected in required.items()
        if completion.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "authoritative publication requires regulation plus causal added "
            f"time without a step budget: {mismatches}"
        )


def verify_video_decode(
    path: str | Path,
    *,
    expected_frame_count: int,
    expected_fps: float,
    expected_width_px: int,
    expected_height_px: int,
    expected_codec: str = "libx264",
    expected_pixel_format: str = "yuv420p",
) -> dict[str, Any]:
    """Cold-path complete decode and stream check for publication."""

    try:
        import imageio_ffmpeg
    except (ImportError, OSError) as error:
        raise RuntimeError("video verification requires imageio-ffmpeg") from error
    video = Path(path)
    try:
        executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
        decoded, duration_s = imageio_ffmpeg.count_frames_and_secs(video)
        reader = imageio_ffmpeg.read_frames(video, pix_fmt="rgb24")
        try:
            stream = next(reader)
        finally:
            reader.close()
    except Exception as error:
        raise RuntimeError(f"video probe/decode failed for {video}: {error}") from error
    size = stream.get("source_size", stream.get("size"))
    actual_codec = stream.get("codec")
    actual_pixel = stream.get("pix_fmt")
    actual_fps = stream.get("fps")
    if type(decoded) is not int or decoded <= 0:
        raise RuntimeError(f"video decode returned invalid frame count: {decoded!r}")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise RuntimeError(f"video probe returned invalid dimensions: {size!r}")
    if not isinstance(actual_fps, (int, float)) or not math.isfinite(float(actual_fps)):
        raise RuntimeError(f"video probe returned invalid fps: {actual_fps!r}")
    if not isinstance(duration_s, (int, float)) or not math.isfinite(float(duration_s)):
        raise RuntimeError(f"video decode returned invalid duration: {duration_s!r}")
    aliases = {"libx264": "h264", "libx264rgb": "h264"}
    expected_codec_normalized = aliases.get(
        expected_codec.lower(), expected_codec.lower()
    )
    actual_pixel_normalized = str(actual_pixel).split("(", 1)[0].strip().lower()
    expected_duration = expected_frame_count / expected_fps
    checks = {
        "frame_count": decoded == expected_frame_count,
        "fps": math.isclose(
            float(actual_fps), expected_fps, rel_tol=1e-6, abs_tol=1e-6
        ),
        "width_px": int(size[0]) == expected_width_px,
        "height_px": int(size[1]) == expected_height_px,
        "codec": str(actual_codec).lower() == expected_codec_normalized,
        "pixel_format": actual_pixel_normalized == expected_pixel_format.lower(),
        "duration_s": math.isclose(
            float(duration_s),
            expected_duration,
            rel_tol=0.0,
            abs_tol=max(1.0 / expected_fps, 1e-3),
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"video decode verification failed for {video}: {failed}")
    return {
        "enabled": True,
        "method": "imageio_ffmpeg_full_decode_plus_stream_probe",
        "ffmpeg_executable": str(executable),
        "ffmpeg_version": stream.get("ffmpeg_version"),
        "actual": {
            "frame_count": decoded,
            "duration_s": float(duration_s),
            "fps": float(actual_fps),
            "width_px": int(size[0]),
            "height_px": int(size[1]),
            "codec": actual_codec,
            "pixel_format": actual_pixel,
        },
        "checks": checks,
    }


def replay_provenance(
    env: Any,
    match_key: jax.Array,
    *,
    policies: Mapping[str, Any],
) -> dict[str, Any]:
    """Return automatic replay provenance without entering a JAX executable."""

    key_data = np.asarray(jax.device_get(jax.random.key_data(match_key)))
    return {
        "environment": environment_fingerprint(env),
        "source_sha256": _source_sha256(),
        "match_key": {
            "implementation": str(jax.random.key_impl(match_key)),
            "data": key_data.astype(np.uint32).tolist(),
        },
        "policies": dict(policies),
    }


def write_completion_manifest(
    staging: Path,
    *,
    completion: Mapping[str, Any],
    outputs: list[Mapping[str, Any]],
) -> Path:
    """Write the last file in a complete, still-private replay directory."""

    path = staging / "completion.json"
    payload = {
        "schema": RENDER_COMPLETION_SCHEMA,
        **dict(completion),
        "outputs": outputs,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        stream.write("\n")
    return path


def _lock_owner_is_alive(lock: Path) -> tuple[bool, os.stat_result]:
    """Return whether a well-formed lock still belongs to a live process."""

    with lock.open("r", encoding="ascii") as stream:
        identity = os.fstat(stream.fileno())
        payload = stream.read().strip()
    fields = {}
    for line in payload.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            fields[name] = value
    owner_host = fields.get("host")
    if owner_host is not None and owner_host != socket.gethostname():
        return True, identity
    if "pid" not in fields:
        return True, identity
    try:
        pid = int(fields["pid"])
    except ValueError:
        return True, identity
    if pid <= 0:
        return True, identity
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, identity
    except (PermissionError, OSError):
        return True, identity
    return True, identity


def _reserve_output_lock(lock: Path) -> int:
    """Reserve a lock, reclaiming it only from a confirmed dead owner."""

    try:
        return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            alive, identity = _lock_owner_is_alive(lock)
            current = lock.stat()
        except FileNotFoundError:
            return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if alive:
            raise
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise FileExistsError(f"output lock changed while inspected: {lock}")
        lock.unlink()
        return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def _release_output_lock(lock: Path, lock_fd: int) -> None:
    """Remove the lock only while the path still names our open inode."""

    identity = os.fstat(lock_fd)
    try:
        current = lock.stat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        return
    lock.unlink()


@contextmanager
def staged_output_directory(target: str | Path) -> Iterator[Path]:
    """Build one replay invocation privately and publish it by directory rename."""

    target = Path(target)
    if not target.name:
        raise ValueError("output_dir must name one directory")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output_dir already exists: {target}")

    lock = parent / f".{target.name}.render.lock"
    try:
        lock_fd = _reserve_output_lock(lock)
    except FileExistsError as exc:
        raise FileExistsError(f"output_dir is already reserved: {target}") from exc
    staging: Path | None = None
    published = False
    try:
        owner = f"host={socket.gethostname()}\npid={os.getpid()}\n"
        os.write(lock_fd, owner.encode("ascii"))
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
        yield staging
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output_dir appeared during rendering: {target}")
        staging.rename(target)
        published = True
    finally:
        try:
            _release_output_lock(lock, lock_fd)
        finally:
            os.close(lock_fd)
        if not published and staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "RENDER_COMPLETION_SCHEMA",
    "artifact_receipt",
    "replay_provenance",
    "stable_json_sha256",
    "staged_output_directory",
    "validate_authoritative_completion",
    "verify_video_decode",
    "write_completion_manifest",
]
