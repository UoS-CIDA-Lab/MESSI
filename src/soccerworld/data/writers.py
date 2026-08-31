"""Host-side storage boundary for captured numeric PyTree batches."""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["NpzShardSink", "RecordSink"]


_ARRAY_TAG = "__soccerworld_ndarray__"
_BYTES_TAG = "__soccerworld_bytes__"
_COMPLEX_TAG = "__soccerworld_complex__"
_FLOAT_TAG = "__soccerworld_float__"
_NONE_PREFIX = "__none__."


@runtime_checkable
class RecordSink(Protocol):
    """Receives captured numeric batches without coupling dynamics to storage."""

    def open(self, metadata: Mapping[str, Any]) -> None: ...

    def write(self, batch: Any) -> None: ...

    def close(self) -> None: ...


def _json_safe(value: Any, path: str) -> Any:
    """Convert metadata to standard JSON values without discarding array identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        label = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        return {_FLOAT_TAG: label}
    if isinstance(value, complex):
        return {
            _COMPLEX_TAG: [
                _json_safe(float(value.real), path),
                _json_safe(float(value.imag), path),
            ]
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return {_BYTES_TAG: encoded}
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise TypeError(
                f"metadata[{path!r}] has unsupported dtype {array.dtype}"
            )
        return {
            _ARRAY_TAG: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "data": _json_safe(array.tolist(), path),
            }
        }
    if isinstance(value, Mapping):
        converted = {}
        keys = list(value)
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(
                    f"metadata mapping keys must be strings, got {type(key).__name__} "
                    f"at {path or '<root>'}"
                )
        for key in sorted(keys):
            child = f"{path}.{key}" if path else key
            converted[key] = _json_safe(value[key], child)
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"metadata[{path!r}] has unsupported type {type(value).__name__}"
    )


def _path_segment(value: str, path: str) -> str:
    if not value.isidentifier() or value.startswith("__"):
        raise TypeError(
            f"record field names must be non-empty identifiers without reserved "
            f"separators, got {value!r} at {path or '<root>'}"
        )
    return value


def _flatten_numeric_tree(batch: Any) -> dict[str, np.ndarray]:
    """Flatten a host NamedTuple/mapping PyTree to stable numeric NPZ members."""

    fields = getattr(type(batch), "_fields", None)
    if not isinstance(batch, Mapping) and not (
        isinstance(batch, tuple) and isinstance(fields, tuple) and fields
    ):
        raise TypeError(
            "batch must be a non-empty NamedTuple or mapping PyTree, got "
            f"{type(batch).__name__}"
        )
    arrays: dict[str, np.ndarray] = {}

    def add(name: str, value: np.ndarray) -> None:
        if name in arrays:
            raise ValueError(f"duplicate flattened record field {name!r}")
        arrays[name] = value

    def visit(value: Any, path: str) -> None:
        fields = getattr(type(value), "_fields", None)
        if isinstance(value, tuple) and isinstance(fields, tuple):
            if not fields:
                raise TypeError(f"empty named tuple is unsupported at {path}")
            for field in fields:
                segment = _path_segment(field, path)
                child = f"{path}.{segment}" if path else segment
                visit(getattr(value, field), child)
            return
        if isinstance(value, Mapping):
            if not value:
                raise TypeError(f"empty mapping is unsupported at {path}")
            keys = list(value)
            for key in keys:
                if not isinstance(key, str):
                    raise TypeError(
                        f"record mapping keys must be strings at {path}, got "
                        f"{type(key).__name__}"
                    )
            for key in sorted(keys):
                segment = _path_segment(key, path)
                visit(value[key], f"{path}.{segment}" if path else segment)
            return
        if isinstance(value, (list, tuple)):
            if not value:
                raise TypeError(f"empty sequence is unsupported at {path}")
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}" if path else str(index))
            return
        if value is None:
            add(f"{_NONE_PREFIX}{path}", np.asarray(True, dtype=np.bool_))
            return
        if not isinstance(value, (np.ndarray, np.generic, bool, int, float, complex)):
            raise TypeError(
                f"record leaf {path!r} must be a numeric array, got "
                f"{type(value).__name__}"
            )
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise TypeError(
                f"record leaf {path!r} has unsupported dtype {array.dtype}"
            )
        if array.dtype.kind not in "biufc":
            raise TypeError(
                f"record leaf {path!r} must have boolean or numeric dtype, "
                f"got {array.dtype}"
            )
        add(path, np.ascontiguousarray(array))

    visit(batch, "")
    return arrays


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform durability fallback
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(target: Path, write_content) -> None:
    """Publish one complete file atomically while refusing an existing target.

    Linking a fully flushed temporary inode gives readers either no target or the
    complete target.  Reserving the target with an empty placeholder first would
    leave a small window in which a dataset consumer could observe a zero-byte
    metadata file or shard.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write_content(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


class NpzShardSink:
    """Single-writer, host-side sink for numbered numeric PyTree NPZ shards.

    ``write`` performs exactly one bulk ``jax.device_get`` before flattening.
    A batch may be a :class:`TransitionRecord` or a larger NamedTuple/mapping
    containing state sequences, transition sidecars, and replay keys. Every
    leaf must remain numeric; object arrays and ambiguous field names fail
    closed.
    The sink is deliberately one-shot: create a new instance for a new dataset.
    Existing metadata or shard paths are never overwritten.
    """

    def __init__(self, directory: str | os.PathLike[str], *, prefix: str = "shard"):
        if not isinstance(prefix, str) or not prefix or prefix.startswith("."):
            raise ValueError("prefix must be a non-empty visible filename prefix")
        if any(character in prefix for character in ("/", "\\")):
            raise ValueError("prefix must not contain path separators")
        self.directory = Path(directory)
        self.prefix = prefix
        self._state = "new"
        self._next_index = 0

    @property
    def shards_written(self) -> int:
        return self._next_index

    def _require(self, expected: str, operation: str) -> None:
        if self._state != expected:
            raise RuntimeError(
                f"cannot {operation}: sink state is {self._state!r}, "
                f"expected {expected!r}"
            )

    def open(self, metadata: Mapping[str, Any]) -> None:
        self._require("new", "open")
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"metadata must be a mapping, got {type(metadata).__name__}"
            )
        import jax

        host_metadata = jax.device_get(dict(metadata))
        payload = _json_safe(host_metadata, "")
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_publish(
            self.directory / "metadata.json",
            lambda stream: stream.write(encoded),
        )
        self._state = "open"

    def write(self, batch: Any) -> None:
        self._require("open", "write")
        target = self.directory / f"{self.prefix}-{self._next_index:06d}.npz"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing shard: {target}")
        import jax

        host_batch = jax.device_get(batch)
        arrays = _flatten_numeric_tree(host_batch)
        _atomic_publish(target, lambda stream: np.savez(stream, **arrays))
        self._next_index += 1

    def close(self) -> None:
        self._require("open", "close")
        self._state = "closed"
