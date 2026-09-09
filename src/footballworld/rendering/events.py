"""Bounded-memory reader for the canonical replay event sidecar."""

from __future__ import annotations

import codecs
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EVENTS_SCHEMA = "footballworld.events/15"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _root_frames_marker(value: str) -> tuple[int, int] | None:
    """Locate a root ``frames`` array without matching nested object keys."""

    depth = 0
    index = 0
    length = len(value)
    while index < length:
        token = value[index]
        if token in " \t\r\n":
            index += 1
            continue
        if token == '"':
            start = index
            index += 1
            escaped = False
            while index < length:
                current = value[index]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
                index += 1
            if index >= length:
                return None
            end = index + 1
            previous = start - 1
            while previous >= 0 and value[previous] in " \t\r\n":
                previous -= 1
            is_root_key = depth == 1 and (previous < 0 or value[previous] in "{,")
            if is_root_key:
                try:
                    key = json.loads(value[start:end])
                except json.JSONDecodeError:
                    key = None
                cursor = end
                while cursor < length and value[cursor] in " \t\r\n":
                    cursor += 1
                if key == "frames":
                    if cursor >= length:
                        return None
                    if value[cursor] != ":":
                        index = end
                        continue
                    cursor += 1
                    while cursor < length and value[cursor] in " \t\r\n":
                        cursor += 1
                    if cursor >= length:
                        return None
                    if value[cursor] == "[":
                        return start, cursor + 1
            index = end
            continue
        if token in "{[":
            depth += 1
        elif token in "}]":
            depth -= 1
        index += 1
    return None


class EventStream:
    """Incrementally read the final ``frames`` member of ``event.json``.

    The header is available immediately.  The digest becomes available only
    after the iterator reaches EOF, so a partial read cannot masquerade as a
    verified artifact.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_schema: str | None = EVENTS_SCHEMA,
        expected_first_frame: int = 0,
    ) -> None:
        if type(expected_first_frame) is not int or expected_first_frame < 0:
            raise ValueError("expected_first_frame must be a non-negative integer")
        self.path = Path(path)
        self._stream = self.path.open("rb")
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._json_decoder = json.JSONDecoder(
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        self._digest = hashlib.sha256()
        self._buffer = ""
        self._index = 0
        self._eof = False
        self._finished = False
        self._iterated = False
        self._count = 0
        self._expected_frame = expected_first_frame
        self._previous_tick: int | None = None
        try:
            self.header = self._read_prefix()
            schema = self.header.get("schema")
            if expected_schema is not None and schema != expected_schema:
                raise ValueError(
                    f"unsupported event schema {schema!r}; expected {expected_schema!r}"
                )
        except Exception:
            self.close()
            raise

    @property
    def sha256(self) -> str:
        if not self._finished:
            raise RuntimeError("event stream must reach EOF before reading SHA-256")
        return self._digest.hexdigest()

    @property
    def count(self) -> int:
        if not self._finished:
            raise RuntimeError("event stream must reach EOF before reading count")
        return self._count

    def _read_more(self) -> bool:
        if self._eof:
            return False
        raw = self._stream.read(1024 * 1024)
        if raw:
            self._digest.update(raw)
            self._buffer += self._decoder.decode(raw, final=False)
            return True
        self._buffer += self._decoder.decode(b"", final=True)
        self._eof = True
        return False

    def _read_prefix(self) -> dict[str, Any]:
        while True:
            marker = _root_frames_marker(self._buffer)
            if marker is not None:
                start, end = marker
                prefix = self._buffer[:start]
                try:
                    header = json.loads(
                        prefix + '"frames":null}',
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_nonfinite_constant,
                    )
                except json.JSONDecodeError as error:
                    self.close()
                    raise ValueError(
                        f"malformed event metadata in {self.path}"
                    ) from error
                self._buffer = self._buffer[end:]
                self._index = 0
                if not isinstance(header, dict):
                    self.close()
                    raise TypeError("event root must be a JSON object")
                header.pop("frames", None)
                return header
            if not self._read_more():
                self.close()
                raise ValueError(f"{self.path} has no final frames array")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._iterated:
            raise RuntimeError("EventStream is a single-pass iterator")
        self._iterated = True
        first = True
        while True:
            if self._index > 1024 * 1024:
                self._buffer = self._buffer[self._index :]
                self._index = 0
            while True:
                while (
                    self._index < len(self._buffer)
                    and self._buffer[self._index] in " \t\r\n"
                ):
                    self._index += 1
                if self._index < len(self._buffer) or self._eof:
                    break
                self._read_more()
            if self._index >= len(self._buffer):
                raise ValueError(f"unterminated frames array in {self.path}")
            if first and self._buffer[self._index] == "]":
                self._index += 1
                while self._read_more():
                    pass
                if self._buffer[self._index :].strip() != "}":
                    raise ValueError(
                        "event frames must be the final root member with no "
                        "trailing data"
                    )
                self._finished = True
                return
            if not first:
                if self._buffer[self._index] == "]":
                    self._index += 1
                    while self._read_more():
                        pass
                    if self._buffer[self._index :].strip() != "}":
                        raise ValueError(
                            "event frames must be the final root member with no "
                            "trailing data"
                        )
                    self._finished = True
                    return
                if self._buffer[self._index] != ",":
                    raise ValueError(
                        "event frame objects must be comma-separated; "
                        f"frame={self._expected_frame}, "
                        f"token={self._buffer[self._index]!r}"
                    )
                self._index += 1
                while True:
                    while (
                        self._index < len(self._buffer)
                        and self._buffer[self._index] in " \t\r\n"
                    ):
                        self._index += 1
                    if self._index < len(self._buffer) or self._eof:
                        break
                    self._read_more()
                if self._index >= len(self._buffer):
                    raise ValueError(f"unterminated frames array in {self.path}")
                if self._buffer[self._index] == "]":
                    raise ValueError(
                        "event frames array must not have a trailing comma"
                    )
            while True:
                try:
                    value, end = self._json_decoder.raw_decode(
                        self._buffer, self._index
                    )
                    break
                except json.JSONDecodeError as error:
                    if self._eof:
                        raise ValueError(
                            f"malformed frame JSON near character {self._index}"
                        ) from error
                    self._read_more()
            self._index = end
            if not isinstance(value, dict):
                raise TypeError("every event frames member must be an object")
            frame = value.get("frame")
            if type(frame) is not int or frame != self._expected_frame:
                raise ValueError(
                    "event frame indices must be contiguous from zero; "
                    f"expected {self._expected_frame}, got {frame!r}"
                )
            tick = value.get("control_tick")
            if type(tick) is not int or (
                self._previous_tick is not None and tick <= self._previous_tick
            ):
                raise ValueError(
                    "event control_tick values must be strictly increasing"
                )
            self._expected_frame += 1
            self._previous_tick = tick
            self._count += 1
            first = False
            yield value

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> EventStream:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_events(
    path: str | Path,
    *,
    expected_schema: str | None = EVENTS_SCHEMA,
    expected_first_frame: int = 0,
) -> EventStream:
    """Open one canonical event sidecar without materializing all frames."""

    return EventStream(
        path,
        expected_schema=expected_schema,
        expected_first_frame=expected_first_frame,
    )


__all__ = ["EVENTS_SCHEMA", "EventStream", "open_events"]
