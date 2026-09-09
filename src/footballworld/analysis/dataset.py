"""Verified, host-only access to one published FootballWorld replay."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from footballworld.rendering.events import EventStream
from footballworld.rendering.events import open_events as open_event_stream
from footballworld.rendering.publication import (
    PublishedReplay,
    authoritative_completion_error,
    open_published_replay,
)
from footballworld.rendering.tracking import open_tracking

SUPPORTED_EVENT_SCHEMAS = {
    "footballworld.events/14",
    "footballworld.events/15",
}


class _VerifiedEventStream:
    """Bind each event pass to the snapshot validated by MatchDataset.open."""

    def __init__(
        self,
        stream: EventStream,
        *,
        expected_count: int,
        expected_sha256: str,
    ) -> None:
        self._stream = stream
        self.header = stream.header
        self._expected_count = expected_count
        self._expected_sha256 = expected_sha256

    def __iter__(self):
        yield from self._stream
        if self._stream.count != self._expected_count:
            raise ValueError("event sidecar frame count changed during analysis")
        if self._stream.sha256 != self._expected_sha256:
            raise ValueError("event sidecar content changed during analysis")

    @property
    def count(self) -> int:
        return self._stream.count

    @property
    def sha256(self) -> str:
        return self._stream.sha256

    def close(self) -> None:
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class MatchDataset:
    """A publication-verified replay plus bounded event access."""

    replay: PublishedReplay
    metadata: dict[str, Any]
    _event_header: dict[str, Any]
    authoritative: bool
    warnings: tuple[str, ...]
    hashes_verified: bool
    _event_frame_count: int
    _event_sha256: str
    _events_cache: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        output_name: str | None = None,
        allow_diagnostic: bool = False,
        verify_hashes: bool = True,
    ) -> MatchDataset:
        """Open a replay and fail closed unless diagnostics are explicitly allowed."""

        if type(allow_diagnostic) is not bool or type(verify_hashes) is not bool:
            raise TypeError("dataset verification switches must be bool")
        replay = open_published_replay(
            path,
            output_name=output_name,
            require_authoritative=not allow_diagnostic,
            verify_hashes=verify_hashes,
        )
        metadata = dict(replay.metadata_record)
        with open_event_stream(replay.event, expected_schema=None) as stream:
            event_header = dict(stream.header)
            for _ in stream:
                pass
            event_frame_count = stream.count
            event_sha256 = stream.sha256
        if event_header.get("schema") not in SUPPORTED_EVENT_SCHEMAS:
            raise ValueError("event sidecar schema is unsupported")
        if event_frame_count != replay.output["event_frame_count"]:
            raise ValueError("event sidecar frame count disagrees with completion")
        if verify_hashes:
            declared_event = replay.output["artifacts"]["event"]
            if event_sha256 != declared_event["sha256"]:
                raise ValueError(
                    "event sidecar content differs from its verified publication"
                )

        completion = replay.completion
        authoritative = (
            authoritative_completion_error(completion, replay.output, metadata=metadata)
            is None
        )
        warnings: list[str] = []
        if not authoritative:
            warnings.append(
                "Diagnostic or partial replay: metrics describe only the captured window."
            )
        exhausted = int(completion.get("event_budget_exhausted_count", 0))
        if exhausted:
            warnings.append(
                f"Exact-event budget was exhausted on {exhausted} control frames."
            )
        missing = event_header.get("missing_frames", [])
        if missing:
            warnings.append("Some exact event frames are unavailable.")
        if event_header.get("action_controls_available", True) is False:
            warnings.append("Some submitted action controls are missing.")
        if not verify_hashes:
            warnings.append("Artifact hashes were not verified for this analysis run.")
        return cls(
            replay=replay,
            metadata=metadata,
            _event_header=event_header,
            authoritative=authoritative,
            warnings=tuple(warnings),
            hashes_verified=verify_hashes,
            _event_frame_count=event_frame_count,
            _event_sha256=event_sha256,
        )

    @property
    def event_header(self) -> dict[str, Any]:
        """Return the small root metadata that precedes the frames array."""

        return self._event_header

    def open_events(self) -> _VerifiedEventStream:
        """Open a new verified, bounded-memory iterator over event frames."""

        return _VerifiedEventStream(
            open_event_stream(
                self.replay.event,
                expected_schema=str(self._event_header["schema"]),
            ),
            expected_count=self._event_frame_count,
            expected_sha256=self._event_sha256,
        )

    @property
    def events(self) -> dict[str, Any]:
        """Return the legacy event document, materializing frames on demand."""

        cached = self._events_cache
        if cached is None:
            with self.open_events() as stream:
                frames = list(stream)
            cached = {**self._event_header, "frames": frames}
            object.__setattr__(self, "_events_cache", cached)
        return cached

    def iter_tracking_rows(self):
        """Yield semantic tracking rows with chunk integrity verification."""

        with open_tracking(self.replay.tracking) as reader:
            yield from reader.iter_rows(verify=True)

    def video_time_s(self, control_tick: int) -> float:
        """Map an absolute control tick onto the replay video's relative clock."""

        control_fps = float(self.metadata["control_fps"])
        axis = self.metadata.get("time_axis", {})
        origin = axis.get("video_origin_time_s")
        if origin is None:
            origin_tick = int(axis.get("video_origin_control_tick", 0))
            origin = origin_tick / control_fps
        value = control_tick / control_fps - float(origin)
        if not math.isfinite(value):
            raise ValueError("computed video time is not finite")
        return max(0.0, value)

    @property
    def match_manifest(self) -> dict[str, Any] | None:
        value = self.metadata.get("match_manifest")
        return value if isinstance(value, dict) else None
