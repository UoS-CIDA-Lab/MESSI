"""Bounded-memory assembly of replay sidecars from fixed rollout chunks."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from footballworld.rendering.events import EVENTS_SCHEMA, open_events
from footballworld.rendering.replay import (
    METADATA_SCHEMA,
    TRACKING_FILENAME,
    _acting_goalkeeper_records,
    _formation_records,
    _substitution_records,
    write_replay_sidecars,
)
from footballworld.rendering.tracking import (
    merge_tracking_archives,
    open_tracking,
    tracking_storage_receipt,
)
from footballworld.rendering.transfer import HostFrame


@dataclass
class _MissingFrames:
    """Chunk-compressed unavailable-frame metadata."""

    spans: list[tuple[int, int]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    count: int = 0

    def extend(self, value: list[int] | str, *, offset: int, count: int) -> None:
        if value == "all":
            self.spans.append((offset, offset + count))
            self.count += count
            return
        rows = [offset + int(item) for item in value]
        self.indices.extend(rows)
        self.count += len(rows)

    def spec(self, total: int) -> list[int] | str:
        if self.count == 0:
            return []
        if self.count == total:
            return "all"
        rows = list(self.indices)
        for start, end in self.spans:
            rows.extend(range(start, end))
        rows.sort()
        return rows


def _validate_chunk_tracking_storage(
    record: Any,
    path: Path,
    *,
    frame_rows: int,
    index: dict[str, Any],
) -> None:
    """Fail closed before a structured chunk enters the final archive."""

    if not isinstance(record, dict):
        raise TypeError("tracking chunk metadata has no storage receipt")
    expected = tracking_storage_receipt(path, index)
    if expected["frame_rows"] != frame_rows:
        raise ValueError("tracking chunk frame count changed before merge")
    mismatches = {
        key: (expected_value, record.get(key))
        for key, expected_value in expected.items()
        if record.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"tracking chunk storage receipt changed: {mismatches}")


class ReplaySidecarSpool:
    """Write bounded chunk files and merge them into one canonical replay."""

    def __init__(
        self,
        spool_dir: str | Path,
        video_path: str | Path,
        *,
        control_fps: float,
        match_index: int,
        stadium: Any,
        halftime_seconds: float,
        fulltime_seconds: float,
        halftime_enabled: bool,
        metadata: Any,
        include_all_action_controls: bool = False,
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.video_path = Path(video_path)
        self.control_fps = float(control_fps)
        self.match_index = int(match_index)
        self.stadium = stadium
        self.halftime_seconds = float(halftime_seconds)
        self.fulltime_seconds = float(fulltime_seconds)
        self.halftime_enabled = bool(halftime_enabled)
        self.metadata = metadata
        if type(include_all_action_controls) is not bool:
            raise TypeError("include_all_action_controls must be bool")
        self.include_all_action_controls = include_all_action_controls
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[tuple[Path, int]] = []
        self._frame_count = 0
        self._pre_frame_management: list[dict[str, Any]] = []
        self._frame_formations: dict[int, list[dict[str, Any]]] = {}

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append(self, frames: list[HostFrame]) -> None:
        """Serialize one source-grid chunk without retaining its frame objects."""

        if not frames:
            return
        chunk = self.spool_dir / f"{len(self._chunks):06d}"
        chunk.mkdir()
        marker = chunk / "chunk.mp4"
        write_replay_sidecars(
            frames,
            marker,
            control_fps=self.control_fps,
            video_sample_fps=self.control_fps,
            sample_every=1,
            video_sample_frame_count=len(frames),
            video_frame_count=len(frames),
            video_fps=self.control_fps,
            match_index=self.match_index,
            stadium=self.stadium,
            halftime_seconds=self.halftime_seconds,
            fulltime_seconds=self.fulltime_seconds,
            halftime_enabled=self.halftime_enabled,
            metadata=self.metadata,
            collect_exact_events=False,
            frame_offset=self._frame_count,
            include_all_action_controls=self.include_all_action_controls,
        )
        self._chunks.append((chunk, len(frames)))
        self._frame_count += len(frames)

    def append_pre_frame_management(
        self,
        *,
        control_tick: int,
        substitution_events: Any,
        acting_goalkeeper_events: Any,
        formation_requested: Any = None,
        formation_layout_index: Any = None,
        formations_applied: Any = None,
    ) -> None:
        """Retain an exact command transaction before the first physics row."""

        substitutions = _substitution_records(substitution_events)
        acting_goalkeepers = _acting_goalkeeper_records(acting_goalkeeper_events)
        formations = _formation_records(
            formation_requested,
            formation_layout_index,
            formations_applied,
        )
        if not substitutions and not acting_goalkeepers and not formations:
            return
        self._pre_frame_management.append(
            {
                "control_tick": int(control_tick),
                "substitutions": substitutions,
                "acting_goalkeepers": acting_goalkeepers,
                "formations": formations,
            }
        )

    def append_frame_formations(
        self,
        *,
        control_tick: int,
        requested: Any,
        layout_index: Any,
        applied: Any,
    ) -> None:
        """Attach a sparse formation receipt to one manager-boundary frame."""

        formations = _formation_records(requested, layout_index, applied)
        if not formations:
            return
        tick = int(control_tick)
        if tick in self._frame_formations:
            raise ValueError("multiple formation decisions share one replay frame")
        self._frame_formations[tick] = formations

    def finalize(
        self,
        *,
        video_sample_frame_count: int,
        video_sample_fps: float,
        video_frame_count: int,
        video_fps: float,
        sample_every: int,
        render_metadata: Any,
        completion: dict[str, Any] | None = None,
    ) -> tuple[Path, Path, Path]:
        """Merge chunk sidecars while preserving absolute control clocks."""

        if not self._chunks:
            raise ValueError("cannot finalize an empty replay")
        event_path = self.video_path.parent / "event.json"
        tracking_path = self.video_path.parent / TRACKING_FILENAME
        metadata_path = self.video_path.parent / "metadata.json"
        event_path.parent.mkdir(parents=True, exist_ok=True)

        header: dict[str, Any] | None = None
        missing_events = _MissingFrames()
        missing_action_controls = _MissingFrames()
        missing_substitutions = _MissingFrames()
        missing_goalkeepers = _MissingFrames()
        offset = 0
        first_metadata = None
        first_tick = None
        last_tick = None
        uniform = True
        previous_tick = None
        tracking_sources: list[Path] = []
        event_contract: dict[str, Any] | None = None
        metadata_contract: dict[str, Any] | None = None
        with nullcontext():
            for chunk, count in self._chunks:
                with open_events(
                    chunk / "event.json", expected_first_frame=offset
                ) as stream:
                    rows = list(stream)
                    payload = {**stream.header, "frames": rows}
                if header is None:
                    header = {
                        key: value for key, value in payload.items() if key != "frames"
                    }
                rows = payload["frames"]
                if len(rows) != count:
                    raise ValueError("event chunk frame count changed during spooling")
                current_event_contract = {
                    key: payload.get(key)
                    for key in (
                        "schema",
                        "match_index",
                        "frame_event_encoding",
                        "omitted_event_slots",
                        "clock_source",
                        "intent_source_schema",
                        "intent_source_names",
                        "action_encoding",
                        "action_scope",
                        "action_categorical_source",
                        "action_continuous_source",
                        "complete_action_reconstruction",
                        "omitted_move_controls",
                        "frame_order",
                        "transition_identity_semantics",
                    )
                }
                if current_event_contract["schema"] != EVENTS_SCHEMA:
                    raise ValueError("event chunk uses an unsupported schema")
                if event_contract is None:
                    event_contract = current_event_contract
                elif current_event_contract != event_contract:
                    raise ValueError("event chunk contract changed during spooling")
                for local, event_row in enumerate(rows):
                    if int(event_row["frame"]) != offset + local:
                        raise ValueError(
                            "event chunk frame indices are not spool-global"
                        )

                missing_events.extend(
                    payload.get("missing_frames", []),
                    offset=offset,
                    count=count,
                )
                missing_action_controls.extend(
                    payload.get("missing_action_control_frames", []),
                    offset=offset,
                    count=count,
                )
                missing_substitutions.extend(
                    payload.get("missing_substitution_frames", []),
                    offset=offset,
                    count=count,
                )
                missing_goalkeepers.extend(
                    payload.get("missing_acting_goalkeeper_frames", []),
                    offset=offset,
                    count=count,
                )
                chunk_tracking = chunk / TRACKING_FILENAME
                with open_tracking(chunk_tracking) as source:
                    if not hasattr(source, "index"):
                        raise TypeError("new spool chunks must use canonical NPZ")
                    tracking_rows = 0
                    for table in source.iter_chunks(verify=True):
                        for local, record_row in enumerate(table):
                            if tracking_rows >= count:
                                raise ValueError(
                                    "tracking chunk frame count changed during spooling"
                                )
                            tick = int(record_row["control_tick"])
                            if int(record_row["frame"]) != offset + tracking_rows:
                                raise ValueError(
                                    "tracking chunk frame indices are not spool-global"
                                )
                            if int(rows[tracking_rows]["control_tick"]) != tick:
                                raise ValueError(
                                    "event and tracking control ticks disagree"
                                )
                            tracking_rows += 1
                            if first_tick is None:
                                first_tick = tick
                            if previous_tick is not None and tick - previous_tick != 1:
                                raise ValueError(
                                    "exact tracking control ticks must be contiguous"
                                )
                            previous_tick = tick
                            last_tick = tick
                    chunk_index = source.index
                with (chunk / "metadata.json").open(encoding="utf-8") as stream:
                    chunk_metadata = json.load(stream)
                _validate_chunk_tracking_storage(
                    chunk_metadata.get("tracking_storage"),
                    chunk_tracking,
                    frame_rows=count,
                    index=chunk_index,
                )
                current_metadata_contract = {
                    key: chunk_metadata.get(key)
                    for key in (
                        "schema",
                        "match_index",
                        "control_fps",
                        "video_sample_fps",
                        "sample_every",
                        "clock",
                        "stadium",
                        "render",
                        "user_metadata",
                    )
                }
                if current_metadata_contract["schema"] != METADATA_SCHEMA:
                    raise ValueError("metadata chunk uses an unsupported schema")
                if metadata_contract is None:
                    metadata_contract = current_metadata_contract
                    first_metadata = chunk_metadata
                elif current_metadata_contract != metadata_contract:
                    raise ValueError("metadata chunk contract changed during spooling")
                if tracking_rows != count:
                    raise ValueError(
                        "tracking chunk frame count changed during spooling"
                    )
                tracking_sources.append(chunk_tracking)
                offset += count
                (chunk / "metadata.json").unlink()

        tracking_index = merge_tracking_archives(tracking_sources, tracking_path)
        for source in tracking_sources:
            source.unlink()

        assert header is not None
        header["available"] = missing_events.count == 0
        header["missing_frames"] = missing_events.spec(self._frame_count)
        header["action_controls_available"] = missing_action_controls.count == 0
        header["missing_action_control_frames"] = missing_action_controls.spec(
            self._frame_count
        )
        header["substitution_events_available"] = missing_substitutions.count == 0
        header["missing_substitution_frames"] = missing_substitutions.spec(
            self._frame_count
        )
        header["acting_goalkeeper_events_available"] = missing_goalkeepers.count == 0
        header["missing_acting_goalkeeper_frames"] = missing_goalkeepers.spec(
            self._frame_count
        )
        header["pre_frame_management"] = self._pre_frame_management
        remaining_formations = dict(self._frame_formations)
        with event_path.open("w", encoding="utf-8") as event_out:
            event_out.write(
                json.dumps(header, ensure_ascii=False, separators=(",", ":"))[:-1]
            )
            event_out.write(',"frames":[')
            first = True
            offset = 0
            for chunk, count in self._chunks:
                with open_events(
                    chunk / "event.json", expected_first_frame=offset
                ) as stream:
                    rows = list(stream)
                    payload = {**stream.header, "frames": rows}
                rows = payload["frames"]
                if len(rows) != count:
                    raise ValueError("event chunk frame count changed during merge")
                for local, row in enumerate(rows):
                    if int(row["frame"]) != offset + local:
                        raise ValueError(
                            "event chunk frame indices changed during merge"
                        )
                    formations = remaining_formations.pop(
                        int(row["control_tick"]), None
                    )
                    if formations is not None:
                        row["formations"] = formations
                    if not first:
                        event_out.write(",")
                    json.dump(
                        row,
                        event_out,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    first = False
                offset += count
                # This chunk is now fully represented by the merged staging
                # sidecars, so its last retained file can be released.
                (chunk / "event.json").unlink()
                chunk.rmdir()
            event_out.write("]}\n")

        if remaining_formations:
            raise ValueError("formation receipt has no matching replay frame")

        assert first_metadata is not None
        if offset != self._frame_count:
            raise ValueError("merged event frame count is inconsistent")
        assert first_tick is not None
        assert last_tick is not None
        expected_sample_frames = round(
            self._frame_count * float(video_sample_fps) / self.control_fps
        )
        if int(video_sample_frame_count) != expected_sample_frames:
            raise ValueError("video sample grid and source duration are inconsistent")
        expected_video_frames = (
            int(video_sample_frame_count) + int(sample_every) - 1
        ) // int(sample_every)
        if int(video_frame_count) != expected_video_frames:
            raise ValueError("video and source frame counts are inconsistent")
        if event_contract is None or metadata_contract is None:
            raise ValueError("replay contracts were not observed during merge")
        first_metadata["source_frame_count"] = self._frame_count
        first_metadata["tracking_frame_count"] = self._frame_count
        first_metadata["video_sample_frame_count"] = int(video_sample_frame_count)
        first_metadata["video_frame_count"] = int(video_frame_count)
        first_metadata["tracking_fps"] = (
            self.control_fps if uniform and self._frame_count > 1 else None
        )
        first_metadata["video_sample_fps"] = float(video_sample_fps)
        first_metadata["video_fps"] = float(video_fps)
        first_metadata["sample_every"] = int(sample_every)
        first_metadata["render"] = render_metadata
        first_metadata["tracking_storage"] = tracking_storage_receipt(
            tracking_path, tracking_index
        )
        time_axis = first_metadata["time_axis"]
        time_axis["video_origin_control_tick"] = first_tick
        time_axis["video_origin_time_s"] = (
            first_tick / self.control_fps
            - 1.0 / self.control_fps
            + 1.0 / float(video_sample_fps)
        )
        time_axis["video_time_step_s"] = 1.0 / float(video_fps)
        time_axis["video_clock"] = "absolute_video_time_s"
        time_axis["tracking_uniform"] = bool(uniform and self._frame_count > 1)
        time_axis["tracking_span_s"] = (last_tick - first_tick) / self.control_fps
        time_axis["video_sample_semantics"] = "authoritative_post_physics_substep_state"
        time_axis["video_sampling"] = "bounded_physics_substep_frame_end"
        time_axis["video_duration_s"] = video_frame_count / video_fps
        first_metadata["completion"] = dict(completion or {})
        first_metadata["verified_counts"] = {
            "source": self._frame_count,
            "tracking": self._frame_count,
            "event": self._frame_count,
            "video": int(video_frame_count),
            "video_verification": "successful_encoder_close_and_segment_count",
        }
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(
                first_metadata,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.write("\n")

        self._chunks.clear()
        return event_path, tracking_path, metadata_path


__all__ = ["ReplaySidecarSpool"]
