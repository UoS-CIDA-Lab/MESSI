"""Lossless, bounded-memory tracking storage outside the JAX runtime."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from footballworld.rendering.transfer import HostFrame, to_jsonable

TRACKING_SCHEMA = "footballworld.tracking/8"
TRACKING_ARCHIVE_SCHEMA = "footballworld.tracking-npz/1"
TRACKING_STORAGE_SCHEMA = "footballworld.tracking-storage/2"
TRACKING_FILENAME = "tracking.npz"
TRACKING_COMPRESSION_LEVEL = 6
_INDEX_KEY = "__index__"
_CHUNK_PREFIX = "chunk-"

_LAST_CONTACT_INT_FIELDS = (
    "actor",
    "mechanism",
    "intent",
    "outcome",
    "restart_kind",
    "law11_effect",
    "intent_source",
)
_PLAYER_VECTOR_FIELDS = (
    ("position", "player_position"),
    ("velocity", "player_velocity"),
    ("body_forward", "player_body_forward"),
    ("view_forward", "player_view_forward"),
)
_PLAYER_FLOAT_FIELDS = (
    ("gaze_yaw", "player_gaze_yaw"),
    ("height", "player_height"),
    ("stamina_long", "player_stamina_long"),
    ("stamina_short", "player_stamina_short"),
)
_PLAYER_INT_FIELDS = (
    ("slot_generation", "player_slot_generation"),
    ("team", "player_team"),
    ("aerial_recovery_substeps", "player_aerial_recovery_substeps"),
    ("yellow_cards", "player_yellow_cards"),
)
_PLAYER_BOOL_FIELDS = (
    ("goalkeeper", "player_goalkeeper"),
    ("active", "player_active"),
    ("on_pitch", "player_on_pitch"),
    ("sent_off", "player_sent_off"),
    ("offside", "player_offside"),
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _dtype_description(dtype: np.dtype[Any]) -> Any:
    return to_jsonable(np.lib.format.dtype_to_descr(dtype))


def _fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    info.create_system = 3
    info._compresslevel = TRACKING_COMPRESSION_LEVEL
    return info


def _write_array_member(
    archive: zipfile.ZipFile,
    name: str,
    value: np.ndarray,
) -> None:
    if value.dtype.hasobject:
        raise TypeError("tracking arrays must not contain object dtype")
    with archive.open(_fixed_zip_info(name + ".npy"), "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, value, allow_pickle=False)


def _named_children(value: Any) -> list[tuple[str, Any]] | None:
    if isinstance(value, Mapping):
        return [(str(key), item) for key, item in value.items()]
    if hasattr(value, "_asdict"):
        return [(str(key), item) for key, item in value._asdict().items()]
    return None


def _flatten_observation(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    children = _named_children(value)
    if children is not None:
        rows: list[tuple[tuple[str, ...], np.ndarray]] = []
        for name, item in children:
            rows.extend(_flatten_observation(item, (*path, name)))
        return rows
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise TypeError(
            "tracking observations require numeric/bool array leaves; "
            f"path={path!r} dtype={array.dtype}"
        )
    return [(path, array)]


def _observation_spec(frame: HostFrame) -> list[dict[str, Any]]:
    if frame.observation is None:
        return []
    return [
        {
            "field": f"observation_{index:03d}",
            "path": list(path),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
        for index, (path, array) in enumerate(_flatten_observation(frame.observation))
    ]


def _tracking_dtype(
    player_count: int,
    observation_spec: Sequence[Mapping[str, Any]],
) -> np.dtype[Any]:
    fields: list[Any] = [
        ("frame", "<i8"),
        ("control_tick", "<i8"),
        ("clock_s", "<f8"),
        ("period", "i1"),
        ("display_clock_s", "<f8"),
        ("added_time_s", "<f8"),
        ("dead_ball_s", "<f8"),
        ("first_half_live_extension_s", "<f8"),
        ("ball_position", "<f4", (3,)),
        ("ball_velocity", "<f4", (3,)),
        ("ball_spin", "<f4", (3,)),
        ("ball_live", "?"),
        ("player_id", "<i8", (player_count,)),
        ("player_position", "<f4", (player_count, 2)),
        ("player_velocity", "<f4", (player_count, 2)),
        ("player_body_forward", "<f4", (player_count, 2)),
        ("player_view_forward", "<f4", (player_count, 2)),
        ("player_gaze_yaw", "<f4", (player_count,)),
        ("player_height", "<f4", (player_count,)),
        ("player_stamina_long", "<f4", (player_count,)),
        ("player_stamina_short", "<f4", (player_count,)),
        ("player_slot_generation", "<i4", (player_count,)),
        ("player_team", "<i4", (player_count,)),
        ("player_aerial_recovery_substeps", "<i4", (player_count,)),
        ("player_yellow_cards", "<i4", (player_count,)),
        ("player_goalkeeper", "?", (player_count,)),
        ("player_active", "?", (player_count,)),
        ("player_on_pitch", "?", (player_count,)),
        ("player_sent_off", "?", (player_count,)),
        ("player_offside", "?", (player_count,)),
        ("score", "<i4", (2,)),
        ("attack_direction", "<f4", (2,)),
        ("possession_team", "<i4"),
        ("possession_player", "<i4"),
        ("possession_previous_team", "<i4"),
        ("possession_control_ticks", "<i4"),
        ("last_contact_known", "?"),
    ]
    fields.extend((f"last_contact_{name}", "<i4") for name in _LAST_CONTACT_INT_FIELDS)
    fields.extend(
        (
            ("last_contact_kick_applied", "?"),
            ("restart_kind", "<i4"),
            ("restart_team", "<i4"),
            ("restart_substeps_remaining", "<i4"),
            ("restart_taker", "<i4"),
            ("restart_indirect", "?"),
        )
    )
    for item in observation_spec:
        fields.append(
            (
                str(item["field"]),
                np.dtype(str(item["dtype"])),
                tuple(int(axis) for axis in item["shape"]),
            )
        )
    return np.dtype(fields, align=False)


def _contact_scalar(contact: Any, name: str) -> Any:
    source = contact[name] if isinstance(contact, Mapping) else getattr(contact, name)
    value = np.asarray(source)
    if value.shape != ():
        raise ValueError(f"last_contact.{name} must be scalar")
    return value.item()


def build_tracking_table(
    frames: Sequence[HostFrame],
    common_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack one fixed frame chunk without object dtype or duration-scaled state."""

    if not frames or len(frames) != len(common_rows):
        raise ValueError(
            "tracking frames and common rows must be non-empty and aligned"
        )
    player_count = int(frames[0].player_position.shape[0])
    observation_spec = _observation_spec(frames[0])
    dtype = _tracking_dtype(player_count, observation_spec)
    table = np.empty(len(frames), dtype=dtype)
    expected_observation = [
        (tuple(item["path"]), np.dtype(item["dtype"]), tuple(item["shape"]))
        for item in observation_spec
    ]

    for index, (frame, common) in enumerate(zip(frames, common_rows, strict=True)):
        if frame.player_position.shape != (player_count, 2):
            raise ValueError("tracking player shape changed inside a chunk")
        table["frame"][index] = common["frame"]
        table["control_tick"][index] = common["control_tick"]
        table["clock_s"][index] = common["clock_s"]
        table["period"][index] = common["period"]
        table["display_clock_s"][index] = common["display_clock_s"]
        table["added_time_s"][index] = common["added_time_s"]
        table["dead_ball_s"][index] = common["dead_ball_s"]
        table["first_half_live_extension_s"][index] = common[
            "first_half_live_extension_s"
        ]
        table["ball_position"][index] = frame.ball_position
        table["ball_velocity"][index] = frame.ball_velocity
        table["ball_spin"][index] = frame.ball_spin
        table["ball_live"][index] = frame.ball_live
        table["player_id"][index] = frame.player_id
        table["player_position"][index] = frame.player_position
        table["player_velocity"][index] = frame.player_velocity
        table["player_body_forward"][index] = frame.player_body_forward
        gaze_cos = np.cos(frame.player_gaze_yaw)
        gaze_sin = np.sin(frame.player_gaze_yaw)
        table["player_view_forward"][index, :, 0] = (
            frame.player_body_forward[:, 0] * gaze_cos
            - frame.player_body_forward[:, 1] * gaze_sin
        )
        table["player_view_forward"][index, :, 1] = (
            frame.player_body_forward[:, 0] * gaze_sin
            + frame.player_body_forward[:, 1] * gaze_cos
        )
        table["player_gaze_yaw"][index] = frame.player_gaze_yaw
        table["player_height"][index] = frame.player_height
        table["player_stamina_long"][index] = frame.stamina_long
        table["player_stamina_short"][index] = frame.stamina_short
        table["player_slot_generation"][index] = frame.slot_generation
        table["player_team"][index] = frame.team_id
        table["player_aerial_recovery_substeps"][index] = frame.aerial_recovery_substeps
        table["player_yellow_cards"][index] = frame.yellow_cards
        table["player_goalkeeper"][index] = frame.is_goalkeeper
        table["player_active"][index] = frame.active
        table["player_on_pitch"][index] = frame.on_pitch
        table["player_sent_off"][index] = frame.sent_off
        table["player_offside"][index] = frame.offside_flagged
        table["score"][index] = frame.score
        table["attack_direction"][index] = frame.attack_direction
        table["possession_team"][index] = frame.possession_team
        table["possession_player"][index] = frame.possession_player
        table["possession_previous_team"][index] = frame.possession_previous_team
        table["possession_control_ticks"][index] = frame.possession_control_ticks
        contact = frame.last_contact
        table["last_contact_known"][index] = contact is not None
        if contact is None:
            for name in _LAST_CONTACT_INT_FIELDS:
                table[f"last_contact_{name}"][index] = 0
            table["last_contact_kick_applied"][index] = False
        else:
            for name in _LAST_CONTACT_INT_FIELDS:
                table[f"last_contact_{name}"][index] = _contact_scalar(contact, name)
            table["last_contact_kick_applied"][index] = _contact_scalar(
                contact, "kick_applied"
            )
        table["restart_kind"][index] = frame.restart_kind
        table["restart_team"][index] = frame.restart_team
        table["restart_substeps_remaining"][index] = frame.restart_substeps_remaining
        table["restart_taker"][index] = frame.restart_taker
        table["restart_indirect"][index] = frame.restart_indirect

        if observation_spec:
            leaves = _flatten_observation(frame.observation)
            observed = [(path, array.dtype, array.shape) for path, array in leaves]
            if observed != expected_observation:
                raise ValueError("tracking observation tree changed inside a chunk")
            for spec, (_, array) in zip(observation_spec, leaves, strict=True):
                table[str(spec["field"])][index] = array
        elif frame.observation is not None:
            raise ValueError("tracking observation presence changed inside a chunk")

    nonfinite_fields = [
        name
        for name in table.dtype.names or ()
        if table.dtype[name].base.kind == "f" and not np.all(np.isfinite(table[name]))
    ]
    if nonfinite_fields:
        raise ValueError(
            f"tracking chunk contains non-finite floating values in {nonfinite_fields}"
        )

    static = {
        "player_count": player_count,
        "player_slot": list(range(player_count)),
        "observation": {
            "present": bool(observation_spec),
            "leaves": observation_spec,
            "semantic_reconstruction": "nested_named_fields",
        },
    }
    return table, static


def _chunk_record(key: str, table: np.ndarray) -> dict[str, Any]:
    if table.ndim != 1 or not table.dtype.names or table.dtype.hasobject:
        raise TypeError("tracking chunk must be one object-free structured array")
    return {
        "key": key,
        "start": int(table["frame"][0]),
        "count": int(table.shape[0]),
        "first_tick": int(table["control_tick"][0]),
        "last_tick": int(table["control_tick"][-1]),
        "record_bytes": int(table.nbytes),
        "record_sha256": hashlib.sha256(table.tobytes(order="C")).hexdigest(),
    }


def _records_digest(dtype: np.dtype[Any], tables: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(_json_bytes(_dtype_description(dtype)))
    for table in tables:
        digest.update(table.tobytes(order="C"))
    return digest.hexdigest()


def _archive_index(
    tables: Sequence[np.ndarray],
    static: Mapping[str, Any],
) -> dict[str, Any]:
    if not tables:
        raise ValueError("tracking archive needs at least one chunk")
    dtype = tables[0].dtype
    if any(table.dtype != dtype for table in tables):
        raise ValueError("tracking chunk dtype changed inside one archive")
    chunks = [
        _chunk_record(f"{_CHUNK_PREFIX}{index:06d}", table)
        for index, table in enumerate(tables)
    ]
    return {
        "schema": TRACKING_ARCHIVE_SCHEMA,
        "semantic_schema": TRACKING_SCHEMA,
        "record_layout": "one_structured_npy_per_chunk",
        "record_dtype": _dtype_description(dtype),
        "allow_pickle": False,
        "compression": "zip_deflate",
        "compression_level": TRACKING_COMPRESSION_LEVEL,
        "frame_rows": sum(int(table.shape[0]) for table in tables),
        "record_bytes": sum(int(table.nbytes) for table in tables),
        "records_sha256": _records_digest(dtype, tables),
        "static": to_jsonable(static),
        "chunks": chunks,
    }


def _write_archive(
    path: Path, tables: Sequence[np.ndarray], static: Mapping[str, Any]
) -> dict[str, Any]:
    index = _archive_index(tables, static)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=TRACKING_COMPRESSION_LEVEL,
        allowZip64=True,
    ) as archive:
        for item, table in zip(index["chunks"], tables, strict=True):
            _write_array_member(archive, str(item["key"]), table)
        _write_array_member(
            archive,
            _INDEX_KEY,
            np.frombuffer(_json_bytes(index), dtype=np.uint8),
        )
    return index


def write_tracking_archive(
    path: str | Path,
    frames: Sequence[HostFrame],
    common_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write one independently readable bounded frame chunk."""

    table, static = build_tracking_table(frames, common_rows)
    return _write_archive(Path(path), (table,), static)


class NpzTrackingReader:
    """Streaming chunk and indexed-row reader over canonical tracking NPZ."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._archive = np.load(self.path, allow_pickle=False)
        if _INDEX_KEY not in self._archive.files:
            self.close()
            raise ValueError("tracking NPZ has no canonical index")
        raw_index = np.asarray(self._archive[_INDEX_KEY])
        if raw_index.dtype != np.uint8 or raw_index.ndim != 1:
            self.close()
            raise TypeError("tracking NPZ index must be one uint8 vector")
        self.index = json.loads(raw_index.tobytes().decode("utf-8"))
        self._validate_index()
        self._cached_chunk = -1
        self._cached_table: np.ndarray | None = None

    def _validate_index(self) -> None:
        index = self.index
        if index.get("schema") != TRACKING_ARCHIVE_SCHEMA:
            raise ValueError("unsupported tracking NPZ schema")
        if index.get("semantic_schema") != TRACKING_SCHEMA:
            raise ValueError("unsupported semantic tracking schema")
        if index.get("allow_pickle") is not False:
            raise ValueError("tracking NPZ must prohibit pickle")
        chunks = index.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("tracking NPZ has no chunk inventory")
        expected_keys = {_INDEX_KEY}
        offset = int(chunks[0].get("start", -1))
        total = 0
        previous_tick: int | None = None
        for number, chunk in enumerate(chunks):
            key = f"{_CHUNK_PREFIX}{number:06d}"
            expected_keys.add(key)
            if chunk.get("key") != key:
                raise ValueError("tracking NPZ chunk key order changed")
            count = chunk.get("count")
            if type(count) is not int or count < 1 or chunk.get("start") != offset:
                raise ValueError("tracking NPZ chunk frame ranges are invalid")
            first_tick = chunk.get("first_tick")
            last_tick = chunk.get("last_tick")
            if type(first_tick) is not int or type(last_tick) is not int:
                raise ValueError("tracking NPZ chunk ticks are invalid")
            if previous_tick is not None and first_tick != previous_tick + 1:
                raise ValueError("tracking NPZ chunk ticks are not contiguous")
            if last_tick - first_tick + 1 != count:
                raise ValueError("tracking NPZ chunk tick span disagrees with count")
            previous_tick = last_tick
            offset += count
            total += count
        if set(self._archive.files) != expected_keys:
            raise ValueError("tracking NPZ members disagree with its index")
        if index.get("frame_rows") != total:
            raise ValueError("tracking NPZ frame count disagrees with its chunks")

    def close(self) -> None:
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()

    def __enter__(self) -> NpzTrackingReader:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.index["frame_rows"])

    @property
    def chunk_count(self) -> int:
        return len(self.index["chunks"])

    def _load_chunk(self, number: int, *, verify: bool) -> np.ndarray:
        if number < 0 or number >= self.chunk_count:
            raise IndexError("tracking chunk index is out of range")
        if self._cached_chunk != number:
            table = np.asarray(self._archive[f"{_CHUNK_PREFIX}{number:06d}"])
            if table.ndim != 1 or not table.dtype.names or table.dtype.hasobject:
                raise TypeError(
                    "tracking member is not an object-free structured array"
                )
            expected_dtype = self.index.get("record_dtype")
            if _dtype_description(table.dtype) != expected_dtype:
                raise ValueError("tracking chunk dtype disagrees with archive index")
            descriptor = self.index["chunks"][number]
            if table.shape != (descriptor["count"],):
                raise ValueError("tracking chunk shape disagrees with archive index")
            if table.nbytes != descriptor["record_bytes"]:
                raise ValueError(
                    "tracking chunk byte count disagrees with archive index"
                )
            if int(table["frame"][0]) != descriptor["start"]:
                raise ValueError(
                    "tracking chunk first frame disagrees with archive index"
                )
            if int(table["control_tick"][0]) != descriptor["first_tick"]:
                raise ValueError(
                    "tracking chunk first tick disagrees with archive index"
                )
            if int(table["control_tick"][-1]) != descriptor["last_tick"]:
                raise ValueError(
                    "tracking chunk last tick disagrees with archive index"
                )
            self._cached_chunk = number
            self._cached_table = table
        assert self._cached_table is not None
        if verify:
            observed = hashlib.sha256(self._cached_table.tobytes(order="C")).hexdigest()
            if observed != self.index["chunks"][number]["record_sha256"]:
                raise ValueError("tracking chunk content digest mismatch")
        return self._cached_table

    def iter_chunks(self, *, verify: bool = False) -> Iterator[np.ndarray]:
        for number in range(self.chunk_count):
            yield self._load_chunk(number, verify=verify)

    def iter_rows(self, *, verify: bool = False) -> Iterator[dict[str, Any]]:
        for table in self.iter_chunks(verify=verify):
            for record in table:
                yield _semantic_row(record, self.index["static"])

    def row(self, frame: int) -> dict[str, Any]:
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise TypeError("frame must be an integer")
        for number, chunk in enumerate(self.index["chunks"]):
            start = int(chunk["start"])
            count = int(chunk["count"])
            if start <= frame < start + count:
                table = self._load_chunk(number, verify=False)
                return _semantic_row(table[frame - start], self.index["static"])
        raise IndexError("tracking frame is outside the archive")

    def verify_records_sha256(self) -> str:
        digest = hashlib.sha256(_json_bytes(self.index["record_dtype"]))
        for table in self.iter_chunks(verify=True):
            digest.update(table.tobytes(order="C"))
        observed = digest.hexdigest()
        if observed != self.index.get("records_sha256"):
            raise ValueError("tracking archive record digest mismatch")
        return observed


class JsonlTrackingReader:
    """Reader for plain and concatenated-gzip JSONL tracking streams."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _stream(self) -> Iterator[BinaryIO]:
        with open_tracking_jsonl(self.path) as stream:
            yield stream

    def close(self) -> None:
        return None

    def __enter__(self) -> JsonlTrackingReader:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_rows(self, *, verify: bool = False) -> Iterator[dict[str, Any]]:
        del verify
        with self._stream() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def row(self, frame: int) -> dict[str, Any]:
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise IndexError("tracking frame must be a non-negative integer")
        for row in self.iter_rows():
            if int(row.get("frame", -1)) == frame:
                return row
        raise IndexError("tracking frame is outside the JSONL stream")


def _restore_observation(
    record: np.void,
    observation: Mapping[str, Any],
) -> Any:
    if not observation.get("present", False):
        return None
    root: dict[str, Any] = {}
    root_leaf: Any = None
    for leaf in observation.get("leaves", []):
        path = [str(item) for item in leaf["path"]]
        value = to_jsonable(record[str(leaf["field"])])
        if not path:
            root_leaf = value
            continue
        cursor = root
        for name in path[:-1]:
            cursor = cursor.setdefault(name, {})
        cursor[path[-1]] = value
    return root_leaf if root_leaf is not None and not root else root


def _semantic_row(record: np.void, static: Mapping[str, Any]) -> dict[str, Any]:
    player_count = int(static["player_count"])
    slots = static["player_slot"]
    players = []
    for player in range(player_count):
        row = {
            "slot": int(slots[player]),
            "player_id": int(record["player_id"][player]),
        }
        for semantic, field in _PLAYER_INT_FIELDS:
            row[semantic] = int(record[field][player])
        row["goalkeeper"] = bool(record["player_goalkeeper"][player])
        row["active"] = bool(record["player_active"][player])
        row["on_pitch"] = bool(record["player_on_pitch"][player])
        row["sent_off"] = bool(record["player_sent_off"][player])
        for semantic, field in _PLAYER_VECTOR_FIELDS:
            row[semantic] = record[field][player].tolist()
        for semantic, field in _PLAYER_FLOAT_FIELDS:
            row[semantic] = float(record[field][player])
        row["offside"] = bool(record["player_offside"][player])
        players.append(row)

    if bool(record["last_contact_known"]):
        last_contact: dict[str, Any] | None = {
            name: int(record[f"last_contact_{name}"])
            for name in _LAST_CONTACT_INT_FIELDS[:-1]
        }
        last_contact["kick_applied"] = bool(record["last_contact_kick_applied"])
        last_contact["intent_source"] = int(record["last_contact_intent_source"])
    else:
        last_contact = None
    return {
        "schema": TRACKING_SCHEMA,
        "frame": int(record["frame"]),
        "control_tick": int(record["control_tick"]),
        "clock_s": float(record["clock_s"]),
        "period": int(record["period"]),
        "display_clock_s": float(record["display_clock_s"]),
        "added_time_s": float(record["added_time_s"]),
        "dead_ball_s": float(record["dead_ball_s"]),
        "first_half_live_extension_s": float(record["first_half_live_extension_s"]),
        "ball": {
            "position": record["ball_position"].tolist(),
            "velocity": record["ball_velocity"].tolist(),
            "spin": record["ball_spin"].tolist(),
            "live": bool(record["ball_live"]),
        },
        "players": players,
        "score": record["score"].tolist(),
        "attack_direction": record["attack_direction"].tolist(),
        "possession": {
            "team": int(record["possession_team"]),
            "player": int(record["possession_player"]),
            "previous_team": int(record["possession_previous_team"]),
            "control_ticks": int(record["possession_control_ticks"]),
            "last_contact": last_contact,
        },
        "restart": {
            "kind": int(record["restart_kind"]),
            "team": int(record["restart_team"]),
            "substeps_remaining": int(record["restart_substeps_remaining"]),
            "taker": int(record["restart_taker"]),
            "indirect": bool(record["restart_indirect"]),
        },
        "observation": _restore_observation(record, static.get("observation", {})),
    }


@contextmanager
def open_tracking(path: str | Path) -> Iterator[Any]:
    """Open NPZ or JSONL tracking through one semantic API."""

    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(4)
    reader: Any
    if magic.startswith(b"PK\x03\x04"):
        reader = NpzTrackingReader(path)
    else:
        reader = JsonlTrackingReader(path)
    try:
        yield reader
    finally:
        reader.close()


@contextmanager
def open_tracking_jsonl(path: str | Path) -> Iterator[BinaryIO]:
    """Open a plain or concatenated-gzip JSONL byte stream."""

    path = Path(path)
    with path.open("rb") as raw:
        magic = raw.read(4)
        raw.seek(0)
        if magic.startswith(b"PK\x03\x04"):
            raise ValueError(
                "canonical NPZ is not a JSONL byte stream; use open_tracking() "
                "or export_tracking_jsonl()"
            )
        if magic[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                yield stream
            return
        if path.name.endswith(".gz"):
            raise ValueError(f"gzip tracking file has an invalid header: {path}")
        yield raw


def export_tracking_jsonl(
    source: str | Path,
    destination: str | Path,
    *,
    compression_level: int = 2,
) -> Path:
    """Export semantic rows as JSONL without retaining duplicate storage."""

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("tracking export source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        if destination.name.endswith(".gz"):
            writer: Any = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compression_level,
                fileobj=target,
                mtime=0,
            )
        else:
            writer = target
        try:
            with open_tracking(source) as reader:
                for row in reader.iter_rows():
                    writer.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
        finally:
            if writer is not target:
                writer.close()
    return destination


def merge_tracking_archives(
    sources: Sequence[str | Path],
    destination: str | Path,
) -> dict[str, Any]:
    """Merge verified chunk archives into one canonical NPZ with bounded memory."""

    if not sources:
        raise ValueError("cannot merge an empty tracking archive sequence")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    static: dict[str, Any] | None = None
    dtype: np.dtype[Any] | None = None
    # Only references to one loaded table are retained.  Tables are written and
    # released inside the loop; the final index is assembled from descriptors.
    descriptors: list[dict[str, Any]] = []
    record_digest: Any | None = None
    record_bytes = 0
    frame_rows = 0
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=TRACKING_COMPRESSION_LEVEL,
        allowZip64=True,
    ) as archive:
        for source in sources:
            with NpzTrackingReader(source) as reader:
                source_static = reader.index["static"]
                if static is None:
                    static = source_static
                elif source_static != static:
                    raise ValueError(
                        "tracking static schema changed across spool chunks"
                    )
                for table in reader.iter_chunks(verify=True):
                    if dtype is None:
                        dtype = table.dtype
                        record_digest = hashlib.sha256(
                            _json_bytes(_dtype_description(dtype))
                        )
                    elif table.dtype != dtype:
                        raise ValueError("tracking dtype changed across spool chunks")
                    assert record_digest is not None
                    key = f"{_CHUNK_PREFIX}{len(descriptors):06d}"
                    descriptor = _chunk_record(key, table)
                    if descriptor["start"] != frame_rows:
                        raise ValueError(
                            "tracking spool frame offsets are not contiguous"
                        )
                    if (
                        descriptors
                        and descriptor["first_tick"] != descriptors[-1]["last_tick"] + 1
                    ):
                        raise ValueError(
                            "tracking spool control ticks are not contiguous"
                        )
                    _write_array_member(archive, key, table)
                    record_digest.update(table.tobytes(order="C"))
                    record_bytes += int(table.nbytes)
                    frame_rows += int(table.shape[0])
                    descriptors.append(descriptor)
        assert static is not None and dtype is not None and record_digest is not None
        index = {
            "schema": TRACKING_ARCHIVE_SCHEMA,
            "semantic_schema": TRACKING_SCHEMA,
            "record_layout": "one_structured_npy_per_chunk",
            "record_dtype": _dtype_description(dtype),
            "allow_pickle": False,
            "compression": "zip_deflate",
            "compression_level": TRACKING_COMPRESSION_LEVEL,
            "frame_rows": frame_rows,
            "record_bytes": record_bytes,
            "records_sha256": record_digest.hexdigest(),
            "static": static,
            "chunks": descriptors,
        }
        _write_array_member(
            archive,
            _INDEX_KEY,
            np.frombuffer(_json_bytes(index), dtype=np.uint8),
        )
    return index


def tracking_storage_receipt(
    path: str | Path,
    index: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe exact canonical content, archive bytes, and access semantics."""

    path = Path(path)
    chunks = index.get("chunks", [])
    return {
        "schema": TRACKING_STORAGE_SCHEMA,
        "filename": path.name,
        "content_format": "numpy_structured_chunks",
        "semantic_schema": TRACKING_SCHEMA,
        "container": "npz_zip",
        "compression": "zip_deflate",
        "compression_level": TRACKING_COMPRESSION_LEVEL,
        "archive_members": len(chunks) + 1,
        "chunk_count": len(chunks),
        "chunk_semantics": "independent_structured_npy_members",
        "frame_rows": int(index["frame_rows"]),
        "record_bytes": int(index["record_bytes"]),
        "records_sha256": str(index["records_sha256"]),
        "compressed_bytes": path.stat().st_size,
        "archive_sha256": _sha256_path(path),
        "lossless": True,
        "allow_pickle": False,
        "streaming_decode": True,
        "random_access": "indexed_chunk_then_row",
    }


__all__ = [
    "TRACKING_ARCHIVE_SCHEMA",
    "TRACKING_COMPRESSION_LEVEL",
    "TRACKING_FILENAME",
    "TRACKING_SCHEMA",
    "TRACKING_STORAGE_SCHEMA",
    "JsonlTrackingReader",
    "NpzTrackingReader",
    "build_tracking_table",
    "export_tracking_jsonl",
    "merge_tracking_archives",
    "open_tracking",
    "open_tracking_jsonl",
    "tracking_storage_receipt",
    "write_tracking_archive",
]
