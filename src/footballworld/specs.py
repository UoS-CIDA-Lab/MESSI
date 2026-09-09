"""Host-side schemas and lossless typed PyTree adapters.

Structured FootballWorld values remain authoritative.  This module exposes
their layout without putting metadata in a JAX transition and flattens values
into separate float32, int32, and bool blocks so discrete facts are never
silently converted to floating point.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.core.action import ACTION_SCHEMA, ACTION_SCHEMA_VERSION, IntentAction
from footballworld.dynamics.action import (
    ACTION_RECEIPT_SCHEMA,
    ACTION_TRACE_SCHEMA,
    ActionReceipt,
    ActionTrace,
)
from footballworld.environment.normalization import (
    MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION,
    MODEL_OBSERVATION_SCHEMA_VERSION,
    MODEL_STATE_SCHEMA_VERSION,
    NormalizationContext,
    NormalizedGlobalRollout,
    NormalizedManagerObservation,
    NormalizedObservation,
)
from footballworld.environment.transition import FrameEvents

TYPED_LAYOUT_SCHEMA = "footballworld.typed-layout/1"
FRAME_EVENTS_SCHEMA = "footballworld.frame-events/1"


class FlatTree(NamedTuple):
    """Three exact-dtype flat blocks with arbitrary shared leading axes."""

    float32: jax.Array
    int32: jax.Array
    boolean: jax.Array


@dataclass(frozen=True, slots=True)
class LeafSpec:
    """Machine-readable meaning and position of one structured array leaf."""

    path: str
    shape: tuple[int | str, ...]
    trailing_shape: tuple[int, ...]
    dtype: str
    block: str
    offset: int
    size: int
    normalized_range: tuple[float | None, float | None] | None
    si_source_unit: str
    normalization_scale_source: str
    normalization_scale: tuple[float, ...] | None
    visibility_rule: str
    masking_rule: str
    semantic_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "trailing_shape": list(self.trailing_shape),
            "dtype": self.dtype,
            "block": self.block,
            "offset": self.offset,
            "size": self.size,
            "normalized_range": (
                None if self.normalized_range is None else list(self.normalized_range)
            ),
            "si_source_unit": self.si_source_unit,
            "normalization_scale_source": self.normalization_scale_source,
            "normalization_scale": (
                None
                if self.normalization_scale is None
                else list(self.normalization_scale)
            ),
            "visibility_rule": self.visibility_rule,
            "masking_rule": self.masking_rule,
            "semantic_version": self.semantic_version,
        }


@dataclass(frozen=True, slots=True)
class TypedLayout:
    """Immutable, schema-fingerprinted recipe for :class:`FlatTree`."""

    contract: str
    semantic_version: int
    root_type: str
    leaves: tuple[LeafSpec, ...]
    float32_size: int
    int32_size: int
    boolean_size: int
    fingerprint: str
    _treedef: Any = field(repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout_schema": TYPED_LAYOUT_SCHEMA,
            "contract": self.contract,
            "semantic_version": self.semantic_version,
            "root_type": self.root_type,
            "block_sizes": {
                "float32": self.float32_size,
                "int32": self.int32_size,
                "boolean": self.boolean_size,
            },
            "leaves": [leaf.as_dict() for leaf in self.leaves],
            "fingerprint": self.fingerprint,
        }


TreeSpec = TypedLayout


def schema_versions() -> MappingProxyType:
    """Return immutable names for every host spec contract."""

    return MappingProxyType(
        {
            "typed_layout": TYPED_LAYOUT_SCHEMA,
            "action": ACTION_SCHEMA,
            "action_trace": ACTION_TRACE_SCHEMA,
            "action_receipt": ACTION_RECEIPT_SCHEMA,
            "player_observation": (
                f"footballworld.player-observation/{MODEL_OBSERVATION_SCHEMA_VERSION}"
            ),
            "manager_observation": (
                f"footballworld.manager-observation/"
                f"{MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION}"
            ),
            "global_state": (
                f"footballworld.global-state/{MODEL_STATE_SCHEMA_VERSION}"
            ),
            "frame_events": FRAME_EVENTS_SCHEMA,
        }
    )


def _root_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _path_name(path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in path:
        if hasattr(key, "name"):
            piece = str(key.name)
            parts.append(piece if not parts else f".{piece}")
        elif hasattr(key, "idx"):
            parts.append(f"[{int(key.idx)}]")
        elif hasattr(key, "key"):
            raw = key.key
            if isinstance(raw, str) and raw.isidentifier():
                parts.append(raw if not parts else f".{raw}")
            elif isinstance(raw, (str, int)) and not isinstance(raw, bool):
                parts.append(f"[{raw!r}]")
            else:
                raise TypeError(f"unsupported stable PyTree key {raw!r}")
        else:
            raise TypeError(f"unsupported PyTree path key {key!r}")
    return "".join(parts) or "$"


def _common_leading_shape(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    if not shapes:
        raise ValueError("tree must contain at least one array leaf")
    prefix = list(shapes[0])
    for shape in shapes[1:]:
        limit = min(len(prefix), len(shape))
        common = 0
        while common < limit and prefix[common] == shape[common]:
            common += 1
        del prefix[common:]
    return tuple(prefix)


def _dtype_block(value: Any, path: str) -> tuple[str, np.dtype[Any]]:
    dtype = np.dtype(jnp.asarray(value).dtype)
    if dtype == np.dtype(np.float32):
        return "float32", dtype
    if dtype == np.dtype(np.int32):
        return "int32", dtype
    # Eventful receipts use compact exact-width integer leaves.  Preserve the
    # established three-block FlatTree wire API by carrying them in its int32
    # block; _unflatten restores the declared source dtype exactly.  uint32 is
    # bit-cast rather than numerically cast, so all 32 flag bits are lossless.
    if dtype in {
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
    }:
        return "int32", dtype
    if dtype == np.dtype(np.bool_):
        return "boolean", dtype
    raise TypeError(
        f"{path} has unsupported dtype {dtype}; lossless layouts accept float32, "
        "int32, int16, uint16, uint32, and bool"
    )


def _default_range(block: str) -> tuple[float | None, float | None] | None:
    if block == "boolean":
        return (0.0, 1.0)
    return (None, None)


def _scale(
    source: str,
    values: tuple[float, ...] | None,
    unit: str,
    bounds: tuple[float | None, float | None] | None = (None, None),
) -> tuple[
    str,
    tuple[float, ...] | None,
    str,
    tuple[float | None, float | None] | None,
]:
    return source, values, unit, bounds


def _clock_scale(path: str, context: NormalizationContext):
    if path.endswith(".period_regulation_progress"):
        durations = (
            (context.halftime_tick, context.fulltime_tick - context.halftime_tick)
            if context.halftime_enabled
            else (context.fulltime_tick,)
        )
        return _scale(
            "configured period duration ticks", durations, "control tick", (0.0, 1.0)
        )
    if path.endswith(".match_regulation_progress"):
        return _scale(
            "configured fulltime_tick",
            (context.fulltime_tick,),
            "control tick",
            (0.0, 1.0),
        )
    if path.endswith(
        (
            ".dead_ball_accrued_fraction",
            ".added_time_elapsed_fraction",
            ".added_time_remaining_fraction",
        )
    ):
        return _scale(
            "configured current-period duration ticks",
            None,
            "control tick",
            (0.0, None),
        )
    if path.endswith(
        (
            ".period_dead_ball_counter",
            ".added_time_elapsed_counter",
            ".added_time_remaining_counter",
        )
    ):
        return _scale(
            "counter_scale_ticks",
            (context.counter_scale_ticks,),
            "control tick",
            (0.0, 1.0),
        )
    return None


def _ability_scale(path: str, context: NormalizationContext):
    table = {
        "max_speed": (
            context.min_player_speed_mps,
            context.max_player_speed_mps,
            "m/s",
        ),
        "height": (context.min_height_m, context.max_height_m, "m"),
        "reach_height": (context.min_reach_height_m, context.max_reach_height_m, "m"),
        "ball_control": (
            context.min_ball_control,
            context.max_ball_control,
            "dimensionless",
        ),
        "endurance_factor": (
            context.min_endurance_factor,
            context.max_endurance_factor,
            "dimensionless",
        ),
    }
    field_name = path.rsplit(".", 1)[-1]
    if field_name not in table:
        return None
    lower, upper, unit = table[field_name]
    return _scale(
        f"affine configured {field_name} bounds",
        (lower, upper),
        unit,
        (0.0, 1.0),
    )


def _normalized_scale(
    contract: str,
    path: str,
    context: NormalizationContext | None,
):
    if context is None:
        return _scale("identity or unavailable without context", None, "dimensionless")
    clock = _clock_scale(path, context)
    if clock is not None:
        return clock
    ability = _ability_scale(path, context)
    if ability is not None and contract != "player_observation":
        return ability
    if path.endswith(
        (
            "self_state.position",
            "players.position",
            "on_field.position",
            "restart_position",
            "formation_anchor",
            "team_centroid",
            "team_spread",
        )
    ):
        return _scale(
            "configured pitch half-extents",
            (context.position_scale_x_m, context.position_scale_y_m),
            "m",
        )
    if path.endswith("players.relative_position"):
        return _scale(
            "configured full pitch extents",
            (context.relative_position_scale_x_m, context.relative_position_scale_y_m),
            "m",
        )
    if path.endswith(("self_state.velocity", "players.velocity", "on_field.velocity")):
        return _scale(
            "configured maximum player speed", (context.player_speed_scale_mps,), "m/s"
        )
    if path.endswith("players.relative_velocity"):
        return _scale(
            "twice configured maximum player speed",
            (context.player_relative_speed_scale_mps,),
            "m/s",
        )
    if path.endswith("gaze_yaw"):
        return _scale(
            "configured gaze yaw limit",
            (context.gaze_yaw_limit_radians,),
            "rad",
            (-1.0, 1.0),
        )
    if path.endswith("ball.relative_state"):
        return _scale(
            "configured relative ball position/energy envelope",
            (
                context.ball_relative_position_scale_x_m,
                context.ball_relative_position_scale_y_m,
                context.ball_height_scale_m,
                context.ball_relative_speed_scale_mps,
                context.ball_relative_speed_scale_mps,
                context.ball_relative_speed_scale_mps,
                context.ball_spin_scale_radps,
                context.ball_spin_scale_radps,
                context.ball_spin_scale_radps,
            ),
            "mixed: m,m,m,m/s,m/s,m/s,rad/s,rad/s,rad/s",
        )
    if path.endswith("ball.position"):
        return _scale(
            "configured ball position/energy envelope",
            (
                context.ball_position_scale_x_m,
                context.ball_position_scale_y_m,
                context.ball_height_scale_m,
            ),
            "m",
        )
    if path.endswith("ball.velocity"):
        return _scale(
            "configured ball energy envelope", (context.ball_speed_scale_mps,), "m/s"
        )
    if path.endswith("ball.spin"):
        return _scale(
            "configured maximum spin", (context.ball_spin_scale_radps,), "rad/s"
        )
    lock_scales = {
        "challenge_recovery_substeps": context.challenge_lock_substeps,
        "contact_lock_substeps": context.contact_lock_substeps,
        "aerial_recovery_substeps": context.aerial_lock_substeps,
        "possession_loss_lock_substeps": context.possession_loss_lock_substeps,
    }
    final = path.rsplit(".", 1)[-1]
    if final in lock_scales:
        return _scale(
            f"configured {final} maximum",
            (lock_scales[final],),
            "physics substep",
            (0.0, 1.0),
        )
    if path.endswith("restart.substeps_remaining"):
        return _scale(
            "restart-kind conditional configured delay",
            (1.0, context.restart_delay_substeps, context.goalkeeper_hold_substeps),
            "physics substep",
            (0.0, 1.0),
        )
    if path.endswith(
        (
            "control_tick",
            "control_ticks",
            "opened_control_tick",
            "dead_ball_control_ticks",
            "first_half_wall_end_tick",
            "first_half_live_extension_ticks",
        )
    ):
        return _scale(
            "counter_scale_ticks", (context.counter_scale_ticks,), "control tick"
        )
    if path.endswith("formation_candidate_attack_depth"):
        return _scale(
            "configured pitch half-length", (context.position_scale_x_m,), "m"
        )
    if path.endswith("formation_candidate_width"):
        return _scale("configured pitch half-width", (context.position_scale_y_m,), "m")
    if path.endswith(
        (
            "stamina_long",
            "stamina_short",
            "formation_candidate_probability",
            "formation_candidate_defender_fraction",
        )
    ):
        return _scale("identity", None, "dimensionless", (0.0, 1.0))
    if path.endswith(("body_forward", "facing_sin", "facing_cos", "attack_direction")):
        return _scale("identity", None, "dimensionless", (-1.0, 1.0))
    return _scale("identity", None, "dimensionless")


def _visibility(contract: str, path: str) -> tuple[str, str]:
    if contract == "player_observation":
        if path == "valid":
            return "observer index validity", "false marks the entire row invalid"
        if path.startswith("players."):
            if path.endswith(
                ("on_pitch", "sent_off", "visible", "contact_may_occur_this_frame")
            ):
                return (
                    "public roster phase; visible is the FOV mask",
                    "invalid observer clears the value",
                )
            return (
                "players.visible under partial observation",
                "hidden slots use zero/false; actor identity uses flags",
            )
        if path.startswith("ball."):
            return (
                "ball.visible under partial observation",
                "hidden kinematics use zero; live stays public",
            )
        if path.startswith("possession.last_contact."):
            return (
                "possession.last_contact.known",
                "unknown contact facts use zero/false",
            )
        if path.startswith("possession."):
            return "possession.known", "unknown teams use NO_TEAM and counters use zero"
        if path.startswith("restart_release."):
            return "restart_release.known", "unknown provenance uses sentinels/false"
        if path == "match.gk_handling_restricted_team":
            return "match.gk_handling_restriction_known", "unknown team uses NO_TEAM"
        return (
            "globally public after observer validation",
            "invalid observer uses zero/sentinel",
        )
    if contract == "manager_observation":
        if path.startswith("team_"):
            return (
                "public active-player aggregate after manager validation",
                "invalid manager or absent team uses zero/false",
            )
        if path.startswith("on_field."):
            return (
                "manager valid and on_field.team_mask",
                "masked float fields use zero",
            )
        if path.startswith("bench."):
            return (
                "manager valid and bench.valid",
                "unavailable/padded ability fields use zero",
            )
        return (
            "private selected-team manager view",
            "invalid manager uses zero/sentinel",
        )
    if contract == "frame_events":
        return (
            "paired occurred fact in the enclosing event",
            "absent events retain fixed-shape sentinels",
        )
    if contract == "global_state":
        return (
            "global; never FOV-filtered",
            "explicit known flags distinguish sentinel channels",
        )
    if contract == "action_trace":
        return (
            "privileged eventful submitted-action telemetry",
            "terminal frames set executed false; never FOV-filtered",
        )
    if contract == "action_receipt":
        return (
            "privileged eventful causal telemetry",
            "fixed bits and sentinels describe absent or suppressed effects; never FOV-filtered",
        )
    return (
        "submitted action slot",
        "invalid categorical codes fail closed during decode",
    )


def _si_event_unit(path: str) -> str:
    if path.endswith("position"):
        return "m"
    if path.endswith("time_fraction"):
        return "physics-substep fraction"
    return "exact category/count/boolean"


def _leaf_semantics(
    contract: str,
    path: str,
    block: str,
    context: NormalizationContext | None,
):
    visibility, masking = _visibility(contract, path)
    if contract == "action":
        bounds = (0.0, 5.0) if path == "intent" else (-1.0, 1.0)
        unit = "intent category" if path == "intent" else "normalized control"
        return bounds, unit, "public action contract", None, visibility, masking
    if contract == "action_trace":
        if path == "requested_intent":
            return (
                (0.0, 5.0),
                "intent category",
                "eventful submitted-action trace",
                None,
                visibility,
                masking,
            )
        if path == "intent_source":
            return (
                (0.0, 2.0),
                "intent-source category",
                "eventful submitted-action trace",
                None,
                visibility,
                masking,
            )
        return (
            (0.0, 1.0),
            "execution boolean",
            "eventful submitted-action trace",
            None,
            visibility,
            masking,
        )
    if contract == "action_receipt":
        bounds_and_units = {
            "requested_intent": ((None, None), "submitted intent category"),
            "effective_intent": ((0.0, 5.0), "effective intent category"),
            "flags": ((0.0, None), "action-fact bit mask"),
            "eligibility_seen": ((0.0, None), "eligibility bit mask"),
            "primary_reason": ((0.0, 12.0), "summary reason category"),
            "parameter_consumed": ((0.0, None), "parameter-use bit mask"),
            "displacement_source": ((0.0, None), "attributed displacement bit mask"),
        }
        bounds, unit = bounds_and_units[path]
        return (
            bounds,
            unit,
            "eventful causal action receipt",
            None,
            visibility,
            masking,
        )
    if contract == "frame_events":
        return None, _si_event_unit(path), "identity", None, visibility, masking
    source, values, unit, bounds = _normalized_scale(contract, path, context)
    if block != "float32":
        bounds = _default_range(block)
        source = "identity"
        values = None
    return bounds, unit, source, values, visibility, masking


def _make_layout(
    value: Any,
    *,
    contract: str,
    semantic_version: int,
    context: NormalizationContext | None = None,
) -> TypedLayout:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    if not path_leaves:
        raise ValueError("tree must contain at least one array leaf")
    arrays = tuple(jnp.asarray(leaf) for _, leaf in path_leaves)
    shapes = tuple(tuple(int(axis) for axis in array.shape) for array in arrays)
    leading_shape = _common_leading_shape(shapes)
    offsets = {"float32": 0, "int32": 0, "boolean": 0}
    specs: list[LeafSpec] = []
    for (path_keys, _), array, shape in zip(path_leaves, arrays, shapes, strict=True):
        path = _path_name(path_keys)
        block, source_dtype = _dtype_block(array, path)
        trailing_shape = shape[len(leading_shape) :]
        size = math.prod(trailing_shape)
        offset = offsets[block]
        offsets[block] += size
        bounds, unit, source, scale_values, visibility, masking = _leaf_semantics(
            contract, path, block, context
        )
        specs.append(
            LeafSpec(
                path=path,
                shape=("...", *trailing_shape),
                trailing_shape=trailing_shape,
                dtype=str(source_dtype),
                block=block,
                offset=offset,
                size=size,
                normalized_range=bounds,
                si_source_unit=unit,
                normalization_scale_source=source,
                normalization_scale=scale_values,
                visibility_rule=visibility,
                masking_rule=masking,
                semantic_version=semantic_version,
            )
        )
    payload = {
        "layout_schema": TYPED_LAYOUT_SCHEMA,
        "contract": contract,
        "semantic_version": semantic_version,
        "root_type": _root_name(value),
        "block_sizes": offsets,
        "leaves": [spec.as_dict() for spec in specs],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TypedLayout(
        contract=contract,
        semantic_version=semantic_version,
        root_type=_root_name(value),
        leaves=tuple(specs),
        float32_size=offsets["float32"],
        int32_size=offsets["int32"],
        boolean_size=offsets["boolean"],
        fingerprint=fingerprint,
        _treedef=treedef,
    )


def _flatten(value: Any, layout: TypedLayout) -> FlatTree:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    if treedef != layout._treedef or _root_name(value) != layout.root_type:
        raise TypeError("value PyTree structure does not match the typed layout")
    arrays = tuple(jnp.asarray(leaf) for _, leaf in path_leaves)
    leading_shape: tuple[int, ...] | None = None
    blocks: dict[str, list[jax.Array]] = {
        "float32": [],
        "int32": [],
        "boolean": [],
    }
    for (path_keys, _), array, spec in zip(
        path_leaves, arrays, layout.leaves, strict=True
    ):
        path = _path_name(path_keys)
        block, source_dtype = _dtype_block(array, path)
        if path != spec.path or block != spec.block or str(source_dtype) != spec.dtype:
            raise TypeError(f"{path} does not match layout leaf {spec.path}")
        trailing_rank = len(spec.trailing_shape)
        if trailing_rank:
            if tuple(array.shape[-trailing_rank:]) != spec.trailing_shape:
                raise ValueError(
                    f"{path} trailing shape must be {spec.trailing_shape}, got {array.shape}"
                )
            prefix = tuple(array.shape[:-trailing_rank])
        else:
            prefix = tuple(array.shape)
        if leading_shape is None:
            leading_shape = prefix
        elif prefix != leading_shape:
            raise ValueError(
                f"{path} leading shape {prefix} differs from {leading_shape}"
            )
        encoded = array.reshape((*prefix, spec.size))
        if source_dtype == np.dtype(np.uint32):
            encoded = jax.lax.bitcast_convert_type(encoded, jnp.int32)
        elif block == "int32" and source_dtype != np.dtype(np.int32):
            encoded = encoded.astype(jnp.int32)
        blocks[block].append(encoded)
    assert leading_shape is not None

    def join(block: str, dtype) -> jax.Array:
        values = blocks[block]
        if values:
            return jnp.concatenate(values, axis=-1)
        return jnp.empty((*leading_shape, 0), dtype=dtype)

    return FlatTree(
        float32=join("float32", jnp.float32),
        int32=join("int32", jnp.int32),
        boolean=join("boolean", jnp.bool_),
    )


def _unflatten(flat: FlatTree, layout: TypedLayout) -> Any:
    if not isinstance(flat, FlatTree):
        raise TypeError("flat must be FlatTree")
    blocks = {
        "float32": jnp.asarray(flat.float32),
        "int32": jnp.asarray(flat.int32),
        "boolean": jnp.asarray(flat.boolean),
    }
    expected = {
        "float32": (np.dtype(np.float32), layout.float32_size),
        "int32": (np.dtype(np.int32), layout.int32_size),
        "boolean": (np.dtype(np.bool_), layout.boolean_size),
    }
    leading_shape: tuple[int, ...] | None = None
    for block, array in blocks.items():
        dtype, size = expected[block]
        if np.dtype(array.dtype) != dtype:
            raise TypeError(f"{block} block must have dtype {dtype}, got {array.dtype}")
        if array.ndim < 1 or array.shape[-1] != size:
            raise ValueError(
                f"{block} block trailing size must be {size}, got {array.shape}"
            )
        prefix = tuple(array.shape[:-1])
        if leading_shape is None:
            leading_shape = prefix
        elif prefix != leading_shape:
            raise ValueError("all flat blocks must share identical leading axes")
    assert leading_shape is not None
    leaves = []
    for spec in layout.leaves:
        array = blocks[spec.block]
        leaf = array[..., spec.offset : spec.offset + spec.size]
        source_dtype = np.dtype(spec.dtype)
        if source_dtype == np.dtype(np.uint32):
            leaf = jax.lax.bitcast_convert_type(leaf, jnp.uint32)
        elif np.dtype(leaf.dtype) != source_dtype:
            leaf = leaf.astype(source_dtype)
        leaves.append(leaf.reshape((*leading_shape, *spec.trailing_shape)))
    return jax.tree_util.tree_unflatten(layout._treedef, leaves)


def action_spec(value: IntentAction) -> TreeSpec:
    if not isinstance(value, IntentAction):
        raise TypeError("value must be IntentAction")
    return _make_layout(
        value,
        contract="action",
        semantic_version=ACTION_SCHEMA_VERSION,
    )


def action_trace_spec(value: ActionTrace) -> TreeSpec:
    """Describe one eventful submitted-action trace."""

    if not isinstance(value, ActionTrace):
        raise TypeError("value must be ActionTrace")
    return _make_layout(value, contract="action_trace", semantic_version=1)


def action_receipt_spec(value: ActionReceipt) -> TreeSpec:
    """Describe one eventful causal action receipt."""

    if not isinstance(value, ActionReceipt):
        raise TypeError("value must be ActionReceipt")
    return _make_layout(value, contract="action_receipt", semantic_version=1)


def player_observation_spec(
    value: NormalizedObservation,
    context: NormalizationContext,
) -> TreeSpec:
    if not isinstance(value, NormalizedObservation):
        raise TypeError("value must be NormalizedObservation")
    return _make_layout(
        value,
        contract="player_observation",
        semantic_version=MODEL_OBSERVATION_SCHEMA_VERSION,
        context=context,
    )


def manager_observation_spec(
    value: NormalizedManagerObservation,
    context: NormalizationContext,
) -> TreeSpec:
    if not isinstance(value, NormalizedManagerObservation):
        raise TypeError("value must be NormalizedManagerObservation")
    return _make_layout(
        value,
        contract="manager_observation",
        semantic_version=MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION,
        context=context,
    )


def global_state_spec(
    value: NormalizedGlobalRollout,
    context: NormalizationContext,
) -> TreeSpec:
    if not isinstance(value, NormalizedGlobalRollout):
        raise TypeError("value must be NormalizedGlobalRollout")
    return _make_layout(
        value,
        contract="global_state",
        semantic_version=MODEL_STATE_SCHEMA_VERSION,
        context=context,
    )


def event_spec(value: FrameEvents) -> TreeSpec:
    if not isinstance(value, FrameEvents):
        raise TypeError("value must be FrameEvents")
    return _make_layout(value, contract="frame_events", semantic_version=1)


def flatten_action(value: IntentAction) -> tuple[FlatTree, TypedLayout]:
    layout = action_spec(value)
    return _flatten(value, layout), layout


def flatten_action_trace(value: ActionTrace) -> tuple[FlatTree, TypedLayout]:
    layout = action_trace_spec(value)
    return _flatten(value, layout), layout


def flatten_action_receipt(value: ActionReceipt) -> tuple[FlatTree, TypedLayout]:
    layout = action_receipt_spec(value)
    return _flatten(value, layout), layout


def _normalized_flatten_layout(
    value: Any,
    *,
    contract: str,
    semantic_version: int,
    context: NormalizationContext | None,
    layout: TypedLayout | None,
) -> TypedLayout:
    if context is None and layout is None:
        raise TypeError(
            "normalized flattening requires either the environment's "
            "NormalizationContext or a precomputed authoritative layout"
        )
    if context is not None and layout is not None:
        raise TypeError("pass context or layout, not both")
    if context is not None:
        if not isinstance(context, NormalizationContext):
            raise TypeError("context must be NormalizationContext")
        return _make_layout(
            value,
            contract=contract,
            semantic_version=semantic_version,
            context=context,
        )
    if not isinstance(layout, TypedLayout):
        raise TypeError("layout must be TypedLayout")
    if layout.contract != contract:
        raise TypeError(f"layout is not a {contract.replace('_', '-')} layout")
    if layout.semantic_version != semantic_version:
        raise ValueError(
            f"layout semantic version must be {semantic_version}, "
            f"got {layout.semantic_version}"
        )
    return layout


def flatten_observation(
    value: NormalizedObservation,
    *,
    context: NormalizationContext | None = None,
    layout: TypedLayout | None = None,
) -> tuple[FlatTree, TypedLayout]:
    """Flatten a normalized player view under one explicit scale contract."""

    if not isinstance(value, NormalizedObservation):
        raise TypeError("value must be NormalizedObservation")
    resolved = _normalized_flatten_layout(
        value,
        contract="player_observation",
        semantic_version=MODEL_OBSERVATION_SCHEMA_VERSION,
        context=context,
        layout=layout,
    )
    return _flatten(value, resolved), resolved


def flatten_manager_observation(
    value: NormalizedManagerObservation,
    *,
    context: NormalizationContext | None = None,
    layout: TypedLayout | None = None,
) -> tuple[FlatTree, TypedLayout]:
    """Flatten a normalized manager view under one explicit scale contract."""

    if not isinstance(value, NormalizedManagerObservation):
        raise TypeError("value must be NormalizedManagerObservation")
    resolved = _normalized_flatten_layout(
        value,
        contract="manager_observation",
        semantic_version=MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION,
        context=context,
        layout=layout,
    )
    return _flatten(value, resolved), resolved


def flatten_global_state(
    value: NormalizedGlobalRollout,
    *,
    context: NormalizationContext | None = None,
    layout: TypedLayout | None = None,
) -> tuple[FlatTree, TypedLayout]:
    """Flatten a normalized global view under one explicit scale contract."""

    if not isinstance(value, NormalizedGlobalRollout):
        raise TypeError("value must be NormalizedGlobalRollout")
    resolved = _normalized_flatten_layout(
        value,
        contract="global_state",
        semantic_version=MODEL_STATE_SCHEMA_VERSION,
        context=context,
        layout=layout,
    )
    return _flatten(value, resolved), resolved


def flatten_events(value: FrameEvents) -> tuple[FlatTree, TypedLayout]:
    layout = event_spec(value)
    return _flatten(value, layout), layout


def unflatten_action(flat: FlatTree, layout: TypedLayout) -> IntentAction:
    if layout.contract != "action":
        raise TypeError("layout is not an action layout")
    return _unflatten(flat, layout)


def unflatten_action_trace(flat: FlatTree, layout: TypedLayout) -> ActionTrace:
    if layout.contract != "action_trace":
        raise TypeError("layout is not an action-trace layout")
    return _unflatten(flat, layout)


def unflatten_action_receipt(flat: FlatTree, layout: TypedLayout) -> ActionReceipt:
    if layout.contract != "action_receipt":
        raise TypeError("layout is not an action-receipt layout")
    return _unflatten(flat, layout)


def unflatten_observation(
    flat: FlatTree,
    layout: TypedLayout,
) -> NormalizedObservation:
    if layout.contract != "player_observation":
        raise TypeError("layout is not a player-observation layout")
    return _unflatten(flat, layout)


def unflatten_manager_observation(
    flat: FlatTree,
    layout: TypedLayout,
) -> NormalizedManagerObservation:
    if layout.contract != "manager_observation":
        raise TypeError("layout is not a manager-observation layout")
    return _unflatten(flat, layout)


def unflatten_global_state(
    flat: FlatTree,
    layout: TypedLayout,
) -> NormalizedGlobalRollout:
    if layout.contract != "global_state":
        raise TypeError("layout is not a global-state layout")
    return _unflatten(flat, layout)


def unflatten_events(flat: FlatTree, layout: TypedLayout) -> FrameEvents:
    if layout.contract != "frame_events":
        raise TypeError("layout is not a frame-events layout")
    return _unflatten(flat, layout)


__all__ = [
    "ACTION_RECEIPT_SCHEMA",
    "ACTION_SCHEMA",
    "ACTION_TRACE_SCHEMA",
    "FRAME_EVENTS_SCHEMA",
    "TYPED_LAYOUT_SCHEMA",
    "FlatTree",
    "LeafSpec",
    "TreeSpec",
    "TypedLayout",
    "action_receipt_spec",
    "action_spec",
    "action_trace_spec",
    "event_spec",
    "flatten_action",
    "flatten_action_receipt",
    "flatten_action_trace",
    "flatten_events",
    "flatten_global_state",
    "flatten_manager_observation",
    "flatten_observation",
    "global_state_spec",
    "manager_observation_spec",
    "player_observation_spec",
    "schema_versions",
    "unflatten_action",
    "unflatten_action_receipt",
    "unflatten_action_trace",
    "unflatten_events",
    "unflatten_global_state",
    "unflatten_manager_observation",
    "unflatten_observation",
]
