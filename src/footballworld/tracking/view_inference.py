"""Host-only body and view trajectory inference for tracking datasets.

The functions in this module deliberately do not enter ``FootballWorld.step``.
Tracking feeds generally contain player centres, not chest or eye direction, so
every returned direction is labelled as an inference and accompanied by masks,
confidence, and per-cell source codes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

INFERRED_VIEW_SCHEMA = "footballworld.inferred-body-view/4"

BODY_SOURCE_INVALID = np.uint8(0)
BODY_SOURCE_TRACKING_VELOCITY = np.uint8(1)
BODY_SOURCE_POSITION_DIFFERENCE = np.uint8(2)
BODY_SOURCE_CARRIED = np.uint8(3)
BODY_SOURCE_BALL_PRIOR = np.uint8(4)
BODY_SOURCE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        int(BODY_SOURCE_INVALID): "invalid",
        int(BODY_SOURCE_TRACKING_VELOCITY): "tracking_velocity",
        int(BODY_SOURCE_POSITION_DIFFERENCE): "causal_position_difference",
        int(BODY_SOURCE_CARRIED): "carried_previous_inference",
        int(BODY_SOURCE_BALL_PRIOR): "ball_facing_design_prior",
    }
)

VIEW_SOURCE_INVALID = np.uint8(0)
VIEW_SOURCE_BALL = np.uint8(1)
VIEW_SOURCE_BODY_CENTRE = np.uint8(2)
VIEW_SOURCE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        int(VIEW_SOURCE_INVALID): "invalid",
        int(VIEW_SOURCE_BALL): "ball_directed_design_prior",
        int(VIEW_SOURCE_BODY_CENTRE): "body_centred_missing_ball_prior",
    }
)


@dataclass(frozen=True, slots=True)
class ViewInferenceConfig:
    """Uncalibrated, replaceable design priors for host-side inference.

    None of these defaults is a measured football constant. They bound a
    deterministic augmentation heuristic until a facing-labelled tracking
    source is available for calibration.
    """

    frame_rate_hz: float = 25.0
    stationary_speed_mps: float = 0.35
    full_confidence_speed_mps: float = 2.0
    body_turn_rate_max_radps: float = 8.0
    body_carry_seconds: float = 0.75
    gaze_yaw_max_rad: float = math.radians(90.0)
    gaze_slew_rate_max_radps: float = math.radians(540.0)
    ball_direction_min_m: float = 0.25
    moving_body_confidence_floor: float = 0.35
    moving_body_confidence_ceiling: float = 0.80
    ball_body_confidence: float = 0.12
    ball_view_confidence_scale: float = 0.65
    neutral_view_confidence_scale: float = 0.10
    chunk_frames: int = 2048

    def __post_init__(self) -> None:
        positive = (
            "frame_rate_hz",
            "stationary_speed_mps",
            "full_confidence_speed_mps",
            "body_turn_rate_max_radps",
            "body_carry_seconds",
            "gaze_slew_rate_max_radps",
            "ball_direction_min_m",
        )
        confidence = (
            "moving_body_confidence_floor",
            "moving_body_confidence_ceiling",
            "ball_body_confidence",
            "ball_view_confidence_scale",
            "neutral_view_confidence_scale",
        )
        for name in (*positive, "gaze_yaw_max_rad", *confidence):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.full_confidence_speed_mps <= self.stationary_speed_mps:
            raise ValueError(
                "full_confidence_speed_mps must exceed stationary_speed_mps"
            )
        if (
            not math.isfinite(self.gaze_yaw_max_rad)
            or not 0.0 < self.gaze_yaw_max_rad < math.pi
        ):
            raise ValueError("gaze_yaw_max_rad must be finite and in (0, pi)")
        for name in confidence:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.moving_body_confidence_ceiling < self.moving_body_confidence_floor:
            raise ValueError(
                "moving_body_confidence_ceiling must be at least its floor"
            )
        if isinstance(self.chunk_frames, bool) or not isinstance(
            self.chunk_frames, int
        ):
            raise TypeError("chunk_frames must be an integer")
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")


def view_inference_config_from_environment(
    environment: Any,
    *,
    frame_rate_hz: float,
    base: ViewInferenceConfig | None = None,
    chunk_frames: int | None = None,
) -> ViewInferenceConfig:
    """Bind environment body/gaze priors to a tracking-source frame rate.

    Dataset sampling rate remains an explicit caller input: it must not be
    inferred from the simulator control rate. Other inference-only confidence
    and missing-data priors come from ``base`` (or the documented defaults).
    This is host-only configuration plumbing and never enters a JAX graph.
    """

    cfg = ViewInferenceConfig() if base is None else base
    if not isinstance(cfg, ViewInferenceConfig):
        raise TypeError("base must be ViewInferenceConfig or None")
    try:
        body_turn_rate_max_radps = environment.player_physics.body_turn_rate_max_radps
        gaze_yaw_limit_degrees = environment.perception.gaze_yaw_limit_degrees
        gaze_slew_rate_degrees_s = environment.perception.gaze_slew_rate_degrees_s
    except AttributeError as error:
        raise TypeError(
            "environment must expose player_physics and perception settings"
        ) from error
    updates = {
        "frame_rate_hz": frame_rate_hz,
        "body_turn_rate_max_radps": body_turn_rate_max_radps,
        "gaze_yaw_max_rad": math.radians(gaze_yaw_limit_degrees),
        "gaze_slew_rate_max_radps": math.radians(gaze_slew_rate_degrees_s),
    }
    if chunk_frames is not None:
        updates["chunk_frames"] = chunk_frames
    return replace(cfg, **updates)


@dataclass(frozen=True, slots=True)
class ViewInferenceProvenance:
    """Dataset-level provenance for one inferred trajectory."""

    schema: str
    semantic_label: str
    method: str
    algorithm_causal_given_inputs: bool
    causal: bool
    causality_scope: str
    input_causality: str
    velocity_source: str
    velocity_causality: str
    coefficient_authority: str
    continuity_guard_enabled: bool
    maximum_continuity_speed_mps: float | None
    chunk_frames: int
    body_source_names: Mapping[int, str]
    view_source_names: Mapping[int, str]
    source_provenance: Mapping[str, Any] | None
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class InferredViewTrajectories:
    """Seam-free inferred player body and view trajectories.

    ``body_forward`` and ``view_forward`` are world-frame unit vectors.
    ``gaze_relative`` is ``[cos(yaw), sin(yaw)]`` in the body frame.
    ``gaze_yaw`` is also included for direct environment-state augmentation;
    it is bounded strictly inside ``(-pi, pi)`` and therefore has no wrap seam.
    Invalid direction rows are zero and must be consumed with their masks.
    """

    body_forward: np.ndarray
    view_forward: np.ndarray
    gaze_relative: np.ndarray
    gaze_yaw: np.ndarray
    gaze_normalized: np.ndarray
    tracking_valid: np.ndarray
    continuity_rejected: np.ndarray
    body_valid: np.ndarray
    view_valid: np.ndarray
    ball_target_used: np.ndarray
    body_confidence: np.ndarray
    view_confidence: np.ndarray
    body_source: np.ndarray
    view_source: np.ndarray
    sequence_start: np.ndarray
    provenance: ViewInferenceProvenance

    @property
    def inferred_not_observed(self) -> bool:
        """Always true: no output in this record is ground-truth gaze."""

        return True

    def as_dict(self) -> dict[str, Any]:
        """Return a dataset-writer-friendly mapping without hiding provenance."""

        provenance = {
            "schema": self.provenance.schema,
            "semantic_label": self.provenance.semantic_label,
            "method": self.provenance.method,
            "algorithm_causal_given_inputs": (
                self.provenance.algorithm_causal_given_inputs
            ),
            "causal": self.provenance.causal,
            "causality_scope": self.provenance.causality_scope,
            "input_causality": self.provenance.input_causality,
            "velocity_source": self.provenance.velocity_source,
            "velocity_causality": self.provenance.velocity_causality,
            "coefficient_authority": self.provenance.coefficient_authority,
            "continuity_guard_enabled": self.provenance.continuity_guard_enabled,
            "maximum_continuity_speed_mps": (
                self.provenance.maximum_continuity_speed_mps
            ),
            "chunk_frames": self.provenance.chunk_frames,
            "body_source_names": dict(self.provenance.body_source_names),
            "view_source_names": dict(self.provenance.view_source_names),
            "source_provenance": deepcopy(self.provenance.source_provenance),
            "config": dict(self.provenance.config),
        }
        return {
            "schema": self.provenance.schema,
            "semantic_label": self.provenance.semantic_label,
            "body_forward": self.body_forward,
            "view_forward": self.view_forward,
            "gaze_relative": self.gaze_relative,
            "gaze_yaw": self.gaze_yaw,
            "gaze_normalized": self.gaze_normalized,
            "tracking_valid": self.tracking_valid,
            "continuity_rejected": self.continuity_rejected,
            "body_valid": self.body_valid,
            "view_valid": self.view_valid,
            "ball_target_used": self.ball_target_used,
            "body_confidence": self.body_confidence,
            "view_confidence": self.view_confidence,
            "body_source": self.body_source,
            "view_source": self.view_source,
            "sequence_start": self.sequence_start,
            "provenance": provenance,
        }


def _boolean_mask(
    value: Any | None,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return result


def _sequence_start_mask(
    value: Any | None,
    frame_count: int,
    player_count: int,
) -> np.ndarray:
    if value is None:
        result = np.zeros((frame_count, player_count), dtype=bool)
    else:
        provided = np.asarray(value)
        if provided.dtype != np.bool_:
            raise TypeError("sequence_start must have boolean dtype")
        if provided.shape == (frame_count,):
            result = np.broadcast_to(
                provided[:, None], (frame_count, player_count)
            ).copy()
        elif provided.shape == (frame_count, player_count):
            result = provided.copy()
        else:
            raise ValueError(
                "sequence_start must have shape "
                f"({frame_count},) or ({frame_count}, {player_count})"
            )
    result[0] = True
    return result


def sequence_start_from_identity(
    player_identity: Any,
    *,
    slot_generation: Any | None = None,
    explicit_sequence_start: Any | None = None,
) -> np.ndarray:
    """Return exact per-player resets for stable or reusable tracking slots.

    Identity values must be integer or fixed-width string arrays shaped
    ``[T, N]``. Optional slot generations must be integral with the same shape.
    A change in either value starts a new sequence, as does an explicit boolean
    ``[T]`` or ``[T, N]`` reset. Frame zero always starts every sequence.

    Generation boundaries are respected without guessing, repairing, or
    certifying provider identities.
    """

    identity = np.asarray(player_identity)
    if identity.ndim != 2:
        raise ValueError("player_identity must have shape [T, N]")
    frame_count, player_count = identity.shape
    if frame_count < 1 or player_count < 1:
        raise ValueError("player_identity must contain at least one frame and player")
    if identity.dtype.kind not in {"i", "u", "S", "U"}:
        raise TypeError(
            "player_identity must have an integer or fixed-width string dtype"
        )
    result = _sequence_start_mask(
        explicit_sequence_start,
        frame_count,
        player_count,
    )
    result[1:] |= identity[1:] != identity[:-1]
    if slot_generation is not None:
        generation = np.asarray(slot_generation)
        if generation.shape != identity.shape:
            raise ValueError(f"slot_generation must have shape {identity.shape}")
        if generation.dtype.kind not in {"i", "u"}:
            raise TypeError("slot_generation must have an integer dtype")
        result[1:] |= generation[1:] != generation[:-1]
    return result


def _time_axis(
    frame_count: int,
    timestamps_s: Any | None,
    frame_rate_hz: float,
    sequence_start_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if timestamps_s is None:
        time = np.arange(frame_count, dtype=np.float64) / frame_rate_hz
    else:
        time = np.asarray(timestamps_s, dtype=np.float64)
        if time.shape != (frame_count,):
            raise ValueError(f"timestamps_s must have shape ({frame_count},)")
        if not np.all(np.isfinite(time)):
            raise ValueError("timestamps_s must be finite")
        if frame_count > 1:
            differences = np.diff(time)
            nonincreasing = differences <= 0.0
            forbidden = nonincreasing & ~sequence_start_all[1:]
            if np.any(forbidden):
                frame = int(np.flatnonzero(forbidden)[0]) + 1
                raise ValueError(
                    "timestamps_s may reset or repeat only when every player "
                    f"starts a new sequence at frame {frame}"
                )
    dt = np.empty(frame_count, dtype=np.float64)
    dt[0] = 1.0 / frame_rate_hz
    if frame_count > 1:
        differences = np.diff(time)
        dt[1:] = np.where(
            differences <= 0.0,
            1.0 / frame_rate_hz,
            differences,
        )
    return time, dt


def _apply_continuity_guard(
    position: np.ndarray,
    tracking_valid: np.ndarray,
    time: np.ndarray,
    sequence_start: np.ndarray,
    maximum_speed_mps: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fail closed on caller-bounded, causally impossible identity motion.

    The optional speed limit is a caller-owned diagnostic guard, not a fitted
    facing coefficient. A rejected sample never advances that identity's last
    trusted position. The first later sample reachable from the trusted anchor
    starts a new inference sequence, so neither velocity difference nor carried
    body/gaze state crosses the rejected interval. This detects some association
    discontinuities; it does not repair or certify provider identity labels.
    """

    rejected = np.zeros(tracking_valid.shape, dtype=bool)
    if maximum_speed_mps is None:
        return tracking_valid, rejected, sequence_start

    guarded_valid = tracking_valid.copy()
    guarded_sequence_start = sequence_start.copy()
    player_count = position.shape[1]
    trusted_position = np.zeros((player_count, 2), dtype=np.float64)
    trusted_time = np.zeros(player_count, dtype=np.float64)
    have_trusted = np.zeros(player_count, dtype=bool)
    rejected_gap = np.zeros(player_count, dtype=bool)

    for frame in range(position.shape[0]):
        reset = sequence_start[frame]
        have_trusted[reset] = False
        rejected_gap[reset] = False

        candidate = guarded_valid[frame]
        checked = candidate & have_trusted
        elapsed = time[frame] - trusted_time
        valid_elapsed = np.isfinite(elapsed) & (elapsed > 0.0)
        displacement = np.linalg.norm(
            position[frame] - trusted_position,
            axis=-1,
        )
        impossible = checked & (
            (~valid_elapsed)
            | (~np.isfinite(displacement))
            | (displacement > maximum_speed_mps * elapsed)
        )
        rejected[frame] = impossible
        guarded_valid[frame, impossible] = False
        rejected_gap |= impossible

        accepted = guarded_valid[frame]
        reconnect = accepted & rejected_gap
        guarded_sequence_start[frame, reconnect] = True
        trusted_position[accepted] = position[frame, accepted]
        trusted_time[accepted] = time[frame]
        have_trusted[accepted] = True
        rejected_gap[reconnect] = False

    return guarded_valid, rejected, guarded_sequence_start


def _normalise(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=-1)
    valid = np.isfinite(norms) & (norms > 1e-8)
    result = np.zeros_like(vectors, dtype=np.float64)
    result[valid] = vectors[valid] / norms[valid, None]
    return result, valid


def _rotate_towards(
    current: np.ndarray,
    target: np.ndarray,
    maximum_angle: float,
) -> np.ndarray:
    """Rotate unit vectors along their shortest arc without angle wrapping."""

    dot = np.clip(np.sum(current * target, axis=-1), -1.0, 1.0)
    cross = current[:, 0] * target[:, 1] - current[:, 1] * target[:, 0]
    angle = np.arccos(dot)
    step = np.minimum(angle, maximum_angle)
    # Exact antipodes have two equally short paths; positive rotation is the
    # deterministic tie-break. No player index or random state enters it.
    sign = np.where(cross < 0.0, -1.0, 1.0)
    signed_step = sign * step
    cosine = np.cos(signed_step)
    sine = np.sin(signed_step)
    rotated = np.column_stack(
        (
            current[:, 0] * cosine - current[:, 1] * sine,
            current[:, 0] * sine + current[:, 1] * cosine,
        )
    )
    reached = angle <= maximum_angle
    rotated[reached] = target[reached]
    return rotated


def infer_body_view_trajectories(
    player_position: Any,
    *,
    ball_position: Any | None = None,
    player_velocity: Any | None = None,
    player_valid: Any | None = None,
    ball_valid: Any | None = None,
    timestamps_s: Any | None = None,
    sequence_start: Any | None = None,
    maximum_continuity_speed_mps: float | None = None,
    input_causality: str = "unknown",
    velocity_causality: str = "unknown",
    source_provenance: Mapping[str, Any] | None = None,
    config: ViewInferenceConfig | None = None,
) -> InferredViewTrajectories:
    """Infer causal body/view trajectories from tracking centres.

    Parameters use time-major shapes: player arrays are ``[T, N, 2]`` and the
    ball is ``[T, 2]`` or ``[T, >=2]``. Explicit validity masks are intersected
    with finite-value masks, never allowed to bless NaN/Inf samples. When
    velocity is absent, only the current and immediately previous valid
    position are differenced; no future frame is read. ``sequence_start`` may
    be boolean ``[T]`` or ``[T, N]``. It cuts all carry and position-difference
    history for the selected player rows; frame zero is always a sequence start.
    An explicitly supplied ``maximum_continuity_speed_mps`` additionally rejects
    causally impossible identity motion. It is a caller diagnostic threshold,
    not an inferred or calibrated player-speed value.

    The implementation is NumPy host code. It processes sequential fixed-size
    chunks while carrying only ``O(N)`` state between them, so arbitrary ``T``
    does not create a new compiled graph and chunk boundaries do not alter the
    result.
    """

    cfg = ViewInferenceConfig() if config is None else config
    if not isinstance(cfg, ViewInferenceConfig):
        raise TypeError("config must be ViewInferenceConfig or None")
    if source_provenance is not None and not isinstance(source_provenance, Mapping):
        raise TypeError("source_provenance must be a mapping or None")
    if velocity_causality not in {"causal", "unknown", "noncausal"}:
        raise ValueError(
            "velocity_causality must be 'causal', 'unknown', or 'noncausal'"
        )
    if input_causality not in {"causal", "unknown", "noncausal"}:
        raise ValueError("input_causality must be 'causal', 'unknown', or 'noncausal'")
    if maximum_continuity_speed_mps is not None:
        if isinstance(maximum_continuity_speed_mps, (bool, np.bool_)):
            raise TypeError("maximum_continuity_speed_mps must be numeric or None")
        if not isinstance(maximum_continuity_speed_mps, Real):
            raise TypeError("maximum_continuity_speed_mps must be numeric or None")
        maximum_continuity_speed_mps = float(maximum_continuity_speed_mps)
        if (
            not math.isfinite(maximum_continuity_speed_mps)
            or maximum_continuity_speed_mps <= 0.0
        ):
            raise ValueError("maximum_continuity_speed_mps must be finite and positive")

    position = np.asarray(player_position, dtype=np.float64)
    if position.ndim != 3 or position.shape[-1] != 2:
        raise ValueError("player_position must have shape [T, N, 2]")
    frame_count, player_count, _ = position.shape
    if frame_count < 1 or player_count < 1:
        raise ValueError("player_position must contain at least one frame and player")
    applied_sequence_start = _sequence_start_mask(
        sequence_start, frame_count, player_count
    )
    explicit_player_valid = _boolean_mask(
        player_valid, (frame_count, player_count), name="player_valid"
    )
    tracking_valid = explicit_player_valid & np.all(np.isfinite(position), axis=-1)

    velocity_was_provided = player_velocity is not None
    if velocity_was_provided:
        velocity = np.asarray(player_velocity, dtype=np.float64)
        if velocity.shape != position.shape:
            raise ValueError(f"player_velocity must have shape {position.shape}")
        velocity_finite = np.all(np.isfinite(velocity), axis=-1)
    else:
        velocity = np.zeros_like(position)
        velocity_finite = np.zeros((frame_count, player_count), dtype=bool)

    if ball_position is None:
        ball_xy = np.zeros((frame_count, 2), dtype=np.float64)
        finite_ball = np.zeros(frame_count, dtype=bool)
        if ball_valid is not None:
            _boolean_mask(ball_valid, (frame_count,), name="ball_valid")
    else:
        ball = np.asarray(ball_position, dtype=np.float64)
        if ball.ndim != 2 or ball.shape[0] != frame_count or ball.shape[1] < 2:
            raise ValueError("ball_position must have shape [T, 2] or [T, >=2]")
        ball_xy = ball[:, :2]
        finite_ball = np.all(np.isfinite(ball_xy), axis=-1)
    explicit_ball_valid = _boolean_mask(ball_valid, (frame_count,), name="ball_valid")
    usable_ball = explicit_ball_valid & finite_ball

    time, dt = _time_axis(
        frame_count,
        timestamps_s,
        cfg.frame_rate_hz,
        np.all(applied_sequence_start, axis=1),
    )
    tracking_valid, continuity_rejected, applied_sequence_start = (
        _apply_continuity_guard(
            position,
            tracking_valid,
            time,
            applied_sequence_start,
            maximum_continuity_speed_mps,
        )
    )

    vector_shape = (frame_count, player_count, 2)
    scalar_shape = (frame_count, player_count)
    body_forward = np.zeros(vector_shape, dtype=np.float32)
    view_forward = np.zeros(vector_shape, dtype=np.float32)
    gaze_relative = np.zeros(vector_shape, dtype=np.float32)
    gaze_yaw = np.zeros(scalar_shape, dtype=np.float32)
    gaze_normalized = np.zeros(scalar_shape, dtype=np.float32)
    body_valid = np.zeros(scalar_shape, dtype=bool)
    view_valid = np.zeros(scalar_shape, dtype=bool)
    ball_target_used = np.zeros(scalar_shape, dtype=bool)
    body_confidence = np.zeros(scalar_shape, dtype=np.float32)
    view_confidence = np.zeros(scalar_shape, dtype=np.float32)
    body_source = np.zeros(scalar_shape, dtype=np.uint8)
    view_source = np.zeros(scalar_shape, dtype=np.uint8)

    previous_body = np.zeros((player_count, 2), dtype=np.float64)
    previous_body_confidence = np.zeros(player_count, dtype=np.float64)
    body_age_s = np.full(player_count, np.inf, dtype=np.float64)
    have_body = np.zeros(player_count, dtype=bool)
    previous_gaze_yaw = np.zeros(player_count, dtype=np.float64)
    have_gaze = np.zeros(player_count, dtype=bool)

    speed_span = cfg.full_confidence_speed_mps - cfg.stationary_speed_mps
    for chunk_start in range(0, frame_count, cfg.chunk_frames):
        chunk_end = min(frame_count, chunk_start + cfg.chunk_frames)
        for frame in range(chunk_start, chunk_end):
            valid_player = tracking_valid[frame]
            reset = applied_sequence_start[frame]
            previous_body[reset] = 0.0
            previous_body_confidence[reset] = 0.0
            body_age_s[reset] = np.inf
            have_body[reset] = False
            previous_gaze_yaw[reset] = 0.0
            have_gaze[reset] = False
            body_age_s[have_body] += dt[frame]
            expired_before_frame = body_age_s > cfg.body_carry_seconds
            have_body[expired_before_frame] = False
            previous_body_confidence[expired_before_frame] = 0.0

            if velocity_was_provided:
                frame_velocity = np.zeros((player_count, 2), dtype=np.float64)
                provided = valid_player & velocity_finite[frame]
                frame_velocity[provided] = velocity[frame, provided]
                velocity_valid = provided.copy()
                velocity_source = np.full(
                    player_count, BODY_SOURCE_INVALID, dtype=np.uint8
                )
                velocity_source[provided] = BODY_SOURCE_TRACKING_VELOCITY
                # A partially missing supplied velocity array need not discard
                # otherwise consecutive, finite position tracking.
                if frame:
                    differenced = (
                        valid_player & tracking_valid[frame - 1] & ~provided & ~reset
                    )
                    frame_velocity[differenced] = (
                        position[frame, differenced] - position[frame - 1, differenced]
                    ) / dt[frame]
                    velocity_valid[differenced] = True
                    velocity_source[differenced] = BODY_SOURCE_POSITION_DIFFERENCE
            else:
                frame_velocity = np.zeros((player_count, 2), dtype=np.float64)
                velocity_source = np.full(
                    player_count, BODY_SOURCE_INVALID, dtype=np.uint8
                )
                if frame:
                    consecutive = valid_player & tracking_valid[frame - 1] & ~reset
                    frame_velocity[consecutive] = (
                        position[frame, consecutive] - position[frame - 1, consecutive]
                    ) / dt[frame]
                    velocity_valid = consecutive
                    velocity_source[consecutive] = BODY_SOURCE_POSITION_DIFFERENCE
                else:
                    velocity_valid = np.zeros(player_count, dtype=bool)

            speed = np.linalg.norm(frame_velocity, axis=-1)
            moving = (
                velocity_valid
                & np.isfinite(speed)
                & (speed >= cfg.stationary_speed_mps)
            )
            movement_direction = np.zeros((player_count, 2), dtype=np.float64)
            movement_direction[moving] = frame_velocity[moving] / speed[moving, None]

            current_body = np.zeros((player_count, 2), dtype=np.float64)
            current_confidence = np.zeros(player_count, dtype=np.float64)
            current_source = np.full(player_count, BODY_SOURCE_INVALID, dtype=np.uint8)
            current_valid = np.zeros(player_count, dtype=bool)

            continuing = moving & have_body
            if np.any(continuing):
                current_body[continuing] = _rotate_towards(
                    previous_body[continuing],
                    movement_direction[continuing],
                    cfg.body_turn_rate_max_radps * dt[frame],
                )
            starting = moving & ~have_body
            current_body[starting] = movement_direction[starting]
            current_valid[moving] = True
            current_source[moving] = velocity_source[moving]
            movement_strength = np.clip(
                (speed - cfg.stationary_speed_mps) / speed_span, 0.0, 1.0
            )
            current_confidence[moving] = (
                cfg.moving_body_confidence_floor
                + (
                    cfg.moving_body_confidence_ceiling
                    - cfg.moving_body_confidence_floor
                )
                * movement_strength[moving]
            )
            body_age_s[moving] = 0.0

            carried = (
                valid_player
                & ~moving
                & have_body
                & (body_age_s <= cfg.body_carry_seconds)
            )
            current_body[carried] = previous_body[carried]
            current_valid[carried] = True
            current_source[carried] = BODY_SOURCE_CARRIED
            decay = np.maximum(0.0, 1.0 - body_age_s / cfg.body_carry_seconds)
            current_confidence[carried] = (
                previous_body_confidence[carried] * decay[carried]
            )

            ball_delta = ball_xy[frame, None, :] - position[frame]
            ball_direction, ball_direction_valid = _normalise(ball_delta)
            ball_distance = np.linalg.norm(ball_delta, axis=-1)
            ball_evidence = (
                valid_player
                & usable_ball[frame]
                & ball_direction_valid
                & (ball_distance >= cfg.ball_direction_min_m)
            )
            ball_initialised = ~current_valid & ball_evidence
            current_body[ball_initialised] = ball_direction[ball_initialised]
            current_valid[ball_initialised] = True
            current_source[ball_initialised] = BODY_SOURCE_BALL_PRIOR
            current_confidence[ball_initialised] = cfg.ball_body_confidence
            body_age_s[ball_initialised] = 0.0

            # Missing tracking never emits a direction, even if latent carry
            # state is retained briefly for a clean reappearance.
            current_valid &= valid_player
            current_body[~current_valid] = 0.0
            current_confidence[~current_valid] = 0.0
            current_source[~current_valid] = BODY_SOURCE_INVALID

            body_forward[frame] = current_body.astype(np.float32)
            body_valid[frame] = current_valid
            body_confidence[frame] = current_confidence.astype(np.float32)
            body_source[frame] = current_source

            attentive = current_valid & ball_evidence
            desired_yaw = np.zeros(player_count, dtype=np.float64)
            if np.any(attentive):
                dot = np.clip(np.sum(current_body * ball_direction, axis=-1), -1.0, 1.0)
                cross = (
                    current_body[:, 0] * ball_direction[:, 1]
                    - current_body[:, 1] * ball_direction[:, 0]
                )
                relative_angle = np.arctan2(cross, dot)
                antipodal = attentive & (dot < 0.0) & (np.abs(cross) < 1e-10)
                antipodal_sign = np.where(
                    have_gaze & (previous_gaze_yaw < 0.0), -1.0, 1.0
                )
                relative_angle[antipodal] = math.pi * antipodal_sign[antipodal]
                desired_yaw[attentive] = np.clip(
                    relative_angle[attentive],
                    -cfg.gaze_yaw_max_rad,
                    cfg.gaze_yaw_max_rad,
                )

            current_gaze = desired_yaw.copy()
            slew = current_valid & have_gaze
            maximum_gaze_step = cfg.gaze_slew_rate_max_radps * dt[frame]
            current_gaze[slew] = previous_gaze_yaw[slew] + np.clip(
                desired_yaw[slew] - previous_gaze_yaw[slew],
                -maximum_gaze_step,
                maximum_gaze_step,
            )
            current_gaze = np.clip(
                current_gaze, -cfg.gaze_yaw_max_rad, cfg.gaze_yaw_max_rad
            )
            cosine = np.cos(current_gaze)
            sine = np.sin(current_gaze)
            relative = np.column_stack((cosine, sine))
            world_view = np.column_stack(
                (
                    current_body[:, 0] * cosine - current_body[:, 1] * sine,
                    current_body[:, 0] * sine + current_body[:, 1] * cosine,
                )
            )
            relative[~current_valid] = 0.0
            world_view[~current_valid] = 0.0

            gaze_yaw[frame] = np.where(current_valid, current_gaze, 0.0).astype(
                np.float32
            )
            gaze_normalized[frame] = np.where(
                current_valid,
                gaze_yaw[frame] / np.float32(cfg.gaze_yaw_max_rad),
                np.float32(0.0),
            ).astype(np.float32)
            gaze_relative[frame] = relative.astype(np.float32)
            view_forward[frame] = world_view.astype(np.float32)
            view_valid[frame] = current_valid
            ball_target_used[frame] = attentive
            view_source[frame, attentive] = VIEW_SOURCE_BALL
            view_source[frame, current_valid & ~attentive] = VIEW_SOURCE_BODY_CENTRE
            view_confidence[frame, attentive] = (
                current_confidence[attentive] * cfg.ball_view_confidence_scale
            ).astype(np.float32)
            neutral = current_valid & ~attentive
            view_confidence[frame, neutral] = (
                current_confidence[neutral] * cfg.neutral_view_confidence_scale
            ).astype(np.float32)

            previous_body[current_valid] = current_body[current_valid]
            reliable_body = moving | ball_initialised
            previous_body_confidence[reliable_body] = current_confidence[reliable_body]
            have_body = have_body | current_valid
            expired = body_age_s > cfg.body_carry_seconds
            have_body[expired] = False
            previous_body_confidence[expired] = 0.0
            previous_gaze_yaw[current_valid] = current_gaze[current_valid]
            have_gaze |= current_valid

    effective_velocity_causality = (
        velocity_causality if velocity_was_provided else "causal"
    )
    algorithm_causal_given_inputs = True
    causal = (
        algorithm_causal_given_inputs
        and input_causality == "causal"
        and effective_velocity_causality == "causal"
    )
    provenance = ViewInferenceProvenance(
        schema=INFERRED_VIEW_SCHEMA,
        semantic_label="inferred_not_observed_or_ground_truth",
        method="velocity_ball_vector_heuristic_v3",
        algorithm_causal_given_inputs=algorithm_causal_given_inputs,
        causal=causal,
        causality_scope=(
            "algorithm reads current/past rows only; sequence_start cuts prior "
            "state and position differences; an enabled continuity guard compares "
            "only with each player's last trusted past sample and performs no "
            "identity correction; caller-declared player/ball input "
            f"causality is {input_causality!r} and supplied velocity causality "
            f"is {effective_velocity_causality!r}; overall causal is true only "
            "when both upstream declarations are causal"
        ),
        input_causality=input_causality,
        velocity_source=(
            "provided_tracking_velocity_with_causal_position_fallback"
            if velocity_was_provided
            else "causal_position_difference"
        ),
        velocity_causality=effective_velocity_causality,
        coefficient_authority="uncalibrated_design_prior",
        continuity_guard_enabled=maximum_continuity_speed_mps is not None,
        maximum_continuity_speed_mps=maximum_continuity_speed_mps,
        chunk_frames=cfg.chunk_frames,
        body_source_names=BODY_SOURCE_NAMES,
        view_source_names=VIEW_SOURCE_NAMES,
        source_provenance=(
            None if source_provenance is None else deepcopy(dict(source_provenance))
        ),
        config=MappingProxyType(asdict(cfg)),
    )
    return InferredViewTrajectories(
        body_forward=body_forward,
        view_forward=view_forward,
        gaze_relative=gaze_relative,
        gaze_yaw=gaze_yaw,
        gaze_normalized=gaze_normalized,
        tracking_valid=tracking_valid,
        continuity_rejected=continuity_rejected,
        body_valid=body_valid,
        view_valid=view_valid,
        ball_target_used=ball_target_used,
        body_confidence=body_confidence,
        view_confidence=view_confidence,
        body_source=body_source,
        view_source=view_source,
        sequence_start=applied_sequence_start,
        provenance=provenance,
    )


__all__ = [
    "BODY_SOURCE_BALL_PRIOR",
    "BODY_SOURCE_CARRIED",
    "BODY_SOURCE_INVALID",
    "BODY_SOURCE_NAMES",
    "BODY_SOURCE_POSITION_DIFFERENCE",
    "BODY_SOURCE_TRACKING_VELOCITY",
    "INFERRED_VIEW_SCHEMA",
    "VIEW_SOURCE_BALL",
    "VIEW_SOURCE_BODY_CENTRE",
    "VIEW_SOURCE_INVALID",
    "VIEW_SOURCE_NAMES",
    "InferredViewTrajectories",
    "ViewInferenceConfig",
    "ViewInferenceProvenance",
    "infer_body_view_trajectories",
    "sequence_start_from_identity",
    "view_inference_config_from_environment",
]
