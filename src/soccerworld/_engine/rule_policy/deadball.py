"""Runtime lookup for the K League-derived dead-ball positioning field."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..constants import RK_FREEKICK, RK_OFFSIDE, RESTART_COUNT


DEADBALL_POSITIONING_VERSION = 2
_PATH = Path(__file__).with_name("deadball_positioning.json")


def _load():
    with _PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != 1:
        raise RuntimeError("unsupported dead-ball positioning data format")
    if payload.get("deadball_positioning_version") != DEADBALL_POSITIONING_VERSION:
        raise RuntimeError("dead-ball positioning version/data mismatch")
    coefficients = np.asarray(payload["coefficients"], np.float32)
    centers = np.asarray(payload["feature_centers"], np.float32)
    valid = np.asarray(payload["valid"], bool)
    counts = np.asarray(payload["sample_count"], np.int32)
    role_mirror = np.asarray(payload["role_mirror"], np.int32)
    expected_coefficients = (RESTART_COUNT, 2, 3, 11, 2, 3)
    expected_cells = expected_coefficients[:-2]
    if coefficients.shape != expected_coefficients:
        raise RuntimeError(f"invalid dead-ball coefficient shape {coefficients.shape}")
    if centers.shape != expected_cells + (2,) or valid.shape != expected_cells:
        raise RuntimeError("invalid dead-ball center/valid shape")
    if counts.shape != expected_cells:
        raise RuntimeError("invalid dead-ball sample-count shape")
    if not np.isfinite(coefficients).all() or not np.isfinite(centers).all():
        raise RuntimeError("non-finite dead-ball positioning table")
    if np.any(counts < 0):
        raise RuntimeError("negative dead-ball sample count")
    minimum_samples = int(payload.get("source", {}).get("minimum_samples_per_cell", 0))
    if minimum_samples <= 0 or np.any(counts[valid] < minimum_samples):
        raise RuntimeError("dead-ball validity/sample-count mismatch")
    if (
        role_mirror.shape != (11,)
        or not np.array_equal(np.sort(role_mirror), np.arange(11))
        or not np.array_equal(role_mirror[role_mirror], np.arange(11))
    ):
        raise RuntimeError("invalid dead-ball role-mirror permutation")
    return payload, coefficients, centers, valid, counts, role_mirror


(METADATA, COEFFICIENTS, FEATURE_CENTERS, VALID, SAMPLE_COUNT,
 ROLE_MIRROR) = _load()


def target(ball_field, slot, is_gk, is_sp_ours, restart_one_hot):
    """Return ``(target[N,2], valid[N])`` in each observer's attack frame."""

    kind = jnp.argmax(restart_one_hot, axis=1).astype(jnp.int32)
    kind = jnp.where(kind == RK_OFFSIDE, jnp.int32(RK_FREEKICK), kind)
    side_index = jnp.where(is_sp_ours > 0.5, 0, 1).astype(jnp.int32)
    zone = jnp.where(ball_field[:, 0] < -17.5, 0,
                     jnp.where(ball_field[:, 0] > 17.5, 2, 1)).astype(jnp.int32)
    role = jnp.where(is_gk, 10, jnp.clip(slot, 0, 9)).astype(jnp.int32)
    # Fitting mirrors negative-y restarts into the positive-y half and swaps
    # left/right formation roles.  Apply the same role permutation on lookup;
    # mirroring coordinates alone would send every short-side player to the
    # far side of a throw-in or corner.
    # 역할 치환과 좌표 복원은 **같은 술어**를 써야 한다. 종전에는 역할을 ``y < 0``으로
    # 뒤집고 좌표는 ``sign(y + 1e-8)``로 되돌려서, ``-1e-8 < y <= 0``에서 역할은 미러인데
    # 좌표는 미러가 아니었다(좌우가 통째로 뒤바뀐다). ``y == -1e-8``에서는 sign이 0이
    # 되어 예측 y가 0으로 무너지기까지 했다.
    mirrored = ball_field[:, 1] < 0.0
    role = jnp.where(mirrored, jnp.asarray(ROLE_MIRROR)[role], role)

    coefficients = jnp.asarray(COEFFICIENTS)[kind, side_index, zone, role]
    centers = jnp.asarray(FEATURE_CENTERS)[kind, side_index, zone, role]
    usable = jnp.asarray(VALID)[kind, side_index, zone, role]
    feature_delta = jnp.stack(
        [ball_field[:, 0], jnp.abs(ball_field[:, 1])], axis=1
    ) - centers
    canonical = coefficients[:, :, 0] + jnp.sum(
        coefficients[:, :, 1:] * feature_delta[:, None, :], axis=2
    )
    side = jnp.where(mirrored, -1.0, 1.0).astype(canonical.dtype)
    predicted = canonical.at[:, 1].set(canonical[:, 1] * side)
    usable = usable & (jnp.abs(is_sp_ours) > 0.5)
    return predicted, usable
