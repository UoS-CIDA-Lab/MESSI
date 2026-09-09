"""Explicit process-level runtime opt-ins for MESSI deployments."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import stat
from dataclasses import asdict, is_dataclass
from pathlib import Path

import jax
import jaxlib

from footballworld._version import __version__
from footballworld.core.action import ACTION_SCHEMA, ActionIntent
from footballworld.core.contact import INTENT_SOURCE_NAMES, INTENT_SOURCE_SCHEMA
from footballworld.environment.management import (
    MANAGER_OBSERVATION_SCHEMA_VERSION as SI_MANAGER_OBSERVATION_SCHEMA_VERSION,
)
from footballworld.environment.management import (
    PLAYER_TACTICAL_OBSERVATION_SCHEMA_VERSION as SI_PLAYER_TACTICAL_OBSERVATION_SCHEMA_VERSION,
)
from footballworld.environment.normalization import (
    MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION,
    MODEL_OBSERVATION_SCHEMA_VERSION,
    MODEL_ROSTER_SCHEMA_VERSION,
    MODEL_STATE_SCHEMA_VERSION,
    MODEL_TACTICAL_OBSERVATION_SCHEMA_VERSION,
)
from footballworld.environment.observation import (
    PLAYER_OBSERVATION_SCHEMA_VERSION as SI_PLAYER_OBSERVATION_SCHEMA_VERSION,
)
from footballworld.environment.observation import (
    ROSTER_METADATA_SCHEMA_VERSION as SI_ROSTER_METADATA_SCHEMA_VERSION,
)


def enable_compilation_cache(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = 2 * 1024**3,
    min_compile_seconds: float = 1.0,
) -> Path:
    """Enable JAX's persistent compilation cache for this process.

    Call this before the first JAX compilation. ``path`` must already exist,
    be writable by the current process, and not be writable by group or other
    users. A JAX compilation cache is a trusted-code boundary, so callers must
    provide a directory they own and do not share with untrusted writers.
    This function never creates a directory and importing the package never
    enables the cache automatically.

    The configured byte cap uses JAX's least-recently-accessed eviction policy.
    ``min_compile_seconds`` avoids filling the cache with cheap compilations.
    """

    if not isinstance(max_bytes, numbers.Integral) or isinstance(max_bytes, bool):
        raise TypeError("max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not isinstance(min_compile_seconds, numbers.Real) or isinstance(
        min_compile_seconds, bool
    ):
        raise TypeError("min_compile_seconds must be a finite non-negative real")
    if not math.isfinite(min_compile_seconds) or min_compile_seconds < 0.0:
        raise ValueError("min_compile_seconds must be finite and non-negative")

    requested = Path(path)
    if requested.is_symlink():
        raise ValueError("compilation cache path must not be a symbolic link")
    try:
        directory = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("compilation cache directory must already exist") from exc
    if not directory.is_dir():
        raise ValueError("compilation cache path must be a directory")

    metadata = directory.stat()
    if hasattr(os, "geteuid"):
        if metadata.st_uid != os.geteuid():
            raise PermissionError(
                "compilation cache directory must be owned by the current user"
            )
        untrusted_write = stat.S_IWGRP | stat.S_IWOTH
        if metadata.st_mode & untrusted_write:
            raise PermissionError(
                "compilation cache directory must not be group- or other-writable"
            )
    if not os.access(directory, os.W_OK | os.X_OK):
        raise PermissionError("compilation cache directory is not writable")

    jax.config.update("jax_compilation_cache_max_size", int(max_bytes))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        float(min_compile_seconds),
    )
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(directory))

    # Package imports create constant JAX arrays and may cache a disabled state
    # before callers configure a directory. Re-evaluate cache enablement on the
    # next compilation without clearing executable or tracing caches.
    from jax.experimental.compilation_cache import compilation_cache

    compilation_cache.reset_cache()
    return directory


def environment_fingerprint(environment=None) -> dict[str, object]:
    """Return host-side runtime facts needed to reproduce a rollout binary.

    The optional environment hash covers every static dataclass field without
    importing it into a JIT graph. Device identity is descriptive; backend and
    numerical settings are the reproducibility-critical fields.
    """

    backend = jax.default_backend()
    devices = jax.devices(backend)
    result: dict[str, object] = {
        "footballworld": __version__,
        "action_schema": ACTION_SCHEMA,
        "action_intents": [intent.name for intent in ActionIntent],
        "intent_source_schema": INTENT_SOURCE_SCHEMA,
        "intent_source_names": list(INTENT_SOURCE_NAMES),
        "model_observation_schema_version": MODEL_OBSERVATION_SCHEMA_VERSION,
        "model_state_schema_version": MODEL_STATE_SCHEMA_VERSION,
        "model_manager_observation_schema_version": (
            MODEL_MANAGER_OBSERVATION_SCHEMA_VERSION
        ),
        "model_roster_schema_version": MODEL_ROSTER_SCHEMA_VERSION,
        "model_tactical_observation_schema_version": (
            MODEL_TACTICAL_OBSERVATION_SCHEMA_VERSION
        ),
        "si_manager_observation_schema_version": (
            SI_MANAGER_OBSERVATION_SCHEMA_VERSION
        ),
        "si_player_observation_schema_version": (SI_PLAYER_OBSERVATION_SCHEMA_VERSION),
        "si_player_tactical_observation_schema_version": (
            SI_PLAYER_TACTICAL_OBSERVATION_SCHEMA_VERSION
        ),
        "si_roster_metadata_schema_version": SI_ROSTER_METADATA_SCHEMA_VERSION,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": backend,
        "enable_x64": bool(jax.config.jax_enable_x64),
        "default_prng_impl": str(jax.config.jax_default_prng_impl),
        "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "device_count": len(devices),
        "device_kinds": sorted({device.device_kind for device in devices}),
    }
    if environment is not None:
        if not is_dataclass(environment):
            raise TypeError("environment must be a dataclass instance")
        canonical_json = json.dumps(
            asdict(environment), sort_keys=True, separators=(",", ":")
        )
        payload = canonical_json.encode("utf-8")
        # The digest proves equality, while the canonical payload makes a
        # custom environment reconstructible.  Keep both host-side: neither
        # belongs in a transition PyTree or a JIT cache key beyond the static
        # environment object that already owns these values.
        result["environment_config"] = json.loads(canonical_json)
        result["environment_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


__all__ = [
    "ACTION_SCHEMA",
    "enable_compilation_cache",
    "environment_fingerprint",
]
