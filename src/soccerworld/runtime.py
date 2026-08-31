"""Process-level runtime configuration that is safe to import before JAX."""

from soccerworld._engine.jax_cache import (
    DEFAULT_CACHE_DIR as DEFAULT_COMPILATION_CACHE_DIR,
)
from soccerworld._engine.jax_cache import enable as enable_compilation_cache
from soccerworld._engine.jax_cache import is_enabled as compilation_cache_enabled

__all__ = [
    "DEFAULT_COMPILATION_CACHE_DIR",
    "compilation_cache_enabled",
    "enable_compilation_cache",
]
