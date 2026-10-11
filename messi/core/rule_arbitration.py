"""한 physics substep의 play-stopping rule 후보를 시간순으로 선택한다."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

STOPPAGE_NONE = 0
STOPPAGE_FOUL = 1
STOPPAGE_RETOUCH = 2
STOPPAGE_OFFSIDE = 3
STOPPAGE_BOUNDARY = 4


class StoppageSelection(NamedTuple):
    """선택된 최초 stoppage의 종류와 substep-local 시간."""

    kind: jax.Array
    time_fraction: jax.Array


def select_first_stoppage(
    foul_occurred: jax.Array,
    foul_time: jax.Array,
    retouch_occurred: jax.Array,
    retouch_time: jax.Array,
    offside_occurred: jax.Array,
    offside_time: jax.Array,
    boundary_occurred: jax.Array,
    boundary_time: jax.Array,
) -> StoppageSelection:
    """가장 이른 stoppage를 고른다.

    동일 timestamp는 IFAB seriousness와 FootballWorld contract에 맞춰
    foul, restart retouch, offside, boundary 순으로 해결한다. 고정된 네 scalar만
    비교하므로 event trajectory나 가변 shape를 만들지 않는다.
    """

    dtype = jnp.result_type(foul_time, retouch_time, offside_time, boundary_time)
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    times = jnp.stack(
        (
            infinity,
            jnp.where(foul_occurred, foul_time, infinity),
            jnp.where(retouch_occurred, retouch_time, infinity),
            jnp.where(offside_occurred, offside_time, infinity),
            jnp.where(boundary_occurred, boundary_time, infinity),
        )
    )
    kind = jnp.argmin(times).astype(jnp.int32)
    selected_time = times[kind]
    any_stoppage = jnp.isfinite(selected_time)
    return StoppageSelection(
        kind=jnp.where(any_stoppage, kind, STOPPAGE_NONE).astype(jnp.int32),
        time_fraction=jnp.where(any_stoppage, selected_time, 1.0).astype(dtype),
    )


__all__ = [
    "STOPPAGE_BOUNDARY",
    "STOPPAGE_FOUL",
    "STOPPAGE_NONE",
    "STOPPAGE_OFFSIDE",
    "STOPPAGE_RETOUCH",
    "StoppageSelection",
    "select_first_stoppage",
]
