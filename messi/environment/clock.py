"""authoritative state에서 파생하는 causal match clock."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MatchClock(NamedTuple):
    period: jax.Array
    regulation_elapsed_ticks: jax.Array
    dead_ball_ticks: jax.Array
    added_time_active: jax.Array
    added_time_elapsed_ticks: jax.Array
    added_time_remaining_ticks: jax.Array


def regulation_elapsed_ticks(state) -> jax.Array:
    return jnp.maximum(
        state.control_tick.astype(jnp.int32)
        - state.dead_ball_control_ticks.astype(jnp.int32)
        - state.first_half_live_extension_ticks.astype(jnp.int32),
        jnp.int32(0),
    )


def match_clock(state, *, halftime_tick: int, fulltime_tick: int) -> MatchClock:
    """미래 dead-ball 시간을 사용하지 않는 causal clock을 반환한다."""
    wall = state.control_tick.astype(jnp.int32)
    live = regulation_elapsed_ticks(state)
    second_half = state.first_half_wall_end_tick >= 0
    regulation_end = jnp.where(second_half, fulltime_tick, halftime_tick)
    first_added = jnp.where(
        second_half,
        jnp.maximum(state.first_half_wall_end_tick - halftime_tick, 0),
        0,
    ).astype(jnp.int32)
    broadcast_tick = wall - first_added
    period_dead = jnp.where(
        second_half,
        jnp.maximum(
            state.dead_ball_control_ticks
            - jnp.maximum(first_added - state.first_half_live_extension_ticks, 0),
            0,
        ),
        state.dead_ball_control_ticks,
    ).astype(jnp.int32)
    added_elapsed = jnp.maximum(broadcast_tick - regulation_end, 0).astype(jnp.int32)
    return MatchClock(
        period=jnp.where(second_half, 2, 1).astype(jnp.int32),
        regulation_elapsed_ticks=jnp.minimum(live, fulltime_tick),
        dead_ball_ticks=period_dead,
        added_time_active=broadcast_tick > regulation_end,
        added_time_elapsed_ticks=added_elapsed,
        added_time_remaining_ticks=jnp.maximum(period_dead - added_elapsed, 0),
    )
