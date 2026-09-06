"""Causal match-clock views shared by model observations and replay output."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.state import State


class MatchClockTicks(NamedTuple):
    """Exact integer clock facts derived from the authoritative rollout state."""

    period: jax.Array
    period_duration_ticks: jax.Array
    period_regulation_elapsed_ticks: jax.Array
    match_regulation_elapsed_ticks: jax.Array
    period_dead_ball_ticks: jax.Array
    added_time_active: jax.Array
    added_time_elapsed_ticks: jax.Array
    added_time_remaining_ticks: jax.Array


class NormalizedMatchClock(NamedTuple):
    """Semantic progress plus power-of-two counters for exact restoration."""

    period: jax.Array
    period_regulation_progress: jax.Array
    match_regulation_progress: jax.Array
    dead_ball_accrued_fraction: jax.Array
    added_time_active: jax.Array
    added_time_elapsed_fraction: jax.Array
    added_time_remaining_fraction: jax.Array
    period_dead_ball_counter: jax.Array
    added_time_elapsed_counter: jax.Array
    added_time_remaining_counter: jax.Array


def regulation_elapsed_ticks(state: State) -> jax.Array:
    """Return regulation-play ticks without dead balls or prior-half extension."""

    return jnp.maximum(
        state.control_tick.astype(jnp.int32)
        - state.dead_ball_control_ticks.astype(jnp.int32)
        - state.first_half_live_extension_ticks.astype(jnp.int32),
        jnp.int32(0),
    )


def match_clock_ticks(
    state: State,
    *,
    halftime_tick: int,
    fulltime_tick: int,
    halftime_enabled: bool,
) -> MatchClockTicks:
    """Derive current-period regulation and added-time counters causally.

    FootballWorld replaces only time spent in out-of-play restarts.  The
    accumulated replacement is known, but future dead balls are not.  The
    returned remaining value can therefore grow during added time without
    leaking the final match duration.
    """

    wall = state.control_tick.astype(jnp.int32)
    dead = state.dead_ball_control_ticks.astype(jnp.int32)
    raw_live = regulation_elapsed_ticks(state)
    second_half = jnp.bool_(halftime_enabled) & (state.first_half_wall_end_tick >= 0)
    regulation_end = jnp.where(
        second_half,
        jnp.int32(fulltime_tick),
        jnp.where(
            jnp.bool_(halftime_enabled),
            jnp.int32(halftime_tick),
            jnp.int32(fulltime_tick),
        ),
    )
    live = jnp.minimum(raw_live, regulation_end)
    first_half_added = jnp.where(
        second_half,
        jnp.maximum(
            state.first_half_wall_end_tick - jnp.int32(halftime_tick),
            jnp.int32(0),
        ),
        jnp.int32(0),
    )
    first_half_dead = jnp.maximum(
        first_half_added - state.first_half_live_extension_ticks.astype(jnp.int32),
        jnp.int32(0),
    )
    period_duration = jnp.where(
        second_half,
        jnp.int32(fulltime_tick - halftime_tick),
        jnp.where(
            jnp.bool_(halftime_enabled),
            jnp.int32(halftime_tick),
            jnp.int32(fulltime_tick),
        ),
    )
    period_live = jnp.where(
        second_half,
        jnp.maximum(live - jnp.int32(halftime_tick), jnp.int32(0)),
        live,
    )
    period_dead = jnp.where(
        second_half,
        jnp.maximum(dead - first_half_dead, jnp.int32(0)),
        dead,
    )
    broadcast_tick = wall - first_half_added
    nominal_end = regulation_end
    added_elapsed = jnp.maximum(broadcast_tick - nominal_end, jnp.int32(0))
    added_active = broadcast_tick > nominal_end
    added_remaining = jnp.maximum(period_dead - added_elapsed, jnp.int32(0))
    return MatchClockTicks(
        period=jnp.where(second_half, jnp.int32(2), jnp.int32(1)),
        period_duration_ticks=period_duration,
        period_regulation_elapsed_ticks=period_live,
        match_regulation_elapsed_ticks=live,
        period_dead_ball_ticks=period_dead,
        added_time_active=added_active,
        added_time_elapsed_ticks=added_elapsed,
        added_time_remaining_ticks=added_remaining,
    )


def normalize_match_clock(
    clock: MatchClockTicks,
    *,
    fulltime_tick: int,
    counter_scale_ticks: int,
) -> NormalizedMatchClock:
    """Normalize an exact clock without clipping or realized-match scales."""

    dtype = jnp.float32
    # Invalid observer rows intentionally carry an all-zero clock sentinel.
    # Keep that row finite without clipping any valid clock value.
    period_scale = jnp.maximum(
        clock.period_duration_ticks.astype(dtype), jnp.float32(1.0)
    )
    match_scale = jnp.asarray(fulltime_tick, dtype=dtype)
    exact_counter_scale = jnp.asarray(counter_scale_ticks, dtype=dtype)
    return NormalizedMatchClock(
        period=clock.period,
        period_regulation_progress=(
            clock.period_regulation_elapsed_ticks.astype(dtype) / period_scale
        ),
        match_regulation_progress=(
            clock.match_regulation_elapsed_ticks.astype(dtype) / match_scale
        ),
        dead_ball_accrued_fraction=(
            clock.period_dead_ball_ticks.astype(dtype) / period_scale
        ),
        added_time_active=clock.added_time_active,
        added_time_elapsed_fraction=(
            clock.added_time_elapsed_ticks.astype(dtype) / period_scale
        ),
        added_time_remaining_fraction=(
            clock.added_time_remaining_ticks.astype(dtype) / period_scale
        ),
        period_dead_ball_counter=(
            clock.period_dead_ball_ticks.astype(dtype) / exact_counter_scale
        ),
        added_time_elapsed_counter=(
            clock.added_time_elapsed_ticks.astype(dtype) / exact_counter_scale
        ),
        added_time_remaining_counter=(
            clock.added_time_remaining_ticks.astype(dtype) / exact_counter_scale
        ),
    )


__all__ = [
    "MatchClockTicks",
    "NormalizedMatchClock",
    "match_clock_ticks",
    "normalize_match_clock",
    "regulation_elapsed_ticks",
]
