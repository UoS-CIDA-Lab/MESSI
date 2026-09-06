"""Pure-JAX helpers for environment-owned restart release timing."""

import jax
import jax.numpy as jnp

from footballworld.config.restart_timing import RestartTiming
from footballworld.core.constants import (
    RESTART_COUNT,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_THROWIN,
)
from footballworld.core.state import State
from footballworld.core.timebase import DEFAULT_TIMEBASE, Timebase


def forced_release_delay_substeps(
    *,
    timebase: Timebase = DEFAULT_TIMEBASE,
    config: RestartTiming = RestartTiming(),
) -> int:
    """Convert the shared three-second deadline once on the host."""

    return timebase.physics_steps_for(
        config.forced_release_delay_s,
        rounding="ceil",
        minimum=1,
    )


def continuous_restart_approach_enabled(
    restart_kind: jax.Array,
    *,
    config: RestartTiming = RestartTiming(),
) -> jax.Array:
    """Return the static-config selection for one traced restart kind."""

    kind = jnp.asarray(restart_kind, dtype=jnp.int32)
    if kind.shape != ():
        raise ValueError("restart_kind must be scalar")
    return (
        ((kind == RK_FREEKICK) & jnp.bool_(config.continuous_freekick_approach))
        | ((kind == RK_OFFSIDE) & jnp.bool_(config.continuous_offside_approach))
        | ((kind == RK_CORNER) & jnp.bool_(config.continuous_corner_approach))
        | ((kind == RK_THROWIN) & jnp.bool_(config.continuous_throwin_approach))
    )


def _ticks_until_release(
    state: State,
    *,
    delay_substeps: int,
    hold_limit_substeps: int,
) -> jax.Array:
    """Return complete physics intervals before the release gate opens."""

    hold_threshold = jnp.maximum(
        jnp.int32(hold_limit_substeps - delay_substeps), jnp.int32(0)
    )
    threshold = jnp.where(
        state.restart.kind == RK_GK_HOLD,
        hold_threshold,
        jnp.int32(0),
    )
    return jnp.maximum(state.restart.substeps_remaining - threshold, 0)


def restart_release_due(
    state: State,
    *,
    delay_substeps: int,
    hold_limit_substeps: int,
) -> jax.Array:
    """Whether an active restart's gate is open on this physics tick."""

    active = (state.restart.kind > RK_NONE) & (state.restart.kind < RESTART_COUNT)
    return active & (
        (state.restart.kind == RK_KICKOFF)
        | (
            _ticks_until_release(
                state,
                delay_substeps=delay_substeps,
                hold_limit_substeps=hold_limit_substeps,
            )
            <= 0
        )
    )


def restart_may_release_within_frame(
    state: State,
    decimation: int,
    *,
    delay_substeps: int,
    hold_limit_substeps: int,
) -> jax.Array:
    """Whether the release gate opens during the upcoming control frame."""

    active = (state.restart.kind > RK_NONE) & (state.restart.kind < RESTART_COUNT)
    return active & (
        (state.restart.kind == RK_KICKOFF)
        | (
            _ticks_until_release(
                state,
                delay_substeps=delay_substeps,
                hold_limit_substeps=hold_limit_substeps,
            )
            < decimation
        )
    )


def advance_restart_release_clock(
    state: State,
    *,
    opened: jax.Array,
    delay_substeps: int,
    countdown_enabled: jax.Array = jnp.bool_(True),
) -> State:
    """Initialize or advance the ordinary-restart release countdown.

    ``RK_GK_HOLD`` retains its law-level counter; its forced-release threshold
    is derived from that same counter without adding recurrent state.
    """

    kind = state.restart.kind
    active = (kind > RK_NONE) & (kind < RESTART_COUNT)
    ordinary = active & (kind != RK_GK_HOLD) & (kind != RK_KICKOFF)
    countdown_enabled = jnp.asarray(countdown_enabled, dtype=jnp.bool_)
    if countdown_enabled.shape != ():
        raise ValueError("countdown_enabled must be scalar")
    continued = jnp.maximum(
        state.restart.substeps_remaining - jnp.int32(1), jnp.int32(0)
    )
    continued = jnp.where(
        countdown_enabled, continued, state.restart.substeps_remaining
    )
    remaining = jnp.where(
        ordinary,
        jnp.where(opened, jnp.int32(delay_substeps), continued),
        jnp.where(active, state.restart.substeps_remaining, jnp.int32(0)),
    ).astype(jnp.int32)
    return state._replace(restart=state.restart._replace(substeps_remaining=remaining))


__all__ = [
    "advance_restart_release_clock",
    "continuous_restart_approach_enabled",
    "forced_release_delay_substeps",
    "restart_may_release_within_frame",
    "restart_release_due",
]
