"""Minimal immutable JAX state carried by an environment rollout."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE
from footballworld.core.contact import MECHANISM_NONE, ContactResult


class BallState(NamedTuple):
    """Dynamic ball state."""

    position: jax.Array
    velocity: jax.Array
    spin: jax.Array
    live: jax.Array


class PlayerState(NamedTuple):
    """Dynamic roster state and abilities needed by physics."""

    position: jax.Array
    velocity: jax.Array
    body_forward: jax.Array
    gaze_yaw: jax.Array
    team_id: jax.Array
    player_id: jax.Array
    on_pitch: jax.Array
    sent_off: jax.Array
    is_goalkeeper: jax.Array

    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array

    stamina_long: jax.Array
    stamina_short: jax.Array
    challenge_recovery_substeps: jax.Array
    contact_lock_substeps: jax.Array
    aerial_recovery_substeps: jax.Array
    possession_loss_lock_substeps: jax.Array
    yellow_cards: jax.Array

    @property
    def active(self):
        """Players currently participating in physics and rules."""

        return self.on_pitch & (~self.sent_off)


class PossessionState(NamedTuple):
    """Current control and the causal contact that produced it."""

    team: jax.Array
    player: jax.Array
    previous_team: jax.Array
    control_ticks: jax.Array
    last_contact: ContactResult


class RestartState(NamedTuple):
    """Pending or active restart."""

    kind: jax.Array
    team: jax.Array
    substeps_remaining: jax.Array
    taker: jax.Array
    indirect: jax.Array
    opened_control_tick: jax.Array = jnp.int32(-1)


class RestartReleaseProvenance(NamedTuple):
    """Executed restart facts retained for retouch and direct-goal rules.

    ``law11_direct_exempt`` is retained for the later offside consumer; this
    state does not itself adjudicate an offside offence.
    """

    active: jax.Array
    untouched: jax.Array
    kind: jax.Array
    team: jax.Array
    taker: jax.Array
    indirect: jax.Array
    law11_direct_exempt: jax.Array
    release_mechanism: jax.Array = jnp.int32(MECHANISM_NONE)


class State(NamedTuple):
    """Authoritative rollout state without observation or audit telemetry."""

    control_tick: jax.Array
    ball: BallState
    players: PlayerState
    attack_direction: jax.Array
    kickoff_team: jax.Array
    possession: PossessionState
    restart: RestartState
    score: jax.Array
    restart_release: RestartReleaseProvenance = RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    # Only the targeted team-mate foot-play cause lives here. Restart-derived
    # goalkeeper restrictions retain their existing causal provenance.
    gk_backpass_team: jax.Array = jnp.int32(NO_TEAM)
    # Added time is exactly the control frames spent in an out-of-play
    # restart. Goalkeeper possession remains live play and is not counted.
    dead_ball_control_ticks: jax.Array = jnp.int32(0)
    # Wall-clock tick at the completed first-half boundary; -1 before it.
    # This lets public clocks remove first-half added time in period two.
    first_half_wall_end_tick: jax.Array = jnp.int32(-1)
    # Live completion of a period-boundary penalty is added time, not part of
    # the following half.  Persist it separately from dead-ball replacement.
    first_half_live_extension_ticks: jax.Array = jnp.int32(0)
    # A penalty taken at a period boundary remains live until its outcome is
    # settled. Pending penalties are represented by ``restart``; this latch
    # covers the post-kick trajectory after that restart has been cleared.
    penalty_completion_active: jax.Array = jnp.bool_(False)
    # Kicking team retained while a period-boundary penalty trajectory is live.
    penalty_completion_team: jax.Array = jnp.int32(NO_TEAM)
    # True only after the current active restart layout has passed complete
    # fixed-shape preparation. Restart creation and any layout-affecting rare
    # event clear the latch before another release may occur.
    restart_layout_ready: jax.Array = jnp.bool_(False)


def initial_player_body_forward(
    team_id: jax.Array,
    attack_direction: jax.Array,
) -> jax.Array:
    """Initialize a seam-free chest direction from each team's attack direction."""

    team_direction = attack_direction[team_id]
    return jnp.stack((team_direction, jnp.zeros_like(team_direction)), axis=-1).astype(
        jnp.float32
    )


def body_forward_from_angle(angle: jax.Array) -> jax.Array:
    """Convert a rare scalar layout heading into the canonical unit vector."""

    angle = jnp.asarray(angle, dtype=jnp.float32)
    return jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1).astype(jnp.float32)


def body_angle_from_forward(body_forward: jax.Array) -> jax.Array:
    """Return the scalar geometric heading of a body-forward vector."""

    body_forward = jnp.asarray(body_forward, dtype=jnp.float32)
    return jnp.arctan2(body_forward[..., 1], body_forward[..., 0]).astype(jnp.float32)
