"""Observation-only goalkeeper hand-claim and foot-play selection.

The contact solver gives an eligible goalkeeper in their own penalty area an
automatic hand contact when the action does *not* select an eligible foot
contact.  Consequently, a lawful claim emits neither foot intent, whereas a
control or clearance emits its explicit categorical intent and a non-zero
encoded ``force_to_ball``.  This helper mirrors that boundary without reading
rollout ``State`` or sampling an outcome.

All geometry is reconstructed from the goalkeeper's own observation row and
the public environment configuration.  ``contact_may_occur_this_frame`` is
still required because it owns the phase, activity, and recovery-lock gates;
the observation deliberately leaves distance, height, and ball-speed tests to
consumers such as this helper.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.reach import Reach
from footballworld.core.action_mapping import linf_radial_encode
from footballworld.core.constants import NO_TEAM, RK_NONE, SAFE_NORM_EPS
from footballworld.environment.observation import Observation, RosterMetadata

_DEFAULT_ACTION_SCALE = ActionScale()
_DEFAULT_BALL = Ball()
_DEFAULT_REACH = Reach()
_DEFAULT_STADIUM = Stadium()


class GoalkeeperContactIntent(NamedTuple):
    """Per-observer mechanism choice and the matching action controls.

    Scalar fields have shape ``(observers,)`` and ``force_to_ball`` has shape
    ``(observers, 2)``.  ``foot_control`` and ``foot_clearance`` are mutually
    exclusive.  A true ``automatic_hand_claim`` has neither foot intent.
    """

    automatic_hand_claim: jax.Array
    foot_control: jax.Array
    foot_clearance: jax.Array
    foot_reachable: jax.Array
    hand_reachable: jax.Array
    handling_restricted: jax.Array
    force_to_ball: jax.Array


def _validate_core_shapes(
    contact_eligible: jax.Array,
    foot_reachable: jax.Array,
    hand_reachable: jax.Array,
    in_own_penalty_area: jax.Array,
    handling_restricted: jax.Array,
    request_foot: jax.Array,
    foot_direction: jax.Array,
    foot_power: jax.Array,
) -> None:
    if contact_eligible.ndim != 1:
        raise ValueError("contact_eligible must have shape (observers,)")
    observers = contact_eligible.shape[0]
    for name, value in (
        ("foot_reachable", foot_reachable),
        ("hand_reachable", hand_reachable),
        ("in_own_penalty_area", in_own_penalty_area),
        ("handling_restricted", handling_restricted),
        ("request_foot", request_foot),
        ("foot_power", foot_power),
    ):
        if value.shape != (observers,):
            raise ValueError(f"{name} must have shape (observers,)")
    if foot_direction.shape != (observers, 2):
        raise ValueError("foot_direction must have shape (observers, 2)")


def select_goalkeeper_contact_intent(
    contact_eligible: jax.Array,
    foot_reachable: jax.Array,
    hand_reachable: jax.Array,
    in_own_penalty_area: jax.Array,
    handling_restricted: jax.Array,
    request_foot: jax.Array,
    foot_direction: jax.Array,
    foot_power: jax.Array,
    *,
    scale: ActionScale = _DEFAULT_ACTION_SCALE,
) -> GoalkeeperContactIntent:
    """Select controls from caller-supplied observation-derived predicates.

    This is the integration-level core.  A policy that already computes ball
    reach and penalty-area masks should call it directly, avoiding a second
    geometry pass.  The caller must derive every predicate from its public
    observation row; no rollout state is accepted.
    """

    contact_eligible = jnp.asarray(contact_eligible, dtype=jnp.bool_)
    foot_reachable = jnp.asarray(foot_reachable, dtype=jnp.bool_)
    hand_reachable = jnp.asarray(hand_reachable, dtype=jnp.bool_)
    in_own_penalty_area = jnp.asarray(in_own_penalty_area, dtype=jnp.bool_)
    handling_restricted = jnp.asarray(handling_restricted, dtype=jnp.bool_)
    request_foot = jnp.asarray(request_foot, dtype=jnp.bool_)
    foot_direction = jnp.asarray(foot_direction, dtype=jnp.float32)
    foot_power = jnp.asarray(foot_power, dtype=jnp.float32)
    _validate_core_shapes(
        contact_eligible,
        foot_reachable,
        hand_reachable,
        in_own_penalty_area,
        handling_restricted,
        request_foot,
        foot_direction,
        foot_power,
    )

    clean_direction = jnp.nan_to_num(foot_direction, nan=0.0, posinf=0.0, neginf=0.0)
    clean_power = jnp.clip(
        jnp.nan_to_num(foot_power, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    nonzero_force = jnp.any(clean_direction != 0.0, axis=-1) & (clean_power > 0.0)
    foot_required = request_foot | handling_restricted | (~in_own_penalty_area)
    foot_play = contact_eligible & foot_required & foot_reachable & nonzero_force

    requested_speed = clean_power * scale.kick_speed_max_mps
    foot_control = foot_play & (requested_speed < scale.control_request_speed_max_mps)
    foot_clearance = foot_play & (~foot_control)
    automatic_hand_claim = (
        contact_eligible
        & in_own_penalty_area
        & (~handling_restricted)
        & (~foot_play)
        & hand_reachable
    )

    encoded_force = linf_radial_encode(clean_direction, clean_power)
    control_power = jnp.clip(
        clean_power * scale.kick_speed_max_mps / scale.control_request_speed_max_mps,
        0.0,
        1.0,
    )
    encoded_control_force = linf_radial_encode(clean_direction, control_power)
    selected_force = jnp.where(
        foot_control[:, None], encoded_control_force, encoded_force
    )
    force_to_ball = jnp.where(foot_play[:, None], selected_force, 0.0).astype(
        jnp.float32
    )
    return GoalkeeperContactIntent(
        automatic_hand_claim=automatic_hand_claim,
        foot_control=foot_control,
        foot_clearance=foot_clearance,
        foot_reachable=contact_eligible & foot_reachable,
        hand_reachable=contact_eligible & hand_reachable,
        handling_restricted=contact_eligible & handling_restricted,
        force_to_ball=force_to_ball,
    )


def _norm(vector: jax.Array) -> jax.Array:
    """Match the contact predicate's float32 Euclidean norm."""

    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + SAFE_NORM_EPS)


def _validate_shapes(
    observations: Observation,
    roster: RosterMetadata,
    request_foot: jax.Array,
    foot_direction: jax.Array,
    foot_power: jax.Array,
) -> tuple[int, int]:
    self_index = observations.self_state.player_index
    if self_index.ndim != 1:
        raise ValueError("observations must have a leading observer axis")
    observers = self_index.shape[0]
    if observations.players.visible.ndim != 2:
        raise ValueError("observed player fields must have shape (observers, players)")
    player_shape = observations.players.visible.shape
    players = player_shape[1]
    if player_shape[0] != observers:
        raise ValueError("player observations must share the observer axis")
    for name, value in (
        ("on_pitch", observations.players.on_pitch),
        ("sent_off", observations.players.sent_off),
        (
            "contact_may_occur_this_frame",
            observations.players.contact_may_occur_this_frame,
        ),
    ):
        if value.shape != player_shape:
            raise ValueError(f"observations.players.{name} must match visible")
    if observations.self_state.position.shape != (observers, 2):
        raise ValueError("self position must have shape (observers, 2)")
    if observations.self_state.velocity.shape != (observers, 2):
        raise ValueError("self velocity must have shape (observers, 2)")
    if observations.ball.relative_state.shape != (observers, 9):
        raise ValueError("ball relative_state must have shape (observers, 9)")
    for name, value in (
        ("ball.live", observations.ball.live),
        ("ball.visible", observations.ball.visible),
        ("restart.kind", observations.restart.kind),
        (
            "match.gk_handling_restricted_team",
            observations.match.gk_handling_restricted_team,
        ),
    ):
        if value.shape != (observers,):
            raise ValueError(f"observations.{name} must have shape (observers,)")
    for name, value in (
        ("team_id", roster.team_id),
        ("is_goalkeeper", roster.is_goalkeeper),
        ("reach_height", roster.reach_height),
        ("height", roster.height),
    ):
        if value.shape != (players,):
            raise ValueError(f"roster.{name} must have shape (players,)")
    if request_foot.shape != (observers,):
        raise ValueError("request_foot must have shape (observers,)")
    if foot_direction.shape != (observers, 2):
        raise ValueError("foot_direction must have shape (observers, 2)")
    if foot_power.shape != (observers,):
        raise ValueError("foot_power must have shape (observers,)")
    return observers, players


def goalkeeper_contact_intent(
    observations: Observation,
    roster: RosterMetadata,
    request_foot: jax.Array,
    foot_direction: jax.Array,
    foot_power: jax.Array,
    *,
    decimation: int = 1,
    ball: Ball = _DEFAULT_BALL,
    stadium: Stadium = _DEFAULT_STADIUM,
    reach: Reach = _DEFAULT_REACH,
    scale: ActionScale = _DEFAULT_ACTION_SCALE,
) -> GoalkeeperContactIntent:
    """Select automatic hand claim, foot control, or foot clearance.

    ``request_foot`` expresses a tactical preference.  A known handling
    restriction, or being outside the own penalty area, forces the foot path.
    The foot path is emitted only when the current public geometry matches the
    engine's foot predicate and the requested force is non-zero.  If a lawful
    hand contact remains the mechanism, the returned controls are neutral so
    the engine's automatic goalkeeper-hand path stays enabled.

    ``decimation`` is the number of upcoming physics substeps covered by the
    held action. A positive recovery timer shorter than that window may expire
    in time for a hand attempt, matching the engine's substep gate. The default
    of one asks only about the current physics instant.

    ``foot_direction`` uses the observation/action attacking frame and
    ``foot_power`` is normalized to ``[0, 1]``.  The engine's own requested
    speed threshold classifies a successful foot request as control or
    clearance; this module introduces no policy tuning constants.
    """

    if not isinstance(decimation, int) or isinstance(decimation, bool):
        raise TypeError("decimation must be an integer")
    if decimation < 1:
        raise ValueError("decimation must be positive")

    request_foot = jnp.asarray(request_foot, dtype=jnp.bool_)
    foot_direction = jnp.asarray(foot_direction, dtype=jnp.float32)
    foot_power = jnp.asarray(foot_power, dtype=jnp.float32)
    observers, players = _validate_shapes(
        observations,
        roster,
        request_foot,
        foot_direction,
        foot_power,
    )

    row = jnp.arange(observers, dtype=jnp.int32)
    self_index = jnp.asarray(observations.self_state.player_index, dtype=jnp.int32)
    valid_self = (self_index >= 0) & (self_index < players)
    safe_self = jnp.clip(self_index, 0, players - 1)

    self_team = roster.team_id[safe_self]
    self_goalkeeper = roster.is_goalkeeper[safe_self]
    self_active = observations.players.on_pitch[row, safe_self] & (
        ~observations.players.sent_off[row, safe_self]
    )
    structural_contact = observations.players.contact_may_occur_this_frame[
        row, safe_self
    ]
    self_visible = observations.players.visible[row, safe_self]
    open_play = observations.restart.kind == RK_NONE
    base = (
        valid_self
        & self_goalkeeper
        & self_active
        & self_visible
        & observations.ball.visible
        & observations.ball.live
        & open_play
        & structural_contact
    )

    relative_ball = observations.ball.relative_state[:, :3]
    distance_xy = _norm(relative_ball[:, :2])
    ball_height = relative_ball[:, 2]
    ball_horizontal_velocity = (
        observations.ball.relative_state[:, 3:5] + observations.self_state.velocity
    )
    horizontal_ball_speed = _norm(ball_horizontal_velocity)

    foot_reachable = (
        (distance_xy <= reach.carry_radius_m + ball.radius)
        & (
            ball_height
            <= roster.height[safe_self] * scale.pelvis_height_factor + ball.radius
        )
        & (
            horizontal_ball_speed + reach.height_speed_penalty_mps_per_m * ball_height
            <= reach.block_speed_limit_mps
        )
    )
    # ``contact_may_occur_this_frame`` is intentionally mechanism-agnostic.
    # The engine is authoritative and blocks every deliberate mechanism while
    # aerial recovery remains positive; this policy-side lookahead may arm a
    # hand attempt that becomes legal later in the same held control frame.
    hand_recovery_ready = (
        observations.players.aerial_recovery_substeps[row, safe_self] < decimation
    )
    hand_reachable = (
        (distance_xy <= reach.goalkeeper_radius_m + ball.radius)
        & (ball_height <= roster.reach_height[safe_self] + ball.radius)
        & hand_recovery_ready
    )

    local_ball_x = observations.self_state.position[:, 0] + relative_ball[:, 0]
    local_ball_y = observations.self_state.position[:, 1] + relative_ball[:, 1]
    own_penalty_depth = stadium.half_length + local_ball_x
    in_own_penalty_area = (
        (own_penalty_depth >= 0.0)
        & (own_penalty_depth <= stadium.penalty_area_length)
        & (jnp.abs(local_ball_y) <= 0.5 * stadium.penalty_area_width)
    )

    handling_team = observations.match.gk_handling_restricted_team
    handling_restricted = (handling_team != NO_TEAM) & (handling_team == self_team)

    return select_goalkeeper_contact_intent(
        base,
        foot_reachable,
        hand_reachable,
        in_own_penalty_area,
        handling_restricted,
        request_foot,
        foot_direction,
        foot_power,
        scale=scale,
    )


__all__ = [
    "GoalkeeperContactIntent",
    "goalkeeper_contact_intent",
    "select_goalkeeper_contact_intent",
]
