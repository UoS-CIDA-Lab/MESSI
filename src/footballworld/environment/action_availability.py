"""Observation-side availability for the six public action intents.

The mask rejects contact intents which cannot begin from the observed pose.
Substep dynamics remain authoritative for swept contact and stochastic
outcomes, but policies are not offered plainly unreachable contacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.reach import Reach
from footballworld.core.constants import (
    NO_TEAM,
    RESTART_COUNT,
    RK_GK_HOLD,
    RK_NONE,
    RK_THROWIN,
    SAFE_NORM_EPS,
)
from footballworld.core.contact import (
    ACTION_INTENT_COUNT,
    INTENT_CHALLENGE,
    INTENT_CLEAR,
    INTENT_CONTROL,
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SHOT,
    MECHANISM_CHEST,
    OUTCOME_TRAP,
)
from footballworld.environment.observation import Observation, RosterMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence

ACTION_INTENT_NAMES = (
    "MOVE",
    "CONTROL",
    "PASS",
    "SHOT",
    "CLEAR",
    "CHALLENGE",
)


def _observer_slot(value: jax.Array, observer: jax.Array) -> jax.Array:
    """Read the observer's own roster slot for scalar or batched views."""

    player_count = value.shape[-1]
    safe_observer = jnp.clip(observer, 0, player_count - 1)
    return jnp.take_along_axis(value, safe_observer[..., None], axis=-1)[..., 0]


def _roster_field(
    roster: RosterMetadata | None,
    name: str,
    player_shape: Sequence[int],
) -> jax.Array | None:
    """Broadcast immutable roster metadata over observation batch axes."""

    if roster is None:
        return None
    value = jnp.asarray(getattr(roster, name))
    target_shape = tuple(player_shape)
    if value.ndim < 1 or value.shape[-1] != target_shape[-1]:
        raise ValueError(
            f"roster.{name} must end in the observation roster-slot axis "
            f"of size {target_shape[-1]}, got {value.shape}"
        )
    if value.ndim > len(target_shape):
        raise ValueError(
            f"roster.{name} has more batch axes than the observation: "
            f"{value.shape} versus {target_shape}"
        )
    # Roster batch axes precede any observation-subject axes.  In particular,
    # a batched observe-all view is [environment, observer, roster_slot], while
    # its metadata is only [environment, roster_slot].
    broadcast_shape = (
        value.shape[:-1] + (1,) * (len(target_shape) - value.ndim) + value.shape[-1:]
    )
    return jnp.broadcast_to(jnp.reshape(value, broadcast_shape), target_shape)


def _norm(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + SAFE_NORM_EPS)


def intent_availability_hint(
    observation: Observation,
    roster: RosterMetadata | None = None,
    *,
    ball: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
) -> jax.Array:
    """Return a strict SI-observation ``[..., 6]`` intent-selection mask.

    Columns are exactly ``MOVE, CONTROL, PASS, SHOT, CLEAR, CHALLENGE``.
    Contact intents require observable phase legality, current horizontal and
    vertical reach, incoming-ball speed compatibility, and (for CHALLENGE) an
    observed opponent-contact context.  Passing SI ``roster`` metadata enables
    the exact player/GK envelope.  Omitting it retains a fail-closed ordinary
    outfield envelope and disables CHALLENGE because team identity cannot be
    recovered causally from ``Observation`` alone.

    ``True`` still does not promise a successful contact: continuous controls,
    swept substeps, contests, and stochastic outcomes remain authoritative in
    dynamics.
    """

    observer = jnp.asarray(observation.self_state.player_index, dtype=jnp.int32)
    player_count = observation.players.on_pitch.shape[-1]
    observer_valid = (observer >= 0) & (observer < player_count)
    observer_valid = observer_valid & jnp.asarray(observation.valid, dtype=jnp.bool_)
    contact_may_occur = _observer_slot(
        observation.players.contact_may_occur_this_frame,
        observer,
    )
    restart_taker = _observer_slot(observation.players.restart_taker, observer)
    observer_active = (
        observer_valid
        & _observer_slot(observation.players.on_pitch, observer)
        & (~_observer_slot(observation.players.sent_off, observer))
    )
    observable_candidate = observer_active & contact_may_occur

    player_shape = observation.players.on_pitch.shape
    ball_state = jnp.asarray(observation.ball.relative_state)
    ball_xy = ball_state[..., :2]
    ball_height = ball_state[..., 2]
    ball_velocity_xy = ball_state[..., 3:5] + observation.self_state.velocity
    distance_xy = _norm(ball_xy)
    horizontal_ball_speed = _norm(ball_velocity_xy)

    team_id = _roster_field(roster, "team_id", player_shape)
    is_goalkeeper = _roster_field(roster, "is_goalkeeper", player_shape)
    stature = _roster_field(roster, "height", player_shape)
    reach_height = _roster_field(roster, "reach_height", player_shape)
    roster_known = team_id is not None
    if roster_known:
        self_team = _observer_slot(team_id, observer)
        self_is_goalkeeper = _observer_slot(is_goalkeeper, observer)
        self_height = _observer_slot(stature, observer)
        self_reach_height = _observer_slot(reach_height, observer)
    else:
        # The lower bounds ensure every True fallback is physically reachable
        # by every valid roster profile. Richer valid contacts fail closed.
        self_team = jnp.full_like(observer, NO_TEAM)
        self_is_goalkeeper = jnp.zeros_like(observer, dtype=jnp.bool_)
        self_height = jnp.full_like(ball_height, 1.45)
        self_reach_height = self_height

    pelvis_height = self_height * jnp.asarray(
        scale.pelvis_height_factor, dtype=ball_height.dtype
    )
    torso_top_height = body.torso_top_height(self_height)
    foot_height = ball_height <= pelvis_height + ball.radius
    chest_height = (ball_height >= pelvis_height - ball.radius) & (
        ball_height <= torso_top_height + ball.radius
    )
    head_height = (ball_height >= torso_top_height - ball.radius) & (
        ball_height <= self_reach_height + ball.radius
    )
    ordinary_height_reachable = foot_height | chest_height | head_height
    foot_speed_reachable = (
        horizontal_ball_speed
        + jnp.asarray(reach.height_speed_penalty_mps_per_m, ball_height.dtype)
        * ball_height
        <= reach.block_speed_limit_mps
    )
    ordinary_speed_reachable = (~foot_height) | foot_speed_reachable

    ball_team_x = observation.self_state.position[..., 0] + ball_xy[..., 0]
    ball_team_y = observation.self_state.position[..., 1] + ball_xy[..., 1]
    own_penalty_area = (
        (ball_team_x >= -stadium.half_length)
        & (ball_team_x <= -stadium.half_length + stadium.penalty_area_length)
        & (jnp.abs(ball_team_y) <= 0.5 * stadium.penalty_area_width)
    )
    handling_known = observation.valid
    hand_restricted = handling_known & (
        observation.match.gk_handling_restricted_team == self_team
    )
    goalkeeper_hand_reachable = (
        self_is_goalkeeper
        & own_penalty_area
        & handling_known
        & (~hand_restricted)
        & (ball_height <= self_reach_height + ball.radius)
        & (distance_xy <= reach.goalkeeper_radius_m + ball.radius)
    )

    ordinary_reachable = (
        (distance_xy <= reach.carry_radius_m + ball.radius)
        & ordinary_height_reachable
        & ordinary_speed_reachable
    )

    possessor = observation.players.possessor
    last_actor = observation.players.last_actor
    observed_other_possessor = jnp.any(
        possessor & observation.players.on_pitch & (~observation.players.sent_off),
        axis=-1,
    ) & (~_observer_slot(possessor, observer))
    if roster_known:
        player_ball_distance = _norm(
            ball_xy[..., None, :] - observation.players.relative_position
        )
        retained_chest_trap = (
            last_actor
            & (observation.possession.last_contact.intent == INTENT_CONTROL)[..., None]
            & (observation.possession.last_contact.outcome == OUTCOME_TRAP)[..., None]
            & (observation.possession.last_contact.mechanism == MECHANISM_CHEST)[
                ..., None
            ]
        )
        carrier_height = jnp.where(
            retained_chest_trap,
            reach_height,
            stature * jnp.asarray(scale.pelvis_height_factor, ball_height.dtype),
        )
        unique_possessor = jnp.sum(possessor, axis=-1) == 1
        verified_possessor = (
            possessor
            & unique_possessor[..., None]
            & observation.players.on_pitch
            & (~observation.players.sent_off)
            & (observation.possession.control_ticks > 0)[..., None]
            & (player_ball_distance <= reach.carry_radius_m + ball.radius)
            & (ball_height[..., None] <= carrier_height + ball.radius)
        )
        opponent_possessor = jnp.any(
            verified_possessor & (team_id != self_team[..., None]), axis=-1
        )
    else:
        opponent_possessor = jnp.zeros_like(observer_active)
    # CHALLENGE is exclusively a tackle request against a physically verified
    # opposing carrier.  Every ownerless-ball reception, including an
    # interception of an opponent release or a just-loosened trap, is CONTROL.
    challenge_context = opponent_possessor
    # Without roster teams, another visible possessor could be either teammate
    # or opponent. Fail closed rather than offering an ordinary kick that may
    # bypass a tackle; the observer's own possession and ownerless balls remain
    # distinguishable directly from the observation.
    control_context = jnp.where(
        roster_known,
        ~challenge_context,
        ~observed_other_possessor,
    )
    challenge_ready = (
        _observer_slot(
            observation.players.challenge_recovery_substeps,
            observer,
        )
        <= 0
    )
    challenge_reachable = (
        challenge_context
        & challenge_ready
        & (distance_xy <= reach.challenge_radius_m + ball.radius)
        & ordinary_height_reachable
        & ordinary_speed_reachable
    )

    kind = observation.restart.kind
    open_play = kind == RK_NONE
    valid_restart = (kind > RK_NONE) & (kind < RESTART_COUNT)
    open_contact = observable_candidate & open_play
    restart_release = observable_candidate & valid_restart & restart_taker
    ordinary_kick_restart = (
        restart_release & (kind != RK_THROWIN) & (kind != RK_GK_HOLD)
    )
    goalkeeper_hold_release = (
        restart_release
        & (kind == RK_GK_HOLD)
        & (distance_xy <= reach.carry_radius_m + ball.radius)
    )
    reachable_restart_release = restart_release & ordinary_reachable

    allowed = jnp.zeros(jnp.shape(kind) + (ACTION_INTENT_COUNT,), dtype=jnp.bool_)
    allowed = allowed.at[..., INTENT_MOVE].set(observer_active)
    allowed = allowed.at[..., INTENT_CONTROL].set(
        open_contact
        & control_context
        & (ordinary_reachable | goalkeeper_hand_reachable)
    )
    allowed = allowed.at[..., INTENT_PASS].set(
        (open_contact & control_context & ordinary_reachable)
        | reachable_restart_release
        | goalkeeper_hold_release
    )
    allowed = allowed.at[..., INTENT_SHOT].set(
        (open_contact & control_context & ordinary_reachable)
        | (ordinary_kick_restart & ordinary_reachable)
    )
    allowed = allowed.at[..., INTENT_CLEAR].set(
        (
            open_contact
            & control_context
            & (ordinary_reachable | goalkeeper_hand_reachable)
        )
        | (ordinary_kick_restart & ordinary_reachable)
    )
    allowed = allowed.at[..., INTENT_CHALLENGE].set(open_contact & challenge_reachable)
    return allowed


__all__ = ["ACTION_INTENT_NAMES", "intent_availability_hint"]
