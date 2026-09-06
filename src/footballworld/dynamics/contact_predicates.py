"""Authoritative physical and rule gates for deliberate ball contact.

Policy observations may consume the individual potential-contact predicates.
``potential`` is the current pre-recovery eligibility mask, not an
anticipatory mask for a future substep; ``possible_now`` additionally applies
the recovery locks. The engine combines ``possible_now`` with one explicit
legal intent. No predicate in this module samples an outcome or mutates state.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.action import ActionScale
from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.config.reach import Reach
from footballworld.core.constants import (
    NO_PLAYER,
    RK_GK_HOLD,
    RK_NONE,
    RK_THROWIN,
    SAFE_NORM_EPS,
    STATIONARY_SPEED_EPS,
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
    MECHANISM_FOOT,
    MECHANISM_GOALKEEPER_HAND,
    MECHANISM_HEAD,
    MECHANISM_THROW,
    OUTCOME_RELEASE,
    OUTCOME_TRAP,
)
from footballworld.core.state import State
from footballworld.rules.action_legality import restart_intent_allowed
from footballworld.rules.gk_handling_restriction import (
    goalkeeper_hand_restricted_mask,
)
from footballworld.rules.restart_legality import restart_actor_mask


class ContactPredicates(NamedTuple):
    """Fixed-shape contact semantics and gates shared by engine consumers."""

    mechanism: jax.Array
    challenge_context: jax.Array
    control_request: jax.Array
    goalkeeper_claim: jax.Array
    goalkeeper_hand_clear: jax.Array
    intent_allowed: jax.Array
    challenge_interception: jax.Array
    designated_restart: jax.Array
    phase_allowed: jax.Array
    horizontal_reach: jax.Array
    height_allowed: jax.Array
    speed_allowed: jax.Array
    recovery_ready: jax.Array
    potential: jax.Array
    possible_now: jax.Array
    verified_controlled_carrier: jax.Array
    distance_xy: jax.Array
    horizontal_reach_effort: jax.Array
    vertical_reach_effort: jax.Array
    athletic_reach_effort: jax.Array


def _norm(vector: jax.Array, *, axis: int = -1) -> jax.Array:
    return jnp.sqrt(jnp.sum(vector * vector, axis=axis) + SAFE_NORM_EPS)


def _extension_fraction(
    required: jax.Array,
    standing: jax.Array,
    maximum: jax.Array,
) -> jax.Array:
    """Return the minimum normalized extension beyond a standing envelope."""

    span = jnp.maximum(maximum - standing, SAFE_NORM_EPS)
    return jnp.clip((required - standing) / span, 0.0, 1.0)


def _in_own_penalty_area(state: State, stadium: Stadium) -> jax.Array:
    team_direction = state.attack_direction[state.players.team_id]
    depth = stadium.half_length + team_direction * state.ball.position[0]
    return (
        (depth >= 0.0)
        & (depth <= stadium.penalty_area_length)
        & (jnp.abs(state.ball.position[1]) <= 0.5 * stadium.penalty_area_width)
    )


def verified_controlled_carrier(
    state: State,
    ball_geometry: Ball = Ball(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
) -> jax.Array:
    """Return the physically verified possession actor, or ``NO_PLAYER``.

    A stored possession identity is authoritative only while its active actor
    still belongs to the stored team and controls the ball within the carry
    radius and control height.  A retained chest trap uses the actor's reach
    height; ordinary foot control uses the configured pelvis height.
    """

    players = state.players
    player_count = players.position.shape[0]
    carrier_in_range = (
        (state.possession.player >= 0)
        & (state.possession.player < player_count)
        & (state.possession.control_ticks > 0)
    )
    safe_carrier = jnp.clip(state.possession.player, 0, player_count - 1)
    carrier_distance = _norm(state.ball.position[:2] - players.position[safe_carrier])
    retained_body_trap = (
        carrier_in_range
        & (state.possession.last_contact.actor == safe_carrier)
        & (state.possession.last_contact.intent == INTENT_CONTROL)
        & (state.possession.last_contact.outcome == OUTCOME_TRAP)
        & (state.possession.last_contact.mechanism == MECHANISM_CHEST)
    )
    carrier_control_height = jnp.where(
        retained_body_trap,
        players.reach_height[safe_carrier],
        players.height[safe_carrier] * scale.pelvis_height_factor,
    )
    carrier_controls = (
        carrier_in_range
        & (state.possession.team == players.team_id[safe_carrier])
        & players.active[safe_carrier]
        & (carrier_distance <= reach.carry_radius_m + ball_geometry.radius)
        & (state.ball.position[2] <= carrier_control_height + ball_geometry.radius)
    )
    return jnp.where(
        carrier_controls,
        state.possession.player,
        NO_PLAYER,
    ).astype(jnp.int32)


def fresh_trap_control_grace(state: State) -> jax.Array:
    """Keep a successful trap visible through its first causal decision.

    The flag preserves only the stored possession identity.  It deliberately
    does not widen :func:`verified_controlled_carrier`: tackle context and all
    physical contact gates therefore continue to use the strict carry radius.
    ``control_ticks == 1`` spans the remaining physics substeps of the contact
    frame and the one following policy frame in which the actor can first
    observe and react to the successful trap.
    """

    player_count = state.players.position.shape[0]
    player = state.possession.player
    player_valid = (player >= 0) & (player < player_count)
    safe_player = jnp.clip(player, 0, player_count - 1)
    return (
        state.ball.live
        & (state.restart.kind == jnp.int32(RK_NONE))
        & player_valid
        & (state.possession.control_ticks == jnp.int32(1))
        & (state.possession.last_contact.actor == player)
        & (state.possession.last_contact.intent == jnp.int32(INTENT_CONTROL))
        & (state.possession.last_contact.outcome == jnp.int32(OUTCOME_TRAP))
        & state.players.active[safe_player]
        & (state.players.team_id[safe_player] == state.possession.team)
    )


def opponent_control_continuation(
    state: State,
    has_verified_carrier: jax.Array,
) -> jax.Array:
    """Return per-player eligibility to challenge a just-loosened trap.

    A physically retained carrier remains a tackle context. This predicate
    covers only the narrow continuation where an opponent's latest deliberate
    ``CONTROL/TRAP`` has already become physically loose, without promoting
    passive deflections or other historical possession into challenges.
    """

    player_count = state.players.position.shape[0]
    last_actor = state.possession.last_contact.actor
    last_actor_valid = (last_actor >= 0) & (last_actor < player_count)
    safe_last_actor = jnp.clip(last_actor, 0, player_count - 1)
    return (
        (~jnp.asarray(has_verified_carrier, dtype=jnp.bool_))
        & state.ball.live
        & last_actor_valid
        & (state.possession.last_contact.intent == INTENT_CONTROL)
        & (state.possession.last_contact.outcome == OUTCOME_TRAP)
        & (state.players.team_id != state.players.team_id[safe_last_actor])
    )


def evaluate_contact_predicates(
    state: State,
    restart_release_allowed: jax.Array,
    *,
    requested_intent: jax.Array,
    ball_geometry: Ball = Ball(),
    stadium: Stadium = Stadium(),
    reach: Reach = Reach(),
    scale: ActionScale = ActionScale(),
    body: BodyContact = BodyContact(),
) -> ContactPredicates:
    """Evaluate current contact eligibility and the action-gated mechanism."""

    players = state.players
    player_count = players.position.shape[0]
    restart_release_allowed = jnp.broadcast_to(
        jnp.asarray(restart_release_allowed, dtype=jnp.bool_),
        (player_count,),
    )
    requested_intent = jnp.asarray(requested_intent, dtype=jnp.int32)
    if requested_intent.shape != (player_count,):
        raise ValueError(
            "requested_intent must have shape "
            f"({player_count},), got {requested_intent.shape}"
        )
    intent_in_range = (requested_intent >= INTENT_MOVE) & (
        requested_intent < ACTION_INTENT_COUNT
    )
    safe_intent = jnp.clip(requested_intent, INTENT_MOVE, ACTION_INTENT_COUNT - 1)
    distance_xy = _norm(state.ball.position[:2] - players.position)

    verified_carrier = verified_controlled_carrier(
        state,
        ball_geometry,
        reach,
        scale,
    )
    safe_carrier = jnp.clip(state.possession.player, 0, player_count - 1)
    carrier_controls = verified_carrier != NO_PLAYER
    opposing_carrier = carrier_controls & (
        players.team_id != players.team_id[safe_carrier]
    )

    ball_height = state.ball.position[2]
    pelvis_height = players.height * scale.pelvis_height_factor
    torso_top_height = body.torso_top_height(players.height)
    height_mechanism = jnp.where(
        ball_height <= pelvis_height + ball_geometry.radius,
        MECHANISM_FOOT,
        jnp.where(
            ball_height <= torso_top_height + ball_geometry.radius,
            MECHANISM_CHEST,
            MECHANISM_HEAD,
        ),
    ).astype(jnp.int32)

    foot_height = ball_height <= pelvis_height + ball_geometry.radius
    horizontal_ball_speed = _norm(state.ball.velocity[:2])
    foot_speed_allowed = (
        horizontal_ball_speed + reach.height_speed_penalty_mps_per_m * ball_height
        <= reach.block_speed_limit_mps
    )
    restart_active = state.restart.kind != RK_NONE
    restart_mechanism = jnp.where(
        state.restart.kind == RK_THROWIN, MECHANISM_THROW, MECHANISM_FOOT
    )
    goalkeeper_in_own_penalty_area = (
        (~restart_active)
        & state.ball.live
        & players.is_goalkeeper
        & _in_own_penalty_area(state, stadium)
    )
    hand_restricted = goalkeeper_hand_restricted_mask(state)
    goalkeeper_hand = (
        goalkeeper_in_own_penalty_area
        & (~hand_restricted)
        & ((safe_intent == INTENT_CONTROL) | (safe_intent == INTENT_CLEAR))
    )
    # A challenge is contest context, not a physical body part. Preserve the
    # physical mechanism while the separate mask supplies wider duel reach.
    mechanism = jnp.where(
        restart_active,
        restart_mechanism,
        jnp.where(
            goalkeeper_hand,
            MECHANISM_GOALKEEPER_HAND,
            height_mechanism,
        ),
    ).astype(jnp.int32)

    last_actor = state.possession.last_contact.actor
    last_actor_valid = (last_actor >= 0) & (last_actor < player_count)
    safe_last_actor = jnp.clip(last_actor, 0, player_count - 1)
    last_intent = state.possession.last_contact.intent
    opponent_deliberate_release = (
        last_actor_valid
        & (state.possession.last_contact.outcome == OUTCOME_RELEASE)
        & (
            (last_intent == INTENT_PASS)
            | (last_intent == INTENT_SHOT)
            | (last_intent == INTENT_CLEAR)
        )
        & state.ball.live
        & (_norm(state.ball.velocity) > STATIONARY_SPEED_EPS)
        & (players.team_id != players.team_id[safe_last_actor])
    )
    control_continuation = opponent_control_continuation(state, carrier_controls)
    opponent_interceptable_play = opponent_deliberate_release | control_continuation
    challenge_interception = (
        (~restart_active)
        & (safe_intent == INTENT_CHALLENGE)
        & opponent_interceptable_play
        & (~opposing_carrier)
    )
    challenge_context = (
        (~restart_active)
        & (safe_intent == INTENT_CHALLENGE)
        & (opposing_carrier | opponent_interceptable_play)
        & (~goalkeeper_hand)
    )

    control_request = (
        (~restart_active)
        & ((safe_intent == INTENT_CONTROL) | (safe_intent == INTENT_CHALLENGE))
        & (mechanism == MECHANISM_FOOT)
    )

    is_goalkeeper_hand = mechanism == MECHANISM_GOALKEEPER_HAND
    contact_radius = jnp.where(
        is_goalkeeper_hand,
        reach.goalkeeper_radius_m,
        jnp.where(challenge_context, reach.challenge_radius_m, reach.carry_radius_m),
    )
    horizontal_reach = distance_xy <= contact_radius + ball_geometry.radius

    chest_height = (ball_height >= pelvis_height - ball_geometry.radius) & (
        ball_height <= torso_top_height + ball_geometry.radius
    )
    head_height = (ball_height >= torso_top_height - ball_geometry.radius) & (
        ball_height <= players.reach_height + ball_geometry.radius
    )
    goalkeeper_height = ball_height <= players.reach_height + ball_geometry.radius
    height_allowed = jnp.where(
        mechanism == MECHANISM_FOOT,
        foot_height,
        jnp.where(
            mechanism == MECHANISM_CHEST,
            chest_height,
            jnp.where(
                mechanism == MECHANISM_HEAD,
                head_height,
                jnp.where(is_goalkeeper_hand, goalkeeper_height, foot_height),
            ),
        ),
    )
    speed_allowed = (mechanism != MECHANISM_FOOT) | foot_speed_allowed

    # Reach effort is inferred at the exact physical state evaluated by this
    # predicate. During swept contact resolution that state is the first entry
    # into the reach volume, so no sampled pose or extra dive action is needed.
    # Subtracting the ball radius measures the player's required extension to
    # the near surface rather than to the ball centre.
    horizontal_extension = jnp.maximum(distance_xy - ball_geometry.radius, 0.0)
    goalkeeper_horizontal_effort = _extension_fraction(
        horizontal_extension,
        jnp.asarray(reach.goalkeeper_standing_radius_m, dtype=distance_xy.dtype),
        jnp.asarray(reach.goalkeeper_radius_m, dtype=distance_xy.dtype),
    )
    ball_bottom_height = ball_height - ball_geometry.radius
    active_vertical_effort = _extension_fraction(
        ball_bottom_height,
        players.height,
        players.reach_height,
    )
    athletic_mechanism = (mechanism == MECHANISM_HEAD) | is_goalkeeper_hand
    horizontal_reach_effort = jnp.where(
        is_goalkeeper_hand, goalkeeper_horizontal_effort, 0.0
    )
    vertical_reach_effort = jnp.where(
        athletic_mechanism, active_vertical_effort, 0.0
    )
    athletic_reach_effort = jnp.where(
        is_goalkeeper_hand,
        jnp.maximum(horizontal_reach_effort, vertical_reach_effort),
        jnp.where(mechanism == MECHANISM_HEAD, vertical_reach_effort, 0.0),
    )

    designated_restart = restart_actor_mask(state)
    structural_intents = restart_intent_allowed(state, restart_release_allowed)
    selected_structural_intent = (
        intent_in_range
        & structural_intents[jnp.arange(player_count, dtype=jnp.int32), safe_intent]
    )
    mechanism_allowed = (
        (
            (safe_intent == INTENT_CONTROL)
            & (
                (mechanism == MECHANISM_FOOT)
                | (mechanism == MECHANISM_CHEST)
                | (mechanism == MECHANISM_HEAD)
                | is_goalkeeper_hand
            )
        )
        | (safe_intent == INTENT_PASS)
        | (safe_intent == INTENT_SHOT)
        | (safe_intent == INTENT_CLEAR)
        | ((safe_intent == INTENT_CHALLENGE) & challenge_context)
    )
    intent_allowed = selected_structural_intent & mechanism_allowed
    phase_allowed = intent_allowed
    release_without_reach = (
        designated_restart
        & restart_release_allowed
        & players.is_goalkeeper
        & (state.restart.kind == RK_GK_HOLD)
    )
    horizontal_reach = horizontal_reach | release_without_reach
    height_allowed = height_allowed | release_without_reach
    speed_allowed = speed_allowed | release_without_reach

    recovery_ready = (
        (players.aerial_recovery_substeps <= 0)
        & ((~challenge_context) | (players.challenge_recovery_substeps <= 0))
        & (players.contact_lock_substeps <= 0)
        & (players.possession_loss_lock_substeps <= 0)
    )
    potential = (
        players.active
        & phase_allowed
        & horizontal_reach
        & height_allowed
        & speed_allowed
    )
    possible_now = potential & recovery_ready
    return ContactPredicates(
        mechanism=mechanism,
        challenge_context=challenge_context,
        control_request=control_request,
        goalkeeper_claim=goalkeeper_hand,
        goalkeeper_hand_clear=goalkeeper_hand & (safe_intent == INTENT_CLEAR),
        intent_allowed=intent_allowed,
        challenge_interception=challenge_interception,
        designated_restart=designated_restart,
        phase_allowed=phase_allowed,
        horizontal_reach=horizontal_reach,
        height_allowed=height_allowed,
        speed_allowed=speed_allowed,
        recovery_ready=recovery_ready,
        potential=potential,
        possible_now=possible_now,
        verified_controlled_carrier=verified_carrier,
        distance_xy=distance_xy,
        horizontal_reach_effort=horizontal_reach_effort,
        vertical_reach_effort=vertical_reach_effort,
        athletic_reach_effort=athletic_reach_effort,
    )
