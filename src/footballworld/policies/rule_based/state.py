"""Small recurrent state for the observation-only rule policy.

The policy owns tactical memory, not environment truth.  Every recurrent
quantity therefore has an observer axis: a view-limited actor may remember
what it previously observed, but it cannot inherit another actor's current
possession observation.  Formation anchors are fixed slot semantics captured
from each slot's own observation row at policy initialization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER, NO_TEAM, RK_NONE, TEAM_0, TEAM_1
from footballworld.core.contact import (
    INTENT_CONTROL,
    INTENT_PASS,
    OUTCOME_RELEASE,
    OUTCOME_TRAP,
)
from footballworld.environment.management import PlayerTacticalObservation
from footballworld.environment.observation import Observation, RosterMetadata
from footballworld.environment.tactics import (
    ROLE_CENTRE_BACK,
    ROLE_CENTRE_FORWARD,
    ROLE_CENTRE_MIDFIELDER,
    ROLE_COUNT,
    ROLE_FULL_BACK,
    ROLE_GOALKEEPER,
    ROLE_WIDE_FORWARD,
    ROLE_WIDE_MIDFIELDER,
    classify_formation_roles,
)
from footballworld.policies.rule_based.opponent_pool import build_team_slot_table
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    tactical_plan_code,
)

ROLE_NAMES = ("GK", "CB", "FB", "CM", "WM", "CF", "WF")

INACTIVE_AGE = -1
_INT32_MAX = 2_147_483_647


class RulePolicyState(NamedTuple):
    """Fixed-shape formation semantics and causal per-observer memory.

    Ages are measured in environment control ticks.  ``INACTIVE_AGE`` means
    that no corresponding episode is active; zero identifies the first
    observed tick of a restart, controlled-possession episode, or counterpress.
    ``possession_team`` is each observer's most recently known team and may be
    retained while its current possession observation is hidden.  Its age also
    bridges a visible, kick-applied PASS release by that same team and a visible
    CONTROL/TRAP lineage whose last actor is still the remembered carrier while
    the ball remains live. Other known loose-ball phases do not receive that
    inference. A visible restart PASS bootstraps the same bridge from its public
    last-contact actor because restart possession begins at NO_TEAM.
    """

    formation_anchor: jax.Array
    role: jax.Array
    team_tactical_plan: jax.Array
    team_slot_index: jax.Array
    team_slot_valid: jax.Array
    restart_kind: jax.Array
    restart_age: jax.Array
    possession_team: jax.Array
    possession_age: jax.Array
    carrier_age: jax.Array
    attack_phase: jax.Array
    current_possessor: jax.Array
    previous_possessor: jax.Array
    counterpress_age: jax.Array
    loose_chaser: jax.Array
    planned_receiver: jax.Array
    planned_receiver_id: jax.Array
    planned_arrival: jax.Array
    planned_eta_ticks: jax.Array
    service_opportunity: jax.Array
    secure_control_age: jax.Array
    last_control_tick: jax.Array


def _player_count(
    observations: Observation,
    roster: RosterMetadata,
) -> int:
    """Validate only the static axes needed by policy state construction."""

    player_count = observations.self_state.player_index.shape[0]
    if observations.self_state.position.shape != (player_count, 2):
        raise ValueError("observations must have one self row per roster slot")
    if observations.players.relative_position.shape != (
        player_count,
        player_count,
        2,
    ):
        raise ValueError("observations must have leading observer and roster axes")
    if observations.restart.kind.shape != (player_count,):
        raise ValueError("restart observations must have an observer axis")
    if observations.possession.team.shape != (player_count,):
        raise ValueError("possession observations must have an observer axis")
    if observations.match.control_tick.shape != (player_count,):
        raise ValueError("match observations must have an observer axis")
    if roster.team_id.shape != (player_count,):
        raise ValueError("roster must describe the observation roster axis")
    if roster.is_goalkeeper.shape != (player_count,):
        raise ValueError("roster goalkeeper flags must match the roster axis")
    return player_count


def _classify_roles(
    formation_anchor: jax.Array,
    team_id: jax.Array,
    is_goalkeeper: jax.Array,
) -> jax.Array:
    """Derive roles from team-relative depth lines and within-line width.

    Distinct depth bands become defence, midfield, and attack.  Within a line
    of at least three outfield players, its two widest slots receive the wide
    variant.  This makes the result formation-relative and invariant to slot
    numbering, team world orientation, and pitch dimensions.
    """

    return classify_formation_roles(formation_anchor, team_id, is_goalkeeper)


def _advance_age(age: jax.Array, elapsed: jax.Array) -> jax.Array:
    """Advance a non-negative int32 age without exposing integer wraparound."""

    remaining = jnp.int32(_INT32_MAX) - elapsed
    return jnp.where(age > remaining, jnp.int32(_INT32_MAX), age + elapsed)


def _next_carrier_age(
    age: jax.Array,
    current_actor: jax.Array,
    observed_actor: jax.Array,
    observed_control_ticks: jax.Array,
    actor_known: jax.Array,
    same_possession: jax.Array,
    possession_known: jax.Array,
    same_actor_control_lineage: jax.Array,
    elapsed: jax.Array,
) -> jax.Array:
    """Track one player's carry across observable CONTROL/TRAP recontacts.

    Environment control_ticks describes an uninterrupted physical control
    segment. A deliberate dribble can remain loose for several control frames,
    so that counter legitimately restarts even though the same player is still
    the tactical carrier. Policy carry age has the latter meaning: it advances
    only when the observer still sees the same player for the same team, and
    resets on a visible handoff or loss.
    """

    observed_age = jnp.maximum(
        jnp.asarray(observed_control_ticks, dtype=jnp.int32) - jnp.int32(1),
        jnp.int32(0),
    )
    same_carrier = (
        actor_known
        & same_possession
        & (age >= 0)
        & (current_actor != NO_PLAYER)
        & (observed_actor == current_actor)
    )
    continued_age = jnp.maximum(
        _advance_age(jnp.maximum(age, 0), elapsed),
        observed_age,
    )
    visible_age = jnp.where(
        actor_known,
        jnp.where(same_carrier, continued_age, observed_age),
        jnp.where(
            same_actor_control_lineage & (age >= 0),
            continued_age,
            jnp.int32(INACTIVE_AGE),
        ),
    ).astype(jnp.int32)
    # Hidden rows cannot prove elapsed control, but visibility loss must not
    # erase remembered tenure and reopen the soft-limit loophole.
    return jnp.where(possession_known, visible_age, age).astype(jnp.int32)


def initialize_rule_policy_state(
    observations: Observation,
    roster: RosterMetadata,
    *,
    team_tactical_plan: jax.Array | None = None,
) -> RulePolicyState:
    """Initialize formation semantics and memory from ``observe_all_si`` output.

    Each slot's own ``self_state.position`` is already expressed in that
    slot's attacking coordinate frame, so anchors need neither a private
    rollout state nor information from a different observer row.
    """

    player_count = _player_count(observations, roster)
    anchor = jnp.asarray(observations.self_state.position, dtype=jnp.float32)
    role = _classify_roles(anchor, roster.team_id, roster.is_goalkeeper)
    team_slots = build_team_slot_table(roster.team_id)
    restart_kind = jnp.asarray(observations.restart.kind, dtype=jnp.int32)
    restart_active = restart_kind != RK_NONE
    possession_known = jnp.asarray(observations.possession.known, dtype=jnp.bool_)
    possession_team = jnp.where(
        possession_known,
        observations.possession.team,
        jnp.int32(NO_TEAM),
    ).astype(jnp.int32)
    controlled = possession_known & (possession_team != NO_TEAM)
    possessor_flag = jnp.asarray(observations.players.possessor, dtype=jnp.bool_)
    has_possessor = controlled & jnp.any(possessor_flag, axis=-1)
    current_possessor = jnp.where(
        has_possessor, jnp.argmax(possessor_flag, axis=-1), jnp.int32(NO_PLAYER)
    ).astype(jnp.int32)

    team_tactical_plan = (
        jnp.full(
            (2,), tactical_plan_code(TacticalPlan.JUEGO_DE_POSICION), dtype=jnp.int32
        )
        if team_tactical_plan is None
        else jnp.asarray(team_tactical_plan, dtype=jnp.int32)
    )
    if team_tactical_plan.shape != (2,):
        raise ValueError("team_tactical_plan must have shape (2,)")
    return RulePolicyState(
        formation_anchor=anchor,
        role=role,
        team_tactical_plan=team_tactical_plan,
        team_slot_index=team_slots.index,
        team_slot_valid=team_slots.valid,
        restart_kind=restart_kind,
        restart_age=jnp.where(restart_active, jnp.int32(0), jnp.int32(INACTIVE_AGE)),
        possession_team=possession_team,
        possession_age=jnp.where(controlled, jnp.int32(0), jnp.int32(INACTIVE_AGE)),
        carrier_age=jnp.where(
            has_possessor,
            jnp.maximum(
                jnp.asarray(observations.possession.control_ticks, dtype=jnp.int32)
                - jnp.int32(1),
                jnp.int32(0),
            ),
            jnp.int32(INACTIVE_AGE),
        ),
        attack_phase=jnp.where(controlled, jnp.int32(0), jnp.int32(INACTIVE_AGE)),
        current_possessor=current_possessor,
        previous_possessor=jnp.full((player_count,), NO_PLAYER, dtype=jnp.int32),
        counterpress_age=jnp.full((player_count,), INACTIVE_AGE, dtype=jnp.int32),
        loose_chaser=jnp.full((player_count,), NO_PLAYER, dtype=jnp.int32),
        planned_receiver=jnp.full((player_count,), NO_PLAYER, dtype=jnp.int32),
        planned_receiver_id=jnp.full((player_count,), NO_PLAYER, dtype=jnp.int32),
        planned_arrival=jnp.zeros((player_count, 2), dtype=jnp.float32),
        planned_eta_ticks=jnp.zeros((player_count,), dtype=jnp.int32),
        service_opportunity=jnp.zeros((player_count,), dtype=jnp.bool_),
        secure_control_age=jnp.full((player_count,), INACTIVE_AGE, dtype=jnp.int32),
        last_control_tick=jnp.asarray(observations.match.control_tick, dtype=jnp.int32),
    )


def update_rule_policy_state(
    state: RulePolicyState,
    observations: Observation,
    roster: RosterMetadata,
) -> RulePolicyState:
    """Advance policy memory using only each actor's current observation row.

    Unknown possession preserves the prior team episode and freezes carrier
    age. A fully observed loose ball bridges the prior episode only when public
    last-contact provenance proves a live, kick-applied same-team PASS release
    or the same remembered actor's CONTROL/TRAP lineage. A newly observed loss
    from the actor's own team starts counterpress age zero; it remains active
    through a loose or hidden ball and ends when own-team possession returns.
    """

    player_count = _player_count(observations, roster)
    if state.formation_anchor.shape != (player_count, 2):
        raise ValueError("policy state does not match the observation roster")
    if state.role.shape != (player_count,):
        raise ValueError("policy role state does not match the roster")
    if state.current_possessor.shape != (player_count,):
        raise ValueError("policy current possessor must match the roster")
    if state.carrier_age.shape != (player_count,):
        raise ValueError("policy carrier age must match the roster")
    if state.previous_possessor.shape != (player_count,):
        raise ValueError("policy previous possessor must match the roster")
    if state.loose_chaser.shape != (player_count,):
        raise ValueError("policy loose-ball chaser must match the roster")
    if state.planned_receiver.shape != (player_count,):
        raise ValueError("policy planned receiver must match the roster")
    if state.planned_receiver_id.shape != (player_count,):
        raise ValueError("policy planned receiver identity must match the roster")
    if state.planned_arrival.shape != (player_count, 2):
        raise ValueError("policy planned arrival must match the roster")
    if state.planned_eta_ticks.shape != (player_count,):
        raise ValueError("policy planned ETA must match the roster")
    if state.service_opportunity.shape != (player_count,):
        raise ValueError("policy service opportunity must match the roster")
    if state.secure_control_age.shape != (player_count,):
        raise ValueError("policy secure-control age must match the roster")
    if state.team_tactical_plan.shape != (2,):
        raise ValueError("policy tactical plan table must have shape (2,)")
    if state.team_slot_index.shape != (2, 11):
        raise ValueError("policy team-slot table must have shape (2, 11)")
    if state.team_slot_valid.shape != (2, 11):
        raise ValueError("policy team-slot validity must have shape (2, 11)")

    current_tick = jnp.asarray(observations.match.control_tick, dtype=jnp.int32)
    elapsed = jnp.maximum(current_tick - state.last_control_tick, 0).astype(jnp.int32)

    current_restart = jnp.asarray(observations.restart.kind, dtype=jnp.int32)
    restart_active = current_restart != RK_NONE
    same_restart = restart_active & (current_restart == state.restart_kind)
    continued_restart_age = _advance_age(jnp.maximum(state.restart_age, 0), elapsed)
    restart_age = jnp.where(
        restart_active,
        jnp.where(same_restart, continued_restart_age, jnp.int32(0)),
        jnp.int32(INACTIVE_AGE),
    )

    known = jnp.asarray(observations.possession.known, dtype=jnp.bool_)
    current_possession = jnp.asarray(observations.possession.team, dtype=jnp.int32)
    current_controlled = known & (current_possession != NO_TEAM)
    last_contact = observations.possession.last_contact
    last_actor_flag = jnp.asarray(observations.players.last_actor, dtype=jnp.bool_)
    # ``last_actor`` is a public one-hot identity.  Do not let argmax turn a
    # malformed multi-hot or empty row into an invented releasing player.
    last_actor_known = jnp.sum(last_actor_flag, axis=-1) == 1
    last_actor_index = jnp.argmax(last_actor_flag, axis=-1).astype(jnp.int32)
    last_actor_team = roster.team_id[last_actor_index]
    last_actor_known = last_actor_known & (
        (last_actor_team == TEAM_0) | (last_actor_team == TEAM_1)
    )
    pass_release_flight = (
        known
        & (~current_controlled)
        & observations.ball.live
        & (observations.restart.kind == RK_NONE)
        & last_contact.known
        & (last_contact.intent == INTENT_PASS)
        & (last_contact.outcome == OUTCOME_RELEASE)
        & last_contact.kick_applied
    )
    continued_pass_flight = (
        pass_release_flight
        & (observations.possession.previous_team == state.possession_team)
        & (state.possession_team != NO_TEAM)
    )
    # A restart begins without controlled possession, so the ordinary bridge
    # above has no prior ``possession_team`` to retain.  Public last-contact
    # provenance identifies a PASS made as a restart, while the public
    # ``last_actor`` flag plus roster metadata identifies its releasing side.
    # This lets that team build one stable receive plan without consulting
    # environment-private state. ``previous_team`` is intentionally not used:
    # a restart begins from NO_TEAM possession, so it is also NO_TEAM here.
    restart_pass_flight = (
        pass_release_flight
        & (last_contact.restart_kind != RK_NONE)
        & last_actor_known
        & (last_actor_team != NO_TEAM)
    )
    reliable_pass_flight = continued_pass_flight | restart_pass_flight
    same_actor_control_lineage = (
        known
        & (~current_controlled)
        & observations.ball.live
        & (observations.restart.kind == RK_NONE)
        & last_contact.known
        & (last_contact.intent == INTENT_CONTROL)
        & (last_contact.outcome == OUTCOME_TRAP)
        & (~last_contact.kick_applied)
        & last_actor_known
        & (state.current_possessor != NO_PLAYER)
        & (last_actor_index == state.current_possessor)
        & (last_actor_team == state.possession_team)
        & (state.carrier_age >= 0)
    )
    reliable_possession_lineage = reliable_pass_flight | same_actor_control_lineage
    pass_flight_team = jnp.where(
        continued_pass_flight,
        state.possession_team,
        last_actor_team,
    ).astype(jnp.int32)
    lineage_team = jnp.where(
        same_actor_control_lineage, state.possession_team, pass_flight_team
    ).astype(jnp.int32)
    restart_pass_started = restart_pass_flight & (
        (state.possession_team == NO_TEAM) | (state.possession_age < 0)
    )
    same_possession = current_controlled & (current_possession == state.possession_team)
    continued_possession_age = _advance_age(
        jnp.maximum(state.possession_age, 0), elapsed
    )
    observed_possession_age = jnp.where(
        current_controlled,
        jnp.where(same_possession, continued_possession_age, jnp.int32(0)),
        jnp.where(
            reliable_possession_lineage,
            jnp.where(
                restart_pass_started,
                jnp.int32(0),
                continued_possession_age,
            ),
            jnp.int32(INACTIVE_AGE),
        ),
    )
    hidden_possession_age = jnp.where(
        state.possession_age >= 0,
        continued_possession_age,
        jnp.int32(INACTIVE_AGE),
    )
    possession_age = jnp.where(known, observed_possession_age, hidden_possession_age)
    possession_team = jnp.where(
        known,
        jnp.where(reliable_possession_lineage, lineage_team, current_possession),
        state.possession_team,
    ).astype(jnp.int32)
    possessor_flag = jnp.asarray(observations.players.possessor, dtype=jnp.bool_)
    actor_known = current_controlled & jnp.any(possessor_flag, axis=-1)
    observed_actor = jnp.argmax(possessor_flag, axis=-1).astype(jnp.int32)
    actor_changed = (
        actor_known
        & same_possession
        & (state.current_possessor != NO_PLAYER)
        & (observed_actor != state.current_possessor)
    )
    carrier_age = _next_carrier_age(
        state.carrier_age,
        state.current_possessor,
        observed_actor,
        observations.possession.control_ticks,
        actor_known,
        same_possession,
        known,
        same_actor_control_lineage,
        elapsed,
    )
    previous_possessor = jnp.where(
        restart_pass_started,
        jnp.int32(NO_PLAYER),
        jnp.where(
            current_controlled & (~same_possession),
            jnp.int32(NO_PLAYER),
            jnp.where(actor_changed, state.current_possessor, state.previous_possessor),
        ),
    ).astype(jnp.int32)
    current_possessor = jnp.where(
        actor_known,
        observed_actor,
        jnp.where(restart_pass_started, last_actor_index, state.current_possessor),
    ).astype(jnp.int32)

    # A possession plan has two bounded phases: setup and execution. It
    # advances only when this observer sees a different same-team possessor.
    # A second handoff completes the pattern and returns the possession to the
    # ordinary policy; the completed sentinel cannot restart until possession
    # changes. Repeated self touches and hidden rows cannot fabricate a pass.
    continued_attack_phase = state.attack_phase
    same_team_handoff = same_possession & actor_changed
    advanced_attack_phase = jnp.where(
        same_team_handoff & (state.attack_phase == 0),
        jnp.int32(1),
        jnp.where(
            same_team_handoff & (state.attack_phase == 1),
            jnp.int32(INACTIVE_AGE),
            state.attack_phase,
        ),
    )
    open_play = observations.ball.live & (current_restart == RK_NONE)
    observed_attack_phase = jnp.where(
        current_controlled,
        jnp.where(
            same_possession,
            advanced_attack_phase,
            jnp.int32(0),
        ),
        jnp.where(
            reliable_possession_lineage,
            jnp.where(restart_pass_started, jnp.int32(0), continued_attack_phase),
            jnp.int32(INACTIVE_AGE),
        ),
    )
    hidden_attack_phase = jnp.where(
        state.attack_phase >= 0,
        continued_attack_phase,
        jnp.int32(INACTIVE_AGE),
    )
    attack_phase = jnp.where(
        open_play,
        jnp.where(known, observed_attack_phase, hidden_attack_phase),
        jnp.int32(INACTIVE_AGE),
    )

    self_index = observations.self_state.player_index.astype(jnp.int32)
    own_team = roster.team_id[self_index]
    plan_row = jnp.arange(player_count, dtype=jnp.int32)
    safe_planned_receiver = jnp.clip(state.planned_receiver, 0, player_count - 1)
    planned_slot_present = (state.planned_receiver >= 0) & (
        state.planned_receiver < player_count
    )
    planned_state_valid = (
        planned_slot_present
        & observations.valid
        & observations.ball.visible
        & (roster.player_id[safe_planned_receiver] == state.planned_receiver_id)
        & (roster.team_id[safe_planned_receiver] == own_team)
        & observations.players.on_pitch[plan_row, safe_planned_receiver]
        & (~observations.players.sent_off[plan_row, safe_planned_receiver])
        & observations.players.visible[plan_row, safe_planned_receiver]
    )
    retain_planned_pass = (
        reliable_pass_flight & planned_state_valid & (state.planned_eta_ticks > 0)
    )
    possession_lost = (
        known
        & (~reliable_possession_lineage)
        & (state.possession_team == own_team)
        & (current_possession != own_team)
    )
    own_possession_observed = known & (
        (current_possession == own_team)
        | (same_actor_control_lineage & (state.possession_team == own_team))
    )
    continued_counterpress_age = _advance_age(
        jnp.maximum(state.counterpress_age, 0), elapsed
    )
    counterpress_age = jnp.where(
        possession_lost,
        jnp.int32(0),
        jnp.where(
            own_possession_observed,
            jnp.int32(INACTIVE_AGE),
            jnp.where(
                state.counterpress_age >= 0,
                continued_counterpress_age,
                jnp.int32(INACTIVE_AGE),
            ),
        ),
    )

    # A completed loose-ball CONTROL starts a short, observer-local secure
    # phase.  The memory is deliberately fail-closed: hidden possession or a
    # different observed actor clears it, and no other observer row is used to
    # fill the gap.  ``loose_chaser`` is written by the preceding policy
    # decision, so acquisition can be identified without engine-private state.
    acquired_from_loose = (
        actor_known
        & observations.ball.visible
        & open_play
        & (
            (planned_state_valid & (observed_actor == state.planned_receiver))
            | (
                (state.loose_chaser != NO_PLAYER)
                & (observed_actor == state.loose_chaser)
            )
        )
    )
    continued_secure_control = (
        actor_known
        & (state.secure_control_age >= 0)
        & (observed_actor == state.current_possessor)
    )
    secure_control_age = jnp.where(
        acquired_from_loose,
        jnp.int32(0),
        jnp.where(
            continued_secure_control,
            _advance_age(state.secure_control_age, elapsed),
            jnp.int32(INACTIVE_AGE),
        ),
    )

    return RulePolicyState(
        formation_anchor=state.formation_anchor,
        role=state.role,
        team_tactical_plan=state.team_tactical_plan,
        team_slot_index=state.team_slot_index,
        team_slot_valid=state.team_slot_valid,
        restart_kind=current_restart,
        restart_age=restart_age,
        possession_team=possession_team,
        possession_age=possession_age,
        carrier_age=carrier_age,
        attack_phase=attack_phase.astype(jnp.int32),
        current_possessor=current_possessor,
        previous_possessor=previous_possessor,
        counterpress_age=counterpress_age,
        loose_chaser=state.loose_chaser,
        planned_receiver=jnp.where(
            retain_planned_pass, state.planned_receiver, jnp.int32(NO_PLAYER)
        ),
        planned_receiver_id=jnp.where(
            retain_planned_pass, state.planned_receiver_id, jnp.int32(NO_PLAYER)
        ),
        planned_arrival=jnp.where(
            retain_planned_pass[:, None], state.planned_arrival, 0.0
        ),
        planned_eta_ticks=jnp.where(
            retain_planned_pass,
            jnp.maximum(state.planned_eta_ticks - elapsed, 0),
            jnp.int32(0),
        ),
        service_opportunity=state.service_opportunity,
        secure_control_age=secure_control_age,
        last_control_tick=current_tick,
    )


def apply_tactical_observation(
    state: RulePolicyState,
    tactics: PlayerTacticalObservation,
    roster: RosterMetadata,
) -> RulePolicyState:
    """Apply one team's visible formation command to policy shape targets."""

    player_count = state.formation_anchor.shape[0]
    if tactics.formation_anchor.shape != (player_count, 2):
        raise ValueError("tactical anchors must match policy roster slots")
    if tactics.formation_role.shape != (player_count,):
        raise ValueError("tactical roles must match policy roster slots")
    own = (roster.team_id == tactics.team) & tactics.valid
    return state._replace(
        formation_anchor=jnp.where(
            own[:, None], tactics.formation_anchor, state.formation_anchor
        ),
        role=jnp.where(own, tactics.formation_role, state.role),
    )


__all__ = [
    "INACTIVE_AGE",
    "ROLE_CENTRE_BACK",
    "ROLE_CENTRE_FORWARD",
    "ROLE_CENTRE_MIDFIELDER",
    "ROLE_COUNT",
    "ROLE_FULL_BACK",
    "ROLE_GOALKEEPER",
    "ROLE_NAMES",
    "ROLE_WIDE_FORWARD",
    "ROLE_WIDE_MIDFIELDER",
    "RulePolicyState",
    "apply_tactical_observation",
    "initialize_rule_policy_state",
    "update_rule_policy_state",
]
