"""One fixed-shape rules transition after a physics substep."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.body_foul import BodyFoul
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    BALL_EVENT_NONE,
    DISCIPLINE_NONE,
    NO_PLAYER,
    NO_TEAM,
    RK_NONE,
    RK_OFFSIDE,
    TEAM_1,
)
from footballworld.core.contact import (
    INTENT_MOVE,
    INTENT_SOURCE_NONE,
    LAW11_DEFLECTION_NO_RESET,
    MECHANISM_NONE,
    MECHANISM_PASSIVE_BODY,
    OUTCOME_DEFLECTION,
    ContactOccurrence,
)
from footballworld.core.state import RestartReleaseProvenance, State
from footballworld.rules.ball_boundary import (
    CROSSING_NONE,
    BoundaryCrossing,
    BoundaryEvent,
)
from footballworld.rules.boundary_resolution import resolve_boundary_crossing
from footballworld.rules.foul import (
    OFFENCE_NONE,
    SEVERITY_NONE,
    TACTICAL_NONE,
    FoulEvent,
    FoulFacts,
    adjudicate_foul,
    classify_existing_contact_discipline,
    facts_from_contest_foul,
    resolve_foul,
    sample_body_foul_facts,
)
from footballworld.rules.offside import (
    OffsideEvent,
    OffsideState,
    clear_offside_state,
    resolve_offside_challenge,
    resolve_offside_contacts,
)
from footballworld.rules.restart import select_restart_taker
from footballworld.rules.restart_retouch import (
    RestartRetouchEvent,
    detect_restart_retouch,
    resolve_restart_retouch,
)
from footballworld.rules.restart_spot import canonical_restart_spot


class RulesTransition(NamedTuple):
    """Authoritative state and mutually exclusive events for one substep."""

    state: State
    offside_state: OffsideState
    foul_event: FoulEvent
    retouch_event: RestartRetouchEvent
    offside_event: OffsideEvent
    boundary_event: BoundaryEvent
    body_impact_consumed: jax.Array


def _contacts_not_after_boundary(
    occurrences: ContactOccurrence,
    boundary: BoundaryCrossing,
) -> ContactOccurrence:
    before_or_at_boundary = (~boundary.occurred) | (
        occurrences.time_fraction <= boundary.time_fraction
    )
    return occurrences._replace(occurred=occurrences.occurred & before_or_at_boundary)


def _select_retouch_event(
    event: RestartRetouchEvent,
    selected: jax.Array,
) -> RestartRetouchEvent:
    zero_position = jnp.zeros_like(event.contact_position)
    return RestartRetouchEvent(
        occurred=selected,
        offender=jnp.where(selected, event.offender, NO_PLAYER).astype(jnp.int32),
        source_restart_kind=jnp.where(
            selected, event.source_restart_kind, RK_NONE
        ).astype(jnp.int32),
        team=jnp.where(selected, event.team, NO_TEAM).astype(jnp.int32),
        restart_kind=jnp.where(selected, event.restart_kind, RK_NONE).astype(jnp.int32),
        indirect=selected & event.indirect,
        goalkeeper_handling=selected & event.goalkeeper_handling,
        time_fraction=jnp.where(selected, event.time_fraction, 0.0),
        contact_position=jnp.where(selected, event.contact_position, zero_position),
        restart_position=jnp.where(selected, event.restart_position, zero_position),
    )


def _select_offside_event(
    event: OffsideEvent,
    selected: jax.Array,
) -> OffsideEvent:
    return OffsideEvent(
        occurred=selected,
        actor=jnp.where(selected, event.actor, NO_PLAYER).astype(jnp.int32),
        team=jnp.where(selected, event.team, NO_TEAM).astype(jnp.int32),
        position=jnp.where(selected, event.position, jnp.zeros_like(event.position)),
        time_fraction=jnp.where(selected, event.time_fraction, 0.0),
    )


def _select_foul_event(event: FoulEvent, selected: jax.Array) -> FoulEvent:
    return FoulEvent(
        occurred=selected,
        contest_source=selected & event.contest_source,
        offender=jnp.where(selected, event.offender, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(selected, event.victim, NO_PLAYER).astype(jnp.int32),
        offender_team=jnp.where(selected, event.offender_team, NO_TEAM).astype(
            jnp.int32
        ),
        offence_type=jnp.where(selected, event.offence_type, OFFENCE_NONE).astype(
            jnp.int32
        ),
        severity=jnp.where(selected, event.severity, SEVERITY_NONE).astype(jnp.int32),
        tactical_effect=jnp.where(
            selected, event.tactical_effect, TACTICAL_NONE
        ).astype(jnp.int32),
        position=jnp.where(selected, event.position, jnp.zeros_like(event.position)),
        time_fraction=jnp.where(selected, event.time_fraction, 0.0),
        restart_kind=jnp.where(selected, event.restart_kind, RK_NONE).astype(jnp.int32),
        discipline=jnp.where(selected, event.discipline, DISCIPLINE_NONE).astype(
            jnp.int32
        ),
        advantage_applied=selected & event.advantage_applied,
    )


def _select_chronological_foul(
    contest_facts: FoulFacts,
    body_facts: FoulFacts,
    contest_time_fraction: jax.Array,
    body_time_fraction: jax.Array,
) -> tuple[FoulFacts, jax.Array, jax.Array]:
    """Select one foul without reopening a duplicate body opportunity.

    A contest and body fact for the same ordered offender/victim pair describe
    the same direct-contact offence, so the contest owns that duplicate. For
    different pairs physical time orders the offences; an exact tie stays with
    the contest because both fact extractors currently emit the same direct-
    free-kick seriousness. The body draw is deliberately completed before
    this arbitration so its eligible impact can still be consumed.
    """

    contest_occurred = jnp.asarray(contest_facts.occurred, dtype=bool)
    body_occurred = jnp.asarray(body_facts.occurred, dtype=bool)
    same_pair = (
        contest_occurred
        & body_occurred
        & (contest_facts.offender == body_facts.offender)
        & (contest_facts.victim == body_facts.victim)
    )
    contest_precedes_or_ties = contest_time_fraction <= body_time_fraction
    contest_selected = contest_occurred & (
        (~body_occurred) | same_pair | contest_precedes_or_ties
    )
    selected_facts = jax.tree_util.tree_map(
        lambda contest_value, body_value: jnp.where(
            contest_selected, contest_value, body_value
        ),
        contest_facts,
        body_facts,
    )
    selected_time_fraction = jnp.where(
        contest_selected,
        contest_time_fraction,
        body_time_fraction,
    )
    return selected_facts, contest_selected, selected_time_fraction


def _restore_late_deliberate_recovery(
    pre_physics_state: State,
    post_physics_state: State,
    restore: jax.Array,
) -> State:
    """Remove recovery installed by a deliberate contact after a whistle.

    The ordinary per-substep countdown must still happen. Only the four
    recovery timers that a later deliberate contact can install are restored;
    locomotion, collision response, and stamina remain at the substep endpoint.
    """

    pre_players = pre_physics_state.players
    post_players = post_physics_state.players

    def no_contact_timer(value):
        return jnp.maximum(jnp.int32(0), value - jnp.int32(1))

    players = post_players._replace(
        challenge_recovery_substeps=jnp.where(
            restore,
            no_contact_timer(pre_players.challenge_recovery_substeps),
            post_players.challenge_recovery_substeps,
        ),
        contact_lock_substeps=jnp.where(
            restore,
            no_contact_timer(pre_players.contact_lock_substeps),
            post_players.contact_lock_substeps,
        ),
        aerial_recovery_substeps=jnp.where(
            restore,
            no_contact_timer(pre_players.aerial_recovery_substeps),
            post_players.aerial_recovery_substeps,
        ),
        possession_loss_lock_substeps=jnp.where(
            restore,
            no_contact_timer(pre_players.possession_loss_lock_substeps),
            post_players.possession_loss_lock_substeps,
        ),
    )
    return post_physics_state._replace(players=players)


def _masked_boundary(
    boundary: BoundaryCrossing,
    enabled: jax.Array,
) -> BoundaryCrossing:
    occurred = boundary.occurred & enabled
    return BoundaryCrossing(
        occurred=occurred,
        axis=jnp.where(occurred, boundary.axis, CROSSING_NONE).astype(jnp.int32),
        time_fraction=jnp.where(occurred, boundary.time_fraction, 0.0),
        position=jnp.where(
            occurred, boundary.position, jnp.zeros_like(boundary.position)
        ),
        through_goal=occurred & boundary.through_goal,
    )


def _empty_boundary_event(boundary: BoundaryCrossing) -> BoundaryEvent:
    return BoundaryEvent(
        occurred=jnp.bool_(False),
        kind=jnp.int32(BALL_EVENT_NONE),
        team=jnp.int32(NO_TEAM),
        scoring_team=jnp.int32(NO_TEAM),
        time_fraction=jnp.asarray(0.0, dtype=boundary.position.dtype),
        position=jnp.zeros_like(boundary.position),
    )


def _apply_offside_offence(
    state: State,
    event: OffsideEvent,
    *,
    stadium: Stadium,
    ball_geometry: Ball,
) -> State:
    restart_team = (TEAM_1 - event.team).astype(jnp.int32)
    restart_position = canonical_restart_spot(
        RK_OFFSIDE,
        restart_team,
        event.position,
        state.attack_direction,
        stadium=stadium,
        ball=ball_geometry,
    )
    taker = select_restart_taker(
        state,
        RK_OFFSIDE,
        restart_team,
        restart_position,
        stadium=stadium,
    )
    ball = state.ball._replace(
        position=restart_position,
        velocity=jnp.zeros_like(state.ball.velocity),
        spin=jnp.zeros_like(state.ball.spin),
        live=jnp.bool_(False),
    )
    possession = state.possession._replace(
        team=jnp.int32(NO_TEAM),
        player=jnp.int32(NO_PLAYER),
        previous_team=state.possession.team.astype(jnp.int32),
        control_ticks=jnp.int32(0),
    )
    restart = state.restart._replace(
        kind=jnp.int32(RK_OFFSIDE),
        team=restart_team,
        substeps_remaining=jnp.int32(0),
        taker=taker,
        indirect=jnp.bool_(True),
        opened_control_tick=state.control_tick,
    )
    cleared_release = RestartReleaseProvenance(
        active=jnp.bool_(False),
        untouched=jnp.bool_(False),
        kind=jnp.int32(RK_NONE),
        team=jnp.int32(NO_TEAM),
        taker=jnp.int32(NO_PLAYER),
        indirect=jnp.bool_(False),
        law11_direct_exempt=jnp.bool_(False),
        release_mechanism=jnp.int32(MECHANISM_NONE),
    )
    candidate = state._replace(
        ball=ball,
        possession=possession,
        restart=restart,
        restart_release=cleared_release,
        gk_backpass_team=jnp.int32(NO_TEAM),
    )
    return jax.tree_util.tree_map(
        lambda changed, current: jnp.where(event.occurred, changed, current),
        candidate,
        state,
    )


def _empty_rules_transition(
    state: State,
    offside_state: OffsideState,
    boundary: BoundaryCrossing,
) -> RulesTransition:
    """Return the canonical fixed-shape result when no rule input occurred."""

    dtype = state.ball.position.dtype
    position = jnp.zeros(3, dtype=dtype)
    return RulesTransition(
        state=state,
        offside_state=offside_state,
        foul_event=FoulEvent(
            occurred=jnp.bool_(False),
            contest_source=jnp.bool_(False),
            offender=jnp.int32(NO_PLAYER),
            victim=jnp.int32(NO_PLAYER),
            offender_team=jnp.int32(NO_TEAM),
            offence_type=jnp.int32(OFFENCE_NONE),
            severity=jnp.int32(SEVERITY_NONE),
            tactical_effect=jnp.int32(TACTICAL_NONE),
            position=position,
            time_fraction=jnp.asarray(0.0, dtype=dtype),
            restart_kind=jnp.int32(RK_NONE),
            discipline=jnp.int32(DISCIPLINE_NONE),
            advantage_applied=jnp.bool_(False),
        ),
        retouch_event=RestartRetouchEvent(
            occurred=jnp.bool_(False),
            offender=jnp.int32(NO_PLAYER),
            source_restart_kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            restart_kind=jnp.int32(RK_NONE),
            indirect=jnp.bool_(False),
            goalkeeper_handling=jnp.bool_(False),
            time_fraction=jnp.asarray(0.0, dtype=dtype),
            contact_position=position,
            restart_position=position,
        ),
        offside_event=OffsideEvent(
            occurred=jnp.bool_(False),
            actor=jnp.int32(NO_PLAYER),
            team=jnp.int32(NO_TEAM),
            position=position,
            time_fraction=jnp.asarray(0.0, dtype=dtype),
        ),
        boundary_event=_empty_boundary_event(boundary),
        body_impact_consumed=jnp.bool_(False),
    )


def _resolve_rules_transition_active(
    pre_physics_state: State,
    physics_substep,
    offside_state: OffsideState,
    requested_intent: jax.Array,
    body_foul_key: jax.Array,
    body_impact_enabled: jax.Array,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    body_foul_config: BodyFoul = BodyFoul(),
    contest_config: Contest = Contest(),
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> RulesTransition:
    """Apply contest, contact, and boundary law in physical time order.

    Contacts after a boundary crossing are ignored defensively even though the
    physics scheduler normally cannot emit them. A validated contest foul owns
    a simultaneous selected-challenger offside offence because Law 5 gives the
    more serious direct-free-kick offence priority. A non-foul challenge by a
    flagged player owns a simultaneous touch offence. If two touch offences
    share an exact timestamp, restart retouch owns the tie.
    """

    active_positions = (
        physics_substep.player_path_start
        + physics_substep.deliberate_time_fraction * physics_substep.player_path_delta
    )
    active_state = physics_substep.state._replace(
        players=physics_substep.state.players._replace(position=active_positions),
        ball=physics_substep.state.ball._replace(live=pre_physics_state.ball.live),
        possession=pre_physics_state.possession,
    )
    contest_foul_facts = facts_from_contest_foul(
        active_state,
        physics_substep.contest,
        ball=ball_geometry,
    )
    # The impact position and time are captured at a locomotion-microstep end,
    # not at a continuous time of impact. Player geometry stays at the physics
    # endpoint because body orientation is fixed before the locomotion
    # microsteps, while the strongest impact retains its own contact geometry.
    # Ball, possession, and restart context deliberately use the causal
    # pre-substep approximation: an exact impact-time snapshot would require
    # sampling the piecewise ball scheduler, whereas its endpoint can include a
    # later contact and must not rewrite an earlier body's attribution or draw.
    # Shoulder alignment is already retained from the collision-pre-response
    # velocity pair in the fixed-shape impact fact.
    enabled_impact = physics_substep.player_impact._replace(
        occurred=(physics_substep.player_impact.occurred & body_impact_enabled)
    )
    # Body extraction stays independent of contest extraction. Even a duplicate
    # pair must consume this frame-local physical opportunity so residual
    # overlap cannot reopen its stochastic lottery in the next substep.
    body_context_state = physics_substep.state._replace(
        ball=pre_physics_state.ball,
        possession=pre_physics_state.possession,
        restart=pre_physics_state.restart,
    )
    body_assessment = sample_body_foul_facts(
        body_context_state,
        enabled_impact,
        requested_intent,
        body_foul_key,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        config=body_foul_config,
        discipline_config=contest_config,
        stadium=stadium,
        ball=ball_geometry,
    )
    body_foul_time_fraction = jnp.asarray(
        physics_substep.player_impact.time_fraction,
        dtype=physics_substep.deliberate_time_fraction.dtype,
    )
    foul_facts, _contest_foul_selected, foul_time_fraction = _select_chronological_foul(
        contest_foul_facts,
        body_assessment.facts,
        physics_substep.deliberate_time_fraction,
        body_foul_time_fraction,
    )
    player_count = active_state.players.position.shape[0]
    safe_offender = jnp.clip(foul_facts.offender, 0, player_count - 1)
    safe_victim = jnp.clip(foul_facts.victim, 0, player_count - 1)
    contest_relative_position = (
        active_state.players.position[safe_victim]
        - active_state.players.position[safe_offender]
    )
    contest_approach = contest_relative_position / jnp.maximum(
        jnp.linalg.norm(contest_relative_position),
        jnp.asarray(1.0e-9, active_state.ball.position.dtype),
    )
    contest_relative_velocity = (
        active_state.players.velocity[safe_offender]
        - active_state.players.velocity[safe_victim]
    )
    contest_closing_speed = jnp.maximum(
        jnp.dot(contest_relative_velocity, contest_approach), 0.0
    )
    offender_speed = jnp.linalg.norm(active_state.players.velocity[safe_offender])
    victim_speed = jnp.linalg.norm(active_state.players.velocity[safe_victim])
    contest_velocity_alignment = jnp.clip(
        jnp.dot(
            active_state.players.velocity[safe_offender],
            active_state.players.velocity[safe_victim],
        )
        / jnp.maximum(
            offender_speed * victim_speed,
            jnp.asarray(1.0e-9, active_state.ball.position.dtype),
        ),
        0.0,
        1.0,
    )
    body_approach = jnp.where(
        body_assessment.facts.offender == physics_substep.player_impact.actor,
        physics_substep.player_impact.contact_normal,
        -physics_substep.player_impact.contact_normal,
    )
    approach_direction = jnp.where(
        _contest_foul_selected, contest_approach, body_approach
    )
    closing_speed = jnp.where(
        _contest_foul_selected,
        contest_closing_speed,
        physics_substep.player_impact.impact_score,
    )
    velocity_alignment = jnp.where(
        _contest_foul_selected,
        contest_velocity_alignment,
        physics_substep.player_impact.velocity_alignment,
    )
    foul_context_state = body_context_state._replace(
        players=body_context_state.players._replace(
            position=jnp.where(
                _contest_foul_selected,
                active_state.players.position,
                body_context_state.players.position,
            )
        )
    )
    severity, tactical_effect = classify_existing_contact_discipline(
        foul_context_state,
        valid=foul_facts.occurred,
        offender=foul_facts.offender,
        victim=foul_facts.victim,
        contact_position=foul_facts.contact_position,
        administrative_discipline=foul_facts.administrative_discipline,
        attempt_to_play_ball=foul_facts.attempt_to_play_ball,
        closing_speed=closing_speed,
        approach_direction=approach_direction,
        velocity_alignment=velocity_alignment,
        stadium=stadium,
    )
    foul_facts = foul_facts._replace(
        severity=severity,
        tactical_effect=tactical_effect,
    )
    foul_adjudication = adjudicate_foul(
        foul_facts, active_state.attack_direction, stadium=stadium
    )
    timed_foul_event = foul_adjudication.event._replace(
        time_fraction=jnp.where(
            foul_adjudication.event.occurred,
            foul_time_fraction,
            0.0,
        ),
        contest_source=(foul_adjudication.event.occurred & _contest_foul_selected),
    )
    challenge_offside = resolve_offside_challenge(
        offside_state,
        active_state,
        physics_substep.contest,
        time_fraction=physics_substep.deliberate_time_fraction,
        ball=ball_geometry,
    )
    occurrences = _contacts_not_after_boundary(
        physics_substep.contact_occurrences,
        physics_substep.boundary,
    )
    retouch_detected = detect_restart_retouch(pre_physics_state, occurrences)
    touch_offside = resolve_offside_contacts(
        offside_state,
        physics_substep.state,
        occurrences,
        player_path_start=physics_substep.player_path_start,
        player_path_delta=physics_substep.player_path_delta,
        ball=ball_geometry,
        body=body,
    )

    retouch_precedes_touch = retouch_detected.occurred & (
        (~touch_offside.event.occurred)
        | (retouch_detected.time_fraction <= touch_offside.event.time_fraction)
    )
    touch_precedes_retouch = touch_offside.event.occurred & (
        (~retouch_detected.occurred)
        | (touch_offside.event.time_fraction < retouch_detected.time_fraction)
    )
    touch_violation = retouch_precedes_touch | touch_precedes_retouch
    touch_violation_time = jnp.where(
        retouch_precedes_touch,
        retouch_detected.time_fraction,
        touch_offside.event.time_fraction,
    )
    active_precedes_touch = (~touch_violation) | (
        physics_substep.deliberate_time_fraction <= touch_violation_time
    )
    foul_precedes_touch = (~touch_violation) | (
        foul_time_fraction <= touch_violation_time
    )
    foul_precedes_boundary = (~physics_substep.boundary.occurred) | (
        foul_time_fraction <= physics_substep.boundary.time_fraction
    )
    foul_selected = (
        timed_foul_event.occurred & foul_precedes_touch & foul_precedes_boundary
    )
    challenge_selected = (
        challenge_offside.occurred & (~foul_selected) & active_precedes_touch
    )
    retouch_selected = retouch_precedes_touch & (~foul_selected) & (~challenge_selected)
    touch_offside_selected = (
        touch_precedes_retouch & (~foul_selected) & (~challenge_selected)
    )
    offside_selected = challenge_selected | touch_offside_selected
    offside_time_fraction = jnp.where(
        challenge_selected,
        challenge_offside.time_fraction,
        touch_offside.event.time_fraction,
    )
    prior_violation = retouch_selected | offside_selected
    boundary_selected = (
        physics_substep.boundary.occurred & (~foul_selected) & (~prior_violation)
    )
    selected_violation = foul_selected | prior_violation | boundary_selected
    dtype = physics_substep.deliberate_time_fraction.dtype
    infinity = jnp.asarray(jnp.inf, dtype=dtype)
    selected_violation_time = jnp.min(
        jnp.stack(
            (
                jnp.where(foul_selected, foul_time_fraction, infinity),
                jnp.where(
                    retouch_selected,
                    retouch_detected.time_fraction,
                    infinity,
                ),
                jnp.where(offside_selected, offside_time_fraction, infinity),
                jnp.where(
                    boundary_selected,
                    physics_substep.boundary.time_fraction,
                    infinity,
                ),
            )
        )
    )
    late_deliberate_after_violation = (
        selected_violation
        & (physics_substep.contest.selected | physics_substep.deliberate_occurred)
        & (physics_substep.deliberate_time_fraction > selected_violation_time)
    )
    chronological_physics_state = _restore_late_deliberate_recovery(
        pre_physics_state,
        physics_substep.state,
        late_deliberate_after_violation,
    )
    # With one active-contact evaluation per physics substep, a selected touch
    # violation that precedes a later deliberate contact must itself be a
    # passive deflection. Preserve that whistle-time last-contact provenance
    # instead of retaining the causally later deliberate result.
    passive_violation = late_deliberate_after_violation & (
        retouch_selected | touch_offside_selected
    )
    passive_actor = jnp.where(
        retouch_selected,
        retouch_detected.offender,
        touch_offside.event.actor,
    ).astype(jnp.int32)
    endpoint_contact = chronological_physics_state.possession.last_contact
    passive_contact = endpoint_contact._replace(
        actor=passive_actor,
        mechanism=jnp.int32(MECHANISM_PASSIVE_BODY),
        intent=jnp.int32(INTENT_MOVE),
        outcome=jnp.int32(OUTCOME_DEFLECTION),
        restart_kind=jnp.int32(RK_NONE),
        law11_effect=jnp.int32(LAW11_DEFLECTION_NO_RESET),
        kick_applied=jnp.bool_(False),
        intent_source=jnp.int32(INTENT_SOURCE_NONE),
    )
    restored_last_contact = jax.tree_util.tree_map(
        lambda restored, endpoint: jnp.where(
            passive_violation,
            restored,
            endpoint,
        ),
        passive_contact,
        endpoint_contact,
    )
    chronological_physics_state = chronological_physics_state._replace(
        possession=chronological_physics_state.possession._replace(
            last_contact=restored_last_contact
        )
    )

    def apply_retouch(_):
        applied = resolve_restart_retouch(
            pre_physics_state,
            chronological_physics_state,
            occurrences,
            stadium=stadium,
            ball_geometry=ball_geometry,
        )
        return applied.state, applied.event

    def skip_retouch(_):
        return chronological_physics_state, retouch_detected

    retouch_state, applied_retouch_event = jax.lax.cond(
        retouch_selected, apply_retouch, skip_retouch, operand=None
    )
    retouch_event = _select_retouch_event(applied_retouch_event, retouch_selected)
    challenge_event = _select_offside_event(challenge_offside, challenge_selected)
    touch_offside_event = _select_offside_event(
        touch_offside.event, touch_offside_selected
    )
    offside_event = jax.tree_util.tree_map(
        lambda challenge_value, touch_value: jnp.where(
            challenge_selected, challenge_value, touch_value
        ),
        challenge_event,
        touch_offside_event,
    )
    violation_state = jax.lax.cond(
        offside_selected,
        lambda _: _apply_offside_offence(
            chronological_physics_state,
            offside_event,
            stadium=stadium,
            ball_geometry=ball_geometry,
        ),
        lambda _: retouch_state,
        operand=None,
    )
    eligible_boundary = _masked_boundary(
        physics_substep.boundary,
        boundary_selected,
    )

    def apply_boundary(_):
        applied = resolve_boundary_crossing(
            violation_state,
            eligible_boundary,
            stadium=stadium,
            ball_geometry=ball_geometry,
        )
        return applied.state, applied.event

    boundary_state, boundary_event = jax.lax.cond(
        eligible_boundary.occurred,
        apply_boundary,
        lambda _: (violation_state, _empty_boundary_event(eligible_boundary)),
        operand=None,
    )

    foul_input_state = chronological_physics_state._replace(
        ball=chronological_physics_state.ball._replace(
            live=pre_physics_state.ball.live
        ),
        possession=pre_physics_state.possession,
    )
    next_state = jax.lax.cond(
        foul_selected,
        lambda _: (
            resolve_foul(
                foul_input_state,
                foul_facts,
                stadium=stadium,
                ball_geometry=ball_geometry,
            ).state
        ),
        lambda _: boundary_state,
        operand=None,
    )
    stoppage = foul_selected | prior_violation | boundary_event.occurred
    next_offside_state = jax.tree_util.tree_map(
        lambda cleared, current: jnp.where(stoppage, cleared, current),
        clear_offside_state(touch_offside.state),
        touch_offside.state,
    )
    return RulesTransition(
        state=next_state,
        offside_state=next_offside_state,
        foul_event=_select_foul_event(timed_foul_event, foul_selected),
        retouch_event=retouch_event,
        offside_event=offside_event,
        boundary_event=boundary_event,
        body_impact_consumed=body_assessment.eligible,
    )


def resolve_rules_transition(
    pre_physics_state: State,
    physics_substep,
    offside_state: OffsideState,
    requested_intent: jax.Array,
    body_foul_key: jax.Array,
    body_impact_enabled: jax.Array,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    body_foul_config: BodyFoul = BodyFoul(),
    contest_config: Contest = Contest(),
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
    body: BodyContact = BodyContact(),
) -> RulesTransition:
    """Resolve only substeps carrying a contest, contact, or boundary input."""

    body_foul_candidate = (
        physics_substep.player_impact.occurred
        & body_impact_enabled
        & pre_physics_state.ball.live
        & (pre_physics_state.restart.kind == RK_NONE)
        & jnp.isfinite(physics_substep.player_impact.impact_score)
        & (
            physics_substep.player_impact.impact_score
            > body_foul_config.minimum_closing_speed_mps
        )
    )
    has_rule_input = (
        physics_substep.contest.selected
        | jnp.any(physics_substep.contact_occurrences.occurred)
        | physics_substep.boundary.occurred
        | body_foul_candidate
    )
    return jax.lax.cond(
        has_rule_input,
        lambda _: _resolve_rules_transition_active(
            pre_physics_state,
            physics_substep,
            offside_state,
            requested_intent,
            body_foul_key,
            body_impact_enabled,
            regulation_elapsed_fraction=regulation_elapsed_fraction,
            body_foul_config=body_foul_config,
            contest_config=contest_config,
            stadium=stadium,
            ball_geometry=ball_geometry,
            body=body,
        ),
        lambda _: _empty_rules_transition(
            physics_substep.state,
            offside_state,
            physics_substep.boundary,
        ),
        operand=None,
    )
