"""Deterministic Law 12 fact extraction and adjudication.

The adjudicator does not infer whether contact was careless, reckless,
excessive, SPA, or DOGSO. After source arbitration, the environment fact path
may conservatively describe an already sampled card using environment-truth
geometry; the sampled discipline remains authoritative. The adjudicator only
maps those facts to the restart and discipline required by IFAB Laws 5 and 12
(2026/27):

https://www.theifab.com/laws/latest/fouls-and-misconduct/
https://www.theifab.com/laws/latest/the-referee/
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_foul import BodyFoul
from footballworld.config.contest import Contest
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    NO_PLAYER,
    NO_TEAM,
    RK_FREEKICK,
    RK_NONE,
    RK_PENALTY,
    TEAM_0,
    TEAM_1,
    YELLOW_CARD_SEND_OFF_COUNT,
)
from footballworld.core.contact import INTENT_CHALLENGE, MECHANISM_NONE, OUTCOME_FOUL
from footballworld.core.randomness import RandomEvent, _event_random_key_unchecked
from footballworld.core.state import RestartReleaseProvenance, State
from footballworld.dynamics.contest import challenge_card_probability
from footballworld.dynamics.separation import PlayerImpactFacts
from footballworld.rules.restart import select_restart_taker
from footballworld.rules.restart_spot import canonical_restart_spot

# Offence families.  These describe an established Law 12 offence, not a
# noisy physical-contact label.
OFFENCE_NONE = 0
OFFENCE_DIRECT_CONTACT = 1
OFFENCE_HANDBALL = 2
OFFENCE_HOLD = 3
OFFENCE_IMPEDE_CONTACT = 4
OFFENCE_THROW_OBJECT = 5
OFFENCE_DANGEROUS_PLAY = 6
OFFENCE_IMPEDE_NO_CONTACT = 7
OFFENCE_PREVENT_GK_RELEASE = 8
# Routine GK handling: rehandling after a hand release, a deliberate
# team-mate kick, or a team-mate throw-in inside the own penalty area.
# Restart second touch is deliberately a separate offence type.
OFFENCE_GK_ILLEGAL_HANDLING = 9
# Prohibited second touch after taking a restart. This is an IFK offence
# that may carry SPA/DOGSO discipline. A handball second touch outside a
# goalkeeper own penalty area must be supplied as OFFENCE_HANDBALL.
OFFENCE_RESTART_SECOND_TOUCH = 10
OFFENCE_DELIBERATE_GK_TRICK = 11
OFFENCE_OTHER_INDIRECT = 12
OFFENCE_VIOLENT_CONDUCT = 13
OFFENCE_COUNT = 14


# Referee-classified challenge severity.
SEVERITY_NONE = 0
SEVERITY_CARELESS = 1
SEVERITY_RECKLESS = 2
SEVERITY_EXCESSIVE_FORCE = 3


# Tactical consequence established by the fact extractor/referee.
TACTICAL_NONE = 0
TACTICAL_SPA = 1
TACTICAL_DOGSO = 2


class FoulFacts(NamedTuple):
    """Fixed-shape facts needed for a deterministic Law 12 decision.

    ``attempt_to_play_ball`` includes a challenge for the ball.  It is used
    only for the penalty-area SPA/DOGSO discipline exceptions.
    ``advantage_goal_scored`` means the non-offending team scored as the
    result of the advantage, not merely later in the match.
    """

    occurred: jax.Array
    offender: jax.Array
    victim: jax.Array
    offender_team: jax.Array
    offence_type: jax.Array
    severity: jax.Array
    tactical_effect: jax.Array
    contact_position: jax.Array
    ball_in_play: jax.Array
    advantage_realised: jax.Array
    advantage_goal_scored: jax.Array
    attempt_to_play_ball: jax.Array
    deliberate_handball: jax.Array
    offender_is_goalkeeper: jax.Array
    administrative_discipline: jax.Array

    @classmethod
    def none(cls, dtype=jnp.float32) -> "FoulFacts":
        """Return a scalar no-offence fact bundle."""

        return cls(
            occurred=jnp.bool_(False),
            offender=jnp.int32(NO_PLAYER),
            victim=jnp.int32(NO_PLAYER),
            offender_team=jnp.int32(NO_TEAM),
            offence_type=jnp.int32(OFFENCE_NONE),
            severity=jnp.int32(SEVERITY_NONE),
            tactical_effect=jnp.int32(TACTICAL_NONE),
            contact_position=jnp.zeros(3, dtype=dtype),
            ball_in_play=jnp.bool_(False),
            advantage_realised=jnp.bool_(False),
            advantage_goal_scored=jnp.bool_(False),
            attempt_to_play_ball=jnp.bool_(False),
            deliberate_handball=jnp.bool_(False),
            offender_is_goalkeeper=jnp.bool_(False),
            administrative_discipline=jnp.int32(DISCIPLINE_NONE),
        )


class FoulDecision(NamedTuple):
    """Restart and disciplinary consequence of one classified offence."""

    offence: jax.Array
    restart_awarded: jax.Array
    play_continues: jax.Array
    restart_kind: jax.Array
    restart_team: jax.Array
    indirect: jax.Array
    penalty: jax.Array
    restart_basis_position: jax.Array
    discipline: jax.Array
    advantage_applied: jax.Array


class FoulEvent(NamedTuple):
    """Compact audit event emitted alongside the decision."""

    occurred: jax.Array
    contest_source: jax.Array
    offender: jax.Array
    victim: jax.Array
    offender_team: jax.Array
    offence_type: jax.Array
    severity: jax.Array
    tactical_effect: jax.Array
    position: jax.Array
    time_fraction: jax.Array
    restart_kind: jax.Array
    discipline: jax.Array
    advantage_applied: jax.Array


class FoulAdjudication(NamedTuple):
    """Pure Law 12 output without mutating the rollout state."""

    decision: FoulDecision
    event: FoulEvent


class BodyFoulAssessment(NamedTuple):
    """One frame-consumable physical impact and its sampled foul facts."""

    eligible: jax.Array
    probability: jax.Array
    facts: FoulFacts


def _is_direct_offence(offence_type: jax.Array) -> jax.Array:
    return (
        (offence_type == OFFENCE_DIRECT_CONTACT)
        | (offence_type == OFFENCE_HANDBALL)
        | (offence_type == OFFENCE_HOLD)
        | (offence_type == OFFENCE_IMPEDE_CONTACT)
        | (offence_type == OFFENCE_THROW_OBJECT)
        | (offence_type == OFFENCE_VIOLENT_CONDUCT)
    )


def _is_indirect_offence(offence_type: jax.Array) -> jax.Array:
    return (
        (offence_type == OFFENCE_DANGEROUS_PLAY)
        | (offence_type == OFFENCE_IMPEDE_NO_CONTACT)
        | (offence_type == OFFENCE_PREVENT_GK_RELEASE)
        | (offence_type == OFFENCE_GK_ILLEGAL_HANDLING)
        | (offence_type == OFFENCE_RESTART_SECOND_TOUCH)
        | (offence_type == OFFENCE_DELIBERATE_GK_TRICK)
        | (offence_type == OFFENCE_OTHER_INDIRECT)
    )


def _inside_own_penalty_area(
    position: jax.Array,
    offender_team: jax.Array,
    attack_direction: jax.Array,
    *,
    stadium: Stadium,
) -> jax.Array:
    safe_team = jnp.clip(offender_team, TEAM_0, TEAM_1)
    direction = attack_direction[safe_team]
    depth = jnp.asarray(stadium.half_length, position.dtype) + (direction * position[0])
    return (
        (depth >= 0.0)
        & (depth <= stadium.penalty_area_length)
        & (jnp.abs(position[1]) <= 0.5 * stadium.penalty_area_width)
    )


def classify_existing_contact_discipline(
    state: State,
    *,
    valid: jax.Array,
    offender: jax.Array,
    victim: jax.Array,
    contact_position: jax.Array,
    administrative_discipline: jax.Array,
    attempt_to_play_ball: jax.Array,
    closing_speed: jax.Array,
    approach_direction: jax.Array,
    velocity_alignment: jax.Array,
    stadium: Stadium,
) -> tuple[jax.Array, jax.Array]:
    """Describe an existing card without changing its disciplinary strength.

    This is deliberately a strict post-classifier, not another foul or card
    model. No-card contact stays careless/non-tactical. Yellow labels are only
    selected when Law 12 would produce at most yellow from the derived facts;
    red labels are only selected when it would produce red. One player-axis
    reduction counts defenders goalward of the offence for the DOGSO prior.
    """

    dtype = state.players.position.dtype
    player_count = state.players.position.shape[0]
    safe_offender = jnp.clip(jnp.asarray(offender, jnp.int32), 0, player_count - 1)
    safe_victim = jnp.clip(jnp.asarray(victim, jnp.int32), 0, player_count - 1)
    offender_team = state.players.team_id[safe_offender].astype(jnp.int32)
    victim_team = state.players.team_id[safe_victim].astype(jnp.int32)
    safe_victim_team = jnp.clip(victim_team, TEAM_0, TEAM_1)

    approach = jnp.asarray(approach_direction, dtype=dtype)
    approach = approach / jnp.maximum(
        jnp.linalg.norm(approach), jnp.asarray(1.0e-9, dtype)
    )
    victim_body = state.players.body_forward[safe_victim]
    end_on_alignment = jnp.abs(jnp.clip(jnp.dot(approach, victim_body), -1.0, 1.0))
    lateral_alignment = 1.0 - end_on_alignment
    shoulder_alignment = (
        jnp.clip(jnp.asarray(velocity_alignment, dtype=dtype), 0.0, 1.0)
        * lateral_alignment
    )
    unsafe_approach = end_on_alignment > shoulder_alignment

    maximum_combined_speed = (
        state.players.max_speed[safe_offender] + state.players.max_speed[safe_victim]
    )
    maximum_individual_speed = jnp.maximum(
        state.players.max_speed[safe_offender],
        state.players.max_speed[safe_victim],
    )
    closing_speed = jnp.maximum(jnp.asarray(closing_speed, dtype=dtype), 0.0)
    reckless_geometry = unsafe_approach & (
        2.0 * closing_speed >= maximum_combined_speed
    )
    excessive_geometry = unsafe_approach & (closing_speed >= maximum_individual_speed)

    contact_xy = jnp.asarray(contact_position, dtype=dtype)[:2]
    victim_direction = state.attack_direction[safe_victim_team].astype(dtype)
    victim_goal = jnp.asarray(
        [victim_direction * stadium.half_length, 0.0], dtype=dtype
    )
    goal_vector = victim_goal - contact_xy
    goal_unit = goal_vector / jnp.maximum(
        jnp.linalg.norm(goal_vector), jnp.asarray(1.0e-9, dtype)
    )
    facing_goal = jnp.dot(victim_body, goal_unit) > 0.0
    moving_goalward = jnp.dot(state.players.velocity[safe_victim], goal_unit) >= 0.0
    victim_possesses = (state.possession.player == victim) & (
        state.possession.team == victim_team
    )
    goalward_attack = victim_possesses & facing_goal & moving_goalward

    progress = contact_xy[0] * victim_direction
    distance_to_goal_line = jnp.asarray(stadium.half_length, dtype) - progress
    near_goal = (distance_to_goal_line >= 0.0) & (
        distance_to_goal_line <= stadium.penalty_area_length
    )
    attacking_half = progress >= 0.0
    goalward_defender = (
        state.players.active
        & (state.players.team_id == offender_team)
        & (jnp.arange(player_count, dtype=jnp.int32) != safe_offender)
        & (state.players.position[:, 0] * victim_direction > progress)
    )
    goalward_defender_count = jnp.sum(goalward_defender.astype(jnp.int32))
    dogso_geometry = (
        goalward_attack & near_goal & (goalward_defender_count <= jnp.int32(1))
    )
    spa_geometry = goalward_attack & attacking_half & (~dogso_geometry)

    inside_penalty_area = _inside_own_penalty_area(
        contact_position,
        offender_team,
        state.attack_direction,
        stadium=stadium,
    )
    valid = jnp.asarray(valid, dtype=bool)
    discipline = jnp.asarray(administrative_discipline, dtype=jnp.int32)
    yellow = valid & (discipline == DISCIPLINE_YELLOW)
    red = valid & (discipline == DISCIPLINE_RED)

    # These gates make the extracted label incapable of upgrading the sampled
    # administrative result when adjudicated. Mutually exclusive priority is
    # DOGSO, physical severity, then SPA.
    yellow_dogso = yellow & dogso_geometry & inside_penalty_area & attempt_to_play_ball
    red_dogso = (
        red & dogso_geometry & ((~inside_penalty_area) | (~attempt_to_play_ball))
    )
    yellow_reckless = yellow & (~yellow_dogso) & reckless_geometry
    red_excessive = red & (~red_dogso) & excessive_geometry
    yellow_spa = (
        yellow
        & (~yellow_dogso)
        & (~yellow_reckless)
        & spa_geometry
        & (~(inside_penalty_area & attempt_to_play_ball))
    )

    severity = jnp.where(valid, SEVERITY_CARELESS, SEVERITY_NONE)
    severity = jnp.where(yellow_reckless, SEVERITY_RECKLESS, severity)
    severity = jnp.where(red_excessive, SEVERITY_EXCESSIVE_FORCE, severity)
    tactical_effect = jnp.where(
        yellow_dogso | red_dogso,
        TACTICAL_DOGSO,
        jnp.where(yellow_spa, TACTICAL_SPA, TACTICAL_NONE),
    )
    return severity.astype(jnp.int32), tactical_effect.astype(jnp.int32)


def _base_discipline(
    facts: FoulFacts,
    offence: jax.Array,
) -> jax.Array:
    severity_applies = (
        (facts.offence_type == OFFENCE_DIRECT_CONTACT)
        | (facts.offence_type == OFFENCE_THROW_OBJECT)
        | (facts.offence_type == OFFENCE_DANGEROUS_PLAY)
    )
    by_severity = jnp.where(
        facts.severity == SEVERITY_EXCESSIVE_FORCE,
        DISCIPLINE_RED,
        jnp.where(
            facts.severity == SEVERITY_RECKLESS,
            DISCIPLINE_YELLOW,
            DISCIPLINE_NONE,
        ),
    )
    mandatory = jnp.where(
        facts.offence_type == OFFENCE_VIOLENT_CONDUCT,
        DISCIPLINE_RED,
        jnp.where(
            facts.offence_type == OFFENCE_DELIBERATE_GK_TRICK,
            DISCIPLINE_YELLOW,
            DISCIPLINE_NONE,
        ),
    )
    return jnp.where(
        offence,
        jnp.maximum(
            mandatory,
            jnp.where(severity_applies, by_severity, DISCIPLINE_NONE),
        ),
        DISCIPLINE_NONE,
    ).astype(jnp.int32)


def _tactical_discipline(
    facts: FoulFacts,
    *,
    offence: jax.Array,
    penalty: jax.Array,
    advantage_applied: jax.Array,
) -> jax.Array:
    handball = facts.offence_type == OFFENCE_HANDBALL
    routine_gk_handling = facts.offence_type == OFFENCE_GK_ILLEGAL_HANDLING
    tactical_offence = offence & facts.ball_in_play & (~routine_gk_handling)

    spa_exception = penalty & jnp.where(
        handball,
        ~facts.deliberate_handball,
        facts.attempt_to_play_ball,
    )
    spa_card = jnp.where(
        advantage_applied | spa_exception,
        DISCIPLINE_NONE,
        DISCIPLINE_YELLOW,
    )

    dogso_stopped_card = jnp.where(
        handball,
        jnp.where(
            facts.deliberate_handball,
            DISCIPLINE_RED,
            jnp.where(penalty, DISCIPLINE_YELLOW, DISCIPLINE_RED),
        ),
        jnp.where(
            penalty & facts.attempt_to_play_ball,
            DISCIPLINE_YELLOW,
            DISCIPLINE_RED,
        ),
    )
    dogso_advantage_card = jnp.where(
        facts.advantage_goal_scored,
        DISCIPLINE_NONE,
        DISCIPLINE_YELLOW,
    )
    dogso_card = jnp.where(
        advantage_applied,
        dogso_advantage_card,
        dogso_stopped_card,
    )

    tactical_card = jnp.where(
        facts.tactical_effect == TACTICAL_SPA,
        spa_card,
        jnp.where(
            facts.tactical_effect == TACTICAL_DOGSO,
            dogso_card,
            DISCIPLINE_NONE,
        ),
    )
    return jnp.where(tactical_offence, tactical_card, DISCIPLINE_NONE).astype(jnp.int32)


def adjudicate_foul(
    facts: FoulFacts,
    attack_direction: jax.Array,
    *,
    stadium: Stadium = Stadium(),
) -> FoulAdjudication:
    """Map one scalar fact bundle to a deterministic Law 12 decision.

    Penalty-area boundary lines count as part of the area.  A successful
    advantage suppresses the immediate restart, but independent reckless,
    excessive-force, trick, or violent-conduct discipline remains.

    Goalkeeper hand/arm control lasting more than eight seconds is outside
    this contact-offence adjudicator. Under 2026/27 Law 12.3 its restart is
    an opponent corner kick, not the indirect free kick used for routine
    illegal handling here.
    """

    valid_team = (facts.offender_team == TEAM_0) | (facts.offender_team == TEAM_1)
    direct_family = _is_direct_offence(facts.offence_type)
    indirect_family = _is_indirect_offence(facts.offence_type)
    challenge_established = (facts.offence_type != OFFENCE_DIRECT_CONTACT) | (
        facts.severity >= SEVERITY_CARELESS
    )
    goalkeeper_handball_exempt = (
        (facts.offence_type == OFFENCE_HANDBALL)
        & facts.offender_is_goalkeeper
        & _inside_own_penalty_area(
            facts.contact_position,
            facts.offender_team,
            attack_direction,
            stadium=stadium,
        )
    )
    offence = (
        facts.occurred
        & valid_team
        & (direct_family | indirect_family)
        & challenge_established
        & (~goalkeeper_handball_exempt)
    )
    in_own_penalty_area = _inside_own_penalty_area(
        facts.contact_position,
        facts.offender_team,
        attack_direction,
        stadium=stadium,
    )
    restart_possible = offence & facts.ball_in_play
    penalty = restart_possible & direct_family & in_own_penalty_area
    advantage_applied = restart_possible & facts.advantage_realised
    restart_awarded = restart_possible & (~advantage_applied)
    restart_kind = jnp.where(
        restart_awarded,
        jnp.where(penalty, RK_PENALTY, RK_FREEKICK),
        RK_NONE,
    ).astype(jnp.int32)
    restart_team = jnp.where(
        restart_awarded,
        TEAM_1 - facts.offender_team,
        NO_TEAM,
    ).astype(jnp.int32)
    indirect = restart_awarded & indirect_family

    base_card = _base_discipline(facts, offence)
    tactical_card = _tactical_discipline(
        facts,
        offence=offence,
        penalty=penalty,
        advantage_applied=advantage_applied,
    )
    supplied_discipline = jnp.asarray(facts.administrative_discipline, dtype=jnp.int32)
    supplied_discipline = jnp.where(
        offence
        & (supplied_discipline >= DISCIPLINE_NONE)
        & (supplied_discipline <= DISCIPLINE_RED),
        supplied_discipline,
        DISCIPLINE_NONE,
    )
    discipline = jnp.maximum(
        jnp.maximum(base_card, tactical_card), supplied_discipline
    ).astype(jnp.int32)
    restart_basis_position = jnp.where(
        restart_awarded,
        facts.contact_position,
        jnp.zeros_like(facts.contact_position),
    )
    play_continues = advantage_applied

    decision = FoulDecision(
        offence=offence,
        restart_awarded=restart_awarded,
        play_continues=play_continues,
        restart_kind=restart_kind,
        restart_team=restart_team,
        indirect=indirect,
        penalty=restart_awarded & penalty,
        restart_basis_position=restart_basis_position,
        discipline=discipline,
        advantage_applied=advantage_applied,
    )
    event = FoulEvent(
        occurred=offence,
        # Source arbitration happens one layer above adjudication, after both
        # contest and player-impact candidates have been sampled.
        contest_source=jnp.bool_(False),
        offender=jnp.where(offence, facts.offender, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(offence, facts.victim, NO_PLAYER).astype(jnp.int32),
        offender_team=jnp.where(offence, facts.offender_team, NO_TEAM).astype(
            jnp.int32
        ),
        offence_type=jnp.where(offence, facts.offence_type, OFFENCE_NONE).astype(
            jnp.int32
        ),
        severity=jnp.where(offence, facts.severity, SEVERITY_NONE).astype(jnp.int32),
        tactical_effect=jnp.where(offence, facts.tactical_effect, TACTICAL_NONE).astype(
            jnp.int32
        ),
        position=jnp.where(
            offence,
            facts.contact_position,
            jnp.zeros_like(facts.contact_position),
        ),
        time_fraction=jnp.asarray(0.0, dtype=facts.contact_position.dtype),
        restart_kind=restart_kind,
        discipline=discipline,
        advantage_applied=advantage_applied,
    )
    return FoulAdjudication(decision=decision, event=event)


class FoulResolution(NamedTuple):
    """Authoritative state plus the Law 12 decision that produced it."""

    state: State
    adjudication: FoulAdjudication


def sample_body_foul_facts(
    state: State,
    impact: PlayerImpactFacts,
    requested_intent: jax.Array,
    key: jax.Array,
    *,
    contest_foul_occurred: jax.Array = False,
    regulation_elapsed_fraction: jax.Array = 0.0,
    config: BodyFoul = BodyFoul(),
    discipline_config: Contest = Contest(),
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
) -> BodyFoulAssessment:
    """Classify and sample one actual opponent body impact.

    The physics layer has already selected the strongest opponent impact in
    this substep. This extractor does not search another pair and never changes
    the collision response. The caller consumes ``eligible`` after the first
    valid frame-local opportunity, whether or not the stochastic foul draw
    succeeds, which prevents residual capsule overlap from creating repeated
    lottery tickets.

    Occurrence uses a dedicated folded key. Conditional card and colour draws
    use separate folded keys and the same established contact-foul discipline
    curve as selected CHALLENGE fouls. A missed CHALLENGE may still become a
    body foul if it physically impacts an opponent. The authoritative rules
    transition leaves ``contest_foul_occurred`` false, completes this draw, and
    then arbitrates same-pair duplicates; that preserves frame-local opportunity
    consumption. The argument remains only as an explicit diagnostic opt-out.
    """

    player_count = state.players.position.shape[0]
    provisional_actor = jnp.asarray(impact.actor, dtype=jnp.int32)
    provisional_victim = jnp.asarray(impact.victim, dtype=jnp.int32)
    actor_in_range = (provisional_actor >= 0) & (provisional_actor < player_count)
    victim_in_range = (provisional_victim >= 0) & (provisional_victim < player_count)
    safe_provisional_actor = jnp.clip(provisional_actor, 0, player_count - 1)
    safe_provisional_victim = jnp.clip(provisional_victim, 0, player_count - 1)
    intent = jnp.asarray(requested_intent, dtype=jnp.int32)

    normal = jnp.asarray(impact.contact_normal, dtype=state.ball.position.dtype)
    normal_norm = jnp.linalg.norm(normal)
    normal = normal / jnp.maximum(normal_norm, jnp.asarray(1.0e-9, normal.dtype))
    actor_challenges = intent[safe_provisional_actor] == jnp.int32(INTENT_CHALLENGE)
    victim_challenges = intent[safe_provisional_victim] == jnp.int32(INTENT_CHALLENGE)
    unique_challenge = actor_challenges != victim_challenges
    provisional_actor_team = state.players.team_id[safe_provisional_actor].astype(
        jnp.int32
    )
    provisional_victim_team = state.players.team_id[safe_provisional_victim].astype(
        jnp.int32
    )
    actor_possesses = (
        actor_in_range
        & state.players.active[safe_provisional_actor]
        & (state.possession.player == provisional_actor)
        & (state.possession.team == provisional_actor_team)
    )
    victim_possesses = (
        victim_in_range
        & state.players.active[safe_provisional_victim]
        & (state.possession.player == provisional_victim)
        & (state.possession.team == provisional_victim_team)
    )
    unique_possession = actor_possesses != victim_possesses
    kinematic_attribution = jnp.asarray(impact.actor_attribution_decisive, dtype=bool)
    tie_attribution_available = unique_challenge | unique_possession
    swap_tied_pair = (~kinematic_attribution) & (
        (unique_challenge & victim_challenges)
        | ((~unique_challenge) & unique_possession & actor_possesses)
    )
    actor = jnp.where(swap_tied_pair, provisional_victim, provisional_actor).astype(
        jnp.int32
    )
    victim = jnp.where(swap_tied_pair, provisional_actor, provisional_victim).astype(
        jnp.int32
    )
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    safe_victim = jnp.clip(victim, 0, player_count - 1)
    actor_team = state.players.team_id[safe_actor].astype(jnp.int32)
    victim_team = state.players.team_id[safe_victim].astype(jnp.int32)
    normal = jnp.where(swap_tied_pair, -normal, normal)
    contact_xy = jnp.asarray(impact.contact_position, dtype=state.ball.position.dtype)
    closing_speed = jnp.asarray(impact.impact_score, dtype=contact_xy.dtype)
    finite_impact = (
        jnp.isfinite(closing_speed)
        & jnp.isfinite(impact.velocity_alignment)
        & jnp.all(jnp.isfinite(normal))
        & jnp.all(jnp.isfinite(contact_xy))
    )

    eligible = (
        impact.occurred
        & (~jnp.asarray(contest_foul_occurred, dtype=bool))
        & state.ball.live
        & (state.restart.kind == RK_NONE)
        & actor_in_range
        & victim_in_range
        & state.players.active[safe_actor]
        & state.players.active[safe_victim]
        & (actor_team != victim_team)
        & ((actor_team == TEAM_0) | (actor_team == TEAM_1))
        & ((victim_team == TEAM_0) | (victim_team == TEAM_1))
        & finite_impact
        & (normal_norm > 1.0e-9)
        & (closing_speed > config.minimum_closing_speed_mps)
        & (kinematic_attribution | tie_attribution_available)
    )

    victim_body = state.players.body_forward[safe_victim]
    behind = jnp.clip(jnp.dot(normal, victim_body), 0.0, 1.0)
    lateral_contact = 1.0 - jnp.abs(jnp.dot(normal, victim_body))
    velocity_alignment = jnp.clip(
        jnp.asarray(impact.velocity_alignment, dtype=contact_xy.dtype),
        0.0,
        1.0,
    )
    shoulder_alignment = velocity_alignment * lateral_contact
    ball_far = (
        jnp.linalg.norm(state.ball.position[:2] - contact_xy)
        > config.ball_near_distance_m
    ).astype(contact_xy.dtype)
    possessed_victim = (
        (state.possession.player == victim) & (state.possession.team == victim_team)
    ).astype(contact_xy.dtype)
    victim_attack_direction = jnp.asarray(
        [state.attack_direction[jnp.clip(victim_team, TEAM_0, TEAM_1)], 0.0],
        dtype=contact_xy.dtype,
    )
    goal_denial = possessed_victim * jnp.clip(
        -jnp.dot(normal, victim_attack_direction), 0.0, 1.0
    )
    excess_speed = jnp.maximum(closing_speed - config.minimum_closing_speed_mps, 0.0)
    logit = (
        config.base_logit
        + config.closing_speed_logit_weight_per_mps * excess_speed
        + config.behind_logit_weight * behind
        - config.shoulder_alignment_logit_discount * shoulder_alignment
        + config.ball_far_logit_weight * ball_far
        + config.possessed_victim_logit_weight * possessed_victim
        + config.goal_denial_logit_weight * goal_denial
    )
    baseline_probability = jnp.clip(
        jax.nn.sigmoid(logit),
        config.probability_floor,
        config.probability_ceiling,
    )
    probability = jnp.clip(
        baseline_probability * config.rare_case_coverage_multiplier, 0.0, 1.0
    )
    occurrence_key = _event_random_key_unchecked(key, RandomEvent.BODY_FOUL)
    card_key = _event_random_key_unchecked(key, RandomEvent.BODY_CARD)
    colour_key = _event_random_key_unchecked(key, RandomEvent.BODY_CARD_COLOUR)
    occurred = eligible & (
        jax.random.uniform(occurrence_key, dtype=contact_xy.dtype) < probability
    )

    attack_progress = jnp.clip(
        0.5
        * (
            contact_xy[0]
            * state.attack_direction[jnp.clip(actor_team, TEAM_0, TEAM_1)]
            / jnp.asarray(stadium.half_length, contact_xy.dtype)
            + 1.0
        ),
        0.0,
        1.0,
    )
    card_probability = challenge_card_probability(
        attack_progress,
        regulation_elapsed_fraction,
        config=discipline_config,
    )
    carded = jax.random.uniform(card_key, dtype=contact_xy.dtype) < card_probability
    direct_red = carded & (
        jax.random.uniform(colour_key, dtype=contact_xy.dtype)
        < discipline_config.direct_red_given_card_probability
    )
    discipline = jnp.where(
        occurred,
        jnp.where(
            direct_red,
            DISCIPLINE_RED,
            jnp.where(carded, DISCIPLINE_YELLOW, DISCIPLINE_NONE),
        ),
        DISCIPLINE_NONE,
    ).astype(jnp.int32)
    contact_position = jnp.concatenate(
        (contact_xy, jnp.asarray([ball.radius], dtype=contact_xy.dtype))
    )
    facts = FoulFacts(
        occurred=occurred,
        offender=jnp.where(occurred, actor, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(occurred, victim, NO_PLAYER).astype(jnp.int32),
        offender_team=jnp.where(occurred, actor_team, NO_TEAM).astype(jnp.int32),
        offence_type=jnp.where(occurred, OFFENCE_DIRECT_CONTACT, OFFENCE_NONE).astype(
            jnp.int32
        ),
        severity=jnp.where(occurred, SEVERITY_CARELESS, SEVERITY_NONE).astype(
            jnp.int32
        ),
        tactical_effect=jnp.int32(TACTICAL_NONE),
        contact_position=jnp.where(
            occurred, contact_position, jnp.zeros_like(contact_position)
        ),
        ball_in_play=occurred,
        advantage_realised=jnp.bool_(False),
        advantage_goal_scored=jnp.bool_(False),
        attempt_to_play_ball=(
            occurred & (intent[safe_actor] == jnp.int32(INTENT_CHALLENGE))
        ),
        deliberate_handball=jnp.bool_(False),
        offender_is_goalkeeper=(occurred & state.players.is_goalkeeper[safe_actor]),
        administrative_discipline=discipline,
    )
    return BodyFoulAssessment(
        eligible=eligible,
        probability=jnp.where(eligible, probability, 0.0).astype(jnp.float32),
        facts=facts,
    )


def facts_from_contest_foul(
    state: State,
    contest,
    *,
    ball: Ball = Ball(),
) -> FoulFacts:
    """Validate and classify the deliberately small contest-foul subset.

    A sampled foul is not trusted merely because its outcome code says so.
    The selected actor must also be the recorded challenger and foul actor,
    while the victim must be the active opposing player who held verified
    control before physics. The contest model samples an aggregate post-foul
    administrative discipline category separately from severity. It does not
    infer SPA/DOGSO or biomechanical injury risk. Its position is consumed
    from the contest result rather than reconstructed from either player or
    the ball.
    """

    player_count = state.players.position.shape[0]
    actor = jnp.asarray(contest.actor, dtype=jnp.int32)
    victim = jnp.asarray(contest.foul_victim, dtype=jnp.int32)
    actor_in_range = (actor >= 0) & (actor < player_count)
    victim_in_range = (victim >= 0) & (victim < player_count)
    safe_actor = jnp.clip(actor, 0, player_count - 1)
    safe_victim = jnp.clip(victim, 0, player_count - 1)
    offender_team = state.players.team_id[safe_actor].astype(jnp.int32)
    victim_team = state.players.team_id[safe_victim].astype(jnp.int32)
    valid = (
        contest.selected
        & contest.occurred
        & (contest.outcome == OUTCOME_FOUL)
        & state.ball.live
        & actor_in_range
        & victim_in_range
        & (contest.challenger == actor)
        & (contest.foul_actor == actor)
        & (state.possession.player == victim)
        & (state.possession.team == victim_team)
        & state.players.active[safe_actor]
        & state.players.active[safe_victim]
        & (offender_team != victim_team)
        & ((offender_team == TEAM_0) | (offender_team == TEAM_1))
        & ((victim_team == TEAM_0) | (victim_team == TEAM_1))
    )
    dtype = state.ball.position.dtype
    position = jnp.concatenate(
        (
            jnp.asarray(contest.offence_position, dtype=dtype),
            jnp.asarray([ball.radius], dtype=dtype),
        )
    )
    return FoulFacts(
        occurred=valid,
        offender=jnp.where(valid, actor, NO_PLAYER).astype(jnp.int32),
        victim=jnp.where(valid, victim, NO_PLAYER).astype(jnp.int32),
        offender_team=jnp.where(valid, offender_team, NO_TEAM).astype(jnp.int32),
        offence_type=jnp.where(valid, OFFENCE_DIRECT_CONTACT, OFFENCE_NONE).astype(
            jnp.int32
        ),
        severity=jnp.where(valid, SEVERITY_CARELESS, SEVERITY_NONE).astype(jnp.int32),
        tactical_effect=jnp.int32(TACTICAL_NONE),
        contact_position=jnp.where(valid, position, jnp.zeros_like(position)),
        ball_in_play=valid,
        advantage_realised=jnp.bool_(False),
        advantage_goal_scored=jnp.bool_(False),
        attempt_to_play_ball=valid,
        deliberate_handball=jnp.bool_(False),
        offender_is_goalkeeper=(valid & state.players.is_goalkeeper[safe_actor]),
        administrative_discipline=jnp.where(
            valid, contest.discipline, DISCIPLINE_NONE
        ).astype(jnp.int32),
    )


def resolve_foul(
    state: State,
    facts: FoulFacts,
    *,
    stadium: Stadium = Stadium(),
    ball_geometry: Ball = Ball(),
) -> FoulResolution:
    """Apply one classified offence without adding rollout state fields."""

    adjudication = adjudicate_foul(
        facts,
        state.attack_direction,
        stadium=stadium,
    )
    decision = adjudication.decision
    player_count = state.players.position.shape[0]
    offender = jnp.asarray(facts.offender, dtype=jnp.int32)
    offender_valid = (offender >= 0) & (offender < player_count)
    safe_offender = jnp.clip(offender, 0, player_count - 1)
    discipline_applies = decision.offence & offender_valid
    yellow = discipline_applies & (decision.discipline == DISCIPLINE_YELLOW)
    straight_red = discipline_applies & (decision.discipline == DISCIPLINE_RED)
    yellow_cards = state.players.yellow_cards.at[safe_offender].add(
        yellow.astype(state.players.yellow_cards.dtype)
    )
    second_yellow = yellow & (yellow_cards[safe_offender] >= YELLOW_CARD_SEND_OFF_COUNT)
    dismissed = straight_red | second_yellow
    disciplined_players = state.players._replace(
        velocity=state.players.velocity.at[safe_offender].set(
            jnp.where(
                dismissed,
                jnp.zeros_like(state.players.velocity[safe_offender]),
                state.players.velocity[safe_offender],
            )
        ),
        on_pitch=state.players.on_pitch.at[safe_offender].set(
            state.players.on_pitch[safe_offender] & (~dismissed)
        ),
        sent_off=state.players.sent_off.at[safe_offender].set(
            state.players.sent_off[safe_offender] | dismissed
        ),
        yellow_cards=yellow_cards,
    )
    disciplined_state = state._replace(players=disciplined_players)

    restart_position = canonical_restart_spot(
        decision.restart_kind,
        decision.restart_team,
        decision.restart_basis_position,
        state.attack_direction,
        indirect=decision.indirect,
        stadium=stadium,
        ball=ball_geometry,
    )
    taker = select_restart_taker(
        disciplined_state,
        decision.restart_kind,
        decision.restart_team,
        restart_position,
        stadium=stadium,
    )
    stopped_state = disciplined_state._replace(
        ball=disciplined_state.ball._replace(
            position=restart_position,
            velocity=jnp.zeros_like(state.ball.velocity),
            spin=jnp.zeros_like(state.ball.spin),
            live=jnp.bool_(False),
        ),
        possession=disciplined_state.possession._replace(
            team=jnp.int32(NO_TEAM),
            player=jnp.int32(NO_PLAYER),
            previous_team=state.possession.team.astype(jnp.int32),
            control_ticks=jnp.int32(0),
        ),
        restart=disciplined_state.restart._replace(
            kind=decision.restart_kind,
            team=decision.restart_team,
            substeps_remaining=jnp.int32(0),
            taker=taker,
            indirect=decision.indirect,
            opened_control_tick=disciplined_state.control_tick,
        ),
        restart_release=RestartReleaseProvenance(
            active=jnp.bool_(False),
            untouched=jnp.bool_(False),
            kind=jnp.int32(RK_NONE),
            team=jnp.int32(NO_TEAM),
            taker=jnp.int32(NO_PLAYER),
            indirect=jnp.bool_(False),
            law11_direct_exempt=jnp.bool_(False),
            release_mechanism=jnp.int32(MECHANISM_NONE),
        ),
        gk_backpass_team=jnp.int32(NO_TEAM),
    )
    next_state = jax.tree_util.tree_map(
        lambda stopped, disciplined: jnp.where(
            decision.restart_awarded, stopped, disciplined
        ),
        stopped_state,
        disciplined_state,
    )
    return FoulResolution(state=next_state, adjudication=adjudication)
