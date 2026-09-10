"""Fixed-shape stochastic selection for player-ball contests.

This module does not decide physical or legal eligibility.  Its ``candidate``
and ``goalkeeper_claim`` masks must be produced by the authoritative reach and
rule gates.  Forced replay values replace stochastic draws only; they can
never add an actor outside those masks or turn an ordinary play into a
challenge or goalkeeper claim. Policies select one of six intents separately
from eight continuous controls. The contest still produces tackle,
interception, deflection, catch, and parry as outcomes rather than
requested-action categories.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.contest import Contest
from footballworld.config.geometry import Stadium
from footballworld.core.constants import (
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    DIV_EPS,
    GEOMETRY_EPS,
    NO_PLAYER,
    NO_TEAM,
    SQUARED_EPS,
)
from footballworld.core.contact import (
    OUTCOME_CATCH,
    OUTCOME_DEFLECTION,
    OUTCOME_FOUL,
    OUTCOME_INTERCEPTION,
    OUTCOME_NONE,
    OUTCOME_PARRY,
    OUTCOME_RELEASE,
    OUTCOME_TACKLE_WON,
)
from footballworld.core.randomness import RandomEvent, _event_random_key_unchecked
from footballworld.core.state import State

SAMPLE_CONTEST = -2


class ContestOverride(NamedTuple):
    """Scalar replay pins; ``SAMPLE_CONTEST`` requests the normal draw."""

    winner: jax.Array
    outcome: jax.Array


class ContestResult(NamedTuple):
    """One fixed-shape contest decision without applying ball physics.

    ``outcome`` is the selected contest context.  The realized physical
    response, including control failure or a miscontrol, is reported by the
    caller's :class:`~footballworld.core.contact.ContactResult`.
    ``selected`` reports that an eligible actor was selected. ``occurred``
    reports a non-``OUTCOME_NONE`` resolution; for ``OUTCOME_FOUL`` this need
    not imply that the actor touched the ball. ``parameters_applied`` tells the
    caller whether the selected actor's submitted ball parameters shape the
    outgoing state. This flag never authorizes energy addition; the selected
    physical mechanism retains that authority. ``offence_position`` is the
    authoritative horizontal capsule-contact point for a realized foul.
    """

    selected: jax.Array
    occurred: jax.Array
    actor: jax.Array
    challenger: jax.Array
    outcome: jax.Array
    parameters_applied: jax.Array
    foul_actor: jax.Array
    foul_victim: jax.Array
    offence_position: jax.Array
    discipline: jax.Array
    override_valid: jax.Array


def sample_contest_override() -> ContestOverride:
    """Return the fixed-shape sentinel used by ordinary forward rollout."""

    return ContestOverride(
        winner=jnp.int32(SAMPLE_CONTEST),
        outcome=jnp.int32(SAMPLE_CONTEST),
    )


def _empty_result(*, override_valid: jax.Array) -> ContestResult:
    return ContestResult(
        selected=jnp.bool_(False),
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        challenger=jnp.int32(NO_PLAYER),
        outcome=jnp.int32(OUTCOME_NONE),
        parameters_applied=jnp.bool_(False),
        foul_actor=jnp.int32(NO_PLAYER),
        foul_victim=jnp.int32(NO_PLAYER),
        offence_position=jnp.zeros(2, dtype=jnp.float32),
        discipline=jnp.int32(DISCIPLINE_NONE),
        override_valid=jnp.asarray(override_valid, dtype=jnp.bool_),
    )


def _point_segment_pair(
    point: jax.Array, start: jax.Array, end: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return ``point`` and its nearest point on one finite segment."""

    segment = end - start
    fraction = jnp.clip(
        jnp.sum((point - start) * segment)
        / (jnp.sum(segment * segment) + jnp.asarray(DIV_EPS, point.dtype)),
        0.0,
        1.0,
    )
    return point, start + fraction * segment


def _capsule_contact_point(
    position_a: jax.Array,
    body_forward_a: jax.Array,
    position_b: jax.Array,
    body_forward_b: jax.Array,
    *,
    body: BodyContact,
) -> jax.Array:
    """Return the midpoint of the closest points on two torso capsules.

    Both torso capsules have the same radius, so the midpoint of their
    closest core-segment points is also the midpoint between the two nearest
    capsule surfaces. This remains defined for overlap and adds no coefficient.
    """

    half_core = jnp.asarray(
        0.5 * (body.shoulder_width_m - body.torso_depth_m),
        dtype=position_a.dtype,
    )
    shoulder_a = jnp.asarray(
        [-body_forward_a[1], body_forward_a[0]], dtype=position_a.dtype
    )
    shoulder_b = jnp.asarray(
        [-body_forward_b[1], body_forward_b[0]], dtype=position_a.dtype
    )
    a_start = position_a - half_core * shoulder_a
    a_end = position_a + half_core * shoulder_a
    b_start = position_b - half_core * shoulder_b
    b_end = position_b + half_core * shoulder_b

    a0, b_on_a0 = _point_segment_pair(a_start, b_start, b_end)
    a1, b_on_a1 = _point_segment_pair(a_end, b_start, b_end)
    b0, a_on_b0 = _point_segment_pair(b_start, a_start, a_end)
    b1, a_on_b1 = _point_segment_pair(b_end, a_start, a_end)
    a_mid, b_on_a_mid = _point_segment_pair(position_a, b_start, b_end)
    b_mid, a_on_b_mid = _point_segment_pair(position_b, a_start, a_end)
    segment_a = a_end - a_start
    segment_b = b_end - b_start
    relative_start = b_start - a_start
    denominator = segment_a[0] * segment_b[1] - segment_a[1] * segment_b[0]
    safe_denominator = jnp.where(jnp.abs(denominator) > GEOMETRY_EPS, denominator, 1.0)
    fraction_a = (
        relative_start[0] * segment_b[1] - relative_start[1] * segment_b[0]
    ) / safe_denominator
    fraction_b = (
        relative_start[0] * segment_a[1] - relative_start[1] * segment_a[0]
    ) / safe_denominator
    crossing = (
        (jnp.abs(denominator) > GEOMETRY_EPS)
        & (fraction_a >= 0.0)
        & (fraction_a <= 1.0)
        & (fraction_b >= 0.0)
        & (fraction_b <= 1.0)
    )
    intersection_a = a_start + fraction_a * segment_a
    intersection_b = b_start + fraction_b * segment_b
    points_a = jnp.stack(
        (
            a0,
            a1,
            a_on_b0,
            a_on_b1,
            a_mid,
            a_on_b_mid,
            intersection_a,
        ),
        axis=0,
    )
    points_b = jnp.stack(
        (
            b_on_a0,
            b_on_a1,
            b0,
            b1,
            b_on_a_mid,
            b_mid,
            intersection_b,
        ),
        axis=0,
    )
    distance_squared = jnp.sum((points_b - points_a) ** 2, axis=-1)
    distance_squared = distance_squared.at[6].set(jnp.where(crossing, 0.0, jnp.inf))
    minimum = jnp.min(distance_squared)
    midpoint = 0.5 * (points_a + points_b)
    centre_midpoint = 0.5 * (position_a + position_b)
    midpoint_distance_squared = jnp.sum((midpoint - centre_midpoint) ** 2, axis=-1)
    nearest = jnp.argmin(
        jnp.where(
            distance_squared <= minimum + SQUARED_EPS,
            midpoint_distance_squared,
            jnp.inf,
        )
    )
    return 0.5 * (points_a[nearest] + points_b[nearest])


def _contest_score(
    state: State,
    distance_xy: jax.Array,
    *,
    config: Contest,
) -> jax.Array:
    time_to_reach = distance_xy / jnp.maximum(state.players.max_speed, DIV_EPS)
    height_fit = jnp.clip(
        1.0 - state.ball.position[2] / jnp.maximum(state.players.reach_height, DIV_EPS),
        0.0,
        1.0,
    )
    possession_side = (
        (state.possession.team != NO_TEAM)
        & (state.players.team_id == state.possession.team)
    ).astype(distance_xy.dtype)
    return (
        -config.distance_weight * distance_xy
        - config.reach_time_weight * time_to_reach
        + config.height_fit_weight * height_fit
        + config.possession_weight * possession_side
        + config.ball_control_weight * state.players.ball_control
    )


def _sample_actor(
    score: jax.Array,
    candidate: jax.Array,
    key: jax.Array,
    forced_winner: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    noise = jax.random.gumbel(key, score.shape, dtype=score.dtype)
    sampled = jnp.argmax(jnp.where(candidate, score + noise, -jnp.inf)).astype(
        jnp.int32
    )

    forced_winner = jnp.asarray(forced_winner, dtype=jnp.int32)
    request_sample = forced_winner == SAMPLE_CONTEST
    request_none = forced_winner == NO_PLAYER
    safe_forced = jnp.clip(forced_winner, 0, candidate.shape[0] - 1)
    forced_eligible = (
        (forced_winner >= 0)
        & (forced_winner < candidate.shape[0])
        & candidate[safe_forced]
    )
    forced_valid = request_sample | request_none | forced_eligible
    actor = jnp.where(
        request_sample,
        sampled,
        jnp.where(forced_eligible, forced_winner, NO_PLAYER),
    ).astype(jnp.int32)
    selected = actor != NO_PLAYER
    return actor, selected, forced_valid


def _goalkeeper_outcome(
    horizontal_speed: jax.Array,
    key: jax.Array,
    forced_outcome: jax.Array,
    require_parry: jax.Array,
    *,
    config: Contest,
) -> tuple[jax.Array, jax.Array]:
    scale = jnp.maximum(
        jnp.asarray(
            config.goalkeeper_catch_speed_scale_mps,
            dtype=horizontal_speed.dtype,
        ),
        DIV_EPS,
    )
    catch_probability = jax.nn.sigmoid(
        (
            jnp.asarray(
                config.goalkeeper_catch_speed_midpoint_mps,
                dtype=horizontal_speed.dtype,
            )
            - horizontal_speed
        )
        / scale
    )
    sampled_catch_or_parry = jnp.where(
        jax.random.uniform(key, dtype=horizontal_speed.dtype) < catch_probability,
        OUTCOME_CATCH,
        OUTCOME_PARRY,
    ).astype(jnp.int32)
    sampled = jnp.where(require_parry, OUTCOME_PARRY, sampled_catch_or_parry).astype(
        jnp.int32
    )
    requested = forced_outcome != SAMPLE_CONTEST
    compatible = jnp.where(
        require_parry,
        forced_outcome == OUTCOME_PARRY,
        (forced_outcome == OUTCOME_CATCH) | (forced_outcome == OUTCOME_PARRY),
    )
    return (
        jnp.where(requested & compatible, forced_outcome, sampled).astype(jnp.int32),
        (~requested) | compatible,
    )


def _challenge_outcome(
    key: jax.Array,
    forced_outcome: jax.Array,
    foul_probability: jax.Array,
    success_probability: jax.Array,
    card_probability: jax.Array,
    *,
    config: Contest,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    foul_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_FOUL)
    success_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_SUCCESS)
    deflect_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_DEFLECTION)
    card_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_CARD)
    color_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_CARD_COLOUR)
    foul = (
        jax.random.uniform(foul_key, dtype=jnp.asarray(foul_probability).dtype)
        < foul_probability
    )
    success_dtype = jnp.asarray(success_probability).dtype
    card_dtype = jnp.asarray(card_probability).dtype
    success = jax.random.uniform(success_key, dtype=success_dtype) < success_probability
    deflection = (
        jax.random.uniform(deflect_key, dtype=success_dtype)
        < config.tackle_deflection_probability
    )
    sampled = jnp.where(
        foul,
        OUTCOME_FOUL,
        jnp.where(
            success,
            OUTCOME_TACKLE_WON,
            jnp.where(deflection, OUTCOME_DEFLECTION, OUTCOME_NONE),
        ),
    ).astype(jnp.int32)
    requested = forced_outcome != SAMPLE_CONTEST
    compatible = (
        (forced_outcome == OUTCOME_NONE)
        | (forced_outcome == OUTCOME_TACKLE_WON)
        | (forced_outcome == OUTCOME_DEFLECTION)
        | (forced_outcome == OUTCOME_FOUL)
    )
    outcome = jnp.where(requested & compatible, forced_outcome, sampled).astype(
        jnp.int32
    )
    carded = jax.random.uniform(card_key, dtype=card_dtype) < card_probability
    direct_red = carded & (
        jax.random.uniform(color_key, dtype=card_dtype)
        < config.direct_red_given_card_probability
    )
    discipline = jnp.where(
        outcome == OUTCOME_FOUL,
        jnp.where(
            direct_red,
            DISCIPLINE_RED,
            jnp.where(carded, DISCIPLINE_YELLOW, DISCIPLINE_NONE),
        ),
        DISCIPLINE_NONE,
    ).astype(jnp.int32)
    return outcome, discipline, (~requested) | compatible


def challenge_success_probability(
    closing_fraction: jax.Array,
    behind_fraction: jax.Array,
    lunge_fraction: jax.Array,
    ball_control_advantage: jax.Array,
    *,
    config: Contest = Contest(),
) -> jax.Array:
    """Return tackle success from bounded, observable duel context."""

    dtype = jnp.result_type(
        closing_fraction,
        behind_fraction,
        lunge_fraction,
        ball_control_advantage,
        jnp.float32,
    )
    closing = jnp.clip(jnp.asarray(closing_fraction, dtype=dtype), 0.0, 1.0)
    behind = jnp.clip(jnp.asarray(behind_fraction, dtype=dtype), 0.0, 1.0)
    lunge = jnp.clip(jnp.asarray(lunge_fraction, dtype=dtype), 0.0, 1.0)
    control = jnp.clip(
        0.5 + 0.5 * jnp.asarray(ball_control_advantage, dtype=dtype),
        0.0,
        1.0,
    )
    favourable = ((1.0 - closing) + (1.0 - behind) + lunge + control) / 4.0
    raw_reference = jnp.asarray(config.tackle_success_probability, dtype=dtype)
    reference = jnp.clip(raw_reference, 1.0e-6, 1.0 - 1.0e-6)
    reference_logit = jnp.log(reference) - jnp.log1p(-reference)
    modifier = jnp.asarray(config.tackle_success_context_logit_limit, dtype=dtype) * (
        2.0 * favourable - 1.0
    )
    contextual = jax.nn.sigmoid(reference_logit + modifier)
    return jnp.where(
        raw_reference <= 0.0,
        jnp.asarray(0.0, dtype=dtype),
        jnp.where(raw_reference >= 1.0, jnp.asarray(1.0, dtype=dtype), contextual),
    ).astype(jnp.float32)


def challenge_foul_probability(
    closing_fraction: jax.Array,
    behind_fraction: jax.Array,
    lunge_fraction: jax.Array,
    *,
    config: Contest = Contest(),
    rare_case_coverage: bool = True,
) -> jax.Array:
    """Return contextual foul risk from bounded observable duel geometry."""

    dtype = jnp.result_type(
        closing_fraction, behind_fraction, lunge_fraction, jnp.float32
    )
    context = (
        jnp.clip(jnp.asarray(closing_fraction, dtype=dtype), 0.0, 1.0)
        + jnp.clip(jnp.asarray(behind_fraction, dtype=dtype), 0.0, 1.0)
        + jnp.clip(jnp.asarray(lunge_fraction, dtype=dtype), 0.0, 1.0)
    ) / 3.0
    raw_reference = jnp.asarray(config.tackle_foul_probability, dtype=dtype)
    reference = jnp.clip(raw_reference, 1.0e-6, 1.0 - 1.0e-6)
    reference_logit = jnp.log(reference) - jnp.log1p(-reference)
    modifier = jnp.asarray(config.tackle_foul_context_logit_limit, dtype=dtype) * (
        2.0 * context - 1.0
    )
    contextual = jax.nn.sigmoid(reference_logit + modifier)
    baseline = jnp.where(
        raw_reference <= 0.0,
        jnp.asarray(0.0, dtype=dtype),
        jnp.where(raw_reference >= 1.0, jnp.asarray(1.0, dtype=dtype), contextual),
    )
    if rare_case_coverage:
        baseline = jnp.clip(
            baseline
            * jnp.asarray(
                config.tackle_foul_rare_case_coverage_multiplier, dtype=dtype
            ),
            0.0,
            1.0,
        )
    return baseline.astype(jnp.float32)


def _challenge_foul_probabilities(
    state: State,
    actor: jax.Array,
    controlled_carrier: jax.Array,
    lunge_fraction: jax.Array,
    *,
    config: Contest,
) -> tuple[jax.Array, jax.Array]:
    """Return baseline and research-coverage challenge-foul probabilities.

    Tackle fouls are conditioned on realized duel context. Separately weighted
    occurrence coefficients are not identifiable from the available provider
    feed, so the runtime uses only three
    dimensionless, already-available kinematic summaries with equal weight and
    bounds their combined logit effect. Submitted ball-force controls are not
    treated as tackle biomechanics.

    ``tackle_foul_probability`` is the pre-coverage probability at context 0.5.
    The second result applies the explicitly separate rare-case multiplier.
    """

    dtype = state.players.position.dtype
    actor = jnp.asarray(actor, dtype=jnp.int32)
    controlled_carrier = jnp.asarray(controlled_carrier, dtype=jnp.int32)
    actor_position = state.players.position[actor]
    carrier_position = state.players.position[controlled_carrier]
    actor_to_carrier = carrier_position - actor_position
    separation = jnp.linalg.norm(actor_to_carrier)
    approach_direction = actor_to_carrier / jnp.maximum(
        separation, jnp.asarray(DIV_EPS, dtype=dtype)
    )
    relative_velocity = (
        state.players.velocity[actor] - state.players.velocity[controlled_carrier]
    )
    closing_speed = jnp.maximum(jnp.dot(relative_velocity, approach_direction), 0.0)
    relative_speed_support = jnp.maximum(
        state.players.max_speed[actor] + state.players.max_speed[controlled_carrier],
        jnp.asarray(DIV_EPS, dtype=dtype),
    )
    closing_fraction = jnp.clip(closing_speed / relative_speed_support, 0.0, 1.0)
    carrier_body = state.players.body_forward[controlled_carrier]
    carrier_body = carrier_body / jnp.maximum(
        jnp.linalg.norm(carrier_body), jnp.asarray(DIV_EPS, dtype=dtype)
    )
    behind_fraction = jnp.clip(jnp.dot(approach_direction, carrier_body), 0.0, 1.0)
    lunge_fraction = jnp.clip(jnp.asarray(lunge_fraction, dtype=dtype), 0.0, 1.0)
    baseline = challenge_foul_probability(
        closing_fraction,
        behind_fraction,
        lunge_fraction,
        config=config,
        rare_case_coverage=False,
    )
    covered = challenge_foul_probability(
        closing_fraction,
        behind_fraction,
        lunge_fraction,
        config=config,
        rare_case_coverage=True,
    )
    return baseline, covered


def challenge_card_probability(
    attack_progress: jax.Array,
    regulation_elapsed_fraction: jax.Array,
    *,
    config: Contest = Contest(),
) -> jax.Array:
    """Return discipline risk conditional on a declared contact foul.

    The K-League receipt identifies this conditional model, not the separate
    probability that a challenge becomes a foul.
    """

    reference = jnp.asarray(config.card_probability_midpoint, jnp.float32)
    safe_reference = jnp.clip(reference, 1.0e-6, 1.0 - 1.0e-6)
    reference_logit = jnp.log(safe_reference) - jnp.log1p(-safe_reference)
    logit = (
        reference_logit
        + jnp.asarray(config.card_attack_progress_logit_weight, jnp.float32)
        * (jnp.clip(attack_progress, 0.0, 1.0) - 0.5)
        + jnp.asarray(config.card_elapsed_fraction_logit_weight, jnp.float32)
        * (jnp.clip(regulation_elapsed_fraction, 0.0, 1.0) - 0.5)
    )
    contextual = jax.nn.sigmoid(logit)
    return jnp.where(
        reference <= 0.0,
        jnp.float32(0.0),
        jnp.where(reference >= 1.0, jnp.float32(1.0), contextual),
    )


def resolve_contest(
    state: State,
    candidate: jax.Array,
    distance_xy: jax.Array,
    goalkeeper_claim: jax.Array,
    controlled_carrier: int | jax.Array,
    key: jax.Array,
    override: ContestOverride,
    *,
    goalkeeper_clear: jax.Array | None = None,
    challenge_request: jax.Array | None = None,
    challenge_interception: jax.Array | None = None,
    challenge_lunge_fraction: jax.Array | None = None,
    regulation_elapsed_fraction: jax.Array = 0.0,
    config: Contest = Contest(),
    body: BodyContact = BodyContact(),
    stadium: Stadium = Stadium(),
) -> ContestResult:
    """Select an eligible actor and sample its context-compatible outcome.

    ``controlled_carrier`` must identify the possession player whose physical
    control was verified by the caller, or ``NO_PLAYER``.  Merely retaining a
    possession-team provenance while a pass travels must not create a tackle.
    Goalkeeper claim eligibility likewise remains a caller-owned physical and
    law gate.  The catch curve intentionally uses horizontal ball speed to
    retain the convention of its compatibility prior.
    """

    candidate = jnp.asarray(candidate, dtype=jnp.bool_) & state.players.active
    distance_xy = jnp.asarray(distance_xy, dtype=state.players.position.dtype)
    goalkeeper_claim = (
        jnp.asarray(goalkeeper_claim, dtype=jnp.bool_)
        & candidate
        & state.players.is_goalkeeper
    )
    controlled_carrier = jnp.asarray(controlled_carrier, dtype=jnp.int32)
    goalkeeper_clear = (
        jnp.zeros_like(candidate)
        if goalkeeper_clear is None
        else jnp.asarray(goalkeeper_clear, dtype=jnp.bool_)
    )
    challenge_request = (
        jnp.zeros_like(candidate)
        if challenge_request is None
        else jnp.asarray(challenge_request, dtype=jnp.bool_)
    )
    challenge_interception = (
        jnp.zeros_like(candidate)
        if challenge_interception is None
        else jnp.asarray(challenge_interception, dtype=jnp.bool_)
    )
    challenge_lunge_fraction = (
        jnp.full_like(distance_xy, 0.5)
        if challenge_lunge_fraction is None
        else jnp.clip(
            jnp.asarray(challenge_lunge_fraction, dtype=distance_xy.dtype),
            0.0,
            1.0,
        )
    )
    override = ContestOverride(
        winner=jnp.asarray(override.winner, dtype=jnp.int32),
        outcome=jnp.asarray(override.outcome, dtype=jnp.int32),
    )

    any_candidate = jnp.any(candidate)

    def resolve_active(_: None) -> ContestResult:
        score = _contest_score(state, distance_xy, config=config)
        winner_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_WINNER)
        outcome_key = _event_random_key_unchecked(key, RandomEvent.CONTEST_OUTCOME)
        actor, selected, winner_override_valid = _sample_actor(
            score / config.temperature,
            candidate,
            winner_key,
            override.winner,
        )
        safe_actor = jnp.clip(actor, 0, candidate.shape[0] - 1)
        carrier_in_range = (controlled_carrier >= 0) & (
            controlled_carrier < candidate.shape[0]
        )
        safe_carrier = jnp.clip(controlled_carrier, 0, candidate.shape[0] - 1)
        carrier_valid = (
            carrier_in_range
            & state.players.active[safe_carrier]
            & (state.possession.player == controlled_carrier)
            & (state.possession.team == state.players.team_id[safe_carrier])
        )
        goalkeeper_context = selected & goalkeeper_claim[safe_actor]
        verified_carrier_challenge = (
            selected
            & carrier_valid
            & (state.players.team_id[safe_actor] != state.players.team_id[safe_carrier])
            & (~goalkeeper_context)
        )
        # The caller physical/rule predicates own whether this actor actually
        # requested a challenge. Proximity to an opposing carrier must never
        # promote CONTROL/PASS/SHOT/CLEAR into a tackle.
        challenge_context = verified_carrier_challenge & challenge_request[safe_actor]
        interception_context = selected & challenge_interception[safe_actor]

        horizontal_speed = jnp.linalg.norm(state.ball.velocity[:2])
        goalkeeper_outcome, goalkeeper_override_valid = _goalkeeper_outcome(
            horizontal_speed,
            outcome_key,
            override.outcome,
            goalkeeper_clear[safe_actor],
            config=config,
        )
        offence_position = _capsule_contact_point(
            state.players.position[safe_actor],
            state.players.body_forward[safe_actor],
            state.players.position[safe_carrier],
            state.players.body_forward[safe_carrier],
            body=body,
        )
        actor_team = jnp.clip(
            state.players.team_id[safe_actor],
            0,
            state.attack_direction.shape[0] - 1,
        )
        attack_progress = jnp.clip(
            0.5
            * (
                offence_position[0]
                * state.attack_direction[actor_team]
                / jnp.asarray(stadium.half_length, offence_position.dtype)
                + 1.0
            ),
            0.0,
            1.0,
        )
        card_probability = challenge_card_probability(
            attack_progress,
            regulation_elapsed_fraction,
            config=config,
        )
        _, challenge_foul_probability = _challenge_foul_probabilities(
            state,
            safe_actor,
            safe_carrier,
            challenge_lunge_fraction[safe_actor],
            config=config,
        )
        actor_to_carrier = (
            state.players.position[safe_carrier] - state.players.position[safe_actor]
        )
        separation = jnp.linalg.norm(actor_to_carrier)
        approach_direction = actor_to_carrier / jnp.maximum(
            separation, jnp.asarray(DIV_EPS, actor_to_carrier.dtype)
        )
        relative_velocity = (
            state.players.velocity[safe_actor] - state.players.velocity[safe_carrier]
        )
        closing_fraction = jnp.clip(
            jnp.maximum(jnp.dot(relative_velocity, approach_direction), 0.0)
            / jnp.maximum(
                state.players.max_speed[safe_actor]
                + state.players.max_speed[safe_carrier],
                jnp.asarray(DIV_EPS, actor_to_carrier.dtype),
            ),
            0.0,
            1.0,
        )
        carrier_body = state.players.body_forward[safe_carrier]
        carrier_body = carrier_body / jnp.maximum(
            jnp.linalg.norm(carrier_body), jnp.asarray(DIV_EPS, carrier_body.dtype)
        )
        behind_fraction = jnp.clip(jnp.dot(approach_direction, carrier_body), 0.0, 1.0)
        success_probability = challenge_success_probability(
            closing_fraction,
            behind_fraction,
            challenge_lunge_fraction[safe_actor],
            state.players.ball_control[safe_actor]
            - state.players.ball_control[safe_carrier],
            config=config,
        )
        (
            challenge_outcome,
            challenge_discipline,
            challenge_override_valid,
        ) = _challenge_outcome(
            outcome_key,
            override.outcome,
            challenge_foul_probability,
            success_probability,
            card_probability,
            config=config,
        )
        ordinary_outcome = jnp.where(
            interception_context, OUTCOME_INTERCEPTION, OUTCOME_RELEASE
        ).astype(jnp.int32)
        deterministic_override_valid = override.outcome == SAMPLE_CONTEST
        outcome = jnp.where(
            goalkeeper_context,
            goalkeeper_outcome,
            jnp.where(challenge_context, challenge_outcome, ordinary_outcome),
        ).astype(jnp.int32)
        outcome = jnp.where(selected, outcome, OUTCOME_NONE).astype(jnp.int32)
        outcome_override_valid = jnp.where(
            ~selected,
            override.outcome == SAMPLE_CONTEST,
            jnp.where(
                goalkeeper_context,
                goalkeeper_override_valid,
                jnp.where(
                    challenge_context,
                    challenge_override_valid,
                    deterministic_override_valid,
                ),
            ),
        )
        parameters_applied = (
            (outcome == OUTCOME_RELEASE)
            | (outcome == OUTCOME_INTERCEPTION)
            | (outcome == OUTCOME_TACKLE_WON)
        )
        foul = selected & (outcome == OUTCOME_FOUL)
        challenger = jnp.where(challenge_context, actor, NO_PLAYER).astype(jnp.int32)
        return ContestResult(
            selected=selected,
            occurred=selected & (outcome != OUTCOME_NONE),
            actor=jnp.where(selected, actor, NO_PLAYER).astype(jnp.int32),
            challenger=challenger,
            outcome=outcome,
            parameters_applied=parameters_applied,
            foul_actor=jnp.where(foul, actor, NO_PLAYER).astype(jnp.int32),
            foul_victim=jnp.where(foul, controlled_carrier, NO_PLAYER).astype(
                jnp.int32
            ),
            offence_position=jnp.where(
                foul, offence_position, jnp.zeros_like(offence_position)
            ),
            discipline=jnp.where(foul, challenge_discipline, DISCIPLINE_NONE).astype(
                jnp.int32
            ),
            override_valid=winner_override_valid & outcome_override_valid,
        )

    no_candidate_override_valid = (
        (override.winner == SAMPLE_CONTEST) | (override.winner == NO_PLAYER)
    ) & (override.outcome == SAMPLE_CONTEST)
    return jax.lax.cond(
        any_candidate,
        resolve_active,
        lambda _: _empty_result(override_valid=no_candidate_override_valid),
        operand=None,
    )
