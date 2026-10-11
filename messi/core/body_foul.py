"""실현된 상대 선수 body impact의 direct-contact foul 판정."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.action import INTENT_CHALLENGE
from messi.core.collision import PlayerImpactFacts
from messi.core.constants import (
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    NO_PLAYER,
    RK_NONE,
    TEAM_0,
    TEAM_1,
)
from messi.core.contact import OUTCOME_FOUL
from messi.core.contest import SAMPLE_CONTEST, ContestPhysics, ContestResult
from messi.core.foul import (
    FoulEvent,
    FoulLaw,
    FoulResolution,
    empty_foul_event,
    resolve_contest_foul,
)
from messi.core.numeric import require_float32_representable
from messi.core.randomness import RandomEvent, event_random_key


@dataclass(frozen=True, slots=True)
class BodyFoulPhysics:
    """실제 충돌 하나를 foul로 선언하는 bounded design prior."""

    minimum_closing_speed_mps: float = 5.5
    ball_near_distance_m: float = 1.5
    base_logit: float = -7.8
    closing_speed_logit_weight_per_mps: float = 0.18
    behind_logit_weight: float = 1.30
    shoulder_alignment_logit_discount: float = 1.20
    ball_far_logit_weight: float = 1.00
    possessed_victim_logit_weight: float = 0.50
    goal_denial_logit_weight: float = 0.50
    probability_floor: float = 0.003
    probability_ceiling: float = 0.10
    rare_case_coverage_multiplier: float = 1.15

    def __post_init__(self) -> None:
        for name in (
            "minimum_closing_speed_mps",
            "ball_near_distance_m",
            "closing_speed_logit_weight_per_mps",
            "behind_logit_weight",
            "shoulder_alignment_logit_discount",
            "ball_far_logit_weight",
            "possessed_victim_logit_weight",
            "goal_denial_logit_weight",
            "rare_case_coverage_multiplier",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        require_float32_representable("base_logit", self.base_logit)
        for name in ("probability_floor", "probability_ceiling"):
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}은 [0, 1] 범위여야 한다")
        if self.probability_floor > self.probability_ceiling:
            raise ValueError("probability_floor는 probability_ceiling 이하여야 한다")


class BodyFoulOverride(NamedTuple):
    """Replay용 occurrence·discipline override. ``SAMPLE_CONTEST``는 sampling이다."""

    occurred: jax.Array
    discipline: jax.Array


class BodyFoulRandomDraws(NamedTuple):
    """Stable substep key에서 미리 생성한 body-foul 확률 draw."""

    occurrence: jax.Array
    card: jax.Array
    colour: jax.Array


class BodyFoulResult(NamedTuple):
    state: object
    event: FoulEvent
    contest: ContestResult
    eligible: jax.Array
    probability: jax.Array
    override_valid: jax.Array


def sample_body_foul_draws(key: jax.Array) -> BodyFoulRandomDraws:
    return BodyFoulRandomDraws(
        occurrence=jax.random.uniform(event_random_key(key, RandomEvent.BODY_FOUL)),
        card=jax.random.uniform(event_random_key(key, RandomEvent.BODY_CARD)),
        colour=jax.random.uniform(
            event_random_key(key, RandomEvent.BODY_CARD_COLOUR)
        ),
    )


def sample_body_foul_override() -> BodyFoulOverride:
    return BodyFoulOverride(jnp.int32(SAMPLE_CONTEST), jnp.int32(SAMPLE_CONTEST))


def resolve_body_impact_foul(
    state,
    roster,
    impact: PlayerImpactFacts,
    requested_intent: jax.Array,
    key: jax.Array,
    field_half_extent: jax.Array,
    override: BodyFoulOverride | None = None,
    random_draws: BodyFoulRandomDraws | None = None,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    ball_radius_m: float = 0.11,
    config: BodyFoulPhysics = BodyFoulPhysics(),
    discipline_config: ContestPhysics = ContestPhysics(),
    law: FoulLaw = FoulLaw(),
    apply_resolution: bool = True,
) -> BodyFoulResult:
    """가장 강한 실현 body impact 하나를 판정하고 restart를 원자적으로 적용한다.

    충돌 pair 검색이나 collision response는 반복하지 않는다. 명시 override 중 하나라도
    현재 context와 맞지 않으면 foul·card를 모두 적용하지 않는 fail-closed 결과를 낸다.
    """

    count = state.players.position.shape[0]
    provisional_actor = jnp.asarray(impact.actor, dtype=jnp.int32)
    provisional_victim = jnp.asarray(impact.victim, dtype=jnp.int32)
    actor_in_range = (provisional_actor >= 0) & (provisional_actor < count)
    victim_in_range = (provisional_victim >= 0) & (provisional_victim < count)
    safe_actor0 = jnp.clip(provisional_actor, 0, count - 1)
    safe_victim0 = jnp.clip(provisional_victim, 0, count - 1)
    intent = jnp.asarray(requested_intent, dtype=jnp.int32)
    actor_challenges = intent[safe_actor0] == INTENT_CHALLENGE
    victim_challenges = intent[safe_victim0] == INTENT_CHALLENGE
    unique_challenge = actor_challenges != victim_challenges
    actor_possesses = state.possession.player == provisional_actor
    victim_possesses = state.possession.player == provisional_victim
    unique_possession = actor_possesses != victim_possesses
    decisive = jnp.asarray(impact.actor_attribution_decisive, dtype=jnp.bool_)
    swap = (~decisive) & (
        (unique_challenge & victim_challenges)
        | ((~unique_challenge) & unique_possession & actor_possesses)
    )
    actor = jnp.where(swap, provisional_victim, provisional_actor).astype(jnp.int32)
    victim = jnp.where(swap, provisional_actor, provisional_victim).astype(jnp.int32)
    safe_actor = jnp.clip(actor, 0, count - 1)
    safe_victim = jnp.clip(victim, 0, count - 1)
    actor_team = roster.team_id[safe_actor].astype(jnp.int32)
    victim_team = roster.team_id[safe_victim].astype(jnp.int32)
    normal0 = jnp.asarray(impact.contact_normal, dtype=state.ball.position.dtype)
    normal_norm = jnp.linalg.norm(normal0)
    normal = jnp.where(swap, -normal0, normal0) / jnp.maximum(normal_norm, 1.0e-9)
    contact_xy = jnp.asarray(impact.contact_position, dtype=state.ball.position.dtype)
    closing_speed = jnp.asarray(impact.impact_score, dtype=contact_xy.dtype)
    finite = (
        jnp.isfinite(closing_speed)
        & jnp.isfinite(impact.velocity_alignment)
        & jnp.all(jnp.isfinite(normal))
        & jnp.all(jnp.isfinite(contact_xy))
    )
    eligible = (
        impact.occurred
        & state.ball.live
        & (state.restart.kind == RK_NONE)
        & actor_in_range
        & victim_in_range
        & state.players.active[safe_actor]
        & state.players.active[safe_victim]
        & (actor_team != victim_team)
        & ((actor_team == TEAM_0) | (actor_team == TEAM_1))
        & ((victim_team == TEAM_0) | (victim_team == TEAM_1))
        & finite
        & (normal_norm > 1.0e-9)
        & (closing_speed > config.minimum_closing_speed_mps)
        & (decisive | unique_challenge | unique_possession)
    )

    victim_forward = state.players.body_forward[safe_victim]
    behind = jnp.clip(jnp.dot(normal, victim_forward), 0.0, 1.0)
    lateral = 1.0 - jnp.abs(jnp.dot(normal, victim_forward))
    shoulder = jnp.clip(impact.velocity_alignment, 0.0, 1.0) * lateral
    ball_far = (
        jnp.linalg.norm(state.ball.position[:2] - contact_xy)
        > config.ball_near_distance_m
    ).astype(contact_xy.dtype)
    possessed_victim = (state.possession.player == victim).astype(contact_xy.dtype)
    victim_attack = jnp.asarray(
        [state.attack_direction[jnp.clip(victim_team, TEAM_0, TEAM_1)], 0.0],
        dtype=contact_xy.dtype,
    )
    goal_denial = possessed_victim * jnp.clip(-jnp.dot(normal, victim_attack), 0.0, 1.0)
    logit = (
        config.base_logit
        + config.closing_speed_logit_weight_per_mps
        * jnp.maximum(closing_speed - config.minimum_closing_speed_mps, 0.0)
        + config.behind_logit_weight * behind
        - config.shoulder_alignment_logit_discount * shoulder
        + config.ball_far_logit_weight * ball_far
        + config.possessed_victim_logit_weight * possessed_victim
        + config.goal_denial_logit_weight * goal_denial
    )
    probability = jnp.where(
        eligible,
        jnp.clip(
            jnp.clip(
                jax.nn.sigmoid(logit),
                config.probability_floor,
                config.probability_ceiling,
            )
            * config.rare_case_coverage_multiplier,
            0.0,
            1.0,
        ),
        0.0,
    ).astype(jnp.float32)
    draws = sample_body_foul_draws(key) if random_draws is None else random_draws
    occurrence_draw = draws.occurrence
    card_draw = draws.card
    colour_draw = draws.colour
    sampled_occurred = eligible & (occurrence_draw < probability)
    card_probability = jnp.clip(
        discipline_config.card_probability_midpoint
        * (0.75 + 0.5 * jnp.clip(regulation_elapsed_fraction, 0.0, 1.0)),
        0.0,
        1.0,
    )
    carded = card_draw < card_probability
    red = carded & (
        colour_draw < discipline_config.direct_red_given_card_probability
    )
    sampled_discipline_given_foul = jnp.where(
        red,
        DISCIPLINE_RED,
        jnp.where(carded, DISCIPLINE_YELLOW, DISCIPLINE_NONE),
    ).astype(jnp.int32)

    if override is None:
        override_valid = jnp.bool_(True)
        occurred = sampled_occurred
        discipline = jnp.where(
            occurred, sampled_discipline_given_foul, DISCIPLINE_NONE
        ).astype(jnp.int32)
    else:
        requested_occurred = jnp.asarray(override.occurred, dtype=jnp.int32)
        requested_discipline = jnp.asarray(override.discipline, dtype=jnp.int32)
        occurrence_explicit = requested_occurred != SAMPLE_CONTEST
        occurrence_code_valid = (
            (requested_occurred == SAMPLE_CONTEST)
            | (requested_occurred == 0)
            | (requested_occurred == 1)
        )
        occurrence_compatible = (~(requested_occurred == 1)) | eligible
        candidate_occurred = jnp.where(
            occurrence_explicit, requested_occurred == 1, sampled_occurred
        )
        discipline_explicit = requested_discipline != SAMPLE_CONTEST
        discipline_code_valid = (
            (requested_discipline == SAMPLE_CONTEST)
            | (requested_discipline == DISCIPLINE_NONE)
            | (requested_discipline == DISCIPLINE_YELLOW)
            | (requested_discipline == DISCIPLINE_RED)
        )
        discipline_compatible = jnp.where(
            candidate_occurred,
            (requested_discipline == SAMPLE_CONTEST)
            | (requested_discipline == DISCIPLINE_NONE)
            | (requested_discipline == DISCIPLINE_YELLOW)
            | (requested_discipline == DISCIPLINE_RED),
            (requested_discipline == SAMPLE_CONTEST)
            | (requested_discipline == DISCIPLINE_NONE),
        )
        override_valid = (
            occurrence_code_valid
            & occurrence_compatible
            & discipline_code_valid
            & discipline_compatible
        )
        occurred = override_valid & candidate_occurred
        discipline = jnp.where(
            occurred,
            jnp.where(
                discipline_explicit,
                requested_discipline,
                sampled_discipline_given_foul,
            ),
            DISCIPLINE_NONE,
        ).astype(jnp.int32)

    contest = ContestResult(
        selected=occurred,
        occurred=occurred,
        actor=actor,
        challenger=actor,
        outcome=jnp.where(occurred, OUTCOME_FOUL, 0).astype(jnp.int32),
        discipline=discipline,
        foul_actor=jnp.where(occurred, actor, NO_PLAYER).astype(jnp.int32),
        foul_victim=jnp.where(occurred, victim, NO_PLAYER).astype(jnp.int32),
        offence_position=contact_xy,
        override_valid=override_valid,
    )
    if apply_resolution:
        resolution: FoulResolution = resolve_contest_foul(
            state,
            roster,
            contest,
            field_half_extent,
            ball_radius_m=ball_radius_m,
            law=law,
        )
    else:
        resolution = FoulResolution(state, empty_foul_event(state.ball.position.dtype))
    yellow = resolution.event.occurred & (discipline == DISCIPLINE_YELLOW)
    direct_red = resolution.event.occurred & (discipline == DISCIPLINE_RED)
    second_yellow = yellow & (state.players.yellow_cards[safe_actor] >= 1)
    players = resolution.state.players._replace(
        yellow_cards=resolution.state.players.yellow_cards.at[safe_actor].add(
            yellow.astype(jnp.int32)
        ),
        sent_off=resolution.state.players.sent_off.at[safe_actor].set(
            resolution.state.players.sent_off[safe_actor]
            | direct_red
            | second_yellow
        ),
    )
    return BodyFoulResult(
        state=resolution.state._replace(players=players),
        event=resolution.event,
        contest=contest,
        eligible=eligible,
        probability=probability,
        override_valid=override_valid,
    )


__all__ = [
    "BodyFoulOverride",
    "BodyFoulPhysics",
    "BodyFoulRandomDraws",
    "BodyFoulResult",
    "resolve_body_impact_foul",
    "sample_body_foul_draws",
    "sample_body_foul_override",
]
