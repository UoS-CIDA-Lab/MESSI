"""명시적 replay override를 지원하는 fixed-shape stochastic contest."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.action import INTENT_CONTROL, INTENT_PASS, INTENT_SHOOT
from messi.core.constants import (
    DISCIPLINE_NONE,
    DISCIPLINE_RED,
    DISCIPLINE_YELLOW,
    NO_PLAYER,
)
from messi.core.contact import (
    OUTCOME_CATCH,
    OUTCOME_DEFLECTION,
    OUTCOME_FOUL,
    OUTCOME_INTERCEPTION,
    OUTCOME_NONE,
    OUTCOME_PARRY,
    OUTCOME_RELEASE,
    OUTCOME_TACKLE_WON,
    ContactGeometry,
)
from messi.core.numeric import require_float32_representable
from messi.core.randomness import RandomEvent, event_random_key, validate_prng_key

SAMPLE_CONTEST = -2
_DIV_EPS = 1.0e-9


class ContestOverride(NamedTuple):
    """각 stochastic 출력의 scalar replay pin.

    `SAMPLE_CONTEST`는 정상 sampling, `winner=NO_PLAYER`는 접촉 없음 강제다.
    """

    winner: jax.Array
    outcome: jax.Array
    discipline: jax.Array


class ContestResult(NamedTuple):
    selected: jax.Array
    occurred: jax.Array
    actor: jax.Array
    challenger: jax.Array
    outcome: jax.Array
    discipline: jax.Array
    foul_actor: jax.Array
    foul_victim: jax.Array
    offence_position: jax.Array
    override_valid: jax.Array


@dataclass(frozen=True, slots=True)
class ContestPhysics:
    """actor selection과 conditional outcome의 bounded prior."""

    distance_weight: float = 1.0
    reach_time_weight: float = 1.0
    height_fit_weight: float = 0.55
    possession_weight: float = 0.0
    ball_control_weight: float = 0.3
    temperature: float = 0.66
    # Neutral ground challenge의 clean-control prior다. DFL7 ground duel에서
    # 무소유 선수가 이긴 비율은 518/1083=47.8%였지만, 그 전부가 즉시 소유권
    # 전환은 아니므로 clean win과 loose-ball disruption을 분리한다.
    tackle_success_probability: float = 0.40
    tackle_success_context_logit_limit: float = 1.10
    tackle_foul_probability: float = 0.04
    tackle_foul_context_logit_limit: float = math.log(4.0)
    tackle_foul_own_penalty_area_multiplier: float = 0.15
    penalty_area_length_m: float = 16.5
    penalty_area_half_width_m: float = 20.16
    # Clean win이 아닌 ground challenge가 loose ball을 만드는 조건부 prior다.
    # 0.40 + (1 - 0.40) * 0.13 = 0.478로 DFL7의 defensive duel-win
    # 비율과 같은 중립 총 disruption prior를 만든다. 이 집계의 피파울 승리는
    # MESSI에 아직 없는 carrier offensive-foul 경로의 attack-stop proxy다.
    tackle_deflection_probability: float = 0.13
    goalkeeper_catch_speed_midpoint_mps: float = 21.3
    goalkeeper_catch_speed_scale_mps: float = 9.9
    card_probability_midpoint: float = 0.1297
    # Card 확률은 파울 지점의 전진도와 경과 시간을 logit에서 조건화한다. 선형
    # midpoint × (0.75 + 0.5f) 근사는 전진도를 아예 쓰지 않았다. 계수는 FootballWorld의
    # K-League receipt 적합값이다.
    card_attack_progress_logit_weight: float = -1.2860
    card_elapsed_fraction_logit_weight: float = 0.8028
    # 실제 sampling에는 coverage 보정본을 쓴다. body foul에만 적용돼 있어 tackle
    # 파울이 그만큼 적게 발생했다.
    tackle_foul_rare_case_coverage_multiplier: float = 1.15
    direct_red_given_card_probability: float = 0.03125
    carry_radius_m: float = 1.1
    challenge_radius_m: float = 1.4

    def __post_init__(self) -> None:
        nonnegative = (
            "distance_weight",
            "reach_time_weight",
            "height_fit_weight",
            "possession_weight",
            "ball_control_weight",
            "tackle_success_context_logit_limit",
            "tackle_foul_context_logit_limit",
        )
        for name in nonnegative:
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 10.0:
                raise ValueError(f"{name}은 [0, 10] 범위여야 한다")
        for name in (
            "tackle_success_probability",
            "tackle_foul_probability",
            "tackle_deflection_probability",
            "card_probability_midpoint",
            "direct_red_given_card_probability",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}은 [0, 1] 범위여야 한다")
        for name in (
            "temperature",
            "goalkeeper_catch_speed_midpoint_mps",
            "goalkeeper_catch_speed_scale_mps",
            "carry_radius_m",
            "challenge_radius_m",
        ):
            value = require_float32_representable(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name}은 0보다 커야 한다")
        for name in ("temperature", "goalkeeper_catch_speed_scale_mps"):
            represented = np.float32(getattr(self, name))
            if represented < np.finfo(np.float32).tiny:
                raise ValueError(
                    f"{name}는 float32 최소 positive normal 이상이어야 한다"
                )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                reciprocal = np.reciprocal(represented)
            if not np.isfinite(reciprocal):
                raise ValueError(f"{name}의 float32 reciprocal은 유한해야 한다")
        if self.challenge_radius_m < self.carry_radius_m:
            raise ValueError("challenge_radius_m은 carry_radius_m 이상이어야 한다")
        for name in (
            "tackle_foul_rare_case_coverage_multiplier",
            "tackle_foul_own_penalty_area_multiplier",
            "card_attack_progress_logit_weight",
            "card_elapsed_fraction_logit_weight",
        ):
            require_float32_representable(name, getattr(self, name))
        if self.tackle_foul_rare_case_coverage_multiplier < 0.0:
            raise ValueError(
                "tackle_foul_rare_case_coverage_multiplier은 0 이상이어야 한다"
            )
        if not 0.0 <= self.tackle_foul_own_penalty_area_multiplier <= 1.0:
            raise ValueError(
                "tackle_foul_own_penalty_area_multiplier은 [0, 1] 범위여야 한다"
            )
        for name in ("penalty_area_length_m", "penalty_area_half_width_m"):
            value = require_float32_representable(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name}은 0보다 커야 한다")



_DEFAULT_CONTEST_PHYSICS = ContestPhysics()


def sample_contest_override() -> ContestOverride:
    """일반 forward rollout에서 모든 stochastic 출력을 sampling한다."""

    return ContestOverride(
        winner=jnp.int32(SAMPLE_CONTEST),
        outcome=jnp.int32(SAMPLE_CONTEST),
        discipline=jnp.int32(SAMPLE_CONTEST),
    )


def _empty_result(valid: jax.Array) -> ContestResult:
    return ContestResult(
        selected=jnp.bool_(False),
        occurred=jnp.bool_(False),
        actor=jnp.int32(NO_PLAYER),
        challenger=jnp.int32(NO_PLAYER),
        outcome=jnp.int32(OUTCOME_NONE),
        discipline=jnp.int32(DISCIPLINE_NONE),
        foul_actor=jnp.int32(NO_PLAYER),
        foul_victim=jnp.int32(NO_PLAYER),
        offence_position=jnp.zeros(2, dtype=jnp.float32),
        override_valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


def _contextual_probability(
    reference: float,
    favourable: jax.Array,
    logit_limit: float,
) -> jax.Array:
    dtype = jnp.asarray(favourable).dtype
    raw = jnp.asarray(reference, dtype=dtype)
    bounded = jnp.clip(raw, 1.0e-6, 1.0 - 1.0e-6)
    logit = jnp.log(bounded) - jnp.log1p(-bounded)
    result = jax.nn.sigmoid(logit + logit_limit * (2.0 * favourable - 1.0))
    return jnp.where(raw <= 0.0, 0.0, jnp.where(raw >= 1.0, 1.0, result))


def _challenge_success_probability(
    closing: jax.Array,
    rear_alignment: jax.Array,
    lunge: jax.Array,
    control_advantage: jax.Array,
    config: ContestPhysics,
) -> jax.Array:
    """중립 0.40에서 자세·reach·속도·능력으로 clean-win odds를 옮긴다."""

    favourable = _challenge_success_favourability(
        closing,
        rear_alignment,
        lunge,
        control_advantage,
    )
    return _contextual_probability(
        config.tackle_success_probability,
        favourable,
        config.tackle_success_context_logit_limit,
    )


def _challenge_success_favourability(
    closing: jax.Array,
    rear_alignment: jax.Array,
    lunge: jax.Array,
    control_advantage: jax.Array,
) -> jax.Array:
    """Ground challenge의 유리함을 중립 0.5인 bounded score로 만든다.

    rear_alignment는 challenger에서 carrier로 향하는 단위 벡터와 carrier
    forward의 내적이다. -1은 정면/goal-side, 0은 측면, +1은 후방 추격이다.
    Position과 실제 reach를 주 prior로 두고, 과도한 closing과 능력 차는 보조
    항으로 둔다. 모든 입력이 중립이면 정확히 0.5다.
    """

    controlled_closing = 1.0 - jnp.clip(closing, 0.0, 1.0)
    position = 0.5 * (1.0 - jnp.clip(rear_alignment, -1.0, 1.0))
    reach = 1.0 - jnp.clip(lunge, 0.0, 1.0)
    control = jnp.clip(control_advantage, 0.0, 1.0)
    return (
        jnp.float32(0.15) * controlled_closing
        + jnp.float32(0.35) * position
        + jnp.float32(0.30) * reach
        + jnp.float32(0.20) * control
    )


def _attack_progress(state, roster, actor, field_half_extent):
    """파울 지점의 공격 방향 전진도를 [0, 1]로 만든다.

    MESSI는 FootballWorld와 달리 ``team_id``를 ``PlayerState``가 아니라
    ``RosterState``에 둔다.
    """

    position = state.players.position[actor]
    team = jnp.clip(roster.team_id[actor], 0, state.attack_direction.shape[0] - 1)
    half_length = jnp.asarray(field_half_extent, dtype=position.dtype)[0]
    return jnp.clip(
        0.5
        * (
            position[0]
            * state.attack_direction[team]
            / jnp.maximum(half_length, 1.0e-6)
            + 1.0
        ),
        0.0,
        1.0,
    )


def challenge_card_probability(
    attack_progress,
    regulation_elapsed_fraction,
    *,
    config: ContestPhysics = _DEFAULT_CONTEST_PHYSICS,
):
    """파울이 선언됐을 때의 card 위험을 전진도와 경과 시간으로 조건화한다.

    선형 ``midpoint * (0.75 + 0.5f)`` 근사는 전진도를 쓰지 않아 파울 위치와 무관하게
    같은 값을 냈다. 수비 진영일수록 card가 잦다는 관계는 전진도 가중치가 음수인
    것으로 표현된다.
    """

    reference = jnp.asarray(config.card_probability_midpoint, jnp.float32)
    bounded = jnp.clip(reference, 1.0e-6, 1.0 - 1.0e-6)
    reference_logit = jnp.log(bounded) - jnp.log1p(-bounded)
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


def _resolve_contest(
    state,
    roster,
    geometry: ContactGeometry,
    key: jax.Array,
    override: ContestOverride | None,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    field_half_extent=(52.5, 34.0),
    config: ContestPhysics = _DEFAULT_CONTEST_PHYSICS,
) -> ContestResult:
    """eligible actor와 context-compatible outcome을 선택한다.

    Override는 random 결과만 대체하며 eligibility, challenge context 또는 goalkeeper
    gate를 새로 만들 수 없다. `override_valid=False`는 요청한 replay가 현재 물리
    context와 호환되지 않았음을 뜻한다.
    """

    validate_prng_key(key, name="key")
    candidate = jnp.asarray(geometry.possible, dtype=jnp.bool_) & state.players.active
    if override is not None:
        override = ContestOverride(
            winner=jnp.asarray(override.winner, dtype=jnp.int32),
            outcome=jnp.asarray(override.outcome, dtype=jnp.int32),
            discipline=jnp.asarray(override.discipline, dtype=jnp.int32),
        )
    any_candidate = jnp.any(candidate)

    def active(_: None) -> ContestResult:
        distance = geometry.distance
        time_to_reach = distance / jnp.maximum(roster.max_speed, _DIV_EPS)
        height_fit = jnp.clip(
            1.0 - state.ball.position[2] / jnp.maximum(roster.reach_height, _DIV_EPS),
            0.0,
            1.0,
        )
        possession_side = (
            (state.possession.team >= 0) & (roster.team_id == state.possession.team)
        ).astype(distance.dtype)
        raw_score = (
            -config.distance_weight * distance
            - config.reach_time_weight * time_to_reach
            + config.height_fit_weight * height_fit
            + config.possession_weight * possession_side
            + config.ball_control_weight * roster.ball_control
        )
        score = raw_score / config.temperature
        # 정상 설정은 기존 연산을 그대로 보존한다. 극단적으로 작은 temperature에서만
        # eligible 공통 최대값을 먼저 빼 Gumbel-max와 동등한 ranking을 유한하게 만든다.
        needs_centering = jnp.any(candidate & (~jnp.isfinite(score)))
        eligible_max = jnp.max(jnp.where(candidate, raw_score, -jnp.inf))
        centered_score = (raw_score - eligible_max) / config.temperature
        selection_score = jnp.where(needs_centering, centered_score, score)
        winner_key = event_random_key(key, RandomEvent.CONTEST_WINNER)
        sampled_actor = jnp.argmax(
            jnp.where(
                candidate,
                selection_score
                + jax.random.gumbel(
                    winner_key, selection_score.shape, dtype=selection_score.dtype
                ),
                -jnp.inf,
            )
        ).astype(jnp.int32)
        if override is None:
            winner_valid = jnp.bool_(True)
            actor = sampled_actor
        else:
            request_sample = override.winner == SAMPLE_CONTEST
            request_none = override.winner == NO_PLAYER
            safe_forced = jnp.clip(override.winner, 0, candidate.shape[0] - 1)
            forced_eligible = (
                (override.winner >= 0)
                & (override.winner < candidate.shape[0])
                & candidate[safe_forced]
            )
            winner_valid = request_sample | request_none | forced_eligible
            actor = jnp.where(
                request_sample,
                sampled_actor,
                jnp.where(forced_eligible, override.winner, NO_PLAYER),
            ).astype(jnp.int32)
        selected = actor != NO_PLAYER
        safe_actor = jnp.clip(actor, 0, candidate.shape[0] - 1)
        goalkeeper = selected & geometry.goalkeeper_hand[safe_actor]
        challenge = selected & geometry.challenge_context[safe_actor]

        last_actor = state.possession.last_contact.actor
        last_valid = (last_actor >= 0) & (last_actor < candidate.shape[0])
        safe_last = jnp.clip(last_actor, 0, candidate.shape[0] - 1)
        last_release = (
            last_valid
            & (state.possession.last_contact.outcome == OUTCOME_RELEASE)
            & (
                (state.possession.last_contact.intent == INTENT_PASS)
                | (state.possession.last_contact.intent == INTENT_SHOOT)
            )
        )
        interception = (
            selected
            & (geometry.requested_intent[safe_actor] == INTENT_CONTROL)
            & (state.possession.player == NO_PLAYER)
            & last_release
            & (roster.team_id[safe_actor] != roster.team_id[safe_last])
            & (state.possession.last_contact.actor != safe_actor)
            & (state.ball.live)
            & (state.ball.velocity @ state.ball.velocity > 1.0e-6)
            & (state.possession.last_contact.outcome == OUTCOME_RELEASE)
            & (state.players.active[safe_actor])
            & (geometry.mechanism[safe_actor] != 0)
            & (state.possession.last_contact.actor != NO_PLAYER)
            & (state.possession.last_contact.intent != INTENT_CONTROL)
        )

        horizontal_speed = jnp.linalg.norm(state.ball.velocity[:2])
        catch_probability = jax.nn.sigmoid(
            (config.goalkeeper_catch_speed_midpoint_mps - horizontal_speed)
            / config.goalkeeper_catch_speed_scale_mps
        )
        outcome_key = event_random_key(key, RandomEvent.CONTEST_OUTCOME)
        goalkeeper_sample = jnp.where(
            jax.random.uniform(outcome_key) < catch_probability,
            OUTCOME_CATCH,
            OUTCOME_PARRY,
        )

        carrier = geometry.verified_carrier
        safe_carrier = jnp.clip(carrier, 0, candidate.shape[0] - 1)
        actor_to_carrier = (
            state.players.position[safe_carrier] - state.players.position[safe_actor]
        )
        separation = jnp.linalg.norm(actor_to_carrier)
        approach = actor_to_carrier / jnp.maximum(separation, _DIV_EPS)
        relative_velocity = (
            state.players.velocity[safe_actor] - state.players.velocity[safe_carrier]
        )
        closing = jnp.clip(
            jnp.maximum(jnp.dot(relative_velocity, approach), 0.0)
            / jnp.maximum(
                roster.max_speed[safe_actor] + roster.max_speed[safe_carrier],
                _DIV_EPS,
            ),
            0.0,
            1.0,
        )
        carrier_forward = state.players.body_forward[safe_carrier]
        carrier_forward = carrier_forward / jnp.maximum(
            jnp.linalg.norm(carrier_forward), _DIV_EPS
        )
        rear_alignment = jnp.clip(jnp.dot(approach, carrier_forward), -1.0, 1.0)
        rear_risk = jnp.clip(rear_alignment, 0.0, 1.0)
        lunge = jnp.clip(
            (geometry.horizontal_extension[safe_actor] - config.carry_radius_m)
            / jnp.maximum(config.challenge_radius_m - config.carry_radius_m, _DIV_EPS),
            0.0,
            1.0,
        )
        control_advantage = jnp.clip(
            0.5
            + 0.5
            * (roster.ball_control[safe_actor] - roster.ball_control[safe_carrier]),
            0.0,
            1.0,
        )
        success_probability = _challenge_success_probability(
            closing,
            rear_alignment,
            lunge,
            control_advantage,
            config,
        )
        field_half_length = jnp.asarray(field_half_extent[0], jnp.float32)
        actor_team = roster.team_id[safe_actor]
        actor_team_x = (
            state.players.position[safe_actor, 0]
            * state.attack_direction[actor_team]
        )
        in_own_penalty_area = (
            actor_team_x
            <= -(field_half_length - jnp.float32(config.penalty_area_length_m))
        ) & (
            jnp.abs(state.players.position[safe_actor, 1])
            <= jnp.float32(config.penalty_area_half_width_m)
        )
        penalty_area_multiplier = jnp.where(
            in_own_penalty_area,
            jnp.float32(config.tackle_foul_own_penalty_area_multiplier),
            jnp.float32(1.0),
        )
        foul_probability = jnp.clip(
            _contextual_probability(
                config.tackle_foul_probability,
                (closing + rear_risk + lunge) / 3.0,
                config.tackle_foul_context_logit_limit,
            )
            * jnp.asarray(
                config.tackle_foul_rare_case_coverage_multiplier, jnp.float32
            ),
            0.0,
            1.0,
        ) * penalty_area_multiplier
        foul = (
            jax.random.uniform(event_random_key(key, RandomEvent.CONTEST_FOUL))
            < foul_probability
        )
        success = (
            jax.random.uniform(event_random_key(key, RandomEvent.CONTEST_SUCCESS))
            < success_probability
        )
        deflection = (
            jax.random.uniform(event_random_key(key, RandomEvent.CONTEST_DEFLECTION))
            < config.tackle_deflection_probability
        )
        challenge_sample = jnp.where(
            foul,
            OUTCOME_FOUL,
            jnp.where(
                success,
                OUTCOME_TACKLE_WON,
                jnp.where(deflection, OUTCOME_DEFLECTION, OUTCOME_NONE),
            ),
        )
        ordinary = jnp.where(interception, OUTCOME_INTERCEPTION, OUTCOME_RELEASE)
        sampled_outcome = jnp.where(
            goalkeeper,
            goalkeeper_sample,
            jnp.where(challenge, challenge_sample, ordinary),
        ).astype(jnp.int32)

        if override is None:
            outcome_valid = jnp.bool_(True)
            outcome = jnp.where(selected, sampled_outcome, OUTCOME_NONE).astype(
                jnp.int32
            )
        else:
            outcome_requested = override.outcome != SAMPLE_CONTEST
            goalkeeper_compatible = (override.outcome == OUTCOME_CATCH) | (
                override.outcome == OUTCOME_PARRY
            )
            challenge_compatible = (
                (override.outcome == OUTCOME_NONE)
                | (override.outcome == OUTCOME_TACKLE_WON)
                | (override.outcome == OUTCOME_DEFLECTION)
                | (override.outcome == OUTCOME_FOUL)
            )
            ordinary_compatible = override.outcome == ordinary
            selected_compatible = jnp.where(
                goalkeeper,
                goalkeeper_compatible,
                jnp.where(challenge, challenge_compatible, ordinary_compatible),
            )
            no_actor_compatible = override.outcome == OUTCOME_NONE
            outcome_compatible = jnp.where(
                selected, selected_compatible, no_actor_compatible
            )
            outcome_valid = (~outcome_requested) | outcome_compatible
            outcome = jnp.where(
                selected,
                jnp.where(
                    outcome_requested & outcome_compatible,
                    override.outcome,
                    sampled_outcome,
                ),
                OUTCOME_NONE,
            ).astype(jnp.int32)

        card_probability = challenge_card_probability(
            _attack_progress(state, roster, safe_actor, field_half_extent),
            regulation_elapsed_fraction,
            config=config,
        )
        carded = (
            jax.random.uniform(event_random_key(key, RandomEvent.CONTEST_CARD))
            < card_probability
        )
        direct_red = carded & (
            jax.random.uniform(event_random_key(key, RandomEvent.CONTEST_CARD_COLOUR))
            < config.direct_red_given_card_probability
        )
        sampled_discipline = jnp.where(
            outcome == OUTCOME_FOUL,
            jnp.where(
                direct_red,
                DISCIPLINE_RED,
                jnp.where(carded, DISCIPLINE_YELLOW, DISCIPLINE_NONE),
            ),
            DISCIPLINE_NONE,
        ).astype(jnp.int32)
        if override is None:
            discipline_valid = jnp.bool_(True)
            discipline = sampled_discipline
        else:
            discipline_requested = override.discipline != SAMPLE_CONTEST
            discipline_compatible = jnp.where(
                outcome == OUTCOME_FOUL,
                (override.discipline == DISCIPLINE_NONE)
                | (override.discipline == DISCIPLINE_YELLOW)
                | (override.discipline == DISCIPLINE_RED),
                override.discipline == DISCIPLINE_NONE,
            )
            discipline_valid = (~discipline_requested) | discipline_compatible
            discipline = jnp.where(
                discipline_requested & discipline_compatible,
                override.discipline,
                sampled_discipline,
            ).astype(jnp.int32)
        foul_realized = selected & (outcome == OUTCOME_FOUL)
        offence_position = 0.5 * (
            state.players.position[safe_actor] + state.players.position[safe_carrier]
        )
        accepted = ContestResult(
            selected=selected,
            occurred=selected & (outcome != OUTCOME_NONE),
            actor=jnp.where(selected, actor, NO_PLAYER).astype(jnp.int32),
            challenger=jnp.where(challenge, actor, NO_PLAYER).astype(jnp.int32),
            outcome=outcome,
            discipline=discipline,
            foul_actor=jnp.where(foul_realized, actor, NO_PLAYER).astype(jnp.int32),
            foul_victim=jnp.where(foul_realized, carrier, NO_PLAYER).astype(jnp.int32),
            offence_position=jnp.where(
                foul_realized, offence_position, jnp.zeros_like(offence_position)
            ),
            override_valid=winner_valid & outcome_valid & discipline_valid,
        )
        if override is None:
            return accepted
        valid = accepted.override_valid
        rejected = _empty_result(jnp.bool_(False))
        return jax.tree.map(
            lambda accepted_leaf, rejected_leaf: jnp.where(
                valid, accepted_leaf, rejected_leaf
            ),
            accepted,
            rejected,
        )

    if override is None:
        no_candidate_valid = jnp.bool_(True)
    else:
        no_candidate_valid = (
            ((override.winner == SAMPLE_CONTEST) | (override.winner == NO_PLAYER))
            & (
                (override.outcome == SAMPLE_CONTEST)
                | (override.outcome == OUTCOME_NONE)
            )
            & (
                (override.discipline == SAMPLE_CONTEST)
                | (override.discipline == DISCIPLINE_NONE)
            )
        )
    return jax.lax.cond(
        any_candidate,
        active,
        lambda _: _empty_result(no_candidate_valid),
        operand=None,
    )


def resolve_contest_sample(
    state,
    roster,
    geometry: ContactGeometry,
    key: jax.Array,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    field_half_extent=(52.5, 34.0),
    config: ContestPhysics = _DEFAULT_CONTEST_PHYSICS,
) -> ContestResult:
    """Override 입력과 검증 연산이 없는 일반 학습용 contest."""

    return _resolve_contest(
        state,
        roster,
        geometry,
        key,
        None,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        field_half_extent=field_half_extent,
        config=config,
    )


def resolve_contest(
    state,
    roster,
    geometry: ContactGeometry,
    key: jax.Array,
    override: ContestOverride,
    *,
    regulation_elapsed_fraction: jax.Array = 0.0,
    field_half_extent=(52.5, 34.0),
    config: ContestPhysics = _DEFAULT_CONTEST_PHYSICS,
) -> ContestResult:
    """Replay override를 검증하고 적용하는 contest."""

    return _resolve_contest(
        state,
        roster,
        geometry,
        key,
        override,
        regulation_elapsed_fraction=regulation_elapsed_fraction,
        field_half_extent=field_half_extent,
        config=config,
    )


__all__ = [
    "SAMPLE_CONTEST",
    "ContestOverride",
    "ContestPhysics",
    "ContestResult",
    "resolve_contest",
    "resolve_contest_sample",
    "sample_contest_override",
]
