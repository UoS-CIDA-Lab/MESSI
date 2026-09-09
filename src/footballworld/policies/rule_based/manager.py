"""Low-frequency rule manager over the public normalized manager view.

The policy uses documented empirical priors for substitution timing, window
size, positional replacement frequency, and set-piece roles.
It improves the random contract: every draw is a pure function of an immutable
match key and stable match facts, never the order in which batches or rollout
chunks happen to be evaluated. Environment legality remains authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    NO_PLAYER,
    RESTART_COUNT,
    RK_CORNER,
    RK_FREEKICK,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_OFFSIDE,
    RK_PENALTY,
    RK_THROWIN,
)
from footballworld.core.numeric import require_float32_representable
from footballworld.environment.management import (
    ManagerActingGoalkeeperCommand,
    ManagerCommand,
    ManagerFormationCommand,
    ManagerSetPieceTakerCommand,
    ManagerSubstitutionCommand,
)
from footballworld.environment.normalization import (
    NormalizationContext,
    NormalizedManagerObservation,
)
from footballworld.policies.manager import (
    NO_POLICY_PARAMETERS,
    ManagerPolicyStep,
    NoPolicyParameters,
)
from footballworld.rules.restart import (
    RESTART_TAKER_TEMPERATURE,
    restart_taker_random_key,
    sample_restart_taker,
    score_restart_takers,
)

_SUBSTITUTION_RANDOM_STREAM = 0x53554253  # ASCII "SUBS".
_SUBSTITUTION_TIMING_STREAM = 0x54494D45  # ASCII "TIME".
_SUBSTITUTION_OUTGOING_STREAM = 0x4F555447  # ASCII "OUTG".
_SUBSTITUTION_INCOMING_STREAM = 0x494E434D  # ASCII "INCM".
# Stage-specific medians from 36 non-halftime same-team windows in the private
# seven-match DFL receipt (stage sample sizes 14, 14, and 8).  The policy uses
# them as scheduling anchors rather than claiming a fitted conditional hazard
# model.
_SUBSTITUTION_PROGRESS = jnp.asarray(
    [65.139 / 90.0, 80.739 / 90.0, 87.392 / 90.0], dtype=jnp.float32
)
# Turn the ordered anchors into disjoint nearest-anchor (Voronoi) intervals.
# Adjacent midpoints are the internal boundaries.  The two outer boundaries
# mirror half the nearest gap, bounded by halftime and regulation full time.
# These remain stratified design-prior targets, not a fitted substitution
# hazard.  No separately tunable spread coefficient is introduced.
_SUBSTITUTION_MIDPOINT = (
    _SUBSTITUTION_PROGRESS[:-1] + _SUBSTITUTION_PROGRESS[1:]
) * jnp.float32(0.5)
_SUBSTITUTION_PROGRESS_LOWER = jnp.concatenate(
    (
        jnp.asarray(
            [
                jnp.maximum(
                    jnp.float32(0.5),
                    _SUBSTITUTION_PROGRESS[0]
                    - jnp.float32(0.5)
                    * (_SUBSTITUTION_PROGRESS[1] - _SUBSTITUTION_PROGRESS[0]),
                )
            ],
            dtype=jnp.float32,
        ),
        _SUBSTITUTION_MIDPOINT,
    )
)
_SUBSTITUTION_PROGRESS_UPPER = jnp.concatenate(
    (
        _SUBSTITUTION_MIDPOINT,
        jnp.asarray(
            [
                jnp.minimum(
                    jnp.float32(1.0),
                    _SUBSTITUTION_PROGRESS[-1]
                    + jnp.float32(0.5)
                    * (_SUBSTITUTION_PROGRESS[-1] - _SUBSTITUTION_PROGRESS[-2]),
                )
            ],
            dtype=jnp.float32,
        ),
    )
)
# Exact descriptive counts from 38 same-team windows in the private seven-match
# DFL receipt: 17 single, 16 double, and 5 triple substitutions.
_WINDOW_SIZE_PROBABILITY = jnp.asarray(
    [17.0 / 38.0, 16.0 / 38.0, 5.0 / 38.0], dtype=jnp.float32
)
# Match-kind preference shares fitted in the private calib simulator against
# the seven-match DFL top-taker shares.  Zero keeps independent event draws;
# one keeps a fully fixed match-kind hierarchy.  Throw-ins remain almost fully
# situational, while corners, free kicks, and kickoffs retain specialists.
_TAKER_PERSISTENCE = (
    jnp.zeros(RESTART_COUNT, dtype=jnp.float32)
    .at[RK_KICKOFF]
    .set(0.75)
    .at[RK_THROWIN]
    .set(0.20)
    .at[RK_GOALKICK]
    .set(0.10)
    .at[RK_CORNER]
    .set(0.50)
    .at[RK_FREEKICK]
    .set(0.45)
    .at[RK_PENALTY]
    .set(0.55)
    .at[RK_OFFSIDE]
    .set(0.45)
)
# These are respectively an aggregate event-size distribution and marginal
# player-event roles. They do not condition on the simulator's remaining
# resources or eligible lineup, so both are ranking/sizing priors rather than
# fitted command probabilities.
_ROLE_SUBSTITUTION_PROPENSITY = jnp.asarray(
    [0.000, 0.049, 0.100, 0.165, 0.220, 0.221, 0.259],
    dtype=jnp.float32,
)


def _prefer_declared_role_candidates(
    candidate: jax.Array,
    preferred_role_known: jax.Array,
    preferred_role_mask: jax.Array,
    outgoing_role: jax.Array,
) -> jax.Array:
    """Prefer declared role matches, preserving fallback when none exist."""

    safe_role = jnp.clip(jnp.asarray(outgoing_role, dtype=jnp.int32), 0, 6)
    declared_match = preferred_role_known & preferred_role_mask[:, safe_role]
    has_match = jnp.any(candidate & declared_match)
    return candidate & ((~has_match) | declared_match)


@dataclass(frozen=True, slots=True)
class RuleManagerConfig:
    """Small set of design priors around inherited aggregate anchors."""

    # The source 15.9% is a share of individual substitution events, not a
    # Bernoulli probability per team-match. This remains a compatibility prior,
    # never a fitted halftime hazard.
    halftime_substitution_probability: float = 0.159
    taker_temperature: float = RESTART_TAKER_TEMPERATURE
    fatigue_weight: float = 1.0
    booking_weight: float = 0.25
    role_frequency_weight: float = 0.15
    # Candidate noise is a declared diversity prior, not a fitted conditional
    # substitution-choice model.
    substitution_choice_temperature: float = 0.30
    # Formation-response values are explicit design priors. They are not
    # inferred from provider events, which do not identify formation changes.
    formation_hold_seconds: float = 300.0
    formation_prior_weight: float = 0.08
    formation_attack_gain: float = 0.90
    formation_protect_gain: float = 0.75
    formation_width_gain: float = 0.12
    formation_incumbent_bonus: float = 0.30
    formation_noise_scale: float = 0.04

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{field.name} must be a real number")
            value = require_float32_representable(field.name, value)
            object.__setattr__(self, field.name, value)
        if not 0.0 <= self.halftime_substitution_probability <= 1.0:
            raise ValueError("halftime_substitution_probability must be in [0, 1]")
        if self.taker_temperature < 0.0:
            raise ValueError("taker_temperature must be non-negative")
        if self.substitution_choice_temperature <= 0.0:
            raise ValueError("substitution_choice_temperature must be positive")
        if self.formation_hold_seconds < 0.0:
            raise ValueError("formation_hold_seconds must be non-negative")
        for name in (
            "fatigue_weight",
            "booking_weight",
            "role_frequency_weight",
            "formation_prior_weight",
            "formation_attack_gain",
            "formation_protect_gain",
            "formation_width_gain",
            "formation_incumbent_bonus",
            "formation_noise_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class RuleManagerState(NamedTuple):
    """Small policy memory kept only in the rare manager executable."""

    processed_restart_tick: jax.Array
    processed_restart_kind: jax.Array
    formation_change_tick: jax.Array


def initialize_rule_manager_state() -> RuleManagerState:
    return RuleManagerState(
        processed_restart_tick=jnp.full(2, -1, dtype=jnp.int32),
        processed_restart_kind=jnp.full(2, RK_NONE, dtype=jnp.int32),
        formation_change_tick=jnp.zeros(2, dtype=jnp.int32),
    )


def _require_manager_observations(observations: NormalizedManagerObservation) -> None:
    if type(observations) is not NormalizedManagerObservation:
        raise TypeError(
            "RuleBasedManager requires normalized manager observations; "
            "use env.observe_managers()"
        )
    if observations.valid.shape != (2,):
        raise ValueError("manager observations must contain exactly two team rows")


def _fold_manager_key(
    match_key: jax.Array,
    restart_tick: jax.Array,
    team: int,
    decision: jax.Array,
) -> jax.Array:
    key = jax.random.fold_in(match_key, _SUBSTITUTION_RANDOM_STREAM)
    key = jax.random.fold_in(key, jnp.asarray(restart_tick, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.uint32(team))
    return jax.random.fold_in(key, jnp.asarray(decision, dtype=jnp.uint32))


def _formation_random_key(
    match_key: jax.Array,
    restart_tick: jax.Array,
    team: int,
    signature: jax.Array,
) -> jax.Array:
    """Key noise by catalog identity, never layout-array position."""

    key = _fold_manager_key(match_key, restart_tick, team, jnp.int32(0x464F524D))
    return jax.random.fold_in(key, signature.astype(jnp.uint32))


def _normalized_candidate_score(
    preference: jax.Array,
    candidate: jax.Array,
) -> jax.Array:
    """Normalize eligible ranking values without changing their order."""

    candidate = jnp.asarray(candidate, dtype=jnp.bool_)
    preference = jnp.asarray(preference, dtype=jnp.float32)
    has_candidate = jnp.any(candidate)
    lower = jnp.where(
        has_candidate,
        jnp.min(jnp.where(candidate, preference, jnp.inf)),
        jnp.float32(0.0),
    )
    upper = jnp.where(
        has_candidate,
        jnp.max(jnp.where(candidate, preference, -jnp.inf)),
        jnp.float32(0.0),
    )
    span = upper - lower
    normalized = jnp.where(
        span > jnp.float32(1.0e-6),
        (preference - lower) / jnp.maximum(span, jnp.float32(1.0e-6)),
        jnp.float32(0.5),
    )
    return jnp.where(candidate, normalized, jnp.float32(0.0)).astype(jnp.float32)


def _effective_formation_change_tick(
    policy_tick: jax.Array,
    tactical_epoch: jax.Array,
    observed_tick: jax.Array,
    counter_scale_ticks: int,
) -> jax.Array:
    """Synchronize the hold boundary with accepted external changes."""

    restored_tick = jnp.rint(observed_tick * jnp.float32(counter_scale_ticks)).astype(
        jnp.int32
    )
    observed_change = (tactical_epoch > 0) & (restored_tick >= 0)
    return jnp.where(
        observed_change,
        jnp.maximum(policy_tick, restored_tick),
        policy_tick,
    ).astype(jnp.int32)


def _identity_gumbel(base_key: jax.Array, identity: jax.Array) -> jax.Array:
    """Draw slot-order-invariant noise addressed by immutable player identity."""

    keys = jax.vmap(
        lambda value: jax.random.fold_in(base_key, value.astype(jnp.uint32))
    )(identity)
    return jax.vmap(lambda key: jax.random.gumbel(key, shape=(), dtype=jnp.float32))(
        keys
    )


def _substitution_target_progress(match_key: jax.Array) -> jax.Array:
    """Sample the immutable team-by-stage ordered target table."""

    key = jax.random.fold_in(match_key, jnp.uint32(_SUBSTITUTION_TIMING_STREAM))
    teams = jnp.arange(2, dtype=jnp.uint32)[:, None]
    stages = jnp.arange(_SUBSTITUTION_PROGRESS.shape[0], dtype=jnp.uint32)[None, :]
    identity = jnp.bitwise_or(jnp.left_shift(teams, jnp.uint32(16)), stages)
    identity = identity.reshape(-1)
    stage_keys = jax.lax.map(lambda value: jax.random.fold_in(key, value), identity)
    lower = jnp.broadcast_to(
        _SUBSTITUTION_PROGRESS_LOWER[None, :], (2, _SUBSTITUTION_PROGRESS.shape[0])
    ).reshape(-1)
    upper = jnp.broadcast_to(
        _SUBSTITUTION_PROGRESS_UPPER[None, :], (2, _SUBSTITUTION_PROGRESS.shape[0])
    ).reshape(-1)
    target = jax.lax.map(
        lambda values: jax.random.uniform(
            values[0],
            shape=(),
            dtype=jnp.float32,
            minval=values[1],
            maxval=values[2],
        ),
        (stage_keys, lower, upper),
    )
    return target.reshape(2, _SUBSTITUTION_PROGRESS.shape[0])


@dataclass(frozen=True)
class RuleBasedManager:
    """Observation-only manager evaluated once when a new restart opens."""

    config: RuleManagerConfig
    max_simultaneous: int
    context: NormalizationContext
    formation_hold_ticks: int

    @staticmethod
    def initialize(
        observations: NormalizedManagerObservation | None = None,
        parameters: NoPolicyParameters = NO_POLICY_PARAMETERS,
    ) -> RuleManagerState:
        """Initialize boundary memory; observations are accepted uniformly."""

        del observations
        if type(parameters) is not NoPolicyParameters:
            raise TypeError("rule manager parameters must be NoPolicyParameters")
        return initialize_rule_manager_state()

    def step(
        self,
        observations: NormalizedManagerObservation,
        match_key: jax.Array,
        state: RuleManagerState,
        parameters: NoPolicyParameters = NO_POLICY_PARAMETERS,
    ) -> ManagerPolicyStep[RuleManagerState]:
        """Return one fixed-shape manager transaction and updated memory."""

        _require_manager_observations(observations)
        if type(state) is not RuleManagerState:
            raise TypeError("state must be RuleManagerState")
        if type(parameters) is not NoPolicyParameters:
            raise TypeError("rule manager parameters must be NoPolicyParameters")

        counter_scale = jnp.float32(self.context.counter_scale_ticks)
        restart_tick = jnp.rint(
            observations.restart_opened_control_tick * counter_scale
        ).astype(jnp.int32)
        new_restart = (
            observations.valid
            & (observations.restart_kind > RK_NONE)
            & (restart_tick >= 0)
            & (
                (restart_tick != state.processed_restart_tick)
                | (observations.restart_kind != state.processed_restart_kind)
            )
        )

        substitutions = ManagerSubstitutionCommand.empty(self.max_simultaneous)
        substitution_target = _substitution_target_progress(match_key)
        acting = ManagerActingGoalkeeperCommand.empty()
        # One permanently-invalid pad cell keeps no-bench squads legal without
        # a separate traced policy shape or unsafe zero-width reductions.
        bench = jax.tree.map(
            lambda value: jnp.pad(
                value,
                ((0, 0), (0, 1)) + ((0, 0),) * (value.ndim - 2),
            ),
            observations.bench,
        )
        player_id = observations.on_field.player_id
        is_goalkeeper = observations.on_field.is_goalkeeper
        active = observations.on_field.active
        max_speed = observations.on_field.max_speed
        height = observations.on_field.height
        reach_height = observations.on_field.reach_height
        ball_control = observations.on_field.ball_control
        endurance = observations.on_field.endurance_factor
        stamina_long = observations.on_field.stamina_long

        for team in range(2):
            team_new_restart = new_restart[team]
            ordinary_restart = (
                observations.valid[team]
                & (observations.restart_kind[team] > RK_NONE)
                & (observations.restart_kind[team] != RK_GK_HOLD)
            )
            field = active[team] & (~is_goalkeeper[team])
            missing_goalkeeper = ~jnp.any(active[team] & is_goalkeeper[team])
            goalkeeper_emergency_boundary = ordinary_restart | (
                observations.valid[team]
                & (observations.restart_kind[team] == RK_GK_HOLD)
                & missing_goalkeeper
            )
            bench_goalkeeper = (
                bench.valid[team] & bench.available[team] & bench.is_goalkeeper[team]
            )
            has_bench_goalkeeper = jnp.any(bench_goalkeeper)
            goalkeeper_bench_index = jnp.argmax(
                bench_goalkeeper.astype(jnp.int32)
            ).astype(jnp.int32)

            attacking_depth = (
                observations.on_field.position[team, :, 0]
                * observations.attack_direction[team]
            )
            emergency_outgoing = jnp.argmax(
                jnp.where(field, attacking_depth, -jnp.inf)
            ).astype(jnp.int32)
            own_goal_x = -observations.attack_direction[team]
            own_goal_distance = (
                observations.on_field.position[team, :, 0] - own_goal_x
            ) ** 2 + observations.on_field.position[team, :, 1] ** 2
            acting_slot = jnp.argmin(
                jnp.where(field, own_goal_distance, jnp.inf)
            ).astype(jnp.int32)
            acting_request = (
                goalkeeper_emergency_boundary & missing_goalkeeper & jnp.any(field)
            )
            acting = acting._replace(
                requested=acting.requested.at[team].set(acting_request),
                player_slot=acting.player_slot.at[team].set(
                    jnp.where(acting_request, acting_slot, jnp.int32(NO_PLAYER))
                ),
            )

            substitutions_left = observations.substitutions_remaining[team]
            windows_left = observations.windows_remaining[team]
            windows_used = observations.windows_max[team] - windows_left
            stage = jnp.clip(windows_used, 0, _SUBSTITUTION_PROGRESS.shape[0] - 1)
            progress = observations.clock.match_regulation_progress[team]
            target_progress = substitution_target[team, stage]
            scheduled = progress >= target_progress
            halftime = (
                (observations.restart_kind[team] == RK_KICKOFF)
                & (observations.clock.period[team] == 2)
                & (progress <= jnp.float32(0.5001))
            )
            decision_key = _fold_manager_key(
                match_key, restart_tick[team], team, windows_used
            )
            halftime_draw = jax.random.bernoulli(
                jax.random.fold_in(decision_key, jnp.uint32(0)),
                self.config.halftime_substitution_probability,
            )
            sampled_window_size = 1 + jax.random.categorical(
                jax.random.fold_in(decision_key, jnp.uint32(1)),
                jnp.log(_WINDOW_SIZE_PROBABILITY),
            ).astype(jnp.int32)
            desired_count = jnp.where(halftime & halftime_draw, 1, sampled_window_size)
            routine_window = (
                team_new_restart
                & ordinary_restart
                & (substitutions_left > 0)
                & ((halftime & halftime_draw) | ((windows_left > 0) & scheduled))
            )
            emergency = (
                goalkeeper_emergency_boundary
                & missing_goalkeeper
                & jnp.any(field)
                & has_bench_goalkeeper
                & (substitutions_left > 0)
            )
            request_limit = jnp.minimum(
                substitutions_left,
                jnp.where(emergency, 1, desired_count),
            )

            used_outgoing = jnp.zeros_like(field)
            used_bench = jnp.zeros_like(bench.valid[team])
            for cell in range(self.max_simultaneous):
                safe_role = jnp.clip(observations.formation_role[team], 0, 6)
                role_frequency = _ROLE_SUBSTITUTION_PROPENSITY[safe_role]
                outgoing_urgency = (
                    self.config.fatigue_weight * (1.0 - stamina_long[team])
                    + self.config.booking_weight
                    * (observations.on_field.yellow_cards[team] > 0)
                    + self.config.role_frequency_weight * role_frequency
                )
                outgoing_candidate = field & (~used_outgoing)
                # Retain fatigue/booking/role ranking, while avoiding a
                # deterministic low-slot tie and argmax.
                # Candidate normalization keeps heterogeneous score units out
                # of the explicit identity-keyed choice temperature.
                outgoing_score = _normalized_candidate_score(
                    outgoing_urgency,
                    outgoing_candidate,
                )
                outgoing_key = jax.random.fold_in(
                    jax.random.fold_in(
                        decision_key,
                        jnp.uint32(_SUBSTITUTION_OUTGOING_STREAM),
                    ),
                    jnp.uint32(cell),
                )
                outgoing_utility = (
                    outgoing_score / self.config.substitution_choice_temperature
                    + _identity_gumbel(outgoing_key, player_id[team])
                )
                routine_outgoing = jnp.argmax(
                    jnp.where(outgoing_candidate, outgoing_utility, -jnp.inf)
                ).astype(jnp.int32)
                outgoing = jnp.where(emergency, emergency_outgoing, routine_outgoing)
                safe_outgoing = jnp.clip(outgoing, 0, player_id.shape[1] - 1)

                bench_candidate = (
                    bench.valid[team]
                    & bench.available[team]
                    & (~bench.is_goalkeeper[team])
                    & (~used_bench)
                )
                outgoing_role = safe_role[safe_outgoing]
                # A declared compatible role is a categorical roster fact,
                # not another weighted coefficient. If none is available,
                # retain the existing normalized ability-profile fallback.
                bench_candidate = _prefer_declared_role_candidates(
                    bench_candidate,
                    bench.preferred_role_known[team],
                    bench.preferred_role_mask[team],
                    outgoing_role,
                )
                profile_distance = (
                    (bench.max_speed[team] - max_speed[team, safe_outgoing]) ** 2
                    + (bench.height[team] - height[team, safe_outgoing]) ** 2
                    + (bench.reach_height[team] - reach_height[team, safe_outgoing])
                    ** 2
                    + (bench.ball_control[team] - ball_control[team, safe_outgoing])
                    ** 2
                    + (bench.endurance_factor[team] - endurance[team, safe_outgoing])
                    ** 2
                )
                incoming_score = _normalized_candidate_score(
                    -profile_distance,
                    bench_candidate,
                )
                incoming_key = jax.random.fold_in(
                    jax.random.fold_in(
                        decision_key,
                        jnp.uint32(_SUBSTITUTION_INCOMING_STREAM),
                    ),
                    jnp.uint32(cell),
                )
                incoming_utility = (
                    incoming_score / self.config.substitution_choice_temperature
                    + _identity_gumbel(incoming_key, bench.player_id[team])
                )
                routine_incoming = jnp.argmax(
                    jnp.where(bench_candidate, incoming_utility, -jnp.inf)
                ).astype(jnp.int32)
                incoming = jnp.where(
                    emergency, goalkeeper_bench_index, routine_incoming
                ).astype(jnp.int32)
                has_pair = jnp.any(outgoing_candidate) & jnp.any(bench_candidate)
                has_pair = jnp.where(
                    emergency, jnp.any(field) & has_bench_goalkeeper, has_pair
                )
                request = (
                    (emergency | routine_window)
                    & (jnp.int32(cell) < request_limit)
                    & has_pair
                )
                substitutions = substitutions._replace(
                    requested=substitutions.requested.at[team, cell].set(request),
                    outgoing_index=substitutions.outgoing_index.at[team, cell].set(
                        jnp.where(request, outgoing, jnp.int32(NO_PLAYER))
                    ),
                    incoming_bench_index=(
                        substitutions.incoming_bench_index.at[team, cell].set(
                            jnp.where(request, incoming, jnp.int32(NO_PLAYER))
                        )
                    ),
                )
                used_outgoing = used_outgoing.at[safe_outgoing].set(
                    used_outgoing[safe_outgoing] | request
                )
                safe_incoming = jnp.clip(incoming, 0, bench.valid.shape[1] - 1)
                used_bench = used_bench.at[safe_incoming].set(
                    used_bench[safe_incoming] | request
                )

        formations = ManagerFormationCommand.empty()
        control_tick = jnp.rint(
            observations.control_tick * jnp.float32(self.context.counter_scale_ticks)
        ).astype(jnp.int32)
        formation_change_tick = _effective_formation_change_tick(
            state.formation_change_tick,
            observations.tactical_epoch,
            observations.formation_changed_control_tick,
            self.context.counter_scale_ticks,
        )
        active_count = jnp.sum(observations.on_field.active, axis=1).astype(jnp.float32)
        for team in range(2):
            other = 1 - team
            margin = (
                observations.score[team, team] - observations.score[team, other]
            ).astype(jnp.float32)
            progress = observations.clock.match_regulation_progress[team]
            urgency = jnp.clip(progress, 0.0, 1.0)
            chase = jnp.maximum(-margin, 0.0) * urgency
            protect = jnp.maximum(margin, 0.0) * urgency + jnp.maximum(
                active_count[other] - active_count[team], 0.0
            )
            mine = observations.on_field.active[team]
            fitness = jnp.sum(
                jnp.where(mine, observations.on_field.stamina_long[team], 0.0)
            ) / jnp.maximum(jnp.sum(mine), 1).astype(jnp.float32)
            valid_layout = observations.formation_candidate_valid[team]
            prior = observations.formation_candidate_probability[team]
            # The public summaries are normalized in physical pitch units, so
            # meaningful layout-to-layout differences are only a few hundredths.
            # Scoring those raw values against the dimensionless incumbent bonus
            # made realistic one- or two-goal contexts effectively unable to
            # change shape. Preserve feature order but put the eligible catalog
            # range on the same dimensionless scale as substitution ranking.
            # No rollout-dependent statistic or new coefficient is introduced.
            attack_depth = _normalized_candidate_score(
                observations.formation_candidate_attack_depth[team], valid_layout
            )
            defender_fraction = _normalized_candidate_score(
                observations.formation_candidate_defender_fraction[team], valid_layout
            )
            width = _normalized_candidate_score(
                observations.formation_candidate_width[team], valid_layout
            )
            base_score = (
                self.config.formation_attack_gain * chase * attack_depth
                + self.config.formation_protect_gain * protect * defender_fraction
                + self.config.formation_width_gain
                * (fitness - jnp.float32(0.5))
                * width
                + self.config.formation_prior_weight * jnp.log(jnp.maximum(prior, 1e-6))
            )
            layout_keys = jax.vmap(
                lambda signature, boundary_tick=restart_tick[team], selected_team=team: (
                    _formation_random_key(
                        match_key,
                        boundary_tick,
                        selected_team,
                        signature,
                    )
                )
            )(observations.formation_candidate_signature[team])
            noise = jax.vmap(jax.random.gumbel)(layout_keys)
            score = base_score + self.config.formation_noise_scale * noise
            current = observations.formation_index[team]
            current_valid = (current >= 0) & (current < score.shape[0])
            safe_current = jnp.clip(current, 0, score.shape[0] - 1)
            score = score.at[safe_current].add(
                jnp.where(
                    current_valid,
                    jnp.float32(self.config.formation_incumbent_bonus),
                    jnp.float32(0.0),
                )
            )
            score = jnp.where(valid_layout, score, -jnp.inf)
            selected = jnp.argmax(score).astype(jnp.int32)
            held = control_tick[team] - formation_change_tick[team] >= jnp.int32(
                self.formation_hold_ticks
            )
            request = (
                new_restart[team]
                & (observations.restart_kind[team] != RK_GK_HOLD)
                & held
                & jnp.any(valid_layout)
                & (selected != current)
            )
            formations = formations._replace(
                requested=formations.requested.at[team].set(request),
                layout_index=formations.layout_index.at[team].set(
                    jnp.where(request, selected, jnp.int32(NO_PLAYER))
                ),
            )

        # Project accepted-looking proposals onto the observation arrays before
        # scoring the visible restart. The environment still validates the
        # transaction and will reject the taker too if a substitution fails.
        projected_id = player_id
        projected_goalkeeper = is_goalkeeper
        projected_reach = reach_height
        projected_stamina = stamina_long
        for team in range(2):
            for cell in range(self.max_simultaneous):
                request = substitutions.requested[team, cell]
                outgoing = jnp.clip(
                    substitutions.outgoing_index[team, cell], 0, player_id.shape[1] - 1
                )
                incoming = jnp.clip(
                    substitutions.incoming_bench_index[team, cell],
                    0,
                    bench.valid.shape[1] - 1,
                )
                projected_id = projected_id.at[team, outgoing].set(
                    jnp.where(
                        request,
                        bench.player_id[team, incoming],
                        projected_id[team, outgoing],
                    )
                )
                projected_goalkeeper = projected_goalkeeper.at[team, outgoing].set(
                    jnp.where(
                        request,
                        bench.is_goalkeeper[team, incoming],
                        projected_goalkeeper[team, outgoing],
                    )
                )
                projected_reach = projected_reach.at[team, outgoing].set(
                    jnp.where(
                        request,
                        bench.reach_height[team, incoming],
                        projected_reach[team, outgoing],
                    )
                )
                projected_stamina = projected_stamina.at[team, outgoing].set(
                    jnp.where(
                        request, jnp.float32(1.0), projected_stamina[team, outgoing]
                    )
                )

        xy = jnp.asarray(
            [self.context.position_scale_x_m, self.context.position_scale_y_m],
            dtype=jnp.float32,
        )
        position_si = observations.on_field.position * xy
        restart_position_si = observations.restart_position * xy

        def restore(value, lower: float, upper: float):
            return jnp.float32(lower) + value * jnp.float32(upper - lower)

        reach_si = restore(
            projected_reach,
            self.context.min_reach_height_m,
            self.context.max_reach_height_m,
        )

        takers = ManagerSetPieceTakerCommand.empty()
        for team in range(2):
            kind = observations.restart_kind[team]
            owns_restart = (
                new_restart[team]
                & (observations.restart_team[team] == team)
                & (kind > RK_NONE)
                & (kind < RESTART_COUNT)
                & (kind != RK_GK_HOLD)
            )
            ranking = score_restart_takers(
                kind=kind,
                position=restart_position_si[team],
                attack_direction=observations.attack_direction[team],
                role=observations.formation_role[team],
                eligible=active[team],
                is_goalkeeper=projected_goalkeeper[team],
                player_position=position_si[team],
                stamina_long=projected_stamina[team],
                reach_height=reach_si[team],
            )
            event_key = restart_taker_random_key(
                match_key, restart_tick[team], jnp.int32(team), kind
            )
            preference_key = restart_taker_random_key(
                match_key, jnp.int32(0), jnp.int32(team), kind
            )
            selected = sample_restart_taker(
                ranking,
                projected_id[team],
                event_key,
                temperature=self.config.taker_temperature,
                preference_key=preference_key,
                persistence=_TAKER_PERSISTENCE[jnp.clip(kind, 0, RESTART_COUNT - 1)],
            )
            safe_kind = jnp.clip(kind, 0, RESTART_COUNT - 1)
            request = owns_restart & (selected != NO_PLAYER)
            takers = takers._replace(
                requested=takers.requested.at[team, safe_kind].set(request),
                player_slot=takers.player_slot.at[team, safe_kind].set(
                    jnp.where(request, selected, jnp.int32(NO_PLAYER))
                ),
            )

        next_state = RuleManagerState(
            processed_restart_tick=jnp.where(
                new_restart, restart_tick, state.processed_restart_tick
            ),
            processed_restart_kind=jnp.where(
                new_restart,
                observations.restart_kind,
                state.processed_restart_kind,
            ),
            formation_change_tick=jnp.where(
                formations.requested,
                control_tick,
                formation_change_tick,
            ),
        )
        return ManagerPolicyStep(
            command=ManagerCommand.empty(self.max_simultaneous)._replace(
                substitutions=substitutions,
                formations=formations,
                acting_goalkeepers=acting,
                set_piece_takers=takers,
            ),
            state=next_state,
        )

    def __call__(
        self,
        observations: NormalizedManagerObservation,
        match_key: jax.Array,
    ) -> ManagerCommand:
        """Stateless convenience call; :meth:`step` is preferred in rollouts."""

        return self.step(observations, match_key, self.initialize()).command


def make_rule_based_manager(
    env,
    config: RuleManagerConfig | None = None,
    *,
    max_simultaneous: int = 3,
) -> RuleBasedManager:
    """Build a manager without adding it to the compiled physics step."""

    config = RuleManagerConfig() if config is None else config
    if not isinstance(config, RuleManagerConfig):
        raise TypeError("config must be RuleManagerConfig or None")
    for name, value in (("max_simultaneous", max_simultaneous),):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return RuleBasedManager(
        config=config,
        max_simultaneous=max_simultaneous,
        context=env.normalization_context(),
        formation_hold_ticks=max(
            1, round(config.formation_hold_seconds / env.timebase.control_dt)
        ),
    )


__all__ = [
    "ManagerPolicyStep",
    "RuleBasedManager",
    "RuleManagerConfig",
    "RuleManagerState",
    "initialize_rule_manager_state",
    "make_rule_based_manager",
]
