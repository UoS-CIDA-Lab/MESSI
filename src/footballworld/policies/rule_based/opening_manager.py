"""Start-only rule and authored-adapter opening manager policies.

The built-in rule policy chooses a registered squad, exact XI, formation, and
unique player-to-slot placement from the public fixed-shape opening view. All
weights below are design priors: no provider event metric identifies squad or
formation selection. Randomness is keyed by immutable player identity or
catalog content, so reordering candidates or layouts reorders the decision
rather than changing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import partial
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.core.constants import NO_PLAYER, TEAM_0, TEAM_1
from footballworld.core.numeric import require_float32_representable
from footballworld.environment.management import ManagerFormationCommand
from footballworld.policies.manager import (
    NO_POLICY_PARAMETERS,
    NoPolicyParameters,
    OpeningManagerDecision,
    OpeningManagerObservation,
    OpeningManagerPolicyStep,
    validate_opening_policy_shapes,
)
from footballworld.policies.opening_formation import RuleBasedOpeningFormationPolicy
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    canonical_tactical_plan,
    tactical_plan_code,
)

_OPENING_PLAYER_STREAM = 0x4C494E45  # ASCII "LINE".
_OPENING_FORMATION_STREAM = 0x464F524D  # ASCII "FORM".

# [speed, height, reach, control, endurance] role weights. These are explicit
# design priors and affect only the start-only reference policy.
_ROLE_ABILITY_WEIGHT = jnp.asarray(
    [
        [0.05, 0.20, 0.35, 0.30, 0.10],  # goalkeeper
        [0.20, 0.25, 0.25, 0.10, 0.20],  # centre-back
        [0.30, 0.05, 0.05, 0.25, 0.35],  # full-back
        [0.15, 0.05, 0.05, 0.45, 0.30],  # centre midfield
        [0.35, 0.00, 0.00, 0.35, 0.30],  # wide midfield
        [0.25, 0.20, 0.15, 0.35, 0.05],  # centre forward
        [0.40, 0.00, 0.00, 0.40, 0.20],  # wide forward
    ],
    dtype=jnp.float32,
)
_REGISTRATION_ABILITY_WEIGHT = jnp.asarray(
    [0.25, 0.10, 0.10, 0.30, 0.25], dtype=jnp.float32
)
# [attack depth, width, defender share, central-midfield share, wide-role share].
# These values only rank structural compatibility after a tactical plan has
# been fixed; they are transparent design priors rather than empirical rates.
_TACTICAL_FORMATION_WEIGHT = jnp.asarray(
    [
        [-0.10, 0.10, 0.00, 0.50, 0.25],  # salida lavolpiana
        [0.20, 0.40, -0.10, 0.35, 0.30],  # juego de posicion
        [0.40, -0.10, -0.10, 0.35, 0.10],  # gegenpress
        [-0.50, -0.30, 0.60, 0.10, -0.10],  # catenaccio
        [-0.10, 0.35, 0.20, 0.10, 0.40],  # zona mista
    ],
    dtype=jnp.float32,
)


class AuthoredOpeningSelection(NamedTuple):
    """Dynamic authored roster/lineup proposal for the compatibility adapter."""

    registered: jax.Array
    starter: jax.Array
    placement_slot: jax.Array


class RuleOpeningManagerState(NamedTuple):
    """One-shot memory, one bit for each team manager row."""

    decided: jax.Array


@dataclass(frozen=True, slots=True)
class RuleOpeningManagerConfig:
    """Start-only reference-policy design priors, not measured constants."""

    team_tactical_plans: tuple[TacticalPlan, TacticalPlan] = (
        TacticalPlan.JUEGO_DE_POSICION,
        TacticalPlan.JUEGO_DE_POSICION,
    )
    position_fit_weight: float = 1.25
    # Softmax logit gain for the deterministic best-XI fit of each formation.
    # This is an explicit design prior, not a measured football constant.
    formation_fit_weight: float = 2.0
    lineup_noise_scale: float = 0.025
    registration_noise_scale: float = 0.02

    def __post_init__(self) -> None:
        plans = self.team_tactical_plans
        if not isinstance(plans, tuple) or len(plans) != 2:
            raise TypeError("team_tactical_plans must be a length-two tuple")
        object.__setattr__(
            self,
            "team_tactical_plans",
            tuple(canonical_tactical_plan(plan) for plan in plans),
        )
        for item in fields(self):
            if item.name == "team_tactical_plans":
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{item.name} must be a real number")
            value = require_float32_representable(item.name, value)
            if value < 0.0:
                raise ValueError(f"{item.name} must be non-negative")
            object.__setattr__(self, item.name, value)


def _validate_state(state: RuleOpeningManagerState) -> None:
    if type(state) is not RuleOpeningManagerState:
        raise TypeError("state must be RuleOpeningManagerState")
    if state.decided.shape != (2,) or state.decided.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("state.decided must be bool with shape [2]")


def _validate_authored_selection(
    selection: AuthoredOpeningSelection,
    candidate_count: int,
) -> None:
    if type(selection) is not AuthoredOpeningSelection:
        raise TypeError("parameters must be AuthoredOpeningSelection")
    shape = (2, candidate_count)
    for name in ("registered", "starter"):
        value = getattr(selection, name)
        if value.shape != shape or value.dtype != jnp.dtype(jnp.bool_):
            raise TypeError(f"parameters.{name} must be bool with shape {shape}")
    if (
        selection.placement_slot.shape != shape
        or selection.placement_slot.dtype != jnp.dtype(jnp.int32)
    ):
        raise TypeError(f"parameters.placement_slot must be int32 with shape {shape}")


def _content_hash(values: jax.Array) -> jax.Array:
    """Return a deterministic uint32 FNV-style hash for one fixed-shape row."""

    flat = values.reshape(-1).astype(jnp.uint32)

    def combine(index, value):
        return (value ^ flat[index]) * jnp.uint32(16_777_619)

    return jax.lax.fori_loop(0, flat.shape[0], combine, jnp.uint32(2_166_136_261))


def _layout_signature(anchor: jax.Array, role: jax.Array) -> jax.Array:
    quantized = jnp.rint(anchor * jnp.float32(100_000.0)).astype(jnp.int32)
    packed = jnp.concatenate((quantized.reshape(-1), role.astype(jnp.int32)))
    return _content_hash(packed)


def _identity_key(
    match_key: jax.Array,
    team: int,
    player_id: jax.Array,
    stream: int,
    context: jax.Array,
) -> jax.Array:
    key = jax.random.fold_in(match_key, jnp.uint32(stream))
    key = jax.random.fold_in(key, jnp.uint32(team))
    key = jax.random.fold_in(key, player_id.astype(jnp.uint32))
    return jax.random.fold_in(key, context.astype(jnp.uint32))


def _formation_tactical_fit(
    anchor: jax.Array,
    role: jax.Array,
    player_mask: jax.Array,
    plan: TacticalPlan,
) -> jax.Array:
    outfield = player_mask[None, :] & (role != 0)
    count = jnp.maximum(jnp.sum(outfield, axis=1), 1).astype(jnp.float32)
    depth = jnp.sum(jnp.where(outfield, anchor[..., 0], 0.0), axis=1) / count
    width = jnp.sum(jnp.where(outfield, jnp.abs(anchor[..., 1]), 0.0), axis=1) / count
    defender = jnp.sum(outfield & ((role == 1) | (role == 2)), axis=1) / count
    central_midfield = jnp.sum(outfield & (role == 3), axis=1) / count
    wide = jnp.sum(outfield & ((role == 2) | (role == 4) | (role == 6)), axis=1) / count
    features = jnp.stack((depth, width, defender, central_midfield, wide), axis=1)
    return features @ _TACTICAL_FORMATION_WEIGHT[tactical_plan_code(plan)]


def _choose_formation(
    probability: jax.Array,
    anchor: jax.Array,
    role: jax.Array,
    lineup_fit: jax.Array,
    lineup_complete: jax.Array,
    tactical_fit: jax.Array,
    match_key: jax.Array,
    team: int,
    fit_weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Sample a layout only after every candidate has a feasible-XI receipt."""

    signatures = jax.vmap(_layout_signature)(anchor, role)
    base_key = jax.random.fold_in(match_key, jnp.uint32(_OPENING_FORMATION_STREAM))
    base_key = jax.random.fold_in(base_key, jnp.uint32(team))
    keys = jax.vmap(lambda value: jax.random.fold_in(base_key, value))(signatures)
    noise = jax.vmap(lambda key: jax.random.gumbel(key, dtype=jnp.float32))(keys)
    usable = (probability > 0.0) & jnp.isfinite(probability) & lineup_complete
    logits = jnp.log(jnp.maximum(probability, 1e-12)) + jnp.float32(fit_weight) * (
        lineup_fit + tactical_fit
    )
    selected = jnp.argmax(jnp.where(usable, logits + noise, -jnp.inf)).astype(jnp.int32)
    return selected, jnp.any(usable)


def _lineup_for_layout(
    anchor: jax.Array,
    role: jax.Array,
    slot_mask: jax.Array,
    valid_candidate: jax.Array,
    player_id: jax.Array,
    is_goalkeeper: jax.Array,
    preferred: jax.Array,
    ability: jax.Array,
    match_key: jax.Array,
    team: int,
    position_fit_weight: float,
    lineup_noise_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return one best-XI proposal and its deterministic mean fit."""

    candidates = player_id.shape[0]
    initial = (
        jnp.zeros((candidates,), dtype=jnp.bool_),
        jnp.zeros((candidates,), dtype=jnp.bool_),
        jnp.full((candidates,), NO_PLAYER, dtype=jnp.int32),
        jnp.float32(0.0),
        jnp.bool_(True),
    )

    def assign_slot(carry, slot_input):
        used, starter, placement, fit_total, complete = carry
        slot, slot_anchor, slot_role, active_slot = slot_input
        selected_role = jnp.clip(slot_role, 0, _ROLE_ABILITY_WEIGHT.shape[0] - 1)
        compatible = valid_candidate & (~used) & (is_goalkeeper == (selected_role == 0))
        distance = jnp.sum((preferred - slot_anchor) ** 2, axis=-1)
        deterministic_score = (
            jnp.sum(ability * _ROLE_ABILITY_WEIGHT[selected_role], axis=-1)
            - jnp.float32(position_fit_weight) * distance
        )
        signature = _layout_signature(slot_anchor[None, :], slot_role[None])
        keys = jax.vmap(
            lambda identity, selected_signature=signature: _identity_key(
                match_key,
                team,
                identity,
                _OPENING_PLAYER_STREAM,
                selected_signature,
            )
        )(player_id)
        noise = jax.vmap(lambda key: jax.random.gumbel(key, dtype=jnp.float32))(keys)
        score = deterministic_score + jnp.float32(lineup_noise_scale) * noise
        chosen = jnp.argmax(jnp.where(compatible, score, -jnp.inf)).astype(jnp.int32)
        has_candidate = jnp.any(compatible)
        applies = active_slot & has_candidate
        starter = starter.at[chosen].set(starter[chosen] | applies)
        placement = placement.at[chosen].set(
            jnp.where(applies, jnp.int32(slot), placement[chosen])
        )
        used = used.at[chosen].set(used[chosen] | applies)
        fit_total += jnp.where(applies, deterministic_score[chosen], 0.0)
        complete &= (~active_slot) | has_candidate
        return (used, starter, placement, fit_total, complete), None

    final, _ = jax.lax.scan(
        assign_slot,
        initial,
        (jnp.arange(anchor.shape[0], dtype=jnp.int32), anchor, role, slot_mask),
    )
    _, starter, placement, fit_total, complete = final
    slot_count = jnp.maximum(jnp.sum(slot_mask), 1).astype(jnp.float32)
    return starter, placement, fit_total / slot_count, complete


def _rank_mask(
    score: jax.Array,
    valid: jax.Array,
    player_id: jax.Array,
    count: jax.Array,
) -> jax.Array:
    """Select an exact top-count set with identity-only deterministic ties."""

    better = score[None, :] > score[:, None]
    equal_but_lower_identity = (score[None, :] == score[:, None]) & (
        player_id[None, :] < player_id[:, None]
    )
    rank = jnp.sum(valid[None, :] & (better | equal_but_lower_identity), axis=1)
    target = jnp.minimum(count, jnp.sum(valid)).astype(jnp.int32)
    return valid & (rank < target)


@dataclass(frozen=True, slots=True)
class RuleBasedOpeningManagerPolicy:
    """Select the complete opening team once from observable candidate facts."""

    config: RuleOpeningManagerConfig = field(default_factory=RuleOpeningManagerConfig)

    def initialize(
        self,
        observations: OpeningManagerObservation,
        parameters: NoPolicyParameters = NO_POLICY_PARAMETERS,
    ) -> RuleOpeningManagerState:
        validate_opening_policy_shapes(observations)
        if type(parameters) is not NoPolicyParameters:
            raise TypeError(
                "rule opening-manager parameters must be NoPolicyParameters"
            )
        return RuleOpeningManagerState(decided=jnp.zeros(2, dtype=jnp.bool_))

    def step(
        self,
        observations: OpeningManagerObservation,
        match_key: jax.Array,
        state: RuleOpeningManagerState,
        parameters: NoPolicyParameters = NO_POLICY_PARAMETERS,
    ) -> OpeningManagerPolicyStep[RuleOpeningManagerState]:
        candidates, _, _ = validate_opening_policy_shapes(observations)
        _validate_state(state)
        if type(parameters) is not NoPolicyParameters:
            raise TypeError(
                "rule opening-manager parameters must be NoPolicyParameters"
            )

        registered = jnp.zeros((2, candidates), dtype=jnp.bool_)
        starter = jnp.zeros((2, candidates), dtype=jnp.bool_)
        placement = jnp.full((2, candidates), NO_PLAYER, dtype=jnp.int32)
        selected_layouts = []
        eligible_rows = []
        ability = jnp.stack(
            (
                observations.players.max_speed,
                observations.players.height,
                observations.players.reach_height,
                observations.players.ball_control,
                observations.players.endurance_factor,
            ),
            axis=-1,
        )

        for team in (TEAM_0, TEAM_1):
            slot_mask = observations.formation.player_mask[team]
            valid_candidate = observations.players.valid[team]
            preferred = observations.players.preferred_position[team]
            lineup_for_team = partial(
                _lineup_for_layout,
                slot_mask=slot_mask,
                valid_candidate=valid_candidate,
                player_id=observations.players.player_id[team],
                is_goalkeeper=observations.players.is_goalkeeper[team],
                preferred=preferred,
                ability=ability[team],
                match_key=match_key,
                team=team,
                position_fit_weight=self.config.position_fit_weight,
                lineup_noise_scale=self.config.lineup_noise_scale,
            )
            (
                layout_starters,
                layout_placements,
                layout_fits,
                layout_complete,
            ) = jax.vmap(lineup_for_team)(
                observations.formation.candidate_anchor[team],
                observations.formation.candidate_role[team],
            )
            tactical_fit = _formation_tactical_fit(
                observations.formation.candidate_anchor[team],
                observations.formation.candidate_role[team],
                slot_mask,
                self.config.team_tactical_plans[team],
            )
            selected_layout, layout_usable = _choose_formation(
                observations.formation.candidate_probability[team],
                observations.formation.candidate_anchor[team],
                observations.formation.candidate_role[team],
                layout_fits,
                layout_complete,
                tactical_fit,
                match_key,
                team,
                self.config.formation_fit_weight,
            )
            eligible = (
                observations.formation.valid[team]
                & (~state.decided[team])
                & layout_usable
            )
            team_starter = layout_starters[selected_layout] & eligible
            team_placement = jnp.where(
                eligible,
                layout_placements[selected_layout],
                jnp.int32(NO_PLAYER),
            )
            selected_layouts.append(selected_layout)
            eligible_rows.append(eligible)

            registration_keys = jax.vmap(
                lambda identity, selected_team=team: _identity_key(
                    match_key,
                    selected_team,
                    identity,
                    _OPENING_PLAYER_STREAM,
                    jnp.uint32(0x52454749),
                )
            )(observations.players.player_id[team])
            registration_noise = jax.vmap(
                lambda key: jax.random.gumbel(key, dtype=jnp.float32)
            )(registration_keys)
            registration_score = (
                jnp.sum(ability[team] * _REGISTRATION_ABILITY_WEIGHT, axis=-1)
                + self.config.registration_noise_scale * registration_noise
                + jnp.where(team_starter, 100.0, 0.0)
            )
            team_registered = _rank_mask(
                registration_score,
                valid_candidate,
                observations.players.player_id[team],
                observations.max_registered_players[team],
            )
            registered = registered.at[team].set(team_registered & eligible)
            starter = starter.at[team].set(team_starter)
            placement = placement.at[team].set(team_placement)

        layout = jnp.stack(selected_layouts)
        eligible = jnp.stack(eligible_rows)
        formation = ManagerFormationCommand(
            requested=eligible,
            layout_index=jnp.where(eligible, layout, jnp.int32(NO_PLAYER)),
        )
        decision = OpeningManagerDecision(
            registered=registered,
            starter=starter,
            placement_slot=placement,
            formation=formation,
        )
        validate_opening_policy_shapes(observations, decision)
        return OpeningManagerPolicyStep(
            decision=decision,
            state=RuleOpeningManagerState(decided=state.decided | eligible),
        )


@dataclass(frozen=True, slots=True)
class AuthoredOpeningManagerPolicy:
    """Preserve caller-authored selection while sampling only formation."""

    formation_policy: RuleBasedOpeningFormationPolicy = field(
        default_factory=RuleBasedOpeningFormationPolicy
    )

    def initialize(
        self,
        observations: OpeningManagerObservation,
        parameters: AuthoredOpeningSelection,
    ) -> RuleOpeningManagerState:
        candidates, _, _ = validate_opening_policy_shapes(observations)
        _validate_authored_selection(parameters, candidates)
        return RuleOpeningManagerState(decided=jnp.zeros(2, dtype=jnp.bool_))

    def step(
        self,
        observations: OpeningManagerObservation,
        match_key: jax.Array,
        state: RuleOpeningManagerState,
        parameters: AuthoredOpeningSelection,
    ) -> OpeningManagerPolicyStep[RuleOpeningManagerState]:
        candidates, _, _ = validate_opening_policy_shapes(observations)
        _validate_authored_selection(parameters, candidates)
        _validate_state(state)
        eligible = observations.formation.valid & (~state.decided)
        visible = observations.players.valid & eligible[:, None]
        registered = parameters.registered & visible
        starter = parameters.starter & registered
        placement_slot = jnp.where(
            starter, parameters.placement_slot, jnp.int32(NO_PLAYER)
        )
        formation = self.formation_policy(observations.formation, match_key)
        requested = formation.requested & eligible
        formation = formation._replace(
            requested=requested,
            layout_index=jnp.where(
                requested, formation.layout_index, jnp.int32(NO_PLAYER)
            ),
        )
        decision = OpeningManagerDecision(
            registered=registered,
            starter=starter,
            placement_slot=placement_slot,
            formation=formation,
        )
        validate_opening_policy_shapes(observations, decision)
        return OpeningManagerPolicyStep(
            decision=decision,
            state=RuleOpeningManagerState(decided=state.decided | eligible),
        )


def make_rule_based_opening_manager_policy(
    config: RuleOpeningManagerConfig | None = None,
) -> RuleBasedOpeningManagerPolicy:
    """Build the complete seeded start-only reference policy."""

    if config is not None and not isinstance(config, RuleOpeningManagerConfig):
        raise TypeError("config must be RuleOpeningManagerConfig or None")
    return RuleBasedOpeningManagerPolicy(
        config=RuleOpeningManagerConfig() if config is None else config
    )


def make_authored_opening_manager_policy() -> AuthoredOpeningManagerPolicy:
    """Build the explicit authored-selection compatibility adapter."""

    return AuthoredOpeningManagerPolicy()


__all__ = [
    "AuthoredOpeningManagerPolicy",
    "AuthoredOpeningSelection",
    "RuleBasedOpeningManagerPolicy",
    "RuleOpeningManagerConfig",
    "RuleOpeningManagerState",
    "make_authored_opening_manager_policy",
    "make_rule_based_opening_manager_policy",
]
