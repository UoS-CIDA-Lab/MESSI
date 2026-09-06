"""Start-only categorical formation-policy contract.

The opening policy uses a categorical layout index and an observation-only
boundary: a manager proposes one registered layout and the environment remains
authoritative. In-match scoreline formation changes are a separate concern.
Opening selection is a one-shot,
low-frequency decision, so candidate tensors and categorical sampling never
enter the player rollout graph.

The registered probabilities are caller-supplied policy priors, not measured
football constants and not an action mask.  A learned policy may select every
layout in the catalog, including one whose rule prior is zero.  Only the
shipped stochastic rule policy excludes zero-prior layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from footballworld.core.constants import (
    NO_TEAM,
    RK_KICKOFF,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.randomness import RandomEvent
from footballworld.core.state import State
from footballworld.environment.management import (
    ManagerFormationCommand,
    ManagerState,
    SquadSetup,
)
from footballworld.environment.normalization import NormalizationContext

OPENING_FORMATION_OBSERVATION_SCHEMA_VERSION = 1
RULE_OPENING_FORMATION_VERSION = 1

# ASCII "FORM".  The immutable match key is folded in this order:
# match -> decision stream -> team.  Evaluation order, batching, and rollout
# chunk boundaries therefore cannot couple the two teams' draws.
_OPENING_FORMATION_RANDOM_STREAM = int(RandomEvent.OPENING_FORMATION)


class NormalizedOpeningFormationObservation(NamedTuple):
    """Private, normalized candidate catalog for both team managers.

    The leading dimension is always the two manager rows.  ``player_mask``
    identifies the roster slots owned by that row; opposing slots in all
    player and candidate leaves are zeroed (roles use ``-1``).  Every one of
    the ``L`` registered layouts is a valid categorical action.  The
    ``candidate_probability`` leaf is only the shipped rule policy's prior.

    Shapes are ``[2]`` for manager leaves, ``[2, N]`` for player leaves,
    ``[2, L]`` for candidate probabilities, ``[2, L, N, 2]`` for candidate
    anchors, and ``[2, L, N]`` for candidate roles.
    """

    valid: jax.Array
    team: jax.Array
    ours_kickoff: jax.Array
    player_mask: jax.Array
    active: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    candidate_probability: jax.Array
    candidate_anchor: jax.Array
    candidate_role: jax.Array


@runtime_checkable
class OpeningFormationPolicy(Protocol):
    """Common deployment contract for rule-based and learned policies."""

    def __call__(
        self,
        observations: NormalizedOpeningFormationObservation,
        match_key: jax.Array,
    ) -> ManagerFormationCommand:
        """Select one registered categorical layout for each valid team."""


def _unit_interval(value: jax.Array, lower: float, upper: float) -> jax.Array:
    """Normalize an ability without clipping the represented value."""

    return (value - jnp.float32(lower)) / jnp.float32(upper - lower)


def _validate_catalog_shapes(
    state: State,
    squad: SquadSetup,
    management: ManagerState,
) -> tuple[int, int]:
    """Reject malformed static catalog trees before they enter a policy graph."""

    player_count = state.players.position.shape[0]
    if state.players.team_id.shape != (player_count,):
        raise ValueError("state team_id must have shape [player_count]")
    if squad.formation_layouts.ndim != 3:
        raise ValueError("formation layouts must have shape [L, player_count, 2]")
    layout_count = squad.formation_layouts.shape[0]
    if layout_count < 1 or squad.formation_layouts.shape[1:] != (player_count, 2):
        raise ValueError("formation layouts must have shape [L, player_count, 2]")
    if squad.formation_roles.shape != (layout_count, player_count):
        raise ValueError("formation roles must have shape [L, player_count]")
    if squad.formation_probabilities.shape != (2, layout_count):
        raise ValueError("formation probabilities must have shape [2, L]")
    if management.opening_formation_committed.shape != (2,):
        raise ValueError("opening formation commitment must have shape [2]")
    return player_count, layout_count


def observe_opening_formations(
    state: State,
    squad: SquadSetup,
    management: ManagerState,
    context: NormalizationContext,
) -> NormalizedOpeningFormationObservation:
    """Build the private candidate catalog at the exact opening boundary.

    A normal mid-match state, a second-half kickoff, a live ball, a malformed
    kickoff team, or an already committed manager row is invalid.  Invalid
    rows are completely masked, making accidental later calls fail closed.
    Candidate anchors are divided by the pitch half extents carried in the
    immutable normalization context and are intentionally not clipped.
    """

    player_count, layout_count = _validate_catalog_shapes(state, squad, management)
    del player_count, layout_count

    restart_team_valid = (state.restart.team == TEAM_0) | (state.restart.team == TEAM_1)
    at_opening = (
        (state.control_tick == 0)
        & (~state.ball.live)
        & (state.restart.kind == RK_KICKOFF)
        & (state.restart.opened_control_tick == 0)
        & (state.first_half_wall_end_tick < 0)
        & restart_team_valid
    )
    valid = at_opening & (~management.opening_formation_committed)
    teams = jnp.arange(2, dtype=jnp.int32)
    player_mask = state.players.team_id[None, :] == teams[:, None]
    visible_player = valid[:, None] & player_mask
    visible_candidate_player = visible_player[:, None, :]

    xy = jnp.asarray(
        [context.position_scale_x_m, context.position_scale_y_m],
        dtype=squad.formation_layouts.dtype,
    )
    normalized_anchor = squad.formation_layouts / xy
    candidate_anchor = jnp.where(
        visible_candidate_player[..., None],
        normalized_anchor[None, ...],
        jnp.zeros((), dtype=normalized_anchor.dtype),
    )
    candidate_role = jnp.where(
        visible_candidate_player,
        squad.formation_roles[None, ...],
        jnp.int32(-1),
    )

    def private_player(value: jax.Array) -> jax.Array:
        return jnp.where(visible_player, value[None, :], jnp.zeros_like(value)[None, :])

    return NormalizedOpeningFormationObservation(
        valid=valid,
        team=jnp.where(valid, teams, jnp.int32(NO_TEAM)),
        ours_kickoff=valid & (teams == state.restart.team),
        player_mask=visible_player,
        active=visible_player & state.players.active[None, :],
        is_goalkeeper=visible_player & state.players.is_goalkeeper[None, :],
        max_speed=private_player(
            _unit_interval(
                state.players.max_speed,
                context.min_player_speed_mps,
                context.max_player_speed_mps,
            )
        ),
        height=private_player(
            _unit_interval(
                state.players.height,
                context.min_height_m,
                context.max_height_m,
            )
        ),
        reach_height=private_player(
            _unit_interval(
                state.players.reach_height,
                context.min_reach_height_m,
                context.max_reach_height_m,
            )
        ),
        ball_control=private_player(
            _unit_interval(
                state.players.ball_control,
                context.min_ball_control,
                context.max_ball_control,
            )
        ),
        endurance_factor=private_player(
            _unit_interval(
                state.players.endurance_factor,
                context.min_endurance_factor,
                context.max_endurance_factor,
            )
        ),
        candidate_probability=jnp.where(
            valid[:, None],
            squad.formation_probabilities,
            jnp.zeros((), dtype=squad.formation_probabilities.dtype),
        ),
        candidate_anchor=candidate_anchor,
        candidate_role=candidate_role,
    )


def opening_formation_command(
    observations: NormalizedOpeningFormationObservation,
    layout_index: jax.Array,
) -> ManagerFormationCommand:
    """Turn learned categorical outputs into the public formation action.

    This helper treats every registered catalog column as selectable and masks
    only invalid manager rows or out-of-range categorical values.  In
    particular, it does not consult ``candidate_probability``.
    """

    _require_opening_observations(observations)
    layout_index = jnp.asarray(layout_index)
    if layout_index.shape != (2,):
        raise ValueError("layout_index must have shape [2]")
    if not jnp.issubdtype(layout_index.dtype, jnp.integer) or jnp.issubdtype(
        layout_index.dtype, jnp.bool_
    ):
        raise TypeError("layout_index must use a non-boolean integer dtype")
    layout_index = layout_index.astype(jnp.int32)
    layout_count = observations.candidate_probability.shape[1]
    in_range = (layout_index >= 0) & (layout_index < layout_count)
    requested = observations.valid & in_range
    return ManagerFormationCommand(
        requested=requested,
        layout_index=jnp.where(requested, layout_index, jnp.int32(-1)),
    )


def _require_opening_observations(
    observations: NormalizedOpeningFormationObservation,
) -> None:
    if type(observations) is not NormalizedOpeningFormationObservation:
        raise TypeError(
            "opening formation policies require NormalizedOpeningFormationObservation"
        )
    if observations.valid.shape != (2,):
        raise ValueError("opening formation observations must have two team rows")
    if observations.candidate_probability.ndim != 2:
        raise ValueError("candidate_probability must have shape [2, L]")
    if observations.candidate_probability.shape[0] != 2:
        raise ValueError("candidate_probability must have shape [2, L]")
    if observations.candidate_probability.shape[1] < 1:
        raise ValueError("the formation catalog must contain at least one layout")


def opening_formation_random_key(match_key: jax.Array, team: int) -> jax.Array:
    """Return one team's stateless opening-layout random stream."""

    if not isinstance(team, int) or isinstance(team, bool) or team not in (0, 1):
        raise ValueError("team must be integer 0 or 1")
    key = jax.random.fold_in(
        match_key, jnp.asarray(_OPENING_FORMATION_RANDOM_STREAM, dtype=jnp.uint32)
    )
    return jax.random.fold_in(key, jnp.uint32(team))


@dataclass(frozen=True, slots=True)
class RuleBasedOpeningFormationPolicy:
    """Sample the registered per-team priors once at the opening boundary."""

    def __call__(
        self,
        observations: NormalizedOpeningFormationObservation,
        match_key: jax.Array,
    ) -> ManagerFormationCommand:
        _require_opening_observations(observations)
        selected = []
        has_mass = []
        for team in (TEAM_0, TEAM_1):
            probability = observations.candidate_probability[team]
            positive = (probability > 0.0) & jnp.isfinite(probability)
            mass = jnp.sum(jnp.where(positive, probability, 0.0))
            usable = jnp.isfinite(mass) & (mass > 0.0)
            # Avoid log(0), while retaining -inf for every excluded category.
            safe_probability = jnp.where(positive, probability, 1.0)
            logits = jnp.where(positive, jnp.log(safe_probability), -jnp.inf)
            logits = jnp.where(
                usable,
                logits,
                jnp.where(
                    jnp.arange(probability.shape[0]) == 0,
                    jnp.float32(0.0),
                    -jnp.inf,
                ),
            )
            selected.append(
                jax.random.categorical(
                    opening_formation_random_key(match_key, team), logits
                ).astype(jnp.int32)
            )
            has_mass.append(usable)

        layout_index = jnp.stack(selected)
        command = opening_formation_command(observations, layout_index)
        requested = command.requested & jnp.stack(has_mass)
        return command._replace(
            requested=requested,
            layout_index=jnp.where(requested, command.layout_index, jnp.int32(-1)),
        )


def make_rule_based_opening_formation_policy() -> RuleBasedOpeningFormationPolicy:
    """Build the opt-in stochastic reference policy."""

    return RuleBasedOpeningFormationPolicy()


__all__ = [
    "OPENING_FORMATION_OBSERVATION_SCHEMA_VERSION",
    "RULE_OPENING_FORMATION_VERSION",
    "NormalizedOpeningFormationObservation",
    "OpeningFormationPolicy",
    "RuleBasedOpeningFormationPolicy",
    "make_rule_based_opening_formation_policy",
    "observe_opening_formations",
    "opening_formation_command",
    "opening_formation_random_key",
]
