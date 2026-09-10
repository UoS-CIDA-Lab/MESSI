"""Host-only materialization of an opening-manager roster proposal.

This module is the authoritative boundary between fixed-shape opening-policy
arrays and FootballWorld's ordinary Python match-construction inputs. It does
not accept a rollout or state, so it cannot be used to rewrite a match that has
already started. Candidate pools are transferred and validated once, then
discarded after producing ``Player`` starters, registered bench
``PlayerProfile`` values, and formation layouts for ``reset`` and
``initialize_management``.

No function here is called by ``FootballWorld.step`` or a rollout scan.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from footballworld.config.roster import (
    Player,
    PlayerProfile,
    player_profile_values_valid,
)
from footballworld.core.constants import NO_PLAYER, TEAM_0, TEAM_1
from footballworld.core.randomness import RandomEvent, validate_prng_key
from footballworld.environment.api import FootballWorld, ResetResult
from footballworld.environment.management import (
    ManagementInitialization,
    ManagementRules,
    ManagerFormationCommand,
)
from footballworld.environment.roster_sampling import (
    ProfileValues,
    sample_profile_values,
)
from footballworld.environment.tactics import classify_formation_roles
from footballworld.environment.validation import (
    validate_formation_inside_pitch,
    validate_kickoff_team,
    validate_no_torso_overlap,
    validate_public_rosters,
)
from footballworld.policies.manager import (
    NO_POLICY_PARAMETERS,
    OpeningManagerDecision,
    OpeningManagerObservation,
    OpeningPlayerPool,
    validate_opening_manager_policy,
    validate_opening_policy_shapes,
)
from footballworld.policies.opening_formation import (
    NormalizedOpeningFormationObservation,
)
from footballworld.policies.rule_based.opening_manager import (
    AuthoredOpeningSelection,
    make_rule_based_opening_manager_policy,
)

_OPENING_CANDIDATE_SAMPLING_STREAM = int(RandomEvent.OPENING_CANDIDATE)


@dataclass(frozen=True, slots=True)
class OpeningMatchInputs:
    """Validated host inputs for one new match.

    ``team_0`` and ``team_1`` go to :meth:`FootballWorld.reset`.
    ``team_0_bench``, ``team_1_bench``, and ``formation_layouts`` go to
    :meth:`FootballWorld.initialize_management` after reset. The selected
    per-team opening layouts are already encoded in each starter's attacking-
    frame ``initial_position``. ``formation_layouts`` retains the original
    registered catalog as alternatives; management initialization prepends
    the selected physical opening layout as its layout zero.
    """

    team_0: tuple[Player, ...]
    team_1: tuple[Player, ...]
    team_0_bench: tuple[PlayerProfile, ...]
    team_1_bench: tuple[PlayerProfile, ...]
    formation_layouts: np.ndarray
    formation_probabilities: np.ndarray
    selected_formation_index: tuple[int, int]
    selected_formation_layout: np.ndarray


@dataclass(frozen=True, slots=True)
class OpeningPolicyInputs:
    """One pre-reset policy view and its authored rule-policy parameters.

    Candidate sequences and layout catalogs are host construction inputs only.
    The returned fixed-shape trees may be passed to a rule or learned opening
    manager and discarded immediately after :func:`create_opening_match`.
    """

    observation: OpeningManagerObservation
    authored_selection: AuthoredOpeningSelection


@dataclass(frozen=True, slots=True)
class CreatedOpeningMatch:
    """A fully reset match with opening management already committed."""

    inputs: OpeningMatchInputs
    reset: ResetResult
    management: ManagementInitialization


@dataclass(frozen=True, slots=True)
class CreatedPolicyOpeningMatch:
    """One-shot policy receipt and the match created from its decision."""

    match: CreatedOpeningMatch
    decision: OpeningManagerDecision
    policy_state: object


def _as_profile_and_position(
    candidate: Player | PlayerProfile,
) -> tuple[PlayerProfile, np.ndarray]:
    if type(candidate) is Player:
        profile = candidate.profile
        position = candidate.initial_position
    elif type(candidate) is PlayerProfile:
        profile = candidate
        position = (0.0, 0.0)
    else:
        raise TypeError("candidate entries must be exactly Player or PlayerProfile")
    if type(profile) is not PlayerProfile:
        raise TypeError("candidate profiles must be exactly PlayerProfile")
    if not isinstance(profile.player_id, numbers.Integral) or isinstance(
        profile.player_id, (bool, np.bool_)
    ):
        raise TypeError("candidate player_id must be a non-boolean integer")
    if type(profile.is_goalkeeper) is not bool:
        raise TypeError("candidate is_goalkeeper must be bool")
    values = (
        profile.max_speed_mps,
        profile.height_m,
        profile.max_reach_height_m,
        profile.ball_control,
        profile.endurance_factor,
    )
    if not all(
        isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("candidate physical values must be finite real numbers")
    try:
        preferred = np.asarray(position, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "candidate preferred position must contain two reals"
        ) from error
    if preferred.shape != (2,) or not np.all(np.isfinite(preferred)):
        raise ValueError("candidate preferred position must be finite with shape [2]")
    represented = tuple(np.float32(value) for value in values)
    return (
        PlayerProfile(
            player_id=int(profile.player_id),
            max_speed_mps=float(represented[0]),
            height_m=float(represented[1]),
            max_reach_height_m=float(represented[2]),
            ball_control=float(represented[3]),
            endurance_factor=float(represented[4]),
            is_goalkeeper=profile.is_goalkeeper,
        ),
        preferred,
    )


def _validate_candidate_profile(env: FootballWorld, profile: PlayerProfile) -> None:
    values = tuple(
        np.float32(value)
        for value in (
            profile.max_speed_mps,
            profile.height_m,
            profile.max_reach_height_m,
            profile.ball_control,
            profile.endurance_factor,
        )
    )
    if not all(np.isfinite(value) for value in values) or not bool(
        player_profile_values_valid(
            profile.player_id,
            *values,
            head_radius=np.float32(env.body.head_radius_m),
        )
    ):
        raise ValueError(
            f"candidate profile is outside its physical domain: "
            f"player_id={profile.player_id}"
        )
    sampling = env.roster_sampling
    reach_margin = float(values[2] - values[1])
    if not (
        sampling.min_max_speed_mps <= values[0] <= sampling.max_max_speed_mps
        and sampling.min_height_m <= values[1] <= sampling.max_height_m
        and sampling.min_reach_margin_m <= reach_margin <= sampling.max_reach_margin_m
        and sampling.min_ball_control <= values[3] <= sampling.max_ball_control
        and sampling.min_endurance_factor <= values[4] <= sampling.max_endurance_factor
    ):
        raise ValueError(
            "candidate profile exceeds the configured normalization domain: "
            f"player_id={profile.player_id}"
        )


def _candidate_rows(
    env: FootballWorld,
    team_0_candidates: Sequence[Player | PlayerProfile],
    team_1_candidates: Sequence[Player | PlayerProfile],
    sampling_key: jax.Array | None,
) -> tuple[
    tuple[tuple[PlayerProfile, ...], tuple[PlayerProfile, ...]],
    tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
]:
    source_rows = (tuple(team_0_candidates), tuple(team_1_candidates))
    if not source_rows[0] or not source_rows[1]:
        raise ValueError("both opening candidate pools must be non-empty")
    profile_rows: list[tuple[PlayerProfile, ...]] = []
    position_rows: list[tuple[np.ndarray, ...]] = []
    seen: set[int] = set()
    for row in source_rows:
        profiles: list[PlayerProfile] = []
        positions: list[np.ndarray] = []
        for candidate in row:
            profile, position = _as_profile_and_position(candidate)
            _validate_candidate_profile(env, profile)
            if profile.player_id in seen:
                raise ValueError(f"duplicate candidate player_id: {profile.player_id}")
            seen.add(profile.player_id)
            profiles.append(profile)
            positions.append(position)
        profile_rows.append(tuple(profiles))
        position_rows.append(tuple(positions))

    if sampling_key is not None and env.roster_sampling.enabled:
        stream_key = jax.random.fold_in(
            sampling_key,
            jnp.uint32(_OPENING_CANDIDATE_SAMPLING_STREAM),
        )
        flat = profile_rows[0] + profile_rows[1]
        identity = jnp.asarray(
            [profile.player_id for profile in flat], dtype=jnp.uint32
        )
        keys = jax.vmap(lambda value: jax.random.fold_in(stream_key, value))(identity)
        columns = tuple(
            jnp.asarray(value, dtype=jnp.float32)
            for value in zip(
                *(
                    (
                        profile.max_speed_mps,
                        profile.height_m,
                        profile.max_reach_height_m,
                        profile.ball_control,
                        profile.endurance_factor,
                    )
                    for profile in flat
                ),
                strict=True,
            )
        )

        def sample_one(
            max_speed,
            height,
            reach_height,
            ball_control,
            endurance_factor,
            candidate_key,
        ):
            return sample_profile_values(
                ProfileValues(
                    max_speed=max_speed,
                    height=height,
                    reach_height=reach_height,
                    ball_control=ball_control,
                    endurance_factor=endurance_factor,
                ),
                candidate_key,
                env.roster_sampling,
                minimum_height_m=float(
                    np.nextafter(
                        np.float32(2.0 * env.body.head_radius_m),
                        np.float32(np.inf),
                    )
                ),
            )

        sampled_values = jax.device_get(jax.vmap(sample_one)(*columns, keys))
        sampled_flat = tuple(
            PlayerProfile(
                player_id=profile.player_id,
                max_speed_mps=float(sampled_values.max_speed[index]),
                height_m=float(sampled_values.height[index]),
                max_reach_height_m=float(sampled_values.reach_height[index]),
                ball_control=float(sampled_values.ball_control[index]),
                endurance_factor=float(sampled_values.endurance_factor[index]),
                is_goalkeeper=profile.is_goalkeeper,
            )
            for index, profile in enumerate(flat)
        )
        cut = len(profile_rows[0])
        profile_rows = [sampled_flat[:cut], sampled_flat[cut:]]
        for profile in sampled_flat:
            _validate_candidate_profile(env, profile)
    return (
        (profile_rows[0], profile_rows[1]),
        (position_rows[0], position_rows[1]),
    )


def _layout_catalog(
    value: np.ndarray | jax.Array,
    *,
    name: str,
) -> np.ndarray:
    catalog = np.asarray(value, dtype=np.float32)
    if catalog.ndim == 2:
        catalog = catalog[None, ...]
    if (
        catalog.ndim != 3
        or catalog.shape[0] < 1
        or catalog.shape[1] < 1
        or catalog.shape[2] != 2
    ):
        raise ValueError(f"{name} must have shape [L, starters, 2]")
    if not np.all(np.isfinite(catalog)):
        raise ValueError(f"{name} must be finite")
    return catalog


def _selection_indices(
    value: Sequence[int] | None,
    *,
    count: int,
    default_count: int | None,
    name: str,
) -> tuple[int, ...]:
    if value is None:
        if default_count is None:
            return tuple(range(count))
        return tuple(range(default_count))
    indices = tuple(value)
    if any(
        not isinstance(index, numbers.Integral) or isinstance(index, (bool, np.bool_))
        for index in indices
    ):
        raise TypeError(f"{name} entries must be non-boolean integers")
    result = tuple(int(index) for index in indices)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    if any(index < 0 or index >= count for index in result):
        raise ValueError(f"{name} entries must address the candidate pool")
    if default_count is not None and len(result) != default_count:
        raise ValueError(f"{name} must contain exactly {default_count} entries")
    return result


def _formation_probabilities(
    value: np.ndarray | jax.Array | None,
    layout_count: int,
) -> np.ndarray:
    if value is None:
        probabilities = np.zeros((2, layout_count), dtype=np.float32)
        probabilities[:, 0] = 1.0
        return probabilities
    probabilities = np.asarray(value, dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = np.broadcast_to(
            probabilities[None, :], (2, probabilities.shape[0])
        ).copy()
    if probabilities.shape != (2, layout_count):
        raise ValueError("formation_probabilities must have shape [L] or [2, L]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("formation_probabilities must be finite and non-negative")
    total = np.sum(probabilities, axis=1, keepdims=True, dtype=np.float64)
    if np.any(total <= 0.0):
        raise ValueError("each team formation prior must have positive mass")
    return np.asarray(probabilities / total, dtype=np.float32)


def build_opening_policy_inputs(
    env: FootballWorld,
    team_0_candidates: Sequence[Player | PlayerProfile],
    team_1_candidates: Sequence[Player | PlayerProfile],
    team_0_formation_layouts: np.ndarray | jax.Array,
    team_1_formation_layouts: np.ndarray | jax.Array,
    *,
    team_0_registered: Sequence[int] | None = None,
    team_1_registered: Sequence[int] | None = None,
    team_0_starters: Sequence[int] | None = None,
    team_1_starters: Sequence[int] | None = None,
    max_registered_players: tuple[int, int] | None = None,
    formation_probabilities: np.ndarray | jax.Array | None = None,
    kickoff_team: int = TEAM_0,
    key: jax.Array | None = None,
) -> OpeningPolicyInputs:
    """Build a pre-reset opening-policy view from host candidate catalogs.

    ``Player.initial_position`` is interpreted only as a normalized-policy
    preference; registered formation slots determine the actual kickoff pose.
    If roster sampling is enabled and ``key`` is supplied, every candidate is
    sampled here exactly once with an identity-keyed stream. The realized
    profiles are what the policy observes and what the authoritative materializer
    later reconstructs. :func:`create_opening_match` deliberately passes
    ``key=None`` to both environment constructors, preventing a second draw.

    The starter sequences define the authored reference lineup and its slot
    order. They are returned as rule-policy parameters, while a learned policy
    remains free to emit another legal selection of the same fixed shape.
    """

    if type(env) is not FootballWorld:
        raise TypeError("env must be exactly FootballWorld")
    validate_kickoff_team(kickoff_team)
    if key is not None:
        validate_prng_key(key, name="key")
    profiles, preferred = _candidate_rows(
        env, team_0_candidates, team_1_candidates, key
    )
    layouts = (
        _layout_catalog(team_0_formation_layouts, name="team_0_formation_layouts"),
        _layout_catalog(team_1_formation_layouts, name="team_1_formation_layouts"),
    )
    if layouts[0].shape[0] != layouts[1].shape[0]:
        raise ValueError("both team formation catalogs must have the same L")
    layout_count = layouts[0].shape[0]
    starter_counts = (layouts[0].shape[1], layouts[1].shape[1])
    registered_indices = (
        _selection_indices(
            team_0_registered,
            count=len(profiles[0]),
            default_count=None,
            name="team_0_registered",
        ),
        _selection_indices(
            team_1_registered,
            count=len(profiles[1]),
            default_count=None,
            name="team_1_registered",
        ),
    )
    starter_indices = (
        _selection_indices(
            team_0_starters,
            count=len(profiles[0]),
            default_count=starter_counts[0],
            name="team_0_starters",
        ),
        _selection_indices(
            team_1_starters,
            count=len(profiles[1]),
            default_count=starter_counts[1],
            name="team_1_starters",
        ),
    )
    for team in (TEAM_0, TEAM_1):
        if not set(starter_indices[team]).issubset(registered_indices[team]):
            raise ValueError(f"team_{team} starters must be registered")
        if (
            sum(profiles[team][index].is_goalkeeper for index in starter_indices[team])
            != 1
        ):
            raise ValueError(
                f"team_{team} authored starters must contain one goalkeeper"
            )

    if max_registered_players is None:
        registered_maxima = tuple(len(row) for row in registered_indices)
    else:
        if (
            type(max_registered_players) is not tuple
            or len(max_registered_players) != 2
            or any(
                not isinstance(value, numbers.Integral)
                or isinstance(value, (bool, np.bool_))
                for value in max_registered_players
            )
        ):
            raise TypeError("max_registered_players must be a pair of integers")
        registered_maxima = tuple(int(value) for value in max_registered_players)
        int32_max = int(np.iinfo(np.int32).max)
        if any(value < 0 or value > int32_max for value in registered_maxima):
            raise ValueError(
                "max_registered_players must be in the non-negative int32 domain"
            )
    for team in (TEAM_0, TEAM_1):
        if registered_maxima[team] < len(registered_indices[team]):
            raise ValueError(f"team_{team} authored registration exceeds its maximum")
        if starter_counts[team] > registered_maxima[team]:
            raise ValueError(
                f"team_{team} starter count exceeds its registration maximum"
            )

    candidate_count = max(len(profiles[0]), len(profiles[1]))
    valid = np.zeros((2, candidate_count), dtype=np.bool_)
    player_id = np.full((2, candidate_count), NO_PLAYER, dtype=np.int32)
    goalkeeper = np.zeros((2, candidate_count), dtype=np.bool_)
    preferred_position = np.zeros((2, candidate_count, 2), dtype=np.float32)
    physical = np.zeros((2, candidate_count, 5), dtype=np.float32)
    registered = np.zeros((2, candidate_count), dtype=np.bool_)
    starter = np.zeros((2, candidate_count), dtype=np.bool_)
    placement = np.full((2, candidate_count), NO_PLAYER, dtype=np.int32)
    slot_offset = 0
    for team in (TEAM_0, TEAM_1):
        size = len(profiles[team])
        valid[team, :size] = True
        for candidate, profile in enumerate(profiles[team]):
            player_id[team, candidate] = profile.player_id
            goalkeeper[team, candidate] = profile.is_goalkeeper
            preferred_position[team, candidate] = preferred[team][candidate]
            physical[team, candidate] = (
                profile.max_speed_mps,
                profile.height_m,
                profile.max_reach_height_m,
                profile.ball_control,
                profile.endurance_factor,
            )
        registered[team, list(registered_indices[team])] = True
        starter[team, list(starter_indices[team])] = True
        for local_slot, candidate in enumerate(starter_indices[team]):
            placement[team, candidate] = slot_offset + local_slot
        slot_offset += starter_counts[team]

    context = env.normalization_context()
    xy = np.asarray(
        [context.position_scale_x_m, context.position_scale_y_m],
        dtype=np.float32,
    )
    if np.any(np.abs(preferred_position[..., 0][valid]) > env.stadium.half_length):
        raise ValueError("candidate preferred x positions must lie on the pitch")
    if np.any(np.abs(preferred_position[..., 1][valid]) > env.stadium.half_width):
        raise ValueError("candidate preferred y positions must lie on the pitch")

    probabilities = _formation_probabilities(formation_probabilities, layout_count)
    full_layouts = np.concatenate(layouts, axis=1).astype(np.float32)
    player_count = full_layouts.shape[1]
    team_id = np.concatenate(
        (
            np.full(starter_counts[0], TEAM_0, dtype=np.int32),
            np.full(starter_counts[1], TEAM_1, dtype=np.int32),
        )
    )
    starter_physical = np.concatenate(
        tuple(
            physical[team, np.asarray(starter_indices[team], dtype=np.int32)]
            for team in (TEAM_0, TEAM_1)
        ),
        axis=0,
    )
    facing = np.concatenate(
        (
            np.zeros(starter_counts[0], dtype=np.float32),
            np.full(starter_counts[1], np.pi, dtype=np.float32),
        )
    )
    for layout in range(layout_count):
        world = full_layouts[layout] * np.where(team_id[:, None] == TEAM_0, 1.0, -1.0)
        validate_formation_inside_pitch(
            world,
            stadium=env.stadium,
            name=f"opening candidate formation {layout}",
        )
        validate_no_torso_overlap(
            world,
            facing,
            body=env.body,
            name=f"opening candidate formation {layout}",
        )

    # Formation slots need identity-independent roles before the policy chooses
    # any candidate. Infer the one goalkeeper slot per team from the deepest,
    # most central anchor in the first registered attacking-frame layout.
    # Using the authored starter order here would make candidate permutation
    # silently change which physical slot is considered the goalkeeper.
    slot_goalkeeper = np.zeros(player_count, dtype=np.bool_)
    for team in (TEAM_0, TEAM_1):
        slots_for_team = np.flatnonzero(team_id == team)
        goalkeeper_score = full_layouts[0, slots_for_team, 0] + np.float32(
            1.0e-4
        ) * np.abs(full_layouts[0, slots_for_team, 1])
        slot_goalkeeper[slots_for_team[int(np.argmin(goalkeeper_score))]] = True
    role = jax.vmap(
        lambda anchor: classify_formation_roles(
            anchor,
            jnp.asarray(team_id),
            jnp.asarray(slot_goalkeeper),
        )
    )(jnp.asarray(full_layouts))
    player_mask = team_id[None, :] == np.arange(2, dtype=np.int32)[:, None]

    def private_player(value: np.ndarray) -> np.ndarray:
        return np.where(player_mask, value[None, :], np.float32(0.0))

    normalized_candidate_anchor = np.zeros(
        (2, layout_count, player_count, 2), dtype=np.float32
    )
    normalized_layouts = full_layouts / xy
    normalized_candidate_anchor[0, :, : starter_counts[0]] = normalized_layouts[
        :, : starter_counts[0]
    ]
    normalized_candidate_anchor[1, :, starter_counts[0] :] = normalized_layouts[
        :, starter_counts[0] :
    ]
    candidate_role = np.where(
        player_mask[:, None, :],
        np.asarray(role)[None, ...],
        np.int32(-1),
    )

    pool = OpeningPlayerPool(
        valid=jnp.asarray(valid),
        player_id=jnp.asarray(player_id),
        is_goalkeeper=jnp.asarray(goalkeeper),
        preferred_position=jnp.asarray(preferred_position / xy),
        max_speed=jnp.asarray(
            (physical[..., 0] - context.min_player_speed_mps)
            / (context.max_player_speed_mps - context.min_player_speed_mps)
        ),
        height=jnp.asarray(
            (physical[..., 1] - context.min_height_m)
            / (context.max_height_m - context.min_height_m)
        ),
        reach_height=jnp.asarray(
            (physical[..., 2] - context.min_reach_height_m)
            / (context.max_reach_height_m - context.min_reach_height_m)
        ),
        ball_control=jnp.asarray(
            (physical[..., 3] - context.min_ball_control)
            / (context.max_ball_control - context.min_ball_control)
        ),
        endurance_factor=jnp.asarray(
            (physical[..., 4] - context.min_endurance_factor)
            / (context.max_endurance_factor - context.min_endurance_factor)
        ),
    )
    formation = NormalizedOpeningFormationObservation(
        valid=jnp.ones(2, dtype=jnp.bool_),
        team=jnp.arange(2, dtype=jnp.int32),
        ours_kickoff=jnp.arange(2, dtype=jnp.int32) == kickoff_team,
        player_mask=jnp.asarray(player_mask),
        active=jnp.asarray(player_mask),
        is_goalkeeper=jnp.asarray(player_mask & slot_goalkeeper[None, :]),
        max_speed=jnp.asarray(
            private_player(
                (starter_physical[:, 0] - context.min_player_speed_mps)
                / (context.max_player_speed_mps - context.min_player_speed_mps)
            )
        ),
        height=jnp.asarray(
            private_player(
                (starter_physical[:, 1] - context.min_height_m)
                / (context.max_height_m - context.min_height_m)
            )
        ),
        reach_height=jnp.asarray(
            private_player(
                (starter_physical[:, 2] - context.min_reach_height_m)
                / (context.max_reach_height_m - context.min_reach_height_m)
            )
        ),
        ball_control=jnp.asarray(
            private_player(
                (starter_physical[:, 3] - context.min_ball_control)
                / (context.max_ball_control - context.min_ball_control)
            )
        ),
        endurance_factor=jnp.asarray(
            private_player(
                (starter_physical[:, 4] - context.min_endurance_factor)
                / (context.max_endurance_factor - context.min_endurance_factor)
            )
        ),
        candidate_probability=jnp.asarray(probabilities),
        candidate_anchor=jnp.asarray(normalized_candidate_anchor),
        candidate_role=jnp.asarray(candidate_role),
    )
    observation = OpeningManagerObservation(
        players=pool,
        formation=formation,
        max_registered_players=jnp.asarray(registered_maxima, dtype=jnp.int32),
        starter_count=jnp.asarray(starter_counts, dtype=jnp.int32),
    )
    validate_opening_policy_shapes(observation)
    _candidate_profiles(env, observation)
    return OpeningPolicyInputs(
        observation=observation,
        authored_selection=AuthoredOpeningSelection(
            registered=jnp.asarray(registered),
            starter=jnp.asarray(starter),
            placement_slot=jnp.asarray(placement),
        ),
    )


def _restore(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.float32(lower) + value.astype(np.float32) * np.float32(upper - lower)


def _candidate_profiles(
    env: FootballWorld,
    observations: OpeningManagerObservation,
) -> tuple[tuple[PlayerProfile | None, ...], tuple[PlayerProfile | None, ...]]:
    players = observations.players
    context = env.normalization_context()
    valid = np.asarray(players.valid, dtype=np.bool_)
    player_id = np.asarray(players.player_id, dtype=np.int64)
    goalkeeper = np.asarray(players.is_goalkeeper, dtype=np.bool_)
    max_speed = _restore(
        np.asarray(players.max_speed),
        context.min_player_speed_mps,
        context.max_player_speed_mps,
    )
    height = _restore(
        np.asarray(players.height), context.min_height_m, context.max_height_m
    )
    reach_height = _restore(
        np.asarray(players.reach_height),
        context.min_reach_height_m,
        context.max_reach_height_m,
    )
    ball_control = _restore(
        np.asarray(players.ball_control),
        context.min_ball_control,
        context.max_ball_control,
    )
    endurance = _restore(
        np.asarray(players.endurance_factor),
        context.min_endurance_factor,
        context.max_endurance_factor,
    )

    seen: set[int] = set()
    result: list[tuple[PlayerProfile | None, ...]] = []
    for team in range(2):
        row: list[PlayerProfile | None] = []
        for candidate in range(valid.shape[1]):
            if not valid[team, candidate]:
                row.append(None)
                continue
            identity = int(player_id[team, candidate])
            if identity in seen:
                raise ValueError(f"duplicate valid candidate player_id: {identity}")
            seen.add(identity)
            values = (
                np.float32(max_speed[team, candidate]),
                np.float32(height[team, candidate]),
                np.float32(reach_height[team, candidate]),
                np.float32(ball_control[team, candidate]),
                np.float32(endurance[team, candidate]),
            )
            if not bool(
                player_profile_values_valid(
                    identity,
                    *values,
                    head_radius=np.float32(env.body.head_radius_m),
                )
            ):
                raise ValueError(
                    f"candidate profile is outside its physical domain: "
                    f"team={team}, candidate={candidate}"
                )
            reach_margin = float(values[2] - values[1])
            sampling = env.roster_sampling
            normalized_domain = (
                sampling.min_max_speed_mps <= values[0] <= sampling.max_max_speed_mps
                and sampling.min_height_m <= values[1] <= sampling.max_height_m
                and sampling.min_reach_margin_m
                <= reach_margin
                <= sampling.max_reach_margin_m
                and sampling.min_ball_control <= values[3] <= sampling.max_ball_control
                and sampling.min_endurance_factor
                <= values[4]
                <= sampling.max_endurance_factor
            )
            if not normalized_domain:
                raise ValueError(
                    "candidate profile exceeds the configured normalization "
                    f"domain: team={team}, candidate={candidate}"
                )
            row.append(
                PlayerProfile(
                    player_id=identity,
                    max_speed_mps=float(values[0]),
                    height_m=float(values[1]),
                    max_reach_height_m=float(values[2]),
                    ball_control=float(values[3]),
                    endurance_factor=float(values[4]),
                    is_goalkeeper=bool(goalkeeper[team, candidate]),
                )
            )
        result.append(tuple(row))
    return result[0], result[1]


def _formation_catalog(
    env: FootballWorld,
    observations: OpeningManagerObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    formation = observations.formation
    valid = np.asarray(formation.valid, dtype=np.bool_)
    team = np.asarray(formation.team, dtype=np.int32)
    if not np.array_equal(valid, np.ones(2, dtype=np.bool_)):
        raise ValueError(
            "opening roster preparation requires both exact initial-boundary rows"
        )
    if not np.array_equal(team, np.arange(2, dtype=np.int32)):
        raise ValueError("opening formation team rows must be [0, 1]")

    player_mask = np.asarray(formation.player_mask, dtype=np.bool_)
    active = np.asarray(formation.active, dtype=np.bool_)
    if player_mask.ndim != 2 or player_mask.shape[0] != 2:
        raise ValueError("formation.player_mask must have shape [2, N]")
    if not np.all(np.sum(player_mask.astype(np.int32), axis=0) == 1):
        raise ValueError("each formation slot must belong to exactly one team")
    if not np.array_equal(active, player_mask):
        raise ValueError("all opening formation slots must be active")

    private_anchor = np.asarray(formation.candidate_anchor, dtype=np.float32)
    if not np.all(np.isfinite(private_anchor)):
        raise ValueError("formation candidate anchors must be finite")
    xy = np.asarray([env.stadium.half_length, env.stadium.half_width], dtype=np.float32)
    private_anchor = private_anchor * xy
    full_catalog = np.where(
        player_mask[0, None, :, None],
        private_anchor[0],
        private_anchor[1],
    ).astype(np.float32)
    return player_mask, private_anchor, full_catalog


def prepare_opening_match_inputs(
    env: FootballWorld,
    observations: OpeningManagerObservation,
    decision: OpeningManagerDecision,
) -> OpeningMatchInputs:
    """Validate and materialize one start-only opening-manager decision.

    This function accepts no ``Rollout``, ``State``, or management state. A
    caller must invoke it during new-match construction, before ``env.reset``.
    Policy arrays are copied to the host once and never become transition or
    scan inputs.
    """

    if type(env) is not FootballWorld:
        raise TypeError("env must be exactly FootballWorld")
    candidates, layouts, starter_slots = validate_opening_policy_shapes(
        observations, decision
    )
    observations, decision = jax.device_get((observations, decision))
    players = observations.players
    preferred_position = np.asarray(players.preferred_position, dtype=np.float32)
    if preferred_position.shape != (2, candidates, 2) or not np.all(
        np.isfinite(preferred_position)
    ):
        raise ValueError("candidate preferred positions must be finite [2, C, 2]")

    valid = np.asarray(players.valid, dtype=np.bool_)
    registered = np.asarray(decision.registered, dtype=np.bool_)
    starter = np.asarray(decision.starter, dtype=np.bool_)
    placement = np.asarray(decision.placement_slot, dtype=np.int32)
    max_registered = np.asarray(observations.max_registered_players, dtype=np.int32)
    starter_count = np.asarray(observations.starter_count, dtype=np.int32)
    if np.any(max_registered < 0) or np.any(starter_count <= 0):
        raise ValueError("registered maxima must be non-negative and starters positive")
    if np.any(starter_count > max_registered):
        raise ValueError("starter_count cannot exceed max_registered_players")
    if np.any(registered & (~valid)):
        raise ValueError("registered players must be valid candidates")
    if np.any(starter & (~registered)):
        raise ValueError("starters must be a subset of the registered squad")
    registered_count = np.sum(registered, axis=1)
    actual_starters = np.sum(starter, axis=1)
    if np.any(registered_count > max_registered):
        raise ValueError("registered squad exceeds max_registered_players")
    if np.any(actual_starters != starter_count):
        raise ValueError("starter mask must contain exactly starter_count per team")

    requested = np.asarray(decision.formation.requested, dtype=np.bool_)
    selected_layout = np.asarray(decision.formation.layout_index, dtype=np.int32)
    if not np.array_equal(requested, np.ones(2, dtype=np.bool_)):
        raise ValueError("both teams must request one opening formation")
    if np.any(selected_layout < 0) or np.any(selected_layout >= layouts):
        raise ValueError("opening formation index lies outside the registered catalog")

    player_mask, private_anchor, full_catalog = _formation_catalog(env, observations)
    if starter_slots != player_mask.shape[1]:
        raise ValueError("formation candidate slot axes are inconsistent")
    if np.any(np.sum(player_mask, axis=1) != starter_count):
        raise ValueError("starter_count must match each team's formation slot count")
    nonstarter_has_slot = (~starter) & (placement != -1)
    if np.any(nonstarter_has_slot):
        raise ValueError("non-starters must use NO_PLAYER placement")

    profiles = _candidate_profiles(env, observations)
    team_players: list[tuple[Player, ...]] = []
    team_benches: list[tuple[PlayerProfile, ...]] = []
    selected_positions: list[np.ndarray] = []
    ordered_catalog_slots: list[int] = []
    for team in range(2):
        team_slots = np.flatnonzero(player_mask[team])
        selected_candidates = np.flatnonzero(starter[team])
        selected_placements = placement[team, selected_candidates]
        if np.any(selected_placements < 0) or np.any(
            selected_placements >= starter_slots
        ):
            raise ValueError(f"team {team} starter placement lies outside the catalog")
        if not np.array_equal(
            np.sort(selected_placements), np.sort(team_slots.astype(np.int32))
        ):
            raise ValueError(
                f"team {team} starters must cover its formation slots exactly once"
            )

        order = np.argsort(selected_placements, kind="stable")
        selected_candidates = selected_candidates[order]
        selected_placements = selected_placements[order]
        row_players: list[Player] = []
        row_positions: list[np.ndarray] = []
        for candidate, slot in zip(
            selected_candidates.tolist(),
            selected_placements.tolist(),
            strict=True,
        ):
            profile = profiles[team][candidate]
            if profile is None:
                raise ValueError("a selected starter has no valid candidate profile")
            position = private_anchor[team, selected_layout[team], slot]
            row_positions.append(position)
            row_players.append(
                Player(
                    profile=profile,
                    initial_position=(float(position[0]), float(position[1])),
                )
            )
        team_players.append(tuple(row_players))
        selected_positions.append(np.asarray(row_positions, dtype=np.float32))
        ordered_catalog_slots.extend(selected_placements.tolist())

        bench: list[PlayerProfile] = []
        for candidate in np.flatnonzero(registered[team] & (~starter[team])):
            profile = profiles[team][int(candidate)]
            if profile is None:
                raise ValueError("a registered bench entry has no candidate profile")
            bench.append(profile)
        team_benches.append(tuple(bench))

    team_0, team_1 = team_players
    validate_public_rosters(
        team_0,
        team_1,
        minimum_team_players=env.match.minimum_team_players,
        body=env.body,
        timebase=env.timebase,
        stadium=env.stadium,
        boundary_margin_m=env.boundary_margin_m,
        roster_sampling=env.roster_sampling,
    )
    world_position = np.concatenate(
        (selected_positions[0], -selected_positions[1]), axis=0
    )
    validate_formation_inside_pitch(
        world_position,
        stadium=env.stadium,
        name="opening manager formation",
    )
    facing = np.concatenate(
        (
            np.zeros(len(team_0), dtype=np.float32),
            np.full(len(team_1), np.pi, dtype=np.float32),
        )
    )
    validate_no_torso_overlap(
        world_position,
        facing,
        body=env.body,
        name="opening manager formation",
    )

    catalog = full_catalog[:, np.asarray(ordered_catalog_slots, dtype=np.int32)]
    for layout in range(catalog.shape[0]):
        validate_formation_inside_pitch(
            catalog[layout],
            stadium=env.stadium,
            name=f"opening manager registered formation {layout}",
        )
    selected_physical = np.concatenate(selected_positions, axis=0)
    original_probability = np.asarray(
        observations.formation.candidate_probability, dtype=np.float32
    )
    registered_probability = np.zeros(
        (2, original_probability.shape[1] + 1), dtype=np.float32
    )
    for team in (TEAM_0, TEAM_1):
        registered_probability[team, 0] = original_probability[
            team, selected_layout[team]
        ]
        registered_probability[team, 1:] = original_probability[team]
        registered_probability[team, 1 + selected_layout[team]] = 0.0
    probability_mass = registered_probability.sum(axis=1, keepdims=True)
    if np.any(probability_mass <= 0.0):
        raise ValueError("selected formation must preserve positive prior mass")
    registered_probability /= probability_mass

    return OpeningMatchInputs(
        team_0=team_0,
        team_1=team_1,
        team_0_bench=team_benches[0],
        team_1_bench=team_benches[1],
        formation_layouts=np.asarray(catalog, dtype=np.float32),
        formation_probabilities=registered_probability,
        selected_formation_index=(int(selected_layout[0]), int(selected_layout[1])),
        selected_formation_layout=np.asarray(selected_physical, dtype=np.float32),
    )


def create_opening_match_from_policy(
    env: FootballWorld,
    inputs: OpeningPolicyInputs,
    match_key: jax.Array,
    *,
    policy=None,
    parameters=NO_POLICY_PARAMETERS,
    kickoff_team: int = TEAM_0,
    rules: ManagementRules | None = None,
) -> CreatedPolicyOpeningMatch:
    """Invoke an opening manager once, then construct exactly one new match.

    This host-only orchestration accepts :class:`OpeningPolicyInputs`, never a
    rollout or checkpoint. Passing ``policy=None`` selects the built-in full
    opening manager only when ``env.policies.rule_based_opening_manager`` is
    enabled. Candidate tensors and opening-policy memory are
    discarded before ordinary rollout, and the committed opening flag prevents
    the separate formation adapter from applying another opening decision.
    """

    if type(env) is not FootballWorld:
        raise TypeError("env must be exactly FootballWorld")
    if type(inputs) is not OpeningPolicyInputs:
        raise TypeError("inputs must be OpeningPolicyInputs")
    if policy is None:
        if not env.policies.rule_based_opening_manager:
            raise ValueError(
                "policy is required when the built-in opening manager is disabled"
            )
        policy = make_rule_based_opening_manager_policy()
    validate_opening_manager_policy(policy)
    validate_prng_key(match_key, name="match_key")

    policy_state = policy.initialize(inputs.observation, parameters)
    policy_step = policy.step(
        inputs.observation,
        match_key,
        policy_state,
        parameters,
    )
    created = create_opening_match(
        env,
        inputs.observation,
        policy_step.decision,
        kickoff_team=kickoff_team,
        rules=rules,
    )
    return CreatedPolicyOpeningMatch(
        match=created,
        decision=policy_step.decision,
        policy_state=policy_step.state,
    )


def create_opening_match(
    env: FootballWorld,
    observations: OpeningManagerObservation,
    decision: OpeningManagerDecision,
    *,
    kickoff_team: int = TEAM_0,
    rules: ManagementRules | None = None,
) -> CreatedOpeningMatch:
    """Materialize and initialize exactly one new match before its first frame.

    There is intentionally no rollout/state argument and no sampling key. Any
    candidate sampling must already have happened in
    :func:`build_opening_policy_inputs`; reset and bench registration preserve
    those realized profiles exactly. A layout-zero command commits the already
    materialized physical pose, preventing a second opening-policy call.
    """

    observed_kickoff = np.flatnonzero(
        np.asarray(observations.formation.ours_kickoff, dtype=np.bool_)
    )
    if observed_kickoff.shape != (1,) or int(observed_kickoff[0]) != kickoff_team:
        raise ValueError("kickoff_team must match the opening-policy observation")
    inputs = prepare_opening_match_inputs(env, observations, decision)
    reset = env.reset(
        inputs.team_0,
        inputs.team_1,
        kickoff_team=kickoff_team,
        key=None,
    )
    management = env.initialize_management(
        reset.rollout,
        inputs.team_0_bench,
        inputs.team_1_bench,
        rules=rules,
        formation_layouts=inputs.formation_layouts,
        formation_probabilities=inputs.formation_probabilities,
        key=None,
    )
    committed = env.opening_formation_command(
        reset.rollout,
        reset.setup,
        management.squad,
        management.state,
        ManagerFormationCommand(
            requested=jnp.ones(2, dtype=jnp.bool_),
            layout_index=jnp.zeros(2, dtype=jnp.int32),
        ),
    )
    if not np.array_equal(np.asarray(committed.applied), np.ones(2, dtype=np.bool_)):
        raise RuntimeError("new-match opening formation could not be committed")
    return CreatedOpeningMatch(
        inputs=inputs,
        reset=ResetResult(rollout=committed.rollout, setup=committed.setup),
        management=ManagementInitialization(
            squad=management.squad,
            state=committed.management,
        ),
    )


__all__ = [
    "CreatedOpeningMatch",
    "CreatedPolicyOpeningMatch",
    "OpeningMatchInputs",
    "OpeningPolicyInputs",
    "build_opening_policy_inputs",
    "create_opening_match",
    "create_opening_match_from_policy",
    "prepare_opening_match_inputs",
]
