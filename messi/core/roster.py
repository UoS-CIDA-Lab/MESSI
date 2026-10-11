"""reset 경계에서 roster 선택과 player ability를 실현한다."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from messi.core.constants import (
    MAX_PLAYERS_PER_TEAM,
    MAX_SUPPORTED_PLAYER_SPEED_MPS,
    STANDARD_PLAYERS_PER_TEAM,
    TEAM_COUNT,
)
from messi.core.formation import FORMATION_COUNT, realize_initial_formation
from messi.core.numeric import require_float32_representable
from messi.core.randomness import RandomEvent, event_random_key, validate_prng_key
from messi.core.state import RosterState

DEFAULT_PLAYERS_PER_TEAM = STANDARD_PLAYERS_PER_TEAM
DEFAULT_CANDIDATES_PER_TEAM = 23
DEFAULT_BENCH_PLAYERS_PER_TEAM = 12
MAX_BENCH_PLAYERS_PER_TEAM = 15


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """호출자가 선택적으로 제공하는 player identity와 ability."""

    player_id: int | None = None
    max_speed_mps: float | None = None
    height_m: float | None = None
    max_reach_height_m: float | None = None
    ball_control: float | None = None
    endurance_factor: float | None = None
    is_goalkeeper: bool | None = None


@dataclass(frozen=True, slots=True)
class TeamRoster:
    """후보 roster와 선택적인 선발 index."""

    players: tuple[PlayerProfile, ...] | None = None
    starting_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.players is not None:
            object.__setattr__(self, "players", tuple(self.players))
        if self.starting_indices is not None:
            object.__setattr__(self, "starting_indices", tuple(self.starting_indices))


@dataclass(frozen=True, slots=True)
class RosterSampling:
    """누락된 ability를 생성하는 bounded Gaussian 설정."""

    mean_max_speed_mps: float = 7.96
    std_max_speed_mps: float = 0.35
    min_max_speed_mps: float = 3.0
    # 논문 근거: "Analyzing soccer match sprint distances: A comparison of GPS-based
    # absolute and relative thresholds"는 elite match peak 32.9±1.4 km/h를 보고했고,
    # "Peak match sprinting speed during soccer matches: analysing the pre- and
    # post-peak speed dynamics"는 반복 근접 관측이 없는 spike를 제외했다. 10.5 m/s는
    # 두 cohort보다 여유가 있는 현실 상한 envelope이며 논문의 직접 fit은 아니다.
    max_max_speed_mps: float = 10.5
    mean_height_m: float = 1.8
    std_height_m: float = 0.05
    min_height_m: float = 1.45
    max_height_m: float = 2.2
    mean_reach_margin_m: float = 0.9
    std_reach_margin_m: float = 0.08
    min_reach_margin_m: float = 0.0
    max_reach_margin_m: float = 1.4
    mean_ball_control: float = 0.5
    std_ball_control: float = 0.08
    min_ball_control: float = 0.0
    max_ball_control: float = 1.0
    mean_endurance_factor: float = 1.0
    std_endurance_factor: float = 0.08
    min_endurance_factor: float = 0.5
    max_endurance_factor: float = 1.5
    clip_standard_deviations: float = 2.5

    def __post_init__(self) -> None:
        fields = (
            "mean_max_speed_mps",
            "std_max_speed_mps",
            "min_max_speed_mps",
            "max_max_speed_mps",
            "mean_height_m",
            "std_height_m",
            "min_height_m",
            "max_height_m",
            "mean_reach_margin_m",
            "std_reach_margin_m",
            "min_reach_margin_m",
            "max_reach_margin_m",
            "mean_ball_control",
            "std_ball_control",
            "min_ball_control",
            "max_ball_control",
            "mean_endurance_factor",
            "std_endurance_factor",
            "min_endurance_factor",
            "max_endurance_factor",
            "clip_standard_deviations",
        )
        for name in fields:
            require_float32_representable(name, getattr(self, name))
        for name in (
            "std_max_speed_mps",
            "std_height_m",
            "std_reach_margin_m",
            "std_ball_control",
            "std_endurance_factor",
            "clip_standard_deviations",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        clip = np.float32(self.clip_standard_deviations)
        for mean_name, std_name in (
            ("mean_max_speed_mps", "std_max_speed_mps"),
            ("mean_height_m", "std_height_m"),
            ("mean_reach_margin_m", "std_reach_margin_m"),
            ("mean_ball_control", "std_ball_control"),
            ("mean_endurance_factor", "std_endurance_factor"),
        ):
            mean = np.float32(getattr(self, mean_name))
            std = np.float32(getattr(self, std_name))
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                spread = std * clip
                lower = mean - spread
                upper = mean + spread
            if not np.isfinite(spread):
                raise ValueError(
                    f"{std_name} * clip_standard_deviations는 float32에서 유한해야 한다"
                )
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(
                    f"{mean_name}의 bounded Gaussian endpoint는 float32에서 "
                    "유한해야 한다"
                )
        for low_name, high_name in (
            ("min_max_speed_mps", "max_max_speed_mps"),
            ("min_height_m", "max_height_m"),
            ("min_reach_margin_m", "max_reach_margin_m"),
            ("min_ball_control", "max_ball_control"),
            ("min_endurance_factor", "max_endurance_factor"),
        ):
            if getattr(self, low_name) > getattr(self, high_name):
                raise ValueError(f"{low_name}은 {high_name}보다 클 수 없다")
        if self.min_max_speed_mps < 0.0:
            raise ValueError("min_max_speed_mps는 0 이상이어야 한다")
        if self.max_max_speed_mps > MAX_SUPPORTED_PLAYER_SPEED_MPS:
            raise ValueError(
                "max_max_speed_mps는 collision solver 지원 상한 "
                f"{MAX_SUPPORTED_PLAYER_SPEED_MPS} 이하여야 한다"
            )
        minimum_normal = float(np.finfo(np.float32).tiny)
        if self.min_height_m < minimum_normal:
            raise ValueError(
                "min_height_m는 float32 최소 positive normal 이상이어야 한다"
            )
        if self.min_reach_margin_m < 0.0:
            raise ValueError("min_reach_margin_m는 0 이상이어야 한다")
        if self.min_endurance_factor < minimum_normal:
            raise ValueError(
                "min_endurance_factor는 float32 최소 positive normal 이상이어야 한다"
            )
        require_float32_representable(
            "min_endurance_factor reciprocal",
            1.0 / self.min_endurance_factor,
        )
        for name, height, margin in (
            (
                "minimum reach height",
                self.min_height_m,
                self.min_reach_margin_m,
            ),
            (
                "maximum reach height",
                self.max_height_m,
                self.max_reach_margin_m,
            ),
        ):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                represented = np.float32(height) + np.float32(margin)
            exact = float(height) + float(margin)
            if not np.isfinite(represented):
                raise ValueError(f"{name}는 float32에서 유한해야 한다")
            if exact != 0.0 and represented == 0.0:
                raise ValueError(f"{name}는 float32에서 0으로 underflow하면 안 된다")


class RealizedRoster(NamedTuple):
    """rollout 입력과 reset 진단용 후보 index."""

    roster: RosterState
    candidate_index: jax.Array


class BenchRoster(NamedTuple):
    """Management 경계에서만 사용하는 고정 폭 등록 bench."""

    valid: jax.Array
    available: jax.Array
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    height: jax.Array
    reach_height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name}은 bool이 아닌 정수여야 한다")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name}은 1 이상이어야 한다")
    return result


def validate_player_capacity(player_count: object) -> int:
    """대칭 team slot capacity를 검증하고 team당 slot 수를 반환한다."""

    slots = _positive_integer("player_count", player_count)
    if slots % TEAM_COUNT:
        raise ValueError("player_count는 두 team에 같은 slot을 주는 짝수여야 한다")
    players_per_team = slots // TEAM_COUNT
    if players_per_team > MAX_PLAYERS_PER_TEAM:
        raise ValueError(f"players_per_team은 {MAX_PLAYERS_PER_TEAM} 이하여야 한다")
    return players_per_team


def _validate_player_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name}은 bool이 아닌 정수여야 한다")
    result = int(value)
    if result < 0 or result > np.iinfo(np.int32).max:
        raise ValueError(f"{name}은 non-negative int32 범위여야 한다")
    return result


def _reserved_player_ids(values: Sequence[object]) -> set[int]:
    """생성 ID가 피해야 할 host-side identity 집합을 검증한다."""

    return {
        _validate_player_id(value, name=f"reserved_player_ids[{index}]")
        for index, value in enumerate(values)
    }


def _bounded_value(
    value: float | None,
    *,
    name: str,
    lower: float,
    upper: float,
) -> float | None:
    if value is None:
        return None
    result = require_float32_representable(name, value)
    if not lower <= result <= upper:
        raise ValueError(f"{name}은 [{lower}, {upper}] 범위여야 한다")
    return result


def _validate_profile_abilities(
    profile: PlayerProfile,
    *,
    name: str,
    sampling: RosterSampling,
) -> PlayerProfile:
    """명시된 ability와 goalkeeper flag를 공통 roster 제한으로 검증한다."""

    speed = _bounded_value(
        profile.max_speed_mps,
        name=f"{name}.max_speed_mps",
        lower=sampling.min_max_speed_mps,
        upper=sampling.max_max_speed_mps,
    )
    height = _bounded_value(
        profile.height_m,
        name=f"{name}.height_m",
        lower=sampling.min_height_m,
        upper=sampling.max_height_m,
    )
    reach = _bounded_value(
        profile.max_reach_height_m,
        name=f"{name}.max_reach_height_m",
        lower=sampling.min_height_m + sampling.min_reach_margin_m,
        upper=sampling.max_height_m + sampling.max_reach_margin_m,
    )
    control = _bounded_value(
        profile.ball_control,
        name=f"{name}.ball_control",
        lower=sampling.min_ball_control,
        upper=sampling.max_ball_control,
    )
    endurance = _bounded_value(
        profile.endurance_factor,
        name=f"{name}.endurance_factor",
        lower=sampling.min_endurance_factor,
        upper=sampling.max_endurance_factor,
    )
    if height is not None and reach is not None:
        margin = reach - height
        if not sampling.min_reach_margin_m <= margin <= sampling.max_reach_margin_m:
            raise ValueError(f"{name}의 reach-height margin은 환경 제한 범위여야 한다")
    if profile.is_goalkeeper is not None and not isinstance(
        profile.is_goalkeeper, bool
    ):
        raise TypeError(f"{name}.is_goalkeeper는 bool이어야 한다")
    return PlayerProfile(
        player_id=profile.player_id,
        max_speed_mps=speed,
        height_m=height,
        max_reach_height_m=reach,
        ball_control=control,
        endurance_factor=endurance,
        is_goalkeeper=profile.is_goalkeeper,
    )


def _normalize_candidates(
    teams: Sequence[TeamRoster],
    *,
    generated_candidates_per_team: int,
    sampling: RosterSampling,
    reserved_player_ids: set[int] | None = None,
) -> tuple[tuple[PlayerProfile, ...], tuple[PlayerProfile, ...]]:
    candidates: list[list[PlayerProfile]] = []
    used_ids: set[int] = set()
    generated_reserved_ids = (
        set() if reserved_player_ids is None else set(reserved_player_ids)
    )
    missing_id_locations: list[tuple[int, int]] = []
    for team_index, team in enumerate(teams):
        if team.players is None:
            current = [PlayerProfile() for _ in range(generated_candidates_per_team)]
        else:
            current = list(team.players)
            if not current:
                raise ValueError(f"team_{team_index}.players는 비어 있을 수 없다")
        for player_index, profile in enumerate(current):
            if not isinstance(profile, PlayerProfile):
                raise TypeError(
                    f"team_{team_index}.players[{player_index}]는 PlayerProfile이어야 한다"
                )
            if profile.player_id is None:
                missing_id_locations.append((team_index, player_index))
            else:
                identity = _validate_player_id(
                    profile.player_id,
                    name=f"team_{team_index}.players[{player_index}].player_id",
                )
                if identity in used_ids:
                    raise ValueError(
                        "player_id는 두 team의 후보 roster 전체에서 고유해야 한다"
                    )
                used_ids.add(identity)
        candidates.append(current)

    next_identity = 0
    for team_index, player_index in missing_id_locations:
        while next_identity in used_ids or next_identity in generated_reserved_ids:
            next_identity += 1
        profile = candidates[team_index][player_index]
        candidates[team_index][player_index] = PlayerProfile(
            player_id=next_identity,
            max_speed_mps=profile.max_speed_mps,
            height_m=profile.height_m,
            max_reach_height_m=profile.max_reach_height_m,
            ball_control=profile.ball_control,
            endurance_factor=profile.endurance_factor,
            is_goalkeeper=profile.is_goalkeeper,
        )
        used_ids.add(next_identity)

    for team_index, current in enumerate(candidates):
        for player_index, profile in enumerate(current):
            prefix = f"team_{team_index}.players[{player_index}]"
            current[player_index] = _validate_profile_abilities(
                profile,
                name=prefix,
                sampling=sampling,
            )
    return tuple(candidates[0]), tuple(candidates[1])


def _require_key(key: jax.Array | None, *, reason: str) -> jax.Array:
    if key is None:
        raise ValueError(f"{reason}에는 명시적인 key가 필요하다")
    return key


def _random_permutation(key: jax.Array, values: np.ndarray) -> np.ndarray:
    order = np.asarray(jax.random.permutation(key, values.shape[0]))
    return values[order]


def _select_team(
    candidates: tuple[PlayerProfile, ...],
    starting_indices: tuple[int, ...] | None,
    *,
    players_per_team: int,
    key: jax.Array | None,
    team_index: int,
) -> np.ndarray:
    candidate_count = len(candidates)
    if candidate_count < players_per_team:
        raise ValueError(
            f"team_{team_index} 후보 수는 players_per_team 이상이어야 한다"
        )
    if starting_indices is not None:
        if len(starting_indices) != players_per_team:
            raise ValueError(
                f"team_{team_index}.starting_indices 길이는 players_per_team과 같아야 한다"
            )
        normalized = []
        for index, value in enumerate(starting_indices):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"team_{team_index}.starting_indices[{index}]는 정수여야 한다"
                )
            selected = int(value)
            if not 0 <= selected < candidate_count:
                raise ValueError(
                    f"team_{team_index}.starting_indices가 범위를 벗어났다"
                )
            normalized.append(selected)
        if len(set(normalized)) != players_per_team:
            raise ValueError(f"team_{team_index}.starting_indices는 고유해야 한다")
        return np.asarray(normalized, dtype=np.int32)
    if candidate_count == players_per_team:
        return np.arange(candidate_count, dtype=np.int32)

    selection_key = event_random_key(
        _require_key(key, reason="무작위 starting roster 선택"),
        RandomEvent.ACTIVE_ROSTER_SAMPLING,
        team_index,
        0,
    )
    goalkeeper = np.asarray(
        [i for i, p in enumerate(candidates) if p.is_goalkeeper is True], dtype=np.int32
    )
    outfield = np.asarray(
        [i for i, p in enumerate(candidates) if p.is_goalkeeper is not True],
        dtype=np.int32,
    )
    if goalkeeper.size:
        if outfield.size < players_per_team - 1:
            raise ValueError(
                f"team_{team_index}에 선발 가능한 outfield 후보가 부족하다"
            )
        chosen_goalkeeper = _random_permutation(selection_key, goalkeeper)[:1]
        chosen_outfield = _random_permutation(
            jax.random.fold_in(selection_key, np.uint32(1)), outfield
        )[: players_per_team - 1]
        return np.concatenate((chosen_goalkeeper, chosen_outfield)).astype(np.int32)
    return _random_permutation(
        selection_key, np.arange(candidate_count, dtype=np.int32)
    )[:players_per_team].astype(np.int32)


def _realize_goalkeepers(
    selected: Sequence[PlayerProfile],
    *,
    key: jax.Array | None,
    team_index: int,
) -> np.ndarray:
    explicit = [i for i, p in enumerate(selected) if p.is_goalkeeper is True]
    if len(explicit) > 1:
        raise ValueError(
            f"team_{team_index} 선발에는 goalkeeper가 최대 한 명이어야 한다"
        )
    result = np.zeros(len(selected), dtype=np.bool_)
    if explicit:
        result[explicit[0]] = True
        return result
    missing = np.asarray(
        [i for i, p in enumerate(selected) if p.is_goalkeeper is None], dtype=np.int32
    )
    if missing.size:
        goalkeeper_key = event_random_key(
            _require_key(key, reason="누락된 goalkeeper 지정"),
            RandomEvent.ACTIVE_ROSTER_SAMPLING,
            team_index,
            1,
        )
        chosen = int(
            np.asarray(jax.random.randint(goalkeeper_key, (), 0, missing.shape[0]))
        )
        result[missing[chosen]] = True
    return result


def _realize_abilities(
    selected: Sequence[PlayerProfile],
    *,
    key: jax.Array | None,
    sampling: RosterSampling,
    random_event: RandomEvent = RandomEvent.ACTIVE_ROSTER_SAMPLING,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    supplied = np.asarray(
        [
            (
                np.nan if p.max_speed_mps is None else p.max_speed_mps,
                np.nan if p.height_m is None else p.height_m,
                np.nan if p.max_reach_height_m is None else p.max_reach_height_m,
                np.nan if p.ball_control is None else p.ball_control,
                np.nan if p.endurance_factor is None else p.endurance_factor,
            )
            for p in selected
        ],
        dtype=np.float32,
    )
    missing = np.isnan(supplied)
    if not bool(missing.any()):
        values = jnp.asarray(supplied, dtype=jnp.float32)
        return values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4]

    ability_key = event_random_key(
        _require_key(key, reason="누락된 player ability sampling"),
        random_event,
        2,
    )
    z = jax.random.normal(ability_key, supplied.shape, dtype=jnp.float32)
    z = jnp.clip(
        z,
        jnp.float32(-sampling.clip_standard_deviations),
        jnp.float32(sampling.clip_standard_deviations),
    )
    supplied_array = jnp.asarray(supplied, dtype=jnp.float32)
    missing_array = jnp.asarray(missing)
    reach_supplied = ~missing_array[:, 2]
    reach_value = supplied_array[:, 2]
    height_lower = jnp.where(
        reach_supplied,
        jnp.maximum(
            sampling.min_height_m,
            reach_value - sampling.max_reach_margin_m,
        ),
        sampling.min_height_m,
    )
    height_upper = jnp.where(
        reach_supplied,
        jnp.minimum(
            sampling.max_height_m,
            reach_value - sampling.min_reach_margin_m,
        ),
        sampling.max_height_m,
    )
    height_lower = jnp.clip(
        height_lower,
        sampling.min_height_m,
        sampling.max_height_m,
    )
    height_upper = jnp.maximum(
        jnp.clip(
            height_upper,
            sampling.min_height_m,
            sampling.max_height_m,
        ),
        height_lower,
    )
    sampled_height = jnp.clip(
        sampling.mean_height_m + sampling.std_height_m * z[:, 1],
        height_lower,
        height_upper,
    )
    height = jnp.where(missing_array[:, 1], sampled_height, supplied_array[:, 1])
    sampled_margin = jnp.clip(
        sampling.mean_reach_margin_m + sampling.std_reach_margin_m * z[:, 2],
        sampling.min_reach_margin_m,
        sampling.max_reach_margin_m,
    )
    speed = jnp.where(
        missing_array[:, 0],
        jnp.clip(
            sampling.mean_max_speed_mps + sampling.std_max_speed_mps * z[:, 0],
            sampling.min_max_speed_mps,
            sampling.max_max_speed_mps,
        ),
        supplied_array[:, 0],
    )
    reach = jnp.where(
        missing_array[:, 2], height + sampled_margin, supplied_array[:, 2]
    )
    control = jnp.where(
        missing_array[:, 3],
        jnp.clip(
            sampling.mean_ball_control + sampling.std_ball_control * z[:, 3],
            sampling.min_ball_control,
            sampling.max_ball_control,
        ),
        supplied_array[:, 3],
    )
    endurance = jnp.where(
        missing_array[:, 4],
        jnp.clip(
            sampling.mean_endurance_factor + sampling.std_endurance_factor * z[:, 4],
            sampling.min_endurance_factor,
            sampling.max_endurance_factor,
        ),
        supplied_array[:, 4],
    )
    return speed, height, reach, control, endurance


def _realize_bench(
    team_0: Sequence[PlayerProfile] | None = None,
    team_1: Sequence[PlayerProfile] | None = None,
    *,
    key: jax.Array | None = None,
    generated_players_per_team: int = DEFAULT_BENCH_PLAYERS_PER_TEAM,
    sampling: RosterSampling | None = None,
    reserved_player_ids: Sequence[object] = (),
) -> BenchRoster:
    """두 team의 등록 bench를 reset 경계에서 고정 폭으로 실현한다."""

    generated_players_per_team = _positive_integer(
        "generated_players_per_team", generated_players_per_team
    )
    if generated_players_per_team > MAX_BENCH_PLAYERS_PER_TEAM:
        raise ValueError(
            f"generated_players_per_team은 {MAX_BENCH_PLAYERS_PER_TEAM} 이하여야 한다"
        )
    if key is not None:
        validate_prng_key(key)
    sampling = RosterSampling() if sampling is None else sampling
    if not isinstance(sampling, RosterSampling):
        raise TypeError("sampling은 RosterSampling이어야 한다")
    supplied = (team_0, team_1)
    materialized: list[tuple[PlayerProfile, ...]] = []
    for team_index, profiles in enumerate(supplied):
        current = (
            tuple(
                PlayerProfile(is_goalkeeper=index == 0)
                for index in range(generated_players_per_team)
            )
            if profiles is None
            else tuple(profiles)
        )
        if len(current) > MAX_BENCH_PLAYERS_PER_TEAM:
            raise ValueError(
                f"team_{team_index} bench는 {MAX_BENCH_PLAYERS_PER_TEAM}명 이하여야 한다"
            )
        for player_index, profile in enumerate(current):
            if not isinstance(profile, PlayerProfile):
                raise TypeError(
                    f"team_{team_index}[{player_index}]는 PlayerProfile이어야 한다"
                )
        materialized.append(current)

    # Missing ID를 배정하기 전에 두 team의 explicit ID를 모두 예약한다. 그래야
    # generated/explicit team 순서를 뒤집어도 같은 identity 집합이 유효하다.
    used_ids = _reserved_player_ids(reserved_player_ids)
    for team_index, current in enumerate(materialized):
        for player_index, profile in enumerate(current):
            identity = profile.player_id
            if identity is None:
                continue
            identity = _validate_player_id(
                identity, name=f"team_{team_index}[{player_index}].player_id"
            )
            if identity in used_ids:
                raise ValueError(
                    "bench player_id는 reserved ID와 두 team 전체에서 고유해야 하며 "
                    "중복될 수 없다"
                )
            used_ids.add(identity)

    next_generated_id = int(np.iinfo(np.int32).max)
    teams: list[tuple[PlayerProfile, ...]] = []
    for team_index, current in enumerate(materialized):
        normalized: list[PlayerProfile] = []
        for player_index, profile in enumerate(current):
            identity = profile.player_id
            if identity is None:
                while next_generated_id in used_ids:
                    next_generated_id -= 1
                if next_generated_id < 0:
                    raise ValueError("생성 가능한 non-negative int32 player_id가 없다")
                identity = next_generated_id
                next_generated_id -= 1
                used_ids.add(identity)
            else:
                identity = _validate_player_id(
                    identity, name=f"team_{team_index}[{player_index}].player_id"
                )
            normalized.append(
                _validate_profile_abilities(
                    PlayerProfile(
                        player_id=identity,
                        max_speed_mps=profile.max_speed_mps,
                        height_m=profile.height_m,
                        max_reach_height_m=profile.max_reach_height_m,
                        ball_control=profile.ball_control,
                        endurance_factor=profile.endurance_factor,
                        is_goalkeeper=profile.is_goalkeeper,
                    ),
                    name=f"team_{team_index}[{player_index}]",
                    sampling=sampling,
                )
            )
        teams.append(tuple(normalized))

    width = max((len(team) for team in teams), default=0)
    if width == 0:
        empty_shape = (2, 0)
        return BenchRoster(
            valid=jnp.zeros(empty_shape, dtype=jnp.bool_),
            available=jnp.zeros(empty_shape, dtype=jnp.bool_),
            player_id=jnp.full(empty_shape, -1, dtype=jnp.int32),
            is_goalkeeper=jnp.zeros(empty_shape, dtype=jnp.bool_),
            max_speed=jnp.zeros(empty_shape, dtype=jnp.float32),
            height=jnp.zeros(empty_shape, dtype=jnp.float32),
            reach_height=jnp.zeros(empty_shape, dtype=jnp.float32),
            ball_control=jnp.zeros(empty_shape, dtype=jnp.float32),
            endurance_factor=jnp.zeros(empty_shape, dtype=jnp.float32),
        )
    flat = tuple(profile for team in teams for profile in team)
    speed, height, reach, control, endurance = _realize_abilities(
        flat,
        key=key,
        sampling=sampling,
        random_event=RandomEvent.BENCH_ROSTER_SAMPLING,
    )
    valid = np.zeros((2, width), dtype=np.bool_)
    player_id = np.full((2, width), -1, dtype=np.int32)
    goalkeeper = np.zeros((2, width), dtype=np.bool_)
    physical = np.zeros((2, width, 5), dtype=np.float32)
    offset = 0
    for team_index, team in enumerate(teams):
        count = len(team)
        valid[team_index, :count] = True
        player_id[team_index, :count] = [profile.player_id for profile in team]
        goalkeeper[team_index, :count] = [
            profile.is_goalkeeper is True for profile in team
        ]
        if count:
            physical[team_index, :count] = np.asarray(
                jnp.stack(
                    (
                        speed[offset : offset + count],
                        height[offset : offset + count],
                        reach[offset : offset + count],
                        control[offset : offset + count],
                        endurance[offset : offset + count],
                    ),
                    axis=-1,
                )
            )
        offset += count
    valid_array = jnp.asarray(valid)
    return BenchRoster(
        valid=valid_array,
        available=valid_array,
        player_id=jnp.asarray(player_id),
        is_goalkeeper=jnp.asarray(goalkeeper),
        max_speed=jnp.asarray(physical[..., 0]),
        height=jnp.asarray(physical[..., 1]),
        reach_height=jnp.asarray(physical[..., 2]),
        ball_control=jnp.asarray(physical[..., 3]),
        endurance_factor=jnp.asarray(physical[..., 4]),
    )


def realize_bench(
    team_0: Sequence[PlayerProfile] | None = None,
    team_1: Sequence[PlayerProfile] | None = None,
    *,
    key: jax.Array | None = None,
    generated_players_per_team: int = DEFAULT_BENCH_PLAYERS_PER_TEAM,
    sampling: RosterSampling | None = None,
) -> BenchRoster:
    """두 team의 등록 bench를 reset 경계에서 고정 폭으로 실현한다."""

    return _realize_bench(
        team_0,
        team_1,
        key=key,
        generated_players_per_team=generated_players_per_team,
        sampling=sampling,
    )


def _realize_roster(
    team_0: TeamRoster | None = None,
    team_1: TeamRoster | None = None,
    *,
    key: jax.Array | None = None,
    players_per_team: int = DEFAULT_PLAYERS_PER_TEAM,
    generated_candidates_per_team: int = DEFAULT_CANDIDATES_PER_TEAM,
    sampling: RosterSampling | None = None,
    formation_ids: tuple[int, int] | None = None,
    reserved_player_ids: Sequence[object] = (),
) -> RealizedRoster:
    """두 team의 선발과 누락된 ability를 한 reset 단위로 실현한다.

    완전히 제공된 입력에는 key가 없어도 된다. 무작위 선택이나 누락값 생성이 필요한
    경우에는 hidden global RNG 대신 호출자가 명시적인 Threefry key를 전달해야 한다.
    """
    players_per_team = _positive_integer("players_per_team", players_per_team)
    if players_per_team > MAX_PLAYERS_PER_TEAM:
        raise ValueError(f"players_per_team은 {MAX_PLAYERS_PER_TEAM} 이하여야 한다")
    generated_candidates_per_team = _positive_integer(
        "generated_candidates_per_team", generated_candidates_per_team
    )
    if generated_candidates_per_team < players_per_team:
        raise ValueError(
            "generated_candidates_per_team은 players_per_team 이상이어야 한다"
        )
    if key is not None:
        validate_prng_key(key)
    if formation_ids is not None and (
        len(formation_ids) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in formation_ids
        )
        or any(
            int(value) < 0 or int(value) >= FORMATION_COUNT for value in formation_ids
        )
    ):
        raise ValueError(
            f"formation_ids는 0..{FORMATION_COUNT - 1}의 정수 두 개여야 한다"
        )
    if formation_ids is not None and players_per_team != STANDARD_PLAYERS_PER_TEAM:
        raise ValueError("formation_ids는 players_per_team=11일 때만 지원한다")
    sampling = RosterSampling() if sampling is None else sampling
    if not isinstance(sampling, RosterSampling):
        raise TypeError("sampling은 RosterSampling이어야 한다")
    teams = (
        TeamRoster() if team_0 is None else team_0,
        TeamRoster() if team_1 is None else team_1,
    )
    if not all(isinstance(team, TeamRoster) for team in teams):
        raise TypeError("team_0과 team_1은 TeamRoster여야 한다")
    reserved_ids = _reserved_player_ids(reserved_player_ids)
    candidates = _normalize_candidates(
        teams,
        generated_candidates_per_team=generated_candidates_per_team,
        sampling=sampling,
        reserved_player_ids=reserved_ids,
    )

    selected_profiles: list[PlayerProfile] = []
    selected_indices: list[np.ndarray] = []
    goalkeeper_flags: list[np.ndarray] = []
    for team_index in range(2):
        indices = _select_team(
            candidates[team_index],
            teams[team_index].starting_indices,
            players_per_team=players_per_team,
            key=key,
            team_index=team_index,
        )
        selected = tuple(candidates[team_index][int(index)] for index in indices)
        selected_profiles.extend(selected)
        selected_indices.append(indices)
        goalkeeper_flags.append(
            _realize_goalkeepers(selected, key=key, team_index=team_index)
        )

    if reserved_ids.intersection(
        int(profile.player_id) for profile in selected_profiles
    ):
        raise ValueError(
            "on-pitch player_id는 registered bench의 명시 ID와 중복될 수 없다"
        )

    speed, height, reach, control, endurance = _realize_abilities(
        selected_profiles, key=key, sampling=sampling
    )
    formation_id, role, formation_anchor = realize_initial_formation(
        players_per_team,
        key=key,
        formation_id=(
            None if formation_ids is None else jnp.asarray(formation_ids, jnp.int32)
        ),
        goalkeeper=jnp.asarray(np.concatenate(goalkeeper_flags)),
    )
    roster = RosterState(
        team_id=jnp.asarray(np.repeat(np.arange(2, dtype=np.int32), players_per_team)),
        player_id=jnp.asarray(
            [p.player_id for p in selected_profiles], dtype=jnp.int32
        ),
        is_goalkeeper=jnp.asarray(np.concatenate(goalkeeper_flags)),
        max_speed=speed,
        reach_height=reach,
        height=height,
        ball_control=control,
        endurance_factor=endurance,
        role=role,
        formation_anchor=formation_anchor,
        formation_id=jnp.repeat(formation_id, players_per_team),
    )
    return RealizedRoster(
        roster=roster,
        candidate_index=jnp.asarray(np.concatenate(selected_indices), dtype=jnp.int32),
    )


def realize_roster(
    team_0: TeamRoster | None = None,
    team_1: TeamRoster | None = None,
    *,
    key: jax.Array | None = None,
    players_per_team: int = DEFAULT_PLAYERS_PER_TEAM,
    generated_candidates_per_team: int = DEFAULT_CANDIDATES_PER_TEAM,
    sampling: RosterSampling | None = None,
    formation_ids: tuple[int, int] | None = None,
) -> RealizedRoster:
    """두 team의 선발과 누락된 ability를 한 reset 단위로 실현한다.

    완전히 제공된 입력에는 key가 없어도 된다. 무작위 선택이나 누락값 생성이 필요한
    경우에는 hidden global RNG 대신 호출자가 명시적인 Threefry key를 전달해야 한다.
    """

    return _realize_roster(
        team_0,
        team_1,
        key=key,
        players_per_team=players_per_team,
        generated_candidates_per_team=generated_candidates_per_team,
        sampling=sampling,
        formation_ids=formation_ids,
    )


def _unselected_candidate_profiles(
    team_0: TeamRoster,
    team_1: TeamRoster,
    candidate_index: jax.Array,
    *,
    players_per_team: int,
    sampling: RosterSampling | None = None,
    reserved_player_ids: Sequence[object] = (),
) -> tuple[tuple[PlayerProfile, ...], tuple[PlayerProfile, ...]]:
    """명시 후보 roster에서 선발되지 않은 profile을 host reset 경계에 반환한다."""

    if not isinstance(team_0, TeamRoster) or not isinstance(team_1, TeamRoster):
        raise TypeError("team_0과 team_1은 TeamRoster여야 한다")
    players_per_team = _positive_integer("players_per_team", players_per_team)
    sampling = RosterSampling() if sampling is None else sampling
    candidates = _normalize_candidates(
        (team_0, team_1),
        generated_candidates_per_team=DEFAULT_CANDIDATES_PER_TEAM,
        sampling=sampling,
        reserved_player_ids=_reserved_player_ids(reserved_player_ids),
    )
    selected = np.asarray(candidate_index, dtype=np.int32)
    if selected.shape != (2 * players_per_team,):
        raise ValueError("candidate_index shape가 players_per_team과 맞지 않는다")
    remaining = []
    for team_index in range(2):
        start = team_index * players_per_team
        team_selected = {
            int(value) for value in selected[start : start + players_per_team]
        }
        remaining.append(
            tuple(
                profile
                for index, profile in enumerate(candidates[team_index])
                if index not in team_selected
            )
        )
    return remaining[0], remaining[1]


def unselected_candidate_profiles(
    team_0: TeamRoster,
    team_1: TeamRoster,
    candidate_index: jax.Array,
    *,
    players_per_team: int,
    sampling: RosterSampling | None = None,
) -> tuple[tuple[PlayerProfile, ...], tuple[PlayerProfile, ...]]:
    """명시 후보 roster에서 선발되지 않은 profile을 host reset 경계에 반환한다."""

    return _unselected_candidate_profiles(
        team_0,
        team_1,
        candidate_index,
        players_per_team=players_per_team,
        sampling=sampling,
    )


__all__ = [
    "DEFAULT_BENCH_PLAYERS_PER_TEAM",
    "DEFAULT_CANDIDATES_PER_TEAM",
    "DEFAULT_PLAYERS_PER_TEAM",
    "MAX_BENCH_PLAYERS_PER_TEAM",
    "BenchRoster",
    "PlayerProfile",
    "RealizedRoster",
    "RosterSampling",
    "TeamRoster",
    "realize_bench",
    "realize_roster",
    "unselected_candidate_profiles",
    "validate_player_capacity",
]
