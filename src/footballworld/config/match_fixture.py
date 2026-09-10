"""Strict host-only configuration for reproducible match initial conditions.

The fixture format describes facts known before kickoff.  It deliberately has
no time-indexed substitutions, tactical changes, set-piece choices, or manager
commands: those remain causal policy decisions after the match starts.

An omitted tactical plan, formation, starting lineup, or player ``abilities``
object means that the corresponding opening policy may choose or sample it.
When present, the complete value is authored and must be retained exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from footballworld.config.roster import PlayerProfile
from footballworld.policies.rule_based.tactical_plan import (
    TacticalPlan,
    canonical_tactical_plan,
)

MATCH_FIXTURE_SCHEMA = "footballworld.match-fixture/1"
MAX_FIXTURE_BYTES = 1_048_576
_INT32_MAX = 2_147_483_647
_ABILITY_FIELDS = frozenset(
    {
        "max_speed_mps",
        "height_m",
        "max_reach_height_m",
        "ball_control",
        "endurance_factor",
    }
)
_ROLE_NAMES = ("GK", "CB", "FB", "CM", "WM", "CF", "WF")
_ROLE_CODES = {name: code for code, name in enumerate(_ROLE_NAMES)}


@dataclass(frozen=True, slots=True)
class FixtureProvenance:
    """Authorship/source declaration; it is not a claim of measurement quality."""

    kind: Literal["recorded", "estimated", "synthetic"]
    source: str
    source_uri: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class FixturePlayerAbilities:
    """A complete, exact player-ability bundle authored by the fixture."""

    max_speed_mps: float
    height_m: float
    max_reach_height_m: float
    ball_control: float
    endurance_factor: float


@dataclass(frozen=True, slots=True)
class FixturePlayer:
    """One registered player and the provenance mode of their abilities."""

    player_id: int
    is_goalkeeper: bool
    preferred_roles: tuple[int, ...]
    abilities: FixturePlayerAbilities | None
    name: str | None = None

    @property
    def ability_mode(self) -> Literal["exact", "sampled"]:
        """Return whether abilities are fixed or must be realized by sampling."""

        return "exact" if self.abilities is not None else "sampled"

    def profile_reference(self) -> PlayerProfile:
        """Build the exact profile or the default mean used for later sampling.

        Callers must inspect :attr:`ability_mode`; a default-valued reference
        produced for ``sampled`` is not itself an authored exact profile.
        """

        values: dict[str, float] = {}
        if self.abilities is not None:
            values = {
                field: getattr(self.abilities, field) for field in _ABILITY_FIELDS
            }
        return PlayerProfile(
            player_id=self.player_id,
            is_goalkeeper=self.is_goalkeeper,
            preferred_roles=self.preferred_roles,
            **values,
        )


@dataclass(frozen=True, slots=True)
class FixtureFormation:
    """An exact formation selection, optionally with exact attacking-frame slots."""

    name: str
    positions: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class FixtureTeam:
    """One team; ``None`` fields are intentionally delegated to opening policy."""

    players: tuple[FixturePlayer, ...]
    name: str | None = None
    tactical_plan: TacticalPlan | None = None
    formation: FixtureFormation | None = None
    starting_player_ids: tuple[int, ...] | None = None

    @property
    def tactical_plan_mode(self) -> Literal["exact", "automatic"]:
        return "exact" if self.tactical_plan is not None else "automatic"

    @property
    def formation_mode(self) -> Literal["exact", "automatic"]:
        return "exact" if self.formation is not None else "automatic"

    @property
    def lineup_mode(self) -> Literal["exact", "automatic"]:
        return "exact" if self.starting_player_ids is not None else "automatic"


@dataclass(frozen=True, slots=True)
class MatchFixture:
    """Validated initial-condition declaration for exactly two teams."""

    schema: str
    match_id: str
    provenance: FixtureProvenance
    teams: tuple[FixtureTeam, FixtureTeam]
    seed: int | None = None
    kickoff_team: int | None = None


@dataclass(frozen=True, slots=True)
class LoadedMatchFixture:
    """Fixture plus hashes needed to reproduce and audit the loaded input."""

    fixture: MatchFixture
    source_path: Path
    source_sha256: str
    configuration_sha256: str


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, label: str, allowed: set[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown fields: {names}")
    return value


def _string(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


def _integer(value: Any, label: str, *, upper: int = _INT32_MAX) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a non-boolean integer")
    if value < 0 or value > upper:
        raise ValueError(f"{label} must be in [0, {upper}]")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} must be a non-boolean real number")
    represented = float(value)
    if not math.isfinite(represented):
        raise ValueError(f"{label} must be finite")
    return represented


def _parse_provenance(value: Any) -> FixtureProvenance:
    item = _mapping(
        value,
        "provenance",
        {"kind", "source", "source_uri", "notes"},
    )
    kind = _string(item.get("kind"), "provenance.kind")
    if kind not in ("recorded", "estimated", "synthetic"):
        raise ValueError("provenance.kind must be recorded, estimated, or synthetic")
    return FixtureProvenance(
        kind=kind,
        source=_string(item.get("source"), "provenance.source"),
        source_uri=_string(
            item.get("source_uri"), "provenance.source_uri", optional=True
        ),
        notes=_string(item.get("notes"), "provenance.notes", optional=True),
    )


def _parse_abilities(value: Any, label: str) -> FixturePlayerAbilities:
    item = _mapping(value, label, set(_ABILITY_FIELDS))
    missing = _ABILITY_FIELDS - set(item)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{label} is partial; missing exact fields: {names}")
    values = {field: _finite(item[field], f"{label}.{field}") for field in item}
    if values["max_speed_mps"] <= 0.0:
        raise ValueError(f"{label}.max_speed_mps must be positive")
    if values["height_m"] <= 0.0:
        raise ValueError(f"{label}.height_m must be positive")
    if values["max_reach_height_m"] < values["height_m"]:
        raise ValueError(f"{label}.max_reach_height_m must be at least height_m")
    if not 0.0 <= values["ball_control"] <= 1.0:
        raise ValueError(f"{label}.ball_control must be in [0, 1]")
    if values["endurance_factor"] <= 0.0:
        raise ValueError(f"{label}.endurance_factor must be positive")
    return FixturePlayerAbilities(**values)


def _parse_roles(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if type(value) is not list:
        raise TypeError(f"{label} must be an array")
    roles: list[int] = []
    for index, role in enumerate(value):
        if type(role) is str:
            try:
                code = _ROLE_CODES[role]
            except KeyError as error:
                raise ValueError(
                    f"{label}[{index}] must use one of {', '.join(_ROLE_NAMES)}"
                ) from error
        else:
            code = _integer(role, f"{label}[{index}]", upper=len(_ROLE_NAMES) - 1)
        roles.append(code)
    if len(set(roles)) != len(roles):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(roles)


def _parse_player(value: Any, label: str) -> FixturePlayer:
    item = _mapping(
        value,
        label,
        {"player_id", "name", "is_goalkeeper", "preferred_roles", "abilities"},
    )
    goalkeeper = item.get("is_goalkeeper", False)
    if type(goalkeeper) is not bool:
        raise TypeError(f"{label}.is_goalkeeper must be bool")
    abilities = (
        None
        if "abilities" not in item
        else _parse_abilities(item["abilities"], f"{label}.abilities")
    )
    return FixturePlayer(
        player_id=_integer(item.get("player_id"), f"{label}.player_id"),
        name=_string(item.get("name"), f"{label}.name", optional=True),
        is_goalkeeper=goalkeeper,
        preferred_roles=_parse_roles(
            item.get("preferred_roles"), f"{label}.preferred_roles"
        ),
        abilities=abilities,
    )


def _parse_formation(value: Any, label: str) -> FixtureFormation:
    item = _mapping(value, label, {"name", "positions"})
    positions_value = item.get("positions")
    positions: tuple[tuple[float, float], ...] | None = None
    if positions_value is not None:
        if type(positions_value) is not list or len(positions_value) != 11:
            raise ValueError(f"{label}.positions must have shape [11, 2]")
        parsed: list[tuple[float, float]] = []
        for index, position in enumerate(positions_value):
            if type(position) is not list or len(position) != 2:
                raise ValueError(f"{label}.positions must have shape [11, 2]")
            parsed.append(
                (
                    _finite(position[0], f"{label}.positions[{index}][0]"),
                    _finite(position[1], f"{label}.positions[{index}][1]"),
                )
            )
        positions = tuple(parsed)
    return FixtureFormation(
        name=_string(item.get("name"), f"{label}.name"), positions=positions
    )


def _parse_team(value: Any, label: str) -> FixtureTeam:
    item = _mapping(
        value,
        label,
        {"name", "tactical_plan", "formation", "starting_player_ids", "players"},
    )
    players_value = item.get("players")
    if type(players_value) is not list:
        raise TypeError(f"{label}.players must be an array")
    if len(players_value) < 11:
        raise ValueError(f"{label}.players must contain at least 11 players")
    players = tuple(
        _parse_player(player, f"{label}.players[{index}]")
        for index, player in enumerate(players_value)
    )
    ids = tuple(player.player_id for player in players)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}.player_id values must be unique")
    if not any(player.is_goalkeeper for player in players):
        raise ValueError(f"{label}.players must include a goalkeeper")

    starters_value = item.get("starting_player_ids")
    starters: tuple[int, ...] | None = None
    if starters_value is not None:
        if type(starters_value) is not list or len(starters_value) != 11:
            raise ValueError(f"{label}.starting_player_ids must contain 11 IDs")
        starters = tuple(
            _integer(player_id, f"{label}.starting_player_ids[{index}]")
            for index, player_id in enumerate(starters_value)
        )
        if len(set(starters)) != 11:
            raise ValueError(f"{label}.starting_player_ids must be unique")
        unknown = set(starters) - set(ids)
        if unknown:
            raise ValueError(
                f"{label}.starting_player_ids contains unregistered IDs: "
                + ", ".join(str(value) for value in sorted(unknown))
            )
        by_id = {player.player_id: player for player in players}
        if sum(by_id[player_id].is_goalkeeper for player_id in starters) != 1:
            raise ValueError(
                f"{label}.starting_player_ids must select exactly one goalkeeper"
            )

    plan_value = item.get("tactical_plan")
    plan = None if plan_value is None else canonical_tactical_plan(plan_value)
    formation_value = item.get("formation")
    formation = (
        None
        if formation_value is None
        else _parse_formation(formation_value, f"{label}.formation")
    )
    return FixtureTeam(
        name=_string(item.get("name"), f"{label}.name", optional=True),
        players=players,
        tactical_plan=plan,
        formation=formation,
        starting_player_ids=starters,
    )


def parse_match_fixture(value: Any) -> MatchFixture:
    """Validate a decoded JSON value and return immutable fixture inputs."""

    item = _mapping(
        value,
        "fixture",
        {"schema", "match_id", "provenance", "seed", "kickoff_team", "teams"},
    )
    schema = _string(item.get("schema"), "schema")
    if schema != MATCH_FIXTURE_SCHEMA:
        raise ValueError(
            f"schema must be exactly {MATCH_FIXTURE_SCHEMA!r}, got {schema!r}"
        )
    teams_value = item.get("teams")
    if type(teams_value) is not list or len(teams_value) != 2:
        raise ValueError("teams must be an array containing exactly two teams")
    teams = (
        _parse_team(teams_value[0], "teams[0]"),
        _parse_team(teams_value[1], "teams[1]"),
    )
    ids = [player.player_id for team in teams for player in team.players]
    if len(set(ids)) != len(ids):
        raise ValueError("player_id values must be unique across both teams")
    seed = None if item.get("seed") is None else _integer(item["seed"], "seed")
    kickoff = (
        None
        if item.get("kickoff_team") is None
        else _integer(item["kickoff_team"], "kickoff_team", upper=1)
    )
    return MatchFixture(
        schema=schema,
        match_id=_string(item.get("match_id"), "match_id"),
        provenance=_parse_provenance(item.get("provenance")),
        teams=teams,
        seed=seed,
        kickoff_team=kickoff,
    )


def load_match_fixture(path: str | Path) -> LoadedMatchFixture:
    """Load one bounded UTF-8 JSON fixture with duplicate-key detection."""

    source_path = Path(path).expanduser().resolve(strict=True)
    payload = source_path.read_bytes()
    if len(payload) > MAX_FIXTURE_BYTES:
        raise ValueError(f"fixture exceeds {MAX_FIXTURE_BYTES} bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("fixture must be UTF-8 JSON") from error
    try:
        decoded = json.loads(text, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid fixture JSON: {error.msg}") from error
    fixture = parse_match_fixture(decoded)
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return LoadedMatchFixture(
        fixture=fixture,
        source_path=source_path,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        configuration_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "MATCH_FIXTURE_SCHEMA",
    "FixtureFormation",
    "FixturePlayer",
    "FixturePlayerAbilities",
    "FixtureProvenance",
    "FixtureTeam",
    "LoadedMatchFixture",
    "MatchFixture",
    "load_match_fixture",
    "parse_match_fixture",
]
