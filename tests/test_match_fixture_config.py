import json

import pytest

from footballworld.config.match_fixture import (
    MATCH_FIXTURE_SCHEMA,
    load_match_fixture,
    parse_match_fixture,
)
from footballworld.policies.rule_based.tactical_plan import TacticalPlan


def _player(player_id: int, *, goalkeeper: bool = False, exact: bool = False):
    player = {
        "player_id": player_id,
        "is_goalkeeper": goalkeeper,
        "preferred_roles": ["GK" if goalkeeper else "CM"],
    }
    if exact:
        player["abilities"] = {
            "max_speed_mps": 8.1,
            "height_m": 1.82,
            "max_reach_height_m": 2.71,
            "ball_control": 0.74,
            "endurance_factor": 1.08,
        }
    return player


def _team(offset: int):
    return {
        "name": f"Team {offset}",
        "players": [
            _player(offset, goalkeeper=True, exact=True),
            *[_player(offset + index) for index in range(1, 11)],
        ],
    }


def _fixture():
    return {
        "schema": MATCH_FIXTURE_SCHEMA,
        "match_id": "recorded-match-001",
        "provenance": {
            "kind": "recorded",
            "source": "official match sheet",
            "source_uri": "https://example.test/matches/001",
        },
        "seed": 29,
        "kickoff_team": 0,
        "teams": [_team(100), _team(200)],
    }


def test_explicit_values_are_exact_and_omitted_values_are_automatic():
    value = _fixture()
    value["teams"][0].update(
        {
            "tactical_plan": "gegenpress",
            "formation": {"name": "4-3-3"},
            "starting_player_ids": list(range(100, 111)),
        }
    )

    fixture = parse_match_fixture(value)
    fixed = fixture.teams[0]
    automatic = fixture.teams[1]

    assert fixed.tactical_plan is TacticalPlan.GEGENPRESS
    assert fixed.tactical_plan_mode == "exact"
    assert fixed.formation_mode == "exact"
    assert fixed.lineup_mode == "exact"
    assert fixed.players[0].ability_mode == "exact"
    assert fixed.players[0].profile_reference().max_speed_mps == 8.1
    assert fixed.players[1].ability_mode == "sampled"
    assert automatic.tactical_plan_mode == "automatic"
    assert automatic.formation_mode == "automatic"
    assert automatic.lineup_mode == "automatic"


def test_partial_ability_bundle_is_rejected_instead_of_silently_sampling():
    value = _fixture()
    value["teams"][0]["players"][1]["abilities"] = {"max_speed_mps": 8.0}

    with pytest.raises(ValueError, match="partial; missing exact fields"):
        parse_match_fixture(value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "8.0", True])
def test_ability_values_must_be_finite_non_boolean_reals(bad):
    value = _fixture()
    value["teams"][0]["players"][1]["abilities"] = {
        "max_speed_mps": bad,
        "height_m": 1.8,
        "max_reach_height_m": 2.7,
        "ball_control": 0.5,
        "endurance_factor": 1.0,
    }

    with pytest.raises((TypeError, ValueError), match="max_speed_mps"):
        parse_match_fixture(value)


def test_ids_must_be_unique_across_teams():
    value = _fixture()
    value["teams"][1]["players"][1]["player_id"] = 101

    with pytest.raises(ValueError, match="unique across both teams"):
        parse_match_fixture(value)


def test_exact_formation_positions_have_strict_shape():
    value = _fixture()
    value["teams"][0]["formation"] = {
        "name": "custom",
        "positions": [[0.0, 0.0]] * 10,
    }

    with pytest.raises(ValueError, match=r"shape \[11, 2\]"):
        parse_match_fixture(value)


def test_unknown_fields_and_scripted_timelines_are_rejected():
    value = _fixture()
    value["teams"][0]["manager_timeline"] = [{"minute": 70}]

    with pytest.raises(ValueError, match="unknown fields: manager_timeline"):
        parse_match_fixture(value)


def test_loader_detects_duplicate_keys_and_records_hashes(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_match_fixture(duplicate)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_fixture(), indent=2), encoding="utf-8")
    second.write_text(json.dumps(_fixture(), separators=(",", ":")), encoding="utf-8")

    loaded_first = load_match_fixture(first)
    loaded_second = load_match_fixture(second)
    assert loaded_first.source_sha256 != loaded_second.source_sha256
    assert loaded_first.configuration_sha256 == loaded_second.configuration_sha256
    assert loaded_first.source_path == first.resolve()
