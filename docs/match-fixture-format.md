# Match fixture format

`footballworld.match-fixture/1` is a strict, host-only JSON format for match
facts known before kickoff. It does not encode a scripted manager timeline,
set-piece decisions, substitutions, or in-match tactical changes. Those remain
causal policy decisions based on the simulated state.

The selection contract is field-local:

- A supplied `tactical_plan` is exact; omission delegates plan selection to the
  roster-aware policy.
- A supplied `formation` is exact; omission delegates formation selection
  until after the tactical plan is known.
- A supplied `starting_player_ids` array is the exact starting XI; omission
  delegates lineup selection.
- A supplied player `abilities` object is exact and must contain all five
  ability fields. Omitting the whole object requests seeded sampling around
  FootballWorld defaults. Partial ability objects are rejected.

The two teams must contain at least eleven registered players each, player IDs
must be unique across the match, and an exact starting XI must contain exactly
one registered goalkeeper. Custom formation positions use eleven finite
attacking-frame `[x, y]` pairs. Pitch legality remains an environment-level
validation because it depends on the selected stadium and kickoff rules.

Every file declares provenance as `recorded`, `estimated`, or `synthetic`.
These labels describe authorship and do not turn an estimate into a measured
football constant. The loader returns both a raw-file SHA-256 and a canonical
configuration SHA-256 so formatting-only JSON changes can be distinguished
from semantic changes.

Load a fixture with:

```python
from footballworld.config.match_fixture import load_match_fixture

loaded = load_match_fixture("examples/fixtures/recorded_match.json")
fixture = loaded.fixture
assert fixture.teams[0].players[0].ability_mode == "exact"
assert fixture.teams[0].players[1].ability_mode == "sampled"
```

`examples/render_full_match.py --match-fixture PATH` integrates this boundary.
It preserves every `exact` value, invokes automatic selection or sampling only
for omitted fields, records both file and canonical hashes, and refuses
publication if the fixture changes during capture. Explicit player ability
bundles bypass roster sampling even when other players in the same team are
sampled from defaults.
