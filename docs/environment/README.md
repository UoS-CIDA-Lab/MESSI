# Environment documentation

This directory documents behavior owned by
[`src/footballworld/environment/`](../../src/footballworld/environment/).
Environment pages define public causal boundaries and model-facing projections;
they do not duplicate physics, rules, rendering, or calibration evidence owned
elsewhere.

## Available contracts

| Topic | Documentation | Primary implementation |
| --- | --- | --- |
| Player action semantics and legality | [Action space](../action-space.md) | [`action.py`](../../src/footballworld/core/action.py), [`action.py`](../../src/footballworld/dynamics/action.py), [`action_legality.py`](../../src/footballworld/rules/action_legality.py) |
| Manager observation, commands, and timing | [Manager command](../manager-command.md) | [`management.py`](../../src/footballworld/environment/management.py), [`manager.py`](../../src/footballworld/policies/manager.py), [`managed.py`](../../src/footballworld/managed.py) |
| Player observations, visibility, and normalization boundary | [Observation](observation.md) | [`observation.py`](../../src/footballworld/environment/observation.py), [`normalization.py`](../../src/footballworld/environment/normalization.py) |

The next environment areas will be migrated here one at a time: episode and
clock semantics, action availability, initialization, roster sampling,
substitution, tactics, and manager-only observation. Until then, their existing
root-level documents remain the public entry points listed in the
[documentation map](../README.md).

## Boundary with neighboring subsystems

- Core state and action types belong under `docs/core/`.
- Physics and contact evolution belong under `docs/dynamics/`.
- Football-law decisions belong under `docs/rules/`.
- Replay presentation and serialized sidecars belong under `docs/rendering/`.
- Public coefficient classifications belong under
  [`docs/reproducibility/`](../reproducibility/); provider data, conversion,
  fitting artifacts, and internal validation receipts are not distributed.
