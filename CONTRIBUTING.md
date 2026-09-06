# Contributing to MESSI

MESSI accepts changes when football semantics, numerical behavior, and
performance consequences are independently inspectable. Keep each change small
enough that its rationale and evidence can be reviewed together.

## Development setup

Use Python 3.10 or newer:

```bash
python -m pip install -e '.[render]'
python -m compileall -q src/footballworld
```

Rendering is optional. A core contribution must continue to import without the
render extra or access to external datasets.

## Environment change contract

Before changing an environment feature, inspect the corresponding current
implementation, design notes, tests, and checked-in reproducibility evidence.
If the repository development contract names an external reference baseline,
inspect it as internal development evidence without exposing migration history
as a public API contract. In the change description:

1. Identify the current behavior being retained and why it remains sound.
2. Identify behavior being changed or rejected and give the correctness,
   realism, API, speed, memory, or compilation reason.
3. Distinguish measurements from transfer priors and design priors.

Keep host validation, rendering, dataset adapters, diagnostics, and release
tooling outside the lean JAX physics/rules transition unless the transition
itself requires the value. Prefer fixed-shape PyTrees, causal state, explicit
keys, and fail-closed traced inputs. Rare host-only behavior must not enlarge
the ordinary rollout graph.

Do not combine a numerical behavior change with an unrelated structural
refactor. A hot-path performance claim needs synchronized before/after
receipts, the intended CPU/GPU and batch conditions, and an output-equivalence
check. Update coefficient and reproducibility receipts only after functional
work is complete.

## Documentation and generated data

Public APIs, action or observation schemas, replay formats, and configuration
defaults must be documented in the same change. Preserve the provenance of
source data and cited research. Generated videos, tracking archives, local
environments, caches, and bulk calibration outputs do not belong in source
control.

## Before opening a change

Run the checks appropriate to the files touched. The minimum release-asset
checks are:

```bash
git diff --check
python -m compileall -q src/footballworld
python -m pip install build
python -m build
```

Install the resulting wheel in a clean temporary environment and import
`footballworld` from outside the checkout. Changes to physics, rules, policy,
state, actions, or observations also require focused semantic and adversarial
validation appropriate to their risk.

Security-sensitive findings must follow [SECURITY.md](SECURITY.md) and must not
be disclosed in a public issue before maintainers can assess them.

Unless explicitly stated otherwise, intentionally submitted contributions are
provided under the project's Apache-2.0 license.
