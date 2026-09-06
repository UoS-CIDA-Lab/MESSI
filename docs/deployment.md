# MESSI deployment guide

MESSI is distributed as the `messi-football` Python distribution and is
imported as `footballworld`. The runtime package contains the fixed-shape JAX
environment, public policy contracts, rule-based policies, management, and
optional replay rendering. Calibration programs and generated match artifacts
are not imported by the runtime.

## Supported runtime

- Python 3.10, 3.11, or 3.12
- JAX 0.4.38 or newer
- NumPy 1.26 or newer
- CPU, CUDA, and other accelerators supported by the installed JAX build

Install the core environment with:

```bash
python -m pip install messi-football
```

Install host rendering separately:

```bash
python -m pip install 'messi-football[render]'
```

MESSI does not select a JAX platform at import time. Backend selection and
accelerator-specific JAX installation belong to the deploying application.

## Runtime profiles

Choose the smallest transition that provides the required output.

| Workload | Interface | Intended output |
| --- | --- | --- |
| Training | `env.step` | Lean authoritative state transition |
| Evaluation | `env.step_with_events` | Exact fixed-shape causal event facts |
| Long rollout | `make_advance` | Final carry without per-frame trajectory |
| Training trajectory | `make_rollout` | Actions and ordinary step results |
| Replay capture | `make_event_rollout` | Actions and exact eventful results |
| Managed match | managed rollout interfaces | Low-frequency manager transactions |

Manager policy work, rendering, encoding, dataset conversion, and validation
remain outside the 80 Hz physics transition. This separation prevents optional
host work from enlarging the training executable.

## Release contract gate

CI runs the small committed `tests/test_release_contract_smoke.py` module for
the public default-timebase, exact-render-grid, and half-time configuration
contracts. It inherits the sound SoccerWorld practice of making release-facing
semantic contracts executable. FootballWorld does not copy the full
SoccerWorld golden, scenario, and soak inventory into the public distribution:
deeper adversarial and calibration checks remain in the ignored
`.validation_tests/` workspace. Both `tests/` and `.validation_tests/` are
explicitly pruned from wheel and sdist, and CI inspects both archives for that
boundary.

## Compilation strategy

The rollout factories return pure functions and do not call `jax.jit`. Build
the one-match function first, apply the desired match batching transform, and
then JIT the resulting callable.

```python
import jax

from footballworld import batch_rollout, make_rollout

one_match = make_rollout(env, policy, num_steps=900)
many_matches = jax.jit(batch_rollout(one_match))
```

`num_steps`, player count, environment configuration, output profile, and
batching strategy define an executable family. Production services should use
a small fixed set of chunk lengths and cache each compiled callable. Match
batching uses `lax.map` by default to keep one scalar match program and avoid
duplicating branch-heavy rule graphs.

## Determinism

Every stochastic transition accepts an explicit JAX key. Environment events
and shipped policy decisions are addressed by stable event, match, frame, team,
and player identities. Reusing the same immutable match key across rollout
chunks is supported because the absolute control tick participates in each
frame address.

A reproducible run records:

- MESSI version and source revision;
- schema versions and layout fingerprints;
- JAX, jaxlib, NumPy, Python, and backend versions;
- immutable environment configuration;
- roster and formation inputs;
- reset and match keys;
- rollout profile, chunk length, and batch strategy.

Use `footballworld.runtime.environment_fingerprint()` and the public
`schema_versions()` mapping when creating experiment metadata.

## Observation and action contracts

The authoritative action is `IntentAction`: one categorical intent and eight
bounded continuous controls. Player, manager, centralized state, action trace,
action receipt, and event schemas have independent semantic versions and
host-level fingerprints. Flattening preserves float, integer, and boolean
blocks separately so categorical and bit-mask values remain lossless.

Model-facing views are normalized with immutable configuration-derived scales
and are never clipped. SI-valued views and explicit restoration functions are
available when physical units are required. Player observations never expose
bench state; bench and substitution resources are manager-only information.
Use the environment's `flatten_observation`, `flatten_manager_observation`, and
`flatten_global_state` methods to retain those exact scale receipts. The
top-level equivalents require an explicit normalization context or a
precomputed environment-bound layout and reject context-free normalized
flattening.

## Replay publication

Authoritative replay capture requires:

- a clean source revision that remains unchanged during capture;
- environment termination after regulation and causal added time;
- zero physics event-budget exhaustion;
- exact source, event, tracking, and video frame agreement;
- artifact size and SHA-256 receipts;
- complete video decoding before every authoritative publication. The public
  full-match CLI enables this automatically when no diagnostic step budget or
  dirty-source override is active.

Diagnostic captures may opt into dirty-source or fixed-step execution, but
their completion manifest marks them non-authoritative. Do not publish those
artifacts as release evidence.

## Release gate

A release candidate is accepted only after the frozen source passes:

1. formatting, static lint, byte compilation, wheel and sdist builds;
2. isolated wheel installation and public import/reset/step smoke checks;
3. lean and eventful same-state transition equivalence;
4. repeated-seed and split-chunk determinism checks;
5. multi-seed 90-minute matches including causal added time;
6. full replay decoding and sidecar integrity audit;
7. CPU and GPU cold compilation, warm throughput, HLO, and memory receipts;
8. coefficient inventory and provenance reconciliation.

Passing these gates establishes software and replay integrity. It does not turn
documented design priors into measured football constants. Dataset-specific
calibration must retain its own estimand, source, split, and held-out receipt.
