# Changelog

All notable changes to MESSI are recorded here. Versions follow semantic versioning; the
`StepCommand` → `TransitionResult` boundary is the public API we intend to keep stable.

## 0.1.0 — 2026-09-07

First tagged release. The project is renamed from **SoccerWorld** to
**MESSI: A Multi-Agent Environment for Soccer Simulation and Intelligence**. The Python import
name `soccerworld` is unchanged; `messi` is provided as an alias package and the distribution is
published as `messi-env`.

### Known behaviours to be aware of

- The rule-based team stops collecting a stationary loose ball once it is inside the unattended-ball
  radius; downstream users may want to lower `unattended_ball_radius` / `possession_release_radius`
  (the CIDA Lab training stack uses 1.5 m) or nudge a static loose ball.
- `kick_gate` (action dim 0) fires only above 0.5; learned policies trained by regressing 0/1 gate
  labels typically need a decision threshold at evaluation time.

### Added

- Installable `src/soccerworld` package with a lazy top-level import boundary.
- Installed `soccerworld-demo` command for the packaged single-match demo.
- Fixed-shape `StepCommand` control for actions, substitutions, formations, and set-piece takers.
- Causal `TransitionRecord` capture profiles for reinforcement and imitation learning.
- Atomic numeric-only NPZ transition shards with strict JSON metadata.
- Semantic test layout, focused/profile/changed selection, and frozen-source conformance receipts.
- Exact CPU/GPU rollout benchmark tooling and machine-readable performance receipts.
- Exact rule-policy output digests in action-only and decision-trace benchmark receipts.
- Fresh-process cumulative phase profiles with exact final-state equivalence checks.
- A fail-closed provenance manifest for all 462 initialized numeric config defaults.
- A reproducible private-input contextual card fit, aggregate validation receipt, and observed
  none/yellow/direct-red reconstruction outcome.
- Variable asymmetric roster and per-agent ability contracts, with inferred asymmetric benches and
  independently configurable substitution budgets (default 5).
- An isolated paper-experiment workspace with one-way public-API and packaging contracts.

### Changed

- Accepted the JAX-native `StepCommand` to `TransitionResult` boundary as the canonical public
  control API; ecosystem-specific interfaces remain non-authoritative adapters.
- The public demo now uses the native command API and a single-match scalar `lax.scan` path.
- The rendering extra now installs the matplotlib dependency required by rich rendering.
- Batched training rollouts preserve per-match scalar control flow by default; dense `vmap` remains
  an explicit measured option.
- Same-team carrier distance ties now resolve to one stable slot, preventing contradictory
  teammate kick decisions while preserving one eligible contest actor per team.
- Dribble recontacts now require observable ball-player relative motion, removing repeated
  low-speed re-kicks without adding hidden policy state or timers.
- Active non-taker players using measured corner structures now receive deterministic, legal,
  same-team-separated targets instead of overlapping within one player diameter.
- Final rule-policy actions are projected to the exact public action Box after float32 assembly.
- Public manager capture limits use canonical scalar `int32` arrays in eager, JIT, and scan records.
- Profile-specific first-run test scheduling hints reflect the current direct-State and full-matrix
  execution costs.
- Rule-policy tracing no longer constructs an unused all-player distance tensor; compiled actions
  and traces are unchanged.
- Calibration and benchmark tools require explicit external roots instead of private workspace
  defaults.
- Cards on declared contact fouls now use data-fitted offender field progress and elapsed-match
  odds; direct red remains an explicitly documented aggregate due to its small sample.
- Corner-to-live domain projection is now marked at the causal environment-write boundary for
  imitation capture, while ordinary live movement clamps remain learner-visible.
- Rule-policy roles now come from formation-relative line and width geometry instead of absolute
  pitch thresholds; the three superseded no-op tuning fields were removed.

### Removed

- Runtime `sys.path` mutation, unconditional trainer imports, temporary command bridges, and the
  legacy training collection surface.
