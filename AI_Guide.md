# FootballWorld AI guide

This checklist applies to AI reviewers and implementation agents working in
this repository.

1. Run commands from `/data/FootballWorld`. This repository uses a `src/`
   layout and is not guaranteed to be installed in the active interpreter.
   Prefix direct Python, pytest, and module invocations with `PYTHONPATH=src`,
   for example `PYTHONPATH=src python -m pytest -q tests/test_name.py`. When a
   suite imports the repository-root `research` namespace, use
   `PYTHONPATH=src:.`; a collection-only `ModuleNotFoundError: research` is an
   invocation error, not a product regression.
2. Select the JAX platform explicitly for reproducible CPU validation:
   `JAX_PLATFORMS=cpu PYTHONPATH=src ...`. On NVIDIA hosts use the explicit
   `JAX_PLATFORMS=cuda` backend; the generic `gpu` alias may probe an installed
   ROCm backend first. Do not assume that an enumerated GPU is usable by the
   active JAX build; verify the backend before promising GPU execution, and do
   not silently fall back when a GPU result is required.
3. Treat the worktree as shared and potentially dirty. Inspect `git status`
   before editing, preserve unrelated user or agent changes, and never use a
   destructive reset or checkout to clean the tree. Re-read overlapping files
   immediately before applying a patch.
4. Read `.validation_tests/POLICY_SESSION_NOTICE.md` before and after policy
   work. It is a cross-session bug ledger. Do not edit policy files merely to
   update a stale expected value; separate selection, execution, physics, and
   symmetry failures first.
5. Use focused tests before broad suites. A test command that fails during
   import because `footballworld` is missing is an invocation error; rerun it
   with `PYTHONPATH=src` and do not report it as a product regression.
6. Keep randomness comparisons controlled. Record the seed, JAX platform,
   configuration fingerprint, roster/formation inputs, and team slot. The
   tactical matrix reuses its parent `--seed` unless explicitly overridden.
   Reversing team 0/team 1 changes the assigned roster, initial direction, and
   restart slot; it is useful counterbalancing but is not an exact coordinate
   mirror or a new random seed. The default plan matrix fixes both demo teams
   to equal candidate ability bundles so identity-keyed sampling is not
   mistaken for a tactical effect; use
   `--no-matrix-equal-roster-abilities` only when independent episode sampling
   is itself part of the experiment.
7. `--matrix-workers N` launches up to N isolated match subprocesses. It is
   process concurrency, not a `vmap`/JAX batch of N environments. The tactical
   matrix defaults to 25 workers so all ordered cells can start together;
   callers on smaller hosts must lower it explicitly. Size it from measured
   per-child memory, compilation pressure, CPU capacity, and output I/O. The
   2026-09-11 CPU run measured roughly 2.9 GB RSS per active child, so the
   default may require about 73 GB before filesystem cache and report work.
8. For non-video experiments use `--report-only`; do not render video and then
   discard it. `--plan-matrix --matrix-legs 2` defaults to the complete 25-cell
   ordered space: twenty slot-counterbalanced distinct-plan games and five
   self-play controls. Use `--no-matrix-include-self-play` only for an explicitly
   incomplete 20-cell diagnostic.
9. Distinguish diagnostic from authoritative artifacts. `--allow-dirty` and
   bounded `--maximum-steps` outputs are diagnostic. Do not label a report
   authoritative unless its publication receipt, full-duration termination,
   source hashes, and clean/stable revision checks support that claim.
10. Keep host-only capture, HTML generation, league aggregation, file I/O, and
    variable-length diagnostics outside the lean JAX transition. Policy and
    environment hot paths should retain fixed shapes and bounded state.
11. For hot-path policy changes, inspect the generated graph/runtime impact as
    well as football semantics. Avoid duplicated pairwise geometry, repeated
    reductions, data-dependent Python branches, and diagnostics that enlarge
    the compiled step for rare cases.
12. Validate behavior at three levels as applicable: a small semantic or
    adversarial unit test, an eager/JIT equivalence or compilation check, and
    a seeded report-only match or paired matrix comparison. Do not infer policy
    quality from one match or silently replace baseline reports after tuning.
13. Run formatting/lint through the repository configuration after functional
    tests. Keep coefficient comments and reproducibility receipts explicit
    about whether values are measured, transferred, compatibility-derived, or
    design priors.
14. In the final review, list files changed, commands actually run, pass/fail
    counts, artifacts created, remaining uncertainties, and the exact behavior
    retained or changed with its rationale. Never claim a command passed
    if it was skipped, timed out, or failed before test collection.
15. Treat `footballworld.match-fixture/1` fields as field-local authority.
    Explicit complete abilities, tactical plans, formations, and starting XIs
    are fixed inputs; only omitted fields may be sampled or policy-selected.
    Do not add time-scripted manager actions to compensate for a rollout that
    diverges from a historical match.
