# FootballWorld development contract

Before implementing or changing any environment feature, inspect the
corresponding implementation, tests, design notes, and reproducibility
evidence in `/data/SoccerWorld`.

For every change:

1. Record which SoccerWorld behavior is being inherited and why it is sound.
2. Record which behavior is being rejected or changed and the concrete
   correctness, realism, API, speed, memory, or compilation reason.
3. Preserve source-data provenance. Never present a compatibility prior or an
   unidentified coefficient as a measured football constant.
4. Keep host validation, rendering, dataset adapters, and diagnostics
   outside the lean JAX physics/rules step unless the transition itself needs
   the value.
5. Prefer fixed-shape PyTrees, causal state, and fail-closed traced inputs.
   Avoid graph growth that materially increases CPU/GPU compilation time for
   rare or host-only work.
6. Keep deployment changes independently inspectable. Add focused semantic
   and adversarial tests only in the dedicated validation phase; benchmark
   graph/runtime/memory there when a hot-path change is nontrivial.
7. Report the comparison and rationale to the user. Update coefficient and
   reproducibility receipts only after functional work is complete.

FootballWorld is not a line-for-line port. SoccerWorld is the mandatory
reference baseline; its advantages are retained and its weaknesses are design
targets.
