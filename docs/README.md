# Documentation map

FootballWorld documentation follows the subsystem that owns each public
contract. The project `README.md` is the installation and first-run entry
point; this page routes readers to implemented environment and policy behavior.

| Area | Entry point | Implementation owner |
| --- | --- | --- |
| Environment observations | [`environment/`](environment/) | `src/footballworld/environment/` |
| Player actions | [`action-space.md`](action-space.md) | `src/footballworld/core/`, `dynamics/`, `rules/` |
| Model projection | [`model-output.md`](model-output.md) | `src/footballworld/environment/normalization.py` |
| Match clock | [`match-clock.md`](match-clock.md) | `src/footballworld/environment/episode.py` |
| Rollout and policy state | [`rollout.md`](rollout.md) | `src/footballworld/rollout.py`, `policies/` |
| Manager commands | [`manager-command.md`](manager-command.md) | `src/footballworld/environment/management.py` |
| Rendering | [`rendering.md`](rendering.md) | `src/footballworld/rendering/` |
| Deployment | [`deployment.md`](deployment.md) | package and CI boundaries |
| Coefficient provenance | [`reproducibility/`](reproducibility/) | runtime receipts and evidence classification |

Calibration claims are kept separate from API documentation: a numerical or
tactical design prior is never presented as a measured football constant.
Rendering, validation, dataset handling, and diagnostics stay outside the lean
JAX transition unless the transition itself requires the value.

Training algorithms and provider-data conversion are not shipped in this
branch. Additional package-aligned documentation homes are planned and will be
linked here only when their corresponding implementation is public.
