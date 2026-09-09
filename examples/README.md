# Public full-match rendering example

The public tree intentionally contains only the host-side synthetic match
renderer used by the README examples.

`render_full_match.py` runs the shipped environment and reference rule policy
without loading provider data:

```bash
python demo_match.py --output output/tactical-demo \
  --team-0-plan catenaccio --team-1-plan random --seed 29
```

The root `demo_match.py` delegates to this renderer. Its `--team-0-plan` and
`--team-1-plan` options independently accept `salida_lavolpiana`,
`juego_de_posicion`, `gegenpress`, `catenaccio`, `zona_mista`, or `random`.
Both default to `juego_de_posicion`. `random` is resolved once per team from a
dedicated key derived from `--seed` and the resolved plan is fixed for the
entire match. The replay receipt retains both resolved plans and the
`RulePolicyConfig` fingerprint.

The canonical renderer can also be invoked directly:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python examples/render_full_match.py \
  --output output/reference-match --maximum-steps 3000 --verify-video
```

For a fast diagnostics pass with exact tracking, events, the interactive
HTML report, and no MP4 rendering:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python examples/render_full_match.py \
  --output output/report-only-match --report-only --allow-dirty \
  --match-report --allow-diagnostic-report
```

Provider datasets, provider-specific conversion and fitting programs, derived
calibration artifacts, and internal semantic/benchmark suites are intentionally
not distributed. Runtime coefficient provenance is documented under
`docs/reproducibility/`.
