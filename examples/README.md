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

To compare every tactical policy against every other policy in isolated
processes, use the plan matrix mode:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python examples/render_full_match.py \
  --plan-matrix --matrix-legs 2 \
  --matrix-workers 2 --matrix-platform cpu \
  --output output/tactical-plan-matrix --report-only --allow-dirty \
  --maximum-steps 600
```

The default two legs reverse the team slots for all ten unordered
distinct-policy pairings and include one diagonal match per policy, producing
the complete 5x5 ordered set of twenty-five matches.
`--no-matrix-include-self-play` explicitly requests an incomplete twenty-cell
diagnostic.
`--matrix-workers` bounds concurrent child processes; each child retains its
own stdout, stderr, replay, match report, and `match-report-status.json`. The
root `matrix-summary.json` lists every resolved plan orientation and returns a
failed status if any child fails. A successful run also writes
`multi-report/report.json` and `multi-report/report.html`, including league
standings that exclude self-play from points. Omit `--maximum-steps`,
`--report-only`, and `--allow-dirty` for authoritative full-match captures from
a clean revision.

Provider datasets, provider-specific conversion and fitting programs, derived
calibration artifacts, and internal semantic/benchmark suites are intentionally
not distributed. Runtime coefficient provenance is documented under
`docs/reproducibility/`.
