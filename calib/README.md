# Public replay fixture

The public tree intentionally contains only the host-side synthetic match
renderer used by the README examples.

`render_full_match.py` runs the shipped environment and reference rule policy
without loading provider data:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python calib/render_full_match.py \
  --output output/reference-match --maximum-steps 3000 --verify-video
```

Provider datasets, provider-specific conversion and fitting programs, derived
calibration artifacts, and internal semantic/benchmark suites are intentionally
not distributed. Runtime coefficient provenance is documented under
`docs/reproducibility/`.
