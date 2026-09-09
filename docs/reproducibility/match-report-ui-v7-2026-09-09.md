# Match-report interaction receipt — 2026-09-09

## Scope

This receipt covers host-side match analysis and the managed report-only
capture boundary: time-resolved ball density, smoothed team occupancy,
positioned shot progression, report-section order, and an empty-render-group
endpoint guard. No environment transition, policy, tracking schema, event
schema, coefficient, or JAX input was changed.

The report contracts advance from `footballworld.match-report/6` and
`footballworld.match-metrics/7` to schema 7 and metrics 8. The bump is required
because ball density gains incremental time-window receipts and shot
`preceding_events` changes from a mixed text list to a positioned,
tracking-derived attack-sequence start followed by same-attack exact actions.
The cumulative `live_seconds` arrays remain available for explicit consumer
migration.

## Mandatory SoccerWorld reference inspection

SoccerWorld has no match-report generation feature. The following files were
inspected only for its host-side rendering, artifact-receipt, source-clock, and
fail-closed conventions; they are not a SoccerWorld report implementation:

- `/data/SoccerWorld/src/soccerworld/demo/reporting.py`
- `/data/SoccerWorld/src/soccerworld/demo/artifact_receipts.py`
- `/data/SoccerWorld/src/soccerworld/data/capture.py`
- `/data/SoccerWorld/src/soccerworld/demo/rollout.py`
- `/data/SoccerWorld/tests/unit/demo/test_showcase.py`
- `/data/SoccerWorld/tests/unit/demo/test_artifact_receipts.py`
- `/data/SoccerWorld/tests/unit/rendering/test_render_contract.py`
- `/data/SoccerWorld/tests/contracts/public_api/test_native_command_capture.py`
- `/data/SoccerWorld/docs/architecture/commands-and-capture.md`
- `/data/SoccerWorld/docs/architecture/render-showcase.md`
- `/data/SoccerWorld/docs/reproducibility/baseline-20260829.md`

Inherited because it is sound: rendering and diagnostics remain host-only;
fixed-grid aggregation is bounded and reproducible; source clocks and receipts
remain authoritative; missing data fails closed; and the HTML stays
dependency-free. Disabled capture products and empty resampling results also
remain explicit, valid empty values instead of being indexed as present data.
These choices add no state leaf, JAX operation, compilation work, or rollout
cost.

Rejected or changed for concrete presentation/API reasons: hard occupancy
rectangles obscured spatial continuity; green turf changed translucent team
colors; a text hover table hid spatial progression and direction; and a
final-only ball density could not explain change over match time. The
schema-7 replacement used a bounded SVG Gaussian blur on neutral navy and
signed team-density layers. Schema 8 superseded those presentation details,
and schema 9 additionally decouples actual tracking routes from event nodes:
it server-renders independent red/blue radial density fields and draws the
pre-shot route from actual tracking ball positions even when a positioned event
node is unavailable. Sparse 30-second density buckets remain prefix-summed in
HTML. No display value is presented as a measured football constant or DFL
coefficient.

SoccerWorld has no managed streaming report-only render-group path. FootballWorld
keeps its useful fixed outer group/control-frame shape but rejects the video-only
assumption that a non-empty outer group has a final visual sample. The manager
boundary now replaces that endpoint only for a non-empty inner video group;
sidecar host-frame replacement is unchanged.

## Validation

- `PYTHONPATH=src pytest -q tests/test_report_only_capture.py tests/test_deployment_regressions.py`
  passed: 38 tests.
- The focused interaction subset passed 5 tests; the broader report/pass/shot
  subset passed 11 tests.
- `PYTHONPATH=src pytest -q tests/test_report_only_capture.py` passed 2 tests,
  including empty report-only groups and unchanged video endpoint replacement.
- `python -m py_compile` passed for `analysis/metrics.py`,
  `analysis/report.py`, and `rendering/capture.py`.
- A bounded real managed CLI smoke used seed 29, 3,000 control steps,
  256-step event chunks, `--report-only`, and `--match-report`. It crossed 7
  manager decisions without an indexing error, published 3,000 source rows and
  both report files, emitted no MP4, and recorded
  `terminal_basis=maximum_steps` rather than claiming a complete match.
  Generated local artifacts are intentionally not retained.
- A final provenance audit rejected the old fixed
  `successful_encoder_close_and_segment_count` receipt for report-only output:
  no encoder exists in that mode. `ReplaySidecarSpool.finalize` now requires an
  explicit, validated verification method. Video capture retains the encoder
  receipt and report-only capture records `not_requested`. The corrected full
  seed-29 run completed 57,642
  control frames with stable source hash, reports `video_generated=false` and
  `video_verification=not_requested`, and has the same event and tracking
  hashes as the pre-fix run. This is a provenance-only correction.
- The existing 90-minute replay correctly failed the authoritative default
  open because its published source authority is invalid. It was then opened
  only with `--allow-diagnostic`, retaining hash verification and the
  diagnostic warning. Generated local report artifacts are not retained.
- That report contains 192 density windows, 15,385 nonzero sparse density rows,
  192 occupancy windows, and 25 realized shots. Positioned routes are present
  for 22 shots (88%): all 22 contain a tracking-derived sequence-start node
  (20 regains and 2 restarts), with 4 retained exact applied-action nodes in
  total. The other 3 fail closed because no distinct earlier positioned start
  is available. Sparse density seconds sum to 5,095.8 s, equal to the cumulative
  receipt within serialized rounding.
- All executable inline scripts extracted from the standalone HTML pass
  `node --check`. Static inspection confirms the old
  `shot-history-bg`/`Previous causal events` table is absent, the Gaussian
  filter and bright team colors are present, both time bars are embedded, and
  `Match control and space` is the final content section.

## Residual risk

No installed headless browser is available in the workspace, so validation
covers generated SVG/HTML structure, JavaScript parsing, mobile viewport CSS,
touch focus markup, and real-data receipts rather than a pixel screenshot.
The source replay remains diagnostic because of its pre-existing authority
receipt; this report does not upgrade that provenance.

## Schema 9 / metrics 10 visibility and tracking correction

A script-blocking HTML viewer exposed that the schema-7 occupancy SVG had an
empty initial `spatial-cells` group: all colors were inserted only by
JavaScript. The signed `p0-p1` display could also erase valid same-cell
occupancy. Schema 8 server-renders the final cumulative frame and shows each
team's own normalized density with red/blue radial SVG fields. The existing
causal 30-second player-second receipts are inherited unchanged. Signed
subtraction, JavaScript-only visibility, Gaussian-filter dependence, and
`mix-blend-mode` are rejected for correctness and viewer-compatibility
reasons. The new overlap color is purple; no green channel is introduced.

The shot route now uses actual attack-normalized ball positions from tracking,
not straight lines between sparse events. Event nodes are independently
defined: sequence start plus at most six latest same-team exact deliberate
CONTROL contacts or kick-applied PASS/SHOT/CLEAR/CHALLENGE contacts. Consecutive
CONTROL contacts from the same player collapse to the latest contact. The
2,048 tracking-point bound is a UI/memory safety prior, not a measured football
constant; longer paths retain actual samples at a deterministic coarser stride.
Shot symbols are emitted in a lower SVG layer and all focus/hover routes in the
final overlay layer.

The mandatory SoccerWorld sources listed above were rechecked. Because
SoccerWorld has no report generator, no occupancy or shot-report behavior is
claimed as inherited. FootballWorld only retains the referenced host-only,
source-clock, artifact-receipt, and fail-closed engineering conventions. The
entire match-report presentation and metrics surface is FootballWorld-specific.
None of these changes touches JAX physics, rules, policy, or rollout
compilation.

On the final seed-29 replay, the regenerated report uses
`footballworld.match-report/9` and `footballworld.match-metrics/10`.
It contains 455 red and 450 blue server-rendered density blobs. Actual tracking
paths are present for 27/27 shots and contain 1,086 source points in total.
Twenty-four paths use the continuous live attacking sequence; the three direct
free-kick shots at 46:25, 72:52, and 75:38 each use 150 actual 10 Hz samples
from a bounded 15-second tracking-position lookback. Those fallback paths are
dotted, attach no earlier-phase event nodes, and are explicitly not labelled a
continuous physical trajectory because dead-ball placement may be present.
The old Regain-only event display falls from 20/27 shots to 2/27 after retaining
26 exact CONTROL-contact nodes. Static inspection confirms that the Team 1
symbol layer precedes its route-overlay layer.
