# Match report pipeline

FootballWorld can turn one atomically published replay into a machine-readable
metric report and an interactive, chart-led HTML report. The analyzer is
host-only: it does not enter a JAX transition, alter a policy, or change match
physics. The HTML is metrics-only: it neither renders, links, nor embeds the
replay MP4.

```bash
PYTHONPATH=src python -m footballworld.analysis output/reference-match
```

An aggregate policy guideline can be overlaid without copying it into tracking,
event, or metadata:

```bash
PYTHONPATH=src python -m footballworld.analysis output/reference-match \
  --policy-reference /path/to/policy-guideline.json
```

The command verifies `completion.json`, the four artifact receipts, the event
frame count, and every tracking chunk before analysis. It writes
`report/report.json` and `report/report.html`. The wide-screen dashboard packs
shot/context views into a three-column row. The match timeline and full metric
tables remain available in one collapsed detail section. `Match control and
space` is the final content section so exploratory event, passing, workload,
and detail views precede the dense match-level spatial summary.

The report uses a cumulative controlled-possession line, a time-weighted
two-dimensional live-ball density map with longitudinal and lateral marginals
and a cumulative 30-second time bar, an interactive cumulative-from-kickoff
team spatial-occupancy map sampled at 30-second endpoints, attack-normalized
realized pass and shot maps, a player-speed histogram, compact exact-event
counts, team comparison bars, and player workload bars. The occupancy view
renders each team's normalized density as server-visible bright red and blue
fields over a neutral dark pitch; it does not blend either team with turf
green. When a policy reference is supplied, current-match bars appear beside
target and prior-policy markers.

```bash
PYTHONPATH=src python -m footballworld.analysis output/smoke-match \
  --allow-diagnostic
```

The report then labels itself as diagnostic and all metrics cover only the
captured interval. `--skip-hash-verification` is available for exploratory
work, but records a data-quality warning in both products.

## Multi-match tactical matrix report

A completed tactical matrix can be aggregated from its already verified
single-match reports without reopening the tracking arrays:

    PYTHONPATH=src python -m footballworld.analysis.multi_cli \
      output/policy-matrix/matrix-summary.json

The footballworld-matrix-report installed command is equivalent. It writes
multi-report/report.json and multi-report/report.html; the HTML links to every
single-match report and adds policy aggregates, a team-slot diagnostic, and
league standings. A successful matrix render invokes this writer automatically.

League standings use only distinct-policy fixtures. A win is three points and
a draw is one; ties are broken by goal difference, goals scored, then stable
policy name. Self-play stays in the matrix as a symmetry and team-slot
diagnostic but is excluded from points because one policy cannot gain a
comparative ranking advantage by playing itself. These are explicit tournament
reporting rules, not measured football coefficients.

The aggregator fails closed on missing artifacts, duplicate ordered cells,
partial matches, bounded rollouts, exhausted event budgets, unstable capture
source, or unverified single-match report hashes. An intentionally incomplete
ordered cell set remains usable only as a visibly incomplete diagnostic: both
JSON quality and the HTML cell count expose the omission. A full-duration report
produced from a dirty worktree likewise remains diagnostic rather than being
promoted to authoritative.

### FootballWorld matrix design

FootballWorld runs controlled matrix cells in fresh processes, retains
per-cell evidence, and performs multi-report and policy-league aggregation only
on immutable single-match outputs. JSON and HTML are built in one staging
directory and become visible through one directory rename, so readers cannot
observe a mixed report generation. Rankings, standings, report links, and
aggregation state stay outside the JAX transition:
they do not affect physics and would enlarge the rollout graph for presentation
work.

## Input ownership

- Tracking owns continuous source state: clock, ball and player kinematics,
  identity and slot generation, active/on-pitch state, score, possession, and
  restart state.
- Event owns confirmed discrete facts and submitted action receipts: exact
  contacts, fouls, boundaries, substitutions, acting goalkeeper changes,
  formation changes, set-piece taker changes, and sparse action rows.
- Replay metadata owns the static match manifest, stadium dimensions,
  timebase, render settings, and provenance.
- The report owns derived values only. It never writes derived values back to
  tracking, event, or metadata.

`footballworld.events/15` and `footballworld.contact-actions/3` add
`intended_receiver_player_id` only to retained `PASS` action rows. The source
is the rule policy's capture-only action receipt at that frame, covering
open-play, restart, and goalkeeper distribution decisions. `null` means that
the policy does not expose a receiver or no valid receiver was selected. It is
an intent receipt, not proof that the kick occurred or the pass was completed.

## Metric contract

`report.json` uses `footballworld.match-report/9` and
`footballworld.match-metrics/11`; every receipt names a metric ID, version,
unit, definition, and quality class.

The schema bump is intentional: realized-shot rows now separate the actual
tracking ball path from positioned event nodes. The path contains only
attack-normalized tracking samples from the sequence start through the frame
strictly before the shot. Nodes retain the start plus at most six latest exact
same-team deliberate CONTROL contacts or kick-applied PASS, SHOT, CLEAR, and
CHALLENGE contacts. Consecutive CONTROL contacts by one player collapse to the
latest contact. No intermediate position is inferred.

- Player distance sums adjacent positions only while player ID, slot
  generation, team, active state, and on-pitch state remain continuous.
- Maximum speed is the largest tracking velocity norm while active and
  on-pitch.
- Controlled possession is accumulated from the beginning of the capture and
  sampled every 30 seconds; loose-ball and dead-ball time are excluded.
- Penalty-area entries are live-ball attacking-direction crossings across
  adjacent frames while the same team retains possession.
- Team comparison uses confirmed goals, corners, fouls, and offsides alongside
  realized shots, shots on target, and same-team next-contact pass-receipt
  proxies.
- Player workload identities show entered/left substitution markers and the
  public match clock when an exact substitution receipt exists.
- Submitted pass and shot counts describe policy action receipts. Exact event
  counts describe the physics/rules event ledger. The two are deliberately not
  conflated.
- Policy-alignment pass and shot cadence count realized deliberate open-play
  contacts, divided by two teams times FootballWorld live minutes.
- Applied pass direction is classified from the submitted `force_to_ball`
  vector only after an exact deliberate-contact event confirms that the kick
  was applied. This is the DFL-facing direction metric. The pass is backward
  when its attack-normalized cosine is at most -0.34, lateral below 0.34, and
  forward otherwise.
- Receipt geometry separately joins each realized open-play pass to the next contact by a
  different actor. A boundary at or before that contact terminates the sequence.
  Repeated contacts by the passer are skipped. It must not be interpreted as
  the applied kick direction.
- Pass receipts are stratified by source half, pitch third, and applied
  direction. Intended-receiver identity matches are reported separately from
  any-teammate next contacts, so a teammate recovery cannot masquerade as the
  planned pass succeeding.
- Same-team next contact is reported as a receipt proxy. It is not renamed as
  provider pass completion.
- The two-dimensional ball density assigns each preceding live tracking
  interval to an approximately 2 m by 2 m absolute-pitch cell. The grid records
  seconds, not duplicated frame counts, and reports live time outside the pitch
  separately instead of clamping it into an edge cell. Schema 7 additionally
  stores deterministic, incremental 30-second sparse buckets with their exact
  clock/tick endpoints and in-pitch/excluded seconds. The standalone report
  prefix-sums those buckets, so its play button and time bar show density from
  kickoff through the selected time without retaining tracking rows in HTML.
- Team spatial occupancy assigns active, on-pitch player-seconds from preceding
  live-ball intervals to approximately 4 m by 4 m cells.
  The sparse receipt stores incremental 30-second buckets, while the report
  prefix-sums them so every selected time shows occupancy accumulated from
  kickoff through that time; the 30-second values are storage increments, never
  a rolling display window. Both teams share the frame in which Team 0 attacks
  positive x. Each occupied cell is rendered exactly once: its hue compares the
  teams' separately normalized match-to-date densities on a bright
  red-purple-blue scale, while its opacity follows their combined density.
  This prevents a later blue layer from hiding valid Team 0 occupancy. The final
  cumulative frame is server-rendered into the SVG, so script-blocking viewers
  still show the data. JavaScript progressively updates the same layer when
  available. It is descriptive occupancy, not inferred territorial control,
  possession probability, or dominance.
- Realized pass-map rows begin only at exact deliberate open-play PASS contacts
  whose kick was applied. Their solid endpoint is the next contact by a
  different actor before a boundary; it remains a receipt proxy. The dashed
  target is the intended registered receiver's tracking position immediately
  before the pass transition. Both teams are reflected into an attack-toward-positive-x frame.
- Realized shot-map rows begin only at exact deliberate SHOT contacts whose kick
  was applied, including restart shots, and use the exact source-contact
  position. A shooting-team goal is a goal; a non-goal is on target only when
  the next opposing controlled contact carries FootballWorld's physics-owned
  deliberate-save Law 11 effect. Passive deflections are followed until the
  next controlled contact, while an opposing deflection resolves a non-goal as
  a block. Woodwork, blocks, other contacts, and non-goal boundaries are not on
  target. A capture that ends before any such fact remains `unresolved` rather
  than being coerced to off target. Team shot-on-target totals include goals and
  saved-on-target shots.
  Hover/focus draws the actual tracking ball path and then places exact event
  nodes over it. Symbol markers are in a lower SVG layer and all interactive
  route overlays are emitted last, so Team 0 and Team 1 routes cannot be hidden
  behind later shot points.
- Each realized-shot interaction separates the tracking path from its event
  nodes. The line uses actual 10 Hz tracking ball positions through the shot
  contact. It starts at the continuous attacking-sequence boundary when that
  boundary is known; otherwise a dotted fallback uses the preceding 15 seconds
  of actual tracking position history without inventing a sequence start. The
  fallback may include dead-ball placement and therefore is not labelled a
  continuous physical trajectory; it does not attach earlier-phase event
  nodes. The hover route is split into chronological SVG segments whose opacity
  increases from 0.2 at the oldest segment to 1.0 at the shot, preserving the
  temporal direction even when a path loops. It retains actual samples up to
  a 2,048-point safety bound; longer paths are deterministically thinned
  without interpolation. The overlay draws at most seven prior positioned
  nodes: the sequence start, when known, plus the latest six exact deliberate
  CONTROL or kick-applied PASS/SHOT/CLEAR/CHALLENGE contacts. Consecutive
  controls by the same player are collapsed to the latest contact; the shot
  point is separate from this limit. The seven-node and 15-second limits are UI
  and memory design priors, not measured football constants. Invalid,
  same-shot-tick, or zero-length starts, opponent actions, actions before the
  sequence, locationless actions, and same-tick exact facts fail closed and are
  omitted. This replaces the old hover table, which hid the spatial development
  and direction the report is meant to explain.
- Pre-shot context is a deterministic report heuristic, not a measured football
  constant. Categories are mutually exclusive in this priority: direct penalty
  kick, direct free kick, other restart attack (direct or within 10 s of the
  restart sequence), counterattack (opponent
  regain, at most 12 s, at least 20 m forward progress), quick after regain (at
  most 5 s), sustained buildup (at least 12 s), other open-play buildup, then
  unclassified. Loose-ball frames do not terminate the last controlling team's
  live attacking sequence. The chart compares each category's share of all
  realized shots with its share of goals and exposes conversion in hover text.

## FootballWorld report design

FootballWorld keeps coefficient and action diagnostics host-side, uses fixed
histogram edges with explicit counts and provenance, and preserves causal
shot-flight semantics: a goal is checked before the next controlled
touch, passive deflections do not masquerade as receptions, opposing
deflections resolve non-goals as blocks, and only goal or saved outcomes count
as shots on target. FootballWorld uses exact contact, boundary, woodwork, and Law 11 receipts
instead of render-time touch heuristics because those facts are already
preserved by the published replay.

Published-match pass maps, two-dimensional ball/team occupancy, and the
buildup/regain/counterattack shot-context taxonomy are host-side analysis
features. Pre-shot categories are visibly labelled report heuristics rather
than measured football constants.

FootballWorld therefore derives both views only from verified replay sidecars.
It rejects adding density, pass outcomes, or visualization state to the JAX
transition, tracking schema, event schema, or metadata. This keeps the source
facts lean and makes every displayed cell and path reproducible in post-process.

This UI revision keeps host rendering separate from simulation,
source-owned clocks, dependency-free artifact discipline, and explicit empty
states: all smoothing, prefix sums, and interactions remain in the standalone
HTML and cannot enlarge a JAX graph or alter a match. It rejects three existing
FootballWorld presentation choices for concrete report-legibility reasons.
Hard occupancy rectangles are replaced by bounded SVG radial fields with a
server-rendered final-frame fallback; turf green is replaced only in that chart
by a neutral navy field so bright `#ff365f`/`#2787ff` team densities remain
visible; and the shot-history text table is replaced by an actual tracking
ball path plus exact contact-position nodes on the pitch. The final-only ball
density is changed to incremental fixed windows
so it can use the same causal cumulative time interaction as occupancy. These
are presentation and receipt-schema decisions, not measured football constants
or compatibility priors.

## External policy reference

The optional reference must use `footballworld.policy-reference/1`. Each metric
has a stable ID matching `report.policy_alignment.metrics`, a target value,
unit, comparability class, and optional prior-policy baseline. Accepted
comparability classes are `closest_comparable`, `proxy`, and
`diagnostic_neighbor`.

The HTML deliberately does not combine unlike metrics into one fidelity score.
It presents each delta independently and preserves source cautions in
`report.json`. A single match, especially a partial capture, remains
descriptive; coefficient decisions require held-out multi-seed intervals.

The initial report does not calculate xG, xT, PPDA, policy utility/logits,
reward, pass completion, or role-fit scores. Those require an explicit model or
a separately versioned inference contract; naming them from weaker evidence
would overstate what the raw replay proves.

## Producing a replay

The public reproducible entry point is:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python examples/render_full_match.py \
  --output output/reference-match --maximum-steps 3000 --verify-video
```

Add `--match-report --allow-diagnostic-report` to that smoke command to create
the metrics-only HTML and JSON immediately after publication. For a clean,
full-duration authoritative render, use `--match-report` without the diagnostic
switch.

Omit `--maximum-steps` and use a clean source tree for an authoritative full
match. The former `calib/` path is a private local coefficient workspace and
is not a public match-generation entry point.
