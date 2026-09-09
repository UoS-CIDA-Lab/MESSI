# Replay and rendering contract

FootballWorld renders one selected match entirely on the host. The environment
rollout remains in JAX; one fixed event chunk is transferred at a time, exact
sidecars are spooled to disk, and video segments are encoded concurrently.
Nothing in this pipeline is added to the ordinary training `env.step` graph.

## Demo tactical plans

The root demo exposes side-specific reference-policy selection:

```bash
python demo_match.py --output output/tactical-demo \
  --team-0-plan salida_lavolpiana --team-1-plan random --seed 29
```

`--team-0-plan` and `--team-1-plan` independently accept
`salida_lavolpiana`, `juego_de_posicion`, `gegenpress`, `catenaccio`,
`zona_mista`, or `random`; each option defaults to `juego_de_posicion`.
`random` is resolved once for each requesting team from a dedicated PRNG key
derived from `--seed`. The two resolved plans are then fixed for the whole
match. Replay provenance records those resolved plan names and the SHA-256
fingerprint of the canonical `RulePolicyConfig`, so the effective policy input
can be audited without treating a requested `random` label as the realized
configuration.

FootballWorld inherits SoccerWorld's sound side-specific selection and seeded,
independent random resolution because those properties make comparisons
reproducible and prevent one side's request from determining the other's.
FootballWorld rejects SoccerWorld's continuous style-vector mechanism for this
surface: its rule policy defines five named, mechanism-specific tactical plans,
so the CLI selects among those explicit plans rather than interpolating an
unidentified vector whose values would not map to FootballWorld's tactical
branches. Plan resolution and receipt construction stay host-side and add no
state, branch, or operation to the JAX transition.

## Full matches and selected intervals

Use the managed entry point for a final match replay:

```python
from footballworld import make_managed_runner
from footballworld.rendering import render_managed_event_match


def main():
    # Build env, reset, management, roster, player_state, and match_key here.
    runner = make_managed_runner(env, chunk_steps=256)
    managed = runner.initialize(
        reset.rollout,
        reset.setup,
        management.squad,
        management.state,
        roster,
        player_state,
    )
    render_managed_event_match(
        runner,
        managed,
        management.squad,
        match_key,
        "output/full-match",
        workers=4,
    )


if __name__ == "__main__":
    main()
```

For `workers > 1`, the encoder intentionally uses multiprocessing
`spawn`, which is safe beside JAX but requires an importable script and the
main guard shown above. Notebook and interactive callers should use
`workers=1`; that path renders in the current process and does not spawn.

With no `maximum_steps`, capture continues to the environment's authoritative
`done` value. It does not assume that 90 regulation minutes are 81,000 wall
frames: dead-ball time extends each half. The final fixed chunk is a valid
prefix followed by absorbing cells. Managed capture transfers the fixed chunk
once, then slices only host arrays; only the prefix through the first terminal
frame is converted to `HostFrame`, written, and rendered. This avoids a new
JAX slice/broadcast executable for every rare manager-boundary prefix length.

`output_dir` must not already exist. One invocation writes every requested
window into a private sibling staging directory, validates every segment and
sidecar, writes `completion.json` last, and publishes the entire directory with
one same-filesystem rename. An exception removes the staging directory and
leaves no partial destination. A successful budget-limited smoke is explicitly
marked `complete=false`; only an authoritative environment termination is a
complete match. Concurrent attempts to reserve the same destination fail
closed.

Capture fingerprints the imported FootballWorld Python source before rollout
and verifies it both before and after sidecar/video finalization. The managed
entry point also accepts a host-only `publication_guard`; the full-match CLI
uses it to require an unchanged Git revision, complete porcelain state, and
fixture-script hash. A mismatch discards the private staging directory instead
of publishing a replay assembled from two source snapshots. These checks do
not enter `step`, `step_with_events`, or either JAX executable.

Authoritative `done` does not by itself prove that 90 regulation minutes were
completed: minimum-player abandonment, an invalid restart team, and the
wall-clock overflow fail-safe are also terminal. `completion.json` therefore
records the exact host-derived `terminal_basis` and a separate
`full_duration_complete` boolean. The long-match CLI returns success without a
step budget only for `terminal_basis=regulation_complete`. This classification
reuses final-state facts and does not add a leaf or operation to the JAX event
graph.

`maximum_steps` is a safety budget for a smoke run, not the full-match
termination rule. Rendering is kept at the environment control grid
(`every=1`) so an incomplete final decimation group cannot make MP4 timestamps
disagree with tracking and metadata.

Several wall-time intervals can be captured in one rollout without retaining
the full trajectory:

```python
windows = (
    ReplayWindow("opening", 0, 5 * 60),
    ReplayWindow("middle-first-half", 20 * 60, 25 * 60),
    ReplayWindow("halftime-band", 42 * 60, 52 * 60),
    ReplayWindow("late", 65 * 60, 70 * 60),
    ReplayWindow("finish", 85 * 60, None),
)
```

Window bounds use elapsed control time, not the broadcast clock. The broad
halftime interval is intentional because first-half added time moves the
second-half wall-time origin. Tracking rows retain `period`,
`display_clock_s`, `added_time_s`, and the authoritative half boundary for
precise post-run selection and audit.

The lower-level `render_event_match` is available for an unmanaged evaluation
trajectory. It also calls `step_with_events`, uses fixed chunks, trims the
terminal suffix, and never changes training `env.step`. It is not the final
match path when substitutions, formations, acting-goalkeeper recovery, or
set-piece-taker selection are enabled.
Exact-event capture accepts only a uniform subset of physics endpoints: the
render rate must be an integer multiple of the control rate and the resulting
samples per control frame must divide the physics decimation. With the default
80 Hz physics and 10 Hz control clocks, the exact rates are 10, 20, 40, and
80 fps. In particular, 30 or 60 fps fails closed instead of labelling the
non-uniform 25/37.5/37.5 ms endpoint gaps as a constant-rate video. SoccerWorld
soundly supports arbitrary rates for its explicitly approximate nearest-sample
resampling path; FootballWorld rejects that behavior here because this API
promises an exact event time axis.
The memory-resident `render_mp4` helper is likewise not a full-match entry
point: it accepts an already materialized trajectory and does not provide
terminal capture, manager boundaries, a liveness watchdog, or invocation-level
atomic publication. Callers that omit `env` must pass the trajectory's positive
`ball_radius_m`; when both are supplied they must agree within float32 input
precision. After a successful encoder close it writes `completion.json` last
and records exact sidecar/video counts in metadata. That receipt uses
`terminal_basis="render_only"` and `full_duration_complete=false`; it never
certifies a complete match. A failed render has no completion manifest.

## Manager boundary and compile separation

The managed event scan stops at the same boundary as `ManagedRunner` before a
three-second restart can progress. Its time-major valid mask is required to be
one prefix. The host applies the manager command, refreshes roster metadata
only if the authoritative transaction reports an identity change, applies
tactical anchors once, and resumes with the same immutable match key.

Only the fixed restart-identity boundary enters the player scan. Manager
parameters, learned recurrent memory, bench observations, command arrays, and
the substitution ledger enter a separately compiled rare-boundary executable.
The eventful scan is a distinct graph from both ordinary `env.step` and the
capture-free managed scan. Choosing it increases evaluation compile time and
device output memory, but cannot enlarge a training executable.

An opening formation is queried once before the first chunk. For a new match
constructed by the opening policy before rendering, its team-local index, name,
and realized 11-by-2 `selected_formation_layout` are match metadata. The
full-match CLI records them under
`user_metadata.user.long_run_fixture.selected_formation`, not as an in-match
event.

A mid-match checkpoint keeps its authored state because the environment opening
predicate makes that transaction a no-op. If an initial manager boundary
produces zero physics frames, its command is applied before frame zero; there is
no invented event frame. Any substitution, acting-goalkeeper, or requested
formation receipt at that zero-step boundary is stored under
`pre_frame_management` in the event header. Later substitution and role events
are attached to the post-command state at the same authoritative control tick
as the boundary frame. Requested formation receipts are sparse `formations`
rows on that boundary frame and contain only team, proposed layout index, and
the
environment-authoritative applied flag. No non-boundary frame carries them.

The action, exact frame events, and transition telemetry on such a boundary row
were produced before that manager transaction, whereas same-tick tracking is
the post-command state. If management changes a slot identity or goalkeeper
role, the event row therefore carries sparse `pre_management_identity`
overrides with `player_id`, `slot_generation`, team, and goalkeeper status.
Resolve an action or event slot against this map first and same-tick tracking
otherwise. This preserves stable actor identity without adding an identity leaf
to the compiled event transition.

Report-only capture preserves one outer render group per retained control frame,
but each inner group is intentionally empty because no visual sample is built.
After a manager transaction, the host control frame is still replaced for
tracking/event identity; the renderer endpoint is replaced only when the final
inner group actually contains a video sample. SoccerWorld's sound treatment of
disabled capture widths and empty resampling outputs as valid empty products is
inherited. SoccerWorld has no managed streaming report-only path, so
FootballWorld rejects the video-only assumption that every non-empty outer
control group contains a renderer sample. This check remains host-only and does
not alter capture shapes, JAX graphs, or video behavior.

A host-only constant-memory watchdog checks each returned chunk. An active
restart whose countdown has expired without a legal layout fails immediately.
A layout-ready restart that does not release is bounded by the configured
forced-release delay, converted once to control steps; kickoff uses its zero
delay boundary. This adds scalar host synchronization only and does not add a
leaf, branch, or calculation to any JAX executable.

Non-boundary frames carry canonical empty substitution and acting-goalkeeper
event trees. They are therefore known-empty, not missing. A valid goalkeeper
hold is not a manager interrupt; malformed holds and missing active
goalkeepers follow the existing management boundary and environment fail-safe
rules.

## Video semantics

The sole presentation path uses a lower 33-degree fixed oblique pinhole camera.
A tighter regulation-surface
crop increases the pitch's screen occupancy while alternating grass, cropped
sloped stands, regulation 3D goal frames and nets, the ball's real height, and
the retained top-down minimap preserve spatial context. Each player is one
team-coloured spherical marker; no separate head, torso, shoulder, or leg icon
is drawn. The aerial recovery countdown drives a bounded sine arc, so the
marker visibly rises and returns after a high-ball contact without requiring
historical frames. Its ground-anchored recovery ring and shrinking shadow
reinforce the jump without changing the player's physical position. The two
stamina bars inherit SoccerWorld's sound full-capacity dark rails, which keep
partly depleted bars legible against either grass stripe. FootballWorld rejects
SoccerWorld's stamina-dependent red gradient because red already identifies the
home side and foul cues, and changing every player's colour every frame adds
host work without adding state information. Brighter fixed green/cyan fills
retain the existing long/short identity, while wider depth-scaled rails and
larger separation survive distant-player antialiasing and H.264 chroma
subsampling. The four shared collections remain host-only and do not enter the
JAX state or transition graph.

The one default presentation standard is 1,920x1,080 at 150 dpi. Raising this
from the historical 1,280x720 default changes only host drawing, RGBA transfer,
and encoding cost. It does not add a state or event field and changes no JAX
trace, executable, HLO operation, or environment rollout result.

Each active player has a translucent white field-of-view fan centred on the
actual current `rotate(body_forward, gaze_yaw)` state, not the requested gaze
action. Under the default reach and ball configuration, the turf-level cue
begins 2.00 m from the player and ends at 3.00 m. The fan remains compact while
the independently scaled action rings retain their physical interference
meaning. A goalkeeper `CONTROL` ring may overlap the fan slightly because its
default maximum reach is 2.11 m. Bright
boundary rays and a separate body-direction needle remain omitted because they
compete with player and intent cues. One update-in-place `PolyCollection` owns
all fans after the vertices pass through the same projection as the players;
there are no per-player patches, trails, or camera changes.

When partial observation is enabled, the fan uses the environment's exact
horizontal observation aperture. Under the default global observation it is
only a gaze-direction cue and uses host-only `RenderStyle.gaze_cue_degrees`;
it must not be read as a visibility mask. This distinction, both angles, and
the resolved radii are recorded in render-settings `/3`.

Requested intent is shown independently from realized contact. Each ring is a
world-space circle projected onto the turf, so it foreshortens with the fixed
camera. Its radius comes from the configured horizontal reach plus the physical
ball radius: 1.21 m for ordinary `CONTROL`, `PASS`, `SHOT`, and `CLEAR`; 1.51 m
for `CHALLENGE`; and 2.11 m for goalkeeper `CONTROL` under the defaults. This is
a maximum horizontal interference envelope, not a promise that contact will
succeed; height, relative motion, recovery, possession, and rule gates remain
authoritative. Two shared update-in-place `LineCollection` objects draw the
dark underlay and coloured ring without constructing per-frame artists.

| Requested intent | Marker | Default radius |
| --- | --- | ---: |
| `MOVE` | no intent ring | - |
| `CONTROL` | cyan | 1.21 m; goalkeeper 2.11 m |
| `PASS` | amber | 1.21 m |
| `SHOT` | orange | 1.21 m |
| `CLEAR` | violet | 1.21 m |
| `CHALLENGE` | magenta | 1.51 m |

A compact bottom-left figure-space legend groups each label beside a ring glyph
using the same palette. The enlarged high-contrast ring is centred on the
player's ground shadow, not the screen-facing marker. The default 185-point-
squared marker area is a presentation choice independent of the collision body.
Intent rings read only `ActionTrace.requested_intent` for an executed action,
do not claim that contact occurred, and are not extended into later frames.

The host-only design retains SoccerWorld's sound use of persistent, turf-
projected artists. FootballWorld deliberately rejects a single fixed display
radius because its environment exposes distinct carry, challenge, goalkeeper,
and ball radii that can be configured independently.

There is no historical event feed or scrolling event window. One transient
adjudication banner may show for 2.5 seconds when an exact
`step_with_events` transition reports a goal, foul, offside, or committed
substitution. Deterministic sporting priority is goal, red-card foul, other
foul, then offside. A same-frame substitution is appended to that decision
rather than silently discarded; otherwise it receives its own `OUT → IN`
banner with team and exact registered player ids. Ordinary contacts, passes,
controls, period boundaries, and non-goal restarts stay out of the video. A
foul names the pre-management player identities and
one-based display slot labels for offender and victim, exact offence type,
source, severity, tactical effect, discipline, restart or advantage. Offside
names its exact team and actor without inventing the involvement branch. Full
substep detail remains in `event.json`.

Banner carry is computed on the host time axis before direct-render frames are
split among workers. Managed capture retains one previous-caption reference
while it flushes bounded chunks. Thus a chunk boundary cannot truncate the
2.5-second display; this adds no device state, JAX work, compilation, or rollout
cost. The direct convenience path already materializes source frames and adds
at most one immutable caption reference per source visual frame so an
`every > 1` sample cannot silently skip the originating transition.
For a clipped window, only the preceding `adjudication_seconds` worth of host
visual frames are inspected to restore a still-active banner. Event rows from
outside the selected window are not copied into that window's `event.json`.

The cached 4:30--4:40 comparison around source frame 4,094 contains the exact
foul banner for player 2007 (display slot 17) on player 1014 (display slot 6).
Both before and after clips contain 150 frames at 1,920x1,080 and 15 Hz. On the
same cached host frames, single-process rendering measured 44.088 fps before
and 42.960 fps after, a 2.56% render-only reduction; simulation and sidecar I/O
were outside both intervals. The generated comparison artifacts are
intentionally not retained in the repository; only this bounded host-specific
receipt is preserved.

## Sidecar schemas and lossless sparse events

Each managed/default output directory contains:

- `match.mp4`: H264 replay on the exact source time axis.
- `tracking.npz`: lossless fixed-dtype state records grouped into bounded
  NumPy chunks, one structured `.npy` member per rollout chunk.
- `event.json`: requested intent plus exact realized events.
- `metadata.json`: geometry, schemas, frame counts, clocks, and provenance.
- `completion.json`: invocation completion status, termination reason, exact
  control-tick and frame counts, and relative paths for every requested output.

The direct memory-resident helper may use a caller-supplied video name. Its
completion manifest is the authority for that relative path; consumers must
not hard-code `match.mp4` for this lower-level path.

Metadata schema `footballworld.replay-metadata/8` includes a top-level
`tracking_storage` receipt. Storage schema
`footballworld.tracking-storage/2` records the selected filename, semantic
tracking schema, ZIP/Deflate level, member and chunk counts, exact frame and
record-byte counts, semantic-record and archive SHA-256 values, and the
pickle-free streaming/random-access contract. Each chunk is one object-free
structured NumPy array. The final `__index__.npy` member contains its fixed
dtype, observation leaf paths, frame/tick extents, and per-chunk digests.
Working memory is bounded by one rollout chunk during writing, merging, and
auditing; the file is privately staged and published only after ZIP close.

Use `with footballworld.rendering.open_tracking(path) as reader:` for both NPZ
and plain/gzip JSONL. `reader.iter_chunks()` exposes the efficient NumPy path
for NPZ, `reader.iter_rows()` reconstructs the stable
`footballworld.tracking/8` semantic dictionaries, and `reader.row(frame)`
performs indexed chunk access.
All NPZ loads set `allow_pickle=False`. Interoperability tooling can call
`export_tracking_jsonl(source, destination)` explicitly; canonical capture
does not store a duplicate JSONL. `open_tracking_jsonl` provides direct JSONL
byte-stream access. The completion output repeats the storage receipt
and remains the authority for the relative tracking path.

Exact-event captures also record `time_axis.video_origin_time_s` and
`time_axis.video_time_step_s`. These fields align a higher-rate video sample
grid with the lower-rate integer control-tick sidecars; for the default
80/10/20 Hz grid the first two rendered states are at 0.05 s and 0.10 s, while
both remain causally associated with control tick 1.

Use `with footballworld.rendering.open_events(path) as reader:` to consume the
final `frames` array incrementally. The reader rejects schema, framing,
non-contiguous frame, and non-increasing tick violations. Its SHA-256 is
available only after EOF, preventing a partial scan from being represented as
a verified file. Working memory is independent of match length.

The canonical NPZ writer and reader preserve semantic fields and validate chunk
ordering, counts, and hashes. Performance receipts and generated artifacts
remain in the private validation suite.

Parquet is not the canonical replay format. It would add the
large `pyarrow` dependency outside FootballWorld's NumPy/JAX base, and the
nested per-frame observation and variable sparse identity context would require
a second schema adapter. NPZ stays within the existing NumPy dependency and
preserves normalized observation leaf dtypes directly. Parquet can remain a
separate analytical export later.

Metadata `/9` retains the earlier top-level contracts and keeps the
runtime-authored `render` receipt separate from caller `user_metadata`. It also
records `footballworld.match-manifest/1` once: registered on-field and bench
identities, realized physical profiles, declared bench role preferences, and
the formation catalog. It records the selected frame rate, requested
resolution/codec/pixel format,
effective host-clamped encoder settings, requested/effective worker counts,
process start method, chunk and segment counts, fixed camera azimuth,
elevation, distance and focal length, environment partial-view state, exact
environment aperture, host gaze-cue aperture, gaze limits, event chunk size
when applicable, the host-only render-worker backend, the two owned
asynchronous RGBA handoff buffers per worker, and installed renderer dependency
versions. It
deliberately does not start a second ffmpeg process merely to probe a version
string. `render_full_match.py` automatically decodes/probes and verifies
an authoritative full-match publication; `--verify-video` requests the same
cold-path check explicitly for a diagnostic capture. This verifies
the actual stream codec, pixel format, rate, dimensions, and frame count.
Completion schema `/2` additionally records byte counts and SHA-256 hashes for
all four child artifacts. An authoritative publication requires clean or
frozen source authority, an unchanged source receipt, no maximum-step budget,
and environment termination at regulation completion including causal added
time. The library API remains fail-closed and requires `verify_video=True`
when its publication guard declares an authoritative capture. The public
full-match CLI supplies that value automatically for an authoritative run;
diagnostic previews may omit that cold-path cost or request it explicitly with
`--verify-video`.

Tracking schema `footballworld.tracking/8` stores seam-free `body_forward`,
bounded `gaze_yaw`, and the derived `view_forward` for every player. Those are
authoritative environment-state values, unlike the separately labelled host
inferences described in [tracking-view-augmentation.md](tracking-view-augmentation.md).

Managed exact-event capture generates replay provenance rather than accepting
it as a user claim. It includes the exact PRNG key representation,
FootballWorld/JAX/JAXlib versions, backend, device kinds, x64 and PRNG settings,
semantic schema versions, environment and imported-source hashes, and policy
class and configuration hashes when available. User metadata remains
separately nested. The direct memory-resident helper still generates its
render-settings and exact-count receipts, but it cannot infer its caller's
environment source, policy, or trajectory origin; any such caller metadata is
kept under `user_metadata` and is not upgraded to runtime provenance.

`footballworld.events/15` uses
`footballworld.occurred-events/2`. It stores only meaningful fixed-tree rows,
identified by event type, physics substep, optional slot, and every original
field. Omitted rows are explicitly the canonical empty sentinels. In
particular, a contest row is retained when it was selected, occurred, or has an
invalid override pin. `restore_sparse_frame_events` reconstructs the exact
fixed event tree from a canonical template.

Ordinary visualization retains sparse non-MOVE/contact-relevant action rows.
Set `exact_actions=True` on capture or `render_mp4` to retain all players'
categorical intent plus move, force, launch, spin, and gaze controls on every
frame. The header then declares complete reconstruction only when no submitted
action frame is missing. This is a host serialization choice and adds neither
JAX leaves nor a second transition graph.

Each frame declares its causal ordering in the event header: action,
`frame_events`, and transition telemetry are pre-management; substitutions and
acting-goalkeeper receipts are the following manager transaction; tracking at
that control tick is post-management. Ordinary frames need no identity override
and remain byte-for-byte sparse. Management-boundary overrides contain only
slots whose identity or goalkeeper role actually changed.

Each exact boundary row also carries `last_contact_at_boundary`, derived on the
host from that frame's post-transition `last_contact`. Invalid actors remain
explicitly unknown rather than being clipped to player slot 0. This is only
last-touch attribution for the boundary decision; it is not a persistent shot
identifier and cannot by itself define goals per shot.

Action traces use `MOVE` and policy provenance as defaults and store only
categorical exceptions. `restore_sparse_action_trace` reconstructs the full
categorical player axis. Exact capture additionally uses
`footballworld.contact-actions/3`: a row retained for a non-`MOVE` request or
a realized deliberate contact carries the submitted `move`, `force_to_ball`,
`launch`, `spin`, and `gaze_center` controls. This is sufficient to audit pass,
cross, shot, control, clearance, challenge, and contact-time view signatures
without duplicating every player's movement on every frame. Retained `PASS`
rows also carry the registered `intended_receiver_player_id` from the rule
policy's capture-only action receipt, or `null` when the policy exposes no such
receipt. This is intent, not a pass completion claim. The extra `[T,N]`
receiver-ID output exists only in managed exact-event capture and never enters
training rollout results, recurrent policy state, or tracking storage.

This bounded continuous trace is intentionally not a rollout-replay contract.
Omitted `MOVE` rows often contain non-zero movement controls, so those values
are unknown and `complete_action_reconstruction` is false. Categorical intent
and provenance come from the authoritative action trace; the eight continuous
values are the submitted policy action. Realized contact intent, provenance,
and outcome remain authoritative in the exact frame-event rows.

Each event frame also retains the compact transition telemetry from
`step_with_events`, including event-budget exhaustion, terminal flags,
restart opening, contact attempt, and kick application.

Tracking-owned clocks and state values are not duplicated in each event row.
The renderer does not infer those values a second time.
Before publication, the merge requires supported and invariant chunk schemas,
spool-global indices, identical event/tracking control ticks, contiguous global
control ticks, and exact event, tracking, source, and encoder-submitted video
frame counts. Successful encoder close plus returned segment counts are the
runtime video integrity source, so `ffprobe` is not a hard deployment
dependency.

Spooling is two-pass and chunk-bounded. Chunk serialization assigns global
frame indices immediately. The first pass parses each tracking row once for
schema, cross-sidecar tick, and contiguous-clock validation, then copies its
canonical source line directly instead of serializing the roughly 856 MB
historical full-match stream a second time. It also collects only headers and
frame-availability metadata. The second pass loads one event chunk, injects any
sparse formation receipt, writes it immediately into the final JSON array, then
releases it. It never accumulates the full fixed event tree or all event-frame
mappings in memory.

## Match reports

The host-only analyzer verifies a published replay and derives versioned JSON
metrics plus a metrics-only HTML report that neither links nor embeds the video.
Tracking remains the source for
continuous spatial and workload calculations; discrete facts remain event-owned.
See [match-report.md](match-report.md) for the command, metric definitions, and
diagnostic-capture rules.

## Presentation and capture design

The renderer uses a fixed oblique pinhole projection, stands, 3D goals and
nets, a minimap, a host-only boundary, lazy dependencies, a bounded encoder
queue, and FFmpeg encoding. One spherical marker represents each player, while
authoritative `body_forward` and `gaze_yaw` produce a translucent fan without
boundary rays. Fixed eventful device chunks, bounded two-pass sidecar spooling,
managed-boundary interruption, sparse exact events, and atomic directory
publication keep capture causal and bounded. Renderer variants,
moving-camera state, historical-frame effects, ball trails, historical feeds,
full-trajectory host materialization, and post-hoc event reconstruction remain
excluded so every segment is independently renderable from exact causal host
frames.

SoccerWorld's renderer established the sound host-side pattern of persistent
artists, blitting, and batched collection drawing. FootballWorld retains those
advantages and keeps all presentation work outside the JAX step. It does not
inherit SoccerWorld's larger pose/path and historical-effect graph: those
features would increase per-frame host work and memory without changing the
authoritative FootballWorld state shown by the current presentation contract.

The fixed-shape hot path additionally reuses camera-space depth, collection
`Path` objects, unchanged color/visibility state, and the invariant layout of
each player number or `GK` label. Labels remain the original individual Agg
text artists and therefore retain the same font hinting and rasterization.
Intent-ring geometry is skipped when no non-move action is displayed. These are
render-only changes; they do not change capture, physics, rules, tracking, or
event facts.

The 2026-09-09 renderer validation used 225 already-decoded tracking frames at
960 x 540 on an AMD EPYC 7763 host (Python 3.11.13, NumPy 1.26.4, Matplotlib
3.10.9, FFmpeg 7.0.2, one `veryfast` encoder thread). With environment stepping
and tracking decoding outside the timed region, warmed Agg-only throughput rose
from a median 69.360 fps to 94.911 fps (1.37x). The same `render_frames` path
including H.264 encoding rose from 61.416 fps to 82.878 fps (1.35x). The four
encoded A/B repetitions all produced the same 621,906-byte MP4 with SHA-256
`602c71f069719d2089a9aba82deb606e93999c174111368eab162ef58f48c3c9`;
an uncompressed frame comparison also had zero changed pixels. These are
host-specific performance receipts, not portable frame-rate guarantees or
football constants.

The follow-up 2026-09-09 font-path experiment kept SoccerWorld's sound
host-only persistent-artist design and removed only repeated resolution of an
unchanged Matplotlib `FontProperties` object. The cache is scoped to one Agg
renderer and retains font loading, sizing, hinting, and rasterization; missing
private backend hooks fail closed to Matplotlib's original method. Across 225
frames from the same tracking artifact, both 960 x 540 and 1920 x 1080 had zero
different uncompressed RGBA frame hashes. Median Agg-only throughput over 900
frames rose from 96.887 to 103.124 fps at 540p and from 92.471 to 98.791 fps at
1080p. A 50--900-frame linear fit estimated about 58 ms baseline and 79 ms
final-candidate fixed setup per segment; the noisier candidate intercept projects
to about 40 seconds, or 3.55% of the observed 508-segment full match. Sharing
figures across independently encoded segments was rejected here because the
roughly 3.5% projected saving does not yet justify weakening process
isolation, bounded lifetime, and failure recovery. Generated raw repetitions
and frame hashes are intentionally excluded from the repository.

The replay audit reports possession and turnovers, challenge
and foul outcomes, submitted intents against realized contacts, shot/pass
geometry, stationary-live streaks, restart delays, goalkeeper holds, event
budget, stamina and speed by half, score, termination, and numerical guards.
Its completed visual review covers the opening, both half middles, the
halftime boundary, late play, terminal play, and a verified aerial effect.

Any DFL comparison must keep submitted intent separate from realized contact.
Current telemetry contact/action opportunity rates use control frames whose
pre-transition state is live; the separately reported live occupancy fraction
uses post-transition `state.ball.live`. Both are environment-specific
estimands, and neither may be labelled equivalent to a provider DFL live-time
field.
