# Match clock and added time

FootballWorld keeps two causal clocks:

- `control_tick` is the monotonic wall-clock frame index.
- `control_tick - dead_ball_control_ticks - first_half_live_extension_ticks` is
  regulation play time.

An out-of-play restart frame is added back at the end of the half. Goalkeeper
possession is live play and is therefore not added. The first-half
period-boundary penalty extension is stored separately so it is not counted as
second-half regulation. The counters are evaluated at control-frame boundaries,
so they are exact to one control frame and add no
substep-only bookkeeping to the physics kernel.

The first-half wall-clock boundary is stored when halftime actually occurs.
The renderer and sidecar writer use the same tick-rounded configured boundary as the JAX
environment. The renderer uses the completed boundary to show `45:00 +MM:SS`, then removes the
first half's added time from the public second-half clock so play resumes at
`45:00`.
Replay sidecars retain both the raw monotonic clock and the public clock.

When `halftime_enabled=False`, `halftime_seconds` is inactive: the environment
and renderer canonicalize it to the full-time boundary. Ordering and positivity
of the half-time boundary are
validated only when the transition can occur. The static gate prevents the half-time transition when the flag is false,
while the constructor validates `0 < halftime_tick < fulltime_tick` only when
that boundary is active. FootballWorld exposes independent half/full-
time seconds for short clips, so validating an inactive boundary would create
configuration-dependent environment/renderer disagreement with no rules
benefit.

## Replay time axes

The replay manifest distinguishes source data from encoded video. Tracking
and event rows retain every supplied source frame; their authoritative time is
`control_tick / control_fps`. Direct non-event `render_mp4` input may first be
causally resampled at `video_sample_fps`, after which `sample_every=N` retains
every Nth sampled frame. For `M` sampled frames and `K` retained frames, the
encoder uses `video_fps = K * video_sample_fps / M`, so the constant-rate video
keeps the exact sampled-grid duration even when `M` is not divisible by `N`.
The exact-event and managed match capture paths instead require
`sample_every=1`; they do not thin the authoritative event-aligned time axis.

Metadata schema v6 records `source_frame_count`, `tracking_frame_count`,
`video_sample_frame_count`, and the actual `video_frame_count` separately. It
also records `tracking_fps`, `video_sample_fps`, and `video_fps`; consumers do
not need to infer one grid from another. `tracking_fps` is null when source
ticks are duplicated or irregular, because per-row `control_tick` remains the
only exact clock in that case. `control_fps` still states the nominal tick
cadence, while `time_axis.tracking_span_s` records the actual first-to-last
source span.

A penalty awarded at either period boundary suspends that boundary. The clock
does not end the period while the kick is pending or while the taken kick is
still moving. The suspension ends from existing transition facts—a goal or other restart, the ball leaving play, or the ball
stopping—so this rule adds no arbitrary penalty timeout or trajectory coefficient. A
moving defending-goalkeeper parry remains live; a catch/hold, stop, boundary,
restart, or new contact by any other player settles it.

## Clock design

Regulation time remains separate from explicitly supplied halftime/fulltime
boundaries. Generated rollouts derive added time only from causal,
already-observed restart frames; the environment never announces a future
allowance to a policy. Callers may still supply explicit match boundaries, and
the period-ending penalty completion latch prevents either boundary mode from
discarding an unfinished penalty kick.
