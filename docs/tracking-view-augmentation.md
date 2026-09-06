# Inferred body and view trajectories from tracking

FootballWorld exposes a host-only augmentation API for tracking sources that
contain player centres but no observed chest or eye direction:

```python
from footballworld.tracking import (
    infer_body_view_trajectories,
    sequence_start_from_identity,
    view_inference_config_from_environment,
)

inference_config = view_inference_config_from_environment(
    env,
    frame_rate_hz=25.0,  # the tracking source rate, not env.control_fps
    chunk_frames=2048,
)

sequence_start = sequence_start_from_identity(
    player_identity,                 # integer or fixed-width string [T, N]
    slot_generation=slot_generation, # optional integer [T, N]
    explicit_sequence_start=period_start,  # optional bool [T] or [T, N]
)
inferred = infer_body_view_trajectories(
    player_position,                 # [T, N, 2]
    player_velocity=player_velocity, # optional [T, N, 2]
    ball_position=ball_position,     # optional [T, 2] or [T, >=2]
    player_valid=player_valid,       # optional bool [T, N]
    ball_valid=ball_valid,           # optional bool [T]
    timestamps_s=timestamps_s,       # optional [T], reset rules below
    sequence_start=sequence_start,   # optional bool [T] or [T, N]
    # Optional caller diagnostic; no built-in/calibrated threshold.
    maximum_continuity_speed_mps=None,
    input_causality="causal",        # positions/ball sampled without future rows
    velocity_causality="causal",     # causal | unknown | noncausal
    source_provenance={              # copied through, never invented
        "provider": provider_name,
        "source_id": match_id,
        "content_sha256": source_sha256,
    },
    config=inference_config,
)

dataset_columns = inferred.as_dict()
```

All numeric inputs use SI units: player and ball positions are metres,
player velocity is metres per second, and timestamps are seconds. The origin
is the pitch centre; world x is the longitudinal axis and world y is the
lateral axis. Team 0 attacks toward positive x and team 1 toward negative x
before a side swap. The transform never changes the `[T, N]` row ordering. The
caller must retain and join its own frame/timestamp, period, player-identity,
team, and attack direction columns by those axes; the API deliberately does
not invent or rewrite them. The N axis should represent stable
provider-scoped player identities. Do not merge IDs from different providers,
matches, or teams merely because their
numeric or string values coincide. If a fixed provider slot is reused by a
substitute, that player-frame must be marked in `sequence_start` so the
replacement cannot inherit the outgoing player's direction or position
difference.

If stable provider IDs cannot be guaranteed, the caller may supply
`maximum_continuity_speed_mps`. Its default is `None`, so no threshold is
silently imposed. When enabled, a finite player row whose displacement from
that player's last trusted past position exceeds the supplied speed bound is
made invalid and marked in `continuity_rejected`. Rejected rows do not update
the trusted position. The first later reachable row is accepted with an
automatically added `sequence_start`, preventing position difference, body
carry, or gaze carry across the rejection. This is a fail-closed diagnostic
guard, not identity association, identity correction, or a calibrated football
coefficient. A caller must justify the bound for its source and preserve it in
provenance.

The result is explicitly labelled
`inferred_not_observed_or_ground_truth`. It must not be described as measured
gaze, body-part tracking, or a calibrated behavioral label. Every player-frame
has a body/view validity mask, a heuristic confidence, and a source code.
`tracking_valid` is the effective finite/explicit/continuity-guard validity;
`continuity_rejected` separately identifies rows rejected only by the optional
guard. The
dataset-level provenance records the method, coefficient authority, complete
configuration, velocity source, schema, and a deep copy of caller-supplied
source provenance. If the caller supplies no source identity or hash, the API
leaves it absent rather than manufacturing one.

`input_causality` records whether player positions, ball positions, masks, and
timestamps were produced without future rows. `velocity_causality` separately
records whether a supplied velocity column was derived from current/past
samples only. Both default to `unknown`. `provenance.algorithm_causal_given_inputs`
is always true for this past/current-row inference, while the stronger
`provenance.causal` is true only when both upstream declarations are explicitly
`"causal"` (velocity is intrinsically causal when the function computes its own
past difference). Use `"noncausal"` for centred differences, smoothers, provider
estimates, or position/ball interpolation that reads a future frame.
The continuity guard itself reads only current and past rows, so enabling it
does not make unknown or noncausal upstream samples causal. Provenance records
whether it was enabled and the exact caller-supplied limit.

The function never parses free-form `source_provenance` to guess causality. A
control-grid trajectory made by interpolation between preceding and following
provider frames must therefore set `input_causality="noncausal"`; it cannot be
upgraded into an online policy input merely because the inference loop itself
reads rows causally.

## Direction and missing-data contract

`body_forward` and `view_forward` are world-frame unit vectors. A relative gaze
is available as `[cos(yaw), sin(yaw)]`; the direct `gaze_yaw` and normalized
value are bounded by `gaze_yaw_max_rad < pi`. Consequently neither the body nor
view representation has a 0/2pi discontinuity.
An exact 180-degree limit is rejected because `-pi` and `+pi` would be the same
direction at two scalar endpoints, reintroducing the seam the bounded action
was designed to remove.

When supplied velocity is absent, the function uses only the current and
immediately previous valid position. It never reads a future frame. Reliable
motion falls back to that difference for individual missing velocity rows
inside an otherwise supplied velocity array; its source code records the
difference rather than provider velocity. The resulting motion direction
initializes or updates body direction with a bounded turn rate. Short
stationary gaps carry that inference with decaying confidence. If no body
direction exists and the ball direction is valid, a deliberately low-confidence
ball-facing prior may initialize it. Missing player samples emit zero vectors
with false masks; explicit masks can never make a NaN or infinity valid.

`sequence_start` is a boolean `[T]` or `[T, N]` mask; `[T]` broadcasts to every
player. Frame zero is always returned as a sequence start. A true player cell
clears body, gaze, carry age, and position-difference history before that frame
is inferred. The applied `[T, N]` mask is returned with the inferred columns.
It includes both caller resets and the first accepted reconnection after a
continuity-guard rejection.
Use it at substitutions, identity changes, period boundaries, and coordinate
frame changes. The combination of player identity and slot generation prevents
velocity windows from crossing a reused slot. Body direction remains a unit
vector rather than a scalar angle with a wrap seam.

Irregular positive timestamp differences are used directly. A repeated or
decreasing timestamp is accepted only when every player starts a new sequence
on that frame, and its inference `dt` is then `1 / frame_rate_hz`. Consequently,
provider periods whose clocks restart must mark the first row of the new period
globally. A positive timestamp gap is treated as one continuous
interval; if it instead denotes a tracking or period discontinuity, the caller
must mark the affected players in `sequence_start`.

Convert positions, velocities, and the ball into one shared FootballWorld world
frame before inference; never attack-normalize the two teams independently.
If the provider changes its coordinate convention at halftime, transform the
new period before calling and mark that boundary as a global sequence start.
The inferred outputs are then already world-frame values. If outputs must be
transformed afterward, transform both `body_forward` and `view_forward`, then
recompute the signed body-relative `gaze_yaw`; a reflection reverses its sign.
Retain the per-period attack direction separately. Duplicate timestamps inside
one continuous period must be consolidated or rejected rather than hidden by a
partial reset.

View direction uses a bounded, rate-limited relative yaw toward the current
ball when it is available. With no usable ball it emits a body-centred prior at
lower confidence. `ball_target_used` and the view source code distinguish those
cases.

A velocity-derived body direction cannot recover a player scanning while
stationary, looking sideways while running, or facing forward while
backpedalling. Ball-directed gaze is likewise a causal augmentation prior,
not a reconstruction of observed attention. Confidence and source fields must
therefore travel with every generated direction.

## Environment state and model-view consumption

`body_forward` and `gaze_yaw` are SI/raw `PlayerState` values. After aligning
the provider identity axis with the environment's current roster-slot axis,
insert those two arrays into a copied raw `Rollout.state.players` before calling
`env.observe`, `env.observe_all`, or `env.global_state_view`. Do not inject
`view_forward`: FootballWorld derives it from body direction and relative gaze.
Do not write an invalid zero direction into an active player state; consume
`body_valid` and either retain an explicitly sourced fallback direction or
exclude that reconstructed state sample.

The normalized global state keeps `body_forward` unchanged and divides
`gaze_yaw` by the environment's fixed gaze limit. Consequently
`gaze_normalized` can be inserted into a normalized global view only when the
inference configuration came from
`view_inference_config_from_environment(the_same_env, ...)`. Prefer inserting
the raw values and letting `global_state_view` normalize them. Player
observations are team-attacking-frame views, and optional partial visibility is
computed from the derived world-frame view centre. Therefore partial-view
observations must also be regenerated from the augmented raw state, not patched
column by column afterward. Full view remains the environment default, while
`global_state_view` is always unfiltered.

For a reusable 22-slot dataset, derive `sequence_start` from a change in either
provider player identity or slot generation, and set the whole row at each
period/coordinate boundary. Both identity and generation decide whether a
past-only velocity window may cross a frame. For an identity-major dataset with one
column per match-scoped player, preserve that identity table as a sidecar and
gather the active identities into current FootballWorld slots before state
augmentation.
`sequence_start_from_identity` implements this exact host-side rule. It accepts
only integer or fixed-width string identities and integer generations; it does
not guess identity from proximity or repair provider associations. This
preserves generation boundaries, vector-facing geometry, and explicit identity
ownership.

The algorithm is deterministic NumPy host code and independent of
`FootballWorld.step`. Within one function call it walks fixed-size chunks and
carries only O(N) recurrent state across boundaries. Changing `chunk_frames`
does not change numeric arrays or masks, including `continuity_rejected` and
guard-added sequence starts, although provenance records the chosen
chunk size by design. Arbitrary T does not compile a new or larger JAX graph.
Calling the function separately on external chunks is not equivalent:
every call resets at frame zero. The current API also materializes the full
inputs and outputs, so total memory remains O(TN); `chunk_frames` does not make
dataset writing bounded-memory.

## Provenance and inference limits

Tracking and dataset adapters remain outside the compiled simulation step,
validity masks are explicit, and every transform retains provenance. Fixed-
chunk host orchestration prevents sequence length from entering a JAX
compilation graph. Callers must supply `sequence_start` explicitly at slot-
generation changes.

Body facing is represented as a unit vector because a scalar angle has a wrap
seam and cannot encode an independently offset view centre robustly. For
centre-only tracking, inferred body and gaze directions are estimates rather
than source ground truth. All default rates, thresholds, confidence values, and
the ball-directed attention rule are recorded as uncalibrated design priors.
They must be replaced or calibrated when facing-labelled data becomes available.
Use `view_inference_config_from_environment` to bind body-turn and gaze bounds
to the exact environment instance whose state will be initialized. The source
frame rate remains explicit because simulator control frequency is not a valid
proxy for provider tracking frequency. Configuration alignment does not turn
any inferred direction or value into a measurement.
