# Model-output and normalization contract

FootballWorld keeps its recurrent physics and rules state in SI units. Model
views are normalized only when requested, so training inputs do not force a
normalize/restore cycle through the 80 Hz physics kernel.

## Public paths

The model-facing methods are normalized by default:

- `observe` and `observe_all`: player observations;
- `global_state_view`: centralized or critic state;
- `observe_manager`: manager-only state, including that team's bench;
- `observe_managers`: two stacked private manager rows for a joint controller;
- `observe_player_tactics`: own-team formation targets;
- `roster_metadata`: cacheable player attributes.

Every path has an explicit `restore_*` inverse. `restore_manager_observation`
recovers the original SI manager fields, while `restore_match_clock` recovers
the added normalized manager clock. The built-in player rule policy uses the
corresponding `_si` methods because it reasons in metres and seconds. The
low-frequency rule manager consumes the normalized manager view so it can use
the same causal normalized match clock; its factory holds only immutable scale
metadata needed to restore metric distance and ability values.
`Rollout.state` is the authoritative SI recurrent carry, not a model input.
Keeping it in the returned rollout is required for pure JAX stepping; callers
that need a centralized model state use `global_state_view`.

## Schema-name boundary

The top-level `footballworld.environment.PLAYER_OBSERVATION_SCHEMA_VERSION`
and `ROSTER_METADATA_SCHEMA_VERSION` names intentionally refer to normalized
model views. SI contracts use explicit `SI_*` names. At the current source
revision the versions are:

| Contract | Version |
| --- | ---: |
| SI player observation | 10 |
| normalized player observation | 7 |
| normalized global state | 6 |
| SI manager observation | 12 |
| normalized manager observation | 10 |
| SI roster metadata | 2 |
| normalized roster metadata | 1 |
| SI / normalized player tactical observation | 2 / 2 |

Consumers should fingerprint the named contract they actually serialize,
rather than treating the short top-level aliases as SI schemas.

## Fixed scales

No output is clipped and no batch, match, sampled roster, or running statistic
is used as a denominator. Scales come only from the immutable environment
definition:

| Feature | Scale |
| --- | --- |
| absolute player x/y | pitch half-length / half-width |
| observer-relative player x/y | full pitch length / width |
| absolute ball x/y | pitch half-extent plus ball radius |
| observer-relative ball x/y | full pitch extent plus ball radius |
| player velocity | configured admissible maximum player speed |
| relative player velocity | twice that maximum |
| ball velocity and height | conservative mechanical-energy envelope from maximum release speed, spin, height, gravity, radius, and inertia |
| player height, reach, speed, control, endurance | affine map over the fixed admissible roster domain |
| facing | radians divided by pi |
| short locks and restart delays | their configured maximum duration |
| regulation progress | configured, tick-rounded period/full-match duration |
| wall and long-lived counters | `2^22` control ticks |

The manager view includes the visible restart position, the managed team's
attack direction, the exact-restorable restart-opening tick, and compact
manager-only formation-catalog summaries. It also exposes both teams' current
active-player centroid and per-axis standard deviation in the observing
manager's attacking frame. These are public kinematic aggregates, paired with
`team_shape_valid`; they reveal neither the opponent's bench nor its registered
formation command. Centroid and spread use the pitch half-extent denominators.
Candidate attack depth and width use the same fixed pitch denominators and
restore to SI units; probability and defender fraction are dimensionless, while
validity and the identity-like content signature keep their exact dtypes. The
signature addresses stateless formation randomness. `restart_opened_control_tick`
is the exact substitution-window identity; neither value is a future or
historical event stream.

Small exact counts and categories—score, cards, team, player, role, restart,
intent, masks, and booleans—retain their integer or boolean dtype. Turning a
one-goal count into a near-zero float would be normalization in name only and
would also discard exact discrete semantics.

The manager's `current_substitution_window_open` is an exact boolean for the
current restart, while `tactical_epoch` and
`formation_changed_control_tick` identify the currently applied tactical
assignment. The epoch increments only when a different registered layout is
accepted; repeated or rejected commands leave both fields unchanged.

Manager substitution resources follow the same rule. Remaining substitutions,
remaining windows, and their configured maxima are exact `int32` leaves. The
model therefore sees both the current resource count and the competition
profile that governs later transitions; no per-match maximum is used as a
normalization denominator.

The wall-counter scale is a power of two. All admitted values through
`2^22 - 1` therefore round-trip exactly in float32. Regulation duration is
limited to `2^20` control ticks, leaving deterministic added-time headroom;
the environment fails safely at the wall horizon before int32 overflow. At the
default 10 Hz these horizons are about 29.1 hours of regulation and 116.5
hours of wall time, well outside a football rollout.

Normalized physical values may exceed `[-1, 1]` when the authoritative state
is outside the ordinary envelope. This is deliberate: clipping would destroy
information and make restoration impossible.

The host-only `restore_global_state_view` boundary enforces an exact typed-state
contract: continuous and normalized-counter leaves are float32,
categorical/counter identities are int32, and predicates are bool.
FootballWorld rejects its earlier permissive restore behavior, where a float
team ID or float64 position could survive normalization and become an invalid
authoritative checkpoint. Finite physical channels are not blanket-clipped or
rejected at `[-1, 1]`: that older check contradicted the reversible global-view
contract and rejected valid positions and velocities outside the ordinary scale
envelope. Intrinsic domains remain strict: normalized roster attributes and
stamina lie in `[0, 1]`, gaze stays in its configured interval, facing vectors
remain unit length, causal counters must fit their int32 tick representation,
and nested contact/restart provenance must use valid enum and address domains.
Validation and its batched device-to-host transfers happen once before
denormalization and never enter `step`, rollout scans, or model observation
graphs.

## Match clock

The clock exposes current period, regulation progress, dead-ball time accrued,
and causal added-time elapsed/remaining fractions. It also carries three
power-of-two-scaled counter channels used for exact restoration; these do not
replace the semantically useful period fractions. The denominator may depend
on the configured match duration because that duration is immutable for the
compiled environment. It never depends on the realized final whistle or a
future dead ball.

Period-boundary penalty completion time is separated from dead-ball
replacement time. It cannot be counted again in the second half. A moving defending-goalkeeper parry remains live until the ball stops, leaves
play, or creates a goal/restart. A new contact by any other player settles the
extended kick.

## Information boundary

`Perception.limit_by_view_angle=False` is the default. It returns full player
and ball visibility and canonicalizes the unused angle so it cannot create
extra JAX compilation keys.

When view limiting is enabled, the horizontal field of view defaults to 160
degrees. Hidden kinematics and actor-linked history are zeroed or replaced by
sentinels; visibility and `known` leaves distinguish hidden facts from real
zeros. For persistent actor-linked facts, `known` means that the causal actor
is currently resolvable in the observer's view. It does not claim that the
observer saw the original event; causal perceptual memory belongs to the
policy's recurrent state rather than the environment carry.
`gk_handling_restriction_known` makes an unseen opponent restriction
distinct from a genuine `NO_TEAM` value. A visible free ball is known to have no current possessor, but the
previous team remains hidden unless its causal actor is visible. Restart
release provenance is hidden when its actor is hidden. The centralized
`global_state_view` is always global and is never filtered by a player's view.

Invalid observer or manager indices return a completely masked finite view;
non-integer and boolean subject identifiers do the same. Eager host inputs are
checked before conversion. The compiled subject contract is scalar int32; callers
must range-check and cast wider host integers before entering `jax.jit`, or use
`observe_all`, because JAX may canonicalize them before the environment sees them.

## Compilation boundary

Normalization is absent from `step`, `step_with_events`, and all five shipped
player-policy rollout kernels. The player rule policy calls `observe_all_si`;
the manager's normalization runs only at a low-frequency restart boundary.
Learned policies can compose normalized observation calls with their own JIT
boundary.

Schema versions are host-level contract and fingerprint metadata, not PyTree
leaves. They therefore cannot create a per-observer batch shape mismatch or
inflate every compiled output. Model and public SI player, manager, roster,
and tactical-view schemas are versioned separately.

## Representation rationale

The observer-relative entity hierarchy makes slot-wise player aggregation
straightforward. Model views exclude per-rollout maxima, runtime clipping,
flattened float categoricals, and bench data. Absolute, relative, and ball-boundary
positions deliberately use different scales; sharing one pitch denominator
would blur their distinct physical bounds and weaken invertibility.
Manager resources are not decoded through policy-owned default caps:
the exact remaining and maximum counts travel together so custom competition
profiles cannot silently change meaning.

The causal goalkeeper-hand restriction remains explicit. Under optional partial
observability, every global relation adds a known bit instead of conflating a
hidden value with no relation.

## Host schemas and lossless flattening

`FootballWorld.action_spec`, `action_trace_spec`, `action_receipt_spec`,
`player_observation_spec`, `manager_observation_spec`, `global_state_spec`, and
`event_spec` expose the current PyTree contracts without entering `step`. Each
immutable spec records
symbolic arbitrary leading axes, exact trailing shapes and dtypes, normalized
ranges, SI source units, configuration-derived scales, visibility/masking
rules, the semantic version, and a SHA-256 layout fingerprint.

The environment-bound `flatten_observation`, `flatten_manager_observation`,
and `flatten_global_state` methods preserve the structured PyTree as the
authority, bind the layout to that environment's immutable normalization
context, and return separate `float32`, `int32`, and `boolean` blocks. The
equivalent top-level helpers require either an explicit `context=` or a
precomputed authoritative `layout=`; context-free normalized flattening fails
closed because it cannot truthfully identify configuration-derived scales.
`unflatten_*` verifies the block dtypes, sizes, common leading axes, tree
structure, and contract before reconstruction. A layout made from one sample
can therefore be reused for time-, batch-, observer-, player-, or substep-
leading arrays without changing its fingerprint.

```python
flat, layout = env.flatten_observation(observation)
same_observation = unflatten_observation(flat, layout)

# Reuse the exact environment-bound layout for compatible leading axes.
flat_batch, same_layout = flatten_observation(batch, layout=layout)
```

Action-receipt leaves retain their compact structured dtypes: `uint32` flags,
`uint16` masks, and the `int16` summary reason are encoded losslessly in the
wire `int32` block and restored to their declared dtype by
`unflatten_action_receipt`. `uint32` uses a bit-preserving cast, so the wire
integer may be negative when its highest bit is set; consumers that inspect
the flat block directly must use the layout dtype instead of treating it as a
numeric category. This preserves the original three-block `FlatTree` API and
its existing action/event fingerprints.

`schema_versions()` includes `action_trace` and `action_receipt` alongside the
action, observation, global-state, and frame-event contracts. The trace and
receipt have independent layout fingerprints because they answer different
questions: what was submitted, and what the environment causally consumed or
overrode. The action layout's semantic version is the numeric suffix of
`ACTION_SCHEMA`; those two values cannot identify different contracts.

The machine-readable slice/layout contract keeps structured PyTrees
authoritative. A single-float-vector representation is not authoritative
because converting
player identities, categories, sentinels, and masks to floats weakens exact
semantics and encourages hard-coded indices. One-hot or embedding model inputs
remain a separate downstream encoding concern. Scale metadata is derived only
from immutable environment configuration; no rollout extrema are recorded.
