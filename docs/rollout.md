# Rollout contract

FootballWorld provides fixed-length rollout factories for any implementation
of the public <code>PlayerPolicy</code> protocol, including the shipped
observation-only rule policy. They connect <code>observe_all_si</code>, policy
memory, and an environment transition in one <code>jax.lax.scan</code> while
keeping management outside the per-frame carry.

## Player policy boundary

<code>PlayerPolicy</code> is a structural, framework-neutral contract:
<code>initialize(observations, roster)</code> returns an explicit policy-state
PyTree, and <code>step(observations, roster, state, match_key)</code> returns an
<code>IntentAction</code> plus the next state. The state tree, every leaf shape,
and every leaf dtype must remain fixed within one compiled scan. FootballWorld
checks the object methods at factory construction and checks the result and
state signatures while tracing the scan body; these checks add no XLA
operations. Replaceable learned parameters should be leaves of the dynamic
policy state rather than arrays captured by the policy object.

The learned-policy boundary is framework-neutral and uses a fast stacked-array
path. The compiled rollout has no per-agent dictionary adapter:
<code>SIObservation</code> and <code>IntentAction</code> remain fixed-shape
PyTrees, avoiding Python dict assembly and preserving categorical intent as a
separate integer leaf.
Reward calculation, checkpoint loading, and MARL framework adapters remain
outside this environment rollout contract.

The low-frequency <code>refresh_policy_state</code> and
<code>apply_management_tactics</code> helpers are still explicitly
<code>RuleBasedPolicy</code>-specific because partial roster-memory refresh and
tactical-anchor mutation depend on <code>RulePolicyState</code> fields. Generic
policies can use all five rollout factories, including
<code>make_managed_advance</code>, but a full <code>ManagedRunner</code> adapter
must define its own roster-change and tactics-update semantics rather than
silently resetting learned recurrent memory.

## Separate compile graphs

Choose the smallest output graph that the caller needs:

| Factory | Result | Per-frame output | Intended use |
| --- | --- | --- | --- |
| <code>make_advance</code> | <code>AdvanceResult</code> | none | advance a match with minimum trajectory memory |
| <code>make_interruptible_advance</code> | <code>InterruptibleAdvanceResult</code> | none | pause a fixed-size chunk for emergency goalkeeper management |
| <code>make_managed_advance</code> | <code>ManagedAdvanceResult</code> | none | pause at every new restart or emergency goalkeeper boundary |
| <code>make_rollout</code> | <code>RolloutResult</code> | actions and ordinary step results | retain an event-free trajectory |
| <code>make_event_rollout</code> | <code>EventRolloutResult</code> | actions and exact eventful step results | evaluation and replay sidecars |

These are five separate player-rollout graphs. Event collection and manager
interruption are not runtime flags on the ordinary rollout, so compiling
either opt-in path cannot enlarge the
advance or event-free graphs. The factories do not call <code>jax.jit</code>;
compose the one-match kernel, optional match batching, and
<code>jax.jit</code> in that order.

<code>num_steps</code> is a static scan length. A different length creates a
different program, so use a small set of fixed chunk lengths and cache each
compiled callable by output mode, batch strategy, length, environment
configuration, and input shape/dtype. Reuse the executable rather than
rebuilding a factory for every chunk.

Player count <code>N</code> is shape-static, not fixed at seven or eleven. A
2v2 rollout is supported with two-player rosters and
<code>MatchConfig(minimum_team_players=(2, 2))</code>; the default remains
seven per team, and changing <code>N</code> requires a separately compiled
executable.

    import jax

    from footballworld import (
        batch_rollout,
        initialize_policy_state,
        make_advance,
        make_rollout,
    )

    # env, policy, reset, and roster were prepared once for this match.
    policy_state = initialize_policy_state(env, policy, reset.rollout, roster)
    advance_900 = jax.jit(make_advance(env, policy, num_steps=900))
    final = advance_900(
        reset.rollout, reset.setup, roster, policy_state, match_key
    )

    # Inputs to this callable have a common leading match axis B.
    one_match = make_rollout(env, policy, num_steps=900)
    many_matches = jax.jit(batch_rollout(one_match))
    trajectory = many_matches(
        rollouts, setups, rosters, policy_states, match_keys
    )

The same immutable match key may be reused across consecutive chunks. Each
transition folds in the absolute pre-step control tick, so changing chunk
boundaries does not restart the random stream. A completed match remains an
absorbing state for the rest of a fixed-length scan.

## Observation-only ball decisions

Ground loose-ball `CONTROL` is an inward settling touch at the policy's native
control scale. This inherits SoccerWorld's bounded reception-control principle
while adapting it to FootballWorld's explicit physical carry-radius ownership:
an outward touch at the radius boundary could otherwise erase a successful
trap on the next physics substep. Aerial cushioning remains momentum-aligned;
contact reach, outcome, and possession remain environment-authoritative.
The primary claimant homes on the forecast ball point and lets the swept
contact solver determine entry; it does not brake to zero on the mathematical
outer tangent. A new trap also gets one causal decision opportunity before
physical carrier verification may release it, and the built-in policy does not
issue a dribble re-touch during its secure-follow window. FootballWorld keeps
player-level possession rather than inheriting SoccerWorld's broader team-only
phase latch.

During an own live pass, the lawful planned receiver now remains the sole
primary runner until the first later contact. This inherits the SoccerWorld
stable receive-runner assignment and fixes a FootballWorld failure in which
the forecast changed runner at the meeting point, sending the intended
receiver back toward formation and turning a controllable pass into a passive
body deflection. The assignment still fails closed immediately when the
receiver is invalid, inactive, hidden, no longer a lawful candidate, or the
ball becomes dead/a restart; no hidden team-shared trajectory state is added.

Reception is also distinct from an automatic relay. A 60-second policy
diagnostic had 23 of 25 different-teammate receptions pass again within two
seconds, with 17 delays exactly 0.4 seconds, exposing decision-cadence pinball
rather than football timing. FootballWorld therefore inherits the SoccerWorld
contextual quick-relay principle: during the first two seconds after a
CONTROL/TRAP, an early pass needs pressure, body alignment, a
completion-qualified first lane, a safe bounded second leg, and one
episode-stable seeded draw. The rounded 0.31 probability is a transfer prior
from the SoccerWorld receipt, not a FootballWorld measurement or physical
constant. FootballWorld rejects the full SoccerWorld quick-relay mechanism and
dense producer because its harder 1v1 contest/carry semantics are intentional;
failed early-relay gates retain normal carry and support movement, and all
contact/tackle geometry remains unchanged.

Ordinary restart positioning follows the same ownership boundary. SoccerWorld
keeps non-kickers in an attacking or defending set-piece phase while its rules
layer enforces legality. FootballWorld now retains that phase continuity; only
the taker's legal approach is left stationary for the environment to execute.
For throw-ins, goal kicks, corners, free kicks, and offside free kicks, the
built-in policy uses compact team centroid, depth, and width moments inherited
from SoccerWorld's data-backed restart-shape abstraction. The source data,
player-level field, derived aggregates, and fit pipeline are private. Individual
targets preserve FootballWorld's formation anchors; unsupported cells,
kick-offs, penalties, goalkeeper holds, and goalkeepers fall back to the
procedural formation field. The transferred shape remains a compatibility
prior rather than a fitted FootballWorld constant. The 3 m/5 m minimum
team-spread floors and 4 m arrival slowdown radius are explicitly
numerical/tactical design priors rather than measured football constants.

The shipped player policy plans from one sparse observer row: the unique
possessor in open play or the designated taker during a restart. Receiver
evaluation compares that row with fixed-shape teammate and opponent pools;
prospective offside eligibility is likewise recomputed only from that actor's
partial-observation row. Neither restart nor goalkeeper distribution
materializes every observer's receiver matrix. The runtime graph never solves
a full ball trajectory.

### Possession-episode attack patterns

`TacticalPlan` remains the team's persistent structural identity. The rule
player policy separately draws one `AttackPattern` for an observed team
possession episode: `PROGRESSIVE_CARRY`, `THIRD_MAN`, `WIDE_OVERLOAD`,
`SWITCH_PLAY`, or `RUN_BEHIND_DIRECT`. The draw is keyed by the immutable match
key, the observed episode start tick, and team, so it is invariant to rollout
chunk boundaries and repeats exactly with the same seed. It uses one scalar
pattern draw and, at most, one scalar run-behind receiver draw rather than a
per-player categorical search. No action, observation, or environment-state
leaf is added. The rule policy adds one observer-local int32 `attack_phase`
leaf: 88 bytes for 22 players.

The episode age continues across a loose ball only when public last-contact
facts prove a live, kick-applied same-team `PASS` release. A known deflection,
non-pass contact, restart, or dead ball ends that inference. If the possession
fact is hidden by partial observation, the long-plan path fails closed and the
ordinary frame policy remains authoritative; retained observer memory is not
used to invent a currently observable attack pattern.

The program advances only after an observed different same-team possessor:
setup 0, execution 1, then inactive/completed -1. A requested pass that never
reaches a teammate cannot advance it. Hidden rows preserve only their own
memory and do not act on it; dead balls and restarts reset it.

Patterns apply bounded movement and receiver-score biases after the existing
team shape has been built. They never make an inactive, invisible, unreachable,
ineligible, or currently offside receiver legal, and final pitch and observable
offside caps still apply. `PROGRESSIVE_CARRY` may commit the carrier for no more
than the existing 2.2 s solo-carry soft limit, but immediately yields to the
base choice under pressure, a good shot, or an urgent clearance. `THIRD_MAN`
uses setup for A-to-B and enables its previous-actor exclusion only in
execution for B-to-C. `SWITCH_PLAY` first overloads the ball side,
`WIDE_OVERLOAD` defers its cross/cutback bonus until execution, and the
selected `RUN_BEHIND_DIRECT` forward starts moving before the release.

Long-run diagnostics also showed that ordinary passes moved backwards on
average while crosses progressed. The rule policy applies a small continuous
backward-pass cost only when the carrier is not under pressure; it never masks
the sole reachable outlet. A moving player's first ground loose-ball CONTROL
uses the opposite of observed player velocity as a capped cushioning request
instead of a fixed goal-facing push. This affects only the built-in policy:
the environment action space and external-policy controls are unchanged.

Pattern receiver bonuses use the square of the existing completion score.
This makes the multi-tick plan a preference among credible lanes, rather than
an instruction to execute a low-quality pass. The third-man program explicitly
prefers a secure connector in setup, then excludes that connector while seeking
a progressing third player in execution; completion after that handoff prevents
the same plan from becoming a recurrent passing loop.

Run-behind timing uncertainty is coupled to one episode-selected active CF/WF.
The wider prospective offside margin applies only to that receiver, and its
shape timing shift applies only when that receiver is the actual selected pass
target. All other receivers keep the ordinary 0.10 m prospective tolerance.
An onside runner can still be led into space behind the current line because
Law 11 eligibility is evaluated at the kick instant, not the arrival target.
The environment's authoritative offside latch and adjudication are unchanged.

Ground passes and aerial services instead use immutable distance tables rolled
from FootballWorld's authoritative ball physics. The default tables ship with
the package and therefore add no calibration compile at policy construction.
An immutable custom physics configuration is calibrated once on CPU through a
cached <code>lax.map</code>. At runtime, fixed one-dimensional interpolation
maps a selected target to kick controls. Crosses first select a visible,
central receiver in the attacking zone, enforce the physical range, and score
the receiver's predicted arrival; only then are they labelled as
<code>PASS</code>. The receiver need not be farther forward than the wide
carrier: this preserves both forward services and cutbacks while the
attacking-zone, offside, centrality, and physical-range gates remain
authoritative. Once a receiver passes those gates, the existing tactical
wide-pass gain also rewards a wide-to-central service in proportion to
attacking depth; this adds no cross-only tuning coefficient or policy-state
field.

Restart choices are type-aware and seeded by the immutable match key and
absolute control tick. Throw-ins prefer an observable short outlet, corners
seek an observable central runner, goal kicks may choose a wider or longer
outlet, direct free kicks probabilistically choose pass or shot from visible
distance, goal angle, and goalkeeper position, and indirect or offside
restarts remain pass-only. If a view-limited taker sees no receiver, it uses a
short bounded release into field space rather than hidden player data or a
forced long shot. Kick-off, free-kick, and offside-restart receiver targets,
plus direct free-kick shot targets, must also point away from the taker's
restart-side body support; this prevents the built-in policy
from releasing the ball directly back through its own solid collider without
making that collider transparent. Penalties vary the visible goal lane. The
initial kick-off boundary and the environment-owned three-second release stay
unchanged. While a released restart remains untouched, the taker is excluded
from loose-ball and aerial-runner selection, but passive body collision remains
solid. On an ordinary release frame, the rule policy gives only the taker a
small one-frame movement opposite the actual kick direction. Opposite and
forward headings share the same capsule axis, so the body still presents its
narrow depth to the departing ball while its velocity now increases separation
instead of following the release. This adds no coefficient and suppresses no
body physics. The authoritative retouch adjudicator remains active as the
backup for any contact that still occurs.

These tactical weights are explicit design priors, not DFL-fitted
probabilities. Factory calibration, type-specific restart handling, and
future-runner evaluation remain separate concerns. Open-play cross targets may
remain level with or behind the carrier so ordinary byline cutbacks stay
available. Lookup values or event-rate tables from a different physics/contact
definition are not treated as measurements for this environment. Goalkeeper-hold long aerial
punts remain disabled until a height-conditioned table models their
pelvis-height release; the current policy uses a physics-derived ground outlet
rather than claiming that the ground-release cross table is exact for a held
ball.

Goalkeeper positioning remains observation-only. For contact intent arming,
the policy additionally projects the observed ball--goalkeeper relative segment
over one control frame and tests its closest point against the existing hand
reach, height, and own-penalty-area bounds. This lets a fast ball that begins
outside the reach circle still request `CONTROL` before it crosses the swept
physics volume. The calculation adds no automatic save: the substep contact
solver and rule gates remain authoritative, and receding, too-high,
out-of-area, or recovery-locked paths stay inactive.

## Restart-layout liveness

Every public restart is exposed only after one of three fixed-shape layouts
passes the same law, field, and player-clearance audit. The first candidate is
the minimum local correction. The second packs only displaced players on
inward-facing rings and centres each occupied ring on its actual population,
so a small group near a corner is not placed at one edge of a nominal
sixteen-player semicircle. The third candidate is a deterministic all-movable
emergency layout. Its displacement cost keeps it unselected whenever either
local candidate is valid; its purpose is bounded liveness, not tactical style.
At a goal- or touch-line restart, the designated taker may stand just outside
the field only at the exact ball-relative release pose. The ball must itself
be on or inside the field boundary, so malformed remote actors or replay balls
cannot use this narrow exception.

At kick-off, the solid-capsule release pose places the taker immediately beyond
the centre mark in the attacking half, facing back toward the ball. IFAB Law 8
explicitly exempts the taker from the own-half requirement, while every other
player retains the existing half and centre-circle projections. SoccerWorld's
behind-ball pose is not inherited here: with FootballWorld's passive body
collision it blocks every trajectory toward a legally positioned own-half
team-mate and forces the built-in policy into a receiverless forward release.
The mirrored pose keeps the same ball/capsule clearance, enables an ordinary
backward team pass, and changes no public state field or restart timing.

The first live frame after a restart PASS now bootstraps the releasing team
from the visible last-contact actor and roster metadata. Restart possession
starts at `NO_TEAM`, so the ordinary previous-possession bridge cannot identify
this flight. The same actor starts a fresh possession episode (with no stale
pre-restart predecessor), so the eventual receiver is consistently the first
same-team handoff after opening, post-goal, or half-time kickoffs. Once
bootstrapped, the same observer-local receiver identity,
arrival point, and countdown used for open-play passes remain causal until
contact or invalidation. This inherits SoccerWorld's sound principle that a
receive runner approaches a physical rendezvous rather than chasing the ball's
current position. FootballWorld also inherits its target-only receive motion:
adding the future ball velocity to player velocity made a runner reverse before
reaching an incoming pass. The full SoccerWorld receive graph is not copied;
FootballWorld retains its fixed current-plus-ten-future-sample supported-ground forecast, public
per-observer memory, braking-distance arrival taper, and fail-closed visibility
and identity checks to bound compilation and avoid hidden shared plans.

Restart projections must repair the collision component they create. The hot
graph uses a fixed candidate set rather than iterative global free-position
search, and it never revives a dead ball without the mandatory kick when a
timer expires. The candidate count,
recurrent state, and release timing therefore remain unchanged.

## Management and roster identity

Apply manager commands only between rollout chunks. Bench, command, and
substitution-ledger tensors therefore stay out of the 10 Hz scan carry. The
roster is nevertheless a dynamic array input to each kernel: after a
substitution, regenerate it with
<code>env.roster_metadata_si(rollout, management)</code> and pass the refreshed
value into the next chunk. Stable shapes and dtypes let the same compiled
executable be reused.

Use <code>make_managed_advance</code> when the rule manager should select every
visible restart taker and schedule ordinary substitutions. It takes the small
<code>ManagerBoundaryState</code> restart identity plus a runtime
<code>step_budget</code>. The rule or learned manager's policy memory remains in
the separately compiled decision call. The latter is capped by the factory's static
<code>num_steps</code>, so independently interrupted batch rows can finish one
common requested horizon without compiling a new scan length or overshooting
completed rows. The result distinguishes a new-restart boundary, a missing-GK
boundary, and ordinary budget exhaustion.

The initial kickoff is intentionally returned with
<code>steps_executed == 0</code>. Resolve it through the separately compiled
<code>make_management_decision</code> callable, then resume with the returned
manager-policy state and the same immutable match key. A restart produced by a
later environment frame is returned after that frame has counted, before its
three-second release delay can expire.

    from footballworld import (
        apply_management_tactics,
        make_managed_advance,
        make_management_decision,
        refresh_policy_state,
    )

    advance = jax.jit(make_managed_advance(env, policy, num_steps=900))
    decide = jax.jit(make_management_decision(env, manager))

    segment = advance(
        rollout,
        setup,
        roster,
        player_policy_state,
        manager_policy_state,
        step_budget,
        match_key,
    )
    if bool(segment.manager_required):
        decision = decide(
            segment.final_rollout,
            squad,
            management,
            manager_policy_state,
            match_key,
        )
        previous_roster = roster
        rollout = decision.step.rollout
        management = decision.step.management
        manager_policy_state = decision.policy_state
        roster = env.roster_metadata_si(rollout, management)
        if bool(decision.step.roster_metadata_changed):
            player_policy_state = refresh_policy_state(
                env,
                policy,
                rollout,
                previous_roster,
                roster,
                segment.final_policy_state,
            )
        else:
            player_policy_state = segment.final_policy_state
        player_policy_state = apply_management_tactics(
            env,
            policy,
            rollout,
            management,
            roster,
            player_policy_state,
        )

Do not JIT this host branch as one monolithic function. The manager decision
has bench-sized tensors, while roster refresh rebuilds an <code>N x N</code>
player observation only after an identity change. Keeping three executables
prevents a taker-only decision from inheriting the refresh graph.

Use <code>make_interruptible_advance</code> when a chunk may cross a goalkeeper
dismissal. Its scan still has exactly <code>num_steps</code> cells, but after a
team without an active goalkeeper reaches an authoritative management
stoppage, every remaining cell carries identity values. The fixed-shape result
contains the final rollout and policy memory plus scalar
<code>steps_executed</code>, scalar <code>manager_required</code>, and
<code>manager_team_mask</code> with shape <code>[2]</code>. Under match batching
these become <code>[B]</code>, <code>[B]</code>, and <code>[B, 2]</code>. A
zero-step result is possible when the input already awaits management.

The interruption predicate deliberately reuses the exact manager-transaction
stoppage boundary. A live-ball advantage continues. A valid
<code>GK_HOLD</code> is not a manager interrupt, but a malformed hold with no
active same-team goalkeeper opens a manager boundary; a caller using direct
<code>env.step</code> instead proceeds through the existing corner-expiry
fail-safe. Falling below the configured minimum player count remains a terminal
match rather than a request to the manager. Only goalkeeper absence and a
genuinely open boundary set the team mask.

When <code>manager_required</code> is true, transfer that small result to the
host, build <code>env.emergency_goalkeeper_command(...)</code>, and call
<code>env.manager_command(...)</code> between chunks. If roster metadata
changed, rebuild it and pass the old and new values through
<code>refresh_policy_state</code>; then apply the refreshed player-tactics
observation before resuming. Squad, bench, command, and manager-ledger tensors
never enter the scan carry. Reuse the same immutable match key on resume. Since
the next active transition again folds in the unchanged absolute control tick,
chunk interruption does not consume or restart the random stream.

The current rule-policy memory is indexed by reusable physical roster slot.
After any committed substitution, retain the prior roster, regenerate the new
roster, and call
<code>refresh_policy_state(env, policy, rollout, previous_roster, roster,
policy_state)</code>. It detects identity changes from either
<code>slot_generation</code> or <code>player_id</code>. Only the changed slots'
restart, possession, counterpress, and clock memory is initialized from the
new observation. Formation anchors, roles, and team-slot tables remain
bit-exact because they describe the tactical slot rather than its current
occupant. Reusing all of the outgoing player's memory would be incorrect;
reinitializing the whole policy state would also erase manager tactics.

Formation and set-piece-taker commands also belong between chunks. After any
required roster refresh, call <code>apply_management_tactics</code> once; it
updates both teams' anchors without rebuilding the full player observation.

### Exact-horizon host runner

<code>make_managed_runner</code> packages the same boundaries into a reusable
scalar host orchestrator. It requests a runtime budget no larger than its one
static <code>chunk_steps</code> scan, handles a boundary, and resumes until the
requested horizon is reached exactly. Opening formation selection is called
once before the initial kickoff; the environment's start predicate makes it a
no-op for every resumed or mid-match state.

    from footballworld.managed import make_managed_runner
    from footballworld.rollout import initialize_policy_state

    runner = make_managed_runner(env, chunk_steps=900)
    player_state = initialize_policy_state(
        env, runner.player_policy, reset.rollout, roster
    )
    managed = runner.initialize(
        reset.rollout,
        reset.setup,
        squad,
        management,
        roster,
        player_state,
    )
    result = runner.run(managed, squad, match_key, num_steps=13_500)

The static <code>env.policies</code> switches install or omit the shipped rule
player, full pre-reset opening manager, post-reset opening-formation adapter,
match-management, and taker policies. The full opening manager is invoked only
by <code>create_opening_match_from_policy</code>; a managed runner has no
candidate pool and therefore owns only the reset-first opening-formation
adapter. An explicitly
supplied policy takes precedence. With both manager axes off and no caller
manager, ordinary restart identities are only acknowledged; a missing
goalkeeper raises because it needs a real management action. For the shipped
combined manager, a separately disabled match or taker component is replaced
by an empty command through a static Python branch.

A learned manager implements the public <code>ManagerPolicy</code> protocol.
Its parameter PyTree and recurrent policy state are inputs only to the separate
manager initializer/decision executable. The high-frequency scan receives only
<code>ManagedManagerState.boundary</code>, the fixed two-team restart identity;
bench arrays, model weights, and learned memory never enter that graph. Roster
metadata and player identity memory are rebuilt only when the authoritative
manager result reports <code>roster_metadata_changed</code>. Tactical anchors
are then applied once at the low-frequency boundary.

### Independent managed match batches

<code>make_managed_batch_runner</code> applies the same host schedule to a
fixed-shape batch. Rollout, setup, squad, management, roster, player-policy
memory, manager-policy memory, and match keys have a leading match axis.
<code>num_steps</code> may be one scalar horizon or an integer
<code>[B]</code> vector. Completed rows receive a zero runtime budget while
other rows continue, so manager interruptions never force a completed match to
overshoot.

    from footballworld import make_managed_batch_runner

    runner = make_managed_batch_runner(env, chunk_steps=900)
    state = runner.initialize(
        rollouts, setups, squads, management, rosters, player_states
    )
    result = runner.run(
        state, squads, match_keys, num_steps=per_match_horizons
    )

The fixed <code>lax.map</code> transform keeps branch-heavy football
transitions on a conservative batching path. The scheduler does not
slice a dynamic active subbatch, which would create changing shapes and extra
compilations. Instead it transfers only
small <code>[B]</code> progress and boundary masks to the host once per chunk.

Manager parameters are shared dynamic inputs and therefore do not need an
artificial leading match axis; recurrent manager state remains batch-major. The
player advance, manager initialization and decision, roster refresh, tactical
refresh, acknowledgement, and opening placement are separate compiled
executables. Consequently enabling managed batching adds new cache entries
keyed by batch size, chunk length, shapes, dtypes, and environment
configuration, but does not enlarge <code>FootballWorld.step</code> or the
ordinary training rollout graph.

## Match batching and rendering

<code>batch_rollout(one_match)</code> always uses <code>lax.map</code>. This
preserves scalar control flow across CPU and GPU and limits accidental graph
expansion. There is no selectable dense whole-match transform and therefore no
second strategy-specific compilation cache entry.

A scalar scan produces time-major <code>[T, ...]</code> leaves. The outer match
transform stacks those as <code>[B, T, ...]</code>. The renderer's batched
transfer convention is instead <code>[T, B, ...]</code>. Prefer selecting the
one match to render while values are still on device, leaving
<code>[T, ...]</code>, rather than transferring or transposing an entire
rollout batch:

    match_index = 0
    states = jax.tree.map(
        lambda value: value[match_index],
        trajectory.steps.rollout.state,
    )
    env.render_mp4(states, "rendered-match")

If a caller genuinely needs to pass a batch to the renderer, it must swap the
leading batch and time axes explicitly before using <code>match_index</code>;
rollout code never silently reorders them.

## Event memory

Use <code>make_event_rollout</code> only when exact event telemetry is
required. A full-match event tree is intentionally much larger than the lean
10 Hz training carry because it retains bounded 80 Hz substep facts. Actions,
normal step outputs, device allocator overhead, and batching add to that
amount. Prefer shorter event chunks, transfer or serialize each
completed chunk, and release it before retaining another. Training-style
advancement should use <code>make_advance</code>; use
<code>make_rollout</code> only when the ordinary trajectory is actually
consumed.

## Implementation rationale

Rollouts use outer <code>lax.map</code>, fixed-length scans, and a caller-owned
JIT boundary. A dense whole-match transform is not exposed: in the current
7-a-side, two-match, 64-step CPU diagnostic it increased cold execution from
17.8 to 39.4 seconds and warm execution from 0.135 to 1.714 seconds. Five
purpose-specific factories keep event data and its compile/memory cost out of
lean advancement while preserving one authoritative environment and policy.

Between-legs passage is recorded as one fixed-shape, non-state-changing nutmeg
event per physics substep only on the eventful graph. Ambiguous bent substep
chords are omitted rather than inferred afterward, and no gait phase or other
unidentified recurrent state is added to the lean rollout.

Emergency-only and all-restart manager-boundary graphs are explicit opt-ins.
The ordinary advance and trajectory graphs remain untouched. The all-restart
graph carries only four manager identity integers and pays for one scalar
conditional plus the stoppage/restart/GK predicate; bench observations,
manager commands, and policy refresh remain separate executables.

Generation-based identity invalidation lives in manager-owned metadata with an
explicit, traceable chunk-boundary refresh for recurrent policies. The
player-id fallback handles callers without a generation ledger while keeping
manager identity tensors out of the per-frame physics carry.
