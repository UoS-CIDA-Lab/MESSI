# Manager command contract

FootballWorld keeps management outside `step` and `step_with_events`. A manager
decision is rare relative to the default 10 Hz player control and 80 Hz
physics, so adding its bench and formation tensors to the recurrent physics
carry would increase compile signatures and device traffic on every frame for
no physical benefit.

Interactive callers that still want one transactional entry point can use
`StepCommand` with `execute_step_command`. The façade applies the player frame
first and the manager transaction second, then exposes both independent
receipts and the final post-management rollout. It is deliberately not used by
rollout scans, so benches and manager ledgers do not enter the high-frequency
JAX graph.

## Design rationale

The manager contract has four core properties:

- one fixed-shape PyTree combines substitutions, formation, acting-goalkeeper,
  and set-piece taker proposals;
- substitutions use a `(2, K)` table, formation uses `(2,)`, and takers use a
  `(2, R)` table, so the same contract works under `jit` and `vmap`;
- masks are authoritative and sentinels in unrequested cells are ignored;
- a manager proposes, while the environment remains the authority on roster,
  match, goalkeeper, substitution-window, and restart legality.

The command is not placed inside the scalar physics transition, does not
temporarily mutate an environment object's active taker command, and never
partially applies a team's multi-substitution request. This keeps manager
compilation separate and makes replay behavior explicit.

Append-only command-reason wire meanings are returned only from the
low-frequency manager APIs. Substitution,
formation, acting-goalkeeper, and set-piece-taker axes each have a fixed-shape
`int32` reason array alongside their applied mask. Explicit
codes for duplicate cells, invalid incoming profiles, identity conflicts, and
team-atomic rollback keep failures auditable. Manager receipts are not placed
in every player transition, so adjudication does not enlarge the lean
physics/rules step or its recurrent carry.

An acting-goalkeeper applied mask describes the resulting environment
intervention. If an invalid explicit nominee is replaced by the deterministic
fallback, the reason still explains why that proposal failed and the event's
`environment_forced` bit identifies the fallback. With no explicit proposal,
the same intervention reports `INTERNAL_FALLBACK`.

The learnable boundary is framework-neutral. `ManagerPolicy` receives dynamic
parameters and recurrent state only in the rare in-match decision graph;
`ManagerBoundaryState` separately tracks restart idempotence with four scalar
integers. `OpeningManagerPolicy` receives a fixed-shape padded candidate pool
and formation catalog once, before reset. Rule implementations satisfy the
same contracts, while `FootballWorld.policies` selects whether the default
player, in-match manager, full pre-reset opening manager, post-reset
opening-formation adapter, and set-piece axes are rule based.
The two opening switches are deliberately distinct: only the full policy owns
registration/XI/placement; the adapter cannot change an already-created roster.

## Opening roster transaction

`build_opening_policy_inputs` creates the pre-reset opening observation from
host `Player` / `PlayerProfile` candidates and per-team formation catalogs. It also returns an explicit authored-selection adapter input for callers that
want their supplied roster preserved. The built-in `RuleBasedOpeningManagerPolicy`
does not consume that adapter: it selects registration, the exact XI, a catalog
formation, and a unique candidate-to-slot placement from observable candidate
profiles and preferred positions. With roster sampling enabled, the builder
draws every candidate once in one identity-keyed vectorized operation; the
realized values are both what the policy sees and what match construction
receives. Candidate and catalog noise is keyed by player identity or formation
content, so reordering either axis only reorders the corresponding output.
Player ids are random-stream identities, never ability inputs.

The long-run render fixture instantiates its
`RuleBasedOpeningManagerPolicy` explicitly rather than relying on an implicit
environment default. Its metadata records the fully qualified policy class,
policy version, canonical-JSON configuration SHA-256, and the identity-level
decision for every valid candidate. The receipt includes the registered squad,
starting XI in both candidate and formation-slot order, and registered bench.
A replay therefore says which opening policy actually ran, not merely which
formation appeared after construction.

An opening decision is authoritative only through
`prepare_opening_match_inputs`. That host transaction validates registration
limits, starter subsets, unique slot coverage, goalkeeper requirements, and
formation geometry, then materializes ordinary starter `Player` tuples, bench
profiles, registered layouts, and their per-team priors. When the selected
physical layout is prepended as layout zero, its prior mass moves with it and
the duplicate alternative receives zero mass. `create_opening_match` performs
that transaction, calls `reset` and `initialize_management` without another
sampling key, and commits the selected physical opening pose before the first
frame. `create_opening_match_from_policy` calls a supplied opening policy
exactly once and immediately performs that transaction. Neither function
accepts a rollout or prior state, so neither can rewrite a mid-match checkpoint.
A rollout that begins mid-match keeps its existing roster, pose, and tactics.

Only this exact opening boundary may physically place players. Formation
commands during the match update tactical anchors and roles but do not
teleport players; ordinary player movement converges toward the new shape.
Candidate pools, bench tensors, opening policy parameters, and opening policy
memory never enter `step`, a rollout carry, or the managed scan.

Opening and in-match management also support 2v2 when the environment uses
`MatchConfig(minimum_team_players=(2, 2))`. Eleven-wide manager policy tables
are padded capacity, not a required active-player count.

Joint substitution/formation ownership, fixed-shape causal proposals, observed
roster abilities, preferred-position matching, and authoritative legality
checks share one explicit boundary. Scheduler identity remains separate from
learned recurrent state, rare policy/roster tensors stay outside the hot graph,
and opening roster materialization is a one-shot host transaction.

## Seeded routine substitution schedule

The fixed-shape proposal remains subject to environment-authoritative legality,
but the two teams are not synchronized on three exact wall-clock-like anchors.
The 57.7, 68.3, and 80.1 minute aggregates are transfer priors for ordering
only. Adjacent-anchor midpoints
form disjoint Voronoi boundaries; the first and last cells mirror half the
nearest gap and are bounded by halftime and regulation full time. A uniform
target inside each cell is keyed solely by immutable match key, team, and
window stage. The same seed is reproducible, team schedules diverge, and
batch/chunk evaluation order cannot consume or shift a draw.

These targets are stratified design priors, not a fitted conditional hazard.
Routine outgoing and incoming candidates use fatigue/booking/role and
profile-fit rankings without a deterministic argmax or low-slot tie. Eligible
scores are normalized within each candidate
set and sampled with an identity-keyed Gumbel draw. The match, restart, team,
window stage, command cell, choice type, and immutable player identity address
the draw, so slot reordering and batch/chunk evaluation order cannot consume or
shift it. The `0.30` temperature is an explicit design prior, not a fitted
lineup-conditioned choice model. Emergency goalkeeper selection remains
deterministic and fail-safe.

## Atomicity

`FootballWorld.manager_command` creates one state boundary for all four axes.
Within each team, every requested substitution cell succeeds or the entire
team batch is rolled back. Duplicate outgoing slots and duplicate bench slots
therefore fail closed. The opposing team is independent: an invalid team-0
batch does not cancel a legal team-1 batch. This matches separate referee
approval and prevents an opponent from cancelling a legal request by issuing
an invalid one.

Formation and taker results have their own applied masks. A rejected formation
or taker never aliases a clipped index and never rolls back already-authorized
substitutions. Eager invalid command-table addresses raise; traced invalid
addresses construct a no-op.

Substitution cells are validated as an unordered final roster. Padding cells
never scatter to slot zero, and a team may temporarily project zero
goalkeepers only because the acting-goalkeeper phase in the same transaction
repairs it. A final projection above one active goalkeeper is rejected. A
successful bench-goalkeeper substitution runs first and suppresses the acting
fallback.

Every committed identity or goalkeeper-role change also reconciles an active
restart before the manager API returns. If the old taker became unusable, the
shared restart selector chooses from the final roster. For `GK_HOLD`, the new
active goalkeeper, held-ball pose, and possession actor change atomically.
Rejected commands remain exact no-ops and therefore do not opportunistically
repair unrelated malformed input. Pending-taker liveness and explicit held-ball
state are maintained by the authoritative transaction.

## Emergency goalkeeper recovery

`FootballWorld.emergency_goalkeeper_command` is the deterministic default
proposal after a goalkeeper dismissal. When a registered bench goalkeeper and
substitution resource are available, it requests that first available keeper
and removes the most advanced active outfielder. It simultaneously nominates
the active outfielder nearest the own-goal centre as a fallback. The proposal
is optimistic about the current window and stoppage; the authoritative
transaction still validates both. `max_simultaneous` changes array shape and
must therefore be a Python-static value under JIT.

If the bench request is absent, unavailable, out of resources, outside a legal
window, or rolled back with another invalid cell, the field-player nomination
becomes the acting goalkeeper. An invalid explicit nomination is replaced by
the same nearest-own-goal rule and its event sets `environment_forced=True`. A
manager boundary with no explicit acting request also repairs a missing keeper
this way; this is deliberate global referee enforcement at a stoppage. It does
not consume a substitution, a window, or a slot generation. No repair occurs
during live play or after the player-count termination boundary. A valid
`GK_HOLD` is not a manager interrupt, but a malformed hold with no active
same-team goalkeeper opens a manager boundary. Without a managed runner,
direct `env.step` handles that malformed hold through its existing
corner-expiry fail-safe.

The rule manager treats that malformed `GK_HOLD` as a goalkeeper-emergency
boundary even though it remains excluded from ordinary formation, routine
substitution, and set-piece-taker decisions. It can therefore request the
bench goalkeeper first; the environment-level acting-player fallback remains
the final safety net if that request is unavailable or rejected.

The new active keeper receives the registered goalkeeper role and anchor. If
that anchor belongs to another active slot, roles, anchors, and physical poses
are exchanged; otherwise the keeper moves to the registered anchor. Every
administratively moved player is stopped without stamina charge. The final
state is then projected once through the configured stadium, ball, and body
restart geometry, so a penalty goalkeeper is already on the goal line in the
observation returned by the manager call.
If malformed manager data has no registered goalkeeper-role source, the
selected player keeps the current anchor instead of aliasing an arbitrary
slot.

## Identity generations and substitution events

A physical roster slot is reusable, so `player_id` alone does not describe its
continuity. `ManagerState.slot_generation` starts at zero and increments only
when a registered substitution commits. It lives outside `Rollout` and is
therefore absent from the default 10 Hz player-control and 80 Hz physics carries.

Single manager substitutions return one `SubstitutionEvent`. Combined manager
commands return the same tree with shape `(2, K)`. A committed cell records the
team, reusable slot, outgoing and incoming player ids, the new slot generation,
and the control tick. Rejected, unrequested, and team-atomic rollback cells use
sentinels and `occurred=False`; they can never appear as historical facts.

Call `roster_metadata_si(rollout, management)` after a managed substitution
when the result feeds the built-in rule policy. Use `roster_metadata` for a
normalized learned-model input. Its `slot_generation` vector is the cache
identity boundary: refresh only slots
whose generation changed. Calling `roster_metadata_si(rollout)` remains valid
for the rule policy; normalized `roster_metadata(rollout)` remains valid for
unmanaged rollouts and returns generation `-1`, meaning unmanaged. The
low-level `substitute` API
still reports `applied` but intentionally owns no cumulative ledger; a caller
that bypasses management also owns its external history.

An acting assignment returns one `ActingGoalkeeperEvent` per team. It records
the player id, reusable slot, unchanged slot generation, control tick, and
whether the environment had to replace the proposed choice.
`roster_metadata_changed` is true for either an identity substitution or an
acting-role change. Rebuild roster metadata on that signal, call
`refresh_policy_state` to update the goalkeeper role without erasing causal
memory for the same player, and apply fresh `observe_player_tactics_si` output so
the swapped manager anchor reaches the player policy.

Slot generation is manager-owned metadata rather than a retired-player ledger
in every physics state. Event streams preserve every outgoing identity without
sending an ever-growing history through each physics step.

## Rendering identity sidecars

Pass the generation trajectory to `render_mp4` through the optional
`slot_generations` argument and pass manager substitution results through
`substitution_events` and acting-role results through
`acting_goalkeeper_events`. `slot_generations` may be either the stacked
generation array or the per-frame manager states from which it is read. Both
event inputs may be stacked fixed-shape trees or per-frame sequences; use
`None` on frames without a manager command. Committed substitutions and
acting-goalkeeper assignments are serialized only to `event.json`. They do not
add an event feed or other overlay to rendered video.

If `slot_generations` is omitted, replay outputs write generation `-1` as an
explicit untracked sentinel. The renderer never guesses a generation or a
substitution from changed player ids. This fail-closed rule prevents an
unmanaged roster edit from being recorded as a historical fact.

Both inputs cross the device boundary with the other render payloads after the
rollout. They are host-side replay metadata, not fields in `Rollout` or inputs
to `step` / `step_with_events`, so enabling exact rendering identity adds no JAX
transition graph, compilation, or per-step rollout cost.

## Formation behavior and information boundary

Management initialization always registers the kickoff shape as layout zero.
Optional layouts are fixed attacking-frame slot anchors appended after it.
Changing formation updates `ManagerState.formation_anchor` and
`formation_role`; by itself it never changes a player's physical position or
velocity. Emergency goalkeeper role repair is the explicit exception described
above.
Both registered benches may be empty. In that case the substitution command
has width zero while formation and taker commands remain fully usable. A scalar
slot request cannot substitute for an explicit player identity. The shared
referee boundary may still assign an acting field goalkeeper when a team has
none.

`observe_player_tactics` exposes normalized model values and only the
observer's own-team anchors and roles.
Opponent entries are zero / `-1`, and no bench field enters the player view.
The manager-only view includes one fixed `[L]` row for valid catalog entries,
content signatures, configured priors, mean attacking depth, maximum width,
and defender fraction. Depth and width use the same immutable pitch scales as
other positions and restore exactly to SI units; invalid manager rows mask all
catalog values. The integer content signature is identity-like metadata for a
seeded rule draw, not a numerical formation feature. These small summaries let
a learned or rule manager select the existing catalog without carrying raw
`[L,N,2]` anchors into the rare decision graph.
The shipped rule policy applies `observe_player_tactics_si` through
`RuleBasedPolicy.apply_tactics` before subsequent policy steps, so its shape
movement targets change gradually
through ordinary locomotion. A learned player policy can consume the same
own-team side input without receiving the manager's bench observation or the
opponent's hidden catalog.

## Set-piece takers

A taker request is accepted only when its team and restart-kind cell matches an
already visible dead-ball restart and the selected slot is an active teammate.
A goalkeeper hold belongs to the player who physically caught the ball and
cannot be reassigned by any manager command. Requests for the current taker,
future or different restarts, and goalkeeper holds are idempotent no-ops.

A different legal taker receives the previous taker's release pose while the
previous taker receives the selected player's pose; both relocated players are
stopped. The ordinary restart projector then audits law, field, and the
configured oriented capsule clearance (default design prior
`0.50 x 0.20 m`). Identity and poses commit together only
when that complete layout is ready. A failed taker audit rolls back just this
taker axis while retaining substitutions, formations, and acting-goalkeeper
changes already authorized in the same manager transaction.

Independently of a manager proposal, each episode-step entry repairs a current
restart taker that is out of range, inactive, on the wrong team, or ineligible
for `GK_HOLD`. A successful replacement invalidates layout readiness, so the
new taker must be positioned and observed before release rather than consuming
the restart in that same frame.

The manager proposes and the environment authorizes legality. A taker update
also updates physical placement, preventing two actors from occupying one
release pose. Restarts created midway through a physics frame expose their
manager command boundary after the opening transition; the restart delay leaves
time to submit the designation before release while keeping the hot path small
and deterministic.

`make_rule_based_manager` supplies the default stochastic nomination. Its
score shares the environment fallback's restart-role propensity, distance,
readiness, and corner-target terms. The manager uses registered
formation roles, while the deterministic fallback infers roles from current
positions because management data deliberately stays outside the physics
carry. A Gumbel-max draw avoids choosing the same highest-scoring player in
every match; the environment remains deterministic when no manager is used.

The random contract is stateless. A dedicated taker stream is folded with the
immutable match key, absolute restart-opening control tick, team, restart kind,
and each `player_id`. Noise therefore follows player identity rather than a
reusable array slot. Reordering a roster, changing rollout chunk boundaries,
or evaluating another subsystem first cannot change the nominated identity.
The same match key and restart facts reproduce the same result exactly.

Use `env.observe_managers(...)` to obtain the two private normalized manager
rows, then call `manager.step(observations, match_key, manager_state)`. The
policy state records the last restart identity and last requested formation
change, making repeated calls during one stoppage idempotent and enforcing a
small hold interval. Candidate formation scores respond smoothly to score,
regulation progress, numerical balance, fitness, width, and the registered
prior. Attack depth, defender fraction, and width are ranked on the eligible
registered catalog range before those dimensionless terms are combined. This
keeps arbitrary pitch-scale catalogs responsive without adding a
rollout-dependent normalizer or a new coefficient. Every response weight and
the default five-minute hold are explicit design priors: DFL event feeds do not
identify tactical formation changes.
The initial kickoff has opening tick zero, so its management boundary remains
immediate but the in-match hold prevents a second formation rewrite directly
after the one-shot opening choice.

For an actual rollout, `make_managed_advance` detects that boundary without
putting the manager in `env.step`; `make_management_decision` then performs
the observation, stochastic decision, and authoritative transaction in a
separate low-frequency graph. Its output state is the one supplied to the next
managed chunk.

## Rule-based substitutions

The shipped manager also proposes low-frequency substitutions. Its scheduling
anchors are external compatibility priors, not a fitted conditional hazard
model. Provider data, derived aggregates, and the extraction pipeline are not
distributed. Fatigue, booking, and role weights remain explicit design priors.

Outgoing players are ranked by fatigue, booking, and role frequency. A bench
player is matched to the outgoing player's normalized ability profile because
FootballWorld does not invent an unobserved bench position label. Goalkeepers
are excluded from routine replacement. If the active goalkeeper is missing,
the first available registered goalkeeper takes priority and the nearest
own-goal outfielder is nominated as the legal acting fallback.

Substitution and taker draws use separate tagged streams. Both depend on the
restart identity and absolute match facts, so batching and chunking cannot
couple their randomness. The manager remains outside `step`; no bench tensor,
Gumbel draw, or substitution ranking enters the 10/80 Hz rollout graph.

`RestartState.opened_control_tick` is also the authoritative substitution-window
identity. Multiple commands during one restart can add simultaneous players
without consuming another window, even if control time advanced while the
ball remained dead. A later restart always has a different opening tick.

## Kickoff and rollout boundary

A coherent kickoff at control tick zero is an immediately open management
boundary, and its release gate remains zero substeps. This gate applies only to
a coherent kickoff; a malformed live-ball or incoherent tick-zero state still
fails closed. Management legality and kick release timing remain separate
predicates.

Management stays outside `step`, `step_with_events`, and the ordinary rollout
scans. `make_interruptible_advance` remains the smaller emergency-GK-only path.
`make_managed_advance` is the explicit all-restart path for stochastic takers
and routine substitutions. Both have a fixed maximum scan length and invoke no
manager callback inside the scan. Live-ball advantage and a valid `GK_HOLD` do
not interrupt. A malformed hold with no active same-team goalkeeper does open
the manager boundary; direct `env.step` instead proceeds through the
corner-expiry fail-safe. Minimum-player termination is not converted into a
manager request.

The host then submits a manager command, rebuilds roster metadata when
signalled, refreshes identity-scoped policy memory, applies player tactics, and
starts the next chunk with the same match key. The absolute control tick did not
advance through the frozen tail, so the next transition receives the same key it
would have received without chunking. Restart release remains fail-closed until
both teams again have an active goalkeeper. Bench tensors, the full restart
projection, and manager events therefore stay out of every 10 Hz player step.

The final design uses bench-goalkeeper priority, fixed-shape commands, restart
projection, same-stoppage window semantics, team-atomic substitutions, a narrow
kickoff gate, immutable taker state, manager-owned ledgers outside the hot
physics carry, and an explicit acting-goalkeeper event.
