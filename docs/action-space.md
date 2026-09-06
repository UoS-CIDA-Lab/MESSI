# FootballWorld action-space contract

FootballWorld uses exactly six selectable player intents:

`MOVE`, `CONTROL`, `PASS`, `SHOT`, `CLEAR`, and `CHALLENGE`.

This vocabulary is deliberately small. A football phrase describes an intent
only when it answers **why the player is acting**. The body part that contacts
the ball is an execution mechanism, what actually happens is an outcome, and
the law phase is restart context. Mechanisms, outcomes, fouls, and restart
names must not be added as top-level actions.

## Public representations

`IntentAction` is the sole public player-action contract. Its categorical
values are exposed as the public `ActionIntent` enum, and it contains one
integer intent per player plus eight continuous controls:

| Control | Shape | Meaning |
|---|---:|---|
| `move` | 2 | Team-relative movement direction and magnitude. |
| `force_to_ball` | 2 | Requested ball-contact direction and magnitude. |
| `launch` | 1 | Vertical launch control. |
| `spin` | 2 | Side-spin and back-spin controls. |
| `gaze_center` | 1 | Target view-centre yaw in the bounded torso-relative interval. |

All continuous controls remain bounded by the public normalized convention.
Movement is independent of contact: a player may move while requesting any
of the five contact intents.

The eight continuous controls are accepted and preserved for every intent,
including `MOVE`. Intent selects the attempted football purpose; it does not
mask a policy's continuous output head. Physics consumes ball-contact values
only when contact is both requested (or the restart clock forces it) and is
physically and legally realizable. Unconsumed values are inert rather than
rewritten to zero.

`gaze_center` is not a periodic world angle. Its normalized scalar `[-1, 1]`
selects a target inside the configured torso-relative yaw interval, which is
strictly smaller than a full turn. The realized `gaze_yaw` state slews toward
that target at a bounded rate and therefore has no 0/2pi action seam. The
default 540 deg/s bound moves at most 54 degrees per 10 Hz control frame; it is
a deliberately responsive head-eye-attention prior, not a calibrated neck-only
biomechanical constant.
The torso-relative half-range must also remain strictly below 180 degrees: at
exactly 180, normalized `-1` and `+1` encode the same direction as `-pi` and
`+pi`, recreating a scalar endpoint seam.

During `MOVE`, a non-zero `force_to_ball` direction instead supplies the body target;
if it is zero, non-zero `move` supplies the fallback body target. During a
contact intent, `force_to_ball` remains solely the ball-force direction and the
body keeps its previously prepared target, preserving backheel and lateral
contact geometry. Movement, body direction, contact force, and view centre are
therefore distinct without adding another categorical intent.

There is no selectable `UNKNOWN` action. Invalid explicit integers fail closed
to `MOVE`. `INTENT_TACKLE` is an event-schema alias for `CHALLENGE`, and
`INTENT_CATCH` is a reserved realized goalkeeper-event value outside the six
selectable actions.

## The six intents

| Intent | Includes | Does not guarantee |
|---|---|---|
| `MOVE` | Running, jogging, turning, holding shape, marking, pressing without attempting contact, and deliberately leaving the ball. | That the player avoids incidental passive collision. |
| `CONTROL` | Receiving, trapping, cushioning a first touch, a dribble touch, retaining possession, and requesting a lawful goalkeeper claim. | Successful control or a goalkeeper catch. |
| `PASS` | Ground and lofted passes, through balls, crosses, cutbacks, back-passes, throw-ins, and distribution to a teammate. | A particular receiver, mechanism, or completed pass. |
| `SHOT` | Any deliberate attempt to score, including a headed attempt when the ball height selects the head mechanism. | A foot strike, on-target trajectory, or goal. |
| `CLEAR` | Deliberately removing danger: a defensive kick, headed clearance, or goalkeeper hand-clear request. | Possession loss, distance, or a particular body part. |
| `CHALLENGE` | A deliberate ball-winning contest, standing tackle, lunge abstraction, or interception attempt against an eligible opponent play. | Winning the ball, making contact, or avoiding a foul. |

A pressure run is `MOVE` until the player deliberately contests the ball.
An incidental block or ricochet is not `CHALLENGE`: the actor may have selected
`MOVE`, while physics records a passive-body mechanism and a deflection.

Football terms map by tactical purpose rather than by animation. A header
towards goal is `SHOT`, a headed layoff is `PASS`, and a headed defensive play
is `CLEAR`. A cross is `PASS`, not a seventh action. Repeated `MOVE` plus
`CONTROL` frames express dribbling without introducing `DRIBBLE`.

## Execution mechanism

The dynamics layer selects a mechanism only after checking phase, designated
restart taker, activity, reach, ball height and speed, and recovery locks.
Selecting an intent cannot bypass those checks.

| Mechanism | Current meaning |
|---|---|
| `NONE` | No realized ball contact. |
| `FOOT` | Active foot contact. A `CONTROL` touch may add bounded dribble energy; `PASS`, `SHOT`, and `CLEAR` may add the larger strike energy and spin budget. |
| `HEAD` | Height-selected head contact. Explicit `CONTROL` can deliberately redirect the ball, but does not assign possession or create kick-like energy. |
| `CHEST` | Height-selected torso contact. Explicit `CONTROL` can cushion the ball, record `TRAP`, and assign possession without creating kick-like energy. |
| `PASSIVE_BODY` | Unrequested swept collision with the represented body geometry. |
| `GOALKEEPER_HAND` | Lawful goalkeeper claim or forced hand-clear abstraction inside the own penalty area. |
| `THROW` | The designated throw-in release. It is the explicit non-foot release exception. |

`CONTROL` with a lawful goalkeeper in handling range requests the goalkeeper
hand mechanism. A later `CATCH` or `PARRY` is an outcome, not a selectable
action. `CLEAR` can require a goalkeeper parry, but it does not create a
separate `PUNCH` action.

For an outfielder, explicit `CONTROL` selects `FOOT`, `CHEST`, or `HEAD` from
ball height. A chest attempt uses the player's relative incoming velocity to
form a passive cushioning target and the bounded physical contact impulse to
approach it. A descending vertical component may first be absorbed passively
down to the non-upward target; horizontal redirection and any upward reversal
still consume that impulse budget. This makes a descending lob easier
to cushion than a flat driven ball of the same total speed without adding a
coefficient. Only a fully reached, energy-feasible target records `TRAP` and
assigns possession. Otherwise the partially cushioned ball remains
free, records `MISCONTROL`, and does not reset Law 11; the rule policy then
continues through its ground or predicted-landing chase path. A head-height
`CONTROL` only performs an energy-neutral redirect, records `MISCONTROL`, and
does not assign possession. Neither body mechanism can add ball energy as a
foot strike can.

Evidence boundary: *Relationship between acceleration of the foot and
lower-leg movement in soccer ball trapping* and *Kinematic factors associated
with first-touch control and trap-to-pass transition in football during a
dynamic receive-to-pass task* support the general mechanics of first-touch
cushioning, but both study foot trapping. Neither supplies a fitted chest
incidence-angle law. The descending-component treatment is therefore a
mechanics-informed design prior, not a fitted chest-angle law. Available DFL
body-reception summaries cannot identify such a curve because they lack
player velocity, the contact normal, and sustained same-player possession
needed to separate incidence angle from a successful controlled reception.

A foot-height `CONTROL` is different: it drives the ball toward the player's
velocity plus the requested player-relative exit velocity. That relative
request is capped by `control_request_speed_max_mps` (currently 9 m/s), while
the available change is bounded by the physical foot-contact impulse limit.
It is independent of `ball_control` once this actor has established contact.
The ability coordinate instead changes relative actor selection when multiple
eligible players contest the ball. This supports an advancing dribble touch
without granting the full kick semantics or new spin of `PASS`, `SHOT`, or
`CLEAR`.

Open-play possession is retained only while `verified_controlled_carrier`
confirms an active same-team actor within the configured carry radius and
control height. If a release leaves that physical envelope, the same
transition clears `team`, `player`, and `control_ticks` while preserving
contact provenance and recording the previous team.

## Realized outcome

The engine records one of these results independently of the requested intent:

| Outcome | Meaning |
|---|---|
| `NONE` | No resolved effect. |
| `RELEASE` | A deliberate release of the ball. |
| `TRAP` | Control succeeded. |
| `INTERCEPTION` | A challenge intercepted a released opponent ball or continued onto an opponent `CONTROL/TRAP` that had already become physically loose. |
| `TACKLE_WON` | A challenge won against a controlled carrier. |
| `DEFLECTION` | Contact redirected the ball without controlled possession. |
| `CATCH` | The goalkeeper gained hand control. |
| `PARRY` | The goalkeeper redirected the ball without catching it. |
| `FOUL` | An eligible `CHALLENGE` against a verified carrier resolved as a direct-contact foul fact. Its occurrence rate is an explicit prior; discipline is sampled conditionally and adjudicated by Law 12. |
| `MISCONTROL` | A control attempt failed. |

Thus `CHALLENGE` is the request, `FOOT` or another height-selected body part is
the mechanism, and `TACKLE_WON`, `INTERCEPTION`, `DEFLECTION`, `FOUL`, or
`NONE` is the result. These labels are not interchangeable. Law 11 effect is
also recorded separately because a deliberate play, deflection, save, or
direct-restart exemption can have different offside consequences.

### Law 11: intent is evidence, not authority

Offside state is updated from the realized `ContactOccurrence`, never from the
requested intent alone. Deliberate and passive contacts share one swept
player/ball timeline, so an explicit contact is resolved only at the first
relative-path intersection that passes the phase, reach, height, recovery, and
mechanism-specific gates. The incoming-speed cap applies to foot contact; body
redirection and goalkeeper handling retain their own response limits. If no
deliberate contact is realized, a later swept body collision remains
`PASSIVE_BODY + DEFLECTION` and does not reset offside even though the policy
requested `CONTROL`, `PASS`, `SHOT`, `CLEAR`, or `CHALLENGE`.

The active actor is excluded for the remainder of the current physics substep.
In later substeps, the latest deliberate-contact actor is suppressed only while
its active-contact lock remains and the ball is either still inside its literal
shape or moving radially outward from the player's reference point. This treats
a release from between the modeled legs and its outward passage past the
kicking leg as one contact episode. The lock is not a whole-reach suppression
envelope: an inward rebound remains collidable even during
the lock, and another contact or lock expiry restores ordinary passive
collision for `MOVE` and every other intent. A restart release also ignores
only a pre-existing body overlap until the ball first exits that shape; it does
not suppress a later block whose swept segment begins outside the body.

Once the deliberate-contact solver has accepted a genuine opportunity to play
the ball, an inaccurate result does not by itself make the play accidental.
Failed foot control and a deliberate headed redirect therefore reset Law 11.
A chest attempt that cannot reach its energy-feasible cushioning target is
instead a no-reset `MISCONTROL`: the intent remains observable, but the
physical response did not establish a controlled play. The other no-reset
cases are passive/realized `DEFLECTION`, a failed limited `CHALLENGE` poke or
block, and a contact that actually prevents a goal threat (a deliberate save).
A non-goal-threatening goalkeeper parry is deliberate
play and does reset. This follows the IFAB distinction between deliberate play,
deflection, and deliberate save without letting a semantic label override
physics.

## Restart context

Restart kind constrains which of the same six intents may be activated. It is
context, not another action. Only the designated taker can release the ball,
and timing and direction legality remain authoritative.

| Context | Structurally selectable contact intents | Current rule-policy choice | Mechanism notes |
|---|---|---|---|
| Open play | `CONTROL`, `PASS`, `SHOT`, `CLEAR`, `CHALLENGE` | Situation-dependent | Height, role, reach, speed, and contest context select the mechanism. |
| Kick-off | `PASS`, `SHOT`, `CLEAR` | `PASS` | Foot; restart direction and goal rules still apply. |
| Throw-in | `PASS` | `PASS` | `THROW`; a throw-in is not a separate top-level intent. |
| Goal kick | `PASS`, `SHOT`, `CLEAR` | `PASS` | Foot. |
| Corner kick | `PASS`, `SHOT`, `CLEAR` | `PASS` | Foot. |
| Free kick | `PASS`, `SHOT`, `CLEAR` | Seeded `PASS` or `SHOT` when direct; `PASS` when indirect | Foot; distance, visible goal angle and goalkeeper position influence direct-shot probability. |
| Penalty kick | `PASS`, `SHOT`, `CLEAR` | `SHOT` | Foot; penalty-specific direction and restart rules remain authoritative. |
| Offside restart | `PASS`, `SHOT`, `CLEAR` | `PASS` | Foot and indirect-restart context. |
| Goalkeeper hold | `PASS` | Environment-forced `PASS` | Currently a foot punt/kick after the configured delay. |

`MOVE` remains the fail-closed value in every phase. During an active restart,
restart positioning may own non-taker poses even though their categorical
action remains `MOVE`.

When the three-second environment release becomes due, submitted continuous
kick values remain authoritative regardless of the submitted intent. A legal
submitted `PASS`, `SHOT`, or `CLEAR` label is retained; an incompatible label
falls back to `SHOT` for a penalty and `PASS` for other restarts. This changes
neither the continuous parameters nor the designated-taker and contact gates.

## Coverage and honest limitations

The following football behaviours fit the six-intent taxonomy, but their full
physical execution is absent or deliberately coarse. Classification here does
not claim implementation.

| Football behaviour | Existing intent classification | Current support and missing physics |
|---|---|---|
| Deliberate outfield handball | Use the tactical purpose (`CONTROL`, `PASS`, `SHOT`, `CLEAR`, or `CHALLENGE`), never a `HANDBALL` action. | Not physically requestable. The Law 12 adjudicator can consume an externally established handball fact, but rollout physics does not generate the deliberate hand/arm act. |
| Pushing, grabbing, holding, or an off-ball professional foul | `CHALLENGE` when deliberately contesting an opponent. | Player overlap separation exists, and the adjudicator accepts classified contact/holding facts, but there is no authored push/grab force or autonomous fact extractor for these acts. |
| Sliding tackle | `CHALLENGE` | Challenge reach, contest outcome, and recovery are represented; a grounded pose, swept legs, momentum transfer, and slide trajectory are not. |
| Jump or aerial leap | Purpose-dependent: commonly `SHOT`, `PASS`, `CLEAR`, or `CHALLENGE`; an off-ball jump remains `MOVE`. | There is no separate jump action or persistent vertical pose. Eligible positive-effort head/GK attempt geometry determines an effort-scaled lock, including a reachable attempt that loses its contest or misses the final contact. The lock blocks every deliberate ball intervention during recovery while movement and passive collision remain available; replay derives a visual rise/fall effect from that lock. |
| Bicycle, scissor, or overhead kick | `SHOT` or `CLEAR` by purpose. | No pose or elevated-foot mechanism. High contacts resolve through the coarse height-selected chest/head model instead. |
| High-foot volley | `PASS`, `SHOT`, or `CLEAR` by purpose. | No articulated leg or high-foot collision volume; foot contact is constrained by the coarse foot-height gate. |
| Goalkeeper dive | `CONTROL` for a claim or `CLEAR` for a parry. | There is no separate dive direction or pose action. Horizontal/vertical reach effort scales the shared all-deliberate-contact recovery lock up to the goalkeeper maximum; movement and passive collision remain enabled, and replay visualizes the recovery interval. The rule policy can arm the intent from a one-control-frame observed relative-path crossing, but swept physics still decides whether contact occurs. |
| Goalkeeper punch | `CLEAR` | A required goalkeeper-hand parry is available as an abstraction; fist contact geometry and punch-generated impulse are not. |
| Goalkeeper hand throw after a catch | `PASS` | Not implemented. `GK_HOLD` currently permits and forces a foot release; only a throw-in uses the `THROW` mechanism. |
| Shielding with active body force | `CONTROL` while retaining the ball, otherwise `MOVE`. | Torso geometry blocks overlap, but there is no requested shoulder/hip force or shielding foul model. |
| Feint or step-over | `MOVE`, optionally followed by `CONTROL`. | It can be approximated through trajectories and touches, but there is no articulated skill mechanism. |

These gaps should be filled, if needed, by adding execution state, physical
mechanisms, or rule fact extraction under the existing intent. They are not a
reason to expand the top-level action taxonomy or introduce unsupported tuning
coefficients.

## Logging and dataset rule

Logs preserve requested intent, its provenance, realized mechanism, realized
outcome, restart kind, and Law 11 effect as separate fields. Provenance is one
of `NONE`, `POLICY`, or `ENVIRONMENT_FORCED`; these are metadata values, not
selectable actions. `step_with_events` retains one compact requested-intent
trace per control frame while each realized contact retains its own provenance.
The lean `step` path does not materialize that trace.

`step_with_events` additionally returns a fixed-shape per-player
`ActionReceipt` under schema `footballworld.action-receipt/1`. Policy agency
and realized action remain separate. Instead of one ambiguous mask, sanitized
input, structural availability,
independent contact predicates seen during the frame, attempted and realized
contact, parameter use, forced release, referee projection, environment
overwrite, and terminal suppression remain separate flags. `primary_reason` is
only a display summary. `parameter_consumed` has independent move, force
direction, force power, launch, spin, and gaze bits; unused values are never
presented as training labels merely because the intent was available.

The same receipt contains a non-exhaustive displacement-source bit mask for
self-motion, collision velocity response, overlap separation, referee
projection, and the halftime reset. Multiple sources may coexist. Collision
and separation bits are deliberately conservative: they identify only the
retained strongest cross-team body-impact participants from each physics
substep, rather than inferring additional contacts from endpoint displacement.
They may therefore omit secondary simultaneous or same-team separations. A
zero collision/separation mask means "not causally attributed by this compact
receipt", not "the player certainly experienced no collision displacement".
Making it exhaustive would require an eventful-only all-participant solver
sidecar; widening the lean collision carry is explicitly not part of this
schema. The receipt is privileged logging/evaluation data and is never
inserted into a player observation.

Host consumers use `footballworld.action-trace/1` and
`footballworld.action-receipt/1`. Their typed specs and layout fingerprints are
available through `action_trace_spec` and `action_receipt_spec`; corresponding
`flatten_*` and `unflatten_*` helpers preserve every structured dtype.

The numeric provenance contract is `footballworld.intent-source/2`:
`NONE=0`, `POLICY=1`, and `ENVIRONMENT_FORCED=2`. Player observations
carrying `last_contact.intent_source` use SI observation schema version 10.
Replay events use `footballworld.events/14` and declare the provenance schema
and ordered names in their header. Tracking rows carrying the same last-contact
field use `footballworld.tracking/8`.

Exact replay capture also retains the eight submitted continuous controls for
the sparse set of non-`MOVE`, environment-forced, or realized-contact actors.
This bounded trace supports contact and cross-signature auditing; it is not a
complete action replay. In particular, omitted `MOVE` controls remain unknown
rather than being reconstructed as zero.

A nutmeg is not a seventh intent and does not change the ball state. The
eventful step may label a clean, low crossing through the fixed triangular leg
gap only when the latest causal contact belongs to a valid opposing deliberate
actor. Failed contact requests do not suppress the label, but a realized
contact, woodwork hit, boundary crossing, or ambiguous ground rebound in the
same physics substep does. This conservative omission policy avoids deriving a
false passage from a trajectory chord that was bent inside the substep. Exact
nutmeg fields are written only through the event sidecar to `event.json`; the
lean step neither computes nor returns them.
