# Player observation contract

This page is the normative description of player-facing observations owned by
[`src/footballworld/environment/observation.py`](../../src/footballworld/environment/observation.py)
and their model projection in
[`src/footballworld/environment/normalization.py`](../../src/footballworld/environment/normalization.py).
The environment's recurrent `Rollout.state` remains the authoritative SI-unit
state; an observation is a pure, fixed-shape projection and never mutates that
state.

## Public entry points

| Method | Result |
| --- | --- |
| `env.observe(rollout, observer_index)` | one normalized player observation |
| `env.observe_all(rollout)` | normalized observations with a leading observer-slot axis |
| `env.observe_si(rollout, observer_index)` | one SI-unit observation used by the shipped rule policy |
| `env.observe_all_si(rollout)` | all SI-unit player observations |
| `env.restore_observation(observation)` | restores the SI values represented by a normalized observation |
| `env.player_observation_spec(observation)` | immutable leaf semantics, shapes, dtypes, scales, sentinel rules, version, and layout fingerprint |
| `env.flatten_observation(observation)` | typed float, integer, and boolean blocks plus the authoritative layout |

Observation construction is separate from `env.step` and
`env.step_with_events`. A learner may compose the desired observation call with
its own JIT boundary; training that needs only recurrent advancement does not
pay for an unused model view.

```python
one = env.observe(result.rollout, observer_index=0)
all_players = env.observe_all(result.rollout)
si_one = env.observe_si(result.rollout, observer_index=0)
restored = env.restore_observation(one)
flat, layout = env.flatten_observation(one)
```

## Structured tree

`Observation` is a typed PyTree rather than one undifferentiated float vector.
Its top-level leaves are:

| Group | Meaning |
| --- | --- |
| `valid` | whether the requested observer index identifies a valid roster slot |
| `self_state` | observer index and team-local position and velocity |
| `players` | slot-wise relative kinematics, facing, stamina, recovery locks, discipline, activity, actor roles, and conservative contact availability |
| `ball` | observer-relative position, velocity, spin, and live state |
| `possession` | previous controlled team, control duration, and latest possession-producing contact |
| `restart` | public restart kind, team, countdown, and indirect status |
| `restart_release` | executed-restart kind, untouched/indirect state, and release mechanism |
| `match` | exact score, control tick, law exemptions, handling restrictions, and causal match clock |

The player view never contains bench identities, registered bench role
preferences, or substitution resources. Those are private manager information
and are exposed only through manager observations. `env.global_state_view` is a
separate centralized view and is a distinct centralized contract.

## Coordinate frame

Each observer receives positions and velocities in that observer's current
team-attacking frame. The observer anchors relative entities, while the away
team's local frame is rotated consistently with its attack direction. Ball
height remains vertical. This makes the observation equivariant to home/away
orientation without editing the authoritative world state.

The structured SI observation and normalized model observation preserve the
same hierarchy. Normalization uses only immutable environment configuration,
does not clip runtime values, and retains discrete counts, categories, masks,
and identities as integer or boolean leaves. Full scale definitions and typed
flattening are documented in the
[model-output contract](../model-output.md).

## Full-information observation

Every valid observer receives every fixed roster slot, the ball, and all public
causal facts. The top-level `valid` bit alone distinguishes the finite
invalid-observer sentinel; redundant per-entity `visible` and per-fact `known`
leaves are absent.

The `contact_may_occur_this_frame` affordance uses current public phase,
activity, recovery, and role facts without reading future outcomes.

## Fail-closed subject selection

An invalid observer index returns a finite, fully masked observation with
`valid=False`; it cannot alias another player. Eager host inputs reject shapes
other than a scalar and treat booleans, fractional values, and out-of-range
integers as invalid subjects.

Inside compiled code the subject contract is scalar `int32`. Callers must
range-check and cast wider host integers before entering `jax.jit`, or use
`observe_all`, because JAX may canonicalize integer widths before the
environment receives the value.

## Tactical side input

`env.observe_player_tactics` is a separate, low-frequency own-team side input.
It exposes the current `formation_index`, slot-wise `formation_anchor` and
`formation_role`, plus `tactical_epoch` and
`formation_changed_control_tick`. Opponent assignments are masked to zero /
`-1`, and no bench field enters this tree. Player policies can therefore reset
or gate role-conditioned memory exactly when a new formation is accepted without
adding manager data to every physics transition.

`formation_role` is the slot's current tactical assignment. It is distinct from
a bench profile's manager-only `preferred_roles`, which is registered roster
metadata and may be unknown. Substitution preserves the on-pitch slot assignment;
the built-in manager first prefers a substitute whose declared compatibility
contains that role and falls back to the existing ability-profile ranking when
no compatible declaration is available.

## Schema boundary

The SI player observation and normalized player observation are independently
versioned. At this frozen preview revision:

| Contract | Version |
| --- | ---: |
| SI player observation | 13 |
| normalized player observation | 10 |

Consumers should persist the named schema and the layout fingerprint returned
by `player_observation_spec`. Hard-coded flat offsets are not a public
contract. Flattening preserves float, integer, and boolean blocks separately,
and unflattening validates dtype, size, leading axes, structure, and contract
identity.

## FootballWorld observation design

FootballWorld guarantees two core behaviors:

- observation assembly is a pure function of state; and
- omitting observation work is a performance choice that does not change the
  authoritative transition.

It also retains fixed shapes and the no-runtime-clipping normalization rule.
Those properties support reproducible JAX compilation and prevent a model-view
choice from changing match physics.

FootballWorld rejects the authoritative dense float vector and its
inclusion of bench attributes in every player observation. It also keeps the `role_pos` cumulative mean out of authoritative physics state: that
value mixes a tactical-epoch activity summary with an assignment and needs an
extra sample count to remain Markov. The replacement separates current tactical
assignment (`formation_anchor`, `formation_role`, epoch and change tick),
manager-only registered compatibility (`preferred_roles`), and public manager
shape summaries (`team_centroid`, `team_spread`).

The structured fixed-shape PyTree, explicit typed flattening, and separate
manager-only bench view prevent categorical and mask dtype loss, keep private
substitution information outside player policies. Full-view player observations use the top-level `valid` bit to distinguish
invalid finite sentinels from real values; no angular visibility or redundant
per-fact availability bits remain. A richer
per-player deployment history, if needed by a learned policy, belongs in that
policy's recurrent state keyed by `player_id`, `slot_generation`, and
`tactical_epoch`, not in the environment's physics carry.

## Related contracts

- [Model output and normalization](../model-output.md)
- [Action space](../action-space.md)
- [Match clock](../match-clock.md)
