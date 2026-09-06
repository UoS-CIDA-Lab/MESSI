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
| `env.player_observation_spec(observation)` | immutable leaf semantics, shapes, dtypes, scales, visibility rules, version, and layout fingerprint |
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
| `self_state` | observer index, team-local position and velocity, and gaze yaw |
| `players` | slot-wise relative kinematics, facing, gaze, stamina, recovery locks, discipline, activity, actor roles, visibility, and conservative contact availability |
| `ball` | observer-relative position, velocity, spin, live state, and visibility |
| `possession` | current and previous team, control duration, resolvability, and latest possession-producing contact |
| `restart` | public restart kind, team, countdown, and indirect status |
| `restart_release` | executed-restart provenance and whether its actor remains resolvable |
| `match` | attack direction, kickoff team, exact score, control tick, handling restrictions, and causal match clock |

The player view never contains bench identities or substitution resources.
Those are private manager information and are exposed only through manager
observations. `env.global_state_view` is a separate centralized view and is not
filtered through one player's visibility.

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

## Full and view-limited observation

`Perception.limit_by_view_angle=False` is the default. Every active player and
the ball are visible, and the unused angle does not create an additional
compiled configuration family.

When view limiting is enabled, visibility is an angular horizontal aperture
around the actual body-forward direction plus gaze yaw. The default configured
aperture is 160 degrees. It is a documented design prior, not a measured human
vision constant. The current visibility model is angular only: it has no
distance limit and no occlusion model.

The observer is always visible to itself. Inactive slots are not visible.
Hidden kinematics and actor-linked fields are zeroed or replaced by canonical
sentinels, while explicit `visible` and `known` leaves distinguish hidden facts
from genuine zero or `NO_*` values.

`known` for possession or restart provenance means that the current causal
actor can be resolved in the current view. It does not claim that the policy
observed the original historical event. Perceptual memory belongs in policy
recurrent state rather than being invented in the environment carry.

The `contact_may_occur_this_frame` affordance deliberately remains
conservative for hidden slots. It uses public phase, team, activity, and role
facts without leaking a hidden actor identity or private recovery timer.

## Fail-closed subject selection

An invalid observer index returns a finite, fully masked observation with
`valid=False`; it cannot alias another player. Eager host inputs reject shapes
other than a scalar and treat booleans, fractional values, and out-of-range
integers as invalid subjects.

Inside compiled code the subject contract is scalar `int32`. Callers must
range-check and cast wider host integers before entering `jax.jit`, or use
`observe_all`, because JAX may canonicalize integer widths before the
environment receives the value.

## Schema boundary

The SI player observation and normalized player observation are independently
versioned. At this frozen preview revision:

| Contract | Version |
| --- | ---: |
| SI player observation | 10 |
| normalized player observation | 7 |

Consumers should persist the named schema and the layout fingerprint returned
by `player_observation_spec`. Hard-coded flat offsets are not a public
contract. Flattening preserves float, integer, and boolean blocks separately,
and unflattening validates dtype, size, leading axes, structure, and contract
identity.

## SoccerWorld baseline and FootballWorld decision

FootballWorld inherits two sound SoccerWorld behaviors:

- observation assembly is a pure function of state; and
- omitting observation work is a performance choice that does not change the
  authoritative transition.

It also retains fixed shapes and the no-runtime-clipping normalization rule.
Those properties support reproducible JAX compilation and prevent a model-view
choice from changing match physics.

FootballWorld rejects SoccerWorld's authoritative dense float vector and its
inclusion of bench attributes in every player observation. The replacement is
a structured fixed-shape PyTree, explicit typed flattening, and a separate
manager-only bench view. This change prevents categorical and mask dtype loss,
keeps private substitution information outside player policies, and makes
partial-observation unknown values distinct from real zeros. Optional angular
visibility and its `visible`/`known` contract are therefore explicit rather
than inferred from one flat token layout.

## Related contracts

- [Model output and normalization](../model-output.md)
- [Action space](../action-space.md)
- [Match clock](../match-clock.md)
- [Tracking-view augmentation](../tracking-view-augmentation.md)
