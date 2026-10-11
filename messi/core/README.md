# Core: match state and control contracts

`messi.core` defines actions, the authoritative dynamic state, rosters, physics, and match rule transitions for a control frame. See [`messi/environment`](../environment/README.md) for training and evaluation interfaces and [`messi/policies`](../policies/README.md) for built-in decision-making.

## Action

`IntentAction` contains a categorical intent and normalised continuous controls for each player.

- The five intents are `MOVE`, `CONTROL`, `PASS`, `SHOOT`, and `CHALLENGE`.
- Continuous controls comprise two movement axes, two ball-force direction axes, one launch axis, and two spin axes.
- Continuous values are decoded into physical controls according to the action schema's ranges and conversion rules.
- `SHOOT` covers both attempts on goal and defensive clearances. There is no separate `CLEAR` intent.

`IntentAction.from_array`, `IntentAction.move_only`, and `neutral_action` are public ways to construct correctly shaped actions. See [`action.py`](action.py) for the schema and constants.

For `P` players in one match, `IntentAction.from_array(intent, continuous)` expects shapes `(P,)` and `(P, 7)`, respectively. Intents must have an integer dtype other than Boolean; returned intents are `int32`. Continuous values become `float32` and are clipped to `[-1, 1]`. `NaN` and either infinity become zero; out-of-range integer intents become `MOVE`. Invalid intent dtypes or input shapes raise an exception.

The shapes of `move`, `force_to_ball`, `launch`, and `spin` are `(P, 2)`, `(P, 2)`, `(P,)`, and `(P, 2)`. Each non-zero two-axis control is decoded into an L2 unit direction and an L-infinity norm magnitude. `launch` maps from `[-1, 1]` to `[0, 1]`. These are normalised commands, not distances or velocities.

## State and roster

`State` is the dynamic carry updated throughout a match. It includes the ball, players, possession, restarts, score, clock ticks, and rule provenance. Observations, event receipts, and the rollout-invariant roster are passed separately.

`TeamRoster` and the realised roster contain player identities and abilities that persist during a rollout. The `reset_environment` function described in the [environment guide](../environment/README.md) creates a new match.

For one match, `state.players.position` and `velocity` each have shape `(P, 2)`; `state.ball.position` and `velocity` each have shape `(3,)`. Positions are in metres and velocities in metres per second; the ball's third coordinate is height. `state.attack_direction` specifies each team's attacking x-direction. The environment observation path converts these to team-relative coordinates for policies.

## Time resolution

The default `Timebase` uses an 80 Hz physics timestep and a 10 Hz control rate, so each control action lasts eight physics substeps. If you change `Timebase`, pass the same value to the match clock configuration and transitions.

## Transition paths

- `step_control`: physics, contact, and rule transitions for one control frame, without event receipts.
- `step_control_with_events`: the same transition with compact event receipts for evaluation and analysis.
- `step_control_replay`: applies stored contest and body-foul overrides and reports replay validity.
- `transition_match`: a match-aware transition that includes the clock, half-time, and end conditions.

Transition actions must have the roster's player axis, and transitions take an explicit PRNG key. Use the `advance_control` family in `messi.control_rollout` for time-axis rollouts and `advance_control_batch` for independent match batches. Consult those functions for multi-device mapping and return contracts.

`step_control` and `advance_control` handle control-frame physics and rules. For half-time, match completion, and match rewards, use `transition_match` or the match-aware `make_managed_advance` path.

## Module map

| Area | Main files |
| --- | --- |
| Actions, state, time | `action.py`, `state.py`, `timebase.py` |
| Rosters and player movement | `roster.py`, `movement.py`, `player_step.py`, `stamina.py` |
| Ball, contact, collisions | `ball.py`, `contact.py`, `collision.py`, `passive_contact.py`, `contest.py` |
| Match rules and restarts | `boundary.py`, `offside.py`, `foul.py`, `discipline.py`, `restart.py`, `goalkeeper.py` |
| Control-frame integration | `transition.py` |

Consult each module's public API and implementation for detailed defaults, boundary handling, and failure behaviour.
