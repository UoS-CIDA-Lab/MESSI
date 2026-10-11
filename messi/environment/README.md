# Environment: match lifecycle and policy inputs

`messi.environment` provides match reset, observations, match timing, and management boundaries. See the [core guide](../core/README.md) for physics and rule transitions and the [policies guide](../policies/README.md) for built-in rule-based policies.

## Starting a match

`reset_environment(key, ...)` returns a `ResetResult` containing the realised roster, bench, manager state, initial dynamic state, half-time setup, and selection results for one match. By default, it creates a roster with the standard number of players per team. You can also supply `TeamRoster` and eligible candidates.

The principal `ResetResult` fields are `state`, `roster`, `bench`, `manager`, and `setup`. Pass `state` to subsequent transitions, `roster` for player identity and abilities, and `setup` to `transition_match` for half-time handling.

Reset takes an explicit PRNG key. Record the key and input configuration for reproducibility. Changing the roster or reset options can change match conditions and the distribution of results.

## Observation

- `observe_all(state, roster)` builds a shared scene with player-relative positions and velocities, team-relative coordinates, ball information, and action readiness.
- `observe_policy(state, roster)` adds player abilities and roles, offside information, and goalkeeper information to produce a `PolicyObservation`.
- The `intent_availability` functions compute a mask of available intents from the state.
- `flatten_observation`, `unflatten_observation`, and the normalisation functions provide explicit control over the observation layout and normalisation settings.

`PolicyObservation` currently contains full information; it is not a partial-observation interface. If preprocessing observations for a neural network, retain the public layout and schema constants alongside the model.

For `P` players in one match, `SharedScene.relative_position` has shape `(P, P, 2)` for observer × target × xy. `position_team` has shape `(P, 2)` and orients the x-axis so that each player's team attacks toward +x. The default `observe_all` output uses unnormalised match units. Apply model-input normalisation explicitly with `make_observation_normalization` and `normalize_policy_observation`.

## Match progression and completion

`transition_match` performs one match-aware frame transition. `step_environment` returns the transition result together with a shared observation of the new state. `match_status` checks whether the current state is finished without taking another physics step.

The default `MatchConfig` sets half-time at 45 minutes, full-time at 90 minutes, up to 10 minutes of added time per period, a 15-minute half-time interval, and a minimum of seven players per team. A match can end if a team falls below the minimum or no restart taker is available. The relevant `MatchStatus` and `MatchTransition` fields report the reason.

`MatchTransition` returns `state`, two-team `reward`, `terminated`, `truncated`, `done`, `halftime_reset`, and `termination_reason`. For a one-frame score change `Δscore`, team 0 receives `2×Δscore[0]−Δscore[1]` and team 1 receives `2×Δscore[1]−Δscore[0]`. `terminated` indicates conditions such as too few players or a missing restart taker; `truncated` indicates the match time limit. `done` is true for either condition or if the caller passes `already_done=True`. An already-finished input freezes the state and yields zero reward.

Use the request/result API in `management.py` for restarts requiring manager input, substitutions, set-piece takers, and acting goalkeepers. A fixed rollout does not automatically supply all management decisions.

## Long rollouts and retained outputs

`messi.control_rollout` accepts control-action trajectories and provides final-state, event-receipt, replay, and independent-match batch paths.

`advance_control` expects an action PyTree with a leading time axis `(T, …)` and returns only the final state. `advance_control_batch` expects a leading match axis `(B, …)` on state, roster, key, and action, with the time axis after the match axis on action. It uses `vmap` on one device and `pmap` on multiple local devices. If `B` is not divisible by the device count, it pads internally and removes the padding from the result.

- Choose the final-state path for ordinary training or the event-receipt path when events are needed.
- Retaining receipts or full trajectories consumes additional output memory.
- `advance_control_batch` selects the device mapping internally. Check its batch contract before wrapping it in another `pmap`.
- For a long match with management boundaries, handle each chunk and boundary response from `make_managed_advance`, then resume explicitly.

Performance depends on batch size, horizon, retained outputs, compilation state, and device count. No throughput or memory figure is guaranteed without measurement.

For an entire match with bounded device output and a detailed report, use [match recording](../analysis/README.md). Its report-only path reduces each captured chunk on the host; video export retains the dense host trajectory only when requested.
