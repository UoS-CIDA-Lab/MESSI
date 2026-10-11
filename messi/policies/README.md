# Policies: built-in rule-based policies

Rule-based policies are part of the MESSI environment. They use the action contract in [`messi.core`](../core/README.md) and policy observations from [`messi.environment`](../environment/README.md).

## Public workflow

1. Create a `PolicyObservation` with `observe_policy(state, roster)`.
2. Compute an `IntentAction` with `rule_policy(observation, ...)`.
3. Pass the action to `transition_match` or a control rollout API.

The policy reads observations rather than private environment state. Stochastic challenges and shot aiming require a caller-provided key. Use a fresh policy key for each frame.

## Tactical plans and configuration

- `RulePolicyConfig` holds the rule-based policy parameters.
- `RuleTacticalPlan` represents tactical profiles and match-up variants as fixed-shape codes.
- `make_rule_tactical_plan` is the public plan constructor.
- `DEFAULT_RULE_TACTICAL_PLAN` and `DEFAULT_RULE_TACTICAL_POOL` are supplied presets.
- `rule_policy_with_diagnostics` returns the policy decision and diagnostics for evaluation or reporting. Use `rule_policy` for inference without diagnostic outputs.

`make_rule_tactical_plan(team_0, team_1)` returns an `int32[2]` code, one entry per team. The public plans are `DFL_BASE`, `ZONA_MISTA`, `TOTAL_FOOTBALL`, `GEGENPRESSING`, and `CATENACCIO`; both teams default to `DFL_BASE`. Pass a selected plan as `rule_policy(..., tactical_plan=plan, key=policy_key)`. See `RULE_TACTICAL_PLAN_SPECS` for each plan's intent and trade-offs.

A policy action does not guarantee a physical outcome. Results also depend on the current state, opposing actions, random key, and physics and rule settings.

## Interpreting policy support

The presence of a policy field alone does not establish that a tactic is active or that a statistic is reproduced. Check the preset, transition wiring, public return value, and relevant tests. A DFL reference schema or hash identifies reference data; it does not by itself demonstrate policy fidelity or statistical equivalence.

Internal feature and specialist code is split across `rule_features.py`, `rule_specialists.py`, `ground_lane.py`, `aerial.py`, and `rule_actions.py`. Internal functions are not part of the public contract.
