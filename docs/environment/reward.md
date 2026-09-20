# Team goal reward contract

FootballWorld exposes one minimal environment-owned reward derived from the
authoritative rules result. It does not infer possession value, expected goals,
territorial value, or policy quality.

## Semantics

Every player transition returns `reward` as a fixed `float32[2]` team vector.

| Event | Reward |
| --- | --- |
| team 0 scores | `[+1, -1]` |
| team 1 scores | `[-1, +1]` |
| no goal | `[0, 0]` |

`reward` is exactly `score_delta - reversed(score_delta)`, converted to
`float32`. The scoring and own-goal laws remain authoritative in
`RuleOutcome.score_delta`; reward calculation does not reclassify a goal.

## APIs and shapes

The same value is available from:

- `env.step(...).reward`
- `env.step_with_events(...).reward`
- `env.step_with_action_receipt(...).reward`
- `execute_step_command(...).reward`
- `make_rollout(...).steps.reward`, with shape `[T, 2]`
- `make_event_rollout(...).steps.reward`, with shape `[T, 2]`

Terminal frozen and padded scan rows have `[0, 0]`. `make_advance` deliberately
retains no per-frame output, so callers needing the reward trajectory must use
an ordinary or eventful rollout.

```python
result = env.step(rollout, setup, action, key)
team_0_reward = result.reward[0]
team_1_reward = result.reward[1]
```

## Learning boundary

Downstream dataset adapters may store the additive team vector without changing
the environment transition. Legacy artifacts that predate the field need an
explicit missing-value path rather than an invented zero reward.

FootballWorld does not prescribe a learning objective for this signal. Any
dense shaping, discounting, per-player credit assignment, or auxiliary loss is
caller-owned and stays outside the lean environment step. Such coefficients are
explicit training choices, not measured football constants.
