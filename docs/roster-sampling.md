# Episode roster sampling

`PlayerProfile` supplies the Gaussian location for each player. Passing a
reset key samples `max_speed_mps`, `height_m`, `max_reach_height_m`,
`ball_control`, and `endurance_factor` once for that episode. The realised
values live in `State.players`, so every later step observes the same player
and the hot transition performs no additional draw.

```python
import jax

from footballworld import FootballWorld, RosterSampling

env = FootballWorld(
    roster_sampling=RosterSampling(
        max_speed_std_mps=0.30,
        ball_control_std=0.06,
    )
)
episode_key = jax.random.key(7)
reset = env.reset(team_0, team_1, key=episode_key)
management = env.initialize_management(
    reset.rollout,
    team_0_bench,
    team_1_bench,
    key=episode_key,
)
```

The same episode key is intentionally passed to both calls. FootballWorld
folds it into separate active-roster and bench streams. The same key and
inputs are bit-exact; a new key produces a new episode roster. `player_id`,
team, goalkeeper designation, formation, and every array shape remain fixed.
Padded bench cells remain zero.

Omitting `key` selects exact-profile mode and uses the supplied profiles
without sampling, even when sampling is enabled in the default configuration.
Set `RosterSampling(enabled=False)` to ignore a supplied key explicitly.
Exact-profile mode avoids hidden host randomness and provides a deterministic
setup contract.

The host-only `build_opening_policy_inputs` boundary also accepts a paired
boolean `sample_abilities` mask. It is intended for strict match fixtures where
complete authored ability bundles remain exact while omitted bundles in the
same candidate pool are sampled. Omitting the mask retains the existing
all-candidates sampling behavior. This mask is consumed before reset and never
enters the recurrent environment step.

Height and reach are not sampled independently. The sampler draws height and
the non-negative margin `max_reach_height_m - height_m`, then reconstructs
reach. Consequently `reach >= height` is structural rather than probabilistic;
the configured body geometry also raises the effective minimum height when
needed. Values close to a configured bound have a clipped distribution, so a
profile is the unbounded Gaussian location, not a promise that the empirical
post-clipping average is identical.

All standard deviations, absolute bounds, and the z clipping point in
`RosterSampling` are explicit design priors. They are not claimed as fitted
population statistics. Replace them with a documented corpus fit when one is
available.

`sample_profile_values` is a pure, fixed-shape JAX operation and is safe under
`jit` and `vmap`. Public roster validation and formation preparation remain
host-side. For a batched experiment, prepare the static roster once and batch
only the sampled profile operation or stack independently keyed reset results;
do not move host validation into the rollout graph.

Roster values are immutable within an episode. A keyed reset samples abilities
once so episode-level population variation remains possible; an unkeyed reset
uses the authored profiles exactly. Neither path introduces hidden host
randomness into later transitions.
