# SoccerWorld

SoccerWorld is a JAX-native football environment for full-match simulation and
multi-agent learning. It supports configurable team sizes up to 11-vs-11,
goalkeepers, three-dimensional ball flight and spin, stamina, fouls and cards,
substitutions, formations, set pieces, and externally controlled match decisions.

This branch contains the reusable environment, its rule-based controllers,
data-capture interfaces, rendering support, and a standalone match demo.

## Installation

Python 3.10 or newer is required.

```bash
git clone -b hw https://github.com/UoS-CIDA-Lab/UOS-FootballMARL-Env.git
cd UOS-FootballMARL-Env
python -m pip install -e .
```

Install the rendering dependencies when MP4 output is needed:

```bash
python -m pip install -e '.[render]'
```

JAX accelerator support depends on the platform-specific JAX installation. The
package does not select or initialize a CPU/GPU backend during top-level import.

## Basic environment use

Every external decision enters through a fixed-shape `StepCommand`. SoccerWorld
checks legality and reports whether each player, substitution, formation, or
set-piece-taker request was accepted, applied, deferred, or rejected.

```python
import jax
import jax.numpy as jnp

from soccerworld import CaptureSpec, SoccerEnv

env = SoccerEnv(halftime=False, game_duration=900)
reset_key, step_key = jax.random.split(jax.random.PRNGKey(0))
observation, state = env.reset_array(reset_key)

command = env.empty_command()
command = command.with_player_actions(jnp.zeros_like(command.player_actions))

result = env.transition(step_key, state, command, capture=CaptureSpec.rl())
observation = result.observation
state = result.state
reward = result.reward
```

For compiled rollouts, keep the capture profile static:

```python
compiled_step = jax.jit(
    lambda key, state, command: env.transition(
        key, state, command, capture=CaptureSpec.rl()
    )
)
```

## External match control

The public command API accepts player actions together with independent manager
decisions. Formation requests may be submitted during live or dead-ball play;
players progressively follow the new layout. Substitution requests are checked
against the configured bench and substitution budget. Goalkeepers may be replaced
when a valid goalkeeper reserve is supplied.

```python
from soccerworld import FORMATION_LAYOUT_NAMES, RestartKind, Team

command = env.empty_command().with_player_actions(actions)
command = command.with_substitutions(
    command.substitutions.with_request(
        Team.HOME, 0, out_slot=3, bench_index=0
    )
)
command = command.with_formations(
    command.formations.with_request(
        Team.HOME,
        layout_index=FORMATION_LAYOUT_NAMES.index("4-3-3 mid normal"),
    )
)
command = command.with_set_piece_takers(
    command.set_piece_takers.with_request(
        Team.HOME, RestartKind.CORNER, player_slot=7
    )
)
```

On-pitch rosters may contain any 1–11 players per team. Player speed, height,
vertical reach, ball control, endurance, goalkeeper status, and initial position
are configurable. Bench width follows the supplied reserves unless explicitly
preallocated, and `max_substitutions` is independently configurable (default: 5).

## Data capture and batching

`CaptureSpec.none()`, `.rl()`, `.imitation()`, `.manager_imitation()`, and
`.audit()` select static output profiles. Imitation profiles expose causal
`pre -> command -> result -> post` records and action provenance without requiring
private engine state. Numeric trajectories can be written through
`soccerworld.data.NpzShardSink`.

For independent batched matches, `soccerworld.batch_rollout(one_rollout)` uses
scalar-control-flow `lax.map` by default. `strategy="vmap"` is an explicit override
that should be selected only after checking numerical and memory behavior for the
target workload.

## Demo match

The demo runs one scalar match and supports rule-based policies, external manager
control, light/rich rendering, and replay export.

```bash
# Fast functional run
python demo_match.py --seconds 30 --no-render

# Light rendering; default output is replays/<timestamp>/match.mp4
python demo_match.py --seconds 60 --mode light

# Rich rendering at an explicit path
python demo_match.py --home tiki_taka --away long_ball \
  --seconds 120 --mode rich --out replays/rich-match.mp4

# Installed entry points
soccerworld-demo --seconds 30 --no-render
python -m soccerworld.demo --seconds 30 --no-render
```

Replay `events.jsonl`, `tracking.jsonl`, and `metadata.json` are written beside the
video by default. Pass `--video-only` when only the MP4 is required. Run
`python demo_match.py --help` for all available options.

## License and attribution

Licensed under the [Apache License 2.0](LICENSE).

Copyright 2026 Hyunwoo Park. Developed at CIDA Lab, University of Seoul. See
[NOTICE](NOTICE) for attribution and [CITATION.cff](CITATION.cff) for software
citation metadata.
