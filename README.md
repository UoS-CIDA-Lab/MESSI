<div align="center">

<img src="docs/assets/logo.svg" alt="MESSI" width="640">

### A Multi-Agent Environment for Soccer Simulation and Intelligence

**Full 90-minute 11-vs-11 football, written end-to-end in JAX — jit-compiled, vmappable, and built for imitation learning and multi-agent reinforcement learning.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-%E2%89%A50.4.38-orange.svg)](https://github.com/jax-ml/jax)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](#project-status)
[![Citation](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

[Why MESSI](#why-messi) · [Features](#features) · [Install](#installation) · [Quickstart](#quickstart) · [Match control](#external-match-control) · [Data capture](#data-capture-and-batching) · [Rule-based teams](#built-in-rule-based-teams) · [Rendering](#rendering-and-replays) · [Showcase](#research-showcase) · [Roadmap](#roadmap) · [Citation](#citation)

</div>

---

**MESSI** is a research-grade football world engine. It simulates complete matches — two halves, offside, fouls and cards, set pieces, substitutions, stamina, three-dimensional ball flight — as a pure JAX function, so an entire 90-minute game (54,000 control steps at 10 Hz) compiles into a single `lax.scan` and thousands of matches batch onto one accelerator.

Every decision that a coach or a player would make enters through one fixed-shape command, and every consequence comes back as a typed, causally ordered record. That makes MESSI equally at home as a **reinforcement-learning environment**, an **imitation-learning target** for real tracking data, and a **reproducible simulator** for football analytics.

> MESSI was formerly published as *SoccerWorld* (`0.0.0.dev0`). The Python import name `soccerworld` is kept for compatibility; `import messi` is an alias.

## Why MESSI

| | MESSI | Typical football RL environments |
|---|---|---|
| Match length | Full 90 minutes with half-time, stoppages and restarts | Short episodes or single scenarios |
| Laws of the game | Offside, fouls, yellow/red cards, set pieces, substitutions (budget 5) | Simplified or absent |
| Physics | 3-D ball flight with spin, contested touches, two-stage stamina | 2-D kinematics |
| Rosters | Any 1–11 per side, asymmetric benches, per-player ability contracts | Fixed 11-vs-11 |
| Control surface | Player actions **and** manager decisions (formation, substitution, set-piece taker) with legality reports | Player actions only |
| Learning interfaces | RL / imitation / manager-imitation / audit capture profiles, NPZ shards | Observation + reward |
| Execution | JAX-native, `jit`/`vmap`/`lax.scan`, CPU and GPU | Python step loops |

## Features

- **Complete match rules** — offside, fouls and cards with a data-fitted card model, set pieces and restarts, half-time, substitutions with independently configurable budgets.
- **Physical realism** — three-dimensional ball flight and spin, contest resolution between players, two-stage (short/long) stamina with speed floors, goalkeeper handling.
- **Formations from data** — layout tables (4-4-2, 4-3-3, 3-4-3, 3-5-2, 4-2-4, 5-4-1) estimated from professional tracking data; formation requests are followed progressively during live play.
- **One command in, one result out** — `StepCommand` carries player actions, substitutions, formations and set-piece takers; `TransitionResult` reports per-request acceptance, deferral or rejection.
- **Learning-ready capture** — `CaptureSpec.rl()`, `.imitation()`, `.manager_imitation()` and `.audit()` produce causal `pre → command → result → post` records with action provenance; `NpzShardSink` writes atomic numeric shards.
- **Built-in opponents** — parameterised rule-based teams (pressing, containment, build-up, set-piece routines) with named styles such as `balanced`, `tiki_taka` and `long_ball`.
- **Batched rollouts** — `batch_rollout` runs independent matches with scalar control flow (`lax.map`) by default and an explicit `vmap` mode when the workload allows it.
- **Rendering and replays** — light and rich MP4 rendering plus `events.jsonl`, `tracking.jsonl` and `metadata.json` exports for analysis tooling.
- **Reproducibility first** — a provenance manifest for every numeric default, frozen-source conformance receipts, and exact rule-policy digests.

## Architecture

```mermaid
flowchart LR
    subgraph Inputs
        A[Player actions<br/>22 × 8 continuous] --> C
        M[Manager decisions<br/>formation · substitution · taker] --> C
    end
    C[StepCommand] --> E
    subgraph MESSI engine (JAX)
        E[Legality &amp; command routing] --> P[Physics<br/>ball flight · contests · stamina]
        P --> L[Laws of the game<br/>offside · fouls · cards · restarts]
        L --> S[(Match state)]
    end
    S --> R[TransitionResult<br/>observation · reward · records]
    R --> RL[RL / MARL]
    R --> IL[Imitation learning]
    R --> V[Rendering &amp; replays]
```

## Project status

MESSI `0.1.0` is an **alpha** release: the engine, rule-based teams, capture interfaces and renderer are usable for research today, and the public command/result boundary is the one we intend to keep. Behavioural and API changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/UoS-CIDA-Lab/MESSI.git
cd MESSI
python -m pip install -e .            # engine only
python -m pip install -e '.[render]'  # + MP4 rendering (imageio-ffmpeg, matplotlib, pillow)
```

JAX accelerator support follows the platform-specific JAX installation. MESSI does not select or initialise a CPU/GPU backend at import time.

## Quickstart

```python
import jax
import jax.numpy as jnp
from soccerworld import CaptureSpec, SoccerEnv   # or: from messi import ...

env = SoccerEnv(halftime=False, game_duration=900)   # 15 in-game minutes at 10 Hz
reset_key, step_key = jax.random.split(jax.random.PRNGKey(0))
observation, state = env.reset_array(reset_key)

command = env.empty_command().with_player_actions(
    jnp.zeros_like(env.empty_command().player_actions)
)
result = env.transition(step_key, state, command, capture=CaptureSpec.rl())
observation, state, reward = result.observation, result.state, result.reward
```

Compile the step once and scan a whole match:

```python
step = jax.jit(lambda k, s, c: env.transition(k, s, c, capture=CaptureSpec.rl()))
```

Run a full match with the built-in teams and render it:

```bash
python demo_match.py --seconds 30 --no-render                       # smoke test
python demo_match.py --home tiki_taka --away long_ball \
       --seconds 120 --mode rich --out replays/rich-match.mp4       # rich render
soccerworld-demo --seconds 30 --no-render                           # installed entry point
```

## External match control

Manager decisions are first-class inputs. Formation requests may be submitted during live or dead-ball play and players progressively adopt the new layout; substitution requests are checked against the bench and the remaining budget; set-piece takers can be nominated per restart kind.

```python
from soccerworld import FORMATION_LAYOUT_NAMES, RestartKind, Team

command = env.empty_command().with_player_actions(actions)
command = command.with_substitutions(
    command.substitutions.with_request(Team.HOME, 0, out_slot=3, bench_index=0))
command = command.with_formations(
    command.formations.with_request(
        Team.HOME, layout_index=FORMATION_LAYOUT_NAMES.index("4-3-3 mid normal")))
command = command.with_set_piece_takers(
    command.set_piece_takers.with_request(Team.HOME, RestartKind.CORNER, player_slot=7))
```

Rosters may contain any 1–11 players per side. Speed, height, vertical reach, ball control, endurance, goalkeeper status and initial position are configurable per player; bench width follows the supplied reserves and `max_substitutions` defaults to 5.

## Data capture and batching

`CaptureSpec.none()`, `.rl()`, `.imitation()`, `.manager_imitation()` and `.audit()` select static output profiles, so a compiled step never changes shape. Imitation profiles expose causal `pre → command → result → post` records and action provenance without touching private engine state, and `soccerworld.data.NpzShardSink` writes atomic NPZ transition shards with strict JSON metadata.

For independent batched matches, `soccerworld.batch_rollout(one_rollout)` uses scalar-control-flow `lax.map` by default; `strategy="vmap"` is an explicit override to enable after checking numerical and memory behaviour for the target workload.

## Built-in rule-based teams

MESSI ships a configurable rule-based team (`RulePolicy`) used as a training opponent, a data generator and a legality oracle for imitation. It presses and contains with measured engagement lines, builds up through role-based support positions, takes set pieces with data-derived corner structures, and is deterministic given its key. Team styles are selected by name in the demo (`--home tiki_taka --away long_ball`).

## Rendering and replays

```bash
python demo_match.py --seconds 60 --mode light     # fast preview → replays/<timestamp>/match.mp4
python demo_match.py --seconds 60 --mode rich      # pitch, kits, ball height, event overlay
```

`events.jsonl`, `tracking.jsonl` and `metadata.json` are written beside the video (pass `--video-only` to skip them), so replays can be re-analysed or re-rendered without re-simulating.

## Research showcase

MESSI is the environment behind the K-League multi-agent learning work at CIDA Lab (University of Seoul). In that pipeline, 22-agent policies are pre-trained by behaviour cloning on professional tracking data and then trained with MAPPO against the built-in rule-based team, using MESSI's imitation capture for DAgger-style supervision and its xG-based shot signals as dense reward. The learned team beat the rule-based opponent in every one of eight full 90-minute matches from a regulation kick-off in our latest run, having started from a policy that could not register a single shot — the capture, manager and full-match interfaces above are what made each diagnostic step along the way measurable.

## Roadmap

- PyPI release of `messi-env`.
- Gymnasium / PettingZoo / JaxMARL adapters over the native command API.
- Learned goalkeeper and manager baselines.
- Public benchmark suite: fixed seeds, kick-off starts, 8-match evaluation protocol with card and stall reporting.
- Documentation site with the engine parameter provenance manifest.

## Contributing

Issues and pull requests are welcome through the [GitHub repository](https://github.com/UoS-CIDA-Lab/MESSI). When reporting a defect please include the MESSI version, the JAX version, the accelerator backend and, where possible, a seed that reproduces the behaviour.

## Citation

If MESSI contributes to your research, please cite the software. Machine-readable metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@software{park_messi_2026,
  author    = {Park, Hyunwoo},
  title     = {{MESSI}: A Multi-Agent Environment for Soccer Simulation and Intelligence},
  version   = {0.1.0},
  year      = {2026},
  license   = {Apache-2.0},
  url       = {https://github.com/UoS-CIDA-Lab/MESSI},
  publisher = {CIDA Lab, University of Seoul}
}
```

## Maintainer

**Hyunwoo Park** — CIDA Lab, University of Seoul.

## License and attribution

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Hyunwoo Park. Developed at CIDA Lab, University of Seoul. See [NOTICE](NOTICE) for attribution and [CITATION.cff](CITATION.cff) for citation metadata.
