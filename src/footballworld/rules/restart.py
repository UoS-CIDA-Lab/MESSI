"""Shared pure-JAX helpers for placing and assigning restarts.

Restart takers are selected exactly when a restart opens. The selector is
deterministic and uses only authoritative rollout state: legal availability,
the restart-specific goalkeeper rule, a lightweight positional role proxy,
technical ability, readiness, and distance. Keeping this here prevents the
different foul, boundary, retouch, and episode paths from silently assigning
different kinds of taker.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from footballworld.config.body_contact import BodyContact
from footballworld.config.geometry import Ball, Stadium
from footballworld.core.constants import (
    NO_PLAYER,
    RK_CORNER,
    RK_GK_HOLD,
    RK_GOALKICK,
    RK_KICKOFF,
    RK_NONE,
    RK_PENALTY,
    RK_THROWIN,
    TEAM_0,
    TEAM_1,
)
from footballworld.core.randomness import RandomEvent, event_random_key
from footballworld.core.state import State

RESTART_TAKER_RANDOM_STREAM = int(RandomEvent.RESTART_TAKER)
RESTART_TAKER_TEMPERATURE = 12.0

ROLE_GK = 0
ROLE_CENTRE_BACK = 1
ROLE_FULL_BACK = 2
ROLE_CENTRE_MID = 3
ROLE_WIDE_MID = 4
ROLE_CENTRE_FORWARD = 5
ROLE_WIDE_FORWARD = 6

# Role-frequency rows for throw-ins, corners, goal kicks are
# calibrated from seven DFL matches in calib/policy/artifacts/
# dfl-policy-reference-v1.json. Pooled provider-role counts receive one
# symmetric pseudo-count per role before normalization. Free-kick depth rows
# and penalties remain compatibility priors because the retained aggregate
# does not identify depth-conditioned free kicks and contains no penalties.
# The 32 observed kickoffs are too sparse to replace the tested opening prior.
_TAKER_PROPENSITY = (
    (0.003, 0.041, 0.839, 0.062, 0.041, 0.003, 0.010),
    (0.018, 0.071, 0.107, 0.589, 0.161, 0.036, 0.018),
    (0.771, 0.157, 0.040, 0.025, 0.004, 0.002, 0.001),
    (0.099, 0.268, 0.200, 0.278, 0.077, 0.018, 0.060),
    (0.000, 0.014, 0.185, 0.230, 0.269, 0.128, 0.174),
    (0.000, 0.000, 0.023, 0.076, 0.229, 0.505, 0.167),
    (0.889, 0.063, 0.016, 0.008, 0.008, 0.008, 0.008),
    (0.000, 0.010, 0.040, 0.160, 0.130, 0.500, 0.160),
)
_TAKER_AFFINITY = jnp.asarray(
    [
        [12.0 * math.log(max(value, 1.0e-3)) for value in row]
        for row in _TAKER_PROPENSITY
    ],
    dtype=jnp.float32,
)
_TAKER_DISTANCE_WEIGHT = jnp.asarray(
    [1.0, 0.35, 0.25, 0.5, 0.4, 0.05, 0.3, 0.2],
    dtype=jnp.float32,
)


class RestartTakerScores(NamedTuple):
    """Fixed-shape restart-taker ranking and its legal-candidate mask."""

    score: jax.Array
    usable: jax.Array


def restart_taker_random_key(
    match_key: jax.Array,
    control_tick: jax.Array,
    team: jax.Array,
    kind: jax.Array,
) -> jax.Array:
    """Return a stateless key for one restart-taker decision.

    The absolute control tick makes the draw independent of rollout chunking.
    Team, restart kind, and a dedicated stream tag isolate it from other random
    systems such as contests and roster sampling.
    """

    return event_random_key(
        match_key,
        RandomEvent.RESTART_TAKER,
        control_tick,
        team,
        kind,
    )


def score_restart_takers(
    *,
    kind: jax.Array,
    position: jax.Array,
    attack_direction: jax.Array,
    role: jax.Array,
    eligible: jax.Array,
    is_goalkeeper: jax.Array,
    player_position: jax.Array,
    stamina_long: jax.Array,
    reach_height: jax.Array,
    stadium: Stadium = Stadium(),
) -> RestartTakerScores:
    """Score legal takers from role, distance, stamina, and aerial value."""

    dtype = player_position.dtype
    kind = jnp.asarray(kind, dtype=jnp.int32)
    direction = jnp.asarray(attack_direction, dtype=dtype)
    restart_position = jnp.asarray(position[:2], dtype=dtype)
    distance = jnp.linalg.norm(player_position - restart_position[None, :], axis=-1)

    half_length = jnp.asarray(stadium.half_length, dtype=dtype)
    restart_progress = jnp.clip(
        (restart_position[0] * direction + half_length) / (2.0 * half_length),
        0.0,
        1.0,
    )
    free_kick_row = jnp.where(
        restart_progress < (1.0 / 3.0),
        2,
        jnp.where(restart_progress < (2.0 / 3.0), 3, 4),
    ).astype(jnp.int32)
    row = free_kick_row
    row = jnp.where(kind == RK_THROWIN, 0, row)
    row = jnp.where(kind == RK_CORNER, 1, row)
    row = jnp.where(kind == RK_PENALTY, 5, row)
    row = jnp.where(kind == RK_GOALKICK, 6, row)
    row = jnp.where(kind == RK_KICKOFF, 7, row).astype(jnp.int32)

    safe_role = jnp.clip(jnp.asarray(role, dtype=jnp.int32), ROLE_GK, ROLE_WIDE_FORWARD)
    score = (
        _TAKER_AFFINITY[row][safe_role]
        + 10.0 * jnp.clip(jnp.asarray(stamina_long, dtype=dtype), 0.0, 1.0)
        - _TAKER_DISTANCE_WEIGHT[row] * distance
    )
    keep_as_corner_target = (kind == RK_CORNER) & (
        (safe_role == ROLE_CENTRE_FORWARD) | (safe_role == ROLE_CENTRE_BACK)
    )
    score = score - jnp.where(
        keep_as_corner_target,
        15.0 * jnp.clip(jnp.asarray(reach_height, dtype=dtype) / 3.0, 0.0, 1.0),
        0.0,
    )
    goalkeeper_only = kind == RK_GK_HOLD
    usable = jnp.asarray(eligible, dtype=jnp.bool_) & jnp.where(
        goalkeeper_only,
        jnp.asarray(is_goalkeeper, dtype=jnp.bool_),
        True,
    )
    return RestartTakerScores(score=score, usable=usable)


def sample_restart_taker(
    ranking: RestartTakerScores,
    player_id: jax.Array,
    event_key: jax.Array,
    *,
    temperature: float = RESTART_TAKER_TEMPERATURE,
    preference_key: jax.Array | None = None,
    persistence: jax.Array | float = 0.0,
) -> jax.Array:
    """Sample a legal taker with event and match preference Gumbel noise.

    Each candidate's draw follows ``player_id`` rather than its array slot, so
    slot reuse and harmless roster permutations do not change the decision.
    ``persistence`` blends repeatable match-kind preference with event variation.
    Temperature zero deliberately recovers the deterministic fallback.
    """

    if temperature < 0.0:
        raise ValueError("restart taker temperature must be non-negative")
    low = jnp.asarray(-jnp.inf, dtype=ranking.score.dtype)
    if temperature == 0.0:
        winner = jnp.argmax(jnp.where(ranking.usable, ranking.score, low))
    else:
        identity = jnp.where(player_id >= 0, player_id, 0).astype(jnp.uint32)
        keys = jax.vmap(lambda value: jax.random.fold_in(event_key, value))(identity)
        event_noise = jax.vmap(
            lambda key: jax.random.gumbel(key, dtype=ranking.score.dtype)
        )(keys)
        preference_key = event_key if preference_key is None else preference_key
        preference_keys = jax.vmap(
            lambda value: jax.random.fold_in(preference_key, value)
        )(identity)
        preference_noise = jax.vmap(
            lambda key: jax.random.gumbel(key, dtype=ranking.score.dtype)
        )(preference_keys)
        weight = jnp.clip(jnp.asarray(persistence, dtype=ranking.score.dtype), 0.0, 1.0)
        noise = (1.0 - weight) * event_noise + weight * preference_noise
        utility = ranking.score / jnp.asarray(temperature, ranking.score.dtype) + noise
        winner = jnp.argmax(jnp.where(ranking.usable, utility, low))
    winner = winner.astype(jnp.int32)
    return jnp.where(jnp.any(ranking.usable), winner, NO_PLAYER).astype(jnp.int32)


def nearest_active_teammate(
    state: State,
    team: jax.Array,
    position: jax.Array,
) -> jax.Array:
    """Return the nearest active player on ``team``, or ``NO_PLAYER``."""

    eligible = state.players.active & (state.players.team_id == team)
    distance_squared = jnp.sum((state.players.position - position[:2]) ** 2, axis=-1)
    winner = jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf)).astype(
        jnp.int32
    )
    return jnp.where(jnp.any(eligible), winner, NO_PLAYER).astype(jnp.int32)


def select_restart_taker(
    state: State,
    kind: jax.Array,
    team: jax.Array,
    position: jax.Array,
    *,
    stadium: Stadium = Stadium(),
) -> jax.Array:
    """Choose one legal, deterministic taker for every restart kind.

    Nearest-only assignment produces implausible penalty, corner, and goal-kick
    takers. Immutable formation anchors are not carried in recurrent physics
    state, so this
    version folds the active team's current depth and width into the same seven
    role classes. Line-wise reductions retain the old "outer two" convention
    without constructing an ``N x N`` role-classification matrix.

    The claimed K-League role table is retained only as an unverified
    ranking prior because its extraction receipt is unavailable. Distance weights
    are authored design priors. Goalkeeper holds are goalkeeper-only; goal kicks
    strongly prefer a goalkeeper but retain the legal outfield fallback.
    ``argmax`` provides a stable lower-slot tie break. This intentionally omits
    Gumbel sampling because the current restart transition has no
    restart-scoped PRNG key; selecting once deterministically is preferable to
    manufacturing state-dependent pseudo-randomness.
    """

    players = state.players
    dtype = players.position.dtype
    safe_team = jnp.clip(jnp.asarray(team, dtype=jnp.int32), TEAM_0, TEAM_1)
    kind = jnp.asarray(kind, dtype=jnp.int32)
    eligible = players.active & (players.team_id == team)

    direction = state.attack_direction[safe_team]
    local_depth = players.position[:, 0] * direction
    lateral = jnp.abs(players.position[:, 1])
    high = jnp.asarray(jnp.inf, dtype=dtype)
    low = jnp.asarray(-jnp.inf, dtype=dtype)
    depth_min = jnp.min(jnp.where(eligible, local_depth, high))
    depth_max = jnp.max(jnp.where(eligible, local_depth, low))
    depth_span = depth_max - depth_min
    depth_fraction = jnp.clip(
        jnp.where(
            depth_span > 1.0,
            (local_depth - depth_min) / jnp.maximum(depth_span, 1.0),
            0.5,
        ),
        0.0,
        1.0,
    )
    band = jnp.where(
        depth_fraction < (1.0 / 3.0),
        0,
        jnp.where(depth_fraction >= (2.0 / 3.0), 2, 1),
    ).astype(jnp.int32)
    wide = jnp.zeros_like(eligible)
    for line in range(3):
        group = eligible & (~players.is_goalkeeper) & (band == line)
        group_count = jnp.sum(group)
        widest = jnp.max(jnp.where(group, lateral, -jnp.inf))
        widest_count = jnp.sum(group & (lateral >= widest - 1.0e-3))
        below_widest = group & (lateral < widest - 1.0e-3)
        second = jnp.max(jnp.where(below_widest, lateral, -jnp.inf))
        threshold = jnp.where(
            (widest_count < 2) & jnp.any(below_widest), second, widest
        )
        wide = wide | (group & (group_count >= 3) & (lateral >= threshold - 1.0e-3))
    role = jnp.where(
        players.is_goalkeeper,
        ROLE_GK,
        jnp.where(
            band == 0,
            jnp.where(wide, ROLE_FULL_BACK, ROLE_CENTRE_BACK),
            jnp.where(
                band == 2,
                jnp.where(wide, ROLE_WIDE_FORWARD, ROLE_CENTRE_FORWARD),
                jnp.where(wide, ROLE_WIDE_MID, ROLE_CENTRE_MID),
            ),
        ),
    ).astype(jnp.int32)

    ranking = score_restart_takers(
        kind=kind,
        position=position,
        attack_direction=direction,
        role=role,
        eligible=eligible,
        is_goalkeeper=players.is_goalkeeper,
        player_position=players.position,
        stamina_long=players.stamina_long,
        reach_height=players.reach_height,
        stadium=stadium,
    )
    winner = jnp.argmax(jnp.where(ranking.usable, ranking.score, low)).astype(jnp.int32)
    return jnp.where(jnp.any(ranking.usable), winner, NO_PLAYER).astype(jnp.int32)


def repair_broken_restart_taker(
    state: State,
    *,
    body: BodyContact = BodyContact(),
    stadium: Stadium = Stadium(),
) -> tuple[State, jax.Array]:
    """Repair an unusable taker and report whether replacement succeeded.

    The common path performs only scalar validity checks.  The ``O(N)`` shared
    taker selector is executed under ``lax.cond`` only when an active restart
    names an out-of-range, inactive, wrong-team, or goalkeeper-ineligible
    player.  Invalidating the layout deliberately prevents same-frame release:
    public restart positioning must first expose the replacement to policies.
    A repaired goalkeeper hold atomically transfers its physical held-ball and
    actor-level possession identity without inventing contact provenance.  If
    no goalkeeper exists, a zero hold counter feeds the existing corner-expiry
    fail-safe on the next environment transition.

    This is the pending-taker liveness repair for the explicit restart layout
    gate.
    """

    taker = jnp.asarray(state.restart.taker, dtype=jnp.int32)
    team = jnp.asarray(state.restart.team, dtype=jnp.int32)
    kind = jnp.asarray(state.restart.kind, dtype=jnp.int32)
    player_count = state.players.position.shape[0]

    valid_slot = (taker >= 0) & (taker < player_count)
    safe_taker = jnp.clip(taker, 0, player_count - 1)
    valid_team = (team == jnp.int32(TEAM_0)) | (team == jnp.int32(TEAM_1))
    usable = (
        valid_slot
        & valid_team
        & state.players.active[safe_taker]
        & (state.players.team_id[safe_taker] == team)
        & ((kind != jnp.int32(RK_GK_HOLD)) | state.players.is_goalkeeper[safe_taker])
    )
    broken = (kind != jnp.int32(RK_NONE)) & (~usable)

    def replace(_):
        replacement = select_restart_taker(
            state,
            kind,
            team,
            state.ball.position,
            stadium=stadium,
        )
        safe_replacement = jnp.clip(replacement, 0, player_count - 1)
        replacement_valid = (
            (replacement >= 0)
            & (replacement < player_count)
            & state.players.active[safe_replacement]
            & (state.players.team_id[safe_replacement] == team)
            & (
                (kind != jnp.int32(RK_GK_HOLD))
                | state.players.is_goalkeeper[safe_replacement]
            )
        )
        synchronize_hold = (kind == jnp.int32(RK_GK_HOLD)) & replacement_valid
        unserviceable_hold = (kind == jnp.int32(RK_GK_HOLD)) & (~replacement_valid)
        held_position = jnp.asarray(
            [
                state.players.position[safe_replacement, 0],
                state.players.position[safe_replacement, 1],
                body.torso_top_height(state.players.height[safe_replacement]),
            ],
            dtype=state.ball.position.dtype,
        )
        return (
            state._replace(
                ball=state.ball._replace(
                    position=jnp.where(
                        synchronize_hold, held_position, state.ball.position
                    ),
                    velocity=jnp.where(
                        synchronize_hold | unserviceable_hold,
                        jnp.zeros_like(state.ball.velocity),
                        state.ball.velocity,
                    ),
                    spin=jnp.where(
                        synchronize_hold | unserviceable_hold,
                        jnp.zeros_like(state.ball.spin),
                        state.ball.spin,
                    ),
                    live=jnp.where(
                        synchronize_hold | unserviceable_hold,
                        jnp.bool_(False),
                        state.ball.live,
                    ),
                ),
                possession=state.possession._replace(
                    team=jnp.where(
                        synchronize_hold, team, state.possession.team
                    ).astype(jnp.int32),
                    player=jnp.where(
                        synchronize_hold, replacement, state.possession.player
                    ).astype(jnp.int32),
                    control_ticks=jnp.where(
                        synchronize_hold,
                        jnp.maximum(state.possession.control_ticks, jnp.int32(1)),
                        state.possession.control_ticks,
                    ).astype(jnp.int32),
                ),
                restart=state.restart._replace(
                    taker=replacement,
                    substeps_remaining=jnp.where(
                        unserviceable_hold,
                        jnp.int32(0),
                        state.restart.substeps_remaining,
                    ).astype(jnp.int32),
                ),
                restart_layout_ready=jnp.bool_(False),
            ),
            replacement_valid,
        )

    return jax.lax.cond(
        broken,
        replace,
        lambda _: (state, jnp.bool_(False)),
        operand=None,
    )


def normalize_indirect_free_kick_position(
    position: jax.Array,
    restart_team: jax.Array,
    attack_direction: jax.Array,
    *,
    stadium: Stadium = Stadium(),
    ball: Ball = Ball(),
) -> jax.Array:
    """Apply the goal-area placement exception to an indirect free kick."""

    dtype = position.dtype
    safe_team = jnp.clip(restart_team, 0, attack_direction.shape[0] - 1)
    direction = attack_direction[safe_team]
    half_length = jnp.asarray(stadium.half_length, dtype=dtype)
    goal_area_length = jnp.asarray(stadium.goal_area_length, dtype=dtype)
    goal_area_half_width = jnp.asarray(0.5 * stadium.goal_area_width, dtype=dtype)
    depth_from_attacking_goal = half_length - direction * position[0]
    inside_opponent_goal_area = (
        (depth_from_attacking_goal >= 0.0)
        & (depth_from_attacking_goal <= goal_area_length)
        & (jnp.abs(position[1]) <= goal_area_half_width)
    )
    line_x = direction * (half_length - goal_area_length)
    restart_position = position.at[0].set(
        jnp.where(inside_opponent_goal_area, line_x, position[0])
    )
    return restart_position.at[2].set(jnp.asarray(ball.radius, dtype=dtype))
