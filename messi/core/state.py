"""physics carry와 rollout-invariant roster를 분리한 immutable JAX state."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from messi.core.contact import ContactResult


class BallState(NamedTuple):
    """매 physics substep에서 변할 수 있는 ball state."""

    position: jax.Array
    velocity: jax.Array
    spin: jax.Array
    live: jax.Array


class RosterState(NamedTuple):
    """한 uninterrupted rollout chunk에서 불변인 player identity와 ability."""

    team_id: jax.Array
    player_id: jax.Array
    is_goalkeeper: jax.Array
    max_speed: jax.Array
    reach_height: jax.Array
    height: jax.Array
    ball_control: jax.Array
    endurance_factor: jax.Array
    role: jax.Array
    formation_anchor: jax.Array
    formation_id: jax.Array


class PlayerState(NamedTuple):
    """physics와 rules가 갱신하는 player별 dynamic carry."""

    position: jax.Array
    velocity: jax.Array
    body_forward: jax.Array
    on_pitch: jax.Array
    sent_off: jax.Array
    stamina_long: jax.Array
    stamina_short: jax.Array
    challenge_recovery_substeps: jax.Array
    contact_lock_substeps: jax.Array
    aerial_recovery_substeps: jax.Array
    possession_loss_lock_substeps: jax.Array
    yellow_cards: jax.Array

    @property
    def active(self) -> jax.Array:
        """현재 physics와 rules에 참여하는 player mask."""

        return self.on_pitch & (~self.sent_off)


class PossessionState(NamedTuple):
    """현재 control과 그 원인이 된 마지막 contact.

    ``control_ticks``는 마지막 control touch로부터의 tick이므로 같은 선수가 드리블 중
    새 ``CONTROL/TRAP``을 성공시키면 1로 다시 시작한다. ``episode_start_tick``은 그와
    달리 같은 team·player가 공을 유지하는 동안 바뀌지 않는 소유 에피소드의 시작
    ``control_tick``이며, 소유가 없으면 ``-1``이다. 두 값은 서로 다른 질문에 답하므로
    함께 보존한다.
    """

    team: jax.Array
    player: jax.Array
    previous_team: jax.Array
    control_ticks: jax.Array
    episode_start_tick: jax.Array
    last_contact: ContactResult


class RestartState(NamedTuple):
    """pending 또는 active restart의 causal state."""

    kind: jax.Array
    team: jax.Array
    substeps_remaining: jax.Array
    taker: jax.Array
    indirect: jax.Array
    opened_control_tick: jax.Array


class RestartReleaseProvenance(NamedTuple):
    """retouch와 direct-goal rule에 필요한 restart release 사실."""

    active: jax.Array
    untouched: jax.Array
    kind: jax.Array
    team: jax.Array
    taker: jax.Array
    indirect: jax.Array
    law11_direct_exempt: jax.Array
    release_mechanism: jax.Array


class State(NamedTuple):
    """observation·event·roster를 제외한 authoritative dynamic rollout carry."""

    control_tick: jax.Array
    ball: BallState
    players: PlayerState
    attack_direction: jax.Array
    kickoff_team: jax.Array
    possession: PossessionState
    restart: RestartState
    score: jax.Array
    restart_release: RestartReleaseProvenance
    offside_flagged: jax.Array
    offside_direct_exempt_team: jax.Array
    gk_backpass_team: jax.Array
    dead_ball_control_ticks: jax.Array
    first_half_wall_end_tick: jax.Array
    first_half_live_extension_ticks: jax.Array
    penalty_completion_active: jax.Array
    penalty_completion_team: jax.Array
    restart_layout_ready: jax.Array
    restart_taker_approach: jax.Array


def initial_player_body_forward(
    team_id: jax.Array,
    attack_direction: jax.Array,
) -> jax.Array:
    """team attack direction에서 seam 없는 initial body direction을 만든다."""

    team_direction = attack_direction[team_id]
    return jnp.stack((team_direction, jnp.zeros_like(team_direction)), axis=-1).astype(
        jnp.float32
    )


def body_forward_from_angle(angle: jax.Array) -> jax.Array:
    """scalar angle을 unit body-forward vector로 변환한다."""

    angle = jnp.asarray(angle, dtype=jnp.float32)
    return jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1).astype(jnp.float32)


def body_angle_from_forward(body_forward: jax.Array) -> jax.Array:
    """body-forward vector의 geometric angle을 반환한다."""

    body_forward = jnp.asarray(body_forward, dtype=jnp.float32)
    return jnp.arctan2(body_forward[..., 1], body_forward[..., 0]).astype(jnp.float32)


__all__ = [
    "BallState",
    "PlayerState",
    "PossessionState",
    "RestartReleaseProvenance",
    "RestartState",
    "RosterState",
    "State",
    "body_angle_from_forward",
    "body_forward_from_angle",
    "initial_player_body_forward",
]
