"""Build deterministic roster metadata cached by ``SoccerEnv`` at construction."""

import jax.numpy as jnp
import numpy as np

from .constants import ROTATE_180, TEAM_0, TEAM_1

__all__ = ["build_static_metadata"]


def build_static_metadata(
    e_cfg,
    n_agents: int,
    n_opponents: int,
    agent_infos: list,
    opponent_infos: list,
):
    n_players = n_agents + n_opponents
    teams = np.array([TEAM_0] * n_agents + [TEAM_1] * n_opponents, np.int32)
    agent_formation = np.array([m.init_pos for m in agent_infos], np.float32)
    opponent_formation = (
        np.array([m.init_pos for m in opponent_infos], np.float32)
        * np.array([ROTATE_180, ROTATE_180], np.float32)
    )  # 반대 공격 방향

    agent_ability = np.array(
        [
            [
                member.speed,
                member.tall,
                member.reach_z_max,
                member.ball_control,
                member.endurance_factor,
            ]
            for member in agent_infos
        ],
        np.float32,
    )
    opponent_ability = np.array(
        [
            [
                member.speed,
                member.tall,
                member.reach_z_max,
                member.ball_control,
                member.endurance_factor,
            ]
            for member in opponent_infos
        ],
        np.float32,
    )

    assert agent_formation.shape[0] == n_agents, (
        f"agent_formation 멤버 {agent_formation.shape[0]} != n_agents {n_agents}"
    )
    assert opponent_formation.shape[0] == n_opponents, (
        "opponent_formation 멤버 "
        f"{opponent_formation.shape[0]} != n_opponents {n_opponents}"
    )
    assert (
        agent_ability.shape[0] == n_agents
        and opponent_ability.shape[0] == n_opponents
    ), "ability 멤버 수가 인원수와 불일치"

    init_position = np.concatenate([agent_formation, opponent_formation], axis=0)
    att_dir = np.where(teams == TEAM_0, 1.0, -1.0).astype(np.float32)

    gk = np.zeros(n_players, np.int32)
    for i, m in enumerate(agent_infos):
        if m.is_gk:
            gk[i] = 1
    for j, m in enumerate(opponent_infos):
        if m.is_gk:
            gk[n_agents + j] = 1

    def _unpack(abil):
        vmax, tall, reach_z, player_ctrl, endurance_factor = (
            abil[:, 0],
            abil[:, 1],
            abil[:, 2],
            abil[:, 3],
            abil[:, 4],
        )
        head_z = tall * e_cfg.reach_height_factor
        return vmax, head_z, reach_z, player_ctrl, endurance_factor

    a_vmax, a_head_z, a_reach_z, a_player_ctrl, a_endurance = _unpack(agent_ability)
    o_vmax, o_head_z, o_reach_z, o_player_ctrl, o_endurance = _unpack(opponent_ability)

    return (
        jnp.array(teams),
        jnp.array(att_dir),
        jnp.array(gk),
        jnp.array(init_position),
        jnp.array(np.concatenate([a_vmax, o_vmax])),
        jnp.array(np.concatenate([a_reach_z, o_reach_z])),
        jnp.array(np.concatenate([a_head_z, o_head_z])),
        jnp.array(np.concatenate([a_player_ctrl, o_player_ctrl])),
        jnp.array(np.concatenate([a_endurance, o_endurance])),
    )
