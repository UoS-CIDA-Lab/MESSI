"""Scalar single-match rollout used by the public demo."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["rollout_single_match"]


def rollout_single_match(
    env,
    policy,
    key,
    n_steps: int,
    *,
    collect_substeps: bool = False,
    start_stamina: float = 1.0,
):
    """Run one match through one jitted temporal scan, with no batch transform."""

    key, reset_key = jax.random.split(key)
    obs0, state0 = env.reset_array(reset_key)
    if start_stamina < 1.0:
        state0 = state0._replace(
            stamina_long=jnp.full_like(state0.stamina_long, start_stamina)
        )
        obs0 = env.get_obs_array(state0)

    policy_apply = getattr(policy, "apply_with_parameters", None)
    runtime_parameters = getattr(policy, "runtime_parameters", ())
    command_template = env.empty_command()

    def run_fn(carry, parameters):
        def step_fn(inner_carry, _):
            obs, state, rng = inner_carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            affordance = env.affordance_view(state)
            action = (
                policy(obs, action_key, affordance)
                if policy_apply is None
                else policy_apply(parameters, obs, action_key, affordance)
            )
            command = command_template.with_player_actions(action)
            obs_next, state_next, _, _, info = env.step_command(
                step_key,
                state,
                command,
                collect_substeps=collect_substeps,
                include_bc_info=False,
                include_decision_trace=False,
            )
            substeps = info["substeps"] if collect_substeps else None
            return (obs_next, state_next, rng), (state_next, substeps)

        return lax.scan(step_fn, carry, xs=None, length=n_steps)

    run = jax.jit(run_fn)
    (_, state_last, _), (trajectory, substeps) = run(
        (obs0, state0, key),
        runtime_parameters,
    )
    states = jax.tree_util.tree_map(
        lambda initial, stacked: jnp.concatenate(
            (initial[None], stacked),
            axis=0,
        ),
        state0,
        trajectory,
    )
    return (
        jax.device_get(states),
        jax.device_get(state_last),
        jax.device_get(substeps) if collect_substeps else None,
    )
