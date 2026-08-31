"""Team-policy composition for the shipped demo."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

TEAM_POLICY_ADAPTER_VERSION = 2

__all__ = [
    "TEAM_POLICY_ADAPTER_VERSION",
    "TeamPolicyAdapter",
    "load_external_policy",
    "policy_label",
    "policy_metadata",
    "style_spec",
]


class TeamPolicyAdapter:
    """Overlay optional learned team policies on the full-roster rule action."""

    def __init__(
        self,
        rule_policy,
        learned_by_team: dict[int, Any],
        *,
        players: int,
        home_slots: int,
        action_dim: int,
    ) -> None:
        if set(learned_by_team) - {0, 1}:
            raise ValueError("learned policy team must be HOME=0 or AWAY=1")
        if not 0 < home_slots < players:
            raise ValueError("team slot split must leave players on both teams")
        if action_dim != 8:
            raise ValueError("learned action policy requires action_dim=8")
        self.rule_policy = rule_policy
        self.players = int(players)
        self.action_dim = int(action_dim)
        self.home_mask = jnp.arange(players) < home_slots
        self.learned_by_team = dict(sorted(learned_by_team.items()))

        # Equal fingerprints share one dynamic parameter tree, preserving JIT reuse.
        parameters = []
        parameter_index: dict[str, int] = {}
        parts = []
        for team, learned in self.learned_by_team.items():
            identity = learned.fingerprint
            index = parameter_index.get(identity)
            if index is None:
                index = len(parameters)
                parameter_index[identity] = index
                parameters.append(learned.parameters)
            parts.append((team, learned, index))
        self.runtime_parameters = tuple(parameters)
        self._parts = tuple(parts)

    def apply_with_parameters(self, parameters, observation, key, affordance):
        if len(self._parts) < 2:
            action = jnp.asarray(
                self.rule_policy(observation, key, affordance),
                jnp.float32,
            )
        else:
            # Both learned teams overwrite every slot, so the rule solver is unnecessary.
            action = jnp.zeros((self.players, self.action_dim), jnp.float32)
        if action.shape != (self.players, self.action_dim):
            raise ValueError("rule policy must return action with shape (N,8)")

        for team, learned, index in self._parts:
            student_key = jax.random.fold_in(key, np.uint32(team + 1))
            student_action = jnp.asarray(
                learned.apply_with_parameters(
                    parameters[index],
                    observation,
                    student_key,
                    affordance,
                )
            )
            if student_action.shape != (self.players, self.action_dim):
                raise ValueError(f"learned team {team} policy must return action with shape (N,8)")
            if not jnp.issubdtype(student_action.dtype, jnp.floating):
                raise TypeError(f"learned team {team} policy must return floating actions")
            student_action = student_action.astype(jnp.float32)
            finite = jnp.all(jnp.isfinite(student_action), axis=-1)
            bounded = (
                (student_action[:, 0] >= jnp.float32(0.0))
                & (student_action[:, 0] <= jnp.float32(1.0))
                & jnp.all(
                    (student_action[:, 1:] >= jnp.float32(-1.0))
                    & (student_action[:, 1:] <= jnp.float32(1.0)),
                    axis=-1,
                )
            )
            # Preserve the action ABI while preventing one malformed external
            # row from entering physics.  A whole invalid row becomes a no-op;
            # clipping would conceal the contract failure.  Experiment-owned
            # strict adapters retain the corresponding diagnostics.
            student_action = jnp.where(
                (finite & bounded)[:, None],
                student_action,
                jnp.zeros_like(student_action),
            )
            mine = self.home_mask if team == 0 else ~self.home_mask
            action = jnp.where(mine[:, None], student_action, action)
        return action

    def __call__(self, observation, key, affordance):
        return self.apply_with_parameters(
            self.runtime_parameters,
            observation,
            key,
            affordance,
        )


def style_spec(name: str):
    return None if name == "random" else name


def policy_label(policy_spec: str, style: str, learned: Any | None) -> str:
    if policy_spec == "rule":
        return f"rule:{style}"
    if learned is None:
        raise ValueError("learned policy label requires a loaded checkpoint")
    return f"learned:{learned.fingerprint[:12]}"


def policy_metadata(
    policy_spec: str,
    style: str,
    learned: Any | None,
) -> dict:
    if policy_spec == "rule":
        return {"kind": "rule", "style": style}
    if learned is None:
        raise ValueError("learned policy metadata requires a loaded checkpoint")
    return learned.as_dict()


def load_external_policy(
    spec: str,
    *,
    observation_spec: dict,
    deterministic: bool,
):
    """Import the optional learned-policy adapter only when explicitly selected."""

    from soccerworld.policies.learned import load_learned_policy

    return load_learned_policy(
        spec,
        observation_spec=observation_spec,
        deterministic=deterministic,
    )
