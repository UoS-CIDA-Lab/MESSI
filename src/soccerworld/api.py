"""Small, supported public construction and transition surface for SoccerWorld."""

from __future__ import annotations

import jax.numpy as jnp

from soccerworld._engine.config import (
    Agent,
    Ball,
    BenchPlayer,
    Engine,
    Foul,
    Reward,
    RulePolicy,
    Stadium,
    Substitution,
)
from soccerworld._engine.constants import DEFAULT_MAX_SUBSTITUTIONS
from soccerworld._engine.env import SoccerEnv as _EngineSoccerEnv
from soccerworld._engine.timebase import DEFAULT_TIMEBASE, Timebase
from soccerworld.core.randomness import RandomEvent, RandomnessControl
from soccerworld.core.results import TransitionResult
from soccerworld.data.capture import CaptureSpec
from soccerworld.data.projection import (
    capture_has_record,
    command_result_from_info,
    transition_record,
)

_DEFAULT_CAPTURE = CaptureSpec.none()
_PRIVATE_INFO_KEYS = frozenset({"manager_decision_trace"})


class SoccerEnv(_EngineSoccerEnv):
    """Supported environment facade with typed command and capture results."""

    def transition(
        self,
        key,
        state,
        command,
        *,
        capture: CaptureSpec = _DEFAULT_CAPTURE,
        external_substitutions: bool = True,
        external_formations: bool = True,
        external_takers: bool = True,
        include_info: bool = False,
        compute_observation: bool = True,
        randomness: RandomnessControl | None = None,
    ) -> TransitionResult:
        """Apply one command and optionally emit a causal on-device record.

        ``capture`` and the boolean options select static output graphs. When supplied as
        arguments to :func:`jax.jit`, mark them static; closing over one profile is usually simpler.
        For the lowest-overhead raw five-tuple path, call :meth:`step_command` directly and disable
        the output flags that are not needed. State-native collectors may set
        ``compute_observation=False`` to receive an ``(N, 0)`` placeholder instead of rebuilding an
        observation that their policy will derive from the next state.
        """

        if not isinstance(capture, CaptureSpec):
            raise TypeError(
                f"capture must be CaptureSpec, got {type(capture).__name__}"
            )
        if type(include_info) is not bool:
            raise TypeError(
                f"include_info must be a Python bool, got {type(include_info).__name__}"
            )
        if type(compute_observation) is not bool:
            raise TypeError(
                "compute_observation must be a Python bool, got "
                f"{type(compute_observation).__name__}"
            )

        pre_observation = (
            self.get_obs_array(state)
            if capture.observation
            else jnp.empty((self.N, 0), dtype=state.ball_pos.dtype)
        )
        observation, post_state, reward, done, info = self.step_command(
            key,
            state,
            command,
            external_substitutions=external_substitutions,
            external_formations=external_formations,
            external_takers=external_takers,
            collect_substeps=capture.substep_telemetry,
            include_bc_info=True,
            compute_observation=compute_observation,
            include_decision_trace=True,
            randomness=randomness,
        )
        command_result = command_result_from_info(
            command,
            info,
            external_substitutions=external_substitutions,
            external_formations=external_formations,
            external_takers=external_takers,
        )
        record = (
            transition_record(
                pre_state=state,
                pre_observation=pre_observation,
                command=command,
                command_result=command_result,
                reward=reward,
                terminated=info["terminated"],
                truncated=info["truncated"],
                post_state=post_state,
                info=info,
                capture=capture,
            )
            if capture_has_record(capture)
            else None
        )
        public_info = (
            {name: value for name, value in info.items() if name not in _PRIVATE_INFO_KEYS}
            if include_info
            else None
        )
        return TransitionResult(
            observation=observation,
            state=post_state,
            reward=reward,
            terminated=info["terminated"],
            truncated=info["truncated"],
            done=done,
            command_result=command_result,
            record=record,
            info=public_info,
        )

__all__ = [
    "Agent",
    "Ball",
    "BenchPlayer",
    "DEFAULT_MAX_SUBSTITUTIONS",
    "DEFAULT_TIMEBASE",
    "Engine",
    "Foul",
    "Reward",
    "RandomEvent",
    "RandomnessControl",
    "RulePolicy",
    "SoccerEnv",
    "Stadium",
    "Substitution",
    "Timebase",
]
