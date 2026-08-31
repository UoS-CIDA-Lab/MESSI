"""Static output profiles used to compile only the requested rollout data."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Hashable, compile-time selection of transition fields.

    A runner passes this object as a static argument.  It must not be converted to a traced boolean:
    disabled fields should disappear from the compiled graph rather than be computed and masked.
    """

    observation: bool = False
    reward: bool = False
    events: bool = False
    action_provenance: bool = False
    manager_provenance: bool = False
    roster_identity: bool = False
    substep_telemetry: bool = False

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not bool:
                raise TypeError(
                    f"CaptureSpec.{field.name} must be a Python bool, "
                    f"got {type(value).__name__}"
                )

    @classmethod
    def none(cls) -> CaptureSpec:
        return cls()

    @classmethod
    def dynamics(cls) -> CaptureSpec:
        return cls.none()

    @classmethod
    def rl(cls) -> CaptureSpec:
        return cls(observation=True, reward=True)

    @classmethod
    def imitation(cls) -> CaptureSpec:
        return cls(
            observation=True,
            reward=True,
            events=True,
            action_provenance=True,
            roster_identity=True,
        )

    @classmethod
    def manager_imitation(cls) -> CaptureSpec:
        return cls(
            observation=True,
            reward=True,
            events=True,
            action_provenance=True,
            manager_provenance=True,
            roster_identity=True,
        )

    @classmethod
    def audit(cls) -> CaptureSpec:
        return cls(
            observation=True,
            reward=True,
            events=True,
            action_provenance=True,
            manager_provenance=True,
            roster_identity=True,
            substep_telemetry=True,
        )
