"""Competition-specific squad-management limits."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManagementRules:
    """Substitution count and opportunity profile for one match."""

    max_substitutions_per_team: int = 5
    max_windows_per_team: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("max_substitutions_per_team", self.max_substitutions_per_team),
            ("max_windows_per_team", self.max_windows_per_team),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


__all__ = ["ManagementRules"]
