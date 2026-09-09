"""Immutable roster inputs used to construct an environment state."""

from dataclasses import dataclass
from numbers import Integral

Position2D = tuple[float, float]
_INT32_MAX = 2_147_483_647


def _finite_scalar(value):
    """Backend-neutral finite predicate for Python, NumPy, and JAX scalars."""

    return (value < float("inf")) & (value > float("-inf"))


def player_profile_values_valid(
    player_id,
    max_speed,
    height,
    reach_height,
    ball_control,
    endurance_factor,
    *,
    head_radius,
):
    """Return the canonical scalar domain predicate for a packed profile.

    Callers must first represent physical values in float32. The expression
    uses scalar arithmetic so it works for host scalars and JAX tracers.
    """

    return (
        (player_id >= 0)
        & (player_id <= _INT32_MAX)
        & _finite_scalar(max_speed)
        & _finite_scalar(height)
        & _finite_scalar(reach_height)
        & _finite_scalar(ball_control)
        & _finite_scalar(endurance_factor)
        & (max_speed > 0.0)
        & (height > 2.0 * head_radius)
        & (reach_height >= height)
        & (ball_control >= 0.0)
        & (ball_control <= 1.0)
        & (endurance_factor > 0.0)
    )


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Identity, authored abilities, and optional registered roles."""

    player_id: int
    max_speed_mps: float = 7.96
    height_m: float = 1.8
    max_reach_height_m: float = 2.7
    # Relative contest-selection coordinate. It does not scale a realized
    # contact's physical impulse or create an independent trapping lottery.
    ball_control: float = 0.5
    endurance_factor: float = 1.0
    is_goalkeeper: bool = False
    # Declared manager-only compatibility with the stable formation-role
    # taxonomy (GK/CB/FB/CM/WM/CF/WF). Empty means unknown.
    preferred_roles: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.preferred_roles) is not tuple:
            raise TypeError("preferred_roles must be a tuple of role codes")
        if any(
            not isinstance(role, Integral) or isinstance(role, bool)
            for role in self.preferred_roles
        ):
            raise TypeError("preferred_roles must contain non-boolean integers")
        if any(role < 0 or role >= 7 for role in self.preferred_roles):
            raise ValueError("preferred_roles must use role codes in [0, 7)")
        if len(set(self.preferred_roles)) != len(self.preferred_roles):
            raise ValueError("preferred_roles must not contain duplicates")


@dataclass(frozen=True, slots=True)
class Player:
    """A player in the initial on-pitch roster."""

    profile: PlayerProfile
    initial_position: Position2D = (0.0, 0.0)
