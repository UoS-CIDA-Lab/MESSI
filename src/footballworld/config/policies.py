"""Static selection of the reference policies supplied by FootballWorld.

These switches select host-runner defaults. They are deliberately not
runtime flags in :meth:`FootballWorld.step`: disabling a reference policy
means that the host supplies another policy (or leaves that decision axis
inactive) at the corresponding low-frequency boundary.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """Choose which built-in rule policies a host runner may install.

    ``rule_based_player`` selects the shipped per-frame player-action policy.
    Match management owns substitutions and in-match formation changes.
    Opening management owns the registered match squad, starting lineup,
    opening formation, and assignment of starters to formation slots. The
    separately named opening-formation adapter only selects a registered layout
    for an already-created authored roster. This reset-first adapter becomes a
    no-op after a full opening commit.
    Set-piece taker selection is separate so a learned manager can coexist
    with the shipped rule-based taker, or vice versa.

    The values are static environment declaration, provenance, and compiled
    runner-cache facts. A caller-provided learned policy always remains an
    explicit callable plus dynamic parameter PyTree; it is never stored in
    this configuration object.
    """

    rule_based_player: bool = True
    rule_based_match_manager: bool = True
    rule_based_opening_manager: bool = True
    rule_based_opening_formation_adapter: bool = True
    rule_based_set_piece_taker: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if type(getattr(self, field.name)) is not bool:
                raise TypeError(f"{field.name} must be a bool")


__all__ = ["PolicySelection"]
