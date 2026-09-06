"""Timing configuration for repeated player-ball interactions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContactTiming:
    """Independent real-time locks used by contact resolution."""

    challenge_recovery_s: float = 0.16
    max_lunge_extra_recovery_s: float = 0.84
    active_contact_interval_s: float = 1.0 / 15.0
    # Maximum recovery after a full vertical-reach attempt. Lower reachable
    # attempts interpolate continuously down to no athletic recovery when the
    # ball can be played at or below the player's standing stature.
    aerial_attempt_recovery_s: float = 0.5
    possession_loss_lock_s: float = 0.16
    # Maximum recovery after a full-extension goalkeeper dive. The horizontal
    # and vertical reach fractions are combined by their maximum, so the
    # engine installs only the minimum recovery implied by the contact point.
    # Appended to preserve the positional constructor order of the original
    # five public fields.
    goalkeeper_dive_recovery_s: float = 1.0
