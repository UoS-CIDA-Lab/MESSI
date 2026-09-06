"""Physical scales used to decode normalized player actions.

The spin limit follows the measured launch envelope in Barber and Carré,
"The effect of surface geometry on soccer ball trajectories" (2010),
doi:10.1007/s12283-010-0048-x. Professional-player averages reached
146 rad/s for one ball, and the trajectory experiments covered 147 rad/s.
Supporting player-measurement context comes from Neilson and Jones,
"Dynamic soccer ball performance measurement", Science and Football V
(2005), pp. 21-27, which recorded a professional-player maximum of
13.89 rev/s, and Kryger, Mitchell, and Forrester,
"Assessment of the accuracy of different systems for measuring football
velocity and spin rate in the field" (2019),
doi:10.1177/1754337119830249, which evaluated field spin measurements over
94-743 rpm. These supporting ranges are context, not separate cap estimates.
The cap describes a high-energy strike envelope, not free angular velocity:
newly imparted foot spin scales with the requested kick power. This preserves
the rounded 150 rad/s design envelope around the 146--147 rad/s observations
while preventing a near-zero impulse from creating maximum spin.

Among direct striking contacts, only the foot may actively add velocity
or spin. Header, chest, challenge, and goalkeeper parry contacts redirect and
damp the ball's existing motion; their retention values are therefore
dimensionless rather than impulse limits. A designated hand release is the
separate exception needed for throw-ins. The header default is an
evidence-informed design value. "The Relationship Between
Biomechanical-Anthropometrical Parameters and the Force Exerted on the Head
When Heading Free Kicks in Soccer" reported a related 17.99 to 15.93 m/s
reduction. That observation anchors the damping direction, but 0.90 is a
rounded design value, not a direct fit; the exact loss depends on contact
geometry and player motion. Chest retention is a solver prior for a
controlled trap. Challenge retention is the explicit energy-neutral upper bound
until challenge-specific response data exist. Goalkeeper catch/parry resolution
is owned by the stochastic contest configuration.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionScale:
    """Physical limits applied after normalized action decoding."""

    kick_speed_max_mps: float = 34.76
    throw_speed_max_mps: float = 21.5
    # This separates a requested player-relative control exit speed from a
    # strike; it is not an incoming-ball controllability limit.
    control_request_speed_max_mps: float = 9.0

    launch_max_radians: float = 1.0
    ground_launch_down_max_radians: float = 0.21
    ground_launch_down_reference_height_m: float = 1.0

    pelvis_height_factor: float = 0.55
    header_speed_retention: float = 0.90
    chest_speed_retention: float = 0.10
    challenge_speed_retention: float = 1.0

    spin_max_radps: float = 150.0
    restart_min_ball_speed_mps: float = 0.5
    throw_release_height_addition_m: float = 0.47
