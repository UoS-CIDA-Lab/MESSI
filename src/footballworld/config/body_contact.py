"""Configuration for passive ball contact with a player's body."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BodyContact:
    """Oriented player silhouette with passive rebound parameters.

    The torso retains the requested 0.50 m shoulder width and 0.20 m depth.
    Below it, two diagonal leg capsules meet at the crotch and end one shoulder
    width apart. Their fixed isosceles stance admits a narrow, physical
    between-legs path without adding an unidentified gait phase to rollout
    state. These are effective collision dimensions, not measured anatomy.
    """

    shoulder_width_m: float = 0.50
    torso_depth_m: float = 0.20
    leg_apex_height_factor: float = 0.50
    leg_radius_m: float = 0.05
    # Law 11 needs the foremost playable foot, not only collision volume.
    # This is a standing/running pose envelope from the player reference point;
    # it does not enlarge passive collision or deliberate control reach.
    foot_forward_extent_m: float = 0.30
    torso_top_height_factor: float = 0.85

    def torso_top_height(self, stature):
        """Return the chest-to-head boundary from literal standing stature."""

        return stature * self.torso_top_height_factor

    def head_center_height(self, stature):
        """Place the head sphere so its top equals literal standing stature."""

        return stature - self.head_radius_m

    head_radius_m: float = 0.10

    restitution: float = 0.50
