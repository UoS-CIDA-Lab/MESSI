"""Explicit design priors for non-challenge player-body fouls."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BodyFoul:
    """Occurrence priors for one physically realized opponent impact.

    The logistic shape conditions a charge foul on causal impact context. Its
    opportunity unit is the strongest realized capsule impact rather than a
    proximity query on every physics substep. These defaults are therefore
    transfer/design priors, not measured football constants.

    ``rare_case_coverage_multiplier`` is deliberately separate from the base
    model. Its default modestly oversamples declared fouls for learning and
    validation coverage; rate receipts must report both the physical candidate
    rate and the post-multiplier declaration rate.
    """

    minimum_closing_speed_mps: float = 5.5
    ball_near_distance_m: float = 1.5

    base_logit: float = -7.8
    closing_speed_logit_weight_per_mps: float = 0.18
    behind_logit_weight: float = 1.30
    shoulder_alignment_logit_discount: float = 1.20
    ball_far_logit_weight: float = 1.00
    possessed_victim_logit_weight: float = 0.50
    goal_denial_logit_weight: float = 0.50

    probability_floor: float = 0.003
    probability_ceiling: float = 0.10
    rare_case_coverage_multiplier: float = 1.15


__all__ = ["BodyFoul"]
