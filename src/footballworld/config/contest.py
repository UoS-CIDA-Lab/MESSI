"""Explicit priors for stochastic player-ball contests.

The model separates candidate eligibility, actor selection, and realized
contact outcomes. Its defaults are transfer and design priors, not independent measured
football constants.  The score weights and temperature form one coupled
coordinate whose available duel data do not identify each term separately.

Likewise, ``tackle_success_probability`` uses a provider-event rate whose
semantic duel unit differs from a simulator attempt, and the goalkeeper curve
mixes action selection with three-dimensional ball response.  Both are
transfer priors pending identity-preserving opportunity data and
match-held-out joint calibration.  The non-zero tackle-foul probability is an
explicit transfer prior, not a fitted occurrence rate: the available
data contain no channel-specific attempt denominator. Conditional discipline
after an already-declared contact foul does have an aggregate K-League receipt.
Stochastic deflections remain disabled because physical swept collisions are
their authoritative source.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Contest:
    """Coupled actor-selection and conditional-outcome priors."""

    distance_weight: float = 1.0
    reach_time_weight: float = 1.0
    height_fit_weight: float = 0.55
    possession_weight: float = 0.0
    ball_control_weight: float = 0.3
    temperature: float = 0.66

    tackle_success_probability: float = 0.28
    # Reference probability at the midpoint of the bounded kinematic context.
    # It remains a transfer/design prior, not a measured attempt rate.
    tackle_foul_probability: float = 0.04
    # At the two context extremes the pre-coverage foul odds may change by at
    # most a factor of two. This is a deliberately small design prior: the
    # available provider feed cannot identify separate biomechanical weights.
    tackle_foul_context_logit_limit: float = 0.6931471805599453
    # Kept separate from the underlying occurrence model so rare-case research
    # coverage cannot be mistaken for a fitted football probability.
    tackle_foul_rare_case_coverage_multiplier: float = 1.15
    tackle_deflection_probability: float = 0.0

    card_probability_midpoint: float = 0.1297
    card_attack_progress_logit_weight: float = -1.2860
    card_elapsed_fraction_logit_weight: float = 0.8028
    direct_red_given_card_probability: float = 0.03125

    goalkeeper_catch_speed_midpoint_mps: float = 21.3
    goalkeeper_catch_speed_scale_mps: float = 9.9
