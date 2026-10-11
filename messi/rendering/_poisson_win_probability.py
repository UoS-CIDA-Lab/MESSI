"""Phase-specific Poisson rates fitted from the Phase S training matches."""

RATE_CURVE_SCHEMA = "messi.poisson-win-probability-rates/1"

# Each row is (displayed match minute, mean HOME goals remaining, mean AWAY goals
# remaining). The source has 250 training episodes; see docs/evaluation.md for
# the collection identity, validation results, and model limitations.
FIRST_HALF_REMAINING_GOALS = (
    (0, 1.600, 1.776),
    (5, 1.544, 1.688),
    (10, 1.472, 1.604),
    (15, 1.412, 1.504),
    (20, 1.344, 1.432),
    (25, 1.240, 1.344),
    (30, 1.156, 1.300),
    (35, 1.116, 1.200),
    (40, 1.076, 1.096),
    (45, 1.040, 1.012),
    (50, 1.000, 0.944),
    (55, 0.928, 0.896),
)

SECOND_HALF_REMAINING_GOALS = (
    (45, 0.928, 0.900),
    (50, 0.832, 0.832),
    (55, 0.772, 0.740),
    (60, 0.704, 0.636),
    (65, 0.648, 0.584),
    (70, 0.576, 0.508),
    (75, 0.508, 0.408),
    (80, 0.424, 0.324),
    (85, 0.348, 0.252),
    (90, 0.212, 0.160),
    (95, 0.108, 0.084),
    (100, 0.000, 0.000),
)

WIN_PROBABILITY_RATE_MODEL = {
    "schema": RATE_CURVE_SCHEMA,
    "first_half": FIRST_HALF_REMAINING_GOALS,
    "second_half": SECOND_HALF_REMAINING_GOALS,
}
