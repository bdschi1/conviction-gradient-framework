"""Adversarial Debate Shift (ADS) — IC probability change.

ADS_t = mean(P_post) - mean(P_pre)

Measures how much the investment committee's collective probability
assessment shifted during adversarial debate. Captures the information
content of structured disagreement.
"""

from __future__ import annotations

import numpy as np


def compute_ads(
    p_pre: list[float],
    p_post: list[float],
) -> float:
    """Compute Adversarial Debate Shift.

    Args:
        p_pre: Pre-debate anonymous thesis probabilities from IC participants.
        p_post: Post-debate anonymous thesis probabilities.

    Returns:
        ADS value. Positive = debate increased conviction, negative = decreased.
    """
    if not p_pre or not p_post:
        return 0.0

    if len(p_pre) != len(p_post):
        # Use mean of each list independently if lengths differ
        pass

    return float(np.mean(p_post) - np.mean(p_pre))
