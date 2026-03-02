"""Independent Thesis Shift (ITS) — conviction delta from thesis testing.

ITS captures how much independent research moved conviction relative to the
PM's starting view. Replaces the prior "Adversarial Debate Shift" (ADS) with
a richer signal that reflects how real investment committees work:

- PM steers inquiry (research directives), not forced conclusions
- Analysts test the thesis independently — they may converge with or
  diverge from the PM's view
- The magnitude and direction of the analyst-PM delta is high-value information
- Large deltas (either direction) are high-information-content events

Sign convention:
    ITS = (pm_conviction - analyst_mean) / scale

    Positive ITS → thesis challenged (analysts less convicted than PM)
    Negative ITS → thesis confirmed/strengthened (analysts more convicted)

    In the gradient: C -= alpha * lambda4 * ITS
    Positive ITS → conviction decreases (protective when analysts see risk PM doesn't)
    Negative ITS → conviction increases (supportive when independent research confirms)

When PM conviction and analyst convictions are unavailable, falls back to
the p_pre/p_post shift computation for backward compatibility.
"""

from __future__ import annotations

import numpy as np


def compute_its(
    p_pre: list[float] | None = None,
    p_post: list[float] | None = None,
    pm_conviction: float | None = None,
    analyst_convictions: list[float] | None = None,
    position_type: str | None = None,
) -> float:
    """Compute Independent Thesis Shift.

    Two modes:

    1. Rich mode (pm_conviction + analyst_convictions available):
       ITS = (pm_conviction - analyst_mean) / scale
       Scale normalizes by the range of analyst convictions to prevent
       outlier sensitivity.

    2. Fallback mode (only p_pre/p_post):
       ITS = mean(p_post) - mean(p_pre)
       Equivalent to the prior ADS computation.

    Args:
        p_pre: Pre-debate thesis probabilities (fallback mode).
        p_post: Post-debate thesis probabilities (fallback mode).
        pm_conviction: PM's conviction level (rich mode). Scale: 0-1.
        analyst_convictions: Analyst conviction levels (rich mode). Scale: 0-1.
        position_type: Position classification. One of:
            alpha_long, core_long, alpha_short, hedge_short.
            Informational — does not change the ITS value directly,
            but is available for downstream consumers.

    Returns:
        ITS value. Positive = thesis challenged, negative = thesis confirmed.
    """
    # Rich mode: PM conviction vs analyst convictions
    if pm_conviction is not None and analyst_convictions:
        return _compute_its_rich(pm_conviction, analyst_convictions)

    # Fallback: p_pre/p_post shift (backward compatible with ADS)
    return _compute_its_fallback(p_pre, p_post)


def _compute_its_rich(
    pm_conviction: float,
    analyst_convictions: list[float],
) -> float:
    """Compute ITS from PM-analyst conviction delta.

    Uses the spread between PM conviction and analyst mean, normalized
    by the range of analyst views to account for disagreement.

    Args:
        pm_conviction: PM's conviction (0-1 scale).
        analyst_convictions: Analyst convictions (0-1 scale each).

    Returns:
        ITS value, typically in [-1, 1].
    """
    analyst_mean = float(np.mean(analyst_convictions))
    analyst_range = float(np.max(analyst_convictions) - np.min(analyst_convictions))

    # Scale by analyst range: high analyst disagreement dilutes the signal
    # Floor at 0.1 to prevent division by near-zero
    scale = max(analyst_range, 0.1) + 0.5

    delta = pm_conviction - analyst_mean
    return float(np.clip(delta / scale, -1.0, 1.0))


def _compute_its_fallback(
    p_pre: list[float] | None,
    p_post: list[float] | None,
) -> float:
    """Compute ITS from pre/post debate probabilities.

    Backward-compatible with the prior ADS computation:
    ITS = mean(p_post) - mean(p_pre)

    Args:
        p_pre: Pre-debate probabilities.
        p_post: Post-debate probabilities.

    Returns:
        ITS value.
    """
    if not p_pre or not p_post:
        return 0.0

    return float(np.mean(p_post) - np.mean(p_pre))


# Backward compatibility alias
compute_ads = compute_its
