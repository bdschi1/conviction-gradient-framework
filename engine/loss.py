"""Total thesis loss computation.

L_t = w1 * FE_t^2 + w2 * FVS_t + w3 * RRS_t + w4 * ADS_t

Aggregates all loss components into a single scalar that measures
how much the investment thesis is under stress.
"""

from __future__ import annotations

from config.defaults import ConvictionParams
from engine.models import LossComponents


def compute_loss(
    fe: float,
    fvs: float,
    rrs: float,
    ads: float,
    params: ConvictionParams | None = None,
) -> LossComponents:
    """Compute total thesis loss from individual components.

    Args:
        fe: Forecast Error (standardized).
        fvs: Fundamental Violation Score (0-1).
        rrs: Risk Regime Shift.
        ads: Adversarial Debate Shift.
        params: Hyperparameters with loss weights w1-w4.

    Returns:
        LossComponents with per-component values and total.
    """
    p = params or ConvictionParams()

    total = p.w1 * (fe ** 2) + p.w2 * fvs + p.w3 * rrs + p.w4 * ads

    return LossComponents(
        fe=fe,
        fvs=fvs,
        rrs=rrs,
        ads=ads,
        total_loss=total,
    )
