"""Total thesis loss computation.

L_t = w1 * FE_t^2 + w2 * FVS_t + w3 * RRS_t + w4 * ITS_t

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
    its: float,
    params: ConvictionParams | None = None,
    weight_overrides: dict[str, float] | None = None,
) -> LossComponents:
    """Compute total thesis loss from individual components.

    Args:
        fe: Forecast Error (standardized).
        fvs: Fundamental Violation Score (0-1).
        rrs: Risk Regime Shift.
        its: Independent Thesis Shift.
        params: Hyperparameters with loss weights w1-w4.
        weight_overrides: Optional dict with keys 'w1','w2','w3','w4' to override
            params weights. Used by adaptive weight system.

    Returns:
        LossComponents with per-component values and total.
    """
    p = params or ConvictionParams()

    w1 = weight_overrides.get("w1", p.w1) if weight_overrides else p.w1
    w2 = weight_overrides.get("w2", p.w2) if weight_overrides else p.w2
    w3 = weight_overrides.get("w3", p.w3) if weight_overrides else p.w3
    w4 = weight_overrides.get("w4", p.w4) if weight_overrides else p.w4

    total = w1 * (fe ** 2) + w2 * fvs + w3 * rrs + w4 * its

    return LossComponents(
        fe=fe,
        fvs=fvs,
        rrs=rrs,
        its=its,
        total_loss=total,
    )
