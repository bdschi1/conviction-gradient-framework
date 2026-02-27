"""Risk Regime Shift (RRS) — change in volatility regime.

RRS_t = (sigma_idio_t - sigma_idio_{t-1}) / sigma_idio_{t-1}
        + (IV_t - HV_t) / HV_t

Captures both the direction of idiosyncratic volatility change and
the implied-vs-historical volatility spread. Positive RRS indicates
increasing risk; negative indicates decreasing risk.
"""

from __future__ import annotations


def compute_rrs(
    sigma_idio_current: float,
    sigma_idio_prev: float,
    implied_vol: float | None = None,
    historical_vol: float | None = None,
) -> float:
    """Compute Risk Regime Shift.

    Args:
        sigma_idio_current: Current idiosyncratic volatility.
        sigma_idio_prev: Previous period idiosyncratic volatility.
        implied_vol: Current implied volatility (optional).
        historical_vol: Current historical volatility (optional).

    Returns:
        RRS value. Positive = increasing risk, negative = decreasing.
    """
    if sigma_idio_prev <= 0:
        raise ValueError(f"sigma_idio_prev must be positive, got {sigma_idio_prev}")

    # Idiosyncratic vol change component
    vol_change = (sigma_idio_current - sigma_idio_prev) / sigma_idio_prev

    # IV-HV spread component (only if both are available)
    iv_hv_spread = 0.0
    if implied_vol is not None and historical_vol is not None and historical_vol > 0:
        iv_hv_spread = (implied_vol - historical_vol) / historical_vol

    return vol_change + iv_hv_spread
