"""Forecast Error (FE) — standardized return miss.

FE_t = (R_t - E_{t-1}[R_t]) / sigma_expected

Measures how far realized returns deviate from the analyst/AI forecast,
normalized by expected volatility so the signal is comparable across names.
"""

from __future__ import annotations


def compute_fe(
    realized_return: float,
    expected_return: float,
    sigma_expected: float,
) -> float:
    """Compute standardized forecast error.

    Args:
        realized_return: Actual return R_t.
        expected_return: Prior expected return E_{t-1}[R_t].
        sigma_expected: Expected volatility for normalization (must be > 0).

    Returns:
        Standardized forecast error.
    """
    if sigma_expected <= 0:
        raise ValueError(f"sigma_expected must be positive, got {sigma_expected}")
    return (realized_return - expected_return) / sigma_expected
