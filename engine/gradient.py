"""Gradient computation and adaptive learning rate.

Gradient: nabla L_t = lambda1 * FE + lambda2 * FVS + lambda3 * RRS + lambda4 * ADS

Learning rate: alpha_t = (kappa / (1 + InfoHalfLife))
                         * (sigma_idio / sigma_expected)
                         * (1 / (1 + TrackRecordScore))

The gradient captures direction and magnitude of thesis deterioration.
The learning rate adapts conviction elasticity to information quality,
volatility regime, and analyst track record.
"""

from __future__ import annotations

from config.defaults import ConvictionParams
from engine.models import GradientResult


def compute_gradient(
    fe: float,
    fvs: float,
    rrs: float,
    ads: float,
    params: ConvictionParams | None = None,
) -> float:
    """Compute the loss gradient for conviction update.

    Args:
        fe: Forecast Error.
        fvs: Fundamental Violation Score.
        rrs: Risk Regime Shift.
        ads: Adversarial Debate Shift.
        params: Hyperparameters with lambda1-4.

    Returns:
        Scalar gradient value.
    """
    p = params or ConvictionParams()
    return p.lambda1 * fe + p.lambda2 * fvs + p.lambda3 * rrs + p.lambda4 * ads


def compute_learning_rate(
    kappa: float,
    info_half_life: float,
    sigma_idio: float,
    sigma_expected: float,
    track_record_score: float,
    alpha_min: float,
    alpha_max: float,
) -> float:
    """Compute adaptive learning rate.

    Higher idiosyncratic vol → more responsive.
    Higher track record → less responsive (trusted analyst, lower learning rate).
    Higher info half-life → less responsive (slower information decay).

    Args:
        kappa: Base scaling constant.
        info_half_life: Information decay measure for this name.
        sigma_idio: Current idiosyncratic volatility.
        sigma_expected: Expected volatility for normalization.
        track_record_score: Analyst calibration score (higher = more trusted).
        alpha_min: Floor for learning rate.
        alpha_max: Ceiling for learning rate.

    Returns:
        Clamped learning rate in [alpha_min, alpha_max].
    """
    if sigma_expected <= 0:
        raise ValueError(f"sigma_expected must be positive, got {sigma_expected}")

    raw_alpha = (
        (kappa / (1.0 + info_half_life))
        * (sigma_idio / sigma_expected)
        * (1.0 / (1.0 + track_record_score))
    )

    return max(alpha_min, min(raw_alpha, alpha_max))


def compute_gradient_result(
    fe: float,
    fvs: float,
    rrs: float,
    ads: float,
    alpha_t: float,
    params: ConvictionParams | None = None,
) -> GradientResult:
    """Compute full gradient result with component breakdown.

    Args:
        fe, fvs, rrs, ads: Loss component values.
        alpha_t: Pre-computed learning rate.
        params: Hyperparameters.

    Returns:
        GradientResult with value, contributions, and learning rate.
    """
    p = params or ConvictionParams()

    contributions = {
        "fe": p.lambda1 * fe,
        "fvs": p.lambda2 * fvs,
        "rrs": p.lambda3 * rrs,
        "ads": p.lambda4 * ads,
    }

    gradient_value = sum(contributions.values())

    return GradientResult(
        gradient_value=gradient_value,
        component_contributions=contributions,
        learning_rate=alpha_t,
    )
