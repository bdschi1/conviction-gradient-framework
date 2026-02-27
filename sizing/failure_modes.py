"""Failure mode controls — oscillation guard and structural reset.

Prevents pathological conviction dynamics:
- Oscillation guard: halve learning rate on sign flips
- Structural reset: force conviction to baseline when thesis breaks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.defaults import ConvictionParams
from engine.stability import count_sign_flips

logger = logging.getLogger(__name__)


@dataclass
class FailureModeAction:
    """Action triggered by a failure mode."""

    action_type: str  # "halve_alpha" | "reset_conviction" | "require_rewrite"
    reason: str
    instrument_id: str
    new_alpha: float | None = None
    new_conviction: float | None = None


def check_oscillation_guard(
    instrument_id: str,
    conviction_history: list[float],
    current_alpha: float,
    params: ConvictionParams | None = None,
) -> FailureModeAction | None:
    """Check if conviction is oscillating and apply guard.

    If conviction sign flips within the configured window, halve the
    learning rate and flag for thesis re-underwrite.

    Args:
        instrument_id: Position identifier.
        conviction_history: Recent convictions, most recent last.
        current_alpha: Current learning rate.
        params: Hyperparameters with flip_window_days.

    Returns:
        FailureModeAction if oscillation detected, None otherwise.
    """
    p = params or ConvictionParams()
    flips = count_sign_flips(conviction_history, window=p.flip_window_days)

    if flips > 0:
        new_alpha = current_alpha / (2 ** flips)  # halve for each flip
        new_alpha = max(new_alpha, p.alpha_min)
        logger.warning(
            "%s: conviction oscillation detected (%d flips in %d days). "
            "Halving alpha: %.4f -> %.4f",
            instrument_id, flips, p.flip_window_days, current_alpha, new_alpha,
        )
        return FailureModeAction(
            action_type="halve_alpha",
            reason=f"{flips} sign flip(s) in {p.flip_window_days} days",
            instrument_id=instrument_id,
            new_alpha=new_alpha,
        )
    return None


def check_structural_reset(
    instrument_id: str,
    fvs: float,
    sigma_idio_current: float,
    sigma_idio_baseline: float,
    params: ConvictionParams | None = None,
) -> FailureModeAction | None:
    """Check if a structural reset is warranted.

    Reset triggers:
    - FVS exceeds threshold (major thesis break)
    - Idiosyncratic vol doubles from baseline

    Args:
        instrument_id: Position identifier.
        fvs: Current Fundamental Violation Score.
        sigma_idio_current: Current idiosyncratic vol.
        sigma_idio_baseline: Baseline (entry) idiosyncratic vol.
        params: Hyperparameters with thresholds.

    Returns:
        FailureModeAction if reset triggered, None otherwise.
    """
    p = params or ConvictionParams()

    if fvs >= p.fvs_reset_threshold:
        logger.warning(
            "%s: FVS %.2f >= threshold %.2f. Structural reset triggered.",
            instrument_id, fvs, p.fvs_reset_threshold,
        )
        return FailureModeAction(
            action_type="reset_conviction",
            reason=f"FVS {fvs:.2f} >= threshold {p.fvs_reset_threshold:.2f}",
            instrument_id=instrument_id,
            new_conviction=0.0,
        )

    if sigma_idio_baseline > 0:
        vol_ratio = sigma_idio_current / sigma_idio_baseline
        if vol_ratio >= p.vol_double_threshold:
            logger.warning(
                "%s: Vol ratio %.2f >= threshold %.2f. Structural reset triggered.",
                instrument_id, vol_ratio, p.vol_double_threshold,
            )
            return FailureModeAction(
                action_type="reset_conviction",
                reason=f"Vol ratio {vol_ratio:.2f}x >= {p.vol_double_threshold:.2f}x threshold",
                instrument_id=instrument_id,
                new_conviction=0.0,
            )

    return None
