"""Conviction-to-weight mapping functions.

Maps per-position conviction scores into portfolio weights.
Two methods:
    1. Basic: w = C / sum|C| (conviction-proportional)
    2. Vol-adjusted: w = (C / sigma_idio) / sum|C / sigma_idio|
"""

from __future__ import annotations

from enum import Enum


class SizingMethod(Enum):
    """Available conviction-to-weight mapping methods."""

    BASIC = "basic"
    VOL_ADJUSTED = "vol_adjusted"


def basic_mapping(convictions: dict[str, float]) -> dict[str, float]:
    """Map convictions to weights proportional to conviction magnitude.

    w_i = C_i / sum_j |C_j|

    Long positions have positive weight, short positions negative.

    Args:
        convictions: {instrument_id: conviction} mapping.

    Returns:
        {instrument_id: weight} mapping summing to <= 1 in absolute terms.
    """
    if not convictions:
        return {}

    total_abs = sum(abs(c) for c in convictions.values())
    if total_abs == 0:
        return {k: 0.0 for k in convictions}

    return {k: c / total_abs for k, c in convictions.items()}


def vol_adjusted_mapping(
    convictions: dict[str, float],
    vols: dict[str, float],
) -> dict[str, float]:
    """Map convictions to weights adjusted for idiosyncratic volatility.

    Adjusts conviction by inverse vol so that high-vol names get smaller
    positions for the same conviction level.

    w_i = (C_i / sigma_i) / sum_j |C_j / sigma_j|

    Args:
        convictions: {instrument_id: conviction} mapping.
        vols: {instrument_id: idiosyncratic_vol} mapping.

    Returns:
        {instrument_id: weight} mapping.
    """
    if not convictions:
        return {}

    adjusted: dict[str, float] = {}
    for k, c in convictions.items():
        vol = vols.get(k, 1.0)
        if vol <= 0:
            vol = 1.0
        adjusted[k] = c / vol

    total_abs = sum(abs(v) for v in adjusted.values())
    if total_abs == 0:
        return {k: 0.0 for k in convictions}

    return {k: v / total_abs for k, v in adjusted.items()}


def map_convictions(
    convictions: dict[str, float],
    method: SizingMethod = SizingMethod.BASIC,
    vols: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map convictions to weights using the specified method.

    Args:
        convictions: Per-position conviction scores.
        method: Sizing method.
        vols: Idiosyncratic vols (required for VOL_ADJUSTED).

    Returns:
        Weight mapping.
    """
    if method == SizingMethod.VOL_ADJUSTED:
        if vols is None:
            raise ValueError("vols required for VOL_ADJUSTED sizing method")
        return vol_adjusted_mapping(convictions, vols)
    return basic_mapping(convictions)
