"""Conviction-to-weight mapping functions.

Maps per-position conviction scores into portfolio weights.
Five methods:
    1. Basic: w = C / sum|C| (conviction-proportional)
    2. Vol-adjusted: w = (C / sigma_idio) / sum|C / sigma_idio|
    3. Kelly: w = (C * E[R]) / sigma^2, capped at half-Kelly
    4. Risk Parity: equal risk contribution, scaled by conviction
    5. Tiered: discrete conviction bands -> fixed weight tiers
"""

from __future__ import annotations

from enum import Enum


class SizingMethod(Enum):
    """Available conviction-to-weight mapping methods."""

    BASIC = "basic"
    VOL_ADJUSTED = "vol_adjusted"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    TIERED = "tiered"


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


def kelly_mapping(
    convictions: dict[str, float],
    expected_returns: dict[str, float],
    vols: dict[str, float],
    half_kelly: bool = True,
) -> dict[str, float]:
    """Kelly criterion position sizing.

    Optimal bet sizing: w_i = (C_i * E[R_i]) / sigma_i^2.
    Scaled by conviction magnitude so higher conviction = larger position.
    Capped at half-Kelly (industry convention) to reduce variance.
    Normalized to sum of absolute weights = 1.

    Args:
        convictions: {instrument_id: conviction} mapping.
        expected_returns: {instrument_id: expected_return} mapping.
        vols: {instrument_id: vol} mapping.
        half_kelly: If True (default), apply 1/2 Kelly scaling.

    Returns:
        {instrument_id: weight} mapping.
    """
    if not convictions:
        return {}

    raw: dict[str, float] = {}
    for k, c in convictions.items():
        er = expected_returns.get(k, 0.0)
        vol = vols.get(k, 1.0)
        if vol <= 0:
            vol = 1.0
        # Kelly fraction: edge / variance, scaled by conviction sign & magnitude
        kelly_w = (abs(c) * er) / (vol**2)
        # Apply sign from conviction (short = negative weight)
        sign = 1.0 if c >= 0 else -1.0
        raw[k] = sign * kelly_w

    if half_kelly:
        raw = {k: v * 0.5 for k, v in raw.items()}

    # Normalize
    total_abs = sum(abs(v) for v in raw.values())
    if total_abs == 0:
        return {k: 0.0 for k in convictions}

    return {k: v / total_abs for k, v in raw.items()}


def risk_parity_mapping(
    convictions: dict[str, float],
    vols: dict[str, float],
) -> dict[str, float]:
    """Risk parity sizing scaled by conviction.

    First compute equal-risk-contribution weights (w_i proportional to 1/sigma_i),
    then scale each by |C_i| / max|C_j| to preserve conviction ordering.
    Ensures no single position dominates portfolio risk while respecting
    conviction magnitude. Normalized to sum of absolute weights = 1.

    Args:
        convictions: {instrument_id: conviction} mapping.
        vols: {instrument_id: vol} mapping.

    Returns:
        {instrument_id: weight} mapping.
    """
    if not convictions:
        return {}

    # Equal risk contribution: inverse vol
    inv_vol: dict[str, float] = {}
    for k in convictions:
        vol = vols.get(k, 1.0)
        if vol <= 0:
            vol = 1.0
        inv_vol[k] = 1.0 / vol

    # Scale by conviction magnitude relative to max
    max_abs_c = max(abs(c) for c in convictions.values())
    if max_abs_c == 0:
        return {k: 0.0 for k in convictions}

    scaled: dict[str, float] = {}
    for k, c in convictions.items():
        conv_scale = abs(c) / max_abs_c
        sign = 1.0 if c >= 0 else -1.0
        scaled[k] = sign * inv_vol[k] * conv_scale

    total_abs = sum(abs(v) for v in scaled.values())
    if total_abs == 0:
        return {k: 0.0 for k in convictions}

    return {k: v / total_abs for k, v in scaled.items()}


def tiered_mapping(
    convictions: dict[str, float],
) -> dict[str, float]:
    """Tiered conviction-to-weight mapping.

    Maps conviction to discrete tiers with fixed weight bands,
    which is how most discretionary L/S PMs actually size:
        |C| >= 3.0  -> "High conviction"   -> 4.5% position
        |C| >= 1.5  -> "Medium conviction"  -> 2.5% position
        |C| > 0     -> "Low conviction"     -> 1.0% position
    Sign preserved. Normalized to sum of absolute weights = 1.

    Args:
        convictions: {instrument_id: conviction} mapping.

    Returns:
        {instrument_id: weight} mapping.
    """
    if not convictions:
        return {}

    raw: dict[str, float] = {}
    for k, c in convictions.items():
        abs_c = abs(c)
        if abs_c >= 3.0:
            tier_weight = 0.045
        elif abs_c >= 1.5:
            tier_weight = 0.025
        elif abs_c > 0:
            tier_weight = 0.01
        else:
            tier_weight = 0.0
        sign = 1.0 if c >= 0 else -1.0
        raw[k] = sign * tier_weight

    total_abs = sum(abs(v) for v in raw.values())
    if total_abs == 0:
        return {k: 0.0 for k in convictions}

    return {k: v / total_abs for k, v in raw.items()}


def map_convictions(
    convictions: dict[str, float],
    method: SizingMethod = SizingMethod.BASIC,
    vols: dict[str, float] | None = None,
    expected_returns: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map convictions to weights using the specified method.

    Args:
        convictions: Per-position conviction scores.
        method: Sizing method.
        vols: Idiosyncratic vols (required for VOL_ADJUSTED, KELLY, RISK_PARITY).
        expected_returns: Expected returns (required for KELLY).

    Returns:
        Weight mapping.
    """
    if method == SizingMethod.VOL_ADJUSTED:
        if vols is None:
            raise ValueError("vols required for VOL_ADJUSTED sizing method")
        return vol_adjusted_mapping(convictions, vols)
    if method == SizingMethod.KELLY:
        if vols is None:
            raise ValueError("vols required for KELLY sizing method")
        if expected_returns is None:
            raise ValueError("expected_returns required for KELLY sizing method")
        return kelly_mapping(convictions, expected_returns, vols)
    if method == SizingMethod.RISK_PARITY:
        if vols is None:
            raise ValueError("vols required for RISK_PARITY sizing method")
        return risk_parity_mapping(convictions, vols)
    if method == SizingMethod.TIERED:
        return tiered_mapping(convictions)
    return basic_mapping(convictions)
