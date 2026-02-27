"""Calibration metrics — Brier score and calibration buckets.

Measures how well conviction levels predict actual outcomes.
Well-calibrated: high conviction → high hit rate.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class CalibrationBucket(BaseModel):
    """One calibration bucket with conviction range and realized metrics."""

    conviction_min: float
    conviction_max: float
    count: int
    hit_rate: float = Field(description="Fraction of correct-direction predictions")
    avg_return: float = Field(description="Average realized return in this bucket")
    avg_conviction: float = Field(description="Mean conviction in bucket")


def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Compute Brier score — mean squared error of probability predictions.

    Lower is better. 0 = perfect calibration, 1 = maximally wrong.

    Args:
        predictions: Predicted probabilities [0, 1].
        outcomes: Binary outcomes (0 or 1).

    Returns:
        Brier score.
    """
    if not predictions or not outcomes:
        return 1.0
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have same length")

    p = np.array(predictions)
    o = np.array(outcomes)
    return float(np.mean((p - o) ** 2))


def calibration_buckets(
    convictions: list[float],
    returns: list[float],
    n_buckets: int = 5,
) -> list[CalibrationBucket]:
    """Bin convictions and compute realized metrics per bucket.

    Args:
        convictions: Conviction scores at time of signal.
        returns: Realized returns over evaluation horizon.
        n_buckets: Number of conviction buckets.

    Returns:
        List of CalibrationBucket objects.
    """
    if not convictions or not returns:
        return []
    if len(convictions) != len(returns):
        raise ValueError("convictions and returns must have same length")

    c_arr = np.array(convictions)
    r_arr = np.array(returns)

    # Create evenly spaced bins across conviction range
    c_min, c_max = float(c_arr.min()), float(c_arr.max())
    if c_min == c_max:
        return [CalibrationBucket(
            conviction_min=c_min,
            conviction_max=c_max,
            count=len(convictions),
            hit_rate=float(np.mean(r_arr > 0)),
            avg_return=float(np.mean(r_arr)),
            avg_conviction=float(np.mean(c_arr)),
        )]

    edges = np.linspace(c_min, c_max, n_buckets + 1)
    buckets = []

    for i in range(n_buckets):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (c_arr >= lo) & (c_arr <= hi) if i == n_buckets - 1 else (c_arr >= lo) & (c_arr < hi)

        if not np.any(mask):
            continue

        bucket_c = c_arr[mask]
        bucket_r = r_arr[mask]

        # Hit rate: for positive conviction, positive return is correct
        # For negative conviction, negative return is correct
        correct = np.where(
            bucket_c >= 0,
            bucket_r > 0,
            bucket_r < 0,
        )

        buckets.append(CalibrationBucket(
            conviction_min=lo,
            conviction_max=hi,
            count=int(np.sum(mask)),
            hit_rate=float(np.mean(correct)),
            avg_return=float(np.mean(bucket_r)),
            avg_conviction=float(np.mean(bucket_c)),
        ))

    return buckets
