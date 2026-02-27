"""Update alignment — compare analyst/AI conviction updates vs model-prescribed.

Measures whether humans/AI over-react or under-react to new information
relative to what the CGF model prescribes.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class AlignmentResult(BaseModel):
    """Result of update alignment analysis."""

    correlation: float = Field(description="Correlation between analyst and model deltas")
    mean_analyst_magnitude: float = Field(description="Mean |delta_analyst|")
    mean_model_magnitude: float = Field(description="Mean |delta_model|")
    bias_direction: str = Field(
        description="'over_reactive', 'under_reactive', or 'aligned'"
    )
    n_observations: int = Field(description="Number of paired observations")


def compute_alignment(
    analyst_deltas: list[float],
    model_deltas: list[float],
) -> AlignmentResult:
    """Compare analyst conviction updates vs model-prescribed updates.

    Args:
        analyst_deltas: Actual conviction changes by analyst/AI per event.
        model_deltas: Model-prescribed conviction changes per event.

    Returns:
        AlignmentResult with correlation and bias direction.
    """
    if not analyst_deltas or not model_deltas:
        return AlignmentResult(
            correlation=0.0,
            mean_analyst_magnitude=0.0,
            mean_model_magnitude=0.0,
            bias_direction="aligned",
            n_observations=0,
        )

    n = min(len(analyst_deltas), len(model_deltas))
    a = np.array(analyst_deltas[:n])
    m = np.array(model_deltas[:n])

    mean_a_mag = float(np.mean(np.abs(a)))
    mean_m_mag = float(np.mean(np.abs(m)))

    # Correlation
    corr = 0.0 if np.std(a) == 0 or np.std(m) == 0 else float(np.corrcoef(a, m)[0, 1])

    # Bias direction
    if mean_m_mag > 0:
        ratio = mean_a_mag / mean_m_mag
        if ratio > 1.2:
            bias = "over_reactive"
        elif ratio < 0.8:
            bias = "under_reactive"
        else:
            bias = "aligned"
    else:
        bias = "aligned"

    return AlignmentResult(
        correlation=corr,
        mean_analyst_magnitude=mean_a_mag,
        mean_model_magnitude=mean_m_mag,
        bias_direction=bias,
        n_observations=n,
    )
