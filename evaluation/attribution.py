"""Forecast attribution decomposition.

Decomposes total forecast error into component sources:
- Alpha miss: idiosyncratic return forecast error
- Beta drift: market exposure error
- Vol misestimate: volatility forecast error
- Structural thesis error: fundamental violation component
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AttributionResult(BaseModel):
    """Decomposition of total forecast error into sources."""

    total_error: float = Field(description="Total forecast error")
    alpha_miss: float = Field(description="Idiosyncratic return miss")
    beta_drift: float = Field(description="Market exposure error")
    vol_misestimate: float = Field(description="Volatility forecast error")
    structural_error: float = Field(description="Fundamental thesis error")
    residual: float = Field(description="Unexplained portion")

    @property
    def explained_pct(self) -> float:
        """Fraction of total error explained by known components."""
        if abs(self.total_error) < 1e-10:
            return 1.0
        explained = (
            abs(self.alpha_miss) + abs(self.beta_drift)
            + abs(self.vol_misestimate) + abs(self.structural_error)
        )
        return min(explained / abs(self.total_error), 1.0)


def decompose_error(
    total_return_error: float,
    market_return_error: float,
    beta: float,
    vol_actual: float,
    vol_forecast: float,
    fvs: float,
) -> AttributionResult:
    """Decompose total forecast error into attributed components.

    Args:
        total_return_error: Realized return - Expected return.
        market_return_error: Market return - Expected market return.
        beta: Position beta to market.
        vol_actual: Realized idiosyncratic vol.
        vol_forecast: Forecast idiosyncratic vol.
        fvs: Fundamental Violation Score (0-1).

    Returns:
        AttributionResult with component breakdown.
    """
    # Beta drift: how much market exposure contributed to the miss
    beta_drift = beta * market_return_error

    # Vol misestimate: scaled by the vol forecast error ratio
    vol_diff = vol_actual - vol_forecast
    vol_misestimate = vol_diff  # raw vol forecast error

    # Structural error: FVS-weighted portion of the total error
    structural_error = fvs * abs(total_return_error) * (1.0 if total_return_error < 0 else -0.5)

    # Alpha miss: residual after removing beta and structural
    alpha_miss = total_return_error - beta_drift - structural_error

    # Residual
    residual = total_return_error - (alpha_miss + beta_drift + structural_error)

    return AttributionResult(
        total_error=total_return_error,
        alpha_miss=alpha_miss,
        beta_drift=beta_drift,
        vol_misestimate=vol_misestimate,
        structural_error=structural_error,
        residual=residual,
    )
