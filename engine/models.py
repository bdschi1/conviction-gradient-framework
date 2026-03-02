"""Core data models for the conviction engine."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class LossComponents(BaseModel):
    """Individual loss components and total loss at time t."""

    fe: float = Field(description="Forecast Error (standardized return miss)")
    fvs: float = Field(description="Fundamental Violation Score (0-1)")
    rrs: float = Field(description="Risk Regime Shift")
    its: float = Field(description="Independent Thesis Shift")
    total_loss: float = Field(description="Weighted total loss L_t")

    @property
    def ads(self) -> float:
        """Backward-compatible alias for ITS."""
        return self.its


class GradientResult(BaseModel):
    """Gradient computation result at time t."""

    gradient_value: float = Field(description="Total gradient nabla L_t")
    component_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component gradient contributions",
    )
    learning_rate: float = Field(description="Adaptive learning rate alpha_t")


class ConvictionState(BaseModel):
    """Full conviction state for one instrument at one point in time."""

    instrument_id: str = Field(description="Ticker or instrument identifier")
    as_of_date: date = Field(description="Date of this state")
    conviction: float = Field(description="Current conviction C_t")
    conviction_prev: float = Field(default=0.0, description="Previous conviction C_{t-1}")
    expected_return: float = Field(default=0.0, description="E[R_12m]")
    idiosyncratic_vol: float = Field(default=0.0, description="sigma_idiosyncratic")
    alpha_t: float = Field(default=0.0, description="Learning rate used for this update")
    loss_components: LossComponents | None = Field(
        default=None, description="Loss breakdown for this update"
    )
    gradient: GradientResult | None = Field(
        default=None, description="Gradient breakdown for this update"
    )


class InstrumentData(BaseModel):
    """Input data for computing a conviction update for one instrument."""

    instrument_id: str
    as_of_date: date
    realized_return: float = Field(description="Actual return R_t over evaluation period")
    expected_return: float = Field(description="Prior expected return E_{t-1}[R_t]")
    sigma_expected: float = Field(gt=0, description="Expected volatility for FE normalization")
    sigma_idio_current: float = Field(gt=0, description="Current idiosyncratic vol")
    sigma_idio_prev: float = Field(gt=0, description="Previous period idiosyncratic vol")
    implied_vol: float | None = Field(default=None, description="Implied vol (if available)")
    historical_vol: float | None = Field(default=None, description="Historical vol (if available)")
    fvs_events: list[dict] = Field(default_factory=list, description="FVS event dicts")
    p_pre: list[float] = Field(default_factory=list, description="Pre-debate probabilities")
    p_post: list[float] = Field(default_factory=list, description="Post-debate probabilities")
    pm_conviction: float | None = Field(default=None, description="PM conviction level (0-1)")
    analyst_convictions: list[float] = Field(
        default_factory=list, description="Analyst conviction levels (0-1 each)"
    )
    position_type: str | None = Field(
        default=None,
        description="Position type: alpha_long, core_long, alpha_short, hedge_short",
    )
    info_half_life: float = Field(default=1.0, ge=0, description="Information decay parameter")
    track_record_score: float = Field(default=0.0, ge=0, description="Analyst track record")
