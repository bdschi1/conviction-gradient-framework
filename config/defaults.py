"""Default hyperparameters for the Conviction Gradient Framework."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConvictionParams(BaseModel):
    """All tunable parameters for the conviction update engine.

    Loss weights (w1-w4) govern relative importance of each loss component.
    Gradient scaling (lambda1-4) control sensitivity of the gradient to each component.
    Learning rate params control how elastic conviction is to new information.
    """

    # Loss weights
    w1: float = Field(default=0.30, ge=0, le=1, description="Forecast error weight")
    w2: float = Field(default=0.25, ge=0, le=1, description="Fundamental violation weight")
    w3: float = Field(default=0.25, ge=0, le=1, description="Risk regime shift weight")
    w4: float = Field(default=0.20, ge=0, le=1, description="Independent thesis shift weight")

    # Gradient scaling
    lambda1: float = Field(default=1.0, ge=0, description="FE gradient scaling")
    lambda2: float = Field(default=1.0, ge=0, description="FVS gradient scaling")
    lambda3: float = Field(default=1.0, ge=0, description="RRS gradient scaling")
    lambda4: float = Field(default=1.0, ge=0, description="ITS gradient scaling")

    # Learning rate
    kappa: float = Field(default=0.1, gt=0, description="Base learning rate scaling constant")
    alpha_min: float = Field(default=0.01, gt=0, description="Minimum learning rate")
    alpha_max: float = Field(default=0.5, gt=0, description="Maximum learning rate")

    # Update rule
    beta: float = Field(default=0.1, ge=0, le=1, description="Momentum stabilizer")
    phi: float = Field(default=0.95, ge=0, le=1, description="AR(1) persistence parameter")
    C_max: float = Field(default=5.0, gt=0, description="Conviction clamp magnitude")

    # Failure modes
    flip_window_days: int = Field(
        default=5, ge=1, description="Days to detect conviction sign flip"
    )
    fvs_reset_threshold: float = Field(
        default=0.7, ge=0, le=1, description="FVS threshold for structural reset"
    )
    vol_double_threshold: float = Field(
        default=2.0, gt=1, description="Vol ratio threshold for structural reset"
    )

    # Adaptive loss weights
    adaptive_weights: bool = Field(
        default=False, description="Enable adaptive component weight learning"
    )
    adaptive_lookback: int = Field(
        default=63, ge=10, description="Lookback window for adaptive weight estimation (days)"
    )
    adaptive_decay: float = Field(
        default=0.97, ge=0.5, le=0.999, description="EMA decay for adaptive weight updates"
    )

    # Continuous regime detection
    continuous_regime: bool = Field(
        default=False, description="Enable continuous regime-aware RRS scaling"
    )
    regime_vol_threshold: float = Field(
        default=1.5, gt=1.0, description="Vol multiple for high-vol regime detection"
    )
    regime_transition_penalty: float = Field(
        default=0.1, ge=0, le=0.5, description="Penalty on regime transitions (anti-whipsaw)"
    )


class SizingParams(BaseModel):
    """Parameters for the position sizing overlay."""

    max_gross_exposure: float = Field(default=2.0, gt=0, description="Max gross exposure (200%)")
    max_net_exposure: float = Field(default=0.5, ge=0, description="Max net exposure (50%)")
    max_position_pct: float = Field(default=0.05, gt=0, le=1, description="Max single position %")
    min_position_pct: float = Field(default=0.005, ge=0, le=1, description="Min single position %")
    max_sector_net: float = Field(default=0.25, ge=0, le=1, description="Max sector net exposure")
