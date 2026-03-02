"""Continuous regime detection for volatility-aware RRS scaling.

Reference: Aydinhan, Kolm, Mulvey & Shu, Annals of Operations Research 2024;
           Hamilton 1989 (Markov-switching models).

Uses an EMA-smoothed volatility baseline with Bayesian posterior updates to
estimate the probability of being in a high-volatility regime. The transition
penalty prevents whipsaw between regimes on noisy vol observations.

When regime detection is active, RRS is scaled by (1 + p_high_vol):
- Low-vol regime (p_high_vol ≈ 0): RRS unchanged
- High-vol regime (p_high_vol ≈ 1): RRS amplified up to 2x
"""

from __future__ import annotations

from dataclasses import dataclass

from components.risk_regime import compute_rrs as _base_compute_rrs


@dataclass
class RegimeDetectorConfig:
    """Configuration for continuous regime detection.

    Args:
        vol_threshold: Multiple of baseline vol that triggers high-vol regime.
            Values > 1 required. Default 1.5 = 50% above baseline.
        transition_penalty: Penalty on regime transitions to prevent whipsaw.
            Higher = more sticky regimes. Range [0, 0.5].
        ema_alpha: EMA smoothing factor for baseline vol. Smaller = smoother.
            Default 0.05 ≈ 20-day effective window.
        min_observations: Minimum observations before regime detection activates.
    """

    vol_threshold: float = 1.5
    transition_penalty: float = 0.1
    ema_alpha: float = 0.05
    min_observations: int = 22


@dataclass
class RegimeState:
    """Current state of the regime detector."""

    p_high_vol: float = 0.5
    smoothed_vol: float = 0.0
    n_observations: int = 0


class RegimeDetector:
    """Bayesian regime detector with EMA baseline and transition penalty.

    Maintains a running estimate of baseline volatility and uses Bayesian
    updating to compute the probability of being in a high-vol regime.
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self.state = RegimeState()

    def update(self, vol_observation: float) -> RegimeState:
        """Update regime state with a new volatility observation.

        Args:
            vol_observation: Current volatility observation (e.g., sigma_idio).

        Returns:
            Updated RegimeState.
        """
        s = self.state
        c = self.config
        s.n_observations += 1

        # Update EMA baseline
        if s.n_observations == 1:
            s.smoothed_vol = vol_observation
            # Stay at uninformative prior during warmup
            return s

        s.smoothed_vol = c.ema_alpha * vol_observation + (1 - c.ema_alpha) * s.smoothed_vol

        # Only compute regime probability after warmup
        if s.n_observations < c.min_observations:
            return s

        # Bayesian update: likelihood ratio based on distance from baseline
        vol_ratio = vol_observation / s.smoothed_vol if s.smoothed_vol > 0 else 1.0

        # Likelihood of high-vol regime: sigmoid of (ratio - threshold)
        # Higher ratio → higher probability of high-vol regime
        distance = vol_ratio - c.vol_threshold
        likelihood_high = 1.0 / (1.0 + _exp_safe(-5.0 * distance))

        # Prior with transition penalty (sticky regimes)
        prior_high = s.p_high_vol * (1.0 - c.transition_penalty) + (
            (1.0 - s.p_high_vol) * c.transition_penalty
        )

        # Posterior update
        numerator = likelihood_high * prior_high
        denominator = numerator + (1.0 - likelihood_high) * (1.0 - prior_high)
        s.p_high_vol = numerator / denominator if denominator > 0 else 0.5

        return s

    def compute_rrs(
        self,
        sigma_current: float,
        sigma_prev: float,
        implied_vol: float | None = None,
        historical_vol: float | None = None,
    ) -> float:
        """Compute regime-scaled RRS.

        Wraps the base compute_rrs() and scales by (1 + p_high_vol).
        Low-vol regime → RRS unchanged. High-vol regime → RRS amplified up to 2x.

        Args:
            sigma_current: Current idiosyncratic vol.
            sigma_prev: Previous period idiosyncratic vol.
            implied_vol: Optional implied vol.
            historical_vol: Optional historical vol.

        Returns:
            Regime-scaled RRS value.
        """
        base_rrs = _base_compute_rrs(sigma_current, sigma_prev, implied_vol, historical_vol)
        return base_rrs * (1.0 + self.state.p_high_vol)

    def reset(self) -> None:
        """Reset to uninformative prior."""
        self.state = RegimeState()


def _exp_safe(x: float) -> float:
    """Numerically safe exponential (clamp to avoid overflow)."""
    import math

    x = max(-500.0, min(500.0, x))
    return math.exp(x)
