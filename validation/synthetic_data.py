"""Synthetic data generators for validation.

Three asset types with known statistical properties:
- GBM (Geometric Brownian Motion) — trending assets
- OU (Ornstein-Uhlenbeck) — mean-reverting assets
- Regime-switching — 2-state Markov with distinct vol regimes

All generators are seed-reproducible and produce 504 trading days (2 years) by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticAsset:
    """Container for synthetic return series with metadata."""

    returns: np.ndarray
    name: str
    asset_type: str
    annualized_vol: float
    expected_drift: float


def generate_gbm(
    n_days: int = 504,
    mu: float = 0.08,
    sigma: float = 0.20,
    seed: int = 42,
    name: str = "GBM_TREND",
) -> SyntheticAsset:
    """Generate trending asset via Geometric Brownian Motion.

    Args:
        n_days: Number of trading days.
        mu: Annualized drift (expected return).
        sigma: Annualized volatility.
        seed: Random seed for reproducibility.
        name: Asset identifier.

    Returns:
        SyntheticAsset with daily log returns.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    return SyntheticAsset(
        returns=daily_returns,
        name=name,
        asset_type="gbm",
        annualized_vol=sigma,
        expected_drift=mu,
    )


def generate_ou(
    n_days: int = 504,
    theta: float = 0.15,
    mu: float = 0.0,
    sigma: float = 0.20,
    seed: int = 43,
    name: str = "OU_REVERT",
) -> SyntheticAsset:
    """Generate mean-reverting asset via Ornstein-Uhlenbeck process.

    Args:
        n_days: Number of trading days.
        theta: Mean-reversion speed.
        mu: Long-run mean (of log price level).
        sigma: Annualized volatility of innovations.
        seed: Random seed.
        name: Asset identifier.

    Returns:
        SyntheticAsset with daily returns derived from OU price level.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    x = np.zeros(n_days + 1)
    x[0] = 0.0

    for t in range(n_days):
        x[t + 1] = x[t] + theta * (mu - x[t]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()

    # Convert level changes to returns
    daily_returns = np.diff(x)
    return SyntheticAsset(
        returns=daily_returns,
        name=name,
        asset_type="ou",
        annualized_vol=sigma,
        expected_drift=0.0,
    )


def generate_regime_switching(
    n_days: int = 504,
    mu_low: float = 0.10,
    mu_high: float = -0.05,
    sigma_low: float = 0.15,
    sigma_high: float = 0.35,
    p_stay_low: float = 0.98,
    p_stay_high: float = 0.95,
    seed: int = 44,
    name: str = "REGIME_SW",
) -> SyntheticAsset:
    """Generate regime-switching asset with 2-state Markov chain.

    Low-vol regime: positive drift, low vol.
    High-vol regime: negative drift, high vol.

    Args:
        n_days: Number of trading days.
        mu_low: Annualized drift in low-vol regime.
        mu_high: Annualized drift in high-vol regime.
        sigma_low: Annualized vol in low-vol regime.
        sigma_high: Annualized vol in high-vol regime.
        p_stay_low: Probability of staying in low-vol regime.
        p_stay_high: Probability of staying in high-vol regime.
        seed: Random seed.
        name: Asset identifier.

    Returns:
        SyntheticAsset with daily returns and blended vol.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    regime = 0  # Start in low-vol regime
    returns = np.zeros(n_days)
    regimes = np.zeros(n_days, dtype=int)

    for t in range(n_days):
        regimes[t] = regime
        if regime == 0:
            drift = (mu_low - 0.5 * sigma_low**2) * dt
            returns[t] = drift + sigma_low * np.sqrt(dt) * rng.standard_normal()
            if rng.random() > p_stay_low:
                regime = 1
        else:
            drift = (mu_high - 0.5 * sigma_high**2) * dt
            returns[t] = drift + sigma_high * np.sqrt(dt) * rng.standard_normal()
            if rng.random() > p_stay_high:
                regime = 0

    blended_vol = (sigma_low + sigma_high) / 2.0
    return SyntheticAsset(
        returns=returns,
        name=name,
        asset_type="regime_switching",
        annualized_vol=blended_vol,
        expected_drift=(mu_low + mu_high) / 2.0,
    )


def generate_market_returns(
    n_days: int = 504,
    mu: float = 0.07,
    sigma: float = 0.16,
    seed: int = 100,
) -> np.ndarray:
    """Generate market index returns (SPY proxy) for factor regressions.

    Args:
        n_days: Number of trading days.
        mu: Annualized market drift.
        sigma: Annualized market vol.
        seed: Random seed.

    Returns:
        Array of daily market returns.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    return (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)


def generate_universe(
    n_days: int = 504,
    seed: int = 42,
) -> tuple[list[SyntheticAsset], np.ndarray]:
    """Generate a 3-asset universe with market returns.

    Returns one of each asset type (GBM, OU, regime-switching) plus market returns.

    Args:
        n_days: Trading days per asset.
        seed: Base seed (each asset offsets by index).

    Returns:
        Tuple of (list of SyntheticAsset, market_returns array).
    """
    assets = [
        generate_gbm(n_days=n_days, seed=seed, name="TREND_A"),
        generate_ou(n_days=n_days, seed=seed + 1, name="REVERT_B"),
        generate_regime_switching(n_days=n_days, seed=seed + 2, name="REGIME_C"),
    ]
    market = generate_market_returns(n_days=n_days, seed=seed + 100)
    return assets, market
