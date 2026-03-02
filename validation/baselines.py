"""Baseline portfolio strategies and performance metrics.

Baselines provide simple, well-understood strategies to compare against
CGF-driven portfolio construction. The goal is not to beat these on every
metric, but to verify CGF adds value on the dimensions it targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PortfolioMetrics:
    """Standard portfolio performance metrics."""

    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    n_days: int


def compute_metrics(
    daily_returns: np.ndarray,
    turnover_series: np.ndarray | None = None,
    risk_free_rate: float = 0.04,
) -> PortfolioMetrics:
    """Compute standard performance metrics from daily portfolio returns.

    Args:
        daily_returns: Array of daily portfolio returns.
        turnover_series: Optional daily turnover values.
        risk_free_rate: Annualized risk-free rate for Sharpe.

    Returns:
        PortfolioMetrics with annualized stats.
    """
    n = len(daily_returns)
    if n == 0:
        return PortfolioMetrics(
            annualized_return=0.0,
            annualized_vol=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            n_days=0,
        )

    ann_ret = float(np.mean(daily_returns)) * 252.0
    ann_vol = float(np.std(daily_returns, ddof=1)) * np.sqrt(252.0) if n > 1 else 0.0
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-10 else 0.0

    # Max drawdown from cumulative returns
    cum = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    avg_turnover = float(np.mean(turnover_series)) * 252.0 if turnover_series is not None else 0.0

    return PortfolioMetrics(
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        turnover=avg_turnover,
        n_days=n,
    )


def equal_weight_strategy(
    asset_returns: np.ndarray,
    rebalance_freq: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight buy-and-rebalance strategy.

    Args:
        asset_returns: Array of shape (n_days, n_assets).
        rebalance_freq: Days between rebalancing.

    Returns:
        Tuple of (portfolio_returns, turnover_series).
    """
    n_days, n_assets = asset_returns.shape
    if n_assets == 0:
        return np.zeros(n_days), np.zeros(n_days)

    target_w = np.ones(n_assets) / n_assets
    weights = target_w.copy()
    port_returns = np.zeros(n_days)
    turnover = np.zeros(n_days)

    for t in range(n_days):
        port_returns[t] = np.dot(weights, asset_returns[t])

        # Update weights for drift
        weights = weights * (1.0 + asset_returns[t])
        weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else target_w

        # Rebalance
        if (t + 1) % rebalance_freq == 0:
            turnover[t] = float(np.sum(np.abs(weights - target_w)))
            weights = target_w.copy()

    return port_returns, turnover


def buy_and_hold_strategy(
    asset_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold: equal-weight at inception, never rebalance.

    Args:
        asset_returns: Array of shape (n_days, n_assets).

    Returns:
        Tuple of (portfolio_returns, turnover_series).
    """
    n_days, n_assets = asset_returns.shape
    if n_assets == 0:
        return np.zeros(n_days), np.zeros(n_days)

    weights = np.ones(n_assets) / n_assets
    port_returns = np.zeros(n_days)

    for t in range(n_days):
        port_returns[t] = np.dot(weights, asset_returns[t])
        weights = weights * (1.0 + asset_returns[t])
        total = np.sum(np.abs(weights))
        if total > 0:
            weights = weights / total

    return port_returns, np.zeros(n_days)


def momentum_strategy(
    asset_returns: np.ndarray,
    lookback: int = 63,
    rebalance_freq: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple cross-sectional momentum: long winners, short losers.

    Weights proportional to trailing cumulative return rank.

    Args:
        asset_returns: Array of shape (n_days, n_assets).
        lookback: Days for momentum calculation.
        rebalance_freq: Days between rebalancing.

    Returns:
        Tuple of (portfolio_returns, turnover_series).
    """
    n_days, n_assets = asset_returns.shape
    if n_assets < 2:
        return equal_weight_strategy(asset_returns, rebalance_freq)

    weights = np.ones(n_assets) / n_assets
    port_returns = np.zeros(n_days)
    turnover = np.zeros(n_days)

    for t in range(n_days):
        port_returns[t] = np.dot(weights, asset_returns[t])

        if (t + 1) % rebalance_freq == 0 and t >= lookback:
            # Compute trailing cumulative returns
            cum_ret = np.sum(asset_returns[t - lookback + 1 : t + 1], axis=0)
            # Demean to create long/short
            demeaned = cum_ret - np.mean(cum_ret)
            abs_sum = np.sum(np.abs(demeaned))
            new_weights = (
                demeaned / abs_sum if abs_sum > 1e-10 else np.ones(n_assets) / n_assets
            )

            turnover[t] = float(np.sum(np.abs(new_weights - weights)))
            weights = new_weights

    return port_returns, turnover
