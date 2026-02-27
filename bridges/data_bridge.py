"""Bridge to financial-data-providers — market data, returns, and volatility.

Wraps the shared bds-data-providers package for CGF-specific data needs:
idiosyncratic vol, historical vol, implied vol, and return series.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)

# Attempt to import from the shared data providers package
try:
    from bds_data_providers.factory import get_provider_safe  # noqa: F401

    __all__ = ["get_provider_safe"]
    _HAS_PROVIDERS = True
except ImportError:
    _HAS_PROVIDERS = False
    logger.info("bds-data-providers not installed; data_bridge will use fallbacks")


def is_available() -> bool:
    """Check if the data providers package is installed."""
    return _HAS_PROVIDERS


def fetch_returns(
    tickers: list[str],
    start: date,
    end: date,
    provider_name: str = "Yahoo Finance",
) -> pl.DataFrame:
    """Fetch daily returns for a set of tickers.

    Args:
        tickers: List of ticker symbols.
        start: Start date.
        end: End date.
        provider_name: Data provider name.

    Returns:
        Polars DataFrame with columns: date, ticker, return.
    """
    if not _HAS_PROVIDERS:
        raise RuntimeError("bds-data-providers not installed")

    provider = get_provider_safe(provider_name)
    prices = provider.fetch_daily_prices(tickers, start, end)

    # Compute daily returns from adjusted close
    returns = (
        prices.sort(["ticker", "date"])
        .with_columns(
            pl.col("adj_close")
            .pct_change()
            .over("ticker")
            .alias("return")
        )
        .filter(pl.col("return").is_not_null())
        .select(["date", "ticker", "return"])
    )
    return returns


def compute_idiosyncratic_vol(
    ticker_returns: list[float],
    market_returns: list[float],
    annualize: bool = True,
) -> float:
    """Compute idiosyncratic volatility as residual vol from CAPM regression.

    Args:
        ticker_returns: Daily returns for the ticker.
        market_returns: Daily returns for the market index.
        annualize: Whether to annualize (multiply by sqrt(252)).

    Returns:
        Idiosyncratic volatility.
    """
    n = min(len(ticker_returns), len(market_returns))
    if n < 20:
        # Not enough data; return total vol as fallback
        vol = float(np.std(ticker_returns))
        return vol * np.sqrt(252) if annualize else vol

    ri = np.array(ticker_returns[:n])
    rm = np.array(market_returns[:n])

    slope, intercept, _, _, _ = stats.linregress(rm, ri)
    residuals = ri - (intercept + slope * rm)
    idio_vol = float(np.std(residuals, ddof=1))

    if annualize:
        idio_vol *= np.sqrt(252)

    return idio_vol


def compute_historical_vol(
    returns: list[float],
    window: int = 21,
    annualize: bool = True,
) -> float:
    """Compute historical realized volatility.

    Args:
        returns: Daily return series.
        window: Rolling window (default 21 = 1 month).
        annualize: Whether to annualize.

    Returns:
        Realized volatility.
    """
    if len(returns) < window:
        window = max(len(returns), 2)

    recent = returns[-window:]
    vol = float(np.std(recent, ddof=1))

    if annualize:
        vol *= np.sqrt(252)

    return vol
