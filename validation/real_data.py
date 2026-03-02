"""Real market data fetcher for CGF validation.

Fetches actual market data and transforms it into SignalEmbeddedAsset format
compatible with run_cgf_over_signal_series. Data sources:

- **Prices / Returns**: yfinance daily OHLCV
- **FVS events**: Earnings surprises from yfinance (large misses → FVS events)
- **IV proxy**: VIX index (^VIX) as market-wide implied vol proxy
- **Forecasts**: Trailing mean + analyst drift (no public consensus time-series)
- **ITS**: No public IC debate data — left empty

Results are cached to disk to avoid repeated downloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from validation.signal_data import SignalEmbeddedAsset

logger = logging.getLogger(__name__)

# Default universe: diversified across sectors
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
DEFAULT_MARKET = "SPY"
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2023-12-31"
CACHE_DIR = Path(".cgf_cache")


@dataclass
class RealDataConfig:
    """Configuration for real data fetching."""

    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    market_ticker: str = DEFAULT_MARKET
    start_date: str = DEFAULT_START
    end_date: str = DEFAULT_END
    cache_dir: Path = CACHE_DIR
    # FVS: earnings miss threshold (standardized)
    earnings_miss_threshold: float = 1.5
    # Forecast: trailing window for expected return
    forecast_window: int = 21


def _cache_key(config: RealDataConfig) -> str:
    """Generate a deterministic cache key from config."""
    raw = f"{sorted(config.tickers)}-{config.market_ticker}-{config.start_date}-{config.end_date}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _load_cache(config: RealDataConfig) -> dict | None:
    """Load cached data if available and fresh."""
    cache_file = config.cache_dir / f"real_data_{_cache_key(config)}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            cached = json.load(f)
        # Cache valid for 7 days
        cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
        age_days = (datetime.now() - cached_at).days
        if age_days > 7:
            logger.info("Cache expired (%d days old), refetching", age_days)
            return None
        logger.info("Using cached real data (age: %d days)", age_days)
        return cached
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cache(config: RealDataConfig, data: dict) -> None:
    """Save fetched data to cache."""
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = config.cache_dir / f"real_data_{_cache_key(config)}.json"
    data["cached_at"] = datetime.now().isoformat()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    logger.info("Cached real data to %s", cache_file)


def _fetch_prices(ticker: str, start: str, end: str) -> np.ndarray:
    """Fetch daily adjusted close prices and compute log returns.

    Args:
        ticker: Stock ticker.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Array of daily log returns.
    """
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No price data for {ticker} from {start} to {end}")

    close = data["Close"].dropna()
    # Handle MultiIndex from yfinance (ticker column)
    if hasattr(close, "columns"):
        close = close.iloc[:, 0] if len(close.shape) > 1 else close

    prices = close.values.flatten().astype(float)
    returns = np.diff(np.log(prices))
    return returns


def _fetch_vix(start: str, end: str) -> np.ndarray:
    """Fetch VIX daily close as market-wide IV proxy.

    Returns annualized VIX level divided by 100 (so 20 VIX → 0.20).
    """
    import yfinance as yf

    data = yf.download("^VIX", start=start, end=end, progress=False)
    if data.empty:
        logger.warning("No VIX data available, using constant 0.20")
        return np.array([])

    close = data["Close"].dropna()
    if hasattr(close, "columns"):
        close = close.iloc[:, 0] if len(close.shape) > 1 else close

    vix = close.values.flatten().astype(float) / 100.0  # Convert to decimal
    return vix


def _fetch_earnings_events(
    ticker: str,
    start: str,
    end: str,
    returns: np.ndarray,
    start_date_dt: datetime,
    miss_threshold: float = 1.5,
) -> list[dict]:
    """Extract FVS events from earnings surprises.

    Uses yfinance earnings_dates to find dates where actual EPS missed
    estimate by more than miss_threshold standard deviations of recent
    earnings variability.

    Args:
        ticker: Stock ticker.
        start: Start date string.
        end: End date string.
        returns: Daily returns array (for day-index mapping).
        start_date_dt: Start date as datetime for day indexing.
        miss_threshold: Std devs for miss classification.

    Returns:
        List of FVS event dicts with day, event_type, severity.
    """
    import yfinance as yf

    events = []
    try:
        stock = yf.Ticker(ticker)
        # earnings_dates has columns: Earnings Date, EPS Estimate, Reported EPS, Surprise(%)
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            logger.info("No earnings data for %s", ticker)
            return events

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        for idx, row in earnings.iterrows():
            try:
                earn_date = idx.to_pydatetime()
                if earn_date.tzinfo is not None:
                    earn_date = earn_date.replace(tzinfo=None)

                if earn_date < start_dt or earn_date > end_dt:
                    continue

                # Get surprise percentage
                surprise_pct = row.get("Surprise(%)")
                if surprise_pct is None or np.isnan(surprise_pct):
                    continue

                # Large negative surprise → FVS event
                if surprise_pct < -miss_threshold:
                    day_idx = (earn_date - start_date_dt).days
                    # Map calendar days to trading day index (approximate)
                    trading_day = int(day_idx * 252 / 365)
                    if 0 <= trading_day < len(returns):
                        severity = min(0.9, 0.3 + abs(surprise_pct) / 100.0)
                        events.append({
                            "day": trading_day,
                            "event_type": "kpi_miss",
                            "severity": float(severity),
                        })
                        logger.debug(
                            "%s earnings miss at day %d: %.1f%% surprise",
                            ticker, trading_day, surprise_pct,
                        )

                # Large positive surprise could indicate guidance_cut coming
                # (market already priced in higher expectations)
                if surprise_pct > miss_threshold * 3:
                    day_idx = (earn_date - start_date_dt).days
                    trading_day = int(day_idx * 252 / 365)
                    if 0 <= trading_day < len(returns):
                        events.append({
                            "day": trading_day,
                            "event_type": "guidance_cut",
                            "severity": 0.3,
                        })

            except (AttributeError, TypeError, ValueError):
                continue

    except Exception as e:
        logger.warning("Failed to fetch earnings for %s: %s", ticker, e)

    return events


def _compute_trailing_forecasts(
    returns: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """Compute daily forecasts as trailing mean return.

    This is a naive forecast with ~50% directional accuracy (no real
    predictive signal). It's the baseline — real alpha comes from FVS
    and RRS components detecting structural breaks.

    Args:
        returns: Daily returns.
        window: Trailing window.

    Returns:
        Array of daily forecasts (same length as returns).
    """
    n = len(returns)
    forecasts = np.zeros(n)
    for t in range(n):
        start = max(0, t - window)
        if t > 0:
            forecasts[t] = np.mean(returns[start:t])
    return forecasts


def _align_vix_to_returns(
    vix: np.ndarray,
    n_returns: int,
) -> np.ndarray:
    """Align VIX series to return series length.

    VIX and stock returns may have different lengths due to trading day
    differences. Truncate or pad as needed.
    """
    if len(vix) == 0:
        return np.full(n_returns, 0.20 / np.sqrt(252))

    # Convert annualized VIX to daily vol for IV field
    daily_vix = vix / np.sqrt(252)

    if len(daily_vix) >= n_returns:
        return daily_vix[:n_returns]

    # Pad with last value
    padded = np.full(n_returns, daily_vix[-1])
    padded[: len(daily_vix)] = daily_vix
    return padded


def fetch_real_asset(
    ticker: str,
    start: str,
    end: str,
    vix_series: np.ndarray | None = None,
    config: RealDataConfig | None = None,
) -> SignalEmbeddedAsset:
    """Fetch real market data for one ticker and produce a SignalEmbeddedAsset.

    Active channels:
    - FE: trailing mean forecast (baseline, ~50% accuracy)
    - FVS: earnings surprise events from yfinance
    - RRS: VIX as IV proxy + realized HV from returns
    - ITS: empty (no public IC data)

    Args:
        ticker: Stock ticker (e.g., "AAPL").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        vix_series: Pre-fetched VIX series (avoids redundant downloads).
        config: Real data config.

    Returns:
        SignalEmbeddedAsset populated from real market data.
    """
    cfg = config or RealDataConfig()
    start_dt = datetime.strptime(start, "%Y-%m-%d")

    # 1. Fetch daily returns
    returns = _fetch_prices(ticker, start, end)
    n = len(returns)
    logger.info("%s: %d trading days of returns", ticker, n)

    # 2. Trailing mean forecasts (no real signal — baseline)
    forecasts = _compute_trailing_forecasts(returns, cfg.forecast_window)

    # 3. FVS events from earnings surprises
    fvs_schedule = _fetch_earnings_events(
        ticker, start, end, returns, start_dt,
        miss_threshold=cfg.earnings_miss_threshold,
    )
    logger.info("%s: %d FVS events from earnings", ticker, len(fvs_schedule))

    # 4. IV proxy from VIX
    if vix_series is not None:
        implied_vols = _align_vix_to_returns(vix_series, n)
    else:
        raw_vix = _fetch_vix(start, end)
        implied_vols = _align_vix_to_returns(raw_vix, n)

    # 5. Regime detection from trailing vol
    window = 63  # ~3 months
    regimes = np.zeros(n, dtype=int)
    for t in range(window, n):
        recent_vol = np.std(returns[t - 21 : t], ddof=1) * np.sqrt(252)
        baseline_vol = np.std(returns[t - window : t], ddof=1) * np.sqrt(252)
        if baseline_vol > 0 and recent_vol / baseline_vol > 1.5:
            regimes[t] = 1

    # 6. Annualized vol from full series
    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if n > 1 else 0.20
    ann_ret = float(np.mean(returns) * 252) if n > 0 else 0.0

    # Active channels: fe (baseline), fvs (if events exist), rrs (always — VIX is real)
    channels = ["fe", "rrs"]
    if fvs_schedule:
        channels.append("fvs")

    return SignalEmbeddedAsset(
        returns=returns,
        name=ticker,
        asset_type="real",
        forecasts=forecasts,
        fvs_schedule=fvs_schedule,
        implied_vols=implied_vols,
        regimes=regimes,
        debate_schedule=[],  # No public IC data
        track_record_score=0.0,
        annualized_vol=ann_vol,
        expected_drift=ann_ret,
        drawdown_days=[],  # Not known a priori on real data
        signal_channels=channels,
    )


def fetch_real_universe(
    config: RealDataConfig | None = None,
) -> tuple[list[SignalEmbeddedAsset], np.ndarray]:
    """Fetch real market data for a universe of stocks + market index.

    Checks disk cache first. Downloads via yfinance if not cached or expired.

    Args:
        config: Real data configuration.

    Returns:
        Tuple of (list of SignalEmbeddedAsset, market_returns array).

    Raises:
        ValueError: If no data could be fetched for any ticker.
    """
    cfg = config or RealDataConfig()

    # Check cache
    cached = _load_cache(cfg)
    if cached is not None:
        return _deserialize_cached(cached)

    # Fetch market returns
    logger.info("Fetching market data (%s)...", cfg.market_ticker)
    market_returns = _fetch_prices(cfg.market_ticker, cfg.start_date, cfg.end_date)

    # Fetch VIX once for all assets
    logger.info("Fetching VIX...")
    vix = _fetch_vix(cfg.start_date, cfg.end_date)

    # Fetch each ticker
    assets = []
    for ticker in cfg.tickers:
        try:
            logger.info("Fetching %s...", ticker)
            asset = fetch_real_asset(
                ticker, cfg.start_date, cfg.end_date,
                vix_series=vix, config=cfg,
            )
            assets.append(asset)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s — skipping", ticker, e)

    if not assets:
        raise ValueError("No data fetched for any ticker")

    # Trim all to same length
    min_len = min(len(a.returns) for a in assets)
    min_len = min(min_len, len(market_returns))
    for asset in assets:
        asset.returns = asset.returns[:min_len]
        asset.forecasts = asset.forecasts[:min_len]
        asset.implied_vols = asset.implied_vols[:min_len]
        asset.regimes = asset.regimes[:min_len]
        # Filter events/debates to valid day range
        asset.fvs_schedule = [e for e in asset.fvs_schedule if e["day"] < min_len]
        asset.debate_schedule = [d for d in asset.debate_schedule if d["day"] < min_len]
    market_returns = market_returns[:min_len]

    # Cache results
    _save_cache(cfg, _serialize_for_cache(assets, market_returns))

    return assets, market_returns


def _serialize_for_cache(
    assets: list[SignalEmbeddedAsset],
    market_returns: np.ndarray,
) -> dict:
    """Serialize assets and market returns to JSON-safe dict."""
    serialized_assets = []
    for a in assets:
        serialized_assets.append({
            "returns": a.returns.tolist(),
            "name": a.name,
            "asset_type": a.asset_type,
            "forecasts": a.forecasts.tolist(),
            "fvs_schedule": a.fvs_schedule,
            "implied_vols": a.implied_vols.tolist(),
            "regimes": a.regimes.tolist(),
            "debate_schedule": a.debate_schedule,
            "track_record_score": a.track_record_score,
            "annualized_vol": a.annualized_vol,
            "expected_drift": a.expected_drift,
            "drawdown_days": a.drawdown_days,
            "signal_channels": a.signal_channels,
        })
    return {
        "assets": serialized_assets,
        "market_returns": market_returns.tolist(),
    }


def _deserialize_cached(cached: dict) -> tuple[list[SignalEmbeddedAsset], np.ndarray]:
    """Deserialize cached data back to SignalEmbeddedAsset objects."""
    assets = []
    for a in cached["assets"]:
        assets.append(SignalEmbeddedAsset(
            returns=np.array(a["returns"]),
            name=a["name"],
            asset_type=a["asset_type"],
            forecasts=np.array(a["forecasts"]),
            fvs_schedule=a["fvs_schedule"],
            implied_vols=np.array(a["implied_vols"]),
            regimes=np.array(a["regimes"], dtype=int),
            debate_schedule=a["debate_schedule"],
            track_record_score=a["track_record_score"],
            annualized_vol=a["annualized_vol"],
            expected_drift=a["expected_drift"],
            drawdown_days=a["drawdown_days"],
            signal_channels=a["signal_channels"],
        ))
    market_returns = np.array(cached["market_returns"])
    return assets, market_returns


def transform_returns_to_asset(
    returns: np.ndarray,
    name: str,
    fvs_events: list[dict] | None = None,
    implied_vols: np.ndarray | None = None,
    forecast_window: int = 21,
) -> SignalEmbeddedAsset:
    """Transform pre-fetched return data into SignalEmbeddedAsset format.

    Utility for users who already have returns data and want to run it
    through the CGF signal pipeline without yfinance.

    Args:
        returns: Daily returns array.
        name: Asset name / ticker.
        fvs_events: Optional list of FVS event dicts.
        implied_vols: Optional implied vol series (daily).
        forecast_window: Window for trailing mean forecast.

    Returns:
        SignalEmbeddedAsset.
    """
    n = len(returns)
    forecasts = _compute_trailing_forecasts(returns, forecast_window)

    if implied_vols is None:
        # Use trailing vol as IV proxy
        implied_vols = np.full(n, np.std(returns, ddof=1) if n > 1 else 0.01)

    channels = ["fe", "rrs"]
    if fvs_events:
        channels.append("fvs")

    return SignalEmbeddedAsset(
        returns=returns,
        name=name,
        asset_type="transformed",
        forecasts=forecasts,
        fvs_schedule=fvs_events or [],
        implied_vols=implied_vols,
        regimes=np.zeros(n, dtype=int),
        debate_schedule=[],
        track_record_score=0.0,
        annualized_vol=float(np.std(returns, ddof=1) * np.sqrt(252)) if n > 1 else 0.20,
        expected_drift=float(np.mean(returns) * 252) if n > 0 else 0.0,
        drawdown_days=[],
        signal_channels=channels,
    )
