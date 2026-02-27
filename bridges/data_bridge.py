"""Bridge to financial-data-providers — market data, returns, and volatility.

Wraps the shared bds-data-providers package for CGF-specific data needs:
idiosyncratic vol, historical vol, implied vol, and return series.

Supports multiple idiosyncratic volatility estimation methods:
    1. CAPM (default) — single-factor regression vs SPY
    2. FF3 — Fama-French 3-factor regression (Mkt-RF, SMB, HML)
    3. EWMA — exponentially weighted residuals (lambda=0.94)
    4. GARCH — GARCH(1,1) conditional variance forecast
"""

from __future__ import annotations

import io
import logging
from datetime import date
from enum import Enum

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

# Attempt to import arch for GARCH
try:
    from arch import arch_model  # noqa: F401

    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False


class VolMethod(Enum):
    """Idiosyncratic volatility estimation methods."""

    CAPM = "capm"
    FF3 = "ff3"
    EWMA = "ewma"
    GARCH = "garch"


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


# ---------------------------------------------------------------------------
# FF3 factor data
# ---------------------------------------------------------------------------
_FF3_CACHE: dict[str, np.ndarray] | None = None


def _fetch_ff3_factors(n_days: int) -> dict[str, np.ndarray] | None:
    """Fetch Fama-French 3-factor daily data from Kenneth French data library.

    Returns dict with keys 'mkt_rf', 'smb', 'hml', 'rf' as numpy arrays
    of daily returns (in decimal, not percent), or None on failure.
    Caches the result for the session.
    """
    global _FF3_CACHE
    if _FF3_CACHE is not None:
        # Trim to requested length
        trimmed = {}
        for k, v in _FF3_CACHE.items():
            trimmed[k] = v[-n_days:] if len(v) >= n_days else v
        return trimmed

    try:
        import urllib.request

        url = (
            "https://mba.tuck.dartmouth.edu/pages/faculty/"
            "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        )
        import zipfile

        req = urllib.request.Request(url, headers={"User-Agent": "CGF/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
            raw = zf.read(csv_name).decode("utf-8")

        # Parse the CSV — skip header lines, stop at annual section
        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if "Mkt-RF" in line and "SMB" in line and "HML" in line:
                header_idx = i
                break
        if header_idx is None:
            return None

        mkt_rf, smb, hml, rf = [], [], [], []
        for line in lines[header_idx + 1 :]:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                int(parts[0].strip())  # date as YYYYMMDD
            except ValueError:
                break  # hit annual section or footer
            try:
                mkt_rf.append(float(parts[1].strip()) / 100.0)
                smb.append(float(parts[2].strip()) / 100.0)
                hml.append(float(parts[3].strip()) / 100.0)
                rf.append(float(parts[4].strip()) / 100.0)
            except ValueError:
                continue

        if len(mkt_rf) < 100:
            return None

        _FF3_CACHE = {
            "mkt_rf": np.array(mkt_rf),
            "smb": np.array(smb),
            "hml": np.array(hml),
            "rf": np.array(rf),
        }
        trimmed = {}
        for k, v in _FF3_CACHE.items():
            trimmed[k] = v[-n_days:] if len(v) >= n_days else v
        return trimmed

    except Exception:
        logger.exception("Failed to fetch FF3 factors")
        return None


# ---------------------------------------------------------------------------
# Idiosyncratic volatility methods
# ---------------------------------------------------------------------------
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
    return compute_idio_vol_capm(ticker_returns, market_returns, annualize)


def compute_idio_vol_capm(
    ticker_returns: list[float],
    market_returns: list[float],
    annualize: bool = True,
) -> float:
    """CAPM single-factor idiosyncratic volatility.

    Regresses stock returns on market returns (SPY). Residual std = idio vol.
    """
    n = min(len(ticker_returns), len(market_returns))
    if n < 20:
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


def compute_idio_vol_ff3(
    ticker_returns: list[float],
    annualize: bool = True,
) -> float:
    """Fama-French 3-factor idiosyncratic volatility.

    Regresses stock returns on Mkt-RF, SMB, HML. Residual std captures vol
    not explained by market, size, or value factors — analogous to what
    Barra-style multi-factor risk models compute.
    """
    n = len(ticker_returns)
    if n < 20:
        vol = float(np.std(ticker_returns))
        return vol * np.sqrt(252) if annualize else vol

    factors = _fetch_ff3_factors(n)
    if factors is None:
        logger.warning("FF3 data unavailable; falling back to CAPM-style total vol")
        vol = float(np.std(ticker_returns, ddof=1))
        return vol * np.sqrt(252) if annualize else vol

    # Align lengths
    min_len = min(n, len(factors["mkt_rf"]))
    if min_len < 20:
        vol = float(np.std(ticker_returns, ddof=1))
        return vol * np.sqrt(252) if annualize else vol

    ri = np.array(ticker_returns[-min_len:])
    mkt_rf = factors["mkt_rf"][-min_len:]
    smb_arr = factors["smb"][-min_len:]
    hml_arr = factors["hml"][-min_len:]
    rf = factors["rf"][-min_len:]

    # Excess returns
    ri_excess = ri - rf

    # Multiple regression: ri_excess = a + b1*mkt_rf + b2*smb + b3*hml + e
    design = np.column_stack([np.ones(min_len), mkt_rf, smb_arr, hml_arr])
    try:
        betas, _, _, _ = np.linalg.lstsq(design, ri_excess, rcond=None)
        predicted = design @ betas
        residuals = ri_excess - predicted
        idio_vol = float(np.std(residuals, ddof=4))  # 4 params estimated
    except np.linalg.LinAlgError:
        idio_vol = float(np.std(ri, ddof=1))

    if annualize:
        idio_vol *= np.sqrt(252)

    return idio_vol


def compute_idio_vol_ewma(
    ticker_returns: list[float],
    market_returns: list[float],
    lam: float = 0.94,
    annualize: bool = True,
) -> float:
    """EWMA (RiskMetrics) idiosyncratic volatility.

    Applies exponential decay (lambda=0.94) to squared CAPM residuals.
    More responsive to recent vol changes than equal-weighted windows.
    """
    n = min(len(ticker_returns), len(market_returns))
    if n < 20:
        vol = float(np.std(ticker_returns))
        return vol * np.sqrt(252) if annualize else vol

    ri = np.array(ticker_returns[:n])
    rm = np.array(market_returns[:n])

    slope, intercept, _, _, _ = stats.linregress(rm, ri)
    residuals = ri - (intercept + slope * rm)

    # EWMA variance
    sq_resid = residuals**2
    ewma_var = float(sq_resid[0])
    for t in range(1, len(sq_resid)):
        ewma_var = lam * ewma_var + (1 - lam) * float(sq_resid[t])

    idio_vol = np.sqrt(ewma_var)

    if annualize:
        idio_vol *= np.sqrt(252)

    return float(idio_vol)


def compute_idio_vol_garch(
    ticker_returns: list[float],
    market_returns: list[float],
    annualize: bool = True,
) -> float:
    """GARCH(1,1) idiosyncratic volatility forecast.

    Fits a GARCH(1,1) model to CAPM residuals and returns the next-period
    conditional standard deviation. This is a forward-looking estimate —
    the forecast predicts tomorrow's vol, not a backward average.

    Requires the `arch` package. Falls back to CAPM if unavailable.
    """
    if not _HAS_ARCH:
        logger.warning("arch package not installed; falling back to CAPM")
        return compute_idio_vol_capm(ticker_returns, market_returns, annualize)

    n = min(len(ticker_returns), len(market_returns))
    if n < 50:
        return compute_idio_vol_capm(ticker_returns, market_returns, annualize)

    ri = np.array(ticker_returns[:n])
    rm = np.array(market_returns[:n])

    slope, intercept, _, _, _ = stats.linregress(rm, ri)
    residuals = ri - (intercept + slope * rm)

    # Scale residuals to percent for numerical stability
    resid_pct = residuals * 100.0

    try:
        model = arch_model(resid_pct, vol="Garch", p=1, q=1, mean="Zero")
        result = model.fit(disp="off", show_warning=False)
        forecast = result.forecast(horizon=1)
        cond_var = forecast.variance.values[-1, 0]
        idio_vol = np.sqrt(cond_var) / 100.0  # back to decimal
    except Exception:
        logger.warning("GARCH fit failed; falling back to CAPM")
        return compute_idio_vol_capm(ticker_returns, market_returns, annualize)

    if annualize:
        idio_vol *= np.sqrt(252)

    return float(idio_vol)


def compute_idio_vol(
    method: VolMethod,
    ticker_returns: list[float],
    market_returns: list[float],
    annualize: bool = True,
) -> float:
    """Dispatch idiosyncratic vol computation to the selected method.

    Args:
        method: Which estimation method to use.
        ticker_returns: Daily returns for the ticker.
        market_returns: Daily returns for the market index (needed for
            CAPM, EWMA, GARCH; ignored for FF3).
        annualize: Whether to annualize.

    Returns:
        Idiosyncratic volatility estimate.
    """
    if method == VolMethod.FF3:
        return compute_idio_vol_ff3(ticker_returns, annualize)
    if method == VolMethod.EWMA:
        return compute_idio_vol_ewma(ticker_returns, market_returns, annualize=annualize)
    if method == VolMethod.GARCH:
        return compute_idio_vol_garch(ticker_returns, market_returns, annualize)
    # Default: CAPM
    return compute_idio_vol_capm(ticker_returns, market_returns, annualize)


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------
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
