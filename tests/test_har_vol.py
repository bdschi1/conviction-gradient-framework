"""Tests for HAR idiosyncratic volatility estimation.

Reference: Corsi 2009; Bekaert, Bergbrant & Kassa, JFE 2025.
"""

import numpy as np
import pytest

from bridges.data_bridge import (
    VolMethod,
    compute_idio_vol,
    compute_idio_vol_ewma,
    compute_idio_vol_har,
)


def _make_returns(n: int, seed: int = 42, vol: float = 0.01) -> tuple[list[float], list[float]]:
    """Generate synthetic ticker and market returns."""
    rng = np.random.default_rng(seed)
    market = (rng.standard_normal(n) * vol).tolist()
    # Ticker = beta * market + idio noise
    beta = 1.2
    idio_noise = rng.standard_normal(n) * vol * 0.5
    ticker = (np.array(market) * beta + idio_noise).tolist()
    return ticker, market


class TestHARVol:
    def test_positive_vol_estimate(self):
        """HAR should produce a positive vol estimate."""
        ticker, market = _make_returns(200)
        vol = compute_idio_vol_har(ticker, market)
        assert vol > 0

    def test_annualization(self):
        """Annualized vol should be ~sqrt(252) times daily vol."""
        ticker, market = _make_returns(200)
        vol_ann = compute_idio_vol_har(ticker, market, annualize=True)
        vol_daily = compute_idio_vol_har(ticker, market, annualize=False)
        ratio = vol_ann / vol_daily
        assert 14.0 < ratio < 17.0  # sqrt(252) ≈ 15.87

    def test_fallback_on_short_data(self):
        """With < 44 observations, should fall back to EWMA."""
        ticker, market = _make_returns(30)
        har_vol = compute_idio_vol_har(ticker, market)
        ewma_vol = compute_idio_vol_ewma(ticker, market)
        assert har_vol == pytest.approx(ewma_vol)

    def test_fallback_on_very_short_data(self):
        """With < 20 observations, should still produce a result (via EWMA fallback)."""
        ticker, market = _make_returns(10)
        vol = compute_idio_vol_har(ticker, market)
        assert vol > 0
        assert np.isfinite(vol)

    def test_dispatch_routing(self):
        """VolMethod.HAR should route to HAR implementation."""
        ticker, market = _make_returns(200)
        vol_dispatch = compute_idio_vol(VolMethod.HAR, ticker, market)
        vol_direct = compute_idio_vol_har(ticker, market)
        assert vol_dispatch == pytest.approx(vol_direct)

    def test_reasonable_range_on_synthetic(self):
        """HAR vol on synthetic data should be in a reasonable annualized range."""
        ticker, market = _make_returns(504, vol=0.01)
        vol = compute_idio_vol_har(ticker, market, annualize=True)
        # With daily vol ~0.5%, annualized should be roughly 5-15%
        assert 0.01 < vol < 0.50

    def test_different_from_capm(self):
        """HAR should generally differ from simple CAPM vol (different methodology)."""
        ticker, market = _make_returns(200)
        from bridges.data_bridge import compute_idio_vol_capm

        har_vol = compute_idio_vol_har(ticker, market)
        capm_vol = compute_idio_vol_capm(ticker, market)
        # They measure different things; very unlikely to be exactly equal
        assert har_vol != capm_vol

    def test_custom_horizons(self):
        """Custom window parameters should produce valid results."""
        ticker, market = _make_returns(200)
        vol = compute_idio_vol_har(
            ticker, market, window_d=1, window_w=10, window_m=44
        )
        assert vol > 0
        assert np.isfinite(vol)

    def test_vol_enum_has_har(self):
        """VolMethod enum should include HAR."""
        assert hasattr(VolMethod, "HAR")
        assert VolMethod.HAR.value == "har"

    def test_finite_with_trending_data(self):
        """HAR should handle trending (non-stationary) data gracefully."""
        rng = np.random.default_rng(42)
        n = 200
        trend = np.linspace(0, 0.001, n)
        noise = rng.standard_normal(n) * 0.01
        ticker = (trend + noise).tolist()
        market = (rng.standard_normal(n) * 0.01).tolist()
        vol = compute_idio_vol_har(ticker, market)
        assert np.isfinite(vol)
        assert vol > 0
