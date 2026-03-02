"""Tests for real data validation pipeline.

Separated into:
- Transformation tests (no network, pure math) — always run
- Network tests (require yfinance download) — skipped if offline
"""

import numpy as np
import pytest

from validation.real_data import (
    RealDataConfig,
    SignalEmbeddedAsset,
    _align_vix_to_returns,
    _compute_trailing_forecasts,
    transform_returns_to_asset,
)

# ---------------------------------------------------------------------------
# Transformation tests (no network)
# ---------------------------------------------------------------------------


class TestTrailingForecasts:
    def test_shape(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 252)
        forecasts = _compute_trailing_forecasts(returns, window=21)
        assert forecasts.shape == returns.shape

    def test_first_value_zero(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        forecasts = _compute_trailing_forecasts(returns, window=21)
        assert forecasts[0] == 0.0

    def test_trailing_mean_correct(self):
        returns = np.ones(50) * 0.01
        forecasts = _compute_trailing_forecasts(returns, window=5)
        # After warmup, forecast should be trailing mean = 0.01
        assert forecasts[10] == pytest.approx(0.01, abs=1e-10)

    def test_window_size_matters(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        f_short = _compute_trailing_forecasts(returns, window=5)
        f_long = _compute_trailing_forecasts(returns, window=63)
        # Short window should be more volatile
        assert np.std(f_short[63:]) > np.std(f_long[63:])


class TestAlignVix:
    def test_empty_vix(self):
        result = _align_vix_to_returns(np.array([]), 100)
        assert len(result) == 100
        assert result[0] == pytest.approx(0.20 / np.sqrt(252))

    def test_longer_vix(self):
        vix = np.full(200, 0.25)
        result = _align_vix_to_returns(vix, 100)
        assert len(result) == 100

    def test_shorter_vix(self):
        vix = np.full(50, 0.25)
        result = _align_vix_to_returns(vix, 100)
        assert len(result) == 100
        # First 50 values from vix, rest padded with last
        assert result[49] == pytest.approx(0.25 / np.sqrt(252))
        assert result[99] == pytest.approx(0.25 / np.sqrt(252))


class TestTransformReturns:
    def test_basic_transform(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        asset = transform_returns_to_asset(returns, "TEST")

        assert isinstance(asset, SignalEmbeddedAsset)
        assert asset.name == "TEST"
        assert asset.asset_type == "transformed"
        assert len(asset.returns) == 252
        assert len(asset.forecasts) == 252
        assert len(asset.implied_vols) == 252
        assert asset.debate_schedule == []
        assert "fe" in asset.signal_channels
        assert "rrs" in asset.signal_channels

    def test_with_fvs_events(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 252)
        events = [{"day": 50, "event_type": "kpi_miss", "severity": 0.5}]
        asset = transform_returns_to_asset(returns, "TEST", fvs_events=events)
        assert "fvs" in asset.signal_channels
        assert len(asset.fvs_schedule) == 1

    def test_with_custom_iv(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 100)
        iv = np.full(100, 0.02)
        asset = transform_returns_to_asset(returns, "TEST", implied_vols=iv)
        np.testing.assert_array_equal(asset.implied_vols, iv)

    def test_annualized_vol_computed(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        asset = transform_returns_to_asset(returns, "TEST")
        expected_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
        assert asset.annualized_vol == pytest.approx(expected_vol)

    def test_empty_returns(self):
        asset = transform_returns_to_asset(np.array([]), "EMPTY")
        assert len(asset.returns) == 0
        assert asset.annualized_vol == 0.20  # fallback


class TestRealDataConfig:
    def test_defaults(self):
        cfg = RealDataConfig()
        assert cfg.tickers == ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
        assert cfg.market_ticker == "SPY"
        assert cfg.start_date == "2022-01-01"
        assert cfg.end_date == "2023-12-31"

    def test_custom_tickers(self):
        cfg = RealDataConfig(tickers=["GOOG", "META"])
        assert cfg.tickers == ["GOOG", "META"]

    def test_custom_dates(self):
        cfg = RealDataConfig(start_date="2023-06-01", end_date="2024-06-01")
        assert cfg.start_date == "2023-06-01"


# ---------------------------------------------------------------------------
# Pipeline integration tests (use synthetic data to simulate real pipeline)
# ---------------------------------------------------------------------------


class TestRealDataPipeline:
    """Test the real data pipeline using transform_returns_to_asset (no network)."""

    def test_signal_runner_accepts_real_asset(self):
        """Real data assets work with run_cgf_over_signal_series."""
        from validation.runner import run_cgf_over_signal_series

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        market = rng.normal(0, 0.008, 252)

        asset = transform_returns_to_asset(returns, "TEST_STOCK")
        port_ret, convictions = run_cgf_over_signal_series(
            [asset], market, warmup=22,
        )
        assert len(port_ret) == 252 - 22
        assert "TEST_STOCK" in convictions
        assert np.all(np.isfinite(port_ret))

    def test_multi_asset_pipeline(self):
        """Multiple real data assets run through signal pipeline."""
        from validation.runner import run_cgf_over_signal_series

        rng = np.random.default_rng(42)
        assets = []
        for name in ["AAPL", "MSFT", "JPM"]:
            returns = rng.normal(0, 0.01, 252)
            assets.append(transform_returns_to_asset(returns, name))
        market = rng.normal(0, 0.008, 252)

        port_ret, convictions = run_cgf_over_signal_series(assets, market)
        assert len(convictions) == 3
        assert np.all(np.isfinite(port_ret))

    def test_with_fvs_events_pipeline(self):
        """FVS events from earnings surprises flow through pipeline."""
        from config.defaults import ConvictionParams
        from validation.runner import run_cgf_over_signal_series

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        events = [
            {"day": 60, "event_type": "kpi_miss", "severity": 0.6},
            {"day": 120, "event_type": "guidance_cut", "severity": 0.4},
        ]
        asset = transform_returns_to_asset(returns, "FVS_TEST", fvs_events=events)
        market = rng.normal(0, 0.008, 252)

        port_ret, convictions = run_cgf_over_signal_series(
            [asset], market, params=ConvictionParams(kappa=0.3),
        )
        assert np.all(np.isfinite(port_ret))
        assert np.all(np.isfinite(convictions["FVS_TEST"]))

    def test_convictions_bounded(self):
        """Convictions stay within [-C_max, C_max]."""
        from config.defaults import ConvictionParams
        from validation.runner import run_cgf_over_signal_series

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 504)  # Higher vol
        market = rng.normal(0, 0.015, 504)
        asset = transform_returns_to_asset(returns, "BOUNDED_TEST")

        params = ConvictionParams(kappa=0.5)
        _, convictions = run_cgf_over_signal_series(
            [asset], market, params=params,
        )
        c = convictions["BOUNDED_TEST"]
        assert np.all(c >= -params.C_max)
        assert np.all(c <= params.C_max)

    def test_ablation_runs(self):
        """Ablation over real-format data produces finite results."""
        from validation.runner import (
            SIGNAL_VARIANTS,
            _make_params_variant,
            run_cgf_over_signal_series,
        )

        rng = np.random.default_rng(42)
        assets = [
            transform_returns_to_asset(rng.normal(0, 0.01, 252), name)
            for name in ["A", "B", "C"]
        ]
        market = rng.normal(0, 0.008, 252)

        for variant in SIGNAL_VARIANTS:
            params = _make_params_variant(variant)
            port_ret, _ = run_cgf_over_signal_series(assets, market, params=params)
            assert np.all(np.isfinite(port_ret)), f"Non-finite returns in {variant}"

    def test_report_format(self):
        """Real data report formatter works."""
        from validation.baselines import PortfolioMetrics
        from validation.report import format_real_data_summary
        from validation.runner import ValidationResult, VariantResult

        result = ValidationResult(
            variants=[
                VariantResult(
                    name="full_model",
                    metrics=PortfolioMetrics(
                        annualized_return=0.08,
                        annualized_vol=0.15,
                        sharpe_ratio=0.53,
                        max_drawdown=-0.10,
                        turnover=0.5,
                        n_days=252,
                    ),
                ),
                VariantResult(
                    name="no_fe",
                    metrics=PortfolioMetrics(
                        annualized_return=0.07,
                        annualized_vol=0.14,
                        sharpe_ratio=0.50,
                        max_drawdown=-0.09,
                        turnover=0.4,
                        n_days=252,
                    ),
                ),
            ],
            baselines=[
                VariantResult(
                    name="equal_weight",
                    metrics=PortfolioMetrics(
                        annualized_return=0.06,
                        annualized_vol=0.14,
                        sharpe_ratio=0.43,
                        max_drawdown=-0.12,
                        turnover=0.1,
                        n_days=252,
                    ),
                ),
            ],
            ablation={"fe": 0.03},
        )
        output = format_real_data_summary(result, tickers=["AAPL", "MSFT"])
        assert "Real Data Validation" in output
        assert "AAPL" in output
        assert "MSFT" in output
        assert "CGF full model" in output

    def test_backward_compatible(self):
        """Original synthetic validation still works."""
        from validation.runner import run_validation
        from validation.synthetic_data import generate_universe

        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        assert len(result.variants) > 0
        assert len(result.baselines) > 0


# ---------------------------------------------------------------------------
# Network tests (require yfinance + internet)
# ---------------------------------------------------------------------------


def _yfinance_available() -> bool:
    """Check if yfinance can fetch data."""
    try:
        import yfinance as yf

        data = yf.download("AAPL", period="5d", progress=False)
        return not data.empty
    except Exception:
        return False


@pytest.mark.skipif(
    not _yfinance_available(),
    reason="yfinance download not available (offline or API issue)",
)
class TestRealDataNetwork:
    """Tests that require network access to yfinance."""

    def test_fetch_prices(self):
        from validation.real_data import _fetch_prices

        returns = _fetch_prices("AAPL", "2023-01-01", "2023-06-30")
        assert len(returns) > 100
        assert np.all(np.isfinite(returns))

    def test_fetch_vix(self):
        from validation.real_data import _fetch_vix

        vix = _fetch_vix("2023-01-01", "2023-06-30")
        assert len(vix) > 100
        assert np.all(vix > 0)
        assert np.all(vix < 1.0)  # Decimal form (not percentage)

    def test_fetch_real_asset(self):
        from validation.real_data import fetch_real_asset

        asset = fetch_real_asset("AAPL", "2023-01-01", "2023-12-31")
        assert isinstance(asset, SignalEmbeddedAsset)
        assert asset.name == "AAPL"
        assert asset.asset_type == "real"
        assert len(asset.returns) > 200
        assert len(asset.forecasts) == len(asset.returns)
        assert len(asset.implied_vols) == len(asset.returns)
        assert np.all(np.isfinite(asset.returns))

    def test_fetch_real_universe(self):
        from validation.real_data import fetch_real_universe

        config = RealDataConfig(
            tickers=["AAPL", "MSFT"],
            start_date="2023-06-01",
            end_date="2023-12-31",
        )
        assets, market = fetch_real_universe(config)
        assert len(assets) == 2
        assert len(market) > 0
        # All arrays same length
        lengths = {len(a.returns) for a in assets}
        lengths.add(len(market))
        assert len(lengths) == 1, f"Mismatched lengths: {lengths}"

    def test_full_real_validation(self):
        """End-to-end: fetch real data → run CGF → get results."""
        from validation.runner import run_real_data_validation

        config = RealDataConfig(
            tickers=["AAPL", "MSFT"],
            start_date="2023-06-01",
            end_date="2023-12-31",
        )
        result = run_real_data_validation(config)
        assert len(result.variants) > 0
        assert len(result.baselines) > 0
        # All metrics finite
        for v in result.variants:
            assert np.isfinite(v.metrics.sharpe_ratio)
        for b in result.baselines:
            assert np.isfinite(b.metrics.sharpe_ratio)

    def test_caching_works(self, tmp_path):
        """Data is cached and reused on second call."""
        from validation.real_data import fetch_real_universe

        config = RealDataConfig(
            tickers=["AAPL"],
            start_date="2023-06-01",
            end_date="2023-09-30",
            cache_dir=tmp_path / "cache",
        )
        # First call: downloads
        assets1, market1 = fetch_real_universe(config)
        # Second call: uses cache
        assets2, market2 = fetch_real_universe(config)

        np.testing.assert_array_equal(assets1[0].returns, assets2[0].returns)
        np.testing.assert_array_equal(market1, market2)
