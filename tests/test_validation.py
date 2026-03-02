"""Tests for the validation harness.

Tests verify:
- Synthetic data generators produce correct shapes and are seed-reproducible
- Baselines run without error and produce finite results
- CGF pipeline runs over synthetic data with bounded convictions
- CGF is not catastrophically worse than equal-weight (mean Sharpe diff > -0.5)
- Ablation reports are populated
"""

import numpy as np

from validation.baselines import (
    buy_and_hold_strategy,
    compute_metrics,
    equal_weight_strategy,
    momentum_strategy,
)
from validation.report import (
    format_ablation_summary,
    format_comparison_table,
    format_multi_seed_summary,
)
from validation.runner import (
    run_cgf_over_series,
    run_validation,
    run_validation_multi_seed,
)
from validation.synthetic_data import (
    generate_gbm,
    generate_market_returns,
    generate_ou,
    generate_regime_switching,
    generate_universe,
)

# --- Synthetic Data ---


class TestSyntheticData:
    def test_gbm_shape(self):
        asset = generate_gbm(n_days=100, seed=1)
        assert len(asset.returns) == 100
        assert asset.asset_type == "gbm"

    def test_gbm_reproducible(self):
        a1 = generate_gbm(n_days=50, seed=42)
        a2 = generate_gbm(n_days=50, seed=42)
        np.testing.assert_array_equal(a1.returns, a2.returns)

    def test_gbm_different_seeds(self):
        a1 = generate_gbm(n_days=50, seed=1)
        a2 = generate_gbm(n_days=50, seed=2)
        assert not np.allclose(a1.returns, a2.returns)

    def test_ou_shape(self):
        asset = generate_ou(n_days=100, seed=1)
        assert len(asset.returns) == 100
        assert asset.asset_type == "ou"

    def test_regime_switching_shape(self):
        asset = generate_regime_switching(n_days=200, seed=1)
        assert len(asset.returns) == 200
        assert asset.asset_type == "regime_switching"

    def test_market_returns_shape(self):
        mkt = generate_market_returns(n_days=252, seed=1)
        assert len(mkt) == 252

    def test_universe_generates_3_assets(self):
        assets, market = generate_universe(n_days=100, seed=42)
        assert len(assets) == 3
        assert len(market) == 100
        types = {a.asset_type for a in assets}
        assert types == {"gbm", "ou", "regime_switching"}

    def test_returns_are_finite(self):
        assets, market = generate_universe(n_days=504, seed=42)
        for asset in assets:
            assert np.all(np.isfinite(asset.returns))
        assert np.all(np.isfinite(market))

    def test_returns_reasonable_magnitude(self):
        """Daily returns should be small (within ~10% for daily)."""
        asset = generate_gbm(n_days=504, seed=42, sigma=0.20)
        assert np.max(np.abs(asset.returns)) < 0.5  # No 50%+ daily moves


# --- Baselines ---


class TestBaselines:
    def _make_returns(self, n_days=252, n_assets=3, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_days, n_assets)) * 0.01

    def test_equal_weight_returns_shape(self):
        ret = self._make_returns()
        port_ret, turnover = equal_weight_strategy(ret)
        assert len(port_ret) == 252
        assert len(turnover) == 252

    def test_buy_and_hold_returns_shape(self):
        ret = self._make_returns()
        port_ret, turnover = buy_and_hold_strategy(ret)
        assert len(port_ret) == 252
        assert np.all(turnover == 0)

    def test_momentum_returns_shape(self):
        ret = self._make_returns()
        port_ret, turnover = momentum_strategy(ret, lookback=21)
        assert len(port_ret) == 252

    def test_metrics_finite(self):
        ret = self._make_returns()
        port_ret, _ = equal_weight_strategy(ret)
        m = compute_metrics(port_ret)
        assert np.isfinite(m.annualized_return)
        assert np.isfinite(m.annualized_vol)
        assert np.isfinite(m.sharpe_ratio)
        assert np.isfinite(m.max_drawdown)

    def test_empty_returns(self):
        m = compute_metrics(np.array([]))
        assert m.annualized_return == 0.0
        assert m.n_days == 0

    def test_empty_assets(self):
        ret = np.zeros((100, 0))
        port_ret, _ = equal_weight_strategy(ret)
        assert len(port_ret) == 100


# --- CGF Runner ---


class TestCGFRunner:
    def test_pipeline_runs(self):
        """Full pipeline runs without error on synthetic data."""
        assets, market = generate_universe(n_days=100, seed=42)
        port_ret, convictions = run_cgf_over_series(assets, market)
        assert len(port_ret) > 0
        assert all(np.all(np.isfinite(c)) for c in convictions.values())

    def test_convictions_bounded(self):
        """Convictions stay within [-C_max, C_max]."""
        assets, market = generate_universe(n_days=200, seed=42)
        _, convictions = run_cgf_over_series(assets, market)
        for name, c_arr in convictions.items():
            assert np.all(np.abs(c_arr) <= 5.0), f"{name} conviction exceeded C_max"

    def test_portfolio_returns_finite(self):
        assets, market = generate_universe(n_days=200, seed=42)
        port_ret, _ = run_cgf_over_series(assets, market)
        assert np.all(np.isfinite(port_ret))

    def test_returns_reasonable(self):
        """Portfolio daily returns should be reasonable (<50% daily)."""
        assets, market = generate_universe(n_days=504, seed=42)
        port_ret, _ = run_cgf_over_series(assets, market)
        assert np.max(np.abs(port_ret)) < 0.5


class TestValidation:
    def test_run_validation(self):
        """Full validation runs and produces results."""
        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        assert len(result.variants) > 0
        assert len(result.baselines) == 3

    def test_ablation_populated(self):
        """Ablation dict has entries for removed components."""
        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        assert "fe" in result.ablation
        assert "fvs" in result.ablation
        assert "rrs" in result.ablation
        assert "ads" in result.ablation

    def test_ablation_values_finite(self):
        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        for comp, val in result.ablation.items():
            assert np.isfinite(val), f"Ablation for {comp} is not finite"

    def test_not_catastrophically_worse(self):
        """CGF should not be catastrophically worse than equal-weight.

        Tests across 5 seeds that mean Sharpe diff > -0.5.
        This is a sanity check, not a performance guarantee.
        """
        results = run_validation_multi_seed(n_seeds=5, n_days=200, base_seed=42)

        full_sharpes = []
        ew_sharpes = []
        for r in results:
            for v in r.variants:
                if v.name == "full_model":
                    full_sharpes.append(v.metrics.sharpe_ratio)
            for b in r.baselines:
                if b.name == "equal_weight":
                    ew_sharpes.append(b.metrics.sharpe_ratio)

        diffs = [f - e for f, e in zip(full_sharpes, ew_sharpes, strict=True)]
        mean_diff = np.mean(diffs)
        assert mean_diff > -0.5, (
            f"CGF catastrophically worse than EW: mean Sharpe diff = {mean_diff:.4f}"
        )


# --- Feature Variants ---


class TestFeatureVariants:
    def test_adaptive_variant_runs(self):
        """full_model_adaptive variant runs without error."""
        from validation.runner import FEATURE_VARIANTS

        assets, market = generate_universe(n_days=150, seed=42)
        result = run_validation(assets, market, variants=["full_model"] + FEATURE_VARIANTS)
        assert len(result.variants) == 4
        for v in result.variants:
            assert np.all(np.isfinite(np.array([v.metrics.sharpe_ratio])))

    def test_regime_variant_runs(self):
        """full_model_continuous_regime variant runs without error."""
        assets, market = generate_universe(n_days=150, seed=42)
        result = run_validation(
            assets, market, variants=["full_model", "full_model_continuous_regime"]
        )
        assert len(result.variants) == 2

    def test_all_features_variant_runs(self):
        """full_model_all_features variant produces finite results."""
        assets, market = generate_universe(n_days=150, seed=42)
        result = run_validation(
            assets, market, variants=["full_model", "full_model_all_features"]
        )
        for v in result.variants:
            assert np.isfinite(v.metrics.annualized_return)
            assert np.isfinite(v.metrics.sharpe_ratio)


# --- Report Formatting ---


class TestReport:
    def test_comparison_table_format(self):
        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        table = format_comparison_table(result)
        assert "equal_weight" in table
        assert "full_model" in table
        assert "Sharpe" in table

    def test_ablation_summary_format(self):
        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        summary = format_ablation_summary(result)
        assert "Component" in summary
        assert "Marginal Sharpe" in summary

    def test_multi_seed_summary_format(self):
        results = run_validation_multi_seed(n_seeds=2, n_days=100, base_seed=42)
        summary = format_multi_seed_summary(results)
        assert "Multi-seed" in summary
        assert "CGF full model" in summary
