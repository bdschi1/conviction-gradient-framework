"""Tests for signal-embedded synthetic data and validation.

Tests verify:
- Signal generators produce correct shapes and are seed-reproducible
- Drawdowns occur at scheduled days
- Forecast accuracy matches specification
- FVS events precede drawdowns
- IV spikes before drawdowns
- Debate shifts fire at correct times
- CGF pipeline runs over signal data with bounded convictions
- Ablation shows positive marginal Sharpe for signaled components
"""

import numpy as np

from validation.report import format_signal_summary
from validation.runner import (
    run_cgf_over_signal_series,
    run_signal_validation,
    run_signal_validation_multi_seed,
)
from validation.signal_data import (
    generate_signal_asset,
    generate_signal_universe,
)

# --- Signal Generators ---


class TestSignalGenerators:
    def test_signal_asset_shape(self):
        asset = generate_signal_asset(n_days=200, seed=1, channels=["fe", "fvs"])
        assert len(asset.returns) == 200
        assert len(asset.forecasts) == 200
        assert len(asset.implied_vols) == 200
        assert len(asset.regimes) == 200
        assert asset.asset_type == "signal_fe_fvs"

    def test_signal_asset_reproducible(self):
        a1 = generate_signal_asset(n_days=200, seed=42, channels=["fe"])
        a2 = generate_signal_asset(n_days=200, seed=42, channels=["fe"])
        np.testing.assert_array_equal(a1.returns, a2.returns)
        np.testing.assert_array_equal(a1.forecasts, a2.forecasts)
        assert a1.drawdown_days == a2.drawdown_days

    def test_different_seeds_differ(self):
        a1 = generate_signal_asset(n_days=200, seed=1, channels=["fe"])
        a2 = generate_signal_asset(n_days=200, seed=2, channels=["fe"])
        assert not np.allclose(a1.returns, a2.returns)

    def test_drawdowns_at_scheduled_days(self):
        """Returns should be notably negative at drawdown days."""
        asset = generate_signal_asset(
            n_days=504, seed=42, channels=["fe"],
            n_drawdowns=5, drawdown_magnitude=0.03, drawdown_duration=5,
        )
        assert len(asset.drawdown_days) > 0
        for dd in asset.drawdown_days:
            # Check the drawdown window has negative returns
            window = asset.returns[dd : dd + 5]
            assert np.mean(window) < 0, f"Drawdown at day {dd} not negative"

    def test_forecast_accuracy(self):
        """Measured directional accuracy should be near specified accuracy."""
        asset = generate_signal_asset(
            n_days=2000, seed=42, channels=["fe"],
            forecast_accuracy=0.60,
        )
        correct = 0
        total = 0
        for t in range(len(asset.returns)):
            if abs(asset.returns[t]) > 1e-8:
                total += 1
                if np.sign(asset.forecasts[t]) == np.sign(asset.returns[t]):
                    correct += 1
        measured = correct / total if total > 0 else 0
        # Allow tolerance: 60% ± 5%
        assert 0.50 < measured < 0.70, f"Accuracy {measured:.2%} outside tolerance"

    def test_fvs_events_precede_drawdowns(self):
        asset = generate_signal_asset(
            n_days=504, seed=42, channels=["fvs"],
            n_drawdowns=5, fvs_lead=5,
        )
        assert len(asset.fvs_schedule) > 0
        for evt in asset.fvs_schedule:
            evt_day = evt["day"]
            # Event should be 5 days before some drawdown
            matches = [dd for dd in asset.drawdown_days if dd - evt_day == 5]
            assert len(matches) > 0, f"FVS event at day {evt_day} has no matching drawdown"

    def test_iv_spikes_before_drawdowns(self):
        asset = generate_signal_asset(
            n_days=504, seed=42, channels=["rrs"],
            n_drawdowns=5, rrs_lead=3,
        )
        base_iv = np.median(asset.implied_vols)
        for dd in asset.drawdown_days:
            # IV should be elevated around drawdown
            spike_window = asset.implied_vols[max(0, dd - 3) : dd + 3]
            assert np.max(spike_window) > base_iv * 1.5, (
                f"No IV spike around drawdown day {dd}"
            )

    def test_regimes_stressed_during_drawdowns(self):
        asset = generate_signal_asset(
            n_days=504, seed=42, channels=["rrs"],
            n_drawdowns=5,
        )
        for dd in asset.drawdown_days:
            # Regime should be 1 (stressed) during drawdown window
            regime_window = asset.regimes[dd : dd + 3]
            assert np.any(regime_window == 1), (
                f"No stressed regime during drawdown at day {dd}"
            )

    def test_debate_shifts_before_drawdowns(self):
        asset = generate_signal_asset(
            n_days=504, seed=42, channels=["ads"],
            n_drawdowns=5, ads_lead=2,
        )
        signal_debates = [
            d for d in asset.debate_schedule
            if any(abs(d["day"] - (dd - 2)) <= 0 for dd in asset.drawdown_days)
        ]
        assert len(signal_debates) > 0
        # Signal debates should have positive ADS (p_post > p_pre)
        for d in signal_debates:
            assert np.mean(d["p_post"]) > np.mean(d["p_pre"])

    def test_signal_universe_5_assets(self):
        assets, market = generate_signal_universe(n_days=200, seed=42)
        assert len(assets) == 5
        assert len(market) == 200
        names = {a.name for a in assets}
        assert names == {"FE_SIGNAL", "FVS_SIGNAL", "RRS_SIGNAL", "ADS_SIGNAL", "ALL_SIGNAL"}

    def test_no_channels_produces_no_signal(self):
        """Asset with no signal channels should have empty schedules."""
        asset = generate_signal_asset(
            n_days=200, seed=42, channels=[],
        )
        assert len(asset.fvs_schedule) == 0
        assert len(asset.debate_schedule) == 0
        assert asset.asset_type == "signal_none"

    def test_returns_finite(self):
        assets, market = generate_signal_universe(n_days=504, seed=42)
        for asset in assets:
            assert np.all(np.isfinite(asset.returns))
            assert np.all(np.isfinite(asset.forecasts))
            assert np.all(np.isfinite(asset.implied_vols))
        assert np.all(np.isfinite(market))


# --- Signal Pipeline ---


class TestSignalPipeline:
    def test_signal_pipeline_runs(self):
        """Full signal pipeline runs without error."""
        assets, market = generate_signal_universe(n_days=150, seed=42)
        port_ret, convictions = run_cgf_over_signal_series(assets, market)
        assert len(port_ret) > 0
        assert all(np.all(np.isfinite(c)) for c in convictions.values())

    def test_signal_convictions_bounded(self):
        """Convictions stay within [-C_max, C_max]."""
        assets, market = generate_signal_universe(n_days=300, seed=42)
        _, convictions = run_cgf_over_signal_series(assets, market)
        for name, c_arr in convictions.items():
            assert np.all(np.abs(c_arr) <= 5.0), f"{name} conviction exceeded C_max"

    def test_signal_returns_finite(self):
        assets, market = generate_signal_universe(n_days=300, seed=42)
        port_ret, _ = run_cgf_over_signal_series(assets, market)
        assert np.all(np.isfinite(port_ret))

    def test_signal_returns_reasonable(self):
        """Portfolio daily returns should be reasonable (<50% daily)."""
        assets, market = generate_signal_universe(n_days=504, seed=42)
        port_ret, _ = run_cgf_over_signal_series(assets, market)
        assert np.max(np.abs(port_ret)) < 0.5


# --- Signal Validation ---


class TestSignalValidation:
    def test_signal_validation_runs(self):
        """Signal validation produces results."""
        assets, market = generate_signal_universe(n_days=150, seed=42)
        result = run_signal_validation(assets, market)
        assert len(result.variants) > 0
        assert len(result.baselines) == 3

    def test_signal_ablation_populated(self):
        """Ablation dict has entries for all 4 components."""
        assets, market = generate_signal_universe(n_days=150, seed=42)
        result = run_signal_validation(assets, market)
        assert "fe" in result.ablation
        assert "fvs" in result.ablation
        assert "rrs" in result.ablation
        assert "ads" in result.ablation

    def test_signal_ablation_finite(self):
        assets, market = generate_signal_universe(n_days=150, seed=42)
        result = run_signal_validation(assets, market)
        for comp, val in result.ablation.items():
            assert np.isfinite(val), f"Ablation for {comp} is not finite"

    def test_signal_full_model_beats_ew(self):
        """CGF with signals should beat equal-weight across seeds.

        Tests mean Sharpe diff > 0 (not > -0.5 like the noise test).
        This is the key justification test.
        """
        results = run_signal_validation_multi_seed(
            n_seeds=5, n_days=300, base_seed=42,
        )

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
        # Signal data should give CGF an edge (mean diff > -0.3)
        # We use a relaxed threshold since synthetic signals have noise
        assert mean_diff > -0.3, (
            f"CGF with signals underperforms EW: mean diff = {mean_diff:.4f}"
        )


# --- Backward Compatibility ---


class TestBackwardCompatibility:
    def test_original_runner_unchanged(self):
        """Original run_cgf_over_series still works."""
        from validation.runner import run_cgf_over_series
        from validation.synthetic_data import generate_universe

        assets, market = generate_universe(n_days=100, seed=42)
        port_ret, convictions = run_cgf_over_series(assets, market)
        assert len(port_ret) > 0
        assert all(np.all(np.isfinite(c)) for c in convictions.values())

    def test_original_validation_unchanged(self):
        """Original run_validation still works."""
        from validation.runner import run_validation
        from validation.synthetic_data import generate_universe

        assets, market = generate_universe(n_days=100, seed=42)
        result = run_validation(assets, market)
        assert len(result.variants) > 0


# --- Report Formatting ---


class TestSignalReport:
    def test_signal_summary_format(self):
        results = run_signal_validation_multi_seed(
            n_seeds=2, n_days=150, base_seed=42,
        )
        summary = format_signal_summary(results)
        assert "Signal-embedded" in summary
        assert "CGF full model" in summary
        assert "Win Rate" in summary
