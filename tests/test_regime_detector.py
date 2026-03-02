"""Tests for continuous regime detection.

Reference: Aydinhan, Kolm, Mulvey & Shu, AOR 2024; Hamilton 1989.
"""

from datetime import date

import numpy as np
import pytest

from components.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeState
from components.risk_regime import compute_rrs
from config.defaults import ConvictionParams
from engine.models import ConvictionState, InstrumentData
from engine.updater import run_single_update


class TestRegimeState:
    def test_initial_state(self):
        """Default state should be uninformative prior."""
        s = RegimeState()
        assert s.p_high_vol == pytest.approx(0.5)
        assert s.smoothed_vol == pytest.approx(0.0)
        assert s.n_observations == 0


class TestRegimeDetectorConfig:
    def test_defaults(self):
        c = RegimeDetectorConfig()
        assert c.vol_threshold == pytest.approx(1.5)
        assert c.transition_penalty == pytest.approx(0.1)
        assert c.ema_alpha == pytest.approx(0.05)
        assert c.min_observations == 22


class TestRegimeDetector:
    def test_first_observation_sets_baseline(self):
        """First observation should set the smoothed vol baseline."""
        d = RegimeDetector()
        d.update(0.20)
        assert d.state.smoothed_vol == pytest.approx(0.20)
        assert d.state.n_observations == 1

    def test_warmup_stays_at_prior(self):
        """During warmup (< min_observations), p_high_vol should stay near 0.5."""
        d = RegimeDetector(RegimeDetectorConfig(min_observations=10))
        for _ in range(5):
            d.update(0.20)
        assert d.state.p_high_vol == pytest.approx(0.5)

    def test_low_vol_detection(self):
        """Stable low vol should drive p_high_vol toward 0."""
        d = RegimeDetector(RegimeDetectorConfig(min_observations=5))
        # Feed stable, low volatility observations
        for _ in range(50):
            d.update(0.15)
        assert d.state.p_high_vol < 0.3

    def test_high_vol_detection(self):
        """Vol spike with slow EMA should drive p_high_vol above 0.5."""
        config = RegimeDetectorConfig(
            min_observations=5, vol_threshold=1.3, ema_alpha=0.01
        )
        d = RegimeDetector(config)
        # Establish baseline at 0.15
        for _ in range(30):
            d.update(0.15)
        # Spike to 3x baseline — slow EMA keeps baseline near 0.15
        for _ in range(20):
            d.update(0.45)
        assert d.state.p_high_vol > 0.5

    def test_transition_penalty_prevents_whipsaw(self):
        """With vs without penalty: penalty provides recovery floor from committed states."""
        config_with = RegimeDetectorConfig(
            min_observations=5, transition_penalty=0.2, vol_threshold=1.5
        )
        config_without = RegimeDetectorConfig(
            min_observations=5, transition_penalty=0.0, vol_threshold=1.5
        )
        d_with = RegimeDetector(config_with)
        d_without = RegimeDetector(config_without)

        # Both establish low-vol baseline
        for _ in range(50):
            d_with.update(0.15)
            d_without.update(0.15)

        # Both should be in low-vol regime
        assert d_with.state.p_high_vol < 0.3
        # Zero penalty converges harder to 0
        assert d_without.state.p_high_vol < d_with.state.p_high_vol

    def test_transition_penalty_as_floor(self):
        """Nonzero penalty provides a floor prior that allows regime recovery.

        With penalty > 0, even after converging to low-vol, a vol spike
        can still push p_high_vol up because the penalty mixes in a
        transition probability floor.
        """
        config = RegimeDetectorConfig(
            min_observations=5, transition_penalty=0.15, vol_threshold=1.3
        )
        d = RegimeDetector(config)
        # Converge to low-vol
        for _ in range(30):
            d.update(0.15)
        low_p = d.state.p_high_vol

        # Now spike vol — with penalty, should still be able to detect
        for _ in range(20):
            d.update(0.40)
        assert d.state.p_high_vol > low_p

    def test_compute_rrs_scaling_low_vol(self):
        """In low-vol regime, RRS should be close to base RRS."""
        d = RegimeDetector(RegimeDetectorConfig(min_observations=5))
        for _ in range(50):
            d.update(0.15)
        # p_high_vol should be low
        assert d.state.p_high_vol < 0.3

        rrs_scaled = d.compute_rrs(0.16, 0.15)
        rrs_base = compute_rrs(0.16, 0.15)
        # Scale factor = 1 + p_high_vol ≈ 1.0-1.3
        assert rrs_scaled >= rrs_base
        assert rrs_scaled < rrs_base * 1.5

    def test_compute_rrs_scaling_high_vol(self):
        """In elevated-vol regime, RRS should be amplified above base."""
        config = RegimeDetectorConfig(
            min_observations=5, vol_threshold=1.3, ema_alpha=0.01
        )
        d = RegimeDetector(config)
        for _ in range(30):
            d.update(0.15)
        for _ in range(20):
            d.update(0.45)
        # p_high_vol should be above 0.5
        assert d.state.p_high_vol > 0.5

        rrs_scaled = d.compute_rrs(0.46, 0.45)
        rrs_base = compute_rrs(0.46, 0.45)
        # Scale factor = 1 + p_high_vol > 1.3
        assert rrs_scaled > rrs_base * 1.3

    def test_compute_rrs_with_iv_hv(self):
        """Regime-scaled RRS should include IV-HV spread."""
        d = RegimeDetector(RegimeDetectorConfig(min_observations=5))
        for _ in range(10):
            d.update(0.15)
        rrs = d.compute_rrs(0.16, 0.15, implied_vol=0.25, historical_vol=0.20)
        assert rrs != 0.0

    def test_reset(self):
        """Reset should return to uninformative prior."""
        d = RegimeDetector()
        for _ in range(30):
            d.update(0.20)
        d.reset()
        assert d.state.p_high_vol == pytest.approx(0.5)
        assert d.state.smoothed_vol == pytest.approx(0.0)
        assert d.state.n_observations == 0

    def test_ema_smoothing(self):
        """EMA should smooth vol observations."""
        d = RegimeDetector(RegimeDetectorConfig(ema_alpha=0.1))
        d.update(0.20)
        d.update(0.30)
        # EMA: 0.1 * 0.30 + 0.9 * 0.20 = 0.21
        assert d.state.smoothed_vol == pytest.approx(0.21)


class TestRegimeUpdaterIntegration:
    def test_backward_compatible_without_regime(self):
        """Without regime detector, updater should work exactly as before."""
        current = ConvictionState(
            instrument_id="TEST",
            as_of_date=date(2024, 5, 31),
            conviction=2.0,
            conviction_prev=1.8,
        )
        data = InstrumentData(
            instrument_id="TEST",
            as_of_date=date(2024, 6, 1),
            realized_return=0.05,
            expected_return=0.08,
            sigma_expected=0.20,
            sigma_idio_current=0.18,
            sigma_idio_prev=0.15,
        )
        # Without regime detector
        result_no_regime = run_single_update(current, data)
        # With regime detector but continuous_regime=False (default)
        d = RegimeDetector()
        result_regime_off = run_single_update(current, data, regime_detector=d)
        assert result_no_regime.conviction == pytest.approx(result_regime_off.conviction)

    def test_regime_detector_active_in_updater(self):
        """With continuous_regime=True and detector, RRS should be scaled."""
        params = ConvictionParams(continuous_regime=True)
        current = ConvictionState(
            instrument_id="TEST",
            as_of_date=date(2024, 5, 31),
            conviction=2.0,
            conviction_prev=1.8,
        )
        data = InstrumentData(
            instrument_id="TEST",
            as_of_date=date(2024, 6, 1),
            realized_return=0.05,
            expected_return=0.08,
            sigma_expected=0.20,
            sigma_idio_current=0.18,
            sigma_idio_prev=0.15,
        )
        d = RegimeDetector(RegimeDetectorConfig(min_observations=1))
        # Feed some vol observations to warm up
        for _ in range(5):
            d.update(0.18)

        result = run_single_update(current, data, params=params, regime_detector=d)
        assert result.conviction != 0.0
        assert np.isfinite(result.conviction)

    def test_config_fields_exist(self):
        """ConvictionParams should have regime config fields."""
        p = ConvictionParams()
        assert p.continuous_regime is False
        assert p.regime_vol_threshold == pytest.approx(1.5)
        assert p.regime_transition_penalty == pytest.approx(0.1)
