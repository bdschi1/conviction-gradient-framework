"""Tests for individual loss components."""

from datetime import date

import pytest

from components.forecast_error import compute_fe
from components.fundamental_violation import (
    FVSEvent,
    FVSTaxonomy,
    compute_fvs,
)
from components.risk_regime import compute_rrs
from components.thesis_shift import compute_its

# --- Forecast Error ---


class TestForecastError:
    def test_perfect_forecast(self):
        fe = compute_fe(realized_return=0.10, expected_return=0.10, sigma_expected=0.20)
        assert fe == pytest.approx(0.0)

    def test_positive_surprise(self):
        fe = compute_fe(realized_return=0.15, expected_return=0.10, sigma_expected=0.20)
        assert fe == pytest.approx(0.25)

    def test_negative_surprise(self):
        fe = compute_fe(realized_return=0.05, expected_return=0.10, sigma_expected=0.20)
        assert fe == pytest.approx(-0.25)

    def test_large_miss(self):
        fe = compute_fe(realized_return=-0.30, expected_return=0.10, sigma_expected=0.20)
        assert fe == pytest.approx(-2.0)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_fe(0.1, 0.1, 0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_fe(0.1, 0.1, -0.1)


# --- Fundamental Violation Score ---


class TestFVS:
    def test_no_events(self):
        assert compute_fvs([]) == 0.0

    def test_single_event(self):
        events = [FVSEvent(event_type="kpi_miss", event_date=date(2024, 6, 1))]
        fvs = compute_fvs(events)
        assert fvs == pytest.approx(0.4)  # base severity

    def test_governance_breach(self):
        events = [FVSEvent(event_type="governance_breach", event_date=date(2024, 6, 1))]
        fvs = compute_fvs(events)
        assert fvs == pytest.approx(0.8)

    def test_multiple_events_accumulate(self):
        events = [
            FVSEvent(event_type="kpi_miss", event_date=date(2024, 6, 1)),
            FVSEvent(event_type="guidance_cut", event_date=date(2024, 6, 2)),
        ]
        fvs = compute_fvs(events)
        # max(0.4, 0.5) + 0.1*(2-1) = 0.5 + 0.1 = 0.6
        assert fvs == pytest.approx(0.6)

    def test_severity_override(self):
        events = [
            FVSEvent(
                event_type="kpi_miss",
                event_date=date(2024, 6, 1),
                severity_override=0.9,
            )
        ]
        fvs = compute_fvs(events)
        assert fvs == pytest.approx(0.9)

    def test_capped_at_one(self):
        events = [
            FVSEvent(event_type="business_model_change", event_date=date(2024, 6, 1)),
            FVSEvent(event_type="governance_breach", event_date=date(2024, 6, 2)),
            FVSEvent(event_type="accounting_restatement", event_date=date(2024, 6, 3)),
            FVSEvent(event_type="mgmt_change", event_date=date(2024, 6, 4)),
        ]
        fvs = compute_fvs(events)
        assert fvs <= 1.0

    def test_unknown_event_type(self):
        events = [FVSEvent(event_type="unknown_type", event_date=date(2024, 6, 1))]
        fvs = compute_fvs(events)
        assert fvs == pytest.approx(0.5)  # default for unknown

    def test_taxonomy_from_dict(self):
        custom = {"custom_event": {"base_severity": 0.3, "description": "test"}}
        tax = FVSTaxonomy(event_types=custom)
        assert tax.get_severity("custom_event") == 0.3

    def test_default_taxonomy_types(self):
        tax = FVSTaxonomy()
        assert "mgmt_change" in tax.event_types
        assert "governance_breach" in tax.event_types
        assert len(tax.event_types) == 7


# --- Risk Regime Shift ---


class TestRRS:
    def test_no_change(self):
        rrs = compute_rrs(sigma_idio_current=0.20, sigma_idio_prev=0.20)
        assert rrs == pytest.approx(0.0)

    def test_vol_increase(self):
        rrs = compute_rrs(sigma_idio_current=0.30, sigma_idio_prev=0.20)
        assert rrs == pytest.approx(0.5)

    def test_vol_decrease(self):
        rrs = compute_rrs(sigma_idio_current=0.15, sigma_idio_prev=0.20)
        assert rrs == pytest.approx(-0.25)

    def test_with_iv_hv_spread(self):
        rrs = compute_rrs(
            sigma_idio_current=0.20,
            sigma_idio_prev=0.20,
            implied_vol=0.25,
            historical_vol=0.20,
        )
        # vol_change = 0, iv_hv = (0.25-0.20)/0.20 = 0.25
        assert rrs == pytest.approx(0.25)

    def test_both_components(self):
        rrs = compute_rrs(
            sigma_idio_current=0.30,
            sigma_idio_prev=0.20,
            implied_vol=0.35,
            historical_vol=0.25,
        )
        # vol_change = 0.5, iv_hv = (0.35-0.25)/0.25 = 0.4
        assert rrs == pytest.approx(0.9)

    def test_zero_prev_vol_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_rrs(0.20, 0.0)

    def test_no_iv_data(self):
        rrs = compute_rrs(0.25, 0.20, implied_vol=None, historical_vol=None)
        assert rrs == pytest.approx(0.25)


# --- Independent Thesis Shift ---


class TestITS:
    def test_fallback_no_change(self):
        its = compute_its(p_pre=[0.7, 0.7], p_post=[0.7, 0.7])
        assert its == pytest.approx(0.0)

    def test_fallback_positive_shift(self):
        its = compute_its(p_pre=[0.5, 0.6], p_post=[0.7, 0.8])
        assert its == pytest.approx(0.2)

    def test_fallback_negative_shift(self):
        its = compute_its(p_pre=[0.8, 0.9], p_post=[0.6, 0.7])
        assert its == pytest.approx(-0.2)

    def test_empty_lists(self):
        assert compute_its(p_pre=[], p_post=[]) == 0.0

    def test_empty_pre(self):
        assert compute_its(p_pre=[], p_post=[0.5]) == 0.0

    def test_fallback_different_lengths(self):
        its = compute_its(p_pre=[0.5], p_post=[0.7, 0.8])
        # mean([0.7, 0.8]) - mean([0.5]) = 0.75 - 0.5 = 0.25
        assert its == pytest.approx(0.25)

    def test_fallback_unanimous_shift(self):
        its = compute_its(p_pre=[0.5, 0.5, 0.5], p_post=[0.9, 0.9, 0.9])
        assert its == pytest.approx(0.4)

    def test_rich_mode_thesis_challenged(self):
        """PM more convicted than analysts → positive ITS (thesis challenged)."""
        its = compute_its(
            pm_conviction=0.8,
            analyst_convictions=[0.4, 0.5, 0.6],
        )
        assert its > 0

    def test_rich_mode_thesis_confirmed(self):
        """Analysts more convicted than PM → negative ITS (thesis confirmed)."""
        its = compute_its(
            pm_conviction=0.4,
            analyst_convictions=[0.7, 0.8, 0.9],
        )
        assert its < 0

    def test_rich_mode_convergence(self):
        """PM and analysts agree → ITS near zero."""
        its = compute_its(
            pm_conviction=0.6,
            analyst_convictions=[0.58, 0.60, 0.62],
        )
        assert abs(its) < 0.1

    def test_rich_mode_bounded(self):
        """ITS should be clipped to [-1, 1]."""
        its = compute_its(
            pm_conviction=1.0,
            analyst_convictions=[0.0],
        )
        assert -1.0 <= its <= 1.0

    def test_none_inputs_returns_zero(self):
        assert compute_its() == 0.0

    def test_position_type_passthrough(self):
        """Position type doesn't affect ITS value (informational only)."""
        its_alpha = compute_its(
            pm_conviction=0.8,
            analyst_convictions=[0.5, 0.6],
            position_type="alpha_long",
        )
        its_hedge = compute_its(
            pm_conviction=0.8,
            analyst_convictions=[0.5, 0.6],
            position_type="hedge_short",
        )
        assert its_alpha == pytest.approx(its_hedge)
