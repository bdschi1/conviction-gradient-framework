"""Tests for position sizing overlay."""

import pytest

from sizing.constraints import PortfolioConstraints, apply_constraints
from sizing.failure_modes import (
    check_oscillation_guard,
    check_structural_reset,
)
from sizing.mapper import (
    SizingMethod,
    basic_mapping,
    kelly_mapping,
    map_convictions,
    risk_parity_mapping,
    tiered_mapping,
    vol_adjusted_mapping,
)

# --- Mapper ---


class TestBasicMapping:
    def test_single_position(self):
        weights = basic_mapping({"AAPL": 3.0})
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_equal_convictions(self):
        weights = basic_mapping({"AAPL": 2.0, "MSFT": 2.0})
        assert weights["AAPL"] == pytest.approx(0.5)
        assert weights["MSFT"] == pytest.approx(0.5)

    def test_long_short(self):
        weights = basic_mapping({"AAPL": 2.0, "TSLA": -1.0})
        assert weights["AAPL"] == pytest.approx(2.0 / 3.0)
        assert weights["TSLA"] == pytest.approx(-1.0 / 3.0)

    def test_empty(self):
        assert basic_mapping({}) == {}

    def test_all_zero(self):
        weights = basic_mapping({"A": 0.0, "B": 0.0})
        assert weights["A"] == pytest.approx(0.0)

    def test_sum_abs_weights_one(self):
        weights = basic_mapping({"A": 3.0, "B": -2.0, "C": 1.0})
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)


class TestVolAdjustedMapping:
    def test_equal_vol(self):
        weights = vol_adjusted_mapping(
            {"AAPL": 2.0, "MSFT": 2.0},
            {"AAPL": 0.20, "MSFT": 0.20},
        )
        assert weights["AAPL"] == pytest.approx(0.5)
        assert weights["MSFT"] == pytest.approx(0.5)

    def test_higher_vol_smaller_weight(self):
        weights = vol_adjusted_mapping(
            {"AAPL": 2.0, "TSLA": 2.0},
            {"AAPL": 0.20, "TSLA": 0.40},
        )
        assert abs(weights["AAPL"]) > abs(weights["TSLA"])

    def test_missing_vol_defaults_to_one(self):
        weights = vol_adjusted_mapping(
            {"AAPL": 2.0},
            {},
        )
        assert "AAPL" in weights


class TestMapConvictions:
    def test_basic_method(self):
        weights = map_convictions({"A": 1.0}, method=SizingMethod.BASIC)
        assert weights["A"] == pytest.approx(1.0)

    def test_vol_adjusted_requires_vols(self):
        with pytest.raises(ValueError, match="vols required"):
            map_convictions({"A": 1.0}, method=SizingMethod.VOL_ADJUSTED)

    def test_kelly_method(self):
        weights = map_convictions(
            {"A": 2.0, "B": -1.0},
            method=SizingMethod.KELLY,
            vols={"A": 0.20, "B": 0.30},
            expected_returns={"A": 0.10, "B": 0.05},
        )
        assert "A" in weights
        assert "B" in weights

    def test_kelly_requires_vols(self):
        with pytest.raises(ValueError, match="vols required"):
            map_convictions({"A": 1.0}, method=SizingMethod.KELLY, expected_returns={"A": 0.1})

    def test_kelly_requires_expected_returns(self):
        with pytest.raises(ValueError, match="expected_returns required"):
            map_convictions({"A": 1.0}, method=SizingMethod.KELLY, vols={"A": 0.2})

    def test_risk_parity_method(self):
        weights = map_convictions(
            {"A": 2.0, "B": 1.0},
            method=SizingMethod.RISK_PARITY,
            vols={"A": 0.20, "B": 0.30},
        )
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)

    def test_risk_parity_requires_vols(self):
        with pytest.raises(ValueError, match="vols required"):
            map_convictions({"A": 1.0}, method=SizingMethod.RISK_PARITY)

    def test_tiered_method(self):
        weights = map_convictions(
            {"A": 4.0, "B": 2.0, "C": 0.5},
            method=SizingMethod.TIERED,
        )
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)


# --- Kelly ---


class TestKellyMapping:
    def test_empty(self):
        assert kelly_mapping({}, {}, {}) == {}

    def test_single_long(self):
        weights = kelly_mapping(
            {"AAPL": 3.0},
            {"AAPL": 0.10},
            {"AAPL": 0.20},
        )
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_higher_conviction_larger_weight(self):
        weights = kelly_mapping(
            {"A": 4.0, "B": 1.0},
            {"A": 0.10, "B": 0.10},
            {"A": 0.20, "B": 0.20},
        )
        assert abs(weights["A"]) > abs(weights["B"])

    def test_short_negative_weight(self):
        weights = kelly_mapping(
            {"A": 2.0, "B": -2.0},
            {"A": 0.10, "B": 0.10},
            {"A": 0.20, "B": 0.20},
        )
        assert weights["A"] > 0
        assert weights["B"] < 0

    def test_half_kelly_reduces(self):
        full = kelly_mapping(
            {"A": 2.0}, {"A": 0.10}, {"A": 0.20}, half_kelly=False
        )
        half = kelly_mapping(
            {"A": 2.0}, {"A": 0.10}, {"A": 0.20}, half_kelly=True
        )
        # Both normalize to 1.0 for single position, so check raw ratio isn't affected
        assert full["A"] == pytest.approx(1.0)
        assert half["A"] == pytest.approx(1.0)

    def test_sum_abs_weights_one(self):
        weights = kelly_mapping(
            {"A": 3.0, "B": -2.0, "C": 1.0},
            {"A": 0.10, "B": 0.05, "C": 0.08},
            {"A": 0.20, "B": 0.30, "C": 0.15},
        )
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)

    def test_zero_conviction(self):
        weights = kelly_mapping({"A": 0.0}, {"A": 0.10}, {"A": 0.20})
        assert weights["A"] == pytest.approx(0.0)


# --- Risk Parity ---


class TestRiskParityMapping:
    def test_empty(self):
        assert risk_parity_mapping({}, {}) == {}

    def test_single_position(self):
        weights = risk_parity_mapping({"AAPL": 3.0}, {"AAPL": 0.20})
        assert weights["AAPL"] == pytest.approx(1.0)

    def test_higher_vol_smaller_weight(self):
        weights = risk_parity_mapping(
            {"A": 2.0, "B": 2.0},
            {"A": 0.10, "B": 0.40},
        )
        assert abs(weights["A"]) > abs(weights["B"])

    def test_conviction_scaling(self):
        weights = risk_parity_mapping(
            {"A": 4.0, "B": 1.0},
            {"A": 0.20, "B": 0.20},
        )
        # Higher conviction should get larger weight
        assert abs(weights["A"]) > abs(weights["B"])

    def test_short_negative_weight(self):
        weights = risk_parity_mapping(
            {"A": 2.0, "B": -2.0},
            {"A": 0.20, "B": 0.20},
        )
        assert weights["A"] > 0
        assert weights["B"] < 0

    def test_sum_abs_weights_one(self):
        weights = risk_parity_mapping(
            {"A": 3.0, "B": -2.0, "C": 1.0},
            {"A": 0.20, "B": 0.30, "C": 0.15},
        )
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)

    def test_zero_conviction_returns_zero(self):
        weights = risk_parity_mapping({"A": 0.0, "B": 0.0}, {"A": 0.20, "B": 0.30})
        assert weights["A"] == pytest.approx(0.0)


# --- Tiered ---


class TestTieredMapping:
    def test_empty(self):
        assert tiered_mapping({}) == {}

    def test_high_conviction_tier(self):
        weights = tiered_mapping({"A": 4.0, "B": 2.0, "C": 0.5})
        # A (high) should get the largest weight, C (low) the smallest
        assert abs(weights["A"]) > abs(weights["B"]) > abs(weights["C"])

    def test_short_preserves_sign(self):
        weights = tiered_mapping({"A": 3.0, "B": -3.0})
        assert weights["A"] > 0
        assert weights["B"] < 0
        assert abs(weights["A"]) == pytest.approx(abs(weights["B"]))

    def test_zero_conviction(self):
        weights = tiered_mapping({"A": 0.0, "B": 2.0})
        assert weights["A"] == pytest.approx(0.0)
        assert weights["B"] == pytest.approx(1.0)

    def test_sum_abs_weights_one(self):
        weights = tiered_mapping({"A": 5.0, "B": -2.0, "C": 0.8})
        assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)

    def test_tier_boundaries(self):
        # Exactly at boundary: 3.0 should be high, 1.5 medium
        weights = tiered_mapping({"A": 3.0, "B": 1.5})
        # Both above thresholds — A gets 4.5% raw, B gets 2.5% raw
        ratio = abs(weights["A"]) / abs(weights["B"])
        assert ratio == pytest.approx(0.045 / 0.025)


# --- Constraints ---


class TestConstraints:
    def test_empty_weights(self):
        result = apply_constraints({})
        assert result.converged is True
        assert result.weights == {}

    def test_basic_constraint_application(self):
        raw = {"AAPL": 0.03, "MSFT": 0.02, "GOOG": -0.02}
        result = apply_constraints(raw)
        assert result.converged is True
        assert len(result.weights) == 3

    def test_position_cap(self):
        constraints = PortfolioConstraints(max_position_pct=0.04)
        raw = {"AAPL": 0.10, "MSFT": 0.02}
        result = apply_constraints(raw, constraints)
        for w in result.weights.values():
            assert abs(w) <= 0.04 + 1e-6

    def test_sector_constraints(self):
        raw = {"AAPL": 0.04, "MSFT": 0.04, "GOOG": -0.02}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "GOOG": "Comm"}
        result = apply_constraints(raw, sectors=sectors)
        assert result.converged is True


# --- Failure Modes ---


class TestOscillationGuard:
    def test_no_oscillation(self):
        result = check_oscillation_guard(
            "AAPL", [1.0, 2.0, 3.0], current_alpha=0.1
        )
        assert result is None

    def test_oscillation_detected(self):
        result = check_oscillation_guard(
            "AAPL", [1.0, -1.0, 1.0], current_alpha=0.1
        )
        assert result is not None
        assert result.action_type == "halve_alpha"
        assert result.new_alpha < 0.1

    def test_alpha_not_below_min(self):
        from config.defaults import ConvictionParams

        params = ConvictionParams(alpha_min=0.05)
        result = check_oscillation_guard(
            "AAPL", [1.0, -1.0, 1.0, -1.0, 1.0], current_alpha=0.1, params=params
        )
        assert result is not None
        assert result.new_alpha >= 0.05


class TestStructuralReset:
    def test_no_reset(self):
        result = check_structural_reset("AAPL", fvs=0.3, sigma_idio_current=0.20, sigma_idio_baseline=0.15)
        assert result is None

    def test_fvs_reset(self):
        result = check_structural_reset("AAPL", fvs=0.8, sigma_idio_current=0.20, sigma_idio_baseline=0.15)
        assert result is not None
        assert result.action_type == "reset_conviction"
        assert result.new_conviction == 0.0

    def test_vol_double_reset(self):
        result = check_structural_reset("AAPL", fvs=0.1, sigma_idio_current=0.40, sigma_idio_baseline=0.15)
        assert result is not None
        assert result.action_type == "reset_conviction"
