"""Tests for position sizing overlay."""

import pytest

from sizing.constraints import PortfolioConstraints, apply_constraints
from sizing.failure_modes import (
    check_oscillation_guard,
    check_structural_reset,
)
from sizing.mapper import SizingMethod, basic_mapping, map_convictions, vol_adjusted_mapping

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
