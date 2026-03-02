"""Tests for core engine: loss, gradient, update rule, stability."""

from datetime import date

import pytest

from config.defaults import ConvictionParams
from engine.gradient import compute_gradient, compute_gradient_result, compute_learning_rate
from engine.loss import compute_loss
from engine.models import ConvictionState, InstrumentData
from engine.stability import apply_clipping, check_stability, count_sign_flips, detect_oscillation
from engine.updater import run_batch_update, run_single_update, update_conviction

# --- Loss ---


class TestLoss:
    def test_zero_components(self):
        result = compute_loss(0, 0, 0, 0)
        assert result.total_loss == pytest.approx(0.0)

    def test_fe_squared(self):
        params = ConvictionParams(w1=1.0, w2=0.0, w3=0.0, w4=0.0)
        result = compute_loss(fe=2.0, fvs=0, rrs=0, its=0, params=params)
        assert result.total_loss == pytest.approx(4.0)

    def test_weighted_sum(self):
        params = ConvictionParams(w1=0.25, w2=0.25, w3=0.25, w4=0.25)
        result = compute_loss(fe=1.0, fvs=0.5, rrs=0.3, its=-0.1, params=params)
        expected = 0.25 * 1.0 + 0.25 * 0.5 + 0.25 * 0.3 + 0.25 * (-0.1)
        assert result.total_loss == pytest.approx(expected)

    def test_components_stored(self):
        result = compute_loss(fe=1.5, fvs=0.6, rrs=0.2, its=-0.3)
        assert result.fe == pytest.approx(1.5)
        assert result.fvs == pytest.approx(0.6)
        assert result.rrs == pytest.approx(0.2)
        assert result.its == pytest.approx(-0.3)


# --- Gradient ---


class TestGradient:
    def test_zero_gradient(self):
        g = compute_gradient(0, 0, 0, 0)
        assert g == pytest.approx(0.0)

    def test_linear_combination(self):
        params = ConvictionParams(lambda1=2.0, lambda2=1.0, lambda3=1.0, lambda4=1.0)
        g = compute_gradient(fe=1.0, fvs=0.5, rrs=0.3, its=0.1, params=params)
        expected = 2.0 * 1.0 + 1.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.1
        assert g == pytest.approx(expected)

    def test_gradient_result_breakdown(self):
        result = compute_gradient_result(
            fe=1.0, fvs=0.5, rrs=0.3, its=0.1, alpha_t=0.05
        )
        assert result.learning_rate == pytest.approx(0.05)
        assert "fe" in result.component_contributions
        assert "fvs" in result.component_contributions
        assert result.gradient_value == pytest.approx(
            sum(result.component_contributions.values())
        )


class TestLearningRate:
    def test_basic_computation(self):
        alpha = compute_learning_rate(
            kappa=0.1,
            info_half_life=1.0,
            sigma_idio=0.20,
            sigma_expected=0.20,
            track_record_score=0.0,
            alpha_min=0.01,
            alpha_max=0.5,
        )
        # 0.1 / (1+1) * (0.20/0.20) * (1/(1+0)) = 0.05
        assert alpha == pytest.approx(0.05)

    def test_high_track_record_lowers_alpha(self):
        alpha = compute_learning_rate(
            kappa=0.1,
            info_half_life=1.0,
            sigma_idio=0.20,
            sigma_expected=0.20,
            track_record_score=4.0,
            alpha_min=0.01,
            alpha_max=0.5,
        )
        # 0.1 / 2 * 1 * (1/5) = 0.01
        assert alpha == pytest.approx(0.01)

    def test_clamped_to_min(self):
        alpha = compute_learning_rate(
            kappa=0.001,
            info_half_life=10.0,
            sigma_idio=0.01,
            sigma_expected=0.20,
            track_record_score=10.0,
            alpha_min=0.01,
            alpha_max=0.5,
        )
        assert alpha == pytest.approx(0.01)

    def test_clamped_to_max(self):
        alpha = compute_learning_rate(
            kappa=10.0,
            info_half_life=0.0,
            sigma_idio=0.50,
            sigma_expected=0.05,
            track_record_score=0.0,
            alpha_min=0.01,
            alpha_max=0.5,
        )
        assert alpha == pytest.approx(0.5)

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            compute_learning_rate(0.1, 1.0, 0.20, 0.0, 0.0, 0.01, 0.5)


# --- Stability ---


class TestStability:
    def test_stable_condition(self):
        assert check_stability(0.1, 5.0) is True  # 0.5 < 2

    def test_unstable_condition(self):
        assert check_stability(0.5, 5.0) is False  # 2.5 >= 2

    def test_boundary(self):
        assert check_stability(0.1, 20.0) is False  # 2.0 not < 2

    def test_zero_alpha(self):
        assert check_stability(0.0, 5.0) is False  # 0 not > 0


class TestClipping:
    def test_no_clipping(self):
        assert apply_clipping(2.0, 5.0) == pytest.approx(2.0)

    def test_clip_high(self):
        assert apply_clipping(10.0, 5.0) == pytest.approx(5.0)

    def test_clip_low(self):
        assert apply_clipping(-10.0, 5.0) == pytest.approx(-5.0)

    def test_zero(self):
        assert apply_clipping(0.0, 5.0) == pytest.approx(0.0)


class TestOscillation:
    def test_no_oscillation(self):
        assert detect_oscillation([1.0, 2.0, 3.0, 4.0]) is False

    def test_oscillation_detected(self):
        assert detect_oscillation([1.0, -1.0, 1.0]) is True

    def test_single_value(self):
        assert detect_oscillation([1.0]) is False

    def test_empty(self):
        assert detect_oscillation([]) is False

    def test_count_flips(self):
        assert count_sign_flips([1.0, -1.0, 1.0, -1.0], window=4) == 3

    def test_count_no_flips(self):
        assert count_sign_flips([1.0, 2.0, 3.0], window=3) == 0


# --- Update Rule ---


class TestUpdateConviction:
    def _make_state(self, c: float, c_prev: float = 0.0) -> ConvictionState:
        return ConvictionState(
            instrument_id="TEST",
            as_of_date=date(2024, 6, 1),
            conviction=c,
            conviction_prev=c_prev,
        )

    def test_no_gradient_no_change(self):
        state = self._make_state(2.0, 2.0)
        new_c = update_conviction(state, gradient_value=0.0, alpha_t=0.1, beta=0.0, c_max=5.0)
        assert new_c == pytest.approx(2.0)

    def test_positive_gradient_decreases(self):
        state = self._make_state(2.0, 2.0)
        new_c = update_conviction(state, gradient_value=1.0, alpha_t=0.1, beta=0.0, c_max=5.0)
        assert new_c == pytest.approx(1.9)

    def test_momentum_effect(self):
        state = self._make_state(2.0, 1.5)
        new_c = update_conviction(state, gradient_value=0.0, alpha_t=0.1, beta=0.5, c_max=5.0)
        # momentum = 0.5 * (2.0 - 1.5) = 0.25
        assert new_c == pytest.approx(2.25)

    def test_clipping(self):
        state = self._make_state(4.9, 4.5)
        new_c = update_conviction(state, gradient_value=-10.0, alpha_t=0.5, beta=0.0, c_max=5.0)
        assert new_c == pytest.approx(5.0)

    def test_negative_clipping(self):
        state = self._make_state(-4.9, -4.5)
        new_c = update_conviction(state, gradient_value=10.0, alpha_t=0.5, beta=0.0, c_max=5.0)
        assert new_c == pytest.approx(-5.0)


# --- Full Update Pipeline ---


class TestRunSingleUpdate:
    def test_basic_update(self):
        current = ConvictionState(
            instrument_id="AAPL",
            as_of_date=date(2024, 5, 31),
            conviction=2.0,
            conviction_prev=1.8,
        )
        data = InstrumentData(
            instrument_id="AAPL",
            as_of_date=date(2024, 6, 1),
            realized_return=0.05,
            expected_return=0.08,
            sigma_expected=0.20,
            sigma_idio_current=0.18,
            sigma_idio_prev=0.15,
            info_half_life=1.0,
            track_record_score=0.5,
        )
        result = run_single_update(current, data)

        assert result.instrument_id == "AAPL"
        assert result.as_of_date == date(2024, 6, 1)
        assert result.conviction_prev == pytest.approx(2.0)
        assert result.loss_components is not None
        assert result.gradient is not None
        assert result.alpha_t > 0

    def test_conviction_changes(self):
        current = ConvictionState(
            instrument_id="MSFT",
            as_of_date=date(2024, 5, 31),
            conviction=1.0,
            conviction_prev=1.0,
        )
        data = InstrumentData(
            instrument_id="MSFT",
            as_of_date=date(2024, 6, 1),
            realized_return=-0.10,
            expected_return=0.05,
            sigma_expected=0.15,
            sigma_idio_current=0.25,
            sigma_idio_prev=0.15,
            info_half_life=1.0,
            track_record_score=0.0,
        )
        result = run_single_update(current, data)

        # Large negative surprise + vol increase should decrease conviction
        assert result.conviction != 1.0

    def test_with_fvs_events(self):
        current = ConvictionState(
            instrument_id="XYZ",
            as_of_date=date(2024, 5, 31),
            conviction=3.0,
            conviction_prev=3.0,
        )
        data = InstrumentData(
            instrument_id="XYZ",
            as_of_date=date(2024, 6, 1),
            realized_return=0.0,
            expected_return=0.05,
            sigma_expected=0.20,
            sigma_idio_current=0.20,
            sigma_idio_prev=0.20,
            fvs_events=[
                {
                    "event_type": "governance_breach",
                    "event_date": "2024-06-01",
                    "description": "Board conflict",
                }
            ],
        )
        result = run_single_update(current, data)
        assert result.loss_components is not None
        assert result.loss_components.fvs > 0


class TestBatchUpdate:
    def test_batch_update(self):
        states = {
            "AAPL": ConvictionState(
                instrument_id="AAPL",
                as_of_date=date(2024, 5, 31),
                conviction=2.0,
                conviction_prev=1.8,
            ),
        }
        data_batch = [
            InstrumentData(
                instrument_id="AAPL",
                as_of_date=date(2024, 6, 1),
                realized_return=0.03,
                expected_return=0.05,
                sigma_expected=0.20,
                sigma_idio_current=0.18,
                sigma_idio_prev=0.18,
            ),
            InstrumentData(
                instrument_id="MSFT",
                as_of_date=date(2024, 6, 1),
                realized_return=0.01,
                expected_return=0.04,
                sigma_expected=0.15,
                sigma_idio_current=0.12,
                sigma_idio_prev=0.12,
            ),
        ]
        results = run_batch_update(states, data_batch)

        assert "AAPL" in results
        assert "MSFT" in results
        # MSFT should start from zero (new position)
        assert results["MSFT"].conviction_prev == pytest.approx(0.0)
