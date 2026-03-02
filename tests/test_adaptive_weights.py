"""Tests for adaptive loss weights.

Reference: Tallman & West, JRSS-B 2024 (BPDS).
"""

from datetime import date

import pytest

from config.defaults import ConvictionParams
from engine.adaptive import (
    AdaptiveWeightTracker,
    compute_component_usefulness,
    update_adaptive_weights,
)
from engine.loss import compute_loss
from engine.models import ConvictionState, InstrumentData
from engine.updater import run_single_update


class TestComponentUsefulness:
    def test_all_useful(self):
        """Components that always predict direction correctly should score high."""
        # Positive FE → negative return (FE predicted trouble correctly)
        history = [{"fe": 1.0, "fvs": 0.5, "rrs": 0.3, "its": -0.2}] * 20
        returns = [-0.05] * 20  # All negative (FE/FVS/RRS positive → trouble)
        scores = compute_component_usefulness(history, returns)
        assert scores["fe"] > 0.8
        assert scores["fvs"] > 0.8
        assert scores["rrs"] > 0.8

    def test_all_useless(self):
        """Components that never predict correctly should score low."""
        # Positive FE but positive returns (wrong direction)
        history = [{"fe": 1.0, "fvs": 0.5, "rrs": 0.3, "its": 0.2}] * 20
        returns = [0.05] * 20
        scores = compute_component_usefulness(history, returns)
        assert scores["fe"] < 0.2
        assert scores["fvs"] < 0.2
        assert scores["rrs"] < 0.2

    def test_mixed_usefulness(self):
        """50% accuracy should score ~0.5."""
        history = []
        returns = []
        for i in range(20):
            if i % 2 == 0:
                history.append({"fe": 1.0, "fvs": 0.5, "rrs": 0.3, "its": 0.0})
                returns.append(-0.05)  # Correct
            else:
                history.append({"fe": 1.0, "fvs": 0.5, "rrs": 0.3, "its": 0.0})
                returns.append(0.05)  # Wrong
        scores = compute_component_usefulness(history, returns)
        assert 0.3 < scores["fe"] < 0.7

    def test_its_directional(self):
        """ITS positive (thesis challenged) → expect negative return."""
        history = [{"fe": 0.0, "fvs": 0.0, "rrs": 0.0, "its": 0.5}] * 20
        returns = [-0.05] * 20  # Negative returns match positive ITS
        scores = compute_component_usefulness(history, returns)
        assert scores["its"] > 0.8

    def test_empty_history(self):
        """Empty history should return neutral scores."""
        scores = compute_component_usefulness([], [])
        for key in ("fe", "fvs", "rrs", "its"):
            assert scores[key] == pytest.approx(0.5)

    def test_zero_components_ignored(self):
        """Zero component values should not count toward accuracy."""
        history = [{"fe": 0.0, "fvs": 0.0, "rrs": 0.0, "its": 0.0}] * 10
        returns = [0.05] * 10
        scores = compute_component_usefulness(history, returns)
        # All zero → no observations → default 0.5
        for key in ("fe", "fvs", "rrs", "its"):
            assert scores[key] == pytest.approx(0.5)


class TestUpdateAdaptiveWeights:
    def test_sum_to_one(self):
        """Updated weights should sum to 1.0."""
        current = {"w1": 0.30, "w2": 0.25, "w3": 0.25, "w4": 0.20}
        usefulness = {"fe": 0.8, "fvs": 0.6, "rrs": 0.4, "its": 0.2}
        updated = update_adaptive_weights(current, usefulness)
        assert sum(updated.values()) == pytest.approx(1.0)

    def test_floor_respected(self):
        """No weight should fall below the floor."""
        current = {"w1": 0.30, "w2": 0.25, "w3": 0.25, "w4": 0.20}
        usefulness = {"fe": 0.0, "fvs": 0.0, "rrs": 0.0, "its": 1.0}
        updated = update_adaptive_weights(current, usefulness, floor=0.05)
        for w in updated.values():
            assert w >= 0.05

    def test_decay_speed(self):
        """Higher decay should produce smaller changes per step."""
        current = {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}
        usefulness = {"fe": 1.0, "fvs": 0.0, "rrs": 0.0, "its": 0.0}

        slow = update_adaptive_weights(current, usefulness, decay=0.99)
        fast = update_adaptive_weights(current, usefulness, decay=0.90)

        # Fast decay should shift w1 more toward usefulness=1.0
        assert fast["w1"] > slow["w1"]

    def test_equal_usefulness_stable(self):
        """Equal usefulness should keep weights close to current."""
        current = {"w1": 0.30, "w2": 0.25, "w3": 0.25, "w4": 0.20}
        usefulness = {"fe": 0.5, "fvs": 0.5, "rrs": 0.5, "its": 0.5}
        updated = update_adaptive_weights(current, usefulness, decay=0.97)
        # With equal usefulness, weights should stay close to current (after renorm)
        for key in current:
            assert abs(updated[key] - current[key]) < 0.05


class TestAdaptiveWeightTracker:
    def test_none_before_lookback(self):
        """Should return None before accumulating lookback observations."""
        tracker = AdaptiveWeightTracker(lookback=10)
        for _ in range(5):
            tracker.record("AAPL", {"fe": 0.1, "fvs": 0.0, "rrs": 0.0, "its": 0.0}, 0.01)
        assert tracker.get_weights("AAPL") is None

    def test_weights_after_lookback(self):
        """Should return weights after lookback observations."""
        tracker = AdaptiveWeightTracker(lookback=10)
        for _ in range(15):
            tracker.record(
                "AAPL",
                {"fe": 0.1, "fvs": 0.0, "rrs": 0.0, "its": 0.0},
                -0.01,  # FE is useful (positive FE → negative return)
            )
        weights = tracker.get_weights("AAPL")
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_weights_evolve(self):
        """Weights should change as new data arrives."""
        tracker = AdaptiveWeightTracker(lookback=10, decay=0.90)
        # Phase 1: FE useful
        for _ in range(12):
            tracker.record("AAPL", {"fe": 1.0, "fvs": 0.0, "rrs": 0.0, "its": 0.0}, -0.05)
        w1 = tracker.get_weights("AAPL")

        # Phase 2: RRS useful
        for _ in range(12):
            tracker.record("AAPL", {"fe": 0.0, "fvs": 0.0, "rrs": 1.0, "its": 0.0}, -0.05)
        w2 = tracker.get_weights("AAPL")

        # w3 (RRS weight) should have increased
        assert w2 is not None
        assert w1 is not None
        assert w2["w3"] > w1["w3"]

    def test_per_instrument_tracking(self):
        """Each instrument should have independent weight tracking."""
        tracker = AdaptiveWeightTracker(lookback=10)
        for _ in range(12):
            tracker.record("AAPL", {"fe": 1.0, "fvs": 0.0, "rrs": 0.0, "its": 0.0}, -0.05)
        for _ in range(5):
            tracker.record("MSFT", {"fe": 0.0, "fvs": 1.0, "rrs": 0.0, "its": 0.0}, -0.05)

        assert tracker.get_weights("AAPL") is not None
        assert tracker.get_weights("MSFT") is None  # Only 5 obs < lookback=10


class TestAdaptiveLossIntegration:
    def test_weight_overrides_in_loss(self):
        """compute_loss should use weight_overrides when provided."""
        params = ConvictionParams(w1=0.30, w2=0.25, w3=0.25, w4=0.20)
        overrides = {"w1": 0.50, "w2": 0.20, "w3": 0.20, "w4": 0.10}

        loss_default = compute_loss(1.0, 0.5, 0.3, 0.1, params)
        loss_override = compute_loss(1.0, 0.5, 0.3, 0.1, params, weight_overrides=overrides)

        assert loss_default.total_loss != loss_override.total_loss
        # Override with higher w1 → higher total (FE squared dominates)
        expected = 0.50 * 1.0 + 0.20 * 0.5 + 0.20 * 0.3 + 0.10 * 0.1
        assert loss_override.total_loss == pytest.approx(expected)

    def test_backward_compatible_without_overrides(self):
        """Without overrides, loss should use params weights."""
        params = ConvictionParams()
        loss1 = compute_loss(1.0, 0.5, 0.3, 0.1, params)
        loss2 = compute_loss(1.0, 0.5, 0.3, 0.1, params, weight_overrides=None)
        assert loss1.total_loss == pytest.approx(loss2.total_loss)

    def test_updater_backward_compatible(self):
        """Without adaptive tracker, updater should work unchanged."""
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
        result_no_tracker = run_single_update(current, data)
        result_tracker_off = run_single_update(current, data, adaptive_tracker=None)
        assert result_no_tracker.conviction == pytest.approx(result_tracker_off.conviction)

    def test_config_fields_exist(self):
        """ConvictionParams should have adaptive weight fields."""
        p = ConvictionParams()
        assert p.adaptive_weights is False
        assert p.adaptive_lookback == 63
        assert p.adaptive_decay == pytest.approx(0.97)
