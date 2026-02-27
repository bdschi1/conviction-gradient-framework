"""Tests for evaluation layer — calibration, alignment, attribution, benchmark."""

import pytest

from evaluation.attribution import decompose_error
from evaluation.benchmark import BenchmarkOutput, score_agent
from evaluation.calibration import brier_score, calibration_buckets
from evaluation.update_alignment import compute_alignment

# --- Brier Score ---


class TestBrierScore:
    def test_perfect_predictions(self):
        score = brier_score([1.0, 0.0, 1.0], [1, 0, 1])
        assert score == pytest.approx(0.0)

    def test_worst_predictions(self):
        score = brier_score([0.0, 1.0, 0.0], [1, 0, 1])
        assert score == pytest.approx(1.0)

    def test_50_50(self):
        score = brier_score([0.5, 0.5], [1, 0])
        assert score == pytest.approx(0.25)

    def test_empty(self):
        assert brier_score([], []) == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            brier_score([0.5], [1, 0])


# --- Calibration Buckets ---


class TestCalibrationBuckets:
    def test_single_bucket(self):
        convictions = [2.0, 2.0, 2.0]
        returns = [0.05, 0.03, -0.01]
        buckets = calibration_buckets(convictions, returns, n_buckets=1)
        assert len(buckets) == 1
        assert buckets[0].count == 3

    def test_multiple_buckets(self):
        convictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        returns = [0.01, 0.02, -0.01, 0.05, 0.08]
        buckets = calibration_buckets(convictions, returns, n_buckets=5)
        assert len(buckets) > 0
        assert sum(b.count for b in buckets) == 5

    def test_hit_rate_positive_conviction(self):
        convictions = [3.0, 3.0, 3.0, 3.0]
        returns = [0.05, 0.03, 0.01, -0.02]  # 3 of 4 positive
        buckets = calibration_buckets(convictions, returns, n_buckets=1)
        assert buckets[0].hit_rate == pytest.approx(0.75)

    def test_empty(self):
        assert calibration_buckets([], []) == []


# --- Update Alignment ---


class TestUpdateAlignment:
    def test_perfect_alignment(self):
        result = compute_alignment([1.0, -0.5, 0.3], [1.0, -0.5, 0.3])
        assert result.correlation == pytest.approx(1.0)
        assert result.bias_direction == "aligned"

    def test_over_reactive(self):
        result = compute_alignment([2.0, -1.0, 0.6], [1.0, -0.5, 0.3])
        assert result.bias_direction == "over_reactive"

    def test_under_reactive(self):
        result = compute_alignment([0.3, -0.1, 0.1], [1.0, -0.5, 0.3])
        assert result.bias_direction == "under_reactive"

    def test_empty(self):
        result = compute_alignment([], [])
        assert result.n_observations == 0

    def test_different_lengths(self):
        result = compute_alignment([1.0, 2.0], [1.0])
        assert result.n_observations == 1


# --- Attribution ---


class TestAttribution:
    def test_basic_decomposition(self):
        result = decompose_error(
            total_return_error=-0.10,
            market_return_error=-0.05,
            beta=1.2,
            vol_actual=0.25,
            vol_forecast=0.20,
            fvs=0.0,
        )
        assert result.total_error == pytest.approx(-0.10)
        assert result.beta_drift == pytest.approx(1.2 * -0.05)

    def test_with_fvs(self):
        result = decompose_error(
            total_return_error=-0.15,
            market_return_error=0.0,
            beta=1.0,
            vol_actual=0.20,
            vol_forecast=0.20,
            fvs=0.8,
        )
        assert result.structural_error != 0.0

    def test_vol_misestimate(self):
        result = decompose_error(
            total_return_error=0.05,
            market_return_error=0.0,
            beta=1.0,
            vol_actual=0.30,
            vol_forecast=0.20,
            fvs=0.0,
        )
        assert result.vol_misestimate == pytest.approx(0.10)


# --- Benchmark ---


class TestBenchmark:
    def test_score_agent(self):
        agent_output = BenchmarkOutput(
            conviction_trajectory=[1.0, 1.5, 2.0, 1.8],
            probability_forecasts=[0.7, 0.8, 0.6, 0.9],
            proposed_weights=[0.03, 0.04, 0.03, 0.04],
        )
        model_trajectory = [1.0, 1.4, 1.9, 1.7]
        realized_returns = [0.02, 0.01, -0.01, 0.03]

        score = score_agent(agent_output, model_trajectory, realized_returns)
        assert 0 <= score.brier <= 1
        assert -1 <= score.conviction_correlation <= 1
        assert 0 <= score.direction_accuracy <= 1
        assert score.overall_score >= 0

    def test_empty_agent(self):
        agent_output = BenchmarkOutput(conviction_trajectory=[])
        score = score_agent(agent_output, [], [])
        assert score.brier == 1.0
