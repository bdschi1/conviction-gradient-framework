"""Tests for bridge modules — graceful degradation when target repos unavailable."""


from bridges.data_bridge import compute_historical_vol, compute_idiosyncratic_vol
from bridges.maic_bridge import extract_conviction_snapshots, extract_its

# --- Data Bridge (math functions, no external deps) ---


class TestDataBridgeMath:
    def test_historical_vol(self):
        import numpy as np

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252).tolist()
        vol = compute_historical_vol(returns, window=21, annualize=True)
        assert 0.05 < vol < 0.30  # reasonable annualized vol range

    def test_historical_vol_short_series(self):
        vol = compute_historical_vol([0.01, -0.01, 0.02], window=21, annualize=True)
        assert vol > 0

    def test_idiosyncratic_vol(self):
        import numpy as np

        rng = np.random.default_rng(42)
        market = rng.normal(0, 0.01, 252).tolist()
        # Stock with beta ~1.5 + idiosyncratic noise
        stock = [1.5 * m + rng.normal(0, 0.005) for m in market]
        idio_vol = compute_idiosyncratic_vol(stock, market, annualize=True)
        assert idio_vol > 0

    def test_idiosyncratic_vol_short_data(self):
        # Should fallback to total vol
        vol = compute_idiosyncratic_vol([0.01, -0.01], [0.005, -0.005])
        assert vol > 0


# --- MAIC Bridge (duck-typed, no import needed) ---


class MockConvictionSnapshot:
    def __init__(self, phase, agent, score, score_type="conviction"):
        self.phase = phase
        self.agent = agent
        self.score = score
        self.score_type = score_type
        self.rationale = ""


class MockCommitteeResult:
    def __init__(self, timeline=None):
        self.conviction_timeline = timeline or []


class TestMAICBridge:
    def test_extract_its(self):
        timeline = [
            MockConvictionSnapshot("Initial Analysis", "Long Analyst", 7.5),
            MockConvictionSnapshot("Initial Analysis", "Short Analyst", 5.0),
            MockConvictionSnapshot("Post-Debate", "Long Analyst", 6.0),
            MockConvictionSnapshot("PM Decision", "Portfolio Manager", 6.5),
        ]
        result = MockCommitteeResult(timeline)
        its = extract_its(result)
        assert isinstance(its, float)

    def test_extract_its_empty(self):
        result = MockCommitteeResult([])
        assert extract_its(result) == 0.0

    def test_extract_its_no_post_debate(self):
        timeline = [
            MockConvictionSnapshot("Initial Analysis", "Long Analyst", 7.0),
        ]
        result = MockCommitteeResult(timeline)
        assert extract_its(result) == 0.0

    def test_extract_snapshots(self):
        timeline = [
            MockConvictionSnapshot("Initial Analysis", "Long Analyst", 7.5),
            MockConvictionSnapshot("Post-Debate", "Long Analyst", 6.0),
        ]
        result = MockCommitteeResult(timeline)
        snapshots = extract_conviction_snapshots(result)
        assert len(snapshots) == 2
        assert snapshots[0]["phase"] == "Initial Analysis"
        assert snapshots[0]["score"] == 7.5


# --- Backtest Bridge ---


class TestBacktestBridge:
    def test_bridge_availability_check(self):
        from bridges.backtest_bridge import is_available

        # May or may not be available depending on test environment
        result = is_available()
        assert isinstance(result, bool)


# --- Portfolio Bridge ---


class TestPortfolioBridge:
    def test_bridge_availability_check(self):
        from bridges.portfolio_bridge import is_available

        result = is_available()
        assert isinstance(result, bool)
