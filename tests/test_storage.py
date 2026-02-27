"""Tests for SQLite storage layer."""

from datetime import date

import pytest

from engine.models import ConvictionState, GradientResult, LossComponents
from storage.database import ConvictionDB


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    database = ConvictionDB(db_path)
    yield database
    database.close()


class TestConvictionDB:
    def test_save_and_load_state(self, db):
        state = ConvictionState(
            instrument_id="AAPL",
            as_of_date=date(2024, 6, 1),
            conviction=2.5,
            conviction_prev=2.0,
            expected_return=0.08,
            idiosyncratic_vol=0.18,
            alpha_t=0.05,
            loss_components=LossComponents(
                fe=1.2, fvs=0.3, rrs=0.1, ads=-0.05, total_loss=0.45,
            ),
            gradient=GradientResult(
                gradient_value=1.55,
                component_contributions={"fe": 1.2, "fvs": 0.3, "rrs": 0.1, "ads": -0.05},
                learning_rate=0.05,
            ),
        )
        db.save_state(state)

        latest = db.get_latest_state("AAPL")
        assert latest is not None
        assert latest["conviction"] == pytest.approx(2.5)
        assert latest["conviction_prev"] == pytest.approx(2.0)
        assert latest["fe"] == pytest.approx(1.2)

    def test_trajectory(self, db):
        for i, d in enumerate([1, 2, 3, 4, 5]):
            state = ConvictionState(
                instrument_id="MSFT",
                as_of_date=date(2024, 6, d),
                conviction=1.0 + i * 0.5,
                conviction_prev=1.0 + max(0, (i - 1) * 0.5),
            )
            db.save_state(state)

        traj = db.get_trajectory("MSFT")
        assert len(traj) == 5
        assert traj[0]["as_of_date"] == "2024-06-01"
        assert traj[-1]["as_of_date"] == "2024-06-05"

    def test_trajectory_date_filter(self, db):
        for d in [1, 2, 3, 4, 5]:
            db.save_state(ConvictionState(
                instrument_id="GOOG",
                as_of_date=date(2024, 6, d),
                conviction=float(d),
            ))

        traj = db.get_trajectory(
            "GOOG", start_date=date(2024, 6, 2), end_date=date(2024, 6, 4)
        )
        assert len(traj) == 3

    def test_no_data(self, db):
        assert db.get_latest_state("NONEXISTENT") is None
        assert db.get_trajectory("NONEXISTENT") == []

    def test_save_event(self, db):
        db.save_event("AAPL", date(2024, 6, 1), "kpi_miss", 0.4, "Q2 revenue miss")
        events = db.get_events("AAPL")
        assert len(events) == 1
        assert events[0]["event_type"] == "kpi_miss"

    def test_save_session(self, db):
        db.save_session(
            session_id="s1",
            instrument_id="AAPL",
            session_date=date(2024, 6, 1),
            status="finalized",
            ads_value=-0.125,
        )
        # Verify by querying directly
        conn = db._get_conn()
        row = conn.execute(
            "SELECT * FROM ic_sessions WHERE session_id = 's1'"
        ).fetchone()
        assert row is not None
        assert dict(row)["ads_value"] == pytest.approx(-0.125)

    def test_track_record(self, db):
        db.save_track_record(
            analyst_id="analyst1",
            evaluation_date=date(2024, 6, 1),
            brier_score=0.15,
            n_forecasts=50,
        )
        records = db.get_track_record("analyst1")
        assert len(records) == 1
        assert records[0]["brier_score"] == pytest.approx(0.15)

    def test_target_weights(self, db):
        weights = {
            "AAPL": {"conviction": 2.5, "raw_weight": 0.04, "constrained_weight": 0.035, "sizing_method": "vol_adjusted"},
            "MSFT": {"conviction": 1.8, "raw_weight": 0.03, "constrained_weight": 0.028, "sizing_method": "vol_adjusted"},
        }
        db.save_target_weights(weights, date(2024, 6, 1))

        loaded = db.get_target_weights(date(2024, 6, 1))
        assert len(loaded) == 2

    def test_upsert_state(self, db):
        state1 = ConvictionState(
            instrument_id="AAPL",
            as_of_date=date(2024, 6, 1),
            conviction=2.0,
        )
        db.save_state(state1)

        state2 = ConvictionState(
            instrument_id="AAPL",
            as_of_date=date(2024, 6, 1),
            conviction=3.0,
        )
        db.save_state(state2)

        latest = db.get_latest_state("AAPL")
        assert latest["conviction"] == pytest.approx(3.0)
