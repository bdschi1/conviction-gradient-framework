"""FastAPI endpoints for the Conviction Gradient Framework."""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="Conviction Gradient Framework",
        description="Gradient-descent conviction engine for L/S portfolio sizing",
        version="1.0.0",
    )
except ImportError:
    app = None
    logger.info("FastAPI not installed; API not available. pip install -e '.[api]'")


# --- Request/Response models ---

class UpdateRequest(BaseModel):
    """Request to update conviction for instruments."""

    instruments: list[dict] = Field(description="List of InstrumentData dicts")


class TrajectoryResponse(BaseModel):
    """Conviction trajectory for an instrument."""

    instrument_id: str
    trajectory: list[dict]


class WeightsResponse(BaseModel):
    """Portfolio target weights."""

    as_of_date: str
    weights: list[dict]


class HealthResponse(BaseModel):
    """API health check."""

    status: str = "healthy"
    version: str = "1.0.0"
    n_instruments: int = 0


# --- Endpoints ---

if app is not None:

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check."""
        from storage.database import ConvictionDB
        db = ConvictionDB()
        try:
            conn = db._get_conn()
            count = conn.execute(
                "SELECT COUNT(DISTINCT instrument_id) FROM conviction_states"
            ).fetchone()[0]
        except Exception:
            count = 0
        finally:
            db.close()
        return HealthResponse(n_instruments=count)

    @app.post("/conviction/update")
    def update_conviction(request: UpdateRequest) -> dict:
        """Recompute conviction for instruments given new data."""
        from engine.models import ConvictionState, InstrumentData
        from engine.updater import run_batch_update
        from storage.database import ConvictionDB

        db = ConvictionDB()
        try:
            # Load current states
            states = {}
            for inst_data in request.instruments:
                inst_id = inst_data["instrument_id"]
                latest = db.get_latest_state(inst_id)
                if latest:
                    states[inst_id] = ConvictionState(
                        instrument_id=inst_id,
                        as_of_date=date.fromisoformat(latest["as_of_date"]),
                        conviction=latest["conviction"],
                        conviction_prev=latest["conviction_prev"],
                    )

            # Run updates
            data_batch = [InstrumentData(**d) for d in request.instruments]
            updated = run_batch_update(states, data_batch)

            # Persist
            for state in updated.values():
                db.save_state(state)

            return {
                "updated": len(updated),
                "instruments": [s.instrument_id for s in updated.values()],
            }
        finally:
            db.close()

    @app.get("/conviction/trajectory", response_model=TrajectoryResponse)
    def get_trajectory(
        instrument_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> TrajectoryResponse:
        """Get conviction trajectory for an instrument."""
        from storage.database import ConvictionDB

        db = ConvictionDB()
        try:
            start_date = date.fromisoformat(start) if start else None
            end_date = date.fromisoformat(end) if end else None
            trajectory = db.get_trajectory(instrument_id, start_date, end_date)
            return TrajectoryResponse(
                instrument_id=instrument_id,
                trajectory=trajectory,
            )
        finally:
            db.close()

    @app.get("/portfolio/target_weights", response_model=WeightsResponse)
    def get_target_weights(as_of_date: str) -> WeightsResponse:
        """Get target portfolio weights for a date."""
        from storage.database import ConvictionDB

        db = ConvictionDB()
        try:
            d = date.fromisoformat(as_of_date)
            weights = db.get_target_weights(d)
            return WeightsResponse(as_of_date=as_of_date, weights=weights)
        finally:
            db.close()

    @app.get("/evaluation/analyst_summary")
    def get_analyst_summary(analyst_id: str) -> dict:
        """Get analyst/AI evaluation summary."""
        from storage.database import ConvictionDB

        db = ConvictionDB()
        try:
            records = db.get_track_record(analyst_id)
            if not records:
                raise HTTPException(status_code=404, detail="Analyst not found")
            return {"analyst_id": analyst_id, "records": records}
        finally:
            db.close()
