"""SQLite persistence for conviction states, events, and IC sessions."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

from engine.models import ConvictionState
from storage.schemas import ALL_TABLES, INDEXES

logger = logging.getLogger(__name__)


class ConvictionDB:
    """SQLite database for persisting conviction engine state."""

    def __init__(self, db_path: str | Path = "conviction.db"):
        self._path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        for table_sql in ALL_TABLES:
            cursor.execute(table_sql)
        for idx_sql in INDEXES:
            cursor.execute(idx_sql)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Conviction States ---

    def save_state(self, state: ConvictionState) -> None:
        """Save or update a conviction state."""
        conn = self._get_conn()
        loss = state.loss_components
        grad = state.gradient

        conn.execute(
            """
            INSERT OR REPLACE INTO conviction_states
            (instrument_id, as_of_date, conviction, conviction_prev,
             expected_return, idiosyncratic_vol, alpha_t,
             fe, fvs, rrs, its, total_loss, gradient_value, learning_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.instrument_id,
                state.as_of_date.isoformat(),
                state.conviction,
                state.conviction_prev,
                state.expected_return,
                state.idiosyncratic_vol,
                state.alpha_t,
                loss.fe if loss else None,
                loss.fvs if loss else None,
                loss.rrs if loss else None,
                loss.its if loss else None,
                loss.total_loss if loss else None,
                grad.gradient_value if grad else None,
                grad.learning_rate if grad else None,
            ),
        )
        conn.commit()

    def get_trajectory(
        self,
        instrument_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get conviction trajectory for an instrument.

        Returns list of dicts with conviction state data, ordered by date.
        """
        conn = self._get_conn()
        query = "SELECT * FROM conviction_states WHERE instrument_id = ?"
        params: list = [instrument_id]

        if start_date:
            query += " AND as_of_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND as_of_date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY as_of_date"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_latest_state(self, instrument_id: str) -> dict | None:
        """Get the most recent conviction state for an instrument."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT * FROM conviction_states
            WHERE instrument_id = ?
            ORDER BY as_of_date DESC LIMIT 1
            """,
            (instrument_id,),
        ).fetchone()
        return dict(row) if row else None

    # --- Events ---

    def save_event(
        self,
        instrument_id: str,
        event_date: date,
        event_type: str,
        severity: float,
        description: str = "",
    ) -> None:
        """Save a fundamental violation event."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO events (instrument_id, event_date, event_type, severity, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (instrument_id, event_date.isoformat(), event_type, severity, description),
        )
        conn.commit()

    def get_events(
        self,
        instrument_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get events for an instrument."""
        conn = self._get_conn()
        query = "SELECT * FROM events WHERE instrument_id = ?"
        params: list = [instrument_id]

        if start_date:
            query += " AND event_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND event_date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY event_date"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # --- IC Sessions ---

    def save_session(
        self,
        session_id: str,
        instrument_id: str,
        session_date: date,
        status: str,
        red_team_analyst: str | None = None,
        notes: str = "",
        its_value: float | None = None,
    ) -> None:
        """Save or update an IC session."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO ic_sessions
            (session_id, instrument_id, session_date, status,
             red_team_analyst, notes, its_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, instrument_id, session_date.isoformat(),
                status, red_team_analyst, notes, its_value,
            ),
        )
        conn.commit()

    # --- Track Records ---

    def save_track_record(
        self,
        analyst_id: str,
        evaluation_date: date,
        brier_score: float | None = None,
        mean_update_alignment: float | None = None,
        n_forecasts: int = 0,
        bias_direction: str = "aligned",
    ) -> None:
        """Save analyst/AI track record metrics."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO track_records
            (analyst_id, evaluation_date, brier_score,
             mean_update_alignment, n_forecasts, bias_direction)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                analyst_id, evaluation_date.isoformat(),
                brier_score, mean_update_alignment, n_forecasts, bias_direction,
            ),
        )
        conn.commit()

    def get_track_record(self, analyst_id: str) -> list[dict]:
        """Get track record for an analyst/AI."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM track_records WHERE analyst_id = ? ORDER BY evaluation_date",
            (analyst_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # --- Target Weights ---

    def save_target_weights(
        self,
        weights: dict[str, dict],
        as_of_date: date,
    ) -> None:
        """Save target portfolio weights.

        Args:
            weights: {instrument_id: {conviction, raw_weight, constrained_weight, sizing_method}}
            as_of_date: Date for these weights.
        """
        conn = self._get_conn()
        for instrument_id, w in weights.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO target_weights
                (instrument_id, as_of_date, conviction, raw_weight,
                 constrained_weight, sizing_method)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    instrument_id,
                    as_of_date.isoformat(),
                    w.get("conviction", 0.0),
                    w.get("raw_weight", 0.0),
                    w.get("constrained_weight", 0.0),
                    w.get("sizing_method", "basic"),
                ),
            )
        conn.commit()

    def get_target_weights(self, as_of_date: date) -> list[dict]:
        """Get target weights for a specific date."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM target_weights WHERE as_of_date = ? ORDER BY instrument_id",
            (as_of_date.isoformat(),),
        ).fetchall()
        return [dict(row) for row in rows]
