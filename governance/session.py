"""IC Session management and probability collection.

Manages the lifecycle of investment committee sessions:
create → collect pre-debate probabilities → debate → collect post-debate → finalize.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from components.thesis_shift import compute_its

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """IC session lifecycle states."""

    CREATED = "created"
    PRE_DEBATE = "pre_debate"
    DEBATING = "debating"
    POST_DEBATE = "post_debate"
    FINALIZED = "finalized"


class ProbabilitySubmission(BaseModel):
    """Anonymous thesis probability from one IC participant."""

    session_id: str
    analyst_id: str
    p_pre: float | None = Field(default=None, ge=0, le=1, description="Pre-debate probability")
    p_post: float | None = Field(default=None, ge=0, le=1, description="Post-debate probability")
    submitted_at: datetime = Field(default_factory=datetime.now)


class ICSession(BaseModel):
    """An investment committee session for one instrument."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    instrument_id: str
    session_date: date
    participants: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.CREATED
    submissions: list[ProbabilitySubmission] = Field(default_factory=list)
    red_team_analyst: str | None = None
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class SessionManager:
    """Manages IC session lifecycle and probability collection."""

    def __init__(self) -> None:
        self._sessions: dict[str, ICSession] = {}

    def create_session(
        self,
        instrument_id: str,
        session_date: date,
        participants: list[str],
        red_team_analyst: str | None = None,
    ) -> ICSession:
        """Create a new IC session."""
        session = ICSession(
            instrument_id=instrument_id,
            session_date=session_date,
            participants=participants,
            red_team_analyst=red_team_analyst,
            status=SessionStatus.PRE_DEBATE,
        )
        self._sessions[session.session_id] = session
        logger.info(
            "Created IC session %s for %s on %s with %d participants",
            session.session_id, instrument_id, session_date, len(participants),
        )
        return session

    def submit_pre_probability(
        self,
        session_id: str,
        analyst_id: str,
        probability: float,
    ) -> None:
        """Submit a pre-debate thesis probability."""
        session = self._get_session(session_id)
        if session.status != SessionStatus.PRE_DEBATE:
            raise ValueError(f"Session {session_id} is not in PRE_DEBATE state")

        # Update existing submission or create new
        existing = self._find_submission(session, analyst_id)
        if existing:
            existing.p_pre = probability
        else:
            session.submissions.append(ProbabilitySubmission(
                session_id=session_id,
                analyst_id=analyst_id,
                p_pre=probability,
            ))

    def start_debate(self, session_id: str) -> None:
        """Transition session to debating state."""
        session = self._get_session(session_id)
        if session.status != SessionStatus.PRE_DEBATE:
            raise ValueError(f"Session {session_id} is not in PRE_DEBATE state")

        pre_count = sum(1 for s in session.submissions if s.p_pre is not None)
        if pre_count == 0:
            raise ValueError("No pre-debate probabilities submitted")

        session.status = SessionStatus.DEBATING

    def end_debate(self, session_id: str) -> None:
        """Transition to post-debate probability collection."""
        session = self._get_session(session_id)
        if session.status != SessionStatus.DEBATING:
            raise ValueError(f"Session {session_id} is not in DEBATING state")
        session.status = SessionStatus.POST_DEBATE

    def submit_post_probability(
        self,
        session_id: str,
        analyst_id: str,
        probability: float,
    ) -> None:
        """Submit a post-debate thesis probability."""
        session = self._get_session(session_id)
        if session.status != SessionStatus.POST_DEBATE:
            raise ValueError(f"Session {session_id} is not in POST_DEBATE state")

        existing = self._find_submission(session, analyst_id)
        if existing:
            existing.p_post = probability
        else:
            session.submissions.append(ProbabilitySubmission(
                session_id=session_id,
                analyst_id=analyst_id,
                p_post=probability,
            ))

    def finalize(self, session_id: str) -> float:
        """Finalize session and compute ITS.

        Returns:
            Independent Thesis Shift value.
        """
        session = self._get_session(session_id)
        if session.status != SessionStatus.POST_DEBATE:
            raise ValueError(f"Session {session_id} is not in POST_DEBATE state")

        p_pre = [s.p_pre for s in session.submissions if s.p_pre is not None]
        p_post = [s.p_post for s in session.submissions if s.p_post is not None]

        if not p_pre or not p_post:
            raise ValueError("Insufficient probability submissions to compute ITS")

        its = compute_its(p_pre=p_pre, p_post=p_post)
        session.status = SessionStatus.FINALIZED
        logger.info(
            "Session %s finalized. ITS = %.4f (pre_mean=%.3f, post_mean=%.3f)",
            session_id, its,
            sum(p_pre) / len(p_pre),
            sum(p_post) / len(p_post),
        )
        return its

    def get_session(self, session_id: str) -> ICSession:
        """Get a session by ID."""
        return self._get_session(session_id)

    def _get_session(self, session_id: str) -> ICSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")
        return session

    @staticmethod
    def _find_submission(
        session: ICSession, analyst_id: str
    ) -> ProbabilitySubmission | None:
        for s in session.submissions:
            if s.analyst_id == analyst_id:
                return s
        return None
