"""Independent thesis testing workflow and ITS computation.

Structures IC debate into quantitative signals. Can consume MAIC
CommitteeResult outputs or standalone debate records.
"""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DebateRecord(BaseModel):
    """Record of an IC thesis testing round."""

    session_id: str
    round_number: int = 1
    lead_analyst: str = ""
    counter_thesis: str = ""
    rebuttal_summary: str = ""
    pre_conviction: float | None = None
    post_conviction: float | None = None
    position_type: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


def extract_its_from_debate_records(records: list[DebateRecord]) -> float:
    """Compute ITS from debate records.

    Uses the conviction shift across thesis testing rounds.

    Args:
        records: Debate records with pre/post conviction scores.

    Returns:
        ITS value (positive = thesis challenged, negative = thesis confirmed).
    """
    if not records:
        return 0.0

    pre_scores = [r.pre_conviction for r in records if r.pre_conviction is not None]
    post_scores = [r.post_conviction for r in records if r.post_conviction is not None]

    if not pre_scores or not post_scores:
        return 0.0

    pre_mean = sum(pre_scores) / len(pre_scores)
    post_mean = sum(post_scores) / len(post_scores)
    return post_mean - pre_mean


def extract_its_from_committee_result(committee_result: object) -> float:
    """Extract ITS from a MAIC CommitteeResult.

    Parses the conviction_timeline to find pre- and post-debate
    conviction scores across all agents.

    Args:
        committee_result: A MAIC CommitteeResult object (duck-typed).

    Returns:
        ITS value.
    """
    timeline = getattr(committee_result, "conviction_timeline", [])
    if not timeline:
        return 0.0

    pre_scores = []
    post_scores = []

    for snapshot in timeline:
        phase = getattr(snapshot, "phase", "")
        score = getattr(snapshot, "score", None)
        if score is None:
            continue

        if "Initial" in phase:
            pre_scores.append(score)
        elif "Post-Debate" in phase or "PM Decision" in phase:
            post_scores.append(score)

    if not pre_scores or not post_scores:
        return 0.0

    # Normalize to 0-1 scale (MAIC uses 0-10)
    pre_mean = sum(pre_scores) / len(pre_scores) / 10.0
    post_mean = sum(post_scores) / len(post_scores) / 10.0
    return post_mean - pre_mean


# Backward compatibility aliases
extract_ads_from_debate_records = extract_its_from_debate_records
extract_ads_from_committee_result = extract_its_from_committee_result
