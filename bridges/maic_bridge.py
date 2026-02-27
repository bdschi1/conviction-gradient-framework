"""Bridge to multi-agent-investment-committee (MAIC).

Consumes MAIC CommitteeResult outputs to extract:
- ADS (Adversarial Debate Shift) from conviction timelines
- FVS events from risk manager analysis
- Conviction snapshots for tracking

Graceful fallback if MAIC is not installed.
"""

from __future__ import annotations

import logging
import sys

from config.settings import settings

logger = logging.getLogger(__name__)

_MAIC_AVAILABLE = False


def _ensure_maic() -> bool:
    """Try to add MAIC to sys.path if available."""
    global _MAIC_AVAILABLE

    if _MAIC_AVAILABLE:
        return True

    maic_path = settings.maic_path
    if maic_path is None:
        return False

    path_str = str(maic_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    try:
        from orchestrator.committee import CommitteeResult, ConvictionSnapshot  # noqa: F401

        _MAIC_AVAILABLE = True
        logger.info("MAIC bridge connected: %s", maic_path)
        return True
    except ImportError:
        logger.info("MAIC not importable from %s", maic_path)
        return False


def is_available() -> bool:
    """Check if MAIC bridge is available."""
    return _ensure_maic()


def extract_ads(committee_result: object) -> float:
    """Extract Adversarial Debate Shift from a CommitteeResult.

    Computes the mean conviction shift from Initial Analysis to
    Post-Debate/PM Decision phases.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        ADS value (positive = conviction increased through debate).
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

    # MAIC uses 0-10 scale; normalize to 0-1 for CGF
    pre_mean = sum(pre_scores) / len(pre_scores) / 10.0
    post_mean = sum(post_scores) / len(post_scores) / 10.0
    return post_mean - pre_mean


def extract_conviction_snapshots(committee_result: object) -> list[dict]:
    """Extract conviction timeline as a list of dicts.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        List of {phase, agent, score, score_type, rationale} dicts.
    """
    timeline = getattr(committee_result, "conviction_timeline", [])
    return [
        {
            "phase": getattr(s, "phase", ""),
            "agent": getattr(s, "agent", ""),
            "score": getattr(s, "score", 0.0),
            "score_type": getattr(s, "score_type", ""),
            "rationale": getattr(s, "rationale", ""),
        }
        for s in timeline
    ]


def extract_fvs_events_from_bear_case(committee_result: object) -> list[dict]:
    """Extract FVS-relevant events from the risk manager's bear case.

    Parses the bear case risks to identify fundamental violation events
    that can be mapped to the FVS taxonomy.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        List of dicts compatible with FVSEvent construction.
    """
    bear_case = getattr(committee_result, "bear_case", None)
    if bear_case is None:
        return []

    events = []
    risks = getattr(bear_case, "risks", [])
    for risk in risks:
        risk_text = str(risk).lower()

        # Heuristic mapping of risk text to FVS event types
        event_type = "kpi_miss"  # default
        if any(kw in risk_text for kw in ["management", "ceo", "cfo", "executive"]):
            event_type = "mgmt_change"
        elif any(kw in risk_text for kw in ["guidance", "outlook", "forecast"]):
            event_type = "guidance_cut"
        elif any(kw in risk_text for kw in ["governance", "board", "audit", "regul"]):
            event_type = "governance_breach"
        elif any(kw in risk_text for kw in ["model", "strategy", "pivot", "restructur"]):
            event_type = "business_model_change"
        elif any(kw in risk_text for kw in ["restate", "accounting", "sec filing"]):
            event_type = "accounting_restatement"
        elif any(kw in risk_text for kw in ["m&a", "acquisition", "buyback", "dividend"]):
            event_type = "capital_allocation"

        events.append({
            "event_type": event_type,
            "description": str(risk),
        })

    return events
