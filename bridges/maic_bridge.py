"""Bridge to multi-agent-investment-committee (MAIC).

Consumes MAIC CommitteeResult outputs to extract:
- ITS (Independent Thesis Shift) from conviction timelines
- PM conviction and analyst convictions for rich ITS mode
- Position type classification
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


def extract_its(committee_result: object) -> float:
    """Extract Independent Thesis Shift from a CommitteeResult.

    Uses PM conviction vs analyst convictions when available (rich mode).
    Falls back to pre/post debate conviction shift.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        ITS value (positive = thesis challenged, negative = confirmed).
    """
    from components.thesis_shift import compute_its

    pm_conv = extract_pm_conviction(committee_result)
    analyst_convs = extract_analyst_convictions(committee_result)

    if pm_conv is not None and analyst_convs:
        return compute_its(
            pm_conviction=pm_conv,
            analyst_convictions=analyst_convs,
        )

    # Fallback: use timeline pre/post shift
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
    p_pre = [s / 10.0 for s in pre_scores]
    p_post = [s / 10.0 for s in post_scores]
    return compute_its(p_pre=p_pre, p_post=p_post)


def extract_pm_conviction(committee_result: object) -> float | None:
    """Extract PM's final conviction from CommitteeResult.

    Looks for PM Decision phase in the conviction timeline.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        PM conviction on 0-1 scale, or None if not available.
    """
    timeline = getattr(committee_result, "conviction_timeline", [])

    for snapshot in timeline:
        phase = getattr(snapshot, "phase", "")
        agent = getattr(snapshot, "agent", "")
        score = getattr(snapshot, "score", None)

        if score is not None and ("PM Decision" in phase or "pm" in agent.lower()):
            return score / 10.0  # Normalize from MAIC 0-10 to 0-1

    return None


def extract_analyst_convictions(committee_result: object) -> list[float]:
    """Extract analyst convictions from CommitteeResult.

    Collects post-debate conviction scores from non-PM agents.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        List of analyst convictions on 0-1 scale.
    """
    timeline = getattr(committee_result, "conviction_timeline", [])
    convictions = []

    for snapshot in timeline:
        phase = getattr(snapshot, "phase", "")
        agent = getattr(snapshot, "agent", "")
        score = getattr(snapshot, "score", None)

        if score is None:
            continue

        # Post-debate scores from non-PM agents
        is_post = "Post-Debate" in phase
        is_pm = "pm" in agent.lower() or "PM Decision" in phase

        if is_post and not is_pm:
            convictions.append(score / 10.0)

    return convictions


def extract_position_type(committee_result: object) -> str | None:
    """Extract position type from CommitteeResult.

    Maps MAIC's recommendation/direction to CGF position types:
    alpha_long, core_long, alpha_short, hedge_short.

    Args:
        committee_result: A MAIC CommitteeResult object.

    Returns:
        Position type string or None.
    """
    recommendation = getattr(committee_result, "recommendation", None)
    if recommendation is None:
        return None

    rec_lower = str(recommendation).lower()

    if "short" in rec_lower:
        # Distinguish alpha vs hedge short
        rationale = getattr(committee_result, "rationale", "")
        if "hedge" in str(rationale).lower() or "risk" in rec_lower:
            return "hedge_short"
        return "alpha_short"

    if "long" in rec_lower or "buy" in rec_lower:
        conviction_score = getattr(committee_result, "conviction_score", None)
        if conviction_score is not None and conviction_score >= 7:
            return "alpha_long"
        return "core_long"

    return None


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


# Backward compatibility alias
extract_ads = extract_its
