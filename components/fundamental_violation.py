"""Fundamental Violation Score (FVS) — rule-based thesis breach scoring.

Combines discrete events (mgmt change, KPI miss, governance breach, etc.)
into a single 0-1 score representing how much the investment thesis has
been structurally violated.

Event types and base severity scores are loaded from a YAML taxonomy.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default taxonomy (used when no YAML file is provided)
DEFAULT_TAXONOMY: dict[str, dict[str, Any]] = {
    "mgmt_change": {
        "base_severity": 0.6,
        "description": "CEO/CFO departure or replacement",
    },
    "guidance_cut": {
        "base_severity": 0.5,
        "description": "Revenue or earnings guidance reduced",
    },
    "kpi_miss": {
        "base_severity": 0.4,
        "description": "Key performance indicator missed consensus",
    },
    "capital_allocation": {
        "base_severity": 0.5,
        "description": "Major M&A, buyback suspension, dividend cut",
    },
    "governance_breach": {
        "base_severity": 0.8,
        "description": "Board conflict, audit concern, regulatory action",
    },
    "business_model_change": {
        "base_severity": 0.9,
        "description": "Fundamental change to revenue model or strategy",
    },
    "accounting_restatement": {
        "base_severity": 0.7,
        "description": "Financial restatement or audit qualification",
    },
}


class FVSEvent(BaseModel):
    """A single fundamental violation event."""

    event_type: str = Field(description="Event type key from taxonomy")
    severity_override: float | None = Field(
        default=None, ge=0, le=1, description="Override base severity"
    )
    event_date: date = Field(description="When the event occurred")
    description: str = Field(default="", description="Free-text event description")


class FVSTaxonomy:
    """Event taxonomy mapping event types to base severity scores."""

    def __init__(self, event_types: dict[str, dict[str, Any]] | None = None):
        self._types = event_types or DEFAULT_TAXONOMY

    @classmethod
    def from_yaml(cls, path: str | Path) -> FVSTaxonomy:
        """Load taxonomy from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(event_types=data.get("event_types", data))

    def get_severity(self, event_type: str) -> float:
        """Get base severity for an event type."""
        entry = self._types.get(event_type)
        if entry is None:
            logger.warning("Unknown FVS event type: %s, using 0.5", event_type)
            return 0.5
        return entry["base_severity"]

    @property
    def event_types(self) -> list[str]:
        return list(self._types.keys())


def compute_fvs(
    events: list[FVSEvent],
    taxonomy: FVSTaxonomy | None = None,
) -> float:
    """Compute aggregate Fundamental Violation Score from events.

    Multiple events combine via max (worst violation dominates) plus
    a small additive penalty for event count to capture accumulation.

    Args:
        events: List of FVS events in the evaluation period.
        taxonomy: Event taxonomy. Uses default if not provided.

    Returns:
        Aggregate FVS in [0, 1].
    """
    if not events:
        return 0.0

    tax = taxonomy or FVSTaxonomy()

    severities = []
    for event in events:
        if event.severity_override is not None:
            severities.append(event.severity_override)
        else:
            severities.append(tax.get_severity(event.event_type))

    # Max severity as base, plus small penalty for multiple events
    max_sev = max(severities)
    count_penalty = min(0.1 * (len(severities) - 1), 0.2)  # cap at 0.2

    return min(max_sev + count_penalty, 1.0)
