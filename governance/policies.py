"""Configurable IC process policies.

Policies define mandatory controls that the conviction engine enforces:
- Mandatory adversarial injection for key positions
- Structural reset triggers
- Oscillation guards
- Minimum data requirements for conviction updates

Policies are loaded from YAML for configurability.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PolicyViolation(BaseModel):
    """A detected policy violation."""

    policy_name: str
    instrument_id: str
    severity: str  # "warning" | "block" | "reset"
    message: str


class ProcessPolicy(BaseModel):
    """Configurable IC process rules."""

    mandatory_adversarial: bool = Field(
        default=True,
        description="Require adversarial debate for all key names",
    )
    fvs_reset_threshold: float = Field(
        default=0.7, ge=0, le=1,
        description="FVS level that triggers structural conviction reset",
    )
    vol_double_trigger: float = Field(
        default=2.0, gt=1,
        description="Vol ratio that triggers structural reset",
    )
    min_fe_std_trigger: float = Field(
        default=2.0, gt=0,
        description="FE standard deviations that trigger review",
    )
    oscillation_window_days: int = Field(
        default=5, ge=1,
        description="Days to detect conviction oscillation",
    )
    min_participants_for_ads: int = Field(
        default=2, ge=1,
        description="Minimum IC participants for ADS computation",
    )
    require_pre_post_probabilities: bool = Field(
        default=True,
        description="Block finalization without pre/post probabilities",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProcessPolicy:
        """Load policy from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("policy", data))


def enforce_policy(
    policy: ProcessPolicy,
    instrument_id: str,
    fvs: float = 0.0,
    fe: float = 0.0,
    vol_ratio: float = 1.0,
    has_adversarial_debate: bool = True,
    participant_count: int = 0,
) -> list[PolicyViolation]:
    """Check all policy rules and return any violations.

    Args:
        policy: Process policy to enforce.
        instrument_id: Position being checked.
        fvs: Current Fundamental Violation Score.
        fe: Current Forecast Error (standardized).
        vol_ratio: Current vol / baseline vol.
        has_adversarial_debate: Whether debate occurred.
        participant_count: Number of IC participants.

    Returns:
        List of policy violations (empty if compliant).
    """
    violations = []

    if policy.mandatory_adversarial and not has_adversarial_debate:
        violations.append(PolicyViolation(
            policy_name="mandatory_adversarial",
            instrument_id=instrument_id,
            severity="block",
            message="Adversarial debate required but not conducted",
        ))

    if fvs >= policy.fvs_reset_threshold:
        violations.append(PolicyViolation(
            policy_name="fvs_reset",
            instrument_id=instrument_id,
            severity="reset",
            message=f"FVS {fvs:.2f} >= threshold {policy.fvs_reset_threshold:.2f}",
        ))

    if vol_ratio >= policy.vol_double_trigger:
        violations.append(PolicyViolation(
            policy_name="vol_reset",
            instrument_id=instrument_id,
            severity="reset",
            message=f"Vol ratio {vol_ratio:.2f}x >= {policy.vol_double_trigger:.2f}x",
        ))

    if abs(fe) >= policy.min_fe_std_trigger:
        violations.append(PolicyViolation(
            policy_name="fe_review",
            instrument_id=instrument_id,
            severity="warning",
            message=f"|FE| {abs(fe):.2f} >= {policy.min_fe_std_trigger:.2f} std devs",
        ))

    if (
        policy.require_pre_post_probabilities
        and participant_count < policy.min_participants_for_ads
    ):
        violations.append(PolicyViolation(
            policy_name="insufficient_participants",
            instrument_id=instrument_id,
            severity="warning",
            message=(
                f"Only {participant_count} participants "
                f"(minimum {policy.min_participants_for_ads})"
            ),
        ))

    return violations
