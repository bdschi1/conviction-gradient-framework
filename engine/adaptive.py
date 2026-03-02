"""Adaptive loss weight estimation.

Reference: Tallman & West, JRSS-B 2024 (Bayesian Predictive Decision Synthesis).

Tracks each loss component's directional prediction accuracy over a lookback
window, then adjusts weights via EMA to emphasize components that have been
useful for predicting subsequent return direction.

When a component's value (e.g., high RRS) correctly predicted the direction
of the next-period return (e.g., negative return following vol spike),
that component's usefulness score increases.

Weights are floored at a minimum to prevent any component from being zeroed out,
then renormalized to sum to 1.0.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

COMPONENT_KEYS = ("fe", "fvs", "rrs", "ads")
WEIGHT_MAP = {"fe": "w1", "fvs": "w2", "rrs": "w3", "ads": "w4"}


def compute_component_usefulness(
    component_history: list[dict[str, float]],
    subsequent_returns: list[float],
) -> dict[str, float]:
    """Score each component's directional prediction accuracy.

    A component is "useful" if its sign/magnitude predicted the direction
    of subsequent returns. FE and RRS are directional (positive = trouble);
    FVS is severity (higher = more damaging); ADS is shift direction.

    Args:
        component_history: List of dicts with keys fe/fvs/rrs/ads.
        subsequent_returns: List of realized returns following each observation.

    Returns:
        Dict mapping component name to usefulness score in [0, 1].
    """
    n = min(len(component_history), len(subsequent_returns))
    if n == 0:
        return {k: 0.5 for k in COMPONENT_KEYS}

    scores: dict[str, float] = {}
    for key in COMPONENT_KEYS:
        correct = 0
        total = 0
        for i in range(n):
            comp_val = component_history[i].get(key, 0.0)
            ret = subsequent_returns[i]
            if abs(comp_val) < 1e-10:
                continue
            total += 1
            # For FE/RRS: positive component → expect negative return
            # For FVS: higher severity → expect negative return
            # For ADS: positive shift → expect positive return (conviction confirmed)
            if key == "ads":
                if (comp_val > 0 and ret > 0) or (comp_val < 0 and ret < 0):
                    correct += 1
            else:
                if (comp_val > 0 and ret < 0) or (comp_val < 0 and ret > 0):
                    correct += 1

        scores[key] = correct / total if total > 0 else 0.5

    return scores


def update_adaptive_weights(
    current_weights: dict[str, float],
    usefulness: dict[str, float],
    decay: float = 0.97,
    floor: float = 0.05,
) -> dict[str, float]:
    """Update loss weights based on component usefulness scores.

    Uses EMA update: new_w = decay * old_w + (1 - decay) * usefulness.
    Floors prevent any component from being completely zeroed.
    Weights are renormalized to sum to 1.0.

    Args:
        current_weights: Current weights as dict (w1/w2/w3/w4).
        usefulness: Per-component usefulness scores.
        decay: EMA decay factor (higher = slower adaptation).
        floor: Minimum weight for any component.

    Returns:
        Updated weights dict (w1/w2/w3/w4), summing to 1.0.
    """
    updated = {}
    for comp_key in COMPONENT_KEYS:
        w_key = WEIGHT_MAP[comp_key]
        old_w = current_weights.get(w_key, 0.25)
        use = usefulness.get(comp_key, 0.5)
        new_w = decay * old_w + (1 - decay) * use
        updated[w_key] = max(new_w, floor)

    # Renormalize
    total = sum(updated.values())
    if total > 0:
        updated = {k: v / total for k, v in updated.items()}

    return updated


@dataclass
class AdaptiveWeightTracker:
    """Tracks component values and returns for adaptive weight estimation.

    Maintains per-instrument history of component values and subsequent
    returns, computes usefulness scores over a lookback window, and
    produces weight overrides.
    """

    lookback: int = 63
    decay: float = 0.97
    floor: float = 0.05
    _history: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _returns: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(
        self,
        instrument_id: str,
        components: dict[str, float],
        realized_return: float,
    ) -> None:
        """Record component values and realized return for an instrument.

        Args:
            instrument_id: Instrument identifier.
            components: Dict with keys fe/fvs/rrs/ads.
            realized_return: Realized return for this period.
        """
        self._history[instrument_id].append(components)
        self._returns[instrument_id].append(realized_return)

        # Trim to lookback
        if len(self._history[instrument_id]) > self.lookback:
            self._history[instrument_id] = self._history[instrument_id][-self.lookback :]
            self._returns[instrument_id] = self._returns[instrument_id][-self.lookback :]

        # Update weights if enough history
        if len(self._history[instrument_id]) >= self.lookback:
            # Use history[:-1] as predictors, returns[1:] as outcomes
            h = self._history[instrument_id][:-1]
            r = self._returns[instrument_id][1:]
            usefulness = compute_component_usefulness(h, r)

            current = self._weights.get(
                instrument_id, {"w1": 0.30, "w2": 0.25, "w3": 0.25, "w4": 0.20}
            )
            self._weights[instrument_id] = update_adaptive_weights(
                current, usefulness, self.decay, self.floor
            )

    def get_weights(self, instrument_id: str) -> dict[str, float] | None:
        """Get adaptive weights for an instrument.

        Returns None if insufficient history (< lookback observations).

        Args:
            instrument_id: Instrument identifier.

        Returns:
            Weight dict or None.
        """
        return self._weights.get(instrument_id)
