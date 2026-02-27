"""AI training benchmark interface.

Provides input/output schemas and scoring for evaluating AI agents
against the CGF model. Agents receive market data and produce conviction
trajectories; scoring compares their outputs to CGF model predictions
and realized outcomes.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from evaluation.calibration import brier_score


class BenchmarkInput(BaseModel):
    """Input data provided to an AI agent for evaluation."""

    instrument_id: str
    prices: list[float] = Field(description="Historical price series")
    returns: list[float] = Field(description="Daily return series")
    fundamentals: dict = Field(default_factory=dict, description="Key financial metrics")
    news_events: list[str] = Field(default_factory=list, description="News/event descriptions")
    debate_summary: str = Field(default="", description="IC debate summary")


class BenchmarkOutput(BaseModel):
    """Expected output from an AI agent."""

    conviction_trajectory: list[float] = Field(description="Conviction over time")
    probability_forecasts: list[float] = Field(
        default_factory=list, description="Directional probability forecasts"
    )
    proposed_weights: list[float] = Field(
        default_factory=list, description="Proposed position weights"
    )


class BenchmarkScore(BaseModel):
    """Scoring result for one benchmark evaluation."""

    brier: float = Field(description="Brier score of probability forecasts")
    conviction_correlation: float = Field(
        description="Correlation of agent vs model conviction trajectory"
    )
    update_mse: float = Field(
        description="MSE between agent and model conviction updates"
    )
    direction_accuracy: float = Field(
        description="Fraction of time agent predicted correct direction"
    )
    overall_score: float = Field(description="Weighted composite score (lower is better)")


def score_agent(
    agent_output: BenchmarkOutput,
    model_trajectory: list[float],
    realized_returns: list[float],
) -> BenchmarkScore:
    """Score an AI agent's output against model predictions and reality.

    Args:
        agent_output: Agent's conviction trajectory and forecasts.
        model_trajectory: CGF model's conviction trajectory.
        realized_returns: Actual returns over the evaluation period.

    Returns:
        BenchmarkScore with per-metric and composite scores.
    """
    # Brier score (if probability forecasts provided)
    if agent_output.probability_forecasts and realized_returns:
        n = min(len(agent_output.probability_forecasts), len(realized_returns))
        outcomes = [1 if r > 0 else 0 for r in realized_returns[:n]]
        brier = brier_score(agent_output.probability_forecasts[:n], outcomes)
    else:
        brier = 1.0

    # Conviction trajectory correlation
    if agent_output.conviction_trajectory and model_trajectory:
        n = min(len(agent_output.conviction_trajectory), len(model_trajectory))
        a = np.array(agent_output.conviction_trajectory[:n])
        m = np.array(model_trajectory[:n])
        corr = float(np.corrcoef(a, m)[0, 1]) if np.std(a) > 0 and np.std(m) > 0 else 0.0
    else:
        corr = 0.0

    # Update MSE
    if len(agent_output.conviction_trajectory) > 1 and len(model_trajectory) > 1:
        n = min(len(agent_output.conviction_trajectory), len(model_trajectory))
        agent_deltas = np.diff(agent_output.conviction_trajectory[:n])
        model_deltas = np.diff(model_trajectory[:n])
        n_deltas = min(len(agent_deltas), len(model_deltas))
        update_mse = float(np.mean((agent_deltas[:n_deltas] - model_deltas[:n_deltas]) ** 2))
    else:
        update_mse = 1.0

    # Direction accuracy
    if agent_output.conviction_trajectory and realized_returns:
        n = min(len(agent_output.conviction_trajectory), len(realized_returns))
        correct = sum(
            1
            for c, r in zip(
                agent_output.conviction_trajectory[:n],
                realized_returns[:n],
                strict=False,
            )
            if (c > 0 and r > 0) or (c < 0 and r < 0) or (c == 0)
        )
        direction_acc = correct / n if n > 0 else 0.0
    else:
        direction_acc = 0.0

    # Composite (lower is better)
    overall = 0.3 * brier + 0.3 * (1 - corr) + 0.2 * update_mse + 0.2 * (1 - direction_acc)

    return BenchmarkScore(
        brier=brier,
        conviction_correlation=corr,
        update_mse=update_mse,
        direction_accuracy=direction_acc,
        overall_score=overall,
    )
