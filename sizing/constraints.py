"""Portfolio constraints — enforce gross/net, sector, and concentration limits.

Uses scipy SLSQP to find the minimum-turnover weight vector that satisfies
all constraints, following the same pattern as ls-portfolio-lab/core/rebalancer.py.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioConstraints(BaseModel):
    """Configurable portfolio constraint parameters."""

    max_gross_exposure: float = Field(default=2.0, gt=0, description="Max gross (sum|w|)")
    max_net_exposure: float = Field(default=0.5, ge=0, description="Max |sum(w)|")
    max_position_pct: float = Field(default=0.05, gt=0, le=1, description="Max single position")
    min_position_pct: float = Field(default=0.005, ge=0, le=1, description="Min position (floor)")
    max_sector_net: float = Field(default=0.25, ge=0, le=1, description="Max sector net exposure")


class ConstrainedResult(BaseModel):
    """Result of constraint application."""

    weights: dict[str, float] = Field(description="Constrained weights")
    converged: bool = Field(description="Whether optimizer converged")
    violations: list[str] = Field(default_factory=list, description="Constraint violations")
    turnover: float = Field(default=0.0, description="Total turnover from raw to constrained")


def apply_constraints(
    raw_weights: dict[str, float],
    constraints: PortfolioConstraints | None = None,
    sectors: dict[str, str] | None = None,
) -> ConstrainedResult:
    """Apply portfolio constraints to raw conviction-derived weights.

    Minimizes turnover (squared deviation from raw weights) subject to
    gross, net, position, and sector constraints.

    Args:
        raw_weights: Unconstrained weights from conviction mapping.
        constraints: Constraint parameters.
        sectors: {instrument_id: sector} mapping for sector constraints.

    Returns:
        ConstrainedResult with adjusted weights.
    """
    cons = constraints or PortfolioConstraints()
    sectors = sectors or {}
    violations: list[str] = []

    if not raw_weights:
        return ConstrainedResult(weights={}, converged=True)

    tickers = list(raw_weights.keys())
    n = len(tickers)
    x0 = np.array([raw_weights[t] for t in tickers])

    # Determine sign (long/short) from raw weights
    signs = np.sign(x0)
    signs[signs == 0] = 1  # treat zero as potential long

    # Work with absolute weights for optimization
    abs_x0 = np.abs(x0)
    abs_x0 = np.clip(abs_x0, cons.min_position_pct, cons.max_position_pct)

    # Objective: minimize turnover from raw weights
    raw_abs = np.abs(x0)

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - raw_abs) ** 2))

    # Constraints
    opt_constraints: list[dict] = []

    # Gross exposure: sum(|w|) <= max_gross
    opt_constraints.append({
        "type": "ineq",
        "fun": lambda x: cons.max_gross_exposure - np.sum(x),
    })

    # Net exposure: |sum(sign * w)| <= max_net
    def net_constraint(x: np.ndarray) -> float:
        net = float(np.abs(np.dot(signs, x)))
        return cons.max_net_exposure - net

    opt_constraints.append({"type": "ineq", "fun": net_constraint})

    # Sector net constraints
    if sectors:
        unique_sectors = set(sectors.values())
        for sector in unique_sectors:
            mask = np.array([
                1.0 if sectors.get(t, "Unknown") == sector else 0.0
                for t in tickers
            ])

            def make_sector_fn(m: np.ndarray = mask):
                def fn(x: np.ndarray) -> float:
                    sector_net = float(np.abs(np.dot(signs * x, m)))
                    return cons.max_sector_net - sector_net
                return fn

            opt_constraints.append({"type": "ineq", "fun": make_sector_fn()})

    # Bounds: each position within [min, max]
    bounds = [(cons.min_position_pct, cons.max_position_pct)] * n

    result = minimize(
        objective,
        abs_x0,
        method="SLSQP",
        bounds=bounds,
        constraints=opt_constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    converged = result.success
    if not converged:
        violations.append(f"Optimizer did not converge: {result.message}")

    # Reconstruct signed weights
    final_abs = result.x
    final_weights = {
        tickers[i]: float(signs[i] * final_abs[i])
        for i in range(n)
    }

    turnover = float(np.sum(np.abs(final_abs - raw_abs)))

    return ConstrainedResult(
        weights=final_weights,
        converged=converged,
        violations=violations,
        turnover=turnover,
    )
