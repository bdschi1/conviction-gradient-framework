"""Bridge to ls-portfolio-lab — build portfolios from convictions.

Converts CGF conviction-derived weights into ls-portfolio-lab Portfolio
objects and leverages its rebalancer for constraint application.

Graceful fallback if ls-portfolio-lab is not installed.
"""

from __future__ import annotations

import logging
import sys

from config.settings import settings

logger = logging.getLogger(__name__)

_PORTFOLIO_LAB_AVAILABLE = False


def _ensure_portfolio_lab() -> bool:
    """Try to add ls-portfolio-lab to sys.path."""
    global _PORTFOLIO_LAB_AVAILABLE

    if _PORTFOLIO_LAB_AVAILABLE:
        return True

    pl_path = settings.portfolio_lab_path
    if pl_path is None:
        return False

    path_str = str(pl_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    try:
        from core.portfolio import Portfolio, Position  # noqa: F401

        _PORTFOLIO_LAB_AVAILABLE = True
        logger.info("ls-portfolio-lab bridge connected: %s", pl_path)
        return True
    except ImportError:
        logger.info("ls-portfolio-lab not importable from %s", pl_path)
        return False


def is_available() -> bool:
    """Check if ls-portfolio-lab bridge is available."""
    return _ensure_portfolio_lab()


def conviction_to_portfolio(
    weights: dict[str, float],
    prices: dict[str, float],
    nav: float = 3_000_000_000.0,
):
    """Build an ls-portfolio-lab Portfolio from conviction-derived weights.

    Args:
        weights: {ticker: signed_weight} from sizing overlay.
        prices: {ticker: current_price} for share calculation.
        nav: Portfolio NAV.

    Returns:
        Portfolio object, or None if ls-portfolio-lab is not available.
    """
    if not _ensure_portfolio_lab():
        logger.warning("ls-portfolio-lab not available")
        return None

    from core.portfolio import Portfolio, Position

    positions = []
    for ticker, weight in weights.items():
        if abs(weight) < 1e-6:
            continue

        price = prices.get(ticker, 100.0)
        notional = abs(weight) * nav
        shares = notional / price if price > 0 else 0

        positions.append(Position(
            ticker=ticker,
            side="LONG" if weight > 0 else "SHORT",
            shares=shares,
            entry_price=price,
            current_price=price,
        ))

    total_invested = sum(p.notional for p in positions)
    cash = nav - total_invested

    return Portfolio(positions=positions, nav=nav, cash=cash)


def apply_rebalance(portfolio, request_params: dict | None = None):
    """Apply ls-portfolio-lab rebalancer to a portfolio.

    Args:
        portfolio: ls-portfolio-lab Portfolio object.
        request_params: Dict of RebalanceRequest parameters.

    Returns:
        RebalanceResult, or None if not available.
    """
    if not _ensure_portfolio_lab():
        return None

    try:
        from core.rebalancer import RebalanceRequest  # noqa: F401

        request = RebalanceRequest(**(request_params or {}))
        # Would need returns_df, betas, current_prices for full rebalance
        # This is a bridge interface — caller provides the data
        logger.info("Rebalance bridge ready; caller must provide market data")
        return request
    except ImportError:
        logger.warning("ls-portfolio-lab rebalancer not importable")
        return None
