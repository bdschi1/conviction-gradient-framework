"""Bridge to backtest-lab — use convictions as signals and leverage regime detection.

Provides:
- ConvictionSignal: implements backtest-lab's Signal ABC so CGF convictions
  can be backtested through the backtest-lab engine.
- RegimeBridge: wraps backtest-lab's VolatilityRegimeDetector for RRS computation.

Graceful fallback if backtest-lab is not installed.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl

from config.settings import settings

logger = logging.getLogger(__name__)

_BACKTEST_AVAILABLE = False


def _ensure_backtest_lab() -> bool:
    """Try to add backtest-lab to sys.path."""
    global _BACKTEST_AVAILABLE

    if _BACKTEST_AVAILABLE:
        return True

    bt_path = settings.backtest_lab_path
    if bt_path is None:
        return False

    path_str = str(bt_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    try:
        from signals.base import Signal  # noqa: F401

        _BACKTEST_AVAILABLE = True
        logger.info("backtest-lab bridge connected: %s", bt_path)
        return True
    except ImportError:
        logger.info("backtest-lab not importable from %s", bt_path)
        return False


def is_available() -> bool:
    """Check if backtest-lab bridge is available."""
    return _ensure_backtest_lab()


def get_conviction_signal(
    conviction_data: dict[date, dict[str, float]] | None = None,
    signal_file: str | Path | None = None,
):
    """Create a ConvictionSignal that implements backtest-lab's Signal ABC.

    Args:
        conviction_data: {date: {ticker: conviction}} mapping.
        signal_file: Path to JSON file with conviction data.

    Returns:
        A Signal instance, or None if backtest-lab is not available.
    """
    if not _ensure_backtest_lab():
        logger.warning("backtest-lab not available; cannot create ConvictionSignal")
        return None

    from signals.base import Signal

    class ConvictionSignal(Signal):
        """Use CGF conviction scores as backtest-lab signals."""

        def __init__(
            self,
            data: dict[date, dict[str, float]] | None = None,
            file_path: str | Path | None = None,
        ):
            self._signals: dict[date, dict[str, float]] = data or {}
            if file_path:
                self._load_from_file(file_path)

        def _load_from_file(self, path: str | Path) -> None:
            path = Path(path)
            if not path.exists():
                logger.warning("Conviction file not found: %s", path)
                return

            with open(path) as f:
                raw = json.load(f)

            for entry in raw:
                d = date.fromisoformat(entry["date"])
                ticker = entry["ticker"]
                conviction = entry["conviction"]
                if d not in self._signals:
                    self._signals[d] = {}
                # Normalize conviction to [-1, 1] for Signal interface
                self._signals[d][ticker] = max(-1.0, min(conviction / 5.0, 1.0))

        @property
        def name(self) -> str:
            return "CGF-Conviction"

        @property
        def lookback_days(self) -> int:
            return 1

        def generate_signals(
            self, prices: pl.DataFrame, current_date: date,
        ) -> dict[str, float]:
            return self._signals.get(current_date, {})

    return ConvictionSignal(data=conviction_data, file_path=signal_file)


def get_regime_bridge():
    """Create a RegimeBridge that wraps backtest-lab's VolatilityRegimeDetector.

    Returns:
        RegimeBridge instance, or None if backtest-lab is not available.
    """
    if not _ensure_backtest_lab():
        logger.warning("backtest-lab not available; cannot create RegimeBridge")
        return None

    from regime.detector import RegimeState, VolatilityRegimeDetector

    class RegimeBridge:
        """Wraps backtest-lab regime detection for RRS computation."""

        def __init__(self, **kwargs):
            self._detector = VolatilityRegimeDetector(**kwargs)

        def update(self, daily_returns: list[float]) -> RegimeState:
            """Update regime classification with latest returns."""
            return self._detector.update(daily_returns)

        @property
        def current_regime(self):
            return self._detector.current_regime

    return RegimeBridge()
