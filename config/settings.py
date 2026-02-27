"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """CGF runtime configuration, loaded from .env or environment."""

    # Database
    db_path: str = "conviction.db"

    # API
    api_port: int = 8000

    # Data provider
    data_provider: str = "Yahoo Finance"

    # Update frequency
    update_frequency: str = "daily"  # daily | event_triggered | both

    # Bridge paths (auto-detected from Tier_1 layout if not set)
    maic_path: Path | None = None
    backtest_lab_path: Path | None = None
    portfolio_lab_path: Path | None = None

    model_config = {"env_prefix": "CGF_", "env_file": ".env"}

    def resolve_bridge_paths(self) -> None:
        """Auto-detect sibling repo paths if not explicitly set."""
        tier1 = Path(__file__).resolve().parent.parent.parent
        if self.maic_path is None:
            candidate = tier1 / "multi-agent-investment-committee"
            if candidate.exists():
                self.maic_path = candidate
        if self.backtest_lab_path is None:
            candidate = tier1 / "backtest-lab"
            if candidate.exists():
                self.backtest_lab_path = candidate
        if self.portfolio_lab_path is None:
            candidate = tier1 / "ls-portfolio-lab"
            if candidate.exists():
                self.portfolio_lab_path = candidate


settings = Settings()
settings.resolve_bridge_paths()
