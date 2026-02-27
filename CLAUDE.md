# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## What This Is
Gradient-descent conviction engine that updates per-position conviction scores using four loss components (forecast error, fundamental violations, vol regime shifts, IC debate shifts), maps conviction into constrained portfolio weights, and scores analyst/AI calibration and update behavior.

## Commands
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Lint
ruff check .
ruff format .

# CLI
cgf update data.json
cgf trajectory AAPL
cgf health

# API
pip install -e ".[api]"
make api

# UI (Streamlit)
pip install -e ".[ui]"
make ui
```

## Architecture
- `engine/` — Core conviction math: loss computation, gradient, adaptive learning rate, update rule, stability/clipping
- `components/` — Loss component implementations: FE (forecast error), FVS (fundamental violation, YAML taxonomy), RRS (risk regime shift), ADS (adversarial debate shift)
- `sizing/` — Conviction-to-weight mapping (basic, vol-adjusted), portfolio constraints (SLSQP), oscillation guard, structural reset
- `governance/` — IC session lifecycle, probability collection, debate records, configurable YAML process policies
- `evaluation/` — Brier score, calibration buckets, update alignment, error attribution decomposition, AI benchmark scoring
- `bridges/` — Integration to financial-data-providers, MAIC, backtest-lab, ls-portfolio-lab (all graceful fallback)
- `storage/` — SQLite persistence: conviction_states, events, ic_sessions, probability_submissions, track_records, target_weights
- `config/` — Pydantic BaseSettings from .env, ConvictionParams + SizingParams defaults
- `api/` — FastAPI endpoints: conviction/update, conviction/trajectory, ic/session, evaluation/analyst_summary, portfolio/target_weights
- `cli.py` — CLI entry point: update, trajectory, health commands

## Key Patterns
- Core update rule: C_{t+1} = C_t - alpha_t * gradient + beta * (C_t - C_{t-1}), clipped to [-C_max, C_max]
- All params (w1-4, lambda1-4, kappa, alpha bounds, beta, C_max) in ConvictionParams Pydantic model
- FVS uses YAML taxonomy for event type → severity mapping
- Bridges use sys.path injection + try/except for graceful degradation
- Polars for all dataframe operations. Pydantic for config validation. SQLite for persistence
- Provider abstraction via financial-data-providers (git dep, Yahoo default)

## Testing Conventions
- Tests in tests/ with synthetic/mock data only
- Bridge tests verify graceful degradation when target repos not on sys.path
- Storage tests use tmp_path fixture for temp SQLite
- Run with `pytest tests/ -v`
