<!-- conviction-gradient-framework/README.md | Last updated: 2026-06-13 -->

# Conviction Gradient Framework

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![tests](https://img.shields.io/badge/tests-293%20passing-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Turns investment conviction into a quantitative state variable, updated over time with a gradient-descent rule, and maps it into position sizes under portfolio-level risk constraints. Integrates IC governance and analyst/AI evaluation.

**Plain English:** Instead of treating conviction as a vague label, CGF scores each position numerically based on how well the thesis holds up against new data — missed forecasts and regime shifts push it down, confirmation pushes it up — and that score drives sizing.

This is a framework, not a trading system. It does not claim alpha from public data and has not been backtested in this form; it formalizes a process PMs already run intuitively so it can be tested and audited.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[api]"        # FastAPI endpoints (optional)
pip install -e ".[bloomberg]"  # or .[ibkr] for live data (optional)
```

## Usage

```
cgf update data.json       # run a conviction update from a data file
cgf trajectory AAPL        # print conviction trajectory
cgf validate --real --tickers "META,INTC,BA"   # validate against real market data
make api                   # FastAPI at localhost:8000
```

## What it does

- **Conviction engine** — updates per-position scores from four signals: forecast error, fundamental violations, volatility-regime shifts, and PM-vs-analyst thesis delta
- **Position sizing** — maps conviction to weights (five methods) under gross/net, sector, and concentration limits via SLSQP
- **Governance & evaluation** — structures IC thesis testing into signals; scores calibration (Brier), update alignment, and error attribution
- **Validation** — synthetic, signal-embedded, and real-data modes with ablation vs. equal-weight / buy-hold / momentum baselines. On public data only vol-regime detection shows positive marginal Sharpe; the components designed to carry signal need proprietary IC and event data

## Tests

```
pytest tests/ -v
ruff check .
```

## License

MIT
