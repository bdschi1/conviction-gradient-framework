# Conviction Gradient Framework

A system that turns investment conviction into a quantitative state variable, updated over time using a gradient-descent-style rule. It maps conviction directly into position sizes under portfolio-level constraints, while integrating IC governance and analyst/AI evaluation.

In plain English: instead of treating conviction as a vague label ("high conviction" or "low conviction"), CGF computes a numeric score per position based on how much the thesis is holding up against new data. When the thesis deteriorates — missed forecasts, structural breaks, vol regime shifts — conviction adjusts down. When it holds or improves, conviction adjusts up. That score drives position sizing.

This is a framework, not a trading system. It doesn't claim to generate alpha from public data. It formalizes a process that every PM runs intuitively — adjusting conviction when new information arrives — and makes it testable, auditable, and systematic. The components that matter most (IC debate signals, proprietary event data) require a real analyst team and live capital context to be meaningful.

**Key questions this project answers:**
- *How should conviction change when new data arrives?*
- *How do you map conviction into portfolio weights under risk constraints?*

---

## Quick Start

```bash
cd conviction-gradient-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

Optional integrations:
```bash
pip install -e ".[api]"        # FastAPI endpoints
pip install -e ".[bloomberg]"  # Bloomberg data provider
pip install -e ".[ibkr]"       # Interactive Brokers
```

### CLI
```bash
cgf update data.json          # Run conviction update from data file
cgf trajectory AAPL           # Print conviction trajectory
cgf health                    # Database stats
```

### Validation
```bash
cgf validate                             # Synthetic data (noise-only)
cgf validate --signal                    # Signal-embedded synthetic data
cgf validate --real                      # Real market data via yfinance
cgf validate --real --tickers "META,INTC,BA" --start 2022-01-01 --end 2022-12-31
```

### API
```bash
make api                      # Start FastAPI at localhost:8000
```

### Python
```python
from engine.updater import run_single_update
from engine.models import ConvictionState, InstrumentData

current = ConvictionState(instrument_id="AAPL", as_of_date=date(2024, 5, 31), conviction=2.0)
data = InstrumentData(
    instrument_id="AAPL", as_of_date=date(2024, 6, 1),
    realized_return=0.05, expected_return=0.08,
    sigma_expected=0.20, sigma_idio_current=0.18, sigma_idio_prev=0.15,
)
updated = run_single_update(current, data)
```

---

## How It Works

### Conviction Engine

Computes and updates per-position conviction scores using four signals — forecast error, fundamental thesis violations, volatility regime shifts, and independent thesis shift (PM-analyst conviction delta from IC debate).

### Position Sizing

Maps conviction into portfolio weights using one of five methods (basic, vol-adjusted, Kelly, risk parity, tiered), then applies gross/net, sector, and concentration constraints via SLSQP optimization. In short: conviction scores go in, position sizes come out, subject to risk limits.

### IC Governance

Structures investment committee thesis testing into quantitative signals. Analysts independently test the thesis; the delta between PM conviction and analyst consensus becomes the ITS signal. Pre/post probability submissions are collected and scored.

### Evaluation

Scores analysts and AI agents on calibration (Brier score), update alignment (conviction changes vs. realized P&L), and error attribution (which loss component drove the miss). Measures whether conviction changes were directionally correct and where forecasts broke down.

### Validation

Three validation modes test the framework against progressively harder benchmarks:

| Mode | Data Source | What It Tests |
|------|------------|---------------|
| Synthetic | GBM / OU / regime-switching noise | Math correctness, no data bugs |
| Signal-embedded | Synthetic with injected signals and known lead times | Whether each component adds marginal Sharpe |
| Real data | yfinance prices, earnings surprises, VIX | End-to-end on actual market data |

Each mode runs ablation: full model vs. model-with-one-component-removed. The marginal Sharpe difference quantifies each component's contribution. All three modes compare against equal-weight, buy-and-hold, and momentum baselines.

On real market data, vol regime detection (RRS via VIX) is the only component that shows positive marginal Sharpe with public data alone. Forecast error hurts at daily frequency with naive trailing-mean estimates. FVS and ITS are neutral — FVS because large-cap earnings surprises are rare, ITS because IC debate data is proprietary. The validation is transparent about what works and what doesn't.

### Core Equations

**Conviction update** — gradient descent with momentum, clipped to $[-C_{\max},\, C_{\max}]$:

$$C_{t+1} = C_t \;-\; \alpha_t \, \nabla L_t \;+\; \beta \left( C_t - C_{t-1} \right)$$

**Total thesis loss** — weighted combination of four stress signals:

$$L_t = w_1 \, FE_t^{\,2} \;+\; w_2 \, FVS_t \;+\; w_3 \, RRS_t \;+\; w_4 \, ITS_t$$

**Loss gradient** — direction and magnitude of thesis deterioration:

$$\nabla L_t = \lambda_1 \, FE_t \;+\; \lambda_2 \, FVS_t \;+\; \lambda_3 \, RRS_t \;+\; \lambda_4 \, ITS_t$$

**Adaptive learning rate** — scales conviction elasticity to information quality:

$$\alpha_t = \frac{\kappa}{1 + \tau} \;\cdot\; \frac{\sigma_{\text{idio}}}{\sigma_{\text{expected}}} \;\cdot\; \frac{1}{1 + s} \qquad \alpha_t \in [\alpha_{\min},\, \alpha_{\max}]$$

where $\tau$ is information half-life and $s$ is analyst track-record score.

**Loss components:**

**Forecast Error** — standardized return miss:

$$FE_t = \frac{R_t - E[R_t]}{\sigma_{\text{expected}}}$$

**Fundamental Violation Score** — thesis-breaking events (0–1):

$$FVS_t = \max(\text{severity}_i) + \min\!\left(0.2,\; 0.05 \cdot n_{\text{events}}\right)$$

**Risk Regime Shift** — vol regime change + IV-HV spread:

$$RRS_t = \frac{\sigma_t^{\text{idio}} - \sigma_{t-1}^{\text{idio}}}{\sigma_{t-1}^{\text{idio}}} + \frac{IV_t - HV_t}{HV_t}$$

**Independent Thesis Shift** — PM-analyst conviction delta from thesis testing:

$$ITS_t = \frac{C_{PM} - \bar{C}_{\text{analysts}}}{\text{scale}}$$

In plain terms: conviction adjusts at a pace that depends on how noisy the environment is and how trusted the analyst is. High vol → faster adjustment. Strong track record → more inertia.

### What This Isn't

CGF is not a backtest showing unrealistic Sharpe ratios. The public-data validation intentionally uses naive forecasts (trailing mean) and limited event data (earnings surprises only) — it's a proof of concept for the infrastructure, not an alpha claim. The components designed to carry signal (ITS from IC debate, FVS from proprietary event feeds) require the context they were designed for: a PM running capital with an analyst team.

### Integration

When installed alongside other Tier_1 repos, CGF auto-detects and bridges to:
- **financial-data-providers**: market data (returns, vol)
- **multi-agent-investment-committee**: IC outputs feed ITS and FVS components
- **backtest-lab**: conviction trajectories usable as backtest signals
- **ls-portfolio-lab**: conviction-derived weights feed into portfolio construction

---

## Architecture

```
engine/       Core math: loss, gradient, update rule, stability checks
components/   Loss component implementations (FE, FVS, RRS, ITS)
sizing/       Conviction-to-weight mapping + portfolio constraints
governance/   IC session management, debate workflow, process policies
evaluation/   Calibration, update alignment, error attribution, AI benchmark
validation/   Synthetic, signal-embedded, and real data validation harness
bridges/      Integration with MAIC, backtest-lab, ls-portfolio-lab, data providers
storage/      SQLite persistence for conviction states, events, IC sessions
config/       Pydantic settings and default hyperparameters
api/          FastAPI endpoints
```

---

## Testing

293 tests across 13 test files. Network-dependent tests (real data via yfinance) are auto-skipped when offline.

```bash
pytest tests/ -v    # Full suite
ruff check .        # Lint
ruff format .       # Auto-format
```

## Contributing

Under active development. Contributions welcome — areas for improvement are a lot -  including loss components, governance workflows, evaluation metrics, and cross-repo bridge integrations.

## License

MIT

---

***Curiosity compounds. Rigor endures.***
