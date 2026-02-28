# Conviction Gradient Framework

A system that turns investment conviction into a quantitative state variable, updated over time using a gradient-descent-style rule. It maps conviction directly into position sizes under portfolio-level constraints, while integrating IC governance and analyst/AI evaluation.

In plain English: instead of treating conviction as a vague label ("high conviction" or "low conviction"), CGF computes a numeric score per position based on how much the thesis is holding up against new data. When the thesis deteriorates — missed forecasts, structural breaks, adversarial debate — conviction adjusts down. When it holds or improves, conviction adjusts up. That score drives position sizing.

## What it does

- **Conviction engine**: Computes and updates per-position conviction scores using four signals — forecast error, fundamental thesis violations, volatility regime shifts, and IC debate outcomes.
- **Position sizing overlay**: Maps conviction into portfolio weights, then applies gross/net, sector, and concentration constraints via constrained optimization.
- **IC governance**: Structures investment committee debate into quantitative signals (pre/post anonymous probability submissions, adversarial debate shifts).
- **Evaluation**: Scores analysts and AI agents on calibration (Brier score), update alignment, and error attribution.

## Setup

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

## Usage

### CLI
```bash
cgf update data.json          # Run conviction update from data file
cgf trajectory AAPL           # Print conviction trajectory
cgf health                    # Database stats
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

## Architecture

```
engine/       Core math: loss, gradient, update rule, stability checks
components/   Loss component implementations (FE, FVS, RRS, ADS)
sizing/       Conviction-to-weight mapping + portfolio constraints
governance/   IC session management, debate workflow, process policies
evaluation/   Calibration, update alignment, error attribution, AI benchmark
bridges/      Integration with MAIC, backtest-lab, ls-portfolio-lab, data providers
storage/      SQLite persistence for conviction states, events, IC sessions
config/       Pydantic settings and default hyperparameters
api/          FastAPI endpoints
```

## Core equations

The conviction update follows a gradient-descent-with-momentum rule:

**C_{t+1} = C_t - alpha_t * gradient_L + beta * (C_t - C_{t-1})**

Where:
- **C_t**: conviction at time t (risk-adjusted expected return: E[R] / sigma_idio)
- **alpha_t**: adaptive learning rate (adjusts for vol regime, information half-life, analyst track record)
- **gradient_L**: weighted sum of forecast error, thesis violations, vol shifts, and debate shifts
- **beta**: momentum stabilizer to prevent oscillation

In plain terms: conviction moves toward better estimates at a pace that depends on how noisy the environment is and how trusted the analyst is. If volatility spikes, conviction responds more quickly. If the analyst has a strong track record, the system gives their existing view more inertia.

## Integration

When installed alongside other Tier_1 repos, CGF auto-detects and bridges to:
- **financial-data-providers**: market data (returns, vol)
- **multi-agent-investment-committee**: IC outputs feed ADS and FVS components
- **backtest-lab**: conviction trajectories usable as backtest signals
- **ls-portfolio-lab**: conviction-derived weights feed into portfolio construction

## Testing

```bash
pytest tests/ -v    # Full suite
make lint           # Ruff linter
make fmt            # Auto-format
```

***Curiosity compounds. Rigor endures.***
