# Conviction Gradient Framework

A system that turns investment conviction into a quantitative state variable, updated over time using a gradient-descent-style rule. It maps conviction directly into position sizes under portfolio-level constraints, while integrating IC governance and analyst/AI evaluation.

In plain English: instead of treating conviction as a vague label ("high conviction" or "low conviction"), CGF computes a numeric score per position based on how much the thesis is holding up against new data. When the thesis deteriorates — missed forecasts, structural breaks, adversarial debate — conviction adjusts down. When it holds or improves, conviction adjusts up. That score drives position sizing.

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

Computes and updates per-position conviction scores using four signals — forecast error, fundamental thesis violations, volatility regime shifts, and IC debate outcomes.

### Position Sizing Overlay

Maps conviction into portfolio weights, then applies gross/net, sector, and concentration constraints via constrained optimization.

### IC Governance

Structures investment committee debate into quantitative signals (pre/post anonymous probability submissions, adversarial debate shifts).

### Evaluation

Scores analysts and AI agents on calibration (Brier score), update alignment, and error attribution.

### Core Equations

**Conviction update** — gradient descent with momentum, clipped to $[-C_{\max},\, C_{\max}]$:

$$C_{t+1} = C_t \;-\; \alpha_t \, \nabla L_t \;+\; \beta \left( C_t - C_{t-1} \right)$$

**Total thesis loss** — weighted combination of four stress signals:

$$L_t = w_1 \, FE_t^{\,2} \;+\; w_2 \, FVS_t \;+\; w_3 \, RRS_t \;+\; w_4 \, ADS_t$$

**Loss gradient** — direction and magnitude of thesis deterioration:

$$\nabla L_t = \lambda_1 \, FE_t \;+\; \lambda_2 \, FVS_t \;+\; \lambda_3 \, RRS_t \;+\; \lambda_4 \, ADS_t$$

**Adaptive learning rate** — scales conviction elasticity to information quality:

$$\alpha_t = \frac{\kappa}{1 + \tau} \;\cdot\; \frac{\sigma_{\text{idio}}}{\sigma_{\text{expected}}} \;\cdot\; \frac{1}{1 + s} \qquad \alpha_t \in [\alpha_{\min},\, \alpha_{\max}]$$

where $\tau$ is information half-life and $s$ is analyst track-record score.

**Loss components:**

| Component | Formula | Interpretation |
|-----------|---------|----------------|
| Forecast Error | $FE_t = \dfrac{R_t - E[R_t]}{\sigma_{\text{expected}}}$ | Standardized return miss |
| Fundamental Violation | $FVS_t = \max(\text{severity}_i) + \min\!\left(0.2,\; 0.05 \cdot n_{\text{events}}\right)$ | Thesis-breaking events (0–1) |
| Risk Regime Shift | $RRS_t = \dfrac{\sigma_t^{\text{idio}} - \sigma_{t-1}^{\text{idio}}}{\sigma_{t-1}^{\text{idio}}} + \dfrac{IV_t - HV_t}{HV_t}$ | Vol regime change + IV-HV spread |
| Adversarial Debate Shift | $ADS_t = \bar{p}_{\text{post}} - \bar{p}_{\text{pre}}$ | IC probability shift during debate |

In plain terms: conviction moves toward better estimates at a pace that depends on how noisy the environment is and how trusted the analyst is. If volatility spikes, conviction responds more quickly. If the analyst has a strong track record, the system gives their existing view more inertia.

### Integration

When installed alongside other Tier_1 repos, CGF auto-detects and bridges to:
- **financial-data-providers**: market data (returns, vol)
- **multi-agent-investment-committee**: IC outputs feed ADS and FVS components
- **backtest-lab**: conviction trajectories usable as backtest signals
- **ls-portfolio-lab**: conviction-derived weights feed into portfolio construction

---

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

---

## Testing

```bash
pytest tests/ -v    # Full suite
make lint           # Ruff linter
make fmt            # Auto-format
```

## Contributing

Under active development. Contributions welcome — areas for improvement include loss components, governance workflows, evaluation metrics, and cross-repo bridge integrations.

## License

MIT

---

***Curiosity compounds. Rigor endures.***
