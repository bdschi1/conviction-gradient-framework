# CGF Simulation Example

## Setup

```bash
cd conviction-gradient-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running a conviction update

Prepare a JSON file with instrument data:

```json
[
  {
    "instrument_id": "AAPL",
    "as_of_date": "2024-06-15",
    "realized_return": 0.05,
    "expected_return": 0.08,
    "sigma_expected": 0.20,
    "sigma_idio_current": 0.18,
    "sigma_idio_prev": 0.15,
    "implied_vol": 0.22,
    "historical_vol": 0.18,
    "info_half_life": 1.0,
    "track_record_score": 0.5,
    "fvs_events": [],
    "p_pre": [0.7, 0.65, 0.8],
    "p_post": [0.6, 0.55, 0.7]
  }
]
```

Run the update:

```bash
cgf update data.json
cgf trajectory AAPL
cgf health
```

## Interpreting results

- **conviction**: Risk-adjusted expected return signal. Positive = long, negative = short.
- **alpha_t**: Learning rate used. Lower = more stable conviction; higher = more responsive.
- **loss components**: FE (forecast error), FVS (thesis violation), RRS (vol shift), ADS (debate shift).
- **gradient**: Direction and magnitude of thesis deterioration/improvement.
