"""Signal-embedded synthetic data generators for validation.

Unlike the noise-only generators in synthetic_data.py, these produce assets
with known, measurable signals in each of the 4 CGF loss components. Drawdown
events are injected at known times, and each signal channel provides advance
warning with known lead times.

This allows ablation to measure whether each component actually helps —
the justification question.

Asset universe:
- FE_SIGNAL:  Forecast accuracy 60%, FE channel only
- FVS_SIGNAL: FVS events 5 days before drawdowns
- RRS_SIGNAL: IV leads HV by 3 days around drawdowns
- ADS_SIGNAL: Debate shifts before drawdowns (contrarian)
- ALL_SIGNAL: All 4 channels active, moderate signal strength
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalEmbeddedAsset:
    """Synthetic asset with embedded signals for all 4 CGF components."""

    returns: np.ndarray
    name: str
    asset_type: str
    forecasts: np.ndarray
    fvs_schedule: list[dict] = field(default_factory=list)
    implied_vols: np.ndarray = field(default_factory=lambda: np.array([]))
    regimes: np.ndarray = field(default_factory=lambda: np.array([]))
    debate_schedule: list[dict] = field(default_factory=list)
    track_record_score: float = 0.0
    annualized_vol: float = 0.20
    expected_drift: float = 0.08
    drawdown_days: list[int] = field(default_factory=list)
    signal_channels: list[str] = field(default_factory=list)


def _generate_base_returns(
    n_days: int,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate GBM base returns."""
    dt = 1.0 / 252.0
    return (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
        n_days
    )


def _inject_drawdowns(
    returns: np.ndarray,
    n_drawdowns: int,
    magnitude: float,
    duration: int,
    rng: np.random.Generator,
    buffer: int = 40,
) -> tuple[np.ndarray, list[int]]:
    """Inject drawdown events at random positions with minimum spacing.

    Args:
        returns: Base return series (modified in place and returned).
        n_drawdowns: Number of drawdown events to inject.
        magnitude: Daily drawdown magnitude (e.g., 0.02 = -2%/day).
        duration: Number of days per drawdown event.
        rng: Random generator.
        buffer: Minimum days between drawdown start and series boundaries/other drawdowns.

    Returns:
        Tuple of (modified returns, sorted list of drawdown start days).
    """
    n = len(returns)
    modified = returns.copy()
    drawdown_days: list[int] = []

    # Need space for signal lead (10 days before) + drawdown duration + buffer after
    lead_buffer = 15
    usable_start = buffer + lead_buffer
    usable_end = n - duration - buffer

    if usable_end <= usable_start:
        return modified, []

    # Pick non-overlapping drawdown positions
    attempts = 0
    while len(drawdown_days) < n_drawdowns and attempts < 500:
        candidate = int(rng.integers(usable_start, usable_end))
        # Check minimum spacing from existing drawdowns
        too_close = any(
            abs(candidate - existing) < duration + buffer
            for existing in drawdown_days
        )
        if not too_close:
            drawdown_days.append(candidate)
        attempts += 1

    drawdown_days.sort()

    # Inject drawdowns
    for day in drawdown_days:
        for d in range(duration):
            if day + d < n:
                modified[day + d] -= magnitude

    return modified, drawdown_days


def _generate_forecasts(
    returns: np.ndarray,
    accuracy: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate daily forecasts with known directional accuracy.

    Uses a noisy oracle model: the forecast is a scaled version of the true
    future return plus noise. The accuracy parameter controls the fraction
    of days where the forecast has the correct sign.

    Args:
        returns: Actual daily return series.
        accuracy: Target directional accuracy (0.5 = random, 1.0 = perfect).
        rng: Random generator.

    Returns:
        Array of daily forecasts (same length as returns).
    """
    n = len(returns)
    forecasts = np.zeros(n)
    noise_std = np.std(returns) if np.std(returns) > 0 else 0.005

    for t in range(n):
        if rng.random() < accuracy:
            # Correct direction: use true return sign with noise
            forecasts[t] = returns[t] + rng.normal(0, noise_std * 0.3)
        else:
            # Wrong direction: flip the sign
            forecasts[t] = -returns[t] + rng.normal(0, noise_std * 0.3)

    return forecasts


def _generate_fvs_events(
    drawdown_days: list[int],
    lead_days: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate FVS events preceding drawdowns.

    Args:
        drawdown_days: Days when drawdowns start.
        lead_days: How many days before drawdown to place event.
        rng: Random generator.

    Returns:
        List of event dicts with day, event_type, severity.
    """
    event_types = [
        "guidance_cut",
        "kpi_miss",
        "mgmt_change",
        "capital_allocation",
    ]
    events = []
    for dd in drawdown_days:
        event_day = dd - lead_days
        if event_day >= 0:
            events.append({
                "day": event_day,
                "event_type": rng.choice(event_types),
                "severity": float(rng.uniform(0.4, 0.8)),
            })
    return events


def _generate_vol_signals(
    n_days: int,
    drawdown_days: list[int],
    base_vol: float,
    lead_days: int,
    duration: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate implied vol series and regime labels.

    IV spikes lead_days before drawdowns. Regime switches to stressed (1)
    around drawdown windows.

    Args:
        n_days: Length of series.
        drawdown_days: Days when drawdowns start.
        base_vol: Baseline annualized vol.
        lead_days: Days before drawdown that IV spikes.
        duration: Drawdown duration (regime=1 during this window).
        rng: Random generator.

    Returns:
        Tuple of (implied_vols array, regimes array).
    """
    # Base IV with small noise
    daily_base = base_vol / np.sqrt(252)
    implied_vols = daily_base * (1 + rng.normal(0, 0.05, n_days))
    implied_vols = np.clip(implied_vols, daily_base * 0.5, daily_base * 5.0)

    regimes = np.zeros(n_days, dtype=int)

    for dd in drawdown_days:
        # IV spike starts lead_days before drawdown
        spike_start = max(0, dd - lead_days)
        spike_end = min(n_days, dd + duration)
        for t in range(spike_start, spike_end):
            implied_vols[t] *= 2.5  # IV roughly 2.5x base during stress

        # Regime = stressed during drawdown window
        regime_start = max(0, dd - 1)
        regime_end = min(n_days, dd + duration + 2)
        regimes[regime_start:regime_end] = 1

    return implied_vols, regimes


def _generate_debate_signals(
    n_days: int,
    drawdown_days: list[int],
    lead_days: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate IC debate schedules preceding drawdowns.

    Positive ADS (p_post > p_pre) before drawdowns acts as a contrarian
    signal: committee becomes more confident right before trouble.

    Also generates some noise debates on random days (no signal).

    Args:
        n_days: Length of series.
        drawdown_days: Days when drawdowns start.
        lead_days: Days before drawdown to schedule debate.
        rng: Random generator.

    Returns:
        List of debate dicts with day, p_pre, p_post.
    """
    debates = []

    # Signal debates: before drawdowns, committee becomes MORE confident
    # (positive ADS → gradient increases → conviction decreases → protective)
    for dd in drawdown_days:
        debate_day = dd - lead_days
        if 0 <= debate_day < n_days:
            # Pre-debate: moderate confidence
            p_pre = [float(rng.uniform(0.4, 0.6)) for _ in range(3)]
            # Post-debate: higher confidence (contrarian — trouble ahead)
            p_post = [p + float(rng.uniform(0.1, 0.25)) for p in p_pre]
            p_post = [min(p, 0.95) for p in p_post]
            debates.append({
                "day": debate_day,
                "p_pre": p_pre,
                "p_post": p_post,
            })

    # Add some noise debates on random non-drawdown days
    n_noise = max(2, len(drawdown_days))
    noise_days = rng.choice(
        [d for d in range(30, n_days - 10) if all(abs(d - dd) > 15 for dd in drawdown_days)],
        size=min(n_noise, max(1, n_days // 50)),
        replace=False,
    )
    for nd in noise_days:
        p_pre = [float(rng.uniform(0.4, 0.6)) for _ in range(3)]
        # Noise: small random shift (no consistent direction)
        p_post = [p + float(rng.uniform(-0.05, 0.05)) for p in p_pre]
        p_post = [max(0.05, min(0.95, p)) for p in p_post]
        debates.append({
            "day": int(nd),
            "p_pre": p_pre,
            "p_post": p_post,
        })

    return sorted(debates, key=lambda x: x["day"])


def generate_signal_asset(
    n_days: int = 504,
    seed: int = 42,
    channels: list[str] | None = None,
    name: str = "SIGNAL",
    mu: float = 0.08,
    sigma: float = 0.20,
    forecast_accuracy: float = 0.60,
    n_drawdowns: int = 5,
    drawdown_magnitude: float = 0.02,
    drawdown_duration: int = 5,
    fvs_lead: int = 5,
    rrs_lead: int = 3,
    ads_lead: int = 2,
    track_record: float = 0.5,
) -> SignalEmbeddedAsset:
    """Generate a single signal-embedded asset.

    Args:
        n_days: Trading days.
        seed: Random seed.
        channels: Active signal channels (subset of ["fe", "fvs", "rrs", "ads"]).
        name: Asset identifier.
        mu: Annualized drift.
        sigma: Annualized vol.
        forecast_accuracy: Directional accuracy for FE channel (0.5-1.0).
        n_drawdowns: Number of drawdown events to inject.
        drawdown_magnitude: Daily magnitude of drawdown (e.g., 0.02 = -2%/day).
        drawdown_duration: Days per drawdown event.
        fvs_lead: Days before drawdown for FVS events.
        rrs_lead: Days before drawdown for IV spike.
        ads_lead: Days before drawdown for debate shift.
        track_record: Analyst track record score.

    Returns:
        SignalEmbeddedAsset with all signal data populated.
    """
    channels = channels or []
    rng = np.random.default_rng(seed)

    # Base returns + drawdowns
    base = _generate_base_returns(n_days, mu, sigma, rng)
    returns, drawdown_days = _inject_drawdowns(
        base, n_drawdowns, drawdown_magnitude, drawdown_duration, rng
    )

    # FE: forecasts (active channel = accurate, inactive = trailing mean proxy)
    if "fe" in channels:
        forecasts = _generate_forecasts(returns, forecast_accuracy, rng)
    else:
        # No signal: use noisy random forecast (50% accuracy)
        forecasts = _generate_forecasts(returns, 0.50, rng)

    # FVS: events before drawdowns (only if channel active)
    fvs_schedule = (
        _generate_fvs_events(drawdown_days, fvs_lead, rng) if "fvs" in channels else []
    )

    # RRS: IV signals and regimes (only if channel active)
    if "rrs" in channels:
        implied_vols, regimes = _generate_vol_signals(
            n_days, drawdown_days, sigma, rrs_lead, drawdown_duration, rng
        )
    else:
        daily_base = sigma / np.sqrt(252)
        implied_vols = daily_base * (1 + rng.normal(0, 0.05, n_days))
        implied_vols = np.clip(implied_vols, daily_base * 0.5, daily_base * 3.0)
        regimes = np.zeros(n_days, dtype=int)

    # ADS: debate signals (only if channel active)
    if "ads" in channels:
        debate_schedule = _generate_debate_signals(
            n_days, drawdown_days, ads_lead, rng
        )
    else:
        debate_schedule = []

    return SignalEmbeddedAsset(
        returns=returns,
        name=name,
        asset_type=f"signal_{'_'.join(channels) if channels else 'none'}",
        forecasts=forecasts,
        fvs_schedule=fvs_schedule,
        implied_vols=implied_vols,
        regimes=regimes,
        debate_schedule=debate_schedule,
        track_record_score=track_record,
        annualized_vol=sigma,
        expected_drift=mu,
        drawdown_days=drawdown_days,
        signal_channels=channels,
    )


def generate_signal_universe(
    n_days: int = 504,
    seed: int = 42,
) -> tuple[list[SignalEmbeddedAsset], np.ndarray]:
    """Generate 5-asset signal-embedded universe with market returns.

    Assets:
    1. FE_SIGNAL:  Forecast accuracy 60%, FE channel only
    2. FVS_SIGNAL: FVS events 5d before drawdowns
    3. RRS_SIGNAL: IV leads HV by 3d, vol regime shifts
    4. ADS_SIGNAL: Debate shifts 2d before drawdowns
    5. ALL_SIGNAL: All 4 channels, moderate signal strength

    Args:
        n_days: Trading days per asset.
        seed: Base seed (each asset offsets).

    Returns:
        Tuple of (list of SignalEmbeddedAsset, market_returns array).
    """
    assets = [
        generate_signal_asset(
            n_days=n_days,
            seed=seed,
            channels=["fe"],
            name="FE_SIGNAL",
            forecast_accuracy=0.60,
        ),
        generate_signal_asset(
            n_days=n_days,
            seed=seed + 1,
            channels=["fvs"],
            name="FVS_SIGNAL",
            fvs_lead=5,
        ),
        generate_signal_asset(
            n_days=n_days,
            seed=seed + 2,
            channels=["rrs"],
            name="RRS_SIGNAL",
            rrs_lead=3,
        ),
        generate_signal_asset(
            n_days=n_days,
            seed=seed + 3,
            channels=["ads"],
            name="ADS_SIGNAL",
            ads_lead=2,
        ),
        generate_signal_asset(
            n_days=n_days,
            seed=seed + 4,
            channels=["fe", "fvs", "rrs", "ads"],
            name="ALL_SIGNAL",
            forecast_accuracy=0.57,
            track_record=0.3,
        ),
    ]

    rng = np.random.default_rng(seed + 100)
    dt = 1.0 / 252.0
    market = (0.07 - 0.5 * 0.16**2) * dt + 0.16 * np.sqrt(dt) * rng.standard_normal(
        n_days
    )

    return assets, market
